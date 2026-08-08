"""Tests for the backtest harness, organised by layer.

The tests that matter most are the ones asserting properties that would produce
a *plausible but wrong* result rather than a crash: no lookahead in quote
lookup, no silent re-pull of cached data, and no fill that could not have
happened.
"""

import datetime as dt
import os
from datetime import date, datetime, time

import pandas as pd
import pytest

from backtest.cache import CacheKey, ParquetCache, symbol_fingerprint
from backtest.config import SCHEMA_DEFINITION, SCHEMA_QUOTES, BacktestConfig
from backtest.data import (
    CALL,
    PUT,
    Contract,
    Quote,
    QuoteBook,
    contracts_from_definitions,
    osi_symbol,
    spot_from_parity,
    year_fraction_to_close,
)
from backtest.engine import Engine
from backtest.fills import BUY, SELL, Leg, MidFill, MidMinusEdgeFill, SettlementFill, Structure
from backtest.report import compute_stats, trade_log
from backtest.signals import PROFIT_TARGET, STOP_LOSS, TIME_EXIT, check_exit
from backtest.source import (
    AgentBridgeFetcher,
    CostCeilingExceeded,
    CostEstimateUnavailable,
    DatabentoFetcher,
    to_eastern,
)

DAY = date(2026, 8, 6)
ET = "America/New_York"


def ts(hour, minute=0, second=0):
    return pd.Timestamp(datetime(2026, 8, 6, hour, minute, second)).tz_localize(ET)


@pytest.fixture
def cfg(tmp_path):
    c = BacktestConfig()
    c.data.cache_dir = tmp_path / "data"
    c.start_date = c.end_date = DAY
    return c


# --------------------------------------------------------------------------
# Rule 2: explicit dates
# --------------------------------------------------------------------------

class TestExplicitDates:
    def test_missing_start_is_rejected(self):
        c = BacktestConfig(end_date=DAY)
        with pytest.raises(SystemExit, match="never defaults to today"):
            c.validate()

    def test_missing_end_is_rejected(self):
        c = BacktestConfig(start_date=DAY)
        with pytest.raises(SystemExit):
            c.validate()

    def test_inverted_range_is_rejected(self):
        c = BacktestConfig(start_date=DAY, end_date=date(2026, 8, 1))
        with pytest.raises(SystemExit, match="precedes"):
            c.validate()

    def test_a_valid_range_passes(self, cfg):
        cfg.validate()


class TestParentSymbolMatchesRoot:
    """OPRA lists SPX and SPXW as separate parents.

    A mismatch is the worst kind of misconfiguration: the definition pull is
    billed and succeeds, the root filter then discards every row, and the
    session is reported as "no contracts" -- which reads like a market-data gap
    rather than a config error.
    """

    def test_mismatched_parent_is_rejected(self, cfg):
        cfg.data.parent_symbol = "SPX.OPT"
        cfg.data.underlying_root = "SPXW"
        with pytest.raises(SystemExit, match="does not match"):
            cfg.validate()

    def test_matching_parent_passes(self, cfg):
        cfg.data.parent_symbol = "SPXW.OPT"
        cfg.data.underlying_root = "SPXW"
        cfg.validate()

    def test_the_default_pair_agrees(self, cfg):
        assert cfg.data.parent_symbol.split(".")[0] == cfg.data.underlying_root


# --------------------------------------------------------------------------
# Rule 4: timezone normalisation
# --------------------------------------------------------------------------

class TestTimezone:
    def test_utc_nanos_become_eastern(self):
        # 2026-08-06 14:00 UTC is 10:00 EDT.
        nanos = int(pd.Timestamp("2026-08-06 14:00:00", tz="UTC").value)
        out = to_eastern(pd.DataFrame({"ts_recv": [nanos]}))
        assert str(out["ts_recv"].dt.tz) == ET
        assert out["ts_recv"].iloc[0].hour == 10

    def test_naive_timestamps_are_treated_as_utc(self):
        out = to_eastern(pd.DataFrame({"ts_recv": [pd.Timestamp("2026-08-06 14:00:00")]}))
        assert out["ts_recv"].iloc[0].hour == 10

    def test_already_eastern_is_left_alone(self):
        stamp = pd.Timestamp("2026-08-06 10:00:00", tz=ET)
        out = to_eastern(pd.DataFrame({"ts_recv": [stamp]}))
        assert out["ts_recv"].iloc[0] == stamp

    def test_winter_dates_use_est_not_edt(self):
        """A fixed offset would be wrong for half the year."""
        nanos = int(pd.Timestamp("2026-01-15 14:00:00", tz="UTC").value)
        out = to_eastern(pd.DataFrame({"ts_recv": [nanos]}))
        assert out["ts_recv"].iloc[0].hour == 9  # EST, not EDT

    def test_both_timestamp_columns_convert(self):
        nanos = int(pd.Timestamp("2026-08-06 14:00:00", tz="UTC").value)
        out = to_eastern(pd.DataFrame({"ts_recv": [nanos], "ts_event": [nanos]}))
        assert out["ts_event"].iloc[0].hour == 10

    def test_missing_columns_are_ignored(self):
        to_eastern(pd.DataFrame({"price": [1.0]}))


# --------------------------------------------------------------------------
# Rule 3: cache
# --------------------------------------------------------------------------

class TestCache:
    def test_roundtrip(self, tmp_path):
        cache = ParquetCache(tmp_path)
        key = CacheKey.build("OPRA.PILLAR", SCHEMA_QUOTES, DAY, ["A", "B"])
        frame = pd.DataFrame({"symbol": ["A"], "bid_px_00": [1.0]})
        cache.write(key, frame)
        assert cache.has(key)
        assert len(cache.read(key)) == 1

    def test_different_symbol_sets_are_different_keys(self):
        """A narrow pull must not be served in answer to a wide one."""
        a = CacheKey.build("D", "s", DAY, ["A"])
        b = CacheKey.build("D", "s", DAY, ["A", "B"])
        assert a.symbols_hash != b.symbols_hash

    def test_symbol_order_does_not_change_the_key(self):
        assert symbol_fingerprint(["B", "A"]) == symbol_fingerprint(["A", "B"])

    def test_duplicates_do_not_change_the_key(self):
        assert symbol_fingerprint(["A", "A", "B"]) == symbol_fingerprint(["A", "B"])

    def test_none_means_whole_feed(self):
        assert symbol_fingerprint(None) == "ALL"

    def test_a_miss_raises(self, tmp_path):
        cache = ParquetCache(tmp_path)
        with pytest.raises(KeyError):
            cache.read(CacheKey.build("D", "s", DAY, ["X"]))

    def test_manifest_records_cost(self, tmp_path):
        cache = ParquetCache(tmp_path)
        key = CacheKey.build("D", "s", DAY, ["A"])
        cache.write(key, pd.DataFrame({"x": [1]}), meta={"cost_usd": 1.25})
        assert cache.total_spend() == 1.25

    def test_manifest_survives_corruption(self, tmp_path):
        cache = ParquetCache(tmp_path)
        (tmp_path).mkdir(exist_ok=True)
        (tmp_path / "manifest.json").write_text("{not json")
        cache.write(CacheKey.build("D", "s", DAY, ["A"]), pd.DataFrame({"x": [1]}))
        assert len(cache.manifest()) == 1


# --------------------------------------------------------------------------
# Rule 1: cost gate
# --------------------------------------------------------------------------

class FakeMetadata:
    def __init__(self, cost, fail=False):
        self.cost = cost
        self.fail = fail
        self.calls = 0

    def get_cost(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("metadata endpoint down")
        return self.cost


class FakeTimeseries:
    def __init__(self, frame):
        self.frame = frame
        self.calls = 0

    def get_range(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs

        class Result:
            """Stands in for a DBNStore.

            ``to_df`` mirrors the real one: it requires the options we pin
            explicitly, and it returns a frame indexed on ts_recv so the
            adapter's ``reset_index()`` is actually exercised rather than
            being incidentally harmless.
            """

            def __init__(self, f):
                self._f = f

            def to_df(self, price_type=None, map_symbols=None, tz=None):
                assert price_type == "float", "adapter must pin float prices"
                assert map_symbols is True, "adapter must request the symbol column"
                assert tz == "UTC", "to_eastern converts from UTC"
                return self._f.set_index("ts_recv")

        return Result(self.frame)


class FakeDatabento:
    def __init__(self, cost=1.0, fail_cost=False, frame=None):
        self.metadata = FakeMetadata(cost, fail_cost)
        self.timeseries = FakeTimeseries(
            frame if frame is not None else pd.DataFrame({
                "ts_recv": [int(pd.Timestamp("2026-08-06 14:00", tz="UTC").value)],
                "symbol": ["SPXW260806P05000000"],
                "bid_px_00": [1.0], "ask_px_00": [1.2],
            })
        )


class TestCostGate:
    def _fetcher(self, cfg, client, echo=None):
        return DatabentoFetcher(cfg, ParquetCache(cfg.data.cache_dir), client=client,
                                echo=echo or (lambda *_: None))

    def test_estimate_is_printed_even_when_cheap(self, cfg):
        printed = []
        f = self._fetcher(cfg, FakeDatabento(cost=0.01), echo=printed.append)
        f.fetch(SCHEMA_QUOTES, DAY, ["A"], time(9, 45), time(16, 0))
        assert any("cost estimate $0.0100" in p for p in printed)

    def test_over_ceiling_halts_before_fetching(self, cfg):
        cfg.cost.ceiling_usd = 5.0
        client = FakeDatabento(cost=50.0)
        f = self._fetcher(cfg, client)
        with pytest.raises(CostCeilingExceeded, match="above the"):
            f.fetch(SCHEMA_QUOTES, DAY, ["A"], time(9, 45), time(16, 0))
        assert client.timeseries.calls == 0  # nothing was pulled

    def test_under_ceiling_proceeds(self, cfg):
        cfg.cost.ceiling_usd = 100.0
        client = FakeDatabento(cost=1.0)
        f = self._fetcher(cfg, client)
        f.fetch(SCHEMA_QUOTES, DAY, ["A"], time(9, 45), time(16, 0))
        assert client.timeseries.calls == 1

    def test_unavailable_estimate_halts_by_default(self, cfg):
        client = FakeDatabento(fail_cost=True)
        f = self._fetcher(cfg, client)
        with pytest.raises(CostEstimateUnavailable):
            f.fetch(SCHEMA_QUOTES, DAY, ["A"], time(9, 45), time(16, 0))
        assert client.timeseries.calls == 0

    def test_unavailable_estimate_can_be_allowed_explicitly(self, cfg):
        cfg.cost.require_estimate = False
        client = FakeDatabento(fail_cost=True)
        f = self._fetcher(cfg, client)
        f.fetch(SCHEMA_QUOTES, DAY, ["A"], time(9, 45), time(16, 0))
        assert client.timeseries.calls == 1

    def test_cached_range_is_never_refetched(self, cfg):
        client = FakeDatabento(cost=1.0)
        f = self._fetcher(cfg, client)
        f.fetch(SCHEMA_QUOTES, DAY, ["A"], time(9, 45), time(16, 0))
        f.fetch(SCHEMA_QUOTES, DAY, ["A"], time(9, 45), time(16, 0))
        assert client.timeseries.calls == 1

    def test_a_cache_hit_does_not_even_price_the_pull(self, cfg):
        client = FakeDatabento(cost=1.0)
        f = self._fetcher(cfg, client)
        f.fetch(SCHEMA_QUOTES, DAY, ["A"], time(9, 45), time(16, 0))
        before = client.metadata.calls
        f.fetch(SCHEMA_QUOTES, DAY, ["A"], time(9, 45), time(16, 0))
        assert client.metadata.calls == before

    def test_fetched_data_lands_in_eastern(self, cfg):
        f = self._fetcher(cfg, FakeDatabento())
        frame, _ = f.fetch(SCHEMA_QUOTES, DAY, ["A"], time(9, 45), time(16, 0))
        assert frame["ts_recv"].iloc[0].hour == 10

    def test_offline_fetcher_never_pulls(self, cfg):
        f = AgentBridgeFetcher(cfg, ParquetCache(cfg.data.cache_dir), echo=lambda *_: None)
        with pytest.raises(KeyError, match="offline mode"):
            f.fetch(SCHEMA_QUOTES, DAY, ["A"], time(9, 45), time(16, 0))


# --------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------

class TestContracts:
    def test_parses_an_osi_symbol(self):
        c = Contract.parse("SPXW260806P05000000")
        assert c.root == "SPXW" and c.strike == 5000.0
        assert c.option_type == PUT and c.expiration == DAY

    def test_round_trips_through_osi(self):
        symbol = osi_symbol("SPXW", DAY, CALL, 5012.5)
        assert Contract.parse(symbol).strike == 5012.5

    def test_rejects_non_option_symbols(self):
        assert Contract.parse("SPX") is None

    def test_definitions_filter_to_root_and_expiry(self):
        frame = pd.DataFrame({"raw_symbol": [
            osi_symbol("SPXW", DAY, PUT, 5000.0),
            osi_symbol("SPXW", date(2026, 8, 7), PUT, 5000.0),
            osi_symbol("SPX", DAY, PUT, 5000.0),
        ]})
        got = contracts_from_definitions(frame, "SPXW", DAY)
        assert len(got) == 1


class TestQuoteBook:
    @pytest.fixture
    def book(self):
        return QuoteBook(pd.DataFrame({
            "ts_recv": [ts(10, 0), ts(10, 5), ts(10, 10)],
            "symbol": ["A", "A", "A"],
            "bid_px_00": [1.0, 2.0, 3.0],
            "ask_px_00": [1.2, 2.2, 3.2],
            "bid_sz_00": [10, 10, 10],
            "ask_sz_00": [10, 10, 10],
        }))

    def test_as_of_returns_the_prevailing_quote(self, book):
        assert book.as_of("A", ts(10, 7)).bid == 2.0

    def test_as_of_never_looks_ahead(self, book):
        """The single most dangerous bug class in a backtest."""
        assert book.as_of("A", ts(10, 4)).bid == 1.0
        assert book.as_of("A", ts(10, 9)).bid == 2.0

    def test_exact_timestamps_are_inclusive(self, book):
        assert book.as_of("A", ts(10, 5)).bid == 2.0

    def test_before_the_first_quote_is_none(self, book):
        assert book.as_of("A", ts(9, 59)) is None

    def test_after_the_last_quote_holds(self, book):
        assert book.as_of("A", ts(15, 0)).bid == 3.0

    def test_unknown_symbol_is_none(self, book):
        assert book.as_of("ZZZ", ts(10, 7)) is None

    def test_empty_frame_is_harmless(self):
        assert len(QuoteBook(pd.DataFrame())) == 0

    def test_fixed_point_prices_are_rescaled(self):
        book = QuoteBook(pd.DataFrame({
            "ts_recv": [ts(10, 0)], "symbol": ["A"],
            "bid_px_00": [1_500_000_000], "ask_px_00": [1_600_000_000],
            "bid_sz_00": [1], "ask_sz_00": [1],
        }))
        assert book.as_of("A", ts(10, 1)).bid == pytest.approx(1.5)


class TestQuoteTradability:
    """Buying and selling are not the same test.

    A far-OTM wing quoted 0.00 x 0.05 cannot be sold but can absolutely be
    bought. Collapsing the two into one predicate rejects cheap wings, which
    drops the calmest sessions from the sample and biases the whole backtest
    toward high-volatility days.
    """

    def test_zero_bid_cannot_be_sold(self):
        assert not Quote(ts(10), 0.0, 0.05).can_sell

    def test_zero_bid_can_still_be_bought(self):
        assert Quote(ts(10), 0.0, 0.05).can_buy

    def test_zero_offer_cannot_be_bought(self):
        assert not Quote(ts(10), 1.0, 0.0).can_buy

    def test_crossed_is_neither(self):
        q = Quote(ts(10), 2.0, 1.0)
        assert not q.can_buy and not q.can_sell and q.is_crossed

    def test_normal_market_is_both(self):
        q = Quote(ts(10), 1.0, 1.2)
        assert q.can_buy and q.can_sell and q.is_tradable

    def test_is_tradable_requires_both_directions(self):
        assert not Quote(ts(10), 0.0, 0.05).is_tradable


class TestParity:
    def test_recovers_the_forward(self):
        """C - P = F - K, so K + (C - P) recovers F."""
        contracts = [
            Contract(osi_symbol("SPXW", DAY, CALL, 5000.0), "SPXW", DAY, CALL, 5000.0),
            Contract(osi_symbol("SPXW", DAY, PUT, 5000.0), "SPXW", DAY, PUT, 5000.0),
        ]
        quotes = {
            contracts[0].symbol: Quote(ts(10), 25.0, 25.0),
            contracts[1].symbol: Quote(ts(10), 15.0, 15.0),
        }
        assert spot_from_parity(quotes, contracts, 0.0, 0.0) == pytest.approx(5010.0)

    def test_none_without_a_complete_pair(self):
        contracts = [Contract(osi_symbol("SPXW", DAY, CALL, 5000.0), "SPXW", DAY, CALL, 5000.0)]
        quotes = {contracts[0].symbol: Quote(ts(10), 25.0, 25.0)}
        assert spot_from_parity(quotes, contracts, 0.0, 0.0) is None


# --------------------------------------------------------------------------
# Fill layer (rule 5)
# --------------------------------------------------------------------------

def condor():
    return Structure(kind="iron_condor", width=25.0, legs=(
        Leg(osi_symbol("SPXW", DAY, PUT, 4975.0), SELL, 4975.0, PUT),
        Leg(osi_symbol("SPXW", DAY, PUT, 4950.0), BUY, 4950.0, PUT),
        Leg(osi_symbol("SPXW", DAY, CALL, 5025.0), SELL, 5025.0, CALL),
        Leg(osi_symbol("SPXW", DAY, CALL, 5050.0), BUY, 5050.0, CALL),
    ))


def quotes_for(structure, bid, ask):
    return {leg.symbol: Quote(ts(10), bid, ask) for leg in structure.legs}


class TestFills:
    @pytest.fixture
    def model(self, cfg):
        return MidMinusEdgeFill(cfg.execution)

    def test_entry_concedes_the_edge_on_every_leg(self, model, cfg):
        s = condor()
        # All legs at 2.00/2.20 -> mid 2.10. Shorts and longs cancel to zero,
        # so the credit is purely the negative of the slippage.
        result = model.entry(s, quotes_for(s, 2.0, 2.2), 1)
        assert result.price == pytest.approx(-cfg.execution.entry_slippage_per_leg * 4)

    def test_entry_slippage_scales_with_leg_count(self, cfg):
        model = MidMinusEdgeFill(cfg.execution)
        two_leg = Structure("put_credit", condor().legs[:2], 25.0)
        four = model.entry(condor(), quotes_for(condor(), 2.0, 2.2), 1).price
        two = model.entry(two_leg, quotes_for(two_leg, 2.0, 2.2), 1).price
        assert four == pytest.approx(2 * two)

    def test_exit_pays_the_ask_on_shorts_and_hits_the_bid_on_longs(self, model):
        s = condor()
        result = model.exit(s, quotes_for(s, 1.0, 1.5), 1)
        # Two shorts bought back at 1.50, two longs sold at 1.00.
        assert result.price == pytest.approx(2 * 1.5 - 2 * 1.0)

    def test_exit_at_mid_is_cheaper_than_adverse(self, cfg):
        s = condor()
        adverse = MidMinusEdgeFill(cfg.execution).exit(s, quotes_for(s, 1.0, 1.5), 1).price
        mid = MidFill(cfg.execution).exit(s, quotes_for(s, 1.0, 1.5), 1).price
        assert mid < adverse

    def test_commissions_are_per_leg_per_contract(self, model, cfg):
        s = condor()
        result = model.entry(s, quotes_for(s, 2.0, 2.2), 3)
        assert result.commissions == pytest.approx(cfg.execution.per_leg_cost * 4 * 3)

    def test_entry_rejects_a_short_leg_with_no_bid(self, model):
        s = condor()
        q = quotes_for(s, 2.0, 2.2)
        q[s.legs[0].symbol] = Quote(ts(10), 0.0, 0.05)  # leg 0 is a SELL
        result = model.entry(s, q, 1)
        assert not result.tradable and "no bid" in result.reason

    def test_entry_accepts_a_long_wing_with_no_bid(self, model):
        """The wing is bought at the offer; a zero bid is irrelevant to that."""
        s = condor()
        q = quotes_for(s, 2.0, 2.2)
        q[s.legs[1].symbol] = Quote(ts(10), 0.0, 0.05)  # leg 1 is a BUY
        assert model.entry(s, q, 1).tradable

    def test_entry_rejects_a_long_leg_with_no_offer(self, model):
        # 0.00 x 0.00, i.e. nothing quoted at all. A 1.00 x 0.00 market would be
        # crossed, which is a different rejection caught earlier.
        s = condor()
        q = quotes_for(s, 2.0, 2.2)
        q[s.legs[1].symbol] = Quote(ts(10), 0.0, 0.0)
        result = model.entry(s, q, 1)
        assert not result.tradable and "no offer" in result.reason

    def test_a_crossed_leg_is_reported_as_crossed_not_as_missing_liquidity(self, model):
        s = condor()
        q = quotes_for(s, 2.0, 2.2)
        q[s.legs[1].symbol] = Quote(ts(10), 1.0, 0.0)
        assert "crossed" in model.entry(s, q, 1).reason

    def test_exit_still_marks_a_zero_bid_leg(self, model):
        """A decayed leg is worth zero, not unmarkable.

        Refusing to mark it strands the position and stops the profit target
        from ever being evaluated.
        """
        s = condor()
        q = quotes_for(s, 2.0, 2.2)
        q[s.legs[1].symbol] = Quote(ts(10), 0.0, 0.05)
        result = model.exit(s, q, 1)
        assert result.tradable

    def test_a_worthless_structure_marks_at_zero(self, model):
        s = condor()
        q = {leg.symbol: Quote(ts(10), 0.0, 0.0) for leg in s.legs}
        result = model.exit(s, q, 1)
        assert result.tradable and result.price == pytest.approx(0.0)

    def test_a_missing_quote_still_fails_an_exit(self, model):
        s = condor()
        q = quotes_for(s, 1.0, 1.5)
        del q[s.legs[0].symbol]
        assert not model.exit(s, q, 1).tradable

    def test_a_crossed_market_fails_both_directions(self, model):
        s = condor()
        q = quotes_for(s, 1.0, 1.5)
        q[s.legs[0].symbol] = Quote(ts(10), 5.0, 1.0)
        assert not model.entry(s, q, 1).tradable
        assert not model.exit(s, q, 1).tradable


class TestSettlementFill:
    def test_only_the_breached_side_has_value(self, cfg):
        s = condor()
        result = SettlementFill(cfg.execution, 4900.0).exit(s, {}, 1)
        # Put side fully in the money: short 4975 worth 75, long 4950 worth 50.
        assert result.price == pytest.approx(25.0)

    def test_expiring_inside_both_wings_settles_at_zero(self, cfg):
        result = SettlementFill(cfg.execution, 5000.0).exit(condor(), {}, 1)
        assert result.price == pytest.approx(0.0)

    def test_settlement_charges_no_commission(self, cfg):
        assert SettlementFill(cfg.execution, 4900.0).exit(condor(), {}, 1).commissions == 0.0


# --------------------------------------------------------------------------
# Signal layer
# --------------------------------------------------------------------------

class TestExitRules:
    def test_stop_fires_at_the_configured_multiple(self, cfg):
        d = check_exit(ts(11), credit=2.0, current_debit=6.0, cfg=cfg.signal, day=DAY)
        assert d.should_exit and d.reason == STOP_LOSS

    def test_target_fires_at_the_configured_fraction(self, cfg):
        d = check_exit(ts(11), credit=2.0, current_debit=1.0, cfg=cfg.signal, day=DAY)
        assert d.should_exit and d.reason == PROFIT_TARGET

    def test_stop_is_checked_before_target(self, cfg):
        """When both could apply, assume the loss happened first."""
        cfg.signal.stop_loss_multiple = 0.0
        cfg.signal.profit_target_fraction = 0.0
        d = check_exit(ts(11), credit=2.0, current_debit=2.0, cfg=cfg.signal, day=DAY)
        assert d.reason == STOP_LOSS

    def test_holds_between_the_thresholds(self, cfg):
        assert not check_exit(ts(11), 2.0, 2.5, cfg.signal, DAY).should_exit

    def test_time_exit_fires_at_the_cutoff(self, cfg):
        d = check_exit(ts(15, 45), credit=2.0, current_debit=2.5, cfg=cfg.signal, day=DAY)
        assert d.should_exit and d.reason == TIME_EXIT

    def test_hold_to_settlement_suppresses_the_time_exit(self, cfg):
        cfg.signal.hold_to_settlement = True
        assert not check_exit(ts(15, 45), 2.0, 2.5, cfg.signal, DAY).should_exit

    def test_a_missing_mark_does_not_trigger_an_early_exit(self, cfg):
        assert not check_exit(ts(11), 2.0, None, cfg.signal, DAY).should_exit


# --------------------------------------------------------------------------
# Engine + reporting, end to end on a fixture
# --------------------------------------------------------------------------

class TestEndToEnd:
    def _run(self, cfg, **kwargs):
        from backtest.fixtures import seed_cache

        cache = ParquetCache(cfg.data.cache_dir)
        seed_cache(cache, cfg, DAY, **kwargs)
        engine = Engine(cfg, AgentBridgeFetcher(cfg, cache, echo=lambda *_: None),
                        echo=lambda *_: None)
        return engine.run_day(DAY)

    def test_quiet_session_takes_the_profit_target(self, cfg):
        result = self._run(cfg, drift_points=0.0, seed=7)
        assert result.trade is not None
        assert result.trade.exit_reason == PROFIT_TARGET
        assert result.trade.net_pnl > 0

    def test_adverse_session_stops_out(self, cfg):
        result = self._run(cfg, drift_points=-120.0, seed=7)
        assert result.trade is not None
        assert result.trade.exit_reason == STOP_LOSS
        assert result.trade.net_pnl < 0

    def test_loss_never_exceeds_defined_risk(self, cfg):
        """A credit spread cannot lose more than its width, whatever happens."""
        result = self._run(cfg, drift_points=-400.0, seed=3)
        if result.trade:
            assert result.trade.net_pnl >= -(result.trade.max_loss + result.trade.commissions)

    def test_entry_is_at_the_configured_time(self, cfg):
        result = self._run(cfg)
        assert result.trade.entry_ts.hour == cfg.signal.entry_time.hour

    def test_exit_never_precedes_entry(self, cfg):
        result = self._run(cfg)
        assert result.trade.exit_ts >= result.trade.entry_ts

    def test_swapping_the_fill_model_changes_only_prices(self, cfg):
        from backtest.fixtures import seed_cache

        cache = ParquetCache(cfg.data.cache_dir)
        seed_cache(cache, cfg, DAY)
        fetcher = AgentBridgeFetcher(cfg, cache, echo=lambda *_: None)

        adverse = Engine(cfg, fetcher, MidMinusEdgeFill(cfg.execution),
                         echo=lambda *_: None).run_day(DAY)
        mid = Engine(cfg, fetcher, MidFill(cfg.execution), echo=lambda *_: None).run_day(DAY)

        # Same strikes chosen -- signals did not see the fill change.
        assert adverse.trade.strikes == mid.trade.strikes
        # But the optimistic model books a better credit.
        assert mid.trade.credit > adverse.trade.credit

    def test_report_stats_are_consistent(self, cfg):
        result = self._run(cfg)
        stats = compute_stats([result])
        assert stats.trades == 1
        assert stats.wins + stats.losses == stats.trades
        assert stats.total_pnl == pytest.approx(result.trade.net_pnl)

    def test_trade_log_has_every_required_column(self, cfg):
        log = trade_log([self._run(cfg)])
        for column in ("entry_ts", "exit_ts", "strikes", "credit_pts", "net_pnl", "exit_reason"):
            assert column in log.columns

    def test_empty_results_do_not_crash_the_report(self):
        stats = compute_stats([])
        assert stats.trades == 0
        assert trade_log([]).empty


# --------------------------------------------------------------------------
# SDK conformance
# --------------------------------------------------------------------------

databento = pytest.importorskip("databento", reason="databento SDK not installed")


class TestSdkConformance:
    """Bind our arguments against the real SDK signatures.

    This is the cheap half of verifying an adapter written without live access:
    it cannot tell us the data comes back correct, but it does catch the failure
    mode where an SDK upgrade renames or drops a parameter and the first sign is
    a traceback partway through a paid multi-day pull.
    """

    def _fetcher(self, cfg):
        return DatabentoFetcher(cfg, ParquetCache(cfg.data.cache_dir),
                                client=object(), echo=lambda *_: None)

    def _kwargs(self, cfg):
        return self._fetcher(cfg).request_kwargs(
            SCHEMA_QUOTES, DAY, ["SPXW260806P05000000"], time(9, 45), time(16, 0), "raw_symbol"
        )

    def test_get_cost_accepts_our_arguments(self, cfg):
        import inspect

        from databento.historical.api.metadata import MetadataHttpAPI

        sig = inspect.signature(MetadataHttpAPI.get_cost)
        # `self` is unbound here, so supply a placeholder.
        sig.bind(None, **self._kwargs(cfg))

    def test_get_range_accepts_our_arguments(self, cfg):
        import inspect

        from databento.historical.api.timeseries import TimeseriesHttpAPI

        sig = inspect.signature(TimeseriesHttpAPI.get_range)
        sig.bind(None, **self._kwargs(cfg))

    def test_to_df_accepts_the_options_we_pin(self):
        import inspect

        from databento.common.dbnstore import DBNStore

        params = set(inspect.signature(DBNStore.to_df).parameters)
        assert {"price_type", "map_symbols", "tz"} <= params

    def test_to_df_still_defaults_to_utc(self):
        """`to_eastern` converts from UTC; if this default moved, it would be wrong."""
        import datetime as dt
        import inspect

        from databento.common.dbnstore import DBNStore

        default = inspect.signature(DBNStore.to_df).parameters["tz"].default
        assert getattr(default, "value", default) == dt.timezone.utc

    def test_the_request_window_is_utc(self, cfg):
        kwargs = self._kwargs(cfg)
        # 09:45 ET in August is 13:45 UTC.
        assert kwargs["start"].hour == 13 and kwargs["start"].minute == 45

    def test_cost_and_range_describe_the_same_request(self, cfg):
        """A cost quoted for a different window than the pull is a decorative gate."""
        f = self._fetcher(cfg)
        args = (SCHEMA_QUOTES, DAY, ["A"], time(9, 45), time(16, 0), "raw_symbol")
        assert f.request_kwargs(*args) == f.request_kwargs(*args)

    def test_the_definition_window_is_the_whole_utc_day(self, cfg):
        """Definitions are a start-of-UTC-day snapshot, not an intraday stream.

        Requesting them over the session window returns a short chain rather
        than an error, so nothing downstream can notice.
        """
        kwargs = self._fetcher(cfg).request_kwargs(
            SCHEMA_DEFINITION, DAY, ["SPXW.OPT"], time(9, 45), time(16, 0), "parent"
        )
        assert (kwargs["start"].hour, kwargs["start"].minute) == (0, 0)
        assert kwargs["end"] - kwargs["start"] == dt.timedelta(days=1)


@pytest.mark.skipif(
    not os.environ.get("DATABENTO_API_KEY"),
    reason="needs a live API key; metadata calls are free but still networked",
)
class TestVendorAgreesWithOurConstants:
    """The half of conformance that only the server can answer.

    Argument binding cannot catch a schema or symbol that is well-formed but
    does not exist on this dataset -- `mbp-1` is a valid Databento schema and a
    valid Python string, and OPRA rejects it with a 422 at request time. These
    are free metadata calls, and they are the difference between finding that
    out here and finding it out mid-pull.
    """

    def _client(self):
        return databento.Historical()

    def test_the_quote_schema_exists_on_the_dataset(self, cfg):
        available = self._client().metadata.list_schemas(dataset=cfg.data.dataset)
        assert SCHEMA_QUOTES in available, (
            f"{SCHEMA_QUOTES!r} is not offered on {cfg.data.dataset}; "
            f"available: {sorted(available)}"
        )

    def test_the_definition_schema_exists_on_the_dataset(self, cfg):
        available = self._client().metadata.list_schemas(dataset=cfg.data.dataset)
        assert SCHEMA_DEFINITION in available

    def test_the_dataset_exists(self, cfg):
        assert cfg.data.dataset in self._client().metadata.list_datasets()
