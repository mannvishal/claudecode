"""End-to-end tests of the watch loop against a fake broker.

These are the tests that matter most for safety. They assert the loop's
*ordering* guarantees -- that positions are checked before entries, that the
daily guard stops new risk without stopping monitoring, and that an unexpected
error does not silently kill a watcher you believe is protecting you.
"""

import math
from datetime import date, datetime, timedelta

import pytest

from spreadscout.alerts import CRITICAL, AlertRouter
from spreadscout.config import Config
from spreadscout.pricing import CALL, ET, PUT, bs_price
from spreadscout.watch import WatchState, run_once

from .conftest import Q, R, vol_for

SPOT = 5000.0
EXPIRY = date(2026, 8, 7)
NOW = datetime(2026, 8, 7, 11, 0, tzinfo=ET)
T = (16 - 11) / (24 * 365)  # 11:00 -> 16:00 ET settlement

# Annualized realized vol the fake history encodes. ATM implied is ~0.18, so the
# measured variance risk premium is comfortably above the 0.15 gate.
CALM_RV = 0.11
STORMY_RV = 0.30


def chain_row(strike: float, option_type: str) -> dict:
    fair = bs_price(SPOT, strike, T, R, Q, vol_for(strike, SPOT), option_type)
    cp = "C" if option_type == CALL else "P"
    return {
        "symbol": f"SPXW260807{cp}{int(strike * 1000):08d}",
        "underlying": "SPX",
        "root_symbol": "SPXW",
        "strike": strike,
        "option_type": option_type,
        "expiration_date": EXPIRY.isoformat(),
        "expiration_type": "weeklys",
        "bid": max(0.0, fair - 0.05),
        "ask": fair + 0.05,
        "bidsize": 25,
        "asksize": 25,
        "volume": 500,
        "open_interest": 5000,
        "contract_size": 100,
        "greeks": {"mid_iv": 0.99, "delta": 0.99, "updated_at": "2026-08-06 20:00:08"},
    }


def daily_bars(n: int, annual_vol: float, start_price: float = SPOT) -> list[dict]:
    """Bars whose open-to-close and close-to-close vol both equal ``annual_vol``."""
    step = annual_vol / math.sqrt(252)
    bars, price = [], start_price
    for i in range(n):
        ret = step if i % 2 == 0 else -step
        open_px = price
        price = price * math.exp(ret)
        bars.append({
            "date": (date(2026, 1, 1) + timedelta(days=i)).isoformat(),
            "open": open_px,
            "close": price,
            "high": max(open_px, price),
            "low": min(open_px, price),
        })
    return bars


class FakeClient:
    """A read-only Tradier stand-in. Records calls so ordering can be asserted."""

    def __init__(self, *, realized_vol=CALM_RV, positions=None, equity=100_000.0,
                 gainloss=None, market_state="open", vix_percentile_high=True):
        self.realized_vol = realized_vol
        self._positions = positions or []
        self._equity = equity
        self._gainloss = gainloss or []
        self.market_state = market_state
        self.vix_high = vix_percentile_high
        self.calls: list[str] = []
        self.fail_next_positions = False

    # --- market data ---
    def spot(self, symbol):
        self.calls.append("spot")
        return SPOT

    def quotes(self, symbols):
        self.calls.append("quotes")
        out = {}
        for sym in symbols:
            out[sym] = {"symbol": sym, "bid": 1.0, "ask": 1.2}
        return out

    def expirations(self, symbol):
        return [EXPIRY]

    def chain(self, symbol, expiration):
        self.calls.append("chain")
        rows = []
        for strike in range(4850, 5151, 5):
            rows.append(chain_row(float(strike), PUT))
            rows.append(chain_row(float(strike), CALL))
        return rows

    def clock(self):
        return {"state": self.market_state, "description": "test"}

    def history(self, symbol, start, end, interval="daily"):
        self.calls.append(f"history:{symbol}")
        if symbol == "VIX":
            level = 20.0 if self.vix_high else 10.0
            # A rising series ending at `level` puts today near the top decile;
            # a flat-high series with a low last value puts it near the bottom.
            closes = [10.0 + i * 0.02 for i in range(300)] if self.vix_high else \
                     [30.0 - i * 0.05 for i in range(300)]
            return [{"date": f"d{i}", "close": c} for i, c in enumerate(closes)]
        if start == end == EXPIRY:
            return [{"date": EXPIRY.isoformat(), "open": SPOT, "close": SPOT}]
        return daily_bars(60, self.realized_vol)

    # --- account (read-only) ---
    def account_ids(self):
        return ["TEST123"]

    def equity(self, account_id):
        self.calls.append("equity")
        return self._equity

    def positions(self, account_id):
        self.calls.append("positions")
        if self.fail_next_positions:
            from spreadscout.tradier import TradierError
            raise TradierError("boom")
        return self._positions

    def gainloss(self, account_id, start, end):
        return self._gainloss


class Recorder:
    def __init__(self):
        self.sent = []

    def send(self, alert):
        self.sent.append(alert)


@pytest.fixture
def rec():
    return Recorder()


@pytest.fixture
def router(rec):
    return AlertRouter([rec], cooldown_minutes=0.0)


@pytest.fixture
def state():
    return WatchState(day=NOW.date())


@pytest.fixture
def cfg():
    c = Config()
    c.risk_free_rate, c.dividend_yield = R, Q
    return c


def kinds(rec):
    return [a.kind for a in rec.sent]


def titles(rec):
    return " | ".join(a.title for a in rec.sent)


class TestEntryGating:
    def test_calm_realized_vol_produces_an_entry_alert(self, cfg, router, rec, state):
        client = FakeClient(realized_vol=CALM_RV)
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert "entry" in kinds(rec), titles(rec)

    def test_realized_vol_above_implied_blocks_entry(self, cfg, router, rec, state):
        client = FakeClient(realized_vol=STORMY_RV)
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert "entry" not in kinds(rec)

    def test_closed_market_blocks_entry(self, cfg, router, rec, state):
        client = FakeClient(market_state="closed")
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert "entry" not in kinds(rec)

    def test_outside_the_time_window_blocks_entry(self, cfg, router, rec, state):
        client = FakeClient()
        early = datetime(2026, 8, 7, 9, 35, tzinfo=ET)
        run_once(client, cfg, router, state, "TEST123", None, now=early)
        assert "entry" not in kinds(rec)

    def test_measured_multiplier_is_applied_to_config(self, cfg, router, rec, state):
        client = FakeClient(realized_vol=CALM_RV)
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        # Started at the 1.0 default; measurement must have moved it below 1.
        assert cfg.beliefs.vol_multiplier < 1.0

    def test_entry_alert_names_its_vol_assumption(self, cfg, router, rec, state):
        client = FakeClient(realized_vol=CALM_RV)
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        entries = [a for a in rec.sent if a.kind == "entry"]
        assert entries and any("vol assumption" in line for line in entries[0].lines)

    def test_entry_alert_carries_a_full_ticket(self, cfg, router, rec, state):
        client = FakeClient(realized_vol=CALM_RV)
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        entry = next(a for a in rec.sent if a.kind == "entry")
        body = "\n".join(entry.lines)
        assert "SELL" in body and "BUY" in body and "multileg credit" in body

    def test_only_one_entry_alert_per_pass(self, cfg, router, rec, state):
        client = FakeClient(realized_vol=CALM_RV)
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert kinds(rec).count("entry") == 1


class TestPositionMonitoring:
    BREACHED = [
        {"symbol": "SPXW260807P05100000", "quantity": -1, "cost_basis": -500.0},
        {"symbol": "SPXW260807P05090000", "quantity": 1, "cost_basis": 300.0},
    ]

    def test_breached_short_strike_alerts_critical(self, cfg, router, rec, state):
        client = FakeClient(positions=self.BREACHED)
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert any(a.severity == CRITICAL for a in rec.sent), titles(rec)

    def test_positions_are_checked_before_the_chain_is_fetched(self, cfg, router, rec, state):
        """A breach warning must never queue behind a slow chain request."""
        client = FakeClient(positions=self.BREACHED)
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert client.calls.index("positions") < client.calls.index("chain")

    def test_position_failure_does_not_stop_the_pass(self, cfg, router, rec, state):
        client = FakeClient(realized_vol=CALM_RV)
        client.fail_next_positions = True
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert "entry" in kinds(rec)  # the rest of the pass still ran

    def test_no_positions_produces_no_risk_alerts(self, cfg, router, rec, state):
        client = FakeClient(realized_vol=CALM_RV, positions=[])
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert "risk" not in kinds(rec)


class TestDailyGuard:
    def losing_day(self, equity=100_000.0):
        return [{"close_date": f"{EXPIRY.isoformat()}T20:00:00Z", "gain_loss": -4_000.0}]

    def test_breach_stops_new_entries(self, cfg, router, rec, state):
        client = FakeClient(realized_vol=CALM_RV, gainloss=self.losing_day())
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert "entry" not in kinds(rec)

    def test_breach_raises_a_critical_guard_alert(self, cfg, router, rec, state):
        client = FakeClient(realized_vol=CALM_RV, gainloss=self.losing_day())
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert any(a.kind == "guard" and a.severity == CRITICAL for a in rec.sent)

    def test_guard_alert_fires_once_not_every_poll(self, cfg, router, rec, state):
        client = FakeClient(realized_vol=CALM_RV, gainloss=self.losing_day())
        for _ in range(3):
            run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert kinds(rec).count("guard") == 1

    def test_positions_are_still_monitored_after_the_guard_trips(self, cfg, router, rec, state):
        client = FakeClient(
            realized_vol=CALM_RV,
            gainloss=self.losing_day(),
            positions=TestPositionMonitoring.BREACHED,
        )
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert any(a.kind == "risk" for a in rec.sent)

    def test_approaching_the_limit_warns_without_blocking(self, cfg, router, rec, state):
        near = [{"close_date": f"{EXPIRY.isoformat()}T20:00:00Z", "gain_loss": -2_500.0}]
        client = FakeClient(realized_vol=CALM_RV, gainloss=near)
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert any(a.kind == "guard" for a in rec.sent)
        assert "entry" in kinds(rec)


class TestStateRollover:
    def test_new_day_resets_the_guard(self, cfg, state):
        state.guard_tripped = True
        state.entries_alerted = 4
        state.roll_if_new_day(date(2026, 8, 10))
        assert not state.guard_tripped and state.entries_alerted == 0

    def test_same_day_preserves_state(self, cfg, state):
        state.entries_alerted = 4
        state.roll_if_new_day(state.day)
        assert state.entries_alerted == 4


class TestCriticalPositionBlocksEntry:
    """Being run over is not the moment to add risk."""

    def test_breached_position_suppresses_new_entries(self, cfg, router, rec, state):
        client = FakeClient(
            realized_vol=CALM_RV, positions=TestPositionMonitoring.BREACHED
        )
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert "entry" not in kinds(rec), titles(rec)

    def test_it_says_why_rather_than_going_quiet(self, cfg, router, rec, state):
        client = FakeClient(
            realized_vol=CALM_RV, positions=TestPositionMonitoring.BREACHED
        )
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert any("suppressed" in a.title for a in rec.sent)

    def test_the_breach_alert_still_fires(self, cfg, router, rec, state):
        client = FakeClient(
            realized_vol=CALM_RV, positions=TestPositionMonitoring.BREACHED
        )
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert any("BREACHED" in a.title for a in rec.sent)

    def test_can_be_disabled_deliberately(self, cfg, router, rec, state):
        cfg.monitor.block_entry_on_critical = False
        client = FakeClient(
            realized_vol=CALM_RV, positions=TestPositionMonitoring.BREACHED
        )
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert "entry" in kinds(rec)

    def test_a_mere_warning_does_not_block(self, cfg, router, rec, state):
        """WARN-level proximity should not stop the whole entry pipeline."""
        near = [
            {"symbol": "SPXW260807P04990000", "quantity": -1, "cost_basis": -500.0},
            {"symbol": "SPXW260807P04980000", "quantity": 1, "cost_basis": 300.0},
        ]
        client = FakeClient(realized_vol=CALM_RV, positions=near)
        run_once(client, cfg, router, state, "TEST123", None, now=NOW)
        assert "entry" in kinds(rec), titles(rec)
