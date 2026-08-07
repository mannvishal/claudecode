"""Tests for candidate construction, filtering and the smile fit."""

import math
from datetime import date, datetime

import pytest

from spreadscout.backtest import Smile, _find_strike_by_delta, _vertical_pnl, fit_smile
from spreadscout.models import OptionContract
from spreadscout.pricing import CALL, ET, PUT
from spreadscout.screener import (
    CALL_CREDIT,
    PUT_CREDIT,
    _passes_liquidity,
    build_condors,
    build_verticals,
    is_pm_settled,
)

from .conftest import PUT_SHORT, SPOT, T, make_contract, vol_for


class TestVerticalConstruction:
    def test_put_shorts_are_below_spot(self, chain, cfg):
        for v in build_verticals(chain, PUT_CREDIT, SPOT, cfg):
            assert v.short.strike < SPOT
            assert v.long.strike < v.short.strike

    def test_call_shorts_are_above_spot(self, chain, cfg):
        for v in build_verticals(chain, CALL_CREDIT, SPOT, cfg):
            assert v.short.strike > SPOT
            assert v.long.strike > v.short.strike

    def test_short_delta_stays_inside_the_configured_band(self, chain, cfg):
        for v in build_verticals(chain, PUT_CREDIT, SPOT, cfg):
            assert cfg.filters.min_short_delta <= abs(v.short.delta) <= cfg.filters.max_short_delta

    def test_widths_come_from_config(self, chain, cfg):
        cfg.filters.widths = [10.0]
        widths = {v.width for v in build_verticals(chain, PUT_CREDIT, SPOT, cfg)}
        assert widths <= {10.0}

    def test_min_credit_ratio_rejects_thin_premium(self, chain, cfg):
        cfg.filters.min_credit_ratio = 0.0
        loose = len(build_verticals(chain, PUT_CREDIT, SPOT, cfg))
        cfg.filters.min_credit_ratio = 0.30
        tight = len(build_verticals(chain, PUT_CREDIT, SPOT, cfg))
        assert tight < loose

    def test_produces_candidates_from_a_realistic_chain(self, chain, cfg):
        assert build_verticals(chain, PUT_CREDIT, SPOT, cfg)
        assert build_verticals(chain, CALL_CREDIT, SPOT, cfg)


class TestLiquidityFilters:
    def test_zero_bid_is_untradeable(self, cfg):
        c = make_contract(4700.0, PUT)
        c.bid = 0.0
        assert not c.is_quotable
        assert not _passes_liquidity(c, cfg)

    def test_crossed_market_is_rejected(self, cfg):
        c = make_contract(4900.0, PUT)
        c.bid, c.ask = 5.0, 4.0
        assert not c.is_quotable

    def test_low_open_interest_is_rejected(self, cfg):
        c = make_contract(4900.0, PUT, open_interest=1)
        assert not _passes_liquidity(c, cfg)

    def test_wide_market_is_rejected(self, cfg):
        c = make_contract(4900.0, PUT, half_spread=0.0)
        mid = c.mid
        c.bid, c.ask = mid * 0.5, mid * 1.5
        assert c.relative_spread > cfg.filters.max_relative_spread
        assert not _passes_liquidity(c, cfg)

    def test_relative_spread_is_infinite_at_zero_mid(self):
        c = make_contract(4900.0, PUT)
        c.bid = c.ask = 0.0
        assert c.relative_spread == float("inf")


class TestGreekRecomputation:
    def test_enrich_recovers_the_vol_used_to_quote(self):
        c = make_contract(PUT_SHORT, PUT, half_spread=0.0)
        assert c.iv == pytest.approx(vol_for(PUT_SHORT), abs=1e-4)

    def test_enrich_ignores_the_vendor_greeks(self):
        c = make_contract(PUT_SHORT, PUT)
        c.vendor_delta = 0.99  # nonsense from a stale batch
        c.enrich(SPOT, T, 0.04, 0.013)
        assert abs(c.delta) < 0.5
        assert c.vendor_delta == 0.99  # kept for display, not used

    def test_greeks_are_populated(self):
        c = make_contract(5100.0, CALL)
        assert c.delta is not None and c.gamma is not None
        assert c.theta is not None and c.vega is not None
        assert c.theta < 0  # long option decays


class TestSettlementDetection:
    def test_spxw_is_pm_settled(self):
        assert is_pm_settled([make_contract(5000.0, CALL)])

    def test_monthly_spx_root_is_am_settled(self):
        c = make_contract(5000.0, CALL)
        c.root_symbol = "SPX"
        assert not is_pm_settled([c])

    def test_mixed_roots_default_to_pm(self):
        a, b = make_contract(5000.0, CALL), make_contract(5000.0, PUT)
        a.root_symbol, b.root_symbol = "SPX", "SPXW"
        assert is_pm_settled([a, b])


class TestCondors:
    def test_pairs_are_delta_balanced(self, chain, cfg):
        puts = build_verticals(chain, PUT_CREDIT, SPOT, cfg)
        calls = build_verticals(chain, CALL_CREDIT, SPOT, cfg)
        for condor in build_condors(puts, calls, SPOT, cfg):
            pd = abs(condor.put_spread.short.delta)
            cd = abs(condor.call_spread.short.delta)
            assert abs(pd - cd) <= 0.05

    def test_wings_share_a_width(self, chain, cfg):
        puts = build_verticals(chain, PUT_CREDIT, SPOT, cfg)
        calls = build_verticals(chain, CALL_CREDIT, SPOT, cfg)
        for condor in build_condors(puts, calls, SPOT, cfg):
            assert condor.put_spread.width == condor.call_spread.width

    def test_no_condors_without_both_sides(self, chain, cfg):
        puts = build_verticals(chain, PUT_CREDIT, SPOT, cfg)
        assert build_condors(puts, [], SPOT, cfg) == []
        assert build_condors([], puts, SPOT, cfg) == []


class TestSmileFit:
    def test_recovers_a_downward_slope(self, chain):
        smile = fit_smile(chain, SPOT, T)
        # The synthetic surface has vol falling as strike rises.
        assert smile.b < 0
        assert smile.factor(-0.02) > smile.factor(0.02)

    def test_factor_is_one_at_the_money(self):
        assert Smile(b=-1.5, c=4.0).factor(0.0) == 1.0

    def test_factor_is_floored_positive(self):
        assert Smile(b=-50.0, c=0.0).factor(0.5) > 0

    def test_degenerate_input_returns_flat_smile(self):
        assert fit_smile([], SPOT, T) == Smile()


class TestBacktestMath:
    def test_strike_search_finds_the_target_delta(self):
        strike = _find_strike_by_delta(
            5000.0, 0.15, PUT, T, 0.18, Smile(), 0.04, 0.013, increment=5.0)
        assert strike is not None and strike < 5000.0

    def test_put_spread_max_profit_above_short_strike(self):
        pnl, breached = _vertical_pnl("put_credit", 4900.0, 4890.0, 1.5, 5000.0, 100)
        assert pnl == pytest.approx(150.0)
        assert not breached

    def test_put_spread_max_loss_below_long_strike(self):
        pnl, breached = _vertical_pnl("put_credit", 4900.0, 4890.0, 1.5, 4800.0, 100)
        assert pnl == pytest.approx((1.5 - 10.0) * 100)
        assert breached

    def test_put_spread_partial_loss_between_strikes(self):
        pnl, _ = _vertical_pnl("put_credit", 4900.0, 4890.0, 1.5, 4895.0, 100)
        assert pnl == pytest.approx((1.5 - 5.0) * 100)

    def test_call_spread_is_the_mirror_image(self):
        pnl, breached = _vertical_pnl("call_credit", 5100.0, 5110.0, 1.5, 5000.0, 100)
        assert pnl == pytest.approx(150.0)
        assert not breached
        loss, breached = _vertical_pnl("call_credit", 5100.0, 5110.0, 1.5, 5200.0, 100)
        assert loss == pytest.approx((1.5 - 10.0) * 100)
        assert breached

    def test_loss_is_capped_at_the_width(self):
        """A defined-risk spread cannot lose more than its width, ever."""
        pnl, _ = _vertical_pnl("put_credit", 4900.0, 4890.0, 1.5, 100.0, 100)
        assert pnl == pytest.approx((1.5 - 10.0) * 100)
