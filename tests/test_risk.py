"""Tests for the expectancy identity and the sizing guard rails."""

import pytest

from spreadscout.models import IronCondor
from spreadscout.pricing import CALL, PUT, Measure
from spreadscout.risk import (
    daily_loss_breached,
    evaluate,
    expected_payoff,
    kelly_fraction,
    open_risk_from_positions,
    position_expected_payoff,
    realized_pnl_today,
    round_trip_costs,
    size_position,
)

from .conftest import Q, R, SPOT, T, make_contract


def test_expected_payoff_matches_risk_neutral_price():
    """E_Q[payoff] compounded back must equal the market price."""
    from spreadscout.pricing import bs_price

    for strike, kind in [(4900.0, PUT), (5100.0, CALL)]:
        price = bs_price(SPOT, strike, T, R, Q, 0.2, kind)
        expected = expected_payoff(SPOT, strike, T, R - Q, 0.2, kind)
        import math

        assert expected == pytest.approx(price * math.exp(R * T), rel=1e-9)


class TestExpectancyIdentity:
    """The central claim of the tool, stated as executable arithmetic."""

    def test_risk_neutral_ev_is_zero_before_costs(self, cfg, put_spread):
        ev = evaluate(put_spread, SPOT, T, cfg)
        assert ev is not None
        # Mid fill, no commissions, market repriced by its own implied vols.
        assert ev.ev_risk_neutral == pytest.approx(0.0, abs=1e-6)

    def test_holds_for_call_spreads_too(self, cfg, call_spread):
        ev = evaluate(call_spread, SPOT, T, cfg)
        assert ev.ev_risk_neutral == pytest.approx(0.0, abs=1e-6)

    def test_holds_for_iron_condors(self, cfg, put_spread, call_spread):
        condor = IronCondor(put_spread=put_spread, call_spread=call_spread)
        ev = evaluate(condor, SPOT, T, cfg)
        assert ev.ev_risk_neutral == pytest.approx(0.0, abs=1e-6)

    def test_costs_make_it_strictly_negative(self, cfg, put_spread):
        cfg.costs.commission_per_contract = 0.35
        cfg.costs.exchange_fee_per_contract = 0.49
        ev = evaluate(put_spread, SPOT, T, cfg)
        assert ev.ev_after_costs < 0
        assert ev.costs == pytest.approx(0.84 * 2 * 2)  # 2 legs, round trip

    def test_crossing_the_spread_makes_it_worse(self, cfg, put_spread):
        cfg.costs.fill_fraction = 0.0
        at_mid = evaluate(put_spread, SPOT, T, cfg).ev_risk_neutral
        cfg.costs.fill_fraction = 1.0
        at_natural = evaluate(put_spread, SPOT, T, cfg).ev_risk_neutral
        assert at_natural < at_mid

    def test_lower_vol_multiplier_is_the_only_thing_creating_edge(self, cfg, put_spread):
        neutral = evaluate(put_spread, SPOT, T, cfg).ev_believed
        cfg.beliefs.vol_multiplier = 0.85
        believing = evaluate(put_spread, SPOT, T, cfg).ev_believed
        assert neutral == pytest.approx(0.0, abs=1e-6)
        assert believing > 0

    def test_higher_vol_multiplier_makes_selling_worse(self, cfg, put_spread):
        cfg.beliefs.vol_multiplier = 1.20
        assert evaluate(put_spread, SPOT, T, cfg).ev_believed < 0


class TestProbabilities:
    def test_pop_is_high_which_is_exactly_the_trap(self, cfg, put_spread):
        ev = evaluate(put_spread, SPOT, T, cfg)
        # A 0.05-0.25 delta short leg wins most of the time...
        assert ev.pop_risk_neutral > 0.70
        # ...and still has zero expectancy.
        assert ev.ev_risk_neutral == pytest.approx(0.0, abs=1e-6)

    def test_condor_pop_is_lower_than_either_wing(self, cfg, put_spread, call_spread):
        condor = IronCondor(put_spread=put_spread, call_spread=call_spread)
        c = evaluate(condor, SPOT, T, cfg)
        p = evaluate(put_spread, SPOT, T, cfg)
        assert c.pop_risk_neutral < p.pop_risk_neutral

    def test_lower_vol_belief_raises_pop(self, cfg, put_spread):
        base = evaluate(put_spread, SPOT, T, cfg).pop_believed
        cfg.beliefs.vol_multiplier = 0.80
        assert evaluate(put_spread, SPOT, T, cfg).pop_believed > base


class TestStructure:
    def test_rejects_credit_exceeding_width(self, cfg, put_spread):
        # A stale quote that implies free money must be dropped, not traded.
        put_spread.short.bid = put_spread.short.ask = 500.0
        put_spread.long.bid = put_spread.long.ask = 0.05
        assert evaluate(put_spread, SPOT, T, cfg) is None

    def test_rejects_debit(self, cfg, put_spread):
        put_spread.short.bid = put_spread.short.ask = 0.10
        put_spread.long.bid = put_spread.long.ask = 5.00
        assert evaluate(put_spread, SPOT, T, cfg) is None

    def test_condor_margin_is_the_wider_wing_not_the_sum(self, put_spread, call_spread):
        condor = IronCondor(put_spread=put_spread, call_spread=call_spread)
        assert condor.width == max(put_spread.width, call_spread.width)
        assert condor.width < put_spread.width + call_spread.width

    def test_max_loss_plus_credit_equals_width(self, put_spread):
        credit = put_spread.credit_at(0.4)
        assert put_spread.max_loss(credit) + credit == pytest.approx(put_spread.width)

    def test_costs_scale_with_leg_count(self, cfg, put_spread, call_spread):
        cfg.costs.commission_per_contract = 0.35
        cfg.costs.exchange_fee_per_contract = 0.49
        condor = IronCondor(put_spread=put_spread, call_spread=call_spread)
        assert round_trip_costs(condor, cfg) == 2 * round_trip_costs(put_spread, cfg)

    def test_one_way_costs_when_held_to_settlement(self, cfg, put_spread):
        cfg.costs.commission_per_contract = 0.35
        cfg.costs.exchange_fee_per_contract = 0.49
        cfg.costs.assume_closing_trade = False
        assert round_trip_costs(put_spread, cfg) == pytest.approx(0.84 * 2)


class TestSizing:
    @pytest.fixture
    def ev(self, cfg, put_spread):
        cfg.beliefs.vol_multiplier = 0.80  # give it positive EV so sizing runs
        return evaluate(put_spread, SPOT, T, cfg)

    def test_respects_per_trade_risk_cap(self, ev, cfg):
        cfg.risk.max_contracts = 1000
        result = size_position(ev, equity=100_000, open_risk=0, cfg=cfg)
        assert result.total_risk <= 100_000 * cfg.risk.max_risk_pct_per_trade

    def test_respects_portfolio_cap(self, ev, cfg):
        cfg.risk.max_contracts = 1000
        exhausted = 100_000 * cfg.risk.max_total_risk_pct
        result = size_position(ev, equity=100_000, open_risk=exhausted, cfg=cfg)
        assert result.contracts == 0
        assert "max_total_risk_pct" in result.reasons[0]

    def test_respects_absolute_contract_cap(self, ev, cfg):
        cfg.risk.max_contracts = 2
        result = size_position(ev, equity=10_000_000, open_risk=0, cfg=cfg)
        assert result.contracts == 2

    def test_refuses_below_min_equity(self, ev, cfg):
        result = size_position(ev, equity=500, open_risk=0, cfg=cfg)
        assert result.contracts == 0
        assert "min_equity" in result.reasons[0]

    def test_refuses_negative_expectancy(self, cfg, put_spread):
        cfg.beliefs.vol_multiplier = 1.0
        cfg.costs.commission_per_contract = 0.35
        negative = evaluate(put_spread, SPOT, T, cfg)
        result = size_position(negative, equity=100_000, open_risk=0, cfg=cfg)
        assert result.contracts == 0
        assert "loses money on average" in result.reasons[0]

    def test_can_be_overridden_but_must_be_deliberate(self, cfg, put_spread):
        cfg.beliefs.vol_multiplier = 1.0
        cfg.costs.commission_per_contract = 0.35
        negative = evaluate(put_spread, SPOT, T, cfg)
        cfg.risk.require_positive_ev = False
        assert size_position(negative, equity=100_000, open_risk=0, cfg=cfg).contracts > 0

    def test_reports_the_binding_constraint(self, ev, cfg):
        cfg.risk.max_contracts = 1
        result = size_position(ev, equity=10_000_000, open_risk=0, cfg=cfg)
        assert "max_contracts" in result.reasons[0]


class TestGuards:
    def test_kelly_is_zero_when_edge_is_absent(self):
        # 90% win rate, but you lose 9x what you win: exactly break-even.
        assert kelly_fraction(0.90, credit=1.0, max_loss=9.0) == pytest.approx(0.0, abs=1e-9)

    def test_kelly_is_zero_when_edge_is_negative(self):
        assert kelly_fraction(0.85, credit=1.0, max_loss=9.0) == 0.0

    def test_kelly_collapses_on_small_winrate_errors(self):
        """Three points of optimism is the difference between a bet and a bust."""
        optimistic = kelly_fraction(0.93, credit=1.0, max_loss=9.0)
        realistic = kelly_fraction(0.90, credit=1.0, max_loss=9.0)
        assert optimistic > 0 and realistic == pytest.approx(0.0, abs=1e-9)

    def test_open_risk_counts_shorts_only(self):
        positions = [
            {"quantity": -2, "cost_basis": -500.0},
            {"quantity": 2, "cost_basis": 200.0},
        ]
        assert open_risk_from_positions(positions) == 500.0

    def test_realized_pnl_filters_to_today(self):
        rows = [
            {"close_date": "2026-08-07T20:00:00.000Z", "gain_loss": -1200.0},
            {"close_date": "2026-08-06T20:00:00.000Z", "gain_loss": 900.0},
        ]
        import datetime as dt

        assert realized_pnl_today(rows, dt.date(2026, 8, 7)) == -1200.0

    def test_daily_loss_limit_triggers(self, cfg):
        assert daily_loss_breached(-3_100, equity=100_000, cfg=cfg)
        assert not daily_loss_breached(-2_900, equity=100_000, cfg=cfg)
