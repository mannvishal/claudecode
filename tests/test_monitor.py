"""Tests for OCC parsing, position grouping and risk alerts."""

from datetime import date

import pytest

from spreadscout.alerts import CRITICAL, WARN
from spreadscout.config import Config
from spreadscout.monitor import (
    Leg,
    PositionGroup,
    evaluate_group,
    group_positions,
    parse_occ,
)
from spreadscout.pricing import CALL, PUT

SPOT = 5000.0


def leg(strike, option_type, quantity, cost_basis, bid=1.0, ask=1.2, delta=None):
    occ = parse_occ(
        f"SPXW260807{'C' if option_type == CALL else 'P'}{int(strike * 1000):08d}"
    )
    return Leg(
        symbol=f"SPXW260807{'C' if option_type == CALL else 'P'}{int(strike * 1000):08d}",
        occ=occ,
        quantity=quantity,
        cost_basis=cost_basis,
        bid=bid,
        ask=ask,
        delta=delta,
    )


class TestOccParsing:
    def test_parses_a_real_spxw_symbol(self):
        occ = parse_occ("SPXW260807C07750000")
        assert occ.root == "SPXW"
        assert occ.expiration == date(2026, 8, 7)
        assert occ.option_type == CALL
        assert occ.strike == 7750.0

    def test_parses_a_put(self):
        assert parse_occ("SPXW260807P04900000").option_type == PUT

    def test_parses_fractional_strikes(self):
        assert parse_occ("SPY260807C00612500").strike == 612.5

    def test_returns_none_for_equity_symbols(self):
        assert parse_occ("SPY") is None
        assert parse_occ("BRK.B") is None

    def test_returns_none_for_impossible_dates(self):
        assert parse_occ("SPXW261332C07750000") is None

    def test_is_case_insensitive_and_trims(self):
        assert parse_occ("  spxw260807c07750000 ").strike == 7750.0


class TestGrouping:
    def test_groups_by_underlying_and_expiry(self):
        positions = [
            {"symbol": "SPXW260807P04900000", "quantity": -1, "cost_basis": -500},
            {"symbol": "SPXW260807P04890000", "quantity": 1, "cost_basis": 300},
        ]
        groups = group_positions(positions)
        assert len(groups) == 1 and len(groups[0].legs) == 2

    def test_separates_different_expiries(self):
        positions = [
            {"symbol": "SPXW260807P04900000", "quantity": -1, "cost_basis": -500},
            {"symbol": "SPXW260814P04900000", "quantity": -1, "cost_basis": -500},
        ]
        assert len(group_positions(positions)) == 2

    def test_ignores_equity_positions(self):
        assert group_positions([{"symbol": "SPY", "quantity": 100, "cost_basis": 50000}]) == []

    def test_ignores_zero_quantity_rows(self):
        assert group_positions(
            [{"symbol": "SPXW260807P04900000", "quantity": 0, "cost_basis": 0}]
        ) == []


class TestGroupEconomics:
    @pytest.fixture
    def spread(self):
        # Sold the 4900 put for 5.00, bought the 4890 for 3.00 -> $200 credit.
        return PositionGroup(
            underlying="SPXW",
            expiration=date(2026, 8, 7),
            legs=[
                leg(4900.0, PUT, -1, -500.0, bid=4.90, ask=5.10),
                leg(4890.0, PUT, 1, 300.0, bid=2.90, ask=3.10),
            ],
        )

    def test_credit_received_is_the_net_of_cost_bases(self, spread):
        assert spread.credit_received == 200.0

    def test_close_price_crosses_the_spread_against_you(self, spread):
        short, long = spread.legs
        assert short.close_price == short.ask  # buying back the short
        assert long.close_price == long.bid  # selling out the long

    def test_cost_to_close_uses_the_unfavourable_side(self, spread):
        # Pay 5.10 to close the short, receive 2.90 for the long -> 2.20 -> $220.
        assert spread.cost_to_close == pytest.approx(220.0)

    def test_open_pnl_is_credit_minus_cost_to_close(self, spread):
        assert spread.open_pnl == pytest.approx(-20.0)

    def test_loss_multiple_is_zero_when_profitable(self, spread):
        for leg_ in spread.legs:
            leg_.bid, leg_.ask = 0.10, 0.20
        assert spread.loss_multiple == 0.0

    def test_loss_multiple_scales_with_the_loss(self, spread):
        spread.legs[0].ask = 11.10  # short leg blew out
        assert spread.loss_multiple == pytest.approx((1110 - 290 - 200) / 200)

    def test_loss_multiple_is_none_for_a_debit_structure(self):
        group = PositionGroup("SPXW", date(2026, 8, 7), [leg(4900.0, PUT, 1, 500.0)])
        assert group.loss_multiple is None


class TestAlerts:
    @pytest.fixture
    def cfg(self):
        return Config()

    def group(self, legs):
        return PositionGroup("SPXW", date(2026, 8, 7), legs)

    def test_breached_short_put_is_critical(self, cfg):
        g = self.group([leg(5100.0, PUT, -1, -500.0, delta=-0.80)])
        alerts = evaluate_group(g, SPOT, cfg)  # spot 5000 is below the 5100 short put
        assert any(a.severity == CRITICAL and "BREACHED" in a.title for a in alerts)

    def test_breached_short_call_is_critical(self, cfg):
        g = self.group([leg(4900.0, CALL, -1, -500.0, delta=0.80)])
        alerts = evaluate_group(g, SPOT, cfg)  # spot 5000 is above the 4900 short call
        assert any(a.severity == CRITICAL and "BREACHED" in a.title for a in alerts)

    def test_safe_side_is_not_treated_as_breached(self, cfg):
        g = self.group([leg(4000.0, PUT, -1, -500.0, delta=-0.02)])
        assert not any("BREACHED" in a.title for a in evaluate_group(g, SPOT, cfg))

    def test_high_delta_is_critical(self, cfg):
        g = self.group([leg(4900.0, PUT, -1, -500.0, delta=-0.50)])
        alerts = evaluate_group(g, SPOT, cfg)
        assert any(a.severity == CRITICAL and "delta" in a.title for a in alerts)

    def test_moderate_delta_warns(self, cfg):
        g = self.group([leg(4900.0, PUT, -1, -500.0, delta=-0.35)])
        alerts = evaluate_group(g, SPOT, cfg)
        assert any(a.severity == WARN and "delta" in a.title for a in alerts)

    def test_low_delta_far_strike_is_silent(self, cfg):
        g = self.group([leg(4000.0, PUT, -1, -500.0, delta=-0.02)])
        assert evaluate_group(g, SPOT, cfg) == []

    def test_proximity_warns_even_at_low_delta(self, cfg):
        g = self.group([leg(4990.0, PUT, -1, -500.0, delta=-0.10)])
        alerts = evaluate_group(g, SPOT, cfg)
        assert any("proximity" in a.key for a in alerts)

    def test_long_legs_do_not_generate_risk_alerts(self, cfg):
        g = self.group([leg(5100.0, PUT, 1, 500.0, delta=-0.80)])
        assert evaluate_group(g, SPOT, cfg) == []

    def test_loss_multiple_breach_alerts(self, cfg):
        g = self.group([
            leg(4900.0, PUT, -1, -500.0, bid=10.0, ask=11.0, delta=-0.20),
            leg(4890.0, PUT, 1, 300.0, bid=2.0, ask=2.2, delta=-0.15),
        ])
        alerts = evaluate_group(g, SPOT, cfg)
        assert any("loss is" in a.title for a in alerts)

    def test_alert_keys_are_stable_for_dedup(self, cfg):
        g = self.group([leg(4900.0, PUT, -1, -500.0, delta=-0.50)])
        first = [a.key for a in evaluate_group(g, SPOT, cfg)]
        second = [a.key for a in evaluate_group(g, SPOT, cfg)]
        assert first == second and all(first)
