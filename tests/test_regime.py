"""Tests for volatility measurement and the entry gates."""

import math
from datetime import datetime

import pytest

from spreadscout.config import Config
from spreadscout.pricing import ET
from spreadscout.regime import (
    Regime,
    _log_returns_close_to_close,
    _log_returns_open_to_close,
    annualized_vol,
    atm_implied_vol,
    check_gates,
    resolve_vol_multiplier,
)

from .conftest import SPOT, make_contract
from spreadscout.pricing import CALL, PUT

IN_WINDOW = datetime(2026, 8, 7, 11, 0, tzinfo=ET)


def bars(closes, opens=None):
    opens = opens or closes
    return [
        {"date": f"2026-01-{i + 1:02d}", "open": o, "close": c}
        for i, (o, c) in enumerate(zip(opens, closes))
    ]


def regime(**kw):
    base = dict(
        spot=5000.0,
        atm_iv=0.20,
        rv_close_to_close=0.16,
        rv_open_to_close=0.14,
        vix=16.0,
        vix_percentile=0.50,
        todays_move_sigma=0.30,
        lookback_days=60,
    )
    base.update(kw)
    return Regime(**base)


class TestVolEstimation:
    def test_close_to_close_uses_consecutive_closes(self):
        rets = _log_returns_close_to_close(bars([100, 110, 121]))
        assert len(rets) == 2
        assert rets[0] == pytest.approx(math.log(1.1))

    def test_open_to_close_uses_the_same_session(self):
        rets = _log_returns_open_to_close(bars(closes=[110, 121], opens=[100, 110]))
        assert all(r == pytest.approx(math.log(1.1)) for r in rets)

    def test_zero_prices_are_skipped_not_crashed_on(self):
        rows = [{"date": "d", "open": 0, "close": 0}, {"date": "e", "open": 10, "close": 11}]
        assert len(_log_returns_open_to_close(rows)) == 1

    def test_annualization_uses_252_trading_days(self):
        daily = 0.01
        vol = annualized_vol([daily, -daily] * 20)
        assert vol == pytest.approx(daily * math.sqrt(252), rel=1e-9)

    def test_too_few_observations_returns_none(self):
        assert annualized_vol([0.01, -0.01]) is None

    def test_atm_iv_averages_the_straddle(self):
        contracts = [make_contract(5000.0, CALL), make_contract(5000.0, PUT)]
        iv = atm_implied_vol(contracts, SPOT)
        expected = (contracts[0].iv + contracts[1].iv) / 2
        assert iv == pytest.approx(expected)

    def test_atm_iv_is_none_without_solvable_contracts(self):
        assert atm_implied_vol([], SPOT) is None


class TestVariancePremium:
    def test_conservative_uses_the_larger_realized_estimate(self):
        r = regime(rv_open_to_close=0.14, rv_close_to_close=0.16)
        assert r.conservative_realized == 0.16
        assert r.realized_for_horizon == 0.14

    def test_conservative_reading_reports_the_smaller_premium(self):
        r = regime()
        assert r.vrp(conservative=True) < r.vrp(conservative=False)

    def test_vrp_is_positive_when_implied_exceeds_realized(self):
        assert regime(atm_iv=0.24, rv_close_to_close=0.16, rv_open_to_close=0.16).vrp() > 0

    def test_vrp_is_negative_when_realized_exceeds_implied(self):
        assert regime(atm_iv=0.14, rv_close_to_close=0.20, rv_open_to_close=0.20).vrp() < 0

    def test_vrp_is_none_without_realized_data(self):
        assert regime(rv_close_to_close=None, rv_open_to_close=None).vrp() is None

    def test_multiplier_is_realized_over_implied(self):
        r = regime(atm_iv=0.20, rv_close_to_close=0.16, rv_open_to_close=0.16)
        assert r.implied_vol_multiplier() == pytest.approx(0.80)


class TestVolMultiplierResolution:
    def test_measured_source_applies_the_haircut(self):
        cfg = Config()
        cfg.beliefs.source = "measured"
        cfg.beliefs.haircut = 0.05
        r = regime(atm_iv=0.20, rv_close_to_close=0.16, rv_open_to_close=0.16)
        value, note = resolve_vol_multiplier(r, cfg)
        assert value == pytest.approx(0.85)
        assert "measured" in note

    def test_haircut_never_pushes_above_one(self):
        """Above 1.0 would be a reason to buy premium, not sell it."""
        cfg = Config()
        cfg.beliefs.haircut = 0.50
        r = regime(atm_iv=0.20, rv_close_to_close=0.19, rv_open_to_close=0.19)
        value, _ = resolve_vol_multiplier(r, cfg)
        assert value == 1.0

    def test_manual_source_is_left_alone(self):
        cfg = Config()
        cfg.beliefs.source = "manual"
        cfg.beliefs.vol_multiplier = 0.9
        value, note = resolve_vol_multiplier(regime(), cfg)
        assert value == 0.9 and "manual" in note

    def test_falls_back_to_manual_without_a_regime(self):
        cfg = Config()
        value, note = resolve_vol_multiplier(None, cfg)
        assert value == cfg.beliefs.vol_multiplier and "manual" in note

    def test_falls_back_when_realized_is_unmeasurable(self):
        cfg = Config()
        r = regime(rv_close_to_close=None, rv_open_to_close=None)
        _value, note = resolve_vol_multiplier(r, cfg)
        assert "could not measure" in note


class TestGates:
    @pytest.fixture
    def cfg(self):
        c = Config()
        c.gates.min_variance_risk_premium = 0.15
        return c

    def test_rich_premium_in_window_passes(self, cfg):
        r = regime(atm_iv=0.30, rv_close_to_close=0.20, rv_open_to_close=0.18)
        assert check_gates(r, cfg, now=IN_WINDOW).passed

    def test_thin_premium_blocks(self, cfg):
        r = regime(atm_iv=0.21, rv_close_to_close=0.20, rv_open_to_close=0.20)
        result = check_gates(r, cfg, now=IN_WINDOW)
        assert not result.passed
        assert any("variance risk premium" in b for b in result.blocks)

    def test_realized_above_implied_blocks(self, cfg):
        r = regime(atm_iv=0.15, rv_close_to_close=0.25, rv_open_to_close=0.25)
        assert not check_gates(r, cfg, now=IN_WINDOW).passed

    def test_vol_at_the_floor_blocks(self, cfg):
        r = regime(atm_iv=0.30, rv_close_to_close=0.20, rv_open_to_close=0.18,
                   vix_percentile=0.05)
        result = check_gates(r, cfg, now=IN_WINDOW)
        assert not result.passed
        assert any("percentile" in b for b in result.blocks)

    def test_trend_day_blocks(self, cfg):
        r = regime(atm_iv=0.30, rv_close_to_close=0.20, rv_open_to_close=0.18,
                   todays_move_sigma=-2.0)
        result = check_gates(r, cfg, now=IN_WINDOW)
        assert not result.passed
        assert any("sigma today" in b for b in result.blocks)

    def test_direction_of_the_move_does_not_matter(self, cfg):
        for move in (2.0, -2.0):
            r = regime(atm_iv=0.30, rv_close_to_close=0.20, rv_open_to_close=0.18,
                       todays_move_sigma=move)
            assert not check_gates(r, cfg, now=IN_WINDOW).passed

    def test_too_early_blocks(self, cfg):
        r = regime(atm_iv=0.30, rv_close_to_close=0.20, rv_open_to_close=0.18)
        early = datetime(2026, 8, 7, 9, 35, tzinfo=ET)
        assert any("before" in b for b in check_gates(r, cfg, now=early).blocks)

    def test_too_late_blocks(self, cfg):
        r = regime(atm_iv=0.30, rv_close_to_close=0.20, rv_open_to_close=0.18)
        late = datetime(2026, 8, 7, 15, 45, tzinfo=ET)
        assert any("cutoff" in b for b in check_gates(r, cfg, now=late).blocks)

    def test_blackout_date_blocks(self, cfg):
        cfg.gates.blackout_dates = ["2026-08-07"]
        r = regime(atm_iv=0.30, rv_close_to_close=0.20, rv_open_to_close=0.18)
        assert any("blackout" in b for b in check_gates(r, cfg, now=IN_WINDOW).blocks)

    def test_missing_regime_blocks(self, cfg):
        assert not check_gates(None, cfg, now=IN_WINDOW).passed

    def test_every_failing_gate_is_named(self, cfg):
        cfg.gates.blackout_dates = ["2026-08-07"]
        r = regime(atm_iv=0.15, rv_close_to_close=0.25, rv_open_to_close=0.25,
                   vix_percentile=0.01, todays_move_sigma=3.0)
        result = check_gates(r, cfg, now=IN_WINDOW)
        assert len(result.blocks) == 4  # premium, percentile, trend, blackout

    def test_default_posture_is_no_entry(self):
        """With nothing measurable, the answer must be no."""
        assert not check_gates(None, Config(), now=IN_WINDOW).passed
