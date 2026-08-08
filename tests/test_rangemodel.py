"""Tests for the underlying-bar range model.

The properties worth asserting here are the ones whose violation produces a
*confident wrong number* rather than a crash: a state that saw the future, a
quantile curve that crosses itself, or a coverage figure that looks calibrated
because the split leaked. A model that is merely inaccurate is a research
result; a model that is accidentally clairvoyant is a bug that will look like
alpha right up until it is traded.
"""

from __future__ import annotations

import math
from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from backtest.rangemodel import (
    DEFAULT_QUANTILES,
    RangeModel,
    VarianceProfile,
    calibration,
    observe,
    split_sessions,
    state_at,
)
from backtest.underlying import (
    RTH_CLOSE,
    RTH_OPEN,
    month_starts,
    normalize_bars,
    regular_hours,
    sessions,
)

ET = "America/New_York"
BARS_PER_SESSION = 390


def synthetic_session(
    day: date, seed: int, sigma_annual: float = 0.16, drift: float = 0.0,
    open_price: float = 5000.0, n: int = BARS_PER_SESSION,
) -> pd.DataFrame:
    """One GBM session on a 1-minute grid, shaped like a real bar frame."""
    rng = np.random.default_rng(seed)
    per_min = sigma_annual * math.sqrt(1.0 / (252 * BARS_PER_SESSION))
    steps = rng.normal(drift / n, per_min, size=n)
    closes = open_price * np.exp(np.cumsum(steps))

    start = pd.Timestamp(f"{day} 09:30", tz=ET)
    ts = pd.date_range(start, periods=n, freq="1min", tz=ET)
    wiggle = np.abs(rng.normal(0, per_min * 0.3, size=n)) * closes
    return pd.DataFrame({
        "ts": ts,
        "open": closes,
        "high": closes + wiggle,
        "low": closes - wiggle,
        "close": closes,
        "volume": 1000,
    })


def synthetic_sessions(n_days: int, seed: int = 0, **kwargs) -> dict[date, pd.DataFrame]:
    out = {}
    day = date(2024, 1, 1)
    for i in range(n_days):
        while day.weekday() >= 5:
            day = date.fromordinal(day.toordinal() + 1)
        out[day] = synthetic_session(day, seed=seed + i, **kwargs)
        day = date.fromordinal(day.toordinal() + 1)
    return out


@pytest.fixture(scope="module")
def many_sessions():
    return synthetic_sessions(400, seed=1234)


# --------------------------------------------------------------------------
# No lookahead. The property everything else depends on.
# --------------------------------------------------------------------------

class TestNoLookahead:
    def test_state_ignores_every_future_bar(self):
        frame = synthetic_session(date(2024, 3, 1), seed=7)
        profile = VarianceProfile.fit({date(2024, 3, 1): frame})

        before, _ = observe(frame, 120, profile)

        tampered = frame.copy()
        tampered.loc[121:, "close"] *= 1.25
        tampered.loc[121:, "high"] *= 1.25
        tampered.loc[121:, "low"] *= 1.25
        after, _ = observe(tampered, 120, profile)

        assert before == after

    def test_outcome_does_change_when_the_future_changes(self):
        """The mirror of the above: proof the tamper was actually visible."""
        frame = synthetic_session(date(2024, 3, 1), seed=7)
        profile = VarianceProfile.fit({date(2024, 3, 1): frame})
        _, before = observe(frame, 120, profile)

        tampered = frame.copy()
        tampered.loc[121:, "close"] *= 1.25
        _, after = observe(tampered, 120, profile)

        assert before.close_return != after.close_return

    def test_high_and_low_so_far_exclude_the_future(self):
        frame = synthetic_session(date(2024, 3, 1), seed=11)
        profile = VarianceProfile.fit({date(2024, 3, 1): frame})
        state, _ = observe(frame, 100, profile)

        assert state.high_so_far == frame["high"].iloc[:101].max()
        assert state.low_so_far == frame["low"].iloc[:101].min()

    def test_observe_refuses_the_last_bar(self):
        frame = synthetic_session(date(2024, 3, 1), seed=3)
        profile = VarianceProfile.fit({date(2024, 3, 1): frame})
        assert observe(frame, len(frame) - 1, profile) is None


# --------------------------------------------------------------------------
# Variance profile
# --------------------------------------------------------------------------

class TestVarianceProfile:
    def test_shares_are_non_decreasing(self, many_sessions):
        profile = VarianceProfile.fit(many_sessions)
        assert np.all(np.diff(profile.shares) >= -1e-12)

    def test_shares_end_at_one(self, many_sessions):
        profile = VarianceProfile.fit(many_sessions)
        assert profile.shares[-1] == pytest.approx(1.0, abs=1e-9)

    def test_constant_vol_gives_a_roughly_linear_profile(self, many_sessions):
        """GBM has no intraday seasonality, so the share should track the clock."""
        profile = VarianceProfile.fit(many_sessions)
        midpoint = profile.share_by(len(profile.shares) // 2)
        assert 0.40 < midpoint < 0.60

    def test_empty_input_is_an_error_not_a_silent_default(self):
        with pytest.raises(ValueError):
            VarianceProfile.fit({})


# --------------------------------------------------------------------------
# Model shape
# --------------------------------------------------------------------------

class TestModelShape:
    def test_quantiles_are_monotonic(self, many_sessions):
        profile = VarianceProfile.fit(many_sessions)
        model = RangeModel.fit(many_sessions, profile, stride=30)
        levels = [model.close_quantile(a) for a in sorted(DEFAULT_QUANTILES)]
        assert levels == sorted(levels)

    def test_put_strike_sits_below_spot_and_call_above(self, many_sessions):
        profile = VarianceProfile.fit(many_sessions)
        model = RangeModel.fit(many_sessions, profile, stride=30)
        frame = next(iter(many_sessions.values()))
        state, _ = observe(frame, 60, profile)

        assert model.short_strike(state, "put", 0.95) < state.price
        assert model.short_strike(state, "call", 0.95) > state.price

    def test_higher_confidence_moves_the_strike_further_away(self, many_sessions):
        profile = VarianceProfile.fit(many_sessions)
        model = RangeModel.fit(many_sessions, profile, stride=30)
        frame = next(iter(many_sessions.values()))
        state, _ = observe(frame, 60, profile)

        assert (model.short_strike(state, "put", 0.99)
                < model.short_strike(state, "put", 0.80))
        assert (model.short_strike(state, "call", 0.99)
                > model.short_strike(state, "call", 0.80))

    def test_an_unknown_side_is_rejected(self, many_sessions):
        profile = VarianceProfile.fit(many_sessions)
        model = RangeModel.fit(many_sessions, profile, stride=30)
        frame = next(iter(many_sessions.values()))
        state, _ = observe(frame, 60, profile)
        with pytest.raises(ValueError, match="put.*call"):
            model.short_strike(state, "sideways", 0.95)

    def test_the_band_narrows_as_the_close_approaches(self, many_sessions):
        """Less time left, less room to travel. If this inverted, the scale
        correction would be backwards."""
        profile = VarianceProfile.fit(many_sessions)
        model = RangeModel.fit(many_sessions, profile, stride=30)
        frame = next(iter(many_sessions.values()))

        early, _ = observe(frame, 30, profile)
        late, _ = observe(frame, 360, profile)
        early_lo, early_hi = model.band(early, 0.10)
        late_lo, late_hi = model.band(late, 0.10)

        assert (early_hi - early_lo) / early.price > (late_hi - late_lo) / late.price


# --------------------------------------------------------------------------
# The claim the model actually makes
# --------------------------------------------------------------------------

class TestCalibration:
    def test_coverage_tracks_stated_confidence(self, many_sessions):
        """The headline property, on data whose generator we control.

        Tolerances are loose because observations within a session are heavily
        correlated -- the effective sample is nearer the session count than the
        observation count -- but a model that were badly miscalibrated would
        miss by far more than this.
        """
        train, test = split_sessions(many_sessions, 0.7)
        profile = VarianceProfile.fit(train)
        model = RangeModel.fit(train, profile, stride=30)
        rows = calibration(model, test, profile, stride=30)

        by_alpha = {row.alpha: row for row in rows}
        assert by_alpha[0.50].empirical == pytest.approx(0.50, abs=0.08)
        assert by_alpha[0.05].empirical == pytest.approx(0.05, abs=0.05)
        assert by_alpha[0.95].empirical == pytest.approx(0.95, abs=0.05)

    def test_coverage_is_monotonic_in_alpha(self, many_sessions):
        train, test = split_sessions(many_sessions, 0.7)
        profile = VarianceProfile.fit(train)
        model = RangeModel.fit(train, profile, stride=30)
        rows = sorted(calibration(model, test, profile, stride=30),
                      key=lambda r: r.alpha)
        empirical = [r.empirical for r in rows]
        assert empirical == sorted(empirical)

    def test_a_deliberately_wrong_model_fails_calibration(self, many_sessions):
        """Guard against a test that would pass for a broken model too.

        Halving every quantile should show up as visible under-coverage in the
        lower tail; if it does not, `calibration` is not measuring anything.
        """
        train, test = split_sessions(many_sessions, 0.7)
        profile = VarianceProfile.fit(train)
        model = RangeModel.fit(train, profile, stride=30)
        honest = {r.alpha: r.empirical for r in calibration(model, test, profile, 30)}

        model.z_close = model.z_close * 0.5
        broken = {r.alpha: r.empirical for r in calibration(model, test, profile, 30)}

        assert broken[0.05] > honest[0.05] + 0.02

    def test_split_is_chronological(self, many_sessions):
        train, test = split_sessions(many_sessions, 0.7)
        assert max(train) < min(test)
        assert len(train) + len(test) == len(many_sessions)


# --------------------------------------------------------------------------
# The "have we bottomed" reading
# --------------------------------------------------------------------------

class TestTurningPointReading:
    def test_probability_is_a_probability(self, many_sessions):
        profile = VarianceProfile.fit(many_sessions)
        model = RangeModel.fit(many_sessions, profile, stride=30)
        frame = next(iter(many_sessions.values()))
        state, _ = observe(frame, 200, profile)

        for p in (model.probability_low_is_in(state),
                  model.probability_high_is_in(state)):
            assert 0.0 <= p <= 1.0

    def test_a_late_session_is_more_settled_than_an_early_one(self, many_sessions):
        """With minutes left there is little room to make a new extreme, so
        both probabilities should be higher than at the open."""
        profile = VarianceProfile.fit(many_sessions)
        model = RangeModel.fit(many_sessions, profile, stride=30)
        frame = next(iter(many_sessions.values()))

        early, _ = observe(frame, 20, profile)
        late, _ = observe(frame, 380, profile)
        early_p = model.probability_low_is_in(early) + model.probability_high_is_in(early)
        late_p = model.probability_low_is_in(late) + model.probability_high_is_in(late)
        assert late_p > early_p


# --------------------------------------------------------------------------
# Bar plumbing
# --------------------------------------------------------------------------

class TestBarHandling:
    def test_month_starts_spans_the_range(self):
        starts = month_starts(date(2024, 1, 15), date(2024, 4, 2))
        assert starts == [date(2024, 1, 1), date(2024, 2, 1),
                          date(2024, 3, 1), date(2024, 4, 1)]

    def test_month_starts_crosses_the_year(self):
        starts = month_starts(date(2024, 11, 5), date(2025, 2, 1))
        assert starts[0] == date(2024, 11, 1) and starts[-1] == date(2025, 2, 1)

    def test_duplicate_minutes_are_dropped(self):
        """A duplicated bar would be squared twice in the realized-variance sum."""
        frame = synthetic_session(date(2024, 3, 1), seed=5).rename(columns={"ts": "ts_event"})
        doubled = pd.concat([frame, frame], ignore_index=True)
        out = normalize_bars(doubled, date(2024, 3, 1), date(2024, 3, 1))
        assert len(out) == len(frame)
        assert out["ts"].is_unique

    def test_rows_outside_the_range_are_trimmed(self):
        a = synthetic_session(date(2024, 3, 1), seed=5)
        b = synthetic_session(date(2024, 3, 4), seed=6)
        both = pd.concat([a, b], ignore_index=True).rename(columns={"ts": "ts_event"})
        out = normalize_bars(both, date(2024, 3, 1), date(2024, 3, 1))
        assert set(out["ts"].dt.date) == {date(2024, 3, 1)}

    def test_regular_hours_keeps_only_the_cash_session(self):
        frame = synthetic_session(date(2024, 3, 1), seed=5)
        overnight = frame.copy()
        overnight["ts"] = overnight["ts"] - pd.Timedelta(hours=6)
        out = regular_hours(pd.concat([overnight, frame], ignore_index=True))
        assert out["ts"].dt.time.min() >= RTH_OPEN
        assert out["ts"].dt.time.max() <= RTH_CLOSE

    def test_short_sessions_are_dropped(self):
        """Half-days would contaminate the variance profile they are pooled into."""
        full = synthetic_session(date(2024, 3, 1), seed=5)
        half = synthetic_session(date(2024, 3, 4), seed=6, n=120)
        bars = pd.concat([full, half], ignore_index=True)
        out = sessions(bars, min_bars=300)
        assert set(out) == {date(2024, 3, 1)}

    def test_state_at_uses_the_last_bar_at_or_before_the_clock(self):
        frame = synthetic_session(date(2024, 3, 1), seed=5)
        profile = VarianceProfile.fit({date(2024, 3, 1): frame})
        state, _ = state_at(frame, time(11, 0), profile)
        assert state.at.time() <= time(11, 0)
        assert state.at.time() >= time(10, 59)


# --------------------------------------------------------------------------
# Transient-failure handling on long pulls
# --------------------------------------------------------------------------

class TestRetry:
    def test_a_transient_failure_is_retried(self):
        from backtest.source import with_retry

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise _server_error(504)
            return "ok"

        assert with_retry(flaky, base_delay=0.0, echo=lambda *_: None) == "ok"
        assert calls["n"] == 3

    def test_a_client_error_is_not_retried(self):
        """A 422 means the request is wrong; repeating it cannot help."""
        from backtest.source import with_retry

        calls = {"n": 0}

        def bad_request():
            calls["n"] += 1
            raise _client_error(422)

        with pytest.raises(Exception):
            with_retry(bad_request, base_delay=0.0, echo=lambda *_: None)
        assert calls["n"] == 1

    def test_retries_are_bounded(self):
        from backtest.source import with_retry

        calls = {"n": 0}

        def always_down():
            calls["n"] += 1
            raise _server_error(503)

        with pytest.raises(Exception):
            with_retry(always_down, attempts=4, base_delay=0.0, echo=lambda *_: None)
        assert calls["n"] == 4

    def test_transient_classification(self):
        from backtest.source import is_transient

        assert is_transient(_server_error(504))
        assert is_transient(ConnectionError("reset"))
        assert not is_transient(_client_error(422))
        assert not is_transient(ValueError("nonsense"))


def _server_error(status):
    from databento.common.error import BentoServerError

    return BentoServerError(http_status=status, message=f"{status} upstream")


def _client_error(status):
    from databento.common.error import BentoClientError

    return BentoClientError(http_status=status, message=f"{status} bad request")
