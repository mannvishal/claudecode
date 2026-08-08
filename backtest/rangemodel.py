"""How far can the index still travel today?

This is the layer that answers the question a 0DTE credit-spread seller
actually has. That question is *not* "have we bottomed" -- a point call on the
turn is the least reliable thing one can ask of intraday data and the easiest
to overfit, because any rule can be tuned to catch the turns in a sample you
have already seen. It is "given where we are and how today has behaved, what is
the distribution of where we can still get to by the close", which is both more
tractable and the thing that actually determines a strike.

The method has three parts.

**Scale.** A move is only large relative to the volatility in force. Realized
variance accumulated since the open is divided by the share of a typical day's
variance that has elapsed by this time, giving a same-day volatility estimate
that adapts to a quiet or a violent morning. This is the standard
variance-profile correction and it is what makes a 10:00 reading and a 15:00
reading comparable at all.

**Shape.** Scaled outcomes are pooled into an empirical distribution. Its
quantiles are read directly -- no distributional assumption, so the fat left
tail that actually kills credit spreads survives into the estimate instead of
being normalised away.

**Proof.** The only claim made is a calibration claim: when the model says 95%,
it should be right 95% of the time, measured out of sample. Coverage is
falsifiable in a way that "did it call the low" is not, and it is reported
whether or not it flatters the model.

Nothing here reads an option price. That is deliberate: this layer is developed
on underlying bars costing fractions of a cent per session, and only a model
that survives calibration is worth spending OPRA money to confirm.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, time

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Reported quantiles. Weighted toward the tails because that is where a credit
# spread lives -- the 50% level is never a strike anyone sells.
DEFAULT_QUANTILES = (0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50,
                     0.75, 0.90, 0.95, 0.975, 0.99, 0.995)

# Guards against a degenerate scale estimate on an unnaturally quiet morning,
# which would otherwise divide a small move by a near-zero sigma and produce an
# enormous z. Expressed as a fraction of the session's median variance share.
MIN_VARIANCE_SHARE = 0.02
MIN_SIGMA = 1e-5


@dataclass
class SessionState:
    """Everything known at one decision moment, before the outcome is seen."""

    day: date
    at: pd.Timestamp
    price: float
    minutes_left: int
    sigma_remaining: float   # stdev of the remaining return, in return units
    high_so_far: float
    low_so_far: float
    open_price: float

    @property
    def return_from_open(self) -> float:
        return self.price / self.open_price - 1.0


@dataclass
class Outcome:
    """What actually happened after a ``SessionState``."""

    close_return: float      # close / price - 1
    min_return: float        # lowest excursion from price, <= 0
    max_return: float        # highest excursion from price, >= 0


@dataclass
class VarianceProfile:
    """Share of a session's realized variance accumulated by each minute.

    Intraday volatility is strongly U-shaped: the first thirty minutes carry a
    disproportionate share and the middle of the day very little. Without this
    correction, an 11:00 realized-variance reading would imply a far calmer day
    than it should, because it is being compared against a full session.
    """

    shares: np.ndarray = field(repr=False)  # cumulative share, one per minute index
    n_sessions: int = 0

    @classmethod
    def fit(cls, sessions: dict[date, pd.DataFrame]) -> "VarianceProfile":
        curves = []
        for frame in sessions.values():
            r = np.diff(np.log(frame["close"].to_numpy()))
            if len(r) < 2:
                continue
            cumulative = np.cumsum(r ** 2)
            total = cumulative[-1]
            if total <= 0:
                continue
            curves.append(cumulative / total)

        if not curves:
            raise ValueError("no usable sessions to fit a variance profile")

        width = min(len(c) for c in curves)
        stacked = np.vstack([c[:width] for c in curves])
        # Median rather than mean: one crash day would otherwise drag the whole
        # profile toward its own shape.
        return cls(shares=np.median(stacked, axis=0), n_sessions=len(curves))

    def share_by(self, index: int) -> float:
        """Fraction of the day's variance expected to have elapsed by ``index``."""
        if index <= 0:
            return MIN_VARIANCE_SHARE
        clipped = min(index, len(self.shares)) - 1
        return max(float(self.shares[clipped]), MIN_VARIANCE_SHARE)


def observe(
    frame: pd.DataFrame, index: int, profile: VarianceProfile,
) -> tuple[SessionState, Outcome] | None:
    """Split one session at ``index`` into what was known and what followed.

    The split is strict: ``SessionState`` is built only from rows up to and
    including ``index``, and ``Outcome`` only from rows after it. Every
    lookahead bug in a backtest of this shape is a row that crossed this line.
    """
    if index < 1 or index >= len(frame) - 1:
        return None

    closes = frame["close"].to_numpy()
    seen, future = closes[: index + 1], closes[index + 1:]
    if len(future) == 0:
        return None

    price = float(seen[-1])
    if price <= 0:
        return None

    # Same-day volatility, rescaled by how much of the day's variance is
    # typically over by now.
    r = np.diff(np.log(seen))
    realized = float(np.sum(r ** 2))
    share = profile.share_by(index)
    daily_variance = realized / share
    remaining_variance = max(daily_variance * (1.0 - share), 0.0)
    sigma_remaining = max(math.sqrt(remaining_variance), MIN_SIGMA)

    highs = frame["high"].to_numpy() if "high" in frame else closes
    lows = frame["low"].to_numpy() if "low" in frame else closes

    state = SessionState(
        day=frame["ts"].iloc[index].date(),
        at=frame["ts"].iloc[index],
        price=price,
        minutes_left=len(future),
        sigma_remaining=sigma_remaining,
        high_so_far=float(np.max(highs[: index + 1])),
        low_so_far=float(np.min(lows[: index + 1])),
        open_price=float(closes[0]),
    )
    outcome = Outcome(
        close_return=float(future[-1]) / price - 1.0,
        min_return=float(np.min(lows[index + 1:])) / price - 1.0,
        max_return=float(np.max(highs[index + 1:])) / price - 1.0,
    )
    return state, outcome


@dataclass
class RangeModel:
    """Empirical quantiles of scale-normalized remaining-day moves."""

    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    z_close: np.ndarray = field(default=None, repr=False)
    z_min: np.ndarray = field(default=None, repr=False)
    z_max: np.ndarray = field(default=None, repr=False)
    profile: VarianceProfile | None = field(default=None, repr=False)
    n_observations: int = 0

    @classmethod
    def fit(
        cls, sessions: dict[date, pd.DataFrame], profile: VarianceProfile,
        stride: int = 5, quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> "RangeModel":
        """Pool normalized outcomes across every session and decision minute.

        Pooling across times of day is what the scale correction buys: once a
        move is expressed in units of its own remaining volatility, a 10:05
        observation and a 14:30 observation are draws from the same
        distribution, and the sample is large enough to estimate a 0.5%
        quantile without fitting a curve to noise.
        """
        zc, zmin, zmax = [], [], []
        for frame in sessions.values():
            for index in range(1, len(frame) - 1, stride):
                observed = observe(frame, index, profile)
                if observed is None:
                    continue
                state, outcome = observed
                scale = state.sigma_remaining
                zc.append(outcome.close_return / scale)
                zmin.append(outcome.min_return / scale)
                zmax.append(outcome.max_return / scale)

        if not zc:
            raise ValueError("no observations; check the session frames")

        return cls(
            quantiles=quantiles,
            z_close=np.array(zc),
            z_min=np.array(zmin),
            z_max=np.array(zmax),
            profile=profile,
            n_observations=len(zc),
        )

    # --- reading the model ------------------------------------------------

    def close_quantile(self, alpha: float) -> float:
        return float(np.quantile(self.z_close, alpha))

    def band(self, state: SessionState, alpha: float) -> tuple[float, float]:
        """Two-sided price band containing the close with probability ``1-alpha``.

        Split evenly between the tails, so ``alpha=0.10`` returns the 5th and
        95th percentiles of the close.
        """
        lo_z = self.close_quantile(alpha / 2.0)
        hi_z = self.close_quantile(1.0 - alpha / 2.0)
        return (state.price * (1.0 + lo_z * state.sigma_remaining),
                state.price * (1.0 + hi_z * state.sigma_remaining))

    def short_strike(self, state: SessionState, side: str, confidence: float) -> float:
        """The strike this model says survives to the close with ``confidence``.

        A put credit spread wants a level the close stays *above*; the relevant
        quantile is therefore the lower tail of the close distribution, and the
        confidence is one-sided. Nothing here rounds to a listed strike -- that
        is the chain's business, not the model's.
        """
        if side == "put":
            z = self.close_quantile(1.0 - confidence)
        elif side == "call":
            z = self.close_quantile(confidence)
        else:
            raise ValueError(f"side must be 'put' or 'call', got {side!r}")
        return state.price * (1.0 + z * state.sigma_remaining)

    def probability_low_is_in(self, state: SessionState) -> float:
        """P(today's low is already set) -- the 'have we bottomed' reading.

        Derived from the excursion distribution rather than asserted: the low
        holds exactly when the remaining downward excursion stays above the low
        already printed. It is a *consequence* of the range model, which is the
        only honest way to produce this number.
        """
        if state.sigma_remaining <= MIN_SIGMA or self.z_min is None:
            return float("nan")
        threshold = (state.low_so_far / state.price - 1.0) / state.sigma_remaining
        return float(np.mean(self.z_min > threshold))

    def probability_high_is_in(self, state: SessionState) -> float:
        if state.sigma_remaining <= MIN_SIGMA or self.z_max is None:
            return float("nan")
        threshold = (state.high_so_far / state.price - 1.0) / state.sigma_remaining
        return float(np.mean(self.z_max < threshold))


# --- validation -----------------------------------------------------------


@dataclass
class Coverage:
    alpha: float
    predicted: float
    empirical: float
    n: int

    @property
    def error(self) -> float:
        return self.empirical - self.predicted


def calibration(
    model: RangeModel, sessions: dict[date, pd.DataFrame],
    profile: VarianceProfile, stride: int = 5,
) -> list[Coverage]:
    """Out-of-sample coverage: does "95%" mean 95%?

    For each target quantile, the fraction of held-out sessions whose actual
    close fell below the level the model predicted. A well-calibrated model
    puts that fraction on the diagonal. This is the whole validation -- a model
    that is sharp but miscalibrated will sell strikes that breach far more
    often than its confidence implies, which is precisely how a credit-spread
    book dies.
    """
    actual, predicted_levels = [], {alpha: [] for alpha in model.quantiles}

    for frame in sessions.values():
        for index in range(1, len(frame) - 1, stride):
            observed = observe(frame, index, profile)
            if observed is None:
                continue
            state, outcome = observed
            actual.append(outcome.close_return)
            for alpha in model.quantiles:
                predicted_levels[alpha].append(
                    model.close_quantile(alpha) * state.sigma_remaining
                )

    if not actual:
        return []

    actual_arr = np.array(actual)
    return [
        Coverage(
            alpha=alpha,
            predicted=alpha,
            empirical=float(np.mean(actual_arr <= np.array(levels))),
            n=len(actual_arr),
        )
        for alpha, levels in predicted_levels.items()
    ]


def split_sessions(
    sessions: dict[date, pd.DataFrame], train_fraction: float = 0.7,
) -> tuple[dict[date, pd.DataFrame], dict[date, pd.DataFrame]]:
    """Chronological train/test split.

    Chronological rather than random: a random split leaks tomorrow's
    volatility regime into today's training set, and volatility is persistent
    enough that the leak alone can make a useless model look calibrated.
    """
    days = sorted(sessions)
    cut = int(len(days) * train_fraction)
    return ({d: sessions[d] for d in days[:cut]},
            {d: sessions[d] for d in days[cut:]})


def state_at(
    frame: pd.DataFrame, clock: time, profile: VarianceProfile,
) -> tuple[SessionState, Outcome] | None:
    """Observe a session at a wall-clock time rather than a bar index."""
    times = frame["ts"].dt.time
    matching = np.flatnonzero((times <= clock).to_numpy())
    if len(matching) == 0:
        return None
    return observe(frame, int(matching[-1]), profile)
