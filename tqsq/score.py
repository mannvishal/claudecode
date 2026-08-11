"""Python port of the 0DTE composite regime score from "Vishal's Upper Pane" v6.40.

The Pine original splits the work in two, and this module keeps that split
because it is what makes the score reproducible:

  f_zdState()  runs *inside* a request.security context and exports the leg's
               recursive state as of the LAST COMPLETED higher-timeframe bar.
               Every recursive term goes out shifted one bar (`[1]`) -- that
               shift is the v6.37 bugfix, and dropping it makes the score lead
               the true value by up to 2.44 points in a trend.

  f_legAt(p)   advances that state exactly one step with a candidate price and
               returns the leg score. Called with the leg's own current close it
               reproduces the live score; called with any other price it answers
               "what would the score be if this bar closed there".

Both are vectorised here: `leg_state` builds the per-minute state arrays and
`leg_score_at` is a pure function over numpy arrays.

Deliberate approximation, and the only one in this file: Pine's `ta.vwap(hlc3)`
is evaluated inside each leg's own timeframe context. This port reproduces that
(session-anchored, developing bar included) rather than substituting a 1-minute
VWAP. The residual difference is that Pine seeds `ta.ema`/`ta.rma` with an SMA
over the first `length` bars while this uses a first-value seed; both converge,
which is what `warmup_bars` in the config exists to discard.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bars import aggregate, bucket_ids, running_bar_state

# Leg timeframes in minutes -- inputs zdTF1..zdTF4.
LEG_TFS: tuple[int, ...] = (2, 5, 15, 60)

# Longer-timeframe context legs (4H / D / W / M in Pine). Context only: they
# feed no machine and raise no alert, so the backtest does not use them.
CONTEXT_TFS: tuple[int, ...] = (240,)


@dataclass(frozen=True)
class Weights:
    """Score component weights -- inputs zdWEma .. zdWMacd."""

    ema: float = 1.00     # EMA 9/14/26 alignment
    price: float = 0.55   # price vs EMA 26
    vwap: float = 0.15    # price vs session VWAP
    rsi: float = 7.10     # RSI(14), the dominant term at 71% of the weight
    macd: float = 1.20    # MACD line vs signal


@dataclass
class LegState:
    """Last-completed-bar state for one leg, broadcast to every 1-minute row."""

    e9: np.ndarray
    e14: np.ndarray
    e26: np.ndarray
    e12: np.ndarray
    sig: np.ndarray      # MACD signal line
    vwap: np.ndarray     # developing-bar session VWAP (NOT shifted -- see module docstring)
    gain: np.ndarray     # Wilder RMA of upward closes
    loss: np.ndarray     # Wilder RMA of downward closes
    close: np.ndarray    # close of the last completed bar
    vwap_ok: np.ndarray  # bool: VWAP term participates


def _ema(x: np.ndarray, length: int) -> np.ndarray:
    """EMA with a first-value seed, matching f_legAt's advance step."""
    alpha = 2.0 / (length + 1.0)
    out = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc = acc + alpha * (x[i] - acc)
        out[i] = acc
    return out


def _rma(x: np.ndarray, length: int) -> np.ndarray:
    """Wilder's smoothing: rma = (rma[1] * (n-1) + x) / n."""
    out = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc = (acc * (length - 1) + x[i]) / length
        out[i] = acc
    return out


def _session_vwap(htf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative hlc3*volume and volume over *completed* bars, reset each day."""
    hlc3 = (htf["high"] + htf["low"] + htf["close"]).to_numpy(float) / 3.0
    vol = htf["volume"].to_numpy(float)
    num = hlc3 * vol
    day = pd.factorize(htf["date"].to_numpy())[0]
    cum_num = np.zeros(len(htf))
    cum_den = np.zeros(len(htf))
    n = d = 0.0
    prev = -1
    for i in range(len(htf)):
        if day[i] != prev:
            n = d = 0.0
            prev = day[i]
        n += num[i]
        d += vol[i]
        cum_num[i], cum_den[i] = n, d
    return cum_num, cum_den


def leg_state(df: pd.DataFrame, tf_minutes: int, anchor: int, rsi_len: int = 14) -> LegState:
    """Build the per-minute exported state for one leg.

    `df` is 1-minute bars for a single symbol and session. The returned arrays
    are aligned row-for-row with `df`.
    """
    buckets = bucket_ids(df, tf_minutes, anchor)
    htf = aggregate(df, buckets)
    close = htf["close"].to_numpy(float)

    e9, e14, e26, e12 = (_ema(close, n) for n in (9, 14, 26, 12))
    macd = e12 - e26
    sig = _ema(macd, 9)

    delta = np.diff(close, prepend=close[0])
    gain = _rma(np.maximum(delta, 0.0), rsi_len)
    loss = _rma(np.maximum(-delta, 0.0), rsi_len)

    cum_num, cum_den = _session_vwap(htf)

    # Pine's `[1]`: state as of the bar BEFORE the developing one. Index -1 is
    # the last completed bar for a row in bucket b, i.e. bucket b-1.
    prev = buckets - 1
    valid = prev >= 0
    idx = np.where(valid, prev, 0)

    def take(a: np.ndarray) -> np.ndarray:
        return np.where(valid, a[idx], np.nan)

    # VWAP is NOT shifted: f_legAt only compares price against it, it never
    # advances it. Include the developing bar's partial contribution.
    run = running_bar_state(df, buckets)
    dev_hlc3 = (run["high"] + run["low"] + run["close"]) / 3.0
    dev_vol = run["volume"]
    # Completed-bar cumulative sums within this session, up to bucket b-1.
    base_num = np.where(valid, cum_num[idx], 0.0)
    base_den = np.where(valid, cum_den[idx], 0.0)
    # A new session resets the accumulator, so drop the carry across the boundary.
    day_of_row = pd.factorize(df["date"].to_numpy())[0]
    day_of_prev = np.where(valid, pd.factorize(htf["date"].to_numpy())[0][idx], -1)
    same_day = day_of_row == day_of_prev
    base_num = np.where(same_day, base_num, 0.0)
    base_den = np.where(same_day, base_den, 0.0)

    den = base_den + dev_vol
    vwap = np.where(den > 0, (base_num + dev_hlc3 * dev_vol) / np.maximum(den, 1e-12), np.nan)

    return LegState(
        e9=take(e9),
        e14=take(e14),
        e26=take(e26),
        e12=take(e12),
        sig=take(sig),
        vwap=vwap,
        gain=take(gain),
        loss=take(loss),
        close=take(close),
        vwap_ok=np.isfinite(vwap),
    )


def leg_score_at(p: np.ndarray | float, st: LegState, w: Weights = Weights()) -> np.ndarray:
    """Score of one leg if its developing bar closed at `p`. Port of f_legAt."""
    p = np.asarray(p, dtype=float)

    q9 = st.e9 + (2.0 / 10.0) * (p - st.e9)
    q14 = st.e14 + (2.0 / 15.0) * (p - st.e14)
    q26 = st.e26 + (2.0 / 27.0) * (p - st.e26)
    q12 = st.e12 + (2.0 / 13.0) * (p - st.e12)

    # EMA alignment: full credit for a stacked fan, 0.4 for a partial read.
    stacked_up = (q9 > q14) & (q14 > q26)
    stacked_dn = (q9 < q14) & (q14 < q26)
    partial = np.where(q9 > q26, 0.4, -0.4)
    ema_sig = np.where(stacked_up, 1.0, np.where(stacked_dn, -1.0, partial))

    price_sig = np.where(p > q26, 1.0, -1.0)
    vwap_sig = np.where(st.vwap_ok, np.where(p > st.vwap, 1.0, -1.0), 0.0)

    # RSI advanced one Wilder step by p.
    up = st.gain * 13.0 + np.maximum(p - st.close, 0.0)
    dn = st.loss * 13.0 + np.maximum(st.close - p, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = np.where(dn <= 0, 100.0, 100.0 - 100.0 / (1.0 + up / np.where(dn <= 0, 1.0, dn)))
    rsi_sig = np.clip((rsi - 50.0) / 25.0, -1.0, 1.0)

    macd_line = q12 - q26
    sig2 = st.sig + (2.0 / 10.0) * (macd_line - st.sig)
    macd_sig = np.where(macd_line > sig2, 1.0, -1.0)

    total_w = w.ema + w.price + np.where(st.vwap_ok, w.vwap, 0.0) + w.rsi + w.macd
    raw = (
        ema_sig * w.ema
        + price_sig * w.price
        + vwap_sig * w.vwap
        + rsi_sig * w.rsi
        + macd_sig * w.macd
    )
    return np.where(total_w == 0, 0.0, raw / total_w * 10.0)


def leg_rsi(st: LegState) -> np.ndarray:
    """RSI of the last completed bar. Port of f_rsiOf -- deliberately NOT advanced.

    The Pine reads `f_rsiOf(aag, aal)` off the shifted RMA state, so the average
    RSI that confirms an exhaustion fire is one higher-timeframe bar stale. That
    lag is part of the signal definition, not an oversight in this port.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(st.loss <= 0, 100.0, 100.0 - 100.0 / (1.0 + st.gain / np.where(st.loss <= 0, 1.0, st.loss)))


@dataclass
class ScoreFrame:
    """Composite score and its inputs, aligned to the 1-minute bars."""

    ts: pd.Series
    close: np.ndarray
    legs: dict[int, np.ndarray]
    avg: np.ndarray
    avg_rsi: np.ndarray
    states: dict[int, LegState]
    weights: Weights

    def as_frame(self) -> pd.DataFrame:
        out = pd.DataFrame({"ts": self.ts, "close": self.close, "score": self.avg, "avg_rsi": self.avg_rsi})
        for tf, v in self.legs.items():
            out[f"leg_{tf}m"] = v
        return out


def compute_score(
    df: pd.DataFrame,
    tfs: tuple[int, ...] = LEG_TFS,
    weights: Weights = Weights(),
    anchor: int | None = None,
) -> ScoreFrame:
    """Composite 0DTE score for a 1-minute frame from `bars.load_bars`.

    The developing higher-timeframe bar's close equals the current 1-minute
    close at every moment, which is why one price feeds all four legs.
    """
    if anchor is None:
        anchor = df.attrs.get("anchor", 9 * 60 + 30)

    close = df["close"].to_numpy(float)
    states = {tf: leg_state(df, tf, anchor) for tf in tfs}
    legs = {tf: leg_score_at(close, st, weights) for tf, st in states.items()}
    rsis = {tf: leg_rsi(st) for tf, st in states.items()}

    # Full precision in the decision path: the Pine rounds only for display
    # (v6.36). Rounding here would move the average by up to +/-0.05 and change
    # which bar crosses a threshold.
    # The opening bars of the series have no completed higher-timeframe bar on
    # the slow legs, so every leg is nan there and nanmean would warn. Those
    # rows are genuinely undefined -- leave them nan rather than inventing a 0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        avg = np.nanmean(np.column_stack(list(legs.values())), axis=1)
        avg_rsi = np.nanmean(np.column_stack(list(rsis.values())), axis=1)

    return ScoreFrame(
        ts=df["ts"].reset_index(drop=True),
        close=close,
        legs=legs,
        avg=avg,
        avg_rsi=avg_rsi,
        states=states,
        weights=weights,
    )


def score_at_price(sf: ScoreFrame, row: int, p: float) -> float:
    """Composite score at row `row` if the bar closed at `p`.

    This is Pine's `f_avgAt`, and it is what the isoline solver bisects to draw
    the score cloud. Exposed here so the live runner can answer "how far does
    price have to move before tier 3 fires" without re-deriving the algebra.
    """
    vals = []
    for tf, st in sf.states.items():
        one = LegState(**{k: np.asarray([getattr(st, k)[row]]) for k in st.__dataclass_fields__})
        vals.append(leg_score_at(np.asarray([p]), one, sf.weights)[0])
    return float(np.mean(vals))
