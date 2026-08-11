"""The zone (A/B/C) and exhaustion (1/2/3) state machines.

Direct port of `f_zdZoneTop/Bot` and `f_zdExhTop/Bot`. Each tier is an
INDEPENDENT one-shot machine with its own peak tracker, stall counter and
"done" latch, which is why a bar can fire tier 1 and tier 3 at once and why
filtering on tier is a real filter rather than a relabelling.

What the two families mean, since the trade hangs on it:

  ZONE A/B/C   the score ENTERING a stretch band. The odometer -- the stretch
               has just begun. Fires on the way out.
  EXHAUST 1/2/3 the stretch failing to make further progress: either the score
               pulls back `rev_delta` off its episode extreme with the average
               RSI confirming, or it stalls in-band for `hold_bars` with no new
               extreme. This is the "something is ENDING" half.

A bottom exhaustion (down-stretch tiring) is the long-TQQQ signal; a top
exhaustion is the long-SQQQ signal. The indicator's own header is explicit that
this is a MARKER and not a validated entry -- see README.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RTH_OPEN_MIN = 9 * 60 + 30
RTH_CLOSE_MIN = 16 * 60


@dataclass(frozen=True)
class SignalConfig:
    """Ports the 0DTE Context / 0DTE Alerts input groups."""

    th1: float = 7.0          # zdTh1 -- zone A / exhaustion 1
    th2: float = 8.0          # zdTh2 -- zone B / exhaustion 2
    th3: float = 8.5          # zdTh3 -- zone C / exhaustion 3
    th_bull: float = 4.0      # zdThBull -- upper edge of the neutral band
    th_bear: float = -4.0     # zdThBear -- lower edge
    rev_delta: float = 1.5    # zdRevDelta -- pullback off the episode extreme
    hold_bars: int = 6        # zdHoldBars -- stall length that also fires
    rsi_ob: float = 70.0      # zdARsiOB -- average-RSI confirm, top side
    rsi_os: float = 30.0      # zdARsiOS -- average-RSI confirm, bottom side
    div_gap: float = 1.0      # zdDivGap -- divergence amplifier
    climax_z: float = 3.0     # zdClimaxZ -- volume climax amplifier
    eth_marks: bool = True    # zdEthMarks -- let machines run outside 09:30-16:00

    @property
    def tiers(self) -> tuple[float, float, float]:
        return (self.th1, self.th2, self.th3)


@dataclass
class SignalFrame:
    """Per-minute machine output. Boolean arrays are one-shot fires."""

    ts: pd.Series
    score: np.ndarray
    avg_rsi: np.ndarray
    in_window: np.ndarray
    zone_top: dict[int, np.ndarray] = field(default_factory=dict)
    zone_bot: dict[int, np.ndarray] = field(default_factory=dict)
    exh_top: dict[int, np.ndarray] = field(default_factory=dict)
    exh_bot: dict[int, np.ndarray] = field(default_factory=dict)
    climax: np.ndarray | None = None
    div_top: np.ndarray | None = None
    div_bot: np.ndarray | None = None

    def exhaustion_tier(self, side: str) -> np.ndarray:
        """Deepest tier firing on each bar, 0 where none. Matches the alert text."""
        src = self.exh_top if side == "top" else self.exh_bot
        out = np.zeros(len(self.score), dtype=int)
        for tier in (1, 2, 3):
            out = np.where(src[tier], tier, out)
        return out


def _in_window(df: pd.DataFrame, eth_marks: bool) -> np.ndarray:
    """Port of `zdInRth`: eth_marks OR the New York regular session."""
    minute = df["minute"].to_numpy()
    rth = (minute >= RTH_OPEN_MIN) & (minute < RTH_CLOSE_MIN)
    return np.ones(len(df), dtype=bool) if eth_marks else rth


def _zone_machine(score, neutral, in_win, threshold, top: bool) -> np.ndarray:
    """One-shot on entering a band; re-arms only in the neutral band."""
    fires = np.zeros(len(score), dtype=bool)
    done = False
    for i in range(len(score)):
        if in_win[i] and neutral[i]:
            done = False
        hit = score[i] >= threshold if top else score[i] <= threshold
        fire = bool(in_win[i] and not done and hit and np.isfinite(score[i]))
        if fire:
            done = True
        fires[i] = fire
    return fires


def _exhaustion_machine(score, avg_rsi, neutral, in_win, threshold, cfg, top: bool) -> np.ndarray:
    """Peak / stall / latch machine. Port of f_zdExhTop and f_zdExhBot."""
    fires = np.zeros(len(score), dtype=bool)
    peak = np.nan
    stall = 0
    done = False

    for i in range(len(score)):
        s = score[i]
        if not np.isfinite(s):
            fires[i] = False
            continue

        if in_win[i]:
            in_band = s >= threshold if top else s <= threshold
            if in_band:
                extended = np.isnan(peak) or (s > peak if top else s < peak)
                if extended:
                    peak, stall = s, 0
                else:
                    stall += 1
            if not np.isnan(peak) and neutral[i]:
                peak, stall, done = np.nan, 0, False

        if in_win[i] and not done and not np.isnan(peak):
            pullback = (peak - s) if top else (s - peak)
            rsi_ok = avg_rsi[i] >= cfg.rsi_ob if top else avg_rsi[i] <= cfg.rsi_os
            in_band = s >= threshold if top else s <= threshold
            fire = (pullback >= cfg.rev_delta and rsi_ok) or (in_band and stall >= cfg.hold_bars)
        else:
            fire = False

        if fire:
            done = True
        fires[i] = fire
    return fires


def _volume_climax(df: pd.DataFrame, in_win: np.ndarray, z_threshold: float) -> np.ndarray:
    """Rolling 30-bar volume z-score that resets each session, never spanning
    the overnight. `zdClimaxNear` is the 4-bar high of that z."""
    vol = df["volume"].to_numpy(float)
    day = pd.factorize(df["date"].to_numpy())[0]
    z = np.zeros(len(df))
    buf: list[float] = []
    prev_day = -1
    for i in range(len(df)):
        if day[i] != prev_day:
            buf.clear()
            prev_day = day[i]
        if in_win[i]:
            if len(buf) >= 30:
                arr = np.asarray(buf)
                sd = arr.std()
                z[i] = (vol[i] - arr.mean()) / sd if sd > 0 else 0.0
            buf.append(vol[i])
            if len(buf) > 30:
                buf.pop(0)
    near = pd.Series(z).rolling(4, min_periods=1).max().to_numpy()
    return near >= z_threshold


def _divergence(df, score, in_win, div_gap) -> tuple[np.ndarray, np.ndarray]:
    """Price retests the session extreme while the score does not follow."""
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    day = pd.factorize(df["date"].to_numpy())[0]

    n = len(df)
    top = np.zeros(n, dtype=bool)
    bot = np.zeros(n, dtype=bool)
    day_hi = day_lo = np.nan
    at_hi = at_lo = np.nan
    prev_day = -1

    for i in range(n):
        if not in_win[i]:
            continue
        if day[i] != prev_day or np.isnan(day_lo):
            day_hi, day_lo, at_hi, at_lo, prev_day = high[i], low[i], score[i], score[i], day[i]
        else:
            if low[i] < day_lo:
                day_lo, at_lo = low[i], score[i]
            if high[i] > day_hi:
                day_hi, at_hi = high[i], score[i]
        # 5 basis points of tolerance on the retest, as in the Pine.
        bot[i] = low[i] <= day_lo * (1 + 5e-4) and score[i] >= at_lo + div_gap
        top[i] = high[i] >= day_hi * (1 - 5e-4) and score[i] <= at_hi - div_gap
    return top, bot


def build_signals(df: pd.DataFrame, score: np.ndarray, avg_rsi: np.ndarray,
                  cfg: SignalConfig = SignalConfig()) -> SignalFrame:
    """Run every machine over a scored 1-minute frame."""
    in_win = _in_window(df, cfg.eth_marks)
    neutral = (score < cfg.th_bull) & (score > cfg.th_bear)

    sf = SignalFrame(ts=df["ts"].reset_index(drop=True), score=score, avg_rsi=avg_rsi, in_window=in_win)
    for tier, th in enumerate(cfg.tiers, start=1):
        sf.zone_top[tier] = _zone_machine(score, neutral, in_win, th, top=True)
        sf.zone_bot[tier] = _zone_machine(score, neutral, in_win, -th, top=False)
        sf.exh_top[tier] = _exhaustion_machine(score, avg_rsi, neutral, in_win, th, cfg, top=True)
        sf.exh_bot[tier] = _exhaustion_machine(score, avg_rsi, neutral, in_win, -th, cfg, top=False)

    sf.climax = _volume_climax(df, in_win, cfg.climax_z)
    sf.div_top, sf.div_bot = _divergence(df, score, in_win, cfg.div_gap)
    return sf
