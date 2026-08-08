"""Underlying minute bars, for research that does not need the option chain.

The question this harness is ultimately asked -- how far can the index still
travel before the close -- is a question about the *underlying*, not about
options. That distinction is worth about four orders of magnitude: a session of
ES continuous minute bars prices at ~$0.0014 against ~$17.69 for the SPXW
`cmbp-1` chain, so several years of signal research costs roughly a dollar.

ES rather than SPY: it is the instrument that actually leads the cash index, it
carries no tracking or dividend drift, and its overnight session gives the
opening gap that a cash-only feed cannot see. Its *level* differs from SPX by
the basis, but every quantity here is a return, and the basis moves far too
slowly to matter over a single session.

There is no SPX index feed among the entitled datasets, which settles the
question of using a proxy at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, time

import pandas as pd

from .source import ET

log = logging.getLogger(__name__)

# The cash session. SPXW 0DTE contracts are PM-settled against the 16:00 close,
# so this is the window whose extremes decide a trade.
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)

BAR_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass
class UnderlyingConfig:
    """Where the underlying bars come from.

    Separate from ``DataConfig`` because it addresses a different venue: CME
    rather than OPRA, continuous-contract symbology rather than option parents.
    """

    dataset: str = "GLBX.MDP3"
    symbol: str = "ES.c.0"
    stype_in: str = "continuous"
    schema: str = "ohlcv-1m"


def month_starts(start: date, end: date) -> list[date]:
    """First of each month spanned by ``start``..``end`` inclusive."""
    out, cursor = [], date(start.year, start.month, 1)
    while cursor <= end:
        out.append(cursor)
        cursor = date(cursor.year + (cursor.month == 12),
                      cursor.month % 12 + 1, 1)
    return out


def load_minutes(
    fetcher, ucfg: UnderlyingConfig, start: date, end: date,
) -> pd.DataFrame:
    """Minute bars over ``start``..``end``, pulled and cached a month at a time.

    Monthly chunks are the compromise between round trips and cache
    granularity: a year is twelve requests rather than 250, and a partially
    completed multi-year pull leaves whole usable months behind rather than
    having to start over.
    """
    # One past the last requested day. The final chunk is clamped to this:
    # asking for a window that runs past the end of the dataset is a 422, not
    # an empty frame, so an unclamped month boundary would fail the whole pull
    # for no reason other than the calendar.
    stop = pd.Timestamp(end, tz=ET) + pd.Timedelta(days=1)

    frames = []
    for first in month_starts(start, end):
        lo = pd.Timestamp(first, tz=ET)
        hi = min((lo + pd.offsets.MonthBegin(1)).tz_convert(ET), stop)
        if hi <= lo:
            continue
        frame, _ = fetcher.fetch_window(
            dataset=ucfg.dataset,
            schema=ucfg.schema,
            symbols=[ucfg.symbol],
            lo=lo,
            hi=hi,
            stype_in=ucfg.stype_in,
            key_day=first,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["ts", *BAR_COLUMNS])

    bars = pd.concat(frames, ignore_index=True)
    return normalize_bars(bars, start, end)


def normalize_bars(bars: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """One tidy frame: ET timestamps, one row per minute, trimmed to the range.

    Chunk boundaries can overlap by a bar and continuous-contract rolls can
    emit two rows for the same minute under different instrument ids, so the
    de-duplication here is load-bearing rather than defensive: a duplicated
    minute would be counted twice in the realized-variance sum.
    """
    ts_col = "ts_event" if "ts_event" in bars.columns else "ts_recv"
    out = bars.rename(columns={ts_col: "ts"})
    keep = ["ts", *[c for c in BAR_COLUMNS if c in out.columns]]
    out = out[keep].dropna(subset=["ts", "close"])

    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    mask = (out["ts"].dt.date >= start) & (out["ts"].dt.date <= end)
    return out.loc[mask].reset_index(drop=True)


def regular_hours(bars: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the cash session, which is what 0DTE settles against."""
    clock = bars["ts"].dt.time
    return bars.loc[(clock >= RTH_OPEN) & (clock <= RTH_CLOSE)].reset_index(drop=True)


def sessions(bars: pd.DataFrame, min_bars: int = 300) -> dict[date, pd.DataFrame]:
    """Split into per-day frames, dropping days too short to be a real session.

    Half-days and feed outages produce sessions whose realized variance is not
    comparable to a full one. Including them would contaminate the intraday
    variance profile that everything downstream is scaled by, so they are
    dropped rather than patched.
    """
    rth = regular_hours(bars)
    out = {}
    for day, frame in rth.groupby(rth["ts"].dt.date):
        if len(frame) >= min_bars:
            out[day] = frame.reset_index(drop=True)
        else:
            log.debug("dropping %s: only %d bars", day, len(frame))
    return out
