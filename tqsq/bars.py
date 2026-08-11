"""Bar loading, session tagging and higher-timeframe bucketing.

TradingView anchors intraday bars to the *session open*, not to the wall clock.
A 60-minute bar on an RTH chart therefore runs 09:30-10:30, not 09:00-10:00.
`bucket_ids` reproduces that, because every leg of the composite score is a
higher-timeframe aggregate and a half-bar offset would shift every EMA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ET = "America/New_York"

# Session windows, in minutes past midnight Eastern.
SESSIONS = {
    "rth": (9 * 60 + 30, 16 * 60),  # 09:30-16:00, what every study behind the
    "eth": (4 * 60, 20 * 60),       # indicator was measured on. 04:00-20:00 is
}                                    # the chart default (zdRthData = false).

OHLCV = ["open", "high", "low", "close", "volume"]


def load_bars(path: str, symbol: str, session: str = "eth") -> pd.DataFrame:
    """Load 1-minute bars for one symbol, clipped to `session`.

    Returns a frame sorted by timestamp with `ts` (tz-aware Eastern), `date`,
    `minute` (minutes past midnight ET) and OHLCV.
    """
    if session not in SESSIONS:
        raise ValueError(f"session must be one of {sorted(SESSIONS)}, got {session!r}")

    df = pd.read_parquet(path)
    df = df[df["symbol"] == symbol].copy()
    if df.empty:
        raise ValueError(f"no rows for symbol {symbol!r} in {path}")

    df["ts"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert(ET)
    df = df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    df["minute"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df["date"] = df["ts"].dt.normalize()

    lo, hi = SESSIONS[session]
    df = df[(df["minute"] >= lo) & (df["minute"] < hi)].reset_index(drop=True)
    df.attrs["session"] = session
    df.attrs["anchor"] = lo
    df.attrs["symbol"] = symbol
    return df[["ts", "date", "minute", *OHLCV]]


def bucket_ids(df: pd.DataFrame, tf_minutes: int, anchor: int) -> np.ndarray:
    """Monotonic higher-timeframe bar index for each 1-minute row.

    Buckets restart every session, so the id is built from (day, offset) and
    then densified. `df` must already be sorted by `ts`, which makes the dense
    codes monotonically increasing.
    """
    day = pd.factorize(df["date"].to_numpy())[0].astype(np.int64)
    offset = (df["minute"].to_numpy().astype(np.int64) - anchor) // tf_minutes
    key = day * 100_000 + offset
    codes = pd.factorize(key)[0].astype(np.int64)
    if np.any(np.diff(codes) < 0):  # pragma: no cover - guards a sorting bug
        raise AssertionError("bucket ids are not monotonic; input was not sorted")
    return codes


def aggregate(df: pd.DataFrame, buckets: np.ndarray) -> pd.DataFrame:
    """Collapse 1-minute rows into completed higher-timeframe bars."""
    g = df.groupby(buckets, sort=True)
    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum(),
            "date": g["date"].first(),
        }
    )
    return out.reset_index(drop=True)


def running_bar_state(df: pd.DataFrame, buckets: np.ndarray) -> dict[str, np.ndarray]:
    """Developing-bar high/low/close/volume at every 1-minute row.

    Pine reads a still-forming higher-timeframe bar, so VWAP at 10:07 includes
    the partial 10:00 bar. These are the running aggregates of that partial bar.
    """
    n = len(df)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    vol = df["volume"].to_numpy(float)

    run_hi = np.empty(n)
    run_lo = np.empty(n)
    run_vol = np.empty(n)
    cur_hi = -np.inf
    cur_lo = np.inf
    cur_vol = 0.0
    prev = -1
    for i in range(n):
        b = buckets[i]
        if b != prev:
            cur_hi, cur_lo, cur_vol, prev = -np.inf, np.inf, 0.0, b
        cur_hi = max(cur_hi, high[i])
        cur_lo = min(cur_lo, low[i])
        cur_vol += vol[i]
        run_hi[i], run_lo[i], run_vol[i] = cur_hi, cur_lo, cur_vol
    return {"high": run_hi, "low": run_lo, "close": close, "volume": run_vol}
