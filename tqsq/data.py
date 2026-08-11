"""Databento ingestion with a cost gate and an on-disk cache.

The cost gate is not ceremony. `metadata.get_cost` is free, the pull is not, and
this project has already lost a pull to `402 account_insufficient_funds` after
the estimate said $2.96 -- the estimate prices the REQUEST, not your remaining
balance, so a cheap-looking call can still fail. `fetch` therefore prices first,
refuses above `ceiling`, and writes chunk by chunk so a mid-way 402 leaves every
completed year on disk instead of nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATASET = "XNAS.ITCH"          # Nasdaq-listed; reaches back to 2018-05-01
SCHEMA = "ohlcv-1m"
DEFAULT_SYMBOLS = ("TQQQ", "SQQQ")


@dataclass(frozen=True)
class FetchPlan:
    start: str
    end: str
    symbols: tuple[str, ...]
    dataset: str = DATASET
    schema: str = SCHEMA


def _client():
    try:
        import databento as db
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pip install databento") from exc
    if not os.environ.get("DATABENTO_API_KEY"):
        raise RuntimeError("DATABENTO_API_KEY is not set")
    return db.Historical()


def estimate_cost(plan: FetchPlan) -> float:
    """Dollar cost of `plan`. Free to call."""
    return float(
        _client().metadata.get_cost(
            dataset=plan.dataset, symbols=list(plan.symbols), schema=plan.schema,
            start=plan.start, end=plan.end, stype_in="raw_symbol",
        )
    )


def yearly_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split a range into one-year chunks, newest first.

    Newest first matters: if the budget runs out partway, you keep the most
    recent history rather than the oldest.
    """
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    cur = e
    while cur > s:
        prev = max(s, cur - pd.DateOffset(years=1))
        out.append((prev.strftime("%Y-%m-%d"), cur.strftime("%Y-%m-%d")))
        cur = prev
    return out


def fetch(
    start: str,
    end: str,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    cache_dir: str | Path = "data/raw",
    ceiling: float = 25.0,
) -> list[Path]:
    """Download 1-minute bars in yearly chunks, skipping what is already cached."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    client = _client()
    written: list[Path] = []

    total = estimate_cost(FetchPlan(start, end, symbols))
    if total > ceiling:
        raise RuntimeError(
            f"estimated ${total:.2f} exceeds ceiling ${ceiling:.2f}; raise ceiling to proceed"
        )

    for chunk_start, chunk_end in yearly_chunks(start, end):
        path = cache / f"{'_'.join(symbols)}_{chunk_start}_{chunk_end}.dbn.zst"
        if path.exists():
            written.append(path)
            continue
        try:
            client.timeseries.get_range(
                dataset=DATASET, symbols=list(symbols), schema=SCHEMA,
                start=chunk_start, end=chunk_end, stype_in="raw_symbol", path=str(path),
            )
            written.append(path)
        except Exception as exc:
            path.unlink(missing_ok=True)
            # 402 means the account balance ran out, not that the plan was
            # invalid. Keep what we have; the caller decides whether it is enough.
            raise RuntimeError(f"chunk {chunk_start}..{chunk_end} failed: {exc}") from exc
    return written


def consolidate(cache_dir: str | Path = "data/raw", out: str = "data/raw/bars_1m.parquet") -> Path:
    """Merge every cached .dbn.zst chunk into one parquet the loaders read."""
    import databento as db

    cache = Path(cache_dir)
    frames = []
    for path in sorted(cache.glob("*.dbn.zst")):
        frames.append(db.DBNStore.from_file(path).to_df().reset_index())
    if not frames:
        raise RuntimeError(f"no .dbn.zst chunks in {cache}")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["symbol", "ts_event"]).drop_duplicates(["symbol", "ts_event"], keep="last")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path
