"""Parquet cache, keyed by dataset + schema + symbol-set + date.

Rule 3: never re-pull a cached range. That is a correctness property as much as
a cost one -- a backtest whose inputs change between runs is not reproducible,
and silently re-fetching makes it impossible to tell whether a result moved
because you changed the strategy or because the vendor revised the data.

Cache keys are content-addressed on the symbol set, because two pulls for the
same day and schema but different strike bands are genuinely different data. A
key that ignored the symbols would serve a narrow pull in answer to a wide one
and quietly truncate the chain.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


def symbol_fingerprint(symbols: list[str] | None) -> str:
    """Stable short hash of a symbol set. ``None`` means "the whole feed"."""
    if not symbols:
        return "ALL"
    joined = "\n".join(sorted(set(symbols)))
    return hashlib.sha256(joined.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class CacheKey:
    dataset: str
    schema: str
    day: date
    symbols_hash: str

    @classmethod
    def build(cls, dataset: str, schema: str, day: date, symbols: list[str] | None) -> "CacheKey":
        return cls(dataset, schema, day, symbol_fingerprint(symbols))

    @property
    def relative_path(self) -> Path:
        safe_dataset = self.dataset.replace(".", "_")
        return Path(safe_dataset) / self.schema / f"{self.day.isoformat()}_{self.symbols_hash}.parquet"


class ParquetCache:
    def __init__(self, root: str | Path = "./data"):
        self.root = Path(root)

    def path_for(self, key: CacheKey) -> Path:
        return self.root / key.relative_path

    def has(self, key: CacheKey) -> bool:
        return self.path_for(key).exists()

    def read(self, key: CacheKey) -> pd.DataFrame:
        path = self.path_for(key)
        if not path.exists():
            raise KeyError(f"cache miss: {path}")
        return pd.read_parquet(path)

    def write(self, key: CacheKey, frame: pd.DataFrame, meta: dict | None = None) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        self._record(key, frame, meta or {})
        log.info("cached %d rows -> %s", len(frame), path)
        return path

    def _record(self, key: CacheKey, frame: pd.DataFrame, meta: dict) -> None:
        """Append to a human-readable manifest.

        The parquet files alone cannot tell you what a pull cost or which symbol
        set produced a given hash. Without that, an unexpected cache miss six
        months later is unattributable.
        """
        manifest_path = self.root / MANIFEST_NAME
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        if manifest_path.exists():
            try:
                entries = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                log.warning("manifest at %s is corrupt; starting a new one", manifest_path)
        entries.append({
            "dataset": key.dataset,
            "schema": key.schema,
            "date": key.day.isoformat(),
            "symbols_hash": key.symbols_hash,
            "rows": int(len(frame)),
            "path": str(key.relative_path),
            **meta,
        })
        manifest_path.write_text(json.dumps(entries, indent=1, default=str))

    def manifest(self) -> list[dict]:
        path = self.root / MANIFEST_NAME
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return []

    def total_spend(self) -> float:
        return sum(float(e.get("cost_usd") or 0.0) for e in self.manifest())

    def cached_days(self, dataset: str, schema: str) -> set[date]:
        return {
            date.fromisoformat(e["date"])
            for e in self.manifest()
            if e.get("dataset") == dataset and e.get("schema") == schema
        }
