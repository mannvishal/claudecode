"""DATA LAYER: fetching from Databento, with the cost gate and the cache.

Rules 1-4 live here and nowhere else. Nothing above this layer knows about
datasets, schemas, dollars or UTC.

On the cost gate (rule 1): ``metadata.get_cost`` is free to call, so the
estimate is printed for every pull, not only expensive ones. Above the ceiling
the fetcher raises ``CostCeilingExceeded`` rather than prompting, because this
harness is meant to run unattended over hundreds of sessions -- a blocking
prompt in the middle of a year-long run is a worse failure than stopping. The
CLI catches it and tells you what to raise the ceiling to.

On why the MCP server is not the fetcher: an MCP tool can only be invoked by an
agent inside a live session. A harness you run yourself cannot call one. The
``AgentBridgeFetcher`` exists for the case where an agent has pre-populated the
cache; it never fetches, and a miss is a hard error rather than a silent pull.

Conformance: every keyword this module sends is checked against the installed
SDK's real signatures by ``TestSdkConformance`` in ``tests/test_backtest.py``,
which skips when ``databento`` is absent. Verified against SDK 0.83.0 --
``metadata.get_cost`` and ``timeseries.get_range`` both accept the argument set
built by ``request_kwargs``, ``to_df()`` indexes on ``ts_recv`` (hence the
``reset_index``), and its timestamps arrive in UTC, which is what ``to_eastern``
converts from. What remains unverified is live behaviour: no request has been
made against the real endpoint from this environment, because outbound access to
``hist.databento.com`` is blocked by the egress policy here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Protocol

import pandas as pd

from .cache import CacheKey, ParquetCache
from .config import BacktestConfig

log = logging.getLogger(__name__)

ET = "America/New_York"
UTC = "UTC"


class CostCeilingExceeded(RuntimeError):
    def __init__(self, estimate: float, ceiling: float, describe: str):
        self.estimate = estimate
        self.ceiling = ceiling
        super().__init__(
            f"estimated ${estimate:,.2f} for {describe}, above the ${ceiling:,.2f} ceiling. "
            f"Raise cost.ceiling_usd or narrow the request."
        )


class CostEstimateUnavailable(RuntimeError):
    """Raised when the cost endpoint fails and estimates are required.

    An unknown cost is not a zero cost. Proceeding here spends real money on a
    request nobody sized.
    """


@dataclass
class Pull:
    """The record of one fetch: what was asked for, what it cost, what came back."""

    dataset: str
    schema: str
    day: date
    symbols: list[str] | None
    rows: int
    cost_usd: float | None
    from_cache: bool

    def describe(self) -> str:
        n = "all" if not self.symbols else str(len(self.symbols))
        origin = "cache" if self.from_cache else f"${self.cost_usd or 0:,.4f}"
        return f"{self.schema} {self.day} ({n} symbols, {self.rows:,} rows, {origin})"


def to_eastern(frame: pd.DataFrame, columns: tuple[str, ...] = ("ts_recv", "ts_event")) -> pd.DataFrame:
    """Rule 4: normalize every timestamp to America/New_York at ingest.

    Databento emits UTC nanoseconds. Converting once here means no module above
    this one has to reason about zones -- and more importantly, session
    boundaries like 09:30 and 16:00 are only meaningful in Eastern. A single
    forgotten conversion silently shifts every entry by four or five hours, and
    the shift changes across the DST boundary, so it does not even fail
    consistently.
    """
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            continue
        series = out[column]
        if pd.api.types.is_integer_dtype(series):
            series = pd.to_datetime(series, unit="ns", utc=True)
        elif not pd.api.types.is_datetime64_any_dtype(series):
            series = pd.to_datetime(series, utc=True)
        elif series.dt.tz is None:
            series = series.dt.tz_localize(UTC)
        out[column] = series.dt.tz_convert(ET)
    return out


def session_bounds(day: date, start: time, end: time) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Session window as tz-aware Eastern timestamps."""
    lo = pd.Timestamp(datetime.combine(day, start)).tz_localize(ET)
    hi = pd.Timestamp(datetime.combine(day, end)).tz_localize(ET)
    return lo, hi


class Fetcher(Protocol):
    def fetch(
        self, schema: str, day: date, symbols: list[str] | None,
        start: time, end: time, stype_in: str,
    ) -> tuple[pd.DataFrame, Pull]: ...


class DatabentoFetcher:
    """Fetches via the official ``databento`` SDK, gated on cost and cached.

    The SDK is imported lazily so the rest of the harness -- signals, fills,
    reporting, and every test -- runs without the dependency or an API key.
    """

    def __init__(self, cfg: BacktestConfig, cache: ParquetCache, client=None, echo=print):
        self.cfg = cfg
        self.cache = cache
        self._client = client
        self.echo = echo
        self.pulls: list[Pull] = []

    @property
    def client(self):
        if self._client is None:
            try:
                import databento as db
            except ImportError as exc:  # pragma: no cover - depends on env
                raise SystemExit(
                    "the databento SDK is not installed. `pip install databento` and set "
                    "DATABENTO_API_KEY, or use --offline to run against cached parquet only."
                ) from exc
            self._client = db.Historical()
        return self._client

    def request_kwargs(
        self, schema: str, day: date, symbols: list[str] | None,
        start: time, end: time, stype_in: str,
    ) -> dict:
        """The arguments shared by ``get_cost`` and ``get_range``.

        Built in one place so the estimate and the pull can never describe
        different requests -- a cost quoted for a narrower window than the one
        actually fetched would make the whole gate decorative. It is also what
        the SDK-conformance test binds against.
        """
        lo, hi = session_bounds(day, start, end)
        return {
            "dataset": self.cfg.data.dataset,
            "schema": schema,
            "symbols": symbols or "ALL_SYMBOLS",
            "stype_in": stype_in,
            "start": lo.tz_convert(UTC).to_pydatetime(),
            "end": hi.tz_convert(UTC).to_pydatetime(),
        }

    def estimate_cost(
        self, schema: str, day: date, symbols: list[str] | None,
        start: time, end: time, stype_in: str,
    ) -> float | None:
        try:
            return float(self.client.metadata.get_cost(
                **self.request_kwargs(schema, day, symbols, start, end, stype_in)
            ))
        except Exception as exc:
            log.warning("cost estimate failed for %s %s: %s", schema, day, exc)
            return None

    def fetch(
        self, schema: str, day: date, symbols: list[str] | None,
        start: time, end: time, stype_in: str = "raw_symbol",
    ) -> tuple[pd.DataFrame, Pull]:
        key = CacheKey.build(self.cfg.data.dataset, schema, day, symbols)

        # Rule 3: a cached range is never re-pulled, and the cost endpoint is
        # not even consulted -- an estimate for data we already hold is noise.
        if self.cache.has(key):
            frame = self.cache.read(key)
            pull = Pull(self.cfg.data.dataset, schema, day, symbols, len(frame), None, True)
            self.pulls.append(pull)
            self.echo(f"  cache hit  {pull.describe()}")
            return frame, pull

        describe = f"{schema} {day} ({'all' if not symbols else len(symbols)} symbols)"
        estimate = self.estimate_cost(schema, day, symbols, start, end, stype_in)

        # Rule 1: print the estimate before every pull, whatever it says.
        if estimate is None:
            self.echo(f"  COST ESTIMATE UNAVAILABLE for {describe}")
            if self.cfg.cost.require_estimate:
                raise CostEstimateUnavailable(
                    f"could not price {describe}, and cost.require_estimate is on. "
                    f"An unknown cost is not a zero cost."
                )
        else:
            self.echo(f"  cost estimate ${estimate:,.4f} for {describe}")
            if estimate > self.cfg.cost.ceiling_usd:
                raise CostCeilingExceeded(estimate, self.cfg.cost.ceiling_usd, describe)

        data = self.client.timeseries.get_range(
            **self.request_kwargs(schema, day, symbols, start, end, stype_in)
        )
        # These three are SDK defaults today, passed explicitly so a future
        # default change cannot silently alter prices, drop the symbol column,
        # or hand us timestamps in a zone `to_eastern` is not expecting.
        #   price_type=float  -> prices as floats, not 1e-9 fixed point
        #   map_symbols       -> adds the `symbol` column QuoteBook keys on
        #   tz=UTC            -> the input `to_eastern` converts from
        # to_df() indexes on ts_recv, so reset_index() is required to make it a
        # column rather than losing it into the index.
        frame = data.to_df(price_type="float", map_symbols=True, tz=UTC).reset_index()
        frame = to_eastern(frame)

        pull = Pull(self.cfg.data.dataset, schema, day, symbols, len(frame), estimate, False)
        self.cache.write(key, frame, meta={"cost_usd": estimate, "stype_in": stype_in})
        self.pulls.append(pull)
        return frame, pull


class AgentBridgeFetcher:
    """Reads only from cache; never fetches.

    Use when an agent (via the Databento MCP server) has written the parquet
    files. A miss is an error rather than a pull, so this mode can never spend
    money by accident -- which is the entire point of separating it from the
    SDK fetcher rather than adding a flag.
    """

    def __init__(self, cfg: BacktestConfig, cache: ParquetCache, echo=print):
        self.cfg = cfg
        self.cache = cache
        self.echo = echo
        self.pulls: list[Pull] = []

    def fetch(
        self, schema: str, day: date, symbols: list[str] | None,
        start: time, end: time, stype_in: str = "raw_symbol",
    ) -> tuple[pd.DataFrame, Pull]:
        key = CacheKey.build(self.cfg.data.dataset, schema, day, symbols)
        if not self.cache.has(key):
            raise KeyError(
                f"offline mode: no cached {schema} for {day} at {self.cache.path_for(key)}. "
                f"Populate the cache first, or run with the SDK fetcher."
            )
        frame = self.cache.read(key)
        pull = Pull(self.cfg.data.dataset, schema, day, symbols, len(frame), None, True)
        self.pulls.append(pull)
        self.echo(f"  cache hit  {pull.describe()}")
        return frame, pull
