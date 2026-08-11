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
converts from. ``TestVendorAgreesWithOurConstants`` asks the server the
questions argument-binding cannot answer, which is how a request for ``mbp-1``
-- a schema OPRA does not offer -- survived 334 passing tests.

Live behaviour is now partly verified: GLBX bar pulls have been made and parsed
against this code. OPRA bulk quote responses have not.

What the gate cannot do: ``get_cost`` prices a request, but Databento exposes
no balance endpoint, so an accurately-priced under-ceiling pull can still be
refused at purchase for want of funds. That arrives as ``BudgetExhausted``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Protocol

import pandas as pd

from .cache import CacheKey, ParquetCache
from .config import SCHEMA_DEFINITION, BacktestConfig

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


class BudgetExhausted(RuntimeError):
    """The vendor refused the request for lack of funds, not for being wrong.

    This is the gap the cost gate structurally cannot close. `get_cost` prices
    a request; there is no endpoint that reports the account's remaining
    balance, so a correctly-priced, under-ceiling, thoroughly-sized request can
    still be refused at the moment of purchase. Worth its own exception because
    the remedy is neither "narrow the request" nor "retry" -- it is to add funds
    -- and because a raw 402 traceback in the middle of a long pull reads like a
    bug in the harness rather than a bill.
    """

    def __init__(self, describe: str, estimate: float | None):
        self.estimate = estimate
        priced = f"priced at ${estimate:,.4f}" if estimate is not None else "unpriced"
        super().__init__(
            f"Databento refused {describe} ({priced}): the account has "
            f"insufficient budget. Nothing was pulled and nothing was billed for "
            f"it. Add funds or raise the budget cap at "
            f"https://databento.com/portal/billing, then re-run -- cached pulls "
            f"are not repeated, so this resumes where it stopped."
        )


def is_insufficient_funds(exc: Exception) -> bool:
    return getattr(exc, "http_status", None) == 402


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


def definition_bounds(day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Full UTC day, which is the only window that returns a complete chain.

    Definitions are not a stream of intraday events but a snapshot stamped at
    the start of the UTC day. Requesting them over the trading session instead
    -- 13:45 UTC onward -- silently returns whatever definitions happened to be
    restated later in the day, so contracts listed at the open go missing and
    the chain comes back short. The SDK warns about this; the warning is right.
    """
    lo = pd.Timestamp(day, tz="UTC")
    return lo, lo + pd.Timedelta(days=1)


# Failures that report no HTTP status because the connection, not the request,
# is what broke. Matched on the message because the SDK raises a bare
# ``BentoError`` for all of them.
TRANSIENT_PHRASES = (
    "ended prematurely",
    "streaming response",
    "connection reset",
    "connection aborted",
    "timed out",
    # http.client spells it without a space; keep both so neither slips past.
    "incompleteread",
    "incomplete read",
)


def is_transient(exc: Exception) -> bool:
    """Whether a failed request is worth repeating.

    `BentoServerError` is the SDK's 5xx class -- the gateway timing out on a
    large range says nothing about whether the request was valid, and a
    multi-month pull is long enough that hitting one is routine. A 4xx is the
    opposite: the request is wrong and repeating it wastes time and money.
    """
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    status = getattr(exc, "http_status", None)
    if isinstance(status, int):
        return 500 <= status < 600
    # A truncated stream carries no status at all: the request was accepted and
    # the connection died mid-download. Multi-hundred-megabyte chain pulls hit
    # this, and treating it as fatal throws away every session after it.
    return any(phrase in str(exc).lower() for phrase in TRANSIENT_PHRASES)


def with_retry(call, attempts: int = 5, base_delay: float = 2.0, echo=print):
    """Run ``call``, repeating transient failures with exponential backoff.

    Deliberately not applied to cost estimation: ``estimate_cost`` already
    treats failure as "unpriced", and an unpriced pull is refused rather than
    retried into a purchase.
    """
    import time as _time

    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == attempts or not is_transient(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            echo(f"  transient {type(exc).__name__} ({exc}); "
                 f"retry {attempt}/{attempts - 1} in {delay:.0f}s")
            _time.sleep(delay)


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
        if schema == SCHEMA_DEFINITION:
            lo, hi = definition_bounds(day)
        else:
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

        try:
            data = with_retry(
                lambda: self.client.timeseries.get_range(
                    **self.request_kwargs(schema, day, symbols, start, end, stype_in)
                ),
                echo=self.echo,
            )
        except Exception as exc:
            if is_insufficient_funds(exc):
                raise BudgetExhausted(describe, estimate) from exc
            raise
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

    def available_end(self, dataset: str) -> pd.Timestamp | None:
        """Last timestamp the entitlement actually covers, or ``None``.

        A dataset's live edge sits at whatever minute it was last appended to,
        not at a day boundary, and asking past it is a 422 for the whole
        request rather than a short frame. Callers building calendar-aligned
        windows need this to clamp the final one; a range that ends a few hours
        into the future fails a pull that is otherwise entirely valid.
        """
        try:
            span = self.client.metadata.get_dataset_range(dataset=dataset)
        except Exception as exc:
            log.warning("could not read the available range for %s: %s", dataset, exc)
            return None
        end = span.get("end") if isinstance(span, dict) else None
        return pd.Timestamp(end).tz_convert(ET) if end else None

    # --- arbitrary windows ------------------------------------------------

    def fetch_window(
        self, dataset: str, schema: str, symbols: list[str] | None,
        lo: pd.Timestamp, hi: pd.Timestamp, stype_in: str, key_day: date,
    ) -> tuple[pd.DataFrame, Pull]:
        """Pull one explicit time window from any dataset, under the same gate.

        ``fetch`` is shaped around one option session: a day, a session window,
        and the OPRA dataset from config. Underlying bars are neither -- they
        come from a different dataset and are far cheaper per unit time, so
        requesting them a day at a time would mean hundreds of round trips for
        data that costs fractions of a cent. The caller chooses the chunking and
        supplies ``key_day`` as the cache identity for the chunk.

        The cost gate, the cache-first rule and the spend manifest are the same
        code path as ``fetch``; only the window construction differs.
        """
        key = CacheKey.build(dataset, schema, key_day, symbols)
        if self.cache.has(key):
            frame = self.cache.read(key)
            pull = Pull(dataset, schema, key_day, symbols, len(frame), None, True)
            self.pulls.append(pull)
            self.echo(f"  cache hit  {pull.describe()}")
            return frame, pull

        kwargs = {
            "dataset": dataset,
            "schema": schema,
            "symbols": symbols or "ALL_SYMBOLS",
            "stype_in": stype_in,
            "start": lo.tz_convert(UTC).to_pydatetime(),
            "end": hi.tz_convert(UTC).to_pydatetime(),
        }
        describe = f"{schema} {lo.date()}..{hi.date()} on {dataset}"

        try:
            estimate = float(self.client.metadata.get_cost(**kwargs))
        except Exception as exc:
            log.warning("cost estimate failed for %s: %s", describe, exc)
            estimate = None

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

        try:
            data = with_retry(
                lambda: self.client.timeseries.get_range(**kwargs), echo=self.echo
            )
        except Exception as exc:
            if is_insufficient_funds(exc):
                raise BudgetExhausted(describe, estimate) from exc
            raise
        frame = data.to_df(price_type="float", map_symbols=True, tz=UTC).reset_index()
        frame = to_eastern(frame)

        pull = Pull(dataset, schema, key_day, symbols, len(frame), estimate, False)
        self.cache.write(key, frame, meta={"cost_usd": estimate, "stype_in": stype_in,
                                           "window": f"{lo.isoformat()}..{hi.isoformat()}"})
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

    def fetch_window(
        self, dataset: str, schema: str, symbols: list[str] | None,
        lo: pd.Timestamp, hi: pd.Timestamp, stype_in: str, key_day: date,
    ) -> tuple[pd.DataFrame, Pull]:
        """Cache-only counterpart of the SDK fetcher's window pull.

        Without this, offline mode cannot read underlying bars at all -- which
        defeats the point of a mode whose whole purpose is to re-analyse data
        already paid for without touching the network.
        """
        key = CacheKey.build(dataset, schema, key_day, symbols)
        if not self.cache.has(key):
            raise KeyError(
                f"offline mode: no cached {schema} for {key_day} at "
                f"{self.cache.path_for(key)}. Populate the cache first, or run "
                f"with the SDK fetcher."
            )
        frame = self.cache.read(key)
        pull = Pull(dataset, schema, key_day, symbols, len(frame), None, True)
        self.pulls.append(pull)
        self.echo(f"  cache hit  {pull.describe()}")
        return frame, pull
