"""DATA LAYER: normalized views over raw Databento frames.

Everything above this module sees ``QuoteBook`` and ``Contract``. Nothing above
it sees a Databento column name, a fixed-point price, or a UTC timestamp.
"""

from __future__ import annotations

import logging
import math
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime, time

import pandas as pd

from .source import ET

log = logging.getLogger(__name__)

CALL, PUT = "call", "put"

# OSI: root padded to 6, YYMMDD, C/P, strike x 1000 in 8 digits.
OSI_RE = re.compile(r"^(?P<root>[A-Z]{1,6})\s*(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")

# Databento reports prices as int nanos (1e-9). ``to_df()`` usually converts,
# but the raw path does not, so anything implausibly large is rescaled.
FIXED_POINT_SCALE = 1e-9
IMPLAUSIBLE_PRICE = 1e6


@dataclass(frozen=True)
class Contract:
    """One listed option, in both spellings of its name.

    ``symbol`` is canonical and is what every internal lookup uses. ``raw`` is
    the vendor's own spelling, preserved exactly as received, and is what must
    be sent back to Databento as a ``raw_symbol``: OPRA pads the root to six
    characters, and a request built from the canonical form asks for symbols
    the feed does not recognise.
    """

    symbol: str
    root: str
    expiration: date
    option_type: str
    strike: float
    raw: str = ""

    def __post_init__(self):
        if not self.raw:
            object.__setattr__(self, "raw", self.symbol)

    @classmethod
    def parse(cls, symbol: str) -> "Contract | None":
        cleaned = symbol.strip().upper()
        match = OSI_RE.match(cleaned.replace(" ", ""))
        if not match:
            return None
        ymd = match.group("ymd")
        try:
            expiry = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
        except ValueError:
            return None
        return cls(
            symbol=cleaned.replace(" ", ""),
            root=match.group("root"),
            expiration=expiry,
            option_type=CALL if match.group("cp") == "C" else PUT,
            strike=int(match.group("strike")) / 1000.0,
            raw=cleaned,
        )


def osi_symbol(root: str, expiry: date, option_type: str, strike: float) -> str:
    cp = "C" if option_type == CALL else "P"
    return f"{root}{expiry:%y%m%d}{cp}{int(round(strike * 1000)):08d}"


def canonical_symbol(symbol: str) -> str:
    """One spelling of an OSI symbol, so two feeds can be compared.

    OPRA pads the root to six characters -- ``SPXW  260202P06300000`` -- while
    ``Contract.parse`` and ``osi_symbol`` both normalise the padding away. Keying
    quotes on the vendor's spelling and looking them up by the parsed one means
    every lookup misses silently: the chain is present, the quotes are present,
    and not one contract finds its price. Every symbol crossing a boundary goes
    through here.
    """
    return symbol.strip().upper().replace(" ", "")


@dataclass(frozen=True)
class Quote:
    ts: pd.Timestamp
    bid: float
    ask: float
    bid_size: int = 0
    ask_size: int = 0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def is_crossed(self) -> bool:
        return self.ask < self.bid

    @property
    def can_sell(self) -> bool:
        """Someone is bidding, so a short can actually be opened.

        Treating a bidless contract as sellable at its mid is the single most
        common way a 0DTE backtest invents premium that never existed.
        """
        return self.bid > 0 and not self.is_crossed

    @property
    def can_buy(self) -> bool:
        """Someone is offering, so a long can be opened.

        Deliberately *not* the same test as ``can_sell``. A far-OTM wing quoted
        0.00 x 0.05 is perfectly buyable -- you pay the nickel. Demanding a
        non-zero bid on a leg you are buying rejects exactly the cheap wings a
        premium seller wants, which silently drops the calmest sessions from the
        sample and biases the result toward high-volatility days.
        """
        return self.ask > 0 and not self.is_crossed

    @property
    def is_tradable(self) -> bool:
        """Tradable in both directions."""
        return self.can_sell and self.can_buy


def _rescale(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.abs().max(skipna=True) is not None and numeric.abs().max(skipna=True) > IMPLAUSIBLE_PRICE:
        return numeric * FIXED_POINT_SCALE
    return numeric


class QuoteBook:
    """Time-ordered top-of-book per symbol, with as-of lookup.

    As-of rather than nearest, always. A backtest that reaches for the closest
    quote in either direction is reading prices from the future, and at 0DTE a
    few seconds of lookahead around a spike is worth more than the whole edge
    being measured.
    """

    def __init__(self, frame: pd.DataFrame):
        self._by_symbol: dict[str, list[Quote]] = {}
        self._times: dict[str, list[pd.Timestamp]] = {}
        if frame is None or frame.empty:
            return

        work = frame.copy()
        if "symbol" not in work.columns:
            raise ValueError("quote frame has no 'symbol' column")
        ts_col = "ts_recv" if "ts_recv" in work.columns else "ts_event"
        if ts_col not in work.columns:
            raise ValueError("quote frame has no ts_recv or ts_event column")

        for col in ("bid_px_00", "ask_px_00"):
            if col in work.columns:
                work[col] = _rescale(work[col])

        # Canonical keys: the vendor pads the root, the parser does not, and a
        # book keyed one way but read the other answers every lookup with None.
        work["symbol"] = work["symbol"].astype(str).map(canonical_symbol)

        work = work.sort_values(ts_col)
        # Column arrays rather than row objects. A full-chain session is several
        # million rows, and `iterrows` materialises a Series per row -- enough to
        # turn one session into minutes of pure overhead.
        for column, default in (("bid_px_00", 0.0), ("ask_px_00", 0.0),
                                ("bid_sz_00", 0), ("ask_sz_00", 0)):
            if column not in work.columns:
                work[column] = default
        work = work.fillna({"bid_px_00": 0.0, "ask_px_00": 0.0,
                            "bid_sz_00": 0, "ask_sz_00": 0})

        for symbol, group in work.groupby("symbol", sort=False):
            stamps = list(group[ts_col])
            quotes = [
                Quote(ts=ts, bid=float(bid), ask=float(ask),
                      bid_size=int(bsz), ask_size=int(asz))
                for ts, bid, ask, bsz, asz in zip(
                    stamps,
                    group["bid_px_00"].to_numpy(),
                    group["ask_px_00"].to_numpy(),
                    group["bid_sz_00"].to_numpy(),
                    group["ask_sz_00"].to_numpy(),
                )
            ]
            self._by_symbol[symbol] = quotes
            self._times[symbol] = stamps

    @property
    def symbols(self) -> list[str]:
        return sorted(self._by_symbol)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_symbol.values())

    def as_of(self, symbol: str, ts: pd.Timestamp) -> Quote | None:
        """Last quote at or before ``ts``. ``None`` if the book had not opened."""
        key = canonical_symbol(symbol)
        times = self._times.get(key)
        if not times:
            return None
        idx = bisect_right(times, ts) - 1
        if idx < 0:
            return None
        return self._by_symbol[key][idx]

    def snapshot(self, ts: pd.Timestamp) -> dict[str, Quote]:
        out = {}
        for symbol in self._by_symbol:
            quote = self.as_of(symbol, ts)
            if quote is not None:
                out[symbol] = quote
        return out

    def timeline(self, start: pd.Timestamp, end: pd.Timestamp, step_seconds: int = 60) -> list[pd.Timestamp]:
        return list(pd.date_range(start, end, freq=f"{step_seconds}s", tz=ET))

    def last_ts(self) -> pd.Timestamp | None:
        alls = [t[-1] for t in self._times.values() if t]
        return max(alls) if alls else None


def contracts_from_definitions(frame: pd.DataFrame, root: str, expiry: date) -> list[Contract]:
    """Extract the tradable chain for one expiry from a definition frame."""
    if frame is None or frame.empty:
        return []
    symbols = frame["raw_symbol"] if "raw_symbol" in frame.columns else frame.get("symbol")
    if symbols is None:
        return []
    out = []
    for symbol in symbols.dropna().unique():
        contract = Contract.parse(str(symbol))
        if contract and contract.root == root and contract.expiration == expiry:
            out.append(contract)
    return sorted(out, key=lambda c: (c.option_type, c.strike))


def spot_from_parity(
    quotes: dict[str, Quote], contracts: list[Contract], r: float, T: float
) -> float | None:
    """Derive the underlying from put-call parity on the most ATM strike pair.

    F = K + e^{rT}(C - P). This costs nothing extra: it reads the option quotes
    already paid for, rather than buying a second feed for the index. It is also
    the *right* number for pricing these contracts -- the synthetic forward the
    options are actually quoted against, which on SPX drifts from cash spot by
    the dividend and financing that parity already embeds.

    The strike used is the one whose call and put mids are closest together,
    which is the standard ATM identification and is robust to a stale wing.
    """
    by_strike: dict[float, dict[str, Quote]] = {}
    for contract in contracts:
        quote = quotes.get(contract.symbol)
        if quote and quote.is_tradable:
            by_strike.setdefault(contract.strike, {})[contract.option_type] = quote

    pairs = [(k, v[CALL], v[PUT]) for k, v in by_strike.items() if CALL in v and PUT in v]
    if not pairs:
        return None

    strike, call, put = min(pairs, key=lambda p: abs(p[1].mid - p[2].mid))
    return strike + math.exp(r * max(T, 0.0)) * (call.mid - put.mid)


def year_fraction_to_close(ts: pd.Timestamp, day: date) -> float:
    """Years from ``ts`` to the 16:00 ET cash settlement of SPXW."""
    close = pd.Timestamp(datetime.combine(day, time(16, 0))).tz_localize(ET)
    seconds = (close - ts).total_seconds()
    return max(seconds, 60.0) / (365.0 * 24.0 * 3600.0)
