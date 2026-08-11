"""Does the range model pick spreads that were actually sellable?

Calibration proves the model's confidence means what it says about the *index*.
It says nothing about whether a trade exists at the level it nominates, and
that gap is where a plausible 0DTE strategy usually dies: the strike a
95%-confident model wants is far enough out that the credit can be a few cents,
or bidless, or wide enough that crossing the spread costs more than the edge.

So this layer is the first thing in the harness that spends real OPRA money,
and it asks only two questions per session:

1. At the entry moment, what credit was genuinely available at the strike the
   model chose -- reading the bid a seller would actually hit, not a midpoint.
2. Did the close respect that strike, and how does the breach rate compare with
   the confidence that selected it?

It runs at one entry time per session and holds to settlement. That is not a
trading strategy -- there is no stop, no management, no sizing -- it is the
narrowest test that can falsify the model economically, on the fewest sessions
of paid data. A strategy is only worth building on top of a model that survives
this.

`cbbo-1m` rather than `cmbp-1`: one-minute consolidated BBO covers the whole
SPXW chain for ~$1.13 a session, against ~$17.69 for full-depth quotes, and an
entry decision made once at 11:00 cannot use sub-minute resolution anyway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, time

import pandas as pd

from .data import (
    CALL,
    PUT,
    Contract,
    QuoteBook,
    contracts_from_definitions,
    spot_from_parity,
    year_fraction_to_close,
)
from .source import ET, session_bounds

log = logging.getLogger(__name__)

SCHEMA_VALIDATION_QUOTES = "cbbo-1m"


@dataclass
class SpreadCheck:
    """One session's answer to both questions."""

    day: date
    entry_ts: pd.Timestamp
    side: str
    spot: float
    model_level: float        # what the model asked for
    short_strike: float       # nearest listed strike that respects it
    long_strike: float
    credit: float             # index points actually receivable
    settlement: float | None
    breached: bool | None
    note: str = ""
    # The same spread marked at both midpoints. Not a tradable price -- nobody
    # is obliged to fill there -- but the gap between this and ``credit`` is
    # what crossing the book costs, which separates "the market does not pay
    # enough" from "the execution is eating it". Only the first of those is
    # fatal; the second is a problem you can work on.
    credit_mid: float = 0.0

    @property
    def wing_width(self) -> float:
        """The width actually obtained, which is not always the one requested.

        SPX lists 5-point strikes near the money and sparser ones further out,
        so a 5-point wing asked for 70 points OTM can come back 25 wide. That
        is five times the risk under the same label, so every number derived
        from width reads this rather than the configured value.
        """
        return abs(self.short_strike - self.long_strike)

    @property
    def max_loss(self) -> float:
        """Defined risk in index points, from the width actually obtained."""
        return self.wing_width - self.credit

    @property
    def tradable(self) -> bool:
        return self.credit > 0 and not self.note


def pick_sessions(days: list[date], count: int) -> list[date]:
    """Evenly spaced sample across the range.

    Even spacing rather than a random draw, and certainly rather than a hand
    picked set: it is reproducible, it cannot be re-rolled until the answer
    flatters the model, and it spreads the sample across volatility regimes
    instead of clustering in whichever weeks happened to be sampled.
    """
    if count >= len(days):
        return list(days)
    step = len(days) / count
    return [days[int(i * step)] for i in range(count)]


def nearest_listed(
    contracts: list[Contract], option_type: str, level: float, side: str,
) -> Contract | None:
    """The listed strike a seller would actually choose for ``level``.

    Rounded *away* from spot, never toward it. Rounding toward spot would hand
    the backtest a strike the model did not ask for and a fatter credit than it
    earned -- a small bias that points consistently in the flattering direction.
    """
    candidates = [c for c in contracts if c.option_type == option_type]
    if not candidates:
        return None
    if side == "put":
        eligible = [c for c in candidates if c.strike <= level]
        return max(eligible, key=lambda c: c.strike) if eligible else None
    eligible = [c for c in candidates if c.strike >= level]
    return min(eligible, key=lambda c: c.strike) if eligible else None


def wing(contracts: list[Contract], option_type: str, short: Contract,
         width: float) -> Contract | None:
    """The long leg, ``width`` points further out from the short."""
    target = short.strike - width if option_type == PUT else short.strike + width
    candidates = [c for c in contracts if c.option_type == option_type]
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(c.strike - target))


def check_session(
    day: date, contracts: list[Contract], book: QuoteBook, entry_ts: pd.Timestamp,
    z_return: float, side: str, width: float, r: float, q: float,
) -> SpreadCheck:
    """Price the model's chosen spread against the real book for one session.

    ``z_return`` is the model's remaining-day move as a fraction, which is
    applied to the *index* level implied by put-call parity rather than to the
    ES price the model was fitted on. ES and SPX differ by a basis that is
    irrelevant to a return but very relevant to a strike.
    """
    option_type = PUT if side == "put" else CALL
    quotes = book.snapshot(entry_ts)
    if not quotes:
        return _empty(day, entry_ts, side, "no quotes at entry")

    T = year_fraction_to_close(entry_ts, day)
    spot = spot_from_parity(quotes, contracts, r, T)
    if spot is None:
        return _empty(day, entry_ts, side, "no ATM pair to imply spot")

    level = spot * (1.0 + z_return)
    short = nearest_listed(contracts, option_type, level, side)
    if short is None:
        return _empty(day, entry_ts, side, f"no listed {side} strike at {level:,.0f}",
                      spot=spot, model_level=level)

    long_leg = wing(contracts, option_type, short, width)
    if long_leg is None or long_leg.strike == short.strike:
        return _empty(day, entry_ts, side, "no wing at the configured width",
                      spot=spot, model_level=level)

    short_quote = quotes.get(short.symbol)
    long_quote = quotes.get(long_leg.symbol)
    if short_quote is None or long_quote is None:
        return _empty(day, entry_ts, side, "a leg had no quote",
                      spot=spot, model_level=level)
    if not short_quote.can_sell:
        return _empty(day, entry_ts, side, "short leg was bidless",
                      spot=spot, model_level=level)

    # Sell the short at its bid, buy the wing at its ask. Nothing here is
    # marked at a midpoint: the whole point of spending on real quotes is to
    # find out what a taker would have received.
    credit = short_quote.bid - long_quote.ask
    credit_mid = short_quote.mid - long_quote.mid

    settlement = _settlement(book, contracts, day, r)
    if settlement is None:
        return _empty(day, entry_ts, side, "could not imply a settlement level",
                      spot=spot, model_level=level)
    breached = None
    if settlement is not None:
        breached = (settlement < short.strike if side == "put"
                    else settlement > short.strike)

    return SpreadCheck(
        day=day, entry_ts=entry_ts, side=side, spot=spot, model_level=level,
        short_strike=short.strike, long_strike=long_leg.strike,
        credit=credit, settlement=settlement, breached=breached,
        credit_mid=credit_mid,
    )


def _settlement(book: QuoteBook, contracts: list[Contract], day: date,
                r: float) -> float | None:
    """Index level implied by the chain at the last quoted minute.

    SPXW is cash-settled against the 16:00 index, and parity on the closing
    book is the closest thing to it that the data already paid for holds.
    """
    last = book.last_ts()
    if last is None:
        return None
    return spot_from_parity(book.snapshot(last), contracts, r,
                            year_fraction_to_close(last, day))


def _empty(day, entry_ts, side, note, spot=float("nan"),
           model_level=float("nan")) -> SpreadCheck:
    return SpreadCheck(
        day=day, entry_ts=entry_ts, side=side, spot=spot, model_level=model_level,
        short_strike=float("nan"), long_strike=float("nan"), credit=0.0,
        settlement=None, breached=None, note=note,
    )


@dataclass
class ValidationSummary:
    checks: list[SpreadCheck]
    confidence: float

    @property
    def tradable(self) -> list[SpreadCheck]:
        return [c for c in self.checks if c.tradable]

    @property
    def resolved(self) -> list[SpreadCheck]:
        return [c for c in self.tradable if c.breached is not None]

    @property
    def breaches(self) -> int:
        return sum(1 for c in self.resolved if c.breached)

    @property
    def breach_rate(self) -> float:
        return self.breaches / len(self.resolved) if self.resolved else float("nan")

    @property
    def expected_breach_rate(self) -> float:
        return 1.0 - self.confidence

    @property
    def median_credit(self) -> float:
        return _median([c.credit for c in self.tradable])

    @property
    def median_width(self) -> float:
        return _median([c.wing_width for c in self.tradable])

    def wrong_width(self, requested: float) -> list[SpreadCheck]:
        """Sessions where the chain could not supply the requested wing."""
        return [c for c in self.tradable if abs(c.wing_width - requested) > 1e-6]

    def expectancy(self) -> float | None:
        """Points per spread, from what actually happened -- no model input.

        A breach is charged its real cost rather than the full width: 0DTE SPX
        settles in cash, so a close between the strikes is a partial loss, and
        assuming total loss on every breach understates the strategy by more
        than the credit it is trying to measure.
        """
        resolved = self.resolved
        if not resolved:
            return None
        total = 0.0
        for c in resolved:
            if not c.breached:
                total += c.credit
                continue
            if c.side == "put":
                intrinsic = min(max(c.short_strike - c.settlement, 0.0), c.wing_width)
            else:
                intrinsic = min(max(c.settlement - c.short_strike, 0.0), c.wing_width)
            total += c.credit - intrinsic
        return total / len(resolved)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
