"""SIGNAL LAYER: which strikes, and when to get out.

Rule 6: nothing here knows about slippage, commissions, or which side of the
book gets crossed. Strike selection reads midpoints because that is the market's
own estimate of value; what you would actually be *filled* at is the fill
layer's problem, and keeping them apart is what lets you re-run a period under a
different execution assumption and get a comparable answer.

Exit rules are expressed against a mark that the engine supplies. They do not
compute it -- the mark is a fill-layer question -- but they do react to it,
which is the honest split: "stop at twice the credit" is a strategy rule, while
"the mark is the ask because you must buy it back" is an execution one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, time

import pandas as pd

from spreadscout.pricing import bs_delta, implied_vol

from .config import SignalConfig
from .data import CALL, PUT, Contract, Quote, osi_symbol, year_fraction_to_close
from .fills import BUY, SELL, Leg, Structure

log = logging.getLogger(__name__)

STOP_LOSS = "stop_loss"
PROFIT_TARGET = "profit_target"
TIME_EXIT = "time_exit"
SETTLEMENT = "settlement"
NO_QUOTE = "no_quote"


@dataclass
class Greeked:
    contract: Contract
    quote: Quote
    iv: float
    delta: float


def solve_greeks(
    contracts: list[Contract], quotes: dict[str, Quote], spot: float, T: float,
    r: float, q: float,
) -> list[Greeked]:
    """Back out IV and delta from live mids.

    Vendor greeks are not used even where a feed supplies them: on a 0DTE chain
    a batch computed hours earlier describes an option with a materially
    different life left. Solving from the quote we are about to trade against
    keeps the selection and the fill consistent.
    """
    out = []
    for contract in contracts:
        quote = quotes.get(contract.symbol)
        if quote is None or not quote.is_tradable:
            continue
        iv = implied_vol(quote.mid, spot, contract.strike, T, r, q, contract.option_type)
        if iv is None:
            continue  # no extrinsic value left; delta is unrecoverable, not zero
        delta = bs_delta(spot, contract.strike, T, r, q, iv, contract.option_type)
        out.append(Greeked(contract=contract, quote=quote, iv=iv, delta=delta))
    return out


def nearest_by_delta(candidates: list[Greeked], target: float, option_type: str) -> Greeked | None:
    """Closest OTM strike to the target delta magnitude."""
    pool = [g for g in candidates if g.contract.option_type == option_type]
    if not pool:
        return None
    return min(pool, key=lambda g: abs(abs(g.delta) - target))


def build_structure(
    contracts: list[Contract], quotes: dict[str, Quote], spot: float, T: float,
    cfg: SignalConfig, r: float, q: float, root: str, expiry: date,
) -> tuple[Structure | None, str]:
    """Select strikes. Returns ``(structure, reason_if_none)``."""
    greeked = solve_greeks(contracts, quotes, spot, T, r, q)
    if not greeked:
        return None, "no contracts with solvable implied vol"

    otm_puts = [g for g in greeked if g.contract.option_type == PUT and g.contract.strike < spot]
    otm_calls = [g for g in greeked if g.contract.option_type == CALL and g.contract.strike > spot]

    legs: list[Leg] = []
    width = cfg.width_points

    def wing(short: Greeked, option_type: str, direction: int) -> tuple[Leg, Leg] | None:
        long_strike = short.contract.strike + direction * width
        long_symbol = osi_symbol(root, expiry, option_type, long_strike)
        long_quote = quotes.get(long_symbol)
        # The wing is bought, so it needs an offer, not a bid. A 0.00 x 0.05
        # market is a perfectly good wing at a nickel.
        if long_quote is None or not long_quote.can_buy:
            return None
        return (
            Leg(short.contract.symbol, SELL, short.contract.strike, option_type),
            Leg(long_symbol, BUY, long_strike, option_type),
        )

    if cfg.structure in ("iron_condor", "put_credit"):
        short_put = nearest_by_delta(otm_puts, cfg.target_short_delta, PUT)
        if short_put is None:
            return None, "no OTM put near the target delta"
        pair = wing(short_put, PUT, -1)
        if pair is None:
            return None, f"long put wing {short_put.contract.strike - width:g} not quotable"
        legs.extend(pair)

    if cfg.structure in ("iron_condor", "call_credit"):
        short_call = nearest_by_delta(otm_calls, cfg.target_short_delta, CALL)
        if short_call is None:
            return None, "no OTM call near the target delta"
        pair = wing(short_call, CALL, +1)
        if pair is None:
            return None, f"long call wing {short_call.contract.strike + width:g} not quotable"
        legs.extend(pair)

    if not legs:
        return None, f"no legs built for structure {cfg.structure!r}"

    return Structure(kind=cfg.structure, legs=tuple(legs), width=width), ""


@dataclass
class ExitDecision:
    should_exit: bool
    reason: str = ""


def check_exit(
    ts: pd.Timestamp, credit: float, current_debit: float | None, cfg: SignalConfig,
    day: date,
) -> ExitDecision:
    """Decide whether to flatten, given a mark supplied by the engine.

    Order matters and is deliberate: the stop is checked before the target. When
    a bar could plausibly have hit both, assuming the loss happened first is the
    conservative reading, and at 0DTE a strike can traverse both thresholds
    between two quote updates.
    """
    hard_stop = pd.Timestamp(
        pd.Timestamp(day).to_pydatetime().replace(
            hour=cfg.exit_time.hour, minute=cfg.exit_time.minute
        )
    ).tz_localize(ts.tz)

    if current_debit is None:
        # The book went dark. Flattening on a stale mark would be inventing a
        # price; the engine falls back to settlement for this case.
        if ts >= hard_stop:
            return ExitDecision(True, NO_QUOTE)
        return ExitDecision(False)

    loss = current_debit - credit
    if credit > 0 and loss >= cfg.stop_loss_multiple * credit:
        return ExitDecision(True, STOP_LOSS)

    if credit > 0 and current_debit <= credit * (1.0 - cfg.profit_target_fraction):
        return ExitDecision(True, PROFIT_TARGET)

    if ts >= hard_stop and not cfg.hold_to_settlement:
        return ExitDecision(True, TIME_EXIT)

    return ExitDecision(False)


def entry_timestamp(day: date, at: time, tz=None) -> pd.Timestamp:
    from .source import ET

    stamp = pd.Timestamp(pd.Timestamp(day).to_pydatetime().replace(hour=at.hour, minute=at.minute))
    return stamp.tz_localize(tz or ET)
