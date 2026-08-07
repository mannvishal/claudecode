"""Candidate construction, filtering and ranking."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from .config import Config
from .models import Evaluation, IronCondor, OptionContract, Vertical
from .pricing import CALL, ET, PUT, year_fraction
from .risk import evaluate
from .tradier import TradierClient

log = logging.getLogger(__name__)

PUT_CREDIT = "put_credit"
CALL_CREDIT = "call_credit"


def pick_expiration(client: TradierClient, symbol: str, dte: int) -> date:
    """Nearest listed expiration at least ``dte`` calendar days out."""
    today = datetime.now(ET).date()
    target = today + timedelta(days=dte)
    available = sorted(e for e in client.expirations(symbol) if e >= target)
    if not available:
        raise SystemExit(f"no {symbol} expirations on or after {target}")
    return available[0]


def is_pm_settled(contracts: list[OptionContract]) -> bool:
    """SPXW weeklies settle PM; the monthly SPX root settles AM.

    A whole trading day of time value hangs on this, so it is read off the
    chain's own ``root_symbol`` rather than assumed.
    """
    roots = {c.root_symbol for c in contracts if c.root_symbol}
    return not (roots == {"SPX"})


def load_chain(
    client: TradierClient, cfg: Config, expiration: date, now: datetime | None = None
) -> tuple[float, float, list[OptionContract]]:
    """Fetch a chain and re-derive its greeks from live mids.

    Returns ``(spot, year_fraction_to_expiry, contracts)``.
    """
    now = now or datetime.now(ET)
    spot = client.spot(cfg.symbol)
    rows = client.chain(cfg.symbol, expiration)
    if not rows:
        raise SystemExit(f"empty chain for {cfg.symbol} {expiration}")

    contracts = [OptionContract.from_tradier(r) for r in rows]
    T = year_fraction(now, expiration, pm_settled=is_pm_settled(contracts))

    for c in contracts:
        if c.is_quotable:
            c.enrich(spot, T, cfg.risk_free_rate, cfg.dividend_yield)
    return spot, T, contracts


def _passes_liquidity(c: OptionContract, cfg: Config) -> bool:
    f = cfg.filters
    return (
        c.is_quotable
        and c.iv is not None
        and c.open_interest >= f.min_open_interest
        and c.volume >= f.min_volume
        and c.relative_spread <= f.max_relative_spread
    )


def build_verticals(
    contracts: list[OptionContract], kind: str, spot: float, cfg: Config
) -> list[Vertical]:
    """Enumerate credit spreads whose short leg sits in the target delta band."""
    f = cfg.filters
    option_type = PUT if kind == PUT_CREDIT else CALL
    pool = [c for c in contracts if c.option_type == option_type and _passes_liquidity(c, cfg)]
    by_strike = {c.strike: c for c in pool}

    out: list[Vertical] = []
    for short in pool:
        # The short leg must be out of the money -- an ITM short turns the
        # position into a directional bet with a very different payoff shape.
        if kind == PUT_CREDIT and short.strike >= spot:
            continue
        if kind == CALL_CREDIT and short.strike <= spot:
            continue

        delta = abs(short.delta or 0.0)
        if not (f.min_short_delta <= delta <= f.max_short_delta):
            continue

        for width in f.widths:
            long_strike = short.strike - width if kind == PUT_CREDIT else short.strike + width
            long_leg = by_strike.get(long_strike)
            if long_leg is None:
                continue
            spread = Vertical(kind=kind, short=short, long=long_leg)
            credit = spread.credit_at(cfg.costs.fill_fraction)
            if credit < f.min_credit or credit / width < f.min_credit_ratio:
                continue
            out.append(spread)
    return out


def build_condors(
    puts: list[Vertical], calls: list[Vertical], spot: float, cfg: Config
) -> list[IronCondor]:
    """Pair put and call verticals into delta-balanced condors.

    Pairing every put with every call is quadratic and mostly produces lopsided
    structures. We sort each side by short delta and match nearest-delta pairs of
    equal width, which is what a condor is actually meant to be: a symmetric bet
    that the index goes nowhere.
    """
    if not puts or not calls:
        return []

    out: list[IronCondor] = []
    for width in cfg.filters.widths:
        p_side = sorted(
            (p for p in puts if p.width == width), key=lambda v: abs(v.short.delta or 0.0)
        )
        c_side = sorted(
            (c for c in calls if c.width == width), key=lambda v: abs(v.short.delta or 0.0)
        )
        for put in p_side:
            p_delta = abs(put.short.delta or 0.0)
            best = min(
                c_side,
                key=lambda c: abs(abs(c.short.delta or 0.0) - p_delta),
                default=None,
            )
            if best is None:
                continue
            # Reject pairs whose deltas are too far apart to call symmetric.
            if abs(abs(best.short.delta or 0.0) - p_delta) > 0.05:
                continue
            out.append(IronCondor(put_spread=put, call_spread=best))
    return out


def annotate(ev: Evaluation, spot: float, T: float, contracts_asof: str | None) -> Evaluation:
    """Attach human-readable warnings to a scored candidate."""
    notes = ev.notes
    days = T * 365.0

    if days < 1.0:
        short_legs = [leg for side, leg in ev.position.legs if side.startswith("sell")]
        worst_gamma = max((abs(leg.gamma or 0.0) for leg in short_legs), default=0.0)
        hours = days * 24.0
        notes.append(
            f"0DTE: {hours:.1f}h to settlement. Short-leg gamma {worst_gamma:.4f}/pt -- "
            f"delta moves ~{worst_gamma * 10:.2f} per 10-point index move, so a strike "
            f"that is safe now can be through the money in minutes."
        )

    for _, leg in ev.position.legs:
        if leg.relative_spread > 0.10:
            notes.append(
                f"{leg.symbol} market is {leg.bid:.2f}/{leg.ask:.2f} "
                f"({leg.relative_spread:.0%} of mid) -- fills will be worse than modelled."
            )
            break

    if ev.ev_risk_neutral < 0 and ev.ev_believed > 0:
        notes.append(
            "Positive EV here comes entirely from your vol_multiplier assumption, "
            "not from the market's pricing."
        )

    if contracts_asof:
        notes.append(f"Vendor greeks were stamped {contracts_asof}; recomputed from live mids.")

    return ev


def screen(client: TradierClient, cfg: Config, now: datetime | None = None) -> dict:
    """Run the full screen. Returns a result dict for the CLI to render."""
    expiration = pick_expiration(client, cfg.symbol, cfg.dte)
    spot, T, contracts = load_chain(client, cfg, expiration, now=now)

    quotable = [c for c in contracts if c.is_quotable]
    liquid = [c for c in quotable if _passes_liquidity(c, cfg)]

    puts = build_verticals(contracts, PUT_CREDIT, spot, cfg)
    calls = build_verticals(contracts, CALL_CREDIT, spot, cfg)
    condors = build_condors(puts, calls, spot, cfg)

    asof = next((c.vendor_greeks_asof for c in contracts if c.vendor_greeks_asof), None)

    scored: list[Evaluation] = []
    for position in [*puts, *calls, *condors]:
        ev = evaluate(position, spot, T, cfg)
        if ev is not None:
            scored.append(annotate(ev, spot, T, asof))

    scored.sort(key=lambda e: e.ev_after_costs, reverse=True)

    return {
        "symbol": cfg.symbol,
        "spot": spot,
        "expiration": expiration,
        "years_to_expiry": T,
        "counts": {
            "listed": len(contracts),
            "quotable": len(quotable),
            "liquid": len(liquid),
            "put_spreads": len(puts),
            "call_spreads": len(calls),
            "condors": len(condors),
        },
        "candidates": scored[: cfg.filters.max_candidates],
        "all_candidates": scored,
        "vendor_greeks_asof": asof,
    }
