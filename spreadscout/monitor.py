"""Watching positions you already hold.

This is the half of the tool that matters most. A missed entry costs nothing; a
0DTE short strike going through the money costs the width, and it can happen
between two polls of a 60-second loop. Position alerts therefore run before the
entry screen on every pass, so a slow chain fetch can never delay a breach
warning.

Tradier reports *legs*, not strategies -- it cannot tell a naked short from the
short leg of a spread. Legs are grouped here by underlying and expiry so that
profit and loss is computed across the structure rather than per leg, which
would show the short side deep underwater while ignoring the long side paying
for it. The grouping is still an inference: two separate spreads on the same
underlying and expiry merge into one group. That errs toward reporting a larger
aggregate exposure, which is the safe direction.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime

from .alerts import CRITICAL, INFO, WARN, Alert
from .config import Config
from .pricing import CALL, ET, PUT, bs_delta, implied_vol, year_fraction
from .tradier import TradierClient, TradierError

log = logging.getLogger(__name__)

# ROOT (1-6 chars) + YYMMDD + C|P + strike * 1000, zero-padded to 8 digits.
OCC_RE = re.compile(r"^(?P<root>[A-Z]{1,6})(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


@dataclass(frozen=True)
class OccSymbol:
    root: str
    expiration: date
    option_type: str
    strike: float


def parse_occ(symbol: str) -> OccSymbol | None:
    """Parse an OCC option symbol. Returns ``None`` for equity symbols."""
    match = OCC_RE.match(symbol.strip().upper())
    if not match:
        return None
    ymd = match.group("ymd")
    try:
        expiry = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
    except ValueError:
        return None
    return OccSymbol(
        root=match.group("root"),
        expiration=expiry,
        option_type=CALL if match.group("cp") == "C" else PUT,
        strike=int(match.group("strike")) / 1000.0,
    )


@dataclass
class Leg:
    symbol: str
    occ: OccSymbol
    quantity: float  # negative for short
    cost_basis: float
    bid: float = 0.0
    ask: float = 0.0
    iv: float | None = None
    delta: float | None = None

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def close_price(self) -> float:
        """Price to flatten this leg, marked against the side you must cross.

        Closing a short means paying the ask; closing a long means hitting the
        bid. Marking both at the mid would understate what getting out costs,
        which is the one number you do not want flattered when deciding whether
        to get out.
        """
        return self.ask if self.is_short else self.bid


@dataclass
class PositionGroup:
    underlying: str
    expiration: date
    legs: list[Leg] = field(default_factory=list)

    @property
    def credit_received(self) -> float:
        """Net credit taken in, in dollars. Negative means it was a debit."""
        return -sum(leg.cost_basis for leg in self.legs)

    @property
    def cost_to_close(self) -> float:
        """Dollars to flatten every leg right now."""
        total = 0.0
        for leg in self.legs:
            contracts = abs(leg.quantity)
            value = leg.close_price * contracts * 100.0
            total += value if leg.is_short else -value
        return total

    @property
    def open_pnl(self) -> float:
        return self.credit_received - self.cost_to_close

    @property
    def loss_multiple(self) -> float | None:
        """Current loss as a multiple of the credit taken in."""
        credit = self.credit_received
        if credit <= 0:
            return None
        loss = self.cost_to_close - credit
        return loss / credit if loss > 0 else 0.0

    @property
    def short_legs(self) -> list[Leg]:
        return [leg for leg in self.legs if leg.is_short]

    @property
    def describe(self) -> str:
        strikes = "/".join(f"{leg.occ.strike:g}" for leg in sorted(self.legs, key=lambda x: x.occ.strike))
        return f"{self.underlying} {self.expiration} {strikes}"


def group_positions(positions: list[dict]) -> list[PositionGroup]:
    """Turn Tradier's flat leg list into option structures."""
    groups: dict[tuple[str, date], PositionGroup] = {}
    for row in positions:
        symbol = str(row.get("symbol", ""))
        occ = parse_occ(symbol)
        if occ is None:
            continue  # equity position, not our concern
        quantity = float(row.get("quantity") or 0)
        if quantity == 0:
            continue
        key = (occ.root, occ.expiration)
        group = groups.setdefault(key, PositionGroup(underlying=occ.root, expiration=occ.expiration))
        group.legs.append(
            Leg(
                symbol=symbol,
                occ=occ,
                quantity=quantity,
                cost_basis=float(row.get("cost_basis") or 0),
            )
        )
    return list(groups.values())


def enrich_groups(
    client: TradierClient,
    groups: list[PositionGroup],
    cfg: Config,
    now: datetime | None = None,
) -> dict[str, float]:
    """Quote every leg and recompute short-leg deltas. Returns spot by underlying."""
    now = now or datetime.now(ET)
    symbols = [leg.symbol for group in groups for leg in group.legs]
    if not symbols:
        return {}

    quotes = client.quotes(symbols)
    spots: dict[str, float] = {}

    for group in groups:
        # The position's root (SPXW) is not the quotable underlying (SPX).
        underlying = cfg.symbol if group.underlying.startswith(cfg.symbol) else group.underlying
        if underlying not in spots:
            try:
                spots[underlying] = client.spot(underlying)
            except TradierError:
                log.warning("no spot for %s; skipping delta recomputation", underlying)
                continue
        spot = spots[underlying]
        pm = group.underlying != cfg.symbol  # weekly roots settle PM
        T = year_fraction(now, group.expiration, pm_settled=pm)

        for leg in group.legs:
            row = quotes.get(leg.symbol) or {}
            leg.bid = float(row.get("bid") or 0.0)
            leg.ask = float(row.get("ask") or 0.0)
            if leg.mid > 0:
                leg.iv = implied_vol(
                    leg.mid, spot, leg.occ.strike, T,
                    cfg.risk_free_rate, cfg.dividend_yield, leg.occ.option_type,
                )
                if leg.iv is not None:
                    leg.delta = bs_delta(
                        spot, leg.occ.strike, T,
                        cfg.risk_free_rate, cfg.dividend_yield, leg.iv, leg.occ.option_type,
                    )
    return spots


def evaluate_group(group: PositionGroup, spot: float, cfg: Config) -> list[Alert]:
    """Produce alerts for one open structure."""
    m = cfg.monitor
    alerts: list[Alert] = []

    for leg in group.short_legs:
        delta = abs(leg.delta or 0.0)
        distance = leg.occ.strike - spot
        side = "call" if leg.occ.option_type == CALL else "put"
        threatened = (leg.occ.option_type == CALL and spot >= leg.occ.strike) or (
            leg.occ.option_type == PUT and spot <= leg.occ.strike
        )

        if threatened:
            # A breached leg is usually so far in the money that it carries no
            # extrinsic value, which makes implied vol -- and therefore delta --
            # unrecoverable. Saying so is more useful than printing 0.00, because
            # "no time value left" is itself the diagnosis.
            delta_line = (
                f"delta {delta:.2f}."
                if leg.delta is not None
                else "delta unavailable: the leg has no extrinsic value left."
            )
            alerts.append(Alert(
                kind="risk",
                severity=CRITICAL,
                title=f"SHORT STRIKE BREACHED: {leg.symbol}",
                key=f"risk:breach:{leg.symbol}",
                lines=[
                    f"{group.describe}",
                    f"spot {spot:,.2f} is through the short {side} at {leg.occ.strike:g}.",
                    f"{delta_line} This leg is in the money and heading for assignment "
                    f"value at settlement.",
                    f"cost to close the structure now: ${group.cost_to_close:,.0f} "
                    f"against ${group.credit_received:,.0f} taken in.",
                ],
            ))
        elif delta >= m.short_delta_critical:
            alerts.append(Alert(
                kind="risk",
                severity=CRITICAL,
                title=f"short {side} delta {delta:.2f}: {leg.symbol}",
                key=f"risk:delta:{leg.symbol}",
                lines=[
                    f"{group.describe}",
                    f"spot {spot:,.2f}, short strike {leg.occ.strike:g} "
                    f"({abs(distance):,.0f} points away).",
                    f"delta has passed {m.short_delta_critical:.2f}; the market now prices this "
                    f"strike as close to a coin flip.",
                ],
            ))
        elif delta >= m.short_delta_alert:
            alerts.append(Alert(
                kind="risk",
                severity=WARN,
                title=f"short {side} delta {delta:.2f}: {leg.symbol}",
                key=f"risk:delta-warn:{leg.symbol}",
                lines=[
                    f"{group.describe}",
                    f"spot {spot:,.2f}, short strike {leg.occ.strike:g} "
                    f"({abs(distance):,.0f} points away).",
                    f"delta has passed the {m.short_delta_alert:.2f} warning level.",
                ],
            ))
        elif abs(distance) <= m.strike_proximity_points:
            alerts.append(Alert(
                kind="risk",
                severity=WARN,
                title=f"spot within {abs(distance):,.0f} pts of short {side} {leg.occ.strike:g}",
                key=f"risk:proximity:{leg.symbol}",
                lines=[
                    f"{group.describe}",
                    f"spot {spot:,.2f} is inside the {m.strike_proximity_points:g}-point "
                    f"proximity band around the short strike.",
                ],
            ))

    multiple = group.loss_multiple
    if multiple is not None and multiple >= m.loss_multiple_alert:
        alerts.append(Alert(
            kind="risk",
            severity=CRITICAL if multiple >= m.loss_multiple_alert * 1.5 else WARN,
            title=f"loss is {multiple:.1f}x credit: {group.describe}",
            key=f"risk:loss:{group.describe}",
            lines=[
                f"took in ${group.credit_received:,.0f}, costs ${group.cost_to_close:,.0f} "
                f"to close now.",
                f"open P&L ${group.open_pnl:,.0f} ({multiple:.1f}x the credit received).",
                f"your configured stop is {m.loss_multiple_alert:.1f}x.",
            ],
        ))

    return alerts


def check_positions(
    client: TradierClient, cfg: Config, account_id: str, now: datetime | None = None
) -> tuple[list[Alert], list[PositionGroup]]:
    """Fetch, enrich and evaluate every open option structure."""
    groups = group_positions(client.positions(account_id))
    if not groups:
        return [], []
    spots = enrich_groups(client, groups, cfg, now=now)

    alerts: list[Alert] = []
    for group in groups:
        underlying = cfg.symbol if group.underlying.startswith(cfg.symbol) else group.underlying
        spot = spots.get(underlying)
        if spot is None:
            continue
        alerts.extend(evaluate_group(group, spot, cfg))
    return alerts, groups
