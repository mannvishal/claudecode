"""Domain types: contracts, verticals, iron condors.

All prices are in index points (SPX quoting convention). Dollar figures are
derived by multiplying by ``contract_size`` (100), and only ever at the
presentation boundary -- keeping points and dollars separate avoids the
factor-of-100 errors that are endemic to options code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from .pricing import Measure, bs_delta, bs_gamma, bs_price, bs_theta, bs_vega, implied_vol


@dataclass
class OptionContract:
    """A single option, normalized from a Tradier chain row."""

    symbol: str
    underlying: str
    root_symbol: str
    strike: float
    option_type: str  # "call" | "put"
    expiration: date
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    volume: int
    open_interest: int
    contract_size: int = 100
    expiration_type: str = ""

    # Recomputed from live mids by ``enrich``; never trusted from the feed.
    iv: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None

    # What the vendor said, kept only so the CLI can show how stale it was.
    vendor_iv: float | None = None
    vendor_delta: float | None = None
    vendor_greeks_asof: str | None = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_width(self) -> float:
        """Absolute bid-ask width in points."""
        return self.ask - self.bid

    @property
    def relative_spread(self) -> float:
        """Bid-ask width as a fraction of mid. ``inf`` when mid is zero."""
        m = self.mid
        return self.spread_width / m if m > 0 else float("inf")

    @property
    def is_quotable(self) -> bool:
        """Can this actually be traded, as opposed to merely listed?

        A zero bid means nobody will buy it from you, which makes it useless as
        the short leg of a credit spread no matter how attractive the mid looks.
        A crossed or inverted market means the quote is stale.
        """
        return self.bid > 0 and self.ask > 0 and self.ask >= self.bid

    @classmethod
    def from_tradier(cls, row: dict) -> "OptionContract":
        greeks = row.get("greeks") or {}
        return cls(
            symbol=row["symbol"],
            underlying=row.get("underlying", ""),
            root_symbol=row.get("root_symbol", ""),
            strike=float(row["strike"]),
            option_type=row["option_type"],
            expiration=date.fromisoformat(row["expiration_date"]),
            bid=float(row.get("bid") or 0.0),
            ask=float(row.get("ask") or 0.0),
            bid_size=int(row.get("bidsize") or 0),
            ask_size=int(row.get("asksize") or 0),
            volume=int(row.get("volume") or 0),
            open_interest=int(row.get("open_interest") or 0),
            contract_size=int(row.get("contract_size") or 100),
            expiration_type=row.get("expiration_type", ""),
            vendor_iv=greeks.get("mid_iv"),
            vendor_delta=greeks.get("delta"),
            vendor_greeks_asof=greeks.get("updated_at"),
        )

    def enrich(self, spot: float, T: float, r: float, q: float) -> "OptionContract":
        """Recompute IV and greeks from the live mid.

        Tradier's greeks come from a periodic ORATS batch and can be many hours
        stale -- on a 0DTE chain that is older than the entire remaining life of
        the contract, so the vendor delta is not merely imprecise, it is
        describing a different option. We back out IV from the current mid and
        rebuild the greeks from that.
        """
        iv = implied_vol(self.mid, spot, self.strike, T, r, q, self.option_type)
        self.iv = iv
        if iv is not None:
            self.delta = bs_delta(spot, self.strike, T, r, q, iv, self.option_type)
            self.gamma = bs_gamma(spot, self.strike, T, r, q, iv)
            self.theta = bs_theta(spot, self.strike, T, r, q, iv, self.option_type) / 365.0
            self.vega = bs_vega(spot, self.strike, T, r, q, iv) / 100.0
        return self

    def theoretical(self, spot: float, T: float, r: float, q: float, measure: Measure) -> float:
        """Value this contract under ``measure`` using its own smile point."""
        if self.iv is None:
            raise ValueError(f"{self.symbol} has no solved IV; call enrich() first")
        return bs_price(
            spot,
            self.strike,
            T,
            r,
            q,
            self.iv * measure.vol_multiplier,
            self.option_type,
        )


@dataclass
class Vertical:
    """A defined-risk credit spread: short one strike, long a further-OTM one."""

    kind: str  # "put_credit" | "call_credit"
    short: OptionContract
    long: OptionContract

    @property
    def width(self) -> float:
        return abs(self.short.strike - self.long.strike)

    @property
    def credit_mid(self) -> float:
        """Credit at the midpoint of both legs. Optimistic but not absurd."""
        return self.short.mid - self.long.mid

    @property
    def credit_natural(self) -> float:
        """Credit if you cross both spreads. The worst realistic fill."""
        return self.short.bid - self.long.ask

    def credit_at(self, fill_fraction: float) -> float:
        """Credit assuming you give up ``fill_fraction`` of the mid-to-natural gap.

        0.0 is a mid fill, 1.0 is paying the full spread on both legs. Real
        multileg fills on liquid SPX strikes land around 0.3-0.5; anything that
        only works at a mid fill does not work.
        """
        return self.credit_mid - fill_fraction * (self.credit_mid - self.credit_natural)

    def max_loss(self, credit: float) -> float:
        """Worst case in points. Cannot go below zero for a real credit spread."""
        return self.width - credit

    @property
    def short_delta(self) -> float | None:
        return self.short.delta

    @property
    def contract_size(self) -> int:
        return self.short.contract_size

    @property
    def legs(self) -> list[tuple[str, OptionContract]]:
        return [("sell_to_open", self.short), ("buy_to_open", self.long)]

    def describe(self) -> str:
        lo, hi = sorted((self.short.strike, self.long.strike))
        side = "put" if self.kind == "put_credit" else "call"
        return f"{lo:g}/{hi:g} {side} credit spread"


@dataclass
class IronCondor:
    """A put credit spread and a call credit spread on the same expiry."""

    put_spread: Vertical
    call_spread: Vertical

    @property
    def width(self) -> float:
        """Margin-relevant width.

        For European cash-settled index options only one side can finish in the
        money, so buying power is reduced by the *wider* wing, not the sum. This
        is a genuine structural advantage of SPX over American-style equity
        condors, where an early assignment can leave you short both sides at once.
        """
        return max(self.put_spread.width, self.call_spread.width)

    def credit_at(self, fill_fraction: float) -> float:
        return self.put_spread.credit_at(fill_fraction) + self.call_spread.credit_at(fill_fraction)

    def max_loss(self, credit: float) -> float:
        return self.width - credit

    @property
    def contract_size(self) -> int:
        return self.put_spread.contract_size

    @property
    def legs(self) -> list[tuple[str, OptionContract]]:
        return self.put_spread.legs + self.call_spread.legs

    def describe(self) -> str:
        return (
            f"{self.put_spread.long.strike:g}/{self.put_spread.short.strike:g}/"
            f"{self.call_spread.short.strike:g}/{self.call_spread.long.strike:g} iron condor"
        )


@dataclass
class Evaluation:
    """A scored candidate, ready to display or reject."""

    position: Vertical | IronCondor
    credit: float
    max_loss: float
    return_on_risk: float
    pop_risk_neutral: float
    pop_believed: float
    ev_risk_neutral: float
    ev_believed: float
    ev_after_costs: float
    costs: float
    contracts: int = 0
    total_credit: float = 0.0
    total_risk: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def describe(self) -> str:
        return self.position.describe()
