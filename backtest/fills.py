"""EXECUTION / FILL LAYER: how a decision becomes a price.

Rule 6: this module is swappable without touching signals. Signals emit a
``Structure`` -- a set of legs with sides -- and know nothing about slippage,
commissions, or which side of the book you cross. Everything price-related
happens here.

Rule 5 is implemented by ``MidMinusEdgeFill``:

  * Entry at mid minus ``entry_slippage_per_leg`` per leg, applied to the net
    credit, so a four-leg condor concedes four times the edge.
  * Exit at the adverse side: pay the ask to close a short, hit the bid to close
    a long. Marking exits at the mid is the standard way a backtest understates
    what getting out actually costs, and getting out is the whole risk-control
    story for a 0DTE seller.
  * Commissions per contract per leg, both ways.

To try a different assumption, implement ``FillModel`` and pass it to the
engine. Nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import ExecutionConfig
from .data import Quote

SELL, BUY = "sell", "buy"


@dataclass(frozen=True)
class Leg:
    symbol: str
    side: str  # SELL to open, BUY to open
    strike: float
    option_type: str

    @property
    def sign(self) -> int:
        """+1 if opening this leg brings in premium, -1 if it costs."""
        return 1 if self.side == SELL else -1


@dataclass(frozen=True)
class Structure:
    """What the signal layer produced. No prices, by design."""

    kind: str
    legs: tuple[Leg, ...]
    width: float

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    def describe(self) -> str:
        strikes = "/".join(f"{leg.strike:g}" for leg in sorted(self.legs, key=lambda x: x.strike))
        return f"{strikes} {self.kind}"


@dataclass(frozen=True)
class FillResult:
    price: float  # net credit (entry) or net debit (exit), in index points
    commissions: float  # dollars, for the whole structure
    tradable: bool
    reason: str = ""


class FillModel(Protocol):
    def entry(self, structure: Structure, quotes: dict[str, Quote], contracts: int) -> FillResult: ...

    def exit(self, structure: Structure, quotes: dict[str, Quote], contracts: int) -> FillResult: ...


class MidMinusEdgeFill:
    """Rule 5. The default."""

    def __init__(self, cfg: ExecutionConfig):
        self.cfg = cfg

    def _commissions(self, structure: Structure, contracts: int) -> float:
        return self.cfg.per_leg_cost * structure.n_legs * contracts

    def _collect(
        self, structure: Structure, quotes: dict[str, Quote], for_exit: bool = False
    ) -> tuple[list[Quote], str]:
        """Gather one quote per leg, with strictness that depends on direction.

        Entry and exit are not symmetric, and treating them as such is a real
        source of error at 0DTE:

        * **Entry** needs every leg genuinely tradable. A zero bid means nobody
          will buy the short leg from you, so the credit you modelled does not
          exist at any size.
        * **Exit** only needs a *mark*. A far-OTM leg that has decayed to a zero
          bid is not missing data -- it is worth nothing, which is a perfectly
          good valuation. Refusing to mark it strands the position: the profit
          target can never be evaluated, so every trade rides to expiry, which
          overstates the wins and fattens the tail on the losses.

        Only an absent quote fails an exit.
        """
        found = []
        for leg in structure.legs:
            quote = quotes.get(leg.symbol)
            if quote is None:
                return [], f"no quote for {leg.symbol}"
            if quote.is_crossed:
                return [], f"{leg.symbol} is crossed ({quote.bid:.2f}/{quote.ask:.2f})"
            if not for_exit:
                # Direction matters: opening a short needs a bid, opening a long
                # needs an offer. A single is_tradable test conflates them and
                # throws away buyable wings.
                ok = quote.can_sell if leg.side == SELL else quote.can_buy
                if not ok:
                    need = "bid" if leg.side == SELL else "offer"
                    return [], (
                        f"{leg.symbol} is {quote.bid:.2f}/{quote.ask:.2f}; "
                        f"no {need} to open a {leg.side}"
                    )
            found.append(quote)
        return found, ""

    def entry(self, structure: Structure, quotes: dict[str, Quote], contracts: int) -> FillResult:
        found, problem = self._collect(structure, quotes)
        if problem:
            return FillResult(0.0, 0.0, False, problem)

        credit_at_mid = sum(leg.sign * q.mid for leg, q in zip(structure.legs, found))
        credit = credit_at_mid - self.cfg.entry_slippage_per_leg * structure.n_legs
        return FillResult(
            price=credit,
            commissions=self._commissions(structure, contracts),
            tradable=True,
            reason=f"mid {credit_at_mid:.2f} less {self.cfg.entry_slippage_per_leg:.2f}/leg",
        )

    def exit(self, structure: Structure, quotes: dict[str, Quote], contracts: int) -> FillResult:
        found, problem = self._collect(structure, quotes, for_exit=True)
        if problem:
            return FillResult(0.0, 0.0, False, problem)

        debit = 0.0
        for leg, quote in zip(structure.legs, found):
            if leg.side == SELL:
                # Closing a short means buying it back: pay the ask.
                debit += quote.ask if self.cfg.exit_at_adverse_side else quote.mid
            else:
                # Closing a long means selling it: hit the bid.
                debit -= quote.bid if self.cfg.exit_at_adverse_side else quote.mid
        return FillResult(
            price=debit,
            commissions=self._commissions(structure, contracts),
            tradable=True,
            reason="adverse side" if self.cfg.exit_at_adverse_side else "mid",
        )


class MidFill(MidMinusEdgeFill):
    """Both sides at the midpoint. Optimistic; useful only as an upper bound.

    Running the same period under this and the default brackets how much of a
    result is strategy and how much is execution assumption. If the two differ
    by more than the edge being claimed, the edge is a fill artifact.
    """

    def __init__(self, cfg: ExecutionConfig):
        relaxed = ExecutionConfig(
            entry_slippage_per_leg=0.0,
            exit_at_adverse_side=False,
            commission_per_contract=cfg.commission_per_contract,
            exchange_fee_per_contract=cfg.exchange_fee_per_contract,
            contract_multiplier=cfg.contract_multiplier,
        )
        super().__init__(relaxed)


class SettlementFill:
    """Cash settlement at expiry: intrinsic value, no spread, no commission.

    SPXW is European and cash-settled, so a position held to the close is
    settled against SET rather than traded out. Charging a bid-ask spread on an
    exit that never happens overstates costs -- the opposite error to the usual
    one, and still an error.
    """

    def __init__(self, cfg: ExecutionConfig, settlement_price: float):
        self.cfg = cfg
        self.settlement = settlement_price

    def entry(self, structure: Structure, quotes: dict[str, Quote], contracts: int) -> FillResult:
        raise NotImplementedError("settlement is an exit-only model")

    def exit(self, structure: Structure, quotes: dict[str, Quote], contracts: int) -> FillResult:
        debit = 0.0
        for leg in structure.legs:
            if leg.option_type == "call":
                intrinsic = max(0.0, self.settlement - leg.strike)
            else:
                intrinsic = max(0.0, leg.strike - self.settlement)
            debit += leg.sign * intrinsic
        return FillResult(price=debit, commissions=0.0, tradable=True, reason="cash settlement")
