"""Expectancy, position sizing, and hard limits.

The central claim this module encodes, and the reason it exists:

    A credit spread's expected value under the market's own implied
    distribution is exactly zero before costs, and negative after them.

That is not a modelling choice, it is arithmetic. We solve each leg's implied
vol from its own mid, so by construction the model reprices the market exactly;
selling at the mid and valuing at the mid nets to zero, and every dollar of
commission and slippage comes straight off the top. A high probability of profit
does not change this -- it is the compensation you are being paid for the tail,
not evidence of an edge.

Positive expected value therefore requires a forecast that differs from
implied. ``BeliefConfig.vol_multiplier`` is where you state that forecast, and
this module reports expectancy under both measures side by side so the
assumption is never invisible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime

from .config import Config
from .models import Evaluation, IronCondor, OptionContract, Vertical
from .pricing import ET, Measure, bs_price, prob_above, prob_below


def expected_payoff(S: float, K: float, T: float, mu: float, sigma: float, kind: str) -> float:
    """E[max(S_T - K, 0)] (call) or E[max(K - S_T, 0)] (put) under GBM drift ``mu``.

    Undiscounted, i.e. valued at expiry. Uses the identity that the real-world
    expected payoff equals the Black-Scholes price computed with ``r = mu``,
    ``q = 0``, then compounded forward at ``mu``. At 0DTE the discount factor is
    ~1 anyway, but keeping it exact means the same code serves the 45-DTE
    backtest without a second implementation.
    """
    return math.exp(mu * T) * bs_price(S, K, T, mu, 0.0, sigma, kind)


def _leg_expected_payoff(
    leg: OptionContract, S: float, T: float, measure: Measure, r: float, q: float
) -> float:
    if leg.iv is None:
        raise ValueError(f"{leg.symbol} has no solved IV")
    return expected_payoff(
        S,
        leg.strike,
        T,
        measure.effective_drift(r, q),
        leg.iv * measure.vol_multiplier,
        leg.option_type,
    )


def position_expected_payoff(
    position: Vertical | IronCondor, S: float, T: float, measure: Measure, r: float, q: float
) -> float:
    """Expected cost to settle the position, in index points, discounted to today.

    The discount factor matters less for its size than for its consistency: the
    credit is cash received now, so the expected settlement cost has to be
    quoted in today's dollars too. Without it the risk-neutral expectancy comes
    out as a small non-zero number, which would look like a real edge and is
    purely a units error. Over a 0DTE horizon it is a third of a cent on a $100
    credit -- immaterial to trade, fatal to a proof.
    """
    total = 0.0
    for side, leg in position.legs:
        payoff = _leg_expected_payoff(leg, S, T, measure, r, q)
        total += payoff if side.startswith("sell") else -payoff
    return math.exp(-r * T) * total


def probability_of_max_profit(
    position: Vertical | IronCondor, S: float, T: float, measure: Measure, r: float, q: float
) -> float:
    """P(every short leg expires worthless) -- the 'win rate' headline.

    Deliberately reported alongside expectancy and never alone. For a typical
    0DTE SPX condor this number is around 0.85-0.92, which is precisely why the
    strategy feels like it works.
    """
    mu = measure.effective_drift(r, q)
    if isinstance(position, IronCondor):
        put_short = position.put_spread.short
        call_short = position.call_spread.short
        sigma_p = (put_short.iv or 0.0) * measure.vol_multiplier
        sigma_c = (call_short.iv or 0.0) * measure.vol_multiplier
        # Both must hold. Using each strike's own smile point means the two
        # probabilities come from slightly different lognormals; taking the
        # complement of the union is the consistent way to combine them.
        p_breach_put = prob_below(S, put_short.strike, T, sigma_p, mu)
        p_breach_call = prob_above(S, call_short.strike, T, sigma_c, mu)
        return max(0.0, 1.0 - p_breach_put - p_breach_call)

    short = position.short
    sigma = (short.iv or 0.0) * measure.vol_multiplier
    if position.kind == "put_credit":
        return prob_above(S, short.strike, T, sigma, mu)
    return prob_below(S, short.strike, T, sigma, mu)


def round_trip_costs(position: Vertical | IronCondor, cfg: Config) -> float:
    """Per-spread transaction costs in *dollars*, for one contract."""
    per_leg = cfg.costs.commission_per_contract + cfg.costs.exchange_fee_per_contract
    legs = len(position.legs)
    trips = 2 if cfg.costs.assume_closing_trade else 1
    return per_leg * legs * trips


def evaluate(
    position: Vertical | IronCondor, spot: float, T: float, cfg: Config
) -> Evaluation | None:
    """Score a candidate. Returns ``None`` if it is structurally unsound."""
    credit = position.credit_at(cfg.costs.fill_fraction)
    width = position.width
    if credit <= 0 or credit >= width:
        # Credit above width is free money, which means a stale quote, not an
        # opportunity. Credit at or below zero is not a credit spread.
        return None

    max_loss = position.max_loss(credit)
    mult = position.contract_size

    risk_neutral = Measure()
    believed = Measure(
        vol_multiplier=cfg.beliefs.vol_multiplier,
        drift=cfg.beliefs.drift,
        label="believed",
    )
    r, q = cfg.risk_free_rate, cfg.dividend_yield

    cost_to_close_rn = position_expected_payoff(position, spot, T, risk_neutral, r, q)
    cost_to_close_bel = position_expected_payoff(position, spot, T, believed, r, q)

    ev_rn = (credit - cost_to_close_rn) * mult
    ev_bel = (credit - cost_to_close_bel) * mult
    costs = round_trip_costs(position, cfg)

    return Evaluation(
        position=position,
        credit=credit,
        max_loss=max_loss,
        return_on_risk=credit / max_loss,
        pop_risk_neutral=probability_of_max_profit(position, spot, T, risk_neutral, r, q),
        pop_believed=probability_of_max_profit(position, spot, T, believed, r, q),
        ev_risk_neutral=ev_rn,
        ev_believed=ev_bel,
        ev_after_costs=ev_bel - costs,
        costs=costs,
    )


def kelly_fraction(win_prob: float, credit: float, max_loss: float) -> float:
    """Kelly-optimal fraction of bankroll to put at risk.

    Reported for reference only, and never used for sizing. Kelly assumes you
    know the true probability; here the probability is an assumption, and Kelly
    is savagely sensitive to overestimating it. On a payoff this skewed, a
    win-rate estimate that is 3 points optimistic can turn the "optimal" bet into
    a negative-growth one. Size off ``RiskConfig`` instead.
    """
    if max_loss <= 0 or credit <= 0:
        return 0.0
    b = credit / max_loss
    f = (win_prob * b - (1.0 - win_prob)) / b
    return max(0.0, f)


@dataclass
class SizingResult:
    contracts: int
    total_credit: float
    total_risk: float
    reasons: list[str]

    @property
    def rejected(self) -> bool:
        return self.contracts == 0


def size_position(
    ev: Evaluation, equity: float, open_risk: float, cfg: Config
) -> SizingResult:
    """Convert a scored candidate into a contract count, or refuse to.

    Every cap is a floor over the others -- the binding one wins, and the reason
    is recorded so a zero is never mysterious.
    """
    reasons: list[str] = []
    mult = ev.position.contract_size
    risk_per_spread = ev.max_loss * mult
    credit_per_spread = ev.credit * mult

    if risk_per_spread <= 0:
        return SizingResult(0, 0.0, 0.0, ["max loss is non-positive; refusing to size"])

    if equity < cfg.risk.min_equity:
        return SizingResult(
            0, 0.0, 0.0, [f"equity ${equity:,.0f} is below min_equity ${cfg.risk.min_equity:,.0f}"]
        )

    if cfg.risk.require_positive_ev and ev.ev_after_costs <= 0:
        return SizingResult(
            0,
            0.0,
            0.0,
            [
                f"expected value after costs is ${ev.ev_after_costs:,.2f}/spread. "
                f"Under your stated beliefs this trade loses money on average."
            ],
        )

    by_trade = int(equity * cfg.risk.max_risk_pct_per_trade // risk_per_spread)
    remaining_budget = equity * cfg.risk.max_total_risk_pct - open_risk
    by_portfolio = int(max(remaining_budget, 0.0) // risk_per_spread)

    caps = {
        "max_risk_pct_per_trade": by_trade,
        "max_total_risk_pct": by_portfolio,
        "max_contracts": cfg.risk.max_contracts,
    }
    contracts = min(caps.values())

    if contracts <= 0:
        binding = min(caps, key=caps.get)
        reasons.append(f"sized to zero: {binding} is the binding constraint")
        return SizingResult(0, 0.0, 0.0, reasons)

    binding = [name for name, cap in caps.items() if cap == contracts]
    reasons.append(f"capped by {', '.join(binding)}")

    return SizingResult(
        contracts=contracts,
        total_credit=credit_per_spread * contracts,
        total_risk=risk_per_spread * contracts,
        reasons=reasons,
    )


def open_risk_from_positions(positions: list[dict]) -> float:
    """Approximate dollars already at risk from open option positions.

    Tradier's positions endpoint reports cost basis per leg, not per strategy,
    so it cannot tell a naked short from the short leg of a spread. This returns
    a conservative estimate based on gross short exposure and is deliberately
    pessimistic: over-stating open risk shrinks the next position, which is the
    safe direction to be wrong in.
    """
    risk = 0.0
    for pos in positions:
        quantity = float(pos.get("quantity") or 0)
        cost_basis = abs(float(pos.get("cost_basis") or 0))
        if quantity < 0:  # short leg
            risk += cost_basis
    return risk


def realized_pnl_today(closed_positions: list[dict], today: date | None = None) -> float:
    """Sum today's realized P&L from the gain/loss endpoint."""
    today = today or datetime.now(ET).date()
    total = 0.0
    for row in closed_positions:
        close_date = row.get("close_date") or ""
        if close_date[:10] == today.isoformat():
            total += float(row.get("gain_loss") or 0)
    return total


def daily_loss_breached(realized: float, equity: float, cfg: Config) -> bool:
    limit = equity * cfg.risk.daily_loss_limit_pct
    return realized <= -limit
