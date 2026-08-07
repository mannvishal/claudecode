"""Modelled historical replay of the selection rules.

READ THIS BEFORE BELIEVING ANY NUMBER THIS MODULE PRINTS.

There is no source of intraday historical option chains wired into this tool,
and for 0DTE that is a fatal gap: an end-of-day chain snapshot on expiry day is
taken *at* settlement, so it cannot tell you what you would have been filled at
in the morning. Rather than ship a backtest that quietly invents entry prices
and calls itself validated, this module is explicit that it *models* entries:

  * Strikes are chosen by Black-Scholes delta at the open.
  * The entry credit is a model price, not a print. The smile shape is
    calibrated from a live chain today and assumed stationary across history,
    with the level driven by each day's VIX close.
  * Settlement is the underlying's actual close, which is real.
  * Fills are assumed to happen at the modelled credit less a haircut.

The direction of the errors is knowable and mostly flattering: real fills are
worse than modelled ones, real chains gap, and no model here can halt trading
or blow through a strike between prints. Treat the output as an upper bound on
what the rules would have produced, not an estimate of it -- and note that an
upper bound which is already unattractive is genuinely informative.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime

from .config import Config
from .models import OptionContract
from .pricing import CALL, ET, PUT, bs_delta, bs_price
from .screener import load_chain, pick_expiration
from .tradier import TradierClient, TradierError

TRADING_DAY_YEARS = 6.5 / (24.0 * 365.0)  # 09:30-16:00 ET, calendar-time convention


@dataclass
class Smile:
    """Quadratic implied-vol smile in *standardized* moneyness, ATM = 1.0.

    The x-axis is ``z = ln(K/S) / (sigma_atm * sqrt(T))`` -- strike distance
    measured in standard deviations rather than percent. This matters more than
    it sounds: at 0DTE one standard deviation of SPX is about 25 index points,
    so a band of +/-15% log-moneyness that looks reasonable for a 45-day chain
    spans thirty standard deviations of a 0DTE one and is populated almost
    entirely by strikes that will never trade. Standardizing also makes the
    fitted shape reusable across vol regimes, which is what the backtest needs
    when it applies today's smile to a day with a different VIX.
    """

    b: float = 0.0
    c: float = 0.0

    def factor(self, z: float) -> float:
        """Multiplier on ATM vol at ``z`` standard deviations from the money."""
        return max(0.2, 1.0 + self.b * z + self.c * z * z)


def standardized_moneyness(strike: float, spot: float, atm_vol: float, T: float) -> float:
    denom = atm_vol * math.sqrt(max(T, 1e-12))
    if denom <= 0:
        return 0.0
    return math.log(strike / spot) / denom


def fit_smile(contracts: list[OptionContract], spot: float, T: float) -> Smile:
    """Least-squares quadratic fit of IV/ATM_IV against standardized moneyness.

    Uses OTM contracts only -- the ITM side of each type is illiquid and its
    implied vol is dominated by quote noise around a large intrinsic value.
    """
    otm = []
    for c in contracts:
        if c.iv is None or c.iv <= 0:
            continue
        if c.option_type == PUT and c.strike > spot:
            continue
        if c.option_type == CALL and c.strike < spot:
            continue
        otm.append(c)

    if len(otm) < 8:
        return Smile()

    atm = min(otm, key=lambda c: abs(c.strike - spot)).iv
    if not atm or atm <= 0:
        return Smile()

    points: list[tuple[float, float]] = []
    for c in otm:
        z = standardized_moneyness(c.strike, spot, atm, T)
        if abs(z) > 3.0:  # beyond 3 sigma the quotes are noise, not a surface
            continue
        points.append((z, c.iv))

    if len(points) < 8:
        return Smile()

    # Normal equations for y = 1 + b*k + c*k^2 (intercept pinned at ATM).
    sk = sk2 = sk3 = sk4 = syk = syk2 = 0.0
    for k, iv in points:
        y = iv / atm - 1.0
        sk += k
        sk2 += k * k
        sk3 += k**3
        sk4 += k**4
        syk += y * k
        syk2 += y * k * k

    det = sk2 * sk4 - sk3 * sk3
    if abs(det) < 1e-18:
        return Smile()
    b = (syk * sk4 - syk2 * sk3) / det
    c = (syk2 * sk2 - syk * sk3) / det
    return Smile(b=b, c=c)


@dataclass
class Trade:
    day: date
    kind: str
    short_strike: float
    long_strike: float
    credit: float
    pnl: float
    settlement: float
    breached: bool


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    skipped: int = 0

    @property
    def wins(self) -> list[Trade]:
        return [t for t in self.trades if t.pnl > 0]

    @property
    def losses(self) -> list[Trade]:
        return [t for t in self.trades if t.pnl <= 0]

    def equity_curve(self) -> list[float]:
        total, curve = 0.0, []
        for t in self.trades:
            total += t.pnl
            curve.append(total)
        return curve

    def max_drawdown(self) -> float:
        peak, worst = 0.0, 0.0
        for value in self.equity_curve():
            peak = max(peak, value)
            worst = min(worst, value - peak)
        return worst


def _find_strike_by_delta(
    spot: float, target_delta: float, option_type: str, T: float,
    atm_vol: float, smile: Smile, r: float, q: float, increment: float,
) -> float | None:
    """Walk the strike ladder outward until delta crosses the target."""
    direction = -1 if option_type == PUT else 1
    strike = round(spot / increment) * increment
    for _ in range(400):
        strike += direction * increment
        if strike <= 0:
            return None
        z = standardized_moneyness(strike, spot, atm_vol, T)
        vol = atm_vol * smile.factor(z)
        delta = abs(bs_delta(spot, strike, T, r, q, vol, option_type))
        if delta <= target_delta:
            return strike
    return None


def _spread_credit(
    spot: float, short_k: float, long_k: float, option_type: str, T: float,
    atm_vol: float, smile: Smile, r: float, q: float,
) -> float:
    def price(strike: float) -> float:
        vol = atm_vol * smile.factor(standardized_moneyness(strike, spot, atm_vol, T))
        return bs_price(spot, strike, T, r, q, vol, option_type)

    return price(short_k) - price(long_k)


def _vertical_pnl(
    kind: str, short_k: float, long_k: float, credit: float, settle: float, mult: int
) -> tuple[float, bool]:
    width = abs(short_k - long_k)
    if kind == "put_credit":
        intrinsic = max(0.0, short_k - settle) - max(0.0, long_k - settle)
    else:
        intrinsic = max(0.0, settle - short_k) - max(0.0, settle - long_k)
    intrinsic = min(intrinsic, width)
    return (credit - intrinsic) * mult, intrinsic > 0


def run_backtest(
    client: TradierClient, cfg: Config, start: date, end: date, args
) -> int:
    underlying = args.underlying
    print(__doc__.split("\n\n", 1)[1].strip())
    print("\n" + "=" * 72)

    # Calibrate the smile from a live chain. This is the one piece of real
    # options data in the whole exercise.
    smile = Smile()
    try:
        expiration = pick_expiration(client, cfg.symbol, cfg.dte)
        live_spot, live_T, contracts = load_chain(client, cfg, expiration)
        smile = fit_smile([c for c in contracts if c.is_quotable], live_spot, live_T)
        print(f"smile calibrated from live {cfg.symbol} {expiration} chain: "
              f"slope {smile.b:+.3f}, curvature {smile.c:+.3f}")
    except (TradierError, SystemExit) as exc:
        print(f"could not calibrate smile from a live chain ({exc}); using flat vol. "
              f"Put-side credits will be understated and call-side overstated.")

    bars = client.history(underlying, start, end)
    if not bars:
        raise SystemExit(f"no history for {underlying} between {start} and {end}")

    try:
        vix_bars = {b["date"]: float(b["close"]) for b in client.history("VIX", start, end)}
    except TradierError:
        vix_bars = {}
    if not vix_bars:
        print("VIX history unavailable; falling back to 20-day realized vol as the ATM level. "
              "That removes the variance risk premium from the simulation entirely, which "
              "biases results *toward* the seller. Read the result accordingly.")

    increment = 5.0 if cfg.symbol.upper() in {"SPX", "SPXW"} else 1.0
    mult = 100
    r, q = cfg.risk_free_rate, cfg.dividend_yield
    target_delta = (cfg.filters.min_short_delta + cfg.filters.max_short_delta) / 2.0
    width = cfg.filters.widths[0]
    # Held to cash settlement, so there is no closing trade to pay for.
    per_leg = cfg.costs.commission_per_contract + cfg.costs.exchange_fee_per_contract
    costs = per_leg * 4  # iron condor, one way

    result = BacktestResult()
    closes = [float(b["close"]) for b in bars]

    for i, bar in enumerate(bars):
        day = date.fromisoformat(bar["date"])
        open_px, close_px = float(bar["open"]), float(bar["close"])
        if open_px <= 0:
            result.skipped += 1
            continue

        vix = vix_bars.get(bar["date"])
        if vix:
            atm_vol = vix / 100.0
        elif i >= 20:
            rets = [math.log(closes[j] / closes[j - 1]) for j in range(i - 19, i + 1)]
            atm_vol = statistics.stdev(rets) * math.sqrt(252)
        else:
            result.skipped += 1
            continue

        T = TRADING_DAY_YEARS
        put_short = _find_strike_by_delta(
            open_px, target_delta, PUT, T, atm_vol, smile, r, q, increment)
        call_short = _find_strike_by_delta(
            open_px, target_delta, CALL, T, atm_vol, smile, r, q, increment)
        if put_short is None or call_short is None:
            result.skipped += 1
            continue

        put_long, call_long = put_short - width, call_short + width
        put_credit = _spread_credit(
            open_px, put_short, put_long, PUT, T, atm_vol, smile, r, q)
        call_credit = _spread_credit(
            open_px, call_short, call_long, CALL, T, atm_vol, smile, r, q)

        # Haircut for crossing the spread, using the configured fill assumption.
        gross = (put_credit + call_credit) * (1.0 - 0.5 * cfg.costs.fill_fraction)
        if gross <= 0:
            result.skipped += 1
            continue

        put_pnl, put_breach = _vertical_pnl(
            "put_credit", put_short, put_long, put_credit, close_px, mult)
        call_pnl, call_breach = _vertical_pnl(
            "call_credit", call_short, call_long, call_credit, close_px, mult)
        haircut = ((put_credit + call_credit) - gross) * mult
        pnl = put_pnl + call_pnl - haircut - costs

        result.trades.append(Trade(
            day=day, kind="iron_condor", short_strike=put_short, long_strike=call_short,
            credit=gross, pnl=pnl, settlement=close_px,
            breached=put_breach or call_breach,
        ))

    _report(result, cfg, width, target_delta, mult)
    return 0


def _report(result: BacktestResult, cfg: Config, width: float, target_delta: float, mult: int) -> None:
    trades = result.trades
    if not trades:
        print("\nno tradable days in range")
        return

    pnls = [t.pnl for t in trades]
    total = sum(pnls)
    wins, losses = result.wins, result.losses
    win_rate = len(wins) / len(trades)
    avg_win = statistics.mean([t.pnl for t in wins]) if wins else 0.0
    avg_loss = statistics.mean([t.pnl for t in losses]) if losses else 0.0
    max_risk = (width - statistics.mean([t.credit for t in trades])) * mult

    print("\n" + "=" * 72)
    print(f"MODELLED RESULT  {trades[0].day} to {trades[-1].day}   "
          f"1 iron condor/day, {width:g}-wide, ~{target_delta:.2f} delta shorts")
    print("=" * 72)
    print(f"  trading days          {len(trades)}  ({result.skipped} skipped)")
    print(f"  win rate              {win_rate:.1%}")
    print(f"  total P&L             ${total:,.0f}")
    print(f"  mean per trade        ${statistics.mean(pnls):,.2f}")
    print(f"  median per trade      ${statistics.median(pnls):,.2f}")
    print(f"  stdev per trade       ${statistics.pstdev(pnls):,.2f}")
    print(f"  average win           ${avg_win:,.2f}")
    print(f"  average loss          ${avg_loss:,.2f}")
    print(f"  worst day             ${min(pnls):,.2f}")
    print(f"  best day              ${max(pnls):,.2f}")
    print(f"  max drawdown          ${result.max_drawdown():,.2f}")
    print(f"  capital at risk/trade ${max_risk:,.0f}")

    if avg_win > 0 and avg_loss < 0:
        wins_to_erase = abs(min(pnls)) / avg_win
        print(f"\n  One worst-day loss erases {wins_to_erase:.0f} winning days.")
    if avg_loss < 0:
        breakeven = abs(avg_loss) / (avg_win - avg_loss) if avg_win > avg_loss else float("nan")
        print(f"  Break-even win rate needed: {breakeven:.1%} "
              f"(you got {win_rate:.1%} in this simulation)")

    if max_risk > 0:
        daily_return = statistics.mean(pnls) / max_risk
        print(f"\n  Mean return on capital at risk: {daily_return:.3%}/day.")
        print("  For scale, the 1%/day target compounds to roughly +1,100%/year.")

    print("\n  Every caveat at the top of this module applies. This is a model of a")
    print("  strategy, not a record of one. Real fills, real gaps and real halts all")
    print("  push in the same direction, and it is not the favourable one.")
