"""Measuring whether conditions actually favour selling premium.

The question "is now a good time to sell a credit spread?" has one defensible
answer: only when implied volatility is rich relative to what the underlying is
actually realizing. That difference -- the variance risk premium -- is the only
thing a premium seller harvests. Everything else (delta bands, widths, strike
selection) shapes the risk, it does not create the return.

So rather than have you guess ``beliefs.vol_multiplier``, this module measures
it. Realized vol comes from Tradier's own daily bars; implied comes from the
live chain's ATM strike. The ratio is the premium on offer today.

Two honest caveats, both enforced in code below:

* **Horizon matching.** A 0DTE seller enters after the open and is flat at the
  close, so the only variance they are exposed to is open-to-close. Comparing
  0DTE implied vol against *close-to-close* realized vol is comparing it against
  a risk you never carry, because close-to-close includes the overnight gap.
  Open-to-close is the correct denominator -- and it is also the smaller one,
  which makes the measured premium look larger. Because that cuts in the
  flattering direction, ``GateConfig.conservative_vrp`` defaults to True and
  takes the *worse* of the two readings.
* **Persistence is an assumption.** Trailing realized vol is a fact; that
  tomorrow resembles it is a forecast. It is the weakest link in the chain, and
  it fails hardest exactly when it costs most -- the day realized vol explodes
  past implied is the day the trailing window says conditions are calm.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .config import Config
from .models import OptionContract
from .pricing import ET
from .tradier import TradierClient, TradierError

TRADING_DAYS = 252


def _log_returns_close_to_close(bars: list[dict]) -> list[float]:
    out = []
    for prev, cur in zip(bars, bars[1:]):
        p, c = float(prev.get("close") or 0), float(cur.get("close") or 0)
        if p > 0 and c > 0:
            out.append(math.log(c / p))
    return out


def _log_returns_open_to_close(bars: list[dict]) -> list[float]:
    out = []
    for bar in bars:
        o, c = float(bar.get("open") or 0), float(bar.get("close") or 0)
        if o > 0 and c > 0:
            out.append(math.log(c / o))
    return out


def annualized_vol(returns: list[float]) -> float | None:
    """Annualized stdev of per-trading-day log returns."""
    if len(returns) < 5:
        return None
    return statistics.pstdev(returns) * math.sqrt(TRADING_DAYS)


@dataclass
class Regime:
    """A measured snapshot of whether premium is worth selling right now."""

    spot: float
    atm_iv: float
    rv_close_to_close: float | None
    rv_open_to_close: float | None
    vix: float | None
    vix_percentile: float | None
    todays_move_sigma: float | None
    lookback_days: int

    @property
    def realized_for_horizon(self) -> float | None:
        """The horizon-matched realized vol: open-to-close for a 0DTE seller."""
        return self.rv_open_to_close or self.rv_close_to_close

    @property
    def conservative_realized(self) -> float | None:
        """The larger of the two realized estimates, i.e. the smaller premium."""
        candidates = [v for v in (self.rv_open_to_close, self.rv_close_to_close) if v]
        return max(candidates) if candidates else None

    def vrp(self, conservative: bool = True) -> float | None:
        """Implied / realized, minus one. 0.20 means implied is 20% rich.

        Positive is the premium you are being paid. Zero or negative means the
        market is charging no more than the index has been delivering, and there
        is nothing for a seller to harvest.
        """
        rv = self.conservative_realized if conservative else self.realized_for_horizon
        if not rv or rv <= 0:
            return None
        return self.atm_iv / rv - 1.0

    def implied_vol_multiplier(self, conservative: bool = True) -> float | None:
        """Realized / implied -- the measured equivalent of beliefs.vol_multiplier.

        Feeding this into the expectancy model replaces a guess with a
        measurement. It is still a forecast, because it assumes the trailing
        window describes tomorrow, but it is one grounded in what the index has
        actually done rather than in what you would like it to do.
        """
        rv = self.conservative_realized if conservative else self.realized_for_horizon
        if not rv or self.atm_iv <= 0:
            return None
        return rv / self.atm_iv

    def describe(self) -> list[str]:
        lines = [
            f"spot                 {self.spot:,.2f}",
            f"ATM implied vol      {self.atm_iv:.1%}",
        ]
        if self.rv_open_to_close:
            lines.append(
                f"realized (o->c)      {self.rv_open_to_close:.1%}  "
                f"<- horizon-matched for 0DTE"
            )
        if self.rv_close_to_close:
            lines.append(
                f"realized (c->c)      {self.rv_close_to_close:.1%}  "
                f"<- includes overnight gaps you never hold"
            )
        vrp = self.vrp()
        if vrp is not None:
            lines.append(f"variance risk prem.  {vrp:+.1%} (conservative reading)")
        if self.vix is not None:
            pct = f", {self.vix_percentile:.0%}ile of last year" if self.vix_percentile else ""
            lines.append(f"VIX                  {self.vix:.2f}{pct}")
        if self.todays_move_sigma is not None:
            lines.append(f"today's move         {self.todays_move_sigma:+.2f} sigma")
        lines.append(f"lookback             {self.lookback_days} trading days")
        return lines


def atm_implied_vol(contracts: list[OptionContract], spot: float) -> float | None:
    """Average the two ATM straddle legs' implied vols.

    Averaging call and put at the same strike cancels most of the put-call
    parity noise that a single leg carries when the synthetic forward drifts
    from spot.
    """
    solved = [c for c in contracts if c.iv is not None and c.iv > 0 and c.is_quotable]
    if not solved:
        return None
    nearest = min(abs(c.strike - spot) for c in solved)
    at_strike = [c.iv for c in solved if abs(abs(c.strike - spot) - nearest) < 1e-9]
    return statistics.mean(at_strike) if at_strike else None


def todays_open(client: TradierClient, symbol: str, today: date) -> float | None:
    """Today's opening print, for measuring how far the session has travelled."""
    try:
        bars = client.history(symbol, today, today)
    except TradierError:
        return None
    for bar in bars:
        if bar.get("date") == today.isoformat():
            value = float(bar.get("open") or 0)
            return value or None
    return None


def measure(
    client: TradierClient,
    cfg: Config,
    spot: float,
    contracts: list[OptionContract],
    lookback_days: int = 60,
    now: datetime | None = None,
) -> Regime | None:
    """Build a regime snapshot from live chain plus trailing daily bars."""
    now = now or datetime.now(ET)
    today = now.date()

    iv = atm_implied_vol(contracts, spot)
    if iv is None:
        return None

    # Ask for extra calendar days so weekends and holidays still leave enough bars.
    start = today - timedelta(days=int(lookback_days * 1.6) + 10)
    try:
        bars = client.history(cfg.symbol, start, today)
    except TradierError:
        bars = []
    bars = [b for b in bars if b.get("date", "") < today.isoformat()][-lookback_days:]

    rv_cc = annualized_vol(_log_returns_close_to_close(bars))
    rv_oc = annualized_vol(_log_returns_open_to_close(bars))

    vix = vix_pct = None
    try:
        vix_bars = client.history("VIX", today - timedelta(days=400), today)
        closes = [float(b["close"]) for b in vix_bars if b.get("close")]
        if closes:
            vix = closes[-1]
            vix_pct = sum(1 for c in closes if c <= vix) / len(closes)
    except (TradierError, KeyError, ValueError):
        pass

    move_sigma = None
    open_px = todays_open(client, cfg.symbol, today)
    daily_rv = rv_cc or rv_oc
    if open_px and daily_rv and spot > 0:
        daily_sigma = daily_rv / math.sqrt(TRADING_DAYS)
        if daily_sigma > 0:
            move_sigma = math.log(spot / open_px) / daily_sigma

    return Regime(
        spot=spot,
        atm_iv=iv,
        rv_close_to_close=rv_cc,
        rv_open_to_close=rv_oc,
        vix=vix,
        vix_percentile=vix_pct,
        todays_move_sigma=move_sigma,
        lookback_days=len(bars),
    )


def resolve_vol_multiplier(regime: "Regime | None", cfg: Config) -> tuple[float, str]:
    """Decide the vol multiplier to price expectancy with.

    Returns the value and a one-line explanation of where it came from. A number
    that silently flips the sign of every expected value in the output should
    never appear without its provenance attached, so the explanation is carried
    alongside it everywhere rather than reconstructed at the display layer.
    """
    if cfg.beliefs.source == "manual" or regime is None:
        return cfg.beliefs.vol_multiplier, f"manual vol_multiplier {cfg.beliefs.vol_multiplier:.2f}"

    measured = regime.implied_vol_multiplier(conservative=cfg.gates.conservative_vrp)
    if measured is None:
        return cfg.beliefs.vol_multiplier, (
            f"could not measure realized vol; fell back to manual "
            f"{cfg.beliefs.vol_multiplier:.2f}"
        )

    # The haircut only ever makes the assumption less favourable, and it can
    # never push the multiplier above 1.0 (which would mean betting that
    # realized exceeds implied -- a reason to buy premium, not sell it).
    applied = min(1.0, measured + cfg.beliefs.haircut)
    return applied, (
        f"measured realized/implied = {measured:.2f}, +{cfg.beliefs.haircut:.2f} haircut "
        f"-> {applied:.2f}"
    )


@dataclass
class GateResult:
    """Whether conditions permit entering, and the reason if not."""

    passed: bool
    blocks: list[str]
    notes: list[str]

    @property
    def summary(self) -> str:
        return "conditions met" if self.passed else "; ".join(self.blocks)


def check_gates(
    regime: Regime | None,
    cfg: Config,
    now: datetime | None = None,
    calendar: "EconCalendar | None" = None,
) -> GateResult:
    """Apply every entry condition. All must pass; each failure is named.

    Gates are deliberately phrased as reasons *not* to trade. The default state
    is "do not enter", and conditions have to argue their way past that.

    ``calendar`` is optional so that callers which only want the volatility
    gates -- and tests which do not want to touch the event seed -- can skip it.
    When it is omitted no event checking happens at all, which is why the watch
    loop always passes one.
    """
    now = now or datetime.now(ET)
    g = cfg.gates
    blocks: list[str] = []
    notes: list[str] = []

    if regime is None:
        return GateResult(False, ["could not measure the regime (no ATM implied vol)"], [])

    # --- the premium actually on offer -----------------------------------
    vrp = regime.vrp(conservative=g.conservative_vrp)
    other = regime.vrp(conservative=not g.conservative_vrp)
    basis = "close-to-close" if g.conservative_vrp else "open-to-close"
    alt_basis = "open-to-close" if g.conservative_vrp else "close-to-close"

    if vrp is None:
        blocks.append("realized volatility unavailable, so the premium cannot be measured")
    elif vrp < g.min_variance_risk_premium:
        # Both readings are quoted, always. The gap between them is often large
        # -- measured on real SPX bars, close-to-close realized ran 13.7% against
        # 10.1% open-to-close, because a third of the variance arrives overnight.
        # A 0DTE seller never holds that overnight risk, so the stricter reading
        # is deliberately mismatched. Showing only the number that blocked would
        # invite you to disable the gate rather than understand it.
        detail = f" (the {alt_basis} reading is {other:+.1%})" if other is not None else ""
        blocks.append(
            f"variance risk premium is {vrp:+.1%} against {basis} realized vol, below the "
            f"{g.min_variance_risk_premium:+.1%} floor{detail} -- implied vol is not rich "
            f"enough versus what the index is actually delivering"
        )
    else:
        detail = f"; {alt_basis} reading {other:+.1%}" if other is not None else ""
        notes.append(f"implied vol is {vrp:+.1%} rich vs {basis} realized{detail}")

    # --- do not sell vol that is already on the floor ---------------------
    if g.min_vix_percentile > 0 and regime.vix_percentile is not None:
        if regime.vix_percentile < g.min_vix_percentile:
            blocks.append(
                f"VIX is at the {regime.vix_percentile:.0%} percentile of the last year, "
                f"below the {g.min_vix_percentile:.0%} floor -- premium is cheap and the "
                f"downside tail is not"
            )

    # --- do not sell into a move already in progress ----------------------
    if regime.todays_move_sigma is not None:
        if abs(regime.todays_move_sigma) > g.max_todays_move_sigma:
            blocks.append(
                f"the index has already moved {regime.todays_move_sigma:+.2f} sigma today, "
                f"past the {g.max_todays_move_sigma:.2f} limit -- trend days are precisely "
                f"when short strikes get run over"
            )

    # --- time of day -------------------------------------------------------
    minutes = now.hour * 60 + now.minute
    start = g.entry_start_hour * 60 + g.entry_start_minute
    end = g.entry_end_hour * 60 + g.entry_end_minute
    if minutes < start:
        blocks.append(
            f"before the {g.entry_start_hour:02d}:{g.entry_start_minute:02d} ET entry window "
            f"-- the opening auction's quotes are wide and its direction is noise"
        )
    elif minutes > end:
        blocks.append(
            f"past the {g.entry_end_hour:02d}:{g.entry_end_minute:02d} ET cutoff -- too little "
            f"time left to manage a position that goes against you"
        )

    # --- explicit blackouts -------------------------------------------------
    today_iso = now.date().isoformat()
    if today_iso in set(g.blackout_dates):
        blocks.append(f"{today_iso} is in your configured blackout list")

    # --- scheduled economic events ------------------------------------------
    if calendar is not None:
        from .events import check_event_gate

        event_blocks, event_notes = check_event_gate(calendar, cfg, now=now)
        blocks.extend(event_blocks)
        notes.extend(event_notes)

    return GateResult(passed=not blocks, blocks=blocks, notes=notes)
