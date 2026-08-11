"""Three-tranche backtest of the exhaustion marks.

The trade under test, stated plainly:

  bottom exhaustion tier k  ->  buy TQQQ  (the down-stretch is tiring)
  top exhaustion tier k     ->  buy SQQQ  (the up-stretch is tiring)

One tranche per tier, so a full episode that runs 1 -> 2 -> 3 ends up three
tranches deep at a worse and worse average price. That is the shape the strategy
was described in, and it is worth being clear that it is a MARTINGALE: each add
happens because the previous one is losing. The stop is therefore the only thing
standing between the strategy and an unbounded loss, which is why every result
in the README is quoted per stop level rather than averaged over them.

No-lookahead discipline:
  * a fire on bar i is filled at bar i+1's OPEN, never bar i's close;
  * stops and targets are tested against bar highs/lows and filled AT the level,
    which flatters the strategy on gaps -- see `gap_fills` in the result;
  * the score at bar i uses only completed higher-timeframe bars plus bar i's
    own close, which is what the Pine does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .signals import RTH_CLOSE_MIN, SignalFrame

SIDE_LONG = "TQQQ"
SIDE_SHORT = "SQQQ"


@dataclass(frozen=True)
class BacktestConfig:
    capital: float = 100_000.0
    # Fraction of capital committed per tranche. Three tranches at 1/3 each puts
    # the whole sleeve to work only on a full 1->2->3 episode.
    tranche_frac: float = 1.0 / 3.0
    max_tranches: int = 3
    stop_pct: float = 0.02          # stop on the position's average cost
    take_profit_pct: float = 0.02   # 0 disables
    exit_on_neutral: bool = True    # score returns inside the +/-4 band
    max_hold_minutes: int = 0       # 0 disables
    flat_at_session_end: bool = True
    session_end_minute: int = RTH_CLOSE_MIN
    slippage_bps: float = 2.0       # per fill, each way
    commission: float = 0.0         # Robinhood equities
    require_amplifier: bool = False  # +climax or +divergence on the fire bar
    min_tier: int = 1               # ignore fires shallower than this


@dataclass
class Trade:
    side: str
    symbol: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    tranches: int
    shares: float
    avg_cost: float
    exit_price: float
    pnl: float
    ret: float
    reason: str
    hold_minutes: int
    max_adverse_pct: float
    gap_fill: bool


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity: pd.Series | None = None
    config: BacktestConfig | None = None

    def frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in self.trades])


class _Position:
    """One side's open exposure across up to three tranches."""

    __slots__ = ("symbol", "side", "shares", "cost", "entry_ts", "tiers", "entry_row", "worst")

    def __init__(self, symbol: str, side: str, ts, row: int):
        self.symbol, self.side = symbol, side
        self.shares = 0.0
        self.cost = 0.0
        self.entry_ts = ts
        self.entry_row = row
        self.tiers: set[int] = set()
        self.worst = 0.0

    @property
    def avg_cost(self) -> float:
        return self.cost / self.shares if self.shares > 0 else float("nan")

    def add(self, shares: float, price: float, tier: int) -> None:
        self.shares += shares
        self.cost += shares * price
        self.tiers.add(tier)


def _run_side(
    bars: pd.DataFrame,
    tiers: np.ndarray,
    amplifier: np.ndarray,
    neutral: np.ndarray,
    episode_reset: np.ndarray,
    cfg: BacktestConfig,
    symbol: str,
    side: str,
) -> list[Trade]:
    """Simulate one side. `tiers[i]` is the deepest tier firing on bar i, 0 if none."""
    ts = bars["ts"].to_numpy()
    minute = bars["minute"].to_numpy()
    op = bars["open"].to_numpy(float)
    hi = bars["high"].to_numpy(float)
    lo = bars["low"].to_numpy(float)
    cl = bars["close"].to_numpy(float)

    slip = cfg.slippage_bps / 1e4
    trades: list[Trade] = []
    pos: _Position | None = None
    # After a position closes, block re-entry until the score has genuinely left
    # the stretch band. Without this the machines re-fire inside the same episode
    # and the "3 tranches" become an unbounded ladder.
    blocked = False

    n = len(bars)
    for i in range(n - 1):
        if episode_reset[i]:
            blocked = False

        # ---- manage an open position on THIS bar, before considering adds ----
        if pos is not None:
            stop_px = pos.avg_cost * (1.0 - cfg.stop_pct)
            tp_px = pos.avg_cost * (1.0 + cfg.take_profit_pct) if cfg.take_profit_pct > 0 else np.inf
            adverse = (lo[i] / pos.avg_cost) - 1.0
            pos.worst = min(pos.worst, adverse)

            exit_px = None
            reason = ""
            gap = False
            # Stop first: within one bar we cannot know the order, and assuming
            # the loss came first is the conservative reading.
            if lo[i] <= stop_px:
                exit_px = min(stop_px, op[i])  # a gap-through fills at the open
                gap = op[i] < stop_px
                reason = "stop"
            elif hi[i] >= tp_px:
                exit_px = max(tp_px, op[i])
                gap = op[i] > tp_px
                reason = "target"
            elif cfg.exit_on_neutral and neutral[i]:
                exit_px = cl[i]
                reason = "score_neutral"
            elif cfg.max_hold_minutes and (i - pos.entry_row) >= cfg.max_hold_minutes:
                exit_px = cl[i]
                reason = "time"
            elif cfg.flat_at_session_end and minute[i] >= cfg.session_end_minute - 1:
                exit_px = cl[i]
                reason = "session_end"

            if exit_px is not None:
                fill = exit_px * (1.0 - slip)
                pnl = pos.shares * (fill - pos.avg_cost) - cfg.commission
                trades.append(
                    Trade(
                        side=side,
                        symbol=symbol,
                        entry_ts=pd.Timestamp(ts[pos.entry_row]),
                        exit_ts=pd.Timestamp(ts[i]),
                        tranches=len(pos.tiers),
                        shares=pos.shares,
                        avg_cost=pos.avg_cost,
                        exit_price=fill,
                        pnl=pnl,
                        ret=fill / pos.avg_cost - 1.0,
                        reason=reason,
                        hold_minutes=i - pos.entry_row,
                        max_adverse_pct=pos.worst,
                        gap_fill=gap,
                    )
                )
                pos = None
                blocked = True

        # ---- new fire on bar i, filled at bar i+1's open ----
        tier = int(tiers[i])
        if tier == 0 or tier < cfg.min_tier:
            continue
        if cfg.require_amplifier and not amplifier[i]:
            continue
        if blocked:
            continue
        if pos is not None and (tier in pos.tiers or len(pos.tiers) >= cfg.max_tranches):
            continue

        fill = op[i + 1] * (1.0 + slip)
        if not np.isfinite(fill) or fill <= 0:
            continue
        notional = cfg.capital * cfg.tranche_frac
        shares = notional / fill
        if pos is None:
            pos = _Position(symbol, side, ts[i + 1], i + 1)
        pos.add(shares, fill, tier)

    return trades


def run_backtest(
    signal_bars: pd.DataFrame,
    signals: SignalFrame,
    long_bars: pd.DataFrame,
    short_bars: pd.DataFrame,
    cfg: BacktestConfig = BacktestConfig(),
) -> BacktestResult:
    """Run both sides and merge into one equity curve.

    `signal_bars`/`signals` are computed on the signal symbol (TQQQ). `long_bars`
    and `short_bars` are the tradeable instruments, reindexed onto the signal
    clock so a fire always has a next bar to fill against.
    """
    idx = signal_bars["ts"]
    lb = long_bars.set_index("ts").reindex(idx).ffill().reset_index()
    sb = short_bars.set_index("ts").reindex(idx).ffill().reset_index()
    lb["minute"] = signal_bars["minute"].to_numpy()
    sb["minute"] = signal_bars["minute"].to_numpy()

    score = signals.score
    neutral = (score < 4.0) & (score > -4.0)
    amp_bot = signals.climax | signals.div_bot
    amp_top = signals.climax | signals.div_top

    trades = _run_side(
        lb, signals.exhaustion_tier("bot"), amp_bot, neutral, neutral, cfg, SIDE_LONG, "long_tqqq"
    ) + _run_side(
        sb, signals.exhaustion_tier("top"), amp_top, neutral, neutral, cfg, SIDE_SHORT, "long_sqqq"
    )
    trades.sort(key=lambda t: t.exit_ts)

    equity = _equity_curve(trades, cfg.capital, idx)
    return BacktestResult(trades=trades, equity=equity, config=cfg)


def _equity_curve(trades: list[Trade], capital: float, idx: pd.Series) -> pd.Series:
    """Realised equity, stamped at each trade's exit and forward-filled daily."""
    if not trades:
        return pd.Series([capital], index=[pd.Timestamp(idx.iloc[0])])
    df = pd.DataFrame({"ts": [t.exit_ts for t in trades], "pnl": [t.pnl for t in trades]})
    daily = df.groupby(df["ts"].dt.normalize())["pnl"].sum()
    all_days = pd.Series(0.0, index=pd.DatetimeIndex(sorted(set(idx.dt.normalize()))))
    daily = daily.reindex(all_days.index).fillna(0.0)
    return capital + daily.cumsum()
