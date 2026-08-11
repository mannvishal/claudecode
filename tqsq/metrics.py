"""Performance statistics, with the honesty machinery attached.

Point estimates on a few dozen trades are close to meaningless, so every
headline number here ships with a bootstrap confidence interval, and the
baselines exist so a positive return can be read against what doing something
arbitrary would have earned over the same tape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import BacktestResult

TRADING_DAYS = 252


@dataclass
class Stats:
    n_trades: int
    total_pnl: float
    total_return: float
    win_rate: float
    mean_trade_ret: float
    mean_trade_ci: tuple[float, float]
    expectancy: float
    profit_factor: float
    sharpe: float
    max_drawdown: float
    avg_hold_minutes: float
    gap_fill_share: float

    def render(self, title: str = "") -> str:
        lo, hi = self.mean_trade_ci
        sign = "+" if self.total_return >= 0 else ""
        return "\n".join(
            [
                f"--- {title} ---" if title else "---",
                f"trades           {self.n_trades}",
                f"total P&L        ${self.total_pnl:,.0f}  ({sign}{self.total_return*100:.2f}%)",
                f"win rate         {self.win_rate*100:.1f}%",
                f"mean trade       {self.mean_trade_ret*100:+.3f}%   95% CI "
                f"[{lo*100:+.3f}%, {hi*100:+.3f}%]",
                f"expectancy       ${self.expectancy:,.2f}/trade",
                f"profit factor    {self.profit_factor:.2f}",
                f"Sharpe (daily)   {self.sharpe:.2f}",
                f"max drawdown     {self.max_drawdown*100:.2f}%",
                f"avg hold         {self.avg_hold_minutes:.0f} min",
                f"gap-through fills {self.gap_fill_share*100:.1f}%  (stops that fired through the level)",
            ]
        )


def bootstrap_ci(x: np.ndarray, n: int = 10_000, alpha: float = 0.05,
                 seed: int = 7) -> tuple[float, float]:
    """Percentile bootstrap on the mean. Returns (nan, nan) below 2 samples."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def summarize(res: BacktestResult) -> Stats:
    df = res.frame()
    cap = res.config.capital if res.config else 100_000.0
    if df.empty:
        nan = float("nan")
        return Stats(0, 0.0, 0.0, nan, nan, (nan, nan), nan, nan, nan, nan, nan, nan)

    rets = df["ret"].to_numpy(float)
    pnl = df["pnl"].to_numpy(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]

    eq = res.equity
    daily_ret = eq.pct_change().dropna() if eq is not None and len(eq) > 2 else pd.Series(dtype=float)
    sharpe = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS))
        if len(daily_ret) > 2 and daily_ret.std() > 0
        else float("nan")
    )
    dd = float(((eq - eq.cummax()) / eq.cummax()).min()) if eq is not None and len(eq) > 1 else 0.0

    return Stats(
        n_trades=len(df),
        total_pnl=float(pnl.sum()),
        total_return=float(pnl.sum() / cap),
        win_rate=float((pnl > 0).mean()),
        mean_trade_ret=float(rets.mean()),
        mean_trade_ci=bootstrap_ci(rets),
        expectancy=float(pnl.mean()),
        profit_factor=float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf"),
        sharpe=sharpe,
        max_drawdown=dd,
        avg_hold_minutes=float(df["hold_minutes"].mean()),
        gap_fill_share=float(df["gap_fill"].mean()),
    )


def buy_and_hold(bars: pd.DataFrame) -> float:
    """Total return of holding the signal instrument over the same window."""
    c = bars["close"].to_numpy(float)
    return float(c[-1] / c[0] - 1.0)


def random_entry_baseline(
    bars: pd.DataFrame,
    n_trades: int,
    hold_minutes: int,
    n_sims: int = 2_000,
    seed: int = 11,
) -> tuple[float, tuple[float, float]]:
    """Mean per-trade return of entering at random times and holding as long.

    This is the number the strategy has to beat. A leveraged ETF in an uptrend
    makes almost any long look profitable, and this baseline prices that in.
    """
    c = bars["close"].to_numpy(float)
    n = len(c) - hold_minutes - 1
    if n <= 1 or n_trades <= 0:
        return (float("nan"), (float("nan"), float("nan")))
    rng = np.random.default_rng(seed)
    sims = np.empty(n_sims)
    for k in range(n_sims):
        idx = rng.integers(0, n, size=n_trades)
        sims[k] = np.mean(c[idx + hold_minutes] / c[idx] - 1.0)
    return (float(sims.mean()), (float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))))


def by_tier(res: BacktestResult) -> pd.DataFrame:
    """Break results out by how many tranches the episode reached."""
    df = res.frame()
    if df.empty:
        return df
    g = df.groupby(["side", "tranches"])
    return pd.DataFrame(
        {
            "n": g.size(),
            "win_rate": g["pnl"].apply(lambda s: (s > 0).mean()),
            "mean_ret_pct": g["ret"].mean() * 100,
            "total_pnl": g["pnl"].sum(),
        }
    ).reset_index()


def by_exit_reason(res: BacktestResult) -> pd.DataFrame:
    df = res.frame()
    if df.empty:
        return df
    g = df.groupby("reason")
    return pd.DataFrame(
        {"n": g.size(), "mean_ret_pct": g["ret"].mean() * 100, "total_pnl": g["pnl"].sum()}
    ).reset_index().sort_values("total_pnl")
