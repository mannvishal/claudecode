"""Rule 7: the per-trade log and the aggregate statistics."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import pandas as pd

from .engine import SessionResult, Trade


def trade_log(results: list[SessionResult]) -> pd.DataFrame:
    rows = [r.trade.as_row() for r in results if r.trade]
    if not rows:
        return pd.DataFrame(columns=[
            "day", "structure", "strikes", "entry_ts", "exit_ts", "contracts",
            "credit_pts", "exit_debit_pts", "gross_pnl", "commissions", "net_pnl",
            "exit_reason", "max_loss",
        ])
    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)


@dataclass
class Stats:
    trades: int
    skipped: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    worst_day: float
    best_day: float
    expectancy: float
    breakeven_win_rate: float
    total_commissions: float
    exit_reasons: dict

    def render(self) -> str:
        lines = [
            f"  trades                {self.trades}  ({self.skipped} sessions skipped)",
            f"  win rate              {self.win_rate:.1%}  ({self.wins}W / {self.losses}L)",
            f"  total net P&L         ${self.total_pnl:,.2f}",
            f"  expectancy / trade    ${self.expectancy:,.2f}",
            f"  average win           ${self.avg_win:,.2f}",
            f"  average loss          ${self.avg_loss:,.2f}",
            f"  worst single day      ${self.worst_day:,.2f}",
            f"  best single day       ${self.best_day:,.2f}",
            f"  max drawdown          ${self.max_drawdown:,.2f}",
            f"  commissions paid      ${self.total_commissions:,.2f}",
        ]
        if self.exit_reasons:
            joined = ", ".join(f"{k} {v}" for k, v in sorted(self.exit_reasons.items()))
            lines.append(f"  exits                 {joined}")
        return "\n".join(lines)


def compute_stats(results: list[SessionResult]) -> Stats:
    trades: list[Trade] = [r.trade for r in results if r.trade]
    skipped = sum(1 for r in results if r.trade is None)

    if not trades:
        return Stats(0, skipped, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})

    pnls = [t.net_pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Drawdown on the per-trade equity curve. With one trade per session this is
    # also the daily curve; if the strategy ever takes several a day, aggregate
    # by day before reading this as a daily figure.
    equity, peak, worst_dd = 0.0, 0.0, 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        worst_dd = min(worst_dd, equity - peak)

    by_day = {}
    for trade in trades:
        by_day[trade.day] = by_day.get(trade.day, 0.0) + trade.net_pnl

    avg_win = statistics.mean(wins) if wins else 0.0
    avg_loss = statistics.mean(losses) if losses else 0.0

    # The win rate you would need for this payoff shape to break even. Printed
    # next to the achieved rate because a high win rate on its own says nothing
    # -- selling premium is supposed to win often, and the question is only ever
    # whether it wins often enough.
    # Undefined with no losing trades, which is a small-sample artifact rather
    # than a strategy with no downside. Reporting 0% there would read as "you
    # need never win", which is the opposite of the truth.
    breakeven = (
        abs(avg_loss) / (avg_win + abs(avg_loss))
        if losses and wins and (avg_win + abs(avg_loss)) > 0
        else float("nan")
    )

    reasons: dict[str, int] = {}
    for trade in trades:
        reasons[trade.exit_reason] = reasons.get(trade.exit_reason, 0) + 1

    return Stats(
        trades=len(trades),
        skipped=skipped,
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(trades),
        total_pnl=sum(pnls),
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_drawdown=worst_dd,
        worst_day=min(by_day.values()),
        best_day=max(by_day.values()),
        expectancy=statistics.mean(pnls),
        breakeven_win_rate=breakeven,
        total_commissions=sum(t.commissions for t in trades),
        exit_reasons=reasons,
    )


def render_report(results: list[SessionResult], echo=print) -> pd.DataFrame:
    log = trade_log(results)
    stats = compute_stats(results)

    echo("\n" + "=" * 78)
    echo("PER-TRADE LOG")
    echo("=" * 78)
    if log.empty:
        echo("  no trades")
    else:
        display = log.copy()
        display["entry_ts"] = display["entry_ts"].dt.strftime("%H:%M:%S")
        display["exit_ts"] = display["exit_ts"].dt.strftime("%H:%M:%S")
        echo(display.to_string(index=False))

    skips = [(r.day, r.skipped) for r in results if r.trade is None and r.skipped]
    if skips:
        echo(f"\nSKIPPED ({len(skips)})")
        for day, why in skips[:20]:
            echo(f"  {day}  {why}")
        if len(skips) > 20:
            echo(f"  ... and {len(skips) - 20} more")

    echo("\n" + "=" * 78)
    echo("AGGREGATE")
    echo("=" * 78)
    echo(stats.render())

    if stats.trades:
        if stats.breakeven_win_rate == stats.breakeven_win_rate:  # not NaN
            echo(
                f"\n  Break-even win rate for this payoff shape: "
                f"{stats.breakeven_win_rate:.1%} (achieved {stats.win_rate:.1%})."
            )
        else:
            echo(
                f"\n  Break-even win rate: undefined -- this sample has "
                f"{'no losses' if not stats.losses else 'no wins'}, which is a sample-size "
                f"artifact, not an absence of downside."
            )
        if stats.avg_win > 0 and stats.worst_day < 0:
            echo(
                f"  The worst single day gave back "
                f"{abs(stats.worst_day) / stats.avg_win:.1f} average wins."
            )
    return log
