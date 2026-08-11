"""Parameter sweeps and the robustness checks that decide whether to believe them.

A sweep over stop and target on 43 sessions will always produce a best cell. The
question this module is built to answer is whether that cell is anything other
than the luckiest draw, so it reports the whole surface rather than the argmax,
and it prices the two assumptions most likely to be carrying the result:
slippage and the extended-hours session.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .metrics import bootstrap_ci, random_entry_baseline, summarize


def grid(
    sig_bars, signals, long_bars, short_bars,
    base: BacktestConfig,
    stops=(0.005, 0.01, 0.015, 0.02, 0.03, 0.05),
    targets=(0.0, 0.005, 0.01, 0.02, 0.03),
) -> pd.DataFrame:
    """Full stop x target surface. Every cell, not just the winner."""
    rows = []
    for stop, tp in itertools.product(stops, targets):
        cfg = replace(base, stop_pct=stop, take_profit_pct=tp)
        st = summarize(run_backtest(sig_bars, signals, long_bars, short_bars, cfg))
        rows.append(
            {
                "stop_pct": stop * 100,
                "tp_pct": tp * 100,
                "trades": st.n_trades,
                "total_return_pct": st.total_return * 100,
                "win_rate": st.win_rate,
                "mean_trade_bps": st.mean_trade_ret * 1e4,
                "ci_lo_bps": st.mean_trade_ci[0] * 1e4,
                "ci_hi_bps": st.mean_trade_ci[1] * 1e4,
                "sharpe": st.sharpe,
                "max_dd_pct": st.max_drawdown * 100,
                "profit_factor": st.profit_factor,
            }
        )
    return pd.DataFrame(rows)


def slippage_curve(
    sig_bars, signals, long_bars, short_bars, base: BacktestConfig,
    bps=(0.0, 2.0, 5.0, 10.0, 20.0, 30.0),
) -> pd.DataFrame:
    """Where does the edge die?

    This is the single most important table in the project. TQQQ quotes a
    penny-wide spread in the regular session (about 1.5 bps at $74) but ten to
    thirty times that at 04:00, and the strategy takes liquidity on entry and on
    every stop. A result that only survives at 2 bps is a result about the
    fill assumption, not about the signal.
    """
    rows = []
    for b in bps:
        cfg = replace(base, slippage_bps=b)
        st = summarize(run_backtest(sig_bars, signals, long_bars, short_bars, cfg))
        rows.append(
            {
                "slippage_bps": b,
                "trades": st.n_trades,
                "total_return_pct": st.total_return * 100,
                "mean_trade_bps": st.mean_trade_ret * 1e4,
                "ci_lo_bps": st.mean_trade_ci[0] * 1e4,
                "ci_hi_bps": st.mean_trade_ci[1] * 1e4,
                "profit_factor": st.profit_factor,
                "sharpe": st.sharpe,
            }
        )
    return pd.DataFrame(rows)


def tier_floor_curve(sig_bars, signals, long_bars, short_bars, base: BacktestConfig) -> pd.DataFrame:
    """Does requiring a deeper tier -- or an amplifier -- actually help?"""
    rows = []
    for tier in (1, 2, 3):
        for amp in (False, True):
            cfg = replace(base, min_tier=tier, require_amplifier=amp)
            st = summarize(run_backtest(sig_bars, signals, long_bars, short_bars, cfg))
            rows.append(
                {
                    "min_tier": tier,
                    "amplifier_required": amp,
                    "trades": st.n_trades,
                    "total_return_pct": st.total_return * 100,
                    "win_rate": st.win_rate,
                    "mean_trade_bps": st.mean_trade_ret * 1e4,
                    "ci_lo_bps": st.mean_trade_ci[0] * 1e4,
                    "ci_hi_bps": st.mean_trade_ci[1] * 1e4,
                }
            )
    return pd.DataFrame(rows)


def baseline_comparison(res, bars, hold_minutes: int | None = None) -> dict:
    """Strategy mean trade against random entries of the same duration."""
    df = res.frame()
    if df.empty:
        return {}
    hold = int(hold_minutes or df["hold_minutes"].mean())
    mean, ci = random_entry_baseline(bars, n_trades=len(df), hold_minutes=max(hold, 1))
    strat = float(df["ret"].mean())
    return {
        "strategy_mean_bps": strat * 1e4,
        "random_mean_bps": mean * 1e4,
        "random_ci_bps": (ci[0] * 1e4, ci[1] * 1e4),
        "beats_random": bool(strat > ci[1]),
        "hold_minutes": hold,
    }


def split_half(res) -> pd.DataFrame:
    """First half vs second half of the trade sequence.

    A signal that only works in one half is a signal about that period.
    """
    df = res.frame()
    if len(df) < 8:
        return pd.DataFrame()
    df = df.sort_values("exit_ts").reset_index(drop=True)
    mid = len(df) // 2
    rows = []
    for label, part in (("first_half", df.iloc[:mid]), ("second_half", df.iloc[mid:])):
        r = part["ret"].to_numpy(float)
        lo, hi = bootstrap_ci(r)
        rows.append(
            {
                "half": label,
                "trades": len(part),
                "mean_trade_bps": r.mean() * 1e4,
                "ci_lo_bps": lo * 1e4,
                "ci_hi_bps": hi * 1e4,
                "win_rate": float((part["pnl"] > 0).mean()),
                "total_pnl": float(part["pnl"].sum()),
            }
        )
    return pd.DataFrame(rows)
