"""Command line entry points: `tqsq cost|fetch|backtest|sweep|decide`."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _load(args):
    from .bars import load_bars
    from .score import compute_score
    from .signals import SignalConfig, build_signals

    sig = load_bars(args.bars, args.signal_symbol, session=args.session)
    long_bars = load_bars(args.bars, "TQQQ", session=args.session)
    short_bars = load_bars(args.bars, "SQQQ", session=args.session)
    sf = compute_score(sig)
    sg = build_signals(sig, sf.avg, sf.avg_rsi, SignalConfig(eth_marks=(args.session == "eth")))
    return sig, sg, long_bars, short_bars


def _base_cfg(args):
    from .backtest import BacktestConfig
    from .bars import SESSIONS

    return BacktestConfig(
        capital=args.capital,
        stop_pct=args.stop / 100.0,
        take_profit_pct=args.target / 100.0,
        slippage_bps=args.slippage,
        session_end_minute=SESSIONS[args.session][1],
        min_tier=args.min_tier,
        require_amplifier=args.amplifier,
    )


def cmd_cost(args) -> int:
    from .data import FetchPlan, estimate_cost

    plan = FetchPlan(args.start, args.end, tuple(args.symbols))
    print(f"${estimate_cost(plan):.4f}  for {plan.symbols} {plan.start}..{plan.end} ({plan.schema})")
    print("note: this prices the REQUEST, not your remaining balance -- a cheap")
    print("      estimate can still return 402 account_insufficient_funds.")
    return 0


def cmd_fetch(args) -> int:
    from .data import consolidate, fetch

    try:
        paths = fetch(args.start, args.end, tuple(args.symbols), ceiling=args.ceiling)
    except RuntimeError as exc:
        print(f"fetch stopped: {exc}", file=sys.stderr)
        paths = []
    out = consolidate()
    print(f"{len(paths)} chunk(s) cached; consolidated -> {out}")
    return 0


def cmd_backtest(args) -> int:
    from . import metrics
    from .backtest import run_backtest

    sig, sg, lb, sb = _load(args)
    res = run_backtest(sig, sg, lb, sb, _base_cfg(args))
    st = metrics.summarize(res)
    print(st.render(f"{args.session.upper()} stop={args.stop}% target={args.target}% slip={args.slippage}bps"))
    print(f"\nbuy & hold TQQQ over the same window: {metrics.buy_and_hold(lb)*100:+.2f}%")
    print("\n-- exits --")
    print(metrics.by_exit_reason(res).to_string(index=False))
    print("\n-- by tranche depth --")
    print(metrics.by_tier(res).to_string(index=False))
    if args.out:
        res.frame().to_csv(args.out, index=False)
        print(f"\ntrades -> {args.out}")
    return 0


def cmd_sweep(args) -> int:
    from . import sweep
    from .backtest import run_backtest

    sig, sg, lb, sb = _load(args)
    base = _base_cfg(args)
    fmt = lambda x: f"{x:8.2f}"  # noqa: E731

    print("== slippage sensitivity ==")
    print(sweep.slippage_curve(sig, sg, lb, sb, base).to_string(index=False, float_format=fmt))
    print("\n== stop x target grid ==")
    print(sweep.grid(sig, sg, lb, sb, base).to_string(index=False, float_format=fmt))
    print("\n== tier / amplifier ==")
    print(sweep.tier_floor_curve(sig, sg, lb, sb, base).to_string(index=False, float_format=fmt))

    res = run_backtest(sig, sg, lb, sb, base)
    print("\n== vs random entry of the same duration ==")
    print(json.dumps(sweep.baseline_comparison(res, lb), indent=2))
    print("\n== split half ==")
    print(sweep.split_half(res).to_string(index=False, float_format=fmt))
    return 0


def cmd_decide(args) -> int:
    from .live import EngineState, RiskLimits, bars_from_json, decide, write_intents
    from .signals import SignalConfig

    bars = bars_from_json(args.bars_json)
    state = EngineState.load(args.state)
    limits = RiskLimits(
        max_position_usd=args.max_position,
        max_daily_loss_usd=args.max_daily_loss,
        stop_pct=args.stop / 100.0,
    )
    quotes = json.loads(args.quotes) if args.quotes else None
    intents, diag = decide(
        bars, state, limits,
        SignalConfig(eth_marks=(args.session == "eth")),
        session=args.session, quotes=quotes,
    )
    write_intents(intents, diag, args.out)
    state.save(args.state)
    print(json.dumps({"diagnostics": diag, "intents": [i.to_dict() for i in intents]}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("tqsq", description="TQQQ/SQQQ exhaustion-mark strategy")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--bars", default="data/raw/bars_1m.parquet")
        sp.add_argument("--signal-symbol", default="TQQQ")
        sp.add_argument("--session", default="rth", choices=["rth", "eth"])
        sp.add_argument("--capital", type=float, default=100_000.0)
        sp.add_argument("--stop", type=float, default=2.0, help="stop loss, percent")
        sp.add_argument("--target", type=float, default=2.0, help="take profit, percent; 0 disables")
        sp.add_argument("--slippage", type=float, default=5.0, help="basis points per fill")
        sp.add_argument("--min-tier", type=int, default=1, choices=[1, 2, 3])
        sp.add_argument("--amplifier", action="store_true", help="require +climax or +divergence")

    c = sub.add_parser("cost", help="price a Databento pull (free)")
    c.add_argument("--start", required=True)
    c.add_argument("--end", required=True)
    c.add_argument("--symbols", nargs="+", default=["TQQQ", "SQQQ"])
    c.set_defaults(func=cmd_cost)

    f = sub.add_parser("fetch", help="download and consolidate 1-minute bars")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--symbols", nargs="+", default=["TQQQ", "SQQQ"])
    f.add_argument("--ceiling", type=float, default=25.0)
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("backtest", help="run one configuration")
    add_common(b)
    b.add_argument("--out", help="write the trade list to CSV")
    b.set_defaults(func=cmd_backtest)

    s = sub.add_parser("sweep", help="robustness tables")
    add_common(s)
    s.set_defaults(func=cmd_sweep)

    d = sub.add_parser("decide", help="one live decision from a bar file")
    d.add_argument("--bars-json", required=True)
    d.add_argument("--state", default="state.json")
    d.add_argument("--out", default="intents.json")
    d.add_argument("--session", default="rth", choices=["rth", "eth"])
    d.add_argument("--stop", type=float, default=2.0)
    d.add_argument("--max-position", type=float, default=10_000.0)
    d.add_argument("--max-daily-loss", type=float, default=500.0)
    d.add_argument("--quotes", help='JSON dict of last prices, e.g. \'{"TQQQ":74.1,"SQQQ":42.3}\'')
    d.set_defaults(func=cmd_decide)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
