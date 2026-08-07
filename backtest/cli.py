"""Backtest CLI.

    smoke     run one date end-to-end and print the trade log (rule 8)
    run       run a date range
    cost      price a range without pulling anything
    cache     show what is cached and what it cost
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from .cache import ParquetCache
from .config import SCHEMA_DEFINITION, SCHEMA_MBP1, BacktestConfig
from .engine import Engine
from .fills import MidFill, MidMinusEdgeFill
from .report import render_report
from .source import (
    AgentBridgeFetcher,
    CostCeilingExceeded,
    CostEstimateUnavailable,
    DatabentoFetcher,
)

BANNER = """\
backtest -- SPX 0DTE credit spreads. Reads Databento, writes nothing to a broker.
Every pull is cost-estimated first and cached to ./data; cached ranges are never
re-pulled."""


def _fetcher(cfg: BacktestConfig, args):
    cache = ParquetCache(cfg.data.cache_dir)
    if args.offline:
        return AgentBridgeFetcher(cfg, cache), cache
    return DatabentoFetcher(cfg, cache), cache


def _fill_model(cfg: BacktestConfig, name: str):
    return {"adverse": MidMinusEdgeFill, "mid": MidFill}[name](cfg.execution)


def _configure(args) -> BacktestConfig:
    cfg = BacktestConfig.load(args.config)
    if args.ceiling is not None:
        cfg.cost.ceiling_usd = args.ceiling
    if args.commission is not None:
        cfg.execution.commission_per_contract = args.commission
    if args.slippage is not None:
        cfg.execution.entry_slippage_per_leg = args.slippage
    return cfg


def cmd_smoke(args) -> int:
    """Rule 8: one known date, end to end, with the trade log printed."""
    cfg = _configure(args)
    day = date.fromisoformat(args.date)
    cfg.start_date = cfg.end_date = day
    cfg.validate()

    print(BANNER)
    print(f"\nSMOKE TEST -- single session {day}")
    print(f"  fill model     {args.fill}")
    print(f"  slippage       ${cfg.execution.entry_slippage_per_leg:.2f}/leg at entry, "
          f"adverse side at exit")
    print(f"  commissions    ${cfg.execution.per_leg_cost:.2f}/contract/leg "
          f"(${cfg.execution.commission_per_contract:.2f} + "
          f"${cfg.execution.exchange_fee_per_contract:.2f} fees)")
    print(f"  cost ceiling   ${cfg.cost.ceiling_usd:,.2f}")

    if args.synthetic:
        from .fixtures import seed_cache

        cache = ParquetCache(cfg.data.cache_dir)
        print("\n  SYNTHETIC DATA -- the Databento MCP server is not connected and no")
        print("  SDK key is configured. Quotes below are generated, not observed.")
        print("  This proves the pipeline runs; it says nothing about the strategy.")
        seed_cache(cache, cfg, day, open_price=args.open_price,
                   annual_vol=args.vol, drift_points=args.drift)
        args.offline = True

    fetcher, cache = _fetcher(cfg, args)
    engine = Engine(cfg, fetcher, _fill_model(cfg, args.fill))

    try:
        result = engine.run_day(day)
    except (CostCeilingExceeded, CostEstimateUnavailable) as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"\nno data: {exc}", file=sys.stderr)
        return 3

    render_report([result])
    spend = cache.total_spend()
    if spend:
        print(f"\n  data spend this cache: ${spend:,.4f}")
    return 0


def cmd_run(args) -> int:
    cfg = _configure(args)
    cfg.start_date = date.fromisoformat(args.start)
    cfg.end_date = date.fromisoformat(args.end)
    cfg.validate()

    print(BANNER)
    print(f"\n{cfg.start_date} .. {cfg.end_date}   fill={args.fill}   "
          f"ceiling=${cfg.cost.ceiling_usd:,.2f}")

    fetcher, cache = _fetcher(cfg, args)
    engine = Engine(cfg, fetcher, _fill_model(cfg, args.fill))

    try:
        results = engine.run(cfg.start_date, cfg.end_date)
    except (CostCeilingExceeded, CostEstimateUnavailable) as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        return 2

    log = render_report(results)
    if args.out and not log.empty:
        log.to_csv(args.out, index=False)
        print(f"\n  trade log -> {args.out}")
    spend = cache.total_spend()
    if spend:
        print(f"  data spend this cache: ${spend:,.4f}")
    return 0


def cmd_cost(args) -> int:
    """Price a range without pulling. Rule 1, in isolation."""
    cfg = _configure(args)
    cfg.start_date = date.fromisoformat(args.start)
    cfg.end_date = date.fromisoformat(args.end)
    cfg.validate()

    cache = ParquetCache(cfg.data.cache_dir)
    fetcher = DatabentoFetcher(cfg, cache)

    total, priced, day = 0.0, 0, cfg.start_date
    from datetime import timedelta

    print(f"pricing {cfg.start_date} .. {cfg.end_date} (no data will be pulled)\n")
    while day <= cfg.end_date:
        if day.weekday() < 5:
            estimate = fetcher.estimate_cost(
                SCHEMA_MBP1, day, None, cfg.data.quote_start, cfg.data.quote_end, "parent"
            )
            if estimate is None:
                print(f"  {day}  estimate unavailable")
            else:
                total += estimate
                priced += 1
                print(f"  {day}  ${estimate:,.4f}")
        day += timedelta(days=1)

    print(f"\n  {priced} sessions priced, ${total:,.2f} total")
    print(f"  per-pull ceiling is ${cfg.cost.ceiling_usd:,.2f}")
    print("\n  Note: this prices the *unfiltered* chain. The engine restricts each")
    print("  pull to a strike band around the money, which is materially cheaper.")
    return 0


def cmd_cache(args) -> int:
    cfg = _configure(args)
    cache = ParquetCache(cfg.data.cache_dir)
    entries = cache.manifest()
    if not entries:
        print(f"nothing cached under {cfg.data.cache_dir}")
        return 0

    print(f"{len(entries)} cached pulls under {cfg.data.cache_dir}\n")
    for entry in entries[-args.limit:]:
        cost = entry.get("cost_usd")
        tag = "synthetic" if entry.get("synthetic") else (
            f"${float(cost):,.4f}" if cost is not None else "-"
        )
        print(f"  {entry['date']}  {entry['schema']:<10} {entry['rows']:>8,} rows  {tag}")
    print(f"\n  total recorded spend ${cache.total_spend():,.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Shared flags live on the subcommands rather than the top level, so
    # `run --offline` works. As top-level-only arguments they would have had to
    # precede the subcommand, which is the opposite of what anyone types.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config")
    common.add_argument("-v", "--verbose", action="store_true")
    common.add_argument("--ceiling", type=float, help="cost ceiling per pull, USD")
    common.add_argument("--commission", type=float, help="commission per contract per leg")
    common.add_argument("--slippage", type=float, help="entry slippage per leg, index points")
    common.add_argument("--offline", action="store_true", help="cache only; never fetch")
    common.add_argument("--fill", choices=["adverse", "mid"], default="adverse")

    p = argparse.ArgumentParser(prog="backtest", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("smoke", parents=[common],
                       help="one date end-to-end with the trade log")
    s.add_argument("--date", required=True, help="YYYY-MM-DD")
    s.add_argument("--synthetic", action="store_true",
                   help="generate a fixture session instead of fetching")
    s.add_argument("--open-price", type=float, default=5000.0)
    s.add_argument("--vol", type=float, default=0.16)
    s.add_argument("--drift", type=float, default=0.0,
                   help="points the synthetic underlying travels over the session")
    s.set_defaults(func=cmd_smoke)

    r = sub.add_parser("run", parents=[common], help="run a date range")
    r.add_argument("--start", required=True)
    r.add_argument("--end", required=True)
    r.add_argument("--out", help="write the trade log to CSV")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("cost", parents=[common], help="price a range without pulling")
    c.add_argument("--start", required=True)
    c.add_argument("--end", required=True)
    c.set_defaults(func=cmd_cost)

    k = sub.add_parser("cache", parents=[common], help="what is cached and what it cost")
    k.add_argument("--limit", type=int, default=30)
    k.set_defaults(func=cmd_cache)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
