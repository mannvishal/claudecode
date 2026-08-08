"""Backtest CLI.

    smoke      run one date end-to-end and print the trade log (rule 8)
    run        run a date range
    cost       price a range without pulling anything
    calibrate  fit the range model and report out-of-sample coverage
    advise     strike bands for one session, from underlying bars only
    cache      show what is cached and what it cost
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from .cache import ParquetCache
from .config import SCHEMA_DEFINITION, SCHEMA_QUOTES, BacktestConfig
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

    total, priced, unpriced, day = 0.0, 0, 0, cfg.start_date
    from datetime import timedelta

    # Price the two requests the engine actually issues, per session: the
    # definition file for one parent, and the quote pull. The quote figure is
    # for the whole SPXW chain, which is an *upper bound* -- the engine narrows
    # each real pull to a strike band it cannot know the size of until the
    # definitions are in hand. Pricing ALL_SYMBOLS instead, as this command
    # once did, quotes a request nothing in the harness ever makes, and on
    # cmbp-1 the estimate endpoint times out trying to compute it.
    parent = [cfg.data.parent_symbol]
    print(f"pricing {cfg.start_date} .. {cfg.end_date} (no data will be pulled)")
    print(f"parent {cfg.data.parent_symbol}, quotes {SCHEMA_QUOTES}\n")
    print(f"  {'session':12s}  {'definitions':>12s}  {'quotes (max)':>14s}")
    while day <= cfg.end_date:
        if day.weekday() < 5:
            defs = fetcher.estimate_cost(
                SCHEMA_DEFINITION, day, parent,
                cfg.data.quote_start, cfg.data.quote_end, "parent",
            )
            quotes = fetcher.estimate_cost(
                SCHEMA_QUOTES, day, parent,
                cfg.data.quote_start, cfg.data.quote_end, "parent",
            )
            if defs is None or quotes is None:
                unpriced += 1
                print(f"  {str(day):12s}  {'unavailable':>12s}  {'unavailable':>14s}")
            else:
                total += defs + quotes
                priced += 1
                print(f"  {str(day):12s}  {'$' + format(defs, ',.4f'):>12s}"
                      f"  {'$' + format(quotes, ',.4f'):>14s}")
        day += timedelta(days=1)

    print(f"\n  {priced} sessions priced, ${total:,.2f} total upper bound")
    if unpriced:
        # Rule 1 again: an unpriced session is not a free one, and a total that
        # silently omits it reads as cheaper than the range actually is.
        print(f"  {unpriced} session(s) could NOT be priced and are excluded from that total")
    print(f"  per-pull ceiling is ${cfg.cost.ceiling_usd:,.2f}")
    if priced:
        print(f"\n  Quote figures cover the full {cfg.data.underlying_root} chain. The engine")
        print(f"  restricts each pull to a ±{cfg.signal.strike_band_pct:.0%} strike band, which is")
        print("  materially cheaper; these are ceilings, not forecasts.")
    return 0


def _load_sessions(args, cfg):
    """Underlying sessions for the requested range, pulled or read from cache."""
    from .underlying import UnderlyingConfig, load_minutes, sessions

    ucfg = UnderlyingConfig()
    fetcher, _cache = _fetcher(cfg, args)
    bars = load_minutes(fetcher, ucfg, cfg.start_date, cfg.end_date)
    if bars.empty:
        print("no underlying bars in that range")
        return None
    return sessions(bars)


def cmd_calibrate(args) -> int:
    """Fit the range model and report out-of-sample coverage.

    The only question this answers is whether the model's confidence means
    anything. It is deliberately the command you must run before `advise`
    prints a strike.
    """
    from .rangemodel import RangeModel, VarianceProfile, calibration, split_sessions

    cfg = _configure(args)
    cfg.start_date = date.fromisoformat(args.start)
    cfg.end_date = date.fromisoformat(args.end)
    cfg.validate()

    by_day = _load_sessions(args, cfg)
    if by_day is None:
        return 1

    train, test = split_sessions(by_day, args.train_fraction)
    print(f"\n{len(by_day)} sessions: {len(train)} train, {len(test)} test "
          f"(chronological split at {sorted(train)[-1] if train else '-'})")
    if not train or not test:
        print("not enough sessions to split; widen the range")
        return 1

    profile = VarianceProfile.fit(train)
    model = RangeModel.fit(train, profile, stride=args.stride)
    print(f"variance profile from {profile.n_sessions} sessions, "
          f"model from {model.n_observations:,} observations\n")

    rows = calibration(model, test, profile, stride=args.stride)
    print("  out-of-sample coverage of the close\n")
    print(f"  {'stated':>8s}  {'actual':>8s}  {'error':>8s}")
    worst = 0.0
    for row in sorted(rows, key=lambda r: r.alpha):
        worst = max(worst, abs(row.error))
        print(f"  {row.alpha:>8.1%}  {row.empirical:>8.1%}  {row.error:>+8.2%}")
    print(f"\n  {rows[0].n:,} held-out observations, worst absolute error {worst:.2%}")
    print("\n  A well-calibrated model puts 'actual' on top of 'stated'. Errors in")
    print("  the lower tail matter most: that is where a put spread's short strike")
    print("  sits, and an understated tail sells strikes that breach too often.")
    return 0


def cmd_advise(args) -> int:
    """What the model says about one session, at one moment.

    Reads history up to the given time only. It cannot see the rest of the day,
    which is the entire point.
    """
    from datetime import time as _time

    from .rangemodel import RangeModel, VarianceProfile, state_at

    cfg = _configure(args)
    asof = date.fromisoformat(args.date)
    cfg.start_date = date.fromisoformat(args.train_start)
    cfg.end_date = asof
    cfg.validate()

    by_day = _load_sessions(args, cfg)
    if by_day is None:
        return 1
    if asof not in by_day:
        print(f"no session for {asof} (holiday, or bars not available)")
        return 1

    # Strictly prior sessions: fitting on the day being advised would be
    # lookahead of the most flattering kind.
    train = {d: f for d, f in by_day.items() if d < asof}
    if len(train) < 60:
        print(f"only {len(train)} prior sessions; widen --train-start")
        return 1

    profile = VarianceProfile.fit(train)
    model = RangeModel.fit(train, profile, stride=args.stride)

    clock = _time.fromisoformat(args.at)
    observed = state_at(by_day[asof], clock, profile)
    if observed is None:
        print(f"no bar at or before {clock} on {asof}")
        return 1
    state, outcome = observed

    print(f"\n{asof} at {state.at.strftime('%H:%M %Z')}   "
          f"fitted on {len(train)} prior sessions")
    print(f"  price {state.price:,.2f}   from open {state.return_from_open:+.2%}   "
          f"{state.minutes_left} min to close")
    print(f"  session range so far {state.low_so_far:,.2f} .. {state.high_so_far:,.2f}")
    print(f"  remaining-day sigma {state.sigma_remaining:.3%}\n")

    print("  short strikes by confidence the close respects them\n")
    print(f"  {'confidence':>10s}  {'put below':>12s}  {'call above':>12s}")
    for confidence in (0.80, 0.90, 0.95, 0.975, 0.99):
        put = model.short_strike(state, "put", confidence)
        call = model.short_strike(state, "call", confidence)
        print(f"  {confidence:>10.1%}  {put:>12,.2f}  {call:>12,.2f}")

    low_in = model.probability_low_is_in(state)
    high_in = model.probability_high_is_in(state)
    print(f"\n  P(today's low is already in)   {low_in:.1%}")
    print(f"  P(today's high is already in)  {high_in:.1%}")

    if args.reveal:
        actual = state.price * (1.0 + outcome.close_return)
        print(f"\n  --reveal: the close was {actual:,.2f} "
              f"({outcome.close_return:+.2%}), "
              f"excursions {outcome.min_return:+.2%} / {outcome.max_return:+.2%}")
    else:
        print("\n  (--reveal shows what actually happened; kept off by default so")
        print("   reading the advice is not contaminated by knowing the answer)")

    print("\n  These are model levels, not listed strikes, and they assume only")
    print("  that today resembles the fitted sample. Nothing here has been")
    print("  checked against an option price -- see `cost` before pulling OPRA.")
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

    cal = sub.add_parser("calibrate", parents=[common],
                         help="fit the range model and report out-of-sample coverage")
    cal.add_argument("--start", required=True)
    cal.add_argument("--end", required=True)
    cal.add_argument("--train-fraction", type=float, default=0.7)
    cal.add_argument("--stride", type=int, default=5,
                     help="minutes between decision points sampled per session")
    cal.set_defaults(func=cmd_calibrate)

    a = sub.add_parser("advise", parents=[common],
                       help="strike bands for one session at one time of day")
    a.add_argument("--date", required=True, help="session to advise, YYYY-MM-DD")
    a.add_argument("--at", default="11:00", help="time of day, HH:MM")
    a.add_argument("--train-start", required=True,
                   help="first session to fit on; only days before --date are used")
    a.add_argument("--stride", type=int, default=5)
    a.add_argument("--reveal", action="store_true",
                   help="also print what actually happened")
    a.set_defaults(func=cmd_advise)

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
