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

import pandas as pd

from .cache import ParquetCache
from .config import SCHEMA_DEFINITION, SCHEMA_QUOTES, BacktestConfig
from .engine import Engine
from .fills import MidFill, MidMinusEdgeFill
from .report import render_report
from .source import (
    ET as ETZ,
    AgentBridgeFetcher,
    BudgetExhausted,
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
    except BudgetExhausted as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        return 4
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
    except BudgetExhausted as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        return 4
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
    from .rangemodel import (
        RangeModel,
        VarianceProfile,
        VolBaseline,
        calibration,
        split_sessions,
    )

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
    # Fitted on every session: it only ever looks strictly backward, so a
    # test day reading yesterday's volatility is not lookahead, it is what
    # a trader has at the open.
    baseline = VolBaseline.fit(by_day)
    model = RangeModel.fit(train, profile, stride=args.stride, baseline=baseline)
    print(f"variance profile from {profile.n_sessions} sessions, "
          f"model from {model.n_observations:,} observations\n")

    rows = calibration(model, test, profile, stride=args.stride, baseline=baseline)
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

    from .rangemodel import RangeModel, VarianceProfile, VolBaseline, state_at

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
    baseline = VolBaseline.fit(by_day)
    model = RangeModel.fit(train, profile, stride=args.stride, baseline=baseline)

    clock = _time.fromisoformat(args.at)
    observed = state_at(by_day[asof], clock, profile, baseline.prior_for(asof))
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


def cmd_validate(args) -> int:
    """Phase 2: check the model's strikes against real option quotes.

    The first command in the harness that spends meaningful OPRA money, so it
    prints its own bill before and after.
    """
    from datetime import time as _time

    from .data import QuoteBook, contracts_from_definitions
    from .rangemodel import RangeModel, VarianceProfile, VolBaseline, state_at
    from .source import session_bounds
    from .validate import (
        SCHEMA_VALIDATION_QUOTES,
        ValidationSummary,
        check_session,
        pick_sessions,
    )

    cfg = _configure(args)
    cfg.start_date = date.fromisoformat(args.train_start)
    cfg.end_date = date.fromisoformat(args.end)
    cfg.validate()

    if args.width is not None:
        cfg.signal.width_points = args.width
    window_start = date.fromisoformat(args.start)
    by_day = _load_sessions(args, cfg)
    if by_day is None:
        return 1

    train = {d: f for d, f in by_day.items() if d < window_start}
    candidates = sorted(d for d in by_day if d >= window_start)
    if len(train) < 60 or not candidates:
        print(f"need more history: {len(train)} training sessions, "
              f"{len(candidates)} candidates")
        return 1

    chosen = pick_sessions(candidates, args.days)
    profile = VarianceProfile.fit(train)
    baseline = VolBaseline.fit(by_day)
    model = RangeModel.fit(train, profile, stride=args.stride, baseline=baseline)
    print(f"\nfitted on {len(train)} sessions strictly before {window_start}")
    print(f"validating {len(chosen)} of {len(candidates)} candidate sessions "
          f"at {args.at}, {args.confidence:.0%} confidence, "
          f"{cfg.signal.width_points:.0f}-point wings\n")

    fetcher, cache = _fetcher(cfg, args)
    spend_before = cache.total_spend()

    clock = _time.fromisoformat(args.at)
    parent = [cfg.data.parent_symbol]
    checks = []
    budget_stop = False

    for day in chosen:
        observed = state_at(by_day[day], clock, profile, baseline.prior_for(day))
        if observed is None:
            print(f"  {day}  no underlying bar at {clock}")
            continue
        state, _ = observed
        z_return = (model.close_quantile(1.0 - args.confidence)
                    if args.side == "put" else model.close_quantile(args.confidence))
        target = z_return * state.sigma_remaining

        try:
            defs, _ = fetcher.fetch(
                SCHEMA_DEFINITION, day, parent,
                cfg.data.quote_start, cfg.data.quote_end, stype_in="parent",
            )
            contracts = contracts_from_definitions(defs, cfg.data.underlying_root, day)
            if not contracts:
                print(f"  {day}  no {cfg.data.underlying_root} contracts expiring today")
                continue

            lo, hi = session_bounds(day, cfg.data.quote_start, cfg.data.quote_end)
            frame, _ = fetcher.fetch_window(
                dataset=cfg.data.dataset, schema=SCHEMA_VALIDATION_QUOTES,
                symbols=parent, lo=lo, hi=hi, stype_in="parent", key_day=day,
            )
        except BudgetExhausted as exc:
            print(f"\n  stopped at {day}: {exc}")
            budget_stop = True
            break
        except (CostCeilingExceeded, CostEstimateUnavailable) as exc:
            print(f"\n  stopped at {day}: {exc}")
            break

        # The pull is the whole SPXW parent -- every expiry listed that day, of
        # which today's is a fortieth. Narrowing before the book is built turns
        # six million rows into a hundred and forty thousand.
        wanted = {c.raw for c in contracts}
        book = QuoteBook(frame[frame["symbol"].isin(wanted)])
        entry_ts = pd.Timestamp(f"{day} {args.at}", tz=ETZ)
        check = check_session(
            day, contracts, book, entry_ts, target, args.side,
            cfg.signal.width_points, cfg.data.risk_free_rate, cfg.data.dividend_yield,
        )
        checks.append(check)

        if check.note:
            print(f"  {day}  skipped: {check.note}")
        else:
            mark = "BREACH" if check.breached else "held"
            flag = "" if abs(check.wing_width - cfg.signal.width_points) < 1e-6 \
                else f"  [wing {check.wing_width:,.0f}pt]"
            print(f"  {day}  spot {check.spot:>9,.2f}  short {check.short_strike:>8,.0f}"
                  f"  credit {check.credit:>6.2f}  settle {check.settlement:>9,.2f}"
                  f"  {mark}{flag}")

    if not checks:
        print("\nno sessions checked")
        return 4 if budget_stop else 1

    summary = ValidationSummary(checks, args.confidence)
    spent = cache.total_spend() - spend_before
    print(f"\n  {len(summary.tradable)} of {len(checks)} sessions produced a tradable spread")
    if summary.resolved:
        print(f"  breaches {summary.breaches}/{len(summary.resolved)} "
              f"= {summary.breach_rate:.1%}, model implied {summary.expected_breach_rate:.1%}")
        print(f"  median credit {summary.median_credit:.2f} index points, "
              f"median wing {summary.median_width:,.0f} points")

        mismatched = summary.wrong_width(cfg.signal.width_points)
        if mismatched:
            # Silence here would report five times the risk under the label of
            # the width that was asked for.
            print(f"  NOTE {len(mismatched)} of {len(summary.tradable)} sessions could not "
                  f"supply a {cfg.signal.width_points:,.0f}-point wing; "
                  f"widths above are what the chain actually listed")

        expectancy = summary.expectancy()
        if expectancy is not None:
            legs = 2 * cfg.execution.per_leg_cost / cfg.execution.contract_multiplier
            net = expectancy - legs
            print(f"\n  realised expectancy {expectancy:+.3f} points per spread "
                  f"({expectancy * cfg.execution.contract_multiplier:+,.0f} per contract)")
            print(f"  after {legs:.3f} points of commission: {net:+.3f} points "
                  f"({net * cfg.execution.contract_multiplier:+,.0f} per contract)")
            print(f"  on median risk {summary.median_width - summary.median_credit:,.2f} points "
                  f"= {(summary.median_width - summary.median_credit) * cfg.execution.contract_multiplier:,.0f} per contract")
    print(f"\n  this run spent ${spent:,.2f}")
    print("\n  Expectancy charges each breach its real settlement cost, not the")
    print("  full width -- these are cash-settled, so a close between the strikes")
    print("  is a partial loss. It holds to settlement with no stop and samples")
    print(f"  {len(summary.resolved)} sessions at one entry time, so it bounds the")
    print("  strategy rather than describing one. At this sample size the credit")
    print("  is the trustworthy number; the breach rate is not.")
    return 0


SWEEP_CONFIDENCES = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def cmd_sweep(args) -> int:
    """Where on the confidence curve, if anywhere, does the trade pay?

    Runs entirely off cached sessions by default, because the pull is a sunk
    cost per day: one session of quotes covers every strike, so sweeping the
    curve costs nothing once the data is bought. This is the command that
    answers whether a strike exists worth selling, rather than whether one
    particular strike was.
    """
    from datetime import time as _time

    from .data import QuoteBook, contracts_from_definitions
    from .rangemodel import (
        RangeModel,
        VarianceProfile,
        VolBaseline,
        measured_breach_rates,
        state_at,
    )
    from .source import session_bounds
    from .validate import SCHEMA_VALIDATION_QUOTES, ValidationSummary, check_session

    cfg = _configure(args)
    if args.width is not None:
        cfg.signal.width_points = args.width
    cfg.start_date = date.fromisoformat(args.train_start)
    cfg.end_date = date.fromisoformat(args.end)
    cfg.validate()
    window_start = date.fromisoformat(args.start)

    by_day = _load_sessions(args, cfg)
    if by_day is None:
        return 1
    train = {d: f for d, f in by_day.items() if d < window_start}
    held_out = {d: f for d, f in by_day.items() if d >= window_start}
    if len(train) < 60 or not held_out:
        print("not enough history to fit and hold out")
        return 1

    profile = VarianceProfile.fit(train)
    baseline = VolBaseline.fit(by_day)
    model = RangeModel.fit(train, profile, stride=args.stride, baseline=baseline)

    # Which sessions do we actually hold option data for? Only those can be
    # priced, and the sweep is deliberately confined to them so it never
    # triggers a pull the user did not ask for.
    fetcher, cache = _fetcher(cfg, args)
    have = sorted(cache.cached_days(cfg.data.dataset, SCHEMA_VALIDATION_QUOTES)
                  & set(held_out))
    if not have:
        print(f"no cached {SCHEMA_VALIDATION_QUOTES} sessions in that window; "
              f"run `validate` first")
        return 1

    print(f"\nsweeping {len(have)} cached sessions, {have[0]} .. {have[-1]}")
    print(f"fitted on {len(train)} sessions before {window_start}, "
          f"{cfg.signal.width_points:.0f}-point wings at {args.at}\n")

    clock = _time.fromisoformat(args.at)
    parent = [cfg.data.parent_symbol]
    context = {}
    for day in have:
        defs, _ = fetcher.fetch(SCHEMA_DEFINITION, day, parent,
                                cfg.data.quote_start, cfg.data.quote_end, "parent")
        contracts = contracts_from_definitions(defs, cfg.data.underlying_root, day)
        lo, hi = session_bounds(day, cfg.data.quote_start, cfg.data.quote_end)
        frame, _ = fetcher.fetch_window(
            dataset=cfg.data.dataset, schema=SCHEMA_VALIDATION_QUOTES,
            symbols=parent, lo=lo, hi=hi, stype_in="parent", key_day=day,
        )
        wanted = {c.raw for c in contracts}
        observed = state_at(by_day[day], clock, profile, baseline.prior_for(day))
        if contracts and observed:
            context[day] = (contracts, QuoteBook(frame[frame["symbol"].isin(wanted)]),
                            observed[0])

    if not context:
        print("no session could be prepared")
        return 1

    # Breakeven belongs against the breach rate the model actually achieved,
    # not the one it claims: it runs conservative, and judging a credit against
    # the nominal figure rejects spreads that are in fact fairly priced.
    measured = measured_breach_rates(model, held_out, profile, SWEEP_CONFIDENCES,
                                     stride=args.stride, baseline=baseline)

    width = cfg.signal.width_points
    commission = 2 * cfg.execution.per_leg_cost / cfg.execution.contract_multiplier
    print(f"  {'conf':>5s} {'OTM':>6s} {'breach':>7s} {'B/E':>6s} "
          f"{'mid':>6s} {'cross':>6s} {'edge@mid':>9s} {'edge@cross':>11s}")
    for confidence in SWEEP_CONFIDENCES:
        z = model.close_quantile(1.0 - confidence)
        checks, otm = [], []
        for day, (contracts, book, state) in context.items():
            check = check_session(
                day, contracts, book, pd.Timestamp(f"{day} {args.at}", tz=ETZ),
                z * state.sigma_remaining, args.side, width,
                cfg.data.risk_free_rate, cfg.data.dividend_yield,
            )
            checks.append(check)
            if not check.note:
                otm.append(abs(check.spot - check.short_strike))
        summary = ValidationSummary(checks, confidence)
        if not summary.tradable:
            print(f"  {confidence:>5.0%} {'--':>6s}  no tradable spread")
            continue

        breach = measured[confidence]
        breakeven = width * breach
        mid = _median_of([c.credit_mid for c in summary.tradable])
        cross = summary.median_credit
        print(f"  {confidence:>5.0%} {sum(otm)/len(otm):>6.0f} {breach:>7.1%} "
              f"{breakeven:>6.2f} {mid:>6.2f} {cross:>6.2f} "
              f"{mid - breakeven - commission:>+9.2f} "
              f"{cross - breakeven - commission:>+11.2f}")

    print(f"\n  Breach is what this model actually did on {len(held_out)} held-out")
    print("  sessions, not what it claims. B/E is the credit that breaks even")
    print("  against it. Edge is net of commission; positive means the market")
    print("  paid more than the risk was worth at that strike.")
    print("\n  mid is not a fillable price. The gap between the two edges is the")
    print("  cost of crossing, and it decides whether better execution could")
    print("  rescue a negative result or whether nothing can.")
    return 0


def _median_of(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


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

    v = sub.add_parser("validate", parents=[common],
                       help="check the model's strikes against real option quotes")
    v.add_argument("--train-start", required=True, help="first session to fit on")
    v.add_argument("--start", required=True, help="first session to validate")
    v.add_argument("--end", required=True)
    v.add_argument("--days", type=int, default=20, help="sessions to sample")
    v.add_argument("--width", type=float,
                   help="wing width in index points (default: config)")
    v.add_argument("--at", default="11:00")
    v.add_argument("--confidence", type=float, default=0.95)
    v.add_argument("--side", choices=["put", "call"], default="put")
    v.add_argument("--stride", type=int, default=5)
    v.set_defaults(func=cmd_validate)

    w = sub.add_parser("sweep", parents=[common],
                       help="edge across the confidence curve, on cached sessions")
    w.add_argument("--train-start", required=True)
    w.add_argument("--start", required=True, help="first held-out session")
    w.add_argument("--end", required=True)
    w.add_argument("--at", default="11:00")
    w.add_argument("--side", choices=["put", "call"], default="put")
    w.add_argument("--width", type=float, help="wing width in index points")
    w.add_argument("--stride", type=int, default=5)
    w.set_defaults(func=cmd_sweep)

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
