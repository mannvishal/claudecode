"""Command line interface.

Commands:
    watch      poll for entry conditions and monitor open positions
    regime     measured conditions right now, and whether they permit entry
    scan       screen the chain and print sized order tickets
    account    show equity, open risk, and daily-loss-limit status
    backtest   replay the selection rules over historical data
    explain    show the expectancy arithmetic for the top candidate
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta

from .config import Config
from .models import Evaluation
from .pricing import ET
from .risk import (
    daily_loss_breached,
    kelly_fraction,
    open_risk_from_positions,
    realized_pnl_today,
    size_position,
)
from .screener import screen
from .tradier import TradierClient, TradierError

BANNER = """\
spreadscout -- recommendation only. This tool never places, modifies or cancels
an order. Every ticket below is for you to review and submit yourself."""


def _client(cfg: Config) -> TradierClient:
    return TradierClient(cfg.token, base_url=cfg.base_url)


def _fmt_money(x: float) -> str:
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def _print_ticket(ev: Evaluation, index: int) -> None:
    pos = ev.position
    mult = pos.contract_size
    print(f"\n[{index}] {ev.describe}")
    print(f"     credit {ev.credit:.2f} pts ({_fmt_money(ev.credit * mult)}/spread)   "
          f"max loss {ev.max_loss:.2f} pts ({_fmt_money(ev.max_loss * mult)})   "
          f"return on risk {ev.return_on_risk:.1%}")
    print(f"     P(max profit)  implied {ev.pop_risk_neutral:.1%}   "
          f"under your beliefs {ev.pop_believed:.1%}")
    print(f"     EV/spread      implied {_fmt_money(ev.ev_risk_neutral)}   "
          f"your beliefs {_fmt_money(ev.ev_believed)}   "
          f"after {_fmt_money(ev.costs)} costs {_fmt_money(ev.ev_after_costs)}")

    if ev.contracts:
        print(f"     SIZE: {ev.contracts} contract(s)  "
              f"credit {_fmt_money(ev.total_credit)}  risk {_fmt_money(ev.total_risk)}")
        print("     legs:")
        for side, leg in pos.legs:
            verb = "SELL" if side.startswith("sell") else "BUY "
            print(f"       {verb} {ev.contracts} {leg.symbol}  "
                  f"({leg.bid:.2f}/{leg.ask:.2f}, delta {leg.delta:+.3f}, iv {leg.iv:.1%})")
        print(f"     order: multileg credit, limit {ev.credit:.2f}, day")
    else:
        print("     SIZE: 0 -- not recommended")

    for note in ev.notes:
        print(f"     note: {note}")


def cmd_scan(cfg: Config, args: argparse.Namespace) -> int:
    print(BANNER)
    client = _client(cfg)

    clock = {}
    try:
        clock = client.clock()
    except TradierError:
        pass
    state = clock.get("state", "unknown")
    if state != "open":
        print(f"\nWARNING: market state is '{state}' ({clock.get('description', '')}). "
              f"Quotes may be stale and the modelled fills will not be achievable.")

    equity, open_risk = args.equity, 0.0
    account_id = cfg.account_id
    if equity is None:
        if not account_id:
            ids = client.account_ids()
            if not ids:
                raise SystemExit("no account found; pass --equity to size manually")
            account_id = ids[0]
        equity = client.equity(account_id)
        open_risk = open_risk_from_positions(client.positions(account_id))

        today = datetime.now(ET).date()
        realized = realized_pnl_today(client.gainloss(account_id, today, today), today)
        if daily_loss_breached(realized, equity, cfg):
            print(f"\nSTOP: realized P&L today is {_fmt_money(realized)}, which is past your "
                  f"{cfg.risk.daily_loss_limit_pct:.1%} daily loss limit "
                  f"({_fmt_money(-equity * cfg.risk.daily_loss_limit_pct)}). "
                  f"No new positions recommended.")
            return 1

    result = screen(client, cfg)
    c = result["counts"]
    hours = result["years_to_expiry"] * 365 * 24

    print(f"\n{result['symbol']} {result['spot']:,.2f}   expiry {result['expiration']} "
          f"({hours:.1f}h)   equity {_fmt_money(equity)}   open risk {_fmt_money(open_risk)}")
    print(f"chain: {c['listed']} listed, {c['quotable']} quotable, {c['liquid']} pass liquidity "
          f"-> {c['put_spreads']} put / {c['call_spreads']} call / {c['condors']} condor candidates")
    print(f"vol assumption: {result['vol_note']}")

    from .regime import check_gates

    from .events import build_calendar

    gates = check_gates(result["regime"], cfg, now=datetime.now(ET),
                        calendar=build_calendar(cfg))
    if not gates.passed:
        print("\nENTRY CONDITIONS NOT MET -- tickets below are shown for reference only:")
        for block in gates.blocks:
            print(f"  - {block}")

    if not result["candidates"]:
        print("\nNothing passed the filters. That is a valid answer -- widen the delta band or "
              "loosen liquidity filters in config only if you understand what you are letting in.")
        return 0

    shown = 0
    for ev in result["candidates"]:
        sizing = size_position(ev, equity, open_risk, cfg)
        ev.contracts = sizing.contracts
        ev.total_credit = sizing.total_credit
        ev.total_risk = sizing.total_risk
        ev.notes.extend(sizing.reasons)
        if sizing.rejected and not args.show_rejected:
            continue
        shown += 1
        _print_ticket(ev, shown)
        if shown >= args.top:
            break

    if shown == 0:
        print("\nEvery candidate sized to zero. Re-run with --show-rejected to see why.")

    print("\nReminder: EV under 'implied' is what the market's own pricing says. It is "
          "negative after costs for every credit spread, always. Anything positive in the "
          "'your beliefs' column is your forecast talking, not the data.")
    return 0


def cmd_account(cfg: Config, args: argparse.Namespace) -> int:
    client = _client(cfg)
    account_id = cfg.account_id or (client.account_ids() or [None])[0]
    if not account_id:
        raise SystemExit("no account found")

    equity = client.equity(account_id)
    positions = client.positions(account_id)
    open_risk = open_risk_from_positions(positions)
    today = datetime.now(ET).date()
    realized = realized_pnl_today(client.gainloss(account_id, today, today), today)
    limit = equity * cfg.risk.daily_loss_limit_pct

    print(f"account       {account_id} ({'sandbox' if client.is_sandbox else 'production'})")
    print(f"equity        {_fmt_money(equity)}")
    print(f"open legs     {len(positions)}")
    print(f"open risk     {_fmt_money(open_risk)} (conservative estimate)")
    print(f"risk budget   {_fmt_money(equity * cfg.risk.max_total_risk_pct)} "
          f"({cfg.risk.max_total_risk_pct:.1%} of equity)")
    print(f"today's P&L   {_fmt_money(realized)}")
    print(f"daily stop    {_fmt_money(-limit)} "
          f"-- {'BREACHED' if daily_loss_breached(realized, equity, cfg) else 'ok'}")
    return 0


def cmd_explain(cfg: Config, args: argparse.Namespace) -> int:
    client = _client(cfg)
    result = screen(client, cfg)
    if not result["candidates"]:
        print("no candidates to explain")
        return 0

    ev = result["candidates"][0]
    mult = ev.position.contract_size
    print(f"Worked example: {ev.describe} on {result['symbol']} @ {result['spot']:,.2f}\n")
    print(f"  You receive                          {_fmt_money(ev.credit * mult)}")
    print(f"  Expected cost to close (implied)     "
          f"{_fmt_money(ev.credit * mult - ev.ev_risk_neutral)}")
    print(f"  = EV before costs, implied           {_fmt_money(ev.ev_risk_neutral)}")
    print("    ^ this is ~zero by construction: we solved each leg's implied vol from")
    print("      its own mid, so the model reprices the market exactly. Selling at the")
    print("      mid and valuing at the mid is a wash. The only thing left is friction.\n")
    print(f"  Commissions + exchange fees          {_fmt_money(-ev.costs)}")
    print(f"  = EV at market-implied odds          "
          f"{_fmt_money(ev.ev_risk_neutral - ev.costs)}   <-- the honest baseline\n")
    print(f"  Your vol_multiplier                  {cfg.beliefs.vol_multiplier:.2f}x implied")
    print(f"  = EV under your beliefs, net         {_fmt_money(ev.ev_after_costs)}\n")

    win = ev.pop_believed
    loss_size = ev.max_loss * mult
    win_size = ev.credit * mult
    print(f"  Win rate you are assuming            {win:.1%}")
    print(f"  Payoff when right                    {_fmt_money(win_size)}")
    print(f"  Payoff when wrong                    {_fmt_money(-loss_size)}")
    print(f"  Losses per win needed to break even  {win_size / loss_size:.2f} "
          f"(you lose {loss_size / win_size:.1f}x what you win)")
    k = kelly_fraction(win, ev.credit, ev.max_loss)
    print(f"  Full-Kelly risk fraction             {k:.1%} of equity "
          f"(your cap: {cfg.risk.max_risk_pct_per_trade:.1%})")
    print("\n  If your assumed win rate is off by even a couple of points, the sign of the")
    print("  EV flips. That sensitivity -- not the win rate -- is the real risk here.")
    return 0


def cmd_regime(cfg: Config, args: argparse.Namespace) -> int:
    """Show what conditions look like right now, and whether they permit entry."""
    from .events import build_calendar
    from .regime import check_gates, measure, resolve_vol_multiplier
    from .screener import load_chain, pick_expiration

    client = _client(cfg)
    now = datetime.now(ET)
    expiration = pick_expiration(client, cfg.symbol, cfg.dte, now=now)
    spot, _T, contracts = load_chain(client, cfg, expiration, now=now)
    regime = measure(client, cfg, spot, contracts, now=now)

    if regime is None:
        print("could not measure the regime: no solvable ATM implied vol on this chain.")
        return 1

    print(f"{cfg.symbol} regime at {now:%Y-%m-%d %H:%M} ET\n")
    for line in regime.describe():
        print(f"  {line}")

    multiplier, note = resolve_vol_multiplier(regime, cfg)
    print(f"\n  vol assumption       {note}")

    gates = check_gates(regime, cfg, now=now, calendar=build_calendar(cfg))
    print(f"\n{'ENTRY PERMITTED' if gates.passed else 'ENTRY BLOCKED'}")
    for note in gates.notes:
        print(f"  + {note}")
    for block in gates.blocks:
        print(f"  - {block}")

    if not gates.passed:
        print(
            "\nNo entry is the default answer, not a failure. Premium selling only pays "
            "when implied vol is genuinely rich against what the index is realizing; "
            "the rest of the time you are taking the tail risk for free."
        )
    return 0


def cmd_calendar(cfg: Config, args: argparse.Namespace) -> int:
    """List upcoming scheduled events and flag any stale source."""
    from .events import IMPACT_RANK, build_calendar

    calendar = build_calendar(cfg)
    now = datetime.now(ET)
    today = now.date()
    status = calendar.status(today, today + timedelta(days=args.days), now=now)

    print(f"sources: {', '.join(p.name for p in calendar.providers) or 'none'}")
    if status.stale_sources:
        print(f"STALE:   {', '.join(sorted(set(status.stale_sources)))}")
        print("         A stale source reports no events, which looks exactly like a clear")
        print("         day. Refresh it, or entries will be blocked while "
              "require_event_calendar is on.")
    if not status.available:
        print("no calendar source produced anything.")
        return 1

    threshold = IMPACT_RANK.get(cfg.gates.block_impact_at_or_above, 2)
    print(f"\nnext {args.days} days ({len(status.events)} events):")
    for event in status.events:
        blocking = IMPACT_RANK.get(event.impact, 0) >= threshold
        marker = "BLOCK" if blocking else "     "
        today_marker = " <- TODAY" if event.day == today else ""
        print(f"  {marker}  {event.describe()}{today_marker}")

    blocks, notes = _event_gate_now(calendar, cfg, now)
    print()
    for note in notes:
        print(f"  + {note}")
    for block in blocks:
        print(f"  - {block}")
    if not blocks:
        print("  no event blocks in force right now.")
    return 0


def _event_gate_now(calendar, cfg, now):
    from .events import check_event_gate

    return check_event_gate(calendar, cfg, now=now)


def cmd_watch(cfg: Config, args: argparse.Namespace) -> int:
    from .watch import watch

    print(BANNER)
    return watch(_client(cfg), cfg, args)


def cmd_backtest(cfg: Config, args: argparse.Namespace) -> int:
    from .backtest import run_backtest

    client = _client(cfg)
    end = date.fromisoformat(args.end) if args.end else datetime.now(ET).date()
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=365)
    return run_backtest(client, cfg, start, end, args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spreadscout", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", help="path to YAML config")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--symbol", help="override underlying")
    p.add_argument("--dte", type=int, help="override days to expiry")
    p.add_argument("--vol-multiplier", type=float,
                   help="your forecast of realized vol as a multiple of implied")
    p.add_argument("--sandbox", action="store_true", help="use the Tradier sandbox")

    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="screen the chain and print sized tickets")
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--equity", type=float, help="size against this equity instead of the account")
    s.add_argument("--show-rejected", action="store_true")
    s.set_defaults(func=cmd_scan)

    a = sub.add_parser("account", help="equity, open risk, daily stop status")
    a.set_defaults(func=cmd_account)

    e = sub.add_parser("explain", help="show the expectancy arithmetic")
    e.set_defaults(func=cmd_explain)

    g = sub.add_parser("regime", help="measured conditions and whether they permit entry")
    g.set_defaults(func=cmd_regime)

    cal = sub.add_parser("calendar", help="upcoming economic events and source freshness")
    cal.add_argument("--days", type=int, default=30)
    cal.set_defaults(func=cmd_calendar)

    w = sub.add_parser("watch", help="poll for entry conditions and monitor open positions")
    w.add_argument("--equity", type=float, help="size against this instead of the account")
    w.add_argument("--once", action="store_true", help="run a single pass and exit")
    w.set_defaults(func=cmd_watch)

    b = sub.add_parser("backtest", help="replay the selection rules over history")
    b.add_argument("--start")
    b.add_argument("--end")
    b.add_argument("--underlying", default="SPY",
                   help="symbol to source historical bars from (default SPY)")
    b.set_defaults(func=cmd_backtest)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = Config.load(args.config)
    if args.symbol:
        cfg.symbol = args.symbol
    if args.dte is not None:
        cfg.dte = args.dte
    if args.vol_multiplier is not None:
        # Passing an explicit multiplier is a statement that you want *your*
        # number used, so it also switches off the measured override that would
        # otherwise silently replace it on the next poll.
        cfg.beliefs.vol_multiplier = args.vol_multiplier
        cfg.beliefs.source = "manual"
    if args.sandbox:
        cfg.environment = "sandbox"
    cfg.validate()

    try:
        return args.func(cfg, args)
    except TradierError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
