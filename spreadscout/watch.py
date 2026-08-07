"""The polling loop: monitor open positions, and alert when entry conditions hold.

Order of operations on every pass is deliberate and should not be rearranged:

  1. Open positions are checked first. A breach alert must never queue behind a
     chain fetch that takes a few seconds.
  2. The account guard runs next. If the daily loss limit is hit, entry
     screening is skipped entirely for the rest of the session -- the loop keeps
     watching what you hold, but stops suggesting new risk.
  3. Regime is measured, and entry gates are applied.
  4. Only if every gate passes is the chain screened and an entry alerted.

The loop never places an order. It tells you what it sees and what it would do.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from .alerts import CRITICAL, INFO, WARN, Alert, AlertRouter, build_router
from .config import Config
from .monitor import check_positions
from .pricing import ET
from .regime import Regime, check_gates, measure, resolve_vol_multiplier
from .risk import daily_loss_breached, realized_pnl_today, size_position
from .screener import load_chain, pick_expiration, screen
from .tradier import TradierClient, TradierError

log = logging.getLogger(__name__)


@dataclass
class WatchState:
    """Session-scoped state that survives across polls."""

    day: date
    entries_alerted: int = 0
    guard_tripped: bool = False
    last_heartbeat: datetime | None = None
    seen_entry_keys: set[str] = field(default_factory=set)

    def roll_if_new_day(self, today: date) -> None:
        if today != self.day:
            self.day = today
            self.entries_alerted = 0
            self.guard_tripped = False
            self.seen_entry_keys.clear()


def _account_guard(
    client: TradierClient, cfg: Config, account_id: str, router: AlertRouter,
    state: WatchState, today: date,
) -> tuple[float, float, bool]:
    """Returns (equity, realized_pnl, ok_to_enter)."""
    equity = client.equity(account_id)
    realized = realized_pnl_today(client.gainloss(account_id, today, today), today)
    limit = equity * cfg.risk.daily_loss_limit_pct

    if daily_loss_breached(realized, equity, cfg):
        if not state.guard_tripped:
            state.guard_tripped = True
            router.send(Alert(
                kind="guard",
                severity=CRITICAL,
                title="DAILY LOSS LIMIT REACHED -- no further entries today",
                key=f"guard:daily:{today}",
                lines=[
                    f"realized P&L today ${realized:,.0f} against a limit of ${-limit:,.0f} "
                    f"({cfg.risk.daily_loss_limit_pct:.1%} of ${equity:,.0f} equity).",
                    "Entry screening is now off for the rest of the session. Open positions "
                    "are still being monitored.",
                    "This is the guard doing its job. Re-entering to make it back is the "
                    "single most reliable way to turn a bad day into a bad month.",
                ],
            ))
        return equity, realized, False

    warn_at = -limit * cfg.monitor.daily_loss_warn_fraction
    if realized <= warn_at:
        router.send(Alert(
            kind="guard",
            severity=WARN,
            title=f"approaching daily loss limit ({realized / -limit:.0%} used)",
            key=f"guard:daily-warn:{today}",
            lines=[
                f"realized P&L today ${realized:,.0f}; limit is ${-limit:,.0f}.",
                "Entries are still permitted but the budget for today is nearly spent.",
            ],
        ))
    return equity, realized, True


def _screen_and_alert(
    client: TradierClient, cfg: Config, router: AlertRouter, state: WatchState,
    equity: float, open_risk: float, vol_note: str, now: datetime,
    preloaded: tuple, expiration, spot: float, T: float,
) -> None:
    result = screen(client, cfg, now=now, preloaded=preloaded)

    for ev in result["candidates"]:
        sizing = size_position(ev, equity, open_risk, cfg)
        if sizing.rejected:
            continue
        ev.contracts = sizing.contracts
        ev.total_credit = sizing.total_credit
        ev.total_risk = sizing.total_risk

        legs = [
            f"  {'SELL' if side.startswith('sell') else 'BUY '} {ev.contracts} {leg.symbol}"
            f"  ({leg.bid:.2f}/{leg.ask:.2f}, delta {leg.delta:+.3f})"
            for side, leg in ev.position.legs
        ]
        router.send(Alert(
            kind="entry",
            severity=INFO,
            title=f"ENTRY: {ev.describe}",
            key=f"entry:{ev.describe}",
            lines=[
                f"{cfg.symbol} {spot:,.2f}, expiry {expiration} "
                f"({T * 365 * 24:.1f}h to settlement)",
                f"credit {ev.credit:.2f} pts (${ev.credit * ev.position.contract_size:,.0f}), "
                f"max loss ${ev.max_loss * ev.position.contract_size:,.0f}, "
                f"return on risk {ev.return_on_risk:.1%}",
                f"P(max profit) {ev.pop_believed:.1%}  |  "
                f"EV/spread ${ev.ev_after_costs:,.2f} after ${ev.costs:,.2f} costs",
                f"vol assumption: {vol_note}",
                f"size {ev.contracts} ({sizing.reasons[0]}), total risk ${ev.total_risk:,.0f}",
                "order (review before submitting -- this tool places nothing):",
                *legs,
                f"  multileg credit, limit {ev.credit:.2f}, day",
            ],
        ))
        state.entries_alerted += 1
        return  # one entry alert per pass; the rest would only be noise

    log.info("gates passed but nothing survived sizing")


def run_once(
    client: TradierClient, cfg: Config, router: AlertRouter, state: WatchState,
    account_id: str | None, equity_override: float | None, now: datetime | None = None,
) -> None:
    """One full pass. Exceptions are caught by the caller so the loop survives."""
    now = now or datetime.now(ET)
    today = now.date()
    state.roll_if_new_day(today)

    # 1. Positions first, always.
    open_risk = 0.0
    positions_critical = False
    if account_id:
        try:
            alerts, groups = check_positions(client, cfg, account_id, now=now)
            for alert in alerts:
                router.send(alert)
            positions_critical = any(
                a.kind == "risk" and a.severity == CRITICAL for a in alerts
            )
            open_risk = sum(max(g.cost_to_close, 0.0) for g in groups)
        except TradierError as exc:
            log.warning("position check failed: %s", exc)

    # 2. Account guard.
    equity, ok_to_enter = equity_override or 0.0, True
    if account_id and equity_override is None:
        equity, _realized, ok_to_enter = _account_guard(
            client, cfg, account_id, router, state, today
        )
    if not ok_to_enter:
        return

    # 3. Market state.
    clock = client.clock()
    if clock.get("state") != "open":
        log.info("market %s; skipping entry screen", clock.get("state"))
        return

    # 4. Regime and gates. The chain is fetched once here and handed to the
    #    screener below, so a pass costs one chain request rather than two.
    expiration = pick_expiration(client, cfg.symbol, cfg.dte)
    spot, T, contracts = load_chain(client, cfg, expiration, now=now)
    preloaded = (expiration, spot, T, contracts)
    regime = measure(client, cfg, spot, contracts, now=now)
    gates = check_gates(regime, cfg, now=now)

    multiplier, vol_note = resolve_vol_multiplier(regime, cfg)
    cfg.beliefs.vol_multiplier = multiplier

    if cfg.alerts.heartbeat_minutes > 0:
        due = state.last_heartbeat is None or (
            now - state.last_heartbeat >= timedelta(minutes=cfg.alerts.heartbeat_minutes)
        )
        if due:
            state.last_heartbeat = now
            router.send(Alert(
                kind="status",
                severity=INFO,
                title=f"watching {cfg.symbol} -- {gates.summary}",
                key=f"status:heartbeat:{now:%H%M}",
                lines=[*(regime.describe() if regime else []), f"vol assumption: {vol_note}"],
            ))

    if positions_critical and cfg.monitor.block_entry_on_critical:
        router.send(Alert(
            kind="guard",
            severity=WARN,
            title="entries suppressed while an open position is in trouble",
            key=f"guard:critical-position:{today}",
            lines=[
                "A short strike is breached or past its critical delta. New entries are "
                "held back until that resolves.",
                "Deal with what you are already holding first. Adding a position while "
                "one is running against you is how a manageable loss becomes a bad day.",
            ],
        ))
        return

    if not gates.passed:
        log.info("gates blocked: %s", gates.summary)
        return

    if not equity:
        log.warning("no equity available; cannot size. Pass --equity or an account.")
        return

    _screen_and_alert(
        client, cfg, router, state, equity, open_risk, vol_note, now,
        preloaded, expiration, spot, T,
    )


def watch(client: TradierClient, cfg: Config, args) -> int:
    router = build_router(cfg)
    now = datetime.now(ET)
    state = WatchState(day=now.date())

    account_id = cfg.account_id
    if args.equity is None and not account_id:
        ids = client.account_ids()
        account_id = ids[0] if ids else None
        if not account_id:
            raise SystemExit("no account found; pass --equity to run without one")

    print(
        f"watching {cfg.symbol} {cfg.dte}DTE every {cfg.alerts.poll_seconds}s. "
        f"Recommendation only -- no orders will be placed. Ctrl-C to stop."
    )
    if cfg.beliefs.source == "measured":
        print(
            "vol assumption is measured from the trailing variance risk premium; "
            "entries are gated on it being genuinely positive."
        )

    while True:
        try:
            run_once(client, cfg, router, state, account_id, args.equity)
        except TradierError as exc:
            log.warning("poll failed, will retry: %s", exc)
        except KeyboardInterrupt:
            print("\nstopped.")
            return 0
        except Exception:
            # A watcher that dies on an unexpected error is worse than useless:
            # you believe you are being monitored when you are not.
            log.exception("unexpected error in poll; continuing")

        if args.once:
            return 0
        try:
            time.sleep(cfg.alerts.poll_seconds)
        except KeyboardInterrupt:
            print("\nstopped.")
            return 0
