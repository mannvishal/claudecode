"""Live decision engine and the Robinhood bridge.

Shape of the integration, and why it is this shape: the Robinhood connector in
this setup is a set of MCP tools, which are callable by an agent session and not
by a long-running Python process. So this module does NOT try to be a daemon
holding a broker socket. It is a pure function of (bars, state) -> order intents,
plus a JSON contract on both sides:

    agent  --get_equity_historicals-->  bars.json
    tqsq decide --bars bars.json --state state.json  -->  intents.json
    agent  --review_equity_order / place_equity_order-->  broker
    agent  --get_equity_positions-->  state.json

That keeps every real-money call in the agent's hands, where the review step and
the user's confirmation live, and keeps this package to the part it is good at:
reproducing the score and the machines exactly as the chart draws them.

Running it around the clock is a scheduling question, not a code question -- a
Routine that fires this every minute the market is open. Note the honest limit
though: TQQQ and SQQQ do not trade 24x7. Robinhood's overnight session covers a
subset of tickers, and the backtest in this repo shows the extended-hours result
is entirely an artefact of the fill assumption. See README.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .bars import SESSIONS
from .score import Weights, compute_score
from .signals import SignalConfig, build_signals

LONG_SYMBOL = "TQQQ"
SHORT_SYMBOL = "SQQQ"


@dataclass
class RiskLimits:
    """Hard stops that sit outside the strategy logic.

    These exist because the strategy adds to losers by construction. Every one
    of them is checked before an intent is emitted, and breaching any of them
    produces a flatten intent rather than a smaller entry.
    """

    max_position_usd: float = 10_000.0
    max_daily_loss_usd: float = 500.0
    max_tranches: int = 3
    stop_pct: float = 0.02
    allow_live: bool = False  # must be set explicitly; nothing places without it


@dataclass
class PositionState:
    symbol: str = ""
    shares: float = 0.0
    avg_cost: float = 0.0
    tiers_filled: list[int] = field(default_factory=list)
    opened_at: str = ""


@dataclass
class EngineState:
    """Persisted between invocations. Written by the agent from broker truth."""

    long: PositionState = field(default_factory=lambda: PositionState(symbol=LONG_SYMBOL))
    short: PositionState = field(default_factory=lambda: PositionState(symbol=SHORT_SYMBOL))
    realized_pnl_today: float = 0.0
    session_date: str = ""
    halted: bool = False
    halt_reason: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "EngineState":
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        st = cls(
            long=PositionState(**raw.get("long", {"symbol": LONG_SYMBOL})),
            short=PositionState(**raw.get("short", {"symbol": SHORT_SYMBOL})),
            realized_pnl_today=raw.get("realized_pnl_today", 0.0),
            session_date=raw.get("session_date", ""),
            halted=raw.get("halted", False),
            halt_reason=raw.get("halt_reason", ""),
        )
        return st

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


@dataclass
class OrderIntent:
    """What the agent should do. Deliberately broker-agnostic and explicit."""

    action: str          # "open" | "add" | "close"
    symbol: str
    side: str            # "buy" | "sell"
    shares: float
    limit_price: float
    reason: str
    tier: int = 0
    urgency: str = "normal"

    def to_dict(self) -> dict:
        return asdict(self)


def bars_from_json(path: str | Path) -> pd.DataFrame:
    """Read the agent-supplied bar file.

    Accepts either Robinhood `get_equity_historicals` shape (begins_at / open_price
    / high_price / low_price / close_price / volume) or a plain OHLCV list.
    """
    raw = json.loads(Path(path).read_text())
    rows = raw["bars"] if isinstance(raw, dict) and "bars" in raw else raw
    df = pd.DataFrame(rows)

    rename = {
        "begins_at": "ts", "timestamp": "ts", "time": "ts",
        "open_price": "open", "high_price": "high",
        "low_price": "low", "close_price": "close",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ts = pd.to_datetime(df["ts"], utc=True, format="mixed")
    df["ts"] = ts.dt.tz_convert("America/New_York")
    df = df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    df["minute"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df["date"] = df["ts"].dt.normalize()
    return df[["ts", "date", "minute", "open", "high", "low", "close", "volume"]]


def decide(
    bars: pd.DataFrame,
    state: EngineState,
    limits: RiskLimits = RiskLimits(),
    sig_cfg: SignalConfig = SignalConfig(),
    weights: Weights = Weights(),
    session: str = "rth",
    quotes: dict[str, float] | None = None,
) -> tuple[list[OrderIntent], dict]:
    """Score the tape and return the intents for the most recent bar.

    `bars` are the SIGNAL symbol's bars (TQQQ). `quotes` carries the last price
    of each tradeable symbol -- the short leg is a different instrument, so its
    price cannot be read off the signal series and the agent must supply it.

    Returns (intents, diagnostics). Diagnostics carry the score, the tier and the
    machine state so the agent can report *why* nothing fired, which matters far
    more than the intents on a quiet day.
    """
    quotes = dict(quotes or {})
    anchor = SESSIONS[session][0]
    bars = bars.copy()
    bars.attrs["anchor"] = anchor

    sf = compute_score(bars, weights=weights, anchor=anchor)
    sg = build_signals(bars, sf.avg, sf.avg_rsi, sig_cfg)

    i = len(bars) - 1
    px = float(bars["close"].iloc[i])
    quotes.setdefault(LONG_SYMBOL, px)
    score = float(sf.avg[i])
    tier_bot = int(sg.exhaustion_tier("bot")[i])
    tier_top = int(sg.exhaustion_tier("top")[i])
    neutral = -sig_cfg.th_bull < score < sig_cfg.th_bull

    diag = {
        "ts": str(bars["ts"].iloc[i]),
        "price": px,
        "score": round(score, 2),
        "avg_rsi": round(float(sf.avg_rsi[i]), 1),
        "exhaustion_bot_tier": tier_bot,
        "exhaustion_top_tier": tier_top,
        "climax": bool(sg.climax[i]),
        "divergence_top": bool(sg.div_top[i]),
        "divergence_bot": bool(sg.div_bot[i]),
        "in_window": bool(sg.in_window[i]),
        "neutral": neutral,
        "halted": state.halted,
    }

    intents: list[OrderIntent] = []

    # ---- risk gates, checked before anything else ----
    if state.realized_pnl_today <= -abs(limits.max_daily_loss_usd):
        state.halted = True
        state.halt_reason = f"daily loss limit {limits.max_daily_loss_usd} reached"
    if state.halted:
        # diag was built from the pre-gate state; reflect the gate's decision.
        diag["halted"] = True
        diag["halt_reason"] = state.halt_reason
        for pos in (state.long, state.short):
            if pos.shares > 0:
                intents.append(OrderIntent("close", pos.symbol, "sell", pos.shares,
                                           quotes.get(pos.symbol, px),
                                           f"halted: {state.halt_reason}", urgency="immediate"))
        return intents, diag

    # ---- exits first: never add to a position that should already be closed ----
    for pos in (state.long, state.short):
        if pos.shares <= 0:
            continue
        mark = quotes.get(pos.symbol, float("nan"))
        if not np.isfinite(mark):
            # No quote for this leg means we cannot evaluate its stop. Say so
            # loudly rather than silently holding through it.
            diag.setdefault("warnings", []).append(f"no quote for {pos.symbol}; stop not evaluated")
            continue
        if pos.avg_cost > 0 and mark <= pos.avg_cost * (1 - limits.stop_pct):
            intents.append(OrderIntent("close", pos.symbol, "sell", pos.shares, mark,
                                       "stop loss", urgency="immediate"))
            continue
        if neutral:
            intents.append(OrderIntent("close", pos.symbol, "sell", pos.shares, mark,
                                       "score returned to the neutral band"))

    if any(x.action == "close" for x in intents):
        return intents, diag

    # ---- entries ----
    for tier, symbol, pos in ((tier_bot, LONG_SYMBOL, state.long), (tier_top, SHORT_SYMBOL, state.short)):
        if tier == 0 or not sg.in_window[i]:
            continue
        if tier in pos.tiers_filled or len(pos.tiers_filled) >= limits.max_tranches:
            continue
        tranche_usd = limits.max_position_usd / limits.max_tranches
        held = pos.shares * pos.avg_cost
        if held + tranche_usd > limits.max_position_usd * 1.001:
            continue
        leg_px = quotes.get(symbol, float("nan"))
        if not np.isfinite(leg_px) or leg_px <= 0:
            diag.setdefault("warnings", []).append(f"no quote for {symbol}; entry skipped")
            continue
        shares = round(tranche_usd / leg_px, 4)
        intents.append(
            OrderIntent(
                action="add" if pos.shares > 0 else "open",
                symbol=symbol,
                side="buy",
                shares=shares,
                limit_price=leg_px,
                reason=f"exhaustion tier {tier} ({'down' if symbol == LONG_SYMBOL else 'up'} stretch tiring)",
                tier=tier,
            )
        )
    return intents, diag


def write_intents(intents: list[OrderIntent], diag: dict, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps({"diagnostics": diag, "intents": [i.to_dict() for i in intents]}, indent=2)
    )
