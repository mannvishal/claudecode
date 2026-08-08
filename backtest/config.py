"""Backtest configuration.

Every number that changes a result lives here rather than being buried in a
module, so a run can be described by its config alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import date, time
from pathlib import Path

import yaml

# Databento identifiers. OPRA is the consolidated US options feed; SPXW is the
# weekly root that carries every 0DTE contract.
OPRA_DATASET = "OPRA.PILLAR"
SCHEMA_DEFINITION = "definition"
# Top-of-book quotes. OPRA does not carry `mbp-1`: because it is a consolidated
# feed rather than a single venue, the equivalent schema is `cmbp-1`, whose
# book columns (bid_px_00/ask_px_00/bid_sz_00/ask_sz_00) are the ones the quote
# parser reads. Named for its role so a schema change does not rename the world.
SCHEMA_QUOTES = "cmbp-1"


@dataclass
class CostConfig:
    """Data spend controls.

    ``ceiling_usd`` is the halt-and-ask threshold from rule 1. It is not a
    budget tracker -- it applies per pull, because that is the granularity at
    which the decision is actually made. ``metadata.get_cost`` is free to call,
    so the estimate is always printed even when it comes in far under.
    """

    ceiling_usd: float = 25.0
    # Refuse to proceed if the cost endpoint itself fails. An unknown cost is
    # not a zero cost, and guessing here spends real money.
    require_estimate: bool = True


@dataclass
class ExecutionConfig:
    """Fill and cost assumptions. Swappable without touching signals."""

    # Entry: you give up this much per leg against the midpoint. Applied to the
    # net credit, so a four-leg condor concedes 4 x this amount.
    entry_slippage_per_leg: float = 0.05
    # Exit marks each leg at the side you must cross: pay the ask to close a
    # short, hit the bid to close a long.
    exit_at_adverse_side: bool = True
    commission_per_contract: float = 0.35
    exchange_fee_per_contract: float = 0.49
    contract_multiplier: int = 100

    @property
    def per_leg_cost(self) -> float:
        return self.commission_per_contract + self.exchange_fee_per_contract


@dataclass
class SignalConfig:
    """Strike selection and exit rules. Knows nothing about fills."""

    entry_time: time = time(10, 0)
    # Hard flat time. SPXW is cash-settled at the close, so holding to
    # settlement is legitimate -- but it is also the choice that produces the
    # fattest tail, so it is opt-in rather than the default.
    exit_time: time = time(15, 45)
    hold_to_settlement: bool = False

    target_short_delta: float = 0.15
    width_points: float = 25.0
    structure: str = "iron_condor"  # iron_condor | put_credit | call_credit

    # Exits, checked in this order on every quote update.
    stop_loss_multiple: float = 2.0  # exit when loss reaches N x credit
    profit_target_fraction: float = 0.50  # exit at N% of max profit

    # Strike band to request quotes for, as a fraction of the anchor price.
    # Wider is safer for strike selection and costs more data.
    strike_band_pct: float = 0.06
    min_credit: float = 0.20
    contracts: int = 1


@dataclass
class DataConfig:
    cache_dir: Path = Path("./data")
    dataset: str = OPRA_DATASET
    underlying_root: str = "SPXW"
    # Must agree with ``underlying_root``: OPRA treats SPX and SPXW as separate
    # parents, so `SPX.OPT` returns the AM-settled monthlies and none of the
    # weeklies that carry 0DTE. Pulling the wrong one yields an empty chain
    # after the root filter rather than an error.
    parent_symbol: str = "SPXW.OPT"
    # Quote window pulled per session. Starting before the entry time gives the
    # strike selector something to look at; ending at the close covers exits.
    quote_start: time = time(9, 45)
    quote_end: time = time(16, 0)
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.013


@dataclass
class BacktestConfig:
    start_date: date | None = None
    end_date: date | None = None
    cost: CostConfig = field(default_factory=CostConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def validate(self) -> None:
        # Rule 2: an implicit "today" is how a backtest silently becomes a
        # live-data request against a date with no settlement yet.
        if self.start_date is None or self.end_date is None:
            raise SystemExit(
                "start_date and end_date are both required. This harness never "
                "defaults to today -- see rule 2."
            )
        if self.end_date < self.start_date:
            raise SystemExit(f"end_date {self.end_date} precedes start_date {self.start_date}")
        if self.cost.ceiling_usd < 0:
            raise SystemExit("cost.ceiling_usd must be >= 0")
        if self.execution.entry_slippage_per_leg < 0:
            raise SystemExit("execution.entry_slippage_per_leg must be >= 0")
        if self.signal.structure not in ("iron_condor", "put_credit", "call_credit"):
            raise SystemExit(f"unknown structure {self.signal.structure!r}")
        if not 0 < self.signal.target_short_delta < 0.5:
            raise SystemExit("signal.target_short_delta must be in (0, 0.5)")
        if self.signal.width_points <= 0:
            raise SystemExit("signal.width_points must be positive")
        if self.signal.entry_time >= self.signal.exit_time:
            raise SystemExit("signal.entry_time must precede signal.exit_time")
        if not 0 < self.signal.strike_band_pct < 0.5:
            raise SystemExit("signal.strike_band_pct must be in (0, 0.5)")
        # A parent that disagrees with the root fails silently: the definition
        # pull succeeds, the root filter discards every row, and the session is
        # skipped as "no contracts" having already been paid for.
        if self.data.parent_symbol.split(".")[0] != self.data.underlying_root:
            raise SystemExit(
                f"data.parent_symbol {self.data.parent_symbol!r} does not match "
                f"data.underlying_root {self.data.underlying_root!r}; the chain "
                f"would come back empty after the root filter."
            )

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides) -> "BacktestConfig":
        cfg = cls()
        if path and Path(path).exists():
            raw = yaml.safe_load(Path(path).read_text()) or {}
            nested = {
                "cost": CostConfig,
                "execution": ExecutionConfig,
                "signal": SignalConfig,
                "data": DataConfig,
            }
            for key, value in raw.items():
                if key in nested:
                    section = getattr(cfg, key)
                    valid = {f.name for f in fields(section)}
                    unknown = set(value or {}) - valid
                    if unknown:
                        raise SystemExit(f"unknown keys in '{key}': {sorted(unknown)}")
                    for k, v in (value or {}).items():
                        current = getattr(section, k)
                        if isinstance(current, time) and isinstance(v, str):
                            v = time.fromisoformat(v)
                        elif isinstance(current, Path):
                            v = Path(v)
                        setattr(section, k, v)
                elif key in {f.name for f in fields(cfg)}:
                    setattr(cfg, key, value)
                else:
                    raise SystemExit(f"unknown config key: {key}")

        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg
