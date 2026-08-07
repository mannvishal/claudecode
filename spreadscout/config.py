"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from .tradier import PRODUCTION, SANDBOX


@dataclass
class FilterConfig:
    """Liquidity and structure filters applied before anything is scored.

    Defaults are tuned for SPX 0DTE, where the near-the-money strikes are among
    the most liquid instruments in the world but the wings four standard
    deviations out are quoted by nobody and fill like it.
    """

    min_open_interest: int = 100
    min_volume: int = 10
    max_relative_spread: float = 0.20  # (ask-bid)/mid on each leg
    min_short_delta: float = 0.05
    max_short_delta: float = 0.25
    # Credit as a fraction of width. The familiar "collect a third of the width"
    # rule comes from 30-45 DTE chains and is unreachable at 0DTE: measured on a
    # live SPX 0DTE chain, 5-25 wide spreads in the 0.05-0.25 delta band priced
    # between 0.102 and 0.178 of width. Setting this to 0.15 leaves a handful of
    # candidates and silently biases them toward the highest-gamma strikes.
    min_credit_ratio: float = 0.10
    min_credit: float = 0.30  # index points
    widths: list[float] = field(default_factory=lambda: [5.0, 10.0, 25.0])
    max_candidates: int = 15


@dataclass
class CostConfig:
    """Transaction costs, in dollars per contract per leg unless noted.

    Tradier charges $0.35/contract on options. SPX index options additionally
    carry an exchange/regulatory fee that runs around $0.49/contract. Both are
    per leg, so a four-leg iron condor pays this eight times round trip -- which
    is exactly why the tool insists on modelling it rather than quoting gross
    credit like most screeners do.
    """

    commission_per_contract: float = 0.35
    exchange_fee_per_contract: float = 0.49
    assume_closing_trade: bool = True  # doubles per-leg costs
    fill_fraction: float = 0.40  # 0 = mid fill, 1 = pay the full spread


@dataclass
class RiskConfig:
    max_risk_pct_per_trade: float = 0.02  # fraction of equity at risk in one position
    max_total_risk_pct: float = 0.06
    max_contracts: int = 10
    daily_loss_limit_pct: float = 0.03
    min_equity: float = 2000.0
    require_positive_ev: bool = True


@dataclass
class BeliefConfig:
    """Your forecast, stated explicitly.

    ``vol_multiplier`` below 1.0 says you expect realized vol to come in under
    implied. It is the only source of positive expected value in this strategy,
    and the tool will not choose it for you -- see ``README.md``.
    """

    vol_multiplier: float = 1.0
    drift: float | None = None


@dataclass
class Config:
    symbol: str = "SPX"
    dte: int = 0
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.013
    account_id: str | None = None
    environment: str = "production"  # "production" | "sandbox"
    filters: FilterConfig = field(default_factory=FilterConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    beliefs: BeliefConfig = field(default_factory=BeliefConfig)

    @property
    def base_url(self) -> str:
        return SANDBOX if self.environment == "sandbox" else PRODUCTION

    @property
    def token(self) -> str:
        var = "TRADIER_SANDBOX_TOKEN" if self.environment == "sandbox" else "TRADIER_ACCESS_TOKEN"
        token = os.environ.get(var) or os.environ.get("TRADIER_ACCESS_TOKEN", "")
        if not token:
            raise SystemExit(
                f"{var} is not set. Export it or put it in a .env file "
                f"(see .env.example). Never commit it."
            )
        return token

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg = cls()
        if path is None:
            for candidate in ("spreadscout.yaml", "config.yaml"):
                if Path(candidate).exists():
                    path = candidate
                    break
        if path is None:
            return cfg

        raw = yaml.safe_load(Path(path).read_text()) or {}
        nested = {
            "filters": FilterConfig,
            "costs": CostConfig,
            "risk": RiskConfig,
            "beliefs": BeliefConfig,
        }
        for key, value in raw.items():
            if key in nested:
                section = getattr(cfg, key)
                valid = {f.name for f in fields(section)}
                unknown = set(value or {}) - valid
                if unknown:
                    raise SystemExit(f"unknown keys in '{key}': {sorted(unknown)}")
                for k, v in (value or {}).items():
                    setattr(section, k, v)
            elif key in {f.name for f in fields(cfg)}:
                setattr(cfg, key, value)
            else:
                raise SystemExit(f"unknown config key: {key}")

        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.environment not in ("production", "sandbox"):
            raise SystemExit(f"environment must be 'production' or 'sandbox', got {self.environment!r}")
        if not 0 < self.risk.max_risk_pct_per_trade <= 0.25:
            raise SystemExit("risk.max_risk_pct_per_trade must be in (0, 0.25]")
        if self.risk.max_total_risk_pct < self.risk.max_risk_pct_per_trade:
            raise SystemExit("risk.max_total_risk_pct must be >= max_risk_pct_per_trade")
        if not 0 <= self.costs.fill_fraction <= 1:
            raise SystemExit("costs.fill_fraction must be in [0, 1]")
        if self.beliefs.vol_multiplier <= 0:
            raise SystemExit("beliefs.vol_multiplier must be positive")
        if self.filters.min_short_delta >= self.filters.max_short_delta:
            raise SystemExit("filters.min_short_delta must be < max_short_delta")
