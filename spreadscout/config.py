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
    """Your forecast about realized vs implied volatility.

    ``source`` decides where it comes from:

    * ``"measured"`` (default) -- derive it from the trailing variance risk
      premium, i.e. what the index has actually been realizing against what the
      chain is charging. Replaces a guess with a measurement. Still a forecast,
      because it assumes the trailing window describes tomorrow.
    * ``"manual"`` -- use ``vol_multiplier`` verbatim. At the 1.0 default this
      makes every candidate negative-EV and the tool will recommend nothing,
      which is the correct answer to "is there free money here".

    ``haircut`` is added to the measured multiplier before use, so a measured
    0.82 with a 0.05 haircut is applied as 0.87. Trailing realized vol
    systematically understates the risk of the day it stops being trailing, and
    this is the knob that admits it.
    """

    source: str = "measured"
    vol_multiplier: float = 1.0
    haircut: float = 0.05
    drift: float | None = None


@dataclass
class GateConfig:
    """Conditions that must hold before an entry is worth alerting on.

    Every gate is a reason *not* to trade. The default posture is no, and
    conditions have to argue past it.
    """

    # The premium actually on offer: implied / realized - 1.
    #
    # Calibration note, measured on 67 real SPX sessions: realized vol ran 13.7%
    # close-to-close against 10.1% open-to-close, because roughly a third of the
    # index's variance arrives overnight. Against a VIX-consistent 15.2% implied,
    # that is a +10.8% premium on the conservative reading and +51% on the
    # horizon-matched one. Which denominator you pick moves the answer by 5x, so
    # both are always reported and the gate names the one it used.
    #
    # conservative_vrp=True compares against close-to-close, which is
    # deliberately mismatched for a 0DTE seller who never carries the overnight
    # gap. It is the stricter test and the default. Setting it False is the
    # economically correct comparison for 0DTE and will fire far more often --
    # which is a reason to raise min_variance_risk_premium at the same time, not
    # a free upgrade.
    min_variance_risk_premium: float = 0.15
    conservative_vrp: bool = True

    # Do not sell volatility that is already scraping its own floor.
    min_vix_percentile: float = 0.20

    # Do not sell into a move already underway.
    max_todays_move_sigma: float = 1.25

    # Entry window, ET. Outside it the quotes are wide (early) or there is too
    # little time to manage a position that turns (late).
    entry_start_hour: int = 10
    entry_start_minute: int = 0
    entry_end_hour: int = 14
    entry_end_minute: int = 0

    blackout_dates: list[str] = field(default_factory=list)


@dataclass
class MonitorConfig:
    """Thresholds for alerting on positions you already hold.

    These matter more than the entry gates. A missed entry costs nothing; a
    0DTE short strike going through the money costs the width.
    """

    short_delta_alert: float = 0.33
    short_delta_critical: float = 0.45
    # Alert when spot comes within this many index points of a short strike.
    strike_proximity_points: float = 15.0
    # Alert when a position's mark-to-market loss reaches this multiple of the
    # credit taken in. 2.0 is the conventional stop for defined-risk sellers.
    loss_multiple_alert: float = 2.0
    # Warn as the daily loss limit is approached, not only once it is breached.
    daily_loss_warn_fraction: float = 0.70
    # Suppress entry suggestions while any open position is in critical trouble.
    # Being run over is not the moment to add risk, and a tool that alerts a
    # breach and then proposes a new condor in the same breath is training you
    # to ignore the first half of its own output.
    block_entry_on_critical: bool = True


@dataclass
class AlertConfig:
    terminal_bell: bool = True
    log_file: str | None = "alerts.jsonl"
    webhook_url: str | None = None  # Slack or Discord incoming webhook
    webhook_min_severity: str = "WARN"
    cooldown_minutes: float = 15.0
    poll_seconds: int = 60
    heartbeat_minutes: float = 60.0


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
    gates: GateConfig = field(default_factory=GateConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)

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
            "gates": GateConfig,
            "monitor": MonitorConfig,
            "alerts": AlertConfig,
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
        if self.beliefs.source not in ("measured", "manual"):
            raise SystemExit("beliefs.source must be 'measured' or 'manual'")
        if self.beliefs.haircut < 0:
            raise SystemExit("beliefs.haircut must be >= 0")
        if self.filters.min_short_delta >= self.filters.max_short_delta:
            raise SystemExit("filters.min_short_delta must be < max_short_delta")
        if self.monitor.short_delta_alert >= self.monitor.short_delta_critical:
            raise SystemExit("monitor.short_delta_alert must be < short_delta_critical")
        if self.alerts.poll_seconds < 5:
            raise SystemExit("alerts.poll_seconds must be >= 5 to stay inside rate limits")
        if self.alerts.webhook_min_severity not in ("INFO", "WARN", "CRITICAL"):
            raise SystemExit("alerts.webhook_min_severity must be INFO, WARN or CRITICAL")
