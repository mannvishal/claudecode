import pytest

from spreadscout.config import Config
from spreadscout.tradier import PRODUCTION, SANDBOX


def test_defaults_are_sane():
    cfg = Config()
    cfg.validate()
    assert cfg.symbol == "SPX"
    assert cfg.dte == 0
    assert cfg.base_url == PRODUCTION


def test_sandbox_switches_base_url():
    cfg = Config(environment="sandbox")
    assert cfg.base_url == SANDBOX


def test_rejects_unknown_environment():
    with pytest.raises(SystemExit):
        Config(environment="live").validate()


def test_rejects_absurd_per_trade_risk():
    cfg = Config()
    cfg.risk.max_risk_pct_per_trade = 0.5
    with pytest.raises(SystemExit):
        cfg.validate()


def test_rejects_portfolio_cap_below_trade_cap():
    cfg = Config()
    cfg.risk.max_risk_pct_per_trade = 0.05
    cfg.risk.max_total_risk_pct = 0.02
    with pytest.raises(SystemExit):
        cfg.validate()


def test_rejects_out_of_range_fill_fraction():
    cfg = Config()
    cfg.costs.fill_fraction = 1.5
    with pytest.raises(SystemExit):
        cfg.validate()


def test_rejects_inverted_delta_band():
    cfg = Config()
    cfg.filters.min_short_delta = 0.30
    cfg.filters.max_short_delta = 0.10
    with pytest.raises(SystemExit):
        cfg.validate()


def test_rejects_nonpositive_vol_multiplier():
    cfg = Config()
    cfg.beliefs.vol_multiplier = 0.0
    with pytest.raises(SystemExit):
        cfg.validate()


def test_loads_yaml_and_overrides_nested_sections(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "symbol: SPY\ndte: 7\nrisk:\n  max_contracts: 3\nbeliefs:\n  vol_multiplier: 0.85\n"
    )
    cfg = Config.load(path)
    assert cfg.symbol == "SPY"
    assert cfg.dte == 7
    assert cfg.risk.max_contracts == 3
    assert cfg.beliefs.vol_multiplier == 0.85
    # Untouched values keep their defaults.
    assert cfg.risk.max_risk_pct_per_trade == 0.02


def test_rejects_unknown_top_level_key(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("symbl: SPY\n")
    with pytest.raises(SystemExit):
        Config.load(path)


def test_rejects_unknown_nested_key(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("risk:\n  max_contract: 3\n")
    with pytest.raises(SystemExit):
        Config.load(path)


def test_missing_token_fails_loudly(monkeypatch):
    monkeypatch.delenv("TRADIER_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TRADIER_SANDBOX_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        _ = Config().token
