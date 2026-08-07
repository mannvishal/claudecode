"""Tests for annotation and ticket rendering."""

import pytest

from spreadscout.cli import _fmt_money, _print_ticket, build_parser
from spreadscout.models import IronCondor
from spreadscout.risk import evaluate, size_position
from spreadscout.screener import annotate

from .conftest import SPOT, T


@pytest.fixture
def sized(cfg, put_spread):
    cfg.beliefs.vol_multiplier = 0.80
    ev = evaluate(put_spread, SPOT, T, cfg)
    result = size_position(ev, equity=100_000, open_risk=0, cfg=cfg)
    ev.contracts = result.contracts
    ev.total_credit = result.total_credit
    ev.total_risk = result.total_risk
    return ev


class TestFormatting:
    def test_negative_money_uses_a_leading_sign(self):
        assert _fmt_money(-1234.5) == "-$1,234.50"
        assert _fmt_money(1234.5) == "$1,234.50"
        assert _fmt_money(0.0) == "$0.00"


class TestAnnotations:
    def test_zero_dte_gets_a_gamma_warning(self, cfg, put_spread):
        ev = annotate(evaluate(put_spread, SPOT, T, cfg), SPOT, T, None)
        assert any("0DTE" in n and "gamma" in n for n in ev.notes)

    def test_longer_dated_does_not(self, cfg, put_spread):
        ev = annotate(evaluate(put_spread, SPOT, T, cfg), SPOT, 30 / 365, None)
        assert not any("0DTE" in n for n in ev.notes)

    def test_flags_belief_driven_edge(self, cfg, put_spread):
        cfg.beliefs.vol_multiplier = 0.80
        ev = annotate(evaluate(put_spread, SPOT, T, cfg), SPOT, T, None)
        assert any("vol_multiplier assumption" in n for n in ev.notes)

    def test_does_not_flag_when_no_belief_is_applied(self, cfg, put_spread):
        ev = annotate(evaluate(put_spread, SPOT, T, cfg), SPOT, T, None)
        assert not any("vol_multiplier assumption" in n for n in ev.notes)

    def test_reports_stale_vendor_greeks(self, cfg, put_spread):
        ev = annotate(evaluate(put_spread, SPOT, T, cfg), SPOT, T, "2026-08-06 20:00:08")
        assert any("2026-08-06" in n and "recomputed" in n for n in ev.notes)


class TestTicketRendering:
    def test_vertical_renders_every_leg(self, sized, capsys):
        _print_ticket(sized, 1)
        out = capsys.readouterr().out
        assert "SELL" in out and "BUY" in out
        assert "multileg credit" in out
        for _, leg in sized.position.legs:
            assert leg.symbol in out

    def test_condor_renders_four_legs(self, cfg, put_spread, call_spread, capsys):
        cfg.beliefs.vol_multiplier = 0.80
        condor = IronCondor(put_spread=put_spread, call_spread=call_spread)
        ev = evaluate(condor, SPOT, T, cfg)
        ev.contracts = 1
        _print_ticket(ev, 1)
        out = capsys.readouterr().out
        assert out.count("SELL") == 2 and out.count("BUY ") == 2

    def test_unsized_candidate_says_not_recommended(self, cfg, put_spread, capsys):
        ev = evaluate(put_spread, SPOT, T, cfg)
        ev.contracts = 0
        _print_ticket(ev, 1)
        out = capsys.readouterr().out
        assert "not recommended" in out
        assert "SELL" not in out  # no ticket for a trade we are not recommending

    def test_shows_both_expectancy_columns(self, sized, capsys):
        _print_ticket(sized, 1)
        out = capsys.readouterr().out
        assert "implied" in out and "your beliefs" in out


class TestParser:
    def test_requires_a_subcommand(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_scan_defaults(self):
        args = build_parser().parse_args(["scan"])
        assert args.top == 5 and args.equity is None and not args.show_rejected

    def test_global_overrides_apply(self):
        args = build_parser().parse_args(
            ["--symbol", "SPY", "--dte", "7", "--vol-multiplier", "0.9", "--sandbox", "scan"]
        )
        assert args.symbol == "SPY" and args.dte == 7
        assert args.vol_multiplier == 0.9 and args.sandbox

    def test_backtest_accepts_a_date_range(self):
        args = build_parser().parse_args(["backtest", "--start", "2024-01-01"])
        assert args.start == "2024-01-01" and args.underlying == "SPY"
