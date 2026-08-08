"""Tests for the option-quote validation layer.

The bugs this layer invites all point the same direction -- they make the
strategy look better than it was. Rounding a strike toward spot, marking a
bidless contract at its midpoint, or sampling the sessions that happened to
work are each individually small and collectively fatal, and none of them
crashes. So these tests are mostly about bias, not correctness.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from backtest.data import CALL, PUT, Contract, QuoteBook, osi_symbol
from backtest.source import ET
from backtest.validate import (
    ValidationSummary,
    check_session,
    nearest_listed,
    pick_sessions,
    wing,
)

DAY = date(2026, 8, 6)
ENTRY = pd.Timestamp(f"{DAY} 11:00", tz=ET)


def contract(strike: float, option_type: str) -> Contract:
    return Contract(
        symbol=osi_symbol("SPXW", DAY, option_type, strike),
        root="SPXW", expiration=DAY, option_type=option_type, strike=strike,
    )


def chain(strikes, types=(PUT, CALL)) -> list[Contract]:
    return [contract(s, t) for s in strikes for t in types]


def book_from(rows) -> QuoteBook:
    """rows: (symbol, bid, ask) at a single timestamp."""
    return QuoteBook(pd.DataFrame([
        {"ts_recv": ENTRY, "symbol": sym, "bid_px_00": bid, "ask_px_00": ask,
         "bid_sz_00": 10, "ask_sz_00": 10}
        for sym, bid, ask in rows
    ]))


class TestStrikeSelection:
    def test_put_strike_rounds_away_from_spot(self):
        """Rounding toward spot would sell a nearer strike than the model asked
        for, collecting a credit it never earned."""
        contracts = chain([6300, 6325, 6350, 6375, 6400])
        picked = nearest_listed(contracts, PUT, 6340.0, "put")
        assert picked.strike == 6325

    def test_call_strike_rounds_away_from_spot(self):
        contracts = chain([6300, 6325, 6350, 6375, 6400])
        picked = nearest_listed(contracts, CALL, 6340.0, "call")
        assert picked.strike == 6350

    def test_an_exact_level_takes_that_strike(self):
        contracts = chain([6300, 6325, 6350])
        assert nearest_listed(contracts, PUT, 6325.0, "put").strike == 6325
        assert nearest_listed(contracts, CALL, 6325.0, "call").strike == 6325

    def test_a_level_beyond_the_chain_is_refused(self):
        """Better to skip the session than to silently sell the furthest strike
        listed, which is a different trade from the one requested."""
        contracts = chain([6300, 6325, 6350])
        assert nearest_listed(contracts, PUT, 6000.0, "put") is None
        assert nearest_listed(contracts, CALL, 6900.0, "call") is None

    def test_the_wing_sits_further_out(self):
        contracts = chain([6250, 6275, 6300, 6325, 6350])
        short = contract(6325, PUT)
        assert wing(contracts, PUT, short, 25.0).strike == 6300

        short_call = contract(6300, CALL)
        assert wing(contracts, CALL, short_call, 25.0).strike == 6325


class TestSessionSampling:
    def test_sampling_is_evenly_spaced(self):
        days = [date(2026, 1, d) for d in range(1, 21)]
        picked = pick_sessions(days, 5)
        assert len(picked) == 5
        assert picked[0] == days[0]
        assert picked == sorted(picked)

    def test_asking_for_more_than_available_returns_all(self):
        days = [date(2026, 1, d) for d in range(1, 6)]
        assert pick_sessions(days, 50) == days

    def test_sampling_is_deterministic(self):
        """A re-rollable sample can be re-rolled until it flatters."""
        days = [date(2026, 1, d) for d in range(1, 31)]
        assert pick_sessions(days, 7) == pick_sessions(days, 7)


class TestCreditIsWhatASellerReceives:
    def _setup(self, short_bid, short_ask, long_bid, long_ask):
        contracts = chain([6300, 6325, 6350, 6375, 6400])
        rows = [
            # ATM pair pins the implied spot near 6375 via parity.
            (osi_symbol("SPXW", DAY, CALL, 6375), 10.0, 10.2),
            (osi_symbol("SPXW", DAY, PUT, 6375), 10.0, 10.2),
            (osi_symbol("SPXW", DAY, PUT, 6325), short_bid, short_ask),
            (osi_symbol("SPXW", DAY, PUT, 6300), long_bid, long_ask),
        ]
        return contracts, book_from(rows)

    def test_credit_is_short_bid_minus_long_ask(self):
        contracts, book = self._setup(3.00, 3.40, 1.00, 1.30)
        check = check_session(DAY, contracts, book, ENTRY, -0.0075, "put",
                              25.0, 0.04, 0.013)
        assert not check.note
        assert check.credit == pytest.approx(3.00 - 1.30)

    def test_a_bidless_short_is_not_a_trade(self):
        """Marking a bidless contract at its mid is how a 0DTE backtest invents
        premium that never existed."""
        contracts, book = self._setup(0.0, 0.30, 0.05, 0.20)
        check = check_session(DAY, contracts, book, ENTRY, -0.0075, "put",
                              25.0, 0.04, 0.013)
        assert "bidless" in check.note
        assert not check.tradable

    def test_a_missing_leg_is_not_a_trade(self):
        contracts = chain([6300, 6325, 6350, 6375, 6400])
        book = book_from([
            (osi_symbol("SPXW", DAY, CALL, 6375), 10.0, 10.2),
            (osi_symbol("SPXW", DAY, PUT, 6375), 10.0, 10.2),
            (osi_symbol("SPXW", DAY, PUT, 6325), 3.0, 3.4),
        ])
        check = check_session(DAY, contracts, book, ENTRY, -0.0075, "put",
                              25.0, 0.04, 0.013)
        assert not check.tradable

    def test_no_quotes_at_entry_is_reported_not_guessed(self):
        contracts = chain([6300, 6325, 6350])
        check = check_session(DAY, contracts, QuoteBook(pd.DataFrame()), ENTRY,
                              -0.0075, "put", 25.0, 0.04, 0.013)
        assert not check.tradable
        assert check.note


class TestSummary:
    def _check(self, breached, credit=1.0, note=""):
        from backtest.validate import SpreadCheck

        return SpreadCheck(
            day=DAY, entry_ts=ENTRY, side="put", spot=6400.0, model_level=6300.0,
            short_strike=6300.0, long_strike=6275.0, credit=credit,
            settlement=6350.0, breached=breached, note=note,
        )

    def test_breach_rate_counts_only_resolved_sessions(self):
        summary = ValidationSummary(
            [self._check(True), self._check(False), self._check(False),
             self._check(None, credit=0.0, note="bidless")],
            confidence=0.95,
        )
        assert len(summary.tradable) == 3
        assert summary.breach_rate == pytest.approx(1 / 3)

    def test_expected_rate_is_the_complement_of_confidence(self):
        assert ValidationSummary([], 0.95).expected_breach_rate == pytest.approx(0.05)

    def test_median_credit_ignores_untradable_sessions(self):
        summary = ValidationSummary(
            [self._check(False, credit=2.0), self._check(False, credit=4.0),
             self._check(None, credit=0.0, note="bidless")],
            confidence=0.95,
        )
        assert summary.median_credit == pytest.approx(3.0)

    def test_no_resolved_sessions_gives_nan_not_zero(self):
        """Zero would read as a perfect record."""
        summary = ValidationSummary([], confidence=0.95)
        assert summary.breach_rate != summary.breach_rate  # NaN
