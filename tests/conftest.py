"""Synthetic chain fixtures.

Contracts are quoted at exactly their Black-Scholes value under a known vol
surface, with an optional bid-ask band around it. That makes the risk-neutral
expectancy identity in ``test_risk.py`` checkable to floating-point precision:
if the model reprices a market it built, expectancy before costs must be zero.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from spreadscout.config import Config
from spreadscout.models import OptionContract, Vertical
from spreadscout.pricing import CALL, PUT, bs_price

SPOT = 5000.0
EXPIRY = date(2026, 8, 7)
R, Q = 0.04, 0.013
T = 6.5 / (24 * 365)  # one trading day


def vol_for(strike: float, spot: float = SPOT, atm: float = 0.18) -> float:
    """A downward-sloping smile, as SPX actually exhibits."""
    k = math.log(strike / spot)
    return max(0.05, atm * (1.0 - 1.5 * k + 4.0 * k * k))


def make_contract(
    strike: float,
    option_type: str,
    *,
    spot: float = SPOT,
    half_spread: float = 0.05,
    volume: int = 500,
    open_interest: int = 5000,
    t: float = T,
) -> OptionContract:
    """Quote a contract at its fair value inside a symmetric bid-ask band.

    The bid is *not* floored at a minimum tick. Far-OTM 0DTE strikes really do
    go bidless, and letting them do so here keeps the fixtures honest: an
    earlier version floored the bid at 0.05, which silently inflated the mid on
    worthless strikes and handed every downstream test a fabricated implied vol.
    """
    fair = bs_price(spot, strike, t, R, Q, vol_for(strike, spot), option_type)
    bid = max(0.0, fair - half_spread)
    ask = fair + half_spread
    c = OptionContract(
        symbol=f"SPXW260807{'C' if option_type == CALL else 'P'}{int(strike * 1000):08d}",
        underlying="SPX",
        root_symbol="SPXW",
        strike=strike,
        option_type=option_type,
        expiration=EXPIRY,
        bid=bid,
        ask=ask,
        bid_size=25,
        ask_size=25,
        volume=volume,
        open_interest=open_interest,
        contract_size=100,
        expiration_type="weeklys",
    )
    return c.enrich(spot, t, R, Q)


@pytest.fixture
def cfg() -> Config:
    c = Config()
    c.risk_free_rate, c.dividend_yield = R, Q
    c.costs.fill_fraction = 0.0  # value at mid so the identity is exact
    c.costs.commission_per_contract = 0.0
    c.costs.exchange_fee_per_contract = 0.0
    return c


# One standard deviation of SPX at 0DTE is only ~25 index points, so a 0.15-delta
# short strike sits about 25 points from spot -- not the several hundred that
# intuition from 45-DTE chains suggests.
PUT_SHORT, PUT_LONG = 4975.0, 4965.0
CALL_SHORT, CALL_LONG = 5025.0, 5035.0


@pytest.fixture
def put_spread() -> Vertical:
    return Vertical(
        kind="put_credit",
        short=make_contract(PUT_SHORT, PUT),
        long=make_contract(PUT_LONG, PUT),
    )


@pytest.fixture
def call_spread() -> Vertical:
    return Vertical(
        kind="call_credit",
        short=make_contract(CALL_SHORT, CALL),
        long=make_contract(CALL_LONG, CALL),
    )


@pytest.fixture
def chain() -> list[OptionContract]:
    out = []
    for strike in range(4850, 5151, 5):
        out.append(make_contract(float(strike), PUT))
        out.append(make_contract(float(strike), CALL))
    return out
