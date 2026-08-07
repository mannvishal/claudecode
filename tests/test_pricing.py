import math
from datetime import date, datetime

import pytest

from spreadscout.pricing import (
    CALL,
    ET,
    PUT,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_vega,
    implied_vol,
    norm_cdf,
    prob_above,
    prob_below,
    year_fraction,
)

S, K, T, R, Q, SIG = 5000.0, 5000.0, 0.05, 0.04, 0.013, 0.20


def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_put_call_parity():
    call = bs_price(S, K, T, R, Q, SIG, CALL)
    put = bs_price(S, K, T, R, Q, SIG, PUT)
    lhs = call - put
    rhs = S * math.exp(-Q * T) - K * math.exp(-R * T)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_deep_itm_call_approaches_discounted_forward():
    price = bs_price(S, 100.0, T, R, Q, SIG, CALL)
    expected = S * math.exp(-Q * T) - 100.0 * math.exp(-R * T)
    assert price == pytest.approx(expected, rel=1e-6)


def test_deep_otm_option_is_worthless_not_negative():
    assert bs_price(S, 20000.0, T, R, Q, SIG, CALL) >= 0.0
    assert bs_price(S, 1.0, T, R, Q, SIG, PUT) >= 0.0


def test_zero_vol_gives_discounted_intrinsic():
    price = bs_price(S, 4000.0, T, R, Q, 0.0, CALL)
    fwd = S * math.exp((R - Q) * T)
    assert price == pytest.approx(math.exp(-R * T) * (fwd - 4000.0))


def test_implied_vol_round_trips_where_vega_is_meaningful():
    for sigma in (0.05, 0.15, 0.40, 1.20):
        for strike in (4500.0, 5000.0, 5500.0):
            for kind in (CALL, PUT):
                price = bs_price(S, strike, T, R, Q, sigma, kind)
                if bs_vega(S, strike, T, R, Q, sigma) < 1e-3:
                    continue  # no extrinsic value to invert; see test below
                solved = implied_vol(price, S, strike, T, R, Q, kind)
                assert solved == pytest.approx(sigma, abs=1e-4), (sigma, strike, kind)


def test_implied_vol_rejects_arbitrage_violating_price():
    # Above the upper no-arb bound: a call worth more than the underlying.
    assert implied_vol(S * 2, S, K, T, R, Q, CALL) is None
    assert implied_vol(0.0, S, K, T, R, Q, CALL) is None


def test_implied_vol_refuses_options_with_no_extrinsic_value():
    """A deep-ITM option is all intrinsic: every vol prices it identically.

    Bisection would return whichever bracket edge it landed on, which looks like
    a vol and carries no information. ``None`` keeps that out of the screener.
    """
    price = bs_price(S, 1000.0, T, R, Q, 0.05, CALL)
    assert implied_vol(price, S, 1000.0, T, R, Q, CALL) is None

    # And a worthless far-OTM one at the other end of the band.
    tiny = bs_price(S, 20000.0, T, R, Q, 0.05, CALL)
    assert implied_vol(tiny, S, 20000.0, T, R, Q, CALL) is None


def test_implied_vol_survives_zero_dte():
    tiny = year_fraction(datetime(2026, 8, 7, 15, 30, tzinfo=ET), date(2026, 8, 7))
    price = bs_price(S, 5010.0, tiny, R, Q, 0.30, CALL)
    assert implied_vol(price, S, 5010.0, tiny, R, Q, CALL) == pytest.approx(0.30, abs=1e-3)


def test_delta_bounds_and_sign():
    assert 0.0 <= bs_delta(S, K, T, R, Q, SIG, CALL) <= 1.0
    assert -1.0 <= bs_delta(S, K, T, R, Q, SIG, PUT) <= 0.0
    # Further OTM means smaller magnitude.
    near = abs(bs_delta(S, 5100.0, T, R, Q, SIG, CALL))
    far = abs(bs_delta(S, 5400.0, T, R, Q, SIG, CALL))
    assert far < near


def test_delta_parity():
    cd = bs_delta(S, K, T, R, Q, SIG, CALL)
    pd = bs_delta(S, K, T, R, Q, SIG, PUT)
    assert cd - pd == pytest.approx(math.exp(-Q * T), abs=1e-9)


def test_gamma_and_vega_are_positive_and_peak_atm():
    assert bs_gamma(S, K, T, R, Q, SIG) > 0
    assert bs_vega(S, K, T, R, Q, SIG) > 0
    assert bs_gamma(S, K, T, R, Q, SIG) > bs_gamma(S, 6000.0, T, R, Q, SIG)


def test_gamma_explodes_as_expiry_approaches():
    """The reason 0DTE is different in kind, not just in degree."""
    one_trading_day = 6.5 / (24 * 365)
    far = bs_gamma(S, K, 0.08, R, Q, SIG)
    near = bs_gamma(S, K, one_trading_day, R, Q, SIG)
    assert near > far * 10


def test_probabilities_are_complementary():
    for strike in (4000.0, 5000.0, 6000.0):
        below = prob_below(S, strike, T, SIG, R - Q)
        assert below + prob_above(S, strike, T, SIG, R - Q) == pytest.approx(1.0)
        assert 0.0 <= below <= 1.0


def test_risk_neutral_prob_matches_digital_from_delta():
    """N(d2) should equal the undiscounted probability of finishing ITM."""
    strike = 5200.0
    p_itm = prob_above(S, strike, T, SIG, R - Q)
    # Numerically differentiate the call price w.r.t. strike: -dC/dK = e^-rT N(d2)
    h = 0.01
    dc = (bs_price(S, strike + h, T, R, Q, SIG, CALL) - bs_price(S, strike - h, T, R, Q, SIG, CALL))
    digital = -dc / (2 * h) * math.exp(R * T)
    assert p_itm == pytest.approx(digital, abs=1e-4)


def test_year_fraction_pm_vs_am_settlement():
    now = datetime(2026, 8, 7, 9, 30, tzinfo=ET)
    pm = year_fraction(now, date(2026, 8, 7), pm_settled=True)
    am = year_fraction(now, date(2026, 8, 7), pm_settled=False)
    assert pm > am
    assert pm * 365 * 24 == pytest.approx(6.5, abs=1e-6)


def test_year_fraction_never_returns_zero_after_the_bell():
    after = datetime(2026, 8, 7, 16, 30, tzinfo=ET)
    assert year_fraction(after, date(2026, 8, 7)) > 0
