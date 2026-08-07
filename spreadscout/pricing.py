"""Black-Scholes pricing and probability math for European index options.

SPX/SPXW options are European-exercise and cash-settled. That makes Black-Scholes
the *exact* model here rather than an approximation: there is no early-exercise
premium to account for, and no discrete dividend on the index itself (the carry
term absorbs the index dividend yield). For American-style equity options this
module would only be an approximation -- see the note in ``README.md``.

Everything is stdlib; no scipy. The normal CDF comes from ``math.erf`` and
implied vol is solved by bisection, which is slower than Newton but cannot
diverge on the near-zero-vega wings where 0DTE screening actually operates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# One minute, expressed in years. Time to expiry is floored at this value so the
# 0DTE math stays finite into the closing bell instead of dividing by zero.
MIN_T = 1.0 / (365.0 * 24.0 * 60.0)

# Implied vol search bracket. 0DTE wings routinely print above 200% vol, hence
# the generous upper bound.
IV_LOW = 1e-4
IV_HIGH = 8.0

# Minimum extrinsic value, in index points, for an implied vol to mean anything.
# Below this the option is all intrinsic, vega is negligible, and the "solved"
# vol is an artifact of the search bracket rather than information.
RESOLVABLE_EXTRINSIC = 1e-4

CALL = "call"
PUT = "put"


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def year_fraction(now: datetime, expiry: date, pm_settled: bool = True) -> float:
    """Calendar-time year fraction from ``now`` until expiry settlement.

    SPXW (the weekly root, which is what every 0DTE contract is) settles on the
    4pm ET close. The monthly SPX root settles AM against the opening print, so
    pass ``pm_settled=False`` for those -- getting this wrong misprices a
    third-Friday contract by a full trading day, which at 0DTE is the entire
    remaining life of the option.
    """
    settle_time = time(16, 0) if pm_settled else time(9, 30)
    settle = datetime.combine(expiry, settle_time, tzinfo=ET)
    if now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    seconds = (settle - now.astimezone(ET)).total_seconds()
    return max(seconds / (365.0 * 24.0 * 3600.0), MIN_T)


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    return d1, d1 - v


def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: str) -> float:
    """Black-Scholes value of a European option."""
    T = max(T, MIN_T)
    if S <= 0 or K <= 0:
        raise ValueError(f"non-positive S={S} or K={K}")
    if sigma <= 0:
        # Degenerate case: value is the discounted forward intrinsic.
        fwd = S * math.exp((r - q) * T)
        intrinsic = fwd - K if kind == CALL else K - fwd
        return math.exp(-r * T) * max(intrinsic, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    df, dq = math.exp(-r * T), math.exp(-q * T)
    if kind == CALL:
        return S * dq * norm_cdf(d1) - K * df * norm_cdf(d2)
    return K * df * norm_cdf(-d2) - S * dq * norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: str) -> float:
    T = max(T, MIN_T)
    if sigma <= 0:
        fwd = S * math.exp((r - q) * T)
        if kind == CALL:
            return 1.0 if fwd > K else 0.0
        return -1.0 if fwd < K else 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    dq = math.exp(-q * T)
    return dq * norm_cdf(d1) if kind == CALL else -dq * norm_cdf(-d1)


def bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    T = max(T, MIN_T)
    if sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return math.exp(-q * T) * norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Vega per 1.00 (100 percentage points) of vol, per 1 unit of index."""
    T = max(T, MIN_T)
    if sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T)


def bs_theta(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: str) -> float:
    """Theta per year. Divide by 365 for the conventional per-day figure."""
    T = max(T, MIN_T)
    if sigma <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    df, dq = math.exp(-r * T), math.exp(-q * T)
    decay = -S * dq * norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
    if kind == CALL:
        return decay - r * K * df * norm_cdf(d2) + q * S * dq * norm_cdf(d1)
    return decay + r * K * df * norm_cdf(-d2) - q * S * dq * norm_cdf(-d1)


def implied_vol(
    price: float, S: float, K: float, T: float, r: float, q: float, kind: str
) -> float | None:
    """Solve for implied vol by bisection. Returns ``None`` if unsolvable.

    ``None`` covers two distinct situations, both of which mean "do not trade
    this contract":

    * The quote sits outside the no-arbitrage band. On a real chain that is
      nearly always a stale or crossed market rather than free money.
    * The contract has no meaningful extrinsic value, so vega is ~0 and every
      vol reprices it equally well. Deep ITM options land here. Bisection would
      happily return whichever bracket edge it converged on, which looks like a
      number and is not one -- returning ``None`` keeps that out of the screener.
    """
    T = max(T, MIN_T)
    if price <= 0 or S <= 0 or K <= 0:
        return None

    lo_price = bs_price(S, K, T, r, q, IV_LOW, kind)
    hi_price = bs_price(S, K, T, r, q, IV_HIGH, kind)
    if not (lo_price <= price <= hi_price):
        return None

    # Extrinsic value must be resolvable above the price grid we can represent.
    if price - lo_price < RESOLVABLE_EXTRINSIC or hi_price - price < RESOLVABLE_EXTRINSIC:
        return None

    # Converge to near machine precision rather than "close enough to display".
    # Downstream, expectancy is computed as the difference between two option
    # values, so solver slop does not cancel -- it shows up as a small non-zero
    # risk-neutral edge, which is exactly the artifact this tool exists to deny.
    # Bisection from an 8-wide bracket reaches 1e-12 in ~43 iterations.
    lo, hi = IV_LOW, IV_HIGH
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, r, q, mid, kind) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def prob_below(S: float, K: float, T: float, sigma: float, mu: float) -> float:
    """P(S_T < K) under GBM with total drift ``mu`` (continuously compounded).

    Pass ``mu = r - q`` for the risk-neutral measure. Note that the risk-neutral
    probability is *not* a forecast: it embeds the variance risk premium, which
    is precisely the thing a credit-spread seller is being paid to bear. Using it
    as a win-rate estimate produces an expectancy of zero by construction. See
    ``risk.expected_value`` for the honest framing.
    """
    T = max(T, MIN_T)
    if sigma <= 0:
        return 1.0 if S * math.exp(mu * T) < K else 0.0
    d2 = (math.log(S / K) + (mu - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return norm_cdf(-d2)


def prob_above(S: float, K: float, T: float, sigma: float, mu: float) -> float:
    return 1.0 - prob_below(S, K, T, sigma, mu)


@dataclass(frozen=True)
class Measure:
    """A probability measure under which to value a position.

    ``vol_multiplier`` scales implied vol to get the vol you actually believe
    will be realized. The empirical variance risk premium in SPX means realized
    vol has historically run below implied -- roughly 0.80-0.90x on average since
    the 1990s -- which is the entire economic case for selling premium. But that
    average is an average *over* crash days, not instead of them, and the
    multiplier is stable right up until it isn't. Setting this below 1.0 is you
    making a forecast; the tool will not make it for you.

    ``drift`` is the continuously-compounded expected return of the underlying.
    Use ``r - q`` for risk-neutral valuation.
    """

    vol_multiplier: float = 1.0
    drift: float | None = None
    label: str = "risk-neutral"

    def effective_drift(self, r: float, q: float) -> float:
        return self.drift if self.drift is not None else (r - q)

    @property
    def is_risk_neutral(self) -> bool:
        return self.vol_multiplier == 1.0 and self.drift is None
