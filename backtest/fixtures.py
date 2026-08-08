"""Synthetic Databento frames, for exercising the pipeline without spending money.

These match the OPRA ``definition`` and ``cmbp-1`` column layouts, so the engine
runs the same code path it will run against live data. They are a test harness,
not a data source: the price path is a generated GBM walk and the quotes are
Black-Scholes values with a fixed spread. Nothing measured against a fixture
says anything about whether the strategy works -- it says the plumbing is
connected. Every number this produces is manufactured.
"""

from __future__ import annotations

import math
import random
from datetime import date, datetime, time

import pandas as pd

from spreadscout.pricing import bs_price

from .data import CALL, PUT, osi_symbol
from .source import ET

SESSION_OPEN = time(9, 45)
SESSION_CLOSE = time(16, 0)


def _smile(strike: float, spot: float, atm_vol: float) -> float:
    """Downward-sloping skew, as SPX exhibits."""
    k = math.log(strike / spot)
    return max(0.04, atm_vol * (1.0 - 1.6 * k + 5.0 * k * k))


def price_path(
    day: date, open_price: float, annual_vol: float, drift_points: float,
    step_seconds: int, seed: int,
) -> list[tuple[pd.Timestamp, float]]:
    rng = random.Random(seed)
    start = pd.Timestamp(datetime.combine(day, SESSION_OPEN)).tz_localize(ET)
    end = pd.Timestamp(datetime.combine(day, SESSION_CLOSE)).tz_localize(ET)
    stamps = list(pd.date_range(start, end, freq=f"{step_seconds}s", tz=ET))

    n = len(stamps)
    per_step_vol = annual_vol * math.sqrt(step_seconds / (365 * 24 * 3600))
    drift_per_step = drift_points / max(n - 1, 1)

    out, price = [], open_price
    for i, ts in enumerate(stamps):
        if i:
            price = price * math.exp(rng.gauss(0.0, per_step_vol)) + drift_per_step
        out.append((ts, price))
    return out


def make_definitions(day: date, strikes: list[float], root: str = "SPXW") -> pd.DataFrame:
    rows = []
    for strike in strikes:
        for option_type in (PUT, CALL):
            rows.append({
                "ts_recv": pd.Timestamp(datetime.combine(day, time(0, 0))).tz_localize(ET),
                "raw_symbol": osi_symbol(root, day, option_type, strike),
                "instrument_class": "P" if option_type == PUT else "C",
                "strike_price": strike,
                "expiration": pd.Timestamp(day),
            })
    return pd.DataFrame(rows)


def make_mbp1(
    day: date, strikes: list[float], path: list[tuple[pd.Timestamp, float]],
    atm_vol: float = 0.16, half_spread: float = 0.10, root: str = "SPXW",
    r: float = 0.04, q: float = 0.013,
) -> pd.DataFrame:
    """Top-of-book quotes for every strike at every step of the path."""
    close = pd.Timestamp(datetime.combine(day, time(16, 0))).tz_localize(ET)
    rows = []
    for ts, spot in path:
        T = max((close - ts).total_seconds(), 60.0) / (365 * 24 * 3600)
        for strike in strikes:
            vol = _smile(strike, spot, atm_vol)
            for option_type in (PUT, CALL):
                fair = bs_price(spot, strike, T, r, q, vol, option_type)
                bid = max(0.0, fair - half_spread)
                ask = fair + half_spread
                rows.append({
                    "ts_recv": ts,
                    "ts_event": ts,
                    "symbol": osi_symbol(root, day, option_type, strike),
                    "bid_px_00": bid,
                    "ask_px_00": ask,
                    "bid_sz_00": 25,
                    "ask_sz_00": 25,
                    "action": "M",
                    "side": "N",
                })
    return pd.DataFrame(rows)


def build_session(
    day: date, open_price: float = 5000.0, annual_vol: float = 0.16,
    drift_points: float = 0.0, strike_step: float = 5.0, n_strikes: int = 61,
    step_seconds: int = 300, seed: int = 7, half_spread: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[pd.Timestamp, float]]]:
    """A full synthetic session: definitions, quotes, and the path that made them."""
    centre = round(open_price / strike_step) * strike_step
    half = n_strikes // 2
    strikes = [centre + (i - half) * strike_step for i in range(n_strikes)]

    path = price_path(day, open_price, annual_vol, drift_points, step_seconds, seed)
    definitions = make_definitions(day, strikes)
    quotes = make_mbp1(day, strikes, path, atm_vol=annual_vol, half_spread=half_spread)
    return definitions, quotes, path


def seed_cache(cache, cfg, day: date, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write a synthetic session into the parquet cache.

    Once seeded, the engine reads it through exactly the same cache path it
    would use for real data, so the fixture exercises the ingest and lookup code
    rather than bypassing it.
    """
    from .cache import CacheKey
    from .config import SCHEMA_DEFINITION, SCHEMA_QUOTES

    definitions, quotes, _path = build_session(day, **kwargs)

    # Keyed on the parent symbol, matching the request the engine issues. A
    # fixture seeded under a different key than the engine looks up is a cache
    # miss that only shows up at run time.
    cache.write(
        CacheKey.build(cfg.data.dataset, SCHEMA_DEFINITION, day, [cfg.data.parent_symbol]),
        definitions,
        meta={"synthetic": True},
    )
    symbols = sorted(quotes["symbol"].unique())
    cache.write(
        CacheKey.build(cfg.data.dataset, SCHEMA_QUOTES, day, symbols),
        quotes,
        meta={"synthetic": True},
    )
    return definitions, quotes
