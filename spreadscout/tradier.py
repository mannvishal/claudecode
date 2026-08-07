"""Tradier REST client -- read-only by construction.

There is deliberately no order-placement method in this module. The tool is a
recommender: it prints tickets for a human to review and submit. If you later
want it to trade, that capability should be added consciously and with the
guard rails in ``risk.py`` wired to it, not inherited by accident from a client
that happened to expose ``POST /orders``.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any

import requests

log = logging.getLogger(__name__)

PRODUCTION = "https://api.tradier.com/v1"
SANDBOX = "https://sandbox.tradier.com/v1"

RETRY_STATUS = {429, 500, 502, 503, 504}


class TradierError(RuntimeError):
    pass


def _as_list(value: Any) -> list:
    """Tradier collapses single-element arrays into bare objects. Undo that."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


class TradierClient:
    def __init__(
        self,
        token: str,
        base_url: str = PRODUCTION,
        timeout: float = 15.0,
        max_retries: int = 4,
        session: requests.Session | None = None,
    ):
        if not token:
            raise TradierError("no Tradier access token supplied")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        )

    @property
    def is_sandbox(self) -> bool:
        return "sandbox" in self.base_url

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        delay = 1.0
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
            else:
                if resp.status_code == 401:
                    raise TradierError(
                        "Tradier rejected the access token (401). Check TRADIER_ACCESS_TOKEN "
                        "and that it matches the environment you are pointing at "
                        f"({'sandbox' if self.is_sandbox else 'production'})."
                    )
                if resp.status_code not in RETRY_STATUS:
                    resp.raise_for_status()
                    return resp.json() or {}
                last_error = TradierError(f"HTTP {resp.status_code} from {path}")

            if attempt < self.max_retries:
                log.warning("Tradier %s failed (%s); retrying in %.0fs", path, last_error, delay)
                time.sleep(delay)
                delay *= 2

        raise TradierError(f"Tradier request to {path} failed after retries: {last_error}")

    # --- market data -----------------------------------------------------

    def quotes(self, symbols: list[str]) -> dict[str, dict]:
        data = self._get("markets/quotes", {"symbols": ",".join(symbols), "greeks": "false"})
        rows = _as_list((data.get("quotes") or {}).get("quote"))
        return {row["symbol"]: row for row in rows}

    def spot(self, symbol: str) -> float:
        """Best available spot for an underlying.

        Cash indices like SPX have no consolidated last sale during the session
        and quote a wide synthetic bid/ask, so the mid of that band is a better
        spot estimate than ``last``, which can be minutes stale.
        """
        row = self.quotes([symbol]).get(symbol)
        if not row:
            raise TradierError(f"no quote returned for {symbol}")
        bid, ask = float(row.get("bid") or 0), float(row.get("ask") or 0)
        last = float(row.get("last") or row.get("lastPrice") or 0)
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            # A cash index's synthetic band can be very wide; if last sits inside
            # it, prefer last as the tighter estimate.
            if last > 0 and bid <= last <= ask:
                return last
            return mid
        if last > 0:
            return last
        raise TradierError(f"no usable price for {symbol}")

    def expirations(self, symbol: str) -> list[date]:
        data = self._get(
            "markets/options/expirations",
            {"symbol": symbol, "includeAllRoots": "true", "strikes": "false"},
        )
        raw = (data.get("expirations") or {}).get("date")
        return [date.fromisoformat(d) for d in _as_list(raw)]

    def chain(self, symbol: str, expiration: date) -> list[dict]:
        data = self._get(
            "markets/options/chains",
            {"symbol": symbol, "expiration": expiration.isoformat(), "greeks": "true"},
        )
        return _as_list((data.get("options") or {}).get("option"))

    def clock(self) -> dict:
        return (self._get("markets/clock") or {}).get("clock", {})

    def calendar(self, month: int, year: int) -> dict:
        return self._get("markets/calendar", {"month": month, "year": year})

    def history(self, symbol: str, start: date, end: date, interval: str = "daily") -> list[dict]:
        data = self._get(
            "markets/history",
            {
                "symbol": symbol,
                "interval": interval,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )
        return _as_list((data.get("history") or {}).get("day"))

    # --- account (read-only) ---------------------------------------------

    def profile(self) -> dict:
        return (self._get("user/profile") or {}).get("profile", {})

    def account_ids(self) -> list[str]:
        accounts = _as_list((self.profile().get("account")))
        return [a["account_number"] for a in accounts if "account_number" in a]

    def balances(self, account_id: str) -> dict:
        return (self._get(f"accounts/{account_id}/balances") or {}).get("balances", {})

    def equity(self, account_id: str) -> float:
        """Total account equity, used as the denominator for position sizing."""
        b = self.balances(account_id)
        for key in ("total_equity", "equity", "account_value"):
            if b.get(key) is not None:
                return float(b[key])
        raise TradierError(f"could not determine equity from balances: {sorted(b)}")

    def positions(self, account_id: str) -> list[dict]:
        data = self._get(f"accounts/{account_id}/positions")
        return _as_list((data.get("positions") or {}).get("position"))

    def gainloss(self, account_id: str, start: date, end: date) -> list[dict]:
        data = self._get(
            f"accounts/{account_id}/gainloss",
            {"start": start.isoformat(), "end": end.isoformat(), "limit": 500},
        )
        return _as_list((data.get("gainloss") or {}).get("closed_position"))
