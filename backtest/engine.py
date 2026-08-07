"""Orchestration: data -> signal -> fill -> trade.

This module owns no pricing, no strike logic and no cost assumptions. It walks
sessions, asks each layer its question in order, and records what came back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from .config import SCHEMA_DEFINITION, SCHEMA_MBP1, BacktestConfig
from .data import (
    CALL,
    PUT,
    Contract,
    QuoteBook,
    contracts_from_definitions,
    osi_symbol,
    spot_from_parity,
    year_fraction_to_close,
)
from .fills import FillModel, MidMinusEdgeFill, SettlementFill, Structure
from .signals import (
    NO_QUOTE,
    SETTLEMENT,
    build_structure,
    check_exit,
    entry_timestamp,
)
from .source import ET, Fetcher

log = logging.getLogger(__name__)


@dataclass
class Trade:
    day: date
    structure: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    strikes: str
    contracts: int
    credit: float          # index points received at entry
    exit_debit: float      # index points paid to close
    gross_pnl: float       # dollars before commissions
    commissions: float
    net_pnl: float
    exit_reason: str
    max_loss: float        # dollars, defined risk
    entry_note: str = ""

    def as_row(self) -> dict:
        return {
            "day": self.day,
            "structure": self.structure,
            "strikes": self.strikes,
            "entry_ts": self.entry_ts,
            "exit_ts": self.exit_ts,
            "contracts": self.contracts,
            "credit_pts": round(self.credit, 4),
            "exit_debit_pts": round(self.exit_debit, 4),
            "gross_pnl": round(self.gross_pnl, 2),
            "commissions": round(self.commissions, 2),
            "net_pnl": round(self.net_pnl, 2),
            "exit_reason": self.exit_reason,
            "max_loss": round(self.max_loss, 2),
        }


@dataclass
class SessionResult:
    day: date
    trade: Trade | None = None
    skipped: str = ""
    pulls: list = field(default_factory=list)


class Engine:
    def __init__(
        self, cfg: BacktestConfig, fetcher: Fetcher, fill_model: FillModel | None = None,
        echo=print,
    ):
        self.cfg = cfg
        self.fetcher = fetcher
        self.fill = fill_model or MidMinusEdgeFill(cfg.execution)
        self.echo = echo

    # --- data assembly ---------------------------------------------------

    def _chain(self, day: date) -> list[Contract]:
        frame, _ = self.fetcher.fetch(
            SCHEMA_DEFINITION, day, None,
            self.cfg.data.quote_start, self.cfg.data.quote_end, stype_in="parent",
        )
        return contracts_from_definitions(frame, self.cfg.data.underlying_root, day)

    def _band_symbols(self, contracts: list[Contract], anchor: float) -> list[str]:
        """Restrict the quote pull to strikes that could plausibly be traded.

        This is the difference between a manageable request and an unusable one.
        A full-chain mbp-1 pull for OPRA is enormous; a band around the anchor
        covering both short strikes and their wings is a small fraction of it,
        and the strikes outside the band are ones the selector would reject
        anyway.
        """
        band = anchor * self.cfg.signal.strike_band_pct
        lo, hi = anchor - band, anchor + band
        # Widen by the wing width so the long legs are inside the pull.
        lo -= self.cfg.signal.width_points
        hi += self.cfg.signal.width_points
        return [c.symbol for c in contracts if lo <= c.strike <= hi]

    def _quotes(self, day: date, symbols: list[str]) -> QuoteBook:
        frame, _ = self.fetcher.fetch(
            SCHEMA_MBP1, day, symbols,
            self.cfg.data.quote_start, self.cfg.data.quote_end, stype_in="raw_symbol",
        )
        return QuoteBook(frame)

    def _anchor(self, day: date, contracts: list[Contract]) -> float | None:
        """Reference price used only to size the strike band.

        Deliberately crude: it just has to be close enough that the band covers
        the strikes the selector will want. The traded spot comes from put-call
        parity on the quotes themselves, once we have them.
        """
        strikes = sorted({c.strike for c in contracts})
        if not strikes:
            return None
        # Listed 0DTE chains are built around the money, so the median listed
        # strike is a serviceable centre without buying an index feed.
        return strikes[len(strikes) // 2]

    # --- one session -----------------------------------------------------

    def run_day(self, day: date) -> SessionResult:
        self.echo(f"\n{day} ---------------------------------------------")
        contracts = self._chain(day)
        if not contracts:
            return SessionResult(day, skipped="no contracts in the definition file")

        anchor = self._anchor(day, contracts)
        if anchor is None:
            return SessionResult(day, skipped="could not anchor a strike band")

        symbols = self._band_symbols(contracts, anchor)
        if not symbols:
            return SessionResult(day, skipped=f"no strikes within the band around {anchor:g}")

        book = self._quotes(day, symbols)
        if len(book) == 0:
            return SessionResult(day, skipped="no quotes returned")

        in_band = [c for c in contracts if c.symbol in set(symbols)]
        entry_ts = entry_timestamp(day, self.cfg.signal.entry_time)
        snapshot = book.snapshot(entry_ts)
        if not snapshot:
            return SessionResult(day, skipped=f"book was empty at {entry_ts:%H:%M}")

        T = year_fraction_to_close(entry_ts, day)
        spot = spot_from_parity(snapshot, in_band, self.cfg.data.risk_free_rate, T)
        if spot is None:
            return SessionResult(day, skipped="could not derive spot from put-call parity")

        structure, why = build_structure(
            in_band, snapshot, spot, T, self.cfg.signal,
            self.cfg.data.risk_free_rate, self.cfg.data.dividend_yield,
            self.cfg.data.underlying_root, day,
        )
        if structure is None:
            return SessionResult(day, skipped=why)

        entry = self.fill.entry(structure, snapshot, self.cfg.signal.contracts)
        if not entry.tradable:
            return SessionResult(day, skipped=f"entry not fillable: {entry.reason}")
        if entry.price < self.cfg.signal.min_credit:
            return SessionResult(
                day,
                skipped=f"credit {entry.price:.2f} below min_credit "
                        f"{self.cfg.signal.min_credit:.2f}",
            )

        self.echo(
            f"  spot {spot:,.2f} (parity)  entry {entry_ts:%H:%M}  "
            f"{structure.describe()}  credit {entry.price:.2f}"
        )
        return SessionResult(day, trade=self._walk_forward(day, book, structure, entry, spot))

    def _walk_forward(self, day, book: QuoteBook, structure: Structure, entry, spot: float) -> Trade:
        cfg = self.cfg
        contracts_n = cfg.signal.contracts
        mult = cfg.execution.contract_multiplier
        entry_ts = entry_timestamp(day, cfg.signal.entry_time)
        end_ts = entry_timestamp(day, cfg.data.quote_end)

        last_debit = 0.0
        exit_ts, exit_reason = end_ts, SETTLEMENT

        for ts in book.timeline(entry_ts, end_ts, step_seconds=60):
            snapshot = book.snapshot(ts)
            mark = self.fill.exit(structure, snapshot, contracts_n)
            debit = mark.price if mark.tradable else None
            if debit is not None:
                last_debit = debit

            decision = check_exit(ts, entry.price, debit, cfg.signal, day)
            if decision.should_exit:
                exit_ts, exit_reason = ts, decision.reason
                break
        else:
            # Never triggered: the position runs to the close and cash-settles.
            exit_reason = SETTLEMENT

        if exit_reason in (SETTLEMENT, NO_QUOTE):
            settlement = self._settlement_price(book, structure, end_ts, spot)
            closer = SettlementFill(cfg.execution, settlement)
            final = closer.exit(structure, {}, contracts_n)
            exit_debit, exit_commissions = final.price, 0.0
            exit_ts = end_ts
        else:
            snapshot = book.snapshot(exit_ts)
            final = self.fill.exit(structure, snapshot, contracts_n)
            exit_debit = final.price if final.tradable else last_debit
            exit_commissions = final.commissions

        gross = (entry.price - exit_debit) * mult * contracts_n
        commissions = entry.commissions + exit_commissions
        max_loss = (structure.width - entry.price) * mult * contracts_n

        return Trade(
            day=day,
            structure=structure.kind,
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            strikes=structure.describe(),
            contracts=contracts_n,
            credit=entry.price,
            exit_debit=exit_debit,
            gross_pnl=gross,
            commissions=commissions,
            net_pnl=gross - commissions,
            exit_reason=exit_reason,
            max_loss=max_loss,
            entry_note=entry.reason,
        )

    def _settlement_price(self, book: QuoteBook, structure: Structure, ts, fallback: float) -> float:
        """Underlying at the close, re-derived from parity on the final book.

        SPXW settles against SET rather than the last quote, which this cannot
        observe. Parity on the closing book is the closest available proxy and
        is documented as such -- it is the largest single approximation in the
        harness, and on a strike that finishes near the money it is the
        difference between a full loss and a full win.
        """
        snapshot = book.snapshot(ts)
        pairs: dict[float, dict[str, float]] = {}
        for symbol, quote in snapshot.items():
            contract = Contract.parse(symbol)
            if contract and quote.is_tradable:
                pairs.setdefault(contract.strike, {})[contract.option_type] = quote.mid
        both = [(k, v[CALL], v[PUT]) for k, v in pairs.items() if CALL in v and PUT in v]
        if not both:
            return fallback
        strike, call, put = min(both, key=lambda p: abs(p[1] - p[2]))
        return strike + (call - put)

    # --- full period -----------------------------------------------------

    def run(self, start: date, end: date) -> list[SessionResult]:
        results = []
        day = start
        while day <= end:
            if day.weekday() < 5:  # holidays surface as empty definition files
                try:
                    results.append(self.run_day(day))
                except KeyError as exc:
                    results.append(SessionResult(day, skipped=str(exc)))
            day += timedelta(days=1)
        return results
