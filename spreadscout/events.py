"""Scheduled economic events, and refusing to trade into them.

For a 0DTE SPX seller the dangerous release is not the one at 08:30 -- by the
time the 10:00 entry window opens that has already printed and its volatility
has been crushed out of the chain. The dangerous one is **FOMC at 14:00 ET**,
which lands mid-session while your position is open and can move the index
several standard deviations in minutes. A condor entered at 13:50 on FOMC day is
not a premium-selling trade, it is a coin flip with the odds paid to someone else.

Three design decisions follow from that, and they are the whole module:

**Over-blocking is nearly free; under-blocking is not.** A false positive costs
one skipped session. A false negative puts you in a 0DTE position through a rate
decision. Every heuristic here is therefore tuned to block when unsure -- an
event whose release time is unknown is treated as blocking the entire day.

**The calendar fails closed.** If no provider can say whether today carries a
high-impact release, the gate blocks rather than assuming a quiet day. Silence
from a data source is not evidence of an empty calendar.

**Hardcoded dates announce their own expiry.** The bundled seed file carries a
``verified_through`` date. Past it, the provider reports itself stale rather than
cheerfully implying that next year has no FOMC meetings. A safety list that rots
silently is worse than no list, because you stop checking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Protocol

import yaml

from .pricing import ET

log = logging.getLogger(__name__)

HIGH, MEDIUM, LOW = "high", "medium", "low"
IMPACT_RANK = {LOW: 0, MEDIUM: 1, HIGH: 2}

SEED_PATH = Path(__file__).parent / "data" / "econ_events.yaml"


@dataclass(frozen=True)
class EconEvent:
    """A scheduled release."""

    day: date
    name: str
    impact: str = HIGH
    time_et: time | None = None
    source: str = "unknown"

    @property
    def blocks_all_day(self) -> bool:
        """No known release time means we cannot time a blackout window.

        Treating it as an all-day block is the fail-safe reading: the cost is a
        skipped session, and the alternative is guessing.
        """
        return self.time_et is None

    def at(self, day: date | None = None) -> datetime | None:
        if self.time_et is None:
            return None
        return datetime.combine(day or self.day, self.time_et, tzinfo=ET)

    def describe(self) -> str:
        when = self.time_et.strftime("%H:%M ET") if self.time_et else "time unknown"
        return f"{self.day} {when} -- {self.name} ({self.impact} impact, via {self.source})"


def first_friday(year: int, month: int) -> date:
    """First Friday of a month. ``date.weekday()`` puts Friday at 4."""
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7)


class CalendarProvider(Protocol):
    name: str
    # A critical provider is one the calendar's coverage actually depends on --
    # in practice the seed and user files, which are the only source of FOMC
    # dates. When one of those goes stale, the calendar genuinely cannot say
    # whether today is clear, and the gate blocks.
    #
    # Supplementary providers are different. An optional third-party feed being
    # unreachable should warn, not block: fail-closed on a source you never
    # depended on means a vendor outage silently ends your trading, which is the
    # fastest route to someone disabling the safety gate altogether. A gate that
    # gets switched off protects nobody.
    critical: bool

    def events(self, start: date, end: date) -> list[EconEvent]: ...

    def stale_on(self, day: date) -> bool: ...


class DerivedProvider:
    """Events derivable from a calendar rule, with no data source required.

    Only the Employment Situation report qualifies. BLS releases it on the first
    Friday of the month at 08:30 ET in the large majority of months, but not all
    -- so it is emitted as a heuristic. Because 08:30 is before the entry window
    opens, this mostly serves to flag the day as one where realized volatility
    is likely to exceed what the trailing window suggests.

    CPI is deliberately *not* derived. It lands somewhere around the 10th to the
    15th with no rule tight enough to be worth guessing at, and a wrong guess
    here would mean blocking an arbitrary quiet day while missing the real one.
    Put CPI dates in the seed or your own file.
    """

    name = "derived"
    critical = False  # a calendar rule cannot go stale, so this never blocks

    def events(self, start: date, end: date) -> list[EconEvent]:
        out: list[EconEvent] = []
        cursor = date(start.year, start.month, 1)
        while cursor <= end:
            nfp = first_friday(cursor.year, cursor.month)
            if start <= nfp <= end:
                out.append(EconEvent(
                    day=nfp,
                    name="Employment Situation (NFP, derived: first Friday)",
                    impact=HIGH,
                    time_et=time(8, 30),
                    source=self.name,
                ))
            cursor = date(cursor.year + (cursor.month == 12), (cursor.month % 12) + 1, 1)
        return out

    def stale_on(self, day: date) -> bool:
        return False  # a calendar rule cannot go stale


class FileProvider:
    """Events from a YAML file, with a self-declared verification horizon.

    Schema::

        verified_through: 2026-12-31
        events:
          - date: 2026-09-16
            time: "14:00"
            name: FOMC rate decision
            impact: high

    ``verified_through`` is what stops a hardcoded list from quietly rotting.
    Past that date the provider reports itself stale, and a stale provider blocks
    entry when ``require_event_calendar`` is on.
    """

    critical = True  # the only source of FOMC dates; staleness here is real

    def __init__(self, path: str | Path, name: str = "file"):
        self.path = Path(path)
        self.name = name
        self._verified_through: date | None = None
        self._events: list[EconEvent] | None = None

    def _load(self) -> list[EconEvent]:
        if self._events is not None:
            return self._events
        if not self.path.exists():
            log.warning("event calendar %s does not exist", self.path)
            self._events = []
            return self._events

        raw = yaml.safe_load(self.path.read_text()) or {}
        through = raw.get("verified_through")
        if isinstance(through, date):
            self._verified_through = through
        elif isinstance(through, str):
            self._verified_through = date.fromisoformat(through)

        events = []
        for row in raw.get("events") or []:
            day = row["date"]
            if isinstance(day, str):
                day = date.fromisoformat(day)
            raw_time = row.get("time")
            parsed_time = None
            if raw_time:
                parsed_time = (
                    raw_time if isinstance(raw_time, time)
                    else time.fromisoformat(str(raw_time))
                )
            events.append(EconEvent(
                day=day,
                name=row.get("name", "unnamed event"),
                impact=str(row.get("impact", HIGH)).lower(),
                time_et=parsed_time,
                source=self.name,
            ))
        self._events = events
        return events

    def events(self, start: date, end: date) -> list[EconEvent]:
        return [e for e in self._load() if start <= e.day <= end]

    def stale_on(self, day: date) -> bool:
        self._load()
        return self._verified_through is None or day > self._verified_through

    @property
    def verified_through(self) -> date | None:
        self._load()
        return self._verified_through


class FmpProvider:
    """Financial Modeling Prep's economic calendar.

    NOTE: this adapter is written to FMP's documented response shape but has not
    been exercised against the live endpoint, because the economics calendar sits
    behind FMP's Starter tier and no key with that entitlement was available.
    Treat it as unverified until you have run ``spreadscout calendar`` with a key
    and confirmed the output looks right. Any failure degrades to the other
    providers rather than raising, so an unverified adapter cannot take the
    watcher down -- but it also cannot be relied on until you have checked it.
    """

    name = "fmp"
    critical = False  # supplementary; an outage warns rather than halting you
    BASE = "https://financialmodelingprep.com/stable/economic-calendar"

    def __init__(self, api_key: str, country: str = "US", timeout: float = 15.0):
        self.api_key = api_key
        self.country = country
        self.timeout = timeout
        self._failed = False

    def events(self, start: date, end: date) -> list[EconEvent]:
        import requests

        try:
            resp = requests.get(
                self.BASE,
                params={"from": start.isoformat(), "to": end.isoformat(), "apikey": self.api_key},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            rows = resp.json() or []
        except Exception as exc:
            self._failed = True
            log.warning("FMP economic calendar unavailable: %s", exc)
            return []

        out: list[EconEvent] = []
        for row in rows:
            if self.country and str(row.get("country", "")).upper() not in (self.country, ""):
                continue
            stamp = str(row.get("date") or "")
            try:
                when = datetime.fromisoformat(stamp.replace("Z", ""))
            except ValueError:
                continue
            out.append(EconEvent(
                day=when.date(),
                name=str(row.get("event", "unnamed")),
                impact=str(row.get("impact", MEDIUM)).lower(),
                time_et=when.time() if when.time() != time(0, 0) else None,
                source=self.name,
            ))
        self._failed = False
        return out

    def stale_on(self, day: date) -> bool:
        return self._failed


@dataclass
class CalendarStatus:
    events: list[EconEvent]
    stale_sources: list[str]
    available: bool
    # Sources whose staleness genuinely undermines coverage, as opposed to a
    # supplementary feed being briefly unreachable. See ``CalendarProvider``.
    stale_critical: list[str] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        return self.available and not self.stale_critical


class EconCalendar:
    """Aggregates providers, with a TTL cache so a 60s poll is not a 60s fetch."""

    def __init__(self, providers: list[CalendarProvider], ttl_minutes: float = 360.0):
        self.providers = providers
        self.ttl = timedelta(minutes=ttl_minutes)
        self._cache: tuple[date, date, datetime, CalendarStatus] | None = None

    def status(self, start: date, end: date, now: datetime | None = None) -> CalendarStatus:
        now = now or datetime.now(ET)
        if self._cache:
            c_start, c_end, fetched, cached = self._cache
            if c_start <= start and end <= c_end and now - fetched < self.ttl:
                return cached

        events: list[EconEvent] = []
        stale: list[str] = []
        stale_critical: list[str] = []
        available = False
        for provider in self.providers:
            is_critical = getattr(provider, "critical", True)
            try:
                found = provider.events(start, end)
                events.extend(found)
                available = True
                degraded = provider.stale_on(end)
            except Exception as exc:
                log.warning("event provider %s failed: %s", provider.name, exc)
                degraded = True
            if degraded:
                stale.append(provider.name)
                if is_critical:
                    stale_critical.append(provider.name)

        events.sort(key=lambda e: (e.day, e.time_et or time(0, 0)))
        status = CalendarStatus(events=events, stale_sources=stale, available=available,
                                stale_critical=stale_critical)
        self._cache = (start, end, now, status)
        return status

    def today(self, now: datetime | None = None) -> CalendarStatus:
        now = now or datetime.now(ET)
        day = now.date()
        # Fetch a window so the cache serves several days of polling at once.
        full = self.status(day - timedelta(days=1), day + timedelta(days=30), now=now)
        return CalendarStatus(
            events=[e for e in full.events if e.day == day],
            stale_sources=full.stale_sources,
            available=full.available,
            stale_critical=full.stale_critical,
        )


def build_calendar(cfg) -> EconCalendar:
    """Assemble providers from config, most trustworthy first."""
    import os

    providers: list[CalendarProvider] = []
    if cfg.gates.use_derived_events:
        providers.append(DerivedProvider())
    if cfg.gates.event_calendar_file:
        providers.append(FileProvider(cfg.gates.event_calendar_file, name="user-file"))
    if SEED_PATH.exists():
        providers.append(FileProvider(SEED_PATH, name="bundled-seed"))
    key = os.environ.get("FMP_API_KEY")
    if key:
        providers.append(FmpProvider(key))
    return EconCalendar(providers, ttl_minutes=cfg.gates.event_cache_minutes)


def check_event_gate(
    calendar: EconCalendar, cfg, now: datetime | None = None
) -> tuple[list[str], list[str]]:
    """Returns ``(blocks, notes)`` for today's scheduled events."""
    now = now or datetime.now(ET)
    g = cfg.gates
    blocks: list[str] = []
    notes: list[str] = []

    if not g.block_on_events:
        return blocks, notes

    status = calendar.today(now=now)

    if not status.available:
        if g.require_event_calendar:
            blocks.append(
                "no economic calendar source is available, so it cannot be confirmed that "
                "today is clear of high-impact releases -- blocking rather than assuming a "
                "quiet day (set gates.require_event_calendar: false to trade anyway)"
            )
        else:
            notes.append("no economic calendar available; event blocking is inactive")
        return blocks, notes

    if status.stale_critical and g.require_event_calendar:
        blocks.append(
            f"the economic calendar is stale ({', '.join(sorted(set(status.stale_critical)))}) "
            f"-- refresh it or extend verified_through in your event file. A hardcoded "
            f"calendar that has passed its horizon reports no events, which is "
            f"indistinguishable from a genuinely empty day"
        )
    elif status.stale_sources:
        supplementary = sorted(set(status.stale_sources) - set(status.stale_critical))
        if supplementary:
            notes.append(
                f"supplementary calendar source unavailable ({', '.join(supplementary)}); "
                f"coverage falls back to the seed and derived rules"
            )

    threshold = IMPACT_RANK.get(g.block_impact_at_or_above, 2)
    for event in status.events:
        if IMPACT_RANK.get(event.impact, 0) < threshold:
            continue

        if event.blocks_all_day:
            blocks.append(
                f"{event.name} is scheduled today with no known release time, so the whole "
                f"session is treated as blocked"
            )
            continue

        moment = event.at(now.date())
        opens = moment - timedelta(minutes=g.event_minutes_before)
        closes = moment + timedelta(minutes=g.event_minutes_after)
        if opens <= now <= closes:
            side = "ahead of" if now < moment else "just after"
            blocks.append(
                f"{now:%H:%M} ET is inside the blackout window {side} {event.name} at "
                f"{event.time_et:%H:%M} ET (-{g.event_minutes_before}/+{g.event_minutes_after} min)"
            )
        elif now < opens:
            # Quote the whole window rather than just its start. For an 08:30
            # release the blackout is already over by the time the entry window
            # opens at 10:00, and saying only "entries close at 06:30" reads as
            # though the day were lost when it is not.
            notes.append(
                f"{event.name} at {event.time_et:%H:%M} ET today; "
                f"entries blocked {opens:%H:%M}-{closes:%H:%M}"
            )

    return blocks, notes
