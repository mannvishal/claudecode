"""Tests for the economic event calendar and its fail-closed semantics.

The property that matters most here is asymmetric: blocking a tradeable day
costs one skipped session, while failing to block puts a 0DTE position through
an FOMC statement. Every ambiguous case must therefore resolve to "block".
"""

from datetime import date, datetime, time, timedelta

import pytest

from spreadscout.config import Config
from spreadscout.events import (
    HIGH,
    LOW,
    MEDIUM,
    SEED_PATH,
    CalendarStatus,
    DerivedProvider,
    EconCalendar,
    EconEvent,
    FileProvider,
    check_event_gate,
    first_friday,
)
from spreadscout.pricing import ET

FOMC_DAY = date(2026, 9, 16)


def at(hour, minute=0, day=FOMC_DAY):
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=ET)


class StubProvider:
    def __init__(self, events, name="stub", stale=False):
        self._events = events
        self.name = name
        self._stale = stale

    def events(self, start, end):
        return [e for e in self._events if start <= e.day <= end]

    def stale_on(self, day):
        return self._stale


class ExplodingProvider:
    name = "exploding"

    def events(self, start, end):
        raise RuntimeError("upstream is down")

    def stale_on(self, day):
        return True


def calendar_with(*events, stale=False, providers=None):
    provs = providers if providers is not None else [StubProvider(list(events), stale=stale)]
    return EconCalendar(provs, ttl_minutes=0.0)


@pytest.fixture
def cfg():
    c = Config()
    c.gates.event_calendar_file = None
    return c


class TestFirstFriday:
    def test_when_the_first_is_a_friday(self):
        assert first_friday(2026, 5) == date(2026, 5, 1)

    def test_when_the_first_is_a_saturday(self):
        assert first_friday(2026, 8) == date(2026, 8, 7)

    def test_result_is_always_a_friday(self):
        for month in range(1, 13):
            assert first_friday(2026, month).weekday() == 4

    def test_result_is_always_in_the_first_week(self):
        for month in range(1, 13):
            assert first_friday(2026, month).day <= 7


class TestDerivedProvider:
    def test_emits_nfp_for_each_month_in_range(self):
        events = DerivedProvider().events(date(2026, 8, 1), date(2026, 10, 31))
        assert [e.day for e in events] == [
            date(2026, 8, 7), date(2026, 9, 4), date(2026, 10, 2)
        ]

    def test_nfp_is_high_impact_at_0830(self):
        event = DerivedProvider().events(date(2026, 8, 1), date(2026, 8, 31))[0]
        assert event.impact == HIGH and event.time_et == time(8, 30)

    def test_range_boundaries_are_respected(self):
        assert DerivedProvider().events(date(2026, 8, 8), date(2026, 8, 31)) == []

    def test_spans_a_year_boundary(self):
        events = DerivedProvider().events(date(2026, 12, 1), date(2027, 1, 31))
        assert len(events) == 2 and events[1].day.year == 2027

    def test_a_calendar_rule_never_goes_stale(self):
        assert not DerivedProvider().stale_on(date(2099, 1, 1))


class TestBundledSeed:
    def test_the_seed_file_ships_and_parses(self):
        assert SEED_PATH.exists()
        provider = FileProvider(SEED_PATH, name="seed")
        assert provider.events(date(2026, 1, 1), date(2026, 12, 31))

    def test_seed_declares_a_verification_horizon(self):
        assert FileProvider(SEED_PATH).verified_through is not None

    def test_seed_reports_itself_stale_past_its_horizon(self):
        """The whole point of the horizon: rot loudly, not silently."""
        provider = FileProvider(SEED_PATH)
        horizon = provider.verified_through
        assert not provider.stale_on(horizon)
        assert provider.stale_on(horizon + timedelta(days=1))

    def test_seeded_fomc_lands_at_1400_et(self):
        events = FileProvider(SEED_PATH).events(date(2026, 1, 1), date(2026, 12, 31))
        fomc = [e for e in events if "FOMC" in e.name]
        assert fomc and all(e.time_et == time(14, 0) for e in fomc)
        assert all(e.impact == HIGH for e in fomc)


class TestFileProvider:
    def test_reads_events_and_horizon(self, tmp_path):
        path = tmp_path / "e.yaml"
        path.write_text(
            "verified_through: 2027-06-30\n"
            "events:\n"
            "  - date: 2026-09-16\n"
            "    time: '14:00'\n"
            "    name: FOMC\n"
            "    impact: high\n"
        )
        provider = FileProvider(path)
        events = provider.events(date(2026, 1, 1), date(2027, 1, 1))
        assert len(events) == 1 and events[0].time_et == time(14, 0)
        assert provider.verified_through == date(2027, 6, 30)

    def test_missing_time_means_all_day(self, tmp_path):
        path = tmp_path / "e.yaml"
        path.write_text("verified_through: 2027-01-01\nevents:\n  - date: 2026-09-16\n    name: X\n")
        assert provider_event(path).blocks_all_day

    def test_a_file_without_a_horizon_is_stale(self, tmp_path):
        path = tmp_path / "e.yaml"
        path.write_text("events:\n  - date: 2026-09-16\n    name: X\n")
        assert FileProvider(path).stale_on(date(2026, 1, 1))

    def test_a_missing_file_yields_nothing_and_is_stale(self, tmp_path):
        provider = FileProvider(tmp_path / "nope.yaml")
        assert provider.events(date(2020, 1, 1), date(2030, 1, 1)) == []
        assert provider.stale_on(date(2026, 1, 1))


def provider_event(path):
    return FileProvider(path).events(date(2020, 1, 1), date(2030, 1, 1))[0]


class TestAggregation:
    def test_merges_and_sorts_across_providers(self):
        a = StubProvider([EconEvent(day=date(2026, 9, 16), name="B", time_et=time(14, 0))], "a")
        b = StubProvider([EconEvent(day=date(2026, 9, 16), name="A", time_et=time(8, 30))], "b")
        events = EconCalendar([a, b], ttl_minutes=0).status(
            date(2026, 9, 1), date(2026, 9, 30), now=at(11)
        ).events
        assert [e.name for e in events] == ["A", "B"]

    def test_a_failing_provider_is_recorded_not_raised(self):
        cal = EconCalendar([ExplodingProvider()], ttl_minutes=0)
        status = cal.status(date(2026, 9, 1), date(2026, 9, 30), now=at(11))
        assert "exploding" in status.stale_sources

    def test_a_failing_provider_does_not_hide_a_working_one(self):
        good = StubProvider([EconEvent(day=FOMC_DAY, name="FOMC", time_et=time(14, 0))])
        cal = EconCalendar([ExplodingProvider(), good], ttl_minutes=0)
        status = cal.status(date(2026, 9, 1), date(2026, 9, 30), now=at(11))
        assert len(status.events) == 1 and status.stale_sources

    def test_no_providers_means_unavailable(self):
        status = EconCalendar([], ttl_minutes=0).status(
            date(2026, 9, 1), date(2026, 9, 30), now=at(11)
        )
        assert not status.available

    def test_results_are_cached_within_the_ttl(self):
        provider = StubProvider([])
        calls = []
        original = provider.events
        provider.events = lambda s, e: (calls.append(1), original(s, e))[1]
        cal = EconCalendar([provider], ttl_minutes=60)
        cal.status(date(2026, 9, 1), date(2026, 9, 30), now=at(11))
        cal.status(date(2026, 9, 1), date(2026, 9, 30), now=at(11, 30))
        assert len(calls) == 1

    def test_today_filters_to_the_current_date(self):
        cal = calendar_with(
            EconEvent(day=FOMC_DAY, name="today", time_et=time(14, 0)),
            EconEvent(day=FOMC_DAY + timedelta(days=1), name="tomorrow", time_et=time(14, 0)),
        )
        assert [e.name for e in cal.today(now=at(11)).events] == ["today"]


class TestEventGate:
    def fomc(self):
        return EconEvent(day=FOMC_DAY, name="FOMC rate decision",
                         impact=HIGH, time_et=time(14, 0), source="test")

    def test_blocks_inside_the_window_before_the_release(self, cfg):
        cal = calendar_with(self.fomc())
        blocks, _ = check_event_gate(cal, cfg, now=at(12, 30))  # 90 min before
        assert blocks and "blackout window" in blocks[0]

    def test_blocks_just_after_the_release(self, cfg):
        cal = calendar_with(self.fomc())
        blocks, _ = check_event_gate(cal, cfg, now=at(15, 0))
        assert blocks and "just after" in blocks[0]

    def test_allows_well_before_the_window_but_warns(self, cfg):
        cal = calendar_with(self.fomc())
        blocks, notes = check_event_gate(cal, cfg, now=at(10, 0))
        assert not blocks
        assert notes and "entries blocked 12:00-15:30" in notes[0]

    def test_the_warning_quotes_the_whole_window_not_just_its_start(self, cfg):
        """An 08:30 release is clear by the 10:00 entry window; say so."""
        nfp = EconEvent(day=FOMC_DAY, name="NFP", impact=HIGH, time_et=time(8, 30))
        _blocks, notes = check_event_gate(calendar_with(nfp), cfg, now=at(6, 0))
        assert "entries blocked 06:30-10:00" in notes[0]

    def test_allows_well_after_the_window(self, cfg):
        cal = calendar_with(self.fomc())
        blocks, _ = check_event_gate(cal, cfg, now=at(15, 45))
        assert not blocks

    def test_unknown_release_time_blocks_the_whole_day(self, cfg):
        """Ambiguity resolves to blocking, never to trading."""
        cal = calendar_with(EconEvent(day=FOMC_DAY, name="Mystery", impact=HIGH))
        blocks, _ = check_event_gate(cal, cfg, now=at(10, 0))
        assert blocks and "whole" in blocks[0]

    def test_low_impact_events_are_ignored(self, cfg):
        cal = calendar_with(EconEvent(day=FOMC_DAY, name="Minor", impact=LOW,
                                      time_et=time(14, 0)))
        blocks, _ = check_event_gate(cal, cfg, now=at(13, 30))
        assert not blocks

    def test_threshold_is_configurable(self, cfg):
        cfg.gates.block_impact_at_or_above = MEDIUM
        cal = calendar_with(EconEvent(day=FOMC_DAY, name="Mid", impact=MEDIUM,
                                      time_et=time(14, 0)))
        blocks, _ = check_event_gate(cal, cfg, now=at(13, 30))
        assert blocks

    def test_window_widths_are_configurable(self, cfg):
        cfg.gates.event_minutes_before = 15
        cal = calendar_with(self.fomc())
        assert not check_event_gate(cal, cfg, now=at(13, 30))[0]
        assert check_event_gate(cal, cfg, now=at(13, 50))[0]

    def test_events_on_other_days_do_not_block(self, cfg):
        cal = calendar_with(EconEvent(day=FOMC_DAY + timedelta(days=1), name="FOMC",
                                      impact=HIGH, time_et=time(14, 0)))
        assert not check_event_gate(cal, cfg, now=at(13, 30))[0]


class TestFailClosed:
    def test_no_calendar_blocks_when_required(self, cfg):
        cfg.gates.require_event_calendar = True
        blocks, _ = check_event_gate(EconCalendar([], ttl_minutes=0), cfg, now=at(11))
        assert blocks and "cannot be confirmed" in blocks[0]

    def test_no_calendar_only_notes_when_not_required(self, cfg):
        cfg.gates.require_event_calendar = False
        blocks, notes = check_event_gate(EconCalendar([], ttl_minutes=0), cfg, now=at(11))
        assert not blocks and notes

    def test_stale_source_blocks_when_required(self, cfg):
        cal = calendar_with(stale=True)
        blocks, _ = check_event_gate(cal, cfg, now=at(11))
        assert blocks and "stale" in blocks[0]

    def test_stale_source_does_not_block_when_not_required(self, cfg):
        cfg.gates.require_event_calendar = False
        cal = calendar_with(stale=True)
        assert not check_event_gate(cal, cfg, now=at(11))[0]

    def test_event_blocking_can_be_switched_off_entirely(self, cfg):
        cfg.block_on_events = False
        cfg.gates.block_on_events = False
        cal = calendar_with(EconEvent(day=FOMC_DAY, name="FOMC", impact=HIGH,
                                      time_et=time(14, 0)))
        assert not check_event_gate(cal, cfg, now=at(13, 30))[0]

    def test_a_dead_provider_blocks_rather_than_reading_as_a_clear_day(self, cfg):
        cal = EconCalendar([ExplodingProvider()], ttl_minutes=0)
        blocks, _ = check_event_gate(cal, cfg, now=at(11))
        assert blocks


class TestCriticalVsSupplementary:
    """A vendor outage must not silently end your trading.

    Fail-closed is right when coverage genuinely lapses, and wrong when an
    optional feed we never depended on is briefly down -- the latter just
    teaches you to switch the gate off, which protects nobody.
    """

    def test_a_stale_supplementary_source_warns_but_does_not_block(self, cfg):
        seed = StubProvider([], name="seed")
        seed.critical = True
        flaky = StubProvider([], name="vendor", stale=True)
        flaky.critical = False
        blocks, notes = check_event_gate(
            calendar_with(providers=[seed, flaky]), cfg, now=at(11)
        )
        assert not blocks
        assert any("supplementary" in n for n in notes)

    def test_a_stale_critical_source_still_blocks(self, cfg):
        seed = StubProvider([], name="seed", stale=True)
        seed.critical = True
        blocks, _ = check_event_gate(calendar_with(providers=[seed]), cfg, now=at(11))
        assert blocks and "stale" in blocks[0]

    def test_exploding_supplementary_provider_does_not_block(self, cfg):
        seed = StubProvider([], name="seed")
        seed.critical = True
        boom = ExplodingProvider()
        boom.critical = False
        blocks, _ = check_event_gate(
            calendar_with(providers=[seed, boom]), cfg, now=at(11)
        )
        assert not blocks

    def test_file_providers_are_critical_by_default(self):
        assert FileProvider(SEED_PATH).critical

    def test_derived_and_fmp_are_supplementary(self):
        from spreadscout.events import FmpProvider

        assert not DerivedProvider().critical
        assert not FmpProvider("key").critical

    def test_unknown_providers_default_to_critical(self, cfg):
        """An unlabelled provider is assumed load-bearing -- the safe assumption."""
        class Unlabelled:
            name = "mystery"
            def events(self, start, end): return []
            def stale_on(self, day): return True

        blocks, _ = check_event_gate(
            calendar_with(providers=[Unlabelled()]), cfg, now=at(11)
        )
        assert blocks
