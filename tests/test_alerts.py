"""Tests for alert routing, de-duplication and sink isolation."""

import json
from datetime import datetime, timedelta

import pytest

from spreadscout.alerts import (
    CRITICAL,
    INFO,
    WARN,
    Alert,
    AlertRouter,
    ConsoleNotifier,
    FileNotifier,
    build_router,
)
from spreadscout.config import Config
from spreadscout.pricing import ET

T0 = datetime(2026, 8, 7, 11, 0, tzinfo=ET)


class Recorder:
    def __init__(self):
        self.sent = []

    def send(self, alert):
        self.sent.append(alert)


class Broken:
    def send(self, alert):
        raise RuntimeError("sink is down")


def alert(severity=INFO, key="k", at=T0, kind="entry"):
    return Alert(kind=kind, severity=severity, title="t", lines=["l"], key=key, at=at)


class TestAlert:
    def test_key_defaults_to_kind_and_title(self):
        a = Alert(kind="risk", severity=WARN, title="something")
        assert a.key == "risk:something"

    def test_text_includes_title_and_lines(self):
        text = Alert(kind="risk", severity=WARN, title="T", lines=["a", "b"]).as_text()
        assert "T" in text and "a" in text and "b" in text

    def test_dict_round_trips_through_json(self):
        payload = json.dumps(alert().as_dict())
        assert json.loads(payload)["kind"] == "entry"


class TestDeduplication:
    def test_first_alert_is_delivered(self):
        rec = Recorder()
        assert AlertRouter([rec], cooldown_minutes=15).send(alert())
        assert len(rec.sent) == 1

    def test_repeat_inside_cooldown_is_suppressed(self):
        rec = Recorder()
        router = AlertRouter([rec], cooldown_minutes=15)
        router.send(alert(at=T0))
        assert not router.send(alert(at=T0 + timedelta(minutes=5)))
        assert len(rec.sent) == 1

    def test_repeat_after_cooldown_is_delivered(self):
        rec = Recorder()
        router = AlertRouter([rec], cooldown_minutes=15)
        router.send(alert(at=T0))
        assert router.send(alert(at=T0 + timedelta(minutes=16)))

    def test_different_keys_do_not_suppress_each_other(self):
        rec = Recorder()
        router = AlertRouter([rec], cooldown_minutes=15)
        router.send(alert(key="a"))
        router.send(alert(key="b"))
        assert len(rec.sent) == 2

    def test_critical_alerts_repeat_sooner(self):
        """A threatened short strike is worth repeating before an entry idea is."""
        rec = Recorder()
        router = AlertRouter([rec], cooldown_minutes=20)
        router.send(alert(severity=CRITICAL, at=T0))
        # 6 minutes is inside the 20-minute window but past the 5-minute quarter.
        assert router.send(alert(severity=CRITICAL, at=T0 + timedelta(minutes=6)))

    def test_info_alert_would_still_be_suppressed_at_that_point(self):
        rec = Recorder()
        router = AlertRouter([rec], cooldown_minutes=20)
        router.send(alert(severity=INFO, at=T0))
        assert not router.send(alert(severity=INFO, at=T0 + timedelta(minutes=6)))

    def test_reset_clears_suppression(self):
        rec = Recorder()
        router = AlertRouter([rec], cooldown_minutes=15)
        router.send(alert(key="a"))
        router.reset("a")
        assert router.send(alert(key="a", at=T0 + timedelta(seconds=1)))


class TestSinkIsolation:
    def test_a_broken_sink_does_not_stop_the_others(self):
        """An unreachable webhook must never silence the console."""
        rec = Recorder()
        AlertRouter([Broken(), rec]).send(alert())
        assert len(rec.sent) == 1

    def test_file_notifier_appends_json_lines(self, tmp_path):
        path = tmp_path / "nested" / "alerts.jsonl"
        notifier = FileNotifier(path)
        notifier.send(alert(key="a"))
        notifier.send(alert(key="b"))
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["key"] for r in rows] == ["a", "b"]

    def test_console_notifier_prints(self, capsys):
        ConsoleNotifier(bell=False).send(alert())
        assert "t" in capsys.readouterr().out


class TestRouterConstruction:
    def test_console_is_always_present(self):
        cfg = Config()
        cfg.alerts.log_file = None
        cfg.alerts.webhook_url = None
        assert len(build_router(cfg).notifiers) == 1

    def test_log_file_adds_a_sink(self, tmp_path):
        cfg = Config()
        cfg.alerts.log_file = str(tmp_path / "a.jsonl")
        cfg.alerts.webhook_url = None
        assert len(build_router(cfg).notifiers) == 2

    def test_webhook_adds_a_sink(self, tmp_path):
        cfg = Config()
        cfg.alerts.log_file = str(tmp_path / "a.jsonl")
        cfg.alerts.webhook_url = "https://hooks.example.com/x"
        assert len(build_router(cfg).notifiers) == 3

    def test_cooldown_comes_from_config(self):
        cfg = Config()
        cfg.alerts.cooldown_minutes = 42
        assert build_router(cfg).cooldown == timedelta(minutes=42)
