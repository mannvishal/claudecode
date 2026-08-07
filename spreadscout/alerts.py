"""Alert delivery and de-duplication.

This module is the only place in the package that makes an outbound POST, and
it only ever posts to a webhook URL you supply. The broker client in
``tradier.py`` remains read-only: nothing here can reach Tradier, and no code
path in this package can place, modify or cancel an order.

De-duplication is not a nicety. A watcher polling every 60 seconds will re-derive
the same conclusion every 60 seconds, and an alert stream that repeats itself is
one you stop reading -- which defeats the purpose on the day it matters. Each
alert carries a stable ``key``; repeats inside the cooldown are suppressed.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import requests

from .pricing import ET

log = logging.getLogger(__name__)

INFO = "INFO"
WARN = "WARN"
CRITICAL = "CRITICAL"

SEVERITY_RANK = {INFO: 0, WARN: 1, CRITICAL: 2}


@dataclass
class Alert:
    kind: str  # "entry" | "risk" | "guard" | "status"
    severity: str
    title: str
    lines: list[str] = field(default_factory=list)
    key: str = ""
    at: datetime = field(default_factory=lambda: datetime.now(ET))

    def __post_init__(self) -> None:
        if not self.key:
            self.key = f"{self.kind}:{self.title}"

    def as_text(self) -> str:
        stamp = self.at.strftime("%H:%M:%S")
        head = f"[{stamp}] {self.severity} {self.title}"
        return "\n".join([head, *(f"    {line}" for line in self.lines)])

    def as_dict(self) -> dict:
        return {
            "at": self.at.isoformat(),
            "kind": self.kind,
            "severity": self.severity,
            "title": self.title,
            "lines": self.lines,
            "key": self.key,
        }


class Notifier(Protocol):
    def send(self, alert: Alert) -> None: ...


class ConsoleNotifier:
    """Prints to stdout, and rings the terminal bell for anything critical."""

    def __init__(self, bell: bool = True):
        self.bell = bell

    def send(self, alert: Alert) -> None:
        if self.bell and alert.severity == CRITICAL:
            sys.stdout.write("\a")
        print(alert.as_text(), flush=True)


class FileNotifier:
    """Appends JSON lines, so an alert history survives the process."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(alert.as_dict()) + "\n")


class WebhookNotifier:
    """POSTs to a Slack or Discord incoming webhook.

    Both payload keys are sent because Slack reads ``text`` and Discord reads
    ``content``; each ignores the other, so one implementation serves both.
    A webhook failure is logged and swallowed -- an unreachable Slack must never
    take down the watcher that is monitoring your open positions.
    """

    def __init__(self, url: str, timeout: float = 10.0, min_severity: str = INFO):
        self.url = url
        self.timeout = timeout
        self.min_severity = min_severity

    def send(self, alert: Alert) -> None:
        if SEVERITY_RANK[alert.severity] < SEVERITY_RANK[self.min_severity]:
            return
        body = alert.as_text()
        try:
            requests.post(
                self.url,
                json={"text": body, "content": body},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            log.warning("webhook delivery failed (continuing): %s", exc)


class AlertRouter:
    """Fans alerts out to every notifier, suppressing repeats."""

    def __init__(self, notifiers: list[Notifier], cooldown_minutes: float = 15.0):
        self.notifiers = notifiers
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self._last: dict[str, datetime] = {}

    def _suppressed(self, alert: Alert) -> bool:
        previous = self._last.get(alert.key)
        if previous is None:
            return False
        # Critical alerts get a quarter of the cooldown: a short strike under
        # threat is worth repeating sooner than an entry opportunity is.
        window = self.cooldown / 4 if alert.severity == CRITICAL else self.cooldown
        return alert.at - previous < window

    def send(self, alert: Alert) -> bool:
        """Deliver unless suppressed. Returns whether it went out."""
        if self._suppressed(alert):
            return False
        self._last[alert.key] = alert.at
        for notifier in self.notifiers:
            try:
                notifier.send(alert)
            except Exception as exc:  # a broken sink must not stop the others
                log.warning("notifier %s failed: %s", type(notifier).__name__, exc)
        return True

    def reset(self, key: str) -> None:
        """Forget a key so its next occurrence alerts immediately."""
        self._last.pop(key, None)


def build_router(cfg) -> AlertRouter:
    """Assemble notifiers from config. Console is always on."""
    notifiers: list[Notifier] = [ConsoleNotifier(bell=cfg.alerts.terminal_bell)]
    if cfg.alerts.log_file:
        notifiers.append(FileNotifier(cfg.alerts.log_file))
    if cfg.alerts.webhook_url:
        notifiers.append(
            WebhookNotifier(cfg.alerts.webhook_url, min_severity=cfg.alerts.webhook_min_severity)
        )
    return AlertRouter(notifiers, cooldown_minutes=cfg.alerts.cooldown_minutes)
