"""Alerts for the operator, by Telegram direct message.

A run collects what went wrong and sends one message at the end. Alerts never break a run: if Telegram is down, the
problem is still logged and the run carries on. Repeating problems (a program failing every hour, a low balance) are
sent at most once per interval, remembered in a small state file.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Protocol

import httpx

from laplace_engine.files import write_atomic

log = logging.getLogger(__name__)

MAX_MESSAGE = 4_000  # Telegram's limit is 4,096 characters


class Notifier(Protocol):
    def send(self, text: str) -> None: ...


@dataclass(frozen=True)
class Telegram:
    token: str = field(repr=False)
    chat: str

    @classmethod
    def from_env(cls) -> Telegram | None:
        token, chat = os.environ.get("LAPLACE_TG_TOKEN"), os.environ.get("LAPLACE_TG_CHAT")
        return cls(token, chat) if token and chat else None

    def send(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat, "text": text[:MAX_MESSAGE], "disable_web_page_preview": True}
        for attempt in range(3):
            try:
                response = httpx.post(url, json=payload, timeout=20.0)
                if response.status_code == httpx.codes.OK:
                    return
                log.warning("telegram answered HTTP %d", response.status_code)
            except httpx.HTTPError as exc:
                log.warning("telegram unreachable: %s", type(exc).__name__)  # never log the URL: it holds the token
            time.sleep(2.0 * (attempt + 1))


class Alerts:
    """Collects a run's problems and sends them as one message, keeping quiet about repeats inside their interval."""

    def __init__(self, notifier: Notifier | None, state: Path) -> None:
        self._notifier = notifier
        self._state = state
        self._lines: list[str] = []
        self._keys: list[str] = []

    def _sent(self) -> dict[str, float]:
        try:
            sent: dict[str, float] = json.loads(self._state.read_bytes())
            return sent
        except (OSError, ValueError):
            return {}

    def add(self, line: str, *, key: str | None = None, every: timedelta = timedelta(0)) -> None:
        log.warning("alert: %s", line)
        if key is not None and time.time() - self._sent().get(key, 0.0) < every.total_seconds():
            return
        self._lines.append(line)
        if key is not None:
            self._keys.append(key)

    @property
    def pending(self) -> bool:
        return bool(self._lines)

    def flush(self, title: str) -> bool:
        """Send everything collected under ``title``; returns whether a message went out."""
        if not self._lines:
            return False
        if self._notifier is not None:
            self._notifier.send("\n".join([title, *(f"• {line}" for line in self._lines)]))
        sent = self._sent()
        now = time.time()
        sent |= {key: now for key in self._keys}
        write_atomic(self._state, json.dumps(sent, sort_keys=True).encode())
        self._lines, self._keys = [], []
        return self._notifier is not None
