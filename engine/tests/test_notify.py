import json
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from laplace_engine import notify


class Inbox:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, text: str) -> None:
        self.messages.append(text)


def test_problems_go_out_as_one_message(tmp_path: Path) -> None:
    inbox = Inbox()
    alerts = notify.Alerts(inbox, tmp_path / "alerts.json")
    assert not alerts.flush("title")  # nothing to say: no message
    alerts.add("issue 1 not committed")
    alerts.add("tails produced no forecast", key="forecaster:tails", every=timedelta(hours=6))
    assert alerts.flush("⚠️ Laplace · issue 1")
    assert inbox.messages == ["⚠️ Laplace · issue 1\n• issue 1 not committed\n• tails produced no forecast"]


def test_repeats_stay_quiet_inside_their_interval(tmp_path: Path) -> None:
    inbox = Inbox()
    for _ in range(3):
        alerts = notify.Alerts(inbox, tmp_path / "alerts.json")  # a new run each time, same state file
        alerts.add("low balance", key="balance", every=timedelta(days=1))
        alerts.flush("t")
    assert len(inbox.messages) == 1
    state = json.loads((tmp_path / "alerts.json").read_text())
    state["balance"] -= 2 * 86_400  # two days later
    (tmp_path / "alerts.json").write_text(json.dumps(state))
    alerts = notify.Alerts(inbox, tmp_path / "alerts.json")
    alerts.add("low balance", key="balance", every=timedelta(days=1))
    alerts.flush("t")
    assert len(inbox.messages) == 2


def test_telegram_failures_never_break_a_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def post(url: str, **kwargs: object) -> httpx.Response:
        calls.append(kwargs)
        raise httpx.ConnectError("down")

    monkeypatch.setattr(notify.httpx, "post", post)
    monkeypatch.setattr(notify.time, "sleep", lambda s: None)
    notify.Telegram("secret-token", "42").send("hello")
    assert len(calls) == 3
    assert calls[0]["json"] == {"chat_id": "42", "text": "hello", "disable_web_page_preview": True}
    assert "secret-token" not in repr(notify.Telegram("secret-token", "42"))


def test_no_credentials_means_no_notifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LAPLACE_TG_TOKEN", raising=False)
    monkeypatch.setenv("LAPLACE_TG_CHAT", "42")
    assert notify.Telegram.from_env() is None
