import json
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from eth_account import Account
from eth_hash.auto import keccak

from laplace_engine import candles, chain, cli, notify, publish, release, settle
from laplace_engine.baselines import NAIVE
from laplace_engine.chain import RPC
from laplace_engine.release import Pool, Program, Release
from laplace_research.evolve.panel import Panel

from .conftest import ISSUE
from .fakes import FakeNode, FakeStore


def test_issue_then_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_panel: Callable[..., Panel],
    capsys: pytest.CaptureFixture[str],
) -> None:
    rel = Release(
        "t1",
        "test",
        (Program("naive", NAIVE, True, "baseline"), Program("double", NAIVE + "\n", False, "test#2")),
        (Pool("demon", (("naive", 500_000), ("double", 500_000))),),
    )
    release.write(rel, tmp_path / "release", tmp_path / "private")
    monkeypatch.setattr(candles, "wait_for", lambda issue, cache_dir, deadline: make_panel(end_issue=issue))
    common = ["--release", str(tmp_path / "release"), "--private", str(tmp_path / "private")]
    data = ["--data", str(tmp_path / "data"), "--issue", str(ISSUE)]

    assert cli.main(["issue", *common, *data, "--workers", "2"]) == 0
    assert "63 forecasts" in capsys.readouterr().out
    assert (tmp_path / "data" / "issues" / str(ISSUE) / "manifest.json").exists()
    assert cli.main(["issue", *common, *data]) == 0  # sealed already: nothing is recomputed
    assert cli.main(["verify", *data, "--offline"]) == 0
    assert "63 forecasts and 1 pool(s) check out" in capsys.readouterr().out


def test_commit_and_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_panel: Callable[..., Panel],
    capsys: pytest.CaptureFixture[str],
) -> None:
    account = Account.create()
    node = FakeNode(account.address, head_timestamp=int(time.time()))
    store = FakeStore()
    rel = Release("r9", "test", (Program("naive", NAIVE, True, "baseline"),), ())
    release.write(rel, tmp_path / "r9", tmp_path / "private")
    number = candles.current_issue()
    monkeypatch.setattr(candles, "wait_for", lambda issue, cache_dir, deadline: make_panel(end_issue=issue))
    monkeypatch.setattr(chain, "COMMIT_WINDOW_S", 3600)  # the test may run at any minute of the hour
    monkeypatch.setattr(chain, "load_account", lambda path: account)
    monkeypatch.setattr(chain, "RPC", lambda url: RPC(url, client=httpx.Client(transport=node.transport())))
    monkeypatch.setattr(publish.R2, "from_env", classmethod(lambda cls: store))
    monkeypatch.setenv("LAPLACE_COMMIT_KEY_FILE", str(tmp_path / "unused.json"))
    argv = ["issue", "--release", str(tmp_path / "r9"), "--private", str(tmp_path / "private")]
    argv += ["--data", str(tmp_path / "data"), "--workers", "1", "--commit", "--publish"]

    assert cli.main(argv) == 0
    assert len(node.sent) == 1
    assert "committed in 0x" in capsys.readouterr().out
    assert json.loads(store.objects["v1/latest.json"][0])["issue"] == number
    assert f"v1/issues/{number}/commitment.json" in store.objects
    assert "v1/releases/r9/programs/naive.py" in store.objects
    assert cli.main(argv) == 0  # a rerun neither seals, commits nor publishes again
    assert len(node.sent) == 1

    assert cli.main(["status", "--data", str(tmp_path / "data"), "--hours", "2"]) == 0
    assert capsys.readouterr().out.splitlines()[-1].endswith("sealed committed published")


def test_a_closed_window_leaves_the_issue_uncommitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_panel: Callable[..., Panel]
) -> None:
    rel = Release("r9", "test", (Program("naive", NAIVE, True, "baseline"),), ())
    release.write(rel, tmp_path / "r9", tmp_path / "private")
    monkeypatch.setattr(candles, "wait_for", lambda issue, cache_dir, deadline: make_panel(end_issue=issue))
    argv = ["issue", "--release", str(tmp_path / "r9"), "--private", str(tmp_path / "private")]
    argv += ["--data", str(tmp_path / "data"), "--issue", str(ISSUE), "--workers", "1", "--commit"]
    assert cli.main(argv) == cli.SOFT_FAILURE
    assert not (tmp_path / "data" / "issues" / str(ISSUE) / "commitment.json").exists()


def test_reconcile_recovers_an_unrecorded_commitment_and_marks_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_panel: Callable[..., Panel]
) -> None:
    account = Account.create()
    number = candles.current_issue()
    node = FakeNode(account.address, head_timestamp=(number - 2) * 3600 + 100)
    rpc = RPC("https://rpc.test", client=httpx.Client(transport=node.transport()))
    rel = Release("r9", "test", (Program("naive", NAIVE, True, "baseline"),), ())
    release.write(rel, tmp_path / "r9", tmp_path / "private")
    monkeypatch.setattr(candles, "wait_for", lambda issue, cache_dir, deadline: make_panel(end_issue=issue))
    base = ["issue", "--release", str(tmp_path / "r9"), "--private", str(tmp_path / "private")]
    base += ["--data", str(tmp_path / "data"), "--workers", "1"]
    for past in (number - 2, number - 1):  # two past hours, sealed in dry runs
        assert cli.main([*base, "--issue", str(past)]) == 0
    orphan = tmp_path / "data" / "issues" / str(number - 2)
    manifest = (orphan / "manifest.json").read_bytes()
    body = json.loads(manifest)
    # The first one reached the chain, but its run died before recording it.
    chain.Ledger(rpc).commit(
        account,
        number - 2,
        bytes.fromhex(body["root"][2:]),
        bytes(keccak(manifest)),
        body["leaves"],
        journal=orphan / cli.JOURNAL,
    )
    store = FakeStore()
    monkeypatch.setattr(chain, "COMMIT_WINDOW_S", 3600)
    monkeypatch.setattr(chain, "load_account", lambda path: account)
    monkeypatch.setattr(chain, "RPC", lambda url: rpc)
    monkeypatch.setattr(publish.R2, "from_env", classmethod(lambda cls: store))
    monkeypatch.setenv("LAPLACE_COMMIT_KEY_FILE", str(tmp_path / "unused.json"))
    node.advance(number * 3600 + 60)  # the current issue's commit lands inside its own window

    inbox = Inbox()
    monkeypatch.setattr(notify.Telegram, "from_env", classmethod(lambda cls: inbox))
    assert cli.main([*base, "--commit", "--publish"]) == 0
    assert json.loads((orphan / "commitment.json").read_bytes())["root"] == body["root"]
    (message,) = inbox.messages  # one message for the whole run
    assert f"issue {number - 2}: recovered a commitment" in message
    assert f"issue {number - 1} was never committed" in message
    assert f"v1/issues/{number - 2}/forecasts.json" in store.objects  # recovered, so published
    gap = tmp_path / "data" / "issues" / str(number - 1)
    assert (gap / cli.GAP).exists()  # never committed: a visible gap, never published
    assert not any(k.startswith(f"v1/issues/{number - 1}/") for k in store.objects)
    assert len(node.sent) == 2  # the orphan's original transaction and this hour's; nothing was sent twice


def test_only_one_run_at_a_time(tmp_path: Path) -> None:
    with cli.run_lock(tmp_path / ".lock") as first:
        assert first
        with cli.run_lock(tmp_path / ".lock") as second:
            assert not second
    with cli.run_lock(tmp_path / ".lock") as again:
        assert again


def test_scoreboard_reads_before_any_issue_has_settled_at_every_horizon(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    issues = tmp_path / "data" / "issues" / "100"
    issues.mkdir(parents=True)
    scores = {"demon": {"BTC": -1.0}, "naive": {"BTC": -1.5}}
    s = settle.Settlement(100, 1, "issue 101", {"BTC": 1.0}, {"BTC": 1.0}, {"BTC": 5}, scores, "r1", {})
    (issues / settle.settlement_name(1)).write_bytes(s.to_json())
    data = ["--data", str(tmp_path / "data")]
    assert cli.main(["scoreboard", *data]) == 0
    demon = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("demon"))
    assert demon.split()[-3:] == ["-", "-", "-"]
    assert cli.main(["scoreboard", *data, "--horizon", "1"]) == 0
    demon = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("demon"))
    assert demon.split()[-3] == "+0.5000"


class Inbox:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, text: str) -> None:
        self.messages.append(text)


def test_crash_alerts_skip_runs_that_already_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    inbox = Inbox()
    monkeypatch.setattr(notify.Telegram, "from_env", classmethod(lambda cls: inbox))
    monkeypatch.setenv("MONITOR_EXIT_STATUS", str(cli.SOFT_FAILURE))
    assert cli.main(["alert"]) == 0
    assert inbox.messages == []
    monkeypatch.setenv("MONITOR_EXIT_STATUS", "1")
    monkeypatch.setenv("MONITOR_SERVICE_RESULT", "exit-code")
    assert cli.main(["alert"]) == 0
    assert "crashed (result: exit-code, exit status: 1)" in inbox.messages[0]


def test_digest_counts_the_last_day_and_reports_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    number = candles.current_issue()
    for n in (number - 1, number):
        d = tmp_path / "data" / "issues" / str(n)
        d.mkdir(parents=True)
        (d / "commitment.json").write_text("{}")
        (d / "published").touch()
    scores = {"demon": {"BTC": -1.0}, "garch": {"BTC": -1.25}}
    s = settle.Settlement(number - 1, 1, "issue x", {"BTC": 1.0}, {"BTC": 1.0}, {"BTC": 5}, scores, "r1", {})
    (tmp_path / "data" / "issues" / str(number - 1) / settle.settlement_name(1)).write_bytes(s.to_json())
    monkeypatch.delenv("LAPLACE_COMMIT_KEY_FILE", raising=False)
    assert cli.main(["digest", "--data", str(tmp_path / "data"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Record: 2/2 hours committed, 2 published" in out  # the record began two hours ago: no gaps
    assert "Demon vs GARCH: 1h +0.250 (n=1)" in out
    assert "Directions only" in out


def test_the_judging_rule_rides_in_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_panel: Callable[..., Panel]
) -> None:
    rel = Release("r9", "test", (Program("naive", NAIVE, True, "baseline"),), ())
    release.write(rel, tmp_path / "r9", tmp_path / "private")
    rule = tmp_path / "rule.md"
    rule.write_text("# a rule\n")
    monkeypatch.setattr(candles, "wait_for", lambda issue, cache_dir, deadline: make_panel(end_issue=issue))
    inbox = Inbox()
    monkeypatch.setattr(notify.Telegram, "from_env", classmethod(lambda cls: inbox))
    base = ["issue", "--release", str(tmp_path / "r9"), "--private", str(tmp_path / "private")]
    base += ["--data", str(tmp_path / "data"), "--workers", "1", "--judging", str(rule)]
    assert cli.main([*base, "--issue", str(ISSUE)]) == 0
    manifest = json.loads((tmp_path / "data" / "issues" / str(ISSUE) / "manifest.json").read_bytes())
    assert manifest["judging"] == [{"path": "rule.md", "keccak256": "0x" + bytes(keccak(b"# a rule\n")).hex()}]
    assert inbox.messages == []
    rule.unlink()  # a missing rule costs nothing but an alert
    assert cli.main([*base, "--issue", str(ISSUE + 1)]) == 0
    assert json.loads((tmp_path / "data" / "issues" / str(ISSUE + 1) / "manifest.json").read_bytes())["judging"] == []
    assert "judging rule" in inbox.messages[0]


def test_the_repository_rule_for_r1_exists() -> None:
    rule = cli.REPO / "docs" / "judging" / "r1.md"
    assert rule.is_file()
    assert "0x97bc9657cb5b4af73b46e0031ac385cab039e72fb9fb8edf591bd457f149914e" in rule.read_text()
