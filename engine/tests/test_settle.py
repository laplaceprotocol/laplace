import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from laplace_engine.settle import BLOCK_ISSUES, Settlement, load_settlements, scoreboard, settle, settle_due

ASSETS = ["BTC", "ETH"]
BPS = [100, 400, 900, 1_600, 2_000, 2_000, 1_600, 900, 400, 100]
IDS = {"demon": "0x" + "d" * 64, "naive": "0x" + "a" * 64}


def manifest(issue: int, closes: dict[str, float]) -> dict[str, Any]:
    return {
        "issue": issue,
        "release": "r1",
        "feed": {"assets": list(closes), "horizons": [1, 4, 24]},
        "forecasters": [{"name": n, "id": i} for n, i in IDS.items()],
        "inputs": {"last_candle": {a: [c, c, c, c, 1.0] for a, c in closes.items()}},
    }


def forecasts(issue: int) -> dict[str, Any]:
    rows = [{"forecaster": n, "asset": a, "horizon": h, "bps": BPS} for n in IDS for a in ASSETS for h in (1, 4, 24)]
    return {"issue": issue, "forecasts": rows}


def test_scores_the_bin_the_return_fell_in() -> None:
    s = settle(manifest(100, {"BTC": 100.0, "ETH": 100.0}), forecasts(100), 1, {"BTC": 100.5, "ETH": 90.0}, "issue 101")
    assert s.bins == {"BTC": 6, "ETH": 0}  # +0.50% is in [0.3%, 0.7%); -10.5% is below -3%
    assert s.scores["demon"]["BTC"] == pytest.approx(math.log(0.16))
    assert s.scores["naive"]["ETH"] == pytest.approx(math.log(0.01))
    assert s.ids == IDS
    assert s.release == "r1"
    assert Settlement.from_json(s.to_json()) == s
    assert json.loads(s.to_json())["returns"]["BTC"] == pytest.approx(math.log(1.005))


def test_a_later_issue_with_more_assets_settles_only_the_issue_s_own() -> None:
    end = {"BTC": 100.5, "ETH": 90.0, "SOL": 10.0}
    s = settle(manifest(100, {"BTC": 100.0, "ETH": 100.0}), forecasts(100), 1, end, "issue 101")
    assert set(s.end) == set(ASSETS)


def write_issue(issues: Path, issue: int, close: float, *, committed: bool = True, extra: str | None = None) -> None:
    d = issues / str(issue)
    d.mkdir(parents=True)
    closes = {"BTC": close, "ETH": close} | ({extra: close} if extra else {})
    (d / "manifest.json").write_text(json.dumps(manifest(issue, closes)))
    (d / "forecasts.json").write_text(json.dumps(forecasts(issue)))
    if committed:
        (d / "commitment.json").write_text("{}")


def test_settles_on_committed_prices_and_falls_back_to_the_exchange_for_gaps(tmp_path: Path) -> None:
    for issue, close in ((100, 100.0), (101, 101.0), (110, 100.0)):
        write_issue(tmp_path, issue, close)
    write_issue(tmp_path, 104, 99.0, committed=False)  # sealed but never committed: its prices are not used
    fetched = []

    def fetch_close(asset: str, open_time: int) -> float:
        fetched.append((asset, open_time))
        return 98.0

    written = settle_due(tmp_path, 110, fetch_close)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in written) == [
        "100/settlement-1h.json",
        "100/settlement-4h.json",
        "101/settlement-1h.json",
        "101/settlement-4h.json",
    ]
    one = Settlement.from_json((tmp_path / "100" / "settlement-1h.json").read_bytes())
    assert one.source == "issue 101"
    assert one.end == {"BTC": 101.0, "ETH": 101.0}
    four = Settlement.from_json((tmp_path / "100" / "settlement-4h.json").read_bytes())
    assert four.source == "coinbase"
    assert four.end["BTC"] == 98.0
    assert ("BTC", 103 * 3600) in fetched
    assert settle_due(tmp_path, 110, fetch_close) == []  # 24h is not due yet; nothing is settled twice


def test_waits_for_the_current_issue_to_commit(tmp_path: Path) -> None:
    write_issue(tmp_path, 109, 100.0)
    write_issue(tmp_path, 110, 100.0, committed=False)
    assert settle_due(tmp_path, 110, lambda asset, t: pytest.fail("must not fetch")) == []


def test_one_unsettleable_question_never_blocks_the_others(tmp_path: Path) -> None:
    write_issue(tmp_path, 100, 100.0)
    write_issue(tmp_path, 101, 100.0, extra="SOL")  # an appended asset in the settling issue

    def flaky(asset: str, open_time: int) -> float:
        raise ValueError("the exchange is down")

    written = settle_due(tmp_path, 200, flaky)
    assert [p.relative_to(tmp_path).as_posix() for p in written] == ["100/settlement-1h.json"]
    later = settle_due(tmp_path, 200, lambda asset, t: 100.0)  # retried on the next run, however old the issue
    assert sorted(p.relative_to(tmp_path).as_posix() for p in later) == [
        "100/settlement-24h.json",
        "100/settlement-4h.json",
        "101/settlement-1h.json",
        "101/settlement-24h.json",
        "101/settlement-4h.json",
    ]


def settlement(issue: int, horizon: int, demon_id: str, demon: float, naive: float) -> Settlement:
    scores = {"demon": {a: demon for a in ASSETS}, "naive": {a: naive for a in ASSETS}}
    return Settlement(issue, horizon, "test", {}, {}, {}, scores, "r1", {"demon": demon_id, "naive": IDS["naive"]})


def test_scoreboard_pairs_by_question_and_keys_forecasters_by_identity() -> None:
    rng = np.random.default_rng(0)
    settlements = []
    for issue in range(1_000, 1_000 + 2 * BLOCK_ISSUES):
        for h in (1, 4, 24):
            base = float(rng.normal(-2.0, 0.5))
            settlements.append(settlement(issue, h, IDS["demon"], base + 0.05, base))
    refit = "0x" + "e" * 64  # a new program under the same name is a new entry
    settlements.append(settlement(2_000, 1, refit, -1.0, -2.0))
    board = scoreboard(settlements)
    demon = board["forecasters"][IDS["demon"]]
    assert demon["name"] == "demon"
    assert demon["questions"] == 3 * 2 * 2 * BLOCK_ISSUES
    assert demon["vs"]["naive"]["mean"] == pytest.approx(0.05)
    assert demon["vs"]["naive"]["interval95"] == pytest.approx([0.05, 0.05])
    assert demon["vs"]["naive"]["by_horizon"]["24"]["issues"] == 2 * BLOCK_ISSUES
    assert board["forecasters"][refit]["vs"]["naive"]["issues"] == 0  # its only issue has one horizon settled
    assert board["forecasters"][refit]["vs"]["naive"]["by_horizon"]["1"]["mean"] == pytest.approx(1.0)
    short = scoreboard(settlements[: 3 * BLOCK_ISSUES])
    assert short["forecasters"][IDS["demon"]]["vs"]["naive"]["interval95"] is None  # too few issues


def test_load_settlements_reads_every_file(tmp_path: Path) -> None:
    for issue in (100, 101, 102):
        write_issue(tmp_path, issue, 100.0)
    settle_due(tmp_path, 102, lambda asset, t: 100.0)
    assert {(s.issue, s.horizon) for s in load_settlements(tmp_path)} == {(100, 1), (101, 1)}


def test_the_primary_statistic_needs_both_horizons_for_every_asset() -> None:
    rng = np.random.default_rng(1)
    settlements = []
    for issue in range(5_000, 5_000 + 2 * BLOCK_ISSUES + 10):
        for h in (1, 4, 24):
            base = {a: float(rng.normal(-2.0, 0.4)) for a in ASSETS}
            demon = {a: base[a] + 0.05 + float(rng.normal(0, 0.1)) for a in ASSETS}
            if issue == 5_000 and h == 4:
                demon.pop("ETH")  # a missing forecast: this issue must not count
            scores = {"demon": demon, "garch": base}
            ids = {"demon": IDS["demon"], "garch": "0x" + "6" * 64}
            settlements.append(Settlement(issue, h, "test", {}, {}, dict.fromkeys(ASSETS, 5), scores, "r1", ids))
    primary = scoreboard(settlements)["forecasters"][IDS["demon"]]["vs"]["garch"]["primary"]
    assert primary["scope"] == "1h+4h"
    assert primary["issues"] == 2 * BLOCK_ISSUES + 9
    assert primary["mean"] == pytest.approx(0.05, abs=0.01)
    assert primary["z"]["72"] > 5
    assert primary["z"]["168"] is None  # fewer than two 168-issue blocks


def test_each_rule_runs_from_its_start_issue_by_identity() -> None:
    from laplace_engine.settle import RULES, rule_statistics

    judged, opponent = "0x" + "1" * 64, "0x" + "2" * 64
    rule = {
        "rule": "docs/judging/rx.md",
        "start_issue": 7_010,
        "judged": ("x", judged),
        "opponents": (("y", opponent),),
    }
    scores: dict[str, dict[tuple[int, int, str], float]] = {judged: {}, opponent: {}}
    expected = {}
    for issue in range(7_000, 7_100):
        for h in (1, 4, 24):
            expected[(issue, h)] = len(ASSETS)
            for a in ASSETS:
                scores[judged][(issue, h, a)] = -2.0 + (0.5 if issue < 7_010 else 0.02)  # before the start: ignored
                scores[opponent][(issue, h, a)] = -2.0
    (stat,) = rule_statistics(scores, expected, (rule,))
    assert stat["status"] == "running, unresolved"
    assert stat["judged"] == {"name": "x", "id": judged}
    assert stat["vs"]["y"]["issues"] == 90
    assert stat["vs"]["y"]["mean"] == pytest.approx(0.02)
    assert [r["rule"] for r in RULES] == ["docs/judging/r1.md", "docs/judging/r2.md"]
    assert all(len(ident) == 66 for r in RULES for _, ident in (r["judged"], *r["opponents"]))
