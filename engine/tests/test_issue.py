import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from eth_hash.auto import keccak

from laplace_engine.baselines import NAIVE
from laplace_engine.issue import Ledger, run_release, seal
from laplace_engine.quantize import mix_basis_points
from laplace_engine.release import HorizonPool, Pool, Program, Release
from laplace_engine.verify import VerificationError, verify
from laplace_research.evolve.feed import ASSETS, HORIZONS
from laplace_research.evolve.panel import Panel

from .conftest import ISSUE, synthetic_panel

LEDGER = Ledger(4663, "0x" + "11" * 20)
WIDER = NAIVE.replace("sd_hour * math.sqrt(ctx.horizon)", "1.5 * sd_hour * math.sqrt(ctx.horizon)")
BROKEN = "def forecast(ctx):\n    return [1.0] * 9\n"


def release(*extra: Program) -> Release:
    programs = (Program("naive", NAIVE, True, "baseline"), Program("wider", WIDER, False, "test#1"), *extra)
    return Release("t1", "test", programs, (Pool("demon", (("naive", 700_000), ("wider", 300_000))),))


@pytest.fixture(scope="module")
def sealed_files(tmp_path_factory: pytest.TempPathFactory) -> tuple[bytes, bytes]:
    rel, panel = release(Program("broken", BROKEN, True, "test")), synthetic_panel()
    finals, failures = run_release(rel, panel, tmp_path_factory.mktemp("work"), workers=3)
    sealed = seal(ISSUE, rel, panel, finals, failures, ledger=LEDGER, commit="abc")
    assert sealed.failures.keys() == {"broken"}
    assert sealed.leaves == 3 * len(ASSETS) * len(HORIZONS)
    return sealed.manifest, sealed.forecasts


def test_a_sealed_issue_verifies_from_its_files_alone(sealed_files: tuple[bytes, bytes]) -> None:
    manifest, forecasts = sealed_files
    result = verify(manifest, forecasts)
    assert result.leaves == 63
    assert result.pools_checked == 1
    m = json.loads(manifest)
    assert m["root"] == json.loads(forecasts)["root"] == "0x" + result.root.hex()
    assert manifest == json.dumps(m, sort_keys=True, separators=(",", ":")).encode()  # canonical bytes
    assert json.loads(forecasts)["manifest"] == "0x" + keccak(manifest).hex()
    assert m["failed"]["broken"].startswith("bad output")
    assert m["inputs"]["last_open"] == "2026-10-23T06:00:00Z"
    assert m["engine"] == {"version": "0.1.0", "commit": "abc"}
    assert m["judging"] == []
    assert {f["name"] for f in m["forecasters"]} == {"naive", "wider", "broken", "demon"}


def shift_member(body: dict[str, Any]) -> None:
    bps = body["forecasts"][0]["bps"]
    bps[0], bps[1] = bps[0] + 1, bps[1] - 1


def drop_one(body: dict[str, Any]) -> None:
    body["forecasts"].pop(0)


def shift_pool(body: dict[str, Any]) -> None:
    bps = body["forecasts"][-1]["bps"]  # the pool is listed last
    bps[4], bps[5] = bps[4] + 1, bps[5] - 1


@pytest.mark.parametrize("tamper", [shift_member, drop_one, shift_pool])
def test_any_change_to_the_forecasts_fails_verification(
    sealed_files: tuple[bytes, bytes], tamper: Callable[[dict[str, Any]], None]
) -> None:
    manifest, forecasts = sealed_files
    body = json.loads(forecasts)
    tamper(body)
    with pytest.raises(VerificationError):
        verify(manifest, json.dumps(body).encode())


def test_any_change_to_the_manifest_fails_verification(sealed_files: tuple[bytes, bytes]) -> None:
    # The forecasts file (and the chain) pin the manifest's hash, so no field of it can change unnoticed.
    manifest, forecasts = sealed_files
    body = json.loads(manifest)
    body["inputs"]["last_candle"]["BTC"][3] *= 1.01
    with pytest.raises(VerificationError, match="another manifest"):
        verify(json.dumps(body, sort_keys=True, separators=(",", ":")).encode(), forecasts)


def test_a_pool_is_skipped_when_a_member_fails(tmp_path: Path, make_panel: Callable[..., Panel]) -> None:
    rel = Release(
        "t2",
        "test",
        (Program("naive", NAIVE, True, "baseline"), Program("broken", BROKEN, True, "test")),
        (Pool("demon", (("naive", 500_000), ("broken", 500_000))),),
    )
    finals, failures = run_release(rel, make_panel(), tmp_path, workers=2)
    sealed = seal(ISSUE, rel, make_panel(), finals, failures, ledger=LEDGER, commit=None)
    assert set(sealed.failures) == {"broken", "demon"}
    assert sealed.leaves == len(ASSETS) * len(HORIZONS)
    assert verify(sealed.manifest, sealed.forecasts).pools_checked == 0


def test_a_per_horizon_pool_is_sealed_and_recomputed_exactly(tmp_path: Path, make_panel: Callable[..., Panel]) -> None:
    programs = (Program("naive", NAIVE, True, "baseline"), Program("wider", WIDER, False, "test#1"))
    by_horizon = tuple(
        (h, (("naive", 1_000_000 - 100_000 * i), ("wider", 100_000 * i))) for i, h in enumerate(HORIZONS, 1)
    )
    rel = Release("t3", "test", programs, (HorizonPool("demon-h", by_horizon),))
    finals, failures = run_release(rel, make_panel(), tmp_path, workers=2)
    sealed = seal(ISSUE, rel, make_panel(), finals, failures, ledger=LEDGER, commit=None)
    assert verify(sealed.manifest, sealed.forecasts).pools_checked == 1
    body = json.loads(sealed.forecasts)
    by_question = {(f["forecaster"], f["asset"], f["horizon"]): f["bps"] for f in body["forecasts"]}
    for h, weights in by_horizon:
        expected = mix_basis_points([by_question[(n, "BTC", h)] for n, _ in weights], [w for _, w in weights])
        assert by_question[("demon-h", "BTC", h)] == list(expected)
    last = body["forecasts"][-1]
    last["bps"][4], last["bps"][5] = last["bps"][4] + 1, last["bps"][5] - 1
    with pytest.raises(VerificationError):
        verify(sealed.manifest, json.dumps(body).encode())


def test_inputs_record_the_senses_programs_saw(make_panel: Callable[..., Panel]) -> None:
    from laplace_engine.issue import inputs

    panel = make_panel()
    rv = np.full(panel.close.shape, 1e-5)
    rv[1, -1] = np.nan
    with_senses = panel.with_senses({"rv": rv})
    plain, sensed = inputs(panel), inputs(with_senses)
    assert "last_senses" not in plain
    assert sensed["keccak256"] != plain["keccak256"]
    assert sensed["keccak256_layout"].endswith("volume, then the senses rv)")
    assert sensed["last_senses"]["BTC"] == {"rv": 1e-5}
    assert sensed["last_senses"]["ETH"] == {"rv": None}  # unknown, and JSON has no NaN
    json.dumps(sensed, allow_nan=False)
