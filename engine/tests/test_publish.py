import json
from pathlib import Path

import numpy as np

from laplace_engine.baselines import NAIVE
from laplace_engine.issue import Ledger, seal
from laplace_engine.publish import FRESH, IMMUTABLE, PUBLISHED, publish_pending, publish_release
from laplace_engine.release import Program, Release
from laplace_research.evolve.feed import ASSETS, HORIZONS, N_BINS

from .conftest import synthetic_panel
from .fakes import FakeStore

RELEASE = Release("r1", "test", (Program("naive", NAIVE, True, "baseline"),), ())


def issue(root: Path, n: int, *, committed: bool = True, published: bool = False, honest: bool = True) -> Path:
    """A really sealed issue, with a commitment that matches it (or, if not ``honest``, one that does not)."""
    finals = {"naive": np.full((len(ASSETS), len(HORIZONS), N_BINS), 1 / N_BINS)}
    sealed = seal(
        n,
        RELEASE,
        synthetic_panel(end_issue=n),
        finals,
        {},
        ledger=Ledger(4663, "0x" + "11" * 20),
        commit=None,
        calendar=None,
    )
    d = root / str(n)
    d.mkdir(parents=True)
    (d / "manifest.json").write_bytes(sealed.manifest)
    (d / "forecasts.json").write_bytes(sealed.forecasts)
    if committed:
        root_hex = "0x" + (sealed.root if honest else bytes(32)).hex()
        commitment = {
            "issue": n,
            "root": root_hex,
            "manifest": "0x" + sealed.manifest_hash.hex(),
            "leaves": sealed.leaves,
        }
        commitment |= {"tx": f"0x{n:064x}", "block": n, "timestamp": n * 3600 + 60}
        (d / "commitment.json").write_text(json.dumps(commitment))
    if published:
        (d / PUBLISHED).touch()
    return d


def test_publishes_committed_issues_once_and_points_latest_at_the_newest(tmp_path: Path) -> None:
    store = FakeStore()
    issue(tmp_path, 10, published=True)
    issue(tmp_path, 11)
    issue(tmp_path, 12)
    issue(tmp_path, 13, committed=False)  # sealed but never committed: not part of the record
    assert publish_pending(store, tmp_path) == [11, 12]
    assert set(store.objects) == {
        *(f"v1/issues/{n}/{f}" for n in (11, 12) for f in ("manifest.json", "forecasts.json", "commitment.json")),
        "v1/latest.json",
    }
    assert json.loads(store.objects["v1/latest.json"][0])["issue"] == 12
    assert store.objects["v1/latest.json"][2] == FRESH
    assert store.objects["v1/issues/11/manifest.json"][1:] == ("application/json", IMMUTABLE)
    assert publish_pending(FakeStore(), tmp_path) == []


def test_catching_up_on_an_older_issue_never_moves_latest_back(tmp_path: Path) -> None:
    store = FakeStore()
    issue(tmp_path, 20)
    issue(tmp_path, 21, published=True)
    assert publish_pending(store, tmp_path) == [20]
    assert "v1/latest.json" not in store.objects


def test_files_that_do_not_match_their_commitment_are_never_published(tmp_path: Path) -> None:
    store = FakeStore()
    issue(tmp_path, 30, honest=False)
    issue(tmp_path, 31)
    assert publish_pending(store, tmp_path) == [31]
    assert not any(key.startswith("v1/issues/30/") for key in store.objects)
    assert not (tmp_path / "30" / PUBLISHED).exists()


def test_publishes_a_release_once(tmp_path: Path) -> None:
    release = tmp_path / "r1"
    (release / "programs").mkdir(parents=True)
    (release / "release.json").write_text("{}")
    (release / "programs" / "naive.py").write_text("def forecast(ctx):\n    return [1] * 10\n")
    store, marker = FakeStore(), tmp_path / "data" / "r1.published"
    publish_release(store, release, "r1", marker)
    assert set(store.objects) == {"v1/releases/r1/release.json", "v1/releases/r1/programs/naive.py"}
    assert store.objects["v1/releases/r1/programs/naive.py"][1] == "text/x-python; charset=utf-8"
    again = FakeStore()
    publish_release(again, release, "r1", marker)
    assert not again.objects
