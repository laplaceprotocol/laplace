import json
from pathlib import Path

import pytest
from eth_hash.auto import keccak

from laplace_engine.release import (
    HorizonPool,
    Pool,
    Program,
    Release,
    ReleaseMismatch,
    canonical,
    load,
    weights_ppm,
    write,
)

SOURCE = "def forecast(ctx):\n    return [1] * 10\n"


def small_release() -> Release:
    programs = (
        Program("secret", SOURCE, public=False, origin="test#1"),
        Program("open", SOURCE.replace("1]", "2]"), public=True, origin="test"),
    )
    return Release("t1", "a test release", programs, (Pool("pool", (("secret", 600_000), ("open", 400_000))),))


def test_identities_commit_to_sources_and_pool_specifications() -> None:
    rel = small_release()
    ids = rel.ids()
    assert ids["secret"] == keccak(SOURCE.encode())
    spec = rel.pools[0].spec(rel.program_ids())
    assert spec["members"][0] == {"forecaster": "0x" + ids["secret"].hex(), "weight": 600_000}
    assert ids["pool"] == keccak(canonical(spec))
    assert list(ids) == ["secret", "open", "pool"]


def test_round_trip_keeps_private_sources_out_of_the_release_directory(tmp_path: Path) -> None:
    public, private = tmp_path / "release", tmp_path / "private"
    write(small_release(), public, private)
    assert (public / "programs" / "open.py").exists()
    assert not (public / "programs" / "secret.py").exists()
    assert SOURCE not in (public / "release.json").read_text()
    assert (private / "secret.py").stat().st_mode & 0o777 == 0o600
    assert load(public, private) == small_release()


def test_load_refuses_a_changed_source(tmp_path: Path) -> None:
    public, private = tmp_path / "release", tmp_path / "private"
    write(small_release(), public, private)
    (private / "secret.py").write_text(SOURCE + "# changed\n")
    with pytest.raises(ReleaseMismatch, match="secret"):
        load(public, private)
    with pytest.raises(ReleaseMismatch, match="private"):
        load(public, None)


def test_load_refuses_a_changed_pool(tmp_path: Path) -> None:
    public, private = tmp_path / "release", tmp_path / "private"
    write(small_release(), public, private)
    body = json.loads((public / "release.json").read_text())
    body["forecasters"][2]["spec"]["members"][0]["weight"] = 500_000
    body["forecasters"][2]["spec"]["members"][1]["weight"] = 500_000
    (public / "release.json").write_text(json.dumps(body))
    with pytest.raises(ReleaseMismatch, match="pool"):
        load(public, private)


def test_fitted_weights_become_parts_per_million() -> None:
    ppm = weights_ppm({"a": 0.447582, "b": 0.245489, "c": 0.306929, "zero": 1e-12})
    assert dict(ppm) == {"a": 447_582, "c": 306_929, "b": 245_489}
    assert sum(w for _, w in weights_ppm({"x": 1 / 3, "y": 1 / 3, "z": 1 / 3})) == 1_000_000
    assert weights_ppm({"x": 1 / 3, "y": 1 / 3, "z": 1 / 3})[0] == ("x", 333_334)


def test_releases_reject_ambiguous_definitions() -> None:
    program = Program("p", SOURCE, public=True, origin="test")
    with pytest.raises(ValueError, match="unique"):
        Release("t", "", (program, program), ())
    with pytest.raises(ValueError, match="outside"):
        Release("t", "", (program,), (Pool("pool", (("q", 1_000_000),)),))
    with pytest.raises(ValueError, match="sum"):
        Pool("pool", (("p", 999_999),))


def test_a_pool_can_weigh_each_horizon_differently(tmp_path: Path) -> None:
    by_horizon = (
        (1, (("secret", 900_000), ("open", 100_000))),
        (4, (("open", 1_000_000),)),
        (24, (("secret", 1_000_000),)),
    )
    rel = Release("t2", "test", small_release().programs, (HorizonPool("demon-h", by_horizon),))
    ids = rel.ids()
    spec = rel.pools[0].spec(rel.program_ids())
    assert spec["kind"] == "linear-pool-by-horizon"
    assert spec["horizons"]["4"] == [{"forecaster": "0x" + ids["open"].hex(), "weight": 1_000_000}]
    assert ids["demon-h"] == keccak(canonical(spec))
    assert rel.pools[0].weights_at(24) == (("secret", 1_000_000),)
    assert rel.pools[0].members() == {"secret", "open"}
    write(rel, tmp_path / "release", tmp_path / "private")
    assert load(tmp_path / "release", tmp_path / "private") == rel
    with pytest.raises(ValueError, match="one set"):
        HorizonPool("x", ((1, (("open", 1_000_000),)), (1, (("open", 1_000_000),))))
    with pytest.raises(ValueError, match="sum"):
        HorizonPool("x", ((1, (("open", 999_999),)),))


def test_two_forecasters_may_not_share_an_identity() -> None:
    programs = small_release().programs
    twin = Pool("twin", (("secret", 600_000), ("open", 400_000)))
    with pytest.raises(ValueError, match="share an identity"):
        Release("t", "", programs, (*small_release().pools, twin))
