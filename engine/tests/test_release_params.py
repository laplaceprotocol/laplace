import json
from pathlib import Path

import numpy as np
import pytest
from eth_hash.auto import keccak

from laplace_engine.release import Pool, Program, Release, ReleaseMismatch, load, params_path, write
from laplace_research.evolve import params

SOURCE = "def forecast(ctx):\n    return [1] * 10\n"
PARAMS = params.encode({"h1.value": np.arange(10.0)})


def with_params() -> Release:
    programs = (
        Program("plain", SOURCE, public=True, origin="test"),
        Program("trees", SOURCE.replace("1]", "2]"), public=True, origin="test", params=PARAMS),
    )
    return Release("t2", "parameter files", programs, (Pool("pool", (("plain", 500_000), ("trees", 500_000))),))


def test_a_parameter_file_is_part_of_the_identity_and_programs_without_one_keep_theirs() -> None:
    rel = with_params()
    ids = rel.ids()
    assert ids["plain"] == keccak(SOURCE.encode())
    source_hash = keccak(SOURCE.replace("1]", "2]").encode())
    assert ids["trees"] == keccak(source_hash + keccak(PARAMS))
    other = Program("trees", SOURCE.replace("1]", "2]"), True, "test", params.encode({"h1.value": np.arange(1.0, 11)}))
    assert other.id != ids["trees"]
    entry = next(e for e in rel.describe() if e["name"] == "trees")
    assert entry["params"] == {"path": "programs/trees.params.json", "keccak256": "0x" + keccak(PARAMS).hex()}
    assert "params" not in next(e for e in rel.describe() if e["name"] == "plain")


def test_parameter_files_round_trip_and_a_changed_one_is_refused(tmp_path: Path) -> None:
    public, private = tmp_path / "release", tmp_path / "private"
    write(with_params(), public, private)
    assert (public / params_path("trees")).read_bytes() == PARAMS
    assert load(public, private) == with_params()
    (public / params_path("trees")).write_bytes(params.encode({"h1.value": np.arange(1.0, 11)}))
    with pytest.raises(ReleaseMismatch, match="parameter file"):
        load(public, private)


def test_a_release_file_cannot_point_a_parameter_file_elsewhere(tmp_path: Path) -> None:
    public, private = tmp_path / "release", tmp_path / "private"
    write(with_params(), public, private)
    body = json.loads((public / "release.json").read_text())
    next(e for e in body["forecasters"] if e["name"] == "trees")["params"]["path"] = "../elsewhere.json"
    (public / "release.json").write_text(json.dumps(body))
    with pytest.raises(ReleaseMismatch, match="parameter file"):
        load(public, private)


def test_private_programs_cannot_carry_parameter_files() -> None:
    with pytest.raises(ValueError, match="public"):
        Release("t3", "", (Program("secret", SOURCE, False, "test", params=PARAMS),), ())


def test_names_that_are_not_plain_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="names"):
        Release("t4", "", (Program("../up", SOURCE, True, "test"),), ())
    public, private = tmp_path / "release", tmp_path / "private"
    write(with_params(), public, private)
    body = json.loads((public / "release.json").read_text())
    next(e for e in body["forecasters"] if e["name"] == "trees")["name"] = "../trees"
    (public / "release.json").write_text(json.dumps(body))
    with pytest.raises(ReleaseMismatch, match="name"):
        load(public, private)
