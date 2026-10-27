"""Releases: the fixed set of forecasters the engine runs, and how each one is identified.

A release lists programs and pools. A program is identified by the keccak256 of its source, or, when it comes with a
parameter file (public arrays it reads as ``ctx.params``), by keccak256(abi.encode(keccak256(source),
keccak256(params))). Public sources and parameter files live in
this repository under the release's directory; private ones are kept outside it, and before every run their hashes are
checked against the release file, so when a private program is revealed anyone can match it to its record. A pool is a
linear pool of the release's programs with integer weights in parts per million; its identity is the keccak256 of its
canonical specification, so it commits to its members and their weights.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eth_hash.auto import keccak

from laplace_engine.quantize import WEIGHT_SCALE

RELEASE_FILE = "release.json"


class ReleaseMismatch(ValueError):
    """A program's source or a pool's specification does not match the identity the release file declares."""


def canonical(obj: Any) -> bytes:
    """Canonical JSON: sorted keys, no whitespace, ASCII only. Hashes are taken over these exact bytes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def hex32(value: bytes) -> str:
    return "0x" + value.hex()


@dataclass(frozen=True)
class Program:
    name: str
    source: str
    public: bool
    origin: str
    params: bytes | None = None  # its parameter file (``evolve.params``), if any

    @property
    def id(self) -> bytes:
        source = bytes(keccak(self.source.encode()))
        if self.params is None:
            return source
        return bytes(keccak(source + bytes(keccak(self.params))))  # abi.encode of two bytes32: their concatenation


Weights = tuple[tuple[str, int], ...]  # (program name, parts per million), summing to WEIGHT_SCALE


def _check(name: str, weights: Weights) -> None:
    if sum(w for _, w in weights) != WEIGHT_SCALE or any(w <= 0 for _, w in weights):
        raise ValueError(f"pool {name}: weights must be positive and sum to {WEIGHT_SCALE}")


def _members(weights: Weights, program_ids: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [{"forecaster": hex32(program_ids[name]), "weight": w} for name, w in weights]


@dataclass(frozen=True)
class Pool:
    """A linear pool with the same weights at every horizon."""

    name: str
    weights: Weights

    def __post_init__(self) -> None:
        _check(self.name, self.weights)

    def members(self) -> set[str]:
        return {name for name, _ in self.weights}

    def weights_at(self, horizon: int) -> Weights:
        return self.weights

    def spec(self, program_ids: Mapping[str, bytes]) -> dict[str, Any]:
        return {"kind": "linear-pool", "members": _members(self.weights, program_ids), "weight_scale": WEIGHT_SCALE}

    def id(self, program_ids: Mapping[str, bytes]) -> bytes:
        return bytes(keccak(canonical(self.spec(program_ids))))


@dataclass(frozen=True)
class HorizonPool:
    """A linear pool with its own weights at each horizon."""

    name: str
    by_horizon: tuple[tuple[int, Weights], ...]

    def __post_init__(self) -> None:
        horizons = [h for h, _ in self.by_horizon]
        if len(set(horizons)) != len(horizons):
            raise ValueError(f"pool {self.name}: each horizon needs exactly one set of weights")
        for h, weights in self.by_horizon:
            _check(f"{self.name} at {h}h", weights)

    def members(self) -> set[str]:
        return {name for _, weights in self.by_horizon for name, _ in weights}

    def weights_at(self, horizon: int) -> Weights:
        return dict(self.by_horizon)[horizon]

    def spec(self, program_ids: Mapping[str, bytes]) -> dict[str, Any]:
        horizons = {str(h): _members(weights, program_ids) for h, weights in self.by_horizon}
        return {"kind": "linear-pool-by-horizon", "horizons": horizons, "weight_scale": WEIGHT_SCALE}

    def id(self, program_ids: Mapping[str, bytes]) -> bytes:
        return bytes(keccak(canonical(self.spec(program_ids))))


AnyPool = Pool | HorizonPool


@dataclass(frozen=True)
class Release:
    name: str
    notes: str
    programs: tuple[Program, ...]
    pools: tuple[AnyPool, ...]

    def __post_init__(self) -> None:
        names = [p.name for p in self.programs] + [p.name for p in self.pools]
        if any(p.params is not None and not p.public for p in self.programs):
            raise ValueError("parameter files are public: a private program cannot carry one")
        if len(set(names)) != len(names):
            raise ValueError("forecaster names must be unique within a release")
        for pool in self.pools:
            unknown = pool.members() - {p.name for p in self.programs}
            if unknown:
                raise ValueError(f"pool {pool.name} has members outside the release: {sorted(unknown)}")
        ids = self.ids()
        if len(set(ids.values())) != len(ids):
            shared = sorted(n for n, i in ids.items() if list(ids.values()).count(i) > 1)
            raise ValueError(f"forecasters {', '.join(shared)} share an identity: their leaves would collide")

    def program_ids(self) -> dict[str, bytes]:
        return {p.name: p.id for p in self.programs}

    def ids(self) -> dict[str, bytes]:
        """Every forecaster's identity, programs first, in release order."""
        programs = self.program_ids()
        return programs | {pool.name: pool.id(programs) for pool in self.pools}

    def describe(self) -> list[dict[str, Any]]:
        """The forecasters as listed in release files and manifests (sources excluded)."""
        programs = self.program_ids()
        out: list[dict[str, Any]] = []
        for p in self.programs:
            entry = {"name": p.name, "kind": "program", "id": hex32(p.id), "public": p.public, "origin": p.origin}
            if p.public:
                entry["path"] = f"programs/{p.name}.py"
            if p.params is not None:
                entry["params"] = {"path": params_path(p.name), "keccak256": hex32(bytes(keccak(p.params)))}
            out.append(entry)
        for pool in self.pools:
            out.append({"name": pool.name, "kind": "pool", "id": hex32(pool.id(programs)), "spec": pool.spec(programs)})
        return out


def params_path(name: str) -> str:
    """Where a program's parameter file lives, relative to its release directory."""
    return f"programs/{name}.params.json"


def write(release: Release, release_dir: Path, private_dir: Path) -> None:
    """Write the release file and public sources to ``release_dir`` and private sources to ``private_dir``."""
    (release_dir / "programs").mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(private_dir, 0o700)
    for p in release.programs:
        path = release_dir / "programs" / f"{p.name}.py" if p.public else private_dir / f"{p.name}.py"
        path.write_bytes(p.source.encode())
        if p.params is not None:
            (release_dir / params_path(p.name)).write_bytes(p.params)
        if not p.public:
            os.chmod(path, 0o600)
    body = {"release": release.name, "notes": release.notes, "forecasters": release.describe()}
    (release_dir / RELEASE_FILE).write_text(json.dumps(body, indent=1) + "\n")


def load(release_dir: Path, private_dir: Path | None) -> Release:
    """Read a release, refusing any source or pool whose identity differs from the release file."""
    body = json.loads((release_dir / RELEASE_FILE).read_text())
    programs: list[Program] = []
    pools: list[AnyPool] = []
    for entry in body["forecasters"]:
        if entry["kind"] == "program":
            if entry["public"]:
                path = release_dir / entry["path"]
            elif private_dir is None:
                raise ReleaseMismatch(f"{entry['name']} is private and no private directory was given")
            else:
                path = private_dir / f"{entry['name']}.py"
            params = None
            if "params" in entry:
                if entry["params"]["path"] != params_path(entry["name"]):
                    raise ReleaseMismatch(f"{entry['name']}: parameter file at {entry['params']['path']}")
                params = (release_dir / params_path(entry["name"])).read_bytes()
                if hex32(bytes(keccak(params))) != entry["params"]["keccak256"]:
                    raise ReleaseMismatch(f"{entry['name']}: parameter file hash differs from the release")
            program = Program(entry["name"], path.read_bytes().decode(), entry["public"], entry["origin"], params)
            if hex32(program.id) != entry["id"]:
                raise ReleaseMismatch(f"{entry['name']}: source hash {hex32(program.id)} != release {entry['id']}")
            programs.append(program)
    ids = {p.name: p.id for p in programs}
    names = {hex32(i): n for n, i in ids.items()}
    for entry in body["forecasters"]:
        if entry["kind"] == "pool":
            spec = entry["spec"]

            def named(members: list[dict[str, Any]]) -> Weights:
                return tuple((names[m["forecaster"]], int(m["weight"])) for m in members)

            pool: AnyPool
            if spec["kind"] == "linear-pool":
                pool = Pool(entry["name"], named(spec["members"]))
            else:
                by_horizon = tuple((int(h), named(members)) for h, members in sorted(spec["horizons"].items()))
                pool = HorizonPool(entry["name"], tuple(sorted(by_horizon)))
            if hex32(pool.id(ids)) != entry["id"]:
                raise ReleaseMismatch(f"{entry['name']}: specification hash differs from the release")
            pools.append(pool)
    return Release(body["release"], body["notes"], tuple(programs), tuple(pools))


def weights_ppm(weights: Mapping[str, float]) -> tuple[tuple[str, int], ...]:
    """Fitted pool weights as parts per million (largest remainder, ties by name); members rounding to 0 are dropped."""
    total = sum(weights.values())
    scaled = {n: w / total * WEIGHT_SCALE for n, w in weights.items()}
    floors = {n: int(s) for n, s in scaled.items()}
    order: Sequence[str] = sorted(scaled, key=lambda n: (-(scaled[n] - floors[n]), n))
    for n in order[: WEIGHT_SCALE - sum(floors.values())]:
        floors[n] += 1
    return tuple((n, w) for n, w in sorted(floors.items(), key=lambda kv: (-kv[1], kv[0])) if w > 0)
