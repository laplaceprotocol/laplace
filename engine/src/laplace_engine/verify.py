"""The reference verifier: from an issue's published manifest and forecasts alone, rebuild every leaf, the Merkle
root and every pool's forecast, and check them against what the issue declares. The chain check (that the root and the
manifest hash were committed in time) is done by ``ledger``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from eth_hash.auto import keccak

from laplace_engine.quantize import mix_basis_points
from laplace_engine.record import Leaf, Tree
from laplace_engine.release import canonical, hex32


class VerificationError(ValueError):
    pass


@dataclass(frozen=True)
class Verified:
    issue: int
    root: bytes
    manifest_hash: bytes
    leaves: int
    pools_checked: int


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise VerificationError(message)


def _members_at(spec: dict[str, Any], horizon: int) -> list[dict[str, Any]]:
    """A pool specification's members and weights at one horizon."""
    if spec["kind"] == "linear-pool":
        members: list[dict[str, Any]] = spec["members"]
        return members
    if spec["kind"] == "linear-pool-by-horizon":
        by_horizon: list[dict[str, Any]] = spec["horizons"][str(horizon)]
        return by_horizon
    raise VerificationError(f"unknown pool kind {spec['kind']}")


def verify(manifest: bytes, forecasts: bytes) -> Verified:
    m, f = json.loads(manifest), json.loads(forecasts)
    manifest_hash = bytes(keccak(manifest))
    _check(f["manifest"] == hex32(manifest_hash), "the forecasts file names another manifest")
    _check(f["issue"] == m["issue"] and f["root"] == m["root"], "issue or root differs between the files")

    ids = {x["name"]: bytes.fromhex(x["id"][2:]) for x in m["forecasters"]}
    assets = m["feed"]["assets"]
    try:
        leaves = [
            Leaf(m["issue"], ids[e["forecaster"]], assets.index(e["asset"]), e["horizon"], tuple(e["bps"]))
            for e in f["forecasts"]
        ]
    except (KeyError, ValueError) as exc:
        raise VerificationError(f"malformed forecast: {exc}") from None
    tree = Tree.build(leaf.hash() for leaf in leaves)
    _check(hex32(tree.root) == m["root"], f"rebuilt root {hex32(tree.root)} != declared {m['root']}")
    _check(len(leaves) == m["leaves"], "leaf count differs from the manifest")

    by_question = {(e["forecaster"], e["asset"], e["horizon"]): e["bps"] for e in f["forecasts"]}
    names = {x["id"]: x["name"] for x in m["forecasters"]}
    pools = 0
    for x in m["forecasters"]:
        if x["kind"] != "pool" or not any(k[0] == x["name"] for k in by_question):
            continue
        _check(hex32(bytes(keccak(canonical(x["spec"])))) == x["id"], f"pool {x['name']}: id is not its spec's hash")
        for asset in assets:
            for horizon in m["feed"]["horizons"]:
                members = [(names[mm["forecaster"]], mm["weight"]) for mm in _members_at(x["spec"], horizon)]
                inputs = [by_question[(n, asset, horizon)] for n, _ in members]
                expected = mix_basis_points(inputs, [w for _, w in members])
                _check(
                    list(expected) == by_question[(x["name"], asset, horizon)], f"pool {x['name']} {asset} {horizon}h"
                )
        pools += 1
    return Verified(m["issue"], tree.root, manifest_hash, len(leaves), pools)
