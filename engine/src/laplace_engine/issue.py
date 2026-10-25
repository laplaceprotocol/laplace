"""Sealing an issue: run a release on the candles that have just closed, round every forecast to basis points, and
build the Merkle root and the manifest that describe the issue.

The manifest's exact bytes are what gets hashed and committed on-chain next to the root; the forecasts file lists every
leaf, so anyone can rebuild the tree (see ``verify``).
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from eth_hash.auto import keccak

from laplace_engine import __version__
from laplace_engine.quantize import mix_basis_points, to_basis_points
from laplace_engine.record import Leaf, Tree
from laplace_engine.release import Release, canonical, hex32
from laplace_research.data.candles import HOUR
from laplace_research.evolve.context import LOOKBACK
from laplace_research.evolve.evaluate import Evaluator
from laplace_research.evolve.feed import ASSETS, EDGES_PCT, HORIZONS, FloatArray, finalize
from laplace_research.evolve.history import FIELDS
from laplace_research.evolve.panel import Panel

TIME_LIMIT_S = 60
MANIFEST_SCHEMA = "laplace/manifest/1"
FORECASTS_SCHEMA = "laplace/forecasts/1"
LEAF_ENCODING = (
    "keccak256(keccak256(abi.encode(uint64 issue, bytes32 forecaster, uint8 asset, uint16 horizon, uint16[10] bps))); "
    "inner nodes: keccak256 of the sorted pair"
)
QUESTION = (
    "For each asset and horizon h: the probability, in basis points, of each of ten bins for the log return from the "
    "close of the last input candle to the close of the hourly candle that ends h hours after the issue time. Bin k "
    "holds returns r with edges[k-1] <= r < edges[k], where edges are -e4, -e3, -e2, -e1, 0, e1, e2, e3, e4 (edges_pct "
    "of the horizon, in percent of log return); the first and last bins are open."
)
PRICES = "Coinbase Exchange hourly candles for <asset>-USD; an hour without trades repeats the previous close."

Forecasts = dict[str, list[list[tuple[int, ...]]]]  # forecaster -> [asset][horizon index] -> bps


@dataclass(frozen=True)
class Ledger:
    chain_id: int
    address: str


@dataclass(frozen=True)
class Sealed:
    issue: int
    root: bytes
    leaves: int
    manifest: bytes  # the exact bytes whose keccak256 is committed
    forecasts: bytes
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def manifest_hash(self) -> bytes:
        return bytes(keccak(self.manifest))


def run_release(
    release: Release, panel: Panel, workdir: Path, *, workers: int, time_limit_s: int = TIME_LIMIT_S
) -> tuple[dict[str, FloatArray], dict[str, str]]:
    """Final forecasts ``[asset, horizon, bin]`` of every program at the panel's last hour, plus failures by name.

    Each program runs in a fresh spawned process that only ever reads the panel up to its last hour.
    """
    path = workdir / "panel.npz"
    panel.save(path)
    times = np.array([len(panel.ts) - 1], dtype=np.int64)
    finals: dict[str, FloatArray] = {}
    failures: dict[str, str] = {}
    with Evaluator(path, workers=workers) as evaluator:
        futures = {p.name: evaluator.submit(p.source, times, time_limit_s) for p in release.programs}
        for name, future in futures.items():
            try:
                outcome = future.result()
            except BrokenProcessPool:
                failures[name] = "worker crashed"
                continue
            if outcome.probs is None:
                failures[name] = outcome.error or "no forecast"
            else:
                finals[name] = finalize(outcome.probs[:, :, 0, :])
    return finals, failures


def basis_points(release: Release, finals: dict[str, FloatArray]) -> tuple[Forecasts, dict[str, str]]:
    """Rounded forecasts of every program that ran, and of every pool whose members all ran (others are failures)."""
    out: Forecasts = {}
    for p in release.programs:
        if p.name in finals:
            f = finals[p.name]
            out[p.name] = [[to_basis_points(f[a, j]) for j in range(len(HORIZONS))] for a in range(len(ASSETS))]
    failures = {}
    for pool in release.pools:
        missing = sorted(m for m in pool.members() if m not in out)
        if missing:
            failures[pool.name] = f"members without a forecast: {', '.join(missing)}"
            continue
        rows: list[list[tuple[int, ...]]] = []
        for a in range(len(ASSETS)):
            row = []
            for j, h in enumerate(HORIZONS):
                weights = pool.weights_at(h)
                row.append(mix_basis_points([out[m][a][j] for m, _ in weights], [w for _, w in weights]))
            rows.append(row)
        out[pool.name] = rows
    return out, failures


def iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def engine_commit() -> str | None:
    """The git commit the engine runs from, marked ``-dirty`` with uncommitted changes; None outside a checkout."""
    here = Path(__file__).parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + ("-dirty" if dirty else "")


def _number(value: float) -> float | None:
    """JSON has no NaN: unknown values are written as null."""
    return float(value) if np.isfinite(value) else None


def inputs(panel: Panel) -> dict[str, Any]:
    """What the programs saw: the last ``LOOKBACK`` hours of candles and senses, identified by hash, and the last
    hour in full."""
    names = sorted(panel.senses)
    series = [getattr(panel, f)[:, -LOOKBACK:] for f in FIELDS] + [panel.senses[n][:, -LOOKBACK:] for n in names]
    window = np.stack(series, axis=-1).transpose(1, 0, 2)
    fields = ", ".join(FIELDS) + (f", then the senses {', '.join(names)}" if names else "")
    out: dict[str, Any] = {
        "candles": LOOKBACK,
        "first_open": iso(int(panel.ts[-LOOKBACK])),
        "last_open": iso(int(panel.ts[-1])),
        "keccak256": hex32(bytes(keccak(np.ascontiguousarray(window, dtype="<f8").tobytes()))),
        "keccak256_layout": f"little-endian float64, ordered hour, asset, field ({fields})",
        "last_candle": {a: [float(getattr(panel, f)[i, -1]) for f in FIELDS] for i, a in enumerate(panel.assets)},
    }
    if names:
        out["last_senses"] = {
            a: {n: _number(panel.senses[n][i, -1]) for n in names} for i, a in enumerate(panel.assets)
        }
    return out


def seal(
    issue: int,
    release: Release,
    panel: Panel,
    finals: dict[str, FloatArray],
    failures: dict[str, str],
    *,
    ledger: Ledger,
    commit: str | None,
    judging: Sequence[dict[str, str]] = (),
) -> Sealed:
    """Build an issue's leaves, root, manifest and forecasts file. ``judging`` lists the pre-registered rules in force
    (path and keccak256), which the manifest carries so that the chain dates them."""
    if panel.assets != ASSETS:
        raise ValueError("the panel's assets must be the feed's registry, in order")
    bps, pool_failures = basis_points(release, finals)
    failures = failures | pool_failures
    ids = release.ids()
    entries = [
        (name, Leaf(issue, ids[name], a, h, bps[name][a][j]))
        for name in ids
        if name in bps
        for a in range(len(ASSETS))
        for j, h in enumerate(HORIZONS)
    ]
    tree = Tree.build(leaf.hash() for _, leaf in entries)
    manifest = canonical(
        {
            "schema": MANIFEST_SCHEMA,
            "issue": issue,
            "time": iso(issue * HOUR),
            "ledger": {"chain_id": ledger.chain_id, "address": ledger.address},
            "engine": {"version": __version__, "commit": commit},
            "release": release.name,
            "feed": {
                "assets": list(ASSETS),
                "horizons": list(HORIZONS),
                "edges_pct": {str(h): list(EDGES_PCT[h]) for h in HORIZONS},
                "question": QUESTION,
                "prices": PRICES,
            },
            "inputs": inputs(panel),
            "forecasters": release.describe(),
            "judging": list(judging),
            "failed": failures,
            "leaf": LEAF_ENCODING,
            "leaves": len(entries),
            "root": hex32(tree.root),
        }
    )
    forecasts = canonical(
        {
            "schema": FORECASTS_SCHEMA,
            "issue": issue,
            "manifest": hex32(bytes(keccak(manifest))),
            "root": hex32(tree.root),
            "forecasts": [
                {"forecaster": name, "asset": ASSETS[leaf.asset], "horizon": leaf.horizon, "bps": list(leaf.bps)}
                for name, leaf in entries
            ],
        }
    )
    return Sealed(issue, tree.root, len(entries), manifest, forecasts, failures)
