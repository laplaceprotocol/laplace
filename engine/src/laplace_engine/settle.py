"""Settling issues and scoring their forecasts from the committed record.

A question of issue N at horizon h settles on the close of the hourly candle that ends at the start of issue N + h.
That candle is issue N + h's last input candle, which its manifest carries and the chain commits, so every score of
issue N at horizon h normally follows from committed files alone. The starting price is issue N's own last input
candle. If issue N + h was never committed, the question settles on the exchange's public candle instead (the last
close at or before that hour, as the feed defines an hour without trades), so a missing hour can never hide an outcome.
Each forecast is scored with the binned log score of its rounded probabilities.

The scoreboard aggregates every settlement per forecaster identity (so a refitted program under an old name is a new
entry): its mean log score, and its paired gain over each baseline with a circular block-bootstrap interval over
issues.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from laplace_engine import bootstrap
from laplace_engine.files import write_atomic
from laplace_engine.quantize import TOTAL
from laplace_engine.release import canonical
from laplace_research.evolve.feed import HORIZONS, bin_index

CLOSE = 3  # position of the close in a manifest's last_candle entries (open, high, low, close, volume)
SCHEMA = "laplace/settlement/1"
BOARD_SCHEMA = "laplace/scoreboard/1"
BLOCK_ISSUES = 72  # bootstrap blocks of 72 hourly issues, as in research
CHECK_BLOCK_ISSUES = 168  # the pre-registered rule's second block length
MIN_BLOCKS = 2  # below two blocks of issues, no interval is shown
BASELINES = ("naive", "garch", "har")
# The pre-registered rules (docs/judging): the judged forecaster and its opponents by identity, from the rule's
# start issue. The scoreboard shows each one's statistic as it runs; only a scheduled look gives a verdict.
RULES = (
    {
        "rule": "docs/judging/r1.md",
        "start_issue": 498020,
        "judged": ("demon", "0x97bc9657cb5b4af73b46e0031ac385cab039e72fb9fb8edf591bd457f149914e"),
        "opponents": (
            ("garch", "0xdf33f0170700f1581334e893a7d08329d5a407c5b0e2913a2c4b9903c7fbe814"),
            ("har", "0x564e31cf84b39fb54ab6fe6e61213eaf70f7cccc792bde19321d357941662dc5"),
        ),
    },
    {
        "rule": "docs/judging/r2.md",
        "start_issue": 498095,
        "judged": ("ev-gbm-r15", "0xb968a022ac9be499bdd06a67a764ee53bdb38c606c7f8e1fc2748f1972845454"),
        "opponents": (
            ("garch-r15", "0xb15791e0ca9a42ca356baf961ad1b7af87f07a28ab37ce5762731453a5f7f767"),
            ("har-r15", "0x9b05a423a031ced018bba945226540dc3edd8dc75d00e9599a3436fa1791e8a5"),
            ("ev-har-r15", "0xdd7e3581c3acac711476e40ae561fd76c19bcc19a55761878249e915ec290c3c"),
        ),
    },
)
PRIMARY_HORIZONS = (1, 4)  # the pre-registered primary questions (docs/judging/r1.md)

FetchClose = Callable[[str, int], float]  # (asset, hour start) -> close of the last candle opening at or before it

log = logging.getLogger(__name__)


def settlement_name(horizon: int) -> str:
    return f"settlement-{horizon}h.json"


def closes(manifest: dict[str, Any]) -> dict[str, float]:
    return {a: float(c[CLOSE]) for a, c in manifest["inputs"]["last_candle"].items()}


@dataclass(frozen=True)
class Settlement:
    issue: int
    horizon: int
    source: str  # "issue <n>" when the settling prices are committed, else "coinbase"
    start: dict[str, float]
    end: dict[str, float]
    bins: dict[str, int]
    scores: dict[str, dict[str, float]]  # forecaster name -> asset -> log score in nats
    release: str = ""
    ids: dict[str, str] = field(default_factory=dict)  # forecaster name -> identity

    def to_json(self) -> bytes:
        returns = {a: math.log(self.end[a] / self.start[a]) for a in self.start}
        return canonical(
            {
                "schema": SCHEMA,
                "issue": self.issue,
                "horizon": self.horizon,
                "source": self.source,
                "release": self.release,
                "ids": self.ids,
                "start": self.start,
                "end": self.end,
                "returns": returns,
                "bins": self.bins,
                "scores": self.scores,
            }
        )

    @classmethod
    def from_json(cls, data: bytes) -> Settlement:
        b = json.loads(data)
        return cls(
            b["issue"], b["horizon"], b["source"], b["start"], b["end"], b["bins"], b["scores"], b["release"], b["ids"]
        )


def settle(
    manifest: dict[str, Any], forecasts: dict[str, Any], horizon: int, end: dict[str, float], source: str
) -> Settlement:
    """Score an issue's forecasts at ``horizon`` given settling closes for (at least) the issue's assets."""
    assets = manifest["feed"]["assets"]
    if horizon not in manifest["feed"]["horizons"]:
        raise ValueError(f"issue {manifest['issue']} has no {horizon}h questions")
    missing = [a for a in assets if a not in end]
    if missing:
        raise ValueError(f"no settling price for {', '.join(missing)}")
    start = closes(manifest)
    end = {a: float(end[a]) for a in assets}  # a later issue may list more assets; only the issue's own are settled
    bins = {a: int(bin_index(np.array([math.log(end[a] / start[a])]), horizon)[0]) for a in assets}
    scores: dict[str, dict[str, float]] = {}
    for f in forecasts["forecasts"]:
        if f["horizon"] == horizon:
            scores.setdefault(f["forecaster"], {})[f["asset"]] = math.log(f["bps"][bins[f["asset"]]] / TOTAL)
    ids = {x["name"]: x["id"] for x in manifest["forecasters"] if x["name"] in scores}
    return Settlement(manifest["issue"], horizon, source, start, end, bins, scores, manifest["release"], ids)


def _committed(issue_dir: Path) -> bool:
    return (issue_dir / "manifest.json").exists() and (issue_dir / "commitment.json").exists()


def _settling_prices(
    issues_dir: Path, assets: Sequence[str], due: int, current: int, fetch_close: FetchClose
) -> tuple[dict[str, float], str] | None:
    """The closes a question due at issue ``due`` settles on, or None while it must wait."""
    later = issues_dir / str(due)
    if _committed(later):
        prices = closes(json.loads((later / "manifest.json").read_bytes()))
        if all(a in prices for a in assets):
            return prices, f"issue {due}"
    elif due >= current:
        return None  # the current issue may still be committed in this hour
    return {a: fetch_close(a, (due - 1) * 3600) for a in assets}, "coinbase"


def settle_due(
    issues_dir: Path, current: int, fetch_close: FetchClose, *, problems: list[tuple[str, str]] | None = None
) -> list[Path]:
    """Write every settlement that has become due for committed issues, however old.

    Issue N at horizon h is due once issue N + h has started. A question that cannot be settled now (the exchange is
    unreachable, say) is logged, reported in ``problems`` as (key, message), and retried on the next run; it never
    blocks the others.
    """
    written = []
    dirs = sorted((d for d in issues_dir.iterdir() if d.name.isdigit()), key=lambda d: int(d.name))
    for issue_dir in dirs:
        issue = int(issue_dir.name)
        if issue >= current or not _committed(issue_dir):
            continue
        manifest: dict[str, Any] | None = None
        forecasts: dict[str, Any] | None = None
        for h in HORIZONS:
            target = issue_dir / settlement_name(h)
            if target.exists() or issue + h > current:
                continue
            try:
                manifest = manifest or json.loads((issue_dir / "manifest.json").read_bytes())
                if h not in manifest["feed"]["horizons"]:
                    continue
                prices = _settling_prices(issues_dir, manifest["feed"]["assets"], issue + h, current, fetch_close)
                if prices is None:
                    continue
                forecasts = forecasts or json.loads((issue_dir / "forecasts.json").read_bytes())
                write_atomic(target, settle(manifest, forecasts, h, *prices).to_json())
                written.append(target)
            except Exception as exc:
                log.exception("issue %d: could not settle the %dh questions yet", issue, h)
                if problems is not None:
                    problems.append((f"settle:{issue}:{h}", f"issue {issue}: {h}h questions not settled yet ({exc})"))
    return written


def load_settlements(issues_dir: Path) -> list[Settlement]:
    return [Settlement.from_json(p.read_bytes()) for p in sorted(issues_dir.glob("*/settlement-*h.json"))]


def _comparison(series: dict[int, float]) -> dict[str, Any]:
    """Mean of a per-issue series and its interval, in issue order; no interval for short records."""
    values = [series[i] for i in sorted(series)]
    ci = None
    if len(values) >= MIN_BLOCKS * BLOCK_ISSUES:
        ci = list(bootstrap.interval(values, BLOCK_ISSUES))
    return {"mean": fmean(values), "interval95": ci, "issues": len(values)}


def _primary(diffs: dict[int, dict[int, list[float]]], expected: dict[tuple[int, int], int]) -> dict[str, Any]:
    """The pre-registered statistic: per issue, the mean difference over every asset at 1 and 4 hours, counted only
    when all of those questions settled for both sides; the gain over issues and its z at both block lengths."""
    series = {
        issue: fmean(d for h in PRIMARY_HORIZONS for d in by_h[h])
        for issue, by_h in diffs.items()
        if all(h in by_h and len(by_h[h]) == expected[(issue, h)] for h in PRIMARY_HORIZONS)
    }
    values = [series[i] for i in sorted(series)]
    z: dict[str, float | None] = {}
    for block in (BLOCK_ISSUES, CHECK_BLOCK_ISSUES):
        se = bootstrap.standard_error(values, block) if len(values) >= MIN_BLOCKS * block else 0.0
        z[str(block)] = fmean(values) / se if se > 0 else None
    scope = "+".join(f"{h}h" for h in PRIMARY_HORIZONS)
    return {"scope": scope, "mean": fmean(values) if values else None, "z": z, "issues": len(values)}


def rule_statistics(
    scores: dict[str, dict[tuple[int, int, str], float]],
    expected: dict[tuple[int, int], int],
    rules: Sequence[dict[str, Any]] = RULES,
) -> list[dict[str, Any]]:
    """Each rule's statistic so far: the judged forecaster against each opponent, by identity, on the issues from
    the rule's start, as ``_primary`` computes it. Labelled as running: no verdict before a scheduled look."""
    out = []
    for rule in rules:
        name, judged = rule["judged"]
        own = scores.get(judged, {})
        entry: dict[str, Any] = {
            "rule": rule["rule"],
            "start_issue": rule["start_issue"],
            "judged": {"name": name, "id": judged},
            "status": "running, unresolved",
            "vs": {},
        }
        for opponent, ident in rule["opponents"]:
            other = scores.get(ident, {})
            diffs: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
            for (issue, h, asset), value in own.items():
                if issue >= rule["start_issue"] and (issue, h, asset) in other:
                    diffs[issue][h].append(value - other[(issue, h, asset)])
            entry["vs"][opponent] = {"id": ident, **_primary(diffs, expected)}
        out.append(entry)
    return out


def scoreboard(settlements: Sequence[Settlement], *, baselines: Iterable[str] = BASELINES) -> dict[str, Any]:
    """Per forecaster identity: questions, mean log score overall and by horizon, and paired gains over each baseline.

    A gain is the mean over issues (each weighted equally) of the issue's mean score difference on the questions both
    answered. Overall gains use only issues whose every horizon has settled for both sides, so a young issue with only
    its 1-hour questions settled does not count as a whole issue; per-horizon gains use every settled issue.
    """
    # identity -> (issue, horizon, asset) -> score; names and releases kept for display and pairing
    scores: dict[str, dict[tuple[int, int, str], float]] = defaultdict(dict)
    labels: dict[str, tuple[str, str]] = {}
    by_name: dict[tuple[int, str], str] = {}  # (issue, name) -> identity, to pair with that issue's baselines
    expected = {(s.issue, s.horizon): len(s.bins) for s in settlements}  # questions per issue and horizon
    for s in settlements:
        for name, by_asset in s.scores.items():
            ident = s.ids.get(name, f"{s.release}:{name}")
            labels[ident] = (name, s.release)
            by_name[(s.issue, name)] = ident
            for asset, value in by_asset.items():
                scores[ident][(s.issue, s.horizon, asset)] = value
    issues = sorted({s.issue for s in settlements})
    board: dict[str, Any] = {
        "schema": BOARD_SCHEMA,
        "first_issue": issues[0] if issues else None,
        "last_issue": issues[-1] if issues else None,
        "settled_issues": len(issues),
        "block_issues": BLOCK_ISSUES,
        "forecasters": {},
    }
    board["rules"] = rule_statistics(scores, expected)
    for ident, own in sorted(scores.items(), key=lambda kv: labels[kv[0]]):
        name, release = labels[ident]
        entry: dict[str, Any] = {
            "name": name,
            "release": release,
            "questions": len(own),
            "mean_log_score": fmean(own.values()),
            "by_horizon": {
                str(h): fmean(v for (_, hh, _), v in own.items() if hh == h)
                for h in HORIZONS
                if any(hh == h for (_, hh, _) in own)
            },
            "vs": {},
        }
        for base in baselines:
            if base == name:
                continue
            diffs: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))  # issue -> h -> diffs
            for (issue, h, asset), value in own.items():
                other = by_name.get((issue, base))
                if other is not None and (issue, h, asset) in scores[other]:
                    diffs[issue][h].append(value - scores[other][(issue, h, asset)])
            if not diffs:
                continue
            complete = {
                i: fmean(d for v in by_h.values() for d in v) for i, by_h in diffs.items() if len(by_h) == len(HORIZONS)
            }
            result: dict[str, Any] = (
                _comparison(complete) if complete else {"mean": None, "interval95": None, "issues": 0}
            )
            result["primary"] = _primary(diffs, expected)
            result["by_horizon"] = {
                str(h): _comparison({i: fmean(by_h[h]) for i, by_h in diffs.items() if h in by_h})
                for h in HORIZONS
                if any(h in by_h for by_h in diffs.values())
            }
            entry["vs"][base] = result
        board["forecasters"][ident] = entry
    return board
