"""The scheduled looks of the pre-registered rules (``docs/judging``): each rule's statistic on a look's issues, and
what it may claim.

Look d of a rule uses the issues of the first d days of its record, ``start`` to ``start + 24 d - 1``, once their
4-hour questions have settled, and is computed after the first midnight (UTC) that follows. Per issue, the statistic
is the mean over every asset and the 1- and 4-hour questions of the judged forecaster's score minus the opponent's;
an issue counts once both horizons have settled for both of them on all of its questions, and every other issue of
the window is left out and counted. The gain is the mean over issues, and z divides it by the standard deviation of
2,000 circular block bootstrap means (seed 0) with blocks of 72 and of 168 issues. A verdict needs z above the look's
threshold with both. Day 30 reports a direction only.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from statistics import fmean
from typing import Any

from laplace_engine import bootstrap
from laplace_engine.settle import BLOCK_ISSUES, CHECK_BLOCK_ISSUES, MIN_BLOCKS, PRIMARY_HORIZONS, RULES, Settlement
from laplace_research.evolve.feed import HORIZONS

LOOKS: dict[int, float | None] = {30: None, 90: 4.36, 180: 2.99, 365: 1.97}  # day -> z needed (30: direction only)
# What each rule may claim, and the opponents a claim needs (docs/judging).
CLAIMS: dict[str, dict[str, tuple[str, ...]]] = {
    "docs/judging/r1.md": {"the Demon beats the classical baselines": ("garch", "har")},
    "docs/judging/r2.md": {
        "A: EV-GBM beats the classical models": ("garch-r15", "har-r15"),
        "B: EV-GBM beats EV-HAR": ("ev-har-r15",),
    },
}
HOUR = 3600


def rule_named(name: str) -> dict[str, Any]:
    """A rule of ``settle.RULES`` by its file's stem (``r1``) or path."""
    for rule in RULES:
        if rule["rule"] in (name, f"docs/judging/{name}.md"):
            return rule
    raise KeyError(f"no pre-registered rule {name!r}")


def issues_of(rule: dict[str, Any], day: int) -> range:
    return range(rule["start_issue"], rule["start_issue"] + 24 * day)


def computed_after(rule: dict[str, Any], day: int) -> datetime:
    """The first midnight (UTC) after the 4-hour questions of the look's last issue settle."""
    last = issues_of(rule, day)[-1]
    settled = datetime.fromtimestamp((last + 4) * HOUR, UTC)
    midnight = settled.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight if midnight == settled else midnight + timedelta(days=1)


def _z(values: list[float], block: int) -> float | None:
    if len(values) < MIN_BLOCKS * block:  # as on the scoreboard: no z before two whole blocks
        return None
    se = bootstrap.standard_error(values, block)
    return fmean(values) / se if se > 0 else None


def _verdict(z: dict[str, float | None], needed: float | None) -> str:
    if needed is None:
        return "direction only"
    if all(v is not None and v >= needed for v in z.values()):
        return "beats"
    if all(v is not None and -v >= needed for v in z.values()):
        return "is beaten by"
    return "no claim at this look"


def _series(
    diffs: dict[int, dict[int, list[float]]],
    expected: dict[tuple[int, int], int],
    window: range,
    horizons: Sequence[int],
) -> list[float]:
    """Per issue of the window, in order, the mean difference at ``horizons``, for the issues where every question at
    those horizons settled for both forecasters; the others are left out."""
    out = []
    for issue in window:
        by_h = diffs.get(issue, {})
        if all(h in by_h and len(by_h[h]) == expected.get((issue, h), -1) for h in horizons):
            out.append(fmean(d for h in horizons for d in by_h[h]))
    return out


def look(settlements: Sequence[Settlement], rule: dict[str, Any], day: int) -> dict[str, Any]:
    """Look ``day`` of ``rule``: per opponent, the gain, both z values, the issues used and left out, and the
    verdict; then the rule's claims, and what is reported but never claimed (24 hours, all horizons, each horizon)."""
    window = issues_of(rule, day)
    judged = rule["judged"][1]
    wanted = {judged} | {ident for _, ident in rule["opponents"]}
    scores: dict[str, dict[tuple[int, int, str], float]] = defaultdict(dict)
    expected: dict[tuple[int, int], int] = {}
    for s in settlements:
        if s.issue not in window:
            continue
        expected[(s.issue, s.horizon)] = len(s.bins)
        for name, by_asset in s.scores.items():
            ident = s.ids.get(name)
            if ident is not None and ident in wanted:
                for asset, value in by_asset.items():
                    scores[ident][(s.issue, s.horizon, asset)] = value
    needed = LOOKS[day]
    out: dict[str, Any] = {
        "rule": rule["rule"],
        "day": day,
        "issues": [window[0], window[-1]],
        "computed_after": computed_after(rule, day).strftime("%Y-%m-%d %H:%M UTC"),
        "z_needed": needed,
        "judged": {"name": rule["judged"][0], "id": judged},
        "vs": {},
    }
    for opponent, ident in rule["opponents"]:
        diffs: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for (issue, h, asset), value in scores[judged].items():
            if (issue, h, asset) in scores[ident]:
                diffs[issue][h].append(value - scores[ident][(issue, h, asset)])

        primary = _series(diffs, expected, window, PRIMARY_HORIZONS)
        z = {str(b): _z(primary, b) for b in (BLOCK_ISSUES, CHECK_BLOCK_ISSUES)}
        reported = {}
        for scope, horizons in {"24h": (24,), "all": tuple(HORIZONS), **{f"{h}h": (h,) for h in (1, 4)}}.items():
            values = _series(diffs, expected, window, horizons)
            reported[scope] = {"gain": fmean(values) if values else None, "issues": len(values)}
        out["vs"][opponent] = {
            "id": ident,
            "gain": fmean(primary) if primary else None,
            "z": z,
            "issues_used": len(primary),
            "issues_left_out": len(window) - len(primary),
            "verdict": _verdict(z, needed),
            "reported_never_claimed": reported,
        }
    out["claims"] = {
        claim: (
            "direction only"
            if needed is None
            else "made"
            if all(out["vs"][o]["verdict"] == "beats" for o in opponents)
            else "not made at this look"
        )
        for claim, opponents in CLAIMS[rule["rule"]].items()
    }
    return out


def render(result: dict[str, Any], *, preview: bool = False) -> str:
    title = f"{result['rule']}, day {result['day']}" + (" (a preview, not the look)" if preview else "")
    lines = [
        f"# {title}",
        "",
        f"Issues {result['issues'][0]} to {result['issues'][1]}; computed after {result['computed_after']}. "
        + (
            f"A verdict needs z >= {result['z_needed']} with both block lengths."
            if result["z_needed"]
            else "A direction report: no verdict either way."
        ),
        "",
        "| opponent | gain (1h + 4h) | z (72) | z (168) | issues used | left out | verdict |",
        "|---|---|---|---|---|---|---|",
    ]

    def num(v: float | None, fmt: str) -> str:
        return "" if v is None else format(v, fmt)

    for opponent, r in result["vs"].items():
        lines.append(
            f"| {opponent} | {num(r['gain'], '+.4f')} | {num(r['z']['72'], '.2f')} | {num(r['z']['168'], '.2f')} | "
            f"{r['issues_used']} | {r['issues_left_out']} | {r['verdict']} |"
        )
    lines += ["", "| claim | status |", "|---|---|"]
    lines += [f"| {claim} | {status} |" for claim, status in result["claims"].items()]
    lines += ["", "Reported, never claimed (gain, issues):", ""]
    for opponent, r in result["vs"].items():
        cells = ", ".join(
            f"{scope} {num(v['gain'], '+.4f')} ({v['issues']})" for scope, v in r["reported_never_claimed"].items()
        )
        lines.append(f"- {opponent}: {cells}")
    return "\n".join(lines) + "\n"
