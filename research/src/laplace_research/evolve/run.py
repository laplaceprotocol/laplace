"""Set up and run an evolution: panel, splits, islands, and the generation loop."""

from __future__ import annotations

import asyncio
import json
import math
import random
import sqlite3
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt

from laplace_research.data.events import load as load_calendar
from laplace_research.evolve import panel as panel_mod
from laplace_research.evolve.baselines import (
    ev_har,
    hourly_returns,
    iv_har,
    naive,
    overlay_features,
    overlay_names,
    trailing_mean,
)
from laplace_research.evolve.db import Database, Program
from laplace_research.evolve.evaluate import (
    Evaluator,
    Split,
    breakdown,
    log_scores,
    mean_gain,
    realized_bins,
    sample_times,
)
from laplace_research.evolve.feed import HORIZONS, FloatArray, IntArray, finalize
from laplace_research.evolve.islands import ISLANDS, SEED_PROGRAM
from laplace_research.evolve.mutate import MUTATORS, NoProgram, Scored, build_prompt, propose
from laplace_research.evolve.panel import Panel
from laplace_research.inference import AuthError, BudgetExceeded, Credentials, InferenceError, WorkersAI

TIME_LIMIT_S = 90
TIME_LIMIT_HOURS = 506  # the forecast hours TIME_LIMIT_S covers: PoC 2's search sample, 10,626 program calls
MODEL_CONCURRENCY = {
    "@cf/deepseek-ai/deepseek-v4.1-flash-1017": 4,
    "@cf/zai-org/glm-5.3": 6,
    "@cf/zai-org/glm-5.3-flash": 8,
}


def parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def time_limit_s(hours: int) -> int:
    """The time limit for evaluating a program at ``hours`` forecast hours: TIME_LIMIT_S per TIME_LIMIT_HOURS (about
    8 ms a call), so a denser sample does not tighten a program's budget."""
    return max(TIME_LIMIT_S, math.ceil(TIME_LIMIT_S * hours / TIME_LIMIT_HOURS))


@dataclass(frozen=True)
class RunConfig:
    """Time windows (UTC, ``[start, end)``) and sampling steps. Steps of 23 and 5 hours rotate through the clock."""

    data_start: str
    data_end: str
    search: tuple[str, str, int]
    validation: tuple[str, str, int]
    holdout: tuple[str, str, int]
    baseline_fit_step: int = 2
    seed: int = 0
    reference: str = "naive"  # what a program's gain is measured against: "naive" or "ev-har"

    @classmethod
    def default(cls, now: datetime) -> RunConfig:
        end = now.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")
        return cls(
            data_start="2025-01-01T00:00",
            data_end=end,
            search=("2025-02-01T00:00", "2026-06-01T00:00", 23),
            validation=("2026-06-01T00:00", "2026-08-16T00:00", 5),
            holdout=("2026-08-16T00:00", end, 5),  # a step coprime with 24 rotates through the clock
        )

    def split(self, panel: Panel, name: str) -> Split:
        start, end, step = getattr(self, name)
        return Split(name, sample_times(panel, parse_utc(start), parse_utc(end), step))

    def fit_times(self, panel: Panel) -> Split:
        start, end, _ = self.search
        return Split("fit", sample_times(panel, parse_utc(start), parse_utc(end), self.baseline_fit_step))


REFERENCES = {"naive": "the naive baseline", "ev-har": "EV-HAR"}


@dataclass(frozen=True)
class Score:
    """A program's result on the search window: either a gain over the run's reference or an error."""

    gain: float | None = None
    breakdown: dict[str, float] | None = None
    error: str | None = None
    eval_seconds: float | None = None


class Run:
    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir
        self.config = RunConfig(**json.loads((run_dir / "config.json").read_text()))
        self.panel = Panel.load(run_dir / "panel.npz")
        self.db = Database(run_dir / "programs.sqlite")
        self.search = self.config.split(self.panel, "search")
        self.search_realized = realized_bins(self.panel, self.search.times)
        if self.config.reference not in REFERENCES:
            raise ValueError(f"unknown reference {self.config.reference!r}")
        self.reference = REFERENCES[self.config.reference]
        self.search_reference = self._reference_scores()
        self.segments = segments(self.panel, self.search.times, self.search_realized)

    def _reference_scores(self) -> FloatArray:
        """The reference's log scores on the search window; EV-HAR's (fitted on the search window, as the baselines
        are) are cached next to the run, since fitting it takes minutes."""
        times = self.search.times
        if self.config.reference == "naive":
            return log_scores(finalize(naive(self.panel.close).probs(times)), self.search_realized)
        cache = self.dir / "reference-ev-har.npz"
        if cache.exists():
            with np.load(cache) as data:
                if np.array_equal(data["times"], times):
                    out: FloatArray = data["scores"]
                    return out
        fit = self.config.fit_times(self.panel)
        realized_fit = realized_bins(self.panel, fit.times)
        senses = self.panel.senses
        iv = iv_har(self.panel.close, senses["rv"], senses["dvol"], self.panel.assets, realized_fit, fit.times)
        ev = ev_har(iv, self.panel.close, self.panel.ts, load_calendar(), fit.times)
        scores: FloatArray = log_scores(finalize(ev.probs(times)), self.search_realized)
        np.savez(cache, times=times, scores=scores)
        return scores

    @staticmethod
    def init(
        run_dir: Path,
        cache_dir: Path,
        *,
        seed_code: str = SEED_PROGRAM,
        reference: str = "naive",
        panel: Panel | None = None,
        islands: Mapping[str, str] = ISLANDS,
        search_step: int | None = None,
    ) -> Run:
        """A new run: one island per ``islands`` hypothesis, each seeded with ``seed_code`` and scored against
        ``reference``, on a fresh panel or on ``panel`` (another run's, senses included: a reference or seed that reads
        senses needs them). ``search_step`` overrides the hours between search samples: a denser sample measures small
        gains more precisely and costs evaluation time only."""
        if not islands:
            raise ValueError("a run needs at least one island")
        if search_step is not None and search_step < 1:
            raise ValueError(f"search step must be at least 1 hour, not {search_step}")
        if (run_dir / "config.json").exists():
            raise FileExistsError(f"{run_dir} already initialized")
        run_dir.mkdir(parents=True)
        if panel is None:
            end = datetime.now(UTC) - timedelta(hours=1)
        else:
            end = datetime.fromtimestamp(int(panel.ts[-1]) + 3600, UTC)
        config = replace(RunConfig.default(end), reference=reference)
        if search_step is not None:
            config = replace(config, search=(config.search[0], config.search[1], search_step))
        if panel is None:
            panel = panel_mod.build(parse_utc(config.data_start), parse_utc(config.data_end), cache_dir)
        panel.save(run_dir / "panel.npz")
        (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=1))
        run = Run(run_dir)
        with Evaluator(run_dir / "panel.npz", workers=1) as evaluator:
            seed = run.score(evaluator, seed_code)
        if seed.error is not None:
            raise RuntimeError(f"the seed program failed: {seed.error}")
        for name, directive in islands.items():
            run.db.add_program(
                island_id=run.db.add_island(name, directive),
                parent_id=None,
                generation=0,
                mutator="seed",
                code=seed_code,
                gain=seed.gain,
                breakdown=seed.breakdown,
                eval_seconds=seed.eval_seconds,
            )
        return run

    def score(self, evaluator: Evaluator, code: str) -> Score:
        outcome = evaluator.run(code, self.search.times, time_limit_s(len(self.search.times)))
        if outcome.probs is None:
            return Score(error=outcome.error, eval_seconds=outcome.seconds)
        log_p = log_scores(finalize(outcome.probs), self.search_realized)
        return Score(
            gain=mean_gain(log_p, self.search_reference),
            breakdown=breakdown(log_p, self.search_reference, self.panel.assets, self.segments),
            eval_seconds=outcome.seconds,
        )

    async def evolve(self, generations: int, children: int, workers: int, max_usd: float | None = None) -> None:
        """Run ``generations`` more generations. With ``max_usd``, no generation starts once the run's recorded cost
        (every generation so far, earlier invocations included) has reached it; one already running finishes. A run
        whose holdout has been measured (``evolve.measure``) evolves no further: its holdout is scored once."""
        if (self.dir / "measure.json").exists():
            raise RuntimeError(f"{self.dir.name}'s holdout has been measured; start a new run to evolve further")
        islands = self.db.islands()
        start = self.db.last_generation() + 1
        if self._spent(max_usd):
            return
        with Evaluator(self.dir / "panel.npz", workers=workers) as evaluator:
            async with WorkersAI(Credentials.from_env(), concurrency=16, model_limits=MODEL_CONCURRENCY) as ai:
                for g in range(start, start + generations):
                    if g > start and self._spent(max_usd):
                        return
                    jobs = [
                        self._child(ai, evaluator, island, index, g, c)
                        for index, island in enumerate(islands)
                        for c in range(children)
                    ]
                    gains = [x for x in await asyncio.gather(*jobs) if x is not None]
                    meta = self.db.meta()
                    best = f"{max(gains):+.4f}" if gains else "none"
                    print(
                        f"generation {g}: {len(gains)}/{len(jobs)} valid, best child {best}, "
                        f"{meta['programs']} programs, ${meta['cost_usd']:.2f} spent",
                        file=sys.stderr,
                        flush=True,
                    )

    def _spent(self, max_usd: float | None) -> bool:
        """Whether the recorded cost has reached ``max_usd`` (never, without a cap)."""
        if max_usd is None:
            return False
        cost = float(self.db.meta()["cost_usd"])
        if cost < max_usd:
            return False
        print(f"stopping: ${cost:.2f} recorded, the cap is ${max_usd:.2f}", file=sys.stderr, flush=True)
        return True

    async def _child(
        self, ai: WorkersAI, evaluator: Evaluator, island: sqlite3.Row, index: int, g: int, c: int
    ) -> float | None:
        rng = random.Random(self.config.seed * 1_000_003 + g * 1009 + index * 31 + c)
        island_id = int(island["id"])
        top = self.db.best(island_id, 4)
        parent = top[0] if rng.random() < 0.6 or len(top) == 1 else rng.choice(top)
        inspirations: list[Program] = []
        others = [i for i in self.db.islands() if i["id"] != island_id]
        if others and rng.random() < 0.5:
            other_best = self.db.best(int(rng.choice(others)["id"]), 1)
            inspirations = [p for p in other_best if p.code != parent.code]
        mutator = MUTATORS[_pick_mutator(g, c, index)]
        prompt = build_prompt(
            island["directive"],
            _scored(parent, self.reference),
            [_scored(p, self.reference) for p in inspirations],
            self.db.recent_failures(island_id, 3),
        )
        record = partial(
            self.db.add_program, island_id=island_id, parent_id=parent.id, generation=g, mutator=mutator.name
        )
        tags: dict[str, str | int] = {
            "run": self.dir.name,
            "island": str(island["name"]),
            "generation": g,
            "mutator": mutator.name,
        }
        try:
            proposal = await propose(ai, mutator, prompt, tags)
        except (AuthError, BudgetExceeded):
            raise
        except NoProgram as exc:  # answered but unusable: billed, so its cost is recorded
            record(code="", error=f"mutation: {exc}"[:400], cost_usd=exc.cost_usd)
            return None
        except (InferenceError, KeyError) as exc:  # no usable answer; any billing is unknown here
            record(code="", error=f"mutation: {exc}"[:400])
            return None
        if self.db.has_code(island_id, proposal.code):
            record(code=proposal.code, title=proposal.title, error="mutation: duplicate", cost_usd=proposal.cost_usd)
            return None
        score = await asyncio.to_thread(self.score, evaluator, proposal.code)
        record(
            code=proposal.code,
            title=proposal.title,
            why=proposal.why,
            cost_usd=proposal.cost_usd,
            gain=score.gain,
            breakdown=score.breakdown,
            error=score.error,
            eval_seconds=score.eval_seconds,
        )
        return score.gain


def segments(panel: Panel, times: IntArray, realized: IntArray) -> dict[str, npt.NDArray[np.bool_]]:
    """Where a forecaster gains or loses, as ``[asset, horizon, sample]`` masks over forecasts made at ``times``: by the
    week's volatility (calm, normal and shock thirds per asset over these times), around scheduled releases, in the US
    session, at weekends, and on questions whose outcome (``realized``, their bins) landed in an outer bin."""
    shape = realized.shape
    week = trailing_mean(hourly_returns(panel.close) ** 2, 168)[:, times]
    low, high = np.quantile(week, [1 / 3, 2 / 3], axis=1, keepdims=True)
    out: dict[str, npt.NDArray[np.bool_]] = {
        "calm": np.broadcast_to((week <= low)[:, None, :], shape),
        "normal": np.broadcast_to(((week > low) & (week < high))[:, None, :], shape),
        "shock": np.broadcast_to((week >= high)[:, None, :], shape),
    }
    calendar = load_calendar()
    at = panel.ts[times] + 3600
    releases = np.zeros(shape, dtype=bool)
    session = np.zeros(shape, dtype=bool)
    for j, h in enumerate(HORIZONS):
        x = overlay_features(calendar, at, h)
        names = overlay_names(h)
        near = x[:, names.index("releases")] + x[:, names.index("recent-releases")] > 0
        releases[:, j, :] = near[None, :]
        session[:, j, :] = (x[:, names.index("us-session")] > 0)[None, :]
    out["releases"] = releases
    out["us session"] = session
    out["weekend"] = np.broadcast_to((((at // 86_400) + 3) % 7 >= 5)[None, None, :], shape)
    out["outer bins"] = np.isin(realized, (0, 1, 8, 9))
    return out


def _scored(p: Program, reference: str) -> Scored:
    """What a mutator is shown of a program's score: per horizon, and per segment where the run records them."""
    assert p.gain is not None
    keep = {k: v for k, v in p.breakdown.items() if k.startswith(("h", "seg:"))}
    return Scored(p.code, p.gain, keep, reference)


def _pick_mutator(g: int, c: int, island_index: int) -> str:
    """Every fourth generation the first child of each island gets the deep model; the rest alternate cheap ones."""
    if c == 0 and g % 4 == 3:
        return "glm-5.3"
    return "glm-5.3-flash" if (island_index + c + g) % 2 == 0 else "deepseek-v4.1-flash"
