import numpy as np
import pytest

from laplace_engine.baselines import gbm_program, ladder_programs
from laplace_research.data.events import load as load_calendar
from laplace_research.evolve import baselines as b
from laplace_research.evolve import gbm, learned, params
from laplace_research.evolve.evaluate import Evaluator, realized_bins, sample_times
from laplace_research.evolve.feed import HORIZONS, N_BINS
from laplace_research.evolve.history import HistoryReader, ensure_history
from laplace_research.evolve.panel import Panel, utc
from laplace_research.evolve.sandbox import load


def random_trees(rng: np.random.Generator, x: np.ndarray, rounds: int, depth: int = 3) -> gbm.Trees:
    """Complete trees of ``depth`` splitting on random inputs at thresholds drawn from the rows ``x``."""
    root, feature, threshold, left, right, value = [], [], [], [], [], []
    for _ in range(rounds * N_BINS):
        root.append(len(feature))
        first = len(feature)
        inner = 2**depth - 1
        for k in range(inner):
            f = int(rng.integers(x.shape[1]))
            feature.append(f)
            threshold.append(float(x[rng.integers(len(x)), f]))
            for side, child in ((left, 2 * k + 1), (right, 2 * k + 2)):
                if child < inner:
                    side.append(first + child)
                else:
                    value.append(float(rng.normal(0, 0.3)))
                    side.append(~(len(value) - 1))
    as_int = lambda v: np.array(v, dtype=np.int32)  # noqa: E731
    return gbm.Trees(as_int(root), as_int(feature), np.array(threshold), as_int(left), as_int(right), np.array(value))


@pytest.fixture
def fitted(panel: Panel):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(5)
    r2 = b.hourly_returns(panel.close) ** 2
    dvol = np.full(panel.close.shape, np.nan)
    dvol[0] = 100 * np.sqrt(b.trailing_mean(r2[0], 168) * b.HOURS_PER_YEAR)
    senses = {"rv": r2 * rng.uniform(0.5, 1.5, r2.shape), "dvol": dvol, "funding": rng.normal(1e-4, 5e-5, r2.shape)}
    fit = sample_times(panel, utc(int(panel.ts[0])), utc(int(panel.ts[900])), 3)
    rungs = b.ladder(
        panel.close, realized_bins(panel, fit), fit, slice(0, 900), senses=senses, assets=panel.assets, ts=panel.ts
    )
    ev = rungs[-1]
    pw = learned.Panelwise.of(panel.close, panel.high, panel.low, senses, panel.assets)
    calendar = load_calendar()
    sample = np.arange(800, 1000, 7)
    trees = {}
    for j, h in enumerate(HORIZONS):
        x = gbm.design(pw, calendar, panel.ts, sample, ev.sd(sample)[:, j], h).reshape(-1, len(gbm.NAMES))
        trees[h] = random_trees(rng, x, rounds=4)
    model = gbm.Gbm("ev-gbm", ev, pw, calendar, panel.ts, trees)
    return panel.with_senses(senses), rungs, model


def test_ev_gbm_program_computes_research_trees_on_its_own_ev_har(fitted, tmp_path) -> None:  # type: ignore[no-untyped-def]
    panel, rungs, model = fitted
    ev = model.ev
    source = gbm_program(ev, panel.assets, "a synthetic window")
    assert len(source.encode()) < 16_000  # the sandbox's limit
    stored = params.decode(params.encode(model.arrays()))
    fn = load(source)
    ev_fn = load(ladder_programs(rungs, panel.assets, "a synthetic window")["ev-har"])
    namespace: dict[str, object] = {}
    exec(source, namespace)  # trusted: our own generated source, to reach its helpers
    ev_har, net_inputs, walk = namespace["_ev_har"], namespace["_net_inputs"], namespace["_walk"]

    path = tmp_path / "panel.npz"
    panel.save(path)
    reader = HistoryReader(*ensure_history(path))
    times = np.array([1060, 1100, 1150, 1198], dtype=np.int64)
    sds = ev.sd(times)
    gaps = []
    research = model.probs(times)
    for n, t in enumerate(times):
        reader.advance_to(int(t))
        for a in (0, 3):
            for j, h in enumerate(HORIZONS):
                ctx = reader.context(a, int(t), h, stored)
                # Given the same spread, the inputs and the trees are research's exactly.
                x = gbm.design(model.panelwise, model.calendar, panel.ts, np.array([t]), sds[:, j, n : n + 1], h)[a, 0]
                y = np.array([*net_inputs(ctx, float(sds[a, j, n])), a, ctx.hour, ctx.weekday])  # type: ignore[operator]
                assert y == pytest.approx(x, rel=1e-9, abs=1e-12), (t, a, h)
                expected = gbm.walk(model.trees[h], x[None, :])[0]
                assert np.abs(np.asarray(walk(stored, h, x)) - expected).max() < 1e-12  # type: ignore[operator]
                # The forecast is EV-HAR's own program, corrected by research's trees on the inputs at its spread.
                sd, base = ev_har(ctx)  # type: ignore[operator]
                ev_out = np.asarray(ev_fn(ctx))
                assert base == pytest.approx(learned.final(ev_out[None, :])[0], abs=1e-12)
                own = np.array([*net_inputs(ctx, sd), a, ctx.hour, ctx.weekday])  # type: ignore[operator]
                logits = np.log(base) + gbm.walk(model.trees[h], own[None, :])[0]
                assert np.asarray(fn(ctx)) == pytest.approx(learned.softmax(logits), abs=1e-12)
                gaps.append(float(np.abs(np.asarray(fn(ctx)) - research[a, j, n]).max()))
    # Where EV-HAR's program differs from research (its 720-hour mean covers 719 hours), a tree can take the other
    # branch; on these windows that stays small.
    assert np.median(gaps) < 1e-3


def test_the_evaluator_hands_a_program_its_parameter_file(fitted, tmp_path) -> None:  # type: ignore[no-untyped-def]
    panel, _, model = fitted
    path = tmp_path / "panel.npz"
    panel.save(path)
    source = gbm_program(model.ev, panel.assets, "a synthetic window")
    raw = params.encode(model.arrays())
    times = np.array([1100, 1198], dtype=np.int64)
    with Evaluator(path, workers=2) as evaluator:
        outcome = evaluator.submit(source, times, 60, raw).result()
        missing = evaluator.submit(source, times, 60).result()
        broken = evaluator.submit(source, times, 60, raw[:-1]).result()
        writer = evaluator.submit(
            "def forecast(ctx):\n    ctx.params['h1.value'][0] = 1.0\n    return [1.0] * 10\n", times, 60, raw
        ).result()
    assert outcome.error is None
    assert outcome.probs is not None
    fn, stored = load(source), params.decode(raw)
    reader = HistoryReader(*ensure_history(path))
    for n, t in enumerate(times):
        reader.advance_to(int(t))
        for a in range(len(panel.assets)):
            for j, h in enumerate(HORIZONS):
                assert outcome.probs[a, j, n] == pytest.approx(np.asarray(fn(reader.context(a, int(t), h, stored))))
    assert missing.probs is None
    assert "KeyError" in (missing.error or "")
    assert broken.probs is None
    assert (broken.error or "").startswith("params:")
    assert writer.probs is None
    assert "read-only" in (writer.error or "")
