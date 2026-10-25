import numpy as np
import pytest

from laplace_engine.baselines import NAIVE, ladder_programs
from laplace_research.evolve import baselines as b
from laplace_research.evolve.baselines import ladder
from laplace_research.evolve.context import LOOKBACK
from laplace_research.evolve.evaluate import realized_bins, sample_times
from laplace_research.evolve.feed import HORIZONS, edges
from laplace_research.evolve.history import HistoryReader, ensure_history, write_history
from laplace_research.evolve.islands import SEED_PROGRAM
from laplace_research.evolve.lib import student_t_probs
from laplace_research.evolve.panel import Panel, utc
from laplace_research.evolve.sandbox import load


def test_the_naive_baseline_is_the_seed_program() -> None:
    assert NAIVE == SEED_PROGRAM


def test_baseline_programs_reproduce_the_research_ladder(panel: Panel, tmp_path) -> None:  # type: ignore[no-untyped-def]
    fit = sample_times(panel, utc(int(panel.ts[0])), utc(int(panel.ts[800])), 3)
    rungs = ladder(panel.close, realized_bins(panel, fit), fit, slice(0, 800))
    sources = ladder_programs(rungs, panel.assets, "a synthetic window")
    assert list(sources) == ["naive", "ewma", "garch", "har"]

    path = tmp_path / "panel.npz"
    panel.save(path)
    reader = HistoryReader(*write_history(panel, path))
    t = len(panel.ts) - 1
    reader.advance_to(t)
    research = dict(zip(sources, (rung.probs(np.array([t])) for rung in rungs), strict=True))
    har = rungs[3]
    r2 = np.diff(np.log(panel.close[:, t + 1 - LOOKBACK : t + 1])) ** 2
    for name, source in sources.items():
        fn = load(source)
        for a in range(len(panel.assets)):
            for j, h in enumerate(HORIZONS):
                got = np.asarray(fn(reader.context(a, t, h)))
                if name != "har":
                    # Recursions restart from the window; after 719 steps the start no longer matters.
                    assert got == pytest.approx(research[name][a, j, 0], abs=1e-9), (name, a, h)
                    continue
                # HARs longest mean covers the windows 719 returns where research has 720: check the formula exactly.
                means = np.array([r2[a, -w:].mean() for w in har.params["windows"]])
                c = np.asarray(har.params["per_asset"][a][str(h)])
                variance = max(c[0] + float(c[1:] @ means) * h, 0.1 * means[2] * h)
                expected = student_t_probs(np.sqrt(variance), har.nu[h], edges(h))
                assert got == pytest.approx(expected, abs=1e-12), (name, a, h)


def test_sense_baselines_follow_their_research_formulas(panel: Panel, tmp_path) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(5)
    r2 = b.hourly_returns(panel.close) ** 2
    rv = r2 * rng.uniform(0.5, 1.5, r2.shape)
    rv[2, 1150] = np.nan  # an hour without 5-minute data falls back to the squared hourly return
    dvol = np.full(panel.close.shape, np.nan)
    dvol[0] = 100 * np.sqrt(b.trailing_mean(r2[0], 168) * b.HOURS_PER_YEAR)
    dvol[0, -1] = np.nan  # no index at the forecast hour: the month's realized variance stands in
    senses = {"rv": rv, "dvol": dvol}
    fit = sample_times(panel, utc(int(panel.ts[0])), utc(int(panel.ts[800])), 3)
    rungs = b.ladder(panel.close, realized_bins(panel, fit), fit, slice(0, 800), senses=senses, assets=panel.assets)
    sources = ladder_programs(rungs, panel.assets, "a synthetic window")
    assert list(sources)[-2:] == ["rv-har", "iv-har"]

    path = tmp_path / "panel.npz"
    panel.with_senses(senses).save(path)
    reader = HistoryReader(*ensure_history(path))
    t = len(panel.ts) - 1
    reader.advance_to(t)
    r = np.diff(np.log(panel.close[:, t + 1 - LOOKBACK : t + 1]))
    window_rv = rv[:, t + 2 - LOOKBACK : t + 1]
    filled = np.where(np.isfinite(window_rv), window_rv, r * r)
    for name in ("rv-har", "iv-har"):
        rung = next(x for x in rungs if x.name == name)
        fn = load(sources[name])
        for a in range(len(panel.assets)):
            means = np.array([filled[a, -w:].mean() for w in rung.params["windows"]])
            for h in HORIZONS:
                c = np.asarray(rung.params["per_asset"][a][str(h)])
                features = means
                if name == "iv-har":
                    implied = dvol[a, t] if np.isfinite(dvol[a, t]) else dvol[0, t]
                    implied = (implied / 100) ** 2 / b.HOURS_PER_YEAR if np.isfinite(implied) else means[3]
                    features = np.append(means, implied)
                variance = max(c[0] + float(c[1:] @ features) * h, 0.1 * means[2] * h)
                expected = student_t_probs(np.sqrt(variance), rung.nu[h], edges(h))
                got = np.asarray(fn(reader.context(a, t, h)))
                assert got == pytest.approx(expected, abs=1e-12), (name, a, h)
