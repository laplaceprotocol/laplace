import numpy as np
import pytest

from laplace_research.data.events import load as load_calendar
from laplace_research.evolve import baselines as b
from laplace_research.evolve import gbm, learned, params
from laplace_research.evolve.evaluate import realized_bins, sample_times
from laplace_research.evolve.feed import HORIZONS, N_BINS
from laplace_research.evolve.panel import Panel, utc

from .test_learned import sensed

lgb = pytest.importorskip("lightgbm")


def test_exported_trees_walk_to_exactly_what_lightgbm_predicts() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3000, 6))
    y = np.digitize(x[:, 0] + 0.5 * x[:, 1] * rng.normal(size=3000), np.linspace(-2, 2, N_BINS - 1))
    init = np.log(np.full((3000, N_BINS), 1 / N_BINS))
    booster = lgb.train(
        gbm.SETTINGS | {"min_data_in_leaf": 20}, lgb.Dataset(x, label=y, init_score=init), num_boost_round=30
    )
    for rounds in (30, 7):
        trees = gbm.export(booster, rounds)
        assert len(trees.root) == rounds * N_BINS
        expected = booster.predict(x, raw_score=True, num_iteration=rounds)
        assert np.abs(gbm.walk(trees, x) - expected).max() < 1e-12


def test_no_rounds_is_ev_har_itself() -> None:
    trees = gbm.export(None, 0)
    assert np.array_equal(gbm.walk(trees, np.zeros((4, len(gbm.NAMES)))), np.zeros((4, N_BINS)))


def test_ev_gbm_starts_from_ev_har_and_its_parameter_file_reproduces_it(panel: Panel) -> None:
    senses = sensed(panel)
    fit = sample_times(panel, utc(int(panel.ts[0])), utc(int(panel.ts[900])), 3)
    rungs = b.ladder(
        panel.close, realized_bins(panel, fit), fit, slice(0, 900), senses=senses, assets=panel.assets, ts=panel.ts
    )
    ev = rungs[-1]
    pw = learned.Panelwise.of(panel.close, panel.high, panel.low, senses, panel.assets)
    model = gbm.ev_gbm(
        ev,
        pw,
        load_calendar(),
        panel.ts,
        np.arange(720, 900),
        np.arange(900, 1000),
        lambda times: realized_bins(panel, times),
    )
    times = np.arange(1000, 1010)
    p = model.probs(times)
    assert p.shape == (len(panel.assets), len(HORIZONS), 10, N_BINS)
    assert np.allclose(p.sum(axis=-1), 1)
    assert set(model.params["gbm"]["rounds"]) == {"1", "4", "24"}
    assert model.params["inputs"] == list(gbm.NAMES)
    stored = params.decode(params.encode(model.arrays()))
    again = gbm.Gbm("ev-gbm", ev, pw, load_calendar(), panel.ts, {h: gbm.Trees.of(stored, f"h{h}") for h in HORIZONS})
    assert np.array_equal(again.probs(times), p)


def test_design_adds_the_asset_and_the_raw_clock(panel: Panel) -> None:
    pw = learned.Panelwise.of(panel.close, panel.high, panel.low, sensed(panel), panel.assets)
    times = np.array([900, 1000], dtype=np.int64)
    x = gbm.design(pw, load_calendar(), panel.ts, times, np.full((len(panel.assets), 2), 0.01), 4)
    assert x.shape == (len(panel.assets), 2, len(gbm.NAMES))
    at = panel.ts[1000] + 3600
    assert x[3, 1, gbm.NAMES.index("asset")] == 3
    assert x[3, 1, gbm.NAMES.index("hour")] == (at // 3600) % 24
    assert x[3, 1, gbm.NAMES.index("weekday")] == utc(int(at)).weekday()
