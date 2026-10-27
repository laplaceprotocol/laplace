from datetime import UTC, datetime

import numpy as np
import pytest

from laplace_research.data.events import load as load_calendar
from laplace_research.evolve import baselines as b
from laplace_research.evolve.panel import Panel
from laplace_research.evolve.rolling import MODELS, PAIRS, SAME_FIT, render, rolling

pytest.importorskip("lightgbm")


def _sensed(panel: Panel) -> Panel:
    rng = np.random.default_rng(5)
    r2 = b.hourly_returns(panel.close) ** 2
    dvol = np.full(panel.close.shape, np.nan)
    dvol[0] = 100 * np.sqrt(b.trailing_mean(r2[0], 168) * b.HOURS_PER_YEAR)
    return panel.with_senses(
        {"rv": r2 * rng.uniform(0.5, 1.5, r2.shape), "dvol": dvol, "funding": np.full(r2.shape, 1e-4)}
    )


def _day(d: int) -> datetime:
    return datetime(2025, 2, d, tzinfo=UTC)


def test_each_fold_is_fitted_before_its_test_period_and_scored_on_every_hour(panel: Panel) -> None:
    result = rolling(
        _sensed(panel), load_calendar(), datetime(2025, 1, 1, tzinfo=UTC), [_day(d) for d in (4, 10, 15, 19)]
    )
    assert [f["test_start"][:10] for f in result["folds"]] == ["2025-02-10", "2025-02-15"]
    assert [f["hours"] for f in result["folds"]] == [5 * 24 - 24, 4 * 24 - 24]
    assert result["hours"] == 96 + 72
    for a, c in PAIRS:
        row = result["pooled"][f"{a} vs {c}"]
        assert np.isfinite(row["1h+4h"]["mean"])
        assert row["1h+4h"]["interval95"][0] <= row["1h+4h"]["mean"] <= row["1h+4h"]["interval95"][1]
    assert [set(r) for r in result["ev_gbm_rounds"]] == [{"test_start", "1", "4", "24"}] * 2
    assert "ages" not in result
    assert "| 2025-02-10 | 96 |" in render(result)


def test_ages_score_each_fit_on_every_later_period(panel: Panel) -> None:
    days = [_day(d) for d in (4, 8, 11, 14, 17)]  # the panel ends on 2025-02-20
    result = rolling(_sensed(panel), load_calendar(), datetime(2025, 1, 1, tzinfo=UTC), days, ages=True)
    stale, same = result["ages"]["stale minus fresh"], result["ages"]["same fit"]
    assert set(stale) == set(MODELS)
    # three folds: two fits are one period old once, one fit is two periods old once
    assert {name: sorted(by) for name, by in stale.items()} == {name: ["1", "2"] for name in MODELS}
    assert stale["ev-har"]["1"]["hours"] == 2 * (3 * 24 - 24)
    assert stale["ev-har"]["2"]["hours"] == 3 * 24 - 24
    assert set(same) == {f"{a} vs {c}" for a, c in SAME_FIT}
    assert sorted(same["ev-gbm vs ev-har"]) == ["0", "1", "2"]
    # a fit's age-0 gain over its own EV-HAR is the fold's own test
    fold_mean = np.mean([f["ev-gbm vs ev-har"]["1h+4h"] * f["hours"] for f in result["folds"]]) * 3 / result["hours"]
    assert same["ev-gbm vs ev-har"]["0"]["1h+4h"]["mean"] == pytest.approx(fold_mean)
    assert "stale minus fresh" in render(result)
