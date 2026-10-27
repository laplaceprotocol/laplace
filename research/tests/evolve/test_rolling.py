from datetime import UTC, datetime

import numpy as np

from laplace_research.data.events import load as load_calendar
from laplace_research.evolve import baselines as b
from laplace_research.evolve.panel import Panel
from laplace_research.evolve.rolling import PAIRS, render, rolling


def test_each_fold_is_fitted_before_its_test_period_and_scored_on_every_hour(panel: Panel) -> None:
    rng = np.random.default_rng(5)
    r2 = b.hourly_returns(panel.close) ** 2
    dvol = np.full(panel.close.shape, np.nan)
    dvol[0] = 100 * np.sqrt(b.trailing_mean(r2[0], 168) * b.HOURS_PER_YEAR)
    sensed = panel.with_senses(
        {"rv": r2 * rng.uniform(0.5, 1.5, r2.shape), "dvol": dvol, "funding": np.full(r2.shape, 1e-4)}
    )
    day = lambda d: datetime(2025, 2, d, tzinfo=UTC)  # noqa: E731
    result = rolling(sensed, load_calendar(), datetime(2025, 1, 1, tzinfo=UTC), [day(4), day(10), day(15), day(19)])
    assert [f["test_start"][:10] for f in result["folds"]] == ["2025-02-10", "2025-02-15"]
    assert [f["hours"] for f in result["folds"]] == [5 * 24 - 24, 4 * 24 - 24]
    assert result["hours"] == 96 + 72
    for a, c in PAIRS:
        row = result["pooled"][f"{a} vs {c}"]
        assert np.isfinite(row["1h+4h"]["mean"])
        assert row["1h+4h"]["interval95"][0] <= row["1h+4h"]["mean"] <= row["1h+4h"]["interval95"][1]
    assert "| 2025-02-10 | 96 |" in render(result)
