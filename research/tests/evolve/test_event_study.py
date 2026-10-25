import numpy as np

from laplace_research.data import events
from laplace_research.evolve.baselines import hourly_returns, trailing_mean
from laplace_research.evolve.event_study import event_study, overlay_screen, render, render_screen
from laplace_research.evolve.panel import Panel
from laplace_research.evolve.run import RunConfig


def test_a_release_that_moves_prices_shows_in_the_study(panel: Panel) -> None:
    calendar = events.load()
    close = panel.close.copy()
    kinds = np.array(calendar.kinds)
    for stamp in calendar.times[kinds == "cpi"]:
        t = int(np.searchsorted(panel.ts, stamp, side="right")) - 1
        if 1 <= t < len(panel.ts):
            close[:, t:] *= 1.03  # a 3% jump in the candle that holds each CPI release
    moved = Panel(panel.assets, panel.ts, panel.open, panel.high, panel.low, close, panel.volume)
    study = event_study(moved, calendar)
    assert study["cpi"]["n"] == 2  # 2025-01-15 and 2025-02-12 fall inside the synthetic panel
    assert min(study["cpi"]["event_hour"]) > 5
    assert 0.3 < np.mean(study["nfp"]["event_hour"]) < 3  # untouched releases look like any other hour
    assert "| cpi | 2 |" in render(study)


def test_the_overlay_screen_scores_every_hour_of_validation_and_holdout(panel: Panel) -> None:
    r2 = hourly_returns(panel.close) ** 2
    rv = r2 * np.random.default_rng(3).uniform(0.5, 1.5, r2.shape)
    dvol = np.full(panel.close.shape, np.nan)
    dvol[0] = 100 * np.sqrt(trailing_mean(r2[0], 168) * 8760)
    sensed = panel.with_senses({"rv": rv, "dvol": dvol})
    config = RunConfig(
        data_start="2025-01-01T00:00",
        data_end="2025-02-20T00:00",
        search=("2025-01-02T00:00", "2025-02-01T00:00", 23),
        validation=("2025-02-01T00:00", "2025-02-10T00:00", 5),
        holdout=("2025-02-10T00:00", "2025-02-18T00:00", 3),
        baseline_fit_step=2,
        seed=0,
    )
    screen = overlay_screen(sensed, config, events.load())
    assert screen["hours"] == 17 * 24  # every hour from 2025-02-01 to 2025-02-18
    for row in screen["ev_har_vs_iv_har"].values():
        assert row["interval95"][0] <= row["mean"] <= row["interval95"][1]
    assert "| us-session |" in render_screen(screen)
