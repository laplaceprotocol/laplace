from datetime import datetime

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


def test_ev_har_program_computes_the_research_overlay(panel: Panel, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from laplace_research.data.events import load as load_calendar

    rng = np.random.default_rng(5)
    r2 = b.hourly_returns(panel.close) ** 2
    rv = r2 * rng.uniform(0.5, 1.5, r2.shape)
    dvol = np.full(panel.close.shape, np.nan)
    dvol[0] = 100 * np.sqrt(b.trailing_mean(r2[0], 168) * b.HOURS_PER_YEAR)
    senses = {"rv": rv, "dvol": dvol}
    fit = sample_times(panel, utc(int(panel.ts[0])), utc(int(panel.ts[800])), 3)
    rungs = b.ladder(
        panel.close,
        realized_bins(panel, fit),
        fit,
        slice(0, 800),
        senses=senses,
        assets=panel.assets,
        ts=panel.ts,
    )
    sources = ladder_programs(rungs, panel.assets, "a synthetic window")
    assert list(sources)[-3:] == ["rv-har", "iv-har", "ev-har"]
    ev = rungs[-1]
    fn = load(sources["ev-har"])

    path = tmp_path / "panel.npz"
    panel.with_senses(senses).save(path)
    reader = HistoryReader(*ensure_history(path))
    calendar = load_calendar()
    # forecasts before, at and after the CPI release of 2026-10-14 (12:30 UTC), and on a session's edges
    for when in ("2026-10-14T10:00", "2026-10-14T12:00", "2026-10-14T14:00", "2026-10-16T19:00", "2026-10-18T09:00"):
        at = int(datetime.fromisoformat(when + "+00:00").timestamp())
        t = int(np.searchsorted(panel.ts, at - 3600))
        reader.advance_to(t)
        r = np.diff(np.log(panel.close[:, t + 1 - LOOKBACK : t + 1]))
        window_rv = rv[:, t + 2 - LOOKBACK : t + 1]
        filled = np.where(np.isfinite(window_rv), window_rv, r * r)
        for a in (0, 3):
            means = np.array([filled[a, -w:].mean() for w in ev.params["windows"]])
            implied = dvol[a, t] if np.isfinite(dvol[a, t]) else dvol[0, t]
            implied = (implied / 100) ** 2 / b.HOURS_PER_YEAR if np.isfinite(implied) else means[3]
            for h in HORIZONS:
                c = np.asarray(ev.params["per_asset"][a][str(h)])
                variance = max(c[0] + float(c[1:] @ np.append(means, implied)) * h, 0.1 * means[2] * h)
                x = b.overlay_features(calendar, np.array([at]), h)[0]
                scale = np.exp(0.5 * x @ np.asarray(ev.params["overlay"][str(h)]["beta"]))
                expected = student_t_probs(np.sqrt(variance) * scale, ev.nu[h], edges(h))
                got = np.asarray(fn(reader.context(a, t, h)))
                assert got == pytest.approx(expected, abs=1e-12), (when, a, h)


def test_ev_net_program_reproduces_the_learned_layer(panel: Panel, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from laplace_engine.baselines import learned_program
    from laplace_research.data.events import load as load_calendar
    from laplace_research.evolve import learned

    rng = np.random.default_rng(5)
    r2 = b.hourly_returns(panel.close) ** 2
    rv = r2 * rng.uniform(0.5, 1.5, r2.shape)
    dvol = np.full(panel.close.shape, np.nan)
    dvol[0] = 100 * np.sqrt(b.trailing_mean(r2[0], 168) * b.HOURS_PER_YEAR)
    funding = rng.normal(1e-4, 5e-5, panel.close.shape)
    senses = {"rv": rv, "dvol": dvol, "funding": funding}
    fit = sample_times(panel, utc(int(panel.ts[0])), utc(int(panel.ts[900])), 3)
    rungs = b.ladder(
        panel.close, realized_bins(panel, fit), fit, slice(0, 900), senses=senses, assets=panel.assets, ts=panel.ts
    )
    ev = rungs[-1]
    pw = learned.Panelwise.of(panel.close, panel.high, panel.low, senses, panel.assets)
    calendar = load_calendar()
    model = learned.ev_net(
        ev, pw, calendar, panel.ts, np.arange(720, 900), np.arange(900, 1050), lambda t: realized_bins(panel, t)
    )
    source = learned_program(ev, model.params["net"], panel.assets, "a synthetic window")
    assert len(source.encode()) < 16_000  # the sandbox's limit
    fn = load(source)
    ev_fn = load(ladder_programs(rungs, panel.assets, "a synthetic window")["ev-har"])
    namespace: dict[str, object] = {}
    exec(source, namespace)  # trusted: our own generated source, to reach its helpers
    net_inputs = namespace["_net_inputs"]

    path = tmp_path / "panel.npz"
    panel.with_senses(senses).save(path)
    reader = HistoryReader(*ensure_history(path))
    times = np.array([1060, 1100, 1150, 1198], dtype=np.int64)
    research = model.probs(times)
    sds = ev.sd(times)
    for n, t in enumerate(times):
        reader.advance_to(int(t))
        for a in (0, 3):
            for j, h in enumerate(HORIZONS):
                ctx = reader.context(a, int(t), h)
                got = np.asarray(fn(ctx))
                # EV-HAR's own program differs from research where its 720-hour mean covers the window's 719 hours;
                # the learned layer may carry that difference, and add nothing of its own.
                ev_got = np.asarray(ev_fn(ctx))
                ev_gap = float(np.abs(ev_got / ev_got.sum() - ev.probs(np.array([t]))[a, j, 0]).max())
                assert float(np.abs(got - research[a, j, n]).max()) <= 3 * ev_gap + 1e-9, (t, a, h)
                x = learned.inputs(pw, calendar, panel.ts, np.array([t]), sds[:, j, n : n + 1], h)[a, 0]
                y = np.asarray(net_inputs(ctx, float(sds[a, j, n])))  # type: ignore[operator]
                assert y == pytest.approx(x, rel=1e-9, abs=1e-12), (t, a, h)
