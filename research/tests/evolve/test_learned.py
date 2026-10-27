import numpy as np
import pytest

from laplace_research.data.events import load as load_calendar
from laplace_research.evolve import baselines as b
from laplace_research.evolve import learned
from laplace_research.evolve.evaluate import realized_bins, sample_times
from laplace_research.evolve.feed import HORIZONS, N_BINS
from laplace_research.evolve.panel import Panel, utc


def sensed(panel: Panel) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    r2 = b.hourly_returns(panel.close) ** 2
    dvol = np.full(panel.close.shape, np.nan)
    dvol[0] = 100 * np.sqrt(b.trailing_mean(r2[0], 168) * b.HOURS_PER_YEAR)
    funding = np.full(panel.close.shape, 1e-4)
    return {"rv": r2 * rng.uniform(0.5, 1.5, r2.shape), "dvol": dvol, "funding": funding}


def test_stored_weights_round_trip_to_within_their_resolution() -> None:
    rng = np.random.default_rng(1)
    net = learned.Net(
        *(rng.normal(0, s, shape) for s, shape in ((1, 4), (2, 4), (0.3, (4, 3)), (0.1, 3), (0.2, (3, 10)), (0.05, 10)))
    )
    stored = net.stored()
    assert all(-(2**15) < v < 2**15 for t in stored.values() for v in t["ints"])
    back = learned.Net.from_stored(stored)
    for name in learned.Net.__dataclass_fields__:
        original, restored = getattr(net, name), getattr(back, name)
        assert np.abs(original - restored).max() <= np.abs(original).max() / 2**15


def test_a_net_learns_a_variance_signal_the_base_ignores() -> None:
    rng = np.random.default_rng(2)
    n = 6_000
    signal = rng.normal(size=n)
    sd = np.exp(0.4 * signal)  # the signal sets the spread of the outcome
    edges = np.linspace(-2, 2, 9)
    y = np.digitize(rng.normal(size=n) * sd, edges)
    base = np.full((n, N_BINS), 0.0)
    base += np.bincount(y[: n // 2], minlength=N_BINS) / (n // 2)  # unconditional frequencies
    base = (1 - 1e-3) * base + 1e-4
    x = np.stack([signal, rng.normal(size=n)], axis=1)
    half = n // 2
    net = learned.fit_net(x[:half], base[:half], y[:half], x[half:], base[half:], y[half:], hidden=4)
    logits = np.log(base[half:]) + net.logits(x[half:])
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    gain = np.mean(np.log(p[np.arange(half), y[half:]])) - np.mean(np.log(base[half:][np.arange(half), y[half:]]))
    assert gain > 0.02


def test_inputs_follow_their_definitions(panel: Panel) -> None:
    senses = sensed(panel)
    pw = learned.Panelwise.of(panel.close, panel.high, panel.low, senses, panel.assets)
    times = np.array([900, 1000], dtype=np.int64)
    sd = np.full((len(panel.assets), 2), 0.01)
    x = learned.inputs(pw, load_calendar(), panel.ts, times, sd, 4)
    assert x.shape == (len(panel.assets), 2, len(learned.NAMES))
    a, t = 2, 1000
    week = b.trailing_mean(np.where(np.isfinite(senses["rv"]), senses["rv"], b.hourly_returns(panel.close) ** 2), 168)
    col = learned.NAMES.index("ret4")
    assert x[a, 1, col] == pytest.approx(np.log(panel.close[a, t] / panel.close[a, t - 4]) / np.sqrt(4 * week[a, t]))
    assert x[a, 1, learned.NAMES.index("funding")] == pytest.approx(1.0)  # 1e-4 in units of 1e-4
    hour = (panel.ts[t] // 3600 + 1) % 24
    assert x[a, 1, learned.NAMES.index("hour_sin")] == pytest.approx(np.sin(2 * np.pi * hour / 24))


def test_the_learned_layer_starts_from_ev_har_and_reports_its_weights(panel: Panel) -> None:
    senses = sensed(panel)
    fit = sample_times(panel, utc(int(panel.ts[0])), utc(int(panel.ts[900])), 3)
    rungs = b.ladder(
        panel.close, realized_bins(panel, fit), fit, slice(0, 900), senses=senses, assets=panel.assets, ts=panel.ts
    )
    ev = rungs[-1]
    pw = learned.Panelwise.of(panel.close, panel.high, panel.low, senses, panel.assets)
    model = learned.ev_net(
        ev,
        pw,
        load_calendar(),
        panel.ts,
        np.arange(720, 900),
        np.arange(900, 1000),
        lambda times: realized_bins(panel, times),
    )
    p = model.probs(np.arange(1000, 1010))
    assert p.shape == (len(panel.assets), len(HORIZONS), 10, N_BINS)
    assert np.allclose(p.sum(axis=-1), 1)
    assert set(model.params["net"]) == {"1", "4", "24"}
    assert model.params["inputs"] == list(learned.NAMES)


def test_a_price_back_where_it_was_gives_exactly_zero_return(panel: Panel) -> None:
    close = panel.close.copy()
    close[2, 1000] = close[2, 996]  # four hours later, the same price after moving in between
    close[2, 1024] = close[2, 1000]
    pw = learned.Panelwise.of(
        close, np.maximum(panel.high, close), np.minimum(panel.low, close), sensed(panel), panel.assets
    )
    assert pw.returns[4][2, 1000] == 0.0
    assert pw.returns[24][2, 1024] == 0.0
    assert pw.returns[1][2, 1001] == np.log(close[2, 1001] / close[2, 1000])
