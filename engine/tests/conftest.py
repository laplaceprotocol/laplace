from collections.abc import Callable

import numpy as np
import pytest

from laplace_research.evolve.feed import ASSETS
from laplace_research.evolve.panel import Panel

ISSUE = 497_983  # 2026-10-23 07:00 UTC


def synthetic_panel(end_issue: int = ISSUE, hours: int = 1_200, seed: int = 7) -> Panel:
    """Hourly candles for every feed asset with GARCH-like volatility clustering, ending with the candle that closes at
    ``end_issue``'s start."""
    rng = np.random.default_rng(seed)
    n = len(ASSETS)
    var = np.full(n, 1e-4)
    r = np.empty((n, hours))
    for t in range(hours):
        r[:, t] = rng.standard_t(5, n) * np.sqrt(var * 3 / 5)
        var = 2e-6 + 0.08 * r[:, t] ** 2 + 0.9 * var
    close = 100 * np.exp(np.cumsum(r, axis=1))
    open_ = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    spread = np.abs(rng.normal(0, 0.002, (n, hours)))
    first = (end_issue - hours) * 3600
    return Panel(
        assets=ASSETS,
        ts=np.arange(first, first + hours * 3600, 3600, dtype=np.int64),
        open=open_,
        high=np.maximum(open_, close) * (1 + spread),
        low=np.minimum(open_, close) * (1 - spread),
        close=close,
        volume=rng.uniform(10, 20, (n, hours)),
    )


@pytest.fixture(scope="session")
def panel() -> Panel:
    return synthetic_panel()


@pytest.fixture
def make_panel() -> Callable[..., Panel]:
    return synthetic_panel


@pytest.fixture(autouse=True)
def no_live_senses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never reach third-party APIs: every live sense is unknown unless a test provides them."""
    from laplace_engine import cli

    def unknown(panel: Panel, cache_dir: object, **kwargs: object) -> dict[str, np.ndarray]:
        return {name: np.full(panel.close.shape, np.nan) for name in cli.sense_data.NAMES}

    monkeypatch.setattr(cli.sense_data, "build", unknown)
