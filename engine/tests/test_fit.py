import numpy as np
import pytest

from laplace_engine.fit import fit_weights, mix
from laplace_engine.quantize import WEIGHT_SCALE
from laplace_research.evolve.feed import HORIZONS, N_BINS, finalize


def test_em_weights_favour_the_member_that_saw_the_outcome() -> None:
    rng = np.random.default_rng(2)
    realized = rng.integers(3, 7, size=(7, len(HORIZONS), 400))
    sharp = np.full((7, len(HORIZONS), 400, N_BINS), 0.02)
    np.put_along_axis(sharp, realized[..., None], 0.82, axis=-1)
    flat = np.full_like(sharp, 1 / N_BINS)
    finals = {"sharp": finalize(sharp), "flat": finalize(flat)}
    weights = dict(fit_weights(finals, realized, HORIZONS))
    assert sum(weights.values()) == WEIGHT_SCALE
    assert weights["sharp"] > 990_000
    one_horizon = dict(fit_weights(finals, realized, (4,)))
    assert one_horizon["sharp"] > 990_000


def test_mix_uses_each_horizon_s_weights() -> None:
    a = finalize(np.ones((7, len(HORIZONS), 5, N_BINS)))
    b = finalize(np.arange(1.0, N_BINS + 1) * np.ones((7, len(HORIZONS), 5, N_BINS)))
    weights = {1: (("a", WEIGHT_SCALE),), 4: (("b", WEIGHT_SCALE),), 24: (("a", 500_000), ("b", 500_000))}
    mixed = mix({"a": a, "b": b}, weights.__getitem__)
    assert mixed[:, 0] == pytest.approx(a[:, 0])
    assert mixed[:, 1] == pytest.approx(b[:, 1])
    assert mixed[:, 2] == pytest.approx((a[:, 2] + b[:, 2]) / 2)
