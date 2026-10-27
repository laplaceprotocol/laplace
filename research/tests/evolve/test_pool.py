import numpy as np
import pytest

from laplace_research.evolve.feed import N_BINS, finalize
from laplace_research.evolve.pool import PoolDidNotConverge, fit_pool, mix, realized_probs, renormalized_contributions


def test_fit_puts_the_weight_on_the_member_that_explains_the_data() -> None:
    p = np.array([[0.9] * 50, [0.1] * 50])
    assert fit_pool(p).weights[0] > 0.99


def test_fit_mixes_complementary_members_and_certifies_the_gap() -> None:
    p = np.array([[0.9, 0.1] * 50, [0.1, 0.9] * 50])
    fit = fit_pool(p, tolerance=1e-10)
    assert fit.weights == pytest.approx([0.5, 0.5], abs=1e-6)
    assert fit.gap <= 1e-10


def test_training_contributions_are_nonnegative_at_the_certified_optimum() -> None:
    p = np.random.default_rng(3).uniform(0.01, 1.0, (6, 400))
    fit = fit_pool(p, tolerance=1e-9)
    contributions = renormalized_contributions(fit.weights, p)
    # At the optimum every renormalized deletion is feasible, so none can beat the full pool beyond the gap.
    assert contributions.min() >= -fit.gap


def _near_copies(seed: int) -> np.ndarray:
    """Realized-bin probabilities of 12 noisy forecasters of one truth; every third is a near-copy of the truth."""
    rng = np.random.default_rng(seed)
    truth = rng.dirichlet(np.full(N_BINS, 2.0), 4000)
    realized = (truth.cumsum(axis=1) < rng.uniform(size=(len(truth), 1))).sum(axis=1)
    members = []
    for k in range(12):
        logit = np.log(truth) * rng.uniform(0.5, 1.2) + rng.normal(0, 0.6 if k % 3 else 0.01, truth.shape)
        prob = np.exp(logit) / np.exp(logit).sum(axis=1, keepdims=True)
        members.append(0.99 * prob[np.arange(len(truth)), realized] + 0.001)
    return np.stack(members)


def test_near_copies_converge_in_few_steps_and_no_weighting_beats_the_certificate() -> None:
    p = _near_copies(7)
    fit = fit_pool(p, max_iterations=2_000)  # plain EM needs over 280,000 steps here
    assert fit.gap <= 1e-8
    best = np.mean(np.log(fit.weights @ p))
    others = np.random.default_rng(8).dirichlet(np.full(len(p), 0.3), 2000)
    assert (np.mean(np.log(others @ p), axis=1) <= best + fit.gap).all()


def test_unconverged_fit_raises() -> None:
    p = np.random.default_rng(4).uniform(0.01, 1.0, (6, 400))
    with pytest.raises(PoolDidNotConverge):
        fit_pool(p, tolerance=1e-15, max_iterations=3)


def test_pooling_identical_final_forecasts_preserves_their_score() -> None:
    final = finalize(np.random.default_rng(5).uniform(0, 1, (2, 3, 7, N_BINS)))
    realized = np.random.default_rng(6).integers(0, N_BINS, (2, 3, 7))
    pooled = mix(np.array([0.3, 0.7]), [final, final])
    assert np.allclose(realized_probs(pooled, realized), realized_probs(final, realized))
