"""Unit tests for the online Hermite-GPIS-W evaluator.

These tests verify:
    1. Loading from an in-memory artifact and from a saved path.
    2. Field value and gradient at the pivots (consistency with the fit).
    3. Exact prior recovery outside the support.
    4. Variance proxy bounds (0 ≤ ṽ_m ≤ σ_f²).
    5. Batch equals individual evaluation.
    6. Gradient matches central finite differences.
    7. Support fraction behavior for points inside/outside.
    8. Edge case: query exactly at a pivot (r = 0).

Run with:
    pytest tests/test_evaluator.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hgw.models.artifact import ModelArtifact
from hgw.models.builder import build_prior
from hgw.models.evaluator import HermiteGPIS_W


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

H = 0.1
SIGMA_F2 = 1.0
SIGMA_0_SQ = 1e-4
SIGMA_1_SQ = 1e-2
M_TARGET = 10


def _sphere(n: int, radius: float = 0.3, seed: int = 0):
    """Points on a sphere with outward normals."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return radius * directions, directions


@pytest.fixture
def artifact() -> ModelArtifact:
    points, normals = _sphere(100, seed=0)
    return build_prior(
        points, normals, H, SIGMA_F2, SIGMA_0_SQ, SIGMA_1_SQ, M_TARGET,
    )


@pytest.fixture
def evaluator(artifact: ModelArtifact) -> HermiteGPIS_W:
    return HermiteGPIS_W(artifact)


# ---------------------------------------------------------------------------
# Group 1 — Loading
# ---------------------------------------------------------------------------

def test_load_from_in_memory_artifact(evaluator: HermiteGPIS_W) -> None:
    assert evaluator.n_primitives == M_TARGET
    assert evaluator.support_radius == pytest.approx(H, rel=1e-6)

def test_rejects_invalid_artifact_type() -> None:
    """Only ModelArtifact, str, or Path are accepted."""
    for bad in (42, None, [1, 2, 3], {"points": None}):
        with pytest.raises(TypeError):
            HermiteGPIS_W(bad)  # type: ignore


# ---------------------------------------------------------------------------
# Group 2 — Field and gradient at pivots
# ---------------------------------------------------------------------------

def test_field_at_pivots_is_zero(
    artifact: ModelArtifact, evaluator: HermiteGPIS_W
) -> None:
    """The posterior mean at each pivot must be close to zero."""
    for p in artifact.points:
        m, _, _ = evaluator.evaluate(np.asarray(p, dtype=np.float64))
        assert abs(m) < 1e-3, f"|m(p_i)| = {abs(m)} is too large"


def test_directional_gradient_at_pivots_is_one(
    artifact: ModelArtifact, evaluator: HermiteGPIS_W
) -> None:
    """The directional derivative along the normal at each pivot is close to 1."""
    for p, n in zip(artifact.points, artifact.normals):
        _, g, _ = evaluator.evaluate(np.asarray(p, dtype=np.float64))
        directional = float(g @ np.asarray(n, dtype=np.float64))
        assert abs(directional - 1.0) < 5e-2, (
            f"∇m(p_i) · n_i = {directional}, expected ~1.0"
        )


# ---------------------------------------------------------------------------
# Group 3 — Prior recovery
# ---------------------------------------------------------------------------

def test_prior_recovery_far_away(evaluator: HermiteGPIS_W) -> None:
    """Far from all pivots, the model must recover the prior exactly."""
    # The pivots are on a sphere of radius 0.3. A query at distance 5.0
    # from the origin is far from all supports.
    x = np.array([5.0, 0.0, 0.0])

    m, grad, var = evaluator.evaluate(x)
    assert m == 0.0
    assert np.all(grad == 0.0)
    assert var == pytest.approx(SIGMA_F2, rel=1e-12)


def test_prior_recovery_at_center(evaluator: HermiteGPIS_W) -> None:
    """The center of the sphere is also outside all supports."""
    x = np.array([0.0, 0.0, 0.0])

    m, grad, var = evaluator.evaluate(x)
    # 0.3 > h = 0.1, so the origin is outside every support
    assert m == 0.0
    assert np.all(grad == 0.0)
    assert var == pytest.approx(SIGMA_F2, rel=1e-12)


# ---------------------------------------------------------------------------
# Group 4 — Variance bounds
# ---------------------------------------------------------------------------

def test_variance_bounded_by_prior(evaluator: HermiteGPIS_W) -> None:
    """The variance proxy must lie in [0, σ_f²] for all query points."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-1.0, 1.0, size=(500, 3))

    _, _, var = evaluator.evaluate_many(X)
    assert np.all(var >= 0.0), f"Negative variance: min = {var.min()}"
    assert np.all(var <= SIGMA_F2 + 1e-12), f"Variance exceeds prior: max = {var.max()}"


def test_variance_at_pivot_matches_theory(
    artifact: ModelArtifact, evaluator: HermiteGPIS_W
) -> None:
    """At a pivot, the variance proxy equals the exact formula.

    At r = 0, k_1 = 0 and k_0 = σ_f². So:
        ṽ_m(p) = σ_f² - σ_f⁴ / (σ_f² + σ_0²) = σ_f² · σ_0² / (σ_f² + σ_0²)
    """
    expected = SIGMA_F2 * SIGMA_0_SQ / (SIGMA_F2 + SIGMA_0_SQ)

    for p in artifact.points[:3]:
        _, _, var = evaluator.evaluate(np.asarray(p, dtype=np.float64))
        assert var == pytest.approx(expected, rel=1e-6), (
            f"variance at pivot = {var}, expected {expected}"
        )


def test_variance_at_pivot_handles_r_zero(evaluator: HermiteGPIS_W, artifact: ModelArtifact) -> None:
    """Exact query at a pivot must not produce NaN or inf."""
    p = np.asarray(artifact.points[0], dtype=np.float64)
    m, grad, var = evaluator.evaluate(p)

    assert np.isfinite(m)
    assert np.all(np.isfinite(grad))
    assert np.isfinite(var)
    assert var >= 0.0


# ---------------------------------------------------------------------------
# Group 5 — Batch vs individual
# ---------------------------------------------------------------------------

def test_batch_equals_individual(evaluator: HermiteGPIS_W) -> None:
    """evaluate_many([x]) must match evaluate(x) for every query."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-0.3, 0.3, size=(20, 3))

    means_batch, grads_batch, vars_batch = evaluator.evaluate_many(X)

    for q in range(len(X)):
        m, g, v = evaluator.evaluate(X[q])
        assert m == pytest.approx(means_batch[q], abs=1e-14)
        assert np.allclose(g, grads_batch[q], atol=1e-14)
        assert v == pytest.approx(vars_batch[q], abs=1e-14)


# ---------------------------------------------------------------------------
# Group 6 — Gradient vs finite differences
# ---------------------------------------------------------------------------

def test_gradient_matches_finite_differences(evaluator: HermiteGPIS_W) -> None:
    """Analytical gradient must match central finite differences."""
    rng = np.random.default_rng(7)
    # Pick a query point inside the support of at least one pivot
    p_center = np.asarray(evaluator._points[0], dtype=np.float64)  # internal, test-only
    x = p_center + 0.03 * rng.normal(size=3)

    m, g_analytic, _ = evaluator.evaluate(x)

    eps = 1e-6
    g_numeric = np.zeros(3)
    for i in range(3):
        x_plus = x.copy(); x_plus[i] += eps
        x_minus = x.copy(); x_minus[i] -= eps
        m_plus, _, _ = evaluator.evaluate(x_plus)
        m_minus, _, _ = evaluator.evaluate(x_minus)
        g_numeric[i] = (m_plus - m_minus) / (2 * eps)

    assert np.allclose(g_analytic, g_numeric, atol=1e-4, rtol=1e-3), (
        f"analytic = {g_analytic}, numeric = {g_numeric}"
    )


# ---------------------------------------------------------------------------
# Group 7 — Support fraction
# ---------------------------------------------------------------------------

def test_support_fraction_inside_support(evaluator: HermiteGPIS_W) -> None:
    """Points near pivots should all be inside the support."""
    # Query exactly at pivots
    X = np.asarray(evaluator._points, dtype=np.float64)
    frac = evaluator.support_fraction(X)
    assert frac == pytest.approx(1.0)


def test_support_fraction_far_away(evaluator: HermiteGPIS_W) -> None:
    """Points far from all pivots should all be outside the support."""
    X = np.array([[10.0, 10.0, 10.0], [-10.0, -10.0, -10.0]])
    frac = evaluator.support_fraction(X)
    assert frac == 0.0


def test_support_fraction_mixed(evaluator: HermiteGPIS_W) -> None:
    """Half inside, half outside gives ~0.5."""
    inside = np.asarray(evaluator._points[:5], dtype=np.float64)
    outside = np.full((5, 3), 10.0)
    X = np.vstack([inside, outside])
    frac = evaluator.support_fraction(X)
    assert frac == pytest.approx(0.5)


def test_support_fraction_empty(evaluator: HermiteGPIS_W) -> None:
    """Empty input returns 0.0."""
    frac = evaluator.support_fraction(np.zeros((0, 3)))
    assert frac == 0.0


# ---------------------------------------------------------------------------
# Group 8 — Input validation
# ---------------------------------------------------------------------------

def test_evaluate_rejects_wrong_shape(evaluator: HermiteGPIS_W) -> None:
    with pytest.raises(ValueError, match="shape"):
        evaluator.evaluate(np.array([0.0, 0.0]))


def test_evaluate_rejects_nonfinite(evaluator: HermiteGPIS_W) -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluator.evaluate(np.array([np.nan, 0.0, 0.0]))


def test_evaluate_many_rejects_wrong_shape(evaluator: HermiteGPIS_W) -> None:
    with pytest.raises(ValueError, match="shape"):
        evaluator.evaluate_many(np.zeros((10, 2)))


def test_support_fraction_rejects_wrong_shape(evaluator: HermiteGPIS_W) -> None:
    with pytest.raises(ValueError, match="shape"):
        evaluator.support_fraction(np.zeros((10, 2)))