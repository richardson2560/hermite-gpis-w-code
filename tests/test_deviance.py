"""Unit tests for the orthogonal residual deviance test.

The variance proxy is intentionally conservative, so the test is
lenient for moderate offsets. The tests here reflect that behavior:
the accepted case is a scene exactly on the surface, and the rejected
cases use large structural displacements or low coverage.

Run with:
    pytest tests/test_deviance.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hgw.models.builder import build_prior
from hgw.models.evaluator import HermiteGPIS_W
from hgw.stats.deviance import (
    DevianceConfig,
    compute_residuals,
    deviance_test,
    expected_deviance_separation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

H = 0.10
SIGMA_F2 = 1.0
SIGMA_0_SQ = 1e-4
SIGMA_1_SQ = 1e-2


def _ellipsoid_points(a: float, b: float, c: float, n: int, seed: int = 0):
    """Points on an axis-aligned ellipsoid with outward normals."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    points = dirs * np.array([a, b, c])
    normals = points / np.array([a * a, b * b, c * c])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


@pytest.fixture(scope="module")
def model() -> HermiteGPIS_W:
    points, normals = _ellipsoid_points(0.3, 0.2, 0.1, 500, seed=0)
    art = build_prior(points, normals, H, SIGMA_F2, SIGMA_0_SQ, SIGMA_1_SQ, 200)
    return HermiteGPIS_W(art)


IDENTITY = np.eye(4)


# ---------------------------------------------------------------------------
# Group 1 — Residual computation
# ---------------------------------------------------------------------------

def test_residuals_shapes(model: HermiteGPIS_W) -> None:
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 50, seed=1)
    r = compute_residuals(scene, IDENTITY, model)

    assert r.residuals.shape == (50,)
    assert r.residual_variances.shape == (50,)
    assert r.gradient_norms.shape == (50,)
    assert r.valid_mask.shape == (50,)


def test_residuals_on_surface_are_small(model: HermiteGPIS_W) -> None:
    """Points on the surface should have residuals near zero."""
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 50, seed=2)
    r = compute_residuals(scene, IDENTITY, model)

    inlier_residuals = np.abs(r.residuals[r.valid_mask])
    assert inlier_residuals.max() < 5e-2, (
        f"max |r_j| = {inlier_residuals.max():.4f}"
    )


def test_residual_variances_are_positive(model: HermiteGPIS_W) -> None:
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 50, seed=3)
    r = compute_residuals(scene, IDENTITY, model)

    valid_variances = r.residual_variances[r.valid_mask]
    assert np.all(valid_variances > 0.0)


# ---------------------------------------------------------------------------
# Group 2 — Gradient floor
# ---------------------------------------------------------------------------

def test_gradient_floor_marks_invalid(model: HermiteGPIS_W) -> None:
    """Points far outside the support have ||∇m|| ≈ 0 and are invalid."""
    far_scene = np.full((20, 3), 10.0)  # 10 m away
    r = compute_residuals(far_scene, IDENTITY, model)

    assert not np.any(r.valid_mask), "points outside support should be invalid"


def test_gradient_floor_keeps_surface_points(model: HermiteGPIS_W) -> None:
    """Points on the surface have ||∇m|| >> g_min and stay valid."""
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 50, seed=4)
    r = compute_residuals(scene, IDENTITY, model)

    assert np.all(r.valid_mask)


# ---------------------------------------------------------------------------
# Group 3 — Full test on compatible scene
# ---------------------------------------------------------------------------

def test_compatible_scene_is_accepted(model: HermiteGPIS_W) -> None:
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 200, seed=5)
    result = deviance_test(scene, IDENTITY, model)

    assert result.status == "ACCEPTED", (
        f"compatible scene rejected: {result.reason}"
    )
    assert result.coverage > 0.9
    assert result.Q_statistic < result.threshold


def test_compatible_scene_statistic_is_finite(model: HermiteGPIS_W) -> None:
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 200, seed=6)
    result = deviance_test(scene, IDENTITY, model)

    assert math.isfinite(result.Q_statistic)
    assert math.isfinite(result.threshold)
    assert 0.0 <= result.p_value <= 1.0


# ---------------------------------------------------------------------------
# Group 4 — Rejection cases
# ---------------------------------------------------------------------------

def test_displaced_scene_is_rejected(model: HermiteGPIS_W) -> None:
    """A scene shifted 2 m away has no active support; coverage is zero."""
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 200, seed=7)
    scene_displaced = scene + np.array([2.0, 0.0, 0.0])

    result = deviance_test(scene_displaced, IDENTITY, model)

    assert result.status == "REJECTED_COVERAGE"
    assert result.coverage < 0.05


def test_coverage_gate_uses_expected_count(model: HermiteGPIS_W) -> None:
    """Doubling the expected count halves the coverage estimate."""
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 100, seed=8)
    r1 = deviance_test(scene, IDENTITY, model, expected_point_count=100)
    r2 = deviance_test(scene, IDENTITY, model, expected_point_count=200)

    assert r1.coverage == pytest.approx(2.0 * r2.coverage, abs=1e-6)


# ---------------------------------------------------------------------------
# Group 5 — Determinism
# ---------------------------------------------------------------------------

def test_determinism(model: HermiteGPIS_W) -> None:
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 100, seed=9)

    r1 = deviance_test(scene, IDENTITY, model)
    r2 = deviance_test(scene, IDENTITY, model)

    assert r1.status == r2.status
    assert r1.Q_statistic == pytest.approx(r2.Q_statistic, abs=1e-14)
    assert np.array_equal(r1.inlier_mask, r2.inlier_mask)


# ---------------------------------------------------------------------------
# Group 6 — Analytical helper
# ---------------------------------------------------------------------------

def test_separation_helper_basic() -> None:
    delta_ll, lambda_m = expected_deviance_separation(
        mean_offset_m=0.075, sigma_r=0.012, M=150
    )
    # ΔLL = 0.075² / (2 · 0.012²) = 0.005625 / 0.000288 ≈ 19.53
    assert delta_ll == pytest.approx(19.53, rel=1e-2)
    # Λ_M = 2 · 150 · 19.53 ≈ 5859
    assert lambda_m == pytest.approx(5859.0, rel=1e-2)


def test_separation_helper_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        expected_deviance_separation(-0.1, 0.012, 100)
    with pytest.raises(ValueError):
        expected_deviance_separation(0.05, -0.01, 100)
    with pytest.raises(ValueError):
        expected_deviance_separation(0.05, 0.01, 0)


# ---------------------------------------------------------------------------
# Group 7 — Input validation
# ---------------------------------------------------------------------------

def test_rejects_wrong_shape_scene(model: HermiteGPIS_W) -> None:
    with pytest.raises(ValueError, match="shape"):
        deviance_test(np.zeros((10, 2)), IDENTITY, model)


def test_rejects_wrong_shape_T(model: HermiteGPIS_W) -> None:
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 50, seed=10)
    with pytest.raises(ValueError, match="T"):
        deviance_test(scene, np.eye(3), model)


def test_rejects_invalid_config() -> None:
    with pytest.raises(ValueError):
        DevianceConfig(alpha_local=0.0)
    with pytest.raises(ValueError):
        DevianceConfig(alpha_global=1.5)
    with pytest.raises(ValueError):
        DevianceConfig(min_coverage_fraction=0.0)
    with pytest.raises(ValueError):
        DevianceConfig(min_gradient_norm=-1.0)