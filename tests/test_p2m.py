"""Unit tests for the P2M Gauss-Newton solver.

The model is an asymmetric ellipsoid so that the registration problem has a
unique solution. A sphere would have SO(3) symmetry and the pose would not
be identifiable.

Run with:
    pytest tests/test_p2m.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from hgw.models.builder import build_prior
from hgw.models.evaluator import HermiteGPIS_W
from hgw.registration.p2m import P2MConfig, p2m_register
from hgw.registration.se3 import exp_se3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

H = 0.15
SIGMA_F2 = 1.0
SIGMA_0_SQ = 1e-4
SIGMA_1_SQ = 1e-2


def _ellipsoid_points(a: float, b: float, c: float, n: int, seed: int = 0):
    """Points on the surface of an axis-aligned ellipsoid with outward normals.

    The three semi-axes are distinct, so the object has no rotational
    symmetry and the registration problem has a unique solution.
    """
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    points = dirs * np.array([a, b, c])
    normals = points / np.array([a * a, b * b, c * c])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


@pytest.fixture
def model() -> HermiteGPIS_W:
    points, normals = _ellipsoid_points(0.3, 0.2, 0.1, 500, seed=0)
    art = build_prior(points, normals, H, SIGMA_F2, SIGMA_0_SQ, SIGMA_1_SQ, 200)
    return HermiteGPIS_W(art)


def _apply(T, points):
    return (T[:3, :3] @ points.T).T + T[:3, 3]


# ---------------------------------------------------------------------------
# Group 1 — Identity pose
# ---------------------------------------------------------------------------

def test_identity_pose_stays(model: HermiteGPIS_W) -> None:
    """Starting at the identity pose should keep it within a tight tolerance."""
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 200, seed=1)
    T_init = np.eye(4)

    result = p2m_register(scene, model, T_init)

    assert result.status == "ACCEPTED"
    assert np.allclose(result.T_estimated, T_init, atol=1e-2), (
        f"pose drifted by {np.linalg.norm(result.T_estimated - T_init):.4f}"
    )


def test_residual_at_identity_is_small(model: HermiteGPIS_W) -> None:
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 200, seed=2)
    result = p2m_register(scene, model, np.eye(4))
    assert result.residual_rms < 5e-3


# ---------------------------------------------------------------------------
# Group 2 — Convergence from nearby poses
# ---------------------------------------------------------------------------

def test_converges_from_small_translation(model: HermiteGPIS_W) -> None:
    """Starting 2 cm off should converge back to identity."""
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 200, seed=3)

    T_init = np.eye(4)
    T_init[:3, 3] = [0.02, 0.0, 0.0]

    result = p2m_register(scene, model, T_init)

    assert result.status == "ACCEPTED"
    assert np.allclose(result.T_estimated[:3, 3], np.zeros(3), atol=5e-3)


def test_converges_from_small_rotation(model: HermiteGPIS_W) -> None:
    """Starting 5° off in yaw should converge back."""
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 200, seed=4)

    xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(5.0)])
    T_init = exp_se3(xi)

    result = p2m_register(scene, model, T_init)

    assert result.status == "ACCEPTED"
    R = result.T_estimated[:3, :3]
    assert np.allclose(R, np.eye(3), atol=1e-2)


def test_reduces_residual(model: HermiteGPIS_W) -> None:
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 200, seed=5)
    T_init = np.eye(4)
    T_init[:3, 3] = [0.03, -0.02, 0.01]

    x_init = _apply(T_init, scene)
    m_init, _, _ = model.evaluate_many(x_init)
    rms_init = float(np.sqrt(np.mean(m_init ** 2)))

    result = p2m_register(scene, model, T_init)

    assert result.residual_rms <= rms_init + 1e-6


# ---------------------------------------------------------------------------
# Group 3 — Robustness to outliers
# ---------------------------------------------------------------------------

def test_huber_weights_attenuate_outliers(model: HermiteGPIS_W) -> None:
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 200, seed=6)

    rng = np.random.default_rng(99)
    outliers = 2.0 + 0.5 * rng.normal(size=(20, 3))
    scene_with_outliers = np.vstack([scene, outliers])

    T_init = np.eye(4)
    T_init[:3, 3] = [0.02, 0.0, 0.0]

    result = p2m_register(scene_with_outliers, model, T_init)

    assert result.status == "ACCEPTED"
    assert np.allclose(result.T_estimated[:3, 3], np.zeros(3), atol=2e-2)


# ---------------------------------------------------------------------------
# Group 4 — Covariance
# ---------------------------------------------------------------------------

def test_covariance_is_spd(model: HermiteGPIS_W) -> None:
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 200, seed=7)
    result = p2m_register(scene, model, np.eye(4))

    C = result.covariance
    assert np.allclose(C, C.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(C)
    assert np.all(eigvals > 0.0)


def test_covariance_finite_on_rejection(model: HermiteGPIS_W) -> None:
    """Rejected results must still return a finite covariance sentinel."""
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 200, seed=8)
    T_far = np.eye(4)
    T_far[:3, 3] = [10.0, 10.0, 10.0]

    result = p2m_register(scene, model, T_far)
    assert result.status == "REJECTED"
    assert np.all(np.isfinite(result.covariance))


# ---------------------------------------------------------------------------
# Group 5 — Status reporting
# ---------------------------------------------------------------------------

def test_status_accepted_on_convergence(model: HermiteGPIS_W) -> None:
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 200, seed=9)
    T_init = np.eye(4)
    T_init[:3, 3] = [0.01, 0.0, 0.0]

    result = p2m_register(scene, model, T_init)
    assert result.status == "ACCEPTED"
    assert result.iterations >= 1


def test_rejected_when_far_from_surface(model: HermiteGPIS_W) -> None:
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 200, seed=10)
    T_far = np.eye(4)
    T_far[:3, 3] = [10.0, 10.0, 10.0]

    result = p2m_register(scene, model, T_far)
    assert result.status == "REJECTED"
    assert result.active_fraction < 0.1


# ---------------------------------------------------------------------------
# Group 6 — Determinism
# ---------------------------------------------------------------------------

def test_deterministic(model: HermiteGPIS_W) -> None:
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 200, seed=11)
    T_init = np.eye(4)
    T_init[:3, 3] = [0.02, 0.01, -0.01]

    r1 = p2m_register(scene, model, T_init)
    r2 = p2m_register(scene, model, T_init)

    assert np.allclose(r1.T_estimated, r2.T_estimated, atol=1e-14)
    assert r1.iterations == r2.iterations
    assert r1.status == r2.status


# ---------------------------------------------------------------------------
# Group 7 — Input validation
# ---------------------------------------------------------------------------

def test_rejects_wrong_shape_scene(model: HermiteGPIS_W) -> None:
    with pytest.raises(ValueError, match="shape"):
        p2m_register(np.zeros((10, 2)), model, np.eye(4))


def test_rejects_too_few_points(model: HermiteGPIS_W) -> None:
    scene = np.zeros((5, 3))
    with pytest.raises(ValueError, match="at least"):
        p2m_register(scene, model, np.eye(4))


def test_rejects_non_model() -> None:
    scene = np.zeros((50, 3))
    with pytest.raises(TypeError):
        p2m_register(scene, "not a model", np.eye(4))


def test_rejects_wrong_shape_T(model: HermiteGPIS_W) -> None:
    scene = np.zeros((50, 3))
    with pytest.raises(ValueError, match="T_initial"):
        p2m_register(scene, model, np.eye(3))


def test_rejects_non_spd_P_prior(model: HermiteGPIS_W) -> None:
    scene, _ = _ellipsoid_points(0.3, 0.2, 0.1, 100, seed=12)
    bad_P = -np.eye(6)
    with pytest.raises(ValueError, match="positive definite"):
        p2m_register(scene, model, np.eye(4), P_prior=bad_P)