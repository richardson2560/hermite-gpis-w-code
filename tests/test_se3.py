"""Unit tests for SE(3) Lie group operations.

These tests verify:
    1. hat/vee are inverse of each other.
    2. exp(0) = identity.
    3. exp/log roundtrip on random vectors.
    4. Known Rodrigues cases (π/2 around z).
    5. rot_align_vectors produces orthogonal matrices with det = 1.
    6. rot_align_vectors actually maps v1 to v2.
    7. Degenerate cases: parallel and antiparallel vectors.
    8. Composition: exp(ξ₁) exp(ξ₂) ≠ exp(ξ₁ + ξ₂) in general (non-abelian).
    9. exp produces valid SE(3) matrices (orthogonal R, bottom row [0,0,0,1]).

Run with:
    pytest tests/test_se3.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from hgw.registration.se3 import hat, vee, exp_se3, log_se3, rot_align_vectors


# ---------------------------------------------------------------------------
# Group 1 — hat / vee
# ---------------------------------------------------------------------------

def test_hat_vee_roundtrip() -> None:
    """vee(hat(xi)) == xi for arbitrary xi."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        xi = rng.normal(size=6)
        assert np.allclose(vee(hat(xi)), xi, atol=1e-14)


def test_hat_shape() -> None:
    xi = np.zeros(6)
    M = hat(xi)
    assert M.shape == (4, 4)


def test_hat_skew_structure() -> None:
    """The top-left 3x3 block of hat(xi) is skew-symmetric."""
    xi = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    M = hat(xi)
    upper = M[:3, :3]
    assert np.allclose(upper, -upper.T, atol=1e-14)


def test_hat_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        hat(np.zeros(5))


# ---------------------------------------------------------------------------
# Group 2 — exp and log
# ---------------------------------------------------------------------------

def test_exp_zero_is_identity() -> None:
    T = exp_se3(np.zeros(6))
    assert np.allclose(T, np.eye(4), atol=1e-14)


def test_exp_returns_valid_se3() -> None:
    """exp(xi) must be a valid SE(3) matrix."""
    rng = np.random.default_rng(1)
    for _ in range(10):
        xi = 0.1 * rng.normal(size=6)
        T = exp_se3(xi)

        # Orthonormality of the rotation block
        R = T[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

        # Bottom row
        assert np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-14)


def test_exp_log_roundtrip() -> None:
    """log(exp(xi)) == xi for small xi."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        xi = 0.5 * rng.normal(size=6)
        T = exp_se3(xi)
        xi_back = log_se3(T)
        assert np.allclose(xi_back, xi, atol=1e-10), (
            f"roundtrip failed: xi = {xi}, xi_back = {xi_back}"
        )


def test_exp_rotation_by_pi_over_2_around_z() -> None:
    """Known case: rotation by π/2 around z, no translation."""
    xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2])
    T = exp_se3(xi)

    expected_R = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.allclose(T[:3, :3], expected_R, atol=1e-12)
    assert np.allclose(T[:3, 3], np.zeros(3), atol=1e-12)


def test_exp_pure_translation() -> None:
    """Pure translation: R = I, t = ρ."""
    xi = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    T = exp_se3(xi)
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-14)
    assert np.allclose(T[:3, 3], xi[:3], atol=1e-14)


def test_log_identity() -> None:
    """log(I) = 0."""
    xi = log_se3(np.eye(4))
    assert np.allclose(xi, np.zeros(6), atol=1e-14)


def test_log_rejects_invalid_rotation() -> None:
    """log of a matrix with non-orthonormal R must raise."""
    bad = np.eye(4)
    bad[0, 0] = 2.0  # not orthogonal
    with pytest.raises(ValueError):
        log_se3(bad)


def test_exp_log_roundtrip_rotation_near_pi() -> None:
    """Roundtrip near θ ≈ π, where the naive formula breaks."""
    # Rotation of π - 1e-3 around z
    theta = np.pi - 1e-3
    xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0, theta])
    T = exp_se3(xi)
    xi_back = log_se3(T)
    assert np.allclose(xi_back, xi, atol=1e-6), (
        f"near-π roundtrip failed: xi = {xi}, xi_back = {xi_back}"
    )


def test_composition_is_non_abelian() -> None:
    """exp(ξ₁) exp(ξ₂) ≠ exp(ξ₁ + ξ₂) for non-commuting ξ."""
    xi1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    xi2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 4])

    T1 = exp_se3(xi1) @ exp_se3(xi2)
    T2 = exp_se3(xi1 + xi2)

    # They should differ
    assert not np.allclose(T1, T2, atol=1e-3)


# ---------------------------------------------------------------------------
# Group 3 — rot_align_vectors
# ---------------------------------------------------------------------------

def test_align_identity() -> None:
    """Aligning a vector with itself gives identity."""
    v = np.array([1.0, 0.0, 0.0])
    R = rot_align_vectors(v, v)
    assert np.allclose(R, np.eye(3), atol=1e-12)


def test_align_orthogonal() -> None:
    """Aligning x to y gives a π/2 rotation around z."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    R = rot_align_vectors(v1, v2)

    assert np.allclose(R @ v1, v2, atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_align_generic() -> None:
    """R @ v1 == v2 for several random pairs."""
    rng = np.random.default_rng(3)
    for _ in range(10):
        v1 = rng.normal(size=3)
        v2 = rng.normal(size=3)
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        R = rot_align_vectors(v1, v2)
        assert np.allclose(R @ v1, v2, atol=1e-10)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_align_antiparallel() -> None:
    """Antiparallel vectors give a π rotation."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    R = rot_align_vectors(v1, v2)

    assert np.allclose(R @ v1, v2, atol=1e-10)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_align_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        rot_align_vectors(np.zeros(3), np.array([1.0, 0.0, 0.0]))


def test_align_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        rot_align_vectors(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]))