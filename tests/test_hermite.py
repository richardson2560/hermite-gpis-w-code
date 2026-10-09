"""Unit tests for the Hermite covariance assembly.

These tests verify:
    1. HermiteObservation validation and normalization.
    2. Correct block values for the four (kind_i, kind_j) combinations.
    3. Compact support and coincident-point limits.
    4. Symmetry of the block function.
    5. Symmetry and positive semidefiniteness of the assembled matrix.
    6. Correct diagonal for value and derivative channels.

Run with:
    pytest tests/test_hermite.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from hgw.core.hermite import (
    HermiteObservation,
    hermite_block,
    assemble_system_matrix,
)


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

H = 0.1
SIGMA_F2 = 1.0
SIGMA_0_SQ = 1e-4
SIGMA_1_SQ = 1e-2


def _obs(point, kind, direction=(0.0, 0.0, 1.0), value=None, noise=None):
    """Build a HermiteObservation with sensible defaults."""
    if value is None:
        value = 0.0 if kind == "value" else 1.0
    if noise is None:
        noise = SIGMA_0_SQ if kind == "value" else SIGMA_1_SQ
    return HermiteObservation(
        point_m=tuple(point),
        kind=kind,
        direction=tuple(direction),
        value=value,
        noise_variance=noise,
    )


def _synthetic_pair(offset=(0.03, -0.02, 0.01), direction=(0.0, 0.0, 1.0)):
    """Build a pair of value/derivative observations at a fixed offset."""
    p_i = np.array(offset, dtype=float)
    p_j = np.zeros(3)
    obs_i_value = _obs(p_i, "value", direction)
    obs_j_value = _obs(p_j, "value", direction)
    obs_i_deriv = _obs(p_i, "derivative", direction)
    obs_j_deriv = _obs(p_j, "derivative", direction)
    return obs_i_value, obs_j_value, obs_i_deriv, obs_j_deriv


# ---------------------------------------------------------------------------
# Test group 1 — HermiteObservation validation
# ---------------------------------------------------------------------------

def test_observation_valid_construction() -> None:
    """A well-formed observation must construct without error."""
    obs = HermiteObservation(
        point_m=(0.1, 0.2, 0.3),
        kind="value",
        direction=(0.0, 0.0, 1.0),
        value=0.0,
        noise_variance=1e-4,
    )
    assert obs.kind == "value"
    assert obs.point_m == (0.1, 0.2, 0.3)
    assert obs.direction == (0.0, 0.0, 1.0)


def test_observation_normalizes_direction() -> None:
    """A non-unit direction must be normalized on construction."""
    obs = HermiteObservation(
        point_m=(0.0, 0.0, 0.0),
        kind="derivative",
        direction=(0.0, 0.0, 5.0),
        value=1.0,
        noise_variance=1e-4,
    )
    assert obs.direction == (0.0, 0.0, 1.0)


def test_observation_rejects_bad_point() -> None:
    """Non-finite or wrong-shape points must raise."""
    with pytest.raises(ValueError):
        HermiteObservation(
            point_m=(np.nan, 0.0, 0.0),
            kind="value",
            direction=(0.0, 0.0, 1.0),
            value=0.0,
            noise_variance=1e-4,
        )
    with pytest.raises(ValueError):
        HermiteObservation(
            point_m=(0.0, 0.0),  # wrong shape
            kind="value",
            direction=(0.0, 0.0, 1.0),
            value=0.0,
            noise_variance=1e-4,
        )


def test_observation_rejects_bad_kind() -> None:
    """Unknown kinds must raise."""
    with pytest.raises(ValueError):
        HermiteObservation(
            point_m=(0.0, 0.0, 0.0),
            kind="integral",   # not a valid kind
            direction=(0.0, 0.0, 1.0),
            value=0.0,
            noise_variance=1e-4,
        )


def test_observation_rejects_zero_direction() -> None:
    """A zero direction vector must raise."""
    with pytest.raises(ValueError):
        HermiteObservation(
            point_m=(0.0, 0.0, 0.0),
            kind="value",
            direction=(0.0, 0.0, 0.0),
            value=0.0,
            noise_variance=1e-4,
        )


def test_observation_rejects_nonpositive_noise() -> None:
    """Zero or negative noise variance must raise."""
    with pytest.raises(ValueError):
        HermiteObservation(
            point_m=(0.0, 0.0, 0.0),
            kind="value",
            direction=(0.0, 0.0, 1.0),
            value=0.0,
            noise_variance=0.0,
        )
    with pytest.raises(ValueError):
        HermiteObservation(
            point_m=(0.0, 0.0, 0.0),
            kind="value",
            direction=(0.0, 0.0, 1.0),
            value=0.0,
            noise_variance=-1e-4,
        )


# ---------------------------------------------------------------------------
# Test group 2 — Block values at the origin
# ---------------------------------------------------------------------------

def test_block_value_value_at_origin() -> None:
    """k_h(p, p) = σ_f²."""
    obs = _obs((0.0, 0.0, 0.0), "value")
    block = hermite_block(obs, obs, H, SIGMA_F2)
    assert block == pytest.approx(SIGMA_F2, abs=1e-12)


def test_block_derivative_derivative_at_origin() -> None:
    """-n^T H(0) n = 20 σ_f² / h²."""
    obs = _obs((0.0, 0.0, 0.0), "derivative", direction=(0.0, 0.0, 1.0))
    block = hermite_block(obs, obs, H, SIGMA_F2)
    expected = 20.0 * SIGMA_F2 / (H ** 2)
    assert block == pytest.approx(expected, rel=1e-10)


def test_block_cross_at_origin_is_zero() -> None:
    """Value-derivative blocks at the same point vanish."""
    obs_v = _obs((0.0, 0.0, 0.0), "value")
    obs_d = _obs((0.0, 0.0, 0.0), "derivative")
    assert hermite_block(obs_v, obs_d, H, SIGMA_F2) == pytest.approx(0.0, abs=1e-14)
    assert hermite_block(obs_d, obs_v, H, SIGMA_F2) == pytest.approx(0.0, abs=1e-14)


# ---------------------------------------------------------------------------
# Test group 3 — Compact support and zero blocks
# ---------------------------------------------------------------------------

def test_block_zero_outside_support() -> None:
    """Every block must be exactly zero for r >= h."""
    for kind_i in ("value", "derivative"):
        for kind_j in ("value", "derivative"):
            obs_i = _obs((H, 0.0, 0.0), kind_i, direction=(1.0, 0.0, 0.0))
            obs_j = _obs((0.0, 0.0, 0.0), kind_j, direction=(1.0, 0.0, 0.0))
            assert hermite_block(obs_i, obs_j, H, SIGMA_F2) == 0.0


# ---------------------------------------------------------------------------
# Test group 4 — Symmetry of the block function
# ---------------------------------------------------------------------------

def test_block_symmetry() -> None:
    """The scalar block must satisfy B_ij = B_ji for all kind pairs."""
    offset = (0.03, -0.02, 0.01)
    direction = (0.0, 0.0, 1.0)
    p_i = np.array(offset)
    p_j = np.zeros(3)

    for kind_i in ("value", "derivative"):
        for kind_j in ("value", "derivative"):
            obs_i = _obs(p_i, kind_i, direction)
            obs_j = _obs(p_j, kind_j, direction)
            b_ij = hermite_block(obs_i, obs_j, H, SIGMA_F2)
            b_ji = hermite_block(obs_j, obs_i, H, SIGMA_F2)
            assert b_ij == pytest.approx(b_ji, abs=1e-14), (
                f"Asymmetry for ({kind_i}, {kind_j}): {b_ij} vs {b_ji}"
            )


# ---------------------------------------------------------------------------
# Test group 5 — Assembled matrix
# ---------------------------------------------------------------------------

def _build_observations(points, normals, sigma_0_sq, sigma_1_sq):
    """Build the 2N Hermite observations from N primitives."""
    observations = []
    for p, n in zip(points, normals):
        observations.append(_obs(p, "value", n, value=0.0, noise=sigma_0_sq))
        observations.append(_obs(p, "derivative", n, value=1.0, noise=sigma_1_sq))
    return observations


def _synthetic_dome(n_points=5, radius=0.05):
    """Points on a small spherical cap with outward normals."""
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(n_points, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    points = radius * directions
    return points, directions


def test_assemble_shape() -> None:
    """The assembled matrix has shape (2M, 2M)."""
    points, normals = _synthetic_dome(n_points=5)
    obs = _build_observations(points, normals, SIGMA_0_SQ, SIGMA_1_SQ)
    K = assemble_system_matrix(obs, H, SIGMA_F2)
    assert K.shape == (10, 10)


def test_assemble_symmetry() -> None:
    """The assembled matrix must be symmetric to machine precision."""
    points, normals = _synthetic_dome(n_points=8)
    obs = _build_observations(points, normals, SIGMA_0_SQ, SIGMA_1_SQ)
    K = assemble_system_matrix(obs, H, SIGMA_F2)
    assert np.allclose(K, K.T, atol=1e-14)


def test_assemble_diagonal() -> None:
    """Diagonal entries match the expected per-channel values."""
    points, normals = _synthetic_dome(n_points=4)
    obs = _build_observations(points, normals, SIGMA_0_SQ, SIGMA_1_SQ)
    K = assemble_system_matrix(obs, H, SIGMA_F2)

    expected_value = SIGMA_F2
    expected_derivative = 20.0 * SIGMA_F2 / (H ** 2)

    for idx, observation in enumerate(obs):
        if observation.kind == "value":
            assert K[idx, idx] == pytest.approx(expected_value, abs=1e-12)
        else:
            assert K[idx, idx] == pytest.approx(expected_derivative, rel=1e-10)


def test_assemble_psd() -> None:
    """The covariance matrix K_H must be positive semidefinite.

    This is the critical algebraic test: an incorrect sign in any of the
    four block cases would break PSD here.
    """
    points, normals = _synthetic_dome(n_points=10)
    obs = _build_observations(points, normals, SIGMA_0_SQ, SIGMA_1_SQ)
    K = assemble_system_matrix(obs, H, SIGMA_F2)

    eigenvalues = np.linalg.eigvalsh(K)
    assert eigenvalues.min() > -1e-10, (
        f"K_H is not PSD: min eigenvalue = {eigenvalues.min()}"
    )


def test_assemble_single_value_observation() -> None:
    """A 1x1 matrix from a single value observation equals σ_f²."""
    obs = [_obs((0.0, 0.0, 0.0), "value")]
    K = assemble_system_matrix(obs, H, SIGMA_F2)
    assert K.shape == (1, 1)
    assert K[0, 0] == pytest.approx(SIGMA_F2, abs=1e-12)


def test_assemble_single_derivative_observation() -> None:
    """A 1x1 matrix from a single derivative observation equals 20 σ_f² / h²."""
    obs = [_obs((0.0, 0.0, 0.0), "derivative", direction=(0.0, 0.0, 1.0))]
    K = assemble_system_matrix(obs, H, SIGMA_F2)
    assert K.shape == (1, 1)
    expected = 20.0 * SIGMA_F2 / (H ** 2)
    assert K[0, 0] == pytest.approx(expected, rel=1e-10)


def test_assemble_empty_raises() -> None:
    """Empty observation list must raise."""
    with pytest.raises(ValueError):
        assemble_system_matrix([], H, SIGMA_F2)


def test_assemble_noise_regularization_makes_spd() -> None:
    """Adding the observation noise makes the matrix strictly positive definite."""
    points, normals = _synthetic_dome(n_points=10)
    obs = _build_observations(points, normals, SIGMA_0_SQ, SIGMA_1_SQ)
    K = assemble_system_matrix(obs, H, SIGMA_F2)
    noise = np.diag([observation.noise_variance for observation in obs])
    A = K + noise

    eigenvalues = np.linalg.eigvalsh(A)
    expected_min = min(SIGMA_0_SQ, SIGMA_1_SQ)
    assert eigenvalues.min() >= expected_min - 1e-12, (
        f"A is not SPD: min eigenvalue = {eigenvalues.min()}"
    )