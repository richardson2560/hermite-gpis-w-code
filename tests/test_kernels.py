"""Unit tests for the Wendland C² kernel and its derivatives.

These tests verify:
    1. Exact values at critical points.
    2. C² continuity across the support boundary.
    3. Compact support properties.
    4. Coincident-point limits.
    5. Analytical gradient against finite differences.
    6. Analytical Hessian against finite differences.
    7. A hand-computed golden value.

Run with:
    pytest tests/test_kernels.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from hgw.core.kernels import (
    wendland_c2_profile,
    wendland_c2_value,
    wendland_c2_gradient,
    wendland_c2_hessian,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

H = 0.1
SIGMA_F2 = 1.0
EPS_DIFF = 1e-5      # step for first-order finite differences
EPS_DIFF2 = 1e-4     # step for second-order finite differences


def _kernel_scalar(d: np.ndarray) -> float:
    """Evaluate the scalar kernel for a given displacement vector.

    k(d) = sigma_f² · Φ(||d||/h)

    This helper is used by the finite-difference tests. It must always
    agree with the analytical value returned by wendland_c2_value().
    """
    r = float(np.linalg.norm(d))
    return float(wendland_c2_value(np.array([r]), H, SIGMA_F2)[0])


# ---------------------------------------------------------------------------
# Test 1 — Values of the profile at critical points
# ---------------------------------------------------------------------------

def test_profile_at_critical_points() -> None:
    """Check Φ, Φ', Φ'' at u = 0 and u >= 1."""
    u = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    phi, dphi, d2phi = wendland_c2_profile(u)

    # Φ(0) = 1, Φ'(0) = 0, Φ''(0) = -20
    assert phi[0] == pytest.approx(1.0, abs=1e-14)
    assert dphi[0] == pytest.approx(0.0, abs=1e-14)
    assert d2phi[0] == pytest.approx(-20.0, abs=1e-14)

    # Φ(u) = 0 for u >= 1, and all derivatives vanish
    for i in (4, 5, 6):
        assert phi[i] == 0.0
        assert dphi[i] == 0.0
        assert d2phi[i] == 0.0

    # Intermediate value sanity check: Φ(0.5) = 0.5^4 * 3 = 0.1875
    assert phi[2] == pytest.approx(0.1875, abs=1e-14)


# ---------------------------------------------------------------------------
# Test 2 — C² continuity across the support boundary
# ---------------------------------------------------------------------------

def test_c2_continuity_at_boundary() -> None:
    """Φ, Φ', Φ'' all approach zero as u → 1⁻."""
    eps = 1e-6
    u_left = np.array([1.0 - eps])
    u_right = np.array([1.0 + eps])

    phi_l, dphi_l, d2phi_l = wendland_c2_profile(u_left)
    phi_r, dphi_r, d2phi_r = wendland_c2_profile(u_right)

    # Approaching from the interior, everything tends to zero
    assert abs(phi_l[0]) < 1e-20
    assert abs(dphi_l[0]) < 1e-15
    assert abs(d2phi_l[0]) < 1e-10

    # Outside the support, everything is exactly zero
    assert phi_r[0] == 0.0
    assert dphi_r[0] == 0.0
    assert d2phi_r[0] == 0.0


# ---------------------------------------------------------------------------
# Test 3 — Compact support of the kernel value
# ---------------------------------------------------------------------------

def test_value_compact_support() -> None:
    """k(r) = 0 exactly for r >= h."""
    r = np.array([H, H + 1e-6, 2 * H, 10 * H])
    k = wendland_c2_value(r, H, SIGMA_F2)
    assert np.all(k == 0.0)


# ---------------------------------------------------------------------------
# Test 4 — Kernel value at the origin
# ---------------------------------------------------------------------------

def test_value_at_origin() -> None:
    """k(0) = σ_f² · Φ(0) = σ_f²."""
    r = np.array([0.0])
    k = wendland_c2_value(r, H, SIGMA_F2)
    assert k[0] == pytest.approx(SIGMA_F2, abs=1e-14)


# ---------------------------------------------------------------------------
# Test 5 — Gradient at the origin
# ---------------------------------------------------------------------------

def test_gradient_at_origin() -> None:
    """∇k(x, p) → 0 as x → p."""
    d = np.zeros(3)
    r = 0.0
    g = wendland_c2_gradient(d, r, H, SIGMA_F2)
    assert np.allclose(g, np.zeros(3), atol=1e-14)


# ---------------------------------------------------------------------------
# Test 6 — Gradient outside the support
# ---------------------------------------------------------------------------

def test_gradient_outside_support() -> None:
    """∇k(x, p) = 0 for r >= h."""
    d = np.array([H, 0.0, 0.0])
    r = H
    g = wendland_c2_gradient(d, r, H, SIGMA_F2)
    assert np.allclose(g, np.zeros(3), atol=1e-14)


# ---------------------------------------------------------------------------
# Test 7 — Hessian at the origin
# ---------------------------------------------------------------------------

def test_hessian_at_origin() -> None:
    """H(0) = -20 · σ_f² / h² · I₃ (coincident-point limit)."""
    d = np.zeros(3)
    r = 0.0
    Hmat = wendland_c2_hessian(d, r, H, SIGMA_F2)

    expected = -20.0 * SIGMA_F2 / (H ** 2) * np.eye(3)
    assert np.allclose(Hmat, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 8 — Hessian outside the support
# ---------------------------------------------------------------------------

def test_hessian_outside_support() -> None:
    """H(x, p) = 0 for r >= h."""
    d = np.array([0.0, H, 0.0])
    r = H
    Hmat = wendland_c2_hessian(d, r, H, SIGMA_F2)
    assert np.allclose(Hmat, np.zeros((3, 3)), atol=1e-14)


# ---------------------------------------------------------------------------
# Test 9 — Analytical gradient matches finite differences
# ---------------------------------------------------------------------------

def test_gradient_matches_finite_diff() -> None:
    """Verify ∇k against central finite differences."""
    d = np.array([0.03, -0.02, 0.01])
    r = float(np.linalg.norm(d))
    assert 0.0 < r < H  # inside the support

    g_analytic = wendland_c2_gradient(d, r, H, SIGMA_F2)

    g_numeric = np.zeros(3)
    for i in range(3):
        d_p = d.copy()
        d_m = d.copy()
        d_p[i] += EPS_DIFF
        d_m[i] -= EPS_DIFF
        g_numeric[i] = (_kernel_scalar(d_p) - _kernel_scalar(d_m)) / (2 * EPS_DIFF)

    assert np.allclose(g_analytic, g_numeric, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Test 10 — Analytical Hessian matches finite differences
# ---------------------------------------------------------------------------

def test_hessian_matches_finite_diff() -> None:
    """Verify H(x, p) against central finite differences."""
    d = np.array([0.03, -0.02, 0.01])
    r = float(np.linalg.norm(d))
    assert 0.0 < r < H

    H_analytic = wendland_c2_hessian(d, r, H, SIGMA_F2)

    H_numeric = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            d_pp = d.copy(); d_pp[i] += EPS_DIFF2; d_pp[j] += EPS_DIFF2
            d_pm = d.copy(); d_pm[i] += EPS_DIFF2; d_pm[j] -= EPS_DIFF2
            d_mp = d.copy(); d_mp[i] -= EPS_DIFF2; d_mp[j] += EPS_DIFF2
            d_mm = d.copy(); d_mm[i] -= EPS_DIFF2; d_mm[j] -= EPS_DIFF2
            H_numeric[i, j] = (
                _kernel_scalar(d_pp)
                - _kernel_scalar(d_pm)
                - _kernel_scalar(d_mp)
                + _kernel_scalar(d_mm)
            ) / (4 * EPS_DIFF2 ** 2)

    # Symmetrize to avoid tiny numerical asymmetries in the FD reference
    H_numeric = 0.5 * (H_numeric + H_numeric.T)

    assert np.allclose(H_analytic, H_numeric, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Test 11 — Hand-computed golden value
# ---------------------------------------------------------------------------

def test_golden_value() -> None:
    """Verify a specific kernel value computed by hand.

    For r = 0.05, h = 0.1, sigma_f² = 1.0:
        u = r / h = 0.5
        Φ(0.5) = (1 - 0.5)^4 * (4 * 0.5 + 1) = 0.0625 * 3 = 0.1875
        k(0.05) = 1.0 * 0.1875 = 0.1875
    """
    r = np.array([0.05])
    k = wendland_c2_value(r, h=0.1, sigma_f2=1.0)
    assert k[0] == pytest.approx(0.1875, abs=1e-12)


# ---------------------------------------------------------------------------
# Test 12 — Gradient radial direction
# ---------------------------------------------------------------------------

def test_gradient_is_radial() -> None:
    """∇k(x, p) must be parallel to d = x - p."""
    d = np.array([0.04, 0.0, 0.0])
    r = float(np.linalg.norm(d))
    g = wendland_c2_gradient(d, r, H, SIGMA_F2)

    # Cross product with d should be zero (parallel vectors)
    cross = np.cross(g, d)
    assert np.allclose(cross, np.zeros(3), atol=1e-14)

    # The gradient should point opposite to d (kernel decreases outward)
    # so g · d < 0 for r > 0 inside the support.
    assert np.dot(g, d) < 0.0


# ---------------------------------------------------------------------------
# Test 13 — Hessian symmetry
# ---------------------------------------------------------------------------

def test_hessian_is_symmetric() -> None:
    """H(x, p) must be a symmetric 3×3 matrix."""
    d = np.array([0.02, -0.03, 0.04])
    r = float(np.linalg.norm(d))
    Hmat = wendland_c2_hessian(d, r, H, SIGMA_F2)
    assert np.allclose(Hmat, Hmat.T, atol=1e-14)