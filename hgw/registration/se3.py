#hgw/registration/se3.py

from __future__ import annotations
import numpy as np

_EPS = 1e-12
_PI_NEAR = 1e-4

def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric 3x3 matrix from a 3-vector."""
    return np.array([
        [0.0,      -v[2],   v[1]],
        [v[2],      0.0,   -v[0]],
        [-v[1],     v[0],   0.0],
    ], dtype=np.float64)

def _vee_skew(M: np.ndarray) -> np.ndarray:
    """Inverse of _skew: extract a 3-vector from a skew-symmetric 3x3 matrix."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=np.float64)

def hat(xi: np.ndarray) -> np.ndarray:
    """Map a 6-vector to the 4x4 se(3) matrix."""
    xi = np.asarray(xi, dtype=np.float64)
    if xi.shape != (6,):
        raise ValueError(f"xi must have shape (6,), got {xi.shape}")
    if not np.all(np.isfinite(xi)):
        raise ValueError("xi must be finite")

    rho = xi[:3]
    phi = xi[3:]

    M = np.zeros((4, 4), dtype=np.float64)
    M[:3, :3] = _skew(phi)
    M[:3, 3] = rho
    return M

def vee(xi_hat: np.ndarray) -> np.ndarray:
    """Inverse of hat: map a 4x4 se(3) matrix to a 6-vector."""
    xi_hat = np.asarray(xi_hat, dtype=np.float64)
    if xi_hat.shape != (4, 4):
        raise ValueError(f"xi_hat must have shape (4, 4), got {xi_hat.shape}")
    if not np.all(np.isfinite(xi_hat)):
        raise ValueError("xi_hat must be finite")

    rho = xi_hat[:3, 3]
    phi = _vee_skew(xi_hat[:3, :3])
    return np.concatenate([rho, phi])

def exp_se3(xi: np.ndarray) -> np.ndarray:
    """Exponential map from se(3) to SE(3)."""
    xi = np.asarray(xi, dtype=np.float64)
    if xi.shape != (6,):
        raise ValueError(f"xi must have shape (6,), got {xi.shape}")
    if not np.all(np.isfinite(xi)):
        raise ValueError("xi must be finite")

    rho = xi[:3]
    phi = xi[3:]
    theta = float(np.linalg.norm(phi))

    if theta < _EPS:
        # Small-angle expansion: R = I + φ^ + 0.5 φ^², V = I + 0.5 φ^
        phi_hat = _skew(phi)
        R = np.eye(3) + phi_hat + 0.5 * (phi_hat @ phi_hat)
        V = np.eye(3) + 0.5 * phi_hat
    else:
        phi_hat = _skew(phi)
        phi_hat_sq = phi_hat @ phi_hat
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        R = (
            np.eye(3)
            + (sin_t / theta) * phi_hat
            + ((1.0 - cos_t) / (theta * theta)) * phi_hat_sq
        )
        V = (
            np.eye(3)
            + ((1.0 - cos_t) / (theta * theta)) * phi_hat
            + ((theta - sin_t) / (theta ** 3)) * phi_hat_sq
        )

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T

def log_se3(T: np.ndarray) -> np.ndarray:
    """Logarithmic map from SE(3) to se(3)."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T must have shape (4, 4), got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError("T must be finite")

    R = T[:3, :3]
    t = T[:3, 3]

    # Check orthogonality of R within tolerance
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        raise ValueError("R block is not orthonormal")

    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < _EPS:
        # Identity rotation: φ = 0, ρ = t
        phi = np.zeros(3)
        rho = t.copy()
    elif theta > np.pi - _PI_NEAR:
        # Near π: extract the rotation axis from the eigenvector of R
        # associated with eigenvalue 1
        eigvals, eigvecs = np.linalg.eigh(R)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        axis = eigvecs[:, idx]
        # Determine the sign of the axis from (R - R^T)
        diff = R - R.T
        if np.linalg.norm(diff) > _PI_NEAR:
            # Non-symmetric case: use the sign of the skew part
            candidate = _vee_skew(diff)
            if np.dot(candidate, axis) < 0:
                axis = -axis
        phi = theta * axis
        # Inverse Jacobian near π
        phi_hat = _skew(phi)
        phi_hat_sq = phi_hat @ phi_hat
        # V^{-1}(π) = I - 0.5 φ^ + (1/π²) (φ^)² · (1 - 0) = I - 0.5 φ^ + φ^²/π²
        V_inv = np.eye(3) - 0.5 * phi_hat + phi_hat_sq / (np.pi * np.pi)
        rho = V_inv @ t
    else:
        # Standard case
        factor = theta / (2.0 * np.sin(theta))
        diff = R - R.T
        phi = factor * _vee_skew(diff)

        phi_hat = _skew(phi)
        phi_hat_sq = phi_hat @ phi_hat
        half = 0.5 * theta
        V_inv_coeff = (
            1.0
            - 0.5 * theta * np.sin(theta) / (1.0 - np.cos(theta))
        ) / (theta * theta)
        V_inv = np.eye(3) - 0.5 * phi_hat + V_inv_coeff * phi_hat_sq
        rho = V_inv @ t

    return np.concatenate([rho, phi])

def rot_align_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix that maps unit vector v1 to unit vector v2."""
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    if v1.shape != (3,) or v2.shape != (3,):
        raise ValueError(f"v1 and v2 must have shape (3,), got {v1.shape}, {v2.shape}")
    if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
        raise ValueError("v1 and v2 must be finite")

    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < _EPS or n2 < _EPS:
        raise ValueError("v1 and v2 must be nonzero")

    v1 = v1 / n1
    v2 = v2 / n2

    c = float(np.dot(v1, v2))
    cross = np.cross(v1, v2)
    s = float(np.linalg.norm(cross))

    if s < _EPS:
        if c > 0.0:
            # Already aligned
            return np.eye(3)
        else:
            # Antiparallel: rotate π around any perpendicular axis
            seed = (
                np.array([1.0, 0.0, 0.0])
                if abs(v1[0]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            axis = np.cross(v1, seed)
            axis = axis / np.linalg.norm(axis)
            return 2.0 * np.outer(axis, axis) - np.eye(3)

    # Generic case
    v_hat = _skew(cross)
    v_hat_sq = v_hat @ v_hat
    return np.eye(3) + v_hat + v_hat_sq * ((1.0 - c) / (s * s))
