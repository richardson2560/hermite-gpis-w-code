#hgw/models/builder.py

from __future__ import annotations
import numpy as np
from hgw.core.hermite import HermiteObservation, assemble_system_matrix
from hgw.core.sampling import pivoted_cholesky
from hgw.models.artifact import ModelArtifact

def build_prior(
        points: np.ndarray,
        normals: np.ndarray,
        h: float,
        sigma_f2: float,
        sigma_0_sq: float,
        sigma_1_sq: float,
        m_target: float,
        eps_tol: float=1e-6,
) -> ModelArtifact:
    """Build a compact HGW prior from a clean point cloud.

    Args:
        points:     (N, 3) array of clean primitive positions in meters.
        normals:    (N, 3) array of unit outward normals.
        h:          positive support radius in meters.
        sigma_f2:   positive kernel amplitude.
        sigma_0_sq: positive value-channel noise variance.
        sigma_1_sq: positive derivative-channel noise variance.
        m_target:   maximum number of primitives to select.
        eps_tol:    stopping tolerance for the pivoted Cholesky.

    Returns:
        A validated ModelArtifact.

    Raises:
        ValueError: on any malformed input, empty pivot selection, or
                    a system matrix that fails Cholesky factorization.
    """
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if normals.shape != points.shape:
        raise ValueError(
            f"normals shape {normals.shape} must match points shape {points.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    if not np.all(np.isfinite(normals)):
        raise ValueError("normals must be finite")

    for name, value in (
        ("h", h),
        ("sigma_f2", sigma_f2),
        ("sigma_0_sq", sigma_0_sq),
        ("sigma_1_sq", sigma_1_sq),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value!r}")

    pivots = pivoted_cholesky(
        points,
        h=h,
        sigma_f2=sigma_f2,
        sigma_0_sq=sigma_0_sq,
        m_target=m_target,
        eps_tol=eps_tol,
    )
    if pivots.size == 0:
        raise ValueError(
            "pivoted Cholesky selected zero pivots; "
            "check eps_tol against sigma_f2 + sigma_0_sq"
        )

    points_sel = points[pivots]
    normals_sel = normals[pivots]

    observations: list[HermiteObservation] = []
    for p, n in zip(points_sel, normals_sel):
        direction = tuple(float(v) for v in n)
        point = tuple(float(v) for v in p)
        observations.append(
            HermiteObservation(
                point_m=point,
                kind="value",
                direction=direction,
                value=0.0,
                noise_variance=sigma_0_sq,
            )
        )
        observations.append(
            HermiteObservation(
                point_m=point,
                kind="derivative",
                direction=direction,
                value=1.0,
                noise_variance=sigma_1_sq,
            )
        )

    K_H = assemble_system_matrix(observations, h=h, sigma_f2=sigma_f2)

    noise = np.array([obs.noise_variance for obs in observations], dtype=np.float64)
    A = K_H + np.diag(noise)

    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "regularized Hermite system is not positive definite; "
            "check noise variances and kernel parameters"
        ) from exc

    y = np.array([obs.value for obs in observations], dtype=np.float64)
    z = np.linalg.solve(L, y)
    alpha = np.linalg.solve(L.T, z)

    return ModelArtifact(
        points=points_sel,
        normals=normals_sel,
        alpha=alpha,
        h=h,
        sigma_f2=sigma_f2,
        sigma_0_sq=sigma_0_sq,
        sigma_1_sq=sigma_1_sq,
    )
    