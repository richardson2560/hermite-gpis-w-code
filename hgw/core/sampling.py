#hgw/core/sampling.py

from __future__ import annotations
from hgw.core.kernels import wendland_c2_value
import numpy as np

def pivoted_cholesky(
        points: np.ndarray,
        h: float,
        sigma_f2: float,
        sigma_0_sq: float,
        m_target: int,
        eps_tol: float = 1e-6,
) -> np.ndarray:
    """
    Select M pivots from N points via pivoted Cholesky.

    The selection maximizes the residual diagonal of the scalar value kernel,
    which corresponds to the "least explained" location given the pivots
    already selected. The algorithm stops either when m_target pivots have
    been selected or when the maximum residual falls below eps_tol.

    Args:
        points:     (N, 3) array of primitive positions in meters.
        h:          positive support radius in meters.
        sigma_f2:   positive kernel amplitude.
        sigma_0_sq: positive value-channel noise variance.
        m_target:   maximum number of pivots to select.
        eps_tol:    stopping tolerance on the residual diagonal.

    Returns:
        (M,) array of pivot indices with M <= m_target.

    Raises:
        ValueError: on any malformed input or inconsistent shape.
    """

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"points must have shape (N, 3) with finite values, got shape {points.shape}")
    n_points = points.shape[0]
    if n_points < 1:
        raise ValueError(f"points must have at least one point: {points}")

    if not isinstance(m_target, int) or m_target < 1:
        raise ValueError(f"m_target must be a positive integer: {m_target}")
    if m_target > n_points:
        raise ValueError(f"m_target ({m_target}) exceeds the number of points ({n_points})")
    for name, value in (("h", h), ("sigma_f2", sigma_f2), ("sigma_0_sq", sigma_0_sq)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value!r}")
    if not np.isfinite(eps_tol) or eps_tol < 0.0:
        raise ValueError(f"eps_tol must be non-negative and finite, got {eps_tol!r}")

    d = np.full(n_points, sigma_f2 + sigma_0_sq, dtype=np.float64)
    L = np.zeros((n_points, m_target), dtype=np.float64)
    pivots = np.zeros(m_target, dtype=np.int64)
    n_selected = 0

    for k in range(m_target):
        j = int(np.argmax(d))
        if d[j] <= 0.0 or d[j] < eps_tol:
            break
        pivots[n_selected] = j
        n_selected += 1
        distances = np.linalg.norm(points - points[j], axis=1)
        k_col = wendland_c2_value(distances, h, sigma_f2)
        k_col[j] += sigma_0_sq
        if k == 0:
            L[:, 0] = k_col / np.sqrt(d[j])
        else:
            correction = L[:, :k] @ L[j, :k]
            L[:, k] = (k_col - correction) / np.sqrt(d[j])
        d = d - L[:, k] ** 2
        np.maximum(d, 0.0, out=d)
        d[pivots[:n_selected]] = 0.0

    return pivots[:n_selected]