#hgw/registration/anchor.py

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from hgw.models.evaluator import HermiteGPIS_W
from hgw.registration.se3 import exp_se3, rot_align_vectors

def _select_anchor(
    scene_points: np.ndarray,
    scene_normals: np.ndarray,
    k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the scene anchor: the point with the most reliable normal."""
    n_points = len(scene_points)
    k_eff = min(k, n_points)
    tree = cKDTree(scene_points)
    _, idx = tree.query(scene_points, k=k_eff)
    neighbor_normals = scene_normals[idx]              # (N, k_eff, 3)
    mean_normals = neighbor_normals.mean(axis=1)       # (N, 3)
    norms = np.linalg.norm(mean_normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    mean_normals /= norms
    scores = np.sum(scene_normals * mean_normals, axis=1)
    anchor_idx = int(np.argmax(scores))
    return scene_points[anchor_idx], scene_normals[anchor_idx]

def _principal_axes(model_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of the model's point covariance."""
    centroid = model_points.mean(axis=0)
    centered = model_points - centroid
    cov = centered.T @ centered / len(model_points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]

def _valid_axes(eigenvalues: np.ndarray, tau_axis: float) -> list[int]:
    """Apply the eigengap gate and return valid axis indices."""
    lam = eigenvalues
    if lam[0] <= 0.0:
        return [0]

    gap_12 = float((lam[0] - lam[1]) / lam[0])
    gap_23 = float((lam[1] - lam[2]) / lam[0])

    if gap_12 > tau_axis and gap_23 > tau_axis:
        return [0, 1, 2]
    if gap_12 > tau_axis:
        return [0]
    return [0]

def _extremal_primitive(model_points: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Return the model point that is extremal along the given axis."""
    projections = model_points @ axis
    idx = int(np.argmax(projections))
    return model_points[idx]

def generate_hermite_seeds(
    scene_points: np.ndarray,
    scene_normals: np.ndarray,
    model: HermiteGPIS_W,
    *,
    tau_axis: float = 0.15,
    n_axial_angles: int = 4,
    n_eval: int = 30,
    min_active_fraction: float = 0.1,
    top_k: int = 2,
) -> list[np.ndarray]:
    """Generate candidate SE(3) poses via Hermite anchor seeding."""
    scene_points = np.asarray(scene_points, dtype=np.float64)
    scene_normals = np.asarray(scene_normals, dtype=np.float64)

    if scene_points.ndim != 2 or scene_points.shape[1] != 3:
        raise ValueError(f"scene_points must have shape (N, 3), got {scene_points.shape}")
    if scene_normals.shape != scene_points.shape:
        raise ValueError(
            f"scene_normals shape {scene_normals.shape} must match "
            f"scene_points shape {scene_points.shape}"
        )
    if not np.all(np.isfinite(scene_points)):
        raise ValueError("scene_points must be finite")
    if not np.all(np.isfinite(scene_normals)):
        raise ValueError("scene_normals must be finite")
    if len(scene_points) < 3:
        raise ValueError("at least 3 scene points are required for anchor selection")

    norms = np.linalg.norm(scene_normals, axis=1)
    if np.any(norms < 1e-10):
        raise ValueError("scene_normals must be nonzero")
    scene_normals = scene_normals / norms[:, None]
    if not isinstance(model, HermiteGPIS_W):
        raise TypeError("model must be a HermiteGPIS_W instance")
    if n_axial_angles < 1:
        raise ValueError("n_axial_angles must be at least 1")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    q_star, n_star = _select_anchor(scene_points, scene_normals)
    model_points = model._points 
    eigenvalues, eigenvectors = _principal_axes(model_points)
    axes = _valid_axes(eigenvalues, tau_axis)
    seeds: list[np.ndarray] = []
    axial_angles = np.linspace(0.0, 2.0 * np.pi, n_axial_angles, endpoint=False)

    for k in axes:
        v_k = eigenvectors[:, k]
        p_k = _extremal_primitive(model_points, v_k)

        # Alignment: n_star -> v_k
        R_align = rot_align_vectors(n_star, v_k)

        for theta in axial_angles:
            # Rotation by theta around v_k
            xi = np.concatenate([np.zeros(3), v_k * theta])
            R_theta = exp_se3(xi)[:3, :3]

            R_final = R_theta @ R_align
            t_final = p_k - R_final @ q_star

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_final
            T[:3, 3] = t_final
            seeds.append(T)

    n_points = len(scene_points)
    step = max(1, n_points // n_eval)
    subsample = scene_points[::step][:n_eval]

    ranked: list[tuple[float, np.ndarray]] = []
    for T in seeds:
        transformed = (T[:3, :3] @ subsample.T).T + T[:3, 3]
        frac = model.support_fraction(transformed)
        if frac >= min_active_fraction:
            ranked.append((frac, T))

    if not ranked:
        raise ValueError(
            "no seed passed the active-support filter; "
            "check h, tau_axis, and min_active_fraction"
        )

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [T for _, T in ranked[:top_k]]
