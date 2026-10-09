#hgw/registration/p2m.py

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from hgw.models.evaluator import HermiteGPIS_W
from hgw.registration.se3 import exp_se3, log_se3
from hgw.stats.deviance import DevianceConfig, deviance_test

# Sentinel for "no valid covariance" (finite, symmetric, PD)
_COV_SENTINEL = 1e12

@dataclass(frozen=True)
class P2MConfig:
    """Configuration for the P2M Gauss-Newton solver."""

    max_iterations: int = 50
    conv_tol: float = 1e-6
    damping: float = 1e-6
    huber_k: float = 1.5
    max_translation_step: float = 0.05
    max_rotation_step: float = 0.5
    min_points: int = 10
    min_active_fraction: float = 0.05
    line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
    cost_increase_tol: float = 1e-8

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.conv_tol <= 0.0:
            raise ValueError("conv_tol must be positive")
        if self.damping < 0.0:
            raise ValueError("damping must be non-negative")
        if self.huber_k <= 0.0:
            raise ValueError("huber_k must be positive")
        if self.max_translation_step <= 0.0:
            raise ValueError("max_translation_step must be positive")
        if self.max_rotation_step <= 0.0:
            raise ValueError("max_rotation_step must be positive")
        if self.min_points < 1:
            raise ValueError("min_points must be at least 1")
        if not 0.0 <= self.min_active_fraction <= 1.0:
            raise ValueError("min_active_fraction must be in [0, 1]")
        if not self.line_search_alphas:
            raise ValueError("line_search_alphas must be non-empty")
        if any(a <= 0.0 or a > 1.0 for a in self.line_search_alphas):
            raise ValueError("line_search_alphas must lie in (0, 1]")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class P2MResult:
    """Result of the P2M Gauss-Newton solver."""

    status: str
    T_estimated: np.ndarray
    covariance: np.ndarray
    residual_rms: float
    active_fraction: float
    n_inliers: int
    iterations: int
    reason: str

    def __post_init__(self) -> None:
        if self.status not in ("ACCEPTED", "REJECTED", "NOT_CONVERGED"):
            raise ValueError(f"invalid status: {self.status}")
        T = np.asarray(self.T_estimated, dtype=np.float64)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise ValueError("T_estimated must be a finite (4, 4) matrix")
        C = np.asarray(self.covariance, dtype=np.float64)
        if C.shape != (6, 6) or not np.all(np.isfinite(C)):
            raise ValueError("covariance must be a finite (6, 6) matrix")
        if self.residual_rms < 0.0:
            raise ValueError("residual_rms must be non-negative")
        if not 0.0 <= self.active_fraction <= 1.0:
            raise ValueError("active_fraction must be in [0, 1]")
        if not self.reason:
            raise ValueError("reason must not be empty")

        object.__setattr__(self, "T_estimated", T)
        object.__setattr__(self, "covariance", C)

def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0,     -v[2],   v[1]],
        [v[2],     0.0,   -v[0]],
        [-v[1],    v[0],   0.0],
    ], dtype=np.float64)


def _inv_se3(T: np.ndarray) -> np.ndarray:
    """Explicit SE(3) inverse: numerically stable."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _huber_weights(z: np.ndarray, k: float) -> np.ndarray:
    abs_z = np.abs(z)
    w = np.ones_like(z)
    outlier = abs_z > k
    w[outlier] = k / abs_z[outlier]
    return w


def _apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ points.T).T + T[:3, 3]


def _residual_jacobian(x_model: np.ndarray, grad_model: np.ndarray) -> np.ndarray:
    """J_j = ∇m(x)^T [I_3 | -x^]."""
    J_data = np.hstack([np.eye(3), -_skew(x_model)])
    return grad_model @ J_data


def _residual_jacobians(points, grads):
    """Jacobian of left SE(3) increments, translation then rotation."""
    return np.hstack([grads, np.cross(points, grads)])


def _huber_cost(z: np.ndarray, k: float) -> float:
    """Sum of Huber loss over standardized residuals."""
    abs_z = np.abs(z)
    quad = 0.5 * z ** 2
    lin = k * (abs_z - 0.5 * k)
    return float(np.sum(np.where(abs_z <= k, quad, lin)))


def _total_cost(
    T: np.ndarray,
    scene_points: np.ndarray,
    model: HermiteGPIS_W,
    P_prior_inv: np.ndarray,
    T_prior: np.ndarray,
    huber_k: float,
) -> float:
    """Evaluate the full weighted cost at a given pose."""
    points_in_model = _apply_transform(T, scene_points)
    means, _, variances = model.evaluate_many(points_in_model)
    sigma_j = np.sqrt(variances + model._sigma_0_sq)
    z = means / sigma_j
    data_cost = _huber_cost(z, huber_k)
    xi_prior = log_se3(T_prior @ _inv_se3(T))
    prior_cost = 0.5 * float(xi_prior @ P_prior_inv @ xi_prior)
    return data_cost + prior_cost


def p2m_register(
    scene_points: np.ndarray,
    model: HermiteGPIS_W,
    T_initial: np.ndarray,
    *,
    T_prior: np.ndarray | None = None,
    P_prior: np.ndarray | None = None,
    config: P2MConfig = P2MConfig(),
    deviance_config: DevianceConfig = DevianceConfig(),
) -> P2MResult:
    """Run Gauss-Newton P2M registration with Huber weights and line search."""
    scene_points = np.asarray(scene_points, dtype=np.float64)
    if scene_points.ndim != 2 or scene_points.shape[1] != 3:
        raise ValueError(
            f"scene_points must have shape (N, 3), got {scene_points.shape}"
        )
    if not np.all(np.isfinite(scene_points)):
        raise ValueError("scene_points must be finite")
    if len(scene_points) < config.min_points:
        raise ValueError(
            f"at least {config.min_points} points required, got {len(scene_points)}"
        )

    if not isinstance(model, HermiteGPIS_W):
        raise TypeError("model must be a HermiteGPIS_W instance")

    T = np.asarray(T_initial, dtype=np.float64).copy()
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("T_initial must be a finite (4, 4) matrix")

    if T_prior is None:
        T_prior = T.copy()
    else:
        T_prior = np.asarray(T_prior, dtype=np.float64)
        if T_prior.shape != (4, 4) or not np.all(np.isfinite(T_prior)):
            raise ValueError("T_prior must be a finite (4, 4) matrix")

    if P_prior is None:
        P_prior = np.eye(6) * 1e6
    else:
        P_prior = np.asarray(P_prior, dtype=np.float64)
        if P_prior.shape != (6, 6) or not np.all(np.isfinite(P_prior)):
            raise ValueError("P_prior must be a finite (6, 6) matrix")
        if np.any(np.linalg.eigvalsh(0.5 * (P_prior + P_prior.T)) <= 0.0):
            raise ValueError("P_prior must be positive definite")

    P_prior_inv = np.linalg.inv(P_prior)
    sigma_0_sq = model._sigma_0_sq

    # --- Initial support check --------------------------------------------
    points_in_model = _apply_transform(T, scene_points)
    active_fraction = model.support_fraction(points_in_model)
    if active_fraction < config.min_active_fraction:
        return P2MResult(
            status="REJECTED",
            T_estimated=T,
            covariance=np.eye(6) * _COV_SENTINEL,
            residual_rms=math.inf,
            active_fraction=active_fraction,
            n_inliers=0,
            iterations=0,
            reason=(
                f"initial active fraction {active_fraction:.3f} below "
                f"threshold {config.min_active_fraction}"
            ),
        )

    # --- Main Gauss-Newton loop -------------------------------------------
    current_cost = _total_cost(
        T, scene_points, model, P_prior_inv, T_prior, config.huber_k
    )
    converged = False
    iterations = 0
    last_reason = "not started"

    for iterations in range(1, config.max_iterations + 1):
        # 1. Evaluate model at transformed points
        points_in_model = _apply_transform(T, scene_points)
        means, grads, variances = model.evaluate_many(points_in_model)

        sigma_j = np.sqrt(variances + sigma_0_sq)
        z = means / sigma_j
        weights = _huber_weights(z, config.huber_k)

        # 2. Weighted Jacobian and residuals (consistent weighting)
        inv_sigma_sq = 1.0 / (sigma_j ** 2)
        w_times_inv = weights * inv_sigma_sq

        J = _residual_jacobians(points_in_model, grads)

        # 3. Normal system
        H = J.T @ (w_times_inv[:, None] * J)
        H += P_prior_inv
        H += config.damping * np.eye(6)

        xi_prior = log_se3(T_prior @ _inv_se3(T))
        b = P_prior_inv @ xi_prior - J.T @ (w_times_inv * means)

        try:
            dxi = np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            last_reason = "singular normal matrix"
            break

        # 4. Clamp the step for stability
        trans_step = float(np.linalg.norm(dxi[:3]))
        rot_step = float(np.linalg.norm(dxi[3:]))
        if trans_step > config.max_translation_step:
            dxi[:3] *= config.max_translation_step / trans_step
        if rot_step > config.max_rotation_step:
            dxi[3:] *= config.max_rotation_step / rot_step

        # 5. Backtracking line search: accept the first alpha that reduces cost
        accepted = False
        best_T = T
        best_cost = current_cost
        for alpha in config.line_search_alphas:
            T_try = exp_se3(alpha * dxi) @ T
            try:
                cost_try = _total_cost(
                    T_try, scene_points, model,
                    P_prior_inv, T_prior, config.huber_k,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            if cost_try < current_cost - config.cost_increase_tol:
                best_T = T_try
                best_cost = cost_try
                accepted = True
                break

        if not accepted:
            # No alpha reduced the cost: we are at a local minimum
            converged = True
            last_reason = "line search exhausted (local minimum)"
            break

        T = best_T
        current_cost = best_cost

        # 6. Convergence test on the applied step
        applied_step = float(np.linalg.norm(alpha * dxi))
        if applied_step < config.conv_tol:
            converged = True
            last_reason = "converged"
            break

    points_in_model = _apply_transform(T, scene_points)
    means, grads, variances = model.evaluate_many(points_in_model)

    sigma_j = np.sqrt(variances + sigma_0_sq)
    z = means / sigma_j
    weights = _huber_weights(z, config.huber_k)
    # Robust optimization weights do not define statistical inliers.
    dev = deviance_test(scene_points, T, model, config=deviance_config)
    inlier_mask = dev.inlier_mask
    n_inliers = int(np.sum(inlier_mask))
    active_fraction = model.support_fraction(points_in_model)

    if n_inliers > 0:
        residual_rms = float(np.sqrt(np.mean(means[inlier_mask] ** 2)))
    else:
        residual_rms = math.inf

    # Final information matrix (without damping)
    J = _residual_jacobians(points_in_model, grads)

    inv_sigma_sq = 1.0 / (sigma_j ** 2)
    w_times_inv = weights * inv_sigma_sq
    H_gn = J.T @ (w_times_inv[:, None] * J) + P_prior_inv

    try:
        covariance = np.linalg.inv(H_gn)
        if not np.all(np.isfinite(covariance)):
            covariance = np.eye(6) * _COV_SENTINEL
    except np.linalg.LinAlgError:
        covariance = np.eye(6) * _COV_SENTINEL

    if not converged:
        status = "NOT_CONVERGED"
        reason = last_reason or f"reached max_iterations ({config.max_iterations})"
    elif active_fraction < config.min_active_fraction:
        status = "REJECTED"
        reason = "final active fraction below threshold"
    elif n_inliers < config.min_points:
        status = "REJECTED"
        reason = f"too few inliers ({n_inliers})"
    else:
        status = "ACCEPTED"
        reason = last_reason or "ok"

    return P2MResult(
        status=status,
        T_estimated=T,
        covariance=covariance,
        residual_rms=residual_rms,
        active_fraction=active_fraction,
        n_inliers=n_inliers,
        iterations=iterations,
        reason=reason,
    )