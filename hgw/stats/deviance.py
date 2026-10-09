#hgw/stats/deviance.py

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.stats import chi2
from hgw.models.evaluator import HermiteGPIS_W

@dataclass(frozen=True)
class DevianceConfig:
    """Configuration for the deviance test."""
    alpha_local: float = 0.01
    alpha_global: float = 0.05
    min_coverage_fraction: float = 0.20
    min_gradient_norm: float = 1e-3
    min_inliers: int = 3
    def __post_init__(self) -> None:
        if not 0.0 < self.alpha_local < 1.0:
            raise ValueError("alpha_local must be in (0, 1)")
        if not 0.0 < self.alpha_global < 1.0:
            raise ValueError("alpha_global must be in (0, 1)")
        if not 0.0 < self.min_coverage_fraction <= 1.0:
            raise ValueError("min_coverage_fraction must be in (0, 1]")
        if self.min_gradient_norm <= 0.0:
            raise ValueError("min_gradient_norm must be positive")
        if self.min_inliers < 1:
            raise ValueError("min_inliers must be at least 1")

@dataclass(frozen=True)
class ResidualData:
    """Orthogonal residuals and their variances."""
    residuals: np.ndarray          # (N,) signed r_j
    residual_variances: np.ndarray # (N,) σ_r,j²
    gradient_norms: np.ndarray     # (N,) ||∇m||
    valid_mask: np.ndarray         # (N,) bool

@dataclass(frozen=True)
class DevianceResult:
    """Result of the full deviance test."""
    status: str
    Q_statistic: float
    threshold: float
    p_value: float
    M_inliers: int
    M_total: int
    coverage: float
    inlier_mask: np.ndarray
    residual_data: ResidualData
    reason: str

    def __post_init__(self) -> None:
        valid_statuses = {
            "ACCEPTED",
            "REJECTED_DEVIANCE",
            "REJECTED_COVERAGE",
            "REJECTED_INSUFFICIENT_INLIERS",
        }
        if self.status not in valid_statuses:
            raise ValueError(f"invalid status: {self.status}")
        if not math.isfinite(self.Q_statistic) and self.Q_statistic != math.inf:
            raise ValueError("Q_statistic must be finite or +inf")
        if self.M_inliers < 0 or self.M_total < 0:
            raise ValueError("counts must be non-negative")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be in [0, 1]")
        if not self.reason:
            raise ValueError("reason must not be empty")

        object.__setattr__(
            self, "inlier_mask", np.asarray(self.inlier_mask, dtype=bool)
        )

def compute_residuals(
    scene_points: np.ndarray,
    T: np.ndarray,
    model: HermiteGPIS_W,
    config: DevianceConfig = DevianceConfig(),
) -> ResidualData:
    """Compute signed orthogonal residuals and their variances."""
    scene_points = np.asarray(scene_points, dtype=np.float64)
    if scene_points.ndim != 2 or scene_points.shape[1] != 3:
        raise ValueError(
            f"scene_points must have shape (N, 3), got {scene_points.shape}"
        )
    if not np.all(np.isfinite(scene_points)):
        raise ValueError("scene_points must be finite")

    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("T must be a finite (4, 4) matrix")

    if not isinstance(model, HermiteGPIS_W):
        raise TypeError("model must be a HermiteGPIS_W instance")

    points_in_model = (T[:3, :3] @ scene_points.T).T + T[:3, 3]
    means, grads, variances = model.evaluate_many(points_in_model)
    grad_norms = np.linalg.norm(grads, axis=1)
    valid_mask = grad_norms >= config.min_gradient_norm

    safe_grad = np.where(valid_mask, grad_norms, 1.0)
    residuals = np.where(valid_mask, means / safe_grad, 0.0)

    sigma_0_sq = float(model._sigma_0_sq)
    residual_variances = np.where(
        valid_mask,
        variances / (safe_grad ** 2) + sigma_0_sq,
        1.0,
    )

    return ResidualData(
        residuals=residuals,
        residual_variances=residual_variances,
        gradient_norms=grad_norms,
        valid_mask=valid_mask,
    )

def deviance_test(
    scene_points: np.ndarray,
    T: np.ndarray,
    model: HermiteGPIS_W,
    *,
    expected_point_count: int | None = None,
    config: DevianceConfig = DevianceConfig(),
) -> DevianceResult:
    """Run nominal deviance diagnostics.

    Local trimming, spatial correlation, and pose fitting invalidate an exact
    chi-square null law. Calibrate thresholds before claiming test size.
    """
    r_data = compute_residuals(scene_points, T, model, config)
    N_total = len(scene_points)
    if expected_point_count is None:
        expected_point_count = N_total
    if expected_point_count < 1:
        raise ValueError("expected_point_count must be at least 1")
    gamma_gate = float(chi2.ppf(1.0 - config.alpha_local, df=1))

    safe_sigma_sq = np.where(
        r_data.residual_variances > 0.0, r_data.residual_variances, 1.0
    )
    e_j = np.where(
        r_data.valid_mask & (r_data.residual_variances > 0.0),
        r_data.residuals ** 2 / safe_sigma_sq,
        np.inf,
    )

    inlier_mask = r_data.valid_mask & (e_j <= gamma_gate)
    M_inliers = int(np.sum(inlier_mask))
    coverage = M_inliers / expected_point_count

    # --- Coverage gate (evaluated first: it is the most fundamental) ------
    if coverage < config.min_coverage_fraction:
        Q = float(np.sum(e_j[inlier_mask])) if M_inliers > 0 else math.inf
        threshold = (
            float(chi2.ppf(1.0 - config.alpha_global, df=M_inliers))
            if M_inliers > 0 else math.inf
        )
        p_value = (
            float(chi2.sf(Q, df=M_inliers))
            if M_inliers > 0 else 0.0
        )
        return DevianceResult(
            status="REJECTED_COVERAGE",
            Q_statistic=Q,
            threshold=threshold,
            p_value=p_value,
            M_inliers=M_inliers,
            M_total=N_total,
            coverage=coverage,
            inlier_mask=inlier_mask,
            residual_data=r_data,
            reason=(
                f"coverage {coverage:.3f} below threshold "
                f"{config.min_coverage_fraction}"
            ),
        )

    # --- Insufficient inliers (only reachable when coverage passes) -------
    if M_inliers < config.min_inliers:
        return DevianceResult(
            status="REJECTED_INSUFFICIENT_INLIERS",
            Q_statistic=math.inf,
            threshold=math.inf,
            p_value=0.0,
            M_inliers=M_inliers,
            M_total=N_total,
            coverage=coverage,
            inlier_mask=inlier_mask,
            residual_data=r_data,
            reason=f"only {M_inliers} inliers, need at least {config.min_inliers}",
        )

    # --- Global statistic --------------------------------------------------
    Q = float(np.sum(e_j[inlier_mask]))
    threshold = float(chi2.ppf(1.0 - config.alpha_global, df=M_inliers))
    p_value = float(chi2.sf(Q, df=M_inliers))

    # --- Deviance gate -----------------------------------------------------
    if Q > threshold:
        return DevianceResult(
            status="REJECTED_DEVIANCE",
            Q_statistic=Q,
            threshold=threshold,
            p_value=p_value,
            M_inliers=M_inliers,
            M_total=N_total,
            coverage=coverage,
            inlier_mask=inlier_mask,
            residual_data=r_data,
            reason=f"Q={Q:.2f} exceeds threshold {threshold:.2f}",
        )

    return DevianceResult(
        status="ACCEPTED",
        Q_statistic=Q,
        threshold=threshold,
        p_value=p_value,
        M_inliers=M_inliers,
        M_total=N_total,
        coverage=coverage,
        inlier_mask=inlier_mask,
        residual_data=r_data,
        reason="ok",
    )

def expected_deviance_separation(
    mean_offset_m: float,
    sigma_r: float,
    M: int,
) -> tuple[float, float]:
    """Analytical log-score separation for a specified offset."""
    if mean_offset_m <= 0.0:
        raise ValueError("mean_offset_m must be positive")
    if sigma_r <= 0.0:
        raise ValueError("sigma_r must be positive")
    if M < 1:
        raise ValueError("M must be at least 1")

    delta_ll = 0.5 * (mean_offset_m ** 2) / (sigma_r ** 2)
    lambda_m = 2.0 * M * delta_ll
    return float(delta_ll), float(lambda_m)
