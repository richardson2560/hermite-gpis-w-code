#hgw/stats/glrt.py

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.stats import chi2
from hgw.models.evaluator import HermiteGPIS_W
from hgw.stats.deviance import (
    DevianceConfig,
    DevianceResult,
    deviance_test,
)

@dataclass(frozen=True)
class ModelCandidate:
    """A named model candidate with a pose hypothesis."""

    name: str
    model: HermiteGPIS_W
    T: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.model, HermiteGPIS_W):
            raise TypeError("model must be a HermiteGPIS_W instance")
        T = np.asarray(self.T, dtype=np.float64)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise ValueError("T must be a finite (4, 4) matrix")
        object.__setattr__(self, "T", T)


@dataclass(frozen=True)
class GLRTConfig:
    """Configuration for the GLRT model selection."""

    delta_score_threshold: float = float(chi2.ppf(0.95, df=1))  # ≈ 3.841
    deviance_config: DevianceConfig = DevianceConfig()

    def __post_init__(self) -> None:
        if self.delta_score_threshold <= 0.0:
            raise ValueError("delta_score_threshold must be positive")

@dataclass(frozen=True)
class PerModelResult:
    """Per-candidate result: deviance outcome plus computed score."""

    name: str
    deviance: DevianceResult
    score: float
    passed: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not math.isfinite(self.score) and self.score != math.inf:
            raise ValueError("score must be finite or +inf")


@dataclass(frozen=True)
class GLRTResult:
    """Result of the multi-model selection."""

    status: str                      # "SELECTED", "AMBIGUOUS", "REJECTED_ALL"
    selected_name: str | None
    delta_score: float
    threshold: float
    per_model: tuple[PerModelResult, ...]
    reason: str

    def __post_init__(self) -> None:
        valid = {"SELECTED", "AMBIGUOUS", "REJECTED_ALL"}
        if self.status not in valid:
            raise ValueError(f"invalid status: {self.status}")
        if self.status == "SELECTED" and self.selected_name is None:
            raise ValueError("SELECTED requires a selected_name")
        if self.status != "SELECTED" and self.selected_name is not None:
            raise ValueError("selected_name must be None unless status is SELECTED")
        if not math.isfinite(self.delta_score) and self.delta_score != math.inf:
            raise ValueError("delta_score must be finite or +inf")
        if not math.isfinite(self.threshold) or self.threshold <= 0.0:
            raise ValueError("threshold must be positive and finite")
        if not self.reason:
            raise ValueError("reason must not be empty")
        object.__setattr__(self, "per_model", tuple(self.per_model))

def _compute_score(dev: DevianceResult) -> float:
    """Compute Score(M) = Q + Σ log(σ_r²) over inliers.

    Returns +inf when there are no inliers.
    """
    if dev.M_inliers == 0:
        return math.inf

    r_data = dev.residual_data
    inlier_mask = dev.inlier_mask
    sigmas_sq = r_data.residual_variances[inlier_mask]

    if np.any(sigmas_sq <= 0.0):
        return math.inf

    log_terms = float(np.sum(np.log(sigmas_sq)))
    return float(dev.Q_statistic) + log_terms

def glrt_select(
    scene_points: np.ndarray,
    candidates: list[ModelCandidate],
    *,
    expected_point_count: int | None = None,
    config: GLRTConfig = GLRTConfig(),
) -> GLRTResult:
    """Select the best model via GLRT."""
    # --- Validate ----------------------------------------------------------
    scene_points = np.asarray(scene_points, dtype=np.float64)
    if scene_points.ndim != 2 or scene_points.shape[1] != 3:
        raise ValueError(
            f"scene_points must have shape (N, 3), got {scene_points.shape}"
        )
    if not np.all(np.isfinite(scene_points)):
        raise ValueError("scene_points must be finite")

    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("at least 2 candidates are required")

    names = [c.name for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")

    if not isinstance(config, GLRTConfig):
        raise TypeError("config must be a GLRTConfig instance")

    # --- Run deviance test for each candidate -----------------------------
    per_model: list[PerModelResult] = []
    for cand in candidates:
        dev = deviance_test(
            scene_points,
            cand.T,
            cand.model,
            expected_point_count=expected_point_count,
            config=config.deviance_config,
        )
        score = _compute_score(dev) if dev.status == "ACCEPTED" else math.inf
        passed = dev.status == "ACCEPTED"
        per_model.append(
            PerModelResult(
                name=cand.name,
                deviance=dev,
                score=score,
                passed=passed,
            )
        )

    # --- Filter to passing candidates -------------------------------------
    passing = [pm for pm in per_model if pm.passed]

    if not passing:
        return GLRTResult(
            status="REJECTED_ALL",
            selected_name=None,
            delta_score=math.inf,
            threshold=config.delta_score_threshold,
            per_model=tuple(per_model),
            reason="all candidates failed the deviance or coverage gates",
        )

    # Sort by ascending score (lower = better)
    passing_sorted = sorted(passing, key=lambda pm: pm.score)

    # --- Exactly one passes: direct selection -----------------------------
    if len(passing_sorted) == 1:
        best = passing_sorted[0]
        return GLRTResult(
            status="SELECTED",
            selected_name=best.name,
            delta_score=math.inf,
            threshold=config.delta_score_threshold,
            per_model=tuple(per_model),
            reason=f"only '{best.name}' passed the deviance test",
        )

    # --- Two or more pass: compare scores ---------------------------------
    best = passing_sorted[0]
    second = passing_sorted[1]
    delta_score = float(second.score - best.score)

    if delta_score < config.delta_score_threshold:
        return GLRTResult(
            status="AMBIGUOUS",
            selected_name=None,
            delta_score=delta_score,
            threshold=config.delta_score_threshold,
            per_model=tuple(per_model),
            reason=(
                f"score gap {delta_score:.3f} below threshold "
                f"{config.delta_score_threshold:.3f}; "
                f"'{best.name}' and '{second.name}' are indistinguishable"
            ),
        )

    return GLRTResult(
        status="SELECTED",
        selected_name=best.name,
        delta_score=delta_score,
        threshold=config.delta_score_threshold,
        per_model=tuple(per_model),
        reason=(
            f"'{best.name}' beats '{second.name}' by "
            f"{delta_score:.3f} > {config.delta_score_threshold:.3f}"
        ),
    )