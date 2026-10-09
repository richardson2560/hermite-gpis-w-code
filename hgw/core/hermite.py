#hgw/core/hermite.py

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from hgw.core.kernels import (
    wendland_c2_value,
    wendland_c2_gradient,
    wendland_c2_hessian,
)

_KIND_VALUE = "value"
_KIND_DERIVATIVE = "derivative"
_VALID_KINDS = frozenset((_KIND_VALUE, _KIND_DERIVATIVE))

@dataclass(frozen=True)
class HermiteObservation:
    """One value or directional derivative observation in model coordinates."""

    point_m: tuple[float, float, float]
    kind: str
    direction: tuple[float, float, float]
    value: float
    noise_variance: float

    def __post_init__(self) -> None:
        point = np.asarray(self.point_m, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f"point_m must be a finite 3D point: {self.point_m}")
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {_VALID_KINDS}: {self.kind}")
        direction = np.asarray(self.direction, dtype=np.float64)
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError(f"direction must be a finite 3D vector: {self.direction}")
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            raise ValueError(f"direction must be a non-zero vector: {self.direction}")
        direction /= norm
        if not np.isfinite(self.value):
            raise ValueError(f"value must be finite: {self.value}")
        if not np.isfinite(self.noise_variance) or self.noise_variance <= 0.0:
            raise ValueError(f"noise_variance must be positive and finite: {self.noise_variance}")

        object.__setattr__(self, "point_m", tuple(float(x) for x in point))
        object.__setattr__(self, "direction", tuple(float(x) for x in direction))

def _block_from_types(
        kind_i: str,
        direction_i: np.ndarray,
        kind_j: str,
        direction_j: np.ndarray,
        value: float,
        gradient: np.ndarray,
        hessian: np.ndarray,
) -> float:
    if kind_i == _KIND_VALUE and kind_j == _KIND_VALUE:
        return value
    elif kind_i == _KIND_VALUE and kind_j == _KIND_DERIVATIVE:
        return - np.dot(gradient, direction_j)
    elif kind_i == _KIND_DERIVATIVE and kind_j == _KIND_VALUE:
        return np.dot(gradient, direction_i)
    elif kind_i == _KIND_DERIVATIVE and kind_j == _KIND_DERIVATIVE:
        return - np.dot(direction_i, hessian @ direction_j)
    else:
        raise ValueError(f"Invalid kinds: {kind_i}, {kind_j}")

def hermite_block(
        obs_i: HermiteObservation,
        obs_j: HermiteObservation,
        h: float,
        sigma_f2: float,
) -> float:
    """Compute the Hermite block for two observations."""
    point_i = np.asarray(obs_i.point_m, dtype=np.float64)
    point_j = np.asarray(obs_j.point_m, dtype=np.float64)
    d = point_i - point_j
    r = np.linalg.norm(d)

    value = float(wendland_c2_value(np.array([r]), h, sigma_f2)[0])
    gradient = wendland_c2_gradient(d, r, h, sigma_f2)
    hessian = wendland_c2_hessian(d, r, h, sigma_f2)

    return _block_from_types(
        obs_i.kind,
        np.asarray(obs_i.direction, dtype=np.float64),
        obs_j.kind,
        np.asarray(obs_j.direction, dtype=np.float64),
        value,
        gradient,
        hessian,
    )

def assemble_system_matrix(
        observations: list[HermiteObservation],
        h: float,
        sigma_f2: float,
) -> np.ndarray:
    """Assemble the system matrix for a list of Hermite observations."""
    if not observations:
        raise ValueError("observations list cannot be empty")
    
    n = len(observations)
    K = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            K[i, j] = hermite_block(observations[i], observations[j], h, sigma_f2)
            if i != j:
                K[j, i] = K[i, j]

    return K