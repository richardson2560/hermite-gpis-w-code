from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RegistrationResult:
    T_estimated: np.ndarray
    # None means the backend did not expose the diagnostic.
    converged: bool | None
    iterations: int | None
    computation_time_ms: float
    status: str
    fitness: float
    inlier_rmse: float

class BaseRegistrationSolver(ABC):
    @abstractmethod
    def align(self, scene_points, model_points, initial_T, *, model_normals=None):
        """Map scene points into the model frame."""
        raise NotImplementedError
