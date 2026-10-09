"""Common capabilities without pretending all fields have calibrated uncertainty."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FieldEvaluation:
    values: np.ndarray
    gradients: np.ndarray
    valid_mask: np.ndarray
    variances: np.ndarray | None = None

class BaseImplicitModel(ABC):
    @abstractmethod
    def field(self, queries, *, with_variance=False) -> FieldEvaluation:
        raise NotImplementedError

    def evaluate_many(self, queries):
        f = self.field(queries, with_variance=True)
        return f.values, f.gradients, f.variances

@dataclass(frozen=True)
class RegistrationResult:
    T_estimated: np.ndarray
    converged: bool | None
    iterations: int | None
    computation_time_ms: float
    status: str
    fitness: float
    inlier_rmse: float
    inlier_mask: np.ndarray | None = None
    reason: str = ""

class BaseRegistrationSolver(ABC):
    @abstractmethod
    def align(self, scene_points, model_points, initial_T, *, model_normals=None):
        """Map scene points into the model frame."""
        raise NotImplementedError
