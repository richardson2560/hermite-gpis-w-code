"""Common unweighted field least-squares for representation ablations.

This is an experimental adapter, not a reproduction of MOP odometry.
No HGW private fields, invented variance, or per-method pose prior.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from hgw.registration.se3 import exp_se3
from .base import FieldEvaluation, RegistrationResult
from ._validation import cloud, pose, positive

class HGWFieldAdapter:
    field_kind = "signed_implicit"
    uncertainty_kind = "conservative_proxy"
    def __init__(self, model):
        self.model = model
    def field(self, queries, *, with_variance=False):
        X = cloud(queries,"queries",min_points=0)
        m,g,v = self.model.evaluate_many(X)
        distances,_ = self.model._tree.query(X,k=1)
        return FieldEvaluation(m,g,distances < self.model.support_radius,
                               v if with_variance else None)
    def size_bytes(self, *, include_variance=True):
        return self.model._artifact.size_bytes()

@dataclass(frozen=True)
class FieldRegistrationConfig:
    max_iterations: int = 50
    min_points: int = 6
    gradient_floor: float = 1e-8
    step_tol: float = 1e-7
    damping: float = 1e-8
    max_translation_step: float = .05
    max_rotation_step: float = .3
    min_valid_fraction: float = .2
    def __post_init__(self):
        for name in ("max_iterations","min_points"):
            value = getattr(self,name)
            if not isinstance(value,int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("gradient_floor","step_tol","max_translation_step","max_rotation_step"):
            positive(getattr(self,name),name)
        positive(self.damping,"damping",allow_zero=True)
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be in (0,1]")

def register_field(scene_points, model, initial_T, *, config=FieldRegistrationConfig()):
    start = perf_counter()
    scene = cloud(scene_points,min_points=config.min_points)
    T = pose(initial_T)
    status,converged,reason = "NOT_CONVERGED",False,"iteration limit"
    def evaluate(T):
        x = scene @ T[:3,:3].T + T[:3,3]
        f = model.field(x,with_variance=False)
        norms = np.linalg.norm(f.gradients,axis=1)
        mask = f.valid_mask & np.isfinite(f.values) & np.isfinite(f.gradients).all(axis=1)
        mask &= norms >= config.gradient_floor
        return x,f,mask
    for iteration in range(1,config.max_iterations+1):
        x,f,mask = evaluate(T)
        if mask.sum() < config.min_points or mask.mean() < config.min_valid_fraction:
            status,reason = "REJECTED","insufficient informative field queries"
            break
        J = np.hstack([f.gradients[mask],np.cross(x[mask],f.gradients[mask])])
        if np.linalg.matrix_rank(J) < 6:
            status,reason = "DEGENERATE","pose is not observable from field gradients"
            break
        H = J.T@J + config.damping*np.eye(6)
        delta = np.linalg.solve(H,-J.T@f.values[mask])
        for sl,limit in ((slice(0,3),config.max_translation_step),(slice(3,6),config.max_rotation_step)):
            norm = np.linalg.norm(delta[sl])
            if norm > limit:
                delta[sl] *= limit/norm
        if np.linalg.norm(delta) < config.step_tol:
            status,converged,reason = "CONVERGED",True,"small Gauss-Newton step"
            break
        current = np.sum(f.values[mask]**2)
        accepted = False
        for alpha in (1.,.5,.25,.125,.0625,.03125):
            trial = exp_se3(alpha*delta)@T
            _,ff,mm = evaluate(trial)
            # A trial cannot reduce the objective by dropping hard points.
            if not np.all(mm[mask]):
                continue
            if np.sum(ff.values[mask]**2) < current:
                T,accepted = trial,True
                break
        if not accepted:
            status,reason = "STALLED","no descent step; convergence not established"
            break
    _,f,mask = evaluate(T)
    rms = float(np.sqrt(np.mean(f.values[mask]**2))) if mask.any() else float("inf")
    return RegistrationResult(T,converged,iteration,1000*(perf_counter()-start),
                              status,float(mask.mean()),rms,mask,reason)
