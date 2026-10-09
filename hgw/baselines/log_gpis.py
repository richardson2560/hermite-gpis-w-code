"""Log-GPIS dense value-observation core, Wu et al. RA-L 2021, (12)-(14).

lambda_param is an inverse length [1/m]. ell=sqrt(3)/lambda_param,
d=-log(|mu|)/lambda_param. This is not the complete Log-GPIS-MOP system.
"""
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from .base import BaseImplicitModel, FieldEvaluation
from ._validation import cloud, positive
from ._hermite import radial

class LogGPIS_Model(BaseImplicitModel):
    field_kind = "unsigned_distance"
    uncertainty_kind = "delta_method"
    def __init__(self, lambda_param=40., sigma_f2=1., sigma_noise_sq=1e-4,
                 eps_floor=1e-12, batch_size=256):
        self.lam = positive(lambda_param, "lambda_param")
        self.ell = np.sqrt(3.)/self.lam
        self.sigma_f2 = positive(sigma_f2, "sigma_f2")
        self.sigma_noise_sq = positive(sigma_noise_sq, "sigma_noise_sq")
        self.eps_floor = positive(eps_floor, "eps_floor")
        if self.eps_floor >= 1:
            raise ValueError("eps_floor must be less than 1")
        if not isinstance(batch_size,int) or batch_size < 1:
            raise ValueError("batch_size must be positive integer")
        self.batch_size, self._points = batch_size, None

    def fit(self, points, normals=None):
        self._points = cloud(points)
        K,_,_ = radial(self._points[:,None]-self._points[None,:],
                       "matern32",self.ell,self.sigma_f2)
        K.flat[::len(K)+1] += self.sigma_noise_sq
        self._L = cholesky(K,lower=True)
        self._alpha = cho_solve((self._L,True), np.ones(len(K)))
        self.n_constraints = len(K)
        return self

    def field(self, queries, *, with_variance=False):
        if self._points is None:
            raise RuntimeError("fit must be called before evaluation")
        X = cloud(queries,"queries",min_points=0)
        values, grads, valid = np.empty(len(X)), np.zeros((len(X),3)), np.empty(len(X),dtype=bool)
        var = np.full(len(X),np.nan) if with_variance else None
        for start in range(0,len(X),self.batch_size):
            sl = slice(start,start+self.batch_size)
            K,G,_ = radial(X[sl,None]-self._points[None,:], "matern32",self.ell,self.sigma_f2)
            mu = K @ self._alpha
            gmu = np.einsum("qnb,n->qb",G,self._alpha)
            ok = np.abs(mu) > self.eps_floor
            safe = np.maximum(np.abs(mu),self.eps_floor)
            values[sl] = -np.log(safe)/self.lam
            # Derivative of log(abs(mu)) is grad(mu)/mu, including negative mu.
            grad = np.zeros_like(gmu)
            grad[ok] = -gmu[ok]/(self.lam*mu[ok,None])
            grads[sl],valid[sl] = grad,ok
            if with_variance:
                V = solve_triangular(self._L,K.T,lower=True)
                vu = np.maximum(0.,self.sigma_f2-np.sum(V*V,axis=0))
                transformed = np.full(len(mu),np.nan)
                transformed[ok] = vu[ok]/(self.lam*mu[ok])**2
                var[sl] = transformed
        return FieldEvaluation(values,grads,valid,var)

    def eikonal_directions(self, queries):
        """Paper's normalized search direction; not an exact scalar derivative."""
        f = self.field(queries)
        norm = np.linalg.norm(f.gradients,axis=1)
        return np.divide(f.gradients,norm[:,None],out=np.zeros_like(f.gradients),
                         where=norm[:,None]>0)

    def size_bytes(self, *, include_variance=True):
        if self._points is None:
            raise RuntimeError("model is not fitted")
        arrays = [self._points,self._alpha]
        if include_variance:
            arrays.append(self._L)
        return sum(a.nbytes for a in arrays)
