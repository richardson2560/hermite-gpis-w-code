"""Dense RBF GPIS: Dragiev et al. (ICRA 2011), Sec. II-C/D.

Default: f(p)=0, full gradient n(p), constant prior 1.
Directional, zero-prior mode is explicitly an HGW kernel ablation.
"""
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from .base import BaseImplicitModel, FieldEvaluation
from ._validation import cloud, normals as unit_normals, positive
from ._hermite import directions, system, features

class GPIS_RBF(BaseImplicitModel):
    field_kind = "signed_implicit"
    uncertainty_kind = "exact_latent_posterior"
    def __init__(self, lengthscale=.05, sigma_f2=1., sigma_0_sq=1e-4,
                 sigma_1_sq=1e-2, *, observation_mode="full", prior_mean=1.,
                 batch_size=256):
        self.ell = positive(lengthscale, "lengthscale")
        self.sigma_f2 = positive(sigma_f2, "sigma_f2")
        self.sigma_0_sq = positive(sigma_0_sq, "sigma_0_sq")
        self.sigma_1_sq = positive(sigma_1_sq, "sigma_1_sq")
        if observation_mode not in ("full", "directional"):
            raise ValueError("observation_mode must be full or directional")
        if not np.isfinite(prior_mean):
            raise ValueError("prior_mean must be finite")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive integer")
        self.mode, self.prior_mean, self.batch_size = observation_mode, float(prior_mean), batch_size
        self._points = None

    def fit(self, points, normals):
        self._points = cloud(points)
        normal_vectors = unit_normals(normals, self._points)
        self._dirs = directions(self._points, normal_vectors, self.mode)
        c = self._dirs.shape[1]+1
        y = np.zeros((len(self._points),c))
        y[:,0] = -self.prior_mean
        y[:,1:] = normal_vectors if self.mode == "full" else 1.
        A = system(self._points, self._dirs, "rbf", self.ell, self.sigma_f2)
        A.flat[::len(A)+1] += np.tile([self.sigma_0_sq]+[self.sigma_1_sq]*(c-1), len(y))
        self._L = cholesky(A, lower=True)
        self._alpha = cho_solve((self._L,True), y.ravel())
        self.n_constraints = y.size
        return self

    def field(self, queries, *, with_variance=False):
        if self._points is None:
            raise RuntimeError("fit must be called before evaluation")
        X = cloud(queries, "queries", min_points=0)
        values, grads = np.empty(len(X)), np.empty((len(X),3))
        var = np.empty(len(X)) if with_variance else None
        for start in range(0,len(X),self.batch_size):
            sl = slice(start,start+self.batch_size)
            F,G = features(X[sl], self._points, self._dirs, "rbf", self.ell, self.sigma_f2)
            values[sl] = self.prior_mean + F @ self._alpha
            grads[sl] = np.einsum("qnb,n->qb", G, self._alpha)
            if with_variance:
                V = solve_triangular(self._L, F.T, lower=True)
                var[sl] = np.maximum(0., self.sigma_f2-np.sum(V*V,axis=0))
        return FieldEvaluation(values, grads, np.ones(len(X),dtype=bool), var)

    def size_bytes(self, *, include_variance=True):
        if self._points is None:
            raise RuntimeError("model is not fitted")
        arrays = [self._points, self._dirs, self._alpha]
        if include_variance:
            arrays.append(self._L)
        return sum(a.nbytes for a in arrays)

class DirectionalRBF(GPIS_RBF):
    """Two-channel zero-mean kernel ablation, not the Dragiev reproduction."""
    def __init__(self, **kwargs):
        super().__init__(observation_mode="directional", prior_mean=0., **kwargs)
