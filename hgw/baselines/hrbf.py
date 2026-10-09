"""Full Hermite cubic RBF with affine polynomial and side constraints.

Macedo, Gois & Velho, Hermite Radial Basis Functions Implicits (2011).
f=sum(alpha*r^3 - beta.grad(r^3)) + b+a.x, f(p)=0, grad f(p)=n.
No fabricated posterior variance. Interpolation is the default.
"""
import numpy as np
from scipy.linalg import solve
from .base import BaseImplicitModel, FieldEvaluation
from ._validation import cloud, normals as unit_normals, positive
from ._hermite import directions, system, features, polynomial

class HRBF_Model(BaseImplicitModel):
    field_kind = "signed_implicit"
    uncertainty_kind = "none"
    def __init__(self, *, ridge_lambda=0., batch_size=256):
        self.ridge = positive(ridge_lambda,"ridge_lambda",allow_zero=True)
        if not isinstance(batch_size,int) or batch_size < 1:
            raise ValueError("batch_size must be positive integer")
        self.batch_size,self._points = batch_size,None

    def fit(self, points, normals):
        points = cloud(points)
        n = unit_normals(normals,points)
        if len(np.unique(points,axis=0)) != len(points):
            raise ValueError("HRBF requires distinct interpolation points")
        self._origin = points.mean(axis=0)
        self._scale = float(np.max(np.linalg.norm(points-self._origin,axis=1)))
        if self._scale <= 0:
            raise ValueError("HRBF needs a nonzero spatial extent")
        # Fit F(z)=f(origin+scale*z)/scale: unit normals remain unchanged.
        self._points = (points-self._origin)/self._scale
        self._dirs = directions(self._points,n,"full")
        K = system(self._points,self._dirs,"cubic",1.)
        K.flat[::len(K)+1] += self.ridge
        P = polynomial(self._points)
        A = np.block([[K,P],[P.T,np.zeros((4,4))]])
        y = np.column_stack([np.zeros(len(points)),n]).ravel()
        solution = solve(A,np.r_[y,np.zeros(4)],assume_a="sym")
        self._alpha,self._poly = solution[:-4],solution[-4:]
        self.n_constraints = len(y)
        return self

    def field(self, queries, *, with_variance=False):
        if self._points is None:
            raise RuntimeError("fit must be called before evaluation")
        X = (cloud(queries,"queries",min_points=0)-self._origin)/self._scale
        values, grads = np.empty(len(X)),np.empty((len(X),3))
        for start in range(0,len(X),self.batch_size):
            sl = slice(start,start+self.batch_size)
            F,G = features(X[sl],self._points,self._dirs,"cubic",1.)
            values[sl] = self._scale*(F@self._alpha+self._poly[0]+X[sl]@self._poly[1:])
            grads[sl] = np.einsum("qnb,n->qb",G,self._alpha)+self._poly[1:]
        return FieldEvaluation(values,grads,np.ones(len(X),dtype=bool),None)

    def size_bytes(self, *, include_variance=True):
        if self._points is None:
            raise RuntimeError("model is not fitted")
        return sum(a.nbytes for a in (self._points,self._dirs,self._alpha,self._poly,self._origin))+8
