#hgw/models/evaluator.py

from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from hgw.core.kernels import (
    wendland_c2_value,
    wendland_c2_gradient,
    wendland_c2_hessian,
)
from hgw.models.artifact import ModelArtifact

class HermiteGPIS_W:
    """Evaluator for a fitted Hermite-GPIS-W prior.

    The evaluator wraps a ModelArtifact and answers three questions about
    any query point x in R^3:

        1. What is the field value m(x)?
        2. What is the spatial gradient ∇m(x)?
        3. What is the conservative variance proxy ṽ_m(x)?

    All queries are exact within the compact support and recover the prior
    exactly outside it.

    Attributes:
        n_primitives: number of stored primitives M.
        support_radius: the kernel support radius h in meters.
    """
    def __init__(self, artifact: ModelArtifact | str | Path) -> None:
        if isinstance(artifact, (str, Path)):
            artifact = ModelArtifact.load(Path(artifact))
        if not isinstance(artifact, ModelArtifact):
            raise TypeError(
                f"artifact must be a ModelArtifact or path, "
                f"got {type(artifact).__name__}"
            )

        self._artifact = artifact
        self._points = artifact.points.astype(np.float64)
        self._normals = artifact.normals.astype(np.float64)
        self._alpha = artifact.alpha.astype(np.float64)
        self._h = float(artifact.h)
        self._sigma_f2 = float(artifact.sigma_f2)
        self._sigma_0_sq = float(artifact.sigma_0_sq)
        self._sigma_1_sq = float(artifact.sigma_1_sq)

        # variance proxy constants
        self._D_0 = self._sigma_f2 + self._sigma_0_sq
        self._D_1 = 20.0 * self._sigma_f2 / (self._h ** 2) + self._sigma_1_sq

        self._tree = cKDTree(self._points)

    @property
    def n_primitives(self) -> int:
        """number of store primitives"""
        return self._points.shape[0]

    @property
    def support_radius(self) -> float:
        """Kernel support radius h in meters."""
        return self._h

    def support_fraction(self, X: np.ndarray) -> float:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"X must have shape (Q, 3), got {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must be finite")
        if len(X) == 0:
            return 0.0

        neighbors = self._tree.query_ball_point(X, r=self._h)
        return float(np.mean([len(nb) > 0 for nb in neighbors]))

    def evaluate(self, x: np.ndarray) -> tuple[float, np.ndarray, float]:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.shape != (3,):
            raise ValueError(f"x must have shape (3,), got {x_arr.shape}")
        if not np.all(np.isfinite(x_arr)):
            raise ValueError("x must be finite")

        m, grad, var = self.evaluate_many(x_arr.reshape(1, 3))
        return float(m[0]), grad[0], float(var[0])

    def evaluate_many(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate m, ∇m, and ṽ_m for a batch of query points."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"X must have shape (Q, 3), got {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must be finite")

        Q = X.shape[0]
        means = np.zeros(Q, dtype=np.float64)
        gradients = np.zeros((Q, 3), dtype=np.float64)
        variances = np.full(Q, self._sigma_f2, dtype=np.float64)

        if Q == 0:
            return means, gradients, variances

        neighbor_lists = self._tree.query_ball_point(X, r=self._h)

        for q in range(Q):
            neighbors = neighbor_lists[q]
            if not neighbors:
                continue
            x_q = X[q]
            m_val = 0.0
            g_val = np.zeros(3, dtype=np.float64)
            best_r = np.inf
            best_i = -1

            for i in neighbors:
                p_i = self._points[i]
                n_i = self._normals[i]
                d = x_q - p_i
                r = float(np.linalg.norm(d))

                k_val = float(wendland_c2_value(np.array([r]), self._h, self._sigma_f2)[0])
                k_grad = wendland_c2_gradient(d, r, self._h, self._sigma_f2)
                k_hess = wendland_c2_hessian(d, r, self._h, self._sigma_f2)

                alpha_v = self._alpha[2 * i]
                alpha_d = self._alpha[2 * i + 1]

                m_val += alpha_v * k_val + alpha_d * (-float(k_grad @ n_i))
                g_val += alpha_v * k_grad + alpha_d * (-k_hess @ n_i)
                
                if r < best_r:
                    best_r = r
                    best_i = i

            means[q] = m_val
            gradients[q] = g_val

            if best_i >= 0:
                p_star = self._points[best_i]
                n_star = self._normals[best_i]
                d_star = x_q - p_star
                r_star = float(np.linalg.norm(d_star))

                k_0 = float(wendland_c2_value(np.array([r_star]), self._h, self._sigma_f2)[0])
                k_grad_star = wendland_c2_gradient(d_star, r_star, self._h, self._sigma_f2)
                k_1 = -float(k_grad_star @ n_star)

                v = self._sigma_f2 - (k_0 ** 2) / self._D_0 - (k_1 ** 2) / self._D_1
                # Numerical floor: variance cannot be negative
                variances[q] = max(0.0, v)

        return means, gradients, variances