"""Shared input contracts; metric coordinates and scene-to-model poses."""
import numpy as np

def cloud(x, name="points", min_points=1):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < min_points or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a finite (N, 3) array with N >= {min_points}")
    return x.copy()

def normals(x, points):
    x = cloud(x, "normals")
    lengths = np.linalg.norm(x, axis=1)
    if x.shape != points.shape or np.any(lengths < 1e-12):
        raise ValueError("normals must be nonzero and match points")
    return x / lengths[:, None]

def positive(x, name, allow_zero=False):
    if not np.isfinite(x) or (x < 0 if allow_zero else x <= 0):
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
    return float(x)

def pose(T):
    T = np.asarray(T, dtype=np.float64)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
        or not np.allclose(T[3], [0,0,0,1], atol=1e-8, rtol=0)
        or not np.allclose(T[:3,:3].T @ T[:3,:3], np.eye(3), atol=1e-7, rtol=0)
        or not np.isclose(np.linalg.det(T[:3,:3]), 1., atol=1e-7, rtol=0)):
        raise ValueError("pose must be an SE(3) matrix")
    return T.copy()
