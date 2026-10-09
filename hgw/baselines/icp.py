"""Lazy Open3D legacy point-to-plane adapter.

The legacy result exposes fitness and RMSE, not actual iterations or a
convergence flag. Successful backend execution is reported as COMPLETED.
"""
from time import perf_counter
import numpy as np
from .base import BaseRegistrationSolver, RegistrationResult

def _cloud(points, name):
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 3
            or not np.isfinite(points).all()):
        raise ValueError(f"{name} must be a finite (N, 3) cloud with N >= 3")
    return points

class PointToPlaneICP(BaseRegistrationSolver):
    def __init__(self, max_correspondence_distance=.05, max_iterations=50,
                 normal_radius=.03):
        for name, value in (("max_correspondence_distance", max_correspondence_distance),
                            ("normal_radius", normal_radius)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.max_dist = max_correspondence_distance
        self.max_iter = max_iterations
        self.normal_radius = normal_radius

    def align(self, scene_points, model_points, initial_T, *, model_normals=None):
        scene_points = _cloud(scene_points, "scene_points")
        model_points = _cloud(model_points, "model_points")
        T = np.asarray(initial_T, dtype=float)
        if (T.shape != (4,4) or not np.isfinite(T).all()
                or not np.allclose(T[3], [0,0,0,1])
                or not np.allclose(T[:3,:3].T @ T[:3,:3], np.eye(3))
                or not np.isclose(np.linalg.det(T[:3,:3]), 1)):
            raise ValueError("initial_T must be an SE(3) matrix")
        if model_normals is not None:
            model_normals = np.asarray(model_normals, dtype=float)
            if (model_normals.shape != model_points.shape
                    or not np.isfinite(model_normals).all()
                    or np.any(np.linalg.norm(model_normals, axis=1) == 0)):
                raise ValueError("model_normals must be finite nonzero vectors matching model_points")
            model_normals = model_normals / np.linalg.norm(model_normals, axis=1)[:,None]
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError("Install the icp extra: pip install 'hgw[icp]'") from exc
        start = perf_counter()
        source, target = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(scene_points)
        target.points = o3d.utility.Vector3dVector(model_points)
        if model_normals is None:
            target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius, max_nn=20))
        else:
            target.normals = o3d.utility.Vector3dVector(model_normals)
        result = o3d.pipelines.registration.registration_icp(
            source, target, self.max_dist, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iter))
        has_matches = len(result.correspondence_set) > 0
        return RegistrationResult(
            T_estimated=np.asarray(result.transformation).copy(),
            converged=None if has_matches else False, iterations=None,
            computation_time_ms=1000 * (perf_counter()-start),
            status="COMPLETED" if has_matches else "NO_CORRESPONDENCES",
            fitness=float(result.fitness), inlier_rmse=float(result.inlier_rmse))
