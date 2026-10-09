"""Open3D L2 ICP adapters, with explicit algorithm settings.

Point-to-plane uses nearest-neighbour correspondences (modern ICP variant of
Chen/Medioni). GICP uses Segal et al.'s planar covariance eigenvalues
(epsilon,1,1) on BOTH clouds, estimated from k neighbours by default.
Open3D does not expose actual iterations or a convergence flag.
"""
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from .base import BaseRegistrationSolver, RegistrationResult
from ._validation import cloud, normals as unit_normals, pose, positive

def planar_covariances(points, *, neighbors=20, epsilon=1e-3, normals=None):
    points = cloud(points,min_points=3)
    epsilon = positive(epsilon,"epsilon")
    if epsilon > 1:
        raise ValueError("epsilon must be <= 1")
    if not isinstance(neighbors,int) or neighbors < 3:
        raise ValueError("neighbors must be an integer >= 3")
    if normals is None:
        _,idx = cKDTree(points).query(points,k=min(neighbors,len(points)))
        centered = points[idx]-points[idx].mean(axis=1,keepdims=True)
        C = np.einsum("nki,nkj->nij",centered,centered)/idx.shape[1]
        _,U = np.linalg.eigh(C)
        n = U[:,:,0]
    else:
        n = unit_normals(normals,points)
    return np.eye(3)[None,:,:]+(epsilon-1.)*n[:,:,None]*n[:,None,:]

class PointToPlaneICP(BaseRegistrationSolver):
    method = "plane"
    huber_scale = None
    def __init__(self,max_correspondence_distance=.05,max_iterations=50,
                 normal_radius=.03,relative_fitness=1e-6,relative_rmse=1e-6):
        self.max_dist = positive(max_correspondence_distance,"max_correspondence_distance")
        self.normal_radius = positive(normal_radius,"normal_radius")
        self.rel_fitness = positive(relative_fitness,"relative_fitness",allow_zero=True)
        self.rel_rmse = positive(relative_rmse,"relative_rmse",allow_zero=True)
        if not isinstance(max_iterations,int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.max_iter = max_iterations

    def align(self,scene_points,model_points,initial_T,*,model_normals=None):
        scene = cloud(scene_points,"scene_points",min_points=3)
        model = cloud(model_points,"model_points",min_points=3)
        T = pose(initial_T)
        n = unit_normals(model_normals,model) if model_normals is not None else None
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError("Install the icp extra: pip install 'hgw[icp]'") from exc
        start = perf_counter()
        source,target = o3d.geometry.PointCloud(),o3d.geometry.PointCloud()
        source.points,target.points = o3d.utility.Vector3dVector(scene),o3d.utility.Vector3dVector(model)
        reg = o3d.pipelines.registration
        criteria = reg.ICPConvergenceCriteria(max_iteration=self.max_iter,
            relative_fitness=self.rel_fitness,relative_rmse=self.rel_rmse)
        if self.method == "plane":
            if n is None:
                target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.normal_radius,max_nn=20))
            else:
                target.normals = o3d.utility.Vector3dVector(n)
            estimator = (reg.TransformationEstimationPointToPlane()
                         if self.huber_scale is None else
                         reg.TransformationEstimationPointToPlane(reg.HuberLoss(self.huber_scale)))
            run = reg.registration_icp
        elif self.method == "point":
            estimator,run = reg.TransformationEstimationPointToPoint(False),reg.registration_icp
        else:
            source.covariances = o3d.utility.Matrix3dVector(planar_covariances(
                scene,neighbors=self.neighbors,epsilon=self.epsilon))
            target.covariances = o3d.utility.Matrix3dVector(planar_covariances(
                model,neighbors=self.neighbors,epsilon=self.epsilon,normals=n))
            estimator = reg.TransformationEstimationForGeneralizedICP(self.epsilon)
            run = reg.registration_generalized_icp
        result = run(source,target,self.max_dist,T,estimator,criteria)
        correspondences = np.asarray(result.correspondence_set,dtype=int)
        mask = np.zeros(len(scene),dtype=bool)
        if len(correspondences):
            # Also accept simple test doubles without a 2D correspondence array.
            if correspondences.ndim == 2:
                mask[correspondences[:,0]] = True
        has_matches = len(correspondences)>0
        return RegistrationResult(
            T_estimated=np.asarray(result.transformation).copy(),
            converged=None if has_matches else False,iterations=None,
            computation_time_ms=1000*(perf_counter()-start),
            status="COMPLETED" if has_matches else "NO_CORRESPONDENCES",
            fitness=float(result.fitness),inlier_rmse=float(result.inlier_rmse),
            inlier_mask=mask,reason="backend does not expose convergence or iterations")

class PointToPointICP(PointToPlaneICP):
    """Besl/McKay-style nearest-neighbour rigid ICP; no scaling."""
    method = "point"

class GeneralizedICP(PointToPlaneICP):
    method = "generalized"
    def __init__(self,*args,epsilon=1e-3,neighbors=20,**kwargs):
        super().__init__(*args,**kwargs)
        self.epsilon = positive(epsilon,"epsilon")
        if self.epsilon > 1:
            raise ValueError("epsilon must be <= 1")
        if not isinstance(neighbors,int) or neighbors < 3:
            raise ValueError("neighbors must be integer >= 3")
        self.neighbors = neighbors


class HuberPointToPlaneICP(PointToPlaneICP):
    """Robust ICP variant; Huber threshold in meters, selected on validation."""
    def __init__(self,*args,huber_scale=.01,**kwargs):
        super().__init__(*args,**kwargs)
        self.huber_scale = positive(huber_scale,"huber_scale")
