"""Common-data geometry metrics and seed selection, independent of native losses."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from ._validation import cloud, pose, positive

def geometry_metrics(scene_points, model_points, T, threshold):
    """One-way scene metrics are appropriate for partial visibility.

    Symmetric Chamfer is also reported, but is penalized by missing surface.
    The gate never changes the all-scene RMSE; failed matches are not discarded.
    """
    scene,model = cloud(scene_points),cloud(model_points)
    T = pose(T)
    threshold = positive(threshold,"threshold")
    transformed = scene@T[:3,:3].T+T[:3,3]
    a,_ = cKDTree(model).query(transformed)
    b,_ = cKDTree(transformed).query(model)
    return dict(scene_rmse_m=float(np.sqrt(np.mean(a*a))),
                scene_median_m=float(np.median(a)),
                scene_q95_m=float(np.quantile(a,.95)),
                scene_overlap=float(np.mean(a<threshold)),
                scene_clipped_rmse_m=float(np.sqrt(np.mean(np.minimum(a,threshold)**2))),
                symmetric_chamfer_squared_m2=float(.5*(np.mean(a*a)+np.mean(b*b))))

def pose_errors(T_estimated,T_true):
    estimate,truth = pose(T_estimated),pose(T_true)
    R = estimate[:3,:3]@truth[:3,:3].T
    return dict(translation_error_m=float(np.linalg.norm(estimate[:3,3]-truth[:3,3])),
                rotation_error_deg=float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))))

def shared_seeds(scene_points, model_points, *, supplied=None):
    """Identity and centroid seeds, or exactly the supplied common poses."""
    scene,model = cloud(scene_points),cloud(model_points)
    if supplied is not None:
        if len(supplied)==0:
            raise ValueError("at least one supplied seed is required")
        return [pose(T) for T in supplied]
    centered = np.eye(4)
    centered[:3,3] = model.mean(axis=0)-scene.mean(axis=0)
    return [np.eye(4),centered]

@dataclass(frozen=True)
class MultiStartResult:
    best_index: int
    results: tuple
    metrics: tuple
    total_registration_ms: float

def run_multistart(align, scene_points, reference_points, seeds, threshold):
    """Select by fixed common geometry, never by ground truth or native score."""
    results,metrics = [],[]
    for T in seeds:
        result = align(pose(T))
        results.append(result)
        metrics.append(geometry_metrics(scene_points,reference_points,result.T_estimated,threshold))
    if not results:
        raise ValueError("at least one seed is required")
    # Native ACCEPTED/COMPLETED flags are deliberately not comparative evidence.
    best = min(range(len(results)),key=lambda i: (
        -metrics[i]["scene_overlap"],metrics[i]["scene_clipped_rmse_m"]))
    return MultiStartResult(best,tuple(results),tuple(metrics),
                            sum(r.computation_time_ms for r in results))
