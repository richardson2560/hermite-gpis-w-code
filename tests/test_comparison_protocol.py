import numpy as np
import pytest
from hgw.baselines.comparison import geometry_metrics,pose_errors,shared_seeds,run_multistart
from hgw.baselines.base import RegistrationResult
from hgw.baselines.icp import planar_covariances,PointToPointICP,PointToPlaneICP,GeneralizedICP,HuberPointToPlaneICP
from hgw.registration.se3 import exp_se3

def test_common_metrics_count_unmatched_points():
    model=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    scene=np.vstack([model,[0.,0.,10.]])
    metrics=geometry_metrics(scene,model,np.eye(4),.1)
    assert metrics["scene_overlap"]==.75
    assert metrics["scene_rmse_m"]==pytest.approx(5.)
    assert metrics["scene_clipped_rmse_m"]==pytest.approx(.05)

def test_multistart_uses_common_geometry_and_total_budget():
    points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    wrong=np.eye(4); wrong[0,3]=2.
    seeds=[wrong,np.eye(4)]
    def align(T):
        return RegistrationResult(T,None,None,3.,"COMPLETED",1.,0.)
    result=run_multistart(align,points,points,seeds,.1)
    assert result.best_index==1
    assert result.total_registration_ms==6.
    with pytest.raises(ValueError): shared_seeds(points,points,supplied=[])

def test_pose_metrics():
    T=exp_se3(np.array([0.,0.,0.,0.,0.,.2]))
    metrics=pose_errors(T,np.eye(4))
    assert metrics["translation_error_m"]==0.
    assert metrics["rotation_error_deg"]==pytest.approx(np.degrees(.2))

def test_gicp_planar_covariance_eigenvalues():
    rng=np.random.default_rng(1)
    p=np.column_stack([rng.normal(size=(50,2)),np.zeros(50)])
    C=planar_covariances(p,epsilon=.001)
    np.testing.assert_allclose(np.linalg.eigvalsh(C),np.tile([.001,1.,1.],(50,1)),atol=1e-12)
    np.testing.assert_allclose(C[:,2,2],.001,atol=1e-12)

@pytest.mark.parametrize("cls", [PointToPointICP,PointToPlaneICP,GeneralizedICP,HuberPointToPlaneICP])
def test_real_open3d_recovers_small_rigid_transform(cls):
    pytest.importorskip("open3d")
    rng=np.random.default_rng(100)
    direction=rng.normal(size=(200,3))
    direction/=np.linalg.norm(direction,axis=1)[:,None]
    axes=np.array([.2,.12,.07])
    model=direction*axes
    normals=direction/axes
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    truth=exp_se3(np.array([.003,-.002,.001,.01,-.02,.015]))
    scene=(model-truth[:3,3])@truth[:3,:3]
    result=cls(max_correspondence_distance=.04,max_iterations=80).align(
        scene,model,np.eye(4),model_normals=normals)
    metrics=pose_errors(result.T_estimated,truth)
    assert metrics["translation_error_m"]<1e-4
    assert metrics["rotation_error_deg"]<.1
    assert result.iterations is None and result.converged is None
    assert result.inlier_mask.all()


def test_benchmark_runs_every_method_with_common_data_and_seeds():
    pytest.importorskip("open3d")
    from experiments.compare_baselines import run_benchmark, BenchmarkConfig, _json_safe
    import json
    rng=np.random.default_rng(51)
    d=rng.normal(size=(30,3)); d/=np.linalg.norm(d,axis=1)[:,None]
    axes=np.array([.2,.13,.07])
    points=d*axes
    normals=d/axes; normals/=np.linalg.norm(normals,axis=1)[:,None]
    config=BenchmarkConfig(m=12,h=.2,lengthscale=.1,log_lambda=20.,
        max_iterations=2,repeats=2,translation_tolerance=.01,rotation_tolerance_deg=5.)
    result=run_benchmark(points[:20],normals[:20],points[20:],points[20:],
        config=config,seeds=[np.eye(4)],truth=np.eye(4))
    assert len(result["rows"])==20
    names={row["method"] for row in result["rows"]}
    assert names=={"hgw_common","hgw_native","gpis_rbf_full","rbf_directional_ablation",
                   "log_gpis_value_core","hrbf_cubic_full","icp_point","icp_plane","icp_plane_huber","gicp"}
    assert len(set(result["training_indices"]))==12
    for row in result["rows"]:
        assert len(row["trials"])==1 and row["selected_seed"]==0
        assert isinstance(row["pose_success"],bool)
        assert row["registration_ms_all_seeds"]>=0
    json.dumps(_json_safe(result),allow_nan=False)

def test_benchmark_requires_complete_success_definition():
    from experiments.compare_baselines import BenchmarkConfig
    with pytest.raises(ValueError,match="both"):
        BenchmarkConfig(translation_tolerance=.01)
