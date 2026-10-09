import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
from hgw.baselines.icp import PointToPlaneICP

@pytest.mark.parametrize("kwargs", [
    dict(max_correspondence_distance=0), dict(normal_radius=float("nan")),
    dict(max_iterations=0)])
def test_icp_config_validation(kwargs):
    with pytest.raises(ValueError):
        PointToPlaneICP(**kwargs)

@pytest.mark.parametrize("matches", [[], [1,2,3]])
def test_icp_does_not_invent_convergence(monkeypatch, matches):
    calls = {}
    def register(source, target, distance, T, estimator, criteria):
        calls["normals"] = target.normals
        calls["T"] = T
        return NS(transformation=T, correspondence_set=matches,
                  fitness=.5 if matches else 0., inlier_rmse=.01 if matches else 0.)
    fake = NS(geometry=NS(PointCloud=lambda: NS()),
              utility=NS(Vector3dVector=np.asarray),
              pipelines=NS(registration=NS(
                  registration_icp=register,
                  TransformationEstimationPointToPlane=lambda: None,
                  ICPConvergenceCriteria=lambda **kwargs: kwargs)))
    monkeypatch.setitem(sys.modules, "open3d", fake)
    points = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]])
    normals = np.tile([0.,0.,2.], (3,1))
    result = PointToPlaneICP().align(points, points, np.eye(4), model_normals=normals)
    assert result.iterations is None
    assert result.converged is (None if matches else False)
    assert result.status == ("COMPLETED" if matches else "NO_CORRESPONDENCES")
    np.testing.assert_allclose(calls["normals"], normals/2)
