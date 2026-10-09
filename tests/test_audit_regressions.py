"""Independent numerical and failure-mode checks for the audit corrections."""
import numpy as np
import pytest
from hgw.core.kernels import wendland_c2_gradient, wendland_c2_hessian_vector_product
from hgw.core.sampling import pivoted_cholesky
from hgw.models.artifact import ModelArtifact
from hgw.models.evaluator import HermiteGPIS_W
from hgw.registration.p2m import _residual_jacobians
from hgw.registration.se3 import exp_se3
from hgw.registration.anchor import generate_hermite_seeds
from experiments.exp_h_m_frontier import sweep_h_and_m, pareto_mask

def tiny_model():
    return HermiteGPIS_W(ModelArtifact(
        points=np.zeros((1, 3)), normals=np.array([[0., 0., 1.]]),
        alpha=np.array([.3, .01]), h=.1, sigma_f2=1.,
        sigma_0_sq=1e-4, sigma_1_sq=1e-2))

@pytest.mark.parametrize("radius", [0., 1e-12, .023, .099, .1, .12])
def test_hessian_product_directional_difference(radius):
    d = np.array([radius, 0., 0.])
    n = np.array([.3, -.7, .4])
    step = 1e-8
    a, b = d + step*n, d - step*n
    numeric = (wendland_c2_gradient(a, np.linalg.norm(a), .1, 1.) -
               wendland_c2_gradient(b, np.linalg.norm(b), .1, 1.)) / (2*step)
    np.testing.assert_allclose(
        wendland_c2_hessian_vector_product(d, radius, n, .1, 1.),
        numeric, atol=.002, rtol=1e-5)

def test_pose_jacobian_against_left_increment():
    rng = np.random.default_rng(42)
    points, grads = rng.normal(size=(2, 8, 3))
    numeric = np.zeros((8, 6))
    for axis in range(6):
        delta = np.eye(6)[axis] * 1e-6
        plus, minus = exp_se3(delta), exp_se3(-delta)
        xp = points @ plus[:3,:3].T + plus[:3,3]
        xm = points @ minus[:3,:3].T + minus[:3,3]
        numeric[:,axis] = np.sum((xp-xm)*grads, axis=1) / 2e-6
    np.testing.assert_allclose(_residual_jacobians(points, grads), numeric, atol=1e-8)

def test_support_boundary_is_inactive():
    model = tiny_model()
    x = np.array([[.1, 0., 0.], [.1001, 0., 0.]])
    assert model.support_fraction(x) == 0
    means, grads, variances = model.evaluate_many(x)
    assert np.all(means == 0) and np.all(grads == 0)
    assert np.all(variances == 1)

def test_pivots_do_not_repeat_with_large_noise():
    points = np.arange(5.)[:,None] * np.array([[1., 0., 0.]])
    selected = pivoted_cholesky(points, .1, 1., 100., 5, eps_tol=0)
    assert len(np.unique(selected)) == 5

def test_no_seed_is_a_controlled_failure(monkeypatch):
    model = tiny_model()
    monkeypatch.setattr(model, "support_fraction", lambda x: 0.)
    points = np.array([[0.,0.,0.], [.01,0.,0.], [0.,.01,0.]])
    normals = np.tile([0.,0.,1.], (3,1))
    assert generate_hermite_seeds(points, normals, model) == []

def test_frontier_excludes_unsupported_zero_field():
    points = np.array([[0.,0.,0.], [.02,0.,0.], [0.,.02,0.]])
    normals = np.tile([0.,0.,1.], (3,1))
    rows = sweep_h_and_m(points, normals, np.array([[10.,0.,0.]]), [.1], [2])
    assert rows[0]["coverage"] == 0 and rows[0]["valid_fraction"] == 0
    assert np.isinf(rows[0]["rms_mm"])

def test_pareto_dominance():
    rows = [dict(valid_fraction=1., rms_mm=1., bytes=10),
            dict(valid_fraction=.5, rms_mm=2., bytes=20),
            dict(valid_fraction=1., rms_mm=.5, bytes=30)]
    assert pareto_mask(rows) == [True, False, True]


def test_p2m_uses_deviance_inliers():
    from hgw.registration.p2m import p2m_register, P2MConfig
    from hgw.stats.deviance import deviance_test, DevianceConfig
    model = tiny_model()
    points = np.array([[0., 0., .001], [.001, 0., .002], [0., .001, .003]])
    cfg = DevianceConfig(alpha_local=.1)
    result = p2m_register(points, model, np.eye(4),
        config=P2MConfig(max_iterations=1, min_points=1), deviance_config=cfg)
    dev = deviance_test(points, result.T_estimated, model, config=cfg)
    assert result.n_inliers == dev.M_inliers

@pytest.mark.parametrize("threshold", [float("nan"), float("inf")])
def test_glrt_rejects_nonfinite_threshold(threshold):
    from hgw.stats.glrt import GLRTConfig
    with pytest.raises(ValueError, match="finite"):
        GLRTConfig(delta_score_threshold=threshold)
