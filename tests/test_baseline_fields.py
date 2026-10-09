import numpy as np
import pytest
from hgw.baselines._hermite import radial, system, features, directions, polynomial
from hgw.baselines.gpis_rbf import GPIS_RBF, DirectionalRBF
from hgw.baselines.log_gpis import LogGPIS_Model
from hgw.baselines.hrbf import HRBF_Model
from hgw.baselines.registration import register_field, FieldRegistrationConfig
from hgw.baselines.base import FieldEvaluation

def sample(n=9):
    rng=np.random.default_rng(23)
    p=rng.normal(size=(n,3))
    p/=np.linalg.norm(p,axis=1)[:,None]
    return p*.2,p.copy()

@pytest.mark.parametrize("family,scale", [("rbf",.3),("matern32",.3),("cubic",1.)])
def test_kernel_derivatives(family,scale):
    d=np.array([.12,-.08,.19])
    k,g,H=radial(d,family,scale)
    step=1e-6
    gn=np.array([(radial(d+step*e,family,scale)[0]-radial(d-step*e,family,scale)[0])/(2*step)
                 for e in np.eye(3)])
    Hn=np.column_stack([(radial(d+step*e,family,scale)[1]-radial(d-step*e,family,scale)[1])/(2*step)
                 for e in np.eye(3)])
    np.testing.assert_allclose(g,gn,atol=1e-8)
    np.testing.assert_allclose(H,Hn,atol=1e-7)
    assert np.isfinite(radial(np.zeros(3),family,scale)[2]).all()

def test_rbf_operator_signs_against_mixed_differences():
    p,n=sample(3)
    D=directions(p,n,"full")
    K=system(p,D,"rbf",.3)
    np.testing.assert_allclose(K,K.T,atol=1e-14)
    eps=1e-5
    x,y=p[0],p[1]
    def kernel(a,b): return np.exp(-np.sum((a-b)**2)/(2*.3**2))
    for a in range(3):
        ea=np.eye(3)[a]*eps
        assert K[0,4+1+a] == pytest.approx((kernel(x,y+ea)-kernel(x,y-ea))/(2*eps),abs=1e-8)
        for b in range(3):
            eb=np.eye(3)[b]*eps
            fd=(kernel(x+ea,y+eb)-kernel(x+ea,y-eb)-kernel(x-ea,y+eb)+kernel(x-ea,y-eb))/(4*eps**2)
            assert K[1+a,4+1+b] == pytest.approx(fd,abs=2e-6)

@pytest.mark.parametrize("model", [GPIS_RBF(lengthscale=.2,sigma_0_sq=1e-10,sigma_1_sq=1e-10),
                                  DirectionalRBF(lengthscale=.2,sigma_0_sq=1e-10,sigma_1_sq=1e-10),
                                  HRBF_Model()])
def test_hermite_interpolation(model):
    p,n=sample()
    model.fit(p,n)
    f=model.field(p,with_variance=True)
    np.testing.assert_allclose(f.values,0.,atol=1e-7)
    if isinstance(model,DirectionalRBF):
        np.testing.assert_allclose(np.sum(f.gradients*n,axis=1),1.,atol=1e-7)
    else:
        np.testing.assert_allclose(f.gradients,n,atol=1e-7)
    if isinstance(model,HRBF_Model):
        assert f.variances is None
        P=polynomial(model._points)
        np.testing.assert_allclose(P.T@model._alpha,0.,atol=1e-8)
    else:
        assert np.all(f.variances>=0)

def test_gp_full_posterior_matches_dense_solve():
    p,n=sample(4)
    model=GPIS_RBF(lengthscale=.25).fit(p,n)
    queries=np.array([[.03,.02,.01],[1.,1.,1.]])
    F,_=features(queries,p,model._dirs,"rbf",model.ell,model.sigma_f2)
    A=system(p,model._dirs,"rbf",model.ell,model.sigma_f2)
    A+=np.diag(np.tile([model.sigma_0_sq]+[model.sigma_1_sq]*3,len(p)))
    y=np.column_stack([-np.ones(len(p)),n]).ravel()
    f=model.field(queries,with_variance=True)
    np.testing.assert_allclose(f.values,1+F@np.linalg.solve(A,y))
    expected=1-np.sum(F*np.linalg.solve(A,F.T).T,axis=1)
    np.testing.assert_allclose(f.variances,expected,atol=1e-12)
    far=model.field(np.array([[10.,0.,0.]]),with_variance=True)
    np.testing.assert_allclose(far.values,1.)
    np.testing.assert_allclose(far.variances,1.)
    assert model.size_bytes() > model.size_bytes(include_variance=False)

def test_log_single_observation_has_tied_scale_and_analytic_distance():
    lam=20.
    model=LogGPIS_Model(lambda_param=lam).fit(np.zeros((1,3)))
    r=.1
    f=model.field(np.array([[r,0.,0.]]),with_variance=True)
    expected=r-np.log1p(lam*r)/lam+np.log1p(model.sigma_noise_sq/model.sigma_f2)/lam
    assert model.ell==pytest.approx(np.sqrt(3)/lam)
    assert f.values[0]==pytest.approx(expected)
    np.testing.assert_allclose(f.gradients[0],[lam*r/(1+lam*r),0,0])
    assert f.variances[0]>0
    far=model.field(np.array([[1e4,0.,0.]]),with_variance=True)
    assert not far.valid_mask[0]
    assert np.isfinite(far.values).all()
    assert np.all(far.gradients==0) and np.isnan(far.variances).all()

def test_log_negative_potential_uses_absolute_value_derivative():
    model=LogGPIS_Model().fit(np.zeros((1,3)))
    q=np.array([[.1,0.,0.]])
    before=model.field(q)
    model._alpha *= -1
    after=model.field(q)
    np.testing.assert_allclose(before.values,after.values)
    np.testing.assert_allclose(before.gradients,after.gradients)

@pytest.mark.parametrize("kind", ["gp","directional","log","hrbf"])
def test_field_gradient_matches_finite_difference_and_batches(kind):
    p,n=sample()
    model={"gp":GPIS_RBF(lengthscale=.2,batch_size=2),
           "directional":DirectionalRBF(lengthscale=.2,batch_size=2),
           "log":LogGPIS_Model(lambda_param=15.,batch_size=2),
           "hrbf":HRBF_Model(batch_size=2)}[kind].fit(p,n)
    q=np.array([[.07,.04,.03],[.13,.05,-.02],[.3,.2,.1]])
    f=model.field(q)
    fd=np.column_stack([(model.field(q+e*1e-6).values-model.field(q-e*1e-6).values)/2e-6
                        for e in np.eye(3)])
    np.testing.assert_allclose(f.gradients,fd,atol=1e-6,rtol=1e-5)
    assert model.field(np.empty((0,3))).gradients.shape==(0,3)
    np.testing.assert_allclose(model.field(q[:1]).values,f.values[:1])

def test_hrbf_affine_plane_and_scaling():
    p=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
    n=np.tile([0.,0.,1.],(4,1))
    model=HRBF_Model().fit(p,n)
    query=np.array([[.3,.2,.4],[-.2,.1,-.7]])
    np.testing.assert_allclose(model.field(query).values,query[:,2],atol=1e-10)
    np.testing.assert_allclose(model.field(query).gradients,n[:2],atol=1e-10)

def test_unobservable_plane_is_not_declared_converged():
    class Plane:
        def field(self,x,**kwargs):
            return FieldEvaluation(x[:,2],np.tile([0.,0.,1.],(len(x),1)),np.ones(len(x),bool))
    p,n=sample(10)
    result=register_field(p,Plane(),np.eye(4))
    assert result.status=="DEGENERATE" and not result.converged

@pytest.mark.parametrize("factory", [GPIS_RBF,DirectionalRBF,LogGPIS_Model,HRBF_Model])
def test_models_validate_inputs(factory):
    model=factory()
    with pytest.raises(RuntimeError): model.field(np.zeros((1,3)))
    with pytest.raises(ValueError): model.fit(np.array([[np.nan,0.,0.]]),np.array([[0.,0.,1.]]))
    p,n=sample(5)
    model.fit(p,n)
    with pytest.raises(ValueError): model.field(np.zeros((2,2)))


def test_common_field_solver_recovers_full_observable_pose():
    from hgw.registration.se3 import exp_se3
    from hgw.baselines.comparison import pose_errors
    class Ellipsoid:
        def field(self,x,**kwargs):
            a=np.array([.2,.13,.07])
            return FieldEvaluation(np.sum((x/a)**2,axis=1)-1.,
                2*x/a**2,np.ones(len(x),bool))
    p,n=sample(40)
    direction=p/.2
    surface=direction*np.array([.2,.13,.07])
    truth=exp_se3(np.array([.003,-.002,.001,.01,-.02,.015]))
    scene=(surface-truth[:3,3])@truth[:3,:3]
    result=register_field(scene,Ellipsoid(),np.eye(4))
    errors=pose_errors(result.T_estimated,truth)
    assert result.converged
    assert errors["translation_error_m"]<1e-6
    assert errors["rotation_error_deg"]<1e-3
