"""Static cloud benchmark with shared data, seeds and external geometry metrics.

python -m experiments.compare_baselines model.npz scene.npz evaluation.npz \
    --m 80 --h .1 --lengthscale .05 --log-lambda 40 --match-distance .03

NPZ: points (meters), normals required only for model. evaluation must be
held out from fitting/tuning. Optional --seeds seeds.npy and --truth truth.npy.
This is not a SLAM, full MOP, or identification benchmark.
"""
import argparse
from dataclasses import asdict,dataclass
from importlib.metadata import version,PackageNotFoundError
import json
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
from hgw.models.builder import build_prior
from hgw.models.evaluator import HermiteGPIS_W
from hgw.registration.p2m import p2m_register,P2MConfig
from hgw.baselines.base import RegistrationResult
from hgw.baselines.gpis_rbf import GPIS_RBF,DirectionalRBF
from hgw.baselines.log_gpis import LogGPIS_Model
from hgw.baselines.hrbf import HRBF_Model
from hgw.baselines.icp import PointToPlaneICP,PointToPointICP,GeneralizedICP,HuberPointToPlaneICP
from hgw.baselines.registration import HGWFieldAdapter,register_field,FieldRegistrationConfig
from hgw.baselines.comparison import shared_seeds,run_multistart,pose_errors
from hgw.baselines._validation import cloud,normals as unit_normals,positive

@dataclass(frozen=True)
class BenchmarkConfig:
    m: int = 80
    h: float = .1
    lengthscale: float = .05
    log_lambda: float = 40.
    match_distance: float = .03
    icp_huber_scale: float = .01
    max_iterations: int = 50
    random_seed: int = 42
    repeats: int = 3
    translation_tolerance: float | None = None
    rotation_tolerance_deg: float | None = None
    def __post_init__(self):
        for name in ("m","max_iterations","repeats"):
            if not isinstance(getattr(self,name),int) or getattr(self,name)<1:
                raise ValueError(f"{name} must be a positive integer")
        if (self.translation_tolerance is None) != (self.rotation_tolerance_deg is None):
            raise ValueError("provide both pose success tolerances or neither")
        if self.translation_tolerance is not None:
            positive(self.translation_tolerance,"translation_tolerance")
            positive(self.rotation_tolerance_deg,"rotation_tolerance_deg")
        for name in ("h","lengthscale","log_lambda","match_distance","icp_huber_scale"):
            positive(getattr(self,name),name)

def _timed(call):
    start=perf_counter()
    result=call()
    return result,1000*(perf_counter()-start)

def _field_metrics(model,queries):
    # Warm up before separately reporting mean/gradient and mean/gradient/variance.
    model.field(queries[:1])
    f,tfield=_timed(lambda:model.field(queries))
    fv,tall=_timed(lambda:model.field(queries,with_variance=True))
    norm=np.linalg.norm(f.gradients,axis=1)
    mask=f.valid_mask & (norm>1e-3)
    if model.field_kind=="unsigned_distance":
        error=np.abs(f.values[mask])
    else:
        error=np.abs(f.values[mask])/norm[mask]
    return dict(field_kind=model.field_kind,uncertainty_kind=model.uncertainty_kind,
        query_count=len(queries),field_ms=tfield,field_and_variance_ms=tall,
        valid_fraction=float(mask.mean()),
        local_surface_rms_m=float(np.sqrt(np.mean(error*error))) if len(error) else None,
        local_surface_quartiles_m=np.quantile(error,[.25,.5,.75]).tolist() if len(error) else None,
        mean_gradient_array_bytes=model.size_bytes(include_variance=False),
        mean_gradient_variance_array_bytes=model.size_bytes(include_variance=True),
        variance_available=fv.variances is not None)

def run_benchmark(points,normals,scene,evaluation,*,config=BenchmarkConfig(),seeds=None,truth=None):
    points=cloud(points)
    normals=unit_normals(normals,points)
    scene=cloud(scene,min_points=6)
    evaluation=cloud(evaluation)
    if config.m>len(points):
        raise ValueError("m exceeds available training points")
    # Common uniform subset: no kernel-dependent selection advantage.
    idx=np.sort(np.random.default_rng(config.random_seed).choice(len(points),config.m,replace=False))
    p,n=points[idx],normals[idx]
    seeds=shared_seeds(scene,points,supplied=seeds)
    fields={}
    start=perf_counter()
    artifact=build_prior(p,n,config.h,1.,1e-4,1e-2,len(p))
    hgw=HermiteGPIS_W(artifact)
    fields["hgw_common"]=(HGWFieldAdapter(hgw),1000*(perf_counter()-start))
    factories={
        "gpis_rbf_full":lambda:GPIS_RBF(lengthscale=config.lengthscale).fit(p,n),
        "rbf_directional_ablation":lambda:DirectionalRBF(lengthscale=config.lengthscale).fit(p,n),
        "log_gpis_value_core":lambda:LogGPIS_Model(lambda_param=config.log_lambda).fit(p),
        "hrbf_cubic_full":lambda:HRBF_Model().fit(p,n)}
    for name,build in factories.items():
        fields[name]=_timed(build)
    fc=FieldRegistrationConfig(max_iterations=config.max_iterations)
    methods={name:(lambda T,model=model:register_field(scene,model,T,config=fc))
             for name,(model,_) in fields.items()}
    for name,cls in (("icp_point",PointToPointICP),("icp_plane",PointToPlaneICP),("gicp",GeneralizedICP)):
        solver=cls(max_correspondence_distance=config.match_distance,max_iterations=config.max_iterations)
        methods[name]=lambda T,solver=solver:solver.align(scene,p,T,model_normals=n)
    robust=HuberPointToPlaneICP(max_correspondence_distance=config.match_distance,
        max_iterations=config.max_iterations,huber_scale=config.icp_huber_scale)
    methods["icp_plane_huber"]=lambda T:robust.align(scene,p,T,model_normals=n)
    def native(T):
        result,ms=_timed(lambda:p2m_register(scene,hgw,T,config=P2MConfig(
            max_iterations=config.max_iterations,min_points=6)))
        return RegistrationResult(result.T_estimated,result.status=="ACCEPTED",result.iterations,
            ms,result.status,result.active_fraction,result.residual_rms,reason=result.reason)
    methods["hgw_native"]=native
    rows=[]
    field_metrics={name:_field_metrics(model,evaluation) for name,(model,_) in fields.items()}
    for repeat in range(config.repeats):
        for name,align in methods.items():
            start=perf_counter()
            result=run_multistart(align,scene,points,seeds,config.match_distance)
            best=result.results[result.best_index]
            row=dict(method=name,repeat=repeat,selected_seed=result.best_index,
                selection_ms=1000*(perf_counter()-start)-result.total_registration_ms,
                registration_ms_all_seeds=result.total_registration_ms,
                T_estimated=best.T_estimated.tolist(),
                geometry=result.metrics[result.best_index],
                trials=[dict(status=r.status,converged=r.converged,iterations=r.iterations,
                    reason=r.reason,geometry=m) for r,m in zip(result.results,result.metrics)])
            if truth is not None:
                row["pose_errors"]=pose_errors(best.T_estimated,truth)
                if config.translation_tolerance is not None:
                    row["pose_success"]=bool(
                        row["pose_errors"]["translation_error_m"] <= config.translation_tolerance
                        and row["pose_errors"]["rotation_error_deg"] <= config.rotation_tolerance_deg)
            if name in fields:
                model,build_ms=fields[name]
                row.update(build_ms=build_ms,n_constraints=(
                    model.model.n_primitives*2 if name=="hgw_common" else model.n_constraints),
                    field=field_metrics[name])
            rows.append(row)
    packages={}
    for package in ("numpy","scipy","open3d"):
        try: packages[package]=version(package)
        except PackageNotFoundError: packages[package]=None
    return dict(config=asdict(config),training_indices=idx.tolist(),seeds=[T.tolist() for T in seeds],
                environment=dict(python=platform.python_version(),platform=platform.platform(),packages=packages),
                protocol="common-uniform-subset; common seeds; fixed external geometric selection",
                rows=rows)

def _json_safe(value):
    if isinstance(value,dict): return {k:_json_safe(v) for k,v in value.items()}
    if isinstance(value,list): return [_json_safe(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value): return None
    return value

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model",type=Path)
    parser.add_argument("scene",type=Path)
    parser.add_argument("evaluation",type=Path)
    parser.add_argument("--seeds",type=Path)
    parser.add_argument("--truth",type=Path)
    parser.add_argument("--output",type=Path,default=Path("results/tables/baseline_comparison.json"))
    for flag,default,kind in (("m",80,int),("h",.1,float),("lengthscale",.05,float),
                             ("log-lambda",40.,float),("match-distance",.03,float),("icp-huber-scale",.01,float),
                             ("max-iterations",50,int),("random-seed",42,int),("repeats",3,int)):
        parser.add_argument("--"+flag,type=kind,default=default)
    parser.add_argument("--translation-tolerance",type=float)
    parser.add_argument("--rotation-tolerance-deg",type=float)
    args=parser.parse_args()
    config=BenchmarkConfig(**{k:getattr(args,k) for k in BenchmarkConfig.__dataclass_fields__})
    with np.load(args.model) as model,np.load(args.scene) as scene,np.load(args.evaluation) as evaluation:
        result=run_benchmark(model["points"],model["normals"],scene["points"],evaluation["points"],
            config=config,seeds=np.load(args.seeds) if args.seeds else None,
            truth=np.load(args.truth) if args.truth else None)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(result),indent=2,allow_nan=False))
    print(args.output)

if __name__=="__main__":
    main()
