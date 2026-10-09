# Static-cloud baseline implementations and comparison protocol

These are independent implementations of the specified mathematical cores,
with explicit adaptations. They are not bit-for-bit reproductions of complete
robotics systems. No performance advantage for HGW is assumed.

| Name in benchmark | Implementation | Supported comparison |
|---|---|---|
| gpis_rbf_full | RBF GP, value + 3 gradient channels, constant prior 1 | field, geometry, memory, exact latent variance |
| rbf_directional_ablation | RBF GP, value + directional derivative, prior 0 | controlled kernel ablation against HGW |
| log_gpis_value_core | dense Mat?rn 3/2 GP on surface heat values 1, log transform | unsigned distance, field alignment, delta-method uncertainty |
| hrbf_cubic_full | cubic full Hermite interpolation, affine polynomial and side constraints | field and geometry, no uncertainty |
| icp_point | Open3D rigid point-to-point ICP | pose and common geometry |
| icp_plane | Open3D nearest-neighbour point-to-plane ICP, L2 | pose and common geometry |
| icp_plane_huber | Open3D point-to-plane with Huber IRLS, threshold in meters | robust native-pipeline comparison |
| gicp | Open3D GICP, k-neighbour planar covariances on both clouds | pose and common geometry |
| hgw_common | current HGW field with shared least-squares solver | representation ablation |
| hgw_native | current P2M, its Huber loss, seed-centred weak pose prior and nominal deviance inliers | native-pipeline comparison |

## Source audit

[Dragiev, Toussaint & Gienger, ICRA 2011, Sec. II-C/D](https://argmin.lis.tu-berlin.de/papers/11-dragiev-ICRA.pdf)
conditions on values and complete gradients and fixes the prior bias to 1.
Contrary to the attachment, this paper does not require manufactured offset
points. Our full RBF GP uses its observation construction; the two-channel GP
is labelled separately as an ablation. Channel-specific positive Gaussian
noise is configurable. Inference uses a Cholesky solve, never an explicit
inverse. Point-normal data are metric; prior bias and target gradient scale
must be recorded when tuning.

[Wu, Lee, Le Gentil & Vidal-Calleja, RA-L 2021, equations 12?14](https://arxiv.org/pdf/2010.11487):
lambda is an inverse distance, ell=sqrt(3)/lambda and
d=-log(mu)/lambda. The attachment's independent ell and distance-scale lambda
do not specify this paper parameterization. The dense value-observation core
is implemented; tree clustering, online fusion and sensor-based sign recovery
are outside this static-cloud comparison.
The [MOP paper](https://arxiv.org/html/2206.09506v3) additionally uses normals,
map fusion, trajectory optimisation, and scalable mapping. This implementation
is deliberately not called a complete Log-GPIS-MOP reproduction.
For non-positive latent means, we use log(abs(mu)) as described in MOP:
its derivative uses the signed mu in the denominator. A numerical floor
defines an invalid-query region with zero derivative and unavailable variance;
it cannot silently become a valid registration point.
The analytic scalar derivative is used by the common optimizer; the normalized
Eikonal search direction is exposed separately. Delta-method distance variance
is an approximation, not a calibrated signed Gaussian likelihood.

[Macedo, Gois & Velho, CGF 2011](https://doi.org/10.1111/j.1467-8659.2010.01785.x)
and the [authors' HRBF formulation](https://www.visgraf.impa.br/Data/RefBib/PS_PDF/npar10/npar10-id23-apr-16.pdf):
full gradients, not just normal directional constraints. Our supported member
is the cubic r^3 basis with affine polynomial and polynomial side conditions.
The default interpolates; ridge regularization is an explicitly labelled
variant, not a mandatory ingredient. Coordinate scaling preserves unit normal
constraints and improves conditioning. The Gaussian two-channel system from
the attachment is not presented as the Macedo baseline.
Deterministic HRBF returns no variance, not a fictitious GP covariance.

[Chen & Medioni 1991/1992](https://graphics.stanford.edu/~smr/ICP/comparison/chen-medioni-align-rob91.pdf):
our point-to-plane objective is conventional nearest-neighbour ICP, rather
than a reconstruction of their original range-image correspondence/control-
point pipeline. Known target normals can be supplied; otherwise Open3D
estimates them using the configured radius.

[Segal, Haehnel & Thrun, RSS 2009](https://www.robots.ox.ac.uk/~avsegal/resources/papers/Generalized_ICP.pdf):
our GICP explicitly constructs covariance eigenvalues (epsilon,1,1) for both
clouds using 20 neighbours by default. It then uses the established
[Open3D GICP solver](https://www.open3d.org/docs/0.19.0/python_api/open3d.pipelines.registration.TransformationEstimationForGeneralizedICP.html).
It does not substitute raw sample covariances for planar regularization.
Source covariances are estimated in the source frame; the backend rotates them.
The target uses supplied normals if available.
All ICP adapters report unavailable iteration/convergence diagnostics as None.
Nonempty correspondences do not establish convergence or pose correctness.
The native fitness/RMSE is not used to compare against HGW's field loss.

The additional point-to-point baseline implements the classical rigid ICP
objective of [Besl & McKay (1992)](https://doi.org/10.1109/34.121791), using Open3D's
rigid estimator without scale. It is a useful simpler correspondence baseline.
Full MOP/SLAM, learned registration and TSDF pipelines require different inputs,
training or sequential data and are not silently reduced to static-cloud clones.

## Reproducible comparison

Install optional ICP dependencies with: pip install -e ".[icp]"

Run from the repository root:

    python -m experiments.compare_baselines model.npz scene.npz heldout.npz \
        --m 80 --h 0.1 --lengthscale 0.05 --log-lambda 40 \
        --match-distance 0.03 --output results/tables/baselines.json

All coordinates are meters. model.npz needs points and normals; scene/heldout
need points. heldout must not be used for fitting or hyperparameter tuning.
--truth truth.npy is only for reporting errors, never seed selection.
--seeds seeds.npy supplies exactly the same (S,4,4) initial poses to all methods;
otherwise identity and centroid initialization are used. Hermite-anchor seeds
can be supplied in this way but should be labelled as an HGW-derived shared
initializer. Seed generation time should be measured separately.

Every method receives the same uniform subset of training points and available
target normals. The subset indices and poses are recorded. Full-gradient
baselines have 4M constraints, HGW/directional RBF 2M, and value Log-GPIS M:
equal points is not equal constraints or memory. Complement these results
with sweeps at equal stored bytes and method-specific best validated settings.
The current runner fixes the input count; it does not claim a memory-matched
study or automatically tune hyperparameters.

All seeds run; total registration cost includes every trial. Seed selection
uses common geometric overlap then clipped all-scene nearest-neighbour RMSE.
The clipping threshold is fixed across methods. This is explicitly a robust
selection rule, not each algorithm's native objective; report all trials and
also common single-start results. Ground truth is used only for translation
and rotation error reporting. Symmetric Chamfer is reported separately because
it penalizes unobserved surfaces in partial scans. Scene RMSE includes outliers
rather than discarding unmatched points. Native ACCEPTED/COMPLETED statuses
are not definitions of benchmark success: set success tolerances on validation
data, apply them to held-out pose errors, and include failures in aggregates.

Field timing separates mean/gradient and mean/gradient/variance requests.
HGW currently bundles its variance proxy even on a mean/gradient request;
GP variances are exact latent values, Log-GPIS uses delta propagation.
Query counts, build costs, available uncertainty and raw retained array bytes
are recorded. Raw arrays exclude Python objects, KD-tree allocations,
temporary solves, metadata and serialized-file overhead; measure peak RSS and
serialized artifacts separately for publication. Mean-only GP deployment can
discard its Cholesky factor; full-uncertainty deployment retains it.
No fixed KB numbers are assumed.

Tune each method on validation scenes with the same search/compute budget;
do not set a universal 5 cm cutoff regardless of scale. Include matched-start
registration, independent clean/noisy surface evaluation, controlled occlusion,
outliers, repetitions and random seeds. Report distributions and failures.
Do not apply HGW chi-square tests to unsigned/deterministic methods or compare
raw GP field likelihoods with different units as a common identification score.

## Verification

Tests check observation signs via mixed differences, field gradients, exact
GP posterior against an independent dense solve, full Hermite constraints,
affine reproduction and polynomial side conditions, logarithmic scale and
floor behaviour, geometric metrics, covariance eigenvalues and genuine
Open3D rigid recovery for all three ICP methods. Plane degeneracy is rejected
by the common optimizer instead of being called convergence.

The runner repeats registration three times by default (--repeats); these are
runtime repeats of the same case, not independent noise/occlusion trials.
Field/build measurements are performed once per fitted model and reused in
each runtime row. Supply --translation-tolerance (meters) together with
--rotation-tolerance-deg and --truth to record pose_success independently of
native solver labels; calibrate tolerances outside the test set.
Symmetric objects require symmetry-aware pose metrics before aggregating
rotation errors; current geodesic errors assume an unambiguous object frame.

HGW retains float32 artifact arrays whereas dense baselines retain float64
arrays. Equal-point kernel ablations therefore also differ in retained
precision; report this difference. Vectorized NumPy, Python neighbour loops
and compiled Open3D have different implementation costs, so these timings
cannot alone establish an intrinsic kernel-complexity advantage.


For occlusion/outliers the runner also includes icp_plane_huber, avoiding a
comparison of HGW's robust optimizer exclusively against L2 competitors.
Its Huber threshold is configurable with --icp-huber-scale (meters);
it is not numerically the same as HGW's dimensionless standardized threshold.
This is a labelled robust variant, not a separate claimed original algorithm.
See [Babin, Giguere & Pomerleau, ICRA 2019](https://www2.ift.ulaval.ca/~pgiguere/papers/RobustFunc.ICRA2019.pdf)
and [Open3D's robust ICP implementation](https://open3d.org/docs/latest/tutorial/pipelines/robust_kernels.html).
