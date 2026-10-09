# Mathematical review of the supplied audit

Accepted: direct Hessian-vector contraction, batch left-increment SE(3)
Jacobians [gradient, point cross gradient], controlled empty seed results,
and statistical inliers separate from Huber optimization weights.

Corrections to the proposal:
- Support is strict: distance < h. At h the kernel and its first two
  derivatives vanish. Empirical fill distance < h only certifies the sampled
  evaluation cloud, not an entire unknown continuous surface.
- Absolute field values are not generally distances. The sweep reports the
  local approximation |m| / ||gradient m||, excluding unsupported points and
  near-zero gradients, and records both support and valid fractions.
- Selection uses a scalar noisy value kernel, not the full Hermite matrix.
  A residual trace ratio is a matrix approximation diagnostic, not a general
  bound on Sobolev reconstruction energy. The pivot column must include its
  noise diagonal, otherwise pivots can repeat.
- No dataset-independent overlap target, optimum (h,M), or speedup follows
  from the formulas. The sweep measures build/query times and Pareto dominance.
  Supply independent evaluation points to avoid training-set optimism.
- Deviance after local chi-square trimming is not exactly chi-square:
  truncation, spatial dependence and estimated poses change its null law.
  Current thresholds/p-values are nominal diagnostics requiring calibration.
- A chi-square(1) score gap for distinct, non-nested candidate objects is
  heuristic, not justified by Wilks' theorem. Candidate-specific inlier sets
  also prevent interpreting the current GLRT score as a common-data likelihood.
  These statistical decisions need a specified outlier model and empirical
  calibration before publication; this review does not silently replace them.
- More than ten ICP correspondences does not prove convergence; neither does
  reporting the configured iteration limit as iterations actually executed.

No manuscript main.tex was present, so its numbered propositions were not
verified. Full surface reconstruction, identification experiments and calibration
remain separate from the supplied frontier diagnostic.
