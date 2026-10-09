"""Measured (h, M) trade-offs on separate training and evaluation clouds.

Run: python -m experiments.exp_h_m_frontier train.npz evaluation.npz
Coordinates must be in meters. Coverage is empirical, not a surface guarantee.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from hgw.models.builder import build_prior
from hgw.models.evaluator import HermiteGPIS_W

def pareto_mask(rows):
    """Minimize orthogonal RMS and bytes, maximize valid evaluation coverage."""
    values = np.array([[-r["valid_fraction"], r["rms_mm"], r["bytes"]]
                       for r in rows], dtype=float)
    return [not any(np.all(v <= x) and np.any(v < x) for v in values)
            for x in values]

def sweep_h_and_m(points, normals, evaluation_points, h_values, m_values,
                  min_gradient_norm=1e-3):
    evaluation_points = np.asarray(evaluation_points, dtype=float)
    if (evaluation_points.ndim != 2 or evaluation_points.shape[1] != 3
            or not len(evaluation_points) or not np.isfinite(evaluation_points).all()):
        raise ValueError("evaluation_points must be a nonempty finite (N, 3) array")
    if not np.isfinite(min_gradient_norm) or min_gradient_norm <= 0:
        raise ValueError("min_gradient_norm must be positive and finite")
    rows = []
    for m in m_values:
        for h in h_values:
            start = perf_counter()
            artifact = build_prior(points, normals, h, 1.0, 1e-4, 1e-2, m)
            build_ms = 1000 * (perf_counter() - start)
            model = HermiteGPIS_W(artifact)
            distances, _ = cKDTree(model._points).query(evaluation_points)
            start = perf_counter()
            means, grads, _ = model.evaluate_many(evaluation_points)
            query_ms = 1000 * (perf_counter() - start)
            norms = np.linalg.norm(grads, axis=1)
            valid = (distances < h) & (norms >= min_gradient_norm)
            # Local first-order distance estimate, not an exact surface distance.
            errors = np.abs(means[valid]) / norms[valid] * 1000
            quartiles = np.percentile(errors, [25, 50, 75]) if len(errors) else [None]*3
            rows.append(dict(M_requested=m, M_actual=artifact.n_primitives, h=h,
                coverage=float(np.mean(distances < h)),
                empirical_fill_distance=float(distances.max()),
                valid_fraction=float(valid.mean()), bytes=artifact.size_bytes(),
                rms_mm=float(np.sqrt(np.mean(errors**2))) if len(errors) else float("inf"),
                q25_mm=quartiles[0], q50_mm=quartiles[1], q75_mm=quartiles[2],
                build_ms=build_ms, query_ms=query_ms))
    for row, is_pareto in zip(rows, pareto_mask(rows)):
        row["pareto"] = bool(is_pareto)
    return rows

def plot_frontier(rows, path):
    """Plot empirical coverage, local-distance quartiles, and Pareto trade-offs."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for m in sorted({r["M_requested"] for r in rows}):
        subset = sorted((r for r in rows if r["M_requested"] == m), key=lambda r: r["h"])
        hs = [r["h"] for r in subset]
        axes[0].plot(hs, [r["coverage"] for r in subset], "o-", label=f"M={m}")
        med = [r["q50_mm"] if r["q50_mm"] is not None else np.nan for r in subset]
        axes[1].plot(hs, med, "o-", label=f"M={m}")
        axes[1].fill_between(hs,
            [r["q25_mm"] if r["q25_mm"] is not None else np.nan for r in subset],
            [r["q75_mm"] if r["q75_mm"] is not None else np.nan for r in subset], alpha=.15)
    finite = [r for r in rows if r["rms_mm"] is not None and np.isfinite(r["rms_mm"])]
    axes[2].scatter([r["bytes"]/1024 for r in finite], [r["rms_mm"] for r in finite],
                    c=[r["valid_fraction"] for r in finite], vmin=0, vmax=1)
    front = [r for r in finite if r["pareto"]]
    axes[2].scatter([r["bytes"]/1024 for r in front], [r["rms_mm"] for r in front],
                    facecolors="none", edgecolors="red", s=100, label="Pareto")
    for ax in axes[:2]:
        ax.set_xlabel("Support h (m)")
        ax.legend()
    axes[0].set_ylabel("Empirical support fraction")
    axes[1].set_ylabel("Local distance estimate (mm), median and IQR")
    axes[2].set_xlabel("Raw artifact (KiB)")
    axes[2].set_ylabel("Local distance RMS (mm)")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--h", type=float, nargs="+", default=[.02, .035, .05, .075, .1])
    parser.add_argument("--m", type=int, nargs="+", default=[50, 100, 150, 200, 300])
    parser.add_argument("--output", type=Path, default=Path("results/h_m_frontier.json"))
    parser.add_argument("--plot", type=Path, help="Optional figure (requires matplotlib)")
    args = parser.parse_args()
    with np.load(args.training) as train, np.load(args.evaluation) as test:
        rows = sweep_h_and_m(train["points"], train["normals"], test["points"], args.h, args.m)
    # JSON null denotes an undefined error (no valid evaluation queries).
    for row in rows:
        if not np.isfinite(row["rms_mm"]):
            row["rms_mm"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, allow_nan=False))
    if args.plot:
        plot_frontier(rows, args.plot)
    print(args.output)

if __name__ == "__main__":
    main()
