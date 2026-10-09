#scripts/build_priors.py

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from hgw.models.artifact import ModelArtifact
from hgw.models.builder import build_prior

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "data" / "artifacts"

def load_clean_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load points and normals from a .npz produced by prepare_data.py.

    Args:
        path: path to the clean .npz file.

    Returns:
        (points, normals) each of shape (N, 3), dtype float64.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if required fields are missing or malformed.
    """
    if not path.exists():
        raise FileNotFoundError(f"clean cloud not found: {path}")

    with np.load(path) as data:
        if "points" not in data.files or "normals" not in data.files:
            raise ValueError(
                f"clean .npz must contain 'points' and 'normals', "
                f"got {sorted(data.files)}"
            )
        points = np.asarray(data["points"], dtype=np.float64)
        normals = np.asarray(data["normals"], dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if normals.shape != points.shape:
        raise ValueError(
            f"normals shape {normals.shape} must match points shape {points.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("points contain non-finite values")
    if not np.all(np.isfinite(normals)):
        raise ValueError("normals contain non-finite values")

    return points, normals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact HGW prior from a clean point cloud."
    )
    parser.add_argument(
        "input_npz",
        type=Path,
        help="Path to the clean .npz file (from prepare_data.py).",
    )
    parser.add_argument(
        "output_name",
        type=str,
        help="Artifact name (saved as data/artifacts/<output_name>.npz).",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=0.05,
        help="Support radius in meters (default: 0.05).",
    )
    parser.add_argument(
        "--sigma-f2",
        type=float,
        default=1.0,
        help="Kernel amplitude (default: 1.0).",
    )
    parser.add_argument(
        "--sigma-0-sq",
        type=float,
        default=1e-4,
        help="Value-channel noise variance (default: 1e-4).",
    )
    parser.add_argument(
        "--sigma-1-sq",
        type=float,
        default=1e-2,
        help="Derivative-channel noise variance (default: 1e-2).",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=200,
        help="Maximum number of primitives (default: 200).",
    )
    parser.add_argument(
        "--eps-tol",
        type=float,
        default=1e-6,
        help="Pivoted Cholesky stopping tolerance (default: 1e-6).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[build_priors] Loading clean cloud from {args.input_npz}")
    points, normals = load_clean_npz(args.input_npz)
    print(f"[build_priors] Loaded {len(points)} points")

    print(
        f"[build_priors] Building prior with M={args.m}, h={args.h}, "
        f"sigma_f2={args.sigma_f2}, sigma_0_sq={args.sigma_0_sq}, "
        f"sigma_1_sq={args.sigma_1_sq}"
    )
    t0 = time.perf_counter()
    artifact = build_prior(
        points=points,
        normals=normals,
        h=args.h,
        sigma_f2=args.sigma_f2,
        sigma_0_sq=args.sigma_0_sq,
        sigma_1_sq=args.sigma_1_sq,
        m_target=args.m,
        eps_tol=args.eps_tol,
    )
    elapsed = time.perf_counter() - t0

    # Ensure the artifact name has the .npz suffix
    output_name = args.output_name
    if not output_name.endswith(".npz"):
        output_name += ".npz"
    output_path = ARTIFACTS_DIR / output_name

    artifact.save(output_path)

    file_kb = output_path.stat().st_size / 1024
    raw_kb = artifact.size_bytes() / 1024

    print("[build_priors] Done.")
    print(f"  primitives:    {artifact.n_primitives}")
    print(f"  raw size:      {raw_kb:.2f} KB")
    print(f"  compressed:    {file_kb:.2f} KB")
    print(f"  build time:    {elapsed:.3f} s")
    print(f"  saved to:      {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())