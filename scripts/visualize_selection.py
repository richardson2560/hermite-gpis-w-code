"""Visualize pivoted Cholesky selections over the dense clean cloud.

Loads a processed .npz (dense clean cloud, N points) and an artifact .npz
(M pivots, normals, and support radius h), and overlays them in a single
Open3D window.

Options:
    --disks         Show a semi-transparent disk of radius h at each pivot,
                    lying in the local tangent plane.
    --normals       Show each pivot's outward normal as a short arrow.
    --max-disks N   Limit the number of displayed disks.

Usage:
    python scripts/visualize_selection.py \\
        data/processed/torus.npz \\
        data/artifacts/torus_hgw.npz \\
        --disks --normals

    python scripts/visualize_selection.py \\
        data/processed/torus.npz \\
        data/artifacts/torus_hgw.npz \\
        --disks --max-disks 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Color constants (RGB in [0, 1])
# ---------------------------------------------------------------------------

_COLOR_DENSE = (0.65, 0.65, 0.65)     # light gray
_COLOR_PIVOT = (1.00, 0.15, 0.15)     # red
_COLOR_DISK = (1.00, 0.15, 0.15)      # green
_COLOR_NORMAL = (0.95, 0.55, 0.00)    # orange


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dense_cloud(path: Path) -> np.ndarray:
    """Load points from a processed .npz (dense cloud).

    Args:
        path: path to the processed .npz file.

    Returns:
        (N, 3) array of points.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if 'points' is missing or malformed.
    """
    if not path.exists():
        raise FileNotFoundError(f"processed cloud not found: {path}")

    with np.load(path) as data:
        if "points" not in data.files:
            raise ValueError(f"{path} is missing 'points'")
        points = np.asarray(data["points"], dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")

    return points


def load_artifact(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load points, normals, and h from an artifact .npz.

    Args:
        path: path to the artifact .npz file.

    Returns:
        (points, normals, h) with shapes (M, 3), (M, 3), and scalar.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if required fields are missing or malformed.
    """
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {path}")

    with np.load(path) as data:
        for field in ("points", "normals", "h"):
            if field not in data.files:
                raise ValueError(f"{path} is missing '{field}'")
        points = np.asarray(data["points"], dtype=np.float64)
        normals = np.asarray(data["normals"], dtype=np.float64)
        h = float(data["h"])

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"artifact points must have shape (M, 3), got {points.shape}")
    if normals.shape != points.shape:
        raise ValueError(
            f"artifact normals shape {normals.shape} must match points {points.shape}"
        )
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError(f"artifact h must be positive and finite, got {h}")

    return points, normals, h


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors spanning the plane perpendicular to normal."""
    n = normal / np.linalg.norm(normal)
    # Pick a seed vector not parallel to n
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, seed)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def make_disk_outline(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    n_segments: int = 32,
) -> o3d.geometry.LineSet:
    """Build a circle outline of radius `radius` in the tangent plane.

    The circle lies in the plane perpendicular to `normal`, centered at
    `center`. This is the disk boundary that a spherical support of radius
    `radius` would trace on a locally flat surface.

    Args:
        center:     (3,) pivot position.
        normal:     (3,) unit outward normal.
        radius:     disk radius.
        n_segments: number of segments around the circle.

    Returns:
        A LineSet representing the closed circle.
    """
    u, v = _tangent_basis(normal)
    angles = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    circle = np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v)
    vertices = circle * radius + center

    lines = [[i, (i + 1) % n_segments] for i in range(n_segments)]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    line_set.paint_uniform_color(_COLOR_DISK)
    return line_set


def make_disk_mesh(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    n_segments: int = 24,
) -> o3d.geometry.TriangleMesh:
    """Build a filled disk (triangle fan) in the tangent plane.

    Warning: the default Open3D viewer does not render per-vertex alpha
    reliably. With many overlapping disks the surface may become invisible.
    Prefer `make_disk_outline` unless the number of disks is small.

    Args:
        center:     (3,) pivot position.
        normal:     (3,) unit outward normal.
        radius:     disk radius.
        n_segments: number of triangles in the fan.

    Returns:
        A TriangleMesh representing the filled disk.
    """
    u, v = _tangent_basis(normal)
    angles = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    circle = np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v)
    circle *= radius

    vertices = np.vstack([np.zeros(3), circle]) + center

    triangles = []
    for i in range(n_segments):
        a = 0
        b = i + 1
        c = ((i + 1) % n_segments) + 1
        triangles.append([a, b, c])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
    mesh.paint_uniform_color(_COLOR_DISK)
    mesh.compute_vertex_normals()
    return mesh


def make_normal_lines(
    points: np.ndarray,
    normals: np.ndarray,
    length: float,
) -> o3d.geometry.LineSet:
    """Build a LineSet showing each normal as a segment of given length."""
    starts = points
    ends = points + length * normals
    all_points = np.vstack([starts, ends])

    lines = [[i, i + len(points)] for i in range(len(points))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    line_set.paint_uniform_color(_COLOR_NORMAL)
    return line_set


def make_point_cloud(points: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    """Build a uniform-colored point cloud."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.paint_uniform_color(list(color))
    return cloud


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize pivoted Cholesky selections over the dense cloud."
    )
    parser.add_argument(
        "processed_npz",
        type=Path,
        help="Path to the processed .npz (dense cloud with 'points' and 'normals').",
    )
    parser.add_argument(
        "artifact_npz",
        type=Path,
        help="Path to the artifact .npz (with 'points', 'normals', 'h').",
    )
    parser.add_argument(
        "--disks",
        action="store_true",
        help="Show disk outlines of radius h at each pivot (tangent plane).",
    )
    parser.add_argument(
        "--filled-disks",
        action="store_true",
        help="Show filled disks instead of outlines (may be dense).",
    )
    parser.add_argument(
        "--normals",
        action="store_true",
        help="Show each pivot's outward normal as a short arrow.",
    )
    parser.add_argument(
        "--normal-length",
        type=float,
        default=0.02,
        help="Length of the normal arrows in meters (default: 0.02).",
    )
    parser.add_argument(
        "--max-disks",
        type=int,
        default=None,
        help="Limit the number of displayed disks (default: show all).",
    )
    parser.add_argument(
        "--disk-radius",
        type=float,
        default=None,
        help="Override the disk radius (default: use h from the artifact).",
    )
    parser.add_argument(
        "--no-frame",
        action="store_true",
        help="Hide the coordinate frame.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[visualize_selection] Loading dense cloud: {args.processed_npz}")
    dense = load_dense_cloud(args.processed_npz)
    print(f"[visualize_selection]   {len(dense)} points")

    print(f"[visualize_selection] Loading artifact: {args.artifact_npz}")
    pivots, normals, h = load_artifact(args.artifact_npz)
    print(f"[visualize_selection]   {len(pivots)} pivots, h = {h:.4f} m")

    geometries: list = []

    # 1. Dense cloud (gray)
    geometries.append(make_point_cloud(dense, _COLOR_DENSE))

    # 2. Pivot points (red)
    geometries.append(make_point_cloud(pivots, _COLOR_PIVOT))

    # 3. Support disks (optional)
    disk_radius = args.disk_radius if args.disk_radius is not None else h
    if args.disks or args.filled_disks:
        n_disks = len(pivots) if args.max_disks is None else min(args.max_disks, len(pivots))
        print(f"[visualize_selection] Adding {n_disks} disks of radius {disk_radius:.4f} m")
        for i in range(n_disks):
            if args.filled_disks:
                disk = make_disk_mesh(pivots[i], normals[i], disk_radius)
            else:
                disk = make_disk_outline(pivots[i], normals[i], disk_radius)
            geometries.append(disk)

    # 4. Normal arrows (optional)
    if args.normals:
        print(f"[visualize_selection] Adding normal arrows (length {args.normal_length})")
        geometries.append(make_normal_lines(pivots, normals, args.normal_length))

    # 5. World frame
    if not args.no_frame:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

    window_name = f"{args.processed_npz.stem} vs {args.artifact_npz.stem}"
    print(f"[visualize_selection] Opening viewer: {window_name}")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1280,
        height=720,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())