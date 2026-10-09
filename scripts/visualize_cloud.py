#scripts/visualize_cloud.py

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load points and normals from a prepared .npz."""
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    data = np.load(path)
    if "points" not in data or "normals" not in data:
        raise ValueError("npz must contain 'points' and 'normals' arrays")
    points = np.asarray(data["points"], dtype=np.float64)
    normals = np.asarray(data["normals"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if normals.shape != points.shape:
        raise ValueError(f"normals shape {normals.shape} != points shape {points.shape}")
    return points, normals


def build_cloud(points: np.ndarray, normals: np.ndarray) -> o3d.geometry.PointCloud:
    """Build an Open3D point cloud with normals attached."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.normals = o3d.utility.Vector3dVector(normals)
    return cloud


def build_frame() -> o3d.geometry.TriangleMesh:
    """Unit-sized world frame for orientation reference."""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)


def visualize(
    cloud: o3d.geometry.PointCloud,
    show_normals: bool,
    normal_length: float,
) -> None:
    """Open the interactive viewer."""
    geometries = [build_frame(), cloud]
    if show_normals:
        geometries.append(
            o3d.geometry.LineSet.create_from_point_cloud_correspondences(
                cloud,
                cloud.translate(cloud.get_center()),   # dummy: se reemplaza abajo
                [(i, i) for i in range(len(cloud.points))],
            )
        )
        # Alternativa: usar el visualizador nativo de normales
        o3d.visualization.draw_geometries(
            geometries[:-1],
            window_name="HGW point cloud (normals via vertex renderer)",
            point_show_normal=True,
            width=1280,
            height=720,
        )
        return
    o3d.visualization.draw_geometries(
        geometries,
        window_name="HGW point cloud",
        width=1280,
        height=720,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize a prepared .npz point cloud.")
    parser.add_argument("npz_path", type=Path, help="Path to the .npz file.")
    parser.add_argument("--normals", action="store_true", help="Show normal vectors.")
    parser.add_argument(
        "--normal-length",
        type=float,
        default=0.01,
        help="Normal vector length for display (only used by Open3D's renderer).",
    )
    parser.add_argument(
        "--no-frame",
        action="store_true",
        help="Hide the world coordinate frame.",
    )
    args = parser.parse_args()

    npz_path = DATA_DIR / args.npz_path
    points, normals = load_npz(npz_path)
    print(f"Loaded {len(points)} points from {npz_path}")
    print(f"  bounds: min={points.min(axis=0)}, max={points.max(axis=0)}")
    print(f"  centroid: {points.mean(axis=0)}")
    print(f"  normal length mean: {np.linalg.norm(normals, axis=1).mean():.6f}")

    cloud = build_cloud(points, normals)

    # Paint by normal direction for visual sanity check
    colors = (normals + 1.0) / 2.0
    cloud.colors = o3d.utility.Vector3dVector(colors)

    geometries = [cloud] if args.no_frame else [build_frame(), cloud]

    o3d.visualization.draw_geometries(
        geometries,
        window_name=str(npz_path.name),
        point_show_normal=args.normals,
        width=1280,
        height=720,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())