#scripts/prepare_data.py

import trimesh
import argparse
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
CLEAN_DATA_DIR = BASE_DIR / "data" / "processed"

def load_mesh(file_path):
    """
    Load a mesh from a file and ensure it is a valid trimesh object.
    """
    file_path = RAW_DATA_DIR / file_path
    mesh = trimesh.load(file_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
    if mesh.is_empty:
        raise ValueError(f"File {file_path} is empty or could not be loaded.")
    return mesh

def normalize_mesh(mesh):
    """
    Normalize the mesh by centering it at the origin. 
    If the mesh is in millimeters, it will be converted to meters.
    """
    centroid = mesh.centroid.copy()
    mesh.apply_translation(-centroid)
    scale = np.max(mesh.extents)
    if scale > 10.0:
        # Convert to meters if the mesh is in millimeters
        scale_factor = 0.001
        mesh.apply_scale(scale_factor)
    else:
        scale_factor = 1.0
    return mesh, centroid, scale_factor

def _sample_by_area(mesh, num_points, seed):
    """
    Sample points uniformly from the surface of a mesh based on face areas.
    """
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]

    cross_prod = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross_prod, axis=1) * 0.5
    valid_faces = areas > 1e-10

    v0, v1, v2 = v0[valid_faces], v1[valid_faces], v2[valid_faces]
    areas = areas[valid_faces]
    face_normals = cross_prod[valid_faces] / (2 * areas[:, np.newaxis])

    rng = np.random.default_rng(seed)
    probs = areas / np.sum(areas)
    face_idx = rng.choice(len(areas), size=num_points, p=probs)

    u = rng.random((num_points, 2))
    flip = u[:, 0] + u[:, 1] > 1
    u[flip] = 1 - u[flip]
    a, b = u[:, 0], u[:, 1]

    points = v0[face_idx] + a[:, np.newaxis] * (v1[face_idx] - v0[face_idx]) + b[:, np.newaxis] * (v2[face_idx] - v0[face_idx])
    normals = face_normals[face_idx]

    return points, normals

def _extract_vertices(mesh):
    """
    Extract vertices and normals from a mesh.
    """
    vertices = mesh.vertices
    normals = mesh.vertex_normals
    return vertices, normals

def _extract_centroids(mesh):
    """
    Extract the centroids and normals from a mesh.
    """
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]

    centroid = (v0 + v1 + v2) / 3.0
    cross_prod = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross_prod, axis=1)
    valid_faces = norms > 1e-10
    normals = cross_prod[valid_faces] / norms[valid_faces][:, np.newaxis]
    return centroid[valid_faces], normals

def mesh_to_point_cloud(mesh, num_points=30000, mode="sample", seed=42): # mode: "sample", "vertices", "centroids"
    if mode == "sample":
        points, normals = _sample_by_area(mesh, num_points, seed)
    elif mode == "vertices":
        points, normals = _extract_vertices(mesh)
    elif mode == "centroids":
        points, normals = _extract_centroids(mesh)
    return points, normals

def save_point_cloud(points, normals, centroid, scale_factor, file_path):
    """
    Save the point cloud data to a .npz file.
    """
    file_path = CLEAN_DATA_DIR / file_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(file_path, points=points, normals=normals, centroid=centroid, scale_factor=scale_factor)
    print(f"Saved point cloud to {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare 3D mesh data for point cloud processing.")
    parser.add_argument("mesh_file", type=str, help="Path to the input mesh file relative to the raw data directory.")
    parser.add_argument("output_file", type=str, help="Path to the output .npz file relative to the processed data directory.")
    parser.add_argument("--mode", type=str, choices=["sample", "vertices", "centroids"], default="sample", help="Mode of point cloud extraction.")
    parser.add_argument("--n_points", type=int, default=30000, help="Number of points to sample from the mesh.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    mesh = load_mesh(args.mesh_file)
    mesh, centroid, scale_factor = normalize_mesh(mesh)
    points, normals = mesh_to_point_cloud(mesh, num_points=args.n_points, mode=args.mode, seed=args.seed)
    save_point_cloud(points, normals, centroid, scale_factor, args.output_file)

if __name__ == "__main__":
    main()