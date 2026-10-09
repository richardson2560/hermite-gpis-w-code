#scripts/download_datasets.py

import trimesh
import urllib.request
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

(DATA_DIR / "synthetic").mkdir(exist_ok=True)
(DATA_DIR / "graphics_benchmarks").mkdir(exist_ok=True)
(DATA_DIR / "ycb").mkdir(exist_ok=True)

def setup_synthetic_dataset():
    print("[1/3]")
    trimesh.primitives.Sphere(radius=0.5, subdivisions=4).export(DATA_DIR / "synthetic" / "sphere.obj")
    trimesh.creation.torus(major_radius=0.2, minor_radius=0.09, major_sections=32, minor_sections=64).export(DATA_DIR / "synthetic" / "torus.obj")
    t1 = trimesh.creation.torus(major_radius=0.2, minor_radius=0.06, major_sections=32, minor_sections=64)
    t2 = trimesh.creation.torus(major_radius=0.2, minor_radius=0.06, major_sections=32, minor_sections=64)
    R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    T = trimesh.transformations.translation_matrix([0.2, 0, 0])
    t2.apply_transform(T @ R)
    chain = trimesh.util.concatenate([t1, t2])
    chain.export(DATA_DIR / "synthetic" / "hopf_link.obj")

def download_graphics_benchmarks():
    print("[2/3]")
    MODELS = {
        "stanford_bunny.obj": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj",
        "fandisk.obj": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/fandisk.obj"
    }
    for name, url in MODELS.items():
        dst = DATA_DIR / "graphics_benchmarks" / name
        if not dst.exists():
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, dst)
        else:
            print(f"{name} already exists, skipping download.")

def download_ycb_dataset():
    print("[3/3]")
    YCB_CORE = [
        "004_sugar_box",       
        "005_tomato_soup_can",  
        "025_mug",              
        "035_power_drill"      
    ]
    S3_BASE = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/berkeley"
    for item in YCB_CORE:
        out_dir = DATA_DIR / "ycb" / item
        out_dir.mkdir(exist_ok=True)
        final_mesh = out_dir / "textured.obj"
        if not final_mesh.exists():
            print(f"Downloading {item}...")
            tgz_url = f"{S3_BASE}/{item}/{item}_berkeley_meshes.tgz"
            tgz_path = out_dir / f"{item}.tgz"
            try:
                urllib.request.urlretrieve(tgz_url, tgz_path)
                import tarfile
                with tarfile.open(tgz_path, "r:gz") as tar:
                    tar.extractall(path=DATA_DIR / "ycb")
                tgz_path.unlink()
            except Exception as e:
                print(f"Error downloading {item}: {e}")

if __name__ == "__main__":
    setup_synthetic_dataset()
    download_graphics_benchmarks()
    download_ycb_dataset()
    print("All datasets have been set up successfully.")