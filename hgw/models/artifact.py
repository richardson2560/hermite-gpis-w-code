#hgw/models/artifact.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "data" / "artifacts"
CLEAN_DATA_DIR = BASE_DIR / "data" / "processed"

_SCHEMA_VERSION = 1
_REQUIRED_FIELDS = frozenset({
    "schema_version",
    "points",
    "normals",
    "alpha",
    "h",
    "sigma_f2",
    "sigma_0_sq",
    "sigma_1_sq",
})

@dataclass(frozen=True)
class ModelArtifact:
    """A model artifact is a named object that can be used to identify a model."""

    points: np.ndarray
    normals: np.ndarray
    alpha: np.ndarray
    h: float
    sigma_f2: float
    sigma_0_sq: float
    sigma_1_sq: float

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"points must be a finite Mx3 array: {self.points}")
        if points.shape[0] < 1:
            raise ValueError(f"points must have at least one point: {self.points}")
        normals = np.asarray(self.normals, dtype=np.float32)
        if normals.shape != points.shape or not np.isfinite(normals).all():
            raise ValueError(f"normals must be a finite Mx3 array matching points: {self.normals}")
        norms = np.linalg.norm(normals, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError(f"normals must be unit vectors: {self.normals}")
        alpha = np.asarray(self.alpha, dtype=np.float32)
        expected_shape = (points.shape[0]*2,)
        if alpha.shape != expected_shape or not np.isfinite(alpha).all():
            raise ValueError(f"alpha must be a finite array of shape {expected_shape}: {self.alpha}")

        for name, value in (
            ("h", self.h),
            ("sigma_f2", self.sigma_f2),
            ("sigma_0_sq", self.sigma_0_sq),
            ("sigma_1_sq", self.sigma_1_sq),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {value!r}")

        object.__setattr__(self, "points", points)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "alpha", alpha)

    @property
    def n_primitives(self) -> int:
        """Return the number of primitives in the artifact."""
        return self.points.shape[0]

    def size_bytes(self) -> int:
        """Approximate size of the raw arrays in bytes (excluding .npz overhead)."""
        return (
            self.points.nbytes
            + self.normals.nbytes
            + self.alpha.nbytes
            + 4 * 4  # four float32 scalars
        )

    def save(self, path: Path) -> None:
        """Save the artifact to a .npz file."""
        path = ARTIFACTS_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.int32(_SCHEMA_VERSION),
            points=self.points,
            normals=self.normals,
            alpha=self.alpha,
            h=self.h,
            sigma_f2=self.sigma_f2,
            sigma_0_sq=self.sigma_0_sq,
            sigma_1_sq=self.sigma_1_sq,
        )
            
    @classmethod
    def load(cls, path: Path) -> ModelArtifact:
        """Load the artifact from a .npz file."""
        path = ARTIFACTS_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"Artifact file not found: {path}")
        with np.load(path) as data:
            present = set(data.files)
            missing = _REQUIRED_FIELDS - present
            if missing:
                raise ValueError(f"Artifact file is missing fields: {missing}")
            version = int(data["schema_version"])
            if version != _SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported schema version {version}; "
                    f"expected {_SCHEMA_VERSION}"
                )

            return cls(
                points=data["points"],
                normals=data["normals"],
                alpha=data["alpha"],
                h=float(data["h"]),
                sigma_f2=float(data["sigma_f2"]),
                sigma_0_sq=float(data["sigma_0_sq"]),
                sigma_1_sq=float(data["sigma_1_sq"]),
            )