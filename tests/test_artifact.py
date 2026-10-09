"""Unit tests for the ModelArtifact DTO.

These tests verify:
    1. Construction and validation of valid artifacts.
    2. Rejection of malformed inputs.
    3. Round-trip save/load preserves all data exactly (up to float32).
    4. Schema-version mismatch is detected.
    5. Missing fields are detected.
    6. Size bounds are respected.

Run with:
    pytest tests/test_artifact.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hgw.models.artifact import ModelArtifact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_artifact(M: int = 5) -> ModelArtifact:
    """Build a small valid artifact for testing."""
    rng = np.random.default_rng(42)

    # Random points in a small box
    points = rng.uniform(-0.1, 0.1, size=(M, 3)).astype(np.float32)

    # Random unit normals
    normals = rng.normal(size=(M, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals.astype(np.float32)

    # Random coefficients
    alpha = rng.normal(size=(2 * M,)).astype(np.float32)

    return ModelArtifact(
        points=points,
        normals=normals,
        alpha=alpha,
        h=0.05,
        sigma_f2=1.0,
        sigma_0_sq=1e-4,
        sigma_1_sq=1e-2,
    )


# ---------------------------------------------------------------------------
# Group 1 — Construction and validation
# ---------------------------------------------------------------------------

def test_valid_construction() -> None:
    """A well-formed artifact must construct without error."""
    art = _valid_artifact(M=5)
    assert art.n_primitives == 5
    assert art.points.shape == (5, 3)
    assert art.normals.shape == (5, 3)
    assert art.alpha.shape == (10,)
    assert art.h == 0.05


def test_rejects_non_2d_points() -> None:
    """Points must be (M, 3)."""
    with pytest.raises(ValueError):
        ModelArtifact(
            points=np.zeros((5, 2), dtype=np.float32),
            normals=np.zeros((5, 3), dtype=np.float32),
            alpha=np.zeros(10, dtype=np.float32),
            h=0.05, sigma_f2=1.0, sigma_0_sq=1e-4, sigma_1_sq=1e-2,
        )


def test_rejects_empty_points() -> None:
    """M must be at least 1."""
    with pytest.raises(ValueError):
        ModelArtifact(
            points=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            alpha=np.zeros(0, dtype=np.float32),
            h=0.05, sigma_f2=1.0, sigma_0_sq=1e-4, sigma_1_sq=1e-2,
        )


def test_rejects_shape_mismatch() -> None:
    """Normals and points must have the same shape."""
    with pytest.raises(ValueError):
        ModelArtifact(
            points=np.zeros((5, 3), dtype=np.float32),
            normals=np.zeros((4, 3), dtype=np.float32),
            alpha=np.zeros(10, dtype=np.float32),
            h=0.05, sigma_f2=1.0, sigma_0_sq=1e-4, sigma_1_sq=1e-2,
        )


def test_rejects_non_unit_normals() -> None:
    """Normals must be unit length."""
    art = _valid_artifact(M=3)
    bad_normals = art.normals * 2.0  # not unit anymore
    with pytest.raises(ValueError):
        ModelArtifact(
            points=art.points,
            normals=bad_normals,
            alpha=art.alpha,
            h=0.05, sigma_f2=1.0, sigma_0_sq=1e-4, sigma_1_sq=1e-2,
        )


def test_rejects_wrong_alpha_length() -> None:
    """Alpha must have length 2M."""
    art = _valid_artifact(M=5)
    with pytest.raises(ValueError):
        ModelArtifact(
            points=art.points,
            normals=art.normals,
            alpha=art.alpha[:8],   # too short
            h=0.05, sigma_f2=1.0, sigma_0_sq=1e-4, sigma_1_sq=1e-2,
        )


def test_rejects_nonfinite_arrays() -> None:
    """Non-finite values must raise."""
    art = _valid_artifact(M=3)
    bad_points = art.points.copy()
    bad_points[0, 0] = np.nan
    with pytest.raises(ValueError):
        ModelArtifact(
            points=bad_points,
            normals=art.normals,
            alpha=art.alpha,
            h=0.05, sigma_f2=1.0, sigma_0_sq=1e-4, sigma_1_sq=1e-2,
        )


@pytest.mark.parametrize("bad_field", ["h", "sigma_f2", "sigma_0_sq", "sigma_1_sq"])
def test_rejects_nonpositive_scalars(bad_field: str) -> None:
    """All scalar hyperparameters must be positive."""
    art = _valid_artifact(M=3)
    kwargs = {
        "points": art.points,
        "normals": art.normals,
        "alpha": art.alpha,
        "h": 0.05,
        "sigma_f2": 1.0,
        "sigma_0_sq": 1e-4,
        "sigma_1_sq": 1e-2,
    }
    kwargs[bad_field] = 0.0
    with pytest.raises(ValueError):
        ModelArtifact(**kwargs)


# ---------------------------------------------------------------------------
# Group 2 — Round-trip save/load
# ---------------------------------------------------------------------------

def test_roundtrip_preserves_data(tmp_path: Path) -> None:
    """Saving and loading preserves all arrays exactly (up to float32)."""
    art = _valid_artifact(M=10)
    out = tmp_path / "art.npz"
    art.save(out)
    loaded = ModelArtifact.load(out)

    assert np.array_equal(loaded.points, art.points)
    assert np.array_equal(loaded.normals, art.normals)
    assert np.array_equal(loaded.alpha, art.alpha)
    assert loaded.h == pytest.approx(art.h, rel=1e-6)
    assert loaded.sigma_f2 == pytest.approx(art.sigma_f2, rel=1e-6)
    assert loaded.sigma_0_sq == pytest.approx(art.sigma_0_sq, rel=1e-6)
    assert loaded.sigma_1_sq == pytest.approx(art.sigma_1_sq, rel=1e-6)


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    """Save must create missing parent directories."""
    art = _valid_artifact(M=3)
    nested = tmp_path / "a" / "b" / "c" / "art.npz"
    art.save(nested)
    assert nested.exists()


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """Loading a missing file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ModelArtifact.load(tmp_path / "does_not_exist.npz")


def test_load_missing_field_raises(tmp_path: Path) -> None:
    """Loading an npz missing a required field must raise."""
    out = tmp_path / "bad.npz"
    np.savez_compressed(
        out,
        points=np.zeros((3, 3), dtype=np.float32),
        normals=np.zeros((3, 3), dtype=np.float32),
        # alpha deliberately missing
        h=np.float32(0.05),
        sigma_f2=np.float32(1.0),
        sigma_0_sq=np.float32(1e-4),
        sigma_1_sq=np.float32(1e-2),
    )
    with pytest.raises(ValueError, match="missing fields"):
        ModelArtifact.load(out)


def test_load_wrong_schema_version_raises(tmp_path: Path) -> None:
    """Loading an artifact with a different schema version must raise."""
    out = tmp_path / "wrong_version.npz"
    np.savez_compressed(
        out,
        schema_version=np.int32(999),
        points=np.zeros((3, 3), dtype=np.float32),
        normals=np.zeros((3, 3), dtype=np.float32),
        alpha=np.zeros(6, dtype=np.float32),
        h=np.float32(0.05),
        sigma_f2=np.float32(1.0),
        sigma_0_sq=np.float32(1e-4),
        sigma_1_sq=np.float32(1e-2),
    )
    with pytest.raises(ValueError, match="schema version"):
        ModelArtifact.load(out)


# ---------------------------------------------------------------------------
# Group 3 — Size and metadata
# ---------------------------------------------------------------------------

def test_size_bytes_scales_with_M() -> None:
    """size_bytes must grow linearly with M."""
    art_small = _valid_artifact(M=10)
    art_large = _valid_artifact(M=100)
    assert art_large.size_bytes() > art_small.size_bytes()
    # Raw bytes for M primitives: 3*4 + 3*4 + 2*4 = 32 bytes per primitive
    # plus 16 bytes for scalars
    expected_small = 10 * 32 + 16
    assert art_small.size_bytes() == expected_small


def test_file_size_under_budget(tmp_path: Path) -> None:
    """A 200-primitive artifact must be well under 50 KB compressed."""
    art = _valid_artifact(M=200)
    out = tmp_path / "art.npz"
    art.save(out)
    size_kb = out.stat().st_size / 1024
    assert size_kb < 50, f"artifact too large: {size_kb:.1f} KB"
    print(f"\nM=200 artifact size: {size_kb:.1f} KB")


def test_immutability() -> None:
    """The frozen dataclass must reject attribute mutation."""
    art = _valid_artifact(M=3)
    with pytest.raises((AttributeError, Exception)):
        art.h = 0.1   # type: ignore