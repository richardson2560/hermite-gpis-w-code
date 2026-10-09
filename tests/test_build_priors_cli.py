"""Integration tests for the build_priors CLI.

These tests verify the end-to-end pipeline:
    clean .npz  ->  build_prior  ->  artifact .npz

They also verify argument handling and error paths.

Run with:
    pytest tests/test_build_priors_cli.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from hgw.models.artifact import ARTIFACTS_DIR, ModelArtifact

# Import the CLI module by file path
_CLI_PATH = Path(__file__).resolve().parent.parent / "scripts" / "build_priors.py"


def _import_cli():
    """Import the CLI module from its file path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_priors_cli", _CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_clean_npz(path: Path, n: int = 100, seed: int = 0) -> None:
    """Write a synthetic clean .npz for testing."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    points = 0.3 * directions
    np.savez_compressed(path, points=points, normals=directions)


# ---------------------------------------------------------------------------
# Group 1 — load_clean_npz
# ---------------------------------------------------------------------------

def test_load_clean_npz_ok(tmp_path: Path) -> None:
    cli = _import_cli()
    clean = tmp_path / "cloud.npz"
    _write_clean_npz(clean, n=50)

    points, normals = cli.load_clean_npz(clean)
    assert points.shape == (50, 3)
    assert normals.shape == (50, 3)
    assert points.dtype == np.float64


def test_load_clean_npz_missing_file(tmp_path: Path) -> None:
    cli = _import_cli()
    with pytest.raises(FileNotFoundError):
        cli.load_clean_npz(tmp_path / "does_not_exist.npz")


def test_load_clean_npz_missing_fields(tmp_path: Path) -> None:
    cli = _import_cli()
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, points=np.zeros((10, 3)))
    with pytest.raises(ValueError, match="must contain"):
        cli.load_clean_npz(bad)


def test_load_clean_npz_mismatched_shapes(tmp_path: Path) -> None:
    cli = _import_cli()
    bad = tmp_path / "bad.npz"
    np.savez_compressed(
        bad,
        points=np.zeros((10, 3)),
        normals=np.zeros((9, 3)),
    )
    with pytest.raises(ValueError, match="shape"):
        cli.load_clean_npz(bad)


# ---------------------------------------------------------------------------
# Group 2 — End-to-end via main() with monkey-patched sys.argv
# ---------------------------------------------------------------------------

def _run_cli(argv: list[str]) -> int:
    """Run the CLI's main() with a custom argv."""
    cli = _import_cli()
    original_argv = sys.argv
    sys.argv = ["build_priors.py"] + argv
    try:
        return cli.main()
    finally:
        sys.argv = original_argv


def test_end_to_end_builds_and_saves(tmp_path: Path) -> None:
    clean = tmp_path / "cloud.npz"
    _write_clean_npz(clean, n=100)

    output_name = f"test_cli_{tmp_path.name}"

    rc = _run_cli([
        str(clean), output_name,
        "--m", "20",
        "--h", "0.1",
    ])
    assert rc == 0

    artifact_path = ARTIFACTS_DIR / f"{output_name}.npz"
    assert artifact_path.exists()

    # Load it back to verify it is well-formed
    art = ModelArtifact.load(f"{output_name}.npz")
    assert art.n_primitives == 20
    assert art.h == 0.1

    # Cleanup
    artifact_path.unlink()


def test_end_to_end_respects_m_flag(tmp_path: Path) -> None:
    clean = tmp_path / "cloud.npz"
    _write_clean_npz(clean, n=200)

    output_name = f"test_cli_m_{tmp_path.name}"
    _run_cli([str(clean), output_name, "--m", "15"])

    art = ModelArtifact.load(f"{output_name}.npz")
    assert art.n_primitives == 15

    (ARTIFACTS_DIR / f"{output_name}.npz").unlink()


def test_end_to_end_respects_hyperparameters(tmp_path: Path) -> None:
    clean = tmp_path / "cloud.npz"
    _write_clean_npz(clean, n=100)

    output_name = f"test_cli_hp_{tmp_path.name}"
    _run_cli([
        str(clean), output_name,
        "--m", "10",
        "--h", "0.08",
        "--sigma-f2", "2.0",
        "--sigma-0-sq", "1e-3",
        "--sigma-1-sq", "5e-2",
    ])

    art = ModelArtifact.load(f"{output_name}.npz")
    assert art.h == pytest.approx(0.08, rel=1e-6)
    assert art.sigma_f2 == pytest.approx(2.0, rel=1e-6)
    assert art.sigma_0_sq == pytest.approx(1e-3, rel=1e-6)
    assert art.sigma_1_sq == pytest.approx(5e-2, rel=1e-6)

    (ARTIFACTS_DIR / f"{output_name}.npz").unlink()


def test_end_to_end_adds_npz_suffix(tmp_path: Path) -> None:
    """The CLI must accept an output name without .npz and add it."""
    clean = tmp_path / "cloud.npz"
    _write_clean_npz(clean, n=50)

    output_name = f"test_cli_suffix_{tmp_path.name}"
    _run_cli([str(clean), output_name, "--m", "5"])

    # File must have been saved with .npz suffix
    assert (ARTIFACTS_DIR / f"{output_name}.npz").exists()

    (ARTIFACTS_DIR / f"{output_name}.npz").unlink()


def test_end_to_end_fails_on_missing_input(tmp_path: Path) -> None:
    """Missing input file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        _run_cli([str(tmp_path / "missing.npz"), "test_fail"])