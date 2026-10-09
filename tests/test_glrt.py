"""Unit tests for the GLRT multi-model selection.

Strategy: use deterministic synthetic ellipsoid models. A scene drawn
from ellipsoid A should:
    - pass only A's test when B is very different,
    - pass both A and A' (a clone of A) tests and be declared ambiguous.

Run with:
    pytest tests/test_glrt.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from hgw.models.builder import build_prior
from hgw.models.evaluator import HermiteGPIS_W
from hgw.stats.glrt import (
    GLRTConfig,
    ModelCandidate,
    glrt_select,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

H = 0.10
SIGMA_F2 = 1.0
SIGMA_0_SQ = 1e-4
SIGMA_1_SQ = 1e-2

IDENTITY = np.eye(4)


def _ellipsoid_points(a: float, b: float, c: float, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    points = dirs * np.array([a, b, c])
    normals = points / np.array([a * a, b * b, c * c])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


def _build_model(a: float, b: float, c: float, seed: int = 0) -> HermiteGPIS_W:
    points, normals = _ellipsoid_points(a, b, c, 400, seed)
    art = build_prior(points, normals, H, SIGMA_F2, SIGMA_0_SQ, SIGMA_1_SQ, 150)
    return HermiteGPIS_W(art)


@pytest.fixture(scope="module")
def model_small() -> HermiteGPIS_W:
    return _build_model(0.30, 0.20, 0.10, seed=0)


@pytest.fixture(scope="module")
def model_small_clone() -> HermiteGPIS_W:
    return _build_model(0.30, 0.20, 0.10, seed=0)


@pytest.fixture(scope="module")
def model_large() -> HermiteGPIS_W:
    return _build_model(0.60, 0.40, 0.20, seed=1)


# ---------------------------------------------------------------------------
# Group 1 — Single passing candidate
# ---------------------------------------------------------------------------

def test_selects_single_passing_model(model_small, model_large) -> None:
    """A scene from the small model should reject the large model."""
    scene, _ = _ellipsoid_points(0.30, 0.20, 0.10, 200, seed=2)

    candidates = [
        ModelCandidate("small", model_small, IDENTITY),
        ModelCandidate("large", model_large, IDENTITY),
    ]
    result = glrt_select(scene, candidates)

    assert result.status == "SELECTED"
    assert result.selected_name == "small"
    assert result.delta_score == float("inf") or result.delta_score > 0


# ---------------------------------------------------------------------------
# Group 2 — Ambiguity
# ---------------------------------------------------------------------------

def test_clone_models_are_ambiguous(model_small, model_small_clone) -> None:
    """Two identical models give identical scores; ambiguity is declared."""
    scene, _ = _ellipsoid_points(0.30, 0.20, 0.10, 200, seed=3)

    candidates = [
        ModelCandidate("A", model_small, IDENTITY),
        ModelCandidate("B", model_small_clone, IDENTITY),
    ]
    result = glrt_select(scene, candidates)

    assert result.status == "AMBIGUOUS"
    assert result.selected_name is None
    assert result.delta_score < result.threshold


def test_clone_models_have_identical_scores(model_small, model_small_clone) -> None:
    scene, _ = _ellipsoid_points(0.30, 0.20, 0.10, 200, seed=4)

    candidates = [
        ModelCandidate("A", model_small, IDENTITY),
        ModelCandidate("B", model_small_clone, IDENTITY),
    ]
    result = glrt_select(scene, candidates)

    assert len(result.per_model) == 2
    scores = [pm.score for pm in result.per_model]
    assert np.isclose(scores[0], scores[1], rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Group 3 — All rejected
# ---------------------------------------------------------------------------

def test_all_rejected_when_scene_far_away(model_small, model_large) -> None:
    """A scene displaced 5 m away from both models is rejected by both."""
    scene, _ = _ellipsoid_points(0.30, 0.20, 0.10, 200, seed=5)
    scene_far = scene + np.array([5.0, 5.0, 5.0])

    candidates = [
        ModelCandidate("small", model_small, IDENTITY),
        ModelCandidate("large", model_large, IDENTITY),
    ]
    result = glrt_select(scene_far, candidates)

    assert result.status == "REJECTED_ALL"
    assert result.selected_name is None
    for pm in result.per_model:
        assert not pm.passed


# ---------------------------------------------------------------------------
# Group 4 — Determinism and structure
# ---------------------------------------------------------------------------

def test_deterministic(model_small, model_large) -> None:
    scene, _ = _ellipsoid_points(0.30, 0.20, 0.10, 200, seed=6)
    candidates = [
        ModelCandidate("small", model_small, IDENTITY),
        ModelCandidate("large", model_large, IDENTITY),
    ]

    r1 = glrt_select(scene, candidates)
    r2 = glrt_select(scene, candidates)

    assert r1.status == r2.status
    assert r1.selected_name == r2.selected_name
    assert r1.delta_score == pytest.approx(r2.delta_score, abs=1e-12)


def test_per_model_result_present_for_each_candidate(model_small, model_large) -> None:
    scene, _ = _ellipsoid_points(0.30, 0.20, 0.10, 200, seed=7)
    candidates = [
        ModelCandidate("small", model_small, IDENTITY),
        ModelCandidate("large", model_large, IDENTITY),
    ]
    result = glrt_select(scene, candidates)

    names = {pm.name for pm in result.per_model}
    assert names == {"small", "large"}


# ---------------------------------------------------------------------------
# Group 5 — Input validation
# ---------------------------------------------------------------------------

def test_rejects_single_candidate(model_small) -> None:
    scene, _ = _ellipsoid_points(0.30, 0.20, 0.10, 100, seed=8)
    with pytest.raises(ValueError, match="at least 2"):
        glrt_select(scene, [ModelCandidate("only", model_small, IDENTITY)])


def test_rejects_empty_candidates() -> None:
    scene = np.zeros((50, 3))
    with pytest.raises(ValueError, match="at least 2"):
        glrt_select(scene, [])


def test_rejects_duplicate_names(model_small, model_small_clone) -> None:
    scene, _ = _ellipsoid_points(0.30, 0.20, 0.10, 100, seed=9)
    candidates = [
        ModelCandidate("dup", model_small, IDENTITY),
        ModelCandidate("dup", model_small_clone, IDENTITY),
    ]
    with pytest.raises(ValueError, match="unique"):
        glrt_select(scene, candidates)


def test_rejects_wrong_shape_scene(model_small, model_large) -> None:
    candidates = [
        ModelCandidate("a", model_small, IDENTITY),
        ModelCandidate("b", model_large, IDENTITY),
    ]
    with pytest.raises(ValueError, match="shape"):
        glrt_select(np.zeros((10, 2)), candidates)


def test_candidate_rejects_non_model() -> None:
    with pytest.raises(TypeError):
        ModelCandidate("x", "not a model", IDENTITY)


def test_candidate_rejects_bad_T(model_small) -> None:
    with pytest.raises(ValueError, match="T"):
        ModelCandidate("x", model_small, np.eye(3))


def test_config_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        GLRTConfig(delta_score_threshold=0.0)
    with pytest.raises(ValueError):
        GLRTConfig(delta_score_threshold=-1.0)