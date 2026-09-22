"""Correctness tests for the optional compiled TAAS offset backend.

Build before running directly:
    python scripts/setup_taas_cython.py build_ext --inplace
"""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from infer_offline import (  # noqa: E402
    distance_offset,
    distance_offset_cython,
    minimization_offset_cython,
    resolve_taas_backend,
)


pytestmark = pytest.mark.skipif(
    distance_offset_cython is None,
    reason=(
        "Cython TAAS extension is not built; run "
        "`python scripts/setup_taas_cython.py build_ext --inplace`"
    ),
)


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_cython_distance_matches_numpy_for_each_supported_offset(offset):
    rng = np.random.default_rng(100 + offset)
    original = rng.random((37, 29, 3), dtype=np.float32)
    reconstruction = rng.random((37, 29, 3), dtype=np.float32)

    expected = distance_offset(original, reconstruction, offset, "numpy")
    actual = distance_offset(original, reconstruction, offset, "cython")

    assert actual.dtype == np.float32
    assert actual.shape == original.shape[:2]
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_cython_matches_numpy_at_image_boundaries():
    original = np.zeros((9, 11, 3), dtype=np.float32)
    reconstruction = original.copy()
    reconstruction[0, 0] = (1.0, 0.5, 0.25)
    reconstruction[-1, -1] = (0.2, 0.4, 0.8)

    expected = distance_offset(original, reconstruction, 3, "numpy")
    actual = distance_offset(original, reconstruction, 3, "cython")

    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_cython_produces_same_taas_score_and_threshold_decision():
    rng = np.random.default_rng(2026)
    original = rng.random((64, 64, 3), dtype=np.float32)
    reconstruction = rng.random((64, 64, 3), dtype=np.float32)

    numpy_map = distance_offset(original, reconstruction, 3, "numpy")
    cython_map = distance_offset(original, reconstruction, 3, "cython")
    numpy_score = float(np.quantile(gaussian_filter(numpy_map, 1.5), 0.99))
    cython_score = float(np.quantile(gaussian_filter(cython_map, 1.5), 0.99))
    threshold = numpy_score * 0.9

    assert np.isclose(cython_score, numpy_score, rtol=1e-6, atol=1e-7)
    # A realistic threshold must not expose more than float32-level drift.
    assert abs(cython_score - numpy_score) < 1e-6
    assert (cython_score > threshold) == (numpy_score > threshold)


def test_explicit_cython_backend_is_resolved():
    assert resolve_taas_backend("cython") == "cython"


def test_compiled_kernel_rejects_different_image_shapes():
    first = np.zeros((8, 8, 3), dtype=np.float32)
    second = np.zeros((7, 8, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="same shape|shapes differ"):
        distance_offset_cython(first, second, 1)


def test_compiled_kernel_rejects_negative_offset():
    image = np.zeros((8, 8, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="non-negative"):
        distance_offset_cython(image, image, -1)


@pytest.mark.skipif(
    minimization_offset_cython is None,
    reason="The enhanced tass_cython_distance extension is not built",
)
@pytest.mark.parametrize("offset", [0, 1, 3])
def test_compiled_minimum_filter_matches_python_reference(offset):
    rng = np.random.default_rng(300 + offset)
    distance = rng.random((19, 23), dtype=np.float32)
    expected = np.empty_like(distance)
    for row in range(distance.shape[0]):
        for col in range(distance.shape[1]):
            expected[row, col] = np.min(distance[
                max(0, row - offset):row + offset + 1,
                max(0, col - offset):col + offset + 1,
            ])

    actual = minimization_offset_cython(distance, offset)

    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)


@pytest.mark.skipif(
    minimization_offset_cython is None,
    reason="The enhanced tass_cython_distance extension is not built",
)
def test_compiled_minimum_filter_rejects_negative_offset():
    distance = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="non-negative"):
        minimization_offset_cython(distance, -1)
