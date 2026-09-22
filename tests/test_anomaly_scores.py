"""Numerical and behavioral tests for reconstruction anomaly scores."""

from pathlib import Path
import sys

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from calibrate_threshold import score_batch, score_pair  # noqa: E402
from infer_offline import distance_offset, distance_offset_cython  # noqa: E402
from test_model_inference import image_metrics  # noqa: E402
from utils import ComputeDifferences, get_anomaly_score_ravi  # noqa: E402


def test_identical_images_have_zero_basic_and_taas_scores():
    image = np.full((16, 16, 3), 0.4, dtype=np.float32)

    metrics = image_metrics(image, image.copy())
    taas, distance = score_pair(image, image.copy(), offset=1, sigma=1.0, quantile=0.99)

    assert metrics["l1"] == 0.0
    assert metrics["mse"] == 0.0
    assert abs(metrics["dssim"]) < 1e-7
    assert taas == 0.0
    assert np.all(distance == 0.0)


def test_large_patch_increases_all_supported_scores():
    normal = np.zeros((32, 32, 3), dtype=np.float32)
    changed = normal.copy()
    changed[8:24, 8:24] = 1.0

    metrics = image_metrics(normal, changed)
    taas, _ = score_pair(normal, changed, offset=1, sigma=1.0, quantile=0.99)

    assert metrics["l1"] > 0
    assert metrics["mse"] > 0
    assert metrics["dssim"] > 0
    assert taas > 0


def test_higher_quantile_cannot_reduce_taas_score():
    normal = np.zeros((16, 16, 3), dtype=np.float32)
    changed = normal.copy()
    changed[3:7, 3:7] = 1.0

    score_95, _ = score_pair(normal, changed, offset=0, sigma=0, quantile=0.95)
    score_99, _ = score_pair(normal, changed, offset=0, sigma=0, quantile=0.99)

    assert score_99 >= score_95


def test_spatial_offset_tolerates_one_pixel_translation():
    original = np.zeros((16, 16, 3), dtype=np.float32)
    original[4:10, 4:10] = 1.0
    shifted = np.roll(original, shift=1, axis=1)

    no_tolerance, _ = score_pair(original, shifted, offset=0, sigma=0, quantile=0.99)
    tolerant, _ = score_pair(original, shifted, offset=1, sigma=0, quantile=0.99)

    assert tolerant < no_tolerance


def test_cython_taas_matches_numpy_when_extension_is_available():
    if distance_offset_cython is None:
        return
    rng = np.random.default_rng(42)
    original = rng.random((32, 32, 3), dtype=np.float32)
    reconstruction = rng.random((32, 32, 3), dtype=np.float32)

    numpy_distance = distance_offset(original, reconstruction, 3, "numpy")
    cython_distance = distance_offset(original, reconstruction, 3, "cython")

    assert np.allclose(cython_distance, numpy_distance, rtol=1e-6, atol=1e-7)


def test_taas_score_is_batch_invariant_and_order_preserving():
    first = torch.zeros(1, 3, 16, 16)
    first_reconstruction = first.clone()
    first_reconstruction[:, :, 4:8, 4:8] = 1.0
    second = torch.ones(1, 3, 16, 16)
    second_reconstruction = torch.zeros_like(second)

    alone = score_batch(first, first_reconstruction, offset=1, sigma=1, quantile=0.99)[0]
    batch = score_batch(
        torch.cat((second, first)), torch.cat((second_reconstruction, first_reconstruction)),
        offset=1, sigma=1, quantile=0.99,
    )

    assert np.isclose(alone, batch[1])


def test_l1_and_l2_are_computed_per_image():
    original = torch.zeros(2, 3, 8, 8)
    reconstruction = original.clone()
    reconstruction[1] = 1.0

    differences = ComputeDifferences(original, reconstruction)
    _, l1 = differences.compute("l1")
    _, l2 = differences.compute("l2")

    assert np.allclose(l1, [0.0, 1.0])
    assert np.isclose(l2[0], 0.0)
    assert l2[1] > 0.0


def test_inference_ravi_is_per_image_and_batch_invariant():
    original = torch.zeros(2, 3, 8, 8)
    reconstruction = original.clone()
    reconstruction[0, :, 0, 0] = 0.25
    reconstruction[1, :, 0, 0] = 1.0

    batch_scores = get_anomaly_score_ravi(original, reconstruction, quantile=1.0)
    alone_score = get_anomaly_score_ravi(
        original[:1], reconstruction[:1], quantile=1.0
    )[0]

    assert batch_scores.shape == (2,)
    assert np.isclose(batch_scores[0], alone_score)
    assert batch_scores[1] > batch_scores[0]
