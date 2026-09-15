"""Training, validation and inference preprocessing tests."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from test_model_inference import make_inference_transform  # noqa: E402
from train_new import get_transforms  # noqa: E402


def rgb_image(rgb, size=(128, 128)) -> Image.Image:
    array = np.empty((size[1], size[0], 3), dtype=np.uint8)
    array[:] = rgb
    return Image.fromarray(array, mode="RGB")


def test_validation_normalizes_zero_and_255_to_minus_one_and_one():
    _, validation = get_transforms("min")

    black = validation(rgb_image((0, 0, 0)))
    white = validation(rgb_image((255, 255, 255)))

    assert torch.allclose(black, torch.full_like(black, -1.0))
    assert torch.allclose(white, torch.full_like(white, 1.0))


def test_preprocessing_preserves_rgb_channel_order():
    _, validation = get_transforms("min")

    tensor = validation(rgb_image((255, 128, 0)))

    assert torch.allclose(tensor[0], torch.ones_like(tensor[0]))
    assert abs(float(tensor[1].mean()) - (128 / 255 * 2 - 1)) < 1e-6
    assert torch.allclose(tensor[2], -torch.ones_like(tensor[2]))


def test_validation_transform_is_deterministic():
    _, validation = get_transforms("custom")
    image = rgb_image((80, 120, 160))

    first = validation(image)
    second = validation(image)

    assert torch.equal(first, second)


def test_minimal_training_and_validation_preprocessing_are_identical():
    training, validation = get_transforms("min")
    image = rgb_image((23, 101, 211))

    assert torch.equal(training(image), validation(image))


def test_inference_matches_validation_for_model_sized_input():
    _, validation = get_transforms("min")
    inference = make_inference_transform((128, 128))
    image = rgb_image((12, 130, 240))

    assert torch.equal(inference(image), validation(image))


def test_custom_augmentation_keeps_shape_and_finite_range():
    training, _ = get_transforms("custom")

    tensor = training(rgb_image((80, 120, 160)))

    assert tensor.shape == (3, 128, 128)
    assert torch.isfinite(tensor).all()
    assert float(tensor.min()) >= -1.0
    assert float(tensor.max()) <= 1.0
