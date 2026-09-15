from pathlib import Path
import sys

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from test_model_inference import (  # noqa: E402
    image_metrics,
    load_reconstruction_models,
    reconstruct_batch,
)


class IdentityEncoder(torch.nn.Module):
    def forward(self, inputs):
        return inputs, torch.zeros_like(inputs)


class ScaledDecoder(torch.nn.Module):
    def forward(self, latent):
        return latent * 0.5


def test_reconstruction_uses_encoder_mean_and_preserves_shape():
    inputs = torch.ones(2, 3, 16, 16)

    outputs = reconstruct_batch(
        IdentityEncoder(), ScaledDecoder(), inputs, torch.device("cpu")
    )

    assert outputs.shape == inputs.shape
    assert torch.allclose(outputs, torch.full_like(inputs, 0.5))
    assert torch.isfinite(outputs).all()


def test_identical_images_have_perfect_metrics():
    image = np.full((16, 16, 3), 0.5, dtype=np.float32)

    metrics = image_metrics(image, image.copy())

    assert metrics["l1"] == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["psnr_db"] == float("inf")
    assert abs(metrics["dssim"]) < 1e-7


def test_changed_image_has_nonzero_reconstruction_error():
    original = np.zeros((16, 16, 3), dtype=np.float32)
    reconstruction = np.ones((16, 16, 3), dtype=np.float32)

    metrics = image_metrics(original, reconstruction)

    assert metrics["l1"] == 1.0
    assert metrics["mse"] == 1.0
    assert metrics["psnr_db"] == 0.0
    assert metrics["dssim"] > 0.0


def test_project_checkpoint_loads_strictly_and_reconstructs():
    """Integration smoke test; skipped in checkouts without trained weights."""
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "results/V6/train/models/model_PLeft_64.pt"
    )
    if not checkpoint.is_file():
        pytest.skip("project PLeft checkpoint is not available")

    encoder, decoder, _ = load_reconstruction_models(
        checkpoint, latent_dim=64, device=torch.device("cpu")
    )
    output = reconstruct_batch(
        encoder, decoder, torch.zeros(1, 3, 128, 128), torch.device("cpu")
    )

    assert output.shape == (1, 3, 128, 128)
    assert torch.isfinite(output).all()
