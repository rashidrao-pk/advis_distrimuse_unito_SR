from pathlib import Path
import sys

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from calibrate_threshold import reconstruct  # noqa: E402


class EncoderWithNoisyPosterior(torch.nn.Module):
    def forward(self, inputs):
        mean = inputs + 0.25
        # A large variance would expose accidental stochastic sampling.
        log_variance = torch.full_like(inputs, 8.0)
        return mean, log_variance


class IdentityDecoder(torch.nn.Module):
    def forward(self, latent):
        return latent


def test_calibration_reconstruction_uses_posterior_mean_deterministically():
    inputs = torch.zeros(2, 3, 8, 8)
    encoder = EncoderWithNoisyPosterior()
    decoder = IdentityDecoder()

    first = reconstruct(encoder, decoder, inputs, torch.device("cpu"))
    second = reconstruct(encoder, decoder, inputs, torch.device("cpu"))

    expected = torch.full_like(inputs, 0.25)
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)
    assert torch.equal(first, second)
