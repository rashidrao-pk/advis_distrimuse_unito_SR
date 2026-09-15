from pathlib import Path
import sys

import pandas as pd
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from check_model_checkpoints import inspect_checkpoint  # noqa: E402


def make_checkpoint(path: Path, *, history_epochs: int = 3) -> None:
    history = pd.DataFrame([{"recon_loss": 1.0 / (i + 1)} for i in range(history_epochs)])
    torch.save({
        "epoch": 10,
        "encoder_state_dict": {"weight": torch.ones(1)},
        "decoder_state_dict": {"weight": torch.ones(1)},
        "discriminator_state_dict": {"weight": torch.ones(1)},
        "optimizer_encdec_state_dict": {},
        "optimizer_d_state_dict": {},
        "loss_history": history,
        "config": {
            "suffix": "PLeft_64",
            "epochs_trained": history_epochs,
            "dataset": {
                "dataset_name": "normal_V6_back_view",
                "train_dir": "/dataset/train/PLeft",
                "n_train_images": 120,
                "n_val_images": 30,
            },
            "training": {"batch_size": 16, "device": "cpu"},
            "params": {"latent_dim": 64, "image_size": [128, 128]},
        },
    }, path)


def test_valid_checkpoint_reports_epochs_and_data_size(tmp_path):
    checkpoint = tmp_path / "model_PLeft_64.pt"
    make_checkpoint(checkpoint, history_epochs=3)

    report = inspect_checkpoint(checkpoint)

    assert report["valid"] is True
    assert report["completed_epochs"] == 3
    assert report["train_images"] == 120
    assert report["validation_images"] == 30
    assert report["training_image_exposures"] == 360
    assert report["latent_dim"] == 64
    assert report["optimizer_format"] == "new"
    assert report["warnings"] == []


def test_missing_model_state_fails_validation(tmp_path):
    checkpoint = tmp_path / "model_broken.pt"
    make_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    del payload["decoder_state_dict"]
    torch.save(payload, checkpoint)

    report = inspect_checkpoint(checkpoint)

    assert report["valid"] is False
    assert any("decoder_state_dict" in error for error in report["errors"])


def test_epoch_metadata_mismatch_is_reported(tmp_path):
    checkpoint = tmp_path / "model_mismatch.pt"
    make_checkpoint(checkpoint, history_epochs=2)
    payload = torch.load(checkpoint, weights_only=False)
    payload["config"]["epochs_trained"] = 8
    torch.save(payload, checkpoint)

    report = inspect_checkpoint(checkpoint)

    assert report["valid"] is True
    assert any("differs from history length" in warning for warning in report["warnings"])


def test_missing_checkpoint_returns_failure(tmp_path):
    report = inspect_checkpoint(tmp_path / "missing.pt")

    assert report["valid"] is False
    assert report["errors"] == ["checkpoint file does not exist"]
