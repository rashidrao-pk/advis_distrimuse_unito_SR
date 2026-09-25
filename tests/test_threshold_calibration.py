from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from calibrate_threshold import (  # noqa: E402
    _build_summary,
    _compute_threshold_f1c,
    _save_threshold_json,
    _setup_params_paths,
    discover_annotated_test_samples,
    reconstruct,
)


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


def make_annotated_test_tree(tmp_path):
    test_root = tmp_path / "test"
    rows = []
    for frame_id, label in enumerate(("normal", "anomalous", "verify")):
        filename = f"s-13_0_s-PRight_f-{frame_id:06d}.png"
        folder = test_root / "13_0" / "PRight" / label
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (128, 128), color=(frame_id, 0, 0)).save(folder / filename)
        rows.append({
            "scenario_id": "13_0",
            "camera": "back_view",
            "safety_area": "PRight",
            "filename": filename,
            "label": label.title(),
        })
    annotation = tmp_path / "scenario_13_0_back_view_annotations.csv"
    pd.DataFrame(rows).to_csv(annotation, index=False)
    return test_root, annotation


def test_current_annotation_layout_supplies_binary_test_labels(tmp_path):
    test_root, annotation = make_annotated_test_tree(tmp_path)

    samples, metadata = discover_annotated_test_samples(
        test_root, "PRight", [annotation], requested_scenarios=["13_0"]
    )

    assert [label for _, label in samples] == [0, 1]
    assert metadata["normal"] == 1
    assert metadata["anomalous"] == 1
    assert metadata["verify_excluded"] == 1
    assert metadata["scenarios"] == ["13_0"]


def test_supervised_calibration_rejects_an_area_with_one_class(tmp_path):
    test_root, annotation = make_annotated_test_tree(tmp_path)
    anomalous = test_root / "13_0" / "PRight" / "anomalous"
    next(anomalous.iterdir()).unlink()

    with pytest.raises(ValueError, match="requires both normal and anomalous"):
        discover_annotated_test_samples(
            test_root, "PRight", [annotation], requested_scenarios=["13_0"]
        )


def test_f1c_selects_an_observed_score_threshold():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert _compute_threshold_f1c(labels, scores) == pytest.approx(0.8)


def test_config_paths_work_when_legacy_host_lookup_has_no_dataset_root(tmp_path):
    dataset_base = tmp_path / "V6"
    training_dir = dataset_base / "train"
    testing_dir = dataset_base / "test"
    masks_dir = dataset_base / "masks"
    args = SimpleNamespace(
        dataset_base=dataset_base,
        training_dir=str(training_dir),
        testing_dir=str(testing_dir),
        masks_dir=str(masks_dir),
        latent_dims=64,
        exp_type="E3",
        batch_size=8,
        dataset_version="V6",
        dataset_type=None,
        mask_image_name=3015,
        checkpoints=str(tmp_path / "models"),
    )

    _, paths = _setup_params_paths("PRight", args)

    assert paths.path_datasets == str(dataset_base.resolve())
    assert paths.train_dir_processed_subgroup == str(
        (training_dir / "PRight").resolve()
    )
    assert paths.test_dir == str(testing_dir.resolve())
    assert paths.mask_dir == str(masks_dir.resolve())


def test_test_summary_and_filename_use_winning_taas_parameters(tmp_path):
    args = SimpleNamespace(
        offset=1,
        sigma=1.0,
        quantile=0.99,
        taas_variant="canonical",
        taas_backend="numpy",
        threshold_strategy="percentile",
        threshold_percentile=99.0,
        threshold_method="f1c",
    )
    scores = pd.DataFrame({"anomaly_score": [0.1, 0.9]})
    summary = _build_summary(
        "PRight", "PRight_64", 12, args, 0.5, scores,
        "scores.csv", "test",
        score_parameters={
            "offset": 3,
            "sigma": 1.5,
            "quantile": 0.98,
            "taas_variant": "canonical",
        },
    )
    path = Path(_save_threshold_json(tmp_path, "PRight", summary, args))

    assert summary["threshold_strategy"] == "f1c"
    assert summary["threshold_percentile"] is None
    assert (summary["offset"], summary["sigma"], summary["quantile"]) == (3, 1.5, 0.98)
    assert summary["score_func"] == "TAAS_OFF3-s_1.5-q_0.98"
    assert path.name == "threshold_PRight_f1c_off3_sig1.5_q0.98.json"
