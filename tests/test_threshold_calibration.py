from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import calibrate_threshold as calibration  # noqa: E402
from calibrate_threshold import (  # noqa: E402
    _build_summary,
    _compute_threshold_f1c,
    _save_threshold_json,
    _setup_params_paths,
    discover_annotated_test_samples,
    load_annotation_index,
    parse_args,
    reconstruct,
    resolve_test_annotation_paths,
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
            "scenario_description": "test scenario",
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
    assert metadata["source_scenarios"] == [{
        "scenario_id": "13_0", "description": "test scenario"
    }]


def test_unified_annotation_uses_scenario_id_from_csv_filename(tmp_path):
    annotation = tmp_path / "scenario_8_16_back_view_annotations.csv"
    pd.DataFrame([{
        "scenario_id": "unified",
        "camera": "back_view",
        "safety_area": "PRight",
        "filename": "s-8_0_s-PRight_f-000000.png",
        "label": "Anomalous",
        "source_scenario_id": "8_0",
    }]).to_csv(annotation, index=False)

    index, used = load_annotation_index(
        [annotation], "back_view", selected_scenarios={"8_16"}
    )

    assert index[("8_16", "PRight", "s-8_0_s-PRight_f-000000.png")] == "anomalous"
    assert used == [annotation.resolve()]


def test_test_discovery_accepts_normal_only_safety_area(tmp_path):
    test_root, annotation = make_annotated_test_tree(tmp_path)
    anomalous = test_root / "13_0" / "PRight" / "anomalous"
    next(anomalous.iterdir()).unlink()

    samples, metadata = discover_annotated_test_samples(
        test_root, "PRight", [annotation], requested_scenarios=["13_0"]
    )

    assert [label for _, label in samples] == [0]
    assert metadata["normal"] == 1
    assert metadata["anomalous"] == 0


def test_test_discovery_reports_anomalous_only_safety_area(tmp_path):
    test_root, annotation = make_annotated_test_tree(tmp_path)
    normal = test_root / "13_0" / "PRight" / "normal"
    next(normal.iterdir()).unlink()

    samples, metadata = discover_annotated_test_samples(
        test_root, "PRight", [annotation], requested_scenarios=["13_0"]
    )

    assert [label for _, label in samples] == [1]
    assert metadata["normal"] == 0
    assert metadata["anomalous"] == 1


def test_f1c_selects_an_observed_score_threshold():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert _compute_threshold_f1c(labels, scores) == pytest.approx(0.8)


def test_test_scenario_auto_resolves_matching_annotation_csv(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    selected = annotations / "scenario_13_1_back_view_annotations.csv"
    unrelated = annotations / "scenario_13_0_back_view_annotations.csv"
    selected.touch()
    unrelated.touch()

    paths = resolve_test_annotation_paths(
        gt_csv=None,
        annotations_dir=annotations,
        test_scenarios=["13_1"],
        camera="back_view",
        project_root=tmp_path,
    )

    assert paths == [selected.resolve()]


def test_singular_test_scenario_cli_alias():
    config = Path(__file__).resolve().parents[1] / "configs" / "cf_dataset_mac.yaml"
    args = parse_args([
        "--config", str(config),
        "--test_scenario", "13_1",
    ])
    assert args.test_scenarios == ["13_1"]
    assert args.test_folder == "/Users/rashid/data/DS/SR/v6/Jul27/test"


def test_test_scenario_reports_expected_annotation_filename(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()

    with pytest.raises(
        FileNotFoundError,
        match="scenario_13_1_back_view_annotations.csv",
    ):
        resolve_test_annotation_paths(
            gt_csv=None,
            annotations_dir=annotations,
            test_scenarios=["13_1"],
            camera="back_view",
            project_root=tmp_path,
        )


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
        mode="test",
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
    assert path.name == "threshold_test_PRight_f1c_off3_sig1.5_q0.98.json"


def test_test_threshold_writes_scenario_archive_and_active_copy(tmp_path):
    args = SimpleNamespace(
        mode="test",
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
    calibration_data = {
        "scenarios": ["8_16"],
        "source_scenarios": [
            {"scenario_id": "8_0", "description": "person entering"},
            {"scenario_id": "9_0", "description": "operator near robot"},
        ],
    }
    summary = _build_summary(
        "PRight", "PRight_64", 12, args, 0.5, scores,
        "scores.csv", "test", calibration_data=calibration_data,
    )

    archived = Path(_save_threshold_json(tmp_path, "PRight", summary, args))
    active = (
        tmp_path / "PRight"
        / "threshold_test_PRight_f1c_off1_sig1.0_q0.99.json"
    )

    assert archived.name == (
        "threshold_PRight_f1c_scenario-8_16_off1_sig1.0_q0.99.json"
    )
    assert active.is_file()
    assert summary["calibration_scenarios"] == ["8_16"]
    assert summary["calibration_source_scenarios"] == calibration_data["source_scenarios"]


def test_normal_only_test_calibration_uses_distribution_threshold(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        calibration,
        "_compute_scores_for_loader",
        lambda *unused: (
            [0.1, 0.2, 0.3],
            [0, 0, 0],
            ["a.png", "b.png", "c.png"],
        ),
    )
    args = SimpleNamespace(
        mode="test",
        offset=1,
        sigma=1.0,
        quantile=0.99,
        taas_variant="canonical",
        taas_backend="numpy",
        threshold_strategy="percentile",
        threshold_percentile=90.0,
        threshold_n_sigma=3.0,
        threshold_method="f1c",
    )
    metadata = {
        "scenarios": ["13_0"],
        "source_scenarios": [{
            "scenario_id": "13_0", "description": "normal-only area"
        }],
        "normal": 3,
        "anomalous": 0,
        "verify_excluded": 0,
    }
    area_out = tmp_path / "PRight"
    area_out.mkdir()

    summary = calibration._run_normal_only_test_calibration(
        "PRight", args, torch.device("cpu"), None, None, metadata,
        None, None, "PRight_64", 12, str(area_out), str(tmp_path),
        "_scenario-13_0",
    )

    assert summary["threshold"] == pytest.approx(0.28)
    assert summary["threshold_strategy"] == "percentile"
    assert summary["calibration_mode"] == "normal_only_fallback"
    assert summary["supervised_metrics_available"] is False
    assert summary["n_anomalous"] == 0
    metrics = pd.read_csv(
        area_out / "anomaly_metrics_PRight_scenario-13_0.csv"
    )
    assert metrics.loc[0, "Status"] == "not_applicable_no_anomalous_samples"
    artifact_dir = (
        area_out / "calibration_plots" / "scenario-13_0"
    )
    assert list((artifact_dir / "plots").glob("*.png"))
    combination_jsons = list((artifact_dir / "json").glob("*.json"))
    assert len(combination_jsons) == 1
    payload = json.loads(combination_jsons[0].read_text(encoding="utf-8"))
    assert payload["is_best"] is True
    assert payload["calibration_mode"] == "normal_only_fallback"
    assert payload["data"]["normal"] == 3
    assert payload["data"]["anomalous"] == 0


def test_combination_results_save_every_candidate_and_mark_winner(tmp_path):
    records = [
        {
            "method": "TAAS_OFF1-s_1.0-q_0.99",
            "safety_area": "PLeft",
            "metrics": {"binormal_auc": 0.72, "recall": 0.80},
            "plot_file": "../plots/first.png",
            "is_best": False,
            "rank": None,
        },
        {
            "method": "TAAS_OFF3-s_1.5-q_0.99",
            "safety_area": "PLeft",
            "metrics": {"binormal_auc": 0.91, "recall": 0.75},
            "plot_file": "../plots/second.png",
            "is_best": False,
            "rank": None,
        },
    ]

    paths = calibration._save_test_combination_results(
        records,
        str(tmp_path / "calibration_plots"),
        "_scenario-13_1",
        "TAAS_OFF3-s_1.5-q_0.99",
        "binormal_auc",
    )

    assert len(paths) == 2
    assert all(
        Path(path).parent
        == tmp_path / "calibration_plots" / "scenario-13_1" / "json"
        for path in paths
    )
    saved = {
        payload["method"]: payload
        for payload in (
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in paths
        )
    }
    assert saved["TAAS_OFF3-s_1.5-q_0.99"]["is_best"] is True
    assert saved["TAAS_OFF3-s_1.5-q_0.99"]["rank"] == 1
    assert saved["TAAS_OFF1-s_1.0-q_0.99"]["is_best"] is False
    assert saved["TAAS_OFF1-s_1.0-q_0.99"]["rank"] == 2


def test_anomalous_only_area_is_skipped_without_overwriting_threshold(tmp_path):
    area_out = tmp_path / "PLeft"
    area_out.mkdir()
    existing_threshold = area_out / "threshold_PLeft_f1c_off1_sig1.0_q0.99.json"
    existing_threshold.write_text('{"threshold": 0.42}', encoding="utf-8")
    metadata = {
        "scenarios": ["13_1"],
        "source_scenarios": [{
            "scenario_id": "13_1", "description": "operator falls"
        }],
        "annotation_csvs": ["scenario_13_1_back_view_annotations.csv"],
        "test_root": "/data/test",
        "normal": 0,
        "anomalous": 215,
        "verify_excluded": 401,
    }

    summary = calibration._skip_test_calibration_without_normal(
        "PLeft", metadata, str(area_out), "_scenario-13_1"
    )

    assert summary["status"] == "skipped"
    assert summary["threshold"] is None
    assert summary["threshold_written"] is False
    assert existing_threshold.read_text(encoding="utf-8") == '{"threshold": 0.42}'
    assert (
        area_out / "calibration_skipped_PLeft_scenario-13_1_no-normal.json"
    ).is_file()
