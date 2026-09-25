from pathlib import Path
import sys
from collections import deque

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from infer_offline import (  # noqa: E402
    amplify_threshold_config,
    apply_rolling_policy,
    calibration_mode_variant_tag,
    camera_name_from_topic,
    compact_sample_name,
    load_threshold,
    load_settings,
    normalize_model_input,
    parse_args,
    public_result,
    resolve_taas_backend,
    resolve_threshold_amplifications,
    resolve_video_scenario,
    rolling_variant_tag,
    taas_variant_tag,
    threshold_variant_tag,
    threshold_amplification_variant_tag,
)


def parse(monkeypatch, *extra):
    monkeypatch.setattr(
        sys, "argv", ["infer_offline.py", "--input_type", "frames", "--input", "/tmp"]
        + list(extra),
    )
    return parse_args()


def test_csv_and_video_are_default(monkeypatch):
    assert parse(monkeypatch).save_video is True


def test_inference_normalization_does_not_require_torchvision():
    tensor = torch.tensor([0.0, 0.5, 1.0])
    assert torch.equal(
        normalize_model_input(tensor), torch.tensor([-1.0, 0.0, 1.0])
    )


def test_scores_only_disables_video_outputs(monkeypatch):
    assert parse(monkeypatch, "--scores-only").save_video is False


def test_save_video_switch_enables_video_outputs(monkeypatch):
    assert parse(monkeypatch, "--save-video").save_video is True


def test_optional_dashboard_details_are_disabled_by_default(monkeypatch):
    args = parse(monkeypatch)
    assert args.add_score_name is False
    assert args.add_fps_details is False


def test_optional_dashboard_details_can_be_enabled(monkeypatch):
    args = parse(monkeypatch, "--add_score_name", "--add_fps_details")
    assert args.add_score_name is True
    assert args.add_fps_details is True


def test_timing_profile_can_be_enabled(monkeypatch):
    assert parse(monkeypatch, "--profile-timing").profile_timing is True


def test_taas_backend_defaults_to_auto(monkeypatch):
    assert parse(monkeypatch).taas_backend == "auto"


def test_taas_variant_is_selectable(monkeypatch):
    assert parse(monkeypatch).taas_variant == "canonical"
    assert (
        parse(monkeypatch, "--taas_variant", "minimization").taas_variant
        == "minimization"
    )


def test_numpy_taas_backend_is_always_available():
    assert resolve_taas_backend("numpy") == "numpy"


def test_rolling_policy_is_disabled_by_default(monkeypatch):
    args = parse(monkeypatch)
    assert args.rolling == "none"
    assert args.rolling_window == 5


def test_rolling_policy_and_window_can_be_selected(monkeypatch):
    args = parse(monkeypatch, "--rolling", "mean", "--rolling_window", "7")
    assert args.rolling == "mean"
    assert args.rolling_window == 7


def test_rolling_policy_accepts_case_insensitive_none(monkeypatch):
    assert parse(monkeypatch, "--rolling", "None").rolling == "none"


def test_rolling_window_must_be_positive(monkeypatch):
    with pytest.raises(SystemExit):
        parse(monkeypatch, "--rolling_window", "0")


def test_one_threshold_amplification_is_broadcast(monkeypatch):
    args = parse(monkeypatch, "--threshold_amplification", "1.1")
    assert list(args.threshold_amplification_by_area.items()) == [
        ("PLeft", 1.1), ("PRight", 1.1),
        ("RoboArm", 1.1), ("ConvBelt", 1.1),
    ]


def test_threshold_amplification_list_follows_selected_area_order(monkeypatch):
    args = parse(
        monkeypatch,
        "--safety_areas", "PRight", "PLeft", "RoboArm", "ConvBelt",
        "--threshold_amplification", "1.1", "1.2", "1.3", "1.4",
    )
    assert list(args.threshold_amplification_by_area.items()) == [
        ("PRight", 1.1), ("PLeft", 1.2),
        ("RoboArm", 1.3), ("ConvBelt", 1.4),
    ]


def test_threshold_amplification_count_and_values_are_validated(monkeypatch):
    with pytest.raises(SystemExit):
        parse(monkeypatch, "--threshold_amplification", "1.1", "1.2")
    with pytest.raises(SystemExit):
        parse(monkeypatch, "--threshold_amplification", "0")


def test_calibration_mode_variant_tag_supports_single_and_mixed_sources():
    assert calibration_mode_variant_tag(["test"]) == "_cal-test"
    assert calibration_mode_variant_tag(["val"]) == "_cal-val"
    assert (
        calibration_mode_variant_tag({"PLeft": "test", "PRight": "val"})
        == "_cal-test-val"
    )


def test_default_inference_outputs_include_loaded_calibration_mode(
    tmp_path, monkeypatch,
):
    import json

    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("models:\n  latent_dims: 64\n", encoding="utf-8")
    threshold_root = tmp_path / "thresholds"
    area_dir = threshold_root / "PLeft"
    area_dir.mkdir(parents=True)
    (area_dir / "threshold_test_PLeft_percentile99.0_off3_sig1.5_q0.99.json").write_text(
        json.dumps({
            "threshold": 0.42,
            "threshold_strategy": "percentile",
            "offset": 3,
            "sigma": 1.5,
            "quantile": 0.99,
            "mode": "test",
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", [
        "infer_offline.py",
        "--input_type", "frames",
        "--input", str(input_dir),
        "--scenario", "8_16",
        "--config", str(config_path),
        "--threshold_dir", str(threshold_root),
        "--safety_areas", "PLeft",
        "--threshold_strategy", "percentile",
        "--offset", "3",
        "--sigma", "1.5",
        "--quantile", "0.99",
        "--scores-only",
    ])

    settings = load_settings(parse_args())

    assert settings.output_csv.name == (
        "frames_8_16_percentile_off3_sig1.5_q0.99_cal-test_scores.csv"
    )
    assert settings.threshold_calibration_by_area == {"PLeft": "test"}


def test_amplified_threshold_preserves_calibrated_value():
    config = amplify_threshold_config({"threshold": 0.5}, 1.2)
    assert config["calibrated_threshold"] == pytest.approx(0.5)
    assert config["threshold_amplification"] == pytest.approx(1.2)
    assert config["threshold"] == pytest.approx(0.6)
    assert threshold_amplification_variant_tag(
        resolve_threshold_amplifications([1.1, 1.2], ["PLeft", "PRight"])
    ) == "_amp-1.1-1.2"


def test_sample_display_does_not_include_full_path():
    sample = compact_sample_name("/beegfs/home/user/videos/input.mp4#frame=42")
    assert sample == "input.mp4#frame=42"


def test_explicit_video_path_enables_video_outputs(monkeypatch):
    args = parse(monkeypatch, "--output_video", "/tmp/detections.mp4")
    assert args.save_video is True


def test_scores_only_rejects_explicit_video_path(monkeypatch):
    with pytest.raises(SystemExit):
        parse(
            monkeypatch, "--scores-only", "--output_video", "/tmp/detections.mp4"
        )


def test_camera_name_is_derived_from_ros_topic():
    assert camera_name_from_topic("/camera/back_view/image_raw") == "back_view"


def test_video_scenario_resolves_generated_dataset_video(tmp_path):
    video = (
        tmp_path
        / "extracted_frames"
        / "13_1"
        / "back_view"
        / "video"
        / "s-13_1_c-back_view.mp4"
    )
    video.parent.mkdir(parents=True)
    video.touch()

    resolved, scenario_id = resolve_video_scenario(
        "13_1", {"dataset_base": str(tmp_path)}, "/camera/back_view/image_raw"
    )

    assert resolved == video.resolve()
    assert scenario_id == "13_1"


def test_video_scenario_reports_expected_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="s-13_1_c-front_view.mp4"):
        resolve_video_scenario(
            "13_1", {"dataset_base": str(tmp_path)}, "/camera/front_view/image_raw"
        )


def test_video_scenario_resolves_epito_videos_directory(tmp_path):
    video = tmp_path / "videos" / "s-8_0_c-back_view.mp4"
    video.parent.mkdir(parents=True)
    video.touch()

    resolved, scenario_id = resolve_video_scenario(
        "8_0", {"dataset_base": str(tmp_path)}, "/camera/back_view/image_raw"
    )

    assert resolved == video.resolve()
    assert scenario_id == "8_0"


def test_threshold_ablation_variant_is_selected(tmp_path):
    import json

    area_dir = tmp_path / "PLeft"
    area_dir.mkdir()
    threshold = area_dir / "threshold_PLeft_percentile99.0_off2_sig1.5_q0.98.json"
    threshold.write_text(json.dumps({
        "threshold": 0.42,
        "threshold_strategy": "percentile",
        "offset": 2,
        "sigma": 1.5,
        "quantile": 0.98,
    }))

    loaded = load_threshold(
        tmp_path, "PLeft", "percentile", 99.0, 2, 1.5, 0.98
    )

    assert loaded["path"] == threshold
    assert loaded["threshold"] == 0.42


def test_mode_prefixed_test_threshold_is_preferred_by_inference(tmp_path):
    import json

    area_dir = tmp_path / "PLeft"
    area_dir.mkdir()
    common = {
        "threshold_strategy": "percentile",
        "offset": 2,
        "sigma": 1.5,
        "quantile": 0.98,
    }
    test_threshold = area_dir / (
        "threshold_test_PLeft_percentile99.0_off2_sig1.5_q0.98.json"
    )
    val_threshold = area_dir / (
        "threshold_val_PLeft_percentile99.0_off2_sig1.5_q0.98.json"
    )
    test_threshold.write_text(
        json.dumps({**common, "threshold": 0.42, "mode": "test"})
    )
    val_threshold.write_text(
        json.dumps({**common, "threshold": 0.21, "mode": "val"})
    )

    loaded = load_threshold(
        tmp_path, "PLeft", "percentile", 99.0, 2, 1.5, 0.98
    )

    assert loaded["path"] == test_threshold
    assert loaded["threshold"] == 0.42
    assert loaded["calibration_mode"] == "test"

    selected_val = load_threshold(
        tmp_path, "PLeft", "percentile", 99.0, 2, 1.5, 0.98,
        "canonical", "val",
    )
    assert selected_val["path"] == val_threshold
    assert selected_val["threshold"] == 0.21
    assert selected_val["calibration_mode"] == "val"


def test_mode_prefixed_val_threshold_is_inference_fallback(tmp_path):
    import json

    area_dir = tmp_path / "PRight"
    area_dir.mkdir()
    val_threshold = area_dir / (
        "threshold_val_PRight_percentile99.0_off1_sig1.0_q0.99.json"
    )
    val_threshold.write_text(json.dumps({
        "threshold": 0.31,
        "threshold_strategy": "percentile",
        "offset": 1,
        "sigma": 1.0,
        "quantile": 0.99,
        "mode": "val",
    }))

    loaded = load_threshold(
        tmp_path, "PRight", "percentile", 99.0, 1, 1.0, 0.99
    )

    assert loaded["path"] == val_threshold
    assert loaded["calibration_mode"] == "val"

    with pytest.raises(FileNotFoundError, match="calibration_mode=test"):
        load_threshold(
            tmp_path, "PRight", "percentile", 99.0, 1, 1.0, 0.99,
            "canonical", "test",
        )


def test_supervised_f1c_threshold_is_loadable(tmp_path):
    import json

    area_dir = tmp_path / "PRight"
    area_dir.mkdir()
    threshold = area_dir / "threshold_PRight_f1c_off3_sig1.5_q0.98.json"
    threshold.write_text(json.dumps({
        "threshold": 0.42,
        "threshold_strategy": "f1c",
        "offset": 3,
        "sigma": 1.5,
        "quantile": 0.98,
    }))

    loaded = load_threshold(
        tmp_path, "PRight", "f1c", 99.0, 3, 1.5, 0.98
    )

    assert loaded["path"] == threshold
    assert loaded["strategy"] == "f1c"


def test_normal_only_max_threshold_is_loadable(tmp_path):
    import json

    area_dir = tmp_path / "PRight"
    area_dir.mkdir()
    threshold = area_dir / "threshold_PRight_max_off1_sig1.0_q0.99.json"
    threshold.write_text(json.dumps({
        "threshold": 0.42,
        "threshold_strategy": "max",
        "offset": 1,
        "sigma": 1.0,
        "quantile": 0.99,
    }))

    loaded = load_threshold(
        tmp_path, "PRight", "max", 99.0, 1, 1.0, 0.99
    )

    assert loaded["path"] == threshold
    assert loaded["strategy"] == "max"


def test_threshold_parameters_must_match_requested_variant(tmp_path):
    import json

    area_dir = tmp_path / "PLeft"
    area_dir.mkdir()
    threshold = area_dir / "threshold_PLeft_percentile.json"
    threshold.write_text(json.dumps({
        "threshold": 0.42,
        "threshold_strategy": "percentile",
        "offset": 3,
        "sigma": 1.5,
        "quantile": 0.98,
    }))

    with pytest.raises(ValueError, match="TAAS parameters do not match"):
        load_threshold(tmp_path, "PLeft", "percentile", 99.0, 2, 1.5, 0.98)


def test_minimization_variant_loads_only_matching_threshold(tmp_path):
    import json

    area_dir = tmp_path / "PLeft"
    area_dir.mkdir()
    threshold = area_dir / (
        "threshold_PLeft_percentile99.0_off3_sig1.5_q0.99"
        "_taas-minimization.json"
    )
    threshold.write_text(json.dumps({
        "threshold": 0.12,
        "threshold_strategy": "percentile",
        "offset": 3,
        "sigma": 1.5,
        "quantile": 0.99,
        "taas_variant": "minimization",
    }))

    loaded = load_threshold(
        tmp_path, "PLeft", "percentile", 99.0, 3, 1.5, 0.99,
        "minimization",
    )

    assert loaded["path"] == threshold
    assert loaded["taas_variant"] == "minimization"


def test_public_result_contains_threshold_provenance():
    result = {
        "safety_area": "PLeft",
        "anomaly_score": 0.3,
        "threshold": 0.2,
        "normalized_score": 1.5,
        "is_anomalous": True,
        "threshold_strategy": "percentile",
        "score_func": "taas",
        "reconstruction_mode": "mean",
        "offset": 2,
        "sigma": 1.5,
        "quantile": 0.98,
        "threshold_file": "threshold_PLeft_percentile_off2_sig1.5_q0.98.json",
        "original_bgr": object(),
    }

    row = public_result(result)

    assert row["threshold_strategy"] == "percentile"
    assert row["offset"] == 2
    assert row["sigma"] == 1.5
    assert row["quantile"] == 0.98
    assert row["calibrated_threshold"] == pytest.approx(0.2)
    assert row["threshold_amplification"] == pytest.approx(1.0)
    assert row["threshold_file"].endswith("_off2_sig1.5_q0.98.json")
    assert "original_bgr" not in row


def test_threshold_variant_tag_matches_calibration_filename_convention():
    assert (
        threshold_variant_tag("percentile", 2, 1.5, 0.98)
        == "percentile_off2_sig1.5_q0.98"
    )


def test_rolling_variant_tag_only_changes_active_rolling_outputs():
    assert rolling_variant_tag("none", 5) == ""
    assert rolling_variant_tag("mean", 5) == "_rollmean_w5"


def test_taas_variant_tag_preserves_legacy_canonical_filenames():
    assert taas_variant_tag("canonical") == ""
    assert taas_variant_tag("minimization") == "_taas-minimization"


@pytest.mark.parametrize(
    ("policy", "expected"),
    (("mean", 0.4), ("min", 0.2), ("max", 0.6), ("none", 0.6)),
)
def test_rolling_policy_controls_detection_score(policy, expected):
    window = deque([0.2, 0.4], maxlen=3)
    result = {
        "anomaly_score": 0.6,
        "normalized_score": 1.2,
        "threshold": 0.5,
        "is_anomalous": True,
    }

    apply_rolling_policy(result, window, policy, 3)

    assert result["instantaneous_anomaly_score"] == 0.6
    assert result["anomaly_score"] == pytest.approx(expected)
    assert result["normalized_score"] == pytest.approx(expected / 0.5)
    assert result["is_anomalous"] == (expected > 0.5)
    assert result["rolling_policy"] == policy
