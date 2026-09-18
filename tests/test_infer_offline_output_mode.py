from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from infer_offline import (  # noqa: E402
    camera_name_from_topic,
    load_threshold,
    parse_args,
    public_result,
    resolve_video_scenario,
    threshold_variant_tag,
)


def parse(monkeypatch, *extra):
    monkeypatch.setattr(
        sys, "argv", ["infer_offline.py", "--input_type", "frames", "--input", "/tmp"]
        + list(extra),
    )
    return parse_args()


def test_csv_and_video_are_default(monkeypatch):
    assert parse(monkeypatch).save_video is True


def test_scores_only_disables_video_outputs(monkeypatch):
    assert parse(monkeypatch, "--scores-only").save_video is False


def test_save_video_switch_enables_video_outputs(monkeypatch):
    assert parse(monkeypatch, "--save-video").save_video is True


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
    threshold = area_dir / "threshold_PLeft_percentile_off2_sig1.5_q0.98.json"
    threshold.write_text(json.dumps({
        "threshold": 0.42,
        "threshold_strategy": "percentile",
        "offset": 2,
        "sigma": 1.5,
        "quantile": 0.98,
    }))

    loaded = load_threshold(tmp_path, "PLeft", "percentile", 2, 1.5, 0.98)

    assert loaded["path"] == threshold
    assert loaded["threshold"] == 0.42


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
        load_threshold(tmp_path, "PLeft", "percentile", 2, 1.5, 0.98)


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
    assert row["threshold_file"].endswith("_off2_sig1.5_q0.98.json")
    assert "original_bgr" not in row


def test_threshold_variant_tag_matches_calibration_filename_convention():
    assert (
        threshold_variant_tag("percentile", 2, 1.5, 0.98)
        == "percentile_off2_sig1.5_q0.98"
    )
