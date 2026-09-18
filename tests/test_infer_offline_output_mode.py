from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from infer_offline import (  # noqa: E402
    camera_name_from_topic,
    parse_args,
    resolve_video_scenario,
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
