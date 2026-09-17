from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from infer_offline import parse_args  # noqa: E402


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
