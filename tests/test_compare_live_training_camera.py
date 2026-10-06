from pathlib import Path
import sys

import cv2
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from compare_live_training_camera import (  # noqa: E402
    crop_live_like_training,
    estimate_alignment,
    find_training_reference,
)


def test_training_crop_maps_back_to_full_raw_frame(tmp_path):
    training = tmp_path / "Jul27" / "train"
    crop = training / "PLeft" / "normal" / "s-4_2_s-PLeft_f-000123.png"
    crop.parent.mkdir(parents=True)
    cv2.imwrite(str(crop), np.zeros((10, 10, 3), dtype=np.uint8))
    raw = (
        tmp_path / "Jul27" / "extracted_frames" / "4_2"
        / "back_view" / "raw" / "frame_000123.png"
    )
    raw.parent.mkdir(parents=True)
    cv2.imwrite(str(raw), np.zeros((20, 30, 3), dtype=np.uint8))

    reference, selected_crop = find_training_reference(
        {"data": {"training": str(training)}}, "PLeft", "back_view"
    )

    assert reference == raw.resolve()
    assert selected_crop == crop.resolve()


def test_alignment_recovers_vertical_translation():
    rng = np.random.default_rng(7)
    training = np.zeros((320, 480, 3), dtype=np.uint8)
    for _ in range(250):
        x = int(rng.integers(10, 470))
        y = int(rng.integers(10, 310))
        color = tuple(int(value) for value in rng.integers(60, 255, size=3))
        cv2.circle(training, (x, y), int(rng.integers(2, 6)), color, -1)
    live = cv2.warpAffine(
        training, np.float32([[1, 0, 0], [0, 1, 14]]), (480, 320),
        borderMode=cv2.BORDER_REPLICATE,
    )

    result = estimate_alignment(training, live)

    assert result["same_camera_angle"] is True
    assert abs(result["shift_x_pixels"]) < 1.0
    assert abs(result["shift_y_pixels"] + 14.0) < 1.0
    assert abs(result["rotation_degrees"]) < 0.2
    assert abs(result["scale"] - 1.0) < 0.01


def test_training_crop_is_used_when_raw_source_was_not_retained(tmp_path):
    training = tmp_path / "Jul27" / "train"
    crop = training / "PLeft" / "normal" / "s-4_2_s-PLeft_f-000123.png"
    crop.parent.mkdir(parents=True)
    cv2.imwrite(str(crop), np.zeros((128, 128, 3), dtype=np.uint8))

    reference, selected_crop = find_training_reference(
        {"data": {"training": str(training)}}, "PLeft", "back_view"
    )

    assert reference == crop.resolve()
    assert selected_crop == crop.resolve()


def test_live_frame_uses_same_mask_crop_as_training(tmp_path):
    frame = np.full((100, 200, 3), 100, dtype=np.uint8)
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[20:80, 40:160] = 255
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask)

    crop, source_scale = crop_live_like_training(
        frame, {"data": {"mask_types": {"PLeft": str(mask_path)}}},
        "PLeft", 128,
    )

    assert crop.shape == (128, 128, 3)
    assert abs(source_scale - (120 / 128)) < 0.01
