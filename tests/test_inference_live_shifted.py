"""Tests for vertical alignment in the shifted live-inference entry point."""

from pathlib import Path
import sys

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from inference_live_shifted import parse_args, shift_frame_vertical  # noqa: E402


def test_positive_shift_moves_content_down_and_replicates_top_edge():
    frame = np.arange(4 * 3, dtype=np.uint8).reshape(4, 3)

    shifted = shift_frame_vertical(frame, 1, "replicate")

    assert np.array_equal(shifted[0], frame[0])
    assert np.array_equal(shifted[1:], frame[:-1])


def test_negative_shift_moves_content_up_and_can_fill_black():
    frame = np.arange(4 * 3, dtype=np.uint8).reshape(4, 3)

    shifted = shift_frame_vertical(frame, -2, "black")

    assert np.array_equal(shifted[:-2], frame[2:])
    assert np.count_nonzero(shifted[-2:]) == 0


def test_shifted_cli_keeps_standard_live_options():
    args = parse_args([
        "--frame_shift_y", "12",
        "--frame_shift_fill", "replicate",
        "--rolling", "min",
        "--rolling_window", "10",
    ])

    assert args.frame_shift_y == 12
    assert args.frame_shift_fill == "replicate"
    assert args.rolling == "min"
    assert args.rolling_window == 10
