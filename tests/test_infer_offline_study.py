from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from infer_offline_study import (  # noqa: E402
    split_study_dashboard,
    study_output_paths,
)


def test_split_study_dashboard_preserves_top_and_bottom_rows():
    dashboard = np.zeros((1000, 1600, 3), dtype=np.uint8)
    dashboard[:500] = 11
    dashboard[500:] = 22

    top, bottom = split_study_dashboard(dashboard)

    assert top.shape == (500, 1600, 3)
    assert bottom.shape == (500, 1600, 3)
    assert np.all(top == 11)
    assert np.all(bottom == 22)
    assert top.flags.c_contiguous
    assert bottom.flags.c_contiguous


def test_study_outputs_use_dedicated_directory_and_row_suffixes():
    args = SimpleNamespace(
        dataset_version="V6",
        output_video=Path(
            "/tmp/video_8_16_percentile_off3_sig1.0_q0.99_detections.mp4"
        ),
        output_csv=Path(
            "/tmp/video_8_16_percentile_off3_sig1.0_q0.99_scores.csv"
        ),
    )

    top, bottom, scores = study_output_paths(args)

    expected_parent = (
        Path(__file__).resolve().parents[1] / "results" / "V6"
        / "offline_inference" / "study_video"
    )
    assert top.parent == expected_parent
    assert bottom.parent == expected_parent
    assert scores.parent == expected_parent
    assert top.name.endswith("_detections_TL_TR.mp4")
    assert bottom.name.endswith("_detections_BL_BR.mp4")
    assert scores.name.endswith("_scores.csv")
