from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from create_unified_anomalous_video import (  # noqa: E402
    LABEL_COLORS,
    discover_scenarios,
    draw_annotated_masks,
    load_annotations,
    scenario_key,
)


def test_scenario_key_is_numeric_not_lexicographic():
    assert scenario_key("9_1") < scenario_key("10_0")


def test_invalid_scenario_id_is_rejected():
    with pytest.raises(ValueError):
        scenario_key("scenario_9_0")


def test_discovery_selects_strictly_after_cutoff_in_numeric_order(tmp_path):
    for name in ("10_0", "8_0", "9_1", "8_1", "7_4", "notes", "11_2"):
        (tmp_path / name).mkdir()
    (tmp_path / "9_0.txt").write_text("not a directory", encoding="utf-8")

    selected = discover_scenarios(tmp_path, "8_0")

    assert selected == ["8_1", "9_1", "10_0", "11_2"]


def test_annotation_csv_is_keyed_by_frame_and_safety_area(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_text(
        "frame_id,safety_area,label,scenario_description\n"
        "0,PLeft,Normal,Test scenario\n"
        "0,PRight,Anomalous,Test scenario\n"
        "1,PLeft,Verify,Test scenario\n",
        encoding="utf-8",
    )

    labels, description = load_annotations(path)

    assert labels[(0, "PLeft")] == "Normal"
    assert labels[(0, "PRight")] == "Anomalous"
    assert labels[(1, "PLeft")] == "Verify"
    assert description == "Test scenario"


def test_mask_overlay_uses_annotation_color():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (30, 30), (70, 70), 255, -1)

    output = draw_annotated_masks(
        frame, {"PLeft": mask}, {(0, "PLeft"): "Anomalous"},
        frame_id=0, scenario="9_0", description="", opacity=1.0,
    )

    # Choose a masked point away from text and the header.
    assert tuple(output[68, 68]) == LABEL_COLORS["Anomalous"]
