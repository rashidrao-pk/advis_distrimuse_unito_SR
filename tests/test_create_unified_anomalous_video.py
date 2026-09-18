import csv
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
    write_unified_annotations,
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


def test_unified_annotations_use_global_video_frame_ids(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "scenario_8_1_back_view_annotations.csv").write_text(
        "frame_id,safety_area,label,scenario_description,note\n"
        "0,PLeft,Normal,First scenario,\n"
        "0,PRight,Anomalous,First scenario,object\n"
        "2,PLeft,Verify,First scenario,check\n"
        "2,PRight,Normal,First scenario,\n",
        encoding="utf-8",
    )
    (annotations / "scenario_9_0_back_view_annotations.csv").write_text(
        "frame_id,safety_area,label,scenario_description,note\n"
        "0,PLeft,Anomalous,Second scenario,\n"
        "0,PRight,Normal,Second scenario,\n",
        encoding="utf-8",
    )
    videos = [
        ("8_1", tmp_path / "8_1.mp4"),
        ("9_0", tmp_path / "9_0.mp4"),
    ]
    output = tmp_path / "unified_annotations.csv"

    rows = write_unified_annotations(
        output, videos, {"8_1": 2, "9_0": 1}, annotations,
        tmp_path / "frames", "back_view", ["PLeft", "PRight"], 2,
    )

    with output.open(newline="", encoding="utf-8") as stream:
        data = list(csv.DictReader(stream))
    assert rows == 6
    assert [int(row["frame_id"]) for row in data] == [0, 0, 1, 1, 2, 2]
    assert [int(row["source_frame_id"]) for row in data] == [0, 0, 2, 2, 0, 0]
    assert data[2]["label"] == "Verify"
    assert data[4]["source_scenario_id"] == "9_0"
