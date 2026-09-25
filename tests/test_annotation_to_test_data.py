"""Tests for building annotated per-scenario test data."""

import csv
from pathlib import Path
import sys

import yaml


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from annotation_to_test_data import main  # noqa: E402


def test_builds_scenario_area_class_hierarchy(tmp_path):
    dataset = tmp_path / "dataset"
    annotations = tmp_path / "annotations"
    output = dataset / "test"
    crop_dir = dataset / "extracted_frames" / "8_0" / "back_view" / "processed" / "PLeft"
    crop_dir.mkdir(parents=True)
    annotations.mkdir()

    rows = []
    for frame_id, label in enumerate(("Normal", "Anomalous", "Verify")):
        filename = f"s-8_0_s-PLeft_f-{frame_id:06d}.png"
        source = crop_dir / filename
        source.write_bytes(f"frame-{frame_id}".encode())
        rows.append({
            "scenario_id": "8_0",
            "scenario_description": "test scenario",
            "camera": "back_view",
            "safety_area": "PLeft",
            "frame_position": str(frame_id + 1),
            "frame_id": str(frame_id),
            "filename": filename,
            "label": label,
            "note": "",
            # An empty saved path exercises the extracted-root fallback.
            "processed_image_path": "",
            "raw_image_path": "",
        })

    annotation_csv = annotations / "scenario_8_0_back_view_annotations.csv"
    with annotation_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({
            "data": {
                "dataset_base": str(dataset),
                "testing": str(output),
            }
        }),
        encoding="utf-8",
    )

    assert main([
        "--config", str(config),
        "--annotations-dir", str(annotations),
    ]) == 0

    for frame_id, label in enumerate(("normal", "anomalous", "verify")):
        expected = (
            output / "8_0" / "PLeft" / label
            / f"s-8_0_s-PLeft_f-{frame_id:06d}.png"
        )
        assert expected.read_bytes() == f"frame-{frame_id}".encode()
    assert (output / "test_manifest.csv").is_file()
