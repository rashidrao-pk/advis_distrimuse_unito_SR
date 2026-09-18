from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from compare_annotations_detection import (  # noqa: E402
    binary_metrics,
    default_evaluation_path,
    load_threshold_metadata,
)


def test_f1_is_undefined_without_annotated_anomalies():
    frames = pd.DataFrame({
        "label": ["Normal", "Normal", "Verify"],
        "detected_anomalous": [False, True, True],
    })

    metrics = binary_metrics(frames)

    assert metrics["f1"] is None
    assert metrics["recall"] is None
    assert metrics["fp"] == 1
    assert metrics["specificity"] == 0.5


def test_f1_is_computed_when_anomalies_are_annotated():
    frames = pd.DataFrame({
        "label": ["Anomalous", "Anomalous", "Normal"],
        "detected_anomalous": [True, False, False],
    })

    metrics = binary_metrics(frames)

    assert metrics["f1"] == 2 / 3
    assert metrics["total_normal"] == 1
    assert metrics["total_anomalous"] == 2
    assert metrics["balanced_accuracy"] == 0.75


def test_threshold_metadata_identifies_taas_variant(tmp_path):
    area_dir = tmp_path / "PLeft"
    area_dir.mkdir()
    (area_dir / "threshold_PLeft_max.json").write_text(
        '{"threshold": 0.5, "threshold_strategy": "max", '
        '"score_func": "TAAS_1-s_1.0-q_0.99", "offset": 1, '
        '"sigma": 1.0, "quantile": 0.99}',
        encoding="utf-8",
    )
    data = pd.DataFrame({
        "score_strategy": ["max"], "safety_area": ["PLeft"],
        "threshold": [0.5],
    })

    record = load_threshold_metadata(data, tmp_path)[0]

    assert record["score_func"] == "TAAS_1-s_1.0-q_0.99"
    assert record["offset"] == 1
    assert record["sigma"] == 1.0
    assert record["quantile"] == 0.99
    assert record["threshold_matches_csv"] is True


def test_default_evaluation_path_removes_input_type_prefix(tmp_path):
    threshold_dir = tmp_path / "results" / "V6" / "thresholds"
    score = tmp_path / "video_8_16_percentile_off1_sig1.0_q0.99_scores.csv"

    output = default_evaluation_path(score, threshold_dir)

    assert output == (
        tmp_path / "results" / "V6" / "evaluation"
        / "evaluation_8_16_percentile_off1_sig1.0_q0.99_scores.json"
    )
