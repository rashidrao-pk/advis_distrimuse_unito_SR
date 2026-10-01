from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from compare_annotations_detection import (  # noqa: E402
    binary_metrics,
    default_evaluation_path,
    evaluation_plot_context,
    load_threshold_metadata,
    ranking_curve_data,
    scenario_id_from_annotation_path,
    scenario_runs,
)


def test_evaluation_plot_context_shows_scenario_and_inference_score():
    data = pd.DataFrame({
        "scenario_id": ["8_16", "8_16"],
        "inference_score_func": [
            "TAAS_OFF1-s_1.0-q_0.97", "TAAS_OFF1-s_1.0-q_0.97",
        ],
    })

    assert evaluation_plot_context(data) == (
        "Scenario 8_16 | TAAS_OFF1-s_1.0-q_0.97"
    )


def test_unified_plot_context_uses_annotation_filename_scenario_id():
    data = pd.DataFrame({
        "scenario_id": ["unified"],
        "inference_score_func": ["TAAS_OFF3-s_1.5-q_0.99"],
    })
    annotation = Path("scenario_8_16_back_view_annotations.csv")

    scenario_id = scenario_id_from_annotation_path(annotation)

    assert scenario_id == "8_16"
    assert evaluation_plot_context(data, scenario_id) == (
        "Scenario 8_16 | TAAS_OFF3-s_1.5-q_0.99"
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


def test_scenario_runs_map_global_frames_to_source_scenarios():
    data = pd.DataFrame({
        "frame_id": [0, 1, 2, 3, 4],
        "source_scenario_id": ["8_0", "8_0", "9_0", "9_0", "9_0"],
        "scenario_description": ["first", "first", "second", "second", "second"],
    })

    assert list(scenario_runs(data)) == [
        (0, 1, "8_0", "first"),
        (2, 4, "9_0", "second"),
    ]


def test_ranking_metrics_are_perfect_for_separated_scores():
    data = pd.DataFrame({
        "label": ["Normal", "Normal", "Anomalous", "Anomalous", "Verify"],
        "normalized_score": [0.1, 0.2, 0.8, 0.9, 2.0],
    })

    curves = ranking_curve_data(data)

    assert curves["auroc"] == 1.0
    assert curves["auprc"] == 1.0
