import json
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from compare_evaluation_results import load_evaluations, ranked  # noqa: E402


def write_evaluation(path, *, offset, f1, balanced_accuracy):
    area = {
        "metrics": {"f1": f1, "balanced_accuracy": balanced_accuracy},
        "inference": {
            "threshold_strategy": "percentile", "offset": offset,
            "sigma": 1.0, "quantile": 0.99, "threshold": 0.2,
            "score_func": f"TAAS_OFF{offset}-s_1.0-q_0.99",
        },
        "threshold_calibration": {"threshold_percentile": 99.0},
        "model_training": {"dataset": {"n_train_images": 100, "n_val_images": 20}},
    }
    payload = {
        "sources": {"html_report": str(path.with_suffix(".html"))},
        "evaluation": {"percentile": {
            "cumulative_metrics": {
                "f1": f1, "balanced_accuracy": balanced_accuracy,
                "precision": 0.5, "recall": 0.8, "specificity": 0.7,
                "accuracy": 0.75, "auroc": 0.9, "auprc": 0.7,
                "false_positive_rate": 0.3, "false_negative_rate": 0.2,
                "tp": 8, "tn": 7, "fp": 3, "fn": 2,
            },
            "safety_areas": {"PLeft": area},
        }},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluations_are_loaded_and_ranked(tmp_path):
    write_evaluation(
        tmp_path / "evaluation_case_off1_sig1.0_q0.99_scores.json",
        offset=1, f1=0.6, balanced_accuracy=0.7,
    )
    write_evaluation(
        tmp_path / "evaluation_case_off2_sig1.0_q0.99_scores.json",
        offset=2, f1=0.7, balanced_accuracy=0.8,
    )

    cumulative, areas = load_evaluations(tmp_path)
    ordered = ranked(cumulative, "balanced_accuracy")

    assert len(cumulative) == 2
    assert len(areas) == 2
    assert ordered.iloc[0]["offset"] == 2


def test_error_count_ranking_uses_lower_values_first(tmp_path):
    write_evaluation(
        tmp_path / "evaluation_case_off1_sig1.0_q0.99_scores.json",
        offset=1, f1=0.6, balanced_accuracy=0.7,
    )
    write_evaluation(
        tmp_path / "evaluation_case_off2_sig1.0_q0.99_scores.json",
        offset=2, f1=0.7, balanced_accuracy=0.8,
    )
    payload = json.loads(
        (tmp_path / "evaluation_case_off2_sig1.0_q0.99_scores.json").read_text()
    )
    payload["evaluation"]["percentile"]["cumulative_metrics"]["fp"] = 1
    (tmp_path / "evaluation_case_off2_sig1.0_q0.99_scores.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    cumulative, _ = load_evaluations(tmp_path)
    assert ranked(cumulative, "fp").iloc[0]["offset"] == 2
