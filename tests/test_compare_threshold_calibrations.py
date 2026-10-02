import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from compare_threshold_calibrations import (  # noqa: E402
    load_combinations,
    winner,
    write_report,
)


def _artifact(method, metrics, is_best=False, plot_file=None):
    artifact = {
        "artifact_type": "test_threshold_combination",
        "method": method,
        "safety_area": "ConvBelt",
        "threshold": 0.2,
        "selection_metric": "binormal_auc",
        "score_parameters": {
            "offset": 1, "sigma": 1.5, "quantile": 0.99,
            "taas_variant": "canonical",
        },
        "metrics": metrics,
        "is_best": is_best,
        "rank": 1 if is_best else 2,
    }
    if plot_file:
        artifact["plot_file"] = plot_file
    return artifact


def test_load_derives_balanced_accuracy_and_renames_legacy_metric(tmp_path):
    json_dir = tmp_path / "scenario-8_16" / "json"
    json_dir.mkdir(parents=True)
    payload = _artifact("TAAS_A", {
        "accuracy": 0.8, "precision": 0.75, "recall": 0.9, "f1": 0.82,
        "auc": 0.95, "binormal_auc": 2.7,
        "true_negative": 80, "false_positive": 20,
        "false_negative": 10, "true_positive": 90,
    }, True)
    (json_dir / "a.json").write_text(json.dumps(payload), encoding="utf-8")

    frame, scenario_dir = load_combinations(tmp_path / "scenario-8_16")

    assert scenario_dir == tmp_path / "scenario-8_16"
    assert frame.loc[0, "specificity"] == 0.8
    assert frame.loc[0, "balanced_accuracy"] == pytest.approx(0.85)
    assert frame.loc[0, "separation_score"] == 2.7


def test_report_ranks_by_balanced_accuracy_without_changing_saved_winner(tmp_path):
    json_dir = tmp_path / "scenario-8_16" / "json"
    json_dir.mkdir(parents=True)
    plots_dir = json_dir.parent / "plots"
    plots_dir.mkdir()
    (plots_dir / "saved.png").write_bytes(b"saved plot")
    (plots_dir / "balanced.png").write_bytes(b"balanced plot")
    artifacts = [
        _artifact("SAVED", {
            "accuracy": .8, "precision": .99, "recall": .5, "f1": .66,
            "auc": .9, "binormal_auc": 4.0, "true_negative": 99,
            "false_positive": 1, "false_negative": 50, "true_positive": 50,
        }, True, "../plots/saved.png"),
        _artifact("BALANCED", {
            "accuracy": .95, "precision": .9, "recall": .95, "f1": .92,
            "auc": .98, "binormal_auc": 2.0, "true_negative": 95,
            "false_positive": 5, "false_negative": 5, "true_positive": 95,
        }, plot_file="../plots/balanced.png"),
    ]
    for index, payload in enumerate(artifacts):
        (json_dir / f"{index}.json").write_text(json.dumps(payload), encoding="utf-8")
    frame, _ = load_combinations(tmp_path / "scenario-8_16")
    output = tmp_path / "comparison.html"

    write_report(output, frame, "balanced_accuracy")

    assert winner(frame, "balanced_accuracy").method == "BALANCED"
    contents = output.read_text(encoding="utf-8")
    assert "The saved winner differs from the report winner" in contents
    assert "BALANCED" in contents
    assert 'href="#calibration-plot-BALANCED"' in contents
    assert 'id="calibration-plot-BALANCED"' in contents
    assert 'src="file://' in contents
    assert "balanced.png" in contents
    assert contents.index('id="calibration-plot-BALANCED"') < contents.index(
        'id="calibration-plot-SAVED"'
    )
    assert frame.loc[frame.method == "SAVED", "saved_winner"].item()
