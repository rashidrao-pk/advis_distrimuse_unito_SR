from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from compare_annotations_detection import binary_metrics  # noqa: E402


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
