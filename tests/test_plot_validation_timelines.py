from pathlib import Path
from types import SimpleNamespace
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from plot_validation_timelines import load_area_data, score_filename  # noqa: E402


def arguments(threshold_root):
    return SimpleNamespace(
        threshold_root=threshold_root,
        threshold_strategy="percentile",
        threshold_percentile=99.0,
        offset=1,
        sigma=1.0,
        quantile=0.99,
    )


def test_score_filename_matches_calibration_convention(tmp_path):
    assert score_filename("PLeft", arguments(tmp_path)) == (
        "val_scores_PLeft_percentile99.0_off1_sig1.0_q0.99.csv"
    )


def test_load_area_data_uses_matching_variant_metadata(tmp_path):
    area_dir = tmp_path / "PLeft"
    area_dir.mkdir()
    csv_path = area_dir / "val_scores_PLeft_percentile99.0_off1_sig1.0_q0.99.csv"
    csv_path.write_text("file_name,anomaly_score\na.png,0.1\n", encoding="utf-8")
    metadata_path = area_dir / (
        "threshold_PLeft_percentile99.0_off1_sig1.0_q0.99.json"
    )
    metadata_path.write_text(
        '{"threshold": 0.2, "threshold_strategy": "percentile", '
        '"score_func": "TAAS_OFF1-s_1.0-q_0.99"}',
        encoding="utf-8",
    )

    frame, threshold, selected_csv, metadata = load_area_data(
        "PLeft", arguments(tmp_path)
    )

    assert len(frame) == 1
    assert threshold == 0.2
    assert selected_csv == csv_path
    assert metadata["score_func"] == "TAAS_OFF1-s_1.0-q_0.99"
