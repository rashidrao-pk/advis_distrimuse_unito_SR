"""Focused tests for the ROS-independent parts of live inference."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from inference_live import (  # noqa: E402
    RULEX_AREA_ENUM_NAMES,
    detection_payload,
    detect_topic_message_type,
    load_live_masks,
    load_live_settings,
    pack_debug_state,
    pack_timeline_state,
    parse_args,
)


def test_rulex_area_mapping_matches_integration_contract():
    assert RULEX_AREA_ENUM_NAMES == {
        "PRight": "AREA_A",
        "PLeft": "AREA_B",
        "RoboArm": "AREA_C",
        "ConvBelt": "AREA_D",
    }


def test_live_cli_uses_auto_type_and_json_output_by_default():
    args = parse_args([])

    assert args.message_type == "auto"
    assert args.detections_topic == "/advis/detections"
    assert args.dashboard_topic == ""
    assert args.safety_areas == ["PLeft", "PRight", "RoboArm", "ConvBelt"]
    assert args.source_name == "live_ros"
    assert args.scenario_id == ""


def test_live_cli_accepts_current_inference_controls():
    args = parse_args([
        "--message_type", "compressed",
        "--safety_areas", "PLeft", "RoboArm",
        "--threshold_strategy", "percentile",
        "--offset", "3",
        "--sigma", "1.5",
        "--quantile", "0.98",
        "--taas_backend", "cython",
        "--taas_variant", "minimization",
        "--rolling", "mean",
        "--rolling_window", "5",
        "--threshold_amplification", "1.1", "1.2",
    ])

    assert args.message_type == "compressed"
    assert args.safety_areas == ["PLeft", "RoboArm"]
    assert args.threshold_strategy == "percentile"
    assert (args.offset, args.sigma, args.quantile) == (3, 1.5, 0.98)
    assert args.taas_backend == "cython"
    assert args.taas_variant == "minimization"
    assert (args.rolling, args.rolling_window) == ("mean", 5)
    assert list(args.threshold_amplification_by_area.items()) == [
        ("PLeft", 1.1), ("RoboArm", 1.2)
    ]


def test_debug_media_requires_and_accepts_zenoh():
    with pytest.raises(SystemExit):
        parse_args(["--debug_mode"])

    args = parse_args(["--publish_zenoh", "--debug_mode"])
    assert args.publish_debug_media is True
    assert args.zenoh_debug_key == "advis/vis/debug/state"


def test_debug_payload_contains_results_and_runtime_metadata():
    class Packer:
        @staticmethod
        def packb(payload, use_bin_type):
            assert use_bin_type is True
            return payload

    result = {
        "safety_area": "PLeft",
        "anomaly_score": 0.6,
        "threshold": 0.5,
        "normalized_score": 1.2,
        "is_anomalous": True,
        "threshold_strategy": "percentile",
        "score_func": "TAAS_OFF3-s_1.5-q_0.99",
        "calibration_score_func": "TAAS_OFF3-s_1.5-q_0.99",
        "inference_score_func": "TAAS_OFF3-s_1.5-q_0.99",
        "inference_score_backend": "cython:tass_cython_distance",
        "reconstruction_mode": "posterior_mean",
        "offset": 3,
        "sigma": 1.5,
        "quantile": 0.99,
        "taas_variant": "canonical",
        "instantaneous_anomaly_score": 0.6,
        "instantaneous_normalized_score": 1.2,
        "rolling_policy": "none",
        "rolling_window": 1,
        "rolling_count": 1,
    }
    payload = pack_debug_state(
        Packer,
        msg_id=4,
        frame_id="back_view",
        stamp=SimpleNamespace(sec=1, nanosec=2),
        results={"PLeft": result},
        runtime_meta={"processing_fps": 5.5},
    )

    assert payload["runtime_meta"]["processing_fps"] == 5.5
    assert payload["latest_results"]["PLeft"]["inference_score_backend"].startswith(
        "cython:"
    )
    assert payload["latest_results"]["PLeft"]["calibration_taas_variant"] == "canonical"


def test_timeline_payload_keeps_aligned_frame_time_and_source_metadata():
    class Packer:
        @staticmethod
        def packb(payload, use_bin_type):
            assert use_bin_type is True
            return payload

    payload = pack_timeline_state(
        Packer,
        msg_id=12,
        frame_id="back_view",
        stamp=SimpleNamespace(sec=100, nanosec=500_000_000),
        histories={"PLeft": [0.4, 0.7]},
        results={},
        history_meta=[
            {"processed_frame": 1, "source_message_index": 10, "ros_elapsed_seconds": 0.0},
            {"processed_frame": 2, "source_message_index": 12, "ros_elapsed_seconds": 0.08},
        ],
        source_meta={"source_name": "Jul27_Scenario_8_0", "scenario_id": "8_0"},
        runtime_meta={"processing_fps": 8.0, "dropped_frames": 4},
    )

    assert payload["history_meta"][-1]["source_message_index"] == 12
    assert payload["source_meta"]["scenario_id"] == "8_0"
    assert payload["runtime_meta"]["processing_fps"] == 8.0


def test_topic_type_discovery_supports_raw_and_compressed():
    assert detect_topic_message_type({"sensor_msgs/msg/CompressedImage"}) == "compressed"
    assert detect_topic_message_type({"sensor_msgs/msg/Image"}) == "raw"
    assert detect_topic_message_type(set()) is None


def test_topic_type_discovery_rejects_ambiguous_topic():
    with pytest.raises(RuntimeError, match="both Image and CompressedImage"):
        detect_topic_message_type({
            "sensor_msgs/msg/CompressedImage",
            "sensor_msgs/msg/Image",
        })


def test_masks_are_loaded_from_yaml_mask_types(tmp_path):
    mask_path = tmp_path / "pleft.png"
    cv2.imwrite(str(mask_path), np.full((12, 16), 255, dtype=np.uint8))
    config = tmp_path / "config.yaml"
    config.write_text(
        "data:\n"
        f"  mask_types:\n    PLeft: {mask_path}\n"
        "models:\n  checkpoints: models\n  latent_dims: 32\n",
        encoding="utf-8",
    )
    args = parse_args([
        "--config", str(config),
        "--safety_areas", "PLeft",
        "--dataset_version", "TEST",
    ])
    args = load_live_settings(args)

    masks = load_live_masks(args)

    assert masks["PLeft"].shape == (12, 16)
    assert args.latent_dims == 32
    assert args.checkpoints.name == "models"


def test_detection_payload_is_json_serializable_and_keeps_score_metadata():
    result = {
        "safety_area": "PLeft",
        "anomaly_score": 0.6,
        "threshold": 0.5,
        "normalized_score": 1.2,
        "is_anomalous": True,
        "threshold_strategy": "percentile",
        "score_func": "TAAS_OFF3-s_1.5-q_0.99",
        "calibration_score_func": "TAAS_OFF3-s_1.5-q_0.99",
        "inference_score_func": "TAAS_OFF3-s_1.5-q_0.99",
        "inference_score_backend": "cython:tass_cython_distance",
        "reconstruction_mode": "posterior_mean",
        "offset": 3,
        "sigma": 1.5,
        "quantile": 0.99,
        "taas_variant": "canonical",
        "instantaneous_anomaly_score": 0.6,
        "instantaneous_normalized_score": 1.2,
        "rolling_policy": "mean",
        "rolling_window": 5,
        "rolling_count": 5,
    }
    stamp = SimpleNamespace(sec=12, nanosec=345)

    payload = detection_payload(
        msg_id=8,
        frame_id="back_view",
        stamp=stamp,
        results={"PLeft": result},
        processing_fps=8.5,
        source_fps=25.0,
        received_frames=10,
        processed_frames=8,
        dropped_frames=2,
    )

    json.dumps(payload)
    assert payload["any_anomalous"] is True
    assert payload["areas"]["PLeft"]["status"] == "UNEXPECTED"
    assert payload["areas"]["PLeft"]["offset"] == 3
    assert payload["areas"]["PLeft"]["rolling_policy"] == "mean"
