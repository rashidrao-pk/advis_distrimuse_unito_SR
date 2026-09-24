"""Status-selection tests for the ADVIS dashboard viewer."""

from pathlib import Path
import sys

import numpy as np


DASHBOARD = Path(__file__).resolve().parents[1] / "zenoh_dashboard"
sys.path.insert(0, str(DASHBOARD))

from dashboard_viewer import (  # noqa: E402
    STATUS_COLORS,
    connection_status,
    draw_connection_banner,
)


def status_key(**overrides):
    values = {
        "now": 10.0,
        "inference_last_received": None,
        "camera_last_received": None,
        "started_at": 0.0,
        "inference_timeout": 3.0,
        "camera_timeout": 2.0,
        "camera_monitor_available": True,
    }
    values.update(overrides)
    return connection_status(**values)[0]


def test_fresh_inference_is_running():
    assert status_key(inference_last_received=9.0) == "running"


def test_live_camera_without_detections_means_inference_stopped():
    assert status_key(camera_last_received=9.0) == "inference_stopped"


def test_stale_camera_and_inference_means_camera_offline():
    assert status_key(
        inference_last_received=4.0,
        camera_last_received=4.0,
    ) == "camera_offline"


def test_startup_grace_period_shows_waiting():
    assert status_key(now=1.0) == "waiting"


def test_missing_ros_monitor_does_not_claim_camera_is_offline():
    assert status_key(camera_monitor_available=False) == "camera_unknown"


def test_running_state_uses_border_and_details_instead_of_top_banner():
    dashboard = np.full((1000, 1600, 3), 235, dtype=np.uint8)
    rendered = draw_connection_banner(
        dashboard,
        "running",
        "INFERENCE RUNNING",
        "Camera stream and detections are live",
        stale=False,
    )

    green = np.asarray(STATUS_COLORS["running"], dtype=np.uint8)
    assert np.array_equal(rendered[4, 4], green)
    assert np.array_equal(rendered[20, 800], dashboard[20, 800])
    details_title_row = rendered[516:560, 816:1584]
    assert np.any(np.all(details_title_row == green, axis=2))
