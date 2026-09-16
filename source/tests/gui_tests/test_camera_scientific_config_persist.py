"""Scientific-camera (FLIR/Ximea) Advanced-tab settings survive the project
config round-trip.

The CameraConfig↔CameraEntry bridge must read exposure/gain from where
``to_json`` emits them (top-level, not ``extra``) and must carry slots for the
structured trigger / line-output / sync-role. Without both, a FLIR camera's
hardware-trigger + strobe + sync config is lost on project save/load, surviving
only in the per-machine camera.json. These pin the full round-trip.
"""
from __future__ import annotations

import json
from dataclasses import asdict

from source.config import experiment as ex
from source.video.framebus.types import (
    CameraConfig,
    LineOutputConfig,
    TriggerConfig,
)


def _flir_config():
    return CameraConfig(
        camera_id="21099999", selected_resolution=(1440, 1080), selected_fps=60,
        probed_modes=[(1440, 1080, 60.0)], grayscale=True, frame_strategy="remux",
        camera_backend="spinnaker", capture_format="mono8",
        exposure_us=5000.0, gain_db=3.5,
        trigger=TriggerConfig(mode="hardware", source="Line3",
                              edge="falling", delay_us=100),
        line_output=LineOutputConfig(enabled=True, line="Line1",
                                     source="frame_active", inverted=True),
        sync_role="primary", user_applied=True)


def _project_round_trip(cfg):
    """cfg -> CameraEntry -> JSON on disk -> CameraEntry -> cfg (the exact
    project save/load bridge, including a real json dump/load)."""
    entry = ex._camera_json_to_entry(cfg.camera_id, cfg.to_json())
    on_disk = json.loads(json.dumps(asdict(entry)))
    entry2 = ex.CameraEntry.from_dict(on_disk)
    return CameraConfig.from_json(ex._entry_to_camera_json(entry2))


def test_exposure_gain_survive_project_config():
    c2 = _project_round_trip(_flir_config())
    assert c2.exposure_us == 5000.0
    assert c2.gain_db == 3.5


def test_trigger_survives_project_config():
    c2 = _project_round_trip(_flir_config())
    assert c2.trigger.to_json() == {
        "mode": "hardware", "source": "Line3", "edge": "falling", "delay_us": 100.0}


def test_line_output_survives_project_config():
    c2 = _project_round_trip(_flir_config())
    assert c2.line_output.to_json() == {
        "enabled": True, "line": "Line1", "source": "frame_active", "inverted": True}


def test_sync_role_survives_project_config():
    assert _project_round_trip(_flir_config()).sync_role == "primary"


def test_opencv_camera_still_defaults_cleanly():
    # A plain webcam has no scientific I/O; it must round-trip to the neutral
    # defaults (free-run trigger, disabled strobe, no sync), not carry junk.
    c = CameraConfig(camera_id="0", selected_resolution=(640, 480),
                     selected_fps=30, probed_modes=[(640, 480, 30.0)],
                     camera_backend="opencv", user_applied=True)
    c2 = _project_round_trip(c)
    assert c2.trigger.mode == "freerun"
    assert c2.line_output.enabled is False
    assert c2.sync_role == "none"
    assert c2.exposure_us is None and c2.gain_db is None


def test_camera_preference_dataclass_round_trips():
    pref = ex.CameraPreference(
        width=800, height=600, codec="mjpeg", exposure_us=1200, gain_db=1.0,
        trigger={"mode": "software", "source": "", "edge": "rising", "delay_us": 0.0},
        line_output={"enabled": False, "line": "", "source": "exposure_active",
                     "inverted": False},
        sync_role="secondary")
    back = ex.CameraPreference.from_dict(json.loads(json.dumps(asdict(pref))))
    assert back == pref


def test_legacy_config_without_scientific_fields_loads():
    # Old project entries have no trigger/line_output/sync_role keys, must load
    # with neutral defaults, never raise.
    pref = ex.CameraPreference.from_dict({"width": 640, "height": 480})
    assert pref.trigger is None
    assert pref.line_output is None
    assert pref.sync_role == "none"
