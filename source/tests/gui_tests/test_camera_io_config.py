"""Per-camera scientific I/O config: exposure/gain + trigger + line-output (strobe)
+ sync role. Phase 1 of the illumination/sync work, the schema only, so it is
fully headless. The backend apply-hooks and UI come later.
"""
from __future__ import annotations

from source.video.framebus.types import (
    CameraConfig, LineOutputConfig, TriggerConfig,
)


# ── defaults reproduce today's behaviour ─────────────────────────────────

def test_defaults_are_standalone_freerun():
    c = CameraConfig(camera_id="0")
    assert c.exposure_us is None and c.gain_db is None
    assert c.trigger.mode == "freerun"
    assert c.line_output.enabled is False
    assert c.sync_role == "none"


# ── round-trip ────────────────────────────────────────────────────────────

def test_full_roundtrip():
    c = CameraConfig(
        camera_id="21290043",
        camera_backend="spinnaker",
        exposure_us=4000.0,
        gain_db=2.0,
        trigger=TriggerConfig(mode="hardware", source="Line2",
                              edge="falling", delay_us=15.0),
        line_output=LineOutputConfig(enabled=True, line="Line1",
                                     source="exposure_active", inverted=True),
        sync_role="secondary",
    )
    back = CameraConfig.from_json(c.to_json())
    assert back.exposure_us == 4000.0
    assert back.gain_db == 2.0
    assert back.trigger == c.trigger
    assert back.line_output == c.line_output
    assert back.sync_role == "secondary"
    assert back.camera_backend == "spinnaker"


def test_nested_json_shapes():
    t = TriggerConfig(mode="hardware", source="Line3", edge="rising", delay_us=5)
    assert TriggerConfig.from_json(t.to_json()) == t
    lo = LineOutputConfig(enabled=True, line="Line1")
    assert LineOutputConfig.from_json(lo.to_json()) == lo


# ── legacy back-compat ─────────────────────────────────────────────────────

def test_legacy_external_trigger_maps_to_hardware():
    # A project saved before the structured schema had a top-level bool.
    old = {"camera_id": "s", "camera_backend": "ximea", "external_trigger": True}
    c = CameraConfig.from_json(old)
    assert c.trigger.mode == "hardware"


def test_legacy_no_trigger_is_freerun():
    old = {"camera_id": "s", "external_trigger": False}
    assert CameraConfig.from_json(old).trigger.mode == "freerun"


def test_new_trigger_block_wins_over_legacy_flag():
    raw = {"camera_id": "s", "external_trigger": True,
           "trigger": {"mode": "freerun"}}
    assert CameraConfig.from_json(raw).trigger.mode == "freerun"


# ── per-camera isolation (the audit's global/transient flaw is gone) ───────

def test_two_cameras_keep_independent_settings():
    a = CameraConfig(camera_id="a", exposure_us=2000.0, sync_role="primary")
    b = CameraConfig(camera_id="b", exposure_us=8000.0, sync_role="secondary")
    # No shared mutable default leaked between instances.
    a.trigger.source = "Line2"
    assert b.trigger.source == ""
    assert a.exposure_us != b.exposure_us
    assert (a.sync_role, b.sync_role) == ("primary", "secondary")


def test_missing_keys_tolerated():
    c = CameraConfig.from_json({"camera_id": "x"})
    assert c.trigger.mode == "freerun" and c.line_output.enabled is False
