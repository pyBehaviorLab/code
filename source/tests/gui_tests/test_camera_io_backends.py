"""Phase 2, FLIR/Ximea strobe + trigger recipes.

Verifies the exact SDK call sequence each backend emits for a given config,
against the primary-source recipe (FLIR app note TAN2016008 + XIMEA wiki):
strobe = LineSource ExposureActive; secondary = FrameStart / RisingEdge /
TriggerOverlap ReadOut, TriggerMode On last. The SDKs aren't installed here, so
we bypass __init__ and patch the node setters / camera handle, this checks the
*recipe logic*; the physical opto behaviour is the rig sign-off.
"""
from __future__ import annotations

from source.video.cameras.spinnaker import SpinnakerCamera
from source.video.cameras.ximea import XimeaCamera


# ── FLIR / Spinnaker ──────────────────────────────────────────────────────

def _flir():
    cam = SpinnakerCamera.__new__(SpinnakerCamera)
    cam._external_trigger = False
    calls = {"enum": [], "bool": [], "float": []}
    cam._set_enum = lambda node, entry: calls["enum"].append((node, entry))
    cam._set_bool = lambda node, val: calls["bool"].append((node, val))
    cam._set_float = lambda node, val: calls["float"].append((node, val))
    return cam, calls


def test_flir_hardware_trigger_recipe():
    cam, calls = _flir()
    cam._apply_trigger({"mode": "hardware", "source": "Line3",
                        "edge": "rising", "delay_us": 10})
    e = calls["enum"]
    # Off first, On last (order matters).
    assert e[0] == ("TriggerMode", "Off")
    assert e[-1] == ("TriggerMode", "On")
    assert ("TriggerSelector", "FrameStart") in e
    assert ("TriggerSource", "Line3") in e
    assert ("TriggerActivation", "RisingEdge") in e
    assert ("TriggerOverlap", "ReadOut") in e          # required for full rate
    assert ("TriggerDelay", 10.0) in calls["float"]
    assert cam._external_trigger is True


def test_flir_falling_edge_and_default_source():
    cam, calls = _flir()
    cam._apply_trigger({"mode": "hardware", "edge": "falling"})
    assert ("TriggerActivation", "FallingEdge") in calls["enum"]
    assert ("TriggerSource", "Line3") in calls["enum"]   # default input line


def test_flir_freerun_leaves_trigger_off():
    cam, calls = _flir()
    cam._apply_trigger({"mode": "freerun"})
    assert calls["enum"] == [("TriggerMode", "Off")]
    assert cam._external_trigger is False


def test_flir_strobe_recipe():
    cam, calls = _flir()
    cam._apply_line_output({"enabled": True, "line": "Line1",
                            "source": "exposure_active", "inverted": False})
    e = calls["enum"]
    assert ("LineSelector", "Line1") in e
    assert ("LineMode", "Output") in e
    assert ("LineSource", "ExposureActive") in e         # the verified enum
    assert ("LineInverter", False) in calls["bool"]
    assert ("V3_3Enable", True) in calls["bool"]         # non-isolated drive


def test_flir_strobe_disabled_is_noop():
    cam, calls = _flir()
    cam._apply_line_output({"enabled": False})
    assert calls["enum"] == [] and calls["bool"] == []


def test_flir_hardware_trigger_enables_trigger_mode():
    cam, calls = _flir()
    cam._apply_trigger({"mode": "hardware"})
    assert ("TriggerMode", "On") in calls["enum"]


# ── Ximea ─────────────────────────────────────────────────────────────────

class _FakeXi:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def rec(*a):
            self.calls.append((name, a))
        return rec


def _ximea():
    cam = XimeaCamera.__new__(XimeaCamera)
    cam._streaming = False
    cam._external_trigger = False
    cam._cam = _FakeXi()
    return cam


def test_ximea_hardware_trigger_recipe():
    cam = _ximea()
    cam._apply_trigger({"mode": "hardware", "edge": "rising"})
    c = cam._cam.calls
    assert ("set_gpi_mode", ("XI_GPI_TRIGGER",)) in c
    assert ("set_trigger_source", ("XI_TRG_EDGE_RISING",)) in c
    assert ("set_trigger_selector", ("XI_TRG_SEL_FRAME_START",)) in c
    assert cam._external_trigger is True


def test_ximea_freerun_is_trg_off():
    cam = _ximea()
    cam._apply_trigger({"mode": "freerun"})
    assert ("set_trigger_source", ("XI_TRG_OFF",)) in cam._cam.calls
    assert cam._external_trigger is False


def test_ximea_strobe_recipe():
    cam = _ximea()
    cam._apply_line_output({"enabled": True, "inverted": False})
    c = cam._cam.calls
    assert ("set_gpo_selector", ("XI_GPO_PORT1",)) in c
    assert ("set_gpo_mode", ("XI_GPO_EXPOSURE_ACTIVE",)) in c


def test_ximea_strobe_inverted():
    cam = _ximea()
    cam._apply_line_output({"enabled": True, "inverted": True})
    assert ("set_gpo_mode", ("XI_GPO_EXPOSURE_ACTIVE_NEG",)) in cam._cam.calls


def test_ximea_strobe_disabled_is_noop():
    cam = _ximea()
    cam._apply_line_output({"enabled": False})
    assert cam._cam.calls == []
