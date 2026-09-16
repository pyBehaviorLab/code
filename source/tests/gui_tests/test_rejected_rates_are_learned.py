"""The rate list offers rates the camera actually has.

A UVC camera exposes a handful of discrete rates and rounds silently to the
nearest one it has, reporting the request back unchanged. The picker builds an
arithmetic ladder up to the measured ceiling, which is a SUPERSET of them.
Measured on this rig at 640x480, where the ceiling is 29.9:

    ladder offered   10, 15, 20, 25, 30
    camera has       15, 20, 30
    asked 25 -> delivered 30.03      asked 10 -> delivered 14.98

So picking 25 always gave 30, and nothing said so. Probing every rate at every
size would multiply an already slow calibration, so the rig learns from use:
each time a rate turns out not to be offered it stops being offered.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest
from PySide6 import QtWidgets

from source.gui.dialogs.camera_connect import CameraConnectDialog as _Dlg
from source.video.cameras import calibration_store as store

KEY = "unit-test-rates"


@pytest.fixture(autouse=True)
def _entry():
    """A store entry to hang rejections on, removed afterwards."""
    store.put(KEY, "opencv", "t", [[640, 480, 29.9]], {},
              variants={"dshow": [[640, 480, 29.9]],
                        "msmf": [[640, 480, 29.9]]})
    yield
    p = (pathlib.Path(os.environ.get("LOCALAPPDATA") or pathlib.Path.home())
         / "pybehaviorlab" / "camera_calibrations.json")
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.pop(KEY, None) is not None:
            p.write_text(json.dumps(d, indent=2), encoding="utf-8")


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_a_rate_the_camera_rounds_away_is_remembered():
    store.note_rate_rejected(KEY, "dshow", (640, 480), 25, 30.03)
    assert store.get_rejected_rates(KEY, "dshow", (640, 480)) == {25}


def test_rejections_are_per_door():
    """The door changes what the camera offers, so it changes the answer."""
    store.note_rate_rejected(KEY, "dshow", (640, 480), 25, 30.0)
    assert store.get_rejected_rates(KEY, "msmf", (640, 480)) == set()


def test_rejections_are_per_size():
    store.note_rate_rejected(KEY, "dshow", (640, 480), 25, 30.0)
    assert store.get_rejected_rates(KEY, "dshow", (1920, 1080)) == set()


def test_the_same_rate_is_not_recorded_twice():
    for _ in range(3):
        store.note_rate_rejected(KEY, "dshow", (640, 480), 10, 15.0)
    assert store.get_rejected_rates(KEY, "dshow", (640, 480)) == {10}


def test_an_unknown_camera_reports_nothing_and_does_not_raise():
    assert store.get_rejected_rates("no-such-camera", "dshow", (640, 480)) == set()
    store.note_rate_rejected("no-such-camera", "dshow", (640, 480), 25, 30.0)


def test_the_picker_stops_offering_what_the_camera_does_not_have(qapp,
                                                                monkeypatch):
    """The rig case, end to end through the real rate ladder."""
    for bad in (5, 10, 25):
        store.note_rate_rejected(KEY, "dshow", (640, 480), bad, 30.0)

    d = _Dlg.__new__(_Dlg)
    monkeypatch.setattr(_Dlg, "_rejected_rates_for",
                        lambda self, data, cam_id=None: store.get_rejected_rates(
                            KEY, (data or {}).get("backend", ""),
                            ((data or {}).get("width"), (data or {}).get("height"))))
    res, fps = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    res.addItem("640x480", {"width": 640, "height": 480, "fps": 29.9,
                            "backend": "dshow"})
    d._repopulate_fps_combo(res_combo=res, fps_combo=fps)
    offered = [fps.itemData(i) for i in range(fps.count())]
    assert offered == [15, 20, 30], offered


def test_the_ladder_never_ends_up_empty(qapp, monkeypatch):
    """Rejecting everything would leave a cell with no rate at all.

    A camera that cannot be set to anything is a different failure and must
    not be produced by this filter; the unfiltered ladder is kept instead.
    """
    for bad in (10, 15, 20, 25, 30):
        store.note_rate_rejected(KEY, "dshow", (640, 480), bad, 30.0)
    d = _Dlg.__new__(_Dlg)
    monkeypatch.setattr(_Dlg, "_rejected_rates_for",
                        lambda self, data, cam_id=None: store.get_rejected_rates(
                            KEY, "dshow", (640, 480)))
    res, fps = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    res.addItem("640x480", {"width": 640, "height": 480, "fps": 29.9,
                            "backend": "dshow"})
    d._repopulate_fps_combo(res_combo=res, fps_combo=fps)
    assert fps.count() > 0 and fps.isEnabled()


# ── a shortfall is a condition, not a missing rate ───────────────────────
#
# Measured on this rig at 1920x1080, minutes apart: 30.13 fps then 24.88 fps,
# the same camera and the same mode. Auto-exposure lengthens integration in
# dimmer light, so the delivered rate tracks the room. Banning 30 fps because
# the light dropped once would strip a rate the camera really has, for good.

def _run_rule(monkeypatch, want, got):
    """Run the real rule and report whether the rate was recorded as missing.

    Only ``measure_delivered_fps`` is replaced, so the decision under test is
    the shipped one; ``monkeypatch`` puts the module global back afterwards.
    """
    from source.video.cameras import opencv as _ocv
    cam = _ocv.OpenCVCamera.__new__(_ocv.OpenCVCamera)
    recorded = []
    cam._remember_rate_rejected = lambda a, g: recorded.append((a, g))
    cam._pending_fps = want
    cam._requested_width, cam._requested_height = 640, 480
    cam._camera_id = 0
    cam.delivered_fps = cam.fps_warning = cam._delivered_for_fps = None
    cam.get_available_images = lambda: {"images": []}
    monkeypatch.setattr(_ocv, "measure_delivered_fps",
                        lambda *a, **k: got)
    _ocv.OpenCVCamera._measure_delivered_rate(cam)
    return recorded, cam


def test_delivering_more_than_asked_means_the_rate_does_not_exist(monkeypatch):
    """25 asked, 30.03 delivered: the camera rounded up to a rate it has."""
    recorded, _cam = _run_rule(monkeypatch, 25, 30.03)
    assert recorded == [(25.0, 30.03)], recorded


def test_delivering_less_than_asked_is_not_held_against_the_camera(monkeypatch):
    """30 asked, 24.7 delivered: the light is limiting it, not the camera."""
    recorded, cam = _run_rule(monkeypatch, 30, 24.7)
    assert recorded == [], (
        "a rate the camera really has was banned because the room was dim")
    assert cam.fps_warning, "but the shortfall is still reported"


def test_an_honoured_rate_is_neither_banned_nor_complained_about(monkeypatch):
    recorded, cam = _run_rule(monkeypatch, 30, 30.02)
    assert recorded == [] and cam.fps_warning is None


# ── a rejection belongs to ONE format, and expires when the camera is asked ──
#
# The rate a size can reach depends on the format: this rig's camera
# advertises 640x480 at 120 fps in MJPEG and at 30 in YUY2. Filed without the
# format, "30 was not honoured at 640x480" was learned while the camera was
# open in YUY2 and then deleted 30 fps from the MJPEG list, which is the one
# rate a behavioural rig wants. The combo showed a list the Detect beside it
# had just contradicted.

def test_a_rejection_in_one_format_does_not_touch_another(monkeypatch):
    store.put(KEY, "dshow", "cam", [(640, 480, 30.0)])
    store.note_rate_rejected(KEY, "dshow", (640, 480), 30, 15.0,
                             pixel_format="yuy2")
    assert store.get_rejected_rates(KEY, "dshow", (640, 480),
                                    pixel_format="yuy2") == {30}
    assert store.get_rejected_rates(KEY, "dshow", (640, 480),
                                    pixel_format="mjpeg") == set(), (
        "a YUY2 rejection deleted a rate from the MJPEG list")


def test_a_fresh_detect_discards_what_the_old_one_learned():
    """Pressing Detect means the stored numbers are not trusted, so the
    rejections learned from them must not outlive the measurement."""
    store.put(KEY, "dshow", "cam", [(640, 480, 30.0)])
    store.note_rate_rejected(KEY, "dshow", (640, 480), 30, 15.0,
                             pixel_format="mjpeg")
    assert store.get_rejected_rates(KEY, "dshow", (640, 480),
                                    pixel_format="mjpeg") == {30}
    store.put(KEY, "dshow", "cam", [(640, 480, 60.0)], replace=True)
    assert store.get_rejected_rates(KEY, "dshow", (640, 480),
                                    pixel_format="mjpeg") == set(), (
        "a fresh Detect left the old rejections deleting rates from it")


def test_a_fresh_detect_does_not_keep_the_higher_old_reading():
    """The merge exists so a failed probe cannot erase good data, but it made
    a wrongly-high measurement permanent: re-measuring could never lower it."""
    store.put(KEY, "dshow", "cam", [], variants={"dshow": [(640, 480, 60.0)]})
    store.put(KEY, "dshow", "cam", [], variants={"dshow": [(640, 480, 28.0)]},
              replace=True)
    got = dict(store.get_variants(KEY)).get("dshow") or []
    assert [round(m[2]) for m in got] == [28], f"kept the stale reading: {got}"


def test_a_format_blind_rejection_from_the_old_shape_is_not_trusted():
    """There is no way to tell which format it was recorded against, and
    applying it to all of them is the bug itself."""
    store.put(KEY, "dshow", "cam", [(640, 480, 30.0)])
    data = store.load()
    data[KEY]["rejected_fps"] = {"dshow": {"640x480": [30]}}   # pre-format key
    data[KEY]["rejected_at"] = "2099-01-01T00:00:00"
    store.save(data)
    assert store.get_rejected_rates(KEY, "dshow", (640, 480),
                                    pixel_format="mjpeg") == set()
