"""The rate list is what the camera offers, not an arithmetic guess.

The ladder it replaces was ``FPS_MIN, +STEP, ... , measured ceiling``. Two
things are wrong with that and the enumeration fixes both:

  * it has no idea where the camera STARTS. It offered 10 and 5 fps to a
    camera whose slowest interval is 15, and both round up to 15 in practice.
  * its ceiling is a measurement, and a measurement is the camera multiplied
    by the room. The same mode read 30.13 fps and then 24.88 minutes apart
    because auto-exposure lengthens integration in dimmer light, so a camera
    calibrated in the dark would never again be offered the 30 it has.

On Linux the enumeration is the camera's exact interval list. On Windows
DirectShow reports only bounds, so the list inside them is still narrowed by
the rates the camera has been seen to round away.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest
from PySide6 import QtWidgets

from source.gui.dialogs.camera_connect import CameraConnectDialog as _Dlg
from source.video.cameras import calibration_store as store
from source.video.cameras.enumerate_modes import OfferedMode

KEY = "unit-test-fps-list"


@pytest.fixture(autouse=True)
def _clean():
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


#: cam3cbb52d3 as FFmpeg reports it: nothing below 15 fps.
OFFERED = [OfferedMode(1920, 1080, "mjpeg", (15.0, 20.0, 24.0, 25.0, 30.0)),
           OfferedMode(640, 480, "mjpeg", (15.0, 20.0, 24.0, 25.0, 30.0)),
           OfferedMode(640, 480, "yuy2", (5.0,))]


def _combo_for(monkeypatch, mode_data, fmt="mjpeg", rejected=frozenset()):
    """The FPS cell for one mode, through the real populator."""
    store.put_offered(KEY, OFFERED)
    d = _Dlg.__new__(_Dlg)
    monkeypatch.setattr(_Dlg, "_active_capture_cam", lambda self: "camA")
    monkeypatch.setattr(_Dlg, "_backendForCameraId", lambda self, c: "opencv")
    monkeypatch.setattr(_Dlg, "_resolve_identity_cached",
                        lambda self, c, b: {"unique_id": KEY})
    monkeypatch.setattr(_Dlg, "_selected_format_for", lambda self, c: fmt)
    monkeypatch.setattr(_Dlg, "_rejected_rates_for",
                        lambda self, data, cam_id=None: set(rejected))
    res, fps = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    res.addItem("m", mode_data)
    d._repopulate_fps_combo(res_combo=res, fps_combo=fps)
    return [fps.itemData(i) for i in range(fps.count())], fps


_MODE = {"width": 640, "height": 480, "fps": 29.9, "backend": "dshow"}


def test_the_list_is_the_cameras_own_intervals(qapp, monkeypatch):
    offered, combo = _combo_for(monkeypatch, _MODE)
    assert offered == [15, 20, 24, 25, 30], offered
    assert combo.isEnabled()


def test_rates_below_the_cameras_slowest_are_not_offered(qapp, monkeypatch):
    """The arithmetic ladder offered 10 and 5 to a camera that starts at 15."""
    offered, _ = _combo_for(monkeypatch, _MODE)
    assert 10 not in offered and 5 not in offered, offered


def test_a_rate_seen_to_round_away_is_dropped(qapp, monkeypatch):
    """DirectShow reports bounds, so the list inside them is still learned."""
    offered, _ = _combo_for(monkeypatch, _MODE, rejected={25})
    assert offered == [15, 20, 24, 30], offered


def test_the_format_changes_the_answer(qapp, monkeypatch):
    """The same size is 30 fps compressed and 5 uncompressed on this camera."""
    offered, _ = _combo_for(monkeypatch, _MODE, fmt="yuy2")
    assert offered == [5], offered


def test_an_un_enumerated_camera_falls_back_to_the_ladder(qapp, monkeypatch):
    """Empty means nobody has asked the camera, not that it has no rates."""
    d = _Dlg.__new__(_Dlg)
    monkeypatch.setattr(_Dlg, "_offered_rates_for", lambda self, data, cam_id=None: ())
    monkeypatch.setattr(_Dlg, "_rejected_rates_for", lambda self, data, cam_id=None: set())
    res, fps = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    res.addItem("m", _MODE)
    d._repopulate_fps_combo(res_combo=res, fps_combo=fps)
    assert [fps.itemData(i) for i in range(fps.count())] == [10, 15, 20, 25, 30]


def test_everything_rejected_still_leaves_a_usable_list(qapp, monkeypatch):
    """A cell with no rate at all is a different failure, and not this one's
    to produce."""
    offered, combo = _combo_for(monkeypatch, _MODE,
                                rejected={15, 20, 24, 25, 30})
    assert offered and combo.isEnabled()


def test_a_size_the_camera_does_not_offer_falls_back(qapp, monkeypatch):
    offered, _ = _combo_for(monkeypatch,
                            {"width": 3840, "height": 2160, "fps": 20.0,
                             "backend": "dshow"})
    assert offered, "an unoffered size must not empty the cell"
