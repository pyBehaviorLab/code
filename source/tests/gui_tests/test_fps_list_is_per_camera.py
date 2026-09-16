"""Each row's rate list must come from THAT row's camera.

``_offered_rates_for`` and ``_rejected_rates_for`` took only a mode and then
resolved the camera themselves, from the Active-camera picker. The picker
names the camera the shared Capture card is pointed at, which has nothing to
do with the row being filled, so on a multi-camera rig every row was given
the active camera's rates: rows 2 and 3 listed row 1's modes.

It reads as incoherent rather than as a wrong lookup, because Detect reports
each camera correctly on the way past and the combo beside it then lists
another camera's rates. The two are reading different cameras.

The rig that found it: a 1080p camera offering up to 60 fps beside one whose
slowest interval is 15.
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

KEY_A = "unit-test-per-camera-A"
KEY_B = "unit-test-per-camera-B"

#: Deliberately disjoint, so a list can only have come from one of them.
OFFERED_A = [OfferedMode(640, 480, "mjpeg", (15.0, 20.0, 30.0))]
OFFERED_B = [OfferedMode(640, 480, "mjpeg", (5.0, 7.0, 10.0, 50.0, 60.0))]

MODE = {"width": 640, "height": 480, "fps": 24.6, "backend": "dshow"}


@pytest.fixture(autouse=True)
def _clean():
    yield
    p = (pathlib.Path(os.environ.get("LOCALAPPDATA") or pathlib.Path.home())
         / "pybehaviorlab" / "camera_calibrations.json")
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        if any(d.pop(k, None) is not None for k in (KEY_A, KEY_B)):
            p.write_text(json.dumps(d, indent=2), encoding="utf-8")


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def two_cameras(monkeypatch):
    """Camera A is ACTIVE; camera B is the one a row is being filled for."""
    store.put_offered(KEY_A, OFFERED_A)
    store.put_offered(KEY_B, OFFERED_B)
    keys = {"camA": KEY_A, "camB": KEY_B}
    monkeypatch.setattr(_Dlg, "_active_capture_cam", lambda self: "camA")
    monkeypatch.setattr(_Dlg, "_firstCameraIdInTable", lambda self: "camA")
    monkeypatch.setattr(_Dlg, "_backendForCameraId", lambda self, c: "opencv")
    monkeypatch.setattr(_Dlg, "_selected_format_for", lambda self, c: "mjpeg")
    monkeypatch.setattr(_Dlg, "_resolve_identity_cached",
                        lambda self, c, b: {"unique_id": keys.get(str(c), "")})
    monkeypatch.setattr(_Dlg, "_rejected_rates_for",
                        lambda self, data, cam_id=None: set())
    return _Dlg.__new__(_Dlg)


def _rates(dlg, cam_id):
    res, fps = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    res.addItem("m", MODE)
    dlg._repopulate_fps_combo(res_combo=res, fps_combo=fps, cam_id=cam_id)
    return [fps.itemData(i) for i in range(fps.count())]


def test_a_rows_list_is_its_own_cameras(qapp, two_cameras):
    """The regression. Without ``cam_id`` this returned camera A's rates."""
    assert _rates(two_cameras, "camB") == [5, 7, 10, 50, 60]


def test_the_other_row_is_unaffected(qapp, two_cameras):
    assert _rates(two_cameras, "camA") == [15, 20, 30]


def test_two_rows_do_not_agree_when_their_cameras_differ(qapp, two_cameras):
    """The symptom was every row showing one list; that must be impossible."""
    assert _rates(two_cameras, "camA") != _rates(two_cameras, "camB")


def test_the_shared_card_still_follows_the_active_camera(qapp, two_cameras):
    """No ``cam_id`` means the shared Capture card, whose subject IS the
    active camera. That fallback is correct and must not be lost."""
    assert _rates(two_cameras, None) == [15, 20, 30]


def test_the_resolution_label_is_also_per_camera(qapp, two_cameras):
    """The Resolution cell prints the ceiling beside the size, from the same
    lookup, so it drifted to the active camera in exactly the same way."""
    res, fps = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    two_cameras._populateResolutionCombo(
        [(640, 480, 24.6, "dshow")], res_combo=res, fps_combo=fps,
        compact=True, cam_id="camB")
    # Camera B tops out at 60; camera A at 30. The room measured 24.6 and
    # that number must appear nowhere.
    assert "60" in res.itemText(0), res.itemText(0)
    assert "24.6" not in res.itemText(0), res.itemText(0)
