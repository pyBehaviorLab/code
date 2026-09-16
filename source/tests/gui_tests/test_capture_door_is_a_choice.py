"""The capture door is the operator's choice, not a side effect.

The Backend cell used to name the camera FAMILY (OpenCV / Spinnaker / XIMEA),
which on a UVC rig reads "OpenCV" on every row and tells nobody anything. The
door the camera is actually opened through (DirectShow / Media Foundation /
V4L2 / GStreamer) was picked automatically and hidden inside the resolution
combo's item data, so choosing a picture size silently chose a backend too.

It is the setting that decides what the camera delivers. Measured on this rig
at 640x480: one camera gave 19.9 fps through DirectShow and 29.9 through Media
Foundation; another gave 28.3 through DirectShow and 23.9 through Media
Foundation. No per-platform constant is right for both, so it is chosen and
measured.
"""
from __future__ import annotations

import pytest

from source.gui.dialogs.camera_connect import CameraConnectDialog as _Dlg
from source.video.cameras import calibration_store as store

KEY = "unit-door-choice"


@pytest.fixture(autouse=True)
def _clean():
    yield
    d = store.load()
    if d.pop(KEY, None) is not None:
        store.save(d)


# ── what the cell offers ─────────────────────────────────────────────────

def test_the_doors_offered_are_this_platforms():
    ids = [d for _label, d in _Dlg._available_doors()]
    assert ids, "no doors offered at all"
    import os
    if os.name == "nt":
        assert ids == ["dshow", "msmf"]
    else:
        assert "dshow" not in ids and "msmf" not in ids


def test_the_door_ids_are_the_ones_the_store_files_under():
    """A door chosen here has to match the door a measurement was filed
    under, or the two tables can never be joined."""
    from source.video.cameras import opencv as O
    offered = {d for _l, d in _Dlg._available_doors()}
    assert offered <= set(O._uvc_backends()), (
        f"{offered - set(O._uvc_backends())} cannot be probed")


def test_the_default_door_is_a_real_one():
    assert _Dlg._default_door() in {d for _l, d in _Dlg._available_doors()}


# ── a Detect answers for the door it measured ────────────────────────────

def test_detecting_one_door_leaves_the_other_alone():
    """Measuring Media Foundation is not a statement about DirectShow."""
    store.put(KEY, "dshow", "c", [], variants={"dshow": [(640, 480, 29.0)]},
              replace=True, pixel_format="mjpeg")
    store.put(KEY, "msmf", "c", [], variants={"msmf": [(640, 480, 19.0)]},
              replace=True, pixel_format="mjpeg")
    mjpeg = store.get_variants_for_format(KEY, "mjpeg")
    assert sorted(mjpeg) == ["dshow", "msmf"], mjpeg
    assert mjpeg["dshow"][0][2] == 29.0, "the DirectShow measurement was lost"


def test_redetecting_a_door_replaces_only_that_door():
    store.put(KEY, "dshow", "c", [], variants={"dshow": [(640, 480, 29.0)]},
              replace=True, pixel_format="mjpeg")
    store.put(KEY, "msmf", "c", [], variants={"msmf": [(640, 480, 19.0)]},
              replace=True, pixel_format="mjpeg")
    store.put(KEY, "dshow", "c", [], variants={"dshow": [(640, 480, 12.0)]},
              replace=True, pixel_format="mjpeg")
    mjpeg = store.get_variants_for_format(KEY, "mjpeg")
    assert mjpeg["dshow"][0][2] == 12.0, "the re-measurement did not take"
    assert mjpeg["msmf"][0][2] == 19.0, "it also wiped a door it never opened"


def test_a_door_measured_in_one_format_does_not_appear_in_another():
    """This camera's 640x480 is 120 fps in MJPEG and 30 in YUY2, so an MJPEG
    figure standing in for a YUY2 entry states a rate nobody measured."""
    store.put(KEY, "msmf", "c", [], variants={"msmf": [(640, 480, 19.0)]},
              replace=True, pixel_format="mjpeg")
    store.put(KEY, "dshow", "c", [], variants={"dshow": [(640, 480, 31.0)]},
              replace=True, pixel_format="yuy2")
    yuy2 = store.get_variants_for_format(KEY, "yuy2")
    assert sorted(yuy2) == ["dshow"], (
        f"a door measured only in MJPEG leaked into YUY2: {yuy2}")


# ── the cell is per ROW ──────────────────────────────────────────────────

def test_the_door_is_read_from_the_row_not_the_camera(monkeypatch):
    """A shared CCTV camera sits on several rows. Looking the door up by
    camera id answered with the first row's for all of them, so editing the
    second row appeared to do nothing."""
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    d = _Dlg.__new__(_Dlg)
    table = QtWidgets.QTableWidget(2, 10)
    d.table = table
    for row, door in ((0, "dshow"), (1, "msmf")):
        c = QtWidgets.QComboBox()
        for label, value in _Dlg._available_doors():
            c.addItem(label, value)
        i = c.findData(door)
        if i >= 0:
            c.setCurrentIndex(i)
        table.setCellWidget(row, _Dlg.COL_BACKEND, c)
    assert d._doorForRow(0) != d._doorForRow(1), (
        "both rows answered with the same door")
    assert d._doorForRow(1) == "msmf"


# ── a scientific camera has no door ──────────────────────────────────────

def _cell_for(family):
    """One Backend cell, filled for a camera of ``family``."""
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    d = _Dlg.__new__(_Dlg)
    d._cam_family = {"camX": family}
    combo = QtWidgets.QComboBox()
    d._fill_door_combo(combo, cam_id="camX")
    return combo


def test_a_uvc_camera_is_offered_the_platforms_doors():
    combo = _cell_for("opencv")
    got = [combo.itemData(i) for i in range(combo.count())]
    assert got == [d for _l, d in _Dlg._available_doors()]
    assert combo.isEnabled()


def test_a_scientific_camera_is_not_offered_a_door():
    """Spinnaker and XIMEA are reached through their SDK. Offering DirectShow
    there is offering a path that does not exist, and it also used to be what
    the family was read from, so the camera answered "dshow" and had its
    calibration filed under a wrong identity."""
    combo = _cell_for("spinnaker")
    assert combo.count() == 1, [combo.itemText(i) for i in range(combo.count())]
    assert not combo.isEnabled(), "a door was offered for an SDK camera"
    assert combo.currentData() == "", "an SDK camera reported a door"


def test_the_family_comes_from_the_enumeration_not_the_cell():
    d = _Dlg.__new__(_Dlg)
    d._cam_family = {"camS": "spinnaker", "camU": "cam0"}
    assert d._family_for_camera("camS") == "spinnaker"
    assert d._family_for_camera("unknown-camera") == "opencv"
