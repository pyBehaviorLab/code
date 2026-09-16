"""Two promises the dialog made and did not keep.

"Any" as a transport format, and a region for a camera that serves one box.
Both were reported from the rig as "the UI says one thing and the camera does
another", and both turned out to be the dialog offering a choice the rest of
the stack could not express.
"""
from __future__ import annotations

import contextlib
import re
from types import SimpleNamespace

import pytest
from PySide6 import QtWidgets

from source.gui.dialogs.camera_connect import CameraConnectDialog as _Dlg
from source.tests.qt_dispose import WidgetBin

_BIN = WidgetBin()


@pytest.fixture(autouse=True)
def _drain():
    yield
    _BIN.drain(close=True)


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ── the transport format ─────────────────────────────────────────────────

def _wanted(fmt):
    """What the capture backend would ask the driver for, given ``fmt``.

    Called on a bare instance: ``_wanted_fourcc`` reads only the configured
    format, so this exercises the real rule without opening a camera.
    """
    from source.video.cameras.opencv import OpenCVCamera
    cam = OpenCVCamera.__new__(OpenCVCamera)
    cam._capture_format = fmt
    cam._warned_unknown_format = False
    cam.serial_number = "test"
    return OpenCVCamera._wanted_fourcc(cam)


def _menu_formats():
    """The format values the row's combo offers, from the dialog itself."""
    import inspect
    src = inspect.getsource(_Dlg._build_row_capture_cells)
    block = src.split("for label, data in (", 1)[1].split("):", 1)[0]
    return re.findall(r'"([a-z0-9]+)"', block)


def test_every_offered_format_is_one_the_backend_can_honour(qapp):
    """The menu had four entries and the backend understood two.

    ``""`` (Any) became MJPEG and ``"h264"`` became uncompressed YUYV, so two
    of the four did the opposite of what they said. Whatever the menu offers,
    the backend must have a rule for.
    """
    offered = _menu_formats()
    assert offered, "the format menu is empty"
    for value in offered:
        if value == "auto":
            continue
        assert _wanted(value) is not None, (
            f"the menu offers {value!r} and the backend has no rule for it, "
            f"so it silently becomes something else")


def test_the_menu_and_the_backend_agree_on_each_name(qapp):
    assert _wanted("mjpeg") == "MJPG"
    assert _wanted("yuy2") == "YUYV"
    assert _wanted("h264") == "H264", (
        "h264 used to fall through to uncompressed YUYV, the opposite of "
        "what was asked for")


def test_auto_is_stored_as_itself(qapp):
    """It was stored as ``""``, which every reader treats as unset.

    ``_fill_row_capture`` replaces a falsy format with mjpeg and so does the
    capture backend, so choosing Any selected MJPEG in both places. A value
    that cannot survive being written is not a choice.
    """
    assert "auto" in _menu_formats()
    assert "" not in _menu_formats(), (
        'an empty string cannot be told apart from "not set"')


def test_auto_leaves_the_driver_alone(qapp):
    """Auto exists so a camera that refuses a forced format can still open."""
    assert _wanted("auto") is None


def test_an_unknown_name_is_reported_not_substituted(qapp, caplog):
    """Silently opening in the wrong format is worse than saying so."""
    import logging
    with caplog.at_level(logging.WARNING):
        assert _wanted("theora") is None
    assert "theora" in caplog.text, caplog.text


# ── the region ───────────────────────────────────────────────────────────

@contextlib.contextmanager
def _dialog_with(shared: bool):
    """A dialog whose boxes look through one shared camera, or one each.

    Disposed on the way out: a leaked dialog keeps its whole tree, and the
    Pipeline under it, alive for the rest of the session.
    """
    from source.video.framebus.controller import Pipeline
    cams = ["camS", "camS"] if shared else ["camA", "camB"]
    pipe = Pipeline()
    for cid in set(cams):
        pipe.install_camera_configs({cid: {
            "camera_id": cid,
            "probed_variants": {"dshow": [[640, 480, 30.0]]}}})
    setups = []
    for n, cid in enumerate(cams, 1):
        s = SimpleNamespace(setup_number=n, save_video_enabled=True,
                            _camera_backend="OpenCV", roi_segment=None,
                            roi_normalized=None)
        s.camera_id_edit = SimpleNamespace(text=lambda c=cid: c)
        setups.append(s)

    class _MW(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.pipeline = pipe
            self.video_target_fps = 30

        def get_all_setup_widgets(self):
            return setups

        def _project_changed(self, **k):
            pass

    mw = _BIN.add(_MW())
    dlg = _BIN.add(_Dlg(mw))
    items = {}
    for n, (cid, s) in enumerate(zip(cams, setups), 1):
        items.setdefault(cid, []).append({"box_number": n, "box_widget": s})
    try:
        yield dlg, items, setups
    finally:
        with contextlib.suppress(Exception):
            pipe.shutdown()


def test_a_camera_serving_one_box_needs_no_region_drawn(qapp):
    """The frame already IS the box, so the whole frame is the honest default.

    Demanding a drawing blocked Connect on a rig with nothing to split, and
    the prompt told the operator the camera was "shared by 1 boxes".
    """
    with _dialog_with(shared=False) as (dlg, items, setups):
        assert dlg._prompt_roi_for_pending_cameras(items) is True, (
            "a per-box camera must not block the connect")
        for s in setups:
            assert s.roi_normalized == (0.0, 0.0, 1.0, 1.0), (
                "the default region must be written, buildSegmentConfig "
                "drops a box that has none")


def test_a_box_at_whole_frame_survives_buildSegmentConfig(qapp):
    """The default has to reach the pipeline, not just the widget."""
    with _dialog_with(shared=False) as (dlg, items, _setups):
        dlg._prompt_roi_for_pending_cameras(items)
        for cid, its in items.items():
            cfg = dlg.buildSegmentConfig(cid, its)
            assert len(cfg["boxes"]) == 1, f"camera {cid} lost its box"
            pct = cfg["boxes"][0]["geometry"]["percent"]
            assert (pct["x"], pct["y"], pct["width"], pct["height"]) == (
                0.0, 0.0, 1.0, 1.0)


def test_a_shared_camera_is_still_asked_to_split(qapp, monkeypatch):
    """The one case where a region is genuinely required stays required."""
    asked = []

    class _Cancelled:
        def __init__(self, camera_id, num_boxes, parent=None):
            asked.append((camera_id, num_boxes))

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr("source.gui.dialogs.camera_connect."
                        "ConfigSelectionDialog", _Cancelled)
    with _dialog_with(shared=True) as (dlg, items, _setups):
        assert dlg._prompt_roi_for_pending_cameras(items) is False
    assert asked and asked[0][1] == 2, (
        f"a shared camera must be prompted, with its real box count: {asked}")
