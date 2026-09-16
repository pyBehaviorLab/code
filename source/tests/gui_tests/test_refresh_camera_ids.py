"""Refresh camera IDs must actually re-scan.

The device list is cached in two places, process-wide in the factory, and once
per combo via a ``_populated`` flag. Clearing either one alone leaves the stale
list on screen, so a camera plugged in after launch stays invisible. These pin
both halves.
"""
from types import SimpleNamespace

import pytest
from PySide6 import QtWidgets

from source.gui.dialogs.camera_connect import CameraConnectDialog
from source.tests.qt_dispose import dispose
from source.video.cameras import factory


class _FakeCfg:
    selected_resolution = (1920, 1080)
    selected_fps = 30
    frame_strategy = "accept"
    camera_backend = "opencv"

    def __init__(self):
        self.probed_modes = []


class _FakePipeline:
    def get_camera_config(self, cam_id):
        return _FakeCfg()

    def update_camera_config(self, cam_id, **kw):
        return _FakeCfg()

    def all_camera_configs(self):
        return {}


def _box(n, cam):
    w = QtWidgets.QWidget()
    w.setup_number = n
    e = QtWidgets.QLineEdit()
    e.setText(cam)
    w.camera_id_edit = e
    w._camera_backend = "OpenCV"
    w.save_video_enabled = True
    w.roi_normalized = None
    return w



def _go_cold(d, timeout=5.0):
    """Put the dialog in the state it is in the instant it opens: nothing
    enumerated yet.

    Opening the dialog starts a prewarm scan and, when it lands, a queued
    ``camera_ids_ready`` refills every picker, which asks for the list again.
    So clearing the cache is only durable once BOTH have finished; clear it
    while either is in flight and the next event loop turn quietly refills it.
    Settle first, then go cold, and do not pump events afterwards.
    """
    import threading
    import time
    end = time.time() + timeout
    while time.time() < end:
        QtWidgets.QApplication.processEvents()
        alive = [t for t in threading.enumerate()
                 if t.name.startswith("camera-enum-")]
        for t in alive:
            t.join(0.05)
        if not alive and d._avail_cam_ids_cache is not None:
            break                      # scanned, delivered, and quiet
        time.sleep(0.01)
    QtWidgets.QApplication.processEvents()
    d._avail_cam_ids_cache = None
    d._enum_worker_running = False


@pytest.fixture
def dlg(mock_qapplication, monkeypatch):
    """A dialog whose enumeration is a controllable fake."""
    listing = {"cams": [{"unique_id": "0-opencv", "model": "Cam A"}]}

    def _fake_list(backends=None):
        return list(listing["cams"])

    monkeypatch.setattr(
        "source.video.cameras.CameraFactory.list_available_cameras",
        staticmethod(_fake_list))

    d = CameraConnectDialog(parent=None)
    d.main_window = SimpleNamespace(
        get_all_setup_widgets=lambda: [_box(1, "0")], pipeline=_FakePipeline(),
        _probed_resolution_modes={},
        video_manager=SimpleNamespace(box_camera_map={}))
    d._machine_cached_modes = lambda i: []
    d.populateBoxes()
    _go_cold(d)
    yield d, listing
    dispose(d)


def _ids_when_ready(d, timeout=5.0):
    """The dialog's camera list, waiting for the worker that builds it.

    Asking on the GUI thread returns immediately, walking the USB bus there
    is a frozen app, so the answer arrives from a worker and the pickers
    fill themselves when it does.
    """
    import time
    end = time.time() + timeout
    while time.time() < end:
        QtWidgets.QApplication.processEvents()
        got = getattr(d, "_avail_cam_ids_cache", None)
        if got is not None:
            return [i for i, _ in got]
        d._available_camera_ids()      # idempotent: starts the worker once
        time.sleep(0.02)
    return None


def test_asking_on_the_gui_thread_never_blocks(dlg):
    """A cold cache must answer instantly, not walk the bus."""
    import time
    d, _listing = dlg
    # Re-established HERE, not only in the fixture: anything that pumps the
    # event loop between the two can deliver the startup refill and warm the
    # cache again, and then this asserts nothing.
    _go_cold(d)
    t0 = time.perf_counter()
    first = d._available_camera_ids()
    assert time.perf_counter() - t0 < 0.5, "the GUI thread waited on a scan"
    assert first == [], "nothing is known yet, and that is the honest answer"
    assert _ids_when_ready(d) == ["0"], "the worker must deliver the list"


def test_refresh_picks_up_a_camera_plugged_in_after_open(dlg):
    d, listing = dlg
    assert _ids_when_ready(d) == ["0"]

    listing["cams"].append({"unique_id": "1-opencv", "model": "Cam B"})
    # Without a refresh the dialog is still holding its cached answer.
    assert [i for i, _ in (d._avail_cam_ids_cache or [])] == ["0"]

    d._onRefreshCameraIds()

    assert [i for i, _ in (d._avail_cam_ids_cache or [])] == ["0", "1"], (
        "refresh did not re-scan")


def test_refresh_drops_the_process_wide_factory_cache(dlg, monkeypatch):
    """The dialog cache is not the only one; the factory holds its own."""
    d, _ = dlg
    called = {"n": 0}
    real = factory.invalidate_camera_enumeration

    def _spy():
        called["n"] += 1
        real()

    monkeypatch.setattr(factory, "invalidate_camera_enumeration", _spy)
    d._onRefreshCameraIds()
    assert called["n"] == 1, (
        "factory cache left intact, a re-scan would return the same list")


def test_refresh_refills_every_combo(dlg):
    """Each combo fills once and then believes it is done; refresh must reset
    that flag or the new camera never appears in the dropdown."""
    d, listing = dlg
    combos = d._camera_id_combos()
    assert combos, "no Camera-ID combos found"
    for c in combos:
        c._populated = True

    listing["cams"].append({"unique_id": "1-opencv", "model": "Cam B"})
    d._onRefreshCameraIds()

    for c in combos:
        items = [c.itemData(i) for i in range(c.count())]
        assert "1" in items, f"combo {c} still shows the stale list: {items}"


def test_refresh_preserves_what_the_operator_typed(dlg):
    d, _ = dlg
    combo = d.table.cellWidget(0, d.COL_CAMERA_ID)
    combo.setEditText("7")
    d._onRefreshCameraIds()
    assert combo.currentText() == "7", "refresh clobbered the typed camera id"


def test_refresh_reports_when_nothing_is_found(dlg, monkeypatch):
    d, listing = dlg
    listing["cams"].clear()
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: shown.append(a)))
    d._onRefreshCameraIds()
    assert shown, "an empty scan gave the operator no feedback"


def test_button_is_re_enabled_even_if_the_scan_raises(dlg, monkeypatch):
    d, _ = dlg
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(
        "source.video.cameras.CameraFactory.list_available_cameras",
        staticmethod(lambda backends=None: (_ for _ in ()).throw(OSError("boom"))))
    # _available_camera_ids swallows the error; the button must still come back.
    d._onRefreshCameraIds()
    assert d.refresh_ids_btn.isEnabled()
    assert d.refresh_ids_btn.text() == "Refresh camera IDs"


def test_scan_runs_off_the_gui_thread(dlg, monkeypatch):
    """A 3-20 s device walk must not block painting; that is the freeze the
    process-wide cache was introduced to avoid in the first place."""
    import threading as _th
    d, _ = dlg
    calls = []

    real = type(d)._available_camera_ids

    def _record(self):
        # Only the first call does the device walk; later ones read the cache
        # it filled (the combo refill on the GUI thread is one of those).
        calls.append((_th.current_thread().name,
                      getattr(self, "_avail_cam_ids_cache", None) is None))
        return real(self)

    monkeypatch.setattr(type(d), "_available_camera_ids", _record)
    d._onRefreshCameraIds()

    scanning = [name for name, was_cold in calls if was_cold]
    assert scanning, "nothing performed a scan"
    assert all(n == "camera-enum-refresh" for n in scanning), (
        f"the device walk ran on {scanning}, not a worker")


def test_the_list_arrives_without_anyone_opening_a_dropdown(dlg):
    """The pickers must fill themselves when the dialog opens.

    Filling lazily on first popup leaves a dialog that has sat open for a
    minute still showing an empty dropdown, with the rows holding whatever id
    the project happened to store, which is how a saved index and an
    enumerated identity come to disagree on screen.
    """
    d, listing = dlg
    listing["cams"] = [{"unique_id": "fpAAA-opencv", "model": "Cam A"},
                       {"unique_id": "fpBBB-opencv", "model": "Cam B"}]
    d._avail_cam_ids_cache = None
    combo = d.table.cellWidget(0, d.COL_CAMERA_ID)
    combo.clear()
    assert combo.count() == 0

    # What the prewarm worker does when its scan finishes. No popup involved.
    d._available_camera_ids()
    d.camera_ids_ready.emit()
    QtWidgets.QApplication.processEvents()

    assert combo.count() == 2, "the dropdown should be filled already"
    assert [combo.itemData(i) for i in range(combo.count())] == ["fpAAA", "fpBBB"]
    assert combo._populated is True, "and it must not re-scan on first popup"
