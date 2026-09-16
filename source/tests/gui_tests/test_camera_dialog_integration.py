"""The camera dialog driven the way an operator drives it, in both rig shapes.

The other camera tests here call one populator or one handler and assert on
its output. Every one of them passed while the rig failed, because the defects
did not live inside a control, they lived in the WIRING between controls: the
row that displayed a size it never stored, the pick that wrote the other
backend, the format that read back as a different format. A test that calls
``_populateResolutionCombo`` directly cannot see any of that.

So these drive the real widgets, in order, and then check the one invariant
that matters:

    what the row is showing IS what the camera is opened with

Both arrangements are covered, because they are different code paths at
Connect and only one of them was ever exercised by hand:

    shared   one camera, N boxes, the frame is split into regions
    per box  N cameras, N boxes, each frame is a whole box

Nothing here opens a device. The seam is ``main_window.connect_camera``,
which is exactly where the real dialog hands off, and the rig below records
what the pipeline held at that instant. That is the value the camera would
have been opened with, so asserting on it asserts on the real handoff rather
than on a mock of it.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
from PySide6 import QtWidgets

from source.gui.dialogs.camera_connect import CameraConnectDialog
from source.tests.qt_dispose import WidgetBin
from source.video.framebus.controller import Pipeline

_BIN = WidgetBin()


@pytest.fixture(autouse=True)
def _drain():
    yield
    _BIN.drain(close=True)


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Modals(list):
    """Every modal Connect raised, in order, instead of a blocked test.

    A modal under pytest waits forever with nothing on screen to click, so the
    hang says nothing about which dialog appeared. Recording them turns that
    into evidence: the success box is expected, and a warning, a critical or a
    region picker each mean the flow asked the operator for something the rig
    should already have supplied.
    """

    @property
    def problems(self):
        return [m for m in self if m[0] not in ("information",)]

    def titles(self):
        return [m[1] for m in self]


@pytest.fixture(autouse=True)
def modals(monkeypatch):
    seen = _Modals()

    def _box(kind):
        def _f(*a, **_k):
            seen.append((kind, a[1] if len(a) > 1 else "",
                         a[2] if len(a) > 2 else ""))
            return QtWidgets.QMessageBox.StandardButton.Ok
        return _f

    for meth in ("information", "warning", "critical", "question", "about"):
        monkeypatch.setattr(QtWidgets.QMessageBox, meth,
                            staticmethod(_box(meth)))

    def _exec(self, *_a, **_k):
        seen.append(("exec", type(self).__name__, ""))
        return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(QtWidgets.QDialog, "exec", _exec)
    return seen


#: Two doors to the same camera, as this rig really measures them: Media
#: Foundation reaches 30 fps at 1080p where DirectShow reaches 5. Taken from
#: camera_calibrations.json on the recording PC.
TWO_DOOR = {"dshow": [[1920, 1080, 5.0], [1280, 720, 10.0], [640, 480, 29.9]],
            "msmf": [[1920, 1080, 30.4], [1280, 720, 29.9], [640, 480, 29.9]]}
#: A second, slower camera, so a per-box rig cannot pass by giving both boxes
#: the same answer.
ONE_DOOR = {"dshow": [[1600, 1200, 9.4], [800, 600, 12.5], [640, 480, 24.9]]}


class _Box(SimpleNamespace):
    """What the dialog needs from a box widget, and nothing else."""

    def __init__(self, number, camera_id):
        super().__init__()
        self.setup_number = number
        self.save_video_enabled = True
        self._camera_backend = "OpenCV"
        self.roi_segment = None
        self.roi_normalized = None
        self._cam = str(camera_id)
        self.camera_id_edit = SimpleNamespace(
            text=lambda: self._cam, setText=self._set_cam)

    def _set_cam(self, value):
        self._cam = str(value)


class _Rig(QtWidgets.QWidget):
    """A main window with a real Pipeline, standing in only for the device.

    ``connect_camera`` is the dialog's handoff. The real one reads the
    camera's config from the pipeline and opens the device; this one reads the
    same config and binds the box without opening anything, so
    ``camera_connect_state`` reports "connected" through the real code path.
    """

    def __init__(self, boxes, configs):
        super().__init__()
        self.pipeline = Pipeline()
        for cam_id, variants in configs.items():
            self.pipeline.install_camera_configs(
                {cam_id: {"camera_id": cam_id, "probed_variants": variants}})
        self._boxes = boxes
        self.video_target_fps = 30
        self.opened: dict = {}          # camera_id -> what it was opened with
        self.bound: list = []           # (box, camera_id) in call order

    def get_all_setup_widgets(self):
        return self._boxes

    def _project_changed(self, **_k):
        pass

    def connect_camera(self, setup_number, camera_backend=None,
                       camera_config=None):
        box = next(b for b in self._boxes if b.setup_number == setup_number)
        cam_id = box.camera_id_edit.text()
        cfg = self.pipeline.get_camera_config(cam_id)
        self.opened[cam_id] = {
            "resolution": tuple(cfg.selected_resolution or ()),
            "fps": cfg.selected_fps,
            "door": cfg.capture_backend,
            "format": cfg.capture_format,
            "flip_h": cfg.flip_horizontal,
            "flip_v": cfg.flip_vertical,
            "sdk": camera_backend,
        }
        self.bound.append((setup_number, cam_id))
        # What "the device came up" looks like to camera_connect_state: the
        # box is mapped and no thread is running, which is the same state the
        # bind-to-an-already-open-camera path leaves behind.
        self.pipeline.video_manager.box_camera_map[setup_number] = cam_id

    def _bind_box_to_shared_camera(self, setup_id, camera_id):
        """The non-first box of a shared camera, as the real window binds it.

        No device is re-opened here in the product either: the box is mapped
        onto the bus the first box already owns, and the widget's id is
        updated so a shared box does not look unconnected.
        """
        self.pipeline.video_manager.box_camera_map[setup_id] = camera_id
        box = next(b for b in self._boxes if b.setup_number == setup_id)
        box.camera_id_edit.setText(str(camera_id))
        self.bound.append((setup_id, str(camera_id)))


def _dialog(assignment, configs):
    """A dialog over ``assignment`` (box number -> camera id)."""
    boxes = [_Box(n, cam) for n, cam in sorted(assignment.items())]
    rig = _BIN.add(_Rig(boxes, configs))
    dlg = _BIN.add(CameraConnectDialog(rig))
    QtWidgets.QApplication.processEvents()
    return dlg, rig, boxes


def _cells(dlg, row):
    return (dlg.table.cellWidget(row, dlg.COL_RES),
            dlg.table.cellWidget(row, dlg.COL_FPS),
            dlg.table.cellWidget(row, dlg.COL_FORMAT),
            dlg.table.cellWidget(row, dlg.COL_FLIP))


def _pick(dlg, row, *, width, door, fps=None, fmt=None, flip_h=False):
    """Do what an operator does to a row, through the widgets.

    The DOOR is chosen in the Backend cell now, not by finding a resolution
    row that happens to carry it. Picking a size used to decide which backend
    the camera was opened through, so the door was a side effect of choosing
    a picture size rather than a decision of its own.
    """
    door_combo = dlg.table.cellWidget(row, dlg.COL_BACKEND)
    j = door_combo.findData(door)
    assert j >= 0, (f"row {row} cannot be set to door {door!r}; it offers "
                    f"{[door_combo.itemData(k) for k in range(door_combo.count())]}")
    door_combo.setCurrentIndex(j)
    QtWidgets.QApplication.processEvents()

    res, fps_combo, fmt_combo, flip = _cells(dlg, row)
    i = next(i for i in range(res.count())
             if isinstance(res.itemData(i), dict)
             and res.itemData(i)["width"] == width)
    res.setCurrentIndex(i)
    QtWidgets.QApplication.processEvents()
    if fps is not None:
        j = fps_combo.findData(fps)
        assert j >= 0, (f"row {row} cannot be set to {fps} fps, it offers "
                        f"{[fps_combo.itemText(k) for k in range(fps_combo.count())]}")
        fps_combo.setCurrentIndex(j)
        QtWidgets.QApplication.processEvents()
    if fmt is not None:
        fmt_combo.setCurrentIndex(fmt_combo.findData(fmt))
        QtWidgets.QApplication.processEvents()
    if flip_h:
        flip.flip_h_btn.setChecked(True)
        QtWidgets.QApplication.processEvents()


def _showing(dlg, row):
    """What the row is displaying, in the same shape ``_Rig.opened`` records."""
    res, fps, fmt, flip = _cells(dlg, row)
    d = res.currentData() or {}
    return {"resolution": (d.get("width"), d.get("height")),
            "fps": fps.currentData(),
            "door": d.get("backend"),
            "format": fmt.currentData(),
            "flip_h": flip.flip_h_btn.isChecked(),
            "flip_v": flip.flip_v_btn.isChecked()}


def _connect(dlg, modals):
    """The real Connect sequence, minus the device open.

    A Connect that raises a warning, a critical or a picker has not succeeded
    quietly, whatever the counters say, so that is checked here rather than
    in each test.
    """
    dlg.connect_cameras()
    QtWidgets.QApplication.processEvents()
    assert not modals.problems, (
        f"Connect stopped to ask the operator something: {modals.problems}")


# ── per box: one camera each ─────────────────────────────────────────────

def test_per_box_each_camera_opens_with_its_own_row(qapp, modals):
    """Two cameras, two different picks, and neither may take the other's."""
    dlg, rig, _boxes = _dialog({1: "camA", 2: "camB"},
                               {"camA": TWO_DOOR, "camB": ONE_DOOR})
    _pick(dlg, 0, width=1920, door="msmf", fps=30, fmt="mjpeg")
    _pick(dlg, 1, width=800, door="dshow", fps=10, fmt="yuy2", flip_h=True)
    shown = {"camA": _showing(dlg, 0), "camB": _showing(dlg, 1)}
    _connect(dlg, modals)

    assert set(rig.opened) == {"camA", "camB"}, "both cameras must be opened"
    for cam in ("camA", "camB"):
        got, want = rig.opened[cam], shown[cam]
        for key in ("resolution", "fps", "door", "format", "flip_h"):
            assert got[key] == want[key], (
                f"{cam}: the row showed {key}={want[key]!r} and the camera "
                f"was opened with {got[key]!r}")


def test_per_box_a_row_nobody_touched_still_opens_with_what_it_shows(qapp, modals):
    """The row displays a default the moment it is drawn.

    Connect used to write only the ACTIVE camera from the Camera options
    card and merely seed defaults for the rest, so the second camera reached
    the device with ``selected_resolution=None`` while its row plainly showed
    a size and a rate. Nothing on screen said so.
    """
    dlg, rig, _boxes = _dialog({1: "camA", 2: "camB"},
                               {"camA": TWO_DOOR, "camB": ONE_DOOR})
    shown = {"camA": _showing(dlg, 0), "camB": _showing(dlg, 1)}
    assert shown["camB"]["resolution"] != (None, None), (
        "the row draws a default, so there is something to honour")
    _connect(dlg, modals)
    for cam in ("camA", "camB"):
        assert rig.opened[cam]["resolution"] == shown[cam]["resolution"], (
            f"{cam} was opened with {rig.opened[cam]['resolution']} while its "
            f"row showed {shown[cam]['resolution']}")
        assert rig.opened[cam]["fps"] == shown[cam]["fps"]


def test_per_box_an_edit_on_one_row_leaves_the_other_camera_alone(qapp):
    dlg, rig, _boxes = _dialog({1: "camA", 2: "camB"},
                               {"camA": TWO_DOOR, "camB": ONE_DOOR})
    before = _showing(dlg, 1)
    _pick(dlg, 0, width=640, door="dshow", fps=15)
    assert _showing(dlg, 1) == before, "row 2 moved when row 1 was edited"
    cfg = rig.pipeline.get_camera_config("camB")
    assert cfg.selected_resolution in (None, ()), (
        "editing row 1 wrote onto camera B")


def test_per_box_the_slower_door_survives_a_redraw(qapp, modals):
    """Picking DirectShow used to store Media Foundation.

    ``blockSignals`` is a flag, not a counter: the populator released a block
    the caller was still holding, and the redraw wrote back the fastest door
    to that size instead of the one that was chosen.
    """
    dlg, rig, _boxes = _dialog({1: "camA"}, {"camA": TWO_DOOR})
    _pick(dlg, 0, width=1920, door="dshow", fps=5)
    dlg._refresh_table_cells()
    QtWidgets.QApplication.processEvents()
    shown = _showing(dlg, 0)
    assert shown["door"] == "dshow", "the row jumped to the other door"
    assert shown["resolution"] == (1920, 1080)
    assert shown["fps"] == 5, "the ladder must belong to the door that was picked"
    _connect(dlg, modals)
    assert rig.opened["camA"]["door"] == "dshow"
    assert rig.opened["camA"]["fps"] == 5


def test_per_box_needs_no_region_drawn(qapp, modals):
    """The frame already is the box, so Connect must not demand a drawing."""
    dlg, rig, boxes = _dialog({1: "camA", 2: "camB"},
                              {"camA": TWO_DOOR, "camB": ONE_DOOR})
    _connect(dlg, modals)
    assert len(rig.bound) == 2, "a per-box rig was blocked by the region step"
    for b in boxes:
        assert b.roi_normalized == (0.0, 0.0, 1.0, 1.0)


# ── shared: one camera, several boxes ────────────────────────────────────

def _shared(qapp, drawn=True):
    dlg, rig, boxes = _dialog({1: "camS", 2: "camS"}, {"camS": TWO_DOOR})
    if drawn:
        # A rig whose regions are already drawn, so Connect has nothing to
        # ask. Splitting is covered by the region tests; what matters here is
        # that the capture settings survive the shared path.
        for i, b in enumerate(boxes):
            b.roi_normalized = (0.5 * i, 0.0, 0.5, 1.0)
            b.roi_segment = (int(320 * i), 0, 320, 480)
    return dlg, rig, boxes


def test_shared_one_camera_opens_once_and_both_boxes_bind(qapp, modals):
    dlg, rig, _boxes = _shared(qapp)
    _pick(dlg, 0, width=1280, door="msmf", fps=25, fmt="mjpeg")
    _connect(dlg, modals)
    assert list(rig.opened) == ["camS"], "a shared camera must open once"
    assert sorted(n for n, _c in rig.bound) == [1, 2], (
        f"both boxes must bind to it: {rig.bound}")


def test_shared_the_camera_opens_with_what_the_row_shows(qapp, modals):
    dlg, rig, _boxes = _shared(qapp)
    _pick(dlg, 0, width=1280, door="msmf", fps=25, fmt="yuy2")
    shown = _showing(dlg, 0)
    _connect(dlg, modals)
    got = rig.opened["camS"]
    for key in ("resolution", "fps", "door", "format"):
        assert got[key] == shown[key], (
            f"the row showed {key}={shown[key]!r} and the camera was opened "
            f"with {got[key]!r}")


def test_shared_rows_cannot_disagree_about_one_camera(qapp):
    """Both rows look through the same device, so both must read the same.

    Two rows showing different rates over one camera is a contradiction the
    operator has no way to resolve: whichever they believed, one was wrong.
    """
    dlg, _rig, _boxes = _shared(qapp)
    _pick(dlg, 0, width=640, door="dshow", fps=15, fmt="mjpeg")
    dlg._refresh_table_cells()
    QtWidgets.QApplication.processEvents()
    assert _showing(dlg, 0) == _showing(dlg, 1), (
        "the two rows of one shared camera disagree")


def test_shared_editing_the_second_row_moves_the_first(qapp, modals):
    """Either row is a control for the same camera, so either may be used."""
    dlg, rig, _boxes = _shared(qapp)
    _pick(dlg, 1, width=1920, door="msmf", fps=30)
    dlg._refresh_table_cells()
    QtWidgets.QApplication.processEvents()
    assert _showing(dlg, 0)["resolution"] == (1920, 1080)
    _connect(dlg, modals)
    assert rig.opened["camS"]["resolution"] == (1920, 1080)


def test_shared_keeps_the_regions_that_were_already_drawn(qapp, modals):
    dlg, rig, boxes = _shared(qapp)
    _connect(dlg, modals)
    assert boxes[0].roi_normalized == (0.0, 0.0, 0.5, 1.0)
    assert boxes[1].roi_normalized == (0.5, 0.0, 0.5, 1.0)
    seg = getattr(rig, "video_segment_config", None)
    if seg is None:
        seg = getattr(dlg.main_window, "video_segment_config", None)
    assert seg and len(seg["boxes"]) == 2, (
        f"both regions must reach the pipeline: {seg}")


# ── the two arrangements must not diverge ────────────────────────────────

@pytest.mark.parametrize("assignment,configs,rows_drawn", [
    ({1: "camA", 2: "camB"}, {"camA": TWO_DOOR, "camB": ONE_DOOR}, False),
    ({1: "camS", 2: "camS"}, {"camS": TWO_DOOR}, True),
])
def test_connect_succeeds_in_both_arrangements(qapp, modals, assignment,
                                               configs, rows_drawn):
    """One test, both rig shapes, so neither can be fixed at the other's cost."""
    dlg, rig, boxes = _dialog(assignment, configs)
    if rows_drawn:
        for i, b in enumerate(boxes):
            b.roi_normalized = (0.5 * i, 0.0, 0.5, 1.0)
            b.roi_segment = (int(320 * i), 0, 320, 480)
    _pick(dlg, 0, width=640, door="dshow", fps=15, fmt="mjpeg")
    _connect(dlg, modals)
    assert sorted(n for n, _c in rig.bound) == [1, 2], (
        f"not every box connected: {rig.bound}")
    for cam, opened in rig.opened.items():
        assert opened["resolution"], f"{cam} opened with no resolution"
        assert opened["fps"], f"{cam} opened with no frame rate"
        assert opened["format"], f"{cam} opened with no transport format"


def _shutdown(rig):
    with contextlib.suppress(Exception):
        rig.pipeline.shutdown()


# ── the controls nobody had driven ───────────────────────────────────────
#
# Everything above covers Resolution, FPS, Format, Flip and the region. The
# rest of the dialog had never been exercised in either arrangement, which is
# exactly the state the audited controls were in before their defects were
# found. These drive the remainder.

def test_the_record_toggle_reaches_the_box(qapp):
    """Save is per box, and it decides whether that box records at all."""
    dlg, _rig, boxes = _dialog({1: "camA", 2: "camB"},
                               {"camA": TWO_DOOR, "camB": ONE_DOOR})
    cb = dlg.table.cellWidget(1, dlg.COL_SAVE)
    inner = cb.findChild(QtWidgets.QCheckBox) or cb
    inner.setChecked(False)
    QtWidgets.QApplication.processEvents()
    picked = {i["box_number"]: i["save_video"]
              for i in dlg.getSelectedBoxesWithCameras() or ()}
    assert picked.get(2) is False, f"box 2 still set to record: {picked}"
    assert picked.get(1) is True, "and box 1 was dragged along with it"


def test_the_backend_column_is_per_box(qapp):
    """The SDK is a per-camera choice, like everything else in the row."""
    dlg, _rig, _boxes = _dialog({1: "camA", 2: "camB"},
                                {"camA": TWO_DOOR, "camB": ONE_DOOR})
    combos = [dlg.table.cellWidget(r, dlg.COL_BACKEND)
              for r in range(dlg.table.rowCount())]
    assert all(c is not None for c in combos)
    assert dlg._backendForCameraId("camA") == "opencv"
    assert dlg._backendForCameraId("camB") == "opencv"


def test_the_card_and_the_row_agree_about_the_selected_camera(qapp):
    """Two surfaces onto one camera must not disagree.

    The Camera options card holds the ACTIVE camera's settings and the row
    holds the same camera's. A value that reads one way in the card and
    another in the row is the contradiction the per-row columns were added to
    remove, and the operator has no way to tell which one the rig will use.
    """
    dlg, rig, _boxes = _dialog({1: "camA", 2: "camB"},
                               {"camA": TWO_DOOR, "camB": ONE_DOOR})
    dlg.table.setCurrentCell(0, dlg.COL_BOX)
    QtWidgets.QApplication.processEvents()
    _pick(dlg, 0, width=1280, door="msmf", fps=25, fmt="yuy2")
    dlg._load_capture_for("camA")
    QtWidgets.QApplication.processEvents()

    row = _showing(dlg, 0)
    assert dlg.getSelectedResolution() == row["resolution"], (
        f"card shows {dlg.getSelectedResolution()}, row shows "
        f"{row['resolution']}")
    assert dlg.getSelectedFPS() == row["fps"], (
        f"card shows {dlg.getSelectedFPS()} fps, row shows {row['fps']}")


def test_save_and_load_a_camera_config_round_trips(qapp, tmp_path, monkeypatch):
    """A saved camera must come back as the same camera.

    Every hop between the dialog and the file is a chance to drop a field,
    and a dropped field is silent: the rig comes back next morning at a
    different rate with nothing saying why.
    """
    dlg, rig, _boxes = _dialog({1: "camA"}, {"camA": TWO_DOOR})
    _pick(dlg, 0, width=1280, door="msmf", fps=25, fmt="yuy2", flip_h=True)
    before = _showing(dlg, 0)

    path = tmp_path / "camA.json"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    dlg._onSaveCameraConfig()
    assert path.exists(), "nothing was written"

    rig.pipeline.update_camera_config(
        "camA", selected_resolution=(640, 480), selected_fps=10,
        capture_backend="dshow", capture_format="mjpeg",
        flip_horizontal=False)
    dlg._row_capture_sig.pop(0, None)
    dlg._refresh_table_cells()
    QtWidgets.QApplication.processEvents()
    assert _showing(dlg, 0) != before, "the test did not actually change it"

    dlg._onLoadCameraConfig()
    dlg._row_capture_sig.pop(0, None)
    dlg._refresh_table_cells()
    QtWidgets.QApplication.processEvents()
    after = _showing(dlg, 0)
    for key in ("resolution", "fps", "door", "format", "flip_h"):
        assert after[key] == before[key], (
            f"{key} did not survive save/load: saved {before[key]!r}, "
            f"loaded {after[key]!r}")


@pytest.mark.parametrize("assignment,configs,drawn", [
    ({1: "camA", 2: "camB"}, {"camA": TWO_DOOR, "camB": ONE_DOOR}, False),
    ({1: "camS", 2: "camS"}, {"camS": TWO_DOOR}, True),
])
def test_the_status_marks_describe_the_rig_in_both_arrangements(
        qapp, assignment, configs, drawn):
    """The footer tells the operator what is still missing.

    It said "2 region(s) still to draw" on a per-box rig that needed none,
    which is a false blocker: the reader cannot tell it from a real one.
    """
    dlg, _rig, boxes = _dialog(assignment, configs)
    if drawn:
        for i, b in enumerate(boxes):
            b.roi_normalized = (0.5 * i, 0.0, 0.5, 1.0)
    dlg._refresh_marks()
    QtWidgets.QApplication.processEvents()
    hint = dlg._footer_hint_text()
    assert "still needs a camera" not in hint, hint
    if not drawn:
        assert "region" not in hint.lower(), (
            f"a per-box rig was told to draw regions it does not need: {hint}")
