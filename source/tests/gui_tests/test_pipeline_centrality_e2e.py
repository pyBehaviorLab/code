"""Both modes must drive the SAME central camera and MCU pipelines.

``pyOperant.py`` and ``pyMaze.py`` differ only in the entry file plus the
per-mode ``MainWindow``. Everything about cameras and the MCU is supposed to
live in ``source/`` and be imported, not reimplemented, so a fix in one mode
is a fix in both.

These tests build the REAL ``MainWindow`` for each mode, side by side, and
assert they reach the same objects through the same calls. A per-mode copy of
pipeline logic shows up here as a divergence, which is the thing that is
expensive to notice by hand.
"""
from __future__ import annotations

import contextlib
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from source.tests.qt_dispose import dispose
from source.video.cameras import capture


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _no_modals(monkeypatch):
    """Offscreen, a modal would block forever."""
    Ok = QtWidgets.QMessageBox.StandardButton.Ok
    Yes = QtWidgets.QMessageBox.StandardButton.Yes
    for name, val in (("question", Yes), ("warning", Ok),
                      ("information", Ok), ("critical", Ok)):
        monkeypatch.setattr(QtWidgets.QMessageBox, name,
                            staticmethod(lambda *a, _v=val, **k: _v))


class _FakeCam:
    """Stand-in CameraThread so no device is opened."""

    def __init__(self, camera_id=None, width=None, height=None,
                 target_fps=None, camera_config=None):
        self.camera_id = camera_id
        self.width = int(width or 640)
        self.height = int(height or 480)
        self.target_fps = target_fps
        self.connected = True
        self.stopped = False
        import threading
        self.connection_checked = threading.Event()
        self.connection_checked.set()
        self._frame = np.full((self.height, self.width, 3), 70, dtype=np.uint8)

    def start(self):
        pass

    def isRunning(self):
        return not self.stopped

    def is_alive(self):
        return not self.stopped

    def stop(self):
        self.stopped = True
        self.connected = False

    def wait(self, timeout=None):
        return True

    def set_target_fps(self, f):
        self.target_fps = f

    def get_last_frame(self):
        return self._frame

    def get_latest_frame_versioned(self, last_version):
        return (self._frame, 0, 1) if last_version < 1 else None

    def drain_recording_buffer(self):
        return []


def _make(mode):
    if mode == "operant":
        from source.gui.operant import MainWindow
    else:
        from source.gui.maze import MainWindow
    mw = MainWindow()
    mw.add_setup()
    return mw


@pytest.fixture(params=["operant", "maze"])
def window(request, qapp, monkeypatch):
    monkeypatch.setattr(capture, "CameraThread", _FakeCam)
    mw = _make(request.param)
    mw.mode_name = request.param
    yield mw
    # Tear down for real. deleteLater() alone only QUEUES the delete; with no
    # event loop spinning in a test run the C++ objects pile up for the whole
    # session, and a later test building another MainWindow trips heap
    # corruption on Windows. close() + processEvents() drains the queue here.
    with contextlib.suppress(Exception):
        mw.pipeline.shutdown()
    with contextlib.suppress(Exception):
        dispose(mw)


# ── one pipeline, owned by the shared base ────────────────────────────────

def test_each_mode_owns_exactly_one_pipeline(window):
    from source.video.framebus.controller import Pipeline
    assert isinstance(window.pipeline, Pipeline)
    # video_manager is an alias, not a second manager.
    assert window.video_manager is window.pipeline.video_manager


def test_pipeline_is_constructed_by_the_shared_base_only(qapp):
    """Neither MainWindow may build its own Pipeline, one construction site
    means one place to change camera lifecycle behaviour."""
    import ast
    import pathlib
    for path in ("source/gui/operant.py", "source/gui/maze.py"):
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
        built = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "Pipeline"]
        assert not built, f"{path} constructs its own Pipeline"


# ── the camera lifecycle is the same call in both modes ──────────────────

def test_camera_connect_and_disconnect_are_the_shared_path(window):
    """Connect then disconnect through the host API both modes inherit."""
    sid = next(iter(window._iter_box_ids()))
    window.pipeline.update_camera_config(
        0, selected_resolution=(640, 480), selected_fps=30,
        camera_backend="opencv")

    assert window.pipeline.connect_camera(0, sid)
    vm = window.video_manager
    assert vm.box_camera_map.get(sid) == 0
    assert window.pipeline.get_bus(0) is not None

    window.disconnect_camera(sid)                # MainWindowBase, shared
    assert sid not in vm.box_camera_map
    assert 0 not in vm.cameras
    assert window.pipeline.get_bus(0) is None


def test_disconnected_box_tile_is_blank_in_both_modes(window):
    """The tile has to stop showing the last frame, the widget classes
    differ per mode, so this is asserted through the shared box protocol."""
    sid = next(iter(window._iter_box_ids()))
    widget = window._setup_widget_for(sid)
    tile = (getattr(widget, "video_label", None)
            or getattr(getattr(widget, "_stream_holder", lambda: None)(),
                       "videoWidget", None))
    if tile is None:
        pytest.skip("mode renders through a holder that needs a live layout")
    tile.update_frame(np.full((48, 64, 3), 180, dtype=np.uint8))
    assert not tile.pixmap().isNull()
    widget.clear_video()
    assert tile.pixmap().isNull(), "disconnected tile still shows a frame"


# ── the MCU path is the same call in both modes ──────────────────────────

def test_mcu_connect_is_one_shared_implementation(window):
    """``connect_pyboard_for_box`` lives once, on the base, and routes to the
    shared ``RunTask.connect_mcu`` rather than a per-mode button handler."""
    from source.gui.base import MainWindowBase
    mode_cls = type(window)
    assert "connect_pyboard_for_box" not in vars(mode_cls), (
        f"{mode_cls.__name__} re-implements connect_pyboard_for_box")
    assert "connect_pyboard_for_box" in vars(MainWindowBase)

    sid = next(iter(window._iter_box_ids()))
    widget = window._setup_widget_for(sid)
    called = []
    widget.connect_mcu = lambda port: called.append(port)
    window.connect_pyboard_for_box(sid, "COM7")
    assert called == ["COM7"]


def test_mcu_connect_never_toggles_a_connected_box(window):
    """Routing through the Connect *button* toggles, so "connect" on an
    already-open board silently disconnects it."""
    sid = next(iter(window._iter_box_ids()))
    widget = window._setup_widget_for(sid)
    calls = []
    widget.connect_mcu = lambda port: calls.append(port)
    widget.disconnect_mcu = lambda *a: calls.append("DISCONNECT")
    # is_connected is derived from the board in both modes, so open one.
    widget.pycboard = object()
    assert widget.is_connected
    window.connect_pyboard_for_box(sid, "COM7")
    assert "DISCONNECT" not in calls
    assert calls == [], "connect on an already-open board should be a no-op"


def test_board_is_constructed_in_exactly_one_place():
    """Every box in both modes gets its Pycboard from the same line."""
    import pathlib
    hits = []
    for path in pathlib.Path("source").rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            if "Pycboard(" in line and "class Pycboard" not in line:
                hits.append(f"{path}:{i}")
    assert len(hits) == 1, f"Pycboard built in several places: {hits}"


def test_gui_never_touches_serial_directly():
    """The GUI talks to the board through source/communication, never a raw
    serial port, that boundary is what keeps both modes on one transport."""
    import pathlib
    offenders = []
    for path in pathlib.Path("source/gui").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "serial.Serial(" in stripped or stripped.startswith("import serial"):
                offenders.append(f"{path}:{i}")
    assert not offenders, f"GUI opens serial directly: {offenders}"


# ── recording goes through the pipeline in both modes ────────────────────

def test_camera_lifecycle_api_lives_on_the_pipeline(window):
    """The pipeline is the single owner of camera -> bus -> sinks -> MCU;
    these are the entry points both modes are expected to call."""
    from source.video.framebus.controller import Pipeline
    for name in ("start_recording", "stop_recording", "connect_camera",
                 "disconnect_camera", "register_box", "get_bus",
                 "update_camera_config", "all_camera_configs"):
        assert hasattr(Pipeline, name), f"Pipeline lost {name}"


def test_modes_do_not_reimplement_camera_lifecycle():
    """A mode redefining these would fork the camera lifecycle, the exact
    scatter the single-Pipeline rule exists to prevent."""
    import ast
    import pathlib
    owned = {"connect_camera", "disconnect_camera", "disconnect_all_cameras",
             "register_box", "update_box_display"}
    for path in ("source/gui/operant.py", "source/gui/maze.py"):
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
        defined = {n.name for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)}
        clash = owned & defined
        assert not clash, f"{path} re-implements {sorted(clash)}"


def test_gui_uses_only_the_pipelines_public_api():
    """Reaching into ``pipeline._x`` from the GUI couples the two layers and
    hides the coupling from anyone reading the Pipeline's API."""
    import pathlib
    import re
    offenders = []
    pattern = re.compile(r"pipeline\._[a-z]")
    for path in pathlib.Path("source/gui").rglob("*.py"):
        for i, line in enumerate(
                path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if line.strip().startswith("#"):
                continue            # a comment naming one is documentation
            if pattern.search(line):
                offenders.append(f"{path}:{i}: {line.strip()}")
    assert not offenders, "GUI reaches into Pipeline internals:\n" + \
        "\n".join(offenders)


def test_shared_camera_bind_is_a_pipeline_call(window):
    """The second box of a shared camera attaches through one public call."""
    sid = next(iter(window._iter_box_ids()))
    window.pipeline.update_camera_config(
        0, selected_resolution=(640, 480), selected_fps=30,
        camera_backend="opencv")
    assert window.pipeline.connect_camera(0, sid)

    # An unopened camera is refused rather than half-binding the box.
    assert window.pipeline.bind_box_to_camera(99, "no-such-camera") is False

    assert window.pipeline.bind_box_to_camera(99, 0) is True
    assert window.video_manager.box_camera_map[99] == 0
    assert 99 in window.pipeline.get_bus(0).known_box_ids()


def test_a_shared_camera_box_reports_connected_too(window):
    """Boxes 2+ of a shared camera bind to an already-open device, so there is
    no streaming poll to fire their readiness hook. Without it only the FIRST
    box ever logged "Camera connected", the Cam field filled in, the status
    line stayed empty."""
    said = []
    window._box_status_say = lambda sid, msg: said.append((sid, msg))
    sid = next(iter(window._iter_box_ids()))
    window.pipeline.update_camera_config(
        0, selected_resolution=(640, 480), selected_fps=30,
        camera_backend="opencv")
    assert window.pipeline.connect_camera(0, sid)

    window._bind_box_to_shared_camera(99, 0)
    assert any(s == 99 and "Camera connected" in m for s, m in said), (
        "shared-camera box never reported readiness")


def test_a_bind_to_a_closed_camera_reports_nothing(window):
    """A refused bind must not claim the box is connected."""
    said = []
    window._box_status_say = lambda sid, msg: said.append((sid, msg))
    window._bind_box_to_shared_camera(99, "no-such-camera")
    assert said == []
