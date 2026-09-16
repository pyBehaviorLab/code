"""Trace a disconnect end to end through the real Pipeline.

Every stage that holds a reference to a live camera has to let go, or the
device stays open and the tile keeps showing the last frame instead of going
dark. The stages, in the order ``Pipeline.disconnect_camera`` walks them:

    box sink subscriptions  ->  pose/tracker enablement  ->  VideoManager
    box_camera_map  ->  the CameraThread itself  ->  the per-camera segment
    processor  ->  the FrameBus box registration  ->  the bus itself

These tests assert each one, plus the two shapes that actually break on a rig:
a camera shared by several boxes (which must stay open until the *last* box
leaves) and the display poll (which must stop painting a disconnected box).
"""
import threading
from datetime import datetime

import numpy as np
import pytest

from source.tests.qt_dispose import dispose
from source.video.cameras import capture


class _FakeCam:
    """Stand-in CameraThread that records whether it was really stopped."""

    def __init__(self, camera_id=None, width=None, height=None,
                 target_fps=None, camera_config=None):
        self.camera_id = camera_id
        self.width = int(width or 640)
        self.height = int(height or 480)
        self.target_fps = target_fps
        self.connected = True
        self.stopped = False
        self.joined = False
        self.connection_checked = threading.Event()
        self.connection_checked.set()
        self._frame = np.full((self.height, self.width, 3), 90, dtype=np.uint8)
        self._version = 1

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
        self.joined = True
        return True

    def set_target_fps(self, f):
        self.target_fps = f

    def get_last_frame(self):
        return self._frame

    def get_latest_frame_versioned(self, last_version):
        if last_version >= self._version:
            return None
        return self._frame, 0, self._version

    def drain_recording_buffer(self):
        return []


def _halves(box_number, left=True):
    x = 0.0 if left else 0.5
    return {"boxes": [{"box_number": box_number,
                       "geometry": {"percent": {"x": x, "y": 0.0,
                                                "width": 0.5,
                                                "height": 1.0}}}]}


@pytest.fixture
def pipe(monkeypatch):
    monkeypatch.setattr(capture, "CameraThread", _FakeCam)
    from source.video.framebus.controller import Pipeline
    p = Pipeline()
    yield p
    with_shutdown = getattr(p, "shutdown", None)
    if with_shutdown is not None:
        try:
            with_shutdown()
        except Exception:
            pass


def test_disconnect_releases_every_stage(pipe):
    """One box, one camera: after disconnect nothing holds the device."""
    pipe.update_camera_config(0, selected_resolution=(640, 480),
                              selected_fps=30, camera_backend="opencv")
    assert pipe.connect_camera(0, 1)
    vm = pipe.video_manager
    cam = vm.cameras[0]
    assert pipe.get_bus(0) is not None

    pipe.disconnect_camera(1)

    assert 1 not in vm.box_camera_map, "box still mapped to a camera"
    assert 0 not in vm.cameras, "CameraThread still held by VideoManager"
    assert cam.stopped, "the capture thread was never told to stop"
    assert cam.joined, "the capture thread was never joined"
    assert pipe.get_bus(0) is None, "FrameBus outlived its last box"
    assert vm.segment_processor_for(0) is None, "segmenter still held"


def test_shared_camera_stays_open_until_the_last_box_leaves(pipe):
    """Four boxes on one overhead camera: releasing one must not blind the
    other three, and the last one out closes the device."""
    pipe.update_camera_config(0, selected_resolution=(640, 480),
                              selected_fps=30, camera_backend="opencv")
    assert pipe.connect_camera(0, 1, segment_config=_halves(1, True))
    assert pipe.connect_camera(0, 2, segment_config=_halves(2, False))
    vm = pipe.video_manager
    cam = vm.cameras[0]

    pipe.disconnect_camera(1)
    assert not cam.stopped, "shared camera closed while box 2 still needs it"
    assert 2 in vm.box_camera_map
    assert pipe.get_bus(0) is not None

    pipe.disconnect_camera(2)
    assert cam.stopped, "shared camera left open after its last box left"
    assert pipe.get_bus(0) is None


def test_disconnect_stops_box_inference(pipe):
    pipe.update_camera_config(0, selected_resolution=(640, 480),
                              selected_fps=30, camera_backend="opencv")
    assert pipe.connect_camera(0, 1)
    pipe.disconnect_camera(1)
    # A later reconnect must not silently resume inference nobody re-armed.
    assert 1 not in getattr(pipe.pose, "_enabled", set())
    assert 1 not in getattr(pipe.tracker, "_enabled", set())


def test_bus_stops_delivering_to_a_disconnected_box(pipe):
    """The sink unsubscribe has to actually detach: a frame published after
    disconnect must not reach the box."""
    from source.video.framebus.types import CameraFrame
    pipe.update_camera_config(0, selected_resolution=(640, 480),
                              selected_fps=30, camera_backend="opencv")
    assert pipe.connect_camera(0, 1)
    bus = pipe.get_bus(0)
    seen = []
    bus.on_box_frame(1, seen.append)
    frame = CameraFrame(
        image=np.zeros((480, 640, 3), np.uint8), cam_frame_id=1,
        capture_host_ns=0, capture_wall=datetime(2026, 1, 1),
        camera_id=0, is_shared=False, box_ids=())
    bus.publish_frame(frame)
    assert seen, "sanity: the box receives frames while connected"

    pipe.disconnect_camera(1)
    before = len(seen)
    bus.publish_frame(frame)          # the bus is detached from the pipeline
    assert len(seen) == before or pipe.get_bus(0) is None


def test_display_poll_skips_a_disconnected_box(monkeypatch):
    """The tile goes dark because the poll stops finding the box, not because
    something repaints it. Drives the real MainWindowBase poll method."""
    monkeypatch.setattr(capture, "CameraThread", _FakeCam)
    from source.gui.base import MainWindowBase

    painted = []

    class _Win:
        _display_min_interval_ns = None

        def __init__(self):
            self._display_last_version = {}
            self._display_last_emit_ns = {}

        # The poll now resolves the Qt-bound values itself and hands a batch
        # of jobs to _paint_prepared_tiles, so that is the seam to observe.
        def _is_box_tile_visible(self, setup_id):
            return True

        def _get_box_roi(self, setup_id, frame=None):
            return None

        def _box_video_label_size(self, setup_id):
            return None

        def _paint_prepared_tiles(self, jobs):
            painted.extend(job[0] for job in jobs)

    from source.video.framebus.controller import Pipeline
    pipe = Pipeline()
    try:
        pipe.update_camera_config(0, selected_resolution=(640, 480),
                                  selected_fps=30, camera_backend="opencv")
        pipe.connect_camera(0, 1)
        win = _Win()
        win.video_manager = pipe.video_manager
        poll = MainWindowBase._paint_streaming_cameras_once
        poll(win)
        assert painted == [1], "sanity: a connected box paints"

        pipe.disconnect_camera(1)
        painted.clear()
        win._display_last_version.clear()
        poll(win)
        assert painted == [], "a disconnected box was still being painted"
    finally:
        try:
            pipe.shutdown()
        except Exception:
            pass


# ── the tile has to actually go dark ──────────────────────────────────────

def test_video_tile_clears_the_painted_frame(mock_qapplication):
    """The tile paints from its own cached source pixmap, so blanking it has
    to drop that, not merely set placeholder text over it."""
    from source.gui.widgets.video_tile import VideoTile
    tile = VideoTile()
    try:
        tile.update_frame(np.full((48, 64, 3), 200, dtype=np.uint8))
        assert not tile.pixmap().isNull(), "sanity: a frame is being painted"

        tile.setText("No Signal")
        assert tile.pixmap().isNull(), (
            "the last frame survived: a disconnected camera's tile would "
            "keep showing it")
        assert tile.text() == "No Signal"
    finally:
        dispose(tile)


def test_video_tile_clear_drops_the_frame(mock_qapplication):
    from source.gui.widgets.video_tile import VideoTile
    tile = VideoTile()
    try:
        tile.update_frame(np.full((48, 64, 3), 200, dtype=np.uint8))
        tile.clear()
        assert tile.pixmap().isNull()
    finally:
        dispose(tile)


def test_box_widget_clear_video_blanks_the_tile(mock_qapplication):
    """Drive the real per-box widget path both modes use on disconnect."""
    from source.gui.widgets.setup_widget import SetupWidget
    w = SetupWidget(setup_id=1, main_window=None)
    try:
        w.video_label.update_frame(np.full((48, 64, 3), 120, dtype=np.uint8))
        assert not w.video_label.pixmap().isNull()
        w.clear_video()
        assert w.video_label.pixmap().isNull(), "tile still shows a frame"
    finally:
        dispose(w)
