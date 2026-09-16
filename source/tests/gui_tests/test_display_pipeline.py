"""Display path: what runs where, what it shares, and what it refuses.

The paint pipeline is split twice. Once by what Qt forbids off the GUI thread,
QPixmap is GUI-only, QImage and cv2 are not, and once by what is shareable
between boxes: the crop-and-resize is identical for every box showing the same
frame at the same size, the overlay is not.

That second split is the one that mattered. A 16-box wall fed by a single
un-segmented camera was resizing the same image sixteen times per tick
(123.9 ms against 6.1 ms) and painting at 8 fps as a result. These tests pin
both splits, plus the two cheap refusals, an invisible tile, a native pixel
format; that came free alongside.
"""
import numpy as np
import pytest
from PySide6 import QtGui

from source.gui.base import MainWindowBase


# ── the pixel format is the handoff cost ──────────────────────────────────

def test_a_bgra_frame_takes_the_native_format(mock_qapplication):
    """Format_RGB32 is the native 32-bit layout, so fromImage is a memcpy.
    Handing Qt BGR888 instead makes it convert every pixel ON THE GUI THREAD:
    measured 0.19 ms vs 11.12 ms for sixteen 240x180 tiles."""
    bgra = np.zeros((40, 60, 4), np.uint8)
    bgra[..., 2] = 255                       # red, in BGRA
    pix = MainWindowBase._numpy_to_pixmap(bgra)
    assert pix is not None and not pix.isNull()
    assert (pix.width(), pix.height()) == (60, 40)
    assert pix.toImage().pixelColor(5, 5).red() == 255, "channel order lost"


def test_bgr_and_grayscale_still_work(mock_qapplication):
    """Callers that hand over 3-channel or single-channel buffers keep
    working, the fast path is an addition, not a replacement."""
    for arr in (np.zeros((20, 30, 3), np.uint8), np.zeros((20, 30), np.uint8)):
        pix = MainWindowBase._numpy_to_pixmap(arr)
        assert pix is not None and not pix.isNull()
        assert (pix.width(), pix.height()) == (30, 20)


def test_a_nonsense_shape_is_refused(mock_qapplication):
    assert MainWindowBase._numpy_to_pixmap(np.zeros((4, 4, 2), np.uint8)) is None
    assert MainWindowBase._numpy_to_pixmap(np.zeros((0, 0, 3), np.uint8)) is None


# ── visibility gate ───────────────────────────────────────────────────────

class _Win:
    """Minimal stand-in exposing only what the poll actually calls."""
    _display_min_interval_ns = None

    def __init__(self, visible):
        self._visible = visible
        self._display_last_version = {}
        self._display_last_emit_ns = {}
        self.jobs = []

    def _is_box_tile_visible(self, setup_id):
        return MainWindowBase._is_box_tile_visible(self, setup_id)

    def _safe_box_call(self, setup_id, method, *a, **kw):
        return self._visible

    def _get_box_roi(self, setup_id, frame=None):
        return None

    def _box_video_label_size(self, setup_id):
        return None

    def _paint_prepared_tiles(self, jobs):
        self.jobs.extend(jobs)


class _Cam:
    connected = True

    def get_latest_frame_versioned(self, last_v):
        return (np.zeros((20, 30, 3), np.uint8), 1234, last_v + 1)


class _VM:
    def __init__(self):
        self.box_camera_map = {1: 0}
        self.cameras = {0: _Cam()}


@pytest.mark.parametrize("visible,expect", [(True, 1), (False, 0)])
def test_an_invisible_tile_is_not_painted(visible, expect):
    """The video grid is a TAB. While the operator is on Live Status every
    tile was still being cropped, resized, converted and painted for nobody."""
    win = _Win(visible)
    win.video_manager = _VM()
    MainWindowBase._paint_streaming_cameras_once(win)
    assert len(win.jobs) == expect


def test_visibility_fails_open():
    """A widget that cannot answer must still paint, a broken check is never
    a reason for a tile to go dark."""
    win = _Win(None)
    win.video_manager = _VM()
    MainWindowBase._paint_streaming_cameras_once(win)
    assert len(win.jobs) == 1


# ── the realtime tick no longer paints ────────────────────────────────────

def test_display_interval_follows_the_setting():
    """An explicit cfg.display.max_fps wins; otherwise the class default."""
    class _W:
        DISPLAY_FPS_DEFAULT = MainWindowBase.DISPLAY_FPS_DEFAULT
        _display_max_fps = 10.0
    assert MainWindowBase._display_interval_ms(_W()) == 100

    class _D:
        DISPLAY_FPS_DEFAULT = MainWindowBase.DISPLAY_FPS_DEFAULT
        _display_max_fps = None
    assert (MainWindowBase._display_interval_ms(_D())
            == round(1000 / MainWindowBase.DISPLAY_FPS_DEFAULT))


def test_the_default_paint_rate_is_not_below_the_cameras():
    """A default under the capture rate throws away frames the operator can
    see. 15 was visibly juddery on a live wall and capped a 30 fps camera."""
    assert MainWindowBase.DISPLAY_FPS_DEFAULT >= 30


def test_there_is_only_one_pacer():
    """The timer paces painting. A second per-tile gate at the same rate,
    measured on a different clock, rejected ticks by beat interference and
    dropped a 15 Hz wall to 8 fps."""
    import inspect
    src = inspect.getsource(MainWindowBase._paint_streaming_cameras_once)
    assert "_display_min_interval_ns" not in src, (
        "the per-tile rate gate is back, it beats against the display timer")


# ── the shared-camera redundancy that caused 8 fps on a 16-box wall ───────

class _Counting:
    """Counts how many times the expensive half actually runs."""

    _tile_aspect_mode = "preserve"
    _overlay: dict = {}
    tracking_zones: dict = {}
    _tile_pool = None

    def __init__(self):
        self.bases = 0
        self.painted = []

    def _draw_overlay_on_frame(self, f, *a, **k):
        return f

    def _tile_base(self, frame, roi, target, mode):
        self.bases += 1
        return MainWindowBase._tile_base(frame, roi, target, mode)

    _tile_has_overlay = MainWindowBase._tile_has_overlay
    _finish_tile = MainWindowBase._finish_tile
    _numpy_to_pixmap = staticmethod(MainWindowBase._numpy_to_pixmap)

    def _render_scaled(self, setup_id, pixmap):
        self.painted.append(setup_id)

    def _setup_widget_for(self, setup_id):
        return None

    def _update_fps_status(self, setup_id, widget):
        pass

    _paint_prepared_tiles = MainWindowBase._paint_prepared_tiles


def test_one_camera_shared_by_many_boxes_resizes_once(mock_qapplication):
    """The 16-box CCTV case. Every box shows the same un-segmented frame, so
    the crop-and-resize must run ONCE, doing it per box measured 123.9 ms
    against 6.1 ms, which is the whole difference between 8 fps and 60."""
    from PySide6 import QtCore
    win = _Counting()
    frame = np.zeros((480, 640, 3), np.uint8)
    target = QtCore.QSize(240, 180)
    jobs = [(bid, frame, None, target) for bid in range(1, 17)]

    win._paint_prepared_tiles(jobs)

    assert win.bases == 1, (
        f"resized {win.bases} times for 16 boxes showing the same frame")
    assert len(win.painted) == 16, "every box must still be painted"


def test_distinct_rois_each_get_their_own_resize(mock_qapplication):
    """Sharing is keyed on the work, not assumed: different ROIs are
    different pixels and must not be collapsed together."""
    from PySide6 import QtCore
    win = _Counting()
    frame = np.zeros((480, 640, 3), np.uint8)
    target = QtCore.QSize(240, 180)
    jobs = [(bid, frame, (bid * 20, 0, 160, 120), target)
            for bid in range(1, 5)]

    win._paint_prepared_tiles(jobs)

    assert win.bases == 4, "distinct ROIs were wrongly shared"
    assert len(win.painted) == 4
