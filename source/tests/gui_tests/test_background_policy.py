"""Background policy: when it is taken for you, refreshed, and refused.

The three rules that make a captured reference something the app maintains
rather than a step the operator has to remember:

  * selecting Blob IS the request for subtraction, so a missing reference is
    captured then and there;
  * a stale one is retaken while the box is doing nothing;
  * one that does not match the current view is refused outright, because
    resizing it mis-registers every arena edge.
"""
import numpy as np

from source.gui.base import MainWindowBase
from source.video.tracking import background as bg


class _Cfg:
    def __init__(self, boxes):
        self.setup_config = type("S", (), {"boxes": boxes})()


class _Box:
    def __init__(self, n, stamp=None, running=False):
        self.setup_number = n
        self.bg_captured_at = stamp
        self.framework_running = running


class _VM:
    def __init__(self, frame):
        self._frame = frame

    def get_last_frame(self, setup_id):
        return self._frame


class _Win:
    """Only the surface these helpers touch."""
    BG_IDLE_REFRESH_SECONDS = MainWindowBase.BG_IDLE_REFRESH_SECONDS

    def __init__(self, boxes, frame=None, bg_dir=None):
        self._active_config = _Cfg(boxes)
        self._boxes = boxes
        self.video_manager = _VM(frame) if frame is not None else None
        self._bg_shape_problem = {}
        self._bg_idle_refresh_ns = {}
        self._dir = bg_dir
        self.captured = []
        self.stamped = []

    def get_all_setup_widgets(self):
        return self._boxes

    def _background_images_dir(self):
        return self._dir

    def _mark_bg_captured(self, setup_id):
        self.stamped.append(setup_id)

    # real implementations under test
    _stale_background_boxes = MainWindowBase._stale_background_boxes
    _bg_captured_at = MainWindowBase._bg_captured_at
    _blob_background_problem = MainWindowBase._blob_background_problem
    _refresh_idle_backgrounds = MainWindowBase._refresh_idle_backgrounds
    _capture_background_for_box = MainWindowBase._capture_background_for_box


# ── staleness detection ───────────────────────────────────────────────────

def test_an_old_stamp_reads_stale():
    win = _Win([_Box(1, "2020-01-01 00:00:00"), _Box(2, "2020-01-01 00:00:00")])
    assert sorted(win._stale_background_boxes()) == [1, 2]


def test_a_box_with_no_stamp_is_not_called_stale():
    """No stamp means no background yet; that is the MISSING case, warned
    about separately. Calling it stale would recapture nothing."""
    assert _Win([_Box(1, None)])._stale_background_boxes() == []


# ── idle refresh ──────────────────────────────────────────────────────────

def test_a_stale_idle_box_is_recaptured(tmp_path, mock_qapplication):
    frames = [np.full((40, 60, 3), 200, np.uint8) for _ in range(20)]
    it = iter(frames)

    win = _Win([_Box(1, "2020-01-01 00:00:00")],
               frame=frames[0], bg_dir=tmp_path)
    win.video_manager.get_last_frame = lambda sid: next(it, frames[0])
    win._refresh_idle_backgrounds()
    assert (tmp_path / "box1.png").exists(), "no background was written"
    assert win.stamped == [1]


def test_a_running_box_is_never_disturbed(tmp_path, mock_qapplication):
    """Recapturing mid-session would swap the reference under a live track."""
    win = _Win([_Box(1, "2020-01-01 00:00:00", running=True)],
               frame=np.zeros((40, 60, 3), np.uint8), bg_dir=tmp_path)
    win._refresh_idle_backgrounds()
    assert not (tmp_path / "box1.png").exists()
    assert win.stamped == []


def test_a_fresh_background_is_left_alone(tmp_path, mock_qapplication):
    """An operator who tuned against a specific reference keeps it."""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    win = _Win([_Box(1, now)], frame=np.zeros((40, 60, 3), np.uint8),
               bg_dir=tmp_path)
    win._refresh_idle_backgrounds()
    assert not (tmp_path / "box1.png").exists()


def test_the_refresh_does_not_re_fire_every_tick(tmp_path, mock_qapplication):
    """The 1 Hz tick calls this constantly; without the cooldown a stale box
    would be recaptured once a second forever."""
    frame = np.full((40, 60, 3), 200, np.uint8)
    win = _Win([_Box(1, "2020-01-01 00:00:00")], frame=frame, bg_dir=tmp_path)
    win._refresh_idle_backgrounds()
    first = win._bg_idle_refresh_ns.get(1)
    win.stamped.clear()
    win._refresh_idle_backgrounds()
    assert win._bg_idle_refresh_ns.get(1) == first
    assert win.stamped == [], "recaptured twice in a row"


def test_no_camera_means_no_capture(tmp_path, mock_qapplication):
    win = _Win([_Box(1, "2020-01-01 00:00:00")], bg_dir=tmp_path)
    win._refresh_idle_backgrounds()          # video_manager is None
    assert not (tmp_path / "box1.png").exists()


# ── capture writes and stamps ─────────────────────────────────────────────

def test_capture_writes_stamps_and_clears_a_block(tmp_path, mock_qapplication):
    frame = np.full((40, 60, 3), 200, np.uint8)
    win = _Win([_Box(1)], frame=frame, bg_dir=tmp_path)
    win._bg_shape_problem[1] = "old mismatch"
    assert win._capture_background_for_box(1) is True
    assert (tmp_path / "box1.png").exists()
    assert win.stamped == [1]
    assert win._blob_background_problem(1) is None, \
        "a fresh reference must clear the mismatch that blocked the box"


def test_capture_without_a_project_is_refused(mock_qapplication):
    win = _Win([_Box(1)], frame=np.zeros((10, 10, 3), np.uint8), bg_dir=None)
    assert win._capture_background_for_box(1) is False


# ── the shape block ───────────────────────────────────────────────────────

def test_a_mismatched_background_blocks_and_says_why():
    win = _Win([_Box(1)])
    code = bg.status(background=np.zeros((240, 320, 3), np.uint8),
                     frame_shape=(120, 200), captured_at=None)
    assert code == bg.MISMATCH
    msg, blocking = bg.describe(code,
                                background=np.zeros((240, 320, 3), np.uint8),
                                frame_shape=(120, 200))
    assert blocking is True
    assert "320x240" in msg and "200x120" in msg, msg
    win._bg_shape_problem[1] = msg
    assert win._blob_background_problem(1) == msg


def test_no_problem_means_no_block():
    assert _Win([_Box(1)])._blob_background_problem(1) is None
