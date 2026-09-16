"""Characterisation of ``BlobTracker.update``.

Pins the behaviour of a 164-line per-frame method BEFORE splitting it. Each
test names a decision the method makes rather than a line it runs:

  * refuse to run at all unless initialised, active, and holding a background
  * take the caller's pre-converted gray view when offered (a shared CCTV
    frame converts once, not once per box)
  * survive a background that no longer matches the frame size
  * on a miss, coast on the last position rather than reporting a jump to None
  * keep the adaptive background from learning a still animal
  * never let a per-frame error kill the tracker
"""
from __future__ import annotations

import numpy as np

from source.video.tracking.blob import BlobTracker

_BG = np.full((240, 320), 200, np.uint8)


def _frame(blobs=((60, 55, 80, 75),)):
    f = _BG.copy()
    for (x0, y0, x1, y1) in blobs:
        f[y0:y1, x0:x1] = 40           # dark animal on a bright background
    return f


def _tracker(bg_mode="static", **kw):
    t = BlobTracker(setup_id=1)
    t.set_background(_BG)
    params = dict(threshold=25, min_area=20, max_area=50000,
                  detect_dark=True, bg_mode=bg_mode)
    params.update(kw)
    t.update_params(**params)
    t.initialize(_BG)
    return t


# ── guards ───────────────────────────────────────────────────────────────

def test_an_uninitialised_tracker_reports_nothing():
    t = BlobTracker(setup_id=1)
    assert t.update(_frame()) == (False, None)


def test_a_stopped_tracker_reports_nothing():
    t = _tracker()
    t.tracking_active = False
    assert t.update(_frame()) == (False, None)


def test_no_background_means_no_detection():
    t = _tracker()
    t.background_gray = None
    assert t.update(_frame()) == (False, None)


# ── the gray view ────────────────────────────────────────────────────────

def test_a_supplied_gray_view_is_used_verbatim():
    """A shared CCTV frame is converted once per CameraFrame and handed to
    every box; converting again per tracker would be wasted work."""
    t = _tracker()
    bgr = np.dstack([_frame()] * 3)          # would convert to a blank-ish gray
    ok, pos = t.update(bgr, frame_gray=_frame())
    assert ok and pos is not None


def test_a_colour_frame_is_converted():
    t = _tracker()
    ok, pos = t.update(np.dstack([_frame()] * 3))
    assert ok and pos is not None


def test_a_grayscale_frame_passes_straight_through():
    t = _tracker()
    ok, pos = t.update(_frame())
    assert ok and pos is not None


def test_a_background_of_the_wrong_size_is_resized_not_fatal():
    """A resolution change mid-session must degrade, not crash."""
    t = _tracker()
    t.background_gray = np.full((120, 160), 200, np.uint8)
    ok, _ = t.update(_frame())
    assert ok is True


# ── hit and miss ─────────────────────────────────────────────────────────

def test_a_hit_returns_a_bounding_box_and_counts():
    t = _tracker()
    before = t.success_count
    ok, pos = t.update(_frame())
    assert ok and len(pos) == 4
    assert t.success_count == before + 1
    assert t.last_position == pos
    assert t.last_centroid is not None


def test_a_miss_coasts_on_the_last_position():
    """Reporting None on a dropped frame would read as 'animal teleported to
    nowhere'; the caller wants the last known box."""
    t = _tracker()
    t.update(_frame())
    known = t.last_position
    ok, pos = t.update(_BG.copy())            # empty frame, nothing to find
    assert ok is False
    assert pos == known


def test_the_frame_counter_advances_on_hit_and_miss():
    t = _tracker()
    t.update(_frame())
    t.update(_BG.copy())
    assert t.frame_count == 2


# ── the callback ─────────────────────────────────────────────────────────

def test_the_callback_receives_the_box_and_timestamp():
    seen = []
    t = _tracker()
    t.callback = lambda sid, x, y, w, h, ts: seen.append((sid, x, y, w, h, ts))
    t.update(_frame(), timestamp=1234)
    assert len(seen) == 1
    assert seen[0][0] == 1 and seen[0][5] == 1234


def test_a_raising_callback_does_not_break_the_frame():
    t = _tracker()

    def _boom(*a):
        raise RuntimeError("subscriber died")
    t.callback = _boom
    ok, pos = t.update(_frame())
    assert ok and pos is not None


def test_no_callback_fires_on_a_miss():
    seen = []
    t = _tracker()
    t.callback = lambda *a: seen.append(a)
    t.update(_BG.copy())
    assert seen == []


# ── quality metrics the calibration UI reads ─────────────────────────────

def test_hit_rate_and_jitter_accumulate():
    t = _tracker()
    for i in range(4):
        x = 60 + 6 * i
        t.update(_frame([(x, 55, x + 20, 75)]))
    assert len(t._recent_results) == 4
    assert all(r == 1 for r in t._recent_results)
    assert t._jitter_px > 0.0            # the animal moved, so it must be > 0


def test_a_miss_is_recorded_in_the_hit_rate():
    t = _tracker()
    t.update(_frame())
    t.update(_BG.copy())
    assert list(t._recent_results)[-1] == 0


# ── adaptive background ──────────────────────────────────────────────────

def test_a_still_animal_freezes_the_adaptive_model():
    """Otherwise a motionless subject is slowly learned into the background
    and disappears."""
    t = _tracker(bg_mode="mog2")
    for _ in range(t._stationary_threshold_frames + 2):
        t.update(_frame())               # identical frame every time
    assert t._adaptive_frozen is True


def test_a_miss_freezes_the_adaptive_model():
    t = _tracker(bg_mode="mog2")
    t.update(_BG.copy())
    assert t._adaptive_frozen is True


def test_running_average_background_updates_only_on_a_hit():
    t = _tracker(bg_mode="running_avg")
    t.update(_frame())
    assert t._bg_running_f32 is not None
    assert t.background_gray is not None


def test_a_static_background_is_never_rewritten():
    t = _tracker(bg_mode="static")
    original = t.background_gray.copy()
    t.update(_frame())
    assert np.array_equal(t.background_gray, original)


# ── robustness ───────────────────────────────────────────────────────────

def test_a_malformed_frame_does_not_kill_the_tracker():
    t = _tracker()
    ok, _ = t.update(np.zeros((0, 0), np.uint8))
    assert ok is False
    # and the tracker still works afterwards
    assert t.update(_frame())[0] is True
