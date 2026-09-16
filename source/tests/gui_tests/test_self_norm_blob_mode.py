"""Wave 5: background-free 'self_norm' blob mode + Li auto-threshold.

The mousefinder-inspired detector: divide each frame by its own blur to cancel
uneven illumination, auto-threshold with Li's method, then run our existing
candidate selection. No captured background required.
"""
import cv2
import numpy as np

from source.video.tracking.blob import BlobTracker, threshold_li
from source.config.experiment import BlobConfig, TrackingConfig


def _uneven_frame_with_dark_animal(cx=160, cy=120, r=18):
    """A horizontal illumination gradient (uneven lighting) with a dark disk."""
    h, w = 240, 320
    frame = np.tile(np.linspace(40, 200, w).astype(np.uint8), (h, 1))
    cv2.circle(frame, (cx, cy), r, 20, -1)
    return frame


def test_threshold_li_is_between_two_modes():
    a = np.concatenate([np.full(1000, 0.7), np.full(1000, 1.0)])
    t = threshold_li(a)
    assert 0.7 < t < 1.0


def test_threshold_li_empty_and_constant():
    assert threshold_li(np.array([])) == 0.0
    assert threshold_li(np.full(10, 5.0)) == 5.0


def _stream(tracker, frames):
    """Feed frames as a camera would. Returns the last (ok, position).

    self_norm holds acquisition for its first frames while it works out which
    blobs are furniture, so a one-frame poke is not a fair test of it.
    """
    result = (False, None)
    for f in frames:
        result = tracker.update(f)
    return result


def _walking(n=40, y=120, x0=140, dx=2):
    """The same animal walking, one frame per step."""
    return [_uneven_frame_with_dark_animal(cx=x0 + i * dx, cy=y)
            for i in range(n)]


def test_self_norm_detects_without_background():
    frames = _walking()
    t = BlobTracker(1)
    t.update_params(bg_mode="self_norm", detect_dark=True)
    # No set_background call at all.
    assert t.initialize(frames[0]) is True
    assert t._self_norm_threshold > 0        # auto-estimated on init
    ok, pos = _stream(t, frames)
    assert ok is True
    cx, cy = t.last_centroid
    last_x = 140 + (len(frames) - 1) * 2
    assert abs(cx - last_x) < 8 and abs(cy - 120) < 8   # found the animal


def test_self_norm_survives_a_lighting_change():
    t = BlobTracker(1)
    t.update_params(bg_mode="self_norm", detect_dark=True)
    frames = _walking(y=100)
    t.initialize(frames[0])
    _stream(t, frames)
    # Brighten the whole scene by 30%, a static background would now light up
    # everywhere; self_norm cancels it via the per-frame division.
    last_x = 140 + (len(frames) - 1) * 2
    bright = np.clip(
        _uneven_frame_with_dark_animal(cx=last_x + 2, cy=100).astype(np.float32) * 1.3,
        0, 255).astype(np.uint8)
    ok, _ = t.update(bright)
    assert ok is True
    cx, cy = t.last_centroid
    assert abs(cx - (last_x + 2)) < 10 and abs(cy - 100) < 10


def test_estimate_threshold_self_norm_returns_ratio():
    t = BlobTracker(1)
    t.update_params(bg_mode="self_norm")
    thr = t.estimate_threshold(_uneven_frame_with_dark_animal())
    assert thr is not None and 0.0 < thr < 2.0


def test_estimate_area_bracket_on_a_fresh_tracker():
    """Auto Threshold calls this before any frame has been tracked, so the
    mask buffer it erodes into does not exist yet. With speck removal on by
    default that raised inside cv2.erode and Auto Threshold failed outright."""
    t = BlobTracker(1)
    t.update_params(bg_mode="self_norm", detect_dark=True,
                    self_norm_smooth_sigma=10.0, self_norm_minsize=10)
    bracket = t.estimate_area_bracket(_uneven_frame_with_dark_animal(r=40))
    assert bracket is not None
    lo, hi = bracket
    assert 0 < lo < hi


def _arena_with_clutter(cx, cy, block=True, strip=True):
    """An arena the way a real one looks to a background-free detector.

    Uneven lighting, a dark animal, and static dark furniture that outweighs
    it: a big compact block (shelf shadow) and a long thin strip (edge
    shadow). Both are exactly what made "largest contour = animal" pick the
    wrong blob on a real recording.
    """
    h, w = 480, 640
    frame = np.tile(np.linspace(40, 200, w).astype(np.uint8), (h, 1))
    if block:
        cv2.rectangle(frame, (60, 40), (260, 200), 30, -1)
    if strip:
        cv2.rectangle(frame, (60, 400), (300, 436), 25, -1)
    cv2.circle(frame, (cx, cy), 26, 20, -1)
    return frame


def _tracker_for_arena():
    t = BlobTracker(1)
    t.update_params(bg_mode="self_norm", detect_dark=True,
                    self_norm_smooth_sigma=10.0, self_norm_minsize=10)
    return t


def _blob_area_at(tracker, frame, center, tol=40):
    """Area of the mask blob sitting at ``center``, i.e. the animal's."""
    mask = tracker._sample_mask(frame)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if (abs(x + w / 2 - center[0]) < tol
                and abs(y + h / 2 - center[1]) < tol):
            return cv2.contourArea(c)
    raise AssertionError(f"no blob near {center}")


def test_area_bracket_picks_the_blob_that_moved():
    """Static clutter can outweigh the animal AND be compact, only its
    stillness across frames gives it away."""
    t = _tracker_for_arena()
    walk = [_arena_with_clutter(cx, 300) for cx in (300, 380, 460, 540)]
    t.estimate_threshold(walk[-1])
    lo, hi = t.estimate_area_bracket(walk)
    animal = _blob_area_at(t, walk[-1], (540, 300))
    assert lo <= animal <= hi                      # animal inside the window
    assert lo == max(10, round(animal * 0.5))      # window derived FROM it


def test_area_bracket_falls_back_to_shape_on_a_single_frame():
    """One frame carries no motion, so the animal is told from edge shadow by
    shape: the clutter that outweighs it is long and thin."""
    t = _tracker_for_arena()
    frame = _arena_with_clutter(400, 300, block=False)
    t.estimate_threshold(frame)
    lo, hi = t.estimate_area_bracket(frame)
    animal = _blob_area_at(t, frame, (400, 300))
    assert lo <= animal <= hi
    assert lo == max(10, round(animal * 0.5))


def test_area_bracket_one_frame_and_that_frame_in_a_list_agree():
    t = _tracker_for_arena()
    frame = _arena_with_clutter(400, 300)
    t.estimate_threshold(frame)
    assert t.estimate_area_bracket(frame) == t.estimate_area_bracket([frame])


def test_area_bracket_ignores_samples_of_another_size():
    """A resolution change mid-sample must not make the estimate fail, the
    odd frames drop out and the newest one still gets bracketed."""
    t = _tracker_for_arena()
    frame = _arena_with_clutter(400, 300)
    small = cv2.resize(frame, (320, 240))
    assert t.estimate_area_bracket([small, frame]) == t.estimate_area_bracket(frame)


def test_area_bracket_survives_frames_that_are_all_none():
    t = _tracker_for_arena()
    assert t.estimate_area_bracket([None, None]) is None
    assert t.estimate_area_bracket(None) is None


def test_self_norm_acquires_the_animal_not_the_furniture():
    """The whole point: a background-free mask contains the arena's dark
    furniture, and on a real recording that furniture outweighed the animal
    4:1. Picking the biggest blob tracked a shadow for a whole session."""
    t = _tracker_for_arena()
    frames = [_arena_with_clutter(300 + i * 4, 300) for i in range(40)]
    t.estimate_threshold(frames[0])
    t.initialize(frames[0])
    ok, _ = _stream(t, frames)
    assert ok is True
    cx, cy = t.last_centroid
    assert abs(cx - (300 + 39 * 4)) < 40 and abs(cy - 300) < 40


def test_self_norm_keeps_an_animal_that_stops_moving():
    """A resting animal looks exactly as static as a shelf. It must not be
    learned as furniture and dropped, the tracker stops learning once the
    subject settles, so the reference never grows to cover it."""
    t = _tracker_for_arena()
    walk = [_arena_with_clutter(300 + i * 4, 300) for i in range(40)]
    t.estimate_threshold(walk[0])
    t.initialize(walk[0])
    _stream(t, walk)
    resting = [_arena_with_clutter(300 + 39 * 4, 300)] * 120   # ~4-8 s still
    ok, _ = _stream(t, resting)
    assert ok is True
    cx, cy = t.last_centroid
    assert abs(cx - (300 + 39 * 4)) < 40 and abs(cy - 300) < 40


def test_self_norm_preview_never_waits_for_a_reference():
    """detect_with_mask is stateless, so it never feeds the furniture
    reference. Letting it wait for one would blank the calibration overlay
    permanently rather than briefly."""
    t = _tracker_for_arena()
    frame = _arena_with_clutter(400, 300)
    success, position, mask = t.detect_with_mask(frame)
    assert success is True and position is not None
    assert t._sn_static is None       # nothing was learned by previewing


def test_estimate_threshold_static_needs_background():
    t = BlobTracker(1)               # static mode, no background set
    assert t.estimate_threshold(_uneven_frame_with_dark_animal()) is None


def test_self_norm_round_trips_through_config():
    tc = TrackingConfig(mode="blob", blob=BlobConfig(bg_mode="self_norm"))
    d = tc.to_compact()
    back = TrackingConfig.from_dict(d)
    assert back.blob.bg_mode == "self_norm"


def test_self_norm_preview_works_without_background():
    # detect_with_mask is the calibration PREVIEW path. For self_norm it must
    # run detection (no background), not early-return a blank mask.
    t = BlobTracker(1)
    t.update_params(bg_mode="self_norm", detect_dark=True)
    frame = _uneven_frame_with_dark_animal()
    success, position, mask = t.detect_with_mask(frame)
    assert success is True and position is not None
    assert mask is not None and mask.any()     # a real mask, not all-zero


def test_static_preview_still_blanks_without_background():
    # Regression guard: non-self_norm modes must STILL show the "no background"
    # blank preview (the self_norm exception must not leak to static).
    t = BlobTracker(1)                          # static, no background
    success, position, mask = t.detect_with_mask(_uneven_frame_with_dark_animal())
    assert success is False and position is None
    assert not mask.any()


def test_self_norm_threshold_latches_and_does_not_re_estimate():
    t = BlobTracker(1)
    t.update_params(bg_mode="self_norm")
    t.initialize(_uneven_frame_with_dark_animal())
    first = t._self_norm_threshold
    assert first is not None and first > 0
    # A later frame must reuse the latched threshold, not re-estimate each time.
    t.update(_uneven_frame_with_dark_animal(cx=100, cy=100))
    assert t._self_norm_threshold == first
