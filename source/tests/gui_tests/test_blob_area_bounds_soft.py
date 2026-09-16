"""A wrong Min/Max Area must degrade the tracker, not blind it.

The area bounds are numbers an operator types, and they were a hard filter.
Set Max Area below the animal and the animal stopped being a candidate at
all, so the tracker either found nothing or locked onto whatever speck of
noise happened to fall inside the range. That is the "the mask shows the
mouse, the box sits somewhere else" failure, and no amount of good tracking
logic downstream could recover from it, because the animal was already gone
before selection ran.

The bounds are now a strong preference rather than a veto: out-of-range
contours stay in the pool at a scoring penalty, so an in-range candidate wins
whenever a reasonable one exists, and the animal still wins over a speck when
the bounds are simply wrong. When that happens the tracker says so via
``tracked_out_of_range`` and the calibration panel turns red.

Only a hard floor survives, a contour of a few pixels is sensor noise on any
rig at any scale.
"""
from __future__ import annotations

import numpy as np
import pytest

from source.video.tracking.blob import BlobTracker

pytestmark = pytest.mark.unit


def _tracker(**params):
    t = BlobTracker(setup_id=1)
    t.background_gray = np.zeros((200, 200), dtype=np.uint8)
    t.use_adaptive_threshold = False
    t.threshold = 25
    t.detect_dark = False          # bright animal on a dark background
    t.min_area = 1
    t.subject_min_area = 1
    t.max_area = 1_000_000
    # Morphology would erode the small synthetic shapes; identity kernels keep
    # this test about the area logic.
    t.open_kernel_size = 1
    t.close_kernel_size = 1
    t.blur_kernel_size = 1
    for k, v in params.items():
        setattr(t, k, v)
    t._ensure_bufs((200, 200))
    return t


def _frame(*rects):
    """A frame with bright rectangles: each is (x, y, w, h)."""
    f = np.zeros((200, 200), dtype=np.uint8)
    for x, y, w, h in rects:
        f[y:y + h, x:x + w] = 255
    return f


ANIMAL = (20, 20, 40, 40)     # 1600 px
SPECK = (150, 150, 5, 5)      # 25 px


def _detect(t, frame):
    return t._detect_contours(t._prepare_gray(frame), t.background_gray)


def _centre(cand):
    return cand[2]


# --- the reported failure ---------------------------------------------------

def test_the_animal_wins_even_when_max_area_excludes_it():
    """Max Area below the animal must not hand the track to the speck."""
    t = _tracker(max_area=100)          # animal is 1600 px, speck is 25 px
    mask, candidates = _detect(t, _frame(ANIMAL, SPECK))

    chosen, _ = t._select_candidate(candidates, None, 0.0)
    assert chosen is not None, "nothing tracked at all"
    cx, cy = _centre(chosen)
    assert 20 <= cx <= 60 and 20 <= cy <= 60, (
        f"tracked {cx:.0f},{cy:.0f}; that is the speck, not the animal")


def test_that_case_is_reported_not_silently_papered_over():
    t = _tracker(max_area=100)
    _mask, candidates = _detect(t, _frame(ANIMAL, SPECK))
    t._select_candidate(candidates, None, 0.0)
    m = t.get_quality_metrics()
    assert m["tracked_out_of_range"] is True, (
        "the operator gets no signal that the bounds are wrong")
    assert m["rejected_too_large"] >= 1
    assert m["largest_rejected_area"] >= 1000


def test_min_area_above_the_animal_is_survivable_too():
    t = _tracker(min_area=5000, subject_min_area=5000)
    _mask, candidates = _detect(t, _frame(ANIMAL))
    chosen, _ = t._select_candidate(candidates, None, 0.0)
    assert chosen is not None
    cx, cy = _centre(chosen)
    assert 20 <= cx <= 60 and 20 <= cy <= 60


# --- correct bounds must not be undermined ----------------------------------

def test_an_in_range_blob_beats_an_out_of_range_one():
    """The demotion has to be strong enough that correct bounds still rule."""
    t = _tracker(max_area=2000)          # animal in range, big blob is not
    huge = (100, 20, 80, 80)             # 6400 px, out of range
    _mask, candidates = _detect(t, _frame(ANIMAL, huge))
    # Seed a track near the animal so scoring, not "largest", decides.
    t._recent_areas.extend([1600.0] * 5)
    chosen, _ = t._select_candidate(candidates, (40, 40), 60.0)
    cx, cy = _centre(chosen)
    assert 20 <= cx <= 60 and 20 <= cy <= 60, (
        f"an out-of-range blob outranked the correct in-range one at {cx},{cy}")
    assert t.get_quality_metrics()["tracked_out_of_range"] is False


def test_with_sane_bounds_nothing_is_reported_out_of_range():
    t = _tracker()
    _mask, candidates = _detect(t, _frame(ANIMAL))
    t._select_candidate(candidates, None, 0.0)
    m = t.get_quality_metrics()
    assert m["tracked_out_of_range"] is False
    assert m["rejected_too_large"] == 0


# --- the one bound that stays hard ------------------------------------------

def test_single_pixel_noise_never_becomes_a_candidate():
    """Without a floor, the demoted pool would fill with sensor noise."""
    t = _tracker(min_area=500, subject_min_area=500)
    dust = _frame((5, 5, 2, 2), (60, 60, 2, 2), (120, 120, 2, 2))
    _mask, candidates = _detect(t, dust)
    chosen, _ = t._select_candidate(candidates, None, 0.0)
    assert chosen is None, "tracked a few pixels of noise"


def test_an_empty_frame_still_reports_nothing():
    t = _tracker()
    _mask, candidates = _detect(t, np.zeros((200, 200), dtype=np.uint8))
    assert t._select_candidate(candidates, None, 0.0) == (None, False)


def test_the_fallback_pool_does_not_leak_between_frames():
    """Stale candidates from an earlier frame would be tracked as if live."""
    t = _tracker(max_area=100)
    _detect(t, _frame(ANIMAL))
    _mask, candidates = _detect(t, np.zeros((200, 200), dtype=np.uint8))
    chosen, _ = t._select_candidate(candidates, None, 0.0)
    assert chosen is None, "a blob from the previous frame was still tracked"


def test_the_warning_names_the_area_the_operator_must_bracket():
    """A Min Area set above the animal rejects nothing for being too LARGE,
    so reporting ``largest_rejected_area`` there read "0 px" and told the
    operator nothing. The number that helps is the area of the blob actually
    being tracked, what Min/Max Area has to be set around."""
    t = _tracker(min_area=5000, subject_min_area=5000)
    _mask, candidates = _detect(t, _frame(ANIMAL))
    chosen, _ = t._select_candidate(candidates, None, 0.0)
    m = t.get_quality_metrics()
    assert m["tracked_out_of_range"] is True
    assert m["largest_rejected_area"] == 0.0, "nothing was too large here"
    assert m["out_of_range_area"] == pytest.approx(chosen[0]), (
        "the warning would not name the tracked blob's own area")
    assert m["out_of_range_area"] > 1000


def test_the_area_is_cleared_once_the_bounds_are_right():
    t = _tracker()
    _mask, candidates = _detect(t, _frame(ANIMAL))
    t._select_candidate(candidates, None, 0.0)
    m = t.get_quality_metrics()
    assert m["tracked_out_of_range"] is False
    assert m["out_of_range_area"] == 0.0, (
        "a stale area would keep the red warning on screen after a fix")
