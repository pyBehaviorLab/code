"""Reference background: median capture, staleness, and shape validation.

Differencing is only as good as the image it subtracts, so these cover the
ways a reference goes wrong, captured from one frame, captured yesterday,
captured at a resolution the camera no longer uses.
"""
import numpy as np
import pytest

from source.video.tracking import background as bg


def _arena(w=200, h=120, level=200):
    return np.full((h, w, 3), level, np.uint8)


def _with_animal(cx, w=200, h=120):
    f = _arena(w, h)
    f[50:70, cx:cx + 20] = 20          # a dark blob somewhere
    return f


# ── median capture ────────────────────────────────────────────────────────

def test_median_erases_a_moving_animal():
    """The whole point: the animal is somewhere different in each sample, so
    it medians out and the arena survives."""
    frames = [_with_animal(cx) for cx in (10, 50, 90, 130, 170)]
    it = iter(frames)
    out = bg.capture_median(lambda: next(it, None), samples=5, seconds=0.0,
                            sleep=lambda _s: None)
    assert out is not None
    # No dark pixels left anywhere, the animal is gone from the reference.
    assert out.min() > 150, "the animal survived into the background"


def test_a_still_animal_is_captured_into_the_background():
    """The honest limitation, asserted so nobody assumes otherwise: an animal
    that never moves during the burst IS the background afterwards."""
    frames = [_with_animal(90) for _ in range(5)]
    it = iter(frames)
    out = bg.capture_median(lambda: next(it, None), samples=5, seconds=0.0,
                            sleep=lambda _s: None)
    assert out is not None
    assert out.min() < 100, "a motionless animal cannot be medianed away"


def test_too_few_frames_is_not_a_median():
    """Two frames is a frame, not a median, returning it would quietly
    reintroduce the single-frame failure this exists to avoid."""
    frames = [_arena(), None, _arena()]
    it = iter(frames)
    assert bg.capture_median(lambda: next(it, None), samples=3, seconds=0.0,
                             sleep=lambda _s: None) is None


def test_dropped_grabs_do_not_abort_the_burst():
    seq = [_with_animal(10), None, _with_animal(90), None, _with_animal(170)]
    it = iter(seq)
    out = bg.capture_median(lambda: next(it, None), samples=5, seconds=0.0,
                            sleep=lambda _s: None)
    assert out is not None


def test_a_resolution_change_restarts_the_burst():
    """Samples of two different views do not describe one arena."""
    small = [np.full((60, 100, 3), 200, np.uint8)] * 2
    big = [_with_animal(cx) for cx in (10, 90, 170)]
    it = iter(small + big)
    out = bg.capture_median(lambda: next(it, None), samples=5, seconds=0.0,
                            sleep=lambda _s: None)
    assert out is not None
    assert out.shape[:2] == (120, 200), "kept the pre-change shape"


# ── status classification ─────────────────────────────────────────────────

def test_missing_background():
    assert bg.status(background=None, frame_shape=(120, 200),
                     captured_at=None) == bg.MISSING


def test_fresh_background_is_ok():
    assert bg.status(background=_arena(), frame_shape=(120, 200),
                     captured_at="2026-08-14 08:00:00",
                     stale_after_hours=1e6) == bg.OK


def test_old_background_is_stale():
    assert bg.status(background=_arena(), frame_shape=(120, 200),
                     captured_at="2020-01-01 00:00:00") == bg.STALE


def test_shape_mismatch_outranks_staleness():
    """A background of the wrong shape is the finding to act on: stretching
    it mis-registers every arena edge and the tracker locks onto that."""
    assert bg.status(background=_arena(w=320, h=240), frame_shape=(120, 200),
                     captured_at="2020-01-01 00:00:00") == bg.MISMATCH


@pytest.mark.parametrize("code,blocks", [
    (bg.OK, False), (bg.MISSING, False), (bg.STALE, False), (bg.MISMATCH, True),
])
def test_only_a_shape_mismatch_blocks_starting(code, blocks):
    _msg, blocking = bg.describe(code, captured_at="2020-01-01 00:00:00",
                                 background=_arena(w=320, h=240),
                                 frame_shape=(120, 200))
    assert blocking is blocks


def test_stale_message_names_the_age():
    msg, _ = bg.describe(bg.STALE, captured_at="2020-01-01 00:00:00")
    assert "captured" in msg and "recapture" in msg.lower()


# ── occupancy sanity check ────────────────────────────────────────────────

def test_a_wholly_changed_view_looks_occupied():
    assert bg.looks_occupied(_arena(level=200), _arena(level=40)) is True


def test_an_animal_sized_difference_does_not():
    assert bg.looks_occupied(_arena(), _with_animal(90)) is False


def test_hours_since_handles_junk():
    assert bg.hours_since(None) is None
    assert bg.hours_since("not a date") is None
    assert bg.hours_since("2020-01-01 00:00:00") > 1.0
