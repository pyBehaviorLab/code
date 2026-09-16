"""What gets burned into a recorded frame, and when it stops.

The annotate callback is handed ``(frame, capture_ts)`` and used to ignore the
timestamp entirely, drawing whatever pose was last cached. Three consequences,
all of which look like "the annotation is slightly off" to whoever watches the
video afterwards:

  * a run that ends, a model that is switched off, or inference that stalls
    left the final keypoints painted on every later frame, fixed at a spot the
    animal had already left;
  * freshness judged against "now" instead of against the frame means the
    answer moves with how far behind the encoder is, so the same frame
    annotates differently under load;
  * a frame sitting in the encoder queue got a pose from well after it.

The live tile has always applied a staleness rule. The recorded video not
applying the same one is the two disagreeing about the same moment.
"""
from __future__ import annotations

import numpy as np
import pytest

from source.gui.base import _OVERLAY_MAX_AGE_NS, MainWindowBase
from source.video.framebus.types import OverlayState

pytest.importorskip("cv2")

BUILD = MainWindowBase._build_annotate_callback
TTL_S = _OVERLAY_MAX_AGE_NS / 1e9


class _Host:
    """Only what the callback reads."""

    _build_annotate_callback = BUILD
    _fresh_overlay_state = MainWindowBase._fresh_overlay_state
    # The saved video draws in the SAME colours as the live tile now; it used
    # to paint every keypoint in one yellow, so the burned-in video could not
    # be read the way the operator had learnt to read the screen.
    _project_part_colours = MainWindowBase._project_part_colours

    def __init__(self, state):
        self._overlay = {1: state}
        self._tracking_dialog_globals = {}


def _pose_state(seen_ns, x=40.0, y=30.0):
    return OverlayState(pose=[(x, y, 0.99)], body_parts=["centroid"],
                        confidence_threshold=0.5, last_seen_ns=seen_ns)


def _frame():
    return np.zeros((80, 120, 3), np.uint8)


def _painted(out):
    return out is not None and bool(np.any(out))


def test_a_pose_from_this_frame_is_drawn():
    now_ns = 10_000_000_000
    draw = _Host(_pose_state(now_ns))._build_annotate_callback(1)
    assert _painted(draw(_frame(), now_ns / 1e9))


def test_a_pose_older_than_the_ttl_is_not_drawn():
    """The stalled/stopped case: the frame keeps coming, the pose does not."""
    pose_ns = 10_000_000_000
    draw = _Host(_pose_state(pose_ns))._build_annotate_callback(1)
    late = (pose_ns / 1e9) + TTL_S * 2
    assert draw(_frame(), late) is None, (
        "a pose this old must not be burned into the frame; that is how a "
        "finished run leaves keypoints painted over an empty box")


def test_a_pose_from_well_after_the_frame_is_not_drawn():
    """The mirrored error: a queued frame annotated with a much later pose."""
    frame_ts = 10.0
    pose_ns = int((frame_ts + TTL_S * 2) * 1e9)
    draw = _Host(_pose_state(pose_ns))._build_annotate_callback(1)
    assert draw(_frame(), frame_ts) is None


def test_freshness_follows_the_frame_not_the_wall_clock():
    """Two frames, same cached pose: the one captured beside the pose is
    annotated and the one captured long after is not, and neither answer
    depends on when the encoder happens to get round to them."""
    pose_ns = 10_000_000_000
    draw = _Host(_pose_state(pose_ns))._build_annotate_callback(1)
    assert _painted(draw(_frame(), pose_ns / 1e9))
    assert draw(_frame(), pose_ns / 1e9 + TTL_S * 3) is None


def test_the_marker_lands_on_the_keypoint_not_a_pixel_up_and_left():
    """``int()`` truncates, and every keypoint moved toward the origin by it.

    Half-integer coordinates are the ordinary case, not a corner one: the
    model's peaks are integers in crop pixels but the letterbox inverse and
    the display scale both produce fractions.
    """
    now_ns = 10_000_000_000
    state = _pose_state(now_ns, x=40.6, y=30.6)
    out = _Host(state)._build_annotate_callback(1)(_frame(), now_ns / 1e9)
    ys, xs = np.nonzero(out[:, :, 1])
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    assert (cx, cy) == (41.0, 31.0), (
        f"marker centred at ({cx}, {cy}); 40.6 rounds to 41, and truncating "
        "to 40 is a bias toward the top-left on every single keypoint")


def test_a_state_with_no_pose_and_no_blob_returns_the_frame_unchanged():
    draw = _Host(OverlayState(last_seen_ns=10_000_000_000))\
        ._build_annotate_callback(1)
    assert draw(_frame(), 10.0) is None


def test_an_unstamped_state_still_draws():
    """``last_seen_ns`` unset is the pre-timestamp path, not a stale pose;
    refusing to draw there would silently disable annotation."""
    draw = _Host(_pose_state(0))._build_annotate_callback(1)
    assert _painted(draw(_frame(), 10.0))
