"""Characterisation of ``MainWindowBase._draw_overlay_on_frame``.

Pins the behaviour of a 174-line per-frame paint method BEFORE splitting it.
It is driven unbound against a stub host, which is how the real caller
(``update_box_display``) reaches it, with the frame already resized to the
tile, so every keypoint and box has to be scaled to match.

Each test names a decision the method makes:

  * a stale overlay draws nothing, a stopped run must not leave a ghost
  * pose mode and blob mode mark the animal differently, on purpose
  * zones burn in for export but composite live, and the occupancy fill
    still follows the animal either way
  * a broken draw returns a frame, never raises into the paint loop
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from source.gui.base import MainWindowBase
from source.video.framebus.types import OverlayState

pytest.importorskip("cv2")

DRAW = MainWindowBase._draw_overlay_on_frame


class _Host:
    """The surface the overlay renderer reads."""

    _DLC_COLORS = MainWindowBase._DLC_COLORS
    # Source->tile coordinate mapping. staticmethod(...) so instance
    # access does not re-bind self; the renderer's try/except turns a
    # missing or mis-bound helper into an unpainted tile and a log line,
    # so a stub that omits it makes every assertion here vacuous.
    _to_tile = staticmethod(MainWindowBase._to_tile)

    def __init__(self, state=None, zones=None):
        self._overlay = {1: state} if state is not None else {}
        self.tracking_zones = {1: zones} if zones is not None else {}
        self.filled = []

    # Real implementations, borrowed from the class under test; only the
    # live zone-fill is stubbed so the test can observe it.
    def _fill_occupied_zones_live(self, frame, zones, centroid):
        self.filled.append(centroid)

    _draw_keypoints_python = MainWindowBase._draw_keypoints_python
    _draw_keypoints_cython = MainWindowBase._draw_keypoints_cython
    # Both drawers hand off to this one for edges, markers and labels.
    _draw_pose_figure = MainWindowBase._draw_pose_figure
    _fresh_overlay_state = MainWindowBase._fresh_overlay_state
    _bbox_centre = staticmethod(MainWindowBase._bbox_centre)
    _draw_pose_layer = MainWindowBase._draw_pose_layer
    _draw_zone_layer = MainWindowBase._draw_zone_layer
    _draw_bbox = staticmethod(MainWindowBase._draw_bbox)
    _draw_trigger_layer = staticmethod(MainWindowBase._draw_trigger_layer)


def _frame(h=120, w=160):
    return np.zeros((h, w, 3), np.uint8)


def _fresh(**kw):
    kw.setdefault("last_seen_ns", time.monotonic_ns())
    return OverlayState(**kw)


def _painted(frame):
    """How many pixels the renderer touched."""
    return int(np.count_nonzero(frame))


# ── freshness gate ───────────────────────────────────────────────────────

def test_no_overlay_state_draws_nothing():
    out = DRAW(_Host(), _frame(), 1)
    assert _painted(out) == 0


def test_a_stale_overlay_draws_nothing():
    """A run that stopped must not leave its last keypoints painted on the
    tile forever."""
    stale = OverlayState(pose=[[10.0, 10.0, 0.99]], body_parts=["nose"],
                         last_seen_ns=time.monotonic_ns() - 10 * 1_000_000_000)
    out = DRAW(_Host(stale), _frame(), 1)
    assert _painted(out) == 0


def test_a_state_that_was_never_stamped_is_treated_as_stale():
    never = OverlayState(pose=[[10.0, 10.0, 0.99]], body_parts=["nose"],
                         last_seen_ns=0)
    assert _painted(DRAW(_Host(never), _frame(), 1)) == 0


# ── the source frame is never mutated ────────────────────────────────────

def test_the_caller_s_frame_is_left_untouched():
    """The bus hands the same buffer to several consumers; drawing in place
    would corrupt what the recorder writes."""
    state = _fresh(pose=[[20.0, 20.0, 0.99]], body_parts=["nose"])
    src = _frame()
    out = DRAW(_Host(state), src, 1)
    assert _painted(src) == 0
    assert out is not src


def test_a_grayscale_frame_comes_back_in_colour():
    state = _fresh(pose=[[20.0, 20.0, 0.99]], body_parts=["nose"])
    out = DRAW(_Host(state), np.zeros((120, 160), np.uint8), 1)
    assert out.ndim == 3 and out.shape[2] == 3


# ── pose mode ────────────────────────────────────────────────────────────

def test_confident_keypoints_are_drawn():
    state = _fresh(pose=[[40.0, 30.0, 0.99], [60.0, 30.0, 0.99]],
                   body_parts=["nose", "tail"], confidence_threshold=0.5)
    assert _painted(DRAW(_Host(state), _frame(), 1)) > 0


def test_a_low_confidence_keypoint_is_not_drawn():
    state = _fresh(pose=[[40.0, 30.0, 0.05]], body_parts=["nose"],
                   confidence_threshold=0.5)
    assert _painted(DRAW(_Host(state), _frame(), 1)) == 0


def test_pose_mode_does_not_add_a_red_centroid_dot():
    """The coloured keypoints already mark the animal; an extra red dot reads
    as a separate body part."""
    state = _fresh(pose=[[40.0, 30.0, 0.99]], body_parts=["nose"])
    out = DRAW(_Host(state), _frame(), 1)
    # OpenCV is BGR: a pure red dot would be (0, 0, 255).
    red_only = ((out[:, :, 2] == 255) & (out[:, :, 0] == 0)
                & (out[:, :, 1] == 0))
    assert not red_only.any()


def test_blob_mode_marks_the_animal_with_a_red_dot():
    state = _fresh(bbox=(40, 30, 20, 20))
    out = DRAW(_Host(state), _frame(), 1)
    red_only = ((out[:, :, 2] == 255) & (out[:, :, 0] == 0)
                & (out[:, :, 1] == 0))
    assert red_only.any()


# ── scaling into tile coordinates ────────────────────────────────────────

def test_keypoints_follow_the_display_scale():
    """The caller resizes the frame first, so a keypoint at x=80 in source
    space must land near x=40 on a half-scale tile."""
    state = _fresh(pose=[[80.0, 60.0, 0.99]], body_parts=["nose"])
    out = DRAW(_Host(state), _frame(), 1, scale_x=0.5, scale_y=0.5)
    ys, xs = np.nonzero(out.any(axis=2))
    assert 30 <= xs.mean() <= 50
    assert 20 <= ys.mean() <= 40


def test_the_bounding_box_follows_the_display_scale():
    state = _fresh(bbox=(80, 60, 40, 40))
    out = DRAW(_Host(state), _frame(), 1, scale_x=0.5, scale_y=0.5)
    _ys, xs = np.nonzero(out.any(axis=2))
    assert xs.max() <= 65                 # (80+40) * 0.5 = 60, plus stroke


# ── zones ────────────────────────────────────────────────────────────────

def test_the_live_path_composites_zones_instead_of_burning_them_in():
    """The display keeps zone outlines in a cached QPainter layer; only the
    occupancy fill is redrawn per frame."""
    state = _fresh(bbox=(40, 30, 20, 20))
    host = _Host(state, zones=[{"name": "left"}])
    DRAW(host, _frame(), 1, draw_zones=False)
    assert len(host.filled) == 1          # fill ran, outlines did not


def test_no_centroid_means_no_occupancy_fill():
    state = _fresh()                      # neither pose nor blob
    host = _Host(state, zones=[{"name": "left"}])
    DRAW(host, _frame(), 1, draw_zones=False)
    assert host.filled == []


# ── robustness ───────────────────────────────────────────────────────────

def test_a_broken_renderer_returns_a_frame_rather_than_raising():
    """This runs inside the paint loop; an exception would kill the tile."""
    class _Bad(_Host):
        def _fill_occupied_zones_live(self, *a):
            raise RuntimeError("zone renderer died")

    state = _fresh(bbox=(40, 30, 20, 20))
    host = _Bad(state, zones=[{"name": "left"}])
    out = DRAW(host, _frame(), 1, draw_zones=False)
    assert out is not None and out.shape == (120, 160, 3)


def test_a_malformed_pose_row_does_not_kill_the_frame():
    state = _fresh(pose=[["x", "y"]], body_parts=["nose"])
    out = DRAW(_Host(state), _frame(), 1)
    assert out is not None
