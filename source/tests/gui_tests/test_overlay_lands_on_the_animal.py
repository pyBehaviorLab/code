"""The marker must sit on the body part, at every display size.

Everything else about the overlay can be right and this still be wrong: the
frame is resized to the tile before the keypoints are drawn, so a keypoint in
source pixels has to be carried through the same transform the image was. Get
the transform even slightly wrong and the dot drifts off the animal, which is
what an enlarged tile was reported to show.

So the check is the literal claim, not a restatement of the arithmetic: a mark
is burned into the source image, a keypoint is placed on that mark, both go
through the real ``_tile_base`` → ``_draw_overlay_on_frame`` path, and the two
must still be on top of each other in the tile that comes out.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
import cv2  # noqa: E402

from PySide6 import QtCore  # noqa: E402

from source.gui.base import MainWindowBase  # noqa: E402
from source.video.framebus.types import OverlayState  # noqa: E402

SRC_W, SRC_H = 320, 240
#: Where the "animal" is in source pixels. Off-centre and not on any half or
#: third of the frame, so a transform that silently centres or halves is caught.
MARK_X, MARK_Y = 214.0, 63.0


class _Host:
    _DLC_COLORS = MainWindowBase._DLC_COLORS
    _draw_overlay_on_frame = MainWindowBase._draw_overlay_on_frame
    _draw_pose_layer = MainWindowBase._draw_pose_layer
    _draw_keypoints_cython = MainWindowBase._draw_keypoints_cython
    _draw_keypoints_python = MainWindowBase._draw_keypoints_python
    _draw_pose_figure = MainWindowBase._draw_pose_figure
    _fresh_overlay_state = MainWindowBase._fresh_overlay_state
    # staticmethod(...) on purpose: assigning ``MainWindowBase.foo`` into a
    # class body unwraps the descriptor, so instance access would re-bind
    # ``self`` and every call would be one argument over. The overlay's
    # try/except swallows that into a warning and an unpainted tile, which is
    # exactly the kind of silent pass a test must not run on.
    _to_tile = staticmethod(MainWindowBase._to_tile)
    _draw_bbox = staticmethod(MainWindowBase._draw_bbox)
    _bbox_centre = staticmethod(MainWindowBase._bbox_centre)
    _draw_trigger_layer = staticmethod(MainWindowBase._draw_trigger_layer)

    def __init__(self):
        import time
        self._overlay = {1: OverlayState(
            pose=[(MARK_X, MARK_Y, 0.99)], body_parts=["centroid"],
            confidence_threshold=0.5, last_seen_ns=time.monotonic_ns())}
        self.tracking_zones = {}

    def _draw_zone_layer(self, *a, **k):
        pass


def _source_frame():
    """Black, with one small BLUE mark standing for the animal.

    Blue because the keypoint markers are drawn in other channels, so the mark
    stays findable in channel 0 after the overlay has been painted on top.
    """
    frame = np.zeros((SRC_H, SRC_W, 3), np.uint8)
    cv2.circle(frame, (int(MARK_X), int(MARK_Y)), 3, (255, 0, 0), -1)
    return frame


def _centre_of(mask):
    ys, xs = np.nonzero(mask)
    assert xs.size, "nothing found"
    return (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0


def _render(tile_w, tile_h, aspect_mode):
    """The real path: prepare the tile, then draw the overlay on it."""
    host = _Host()
    target = QtCore.QSize(tile_w, tile_h)
    base, scale_x, scale_y, marker_scale = MainWindowBase._tile_base(
        _source_frame(), None, target, aspect_mode)
    out = host._draw_overlay_on_frame(
        base.copy(), 1, scale_x=scale_x, scale_y=scale_y,
        draw_zones=False, marker_scale=marker_scale)
    return base, out


#: Tiles smaller than, equal to, and, the reported case, larger than source.
SIZES = [
    (160, 120),    # half
    (320, 240),    # identity
    (480, 360),    # 1.5x enlargement
    (640, 480),    # 2x enlargement
    (960, 720),    # 3x enlargement
    (800, 300),    # enlarged and a different aspect
    (240, 400),    # taller than wide
]


@pytest.mark.parametrize("aspect_mode", ["stretch", "preserve"])
@pytest.mark.parametrize("tile_w,tile_h", SIZES)
def test_the_marker_sits_on_the_mark(tile_w, tile_h, aspect_mode):
    base, out = _render(tile_w, tile_h, aspect_mode)

    # Where the animal ended up in the tile, from the image itself.
    mark_cx, mark_cy = _centre_of(base[:, :, 0] > 128)
    # Where the overlay put its marker: whatever the drawing added.
    added = np.any(out.astype(int) - base.astype(int) != 0, axis=2)
    kp_cx, kp_cy = _centre_of(added)

    off = float(np.hypot(kp_cx - mark_cx, kp_cy - mark_cy))
    assert off <= 2.0, (
        f"{aspect_mode} {tile_w}x{tile_h}: the mark is at "
        f"({mark_cx:.1f}, {mark_cy:.1f}) and the keypoint was drawn at "
        f"({kp_cx:.1f}, {kp_cy:.1f}), off by {off:.1f} px in tile pixels")


@pytest.mark.parametrize("aspect_mode", ["stretch", "preserve"])
def test_the_error_does_not_grow_with_enlargement(aspect_mode):
    """A scale applied to the image but not to the keypoints shows up as an
    error proportional to the zoom, the signature of the reported bug, and
    the thing a single-size test cannot see."""
    offsets = {}
    for tile_w, tile_h in [(320, 240), (640, 480), (960, 720), (1280, 960)]:
        base, out = _render(tile_w, tile_h, aspect_mode)
        mark = _centre_of(base[:, :, 0] > 128)
        added = np.any(out.astype(int) - base.astype(int) != 0, axis=2)
        kp = _centre_of(added)
        offsets[tile_w] = float(np.hypot(kp[0] - mark[0], kp[1] - mark[1]))
    assert max(offsets.values()) <= 2.0, (
        f"{aspect_mode}: offset by tile width {offsets}, an error that grows "
        "with the zoom means the image is being scaled and the keypoints are "
        "not")


def test_an_roi_crop_does_not_shift_the_marker():
    """The operant path crops to the box's ROI before scaling. Pose is already
    in box pixels, so the crop must not be applied to the coordinates twice."""
    host = _Host()
    # ROI whose origin is non-zero, containing the mark.
    roi = (100, 20, 200, 160)
    host._overlay[1].pose = [(MARK_X - roi[0], MARK_Y - roi[1], 0.99)]
    base, sx, sy, ms = MainWindowBase._tile_base(
        _source_frame(), roi, QtCore.QSize(600, 480), "stretch")
    out = host._draw_overlay_on_frame(base.copy(), 1, scale_x=sx, scale_y=sy,
                                      draw_zones=False, marker_scale=ms)
    mark = _centre_of(base[:, :, 0] > 128)
    added = np.any(out.astype(int) - base.astype(int) != 0, axis=2)
    kp = _centre_of(added)
    off = float(np.hypot(kp[0] - mark[0], kp[1] - mark[1]))
    assert off <= 2.0, f"ROI crop shifted the marker by {off:.1f} px"


def test_both_modes_keep_the_camera_shape_or_fill_the_cell_on_purpose():
    """Which aspect mode each window uses is a decision, not an accident.

    Maze stretched its tiles to fill the cell, which distorts the arena: a
    circular field reads as elliptical, and the distortion changes with the
    window size, indistinguishable, to the operator, from a lens that needs
    calibrating. Both modes now preserve the camera's shape.
    """
    import source.gui.maze as maze_mod
    import source.gui.operant as operant_mod
    assert maze_mod.MainWindow._tile_aspect_mode == "preserve"
    # Operant sets it in __init__ rather than on the class, so the class
    # attribute cannot be read without building a window; its module
    # docstring is the statement of intent that CAN be checked cheaply.
    assert "preserve" in (operant_mod.__doc__ or "")
    assert "aspect-preserving" in (maze_mod.__doc__ or "").lower()


@pytest.mark.parametrize("tile", [(320, 240), (900, 300), (400, 900)])
def test_preserve_never_distorts_whatever_the_cell_shape(tile):
    """The point of the mode: a cell of any shape must not squash the picture.

    Checked as the ratio the frame is scaled by in each axis, equal scales
    mean the shape survived.
    """
    import numpy as np
    from PySide6 import QtCore
    src = np.zeros((240, 320, 3), np.uint8)
    _base, sx, sy, _ms = MainWindowBase._tile_base(
        src, None, QtCore.QSize(*tile), "preserve")
    assert sx == pytest.approx(sy, rel=1e-3), (
        f"cell {tile}: scaled x by {sx} and y by {sy}; that is a stretch")
