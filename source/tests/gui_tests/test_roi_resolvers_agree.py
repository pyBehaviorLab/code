"""The pose and the picture must be cut from the same rectangle.

Two modules turn one normalized ROI into pixels:

  * ``VideoSegmentProcessor.extract_segment`` cuts the slice the pose is
    inferred on, so keypoint coordinates are measured from ITS origin;
  * ``MainWindowBase._get_box_roi`` cuts the frame shown and recorded.

They rounded differently, one ``round``, one ``int``, so for any normalized
value whose product with the frame size reached a half pixel, the picture was
cut one pixel up and left of the coordinate origin, and every keypoint drawn on
it sat a pixel off. Small, constant, and invisible to any test that only ever
exercised one of the two.

Checked against each other on real numbers rather than against a restatement of
either one's arithmetic.
"""
from __future__ import annotations

import numpy as np
import pytest

from source.gui.base import MainWindowBase
from source.video.cameras.capture import VideoSegmentProcessor

pytest.importorskip("cv2")


class _Host:
    _get_box_roi = MainWindowBase._get_box_roi

    def __init__(self, boxes):
        self.video_segment_config = {"boxes": boxes}

    def _box_roi_lookup(self, setup_id, frame=None):
        # No widget and no session pixel cache: the percent branch is what is
        # under test, and it is the branch that survives a resolution change.
        raise NotImplementedError


def _both(percent, w, h):
    """``(bus_rect, display_rect)`` for one normalized ROI at ``w`` x ``h``."""
    seg = VideoSegmentProcessor({"boxes": [
        {"box_id": 1, "box_number": 1, "geometry": {"percent": dict(percent)}}]})
    frame = np.zeros((h, w, 3), np.uint8)
    cut = seg.extract_segment(frame, 1)
    key = (1, h, w)
    bus = seg._coord_cache.get(key)

    host = _Host([{"box_id": 1, "box_number": 1,
                   "geometry": {"percent": dict(percent)}}])
    disp = host._get_box_roi(1, frame)
    assert cut is not None, "the bus refused to cut this ROI"
    return bus, disp


#: Normalized ROIs whose pixel products land on and around a half pixel, the
#: only place the two roundings can disagree, and unremarkable values a person
#: would actually draw.
CASES = [
    ({"x": 0.1234, "y": 0.2345, "width": 0.5, "height": 0.5}, 640, 480),
    ({"x": 0.3, "y": 0.3, "width": 0.4, "height": 0.4}, 641, 481),
    ({"x": 0.16255, "y": 0.20416, "width": 0.6, "height": 0.6}, 1280, 720),
    ({"x": 0.05, "y": 0.05, "width": 0.9, "height": 0.9}, 1920, 1080),
    ({"x": 1 / 3, "y": 1 / 7, "width": 0.5, "height": 0.5}, 800, 600),
    ({"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}, 640, 480),
]


@pytest.mark.parametrize("percent,w,h", CASES)
def test_the_two_resolvers_return_the_same_rectangle(percent, w, h):
    bus, disp = _both(percent, w, h)
    assert tuple(bus) == tuple(disp), (
        f"the pose is measured from {tuple(bus)} and the picture is cut at "
        f"{tuple(disp)}, the overlay is off by "
        f"({bus[0] - disp[0]}, {bus[1] - disp[1]}) px")


def test_a_case_that_actually_used_to_differ():
    """Guards the guard: at least one case must exercise a half-pixel product,
    or this whole file passes for the wrong reason."""
    exercised = False
    for percent, w, h in CASES:
        for key, dim in (("x", w), ("y", h), ("width", w), ("height", h)):
            frac = (percent[key] * dim) % 1.0
            if frac >= 0.5:
                exercised = True
    assert exercised, ("no case rounds up anywhere, so truncation and "
                       "rounding agree everywhere and the test proves nothing")
