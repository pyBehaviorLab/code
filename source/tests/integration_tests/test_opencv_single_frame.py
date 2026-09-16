"""OpenCV backend returns ONE fresh frame per get_available_images call.

Low-latency live mode (CAP_PROP_BUFFERSIZE=1): the driver holds only the
freshest frame, so each call yields exactly one frame and the capture loop
publishes every frame, display runs at the full capture rate instead of
capture_rate / batch_size (the 7-of-20 fps regression). This guards against a
re-introduction of the blocking-lookahead batch drain.
"""
from __future__ import annotations

import numpy as np
import pytest

from source.video.cameras.opencv import OpenCVCamera


class _FakeCap:
    """Minimal cv2.VideoCapture stand-in: read() yields one frame, counts grabs."""
    def __init__(self):
        self.read_calls = 0
        self.grab_calls = 0

    def isOpened(self):
        return True

    def read(self):
        self.read_calls += 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def grab(self):
        self.grab_calls += 1
        return True

    def retrieve(self):
        return True, np.zeros((4, 4, 3), dtype=np.uint8)


def _cam_with_fake_cap():
    cam = OpenCVCamera.__new__(OpenCVCamera)
    cam._cap = _FakeCap()
    return cam


def test_returns_single_frame_per_call():
    cam = _cam_with_fake_cap()
    out = cam.get_available_images()
    assert out is not None
    assert len(out["images"]) == 1
    assert len(out["timestamps"]) == 1
    assert out["dropped_frames"] == 0


def test_does_not_lookahead_grab():
    """The freshest-frame path must not block-grab a future frame, that
    lookahead is exactly what batched 2 frames per call and halved display."""
    cam = _cam_with_fake_cap()
    cam.get_available_images()
    assert cam._cap.read_calls == 1
    assert cam._cap.grab_calls == 0   # no extra lookahead grab


def test_returns_none_when_read_fails():
    cam = OpenCVCamera.__new__(OpenCVCamera)

    class _DeadCap(_FakeCap):
        def read(self):
            return False, None
    cam._cap = _DeadCap()
    assert cam.get_available_images() is None


def test_returns_none_when_closed():
    cam = OpenCVCamera.__new__(OpenCVCamera)
    cam._cap = None
    assert cam.get_available_images() is None
