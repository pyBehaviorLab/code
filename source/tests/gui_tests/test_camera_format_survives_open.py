"""The pixel format must still be MJPG when the camera starts delivering.

A UVC camera advertising MJPG at 800x600 up to 60 fps was opening in YUY2 at
30, with the operator's MJPEG choice intact in the project and correct at
every layer down to ``VideoManager.start_camera``. The request was being made
and then undone.

Two separate lines undid it, which is why the retry never helped:

* ``_apply_fourcc`` set the format, then the size. Setting the size makes the
  driver renegotiate and answer from its uncompressed list, so the format has
  to be re-asserted LAST, after every other property.
* ``configure()`` sets ``CAP_PROP_FPS``, and it runs at the END of
  ``begin_capturing``. So the format was established correctly and reverted
  four lines later, which also explains the silence: the format check ran
  before the line that broke it.

Measured on the rig, DirectShow, 800x600, 2026-09-10::

    FOURCC, W, H                -> YUY2 at 30
    FOURCC, W, H, FPS           -> YUY2 at 30
    FOURCC, W, H, FPS, FOURCC   -> MJPG at 30
    FOURCC, W, H, FOURCC        -> MJPG at 60

The fake below reproduces exactly that rule, so the test travels without the
camera: any property other than FOURCC resets the format to the uncompressed
default.
"""
from __future__ import annotations

import cv2
import pytest

from source.video.cameras.opencv import OpenCVCamera


class _FakeCap:
    """A UVC driver that renegotiates the format on any other property."""

    def __init__(self, *_a, **_k):
        self._fourcc = cv2.VideoWriter_fourcc(*"YUYV")
        self._w, self._h, self._fps = 640, 480, 30.0
        self.sets = []

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.sets.append(prop)
        if prop == cv2.CAP_PROP_FOURCC:
            self._fourcc = int(value)
        else:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                self._w = int(value)
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                self._h = int(value)
            elif prop == cv2.CAP_PROP_FPS:
                self._fps = float(value)
            # The behaviour under test: anything that is not the format
            # renegotiates, and the driver picks uncompressed.
            self._fourcc = cv2.VideoWriter_fourcc(*"YUYV")
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FOURCC: float(self._fourcc),
                cv2.CAP_PROP_FRAME_WIDTH: float(self._w),
                cv2.CAP_PROP_FRAME_HEIGHT: float(self._h),
                cv2.CAP_PROP_FPS: self._fps}.get(prop, 0.0)

    def read(self):
        return False, None

    def release(self):
        pass


@pytest.fixture
def fake_cap(monkeypatch):
    made = []

    def _factory(*a, **k):
        c = _FakeCap()
        made.append(c)
        return c

    monkeypatch.setattr(cv2, "VideoCapture", _factory)
    return made


def _open(fmt="mjpeg", fps=30):
    cam = OpenCVCamera(0, width=800, height=600, fps=fps, capture_format=fmt)
    cam.begin_capturing()
    return cam


def test_mjpg_survives_the_whole_open(fake_cap):
    cam = _open("mjpeg")
    assert cam._read_fourcc_str() == "MJPG", (
        "the format was requested and then undone before the camera "
        "delivered its first frame")


def test_the_format_is_the_last_property_set(fake_cap):
    """Whichever property is set last decides the format on this driver."""
    _open("mjpeg")
    cap = fake_cap[0]
    assert cap.sets, "nothing was configured"
    assert cap.sets[-1] == cv2.CAP_PROP_FOURCC, (
        "the format is not sealed last, so the driver renegotiated after it")


def test_setting_the_rate_afterwards_does_not_lose_it(fake_cap):
    """``configure()`` runs at the end of the open and again on every rate
    change; each one used to put the camera back to uncompressed."""
    cam = _open("mjpeg")
    cam.configure(fps=60)
    assert cam._read_fourcc_str() == "MJPG", (
        "changing the frame rate dropped the pixel format")


def test_an_explicit_uncompressed_request_is_still_honoured(fake_cap):
    """The fix must not force MJPG on someone who asked for raw."""
    cam = _open("yuv")
    assert cam._read_fourcc_str() != "MJPG"


def test_the_size_still_reaches_the_driver(fake_cap):
    """Sealing the format must not cost the resolution."""
    cam = _open("mjpeg")
    assert (int(cam._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cam._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (800, 600)


# ── which doors the camera is tried through, per platform ────────────────

def test_linux_tries_v4l2_then_gstreamer_then_any():
    """Linux has no Media Foundation, and on a Jetson the CSI nodes are
    libcamera or nvargus managed: GStreamer is the only backend that opens
    them at all. CAP_ANY does try GStreamer, but it reports the same failure
    for "this build has no GStreamer" as for "this device refused", so it is
    named explicitly rather than left to the fallback."""
    import sys
    import unittest.mock as mock

    import source.video.cameras.opencv as m
    with mock.patch.object(m.os, "name", "posix"), \
         mock.patch.object(sys, "platform", "linux"):
        order = m._opencv_backend_order()
    expected = [cv2.CAP_V4L2]
    gst = getattr(cv2, "CAP_GSTREAMER", None)
    if gst is not None:
        expected.append(gst)
    expected.append(cv2.CAP_ANY)
    assert order == expected, f"Linux backend order is {order}"


def test_windows_keeps_dshow_first():
    """MSMF was tried in front of DSHOW and reverted: it took 37 s to a first
    frame and then delivered nothing. Reliable beats fast for an unattended
    session, so this order is a decision and not an accident."""
    import sys
    import unittest.mock as mock

    import source.video.cameras.opencv as m
    with mock.patch.object(m.os, "name", "nt"), \
         mock.patch.object(sys, "platform", "win32"):
        order = m._opencv_backend_order()
    assert order[0] == cv2.CAP_DSHOW
    assert cv2.CAP_MSMF in order


def test_a_build_without_gstreamer_still_returns_a_usable_order(monkeypatch):
    """``cv2.CAP_GSTREAMER`` is looked up defensively so an OpenCV wheel
    without it cannot raise on a machine that never needed it."""
    import sys
    import unittest.mock as mock

    import source.video.cameras.opencv as m
    monkeypatch.delattr(cv2, "CAP_GSTREAMER", raising=False)
    with mock.patch.object(m.os, "name", "posix"), \
         mock.patch.object(sys, "platform", "linux"):
        order = m._opencv_backend_order()
    assert order == [cv2.CAP_V4L2, cv2.CAP_ANY]
