"""On V4L2 (Linux, Jetson) the rate that is asked for is the rate that runs.

Three defects, measured on the Jetson rig on 2026-09-15:

* A one-frame capture queue halves a fast camera. 640x480 MJPEG, a 120 fps
  mode: queue of 1 -> 49.5 fps, queue of 2 -> 98.8 fps, frame age the same
  (median 14 ms). Detect never set the queue and measured 98.8; capture set it
  and delivered 49.5, so Detect said one thing and the camera did another.
* uvcvideo refuses a frame-interval change on a running stream
  (``cap.set(CAP_PROP_FPS)`` returns False). A rate picked for a camera that
  was already live never took effect: asked 15, still delivering 30.
* The shortfall warning said "the driver reported <the request>".

Every other backend must behave exactly as before; the DirectShow cases pin it.
"""
from __future__ import annotations

import cv2
import pytest

from source.video.cameras import opencv as ocv
from source.video.cameras.opencv import OpenCVCamera


class _FakeCap:
    """Honours every property until ``streaming``; then refuses a rate change,
    as uvcvideo does."""

    def __init__(self, *_a, **_k):
        self.props = {cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
                      cv2.CAP_PROP_FRAME_WIDTH: 640.0,
                      cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
                      cv2.CAP_PROP_FPS: 30.0}
        self.buffersizes = []
        self.fps_sets = []
        self.streaming = False
        self.released = False

    def isOpened(self):
        return not self.released

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_BUFFERSIZE:
            self.buffersizes.append(int(value))
        if prop == cv2.CAP_PROP_FPS:
            self.fps_sets.append(float(value))
            if self.streaming:
                return False
        self.props[prop] = float(value)
        return True

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def read(self):
        return False, None

    def release(self):
        self.released = True


@pytest.fixture
def caps(monkeypatch):
    made = []

    def _factory(*_a, **_k):
        c = _FakeCap()
        made.append(c)
        return c

    monkeypatch.setattr(cv2, "VideoCapture", _factory)
    # No frames to count, and no 1.6 s drain-and-measure per open.
    monkeypatch.setattr(ocv, "measure_delivered_fps", lambda *a, **k: 0.0)
    return made


def _open(monkeypatch, backend, fps=30):
    monkeypatch.setattr(ocv, "_opencv_backend_order", lambda: [backend])
    cam = OpenCVCamera(0, width=640, height=480, fps=fps, capture_format="mjpeg")
    cam.begin_capturing()
    return cam


def test_v4l2_opens_with_a_two_frame_queue(caps, monkeypatch):
    _open(monkeypatch, cv2.CAP_V4L2)
    assert caps[0].buffersizes, "the capture queue depth was never set"
    assert set(caps[0].buffersizes) == {2}, (
        "a one-frame queue drops every other frame of a camera faster than "
        "the host's per-frame work")


def test_other_backends_keep_a_one_frame_queue(caps, monkeypatch):
    _open(monkeypatch, cv2.CAP_DSHOW)
    assert set(caps[0].buffersizes) == {1}


def test_a_new_rate_on_a_running_v4l2_camera_reopens_it(caps, monkeypatch):
    cam = _open(monkeypatch, cv2.CAP_V4L2, fps=30)
    caps[0].streaming = True
    cfg = cam.configure(fps=15)
    assert len(caps) == 2, "the refused rate was left unapplied"
    assert caps[0].released
    assert 15.0 in caps[1].fps_sets
    assert cfg.target_fps == 15.0
    assert cam._rate_on_device == 15.0


def test_the_same_rate_reapplied_does_not_reopen(caps, monkeypatch):
    """begin_capturing re-applies its own rate through configure(); that must
    not look like a change, or every open would reopen forever."""
    cam = _open(monkeypatch, cv2.CAP_V4L2, fps=30)
    caps[0].streaming = True
    cam.configure(fps=30)
    assert len(caps) == 1


def test_other_backends_do_not_reopen_on_a_refused_rate(caps, monkeypatch):
    cam = _open(monkeypatch, cv2.CAP_DSHOW, fps=30)
    caps[0].streaming = True
    cam.configure(fps=15)
    assert len(caps) == 1


def test_a_rate_that_will_not_open_falls_back_to_the_previous_one(caps, monkeypatch):
    cam = _open(monkeypatch, cv2.CAP_V4L2, fps=30)
    caps[0].streaming = True
    real_begin = OpenCVCamera.begin_capturing
    calls = {"n": 0}

    def flaky_begin(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("device busy")
        return real_begin(self)

    monkeypatch.setattr(OpenCVCamera, "begin_capturing", flaky_begin)
    cfg = cam.configure(fps=15)
    assert cam._cap is not None and cam._cap.isOpened(), (
        "a refused rate change left the box without a camera")
    assert cfg.target_fps == 30.0
    assert cam._rate_on_device == 30.0


def test_the_shortfall_warning_names_the_drivers_rate_not_the_request(monkeypatch):
    class _OneIntervalCap(_FakeCap):
        """640x480 MJPEG on the rig's camera: 120 fps whatever is asked."""

        def set(self, prop, value):
            ok = super().set(prop, value)
            if prop == cv2.CAP_PROP_FPS:
                self.props[prop] = 120.0
            return ok

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: _OneIntervalCap())
    monkeypatch.setattr(ocv, "_opencv_backend_order", lambda: [cv2.CAP_V4L2])
    monkeypatch.setattr(ocv, "measure_delivered_fps", lambda *a, **k: 98.8)
    # Keep the per-machine calibration file out of a unit test.
    monkeypatch.setattr(OpenCVCamera, "_remember_rate_rejected",
                        lambda self, asked, got: None)
    cam = OpenCVCamera(0, width=640, height=480, fps=30, capture_format="mjpeg")
    cam.begin_capturing()
    assert cam.fps_warning, "a 30 fps request delivering 98.8 raised no warning"
    assert "driver reports 120" in cam.fps_warning
    assert "reported 30" not in cam.fps_warning
