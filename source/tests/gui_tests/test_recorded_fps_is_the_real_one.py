"""A recording is stamped with the rate the camera actually delivers.

``cap.set(CAP_PROP_FPS, 10)`` returns success and ``cap.get`` reads back 10 on
a camera that then delivers 15. Measured on this rig at 640x480:

    asked 30 -> driver reports 30.0 -> delivered 30.01   honoured
    asked 20 -> driver reports 20.0 -> delivered 19.92   honoured
    asked 10 -> driver reports 10.0 -> delivered 15.00   NOT honoured

Nothing decimates to hold the request, so the recorder receives 15 fps of
frames. Stamping the file with the REQUEST wrote 15 fps of frames into a file
declared as 10: it plays back 1.5 times slow and every timestamp derived from
its nominal rate is wrong. Silently, because every property still reads back
as asked.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from source.video.recording.recorder import VideoRecorder


def _adopt(target, delivered):
    """Run the recorder's own rate decision over one camera outcome."""
    rec = VideoRecorder.__new__(VideoRecorder)
    rec.camera_id = "camtest"
    rec.fps = 0.0
    rec._grayscale = False
    rec.frame_queue = []
    import threading
    rec._queue_lock = threading.Lock()
    rec._adopt_camera_settings(
        SimpleNamespace(target_fps=target, delivered_fps=delivered,
                        grayscale=False))
    return rec.fps


def test_a_rate_the_camera_will_not_hold_is_not_what_gets_written():
    """The rig case: asked 10, delivered 15."""
    assert _adopt(10.0, 15.0) == 15.0, (
        "the file would claim 10 fps while holding 15 fps of frames, so it "
        "plays back 1.5x slow")


def test_an_honoured_rate_keeps_the_round_number():
    """A measurement carries about a frame of noise.

    Stamping 30.01 on a camera that really runs at 30 is less accurate, not
    more, so the request wins whenever the camera is holding it.
    """
    assert _adopt(30.0, 30.01) == 30.0
    assert _adopt(20.0, 19.92) == 20.0


def test_an_unmeasured_camera_falls_back_to_the_request():
    """No measurement is not the same as a bad one."""
    assert _adopt(25.0, None) == 25.0
    assert _adopt(25.0, 0.0) == 25.0


@pytest.mark.parametrize("want,got", [(10.0, 15.0), (30.0, 5.0), (60.0, 30.0)])
def test_any_rate_the_driver_lies_about_is_caught(want, got):
    assert _adopt(want, got) == got


def _cam(delivered, measured_at):
    from source.video.cameras.opencv import OpenCVCamera
    cam = OpenCVCamera.__new__(OpenCVCamera)
    cam.delivered_fps = delivered
    cam._delivered_for_fps = measured_at
    cam.fps_warning = "asked %s, got %s" % (measured_at, delivered)
    return cam


def test_a_measurement_never_travels_across_a_rate_change():
    """A measurement belongs to the rate it was taken at.

    ``configure(fps=...)`` runs again on every rate change, and carrying the
    previous rate's measurement would hand the recorder a number for a mode
    the camera is no longer running. Dropping it is safe, because an absent
    measurement makes the recorder fall back to the request; a wrong one
    makes it stamp the file confidently and incorrectly.
    """
    from source.video.cameras.opencv import OpenCVCamera
    cam = _cam(15.0, 10.0)
    assert OpenCVCamera._delivered_at(cam, 30.0) is None, (
        "a 10 fps measurement was carried onto a 30 fps mode")
    assert cam.fps_warning is None, "and its complaint went with it"


def test_a_measurement_survives_a_configure_at_the_same_rate():
    from source.video.cameras.opencv import OpenCVCamera
    cam = _cam(15.0, 10.0)
    assert OpenCVCamera._delivered_at(cam, 10.0) == 15.0
    assert cam.fps_warning, "and so does the complaint about it"


def test_a_camera_that_was_never_measured_reports_nothing():
    from source.video.cameras.opencv import OpenCVCamera
    cam = _cam(None, None)
    assert OpenCVCamera._delivered_at(cam, 30.0) is None
