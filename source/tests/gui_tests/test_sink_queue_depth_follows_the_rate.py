"""A queue bounded in frames is a queue bounded for an assumed rate.

RecorderSink held a fixed 90 frames, described in its own comment as "~3 s at
30 fps". At 60 fps that is one and a half seconds: the depth it was designed
to have quietly halved on a faster camera, and the sink is LOSSLESS, so
overflow is a user-visible alarm rather than a dropped frame.

TrackerSink was worse. Its depth is ``1 s x fps_hint`` and ``fps_hint``
defaults to 30 with nothing in the codebase ever passing it, so the queue was
30 frames whatever the camera did.

Both are now bounded in seconds and told the rate the camera actually
delivers. See docs/dev/camera-modes-and-rates.md section 6.
"""
from __future__ import annotations

import pytest

from source.video.framebus.recorder_sink import RecorderSink
from source.video.framebus.sink_base import DropPolicy, Sink
from source.video.framebus.tracker_sink import TrackerSink


def _sink(seconds=None, maxsize=10):
    return Sink(name="t", maxsize=maxsize, seconds=seconds,
                policy=DropPolicy.DROP_OLDEST, workers=1)


def test_a_seconds_budget_follows_the_rate():
    s = _sink(seconds=3.0, maxsize=90)
    s.set_rate_hint(60.0)
    assert s._maxsize == 180, "three seconds at 60 fps is 180 frames"
    s.set_rate_hint(30.0)
    assert s._maxsize == 90


def test_it_never_shrinks_below_what_it_was_built_with():
    """A slow camera must not make a queue too shallow to absorb a burst."""
    s = _sink(seconds=3.0, maxsize=90)
    s.set_rate_hint(5.0)
    assert s._maxsize == 90, "3 s at 5 fps is 15 frames, which is too few"


def test_a_sink_with_no_budget_is_left_alone():
    """PoseSink's queue of 1 is a POLICY, not a duration."""
    s = _sink(seconds=None, maxsize=1)
    s.set_rate_hint(60.0)
    assert s._maxsize == 1


@pytest.mark.parametrize("bad", [0, -1, None])
def test_a_rate_that_says_nothing_changes_nothing(bad):
    s = _sink(seconds=3.0, maxsize=90)
    s.set_rate_hint(bad)
    assert s._maxsize == 90


def test_the_recorder_declares_three_seconds():
    r = RecorderSink()
    try:
        assert r._seconds == 3.0
        r.set_rate_hint(60.0)
        assert r._maxsize == 180, (
            "at 60 fps the lossless queue held 1.5 s, half its design depth")
    finally:
        r.stop()


def test_the_tracker_declares_one_second_and_stops_assuming_30():
    t = TrackerSink()
    try:
        assert t._seconds == 1.0
        assert t._maxsize == 30, "the fps_hint default nobody ever passed"
        t.set_rate_hint(60.0)
        assert t._maxsize == 60
    finally:
        t.stop()


# ── the session header says whose clock the frame times are ──────────────
#
# A UVC camera through OpenCV has no per-frame hardware timestamp, so every
# time in a session log is taken as the frame is handed over and carries the
# whole transport delay: 72 ms on a direct camera here, 121 to 125 ms through
# a CCTV chain. Spinnaker and XIMEA stamp each frame themselves. An analysis
# aligning video to controller time is holding a different quantity in each
# case, and the file did not say which.

def test_each_backend_declares_whose_clock_it_uses():
    from source.video.cameras.base import GenericCamera
    from source.video.cameras.opencv import OpenCVCamera
    from source.video.cameras.spinnaker import SpinnakerCamera
    assert GenericCamera.TIMESTAMP_SOURCE == "host", "the safe default"
    assert OpenCVCamera.TIMESTAMP_SOURCE == "host"
    assert SpinnakerCamera.TIMESTAMP_SOURCE == "hw"


def test_the_recorder_keeps_it_with_the_rate_it_decided():
    """It reaches the session header through the recorder, which already
    receives the resolved settings; an earlier version of this looked it up
    through attributes that do not exist and would have written "host" for
    every camera including the ones that stamp their own frames."""
    import threading
    from types import SimpleNamespace

    from source.video.recording.recorder import VideoRecorder
    rec = VideoRecorder.__new__(VideoRecorder)
    rec.camera_id, rec.fps, rec._grayscale = "c", 0.0, False
    rec.frame_queue, rec._queue_lock = [], threading.Lock()
    rec._adopt_camera_settings(SimpleNamespace(
        target_fps=30.0, delivered_fps=30.0, grayscale=False,
        timestamp_source="hw"))
    assert rec.timestamp_source == "hw"
    assert rec.requested_fps == 30.0
