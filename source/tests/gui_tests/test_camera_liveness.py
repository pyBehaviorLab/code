"""Camera liveness contract.

Before: a brief camera loss was silent, the run loop served the stale last
frame, ``connected`` stayed True, no health event fired, and a permanent loss
killed the thread while still marked connected (so a box "recorded" from a
dead camera). Now the CameraThread tracks a frame-flow state driven by frame
age (streaming → stalled → reconnecting → streaming) and the Pipeline
watchdog banners every box on a stalled/lost/recovered camera.

Assertions read the liveness state directly: ``_liveness`` is the frame-flow
state machine, ``_streaming`` the flowing/not-flowing flag the Pipeline and
``VideoManager.is_camera_streaming`` consume.
"""
import time

import numpy as np

from source.video.cameras.capture import CameraThread
from source.video.framebus.controller import Pipeline


def _thread():
    # __init__ only, no device opened, no run loop started.
    t = CameraThread(camera_id=0)
    return t


def _batch():
    return {"images": [np.zeros((8, 8, 3), np.uint8)],
            "timestamps": [time.monotonic_ns()]}


def test_first_frame_marks_streaming():
    t = _thread()
    assert t._liveness == "starting"
    assert not t._streaming
    t._process_batch(_batch())
    assert t._liveness == "streaming"
    assert t._streaming


def test_stall_detected_after_silence():
    t = _thread()
    t._process_batch(_batch())                     # streaming
    # Backdate the last-good time beyond the stall threshold, then poll empty.
    t._last_good_ns = time.monotonic_ns() - int((t._stall_after_s + 0.2) * 1e9)
    t._handle_empty_batch()
    assert t._liveness == "stalled"
    assert not t._streaming


def test_brief_gap_does_not_trip_stall():
    t = _thread()
    t._process_batch(_batch())
    # A single empty poll right after a good frame must NOT flip to stalled.
    t._handle_empty_batch()
    assert t._liveness == "streaming"
    assert t._streaming


def test_recovers_on_next_frame():
    t = _thread()
    t._process_batch(_batch())
    t._last_good_ns = time.monotonic_ns() - int((t._stall_after_s + 0.2) * 1e9)
    t._handle_empty_batch()
    assert t._liveness == "stalled"
    # A real frame returns → back to streaming.
    t._process_batch(_batch())
    assert t._liveness == "streaming"
    assert t._streaming


def test_empty_batch_never_kills_the_loop():
    # A down camera keeps retrying without ever signalling loop-exit; the run
    # loop exits only on self.running=False. Calling it repeatedly must neither
    # raise nor leave a nonsensical state.
    t = _thread()
    t._process_batch(_batch())
    t._last_good_ns = time.monotonic_ns() - int((t._stall_after_s + 0.2) * 1e9)
    for _ in range(5):
        t._handle_empty_batch()                     # no return, must not raise
    assert t._liveness in ("stalled", "reconnecting")
    assert not t._streaming


# ── Pipeline watchdog fans health on transitions ────────────────────────

class _FakeCam:
    def __init__(self, state): self._liveness = state


class _FakeVM:
    def __init__(self, cams, box_map):
        self.cameras = cams
        self.box_camera_map = box_map


def _watchdog_stub(cams, box_map):
    stub = type("S", (), {})()
    stub._observe_camera_liveness = Pipeline._observe_camera_liveness.__get__(stub)
    stub.video_manager = _FakeVM(cams, box_map)
    stub._cam_liveness = {}
    stub.events = []
    stub._notify_health = lambda sid, reason: stub.events.append((sid, reason))
    return stub


def test_watchdog_banners_boxes_on_stall_and_recovery():
    cam = _FakeCam("streaming")
    stub = _watchdog_stub({7: cam}, {1: 7, 2: 7, 3: 99})

    stub._observe_camera_liveness()                 # initial streaming → quiet
    assert stub.events == []

    cam._liveness = "stalled"
    stub._observe_camera_liveness()
    # both boxes on camera 7 bannered; box 3 (other camera) untouched
    assert (1, "Camera stopped delivering frames") in stub.events
    assert (2, "Camera stopped delivering frames") in stub.events
    assert all(sid != 3 for sid, _ in stub.events)

    stub.events.clear()
    cam._liveness = "reconnecting"
    stub._observe_camera_liveness()
    assert (1, "Camera lost, reconnecting…") in stub.events

    stub.events.clear()
    cam._liveness = "streaming"
    stub._observe_camera_liveness()
    assert (1, "Camera recovered") in stub.events


def test_watchdog_quiet_when_state_unchanged():
    cam = _FakeCam("streaming")
    stub = _watchdog_stub({7: cam}, {1: 7})
    stub._observe_camera_liveness()
    stub._observe_camera_liveness()
    stub._observe_camera_liveness()
    assert stub.events == []


class _FakeThreadCam:
    """Camera whose thread liveness (isRunning/connected) can be toggled, a
    plain ``_FakeCam`` has neither, so it exercises the normal state path."""
    def __init__(self, running=True, connected=True, state="streaming"):
        self._running = running
        self.connected = connected
        self._liveness = state

    def isRunning(self):
        return self._running


def test_watchdog_banners_a_dead_capture_thread():
    # A thread that dies mid-stream leaves _liveness frozen at "streaming" but
    # isRunning() goes False, must still banner every box on that camera.
    cam = _FakeThreadCam(running=True, connected=True, state="streaming")
    stub = _watchdog_stub({7: cam}, {1: 7, 2: 7})
    stub._observe_camera_liveness()                 # streaming → quiet
    assert stub.events == []
    cam._running = False                            # thread died
    stub._observe_camera_liveness()
    assert (1, "Camera disconnected, capture thread stopped") in stub.events
    assert (2, "Camera disconnected, capture thread stopped") in stub.events


def test_watchdog_ignores_thread_that_never_came_up():
    # prev is None (never streamed) → a not-running thread is NOT a death event
    # (avoids false banners at startup / on intentional teardown).
    cam = _FakeThreadCam(running=False, connected=False, state="starting")
    stub = _watchdog_stub({7: cam}, {1: 7})
    stub._observe_camera_liveness()
    assert stub.events == []
