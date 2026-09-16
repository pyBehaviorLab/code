"""Per-camera VideoManager behaviour, the multi-camera fix.

Two cameras must be able to run at different FPS/resolution, and setting the
manager DEFAULT fps must not silently retune already-running cameras (the old
`set_target_fps` broadcast retuned every camera when a second one connected).
Headless: CameraThread is stubbed so no real device is opened.
"""
from __future__ import annotations

import threading
from datetime import datetime

import numpy as np

from source.video.cameras import capture


class _FakeThread:
    def __init__(self, camera_id=None, width=None, height=None,
                 target_fps=None, camera_config=None):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.camera_config = camera_config
        self.connected = True
        self.fps_calls = []

    def start(self):
        pass

    def is_alive(self):
        return True

    def stop(self):
        pass

    def set_target_fps(self, fps):
        self.fps_calls.append(fps)
        self.target_fps = fps


def test_start_camera_uses_per_camera_fps_and_resolution(monkeypatch):
    monkeypatch.setattr(capture, "CameraThread", _FakeThread)
    vm = capture.VideoManager()
    vm.target_fps = 30                      # manager default

    ok = vm.start_camera(
        0, 1, camera_config={"width": 1280, "height": 720, "target_fps": 60})
    assert ok
    thread = vm.cameras[0]
    assert thread.target_fps == 60          # its own fps, not the 30 default
    assert (thread.width, thread.height) == (1280, 720)


def test_default_fps_does_not_retune_running_cameras():
    vm = capture.VideoManager.__new__(capture.VideoManager)
    vm.target_fps = 30
    cam_a, cam_b = _FakeThread(), _FakeThread()
    vm.cameras = {0: cam_a, 1: cam_b}

    vm.set_target_fps(60)                    # DEFAULT, must not touch running cams
    assert vm.target_fps == 60
    assert cam_a.fps_calls == [] and cam_b.fps_calls == []

    vm.set_target_fps(45, camera_id=0)       # targeted, only camera 0
    assert cam_a.fps_calls == [45]
    assert cam_b.fps_calls == []


def test_controller_carries_per_camera_resolution_into_start_camera():
    """controller.connect_camera must put each camera's own resolution/fps in
    the camera_config it hands to start_camera."""
    from source.video.framebus.controller import Pipeline

    pipe = Pipeline.__new__(Pipeline)
    pipe._registered_boxes = set()
    pipe._buses = {}
    pipe._frame_id = {}
    pipe._frame_version = {}
    pipe._box_unsubs = {}
    pipe._segment_config = None
    pipe.lens = None  # real Pipeline injects a LensCorrectionCache into each bus
    captured = {}

    class _VM:
        box_camera_map = {}

        def segment_processor_for(self, camera_id):
            return None

        def set_target_resolution(self, w, h):
            pass

        def set_target_fps(self, f):
            pass

        def set_frame_strategy(self, s):
            pass

        def start_camera(self, camera_id, setup_id, *, segment_config=None,
                         camera_backend=None, camera_config=None):
            captured.update(camera_config or {})
            return True

    pipe.video_manager = _VM()

    from source.video.framebus.types import LineOutputConfig, TriggerConfig

    class _Cfg:
        camera_backend = "opencv"
        grayscale = False
        flip_horizontal = False
        flip_vertical = False
        selected_resolution = (1024, 768)
        selected_fps = 45
        frame_strategy = "accept"
        capture_format = None
        exposure_us = None
        gain_db = None
        trigger = TriggerConfig()
        line_output = LineOutputConfig()
        extra = {}

    pipe.get_camera_config = lambda cid: _Cfg()
    pipe.register_box = lambda sid, **k: pipe._registered_boxes.add(sid)
    pipe._subscribe_sinks_for_box = lambda bus, sid: None
    pipe._ensure_tick_running = lambda: None

    ok = pipe.connect_camera(0, 1)
    assert ok
    assert captured.get("width") == 1024
    assert captured.get("height") == 768
    assert captured.get("target_fps") == 45


# ── End-to-end: two synthetic cameras through the real pipeline ──────────

class _FakeCam:
    """Stand-in CameraThread, delivers a distinct solid-colour frame per
    camera so crops from different cameras are distinguishable."""

    def __init__(self, camera_id=None, width=None, height=None,
                 target_fps=None, camera_config=None):
        self.camera_id = camera_id
        self.width = int(width or 640)
        self.height = int(height or 480)
        self.target_fps = target_fps
        self.connected = True
        self.connection_checked = threading.Event()
        self.connection_checked.set()
        val = 50 if camera_id in (0, "0") else 200
        self._frame = np.full((self.height, self.width, 3), val, dtype=np.uint8)

    def start(self):
        pass

    def isRunning(self):
        return True

    def is_alive(self):
        return True

    def stop(self):
        self.connected = False

    def wait(self, timeout=None):
        pass

    def set_target_fps(self, f):
        self.target_fps = f

    def get_last_frame(self):
        return self._frame

    def drain_recording_buffer(self):
        return []


def _left_half(box_number):
    return {"boxes": [{"box_number": box_number,
                       "geometry": {"percent": {"x": 0.0, "y": 0.0,
                                                "width": 0.5, "height": 1.0}}}]}


def _right_half(box_number):
    return {"boxes": [{"box_number": box_number,
                       "geometry": {"percent": {"x": 0.5, "y": 0.0,
                                                "width": 0.5, "height": 1.0}}}]}


def test_e2e_two_cameras_independent_resolution_and_segmentation(monkeypatch):
    """Full pipeline: two cameras at different resolutions/FPS, each with its
    OWN ROI segmentation, connecting the 2nd must not clobber the 1st."""
    monkeypatch.setattr(capture, "CameraThread", _FakeCam)
    from source.video.framebus.controller import Pipeline
    from source.video.framebus.types import CameraFrame

    pipe = Pipeline()
    try:
        pipe.update_camera_config(0, selected_resolution=(640, 480),
                                  selected_fps=30, camera_backend="opencv")
        pipe.update_camera_config(1, selected_resolution=(1280, 720),
                                  selected_fps=60, camera_backend="opencv")

        assert pipe.connect_camera(0, 1, segment_config=_left_half(1))
        assert pipe.connect_camera(1, 2, segment_config=_right_half(2))

        vm = pipe.video_manager
        # Each camera opened at ITS OWN resolution + fps.
        assert (vm.cameras[0].width, vm.cameras[0].height) == (640, 480)
        assert vm.cameras[0].target_fps == 30
        assert (vm.cameras[1].width, vm.cameras[1].height) == (1280, 720)
        assert vm.cameras[1].target_fps == 60

        # Independent, per-camera segmenters, connecting cam1 didn't wipe cam0.
        seg0 = vm.segment_processor_for(0)
        seg1 = vm.segment_processor_for(1)
        assert seg0 is not None and seg1 is not None and seg0 is not seg1
        assert 1 in seg0.box_index and 2 not in seg0.box_index
        assert 2 in seg1.box_index and 1 not in seg1.box_index

        # get_last_frame returns each box's OWN crop from its OWN camera.
        f1 = vm.get_last_frame(1)     # box1 on cam0: left half of 640 = 320
        f2 = vm.get_last_frame(2)     # box2 on cam1: right half of 1280 = 640
        assert f1.shape[1] == 320 and int(f1[0, 0, 0]) == 50   # cam0 colour
        assert f2.shape[1] == 640 and int(f2[0, 0, 0]) == 200  # cam1 colour

        # The bus fan-out crops per box too (the recording/display path).
        got = []
        bus1 = pipe.get_bus(1)
        bus1.on_box_frame(2, lambda bf: got.append(bf))
        bus1.publish_frame(CameraFrame(
            image=vm.cameras[1]._frame, cam_frame_id=1, capture_host_ns=0,
            capture_wall=datetime(2026, 1, 1), camera_id=1,
            is_shared=False, box_ids=()))
        assert got and got[0].image.shape[1] == 640
    finally:
        pipe.disconnect_camera(1)
        pipe.disconnect_camera(2)
        if hasattr(pipe, "shutdown"):
            try:
                pipe.shutdown()
            except Exception:
                pass
