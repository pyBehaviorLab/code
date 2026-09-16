"""WS8 host-efficiency: contiguous recorder ingest + cached crop-rect reuse."""
from __future__ import annotations

import time

from source import host_clock
from datetime import datetime

import numpy as np

from source.video.cameras.capture import VideoSegmentProcessor
from source.video.framebus.frame_bus import FrameBus
from source.video.framebus.recorder_sink import RecorderSink
from source.video.framebus.types import BoxFrame, CameraFrame


def _seg(geom):
    return VideoSegmentProcessor({"boxes": [{"box_id": 1, "geometry": geom}]})


def _cam_frame():
    return CameraFrame(
        image=np.zeros((48, 64, 3), np.uint8), cam_frame_id=1,
        capture_host_ns=1, capture_wall=datetime.now(),
        camera_id=0, is_shared=False, box_ids=())


# ── I6/I7: recorder gets an owned, contiguous ROI (never a view) ───────

def test_recorder_ingest_is_contiguous():
    got = {}

    class StubRecorder:
        recording = True

        def add_frame(self, img, ts):
            got["contiguous"] = bool(img.flags["C_CONTIGUOUS"])
            got["shape"] = img.shape
            return True

    sink = RecorderSink()
    sink.start_recording(1, recorder=StubRecorder())

    # A non-contiguous view, exactly what a shared-camera BoxFrame carries.
    parent = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    view = parent[8:40, 8:56]
    assert not view.flags["C_CONTIGUOUS"]

    bf = BoxFrame(
        image=view, setup_id=1, cam_frame_id=5, camera_id=0,
        capture_host_ns=host_clock.host_ns(), capture_wall=datetime.now(),
        is_shared_camera=True, poll_host_ns=host_clock.host_ns())
    sink.process(bf)

    # The writer must receive an already-contiguous ROI-sized buffer, so it
    # neither re-copies per write nor pins the whole parent frame.
    assert got["contiguous"] is True
    assert got["shape"] == (32, 48, 3)


# ── I11: crop rect comes from the extract_segment cache ────────────────

def test_segment_rect_percent_cached():
    seg = _seg({"percent": {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5}})
    frame = np.zeros((48, 64, 3), np.uint8)
    seg.extract_segment(frame, 1)  # warms the cache
    assert seg.segment_rect(1, 48, 64) == (16, 12, 32, 24)


def test_publish_sets_crop_from_cache_percent():
    seg = _seg({"percent": {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5}})
    bus = FrameBus(camera_id=0)
    bus.register_box(1)
    bus.set_segment_processor(seg)
    got = []
    bus.on_box_frame(1, got.append)
    bus.publish_frame(_cam_frame())
    bf = got[0]
    assert bf.crop_origin == (16, 12)
    assert bf.crop_size == (32, 24)
    assert bf.image.shape[:2] == (24, 32)  # (h, w) of the ROI


def test_publish_handles_pixel_geometry():
    """Latent-bug fix: the old publish_frame recomputed crop_origin/size from
    'percent' only, silently leaving them None for 'pixel' geometry. Reading
    the cache handles both."""
    seg = _seg({"pixel": {"x": 10, "y": 6, "width": 20, "height": 15}})
    bus = FrameBus(camera_id=0)
    bus.register_box(1)
    bus.set_segment_processor(seg)
    got = []
    bus.on_box_frame(1, got.append)
    bus.publish_frame(_cam_frame())
    bf = got[0]
    assert bf.crop_origin == (10, 6)
    assert bf.crop_size == (20, 15)
    assert bf.image.shape[:2] == (15, 20)
