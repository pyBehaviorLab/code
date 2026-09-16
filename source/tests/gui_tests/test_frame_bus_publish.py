"""FrameBus.publish_frame delivers per-box BoxFrames end-to-end.

The bus does not tag MCU framework time; RecorderSink reads the raw
``pycboard.timestamp`` per box at frame-handling time.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from source.video.framebus.frame_bus import FrameBus
from source.video.framebus.types import CameraFrame


def _cam_frame(frame_id=7, mono=1_000_500_000):
    return CameraFrame(
        image=np.zeros((48, 64, 3), np.uint8), cam_frame_id=frame_id,
        capture_host_ns=mono, capture_wall=datetime.now(),
        camera_id=0, is_shared=False, box_ids=())


def _bus_with_box():
    bus = FrameBus(camera_id=0)
    bus.register_box(1)
    got = []
    bus.on_box_frame(1, got.append)
    return bus, got


def test_publish_frame_delivers_box_frame():
    bus, got = _bus_with_box()
    bus.publish_frame(_cam_frame())
    assert len(got) == 1
    bf = got[0]
    assert bf.setup_id == 1
    assert bf.cam_frame_id == 7
    assert bf.capture_host_ns == 1_000_500_000


def test_publish_frame_carries_raw_capture_fields():
    # The capture identity a producer stamps on the CameraFrame (frame id,
    # capture timestamps) must reach the derived BoxFrame unchanged, and
    # the bus must fill in this bus's box membership.
    bus, got = _bus_with_box()
    wall = datetime.now()
    cf = CameraFrame(
        image=np.zeros((48, 64, 3), np.uint8), cam_frame_id=9,
        capture_host_ns=1_000_500_000, capture_wall=wall,
        camera_id=0, is_shared=False, box_ids=())
    bus.publish_frame(cf)
    assert len(got) == 1
    bf = got[0]
    assert bf.cam_frame_id == 9
    assert bf.capture_host_ns == 1_000_500_000
    assert bf.capture_wall is wall
    assert bf.is_shared_camera is False
    assert cf.box_ids == (1,)


def test_publish_frame_no_boxes_is_noop():
    bus = FrameBus(camera_id=0)
    bus.publish_frame(_cam_frame())   # no registered box → just raw fan-out
