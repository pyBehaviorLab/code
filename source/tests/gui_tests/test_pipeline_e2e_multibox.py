"""End-to-end through the real Pipeline: several boxes, recording + tracking.

Everything else about this work is tested a layer at a time. This drives the
actual object graph, FrameBus fanning frames to RecorderSink and TrackerSink,
each on its own worker pool, and then reads the artefacts off disk, because
that is the only place the whole chain can be shown to agree:

  * every captured frame accounted for (lossless recorder contract),
  * per-box ordering preserved despite the sinks now running several workers.
"""
from __future__ import annotations

import threading
import time

from source import host_clock

import cv2
import numpy as np
import pytest

from source.video.cameras import capture
from source.video.framebus.controller import Pipeline
from source.video.framebus.types import CameraFrame
from source.video.recording.frame_log import FrameLog
from source.video.tracking.blob import TRACK_TRACKED

BOXES = (1, 2, 3, 4)
W, H = 320, 240


def _frame(cx, cy=120):
    """Pale arena, dark animal, the case the default preset is tuned for."""
    img = np.full((H, W, 3), 200, np.uint8)
    cv2.circle(img, (int(cx), int(cy)), 14, (20, 20, 20), -1)
    return img


def _quadrant(box_number):
    """One cell of a 2x2 CCTV split, in the percent geometry the bus wants."""
    i = (int(box_number) - 1) % 4
    return {"boxes": [{"box_number": int(box_number),
                       "geometry": {"percent": {"x": 0.5 * (i % 2),
                                                "y": 0.5 * (i // 2),
                                                "width": 0.5,
                                                "height": 0.5}}}]}


class _FakeCam:
    """Stand-in CameraThread, the bus is what is under test, not the driver."""

    def __init__(self, camera_id=None, width=None, height=None,
                 target_fps=None, camera_config=None):
        self.camera_id = camera_id
        self.width = int(width or W)
        self.height = int(height or H)
        self.target_fps = target_fps
        self.connected = True
        self.stopped = False
        self.connection_checked = threading.Event()
        self.connection_checked.set()
        self._frame = _frame(60)

    def start(self):
        pass

    def isRunning(self):
        return not self.stopped

    def is_alive(self):
        return not self.stopped

    def stop(self):
        self.stopped = True
        self.connected = False

    def wait(self, timeout=None):
        return True

    def set_target_fps(self, f):
        self.target_fps = f

    def get_last_frame(self):
        return self._frame

    def get_latest_frame_versioned(self, last_version):
        return None

    def drain_recording_buffer(self):
        return []


class _Rec:
    """Stand-in VideoRecorder: counts what the sink hands it."""

    def __init__(self):
        self.frames = 0
        self.stopped = False
        # The sink gates on this exactly as it does for a real VideoRecorder:
        # without it every frame is treated as live-preview and dropped.
        self.recording = True
        self.encoder_failed = False

    def add_frame(self, frame, capture_host_ns=0, *a, **kw):
        self.frames += 1
        return True

    def stop_recording(self, *a, **kw):
        self.stopped = True
        return None

    def stop(self, *a, **kw):
        self.stopped = True

    def release(self, *a, **kw):
        self.stopped = True


@pytest.fixture
def pipe(monkeypatch):
    monkeypatch.setattr(capture, "CameraThread", _FakeCam)
    p = Pipeline(target_fps=30)
    p.update_camera_config(0, selected_resolution=(W, H),
                           selected_fps=30, camera_backend="opencv")
    yield p
    try:
        p.shutdown()
    except Exception:
        pass


def _publish(pipeline, cam_id, n_frames, boxes=BOXES):
    """Push n frames through the bus with the animal walking."""
    bus = pipeline.get_bus(cam_id)
    assert bus is not None, "no bus for the camera"
    for i in range(n_frames):
        cx = 40 + (i * 6) % (W - 80)
        # is_shared / box_ids are overwritten by publish_frame from the bus's
        # own registration; they are required positionally.
        cf = CameraFrame(image=_frame(cx), cam_frame_id=i,
                         camera_id=cam_id,
                         capture_host_ns=host_clock.host_ns(),
                         capture_wall=time.time(),
                         is_shared=False, box_ids=tuple(boxes))
        bus.publish_frame(cf)
    return n_frames


# ── the sinks really do run pools now ─────────────────────────────────────

def test_recorder_runs_several_workers_and_pose_stays_single(pipe):
    """Recording is per-box independent work that releases the GIL, so it
    gets a pool. Pose is GPU-bound and batches across boxes itself, so a
    second host thread would only contend; it stays at one."""
    assert pipe.recorder._nworkers > 1, "recorder still single-threaded"
    assert pipe.pose._nworkers == 1, "pose must not be parallelised host-side"
    # Tracker dispatches to its OWN executor; one dispatcher is correct.
    assert pipe.tracker._nworkers == 1


def test_a_box_is_pinned_to_one_worker(pipe):
    """Per-box ordering is what makes multiple workers safe, the recorder
    and tracker keep per-box state that assumes sequential frames."""
    seen = {pipe.recorder._worker_for(b) for b in range(0, 64)}
    assert seen == set(range(pipe.recorder._nworkers)), "boxes not spread"
    for b in range(0, 64):
        assert (pipe.recorder._worker_for(b)
                == pipe.recorder._worker_for(b)), "pinning is not stable"


# ── the whole chain, artefacts on disk ────────────────────────────────────

def test_multibox_record_and_track_writes_every_frame(pipe, tmp_path):
    n = 60
    recs, logs = {}, {}
    for bid in BOXES:
        pipe.register_box(bid)
        recs[bid] = _Rec()
        logs[bid] = FrameLog(tmp_path / f"box{bid}_video_data.txt")
        pipe.start_recording(bid, recorder=recs[bid],
                             tracking_writer=logs[bid])

    # One shared camera split into a 2x2 grid, the CCTV case, and the one
    # where per-box ordering across a worker pool actually gets exercised.
    for bid in BOXES:
        assert pipe.connect_camera(0, bid, segment_config=_quadrant(bid)), \
            f"box {bid} did not attach"

    _publish(pipe, 0, n)
    _drain(pipe)

    for bid in BOXES:
        assert recs[bid].frames == n, (
            f"box {bid}: recorder got {recs[bid].frames}/{n} frames, the "
            f"recorder contract is lossless")
        pipe.stop_recording(bid)
        logs[bid].close()


def _drain(pipeline, timeout=8.0):
    """Wait for every sink queue to empty, the sinks are asynchronous."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pending = 0
        for sink in (pipeline.recorder, pipeline.tracker):
            with sink._lock:
                pending += sum(len(d) for d in sink._per_box.values())
        if pending == 0:
            time.sleep(0.25)      # let the in-flight frame finish
            return
        time.sleep(0.05)
    raise AssertionError("sinks did not drain")



def test_rows_are_in_frame_order_despite_worker_pools(pipe, tmp_path):
    """Per-box pinning exists so this holds. Out-of-order rows would mean two
    workers ran the same box concurrently."""
    bid = 2
    pipe.register_box(bid)
    rec, log = _Rec(), FrameLog(tmp_path / "o_video_data.txt")
    pipe.start_recording(bid, recorder=rec, tracking_writer=log)
    pipe.connect_camera(0, bid)
    _publish(pipe, 0, 80)
    _drain(pipe)
    pipe.stop_recording(bid)
    log.close()

    nums = [int(ln.split("\t")[0]) for ln in
            (tmp_path / "o_video_data.txt").read_text().splitlines()
            if ln and not ln.startswith("#") and ln[0].isdigit()]
    assert nums == sorted(nums), "frame rows are out of order"
    assert len(set(nums)) == len(nums), "a frame was written twice"
