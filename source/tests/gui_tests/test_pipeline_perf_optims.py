"""Host-side tests for the multi-board efficiency changes.

Covers lazy color conversion + shared-camera prime-once cache, OpenCV thread
tuning, CPU-encoder thread caps, and removal of the dead Tracker ABC. All
pure host logic, no MCU, no live camera.
"""
from datetime import datetime

import numpy as np
import pytest

from source.video.framebus.types import CameraFrame, BoxFrame


def _camera_frame(img, *, shared, box_ids):
    return CameraFrame(
        image=img, cam_frame_id=1, capture_host_ns=0,
        capture_wall=datetime(2026, 6, 21), camera_id=0,
        is_shared=shared, box_ids=tuple(box_ids),
    )


def _box_frame(img, *, setup_id, shared, parent, crop_origin=None, crop_size=None):
    return BoxFrame(
        image=img, setup_id=setup_id, cam_frame_id=1, camera_id=0,
        capture_host_ns=0, capture_wall=datetime(2026, 6, 21),
        is_shared_camera=shared, crop_origin=crop_origin, crop_size=crop_size,
        _parent_camera_frame=parent,
    )


# ── lazy conversion, a box only converts what its consumer touches ──────────

def test_gray_access_does_not_compute_rgb():
    """A blob-only / record path that reads image_gray must NOT trigger the
    BGR→RGB (DLC) conversion. (no DLC ⇒ no RGB)"""
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    bf = _box_frame(img, setup_id=1, shared=False, parent=None)
    _ = bf.image_gray
    assert "gray" in bf._color_cache
    assert "rgb" not in bf._color_cache  # never converted to color


def test_untouched_box_converts_nothing():
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    bf = _box_frame(img, setup_id=1, shared=False, parent=None)
    assert bf._color_cache == {}  # record-only box: zero cvtColor


# ── shared camera primes the parent cache ONCE; siblings slice a view ────────

def test_shared_camera_primes_parent_once_and_siblings_slice():
    parent_img = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
    cf = _camera_frame(parent_img, shared=True, box_ids=[1, 2])

    # Two boxes split the 80px width into left/right halves.
    left = _box_frame(parent_img[:, :40], setup_id=1, shared=True, parent=cf,
                      crop_origin=(0, 0), crop_size=(40, 100))
    right = _box_frame(parent_img[:, 40:], setup_id=2, shared=True, parent=cf,
                       crop_origin=(40, 0), crop_size=(40, 100))

    g_left = left.image_gray
    g_right = right.image_gray

    # Parent whole-frame gray was primed exactly once and cached.
    assert "gray" in cf._color_cache
    parent_gray = cf._color_cache["gray"]
    assert parent_gray.shape == (100, 80)
    # Each box's gray is a VIEW into the primed parent buffer (no per-box cvtColor).
    assert np.shares_memory(g_left, parent_gray)
    assert np.shares_memory(g_right, parent_gray)
    # And they correspond to the right crop regions.
    assert g_left.shape == (100, 40)
    assert g_right.shape == (100, 40)


def test_dedicated_camera_does_not_prime_parent():
    """One box per camera: convert the box's own image directly, do NOT prime
    the parent (no sibling to share to → priming would convert more pixels)."""
    parent_img = np.zeros((50, 50, 3), dtype=np.uint8)
    cf = _camera_frame(parent_img, shared=False, box_ids=[1])
    bf = _box_frame(parent_img, setup_id=1, shared=False, parent=cf)
    g = bf.image_gray
    assert g is not None and g.shape == (50, 50)
    assert "gray" not in cf._color_cache  # parent NOT primed for a dedicated cam


# ── OpenCV thread tuning ─────────────────────────────────────────────────────

def test_opencv_thread_count_scaling():
    from source.video.framebus.controller import _opencv_thread_count
    # Small/Jetson hosts stay pinned to 1; larger hosts get ~cores/4 intra-op
    # threads (capped at 6) so a 16-box rig doesn't starve the serial per-tile
    # resizes and encoder opens.
    assert _opencv_thread_count(2) == 1
    assert _opencv_thread_count(4) == 1
    assert _opencv_thread_count(8) == 2
    assert _opencv_thread_count(16) == 4
    assert _opencv_thread_count(24) == 6
    assert _opencv_thread_count(64) == 6      # ceiling


def test_tune_opencv_threads_applies_low_count():
    import cv2
    from source.video.framebus.controller import _opencv_thread_count, _tune_opencv_threads
    _tune_opencv_threads()
    assert cv2.getNumThreads() == _opencv_thread_count()


# ── CPU encoders get a per-encoder thread cap; GPU does not ──────────────────

def test_cpu_encoders_get_thread_cap(monkeypatch):
    import os
    from source.video.recording import ffmpeg as ff
    monkeypatch.setattr(ff.FFmpegVideoWriter, "_start_ffmpeg", lambda self: None)
    expect = str(max(1, (os.cpu_count() or 4) // 2))
    for enc in ("libx264", "libx265"):
        w = ff.FFmpegVideoWriter("out.mp4", 30, (640, 480),
                                 encoder=enc, crf=23, preset="fast")
        cmd = w._build_command()
        assert "-threads" in cmd, enc
        assert cmd[cmd.index("-threads") + 1] == expect, enc


def test_gpu_encoder_has_no_cpu_thread_cap(monkeypatch):
    from source.video.recording import ffmpeg as ff
    monkeypatch.setattr(ff.FFmpegVideoWriter, "_start_ffmpeg", lambda self: None)
    w = ff.FFmpegVideoWriter("out.mp4", 30, (640, 480),
                             encoder="h264_nvenc", crf=23, preset="fast")
    assert "-threads" not in w._build_command()  # GPU = session-bound, not threads


# ── the dead Tracker ABC is gone ─────────────────────────────────────────────

def test_tracker_abc_removed():
    import source.video.tracking as trk
    assert not hasattr(trk, "Tracker")
    with pytest.raises(ModuleNotFoundError):
        __import__("source.video.tracking.base")
