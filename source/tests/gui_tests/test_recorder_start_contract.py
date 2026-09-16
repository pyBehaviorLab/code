"""Characterisation of ``VideoRecorder.start_recording``.

Written to pin the behaviour of a 193-line method BEFORE splitting it, so the
split is provably behaviour-preserving. Each test names a decision the method
makes rather than a line it executes:

  * refuse to start twice
  * adopt the camera's rate and colour mode
  * name the file (explicit stem / session stem / bare subject+timestamp)
  * size the output (ROI crop scaled to ~420 with even sides, else a
    standard resolution)
  * pick an encoder, and abort rather than silently drop to CPU when the
    operator forbade it
  * reset per-session counters so a reused recorder never starts tripped
"""
from __future__ import annotations

import types
from datetime import datetime

import pytest

from source.video.recording.recorder import VideoRecorder

pytest.importorskip("cv2")

WHEN = datetime(2026, 8, 5, 14, 30, 15)


def _cfg(fps=30, grayscale=False):
    return types.SimpleNamespace(target_fps=fps, grayscale=grayscale)


class _FakeWriter:
    encoder = "libx264"

    def isOpened(self):
        return True

    def write(self, frame):
        return True


def _patch_factory(monkeypatch, seen=None, writer=None):
    """Capture what the encoder factory was asked for."""
    seen = {} if seen is None else seen

    def _create(path, fps, size, **kw):
        seen["path"], seen["fps"], seen["size"] = path, fps, size
        seen.update(kw)
        return _FakeWriter() if writer is None else writer
    monkeypatch.setattr(
        "source.video.recording.ffmpeg.VideoWriterFactory.create_writer",
        staticmethod(_create))
    return seen


def _rec(**kw):
    kw.setdefault("camera_id", 0)
    kw.setdefault("fps", 30)
    kw.setdefault("resolution", (640, 480))
    return VideoRecorder(**kw)


def _stop(rec):
    rec.recording = False


# ── guard ────────────────────────────────────────────────────────────────

def test_refuses_to_start_twice(monkeypatch, tmp_path):
    _patch_factory(monkeypatch)
    rec = _rec()
    assert rec.start_recording(str(tmp_path), "subj", camera_config=_cfg())
    try:
        assert rec.start_recording(str(tmp_path), "subj") is False
    finally:
        _stop(rec)


# ── the camera dictates rate and colour ──────────────────────────────────

def test_adopts_camera_rate_and_grayscale(monkeypatch, tmp_path):
    seen = _patch_factory(monkeypatch)
    rec = _rec(fps=30)
    assert rec.start_recording(str(tmp_path), "subj",
                               camera_config=_cfg(fps=60, grayscale=True))
    try:
        assert rec.fps == 60.0            # the camera's rate is used verbatim
        assert seen["fps"] == 60.0
        assert seen["grayscale"] is True
    finally:
        _stop(rec)


def test_keeps_own_rate_when_camera_reports_none(monkeypatch, tmp_path):
    seen = _patch_factory(monkeypatch)
    rec = _rec(fps=18)
    assert rec.start_recording(str(tmp_path), "subj",
                               camera_config=_cfg(fps=0))
    try:
        assert rec.fps == 18 and seen["fps"] == 18
    finally:
        _stop(rec)


def test_mjpeg_keeps_the_configured_rate(monkeypatch, tmp_path):
    """target_fps is the only FPS driving capture, the constructor must
    not silently cap MJPEG to 20 (a wrong-fps header plays back at the
    wrong speed if start_recording ever runs without a camera config)."""
    _patch_factory(monkeypatch)
    rec = _rec(fps=25, codec="MJPEG")
    assert rec.fps == 25
    # A camera reporting a real rate still wins.
    assert rec.start_recording(str(tmp_path), "subj", camera_config=_cfg(60))
    try:
        assert rec.fps == 60.0
    finally:
        _stop(rec)


# ── file naming ──────────────────────────────────────────────────────────

def test_explicit_stem_wins(monkeypatch, tmp_path):
    _patch_factory(monkeypatch)
    rec = _rec()
    assert rec.start_recording(str(tmp_path), "subj", datetime_now=WHEN,
                               box_ID=2, camera_config=_cfg(),
                               file_stem="video_Box2")
    try:
        assert rec.video_path.endswith("video_Box2.mp4")
    finally:
        _stop(rec)


def test_box_id_uses_the_shared_session_stem(monkeypatch, tmp_path):
    from source.video.recording import build_session_stem
    _patch_factory(monkeypatch)
    rec = _rec()
    assert rec.start_recording(str(tmp_path), "M01", datetime_now=WHEN,
                               box_ID=3, camera_config=_cfg())
    try:
        expected = build_session_stem("M01", 3, WHEN)
        assert rec.video_path.endswith(expected + ".mp4")
    finally:
        _stop(rec)


def test_without_box_id_falls_back_to_subject_and_timestamp(monkeypatch,
                                                            tmp_path):
    _patch_factory(monkeypatch)
    rec = _rec()
    assert rec.start_recording(str(tmp_path), "M01", datetime_now=WHEN,
                               camera_config=_cfg())
    try:
        assert rec.video_path.endswith("M01-2026-08-05-143015.mp4")
    finally:
        _stop(rec)


def test_creates_the_output_directory(monkeypatch, tmp_path):
    _patch_factory(monkeypatch)
    target = tmp_path / "nested" / "video"
    rec = _rec()
    assert rec.start_recording(str(target), "subj", camera_config=_cfg())
    try:
        assert target.is_dir()
    finally:
        _stop(rec)


# ── output size ──────────────────────────────────────────────────────────

def test_no_roi_normalises_to_a_standard_resolution(monkeypatch, tmp_path):
    from source.video.recording.recorder import get_standard_resolution
    seen = _patch_factory(monkeypatch)
    rec = _rec(resolution=(1280, 720))
    assert rec.start_recording(str(tmp_path), "subj", camera_config=_cfg())
    try:
        assert seen["size"] == get_standard_resolution(1280, 720)
    finally:
        _stop(rec)


@pytest.mark.parametrize("roi,expected", [
    # wide crop: width pinned to 420, height follows the aspect, both even
    ((0, 0, 400, 200), (420, 210)),
    # tall crop: height pinned to 420
    ((0, 0, 200, 400), (210, 420)),
    # square crop
    ((10, 10, 300, 300), (420, 420)),
])
def test_roi_is_scaled_to_420_with_even_sides(monkeypatch, tmp_path,
                                               roi, expected):
    seen = _patch_factory(monkeypatch)
    rec = _rec(resolution=(640, 480))
    rec.roi = roi
    assert rec.start_recording(str(tmp_path), "subj", camera_config=_cfg())
    try:
        assert seen["size"] == expected
        assert seen["size"][0] % 2 == 0 and seen["size"][1] % 2 == 0
        assert rec._roi_valid is True
    finally:
        _stop(rec)


@pytest.mark.parametrize("roi", [
    (-1, 0, 100, 100),          # negative origin
    (0, 0, 0, 100),             # zero width
    (600, 0, 100, 100),         # runs off the right edge
    (0, 400, 100, 200),         # runs off the bottom
])
def test_an_out_of_frame_roi_is_ignored_not_applied(monkeypatch, tmp_path,
                                                     roi):
    from source.video.recording.recorder import get_standard_resolution
    seen = _patch_factory(monkeypatch)
    rec = _rec(resolution=(640, 480))
    rec.roi = roi
    assert rec.start_recording(str(tmp_path), "subj", camera_config=_cfg())
    try:
        # Falls back to the full-frame size, and does not claim a valid ROI.
        assert seen["size"] == (640, 480) or seen["size"] == \
            get_standard_resolution(640, 480)
        assert rec._roi_valid is False
    finally:
        _stop(rec)


def test_a_malformed_roi_does_not_raise(monkeypatch, tmp_path):
    _patch_factory(monkeypatch)
    rec = _rec(resolution=(640, 480))
    rec.roi = ("not", "a", "roi", "!")
    assert rec.start_recording(str(tmp_path), "subj", camera_config=_cfg())
    _stop(rec)


# ── encoder selection ────────────────────────────────────────────────────

def test_gpu_encoder_is_reported_and_extension_becomes_mp4(monkeypatch,
                                                            tmp_path):
    class _Nvenc(_FakeWriter):
        encoder = "h264_nvenc"
    _patch_factory(monkeypatch, writer=_Nvenc())
    rec = _rec()
    assert rec.start_recording(str(tmp_path), "subj", camera_config=_cfg())
    try:
        assert rec._using_gpu is True
        assert rec._actual_encoder == "h264_nvenc"
        assert rec.video_path.endswith(".mp4")
    finally:
        _stop(rec)


def test_cpu_encoder_is_not_reported_as_gpu(monkeypatch, tmp_path):
    _patch_factory(monkeypatch)           # encoder = libx264
    rec = _rec()
    assert rec.start_recording(str(tmp_path), "subj", camera_config=_cfg())
    try:
        assert rec._using_gpu is False
        assert rec._actual_encoder == "libx264"
    finally:
        _stop(rec)


def test_strict_hardware_mode_aborts_rather_than_dropping_to_cpu(monkeypatch,
                                                                 tmp_path):
    """allow_cpu_fallback=False means the operator wants the GPU or nothing;
    silently recording on CPU would misreport the session."""
    from source.video.recording.recorder import EncoderCapabilities
    caps = EncoderCapabilities.get_instance()
    monkeypatch.setattr(type(caps), "gpu_force_required", lambda self: True)
    monkeypatch.setattr(type(caps), "has_ffmpeg", property(lambda self: True))
    monkeypatch.setattr(
        "source.video.recording.ffmpeg.VideoWriterFactory.create_writer",
        staticmethod(lambda *a, **k: None))
    rec = _rec(allow_cpu_fallback=False)
    assert rec.start_recording(str(tmp_path), "subj",
                               camera_config=_cfg()) is False
    assert rec.recording is False


# ── per-session state ────────────────────────────────────────────────────

def test_reused_recorder_does_not_start_already_tripped(monkeypatch, tmp_path):
    _patch_factory(monkeypatch)
    rec = _rec()
    rec.encoder_failed = True
    rec._write_errors = 7
    rec.frame_count = 99
    assert rec.start_recording(str(tmp_path), "subj", camera_config=_cfg())
    try:
        assert rec.encoder_failed is False
        assert rec._write_errors == 0
        assert rec.frame_count == 0
        assert rec.recording is True
        assert rec.rec_start_host_ns > 0 and rec.start_time > 0
        assert rec.thread is not None and rec.thread.daemon
    finally:
        _stop(rec)


def test_queue_is_emptied_so_a_new_session_starts_clean(monkeypatch, tmp_path):
    _patch_factory(monkeypatch)
    rec = _rec()
    rec.frame_queue.append(("stale", 0))
    assert rec.start_recording(str(tmp_path), "subj", camera_config=_cfg())
    try:
        assert not any(f[0] == "stale" for f in list(rec.frame_queue))
    finally:
        _stop(rec)
