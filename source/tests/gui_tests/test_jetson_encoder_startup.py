"""A Jetson hardware encoder that cannot start must not cost a box its video,
and the session file must say which encoder recorded it.

Before this, ``_try_ffmpeg`` judged a writer by ``isOpened()`` straight after
``Popen``, when an FFmpeg that is about to fail has not failed yet. An
h264_nvmpi encoder that refused to start (library missing, out of sessions,
refused size) was therefore accepted, died on the first frames, and the box
recorded nothing. And every ``_video_data.txt`` said ``"encoder":"unknown"``.

Everything here is scoped to h264_nvmpi or to a Jetson, so no other machine's
recording or files change; the non-Jetson cases pin that.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from source import host_clock
from source.video.recording import ffmpeg as ff
from source.video.recording import recorder as recmod
from source.video.recording.frame_log import FrameLog
from source.video.recording.recorder import VideoRecorder


# ── the test encode before a hardware writer opens ──────────────────────

class _OpenWriter:
    def __init__(self, path, fps, size, encoder="libx264", crf=23, grayscale=False, **_k):
        self.path, self.encoder = path, encoder

    def isOpened(self):
        return True

    def release(self):
        pass


def test_a_failed_test_encode_skips_nvmpi_without_opening_a_writer(monkeypatch, tmp_path):
    monkeypatch.setattr(ff, "_nvmpi_encodes_at", lambda size: False)

    def _never(*a, **k):
        raise AssertionError("a writer was opened on an encoder that failed its test")

    monkeypatch.setattr(ff, "FFmpegVideoWriter", _never)
    start = ff.hw_encoder_sessions()
    w = ff.VideoWriterFactory._try_ffmpeg(
        Path(tmp_path / "v.mp4"), 30, (240, 240), "h264_nvmpi", 23, False,
        "Jetson HW H.264 (h264_nvmpi)", is_hardware=True)
    assert w is None
    assert ff.hw_encoder_sessions() == start


def test_a_passed_test_encode_opens_the_hardware_writer(monkeypatch, tmp_path):
    monkeypatch.setattr(ff, "_nvmpi_encodes_at", lambda size: True)
    monkeypatch.setattr(ff, "FFmpegVideoWriter", _OpenWriter)
    w = ff.VideoWriterFactory._try_ffmpeg(
        Path(tmp_path / "v.mp4"), 30, (240, 240), "h264_nvmpi", 23, False,
        "Jetson HW H.264 (h264_nvmpi)", is_hardware=True)
    assert w is not None and w.encoder == "h264_nvmpi"
    ff._dec_hw_sessions()


def test_other_encoders_are_never_test_encoded(monkeypatch, tmp_path):
    def _boom(size):
        raise AssertionError("a non-Jetson encoder paid for the Jetson test")

    monkeypatch.setattr(ff, "_nvmpi_encodes_at", _boom)
    monkeypatch.setattr(ff, "FFmpegVideoWriter", _OpenWriter)
    for enc in ("libx264", "h264_nvenc", "h264_qsv", "h264_mf"):
        w = ff.VideoWriterFactory._try_ffmpeg(
            Path(tmp_path / f"{enc}.mp4"), 30, (240, 240), enc, 23, False, enc)
        assert w is not None


class _JetsonCaps:
    def gpu_force_required(self): return True
    def detect_hevc_nvenc(self): return False
    def detect_nvenc(self): return False
    def detect_qsv(self): return False
    def detect_amf(self): return False
    def detect_mf(self): return False
    def detect_nvmpi(self): return True
    def detect_v4l2m2m(self): return False
    def detect_libx264(self): return True
    def detect_libx265(self): return False
    def is_jetson(self): return True


def test_a_jetson_whose_encoder_fails_its_test_records_on_libx264(monkeypatch, tmp_path):
    monkeypatch.setattr(ff.EncoderCapabilities, "get_instance",
                        staticmethod(lambda: _JetsonCaps()))
    monkeypatch.setattr(ff, "_nvmpi_encodes_at", lambda size: False)
    monkeypatch.setattr(ff, "FFmpegVideoWriter", _OpenWriter)
    w = ff.VideoWriterFactory.create_writer(
        str(tmp_path / "v.mp4"), 30.0, (240, 240), use_gpu=True)
    assert w is not None and w.encoder == "libx264"


def test_a_hung_test_encode_counts_as_a_failure(monkeypatch):
    def _hang(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=k.get("timeout"))

    monkeypatch.setattr(ff.subprocess, "run", _hang)
    assert ff._nvmpi_encodes_at((240, 240)) is False


def test_a_failing_test_encode_counts_as_a_failure(monkeypatch):
    class _Done:
        returncode = 1
        stderr = b"NvVideoEncTransferCaptureBufferToBlock: DoWork failed"

    monkeypatch.setattr(ff.subprocess, "run", lambda *a, **k: _Done())
    assert ff._nvmpi_encodes_at((240, 240)) is False


# ── an h264_nvmpi writer that dies at start-up ──────────────────────────

class _DeadWriter:
    encoder = "h264_nvmpi"

    def __init__(self):
        self.released = False

    def write(self, frame):
        return False

    def is_healthy(self):
        return False

    def release(self):
        self.released = True


class _GoodWriter:
    encoder = "libx264"

    def __init__(self, path):
        self.path, self.frames = path, 0

    def write(self, frame):
        self.frames += 1
        return True

    def is_healthy(self):
        return True

    def release(self):
        pass


class _Drops:
    def __init__(self):
        self.reasons = []

    def record(self, *a, **k):
        self.reasons.append(k.get("reason"))


def _recorder(monkeypatch, tmp_path, encoder="h264_nvmpi", started_s_ago=0.5):
    rec = VideoRecorder(fps=30)
    rec.video_path = str(tmp_path / "v.mp4")
    rec._target_size = (240, 240)
    rec._actual_encoder = encoder
    rec.writer = _DeadWriter()
    rec.writer.encoder = encoder
    rec.rec_start_host_ns = host_clock.host_ns() - int(started_s_ago * 1e9)
    opened = {}

    def fake_try(path_obj, fps, size, enc, crf, grayscale, label, is_hardware=False):
        opened["args"] = (str(path_obj), enc, tuple(size))
        opened["writer"] = _GoodWriter(str(path_obj))
        return opened["writer"]

    monkeypatch.setattr(ff.VideoWriterFactory, "_try_ffmpeg", staticmethod(fake_try))
    drops = _Drops()
    monkeypatch.setattr(recmod, "_drop_log", drops)
    return rec, opened, drops


def test_a_writer_dying_at_start_continues_on_libx264_in_the_same_file(monkeypatch, tmp_path):
    rec, opened, drops = _recorder(monkeypatch, tmp_path)
    dead = rec.writer
    told = []
    rec.drop_callback = lambda n, reason: told.append((n, reason))
    frame = np.zeros((240, 240, 3), np.uint8)
    for _ in range(3):
        rec.frame_queue.append((frame, host_clock.host_ns()))
    rec.recording = True
    t = threading.Thread(target=rec._external_record_loop, daemon=True)
    t.start()
    deadline = time.time() + 3
    while rec.frame_queue and time.time() < deadline:
        time.sleep(0.01)
    rec.recording = False
    t.join(timeout=3)

    assert dead.released
    assert opened["args"] == (str(tmp_path / "v.mp4"), "libx264", (240, 240)), (
        "the fallback must reopen the same file, which the MCU TSV already names")
    assert opened["writer"].frames == 3, "a death seen before the write loses no frame"
    assert rec._actual_encoder == "libx264" and rec.frame_count == 3
    assert not rec.encoder_failed
    assert told == [(0, "nvmpi_stopped_fallback_libx264")]
    assert any("nvmpi_stopped_fallback_libx264" in (r or "") for r in drops.reasons)


class _Exited:
    def poll(self):
        return -9


class _Running:
    def poll(self):
        return None


class _BufferingDeadWriter(_DeadWriter):
    """FFmpeg has exited, but the pipe buffer still takes frames."""
    _process = _Exited()

    def write(self, frame):
        return True

    def is_healthy(self):
        return True


class _LiveWriter(_DeadWriter):
    _process = _Running()

    def is_healthy(self):
        return True


def test_an_exited_ffmpeg_is_caught_while_its_pipe_still_accepts_frames(monkeypatch, tmp_path):
    rec, opened, _ = _recorder(monkeypatch, tmp_path)
    rec.writer = _BufferingDeadWriter()
    rec.frame_count = 12
    told = []
    rec.drop_callback = lambda n, reason: told.append((n, reason))
    assert rec._recover_dead_nvmpi_writer(rejected_frame=False) is True
    assert rec._actual_encoder == "libx264"
    assert told == [(12, "nvmpi_stopped_fallback_libx264")]


def test_a_running_nvmpi_writer_is_left_alone(monkeypatch, tmp_path):
    rec, opened, _ = _recorder(monkeypatch, tmp_path)
    rec.writer = _LiveWriter()
    assert rec._recover_dead_nvmpi_writer(rejected_frame=False) is False
    assert "args" not in opened


def test_a_writer_dying_late_is_not_reopened(monkeypatch, tmp_path):
    """Minutes into a session the file holds real video; reopening would
    overwrite it, so a late failure keeps the existing encoder-dead path."""
    rec, opened, _ = _recorder(monkeypatch, tmp_path, started_s_ago=60)
    assert rec._recover_dead_nvmpi_writer() is False
    assert "args" not in opened


def test_other_encoders_are_never_swapped(monkeypatch, tmp_path):
    for enc in ("h264_nvenc", "libx264", "h264_qsv"):
        rec, opened, _ = _recorder(monkeypatch, tmp_path, encoder=enc)
        assert rec._recover_dead_nvmpi_writer() is False
        assert "args" not in opened


def test_strict_hardware_mode_is_respected(monkeypatch, tmp_path):
    rec, opened, _ = _recorder(monkeypatch, tmp_path)
    rec.allow_cpu_fallback = False
    assert rec._recover_dead_nvmpi_writer() is False


def test_the_swap_happens_once_per_session(monkeypatch, tmp_path):
    rec, opened, _ = _recorder(monkeypatch, tmp_path)
    assert rec._recover_dead_nvmpi_writer() is True
    rec.writer = _DeadWriter()
    rec._actual_encoder = "h264_nvmpi"
    assert rec._recover_dead_nvmpi_writer() is False


# ── the encoder in the _video_data.txt header ────────────────────────────

class _Caps:
    def __init__(self, jetson):
        self._jetson = jetson

    def is_jetson(self):
        return self._jetson


def test_the_header_encoder_is_named_on_a_jetson_only(monkeypatch):
    rec = VideoRecorder()
    rec._actual_encoder = "h264_nvmpi"
    monkeypatch.setattr(ff.EncoderCapabilities, "get_instance",
                        staticmethod(lambda: _Caps(True)))
    assert rec.header_encoder() == "h264_nvmpi"
    monkeypatch.setattr(ff.EncoderCapabilities, "get_instance",
                        staticmethod(lambda: _Caps(False)))
    assert rec.header_encoder() is None


def _header(tmp_path, encoder=None):
    log = FrameLog(tmp_path / "s_video_data.txt", video_file="s.mp4",
                   rebind_drop_log=False)
    log.write_info("subject_id", "s")
    log.write_info("box_id", 1)
    if encoder:
        log.write_info("video_encoder", encoder)
    log.write_frame(0, 0)
    log.close()
    return (tmp_path / "s_video_data.txt").read_text(encoding="utf-8")


def test_the_session_header_names_the_encoder_it_is_given(tmp_path):
    text = _header(tmp_path, "h264_nvmpi")
    line = next(l for l in text.splitlines() if l.startswith("#video_codec"))
    assert '"h264_nvmpi"' in line and "unknown" not in line


def test_the_session_header_is_unchanged_when_no_encoder_is_given(tmp_path):
    text = _header(tmp_path)
    line = next(l for l in text.splitlines() if l.startswith("#video_codec"))
    assert '"unknown"' in line


# ── a switch mid-session is stated in the session file ──────────────────

def test_the_swap_tells_the_session_file(monkeypatch, tmp_path):
    rec, opened, _ = _recorder(monkeypatch, tmp_path)
    rec.frame_count = 5
    told = []
    rec.encoder_changed_callback = lambda enc, replaced, lost: told.append(
        (enc, replaced, lost))
    assert rec._recover_dead_nvmpi_writer(rejected_frame=False) is True
    assert told == [("libx264", "h264_nvmpi", 5)]


def _switched_log(tmp_path, before_header):
    path = tmp_path / "s_video_data.txt"
    log = FrameLog(path, video_file="s.mp4", rebind_drop_log=False)
    log.write_info("subject_id", "s")
    log.write_info("box_id", 1)
    log.write_info("video_encoder", "h264_nvmpi")
    if before_header:
        log.note_encoder_change("libx264", "h264_nvmpi", 0)
    log.write_frame(0, 0)
    if not before_header:
        log.note_encoder_change("libx264", "h264_nvmpi", 24)
    log.write_frame(1, 33)
    log.close()
    return path, path.read_text(encoding="utf-8").splitlines()


def test_a_switch_after_the_header_is_stated_in_a_closing_line(tmp_path):
    """The header at the top of an open file cannot be rewritten, so the file
    would otherwise name an encoder that wrote none of it."""
    _, lines = _switched_log(tmp_path, before_header=False)
    header = next(l for l in lines if l.startswith("#video_codec "))
    final = [l for l in lines if l.startswith("#video_codec_final")]
    end = next(i for i, l in enumerate(lines) if l.startswith("#session_end"))
    assert '"h264_nvmpi"' in header, "the header at the top is left as written"
    assert len(final) == 1 and '"encoder":"libx264"' in final[0]
    assert '"lost_frames":24' in final[0]
    assert lines.index(final[0]) == end - 1, "stated right before #session_end"


def test_a_switch_before_the_header_is_simply_what_the_header_says(tmp_path):
    _, lines = _switched_log(tmp_path, before_header=True)
    header = next(l for l in lines if l.startswith("#video_codec "))
    assert '"libx264"' in header


def test_a_file_without_a_switch_has_no_closing_codec_line(tmp_path):
    assert "#video_codec_final" not in _header(tmp_path, "h264_nvmpi")


def test_the_offline_parser_reads_the_switched_file(tmp_path):
    parser = pytest.importorskip("tools.offline_analysis.video_data_parser")
    path, _ = _switched_log(tmp_path, before_header=False)
    session = parser.parse_video_data(str(path))
    assert session.info["video_codec"]["encoder"] == "h264_nvmpi"
    assert session.info["video_codec_final"]["encoder"] == "libx264"
