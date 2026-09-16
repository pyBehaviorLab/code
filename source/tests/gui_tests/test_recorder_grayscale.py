"""Grayscale recording, the NVENC ``-pix_fmt gray`` fast path.

A grayscale camera delivers true single-channel frames; the recorder must tell
the encoder ``grayscale=True`` (so the pipe stays 1 byte/pixel) and coerce every
frame to match that pixel format, otherwise the FFmpeg byte count desyncs.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from source.video.recording.recorder import VideoRecorder

cv2 = pytest.importorskip("cv2")


def _cfg(grayscale):
    return types.SimpleNamespace(target_fps=30, grayscale=grayscale)


class _FakeWriter:
    def __init__(self):
        self.frames = []

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame)
        return True


def _recorder():
    return VideoRecorder(camera_id=0, fps=30, resolution=(64, 48))


# ── the flag reaches the writer factory ──────────────────────────────────

def test_grayscale_flag_passed_to_writer(monkeypatch, tmp_path):
    seen = {}

    def _fake_create(path, fps, size, **kw):
        seen.update(kw)
        return _FakeWriter()

    monkeypatch.setattr(
        "source.video.recording.ffmpeg.VideoWriterFactory.create_writer",
        staticmethod(_fake_create))

    rec = _recorder()
    assert rec.start_recording(str(tmp_path), "subj", camera_config=_cfg(True))
    assert seen.get("grayscale") is True
    rec.recording = False


def test_colour_by_default(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        "source.video.recording.ffmpeg.VideoWriterFactory.create_writer",
        staticmethod(lambda p, f, s, **kw: (seen.update(kw), _FakeWriter())[1]))

    rec = _recorder()
    assert rec.start_recording(str(tmp_path), "subj", camera_config=_cfg(False))
    assert seen.get("grayscale") is False
    rec.recording = False


# ── channel-coercion invariant ───────────────────────────────────────────

def test_coerce_grayscale_from_bgr():
    bgr = np.zeros((48, 64, 3), np.uint8)
    out = VideoRecorder._coerce_channels(bgr, grayscale=True)
    assert out.ndim == 2


def test_coerce_grayscale_passthrough():
    gray = np.zeros((48, 64), np.uint8)
    out = VideoRecorder._coerce_channels(gray, grayscale=True)
    assert out.ndim == 2 and out is gray


def test_coerce_colour_from_gray():
    gray = np.zeros((48, 64), np.uint8)
    out = VideoRecorder._coerce_channels(gray, grayscale=False)
    assert out.ndim == 3 and out.shape[2] == 3


def test_coerce_colour_passthrough():
    bgr = np.zeros((48, 64, 3), np.uint8)
    out = VideoRecorder._coerce_channels(bgr, grayscale=False)
    assert out is bgr
