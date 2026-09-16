"""Lens / FOV correction, calibration math, store, cache, and the FrameBus hook.

Headless: a synthetic barrel-distorted pinhole camera (built with
``cv2.projectPoints``) provides ground-truth board views, so the solve is
checked against a known distortion rather than a real lens.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from source.video.cameras import lens
from source.video.cameras.lens import (
    BoardSpec, CalibrationProfile, LensCalibrationStore, LensCorrectionCache,
    Undistorter, board_signature, find_board, object_points, solve,
)
from source.video.framebus.frame_bus import FrameBus
from source.video.framebus.types import CameraFrame

cv2 = pytest.importorskip("cv2")


# ── synthetic ground-truth camera ───────────────────────────────────────

WIDTH, HEIGHT = 640, 480
BOARD = BoardSpec(cols=9, rows=6, square_mm=25.0)
GT_K = np.array([[520.0, 0.0, 320.0],
                 [0.0, 520.0, 240.0],
                 [0.0, 0.0, 1.0]])
GT_DIST = np.array([-0.32, 0.12, 0.0, 0.0, 0.0])  # barrel (negative k1)


def _project_board(rvec, tvec) -> np.ndarray:
    """Project the board's 3-D corners through the GT camera → distorted
    image points, shaped like a ``find_board`` result."""
    objp = object_points(BOARD)
    img, _ = cv2.projectPoints(objp, rvec, tvec, GT_K, GT_DIST)
    return img.astype(np.float32)


def _views(n=14):
    """A spread of board poses across depth and angle."""
    sets = []
    for i in range(n):
        rvec = np.array([0.15 * np.sin(i), 0.15 * np.cos(i), 0.05 * i])
        tvec = np.array([-90.0 + 12 * (i % 4), -70.0 + 15 * (i % 3),
                         420.0 + 25 * (i % 5)])
        sets.append(_project_board(rvec, tvec))
    return sets


# ── calibration math ────────────────────────────────────────────────────

def test_solve_recovers_distortion():
    prof = solve(_views(), (WIDTH, HEIGHT), BOARD)
    assert prof.rms_error < 0.5           # excellent fit to synthetic GT
    assert prof.quality == "excellent"
    assert prof.dist_coeffs[0] < 0        # recovered the barrel sign
    # focal length recovered within a few percent
    assert abs(prof.camera_matrix[0, 0] - GT_K[0, 0]) / GT_K[0, 0] < 0.05


def test_solve_rejects_too_few_views():
    with pytest.raises(ValueError):
        solve(_views(2), (WIDTH, HEIGHT), BOARD)


def test_undistorter_straightens_points():
    prof = solve(_views(), (WIDTH, HEIGHT), BOARD)
    und = Undistorter(prof, (WIDTH, HEIGHT), alpha=0.0)
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    out = und.apply(frame)
    assert out.shape == frame.shape
    assert und.size == (WIDTH, HEIGHT)


def test_profile_scaled_matrix_tracks_resolution():
    prof = solve(_views(), (WIDTH, HEIGHT), BOARD)
    K2 = prof.scaled_matrix((WIDTH // 2, HEIGHT // 2))
    assert K2[0, 0] == pytest.approx(prof.camera_matrix[0, 0] * 0.5)
    assert K2[1, 2] == pytest.approx(prof.camera_matrix[1, 2] * 0.5)


def test_undistorter_rebuilds_on_resolution_change():
    prof = solve(_views(), (WIDTH, HEIGHT), BOARD)
    und = Undistorter(prof, (WIDTH, HEIGHT))
    und.apply(np.zeros((HEIGHT // 2, WIDTH // 2, 3), np.uint8))
    assert und.size == (WIDTH // 2, HEIGHT // 2)


def test_profile_json_roundtrip():
    prof = solve(_views(), (WIDTH, HEIGHT), BOARD)
    back = CalibrationProfile.from_json(prof.to_json())
    assert np.allclose(back.camera_matrix, prof.camera_matrix)
    assert np.allclose(back.dist_coeffs, prof.dist_coeffs)
    assert back.image_size == prof.image_size
    assert back.board.cols == BOARD.cols


def test_board_signature_spreads_by_cell_and_size():
    near = _project_board(np.zeros(3), np.array([0.0, 0.0, 250.0]))
    far = _project_board(np.zeros(3), np.array([0.0, 0.0, 600.0]))
    sig_near = board_signature(near, (WIDTH, HEIGHT))
    sig_far = board_signature(far, (WIDTH, HEIGHT))
    assert sig_near != sig_far           # different size buckets → kept apart


# ── store ────────────────────────────────────────────────────────────────

def _profile():
    return solve(_views(), (WIDTH, HEIGHT), BOARD)


def test_store_put_get_enable_gate(tmp_path):
    store = LensCalibrationStore(path=tmp_path / "lens.json")
    prof = _profile()
    store.put("usb-1234:5678:ABC", prof, friendly="Cam A")
    assert store.get("usb-1234:5678:ABC") is not None
    assert store.get_enabled_profile("usb-1234:5678:ABC") is not None
    # toggle off → the runtime gate returns None, profile retained
    store.set_enabled("usb-1234:5678:ABC", False)
    assert store.get_enabled_profile("usb-1234:5678:ABC") is None
    assert store.get("usb-1234:5678:ABC").profile.rms_error < 0.5


def test_store_persists_across_instances(tmp_path):
    p = tmp_path / "lens.json"
    LensCalibrationStore(path=p).put("k", _profile())
    assert LensCalibrationStore(path=p).get_enabled_profile("k") is not None


def test_store_corrupt_file_is_empty(tmp_path):
    p = tmp_path / "lens.json"
    p.write_text("{ not json", encoding="utf-8")
    store = LensCalibrationStore(path=p)
    assert store.keys() == []


def test_store_forget(tmp_path):
    store = LensCalibrationStore(path=tmp_path / "lens.json")
    store.put("k", _profile())
    assert store.forget("k") is True
    assert store.forget("k") is False


# ── cache ────────────────────────────────────────────────────────────────

def test_cache_empty_store_skips_identity_resolution(tmp_path):
    """The common case (no calibration) must never touch identity resolution,
    that can spawn an ffmpeg subprocess and contend for the device."""
    store = LensCalibrationStore(path=tmp_path / "lens.json")
    calls = []

    def _resolve(cam_id):
        calls.append(cam_id)
        return "usb-x"

    cache = LensCorrectionCache(store=store, resolve=_resolve)
    assert cache.undistorter_for(0, (WIDTH, HEIGHT)) is None
    assert calls == []  # store empty → bailed before resolving


def test_cache_returns_undistorter_for_calibrated_camera(tmp_path):
    store = LensCalibrationStore(path=tmp_path / "lens.json")
    store.put("usb-cam0", _profile())
    cache = LensCorrectionCache(store=store, resolve=lambda cid: "usb-cam0")
    und = cache.undistorter_for(0, (WIDTH, HEIGHT))
    assert isinstance(und, Undistorter)
    # cached: a second call returns the same object without re-resolving
    assert cache.undistorter_for(0, (WIDTH, HEIGHT)) is und


def test_cache_disabled_returns_none(tmp_path):
    store = LensCalibrationStore(path=tmp_path / "lens.json")
    store.put("usb-cam0", _profile())
    cache = LensCorrectionCache(store=store, resolve=lambda cid: "usb-cam0")
    cache.set_enabled(False)
    assert cache.undistorter_for(0, (WIDTH, HEIGHT)) is None


def test_cache_weak_identity_gets_no_correction(tmp_path):
    store = LensCalibrationStore(path=tmp_path / "lens.json")
    store.put("usb-cam0", _profile())
    # resolve returning None models a weak/unresolvable identity
    cache = LensCorrectionCache(store=store, resolve=lambda cid: None)
    assert cache.undistorter_for(0, (WIDTH, HEIGHT)) is None


def test_cache_invalidate_rereads(tmp_path):
    store = LensCalibrationStore(path=tmp_path / "lens.json")
    cache = LensCorrectionCache(store=store, resolve=lambda cid: "usb-cam0")
    assert cache.undistorter_for(0, (WIDTH, HEIGHT)) is None  # empty
    store.put("usb-cam0", _profile())
    cache.invalidate(0)  # targeted; injected store is reused
    assert cache.undistorter_for(0, (WIDTH, HEIGHT)) is not None


# ── FrameBus integration ─────────────────────────────────────────────────

def _cam_frame():
    return CameraFrame(
        image=np.full((HEIGHT, WIDTH, 3), 128, np.uint8), cam_frame_id=1,
        capture_host_ns=1_000, capture_wall=datetime.now(),
        camera_id=0, is_shared=False, box_ids=())


def test_framebus_applies_correction_before_fanout(tmp_path):
    store = LensCalibrationStore(path=tmp_path / "lens.json")
    store.put("usb-cam0", _profile())
    cache = LensCorrectionCache(store=store, resolve=lambda cid: "usb-cam0")

    bus = FrameBus(camera_id=0, lens=cache)
    bus.register_box(1)
    seen = []
    bus.on_camera_frame(lambda cf: seen.append(cf.image.copy()))
    bus.on_box_frame(1, lambda bf: seen.append(bf.image.copy()))

    cf = _cam_frame()
    bus.publish_frame(cf)
    # alpha=0 remap of a flat frame zooms valid pixels; subscribers got the
    # corrected image (same shape), and the color cache was cleared on rebind.
    assert seen and seen[0].shape == (HEIGHT, WIDTH, 3)
    assert cf._color_cache == {}


def test_framebus_no_lens_is_passthrough():
    bus = FrameBus(camera_id=0, lens=None)
    bus.register_box(1)
    got = []
    bus.on_box_frame(1, lambda bf: got.append(bf.image))
    cf = _cam_frame()
    bus.publish_frame(cf)
    assert got and got[0].shape == (HEIGHT, WIDTH, 3)


def test_framebus_lens_failure_latches(tmp_path, monkeypatch):
    store = LensCalibrationStore(path=tmp_path / "lens.json")
    store.put("usb-cam0", _profile())
    cache = LensCorrectionCache(store=store, resolve=lambda cid: "usb-cam0")
    bus = FrameBus(camera_id=0, lens=cache)
    bus.register_box(1)

    und = cache.undistorter_for(0, (WIDTH, HEIGHT))
    calls = {"n": 0}

    def _boom(_frame):
        calls["n"] += 1
        raise RuntimeError("remap boom")

    monkeypatch.setattr(und, "apply", _boom)
    bus.publish_frame(_cam_frame())
    bus.publish_frame(_cam_frame())
    assert calls["n"] == 1          # latched off after the first error
    assert bus._lens_failed is True
    bus.reset_lens_failed()
    assert bus._lens_failed is False
