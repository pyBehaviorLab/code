"""Wave 2: ROI / zone coordinate-space unification (Z1-Z5).

Pure-logic tests for the coordinate fixes that stopped ROI corruption and
zone offsets across a save/reload or a resolution change:

  Z3  get_last_frame returns None (blackout), not the full frame, when a
      configured segment fails to extract.
  Z5  _check_zone_shape_dim warns when zones were drawn on a frame whose
      aspect differs from the live crop.

The Z1 (ROI dialog seed-from-normalized) and Z4 (editor no-clamp) paths need
Qt widgets / a live camera and are exercised by the rig-verification matrix;
here we cover the host-side logic that is unit-testable.
"""
from types import SimpleNamespace

import numpy as np

from source.gui.base import MainWindowBase
from source.video.cameras.capture import VideoManager, VideoSegmentProcessor


# ── Z3: blackout instead of full-frame substitute ─────────────────────────

def _vm_with_segment(box_index):
    vm = VideoManager.__new__(VideoManager)
    cam_id = "cam0"
    seg = VideoSegmentProcessor.__new__(VideoSegmentProcessor)
    seg.box_index = box_index
    seg.config = {"boxes": [{"geometry": {"pixel": {}}} for _ in box_index]}
    seg._coord_cache = {}
    vm._segment_processors = {cam_id: seg}
    # Both box 1 (segmented) and box 2 (not) sit on this camera.
    vm.box_camera_map = {1: cam_id, 2: cam_id}
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    thread = SimpleNamespace(get_last_frame=lambda: frame)
    vm.cameras = {cam_id: thread}
    return vm, seg


def test_get_last_frame_blackout_on_failed_segment(monkeypatch):
    # Box 1 is a configured segment; extraction fails (returns None).
    vm, seg = _vm_with_segment({1: 0})
    monkeypatch.setattr(seg, "extract_segment", lambda frame, sid: None)
    # Must be None (blackout), NOT the full 320x240 frame.
    assert vm.get_last_frame(1) is None


def test_get_last_frame_returns_segment_when_ok(monkeypatch):
    vm, seg = _vm_with_segment({1: 0})
    crop = np.ones((60, 80, 3), dtype=np.uint8)
    monkeypatch.setattr(seg, "extract_segment", lambda frame, sid: crop)
    out = vm.get_last_frame(1)
    assert out is crop


def test_get_last_frame_full_frame_when_not_segmented(monkeypatch):
    # Box 2 is NOT in the segment's box_index → full frame is its view.
    vm, seg = _vm_with_segment({1: 0})
    out = vm.get_last_frame(2)
    assert out is not None and out.shape == (240, 320, 3)


# ── Z5: shape_dim aspect tripwire ──────────────────────────────────────────

def _host_for_shapecheck(live_wh):
    said = []
    h = SimpleNamespace(
        _box_frame_wh=lambda sid: live_wh,
        _box_status_say=lambda sid, msg: said.append((sid, msg)),
    )
    h._said = said
    return h


def test_shape_dim_mismatch_warns():
    # Zones drawn on 229x189 (aspect 1.21); live crop 70x66 (aspect ~1.06).
    h = _host_for_shapecheck((70, 66))
    zones = [{"name": "A", "points": [[0.1, 0.1]], "shape_dim": [229, 189]}]
    MainWindowBase._check_zone_shape_dim(h, 1, zones)
    assert h._said and "re-check roi" in h._said[0][1].lower()


def test_shape_dim_match_is_silent():
    # Same aspect (both 4:3) → no warning.
    h = _host_for_shapecheck((320, 240))
    zones = [{"name": "A", "points": [[0.1, 0.1]], "shape_dim": [640, 480]}]
    MainWindowBase._check_zone_shape_dim(h, 1, zones)
    assert h._said == []


def test_shape_dim_unknown_frame_is_silent():
    h = _host_for_shapecheck(None)
    zones = [{"name": "A", "points": [[0.1, 0.1]], "shape_dim": [229, 189]}]
    MainWindowBase._check_zone_shape_dim(h, 1, zones)
    assert h._said == []


def test_shape_dim_absent_is_silent():
    h = _host_for_shapecheck((70, 66))
    zones = [{"name": "A", "points": [[0.1, 0.1]]}]   # no shape_dim
    MainWindowBase._check_zone_shape_dim(h, 1, zones)
    assert h._said == []
