"""Phase C: one shared _video_recorder_geometry for both modes.

The two per-mode overrides (operant box-ROI tuple, maze full-frame tuple)
collapsed into a single capability-branched resolver on MainWindowBase. This
pins each branch reproduces the tuple its mode produced before:

  * segment-processor active  → (segment_size, None)      [both modes]
  * else, widget has _get_box_roi (operant) → (None, box_roi)
  * else (maze)               → (full_camera_wh, None)
"""
from __future__ import annotations

from types import MethodType, SimpleNamespace

from source.gui.base import MainWindowBase

# The resolver calls sibling base helpers on self; bind the real ones onto the
# stub host so we exercise the actual MainWindowBase logic (not re-stubs).
_BASE_HELPERS = ("_box_segment_size", "_full_camera_wh",
                 "_box_recorder_roi", "_recording_fps")


class _Seg:
    box_index = {1: 0}

    def get_segment_size(self, sid, shape):
        return (640, 720)


class _VM:
    box_camera_map = {1: 0, 2: 0}
    cameras = {0: SimpleNamespace(width=1280, height=720)}

    def segment_processor_for_box(self, sid):
        return _Seg() if sid == 1 else None

    def resolved_settings_for_box(self, sid):
        return SimpleNamespace(target_fps=30)

    def get_full_frame(self, cid):
        return None


def _host(widget):
    h = SimpleNamespace(
        video_manager=_VM(),
        pipeline=SimpleNamespace(
            get_camera_config=lambda cid: SimpleNamespace(selected_fps=60)),
        video_target_fps=20,
        _setup_widget_for=lambda sid: widget,
        _box_camera_id_text=lambda sid: "0",
    )
    for name in _BASE_HELPERS:
        setattr(h, name, MethodType(getattr(MainWindowBase, name), h))
    return h


def _geom(host, sid):
    return MainWindowBase._video_recorder_geometry(host, sid)


def test_segmented_box_uses_segment_size_no_roi():
    # Box 1 has an active segment processor → segment size, no recorder ROI.
    res, roi, fps, cam_id, cfg = _geom(_host(SimpleNamespace(
        _get_box_roi=lambda: (10, 20, 300, 400))), 1)
    assert res == (640, 720)
    assert roi is None
    assert fps == 60.0                     # pipeline selected_fps wins
    assert cam_id == 0


def test_operant_non_segmented_crops_box_roi():
    # Box 2, operant widget (has _get_box_roi), no segment → ROI drives the
    # crop, resolution left None (recorder derives it). Matches old operant.
    res, roi, fps, cam_id, cfg = _geom(_host(SimpleNamespace(
        _get_box_roi=lambda: (10, 20, 300, 400))), 2)
    assert res is None
    assert roi == (10, 20, 300, 400)


def test_maze_non_segmented_uses_full_camera_frame():
    # Box 2, maze widget (NO _get_box_roi), no segment → full camera frame,
    # no ROI. Matches old maze tuple ((w,h), None, ...).
    res, roi, fps, cam_id, cfg = _geom(_host(SimpleNamespace()), 2)
    assert res == (1280, 720)
    assert roi is None


def test_fps_falls_back_to_cam_cfg_then_rig_default():
    # No pipeline selected_fps → cam_cfg.target_fps; no cam_cfg → rig default.
    host = _host(SimpleNamespace())
    host.pipeline = SimpleNamespace(
        get_camera_config=lambda cid: SimpleNamespace(selected_fps=None))
    _res, _roi, fps, _cid, _cfg = _geom(host, 2)
    assert fps == 30.0                     # cam_cfg.target_fps
