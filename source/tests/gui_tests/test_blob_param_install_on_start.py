"""Operant record-start must push saved blob params into the tracker.

``MainWindowBase.start_tracking_for_box`` must install the saved bg_mode /
blur / morph, not just call ``enable_blob_tracking``: without them the tracker
runs class defaults (bg_mode "running_avg") and drifts off the captured
background. Maze happens to be covered by its own override, operant is not, so
the install belongs in the shared branch, right after
enable. This test exercises the base path with light stubs.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from source.gui.base import MainWindowBase


def _make_tc():
    """A blob (non-DLC) TrackingConfig stub with the fields the settings
    builder + installer read."""
    tc = SimpleNamespace(
        online_tracking_enabled=True,
        tracker_type="blob",
        dlc_model_path="",
        keypoint_names=[],
        confidence_threshold=0.5,
        pose_resize_factor=1.0,
        pose_n_instances=1,
        blob_threshold=30,
        blob_min_area=50,
        blob_max_area=500,
        blob_detect_dark=True,
        blob_blur_mode="gaussian",
        blob_blur_kernel_size=5,
        blob_bg_mode="static",          # the field that was being dropped
        blob_open_kernel_size=3,
        blob_close_kernel_size=7,
        blob_use_clahe=True,             # contrast/threshold preprocessing
        blob_clahe_clip_limit=3.0,
        blob_clahe_tile_size=8,
        blob_use_adaptive_threshold=True,
        blob_smooth_tracking=False,
        blob_background_path="",         # empty → installer skips BG load
        blob_self_norm_ratio=0.0,        # Simple mode is off in this fixture
        blob_self_norm_sigma=0.0,
        blob_self_norm_smooth_sigma=0.0,
        blob_self_norm_minsize=0,
    )
    tc.has_dlc = lambda: False
    tc.has_blob = lambda: True
    return tc


def test_blob_start_installs_saved_params():
    tc = _make_tc()
    tracker = MagicMock()
    tm = MagicMock()
    tm.get_tracker.return_value = tracker

    pipe = MagicMock()
    pipe.get_tracking_config.return_value = tc

    w = MainWindowBase.__new__(MainWindowBase)
    w.pipeline = pipe
    w.tracker_manager = tm
    w.video_manager = None          # installer skips auto-init
    w.tracking_enabled = {}
    w._tracking_dialog_globals = {}

    ok = MainWindowBase.start_tracking_for_box(w, 1)

    assert ok is True
    pipe.enable_blob_tracking.assert_called_once_with(1)
    # The saved blob params reached the tracker, bg_mode in particular.
    tracker.update_params.assert_called_once()
    kwargs = tracker.update_params.call_args.kwargs
    assert kwargs["bg_mode"] == "static"
    assert kwargs["blur_mode"] == "gaussian"
    assert kwargs["blur_kernel_size"] == 5
    assert kwargs["open_kernel_size"] == 3
    assert kwargs["close_kernel_size"] == 7
    assert kwargs["threshold"] == 30
    # CLAHE / adaptive-threshold preprocessing must reach the tracker too: it
    # round-trips through the schema and the dialog, and the last hop into the
    # live tracker is the one that drops it silently.
    assert kwargs["use_clahe"] is True
    assert kwargs["clahe_clip_limit"] == 3.0
    assert kwargs["clahe_tile_size"] == 8
    assert kwargs["use_adaptive_threshold"] is True
    assert w.tracking_enabled[1] is True
