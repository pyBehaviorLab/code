"""Central post-load sequence (cameras, pose readiness, MCU).

Both modes inherit ``MainWindowBase._auto_flow_after_load``, maze keeps no
parallel copy. These pin the gating and the readiness decision without a Qt
event loop (``QTimer.singleShot`` is patched to capture scheduled calls).

Readiness follows the CAMERA, not the auto-connect preference: a project opened
with auto-connect off still prepares its models the moment the operator
connects by hand. Hanging it off the preference left those rigs loading their
model at the first Record click, with the animal already in the box.
"""
from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from source.gui.base import MainWindowBase


def _bare_host():
    w = MainWindowBase.__new__(MainWindowBase)
    w.connect_all_cameras = MagicMock()
    w._iter_box_ids = lambda: [1]
    w._setup_widget_for = lambda sid: None
    w.pipeline = MagicMock()
    w.pipeline.all_tracking_configs.return_value = {}
    return w


def _meta(on: bool):
    return SimpleNamespace(meta=SimpleNamespace(auto_connect_cameras_on_load=on))


def _tc(has_dlc: bool, online: bool = True):
    return SimpleNamespace(has_dlc=lambda: has_dlc,
                           online_tracking_enabled=online)


def test_pref_off_does_not_connect_cameras():
    w = _bare_host()
    with patch("source.gui.base.QtCore.QTimer.singleShot"):
        w._auto_flow_after_load(_meta(False))
    w.connect_all_cameras.assert_not_called()


def test_pref_on_connects_cameras():
    w = _bare_host()
    with patch("source.gui.base.QtCore.QTimer.singleShot"):
        w._auto_flow_after_load(_meta(True))
    w.connect_all_cameras.assert_called_once()


def test_blob_mode_schedules_no_pose_readiness():
    """Blob needs no model, so nothing is scheduled and nothing is asked."""
    w = _bare_host()
    w.pipeline.all_tracking_configs.return_value = {1: _tc(False)}
    with patch("source.gui.base.QtCore.QTimer.singleShot") as shot:
        w._auto_flow_after_load(_meta(True))
    assert getattr(w, "_pose_ready_pending", False) is False
    shot.assert_not_called()


def test_a_configured_box_schedules_a_readiness_pass():
    w = _bare_host()
    w.pipeline.all_tracking_configs.return_value = {1: _tc(True)}
    with patch("source.gui.base.QtCore.QTimer.singleShot") as shot:
        w._auto_flow_after_load(_meta(True))
    assert w._pose_ready_pending is True
    shot.assert_called()


def test_readiness_is_scheduled_even_with_auto_connect_off():
    """The preference decides whether CAMERAS open by themselves, not whether
    a model is ever prepared."""
    w = _bare_host()
    w.pipeline.all_tracking_configs.return_value = {1: _tc(True)}
    with patch("source.gui.base.QtCore.QTimer.singleShot") as shot:
        w._auto_flow_after_load(_meta(False))
    assert w._pose_ready_pending is True
    shot.assert_called_once()


def test_a_box_that_opted_out_of_tracking_is_not_prepared():
    """"No tracking configured" must mean nothing happens for that box: no
    load, no prompt, no entry in any list."""
    w = _bare_host()
    w.pipeline.all_tracking_configs.return_value = {1: _tc(True, online=False)}
    assert w.pose_configured_boxes() == []
    with patch("source.gui.base.QtCore.QTimer.singleShot") as shot:
        w._auto_flow_after_load(_meta(True))
    shot.assert_not_called()


def test_repeat_requests_coalesce_into_one_pass():
    """Sixteen boxes finishing their camera open in the same second must
    produce ONE load, not sixteen."""
    w = _bare_host()
    w.pipeline.all_tracking_configs.return_value = {1: _tc(True), 2: _tc(True)}
    with patch("source.gui.base.QtCore.QTimer.singleShot") as shot:
        w.schedule_pose_ready()
        w.schedule_pose_ready()
        w.schedule_pose_ready()
    assert shot.call_count == 1


# ── the readiness pass itself ─────────────────────────────────────────


def _ready_host(streaming: bool, *, model_dir: str, already_ready: bool = False):
    w = MainWindowBase.__new__(MainWindowBase)
    w._iter_box_ids = lambda: [1]
    vm = MagicMock()
    vm.box_camera_map = {1: "cam0"}
    vm.cameras = {}
    vm.is_camera_streaming = lambda bid: streaming
    w.video_manager = vm
    w._overlay = {}
    w.tracking_enabled = {}
    w.pipeline = MagicMock()
    w.pipeline.all_tracking_configs.return_value = {1: _tc(True)}
    w.pose_settings_for_box = MagicMock(
        return_value={"mode": "dlc", "model_path": model_dir,
                      "body_parts": ["snout"]})
    w._configure_pose_for_box = MagicMock(return_value=True)
    w._ready_patch = patch("source.gui.base.pose_ready_for",
                           return_value=already_ready)
    return w


def test_readiness_retries_while_the_camera_is_still_opening():
    """A cold open plus the driver settle outlasts the first schedule. Giving
    up on the first miss is how the model came to load at Record instead."""
    with tempfile.TemporaryDirectory() as d:
        w = _ready_host(False, model_dir=d)
        with w._ready_patch, patch(
                "source.gui.base.QtCore.QTimer.singleShot") as shot:
            assert w.ensure_pose_ready() == 0
        assert w._pose_ready_attempts == 1
        assert w._pose_ready_pending is True
        shot.assert_called_once()


def test_readiness_gives_up_loudly_after_the_cap():
    with tempfile.TemporaryDirectory() as d:
        w = _ready_host(False, model_dir=d)
        w._pose_ready_attempts = 12
        with w._ready_patch, patch(
                "source.gui.base.QtCore.QTimer.singleShot") as shot:
            assert w.ensure_pose_ready() == 0
        assert w._pose_ready_pending is False
        shot.assert_not_called()


def test_a_streaming_camera_loads_the_model_without_starting_inference():
    """Loads and warms. Inference starts at Test Tracking or Record, so
    opening a project does not quietly begin tracking every animal."""
    with tempfile.TemporaryDirectory() as d:
        w = _ready_host(True, model_dir=d)
        w.statusbar = MagicMock()
        with w._ready_patch, patch(
                "source.gui.base.QtCore.QTimer.singleShot"):
            assert w.ensure_pose_ready() == 1
        w._configure_pose_for_box.assert_called_once()
        assert w.tracking_enabled[1] is False
        w.pipeline.enable_pose.assert_not_called()


def test_an_already_ready_box_is_not_loaded_again():
    with tempfile.TemporaryDirectory() as d:
        w = _ready_host(True, model_dir=d, already_ready=True)
        w.statusbar = MagicMock()
        with w._ready_patch, patch(
                "source.gui.base.QtCore.QTimer.singleShot"):
            assert w.ensure_pose_ready() == 1
        w._configure_pose_for_box.assert_not_called()


def test_a_model_that_is_not_on_disk_says_so_and_does_not_load():
    w = _ready_host(True, model_dir=os.path.join("no", "such", "model"))
    w.statusbar = MagicMock()
    with w._ready_patch, patch("source.gui.base.QtCore.QTimer.singleShot"):
        assert w.ensure_pose_ready() == 0
    w._configure_pose_for_box.assert_not_called()
