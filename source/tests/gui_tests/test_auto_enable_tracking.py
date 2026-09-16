"""Tests for ``RunTask._auto_enable_tracking``, the Record-click
pre-tracking decision tree spec'd in Section 3 of
``docs/CENTRALIZATION_PLAN.md``.

Decision matrix covered:
  * no main_window / no pipeline                       → proceed silently
  * online_tracking_enabled=False                      → proceed silently
  * neither has_dlc() nor has_blob()                   → proceed silently
  * DLC configured + camera NOT connected              → proceed silently (no modal)
  * Blob configured + camera NOT connected             → proceed silently
  * DLC configured + camera + model not loaded         → modal fires
  * DLC configured + camera + settings changed         → modal fires
  * Modal "Init now"     → proceed (init called with THIS box's settings)
  * Modal "Disable"      → proceed; online_tracking_enabled=False; model_path UNCHANGED
  * Modal "Cancel"       → ABORT Record click (returns False)
  * DLC configured + camera + box already ready        → enable_pose, no modal

Readiness is one call to ``pose_ready_for``, which both the dialog and the load
path also use. The modal here is the safety net for a deferred init, a model
file that moved, or a load that failed, not the normal path. It used to be the
normal path, because the check compared two different descriptions of the same
model and they could never be equal.
"""
from unittest.mock import MagicMock, patch

from source.gui.widgets.run_task import RunTask


def _make_widget(*, has_dlc=False, has_blob=False, online=True,
                 cam_connected=True, model_loaded=True, sig_matches=True,
                 pose_already_enabled=False, prompt_choice="cancel"):
    """Build a RunTask + main_window test double wired for one scenario."""
    mw = MagicMock()
    tc = MagicMock()
    tc.has_dlc.return_value = has_dlc
    tc.has_blob.return_value = has_blob
    tc.online_tracking_enabled = online
    tc.dlc_model_path = "/x" if has_dlc else None
    tc.pose_resize_factor = 1.0
    tc.confidence_threshold = 0.5
    tc.keypoint_names = ("head",)
    tc.tracker_type = "dlc"
    tc.pose_n_instances = 1
    # DLC config → no SLEAP options; the signature's 6th element is "".
    tc.derived_sleap_opts.return_value = {}

    mw.pipeline.get_tracking_config.return_value = tc
    mw.pipeline.has_pose_model.return_value = model_loaded
    mw.pipeline.pose.is_enabled.return_value = pose_already_enabled

    # The box's own settings, through the one builder the real host uses.
    mw.pose_settings_for_box.return_value = {
        "mode": "dlc", "model_path": "/x", "body_parts": ["head"]}
    # Readiness: one answer, patched per scenario in _run below.
    mw._ready = bool(sig_matches and model_loaded)

    # Camera registry
    mw.video_manager.box_camera_map = {1: 99} if cam_connected else {}

    # Prompt
    mw.prompt_pose_init_required.return_value = prompt_choice

    # Init success by default
    mw._enable_pose_for_box.return_value = True

    mw.tracker_manager = None

    class W(RunTask):
        def __init__(self):
            self.setup_id = 1
            self.main_window = mw
            self._log_lines = []
        @property
        def _log_func(self):
            return lambda m: self._log_lines.append(m)

    return W(), mw, tc


def _run(w, mw):
    """Call the Record precheck with readiness answered for this scenario.

    Patched at the source, not simulated: the widget asks
    ``pose_subsystem.pose_ready_for`` exactly once, and every other caller
    (the dialog, the load path) asks the same function.
    """
    with patch("source.gui.pose_subsystem.pose_ready_for",
               return_value=mw._ready):
        return RunTask._auto_enable_tracking(w)


# ─── silent-proceed scenarios ─────────────────────────────────────


def test_no_main_window_proceeds_silently():
    class W(RunTask):
        def __init__(self):
            self.setup_id = 1; self.main_window = None
        @property
        def _log_func(self): return lambda m: None
    assert W()._auto_enable_tracking() is True


def test_no_pipeline_proceeds_silently():
    class W(RunTask):
        def __init__(self):
            self.setup_id = 1
            self.main_window = type("X", (), {})()
        @property
        def _log_func(self): return lambda m: None
    assert W()._auto_enable_tracking() is True


def test_online_tracking_disabled_proceeds_silently():
    w, mw, _ = _make_widget(has_dlc=True, online=False)
    assert _run(w, mw) is True
    mw.prompt_pose_init_required.assert_not_called()
    mw.pipeline.enable_pose.assert_not_called()


def test_nothing_configured_proceeds_silently():
    w, mw, _ = _make_widget(has_dlc=False, has_blob=False)
    assert _run(w, mw) is True
    mw.prompt_pose_init_required.assert_not_called()


def test_dlc_configured_no_camera_proceeds_silently():
    """SPEC: camera not connected + tracking enabled → silent.
    User can't track without camera; don't warn."""
    w, mw, _ = _make_widget(has_dlc=True, cam_connected=False)
    assert _run(w, mw) is True
    mw.prompt_pose_init_required.assert_not_called()
    mw.pipeline.enable_pose.assert_not_called()


def test_blob_configured_no_camera_proceeds_silently():
    w, mw, _ = _make_widget(has_blob=True, cam_connected=False)
    assert _run(w, mw) is True
    mw.pipeline.enable_blob_tracking.assert_not_called()


# ─── modal-firing scenarios (DLC + camera) ─────────────────────────


def test_dlc_camera_no_model_fires_modal():
    w, mw, _ = _make_widget(has_dlc=True, cam_connected=True,
                             model_loaded=False, prompt_choice="cancel")
    result = _run(w, mw)
    mw.prompt_pose_init_required.assert_called_once_with(1)
    assert result is False  # cancel aborts


def test_dlc_camera_settings_changed_fires_modal():
    w, mw, _ = _make_widget(has_dlc=True, cam_connected=True,
                             model_loaded=True, sig_matches=False,
                             prompt_choice="cancel")
    result = _run(w, mw)
    mw.prompt_pose_init_required.assert_called_once_with(1)
    assert result is False


# ─── modal choice handling ────────────────────────────────────────


def test_modal_init_now_calls_enable_pose_for_box_and_proceeds():
    w, mw, _ = _make_widget(has_dlc=True, cam_connected=True,
                             model_loaded=False, prompt_choice="init")
    result = _run(w, mw)
    mw._enable_pose_for_box.assert_called_once_with(
        1, mw.pose_settings_for_box.return_value)
    assert result is True


def test_modal_disable_sets_online_off_but_keeps_model_path():
    w, mw, tc = _make_widget(has_dlc=True, cam_connected=True,
                              model_loaded=False, prompt_choice="disable")
    result = _run(w, mw)
    # update_tracking_config called with online=False; model_path NOT touched
    call = mw.pipeline.update_tracking_config.call_args
    assert call.args == (1,)
    assert call.kwargs == {"online_tracking_enabled": False}
    # tc.dlc_model_path NEVER reassigned
    assert tc.dlc_model_path == "/x"
    assert result is True


def test_modal_cancel_aborts_record():
    w, mw, _ = _make_widget(has_dlc=True, cam_connected=True,
                             model_loaded=False, prompt_choice="cancel")
    result = _run(w, mw)
    assert result is False
    mw._enable_pose_for_box.assert_not_called()
    mw.pipeline.update_tracking_config.assert_not_called()


# ─── happy path: model loaded + signature matches ────────────────


def test_a_ready_box_enables_pose_with_no_modal():
    w, mw, _ = _make_widget(has_dlc=True, cam_connected=True,
                             model_loaded=True, sig_matches=True,
                             pose_already_enabled=False)
    result = _run(w, mw)
    mw.prompt_pose_init_required.assert_not_called()
    mw.pipeline.enable_pose.assert_called_once_with(1, zone_lookup=None)
    assert result is True


def test_a_ready_box_already_enabled_is_not_enabled_twice():
    w, mw, _ = _make_widget(has_dlc=True, cam_connected=True,
                             model_loaded=True, sig_matches=True,
                             pose_already_enabled=True)
    result = _run(w, mw)
    mw.prompt_pose_init_required.assert_not_called()
    mw.pipeline.enable_pose.assert_not_called()
    assert result is True


# ─── blob path ───────────────────────────────────────────────────


def test_blob_camera_enables_blob_tracking():
    w, mw, _ = _make_widget(has_blob=True, cam_connected=True)
    result = _run(w, mw)
    mw.pipeline.enable_blob_tracking.assert_called_once_with(1)
    assert result is True


# ─── dynamic adoption: apply_tracking_config always called ───────


def test_apply_tracking_config_always_called_at_record():
    """Even on silent-proceed paths, the latest TC must be pushed to
    the policy so mid-session dialog edits take effect."""
    w, mw, _ = _make_widget(has_dlc=True, cam_connected=False)
    _run(w, mw)
    mw.pipeline.apply_tracking_config.assert_called_once_with(1)
