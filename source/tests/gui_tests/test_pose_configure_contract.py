"""Characterisation of ``PoseSubsystemMixin._configure_pose_for_box``.

Written against the unmodified 194-line version, before splitting it, and the
only coverage this method has. It validates the camera, forces the stream to
colour, loads the model and seeds per-box state, but deliberately does NOT
start inference; the caller decides when to activate.

The behaviour worth pinning:

  * refuse, with a message, when there is no camera / no model / no frames
  * force grayscale off on all three surfaces that can silently re-enable it
  * wait for a genuinely colour frame after that toggle, rather than handing
    the model a stale grayscale one
  * accept both the modern and legacy config keys for the same setting
"""
from __future__ import annotations

import numpy as np

from source.gui.pose_subsystem import PoseSubsystemMixin

CONFIGURE = PoseSubsystemMixin._configure_pose_for_box


class _CamThread:
    def __init__(self, grayscale=False):
        self.grayscale = grayscale


class _VM:
    def __init__(self, cam_id=0, frame=None, grayscale=False):
        self.box_camera_map = {1: cam_id} if cam_id is not None else {}
        self.cameras = {cam_id: _CamThread(grayscale)} if cam_id is not None else {}
        self._frame = frame
        self.frames_served = 0

    def get_last_frame(self, setup_id):
        self.frames_served += 1
        return self._frame


class _Pipeline:
    """Models the real Pipeline surface the pose subsystem uses.

    Camera lookup and the grayscale force go through the Pipeline now, not
    through the ``video_manager`` alias, live capture state belongs to the
    pipeline that owns the thread. ``vm`` is still supplied so this double
    answers from the same fake camera map the test set up.
    """

    def __init__(self, handle="H", vm=None):
        self.handle = handle
        self.calls = []
        self.vm = vm

    def camera_id_for_box(self, setup_id):
        return (self.vm.box_camera_map.get(setup_id)
                if self.vm is not None else None)

    def force_color_for_box(self, setup_id):
        cam_id = self.camera_id_for_box(setup_id)
        thread = (self.vm.cameras.get(cam_id)
                  if self.vm is not None and cam_id is not None else None)
        if thread is None:
            return False
        was_gray = bool(getattr(thread, "grayscale", False))
        thread.grayscale = False
        return was_gray

    def configure_pose_model(self, **kw):
        self.calls.append(kw)
        return self.handle


class _Host(PoseSubsystemMixin):
    """Inherits the real mixin, the method under test and every helper it
    calls are the production ones; only the collaborators are stubbed."""

    def __init__(self, *, vm=None, pipeline=None, model_path="/m/net.yml",
                 video_grayscale=False):
        self.video_manager = vm if vm is not None else _VM(frame=_bgr())
        self.pipeline = (pipeline if pipeline is not None
                         else _Pipeline(vm=self.video_manager))
        # A caller-supplied pipeline still has to answer camera questions
        # against this host's camera map.
        if getattr(self.pipeline, "vm", None) is None:
            self.pipeline.vm = self.video_manager
        self.video_grayscale = video_grayscale
        self.pose_configs = {}
        self.pose_zone_state = {}
        self.tracking_zones = {}
        self.errors = []
        self.after_configured = []
        self.push_applied = []
        self._model_path = model_path

    def showError(self, msg):
        self.errors.append(msg)

    def _pose_resolve_model_path(self, cfg):
        return self._model_path

    def _apply_push_policy(self, setup_id):
        self.push_applied.append(setup_id)

    def _rebuild_zone_manager(self, setup_id, zones):
        pass

    def _pose_after_configured(self, setup_id, cfg, handle):
        self.after_configured.append((setup_id, handle))


def _bgr(h=48, w=64):
    return np.zeros((h, w, 3), np.uint8)


def _gray(h=48, w=64):
    return np.zeros((h, w), np.uint8)


# ── validation refusals ──────────────────────────────────────────────────

def test_no_camera_is_refused_with_a_message():
    host = _Host(vm=_VM(cam_id=None))
    assert CONFIGURE(host, 1, {"mode": "dlc"}) is False
    assert any("Connect a camera" in e for e in host.errors)


def test_no_model_path_is_refused_with_a_message():
    host = _Host(model_path="")
    assert CONFIGURE(host, 1, {"mode": "dlc"}) is False
    assert any("model path missing" in e for e in host.errors)


def test_no_frames_is_refused_with_a_message():
    host = _Host(vm=_VM(frame=None))
    assert CONFIGURE(host, 1, {"mode": "dlc"}) is False
    assert any("not delivering frames" in e for e in host.errors)


def test_a_failed_model_load_is_reported_as_failure():
    host = _Host(pipeline=_Pipeline(handle=None))
    assert CONFIGURE(host, 1, {"mode": "dlc"}) is False
    assert host.pose_configs == {}


def test_a_missing_backend_package_is_refused_with_install_advice():
    class _NoPkg(_Pipeline):
        def configure_pose_model(self, **kw):
            raise ImportError("no dlclive")
    host = _Host(pipeline=_NoPkg())
    assert CONFIGURE(host, 1, {"mode": "dlc"}) is False
    assert any("not installed" in e for e in host.errors)


# ── grayscale is forced off, everywhere ──────────────────────────────────

def test_grayscale_is_disabled_on_the_live_camera_thread():
    """Pose models need 3-channel BGR; a grayscale stream returns empty
    results with no error at all."""
    vm = _VM(frame=_bgr(), grayscale=True)
    host = _Host(vm=vm)
    assert CONFIGURE(host, 1, {"mode": "dlc"}) is True
    assert vm.cameras[0].grayscale is False


def test_the_persistent_grayscale_flag_is_cleared():
    """Otherwise the next camera reconnect silently re-enables it."""
    host = _Host(video_grayscale=True)
    assert CONFIGURE(host, 1, {"mode": "dlc"}) is True
    assert host.video_grayscale is False


def test_a_stale_grayscale_frame_is_not_handed_to_the_model():
    """After the toggle the camera's cached frame is still the last grayscale
    capture; the model must wait for a real colour one."""
    class _EventualColour(_VM):
        def get_last_frame(self, setup_id):
            self.frames_served += 1
            return _gray() if self.frames_served < 3 else _bgr()

    vm = _EventualColour(frame=None, grayscale=True)
    host = _Host(vm=vm)
    assert CONFIGURE(host, 1, {"mode": "dlc"}) is True
    probe = host.pipeline.calls[0]["probe_frame"]
    assert probe.ndim == 3, "a 2D frame reached the model"


def test_a_colour_camera_does_not_wait():
    vm = _VM(frame=_bgr(), grayscale=False)
    host = _Host(vm=vm)
    assert CONFIGURE(host, 1, {"mode": "dlc"}) is True
    assert vm.frames_served == 1


# ── model parameters ─────────────────────────────────────────────────────

def test_model_parameters_reach_the_pipeline():
    host = _Host()
    assert CONFIGURE(host, 1, {"mode": "dlc", "dlc_confidence": 0.8,
                               "dlc_resize": 0.5,
                               "body_parts": ["nose", "tail"]}) is True
    call = host.pipeline.calls[0]
    assert call["tracker_type"] == "dlc"
    assert call["confidence"] == 0.8
    assert call["resize_factor"] == 0.5
    assert call["body_parts"] == ["nose", "tail"]


def test_the_legacy_tracker_type_key_is_accepted():
    host = _Host()
    CONFIGURE(host, 1, {"tracker_type": "SLEAP"})
    assert host.pipeline.calls[0]["tracker_type"] == "sleap"


def test_the_mode_defaults_to_dlc():
    host = _Host()
    CONFIGURE(host, 1, {})
    assert host.pipeline.calls[0]["tracker_type"] == "dlc"


def test_sleap_options_are_only_sent_for_sleap():
    host = _Host()
    CONFIGURE(host, 1, {"mode": "dlc"})
    assert host.pipeline.calls[0]["sleap_opts"] is None

    host2 = _Host()
    CONFIGURE(host2, 1, {"mode": "sleap", "sleap_device": "cuda",
                         "sleap_fp16": True})
    opts = host2.pipeline.calls[0]["sleap_opts"]
    assert opts["device"] == "cuda" and opts["fp16"] is True


# ── per-box state + hooks ────────────────────────────────────────────────

def test_success_seeds_per_box_state_and_runs_the_hooks():
    host = _Host()
    assert CONFIGURE(host, 1, {"mode": "dlc"}) is True
    assert host.pose_configs[1]["mode"] == "dlc"
    assert 1 in host.pose_zone_state
    assert host.push_applied == [1]
    assert host.after_configured == [(1, "H")]


def test_a_raising_subclass_hook_does_not_fail_the_configure():
    class _BadHook(_Host):
        def _pose_after_configured(self, setup_id, cfg, handle):
            raise RuntimeError("writer failed to open")
    host = _BadHook()
    assert CONFIGURE(host, 1, {"mode": "dlc"}) is True


def test_the_stored_config_is_a_copy_not_the_caller_s_dict():
    """The caller keeps editing its dict; the per-box snapshot must not move
    underneath the pipeline."""
    host = _Host()
    cfg = {"mode": "dlc"}
    CONFIGURE(host, 1, cfg)
    cfg["mode"] = "sleap"
    assert host.pose_configs[1]["mode"] == "dlc"


def test_inference_is_not_started_here():
    """Activation is the caller's decision, maze defers it to Test Tracking."""
    host = _Host()
    CONFIGURE(host, 1, {"mode": "dlc"})
    assert not hasattr(host.pipeline, "enabled")
