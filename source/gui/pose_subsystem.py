"""Pose-tracking subsystem mixin, DLC / SLEAP integration for the main
windows.

One module by intent: debugging a pose issue (model load, grayscale
lockout, per-box config, dialog init flow, shutdown) means opening this
file, not skimming 4000 lines of base.py.

The mixin assumes the host class provides (initialised in
``MainWindowBase.__init__``):

    self.pipeline, the unified video pipeline.
    self.video_manager, multi-camera coordinator.
    self.pose_configs, dict[box_id -> cfg]
    self.pose_zone_state, dict[box_id -> {}]
    self.tracking_enabled, dict[box_id -> bool]
    self.tracking_zones, dict[box_id -> [...]]
    self._overlay, dict[box_id -> OverlayState]
    self.video_grayscale, bool (persistent grayscale flag)
    self.showError(msg), modal error helper
    self._rebuild_zone_manager(box_id, zones)
    self._apply_push_policy(box_id)
    self._refresh_status_badges()

Subclass hooks (operant + maze override as needed):
    _pose_resolve_model_path(cfg) -> Optional[str]
    _pose_after_configured(box_id, cfg, handle)
    _pose_after_disabled(box_id)
    _pose_after_dialog_init(cfg, dialog, success_count)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Tuple

from source.log import get_logger

logger = get_logger()


# ==================================================================
# The one settings builder, and the one readiness question
# ==================================================================

def pose_settings_from_config(tc: Any, *, dialog_globals: Optional[dict] = None,
                              model_path: Optional[str] = None) -> dict:
    """Every setting a pose model is built from, taken from one config.

    THE builder. Each path that loads a model used to hand-write its own dict
    and each forgot a different set of fields: the project-load path sent no
    backend options at all, the Record path sent an empty dict, and none of
    them sent the DLC engine options, so a project saved as FP16 always ran
    FP32 without a word. Worse, the dict at Record was not the dict at load,
    so the loader's cache missed and the model was built a second time with
    the operator waiting on it.

    The backend option dicts come from the config's own ``derived_*`` methods,
    which are also what the sink's cache key is built from, so there is one
    derivation rather than two that can drift apart.

    ``model_path`` overrides the config's, for the mode hook that resolves a
    ``.yml`` pick to the exported folder DeepLabCut-Live wants. Resolving it
    HERE means the load and the readiness check see the same string; resolving
    it only on the way in made every operant box look uninitialised.
    """
    g = dialog_globals or {}
    mode = str(getattr(tc, "tracker_type", "dlc") or "dlc").lower()
    if mode in ("sleap-nn", "sleap_nn"):
        mode = "sleap"
    path = model_path if model_path is not None else (
        getattr(tc, "dlc_model_path", None) or "")

    def _f(name, default):
        try:
            return float(getattr(tc, name, default))
        except (TypeError, ValueError):
            return float(default)

    return {
        "mode": mode,
        "tracker_type": mode,
        "model_path": str(path or ""),
        "body_parts": list(getattr(tc, "keypoint_names", ()) or ()),
        "skeleton": [list(e) for e in (getattr(tc, "skeleton", ()) or ())],
        "dlc_confidence": _f("confidence_threshold", 0.5),
        "resize_factor": _f("pose_resize_factor", 1.0),
        "pose_instances": int(getattr(tc, "pose_n_instances", 1) or 1),
        # Built once, by the config, in the shape the sink takes them.
        "sleap_opts": (tc.derived_sleap_opts()
                       if callable(getattr(tc, "derived_sleap_opts", None))
                       else {}) or {},
        "dlc_opts": (tc.derived_dlc_opts()
                     if callable(getattr(tc, "derived_dlc_opts", None))
                     else {}) or {},
        "pose_colour_mode": str(getattr(tc, "pose_colour_mode", "auto") or "auto"),
        "pose_input_mode": str(getattr(tc, "pose_input_mode", "auto") or "auto"),
        "pose_input_w": int(getattr(tc, "pose_input_w", 0) or 0),
        "pose_input_h": int(getattr(tc, "pose_input_h", 0) or 0),
        "pose_crop_conf_min": _f("pose_crop_conf_min", 0.20),
        "pose_crop_good_min": int(getattr(tc, "pose_crop_good_min", 3) or 3),
        "pose_crop_reacquire": bool(getattr(tc, "pose_crop_reacquire", True)),
        # Not model state - it filters the model's output - so it is applied
        # live at enable time and never asks for a re-init.
        "pose_gap_frames": int(getattr(tc, "pose_gap_frames", 5) or 0),
        # Not model state, but every caller of the loader also seeds per-box
        # display and smoothing state from the same dict.
        "smooth_tracking": bool(getattr(tc, "blob_smooth_tracking", True)),
        "annotate_parts": list(g.get("annotate_parts") or []),
        "zone_body_part": g.get("zone_body_part") or "",
        "marker_size": int(g.get("marker_size", 4) or 4),
        "coord_mapping": dict(g.get("coord_mapping") or {}),
        "triggers": list(g.get("triggers") or []),
        "enabled_boxes": [int(getattr(tc, "setup_id", 0) or 0)],
    }


def settings_fingerprint(settings: dict) -> Optional[Tuple]:
    """What :func:`pose_settings_from_config` says the model must be.

    Built by the SAME function the sink stores its own fingerprint with, so
    the two cannot describe one model differently. ``None`` when no model path
    is set, meaning there is nothing to compare.
    """
    if not settings or not settings.get("model_path"):
        return None
    from source.video.framebus.pose_sink import pose_fingerprint

    params = PoseSubsystemMixin._pose_model_params(settings)
    return pose_fingerprint(
        tracker_type=params["mode"], model_path=settings["model_path"],
        resize_factor=params["resize"], confidence=params["confidence"],
        body_parts=settings.get("body_parts") or (),
        sleap_opts=params["sleap_opts"], dlc_opts=params["dlc_opts"],
        colour_mode=params["colour_mode"], input_mode=params["input_mode"],
        input_wh=params["input_wh"], crop_opts=params["crop_opts"])


def pose_ready_for(pipeline: Any, settings: dict) -> bool:
    """True when the loaded model is exactly the one these settings ask for.

    The whole init question, in one place. A box whose settings produce the
    fingerprint the sink is holding needs nothing done to it: Record enables
    inference and the first frame goes straight through.
    """
    want = settings_fingerprint(settings)
    if want is None or pipeline is None:
        return False
    try:
        if not pipeline.has_pose_model():
            return False
        return pipeline.pose_fingerprint() == want
    except Exception as e:
        logger.debug("pose readiness check failed: %s", e)
        return False


def pose_settings_differ(loaded: Optional[Tuple],
                         settings: dict) -> Tuple[str, ...]:
    """Which parts of the model settings changed, for the log line.

    Names rather than two opaque tuples: "model_path, dlc_opts" tells an
    operator what they altered; a pair of tuples tells them nothing.
    """
    want = settings_fingerprint(settings)
    if want is None or loaded is None:
        return ("no model loaded",) if want is not None else ()
    names = ("tracker_type", "model_path", "resize", "confidence",
             "body_parts", "sleap_opts", "dlc_opts", "colour_mode",
             "input_mode", "input_size", "crop_opts")
    return tuple(n for n, a, b in zip(names, loaded, want) if a != b)


class PoseSubsystemMixin:
    """All DLC/SLEAP code paths in one place.  Mixed into MainWindowBase."""

    # ==================================================================
    # State predicates
    # ==================================================================

    def _pose_active(self) -> bool:
        """True if any DLC / SLEAP pose model is configured or running.

        Used as the grayscale lockout: pose models need 3-channel BGR
        input, so when this returns True every grayscale-toggle path
        force-overrides ``grayscale=True`` to ``False`` and logs a
        WARNING.  Silent grayscale-on with pose active produced
        empty pose results with no error message, that path is now
        guarded everywhere via this helper.
        """
        # 1. Per-box pose configs populated by _configure_pose_for_box
        #    (operant + maze both write here via the base class).
        if getattr(self, "pose_configs", None):
            return True
        # 2. Pipeline-side check: pose model loaded but no per-box config
        #    yet (rare race window during dialog Init DLC).
        try:
            if self.pipeline is not None and self.pipeline.has_pose_model():
                return True
        except Exception:
            pass
        # 3. Per-box TrackingConfig: user has configured DLC/SLEAP via
        #    the tracking dialog but hasn't started inference yet.  A
        #    DLC model path on any box is enough to lock grayscale off.
        try:
            for tc in self.pipeline.all_tracking_configs().values():
                if tc.has_dlc():
                    return True
        except Exception:
            pass
        return False

    def _resolve_grayscale_for_camera(self, requested: bool, *,
                                      setup_id=None) -> bool:
        """Apply the pose-mode grayscale lockout with a visible WARNING.

        Returns the effective grayscale setting.  When the request is
        True but a pose model is active, this returns False and emits
        a WARNING-level log that shows up in the user-visible log
        sidebar, no silent overrides.
        """
        if requested and self._pose_active():
            box_str = f" (box {setup_id})" if setup_id is not None else ""
            logger.warning(
                "Grayscale OFF forced%s, DLC/SLEAP active and pose models "
                "require 3-channel BGR input.  User-requested grayscale=True "
                "is being overridden to keep tracking working.",
                box_str,
            )
            return False
        return bool(requested)

    # ==================================================================
    # Subclass hooks (overridable; no-op by default)
    # ==================================================================

    def _pose_resolve_model_path(self, cfg) -> Optional[str]:
        mp = cfg.get("model_path") or cfg.get("dlc_model_path")
        if not mp:
            return None
        try:
            if not Path(mp).exists():
                return None
        except Exception:
            return None
        return mp

    def _pose_after_configured(self, setup_id: int, cfg, handle) -> None:
        """Subclass extras after a model has loaded for one box.  No-op default."""

    def _pose_after_disabled(self, setup_id: int) -> None:
        """Drop every piece of this box's pose runtime state.

        Both modes carry both pieces, ``pose_zone_state`` is declared in
        each mode's ``__init__``, ``tracking_enhancers`` lazily on the base,
        and BOTH must be cleared here: a partial clear leaves a box
        re-enabled after a disable running on the previous run's state,
        which for the enhancer means the predictor extrapolating from a
        stale position.

        Subclasses may still override for genuinely mode-specific extras, but
        must call ``super()``.
        """
        state = getattr(self, "pose_zone_state", None)
        if isinstance(state, dict):
            state.pop(setup_id, None)
        # Shared teardown: resets the enhancer, drops it, and clears it from
        # the tracker manager and the pose sink.
        detach = getattr(self, "_detach_tracking_enhancer", None)
        if callable(detach):
            try:
                detach(setup_id)
            except Exception as e:
                logger.debug("Box %s: enhancer detach failed: %s", setup_id, e)

    def _pose_after_dialog_init(self, cfg, dialog, success_count: int) -> None:
        """Hook for subclass-specific dialog post-processing.  No-op default."""

    # ==================================================================
    # Per-box configure / enable / disable
    # ==================================================================

    def pose_settings_for_box(self, setup_id: int) -> dict:
        """This box's pose settings, resolved, from its committed config.

        The single entry every path uses: project load, Test Tracking, Record,
        and the readiness check. Because they all get the same dict, the model
        loaded when the project opened is the model Record wants, the loader's
        cache hits, and starting a run costs one ``enable_pose`` call.
        """
        try:
            tc = self.pipeline.get_tracking_config(setup_id)
        except Exception as e:
            logger.debug("Box %s: no tracking config: %s", setup_id, e)
            return {}
        settings = pose_settings_from_config(
            tc, dialog_globals=getattr(self, "_tracking_dialog_globals", None))
        resolved = self._pose_resolve_model_path(settings)
        if resolved:
            settings["model_path"] = resolved
        return settings

    def pose_box_is_ready(self, setup_id: int) -> bool:
        """True when this box could start inferring right now."""
        settings = self.pose_settings_for_box(setup_id)
        return bool(settings) and pose_ready_for(self.pipeline, settings)

    def pose_configured_boxes(self) -> list:
        """Box ids that ask for pose: a model path, and tracking not opted out.

        A box with nothing configured is not asked about and not prepared. It
        records video and nothing else, which is what "no tracking configured"
        should mean.
        """
        out = []
        try:
            configs = self.pipeline.all_tracking_configs() or {}
        except Exception:
            return out
        for bid, tc in configs.items():
            if not getattr(tc, "online_tracking_enabled", True):
                continue
            if tc.has_dlc():
                out.append(int(bid))
        return sorted(out)

    def _configure_pose_for_box(self, setup_id, cfg) -> bool:
        """Validate the camera, load the pose model, seed per-box state.

        Does NOT activate inference, the caller decides whether to start
        immediately (operant, single step) or defer (maze: dialog init now,
        Test Tracking later).

        Returns True on success, False on any validation or init failure;
        each failure path has already told the operator what to fix.
        """
        try:
            cfg = dict(cfg or {})
            cam_id = self._pose_validate_camera(setup_id)
            if cam_id is None:
                return False
            cam_was_gray = self._pose_force_colour(setup_id, cam_id)

            model_path = self._pose_resolve_model_path(cfg)
            if not model_path:
                logger.error("Box %s: configure_pose: no/invalid model_path",
                             setup_id)
                self._pose_say(
                    f"Box {setup_id}: model path missing or not found.\n"
                    "Pick a .yml config or model directory in the dialog.")
                return False

            probe_frame = self._pose_probe_frame(setup_id,
                                                 require_color=cam_was_gray)
            if probe_frame is None:
                logger.error("Box %s: configure_pose: no probe frame", setup_id)
                self._pose_say(
                    f"Box {setup_id}: camera not delivering frames yet.\n"
                    "Wait a moment after connect, then retry.")
                return False

            params = self._pose_model_params(cfg)
            handle = self._pose_load_model(setup_id, params, model_path,
                                           probe_frame)
            if handle is None:
                return False

            self._pose_seed_box_state(setup_id, cfg, handle)
            logger.info("Box %s: pose configured (mode=%s)",
                        setup_id, params["mode"])
            return True

        except Exception as e:
            logger.error("Box %s: configure_pose error: %s", setup_id, e)
            return False

    # ── _configure_pose_for_box steps ─────────────────────────────────

    def _pose_say(self, message: str) -> None:
        """Surface a failure to the operator, tolerating a host without a
        message surface (headless / tests)."""
        try:
            self.showError(message)
        except Exception:
            pass

    def _pose_validate_camera(self, setup_id):
        """The camera id bound to this box, or ``None`` after complaining."""
        cam_id = self.pipeline.camera_id_for_box(setup_id)
        if cam_id is None:
            logger.error("Box %s: configure_pose: no camera connected", setup_id)
            self._pose_say(f"Connect a camera for Box {setup_id} first.")
        return cam_id

    def _pose_force_colour(self, setup_id, cam_id) -> bool:
        """Turn grayscale off everywhere it can be re-enabled from.

        Pose models need 3-channel BGR, and a grayscale stream makes them
        return empty results with no error at all. Three surfaces have to
        agree or the next reconnect, or an "Apply" in the camera dialog,
        silently switches it back on:

          a) the live camera thread (mid-stream toggle)
          b) the window's persistent flag (next connect_camera)
          c) the tracking config snapshot (saved-config reload)

        Returns whether the live camera *was* grayscale, which tells the probe
        step it must wait for a genuinely colour frame.
        """
        cam_was_gray = False
        try:
            # Through the Pipeline, not by reaching into
            # video_manager.cameras[...] and assigning the thread's flag,
            # live capture state belongs to the pipeline that owns the thread.
            cam_was_gray = self.pipeline.force_color_for_box(setup_id)
            if cam_was_gray:
                logger.warning(
                    "Box %s: grayscale FORCE-DISABLED on live camera, "
                    "DLC/SLEAP requires 3-channel BGR. Reverting "
                    "grayscale=True to grayscale=False for this session.",
                    setup_id)
        except Exception as e:
            logger.error(
                "Box %s: failed to disable grayscale on camera thread: %s",
                setup_id, e)
        if getattr(self, "video_grayscale", False):
            self.video_grayscale = False
            logger.warning(
                "Box %s: video_grayscale flag cleared, pose mode locks the "
                "camera to colour for the rest of this session.", setup_id)
        return cam_was_gray

    def _pose_probe_frame(self, setup_id, require_color: bool):
        """A frame to initialise the model on, or ``None`` after ~2 s.

        When grayscale was just toggled off, the camera thread's cached frame
        is still the last 2D capture from before the toggle. DLC will accept
        it, but the model then initialises on a BGR image synthesised from
        grayscale, which mis-trains the first batch shape on some graphs,
        so wait for a real colour frame (typically 1-2 frame intervals),
        capped so a stalled camera cannot hang the dialog.
        """
        for attempt in range(20):                    # 20 x 0.1 s = 2 s
            try:
                frame = self.video_manager.get_last_frame(setup_id)
            except Exception:
                frame = None
            if frame is None:
                time.sleep(0.1)
                continue
            if require_color and frame.ndim == 2:
                if attempt % 5 == 0:
                    logger.info(
                        "Box %s: waiting for fresh BGR frame after grayscale "
                        "toggle (attempt %d/20)", setup_id, attempt + 1)
                time.sleep(0.1)
                continue
            return frame
        return None

    @staticmethod
    def _pose_model_params(cfg) -> dict:
        """Model settings, accepting both the current and legacy config keys.

        SLEAP-only options stay ``None`` for DLC and blob so their
        constructors are not handed arguments they do not understand.
        """
        mode = (cfg.get("mode") or cfg.get("tracker_type") or "dlc").lower()
        # Already built by the config itself (pose_settings_from_config).
        # Preferred over re-deriving from loose keys, because the config's
        # ``derived_*`` methods are also what the sink's cache key is built
        # from: one derivation, so the loaded model and the readiness check
        # cannot disagree about what was asked for.
        prebuilt_sleap = cfg.get("sleap_opts")
        prebuilt_dlc = cfg.get("dlc_opts")
        sleap_opts = None
        if mode in ("sleap", "sleap-nn", "sleap_nn"):
            sleap_opts = {
                # Namespaced: ``filter_options`` drops a bare ``model_type`` as
                # ambiguous between the two backends, so this key is the only
                # spelling that reaches the SLEAP constructor.
                "sleap_model_type": cfg.get("sleap_model_type", "auto"),
                "centroid_path":  cfg.get("sleap_centroid_path") or None,
                "runtime":        cfg.get("sleap_runtime", "auto"),
                "device":         cfg.get("sleap_device", "auto"),
                "fp16":           bool(cfg.get("sleap_fp16", False)),
                "compile":        bool(cfg.get("sleap_compile", False)),
                "peak_threshold": float(cfg.get("sleap_peak_threshold", 0.2)),
            }
        # DLC-only options, mirroring the SLEAP block: left None for other
        # backends so their constructors never see arguments they do not take.
        dlc_opts = None
        if mode in ("dlc", "deeplabcut"):
            dlc_opts = {
                # Namespaced for the same reason as sleap_model_type above.
                "dlc_model_type": cfg.get("dlc_model_type", "auto"),
                "precision":  cfg.get("dlc_precision", "FP32"),
                "device":     cfg.get("dlc_device", "auto"),
            }
        if prebuilt_sleap:
            sleap_opts = dict(prebuilt_sleap)
        if prebuilt_dlc:
            dlc_opts = dict(prebuilt_dlc)
        return {
            "mode": mode,
            "confidence": float(cfg.get("dlc_confidence")
                                or cfg.get("sleap_confidence")
                                or cfg.get("confidence", 0.5)),
            "resize": float(cfg.get("resize_factor")
                            or cfg.get("dlc_resize") or 1.0),
            "body_parts": cfg.get("body_parts") or None,
            "sleap_opts": sleap_opts,
            "dlc_opts": dlc_opts,
            "colour_mode": cfg.get("pose_colour_mode", "auto") or "auto",
            # How the frame becomes the model's input, and the window it cuts
            # when it follows the animal. Absent means letterbox, which is what
            # a project that never set it expects.
            "input_mode": cfg.get("pose_input_mode", "auto") or "auto",
            "input_wh": (int(cfg.get("pose_input_w") or 0),
                         int(cfg.get("pose_input_h") or 0)),
            "crop_opts": {
                "conf_min": float(cfg.get("pose_crop_conf_min", 0.20) or 0.20),
                "good_min": int(cfg.get("pose_crop_good_min", 3) or 3),
                "reacquire": bool(cfg.get("pose_crop_reacquire", True)),
            },
        }

    def _pose_load_model(self, setup_id, params, model_path, probe_frame):
        """Load through the pipeline, which caches on the
        path/shape/resize/type/sleap signature, identical settings reuse the
        already-loaded model rather than paying for a second init.
        """
        try:
            handle = self.pipeline.configure_pose_model(
                tracker_type=params["mode"],
                model_path=model_path,
                resize_factor=params["resize"],
                body_parts=params["body_parts"],
                confidence=params["confidence"],
                probe_frame=probe_frame,
                sleap_opts=params["sleap_opts"],
                input_mode=params["input_mode"],
                input_wh=params["input_wh"],
                crop_opts=params["crop_opts"],
                dlc_opts=params["dlc_opts"],
                colour_mode=params["colour_mode"],
            )
        except ImportError:
            self._pose_say(
                f"{params['mode'].upper()} package is not installed.\n\n"
                f"Install it with:  pip install dlclive")
            return None
        if handle is None:
            logger.error("Box %s: pose model init failed (mode=%s)",
                         setup_id, params["mode"])
        return handle

    def _pose_seed_box_state(self, setup_id, cfg, handle) -> None:
        """Per-box state, push policy, and the subclass hook.

        The push policy applies zone-derived rules first and then the dialog's
        Coord Mapping / Triggers on top (dialog wins on conflict), the same
        shape blob uses. Both paths must apply both layers, or the operator's
        ``c.<var>`` push silently never happens.
        """
        self.pose_configs[setup_id] = cfg
        self.pose_zone_state.setdefault(setup_id, {})

        # Drawing facts, written once per enable rather than per frame: which
        # parts join to which, and whose name is written beside its marker.
        # Both are properties of the configured model, not of any one result.
        try:
            self._on_overlay_update(
                setup_id,
                skeleton=[tuple(e) for e in (cfg.get("skeleton") or [])
                          if len(e) == 2],
                annotate_parts=list(cfg.get("annotate_parts") or []),
            )
        except Exception as e:
            logger.debug("overlay skeleton seed (%s) failed: %s", setup_id, e)

        zones = getattr(self, "tracking_zones", {}).get(setup_id, [])
        if zones:
            try:
                self._rebuild_zone_manager(setup_id, zones)
            except Exception:
                pass
        self._apply_push_policy(setup_id)

        # Subclass extras: open the tracking writer, store the global model
        # path, attach the Kalman enhancer. A failure here must not undo a
        # model that loaded fine.
        # The smoothing hook was only ever called from the BLOB path, so a
        # pose box carried no enhancer whatever the dialog said. The same one
        # serves both: PoseSink.set_enhancer exists precisely so pose can share
        # the Kalman state without going through the blob tracker.
        attach = getattr(self, "_attach_kalman_enhancer", None)
        if callable(attach):
            try:
                attach(setup_id, {"smooth_tracking":
                                  bool(cfg.get("smooth_tracking", True))})
            except Exception as e:
                logger.warning("Box %s: pose smoothing not attached: %s",
                               setup_id, e)

        try:
            self._pose_after_configured(setup_id, cfg, handle)
        except Exception as e:
            logger.debug("_pose_after_configured(%s) error: %s", setup_id, e)

    def _enable_pose_for_box(self, setup_id, cfg) -> bool:
        """Configure + activate pose tracking (PoseSink starts inferring)."""
        if not self._configure_pose_for_box(setup_id, cfg):
            return False
        self.tracking_enabled[setup_id] = True
        zm = None
        try:
            tm = getattr(self, "tracker_manager", None)
            if tm is not None and hasattr(tm, "get_zone_manager"):
                zm = tm.get_zone_manager(setup_id)
        except Exception:
            zm = None
        # Per-keypoint filtering and gap fill. Applied at enable, not at load:
        # it filters the model's output rather than changing the model, so
        # altering it must never cost a re-init.
        try:
            gap = int((cfg or {}).get("pose_gap_frames", 5) or 0)
            if bool((cfg or {}).get("smooth_tracking", True)):
                self.pipeline.pose.set_gap_fill(setup_id, gap)
            else:
                self.pipeline.pose.set_gap_fill(setup_id, 0)
        except Exception as e:
            logger.debug("Box %s: gap fill not configured: %s", setup_id, e)
        try:
            self.pipeline.enable_pose(setup_id, zone_lookup=zm)
        except Exception as e:
            logger.error("Box %s: pipeline.enable_pose error: %s", setup_id, e)
            return False
        return True

    def _disable_pose_for_box(self, setup_id) -> None:
        """Disable pose for one box. Reversible, use ``_enable_pose_for_box``
        again to restart with the cached model (no re-init)."""
        try:
            self.pipeline.disable_pose(setup_id)
        except Exception as e:
            logger.debug("pipeline.disable_pose(%s) error: %s", setup_id, e)
        try:
            self.tracking_enabled[setup_id] = False
        except Exception:
            pass
        for d in (getattr(self, "pose_configs", None),
                  getattr(self, "pose_zone_state", None)):
            try:
                if isinstance(d, dict):
                    d.pop(setup_id, None)
            except Exception:
                pass
        # Clear pose fields on the OverlayState (leave blob fields if blob
        # is still active, they'll age out separately via TTL).
        state = self._overlay.get(setup_id)
        if state is not None:
            state.pose = []
            state.body_parts = []
            state.skeleton = []
            state.annotate_parts = []
        try:
            self._pose_after_disabled(setup_id)
        except Exception as e:
            logger.debug("_pose_after_disabled(%s) error: %s", setup_id, e)

    def _shutdown_pose(self) -> None:
        """Disable pose for every active box. Called on app close."""
        for setup_id in list(getattr(self, "pose_configs", {}).keys()):
            try:
                self._disable_pose_for_box(setup_id)
            except Exception:
                pass
        try:
            self.pose_configs.clear()
            self._overlay.clear()
            self.pose_zone_state.clear()
        except Exception:
            pass

    # ==================================================================
    # Tracking-config dialog DLC init flow
    # ==================================================================

    def _handle_dlc_init_from_dialog(self, cfg, dialog) -> None:
        """Common DLC/SLEAP init flow used by both modes.

        Loads the model on every enabled+camera-connected box (same
        cached handle is reused across boxes, InferenceBackend keys
        on path/shape/resize/type). Reports back to the dialog.
        Subclasses can override for mode-specific extras (e.g. maze
        defers inference activation).

        Shows a modal BusyDialog popup while the model is loaded +
        configured (DLC/SLEAP init can take many seconds on large
        models). The popup advances per box via ``set_step``; the
        dialog repaints between steps via processEvents.
        """
        # Lazy import, common.py is in source.gui.widgets, which we
        # don't want to drag into base.py's module-load graph.
        from source.gui.widgets.common import BusyDialog
        busy = None
        try:
            enabled_boxes = cfg.get("enabled_boxes", [])
            if not enabled_boxes:
                dialog.settings_panel.set_dlc_init_result(False, "No boxes selected")
                return

            connected_with_cam = [
                bid for bid in enabled_boxes
                if (hasattr(self, "video_manager")
                    and bid in self.video_manager.box_camera_map
                    and self.video_manager.box_camera_map[bid] is not None)
            ]
            if not connected_with_cam:
                dialog.settings_panel.set_dlc_init_result(
                    False, "No connected cameras for selected boxes")
                return

            # Force grayscale OFF for every camera that this dialog touches.
            for bid in connected_with_cam:
                cid = self.video_manager.box_camera_map.get(bid)
                cam = self.video_manager.cameras.get(cid)
                if cam is not None and getattr(cam, "grayscale", False):
                    cam.grayscale = False

            # Per-box settings through the ONE builder. The dialog payload is
            # mapped onto a config with the same functions the commit uses, so
            # the model this button loads is bit-for-bit the model the box will
            # ask for afterwards. Hand-building a settings dict here is what
            # made a freshly initialised box report "needs init" at Record.
            from source.gui.base import tracking_config_from_dialog
            tracker_type = cfg.get("tracker_type", "dlc")
            model_path = cfg.get("model_path", "")
            settings = pose_settings_from_config(
                tracking_config_from_dialog(cfg, 0),
                dialog_globals={
                    "annotate_parts": cfg.get("annotate_parts") or [],
                    "zone_body_part": cfg.get("zone_body_part") or "",
                    "marker_size": cfg.get("marker_size", 4),
                    "coord_mapping": cfg.get("coord_mapping") or {},
                    "triggers": cfg.get("triggers") or [],
                })
            # The Smoothing checkbox lives on the dialog, not on a committed
            # config yet. Pose needs it more than blob does: keypoint jitter is
            # per-part and per-frame, and it lands in speed and in zone
            # crossings before anything downstream can see it.
            settings["smooth_tracking"] = bool(cfg.get("smooth_tracking", True))
            # Resolve the path here, not inside the load: a ``.yml`` pick
            # becomes the exported folder, and the fingerprint the sink stores
            # must be the one the box's own settings will produce later.
            settings["model_path"] = (self._pose_resolve_model_path(settings)
                                      or settings["model_path"])

            # Reset the per-box overlay cache so the dialog probe is the
            # only frame the model has seen.
            try:
                self._overlay.clear()
            except Exception:
                pass

            # -- Progress popup --
            # n_steps = 1 for the initial model-load + n boxes, plus 1
            # final wrap-up tick.
            n_steps = 1 + len(connected_with_cam) + 1
            # "Pose" when the type is unknown, never "DLC": naming one
            # backend by default sent SLEAP operators to the wrong install.
            label_kind = tracker_type.upper() if tracker_type else "Pose"
            model_short = ("..." + model_path[-40:]) if len(model_path) > 40 else model_path
            busy = BusyDialog.stepped(
                dialog if dialog is not None else self,
                f"Initializing {label_kind} model",
                n_steps,
                detail=f"Loading model from {model_short}..." if model_path else
                       f"Loading {label_kind} model...",
            )

            # Multi-instance pose: user-controlled via the
            # ``Instances`` spinner in the tracking config dialog
            # (default 1 = single shared session).  Must be set BEFORE
            # configure_pose_for_box so the model loads onto the
            # correct backend.
            try:
                n_inst = max(1, int(cfg.get("pose_instances", 1)))
                self.pipeline.set_pose_n_instances(n_inst)
            except Exception as e:
                logger.debug("set_pose_n_instances raised: %s", e)
            busy.set_step(1, f"Model loaded - configuring {len(connected_with_cam)} box(es)...")

            success_count = 0
            for i, bid in enumerate(connected_with_cam, start=1):
                busy.set_step(1 + i, f"Configuring Box {bid}...")
                try:
                    if self._configure_pose_for_box(bid, settings):
                        success_count += 1
                except Exception as e:
                    logger.error("Box %s: dialog DLC init failed: %s", bid, e)

            busy.set_step(n_steps, "Finalizing...")

            if success_count > 0 and self.pipeline.has_pose_model():
                n_parts = len(self.pipeline.pose_model_info()[0])
                # Name the runtime that is ACTUALLY running. A TensorRT engine
                # that fails to build falls back to ONNX and then to native
                # torch, correct behaviour, but invisible if only logged, and
                # an operator on FP32 torch will not think to ask why the rig
                # is slow.
                running = self.pipeline.pose_resolved_backend()
                final_msg = (
                    f"Model loaded ({n_parts} parts, {success_count} box"
                    f"{'es' if success_count > 1 else ''}"
                    f"{', ' + running if running else ''})"
                )
                dialog.settings_panel.set_dlc_init_result(True, final_msg)
                if busy is not None:
                    busy.finish(success=True, msg=final_msg)
            else:
                fail_msg = (f"Init failed - check the model path and the "
                            f"{label_kind} install")
                dialog.settings_panel.set_dlc_init_result(False, fail_msg)
                if busy is not None:
                    busy.finish(success=False, msg=fail_msg)

            try:
                self._pose_after_dialog_init(cfg, dialog, success_count)
            except Exception as e:
                logger.debug("_pose_after_dialog_init error: %s", e)

            # DLC-init state just changed - repaint the bottom-bar
            # tracking status so the "click Init" hint clears.
            try:
                self._refresh_status_badges()
            except Exception:
                pass
        except Exception as e:
            logger.error("DLC init from dialog failed: %s", e)
            try:
                dialog.settings_panel.set_dlc_init_result(False, str(e))
            except Exception:
                pass
            if busy is not None:
                try:
                    busy.finish(success=False, msg=str(e))
                except Exception:
                    pass
