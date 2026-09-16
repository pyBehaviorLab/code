"""
source/gui/base.py - Shared MainWindow base class.

Houses every method whose body is the same (or differs only by which
per-box widget collection it consults) between pyOperant and pyMaze.
Mode-specific behavior is dispatched through abstract hooks the
subclasses override.

Inheritance chain:
    QtWidgets.QMainWindow
            ↑
    MainWindowUtilsMixin (file/theme/error utilities -- main_window_mixins.py)
            ↑
    MainWindowBase (this file -- camera, display, state, lifecycle)
            ↑
    MainWindow (operant or maze -- layout + dialogs + mode-specific events)

Every hook below raises NotImplementedError by default. Anything
missing on a subclass surfaces immediately rather than silently
bypassing.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import ClassVar, Iterable, Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from source import paths as app_paths
from source.config import experiment as _cfg_schema
from source.video.recording.drop_log import drop_log as _drop_log
from source.video.tracking import background as _bg_mod
from source.gui.main_window_mixins import MainWindowUtilsMixin
from source.gui.pose_subsystem import (PoseSubsystemMixin, pose_ready_for,
                                       pose_settings_differ)
from source.gui.window_behavior import close_independent_windows
from source.gui import project_workflow as _pw
from source.datetime_formats import format_run_clock

logger = logging.getLogger(__name__)

#: Annotation marker radius in pixels when a box has not chosen one. Module
#: level so the drawing path can fall back without depending on the host
#: object carrying the setting.
MARKER_SIZE_DEFAULT = 4


# Optional Cython acceleration for zone overlay drawing.
try:
    from source.cython.drawing_ops import (
        prepare_dlc_keypoints as _cy_prepare_dlc_keypoints,
    )
    _HAS_CY_DRAWING = True
    logger.info("Cython drawing_ops loaded (fast keypoint path active)")
except Exception as _cy_err:
    _HAS_CY_DRAWING = False
    _cy_prepare_dlc_keypoints = None
    logger.warning(
        "Cython drawing_ops unavailable (%s), using pure-Python keypoint loop. "
        "Run 'python -m source.cython.build' to compile.", _cy_err)


# Maximum age (in monotonic nanoseconds) of a cached overlay result before
# the renderer drops it.  200 ms ≈ 6 frames at 30 fps and 12 frames at
# 60 fps, generous enough that an in-flight inference completing during
# normal latency (typ. 30–60 ms) still draws, but tight enough that when
# pose / blob stops flowing (Stop, framework error, DLC stall) the live
# preview clears within ~⅕ s.
_OVERLAY_MAX_AGE_NS = 200_000_000


def _merge_draft_tree(src_root: Path, dst_root: Path) -> None:
    """Copy a draft snapshot workspace into a named project folder on first
    Save. ``source/`` + ``_pending/`` files are content-addressed (``<djb2>``)
    so a plain copy-if-absent is safe; ``change_log.jsonl`` is appended (the
    project may already have a 'create' line). One-time, off the hot path."""
    import shutil
    for sub in ("source", "_pending"):
        s = src_root / sub
        if not s.is_dir():
            continue
        for f in s.rglob("*"):
            if not f.is_file():
                continue
            dst = dst_root / f.relative_to(src_root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(f, dst)
    s_log = src_root / "change_log.jsonl"
    if s_log.is_file():
        d_log = dst_root / "change_log.jsonl"
        d_log.parent.mkdir(parents=True, exist_ok=True)
        with open(s_log, "r", encoding="utf-8") as fh:
            data = fh.read()
        if data:
            with open(d_log, "a", encoding="utf-8") as fh:
                fh.write(data)


class _AutosaveSignals(QtCore.QObject):
    """Worker → GUI completion bridge for the autosave QRunnable.
    QRunnable can't emit signals on its own, every worker carries a
    short-lived QObject helper. Signal payload: (ok, msg, cfg).
    """
    done = QtCore.Signal(bool, str, object)


class _AutosaveRunnable(QtCore.QRunnable):
    """Disk-I/O half of an autosave. Runs on QThreadPool, never touches Qt.
    The GUI thread snapshots the live widgets into ``cfg`` first; this
    worker only serializes the dataclass tree and writes the JSON
    files. Single-flight is enforced by the host's
    ``_autosave_in_flight`` flag; cross-thread mutation of the
    ProjectFileGuard is guarded by ``save_lock``.
    """

    def __init__(self, cfg, project_dir, guard, save_lock):
        super().__init__()
        self.setAutoDelete(True)
        self.cfg = cfg
        self.project_dir = project_dir
        self.guard = guard
        self.save_lock = save_lock
        self.signals = _AutosaveSignals()

    def run(self):  # noqa: C901 - thin wrapper, control flow is the catch list
        try:
            from source.config.experiment import (
                save_experiment, save_template, serialize_for_save,
            )
            from source.config.multi_instance import SaveConflictError
            from source.gui.project_workflow import _ensure_history_dir
            with self.save_lock:
                payload = serialize_for_save(self.cfg)
                try:
                    save_experiment(
                        self.cfg,
                        project_dir_path=self.project_dir,
                        guard=self.guard,
                        on_conflict=(lambda _c: "cancel") if self.guard else None,
                        cached_payload=payload,
                    )
                except SaveConflictError as ce:
                    paths = "; ".join(".".join(c.path) for c in ce.conflicts[:3])
                    self.signals.done.emit(
                        False,
                        f"deferred:{len(ce.conflicts)}:{paths}",
                        self.cfg,
                    )
                    return
                save_template(self.cfg, self.project_dir, cached_payload=payload)
                _ensure_history_dir(self.project_dir)
            self.signals.done.emit(True, "", self.cfg)
        except Exception as e:
            self.signals.done.emit(False, f"error:{e}", self.cfg)


# ── tracking-dialog fan-out helpers ──────────────────────────────────────
# Pure translations from the dialog's key names to the TrackingConfig
# registry's. Module-level because they need nothing from the window.

def _dialog_tracker_type(tracking: dict) -> str:
    """Registry name for the dialog's mode. The dialog says "normal" where
    the registry says "blob"."""
    tracker_type = str(tracking.get("tracker_type")
                       or tracking.get("mode") or "blob").lower()
    return "blob" if tracker_type == "normal" else tracker_type


def _dialog_box_ids(tracking: dict):
    """``(enabled, target_boxes)``.

    Rig-level settings land on every box the dialog rendered, not just the
    ticked ones, the checkbox only gates ``online_tracking_enabled``, so an
    operator can tick a box later without re-picking the model path. When the
    dialog did not report what it rendered, fall back to the ticked boxes.
    """
    def _ints(seq):
        out = []
        for raw in seq or []:
            try:
                out.append(int(raw))
            except (TypeError, ValueError):
                continue
        return out

    enabled = set(_ints(tracking.get("enabled_boxes")))
    rendered = _ints(tracking.get("all_dialog_boxes"))
    return enabled, (rendered or sorted(enabled))


def _dialog_shared_fields(tracking: dict, tracker_type: str,
                          online: bool) -> dict:
    """Fields every box gets, whatever the tracker.

    ``user_applied`` marks this as a config the operator chose, the project
    save filter uses it to drop pipeline defaults. Push gates are only written
    when the dialog sent them, so an absent key keeps the dataclass default.
    """
    fields = {
        "tracker_type": tracker_type,
        "user_applied": True,
        "online_tracking_enabled": online,
    }
    for key in ("push_zones_to_mcu", "push_coords_to_mcu",
                "push_frame_event"):
        if key in tracking:
            fields[key] = bool(tracking[key])
    if "zone_change_body_part" in tracking:
        fields["zone_change_body_part"] = str(
            tracking.get("zone_change_body_part") or "centroid")
    # One authored trigger table in the dialog, applied to every box, like
    # coord_mapping. Reaches the MCU policy via apply_tracking_config.
    if "triggers" in tracking:
        fields["triggers"] = list(tracking.get("triggers") or [])
    return fields


def _pose_field_names() -> frozenset:
    """TrackingConfig fields the dialog may write straight through.

    Read off the dataclass rather than listed by hand: a pose option added to
    the config and to the panel then persists without anyone remembering to
    extend a table here. That table is exactly what dropped ``dlc_precision``,
    ``dlc_device``, ``pose_colour_mode`` and the whole pose-input group on the
    floor, so a project could show FP16 in the dialog and load FP32 for ever.
    """
    import dataclasses

    from source.video.framebus.types import TrackingConfig
    return frozenset(
        f.name for f in dataclasses.fields(TrackingConfig)
        if (f.name.startswith(("sleap_", "pose_", "dlc_"))
            or f.name in ("skeleton", "keypoint_names", "rotation_enabled",
                          "rotation_keypoints", "n_animals", "identity_method"))
        and f.name != "dlc_model_path")            # set explicitly below


def _dialog_pose_fields(tracking: dict, body_parts) -> dict:
    """DLC / SLEAP fields, under their registry names."""
    fields = {"dlc_model_path": tracking.get("model_path") or None}
    if body_parts:
        fields["keypoint_names"] = tuple(body_parts)
    if "dlc_confidence" in tracking:
        fields["confidence_threshold"] = float(tracking["dlc_confidence"])
    if "dlc_resize" in tracking:
        fields["pose_resize_factor"] = float(tracking["dlc_resize"])
    if "pose_instances" in tracking:
        try:
            fields["pose_n_instances"] = int(tracking["pose_instances"])
        except (TypeError, ValueError):
            pass
    # Everything else the panel emits under its own registry name: the SLEAP
    # options, the DLC engine options, the colour mode and the pose-input
    # group. Names are checked against the dataclass, so a stray dialog key
    # cannot invent a config field.
    for key in _pose_field_names():
        if key in tracking:
            value = tracking[key]
            if key == "skeleton":
                value = tuple(tuple(e) for e in (value or ()) if len(e) == 2)
            fields[key] = value
    return fields


def tracking_config_from_dialog(tracking: dict, setup_id: int):
    """A throwaway ``TrackingConfig`` carrying what the dialog is showing.

    The dialog's Init runs on a payload the operator has not applied yet, so
    there is no committed config to read. Rather than hand-build a second
    settings dict there (which is how the dialog's model and the Record check
    came to disagree), the payload is mapped onto a config with the SAME
    functions the commit uses, and the one builder takes it from there.
    """
    import dataclasses

    from source.video.framebus.types import TrackingConfig

    tracker_type = _dialog_tracker_type(tracking)
    fields = _dialog_shared_fields(tracking, tracker_type, True)
    fields.update(_dialog_pose_fields(tracking, tracking.get("body_parts") or []))
    known = {f.name for f in dataclasses.fields(TrackingConfig)}
    return TrackingConfig(setup_id=int(setup_id),
                          **{k: v for k, v in fields.items() if k in known})


def _dialog_blob_fields(tracking: dict, smooth: bool) -> dict:
    """Blob fields, under their registry names.

    ``dlc_model_path`` is cleared explicitly: switching a box from pose to
    blob must not leave a stale model path behind for the next reload to
    resurrect.
    """
    fields = {"dlc_model_path": None, "blob_smooth_tracking": smooth}
    for src, dst in (("threshold", "blob_threshold"),
                     ("min_area", "blob_min_area"),
                     ("max_area", "blob_max_area"),
                     ("detect_dark", "blob_detect_dark"),
                     ("blur_mode", "blob_blur_mode"),
                     ("blur_kernel_size", "blob_blur_kernel_size"),
                     ("bg_mode", "blob_bg_mode"),
                     ("open_kernel_size", "blob_open_kernel_size"),
                     ("close_kernel_size", "blob_close_kernel_size"),
                     ("self_norm_ratio", "blob_self_norm_ratio"),
                     ("self_norm_sigma", "blob_self_norm_sigma"),
                     ("self_norm_smooth_sigma", "blob_self_norm_smooth_sigma"),
                     ("self_norm_minsize", "blob_self_norm_minsize")):
        if src in tracking:
            fields[dst] = tracking[src]
    return fields


def _dialog_annotate_flag(annotate_saved: dict, bid: int):
    """Per-box "save annotated video", or ``None``. JSON round-trips the key
    as a string; the live dialog hands it back as an int."""
    flag = annotate_saved.get(str(bid))
    return annotate_saved.get(bid) if flag is None else flag


def _dialog_global_overrides(full_cfg: dict, tracking: dict,
                             tracker_type: str, annotate_saved: dict) -> dict:
    """The residual dialog state with no per-box TrackingConfig home."""
    overrides = {
        "coord_mapping": (tracking.get("coord_mapping")
                          or full_cfg.get("coord_mapping") or {}),
        "triggers": list(tracking.get("triggers") or []),
        "annotate_parts": list(tracking.get("annotate_parts") or []),
        "zone_body_part": str(tracking.get("zone_body_part") or ""),
        "marker_size": int(tracking.get("marker_size", 4) or 4),
        "annotate_saved": {str(k): bool(v) for k, v in annotate_saved.items()},
        "task_name": str(full_cfg.get("task_name", "") or ""),
        "tracker_type": tracker_type,
    }
    # Per-box pixel/mm scale from the zone editor. Pure analysis metadata,
    # round-tripped so the dialog reopens with the calibration intact.
    scale_map = full_cfg.get("scale") or {}
    if isinstance(scale_map, dict):
        overrides["scale"] = {str(k): v for k, v in scale_map.items() if v}
    return overrides



#: Cache keyed by part count, so the colormap is sampled once per model rather
#: than per frame.
_PART_COLOUR_CACHE: dict = {}

#: Hue band the markers may use, in OpenCV's 0-179 scale. It starts past red
#: and stops before it comes back round, because red is the interface's alarm
#: colour: an error tile border, a "camera not connected" prompt, the stop
#: button. A keypoint drawn in it reads as a fault at a glance across a wall of
#: box tiles. 15 is orange, 165 is magenta, and everything between is fair game.
_HUE_MIN, _HUE_MAX = 15.0, 165.0


def part_colours(n_parts: int):
    """One distinct colour per part, spread across the usable hue band.

    DeepLabCut assigns body-part colours by sampling a colormap across the
    number of parts, and SLEAP likewise gives each node its own hue. Sampling
    keeps them distinct at any part count, where a fixed list indexed modulo
    its length drew two different parts in the same colour.

    Two rules the sweep has to respect. It must not wrap: running 0 to 170
    puts the first and last part at either end of red, which on six parts made
    snout and tail base near enough the same dot. And it must stay out of red
    altogether, which is what ``_HUE_MIN``/``_HUE_MAX`` do.

    Order still carries meaning: parts listed nose to tail run through the
    band in that order, so a glance says which end of the animal a dot is.

    A module-level function rather than a method: it depends on nothing but the
    part count, and both drawing paths plus the offline overlay need it.
    """
    n = max(1, int(n_parts))
    cached = _PART_COLOUR_CACHE.get(n)
    if cached is not None:
        return cached
    # Hue sweep done directly so the result does not depend on which colormaps
    # this OpenCV build ships. High saturation and value keep every part
    # readable on a dark arena floor.
    if n == 1:
        ramp = np.array([(_HUE_MIN + _HUE_MAX) / 2.0])
    else:
        ramp = np.linspace(_HUE_MIN, _HUE_MAX, n)
    ramp = ramp.astype(np.uint8).reshape(-1, 1, 1)
    hsv = np.concatenate([ramp, np.full_like(ramp, 235),
                          np.full_like(ramp, 255)], axis=2)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    colours = [tuple(int(c) for c in row) for row in bgr]
    _PART_COLOUR_CACHE[n] = colours
    return colours


def part_colour_map(names, saved: Optional[dict] = None) -> dict:
    """``{part name: BGR}``, keeping whatever a project already chose.

    Colour follows the NAME, not the position in the list. Indexing by
    position means adding one keypoint repaints every other part, so the
    snout is green in one session and cyan in the next and an operator who
    has learnt the overlay has to learn it again. A project stores this map
    and hands it back, so a part keeps its colour for the life of the project.

    A part with no stored colour takes the free hue furthest from the ones
    already in use, which keeps a late addition distinct from its neighbours
    instead of landing on top of one.
    """
    names = [str(n) for n in (names or [])]
    out = {n: tuple(int(c) for c in saved[n])
           for n in names if saved and n in saved and len(saved[n] or ()) == 3}
    missing = [n for n in names if n not in out]
    if not missing:
        return out

    def _hue(bgr):
        px = np.uint8([[list(bgr)]])
        return float(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0, 0])

    used = sorted(_hue(c) for c in out.values())
    fresh = list(part_colours(len(names)))
    for name in missing:
        if not used:
            out[name] = fresh.pop(0)
            used.append(_hue(out[name]))
            continue
        # Widest gap in the band, including the two ends: the middle of it is
        # as far from every taken hue as this band allows.
        edges = [_HUE_MIN] + used + [_HUE_MAX]
        gap_at, gap = 0, -1.0
        for a, b in zip(edges, edges[1:]):
            if b - a > gap:
                gap, gap_at = b - a, (a + b) / 2.0
        h = np.uint8([[[int(round(gap_at)), 235, 255]]])
        out[name] = tuple(int(c) for c in
                          cv2.cvtColor(h, cv2.COLOR_HSV2BGR)[0, 0])
        used = sorted(used + [gap_at])
    return out


class MainWindowBase(QtWidgets.QMainWindow,
                     MainWindowUtilsMixin,
                     PoseSubsystemMixin):
    """Shared MainWindow logic for both pyOperant and pyMaze.

    Subclasses must implement every hook in the *Hook surface* section.
    Concrete methods that live on this base call those hooks; nothing
    else is mode-aware.
    """

    #: How a live tile fills its cell:
    #:
    #:   ``stretch``  - resize the source to the tile and paint it edge to
    #:                  edge. A grid of sixteen chambers is read by glancing
    #:                  across it, so the height goes to the animals rather
    #:                  than to letterbox bars. Operant.
    #:   ``preserve`` - fit the source inside the tile and letterbox the rest,
    #:                  so a circular arena stays circular. Maze.
    #:
    #: A CLASS attribute, so a mode declaring its own value at class level
    #: actually gets it. This was assigned in ``__init__`` instead, and an
    #: instance attribute shadows the subclass's class attribute: maze declared
    #: ``preserve``, the base overwrote it with ``stretch`` on every window it
    #: built, and every arena has been stretched since.
    _tile_aspect_mode: str = "stretch"

    # Keypoint colors for pose annotation (BGR), cycled across body parts.
    #: Fallback palette, used only when the part count is unknown. The real
    #: colours come from :meth:`_part_colours`, which spreads a colormap across
    #: however many parts the model has.
    _DLC_COLORS = [
        (0, 255, 0),    # green
        (0, 165, 255),  # orange
        (255, 0, 0),    # blue
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (255, 255, 0),  # cyan
        (128, 0, 255),  # pink
        (0, 128, 255),  # gold
    ]

    @classmethod
    def _part_colours(cls, n_parts: int):
        """Deprecated alias; see :func:`part_colours`."""
        return part_colours(n_parts)

    # Zone outline colors (BGR), cycled across zones.
    _ZONE_COLORS = [
        (76, 175, 80),    # green
        (33, 150, 243),   # blue
        (156, 39, 176),   # purple
        (255, 87, 34),    # deep orange
        (255, 235, 59),   # yellow
        (0, 188, 212),    # cyan
    ]

    # ==================================================================
    # Hook surface, every subclass must implement these.
    # ==================================================================

    # ---- box-widget lookup -------------------------------------------
    # All helpers below read ONE base-owned mapping, ``_box_widgets``
    # ({setup_id -> widget}), created in ``_setup_pipeline``. Both modes
    # register/remove widgets in it directly, no per-mode dict, no shims.

    @property
    def video_segment_config(self):
        """The shared-camera ROI segment table, Pipeline-owned; this
        property keeps the many existing read/write sites working."""
        return self.pipeline.segment_config

    @video_segment_config.setter
    def video_segment_config(self, value):
        self.pipeline.set_segment_config(value)

    def _setup_widget_for(self, setup_id: int):
        """Return the per-box GUI widget (BoxControlWidget or SetupWidget)."""
        return self._box_widgets.get(setup_id)

    def _iter_box_ids(self) -> Iterable[int]:
        """Currently registered box ids, ascending."""
        return sorted(self._box_widgets.keys())

    def iter_box_widgets(self):
        """Yield ``(setup_id, widget)`` for every box. Host hook for the
        GUI <-> Config bridge."""
        for sid, widget in sorted(self._box_widgets.items()):
            yield int(sid), widget

    def iter_box_subject_widgets(self):
        """Yield ``(setup_id, subject_id_edit)`` for every box that has one.
        Host hook for MetadataManager."""
        for sid, widget in sorted(self._box_widgets.items()):
            edit = getattr(widget, "subject_id_edit", None)
            if edit is None:
                continue
            yield int(sid), edit

    # ---- per-box status / video display ------------------------------
    # All bodies delegate through the per-box-widget protocol exposed by
    # RunTask + BoxControlWidget + SetupWidget (append_status, clear_video,
    # update_frame, video_size, set_fps_text, get_roi, camera_id_text,
    # notify_camera_starting, notify_camera_disconnected). Mode subclasses
    # only override the helpers that touch MainWindow-side state (track
    # button, recording_boxes, cache dicts).

    # Void delegations catch (AttributeError, RuntimeError) only; those are
    # the two failure modes that actually happen (missing method on a custom
    # box widget; Qt widget already destroyed). Value-returning delegations
    # swallow any Exception and return ``default`` so callers never get a raise
    # from a degraded widget.
    def _safe_box_call(self, setup_id, method, *args,
                       default=None, exc=(AttributeError, RuntimeError)):
        bw = self._setup_widget_for(setup_id)
        if bw is None:
            return default
        try:
            return getattr(bw, method)(*args)
        except exc:
            return default

    def _box_status_say(self, setup_id: int, msg: str) -> None:
        self._safe_box_call(setup_id, 'append_status', msg)

    def _box_video_clear(self, setup_id: int) -> None:
        self._safe_box_call(setup_id, 'clear_video')

    def _box_camera_id_text(self, setup_id: int) -> str:
        return self._safe_box_call(setup_id, 'camera_id_text', default="", exc=Exception)

    def _box_camera_started_ui(self, setup_id: int, camera_id) -> None:
        self._safe_box_call(setup_id, 'notify_camera_starting', camera_id)

    def _box_camera_failed_ui(self, setup_id: int, err: str) -> None:
        # Log the failure via the box's status surface. Mode subclasses
        # override if they need to also block further UI.
        self._box_status_say(setup_id, err)

    def _box_camera_disconnected_cleanup(self, setup_id: int) -> None:
        self._safe_box_call(setup_id, 'notify_camera_disconnected')
        # Drop per-box transient state so a long session with box cycling
        # (connect → disconnect → reconnect with a new id) doesn't leak
        # one OverlayState / display-version / zone-layer entry per
        # cycle. These dicts are tiny but unbounded in principle.
        # Every per-box dict the BASE owns is cleared here; a mode
        # override adds only its own dicts.
        for attr in ("_overlay", "_display_last_version",
                     "_display_last_emit_ns", "_zone_layer_cache",
                     "_zone_version",
                     "_display_frame_counts", "_fps_status_times"):
            d = getattr(self, attr, None)
            if isinstance(d, dict):
                d.pop(setup_id, None)
        # If this was the last streaming camera and no box is running,
        # process_timer can stop. _sync_timer_mode is idempotent.
        try:
            self._sync_timer_mode()
        except Exception:
            pass

    def _camera_streaming_ready(self, setup_id: int) -> None:
        """Polling confirmed the camera is streaming. Default: tell the box.
        Operant overrides to also enable its global Track button."""
        self._box_status_say(setup_id, "Camera connected")
        # A streaming camera makes the GUI 'active' even when no box is
        # running, _sync_timer_mode starts process_timer so frames get
        # painted (refresh_timer keeps running alongside it).
        try:
            self._sync_timer_mode()
        except Exception:
            pass
        # A live camera is the ONE thing pose init was waiting for, so this is
        # where readiness is triggered, for every route that opens a camera:
        # auto-connect on load, the Camera Connect dialog, or a reconnect.
        try:
            self.schedule_pose_ready()
        except Exception as e:
            logger.debug("Box %s: pose readiness schedule: %s", setup_id, e)

    def _apply_box_timer(self, bw, txt: str) -> None:
        """Write the per-box elapsed clock to EVERY surface at once, box card
        label, Live Status timer, and (via ``_stamp_box_timer``) the stats
        table + camera tile. THE single path, used by the 10 ms process tick
        AND by the start-reset / stop-freeze, so the surfaces can never drift
        apart at the run boundaries. setText is text-gated so an unchanged
        value doesn't trigger a needless Qt repaint."""
        tl = getattr(bw, "_timer_label", None)
        if tl is not None:
            try:
                if tl.text() != txt:
                    tl.setText(txt)
            except (AttributeError, RuntimeError):
                pass
        ls_by_box = getattr(self, "_live_status_by_box", None)
        if ls_by_box:
            ls = ls_by_box.get(getattr(bw, "setup_number", None))
            lsl = getattr(ls, "timerLabel", None) if ls is not None else None
            if lsl is not None:
                try:
                    if lsl.text() != txt:
                        lsl.setText(txt)
                except (AttributeError, RuntimeError):
                    pass
        try:
            self._stamp_box_timer(bw, txt)
        except Exception:
            pass

    def _stamp_box_timer(self, bw, txt: str) -> None:
        """Hook: stamp the per-box elapsed clock onto mode-specific surfaces
        beyond the box card + Live Status (operant: stats table Timer column
        + camera tile header). Default no-op, maze has neither."""
        return

    def _camera_streaming_timeout(self, setup_id: int) -> None:
        """Polling exceeded max_attempts. Default: tell the box. Operant
        overrides to also enable Track with a 'slow start' tooltip.

        Even on timeout the camera may eventually produce frames, so
        sync the timer mode, if the underlying CameraThread is
        ``connected``, process_timer should be on to paint them.
        """
        self._box_status_say(setup_id, "Camera connected (slow start)")
        try:
            self._sync_timer_mode()
        except Exception:
            pass

    # ---- frame display -----------------------------------------------

    def _box_render_pixmap(self, setup_id: int, pixmap) -> None:
        self._safe_box_call(setup_id, 'update_frame', pixmap)

    def _box_video_label_size(self, setup_id: int):
        return self._safe_box_call(setup_id, 'video_size', exc=Exception)

    def _is_box_tile_visible(self, setup_id: int) -> bool:
        """Is this box's video tile actually on screen?

        Qt reports a widget on a non-current tab (or under a minimised window)
        as not visible, which is what makes this worth asking: the video grid
        is a tab, so an operator on Live Status was paying the full paint cost
        for every tile with nothing to show for it.

        Fails OPEN. A widget that does not implement ``video_visible``, or a
        lookup that raises, paints as before, a broken check must never be
        the reason a tile goes dark.
        """
        visible = self._safe_box_call(setup_id, 'video_visible', exc=Exception)
        return True if visible is None else bool(visible)

    def _box_set_fps_text(self, setup_id: int, text: str, tooltip: str = "") -> None:
        self._safe_box_call(setup_id, 'set_fps_text', text, tooltip)

    # ---- ROI lookup --------------------------------------------------

    def _box_roi_lookup(self, setup_id: int, frame=None):
        return self._safe_box_call(setup_id, 'get_roi', frame, exc=Exception)

    # ---- lifecycle event extras --------------------------------------

    def _close_event_extras(self, event) -> None:
        """Mode-specific cleanup during closeEvent (after pipeline shutdown).
        Default is no-op; mode subclasses override only if they need to."""
        return

    def _resize_event_extras(self, event) -> None:
        """Default: keep every registered sidebar at full central-widget
        height. Subclass may override and call super() for additional
        per-mode resize actions."""
        self._sync_sidebar_heights()

    def _move_event_extras(self, event) -> None:
        """Default: same height-sync as resize (covers monitor changes
        and quick window-snap moves)."""
        self._sync_sidebar_heights()

    # ==================================================================
    # Two timers with DISJOINT jobs. They are NOT mutually exclusive:
    #
    #   refresh_timer  (1 Hz)     ALWAYS running, housekeeping.
    #                             jobs:  tasks/ folder rescan + idle-button
    #                                    gating + per-box on-disk task/HD
    #                                    staleness poll, skipping boxes that
    #                                    are THEMSELVES running. Host-only and
    #                                    cheap (stat + mtime-gated djb2), so it
    #                                    keeps running while cameras stream and
    #                                    other boxes run. This is what flips an
    #                                    idle box's Reset→Upload within ~1 s.
    #
    #   process_timer  (10 ms)    runs ONLY when there is realtime work
    #                             (≥1 box running OR ≥1 camera streaming).
    #                             jobs:  per running box → tick_active() +
    #                                    1 Hz HH:MM:SS clock; per streaming
    #                                    camera → pull latest frame + paint.
    #
    # The staleness poll is gated PER BOX (this box running?), never globally:
    # with 4-8 boxes, box 1 must still flip to Upload while box 2 runs and a
    # camera streams, so refresh_timer stays always-on.
    #
    # _autosave_timer (debounced singleShot) is orthogonal and lives below.
    #
    # ``_sync_timer_mode()`` is the single switch for process_timer, called
    # from every site that flips a box's framework_running or a camera's
    # connected flag (mcu_start, mcu_stop, mcu_disconnect, camera connect/
    # disconnect). refresh_timer is started once and never stopped until close.
    # ==================================================================

    REFRESH_INTERVAL_MS = 1000          # 1 Hz housekeeping / UI refresh
    PROCESS_INTERVAL_MS = 10            # realtime tick (MCU drain + clocks)
    # Queued host→MCU writes (coords, triggers, per-frame events) leave on this
    # tick, so the tick period IS the delay on every one of them: uniform
    # 0-PROCESS_INTERVAL_MS, mean half of it. Measured against the board's own
    # clock, the ping/pong round trip sat at 16 ms p50 with a 10 ms tick.
    # 3 ms while a box is running closed-loop cuts that to ~1.5 ms mean and,
    # more importantly, removes the jitter, for the cost of ~330 extra
    # wake-ups per second on a thread that is already awake for the camera.
    # Pycboard stays single-threaded (see its _drain_pending_writes): this
    # changes WHEN the one writer runs, never how many writers there are.
    CLOSED_LOOP_INTERVAL_MS = 3
    # Tile paint rate; ``cfg.display.max_fps`` overrides. 30 because that is
    # what the cameras run at, a cap below the capture rate throws away
    # frames the operator can see, and 15 was visibly juddery on a live wall.
    DISPLAY_FPS_DEFAULT = 30
    # On-disk task/HD hashes are polled every 1 Hz housekeeping tick so an
    # external edit of the task .py / HD flips Reset→Upload within ~1 s even
    # while cameras stream. Cheap: _check_task_consistency gates the djb2 on
    # the file's mtime, so the hash only runs when the file actually changed.

    def _tasks_dir(self) -> Path:
        return Path(__file__).resolve().parents[2] / "tasks"

    def _scan_task_files(self):
        """Snapshot of (relpath, mtime) for every .py task, same value
        means no change, so we don't waste work rebuilding menus."""
        tasks_dir = self._tasks_dir()
        out = set()
        if not tasks_dir.is_dir():
            return out
        try:
            for p in tasks_dir.rglob("*.py"):
                rel = p.relative_to(tasks_dir)
                if any(part.startswith(("__", ".")) for part in rel.parts):
                    continue
                try:
                    out.add((str(rel), p.stat().st_mtime))
                except OSError:
                    pass
        except OSError:
            pass
        return out

    def _start_refresh_timer(self) -> None:
        """Create both timers once and start the 1 Hz refresh_timer (it then
        runs continuously). process_timer is started on demand by
        ``_sync_timer_mode``. Idempotent. Each subclass __init__ calls this
        once after ``_setup_pipeline``."""
        if getattr(self, "refresh_timer", None) is not None:
            return
        self._available_task_files = self._scan_task_files()
        self._refresh_tick_count = 0
        self._timer_mode = "idle"
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self._on_refresh_tick)
        self.process_timer = QtCore.QTimer(self)
        self.process_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.process_timer.timeout.connect(self._on_process_tick)
        # Display runs on its OWN clock, not the 10 ms realtime tick. Painting
        # is the only consumer allowed to skip, so it must not be able to make
        # the MCU drain late by sharing a deadline with it, and nobody can
        # read more than ~15 fps across a wall of tiles anyway.
        self.display_timer = QtCore.QTimer(self)
        # PreciseTimer, like the realtime tick. Qt's default is Coarse, which
        # on Windows quantises to the ~15.6 ms system tick: a 33 ms request
        # becomes 46.8 ms (21 Hz) and a 67 ms request becomes 78 ms (12.8 Hz).
        # The paint rate would silently be a third below whatever was asked
        # for, and would not even be the same third on another platform.
        self.display_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.display_timer.timeout.connect(self._on_display_tick)
        self.refresh_timer.start(self.REFRESH_INTERVAL_MS)

    def _any_box_running(self) -> bool:
        return any(getattr(bw, "framework_running", False)
                   for bw in self.get_all_setup_widgets())

    def _any_camera_streaming(self) -> bool:
        vm = getattr(self, "video_manager", None)
        if vm is None or not getattr(vm, "box_camera_map", None):
            return False
        for cam_id in vm.box_camera_map.values():
            ct = vm.cameras.get(cam_id)
            if ct is not None and getattr(ct, "connected", False):
                return True
        return False

    def _sync_timer_mode(self) -> None:
        """Start/stop ONLY the 10 ms realtime tick, based on whether there is
        realtime work (a running box or a streaming camera). The 1 Hz
        housekeeping tick (refresh_timer) runs continuously in both modes and
        is never stopped here; it does the per-box on-disk staleness poll
        (skipping running boxes) + idle-button gating, which must keep working
        while cameras stream / other boxes run (else a box's Reset→Upload flip
        only lands on the next user input while a camera is up). Idempotent.
        Called from per-box mcu_start / mcu_stop / mcu_disconnect and from
        camera connect / disconnect."""
        if getattr(self, "process_timer", None) is None:
            return                       # timers not initialised yet
        want_active = self._any_box_running() or self._any_camera_streaming()
        # A running box means queued events are in flight to the MCU, and the
        # tick period is their delay. A streaming camera alone only needs the
        # frame pull, which the 10 ms tick serves fine.
        interval = (self.CLOSED_LOOP_INTERVAL_MS if self._any_box_running()
                    else self.PROCESS_INTERVAL_MS)
        if want_active and (self._timer_mode != "active"
                            or self.process_timer.interval() != interval):
            self.process_timer.start(interval)
            self._timer_mode = "active"
        elif not want_active and self._timer_mode != "idle":
            self.process_timer.stop()
            self._timer_mode = "idle"
        # Display only has work while a camera is streaming, a running box
        # with no camera has nothing to paint.
        dt = getattr(self, "display_timer", None)
        if dt is not None:
            streaming = self._any_camera_streaming()
            if streaming and not dt.isActive():
                dt.start(self._display_interval_ms())
            elif not streaming and dt.isActive():
                dt.stop()

    def _display_interval_ms(self) -> int:
        """Paint period from ``cfg.display.max_fps``.

        Defaults to DISPLAY_FPS_DEFAULT rather than the camera rate: past
        roughly 15 fps a wall of tiles conveys nothing extra to a human, while
        every frame beyond it costs the same CPU as one that matters. Capture,
        recording and tracking are untouched by this, it caps painting only.
        """
        fps = getattr(self, "_display_max_fps", None) or self.DISPLAY_FPS_DEFAULT
        return max(1, int(round(1000.0 / float(fps))))

    def _on_display_tick(self) -> None:
        """Paint whatever is current. Never queues, never catches up."""
        try:
            self._paint_streaming_cameras_once()
        except Exception as e:
            logger.debug("display tick error: %s", e)

    def _on_refresh_tick(self) -> None:
        """1 Hz housekeeping tick, runs CONTINUOUSLY (both idle and active
        mode): rescan tasks/ folder; re-gate idle-only buttons; poll on-disk
        task/HD hashes on every connected box that isn't itself running.

        Everything here is host-only and cheap, and the per-box hash poll
        skips running boxes, so running this while ``process_timer`` also
        ticks (cameras streaming / other boxes running) is safe. The
        idle-only button gating (``_idle_button_refresh`` →
        ``update_metadata_button_states``) gates on ``any_box_running``
        itself, so it stays correct in active mode too."""
        self._refresh_tick_count += 1
        try:
            # With no boxes there is no task_combo to feed, a full
            # rglob+stat of tasks/ every second is wasted then; scan every
            # 10th tick until a box exists.
            if self._box_widgets or self._refresh_tick_count % 10 == 0:
                self._refresh_task_folders()
        except Exception as e:
            logger.debug("refresh tick folder error: %s", e)
        # Re-gate idle-only buttons every tick (cheap; subclass no-op by
        # default). Operant uses this to re-enable Clear Meta once idle.
        try:
            self._idle_button_refresh()
        except Exception as e:
            logger.debug("idle button refresh: %s", e)
        for bw in self.get_all_setup_widgets():
            if getattr(bw, "framework_running", False):
                continue
            check = getattr(bw, "_check_task_consistency", None)
            if callable(check):
                try:
                    check()
                except Exception as e:
                    logger.debug("hash poll box %s: %s",
                                 getattr(bw, "setup_number", "?"), e)
        try:
            self._refresh_idle_backgrounds()
        except Exception as e:
            logger.debug("idle background refresh: %s", e)

    # Minimum gap between automatic background refreshes for one box. Long
    # enough that this is a between-trials housekeeping job, not something
    # that fires while the operator is still setting the box up.
    BG_IDLE_REFRESH_SECONDS = 15 * 60

    def _refresh_idle_backgrounds(self) -> None:
        """Re-take a stale background while its box is doing nothing.

        This is what keeps differencing honest over a long day: a feeder gets
        moved, a lamp drifts, someone nudges the camera, and a reference from
        this morning slowly stops describing the arena. Recapturing between
        trials fixes that without anyone remembering to.

        Deliberately narrow. It runs only when the box is idle, the camera is
        live, the existing background is already past the staleness bound, and
        nothing has been recaptured for this box recently. It never touches a
        running box, and never replaces a background that is still fresh,
        an operator who tuned against a specific reference keeps it.
        """
        if not getattr(self, "_active_config", None):
            return
        stale = set(self._stale_background_boxes())
        if not stale:
            return
        now = time.monotonic_ns()
        for bw in self.get_all_setup_widgets():
            setup_id = getattr(bw, "setup_number", None)
            if setup_id is None or setup_id not in stale:
                continue
            if getattr(bw, "framework_running", False):
                continue
            last = self._bg_idle_refresh_ns.get(setup_id)
            if last is not None and (now - last) < self.BG_IDLE_REFRESH_SECONDS * 1e9:
                continue
            if getattr(self, "video_manager", None) is None:
                continue
            try:
                if self.video_manager.get_last_frame(setup_id) is None:
                    continue          # camera not delivering; nothing to take
            except Exception:
                continue
            self._bg_idle_refresh_ns[setup_id] = now
            logger.info("Box %s: background is stale and the box is idle, "
                        "recapturing", setup_id)
            self._capture_background_for_box(setup_id)

    def _idle_button_refresh(self) -> None:
        """Subclass hook, re-evaluate idle-only button gating on the 1 Hz
        refresh tick. Default no-op (maze has no idle-gated master buttons
        that need per-second refresh)."""
        return None

    def _on_process_tick(self) -> None:
        """10 ms active tick: drain MCU + update plots + advance clock
        for every running box; paint latest frame for every streaming
        camera. Mode-agnostic; no subclass overrides.

        The HH:MM:SS string is computed once per running box and
        stamped onto:
          * ``bw._timer_label``: the box-card label.
          * the matching ``LiveStatusWidget.timerLabel`` if operant
            mirrors it (maze has no LiveStatusWidget, so the lookup
            short-circuits on an empty list).
        ``self._live_status_by_box`` is a dict[box_number → widget]
        operant builds once when it creates the LiveStatus tab; the
        central tick only does an O(1) lookup per running box.

        The clock is the MCU framework time (``pycboard.get_timestamp()``):
        the SAME value written into the TSV, re-anchored on every MCU
        message + monotonic-interpolated between them. No host wall clock
        is involved, so the display can never drift from the data file.
        """
        for bw in self.get_all_setup_widgets():
            if not getattr(bw, "framework_running", False):
                continue
            try:
                bw.tick_active()
            except Exception as e:
                logger.debug("process tick box %s: %s",
                             getattr(bw, "setup_number", "?"), e)
            # tick_active() may have auto-stopped this box (MCU sent EOR /
            # board error). mcu_stop already froze the label at the exact
            # final fw timestamp, don't overwrite it with one more update.
            if not getattr(bw, "framework_running", False):
                continue
            # HH:MM:SS from MCU framework ms. setText is gated by a text
            # compare so the actual Qt repaint only fires when the second
            # flips.
            pyc = getattr(bw, "pycboard", None)
            if pyc is None:
                continue
            try:
                ts = pyc.get_timestamp()
                txt = format_run_clock(ts)
            except Exception:
                continue
            # Per-state elapsed (maze Live Status "dur"). Same fw clock as
            # the session timer, so the two can't drift. No-op for boxes
            # whose setup widget has no state-duration label (operant).
            tsd = getattr(bw, "tick_state_duration", None)
            if callable(tsd):
                try:
                    tsd(ts)
                except Exception:
                    pass
            # ONE fan-out to every timer surface (box card + Live Status +
            # stats table + camera tile). Also used by the start-reset and
            # stop-freeze so all surfaces always agree.
            self._apply_box_timer(bw, txt)
        # Painting is NOT done here; it has its own timer (display_timer), so
        # a slow repaint can no longer delay the MCU drain above it.
        # State may have flipped during the tick (auto-stop, camera
        # disconnect). Re-evaluate so the next tick is from the right
        # timer.
        self._sync_timer_mode()

    def _refresh_task_folders(self) -> None:
        """Timer callback: rescan tasks/, push update_menu to every box's
        ``task_combo`` if anything changed.  Cheap when nothing changed,
        the equality check skips the per-box update_menu calls."""
        try:
            current = self._scan_task_files()
        except Exception:
            return
        if current == getattr(self, "_available_task_files", None):
            return
        self._available_task_files = current
        tasks_dir = str(self._tasks_dir())
        try:
            box_ids = list(self._iter_box_ids())
        except NotImplementedError:
            return
        for setup_id in box_ids:
            try:
                bw = self._setup_widget_for(setup_id)
            except NotImplementedError:
                continue
            if bw is None:
                continue
            combo = getattr(bw, "task_combo", None)
            if combo is None or not hasattr(combo, "update_menu"):
                continue
            try:
                combo.update_menu(tasks_dir)
            except (AttributeError, RuntimeError) as e:
                logger.debug("task_combo refresh for box %s: %s", setup_id, e)

    # ==================================================================
    # Concrete shared behavior, pipeline + camera lifecycle
    # ==================================================================

    def _setup_pipeline(self, *, target_fps: int = 30) -> None:
        """Construct the unified Pipeline + QtBridge and wire bridge signals.

        Subclasses call this once from ``__init__`` (after ``super().__init__()``).
        After this returns, ``self.pipeline`` is the only thing the subclass
        ever needs to talk to for camera / recording / tracking / pose / push.
        ``self.video_manager`` is exposed as a thin alias for code (mostly
        dialogs) that still talks directly to the camera registry.
        """
        from source.video.framebus import Pipeline, QtBridge
        from source.communication.controller import MCUController

        self.mcu = MCUController(self)
        # ``video_segment_config`` is a property over the Pipeline's owned
        # table (defined on the class); subclasses must NOT re-init it.
        # THE per-box widget registry ({setup_id -> BoxControlWidget /
        # SetupWidget}). Single source for every box lookup in base,
        # dialogs and both modes.
        self._box_widgets: dict = {}
        # {setup_id: run_id} for runs opened this session (runs JSON rows).
        self._open_runs: dict = {}
        self.pipeline = Pipeline(target_fps=target_fps)
        self.bridge = QtBridge(self.pipeline, parent=self)
        # Alias so existing dialogs / widgets that call self.video_manager
        # keep working, VideoManager is now slim (just camera lifecycle).
        self.video_manager = self.pipeline.video_manager
        # Wire bridge signals to base slots (queued by default → main thread).
        self.bridge.pose_ready.connect(self._on_box_pose)
        self.bridge.tracker_ready.connect(self._on_box_tracker)
        self.bridge.health.connect(self._on_box_health)
        self.bridge.pose_failed.connect(self._on_pose_failed)
        self.bridge.trigger_ready.connect(self._on_box_triggers)
        # ── Display state (no standalone timer) ─────────────────────────
        # Per-tile last drawn version / emit timestamp / optional rate
        # cap. Frames are pulled + painted by ``process_timer`` via
        # ``_paint_streaming_cameras_once``; see the timer pipeline
        # block above. At ~100 Hz tick rate the overhead is microseconds
        # per camera (a lock + version compare) when nothing changed;
        # the actual paint runs only when ``last_version`` advanced
        # past what we last drew. ``cfg.display.max_fps`` caps the
        # per-tile paint rate without affecting capture.
        self._display_last_version: dict = {}    # box_id -> last drawn version
        self._display_last_emit_ns: dict = {}    # box_id -> monotonic_ns of last paint
        # box_id -> message, set when a loaded background does not match the
        # camera's current view. Blocks that box from starting blob tracking.
        self._bg_shape_problem: dict = {}
        # box_id -> monotonic_ns of the last idle background refresh, so the
        # refresh cannot re-fire every tick.
        self._bg_idle_refresh_ns: dict = {}
        # Tile-preparation pool. Sized to leave cores for the sinks, which do
        # the work that must not be late (recording, tracking); display is the
        # layer allowed to fall behind, so it gets what is left over and never
        # the whole machine. One box is done inline, a pool round-trip costs
        # more than the work.
        from concurrent.futures import ThreadPoolExecutor
        _cores = os.cpu_count() or 4
        self._tile_pool = ThreadPoolExecutor(
            max_workers=max(2, min(6, _cores - 2)),
            thread_name_prefix="tile")
        from source.config.settings import get_setting as _get_setting
        _max_fps = _get_setting("display", "max_fps")
        # The display TIMER is the only pacer. There was briefly a second,
        # per-tile gate at the same rate, measured on camera timestamps while
        # the timer ran on Qt's clock, two 15 Hz gates that were not phase
        # locked, so a tick whose newest frame was a few ms "too fresh" got
        # rejected and the tile painted at a beat frequency well under either.
        # One gate, one clock.
        self._display_max_fps = (float(_max_fps)
                                 if isinstance(_max_fps, (int, float)) and _max_fps > 0
                                 else float(self.DISPLAY_FPS_DEFAULT))
        self._display_min_interval_ns: Optional[int] = None
        # Per-box overlay state, ONE dataclass per box, shared by the
        # display renderer + the zone editor + result fan-out.  Both pose
        # and blob result callbacks write fields on the same object via
        # ``_on_overlay_update``; the renderer reads one object and uses
        # ``last_seen_ns`` for the TTL freshness gate.
        self._overlay: dict = {}  # box_id -> OverlayState
        # Global dialog-only state, fields the unified tracking dialog
        # owns that don't live per-box on ``TrackingConfig`` (coord_mapping,
        # triggers, annotate_parts, zone_body_part, marker_size,
        # annotate_saved per-box flags, task_name).  Kept compact; the
        # per-box source of truth is ``pipeline.get_tracking_config``.
        self._tracking_dialog_globals: dict = {}
        # Cached zone QPixmap layer per box.
        # Zones don't change frame-to-frame, so we render them ONCE into a
        # transparent QPixmap and composite via QPainter onto each live
        # tile.  Eliminates per-frame cv2 zone burn-in on the main thread.
        # Key: (zone_version, fit_w, fit_h).  Invalidated by
        # _invalidate_zone_layer() whenever zones mutate.
        self._zone_layer_cache: dict = {}    # box_id -> (key_tuple, QPixmap)
        self._zone_version: dict = {}        # box_id -> int (bumped on any mutation)

    # _pose_active and _resolve_grayscale_for_camera live in
    # source/gui/pose_subsystem.py (PoseSubsystemMixin). MainWindowBase
    # inherits the mixin so ``self._pose_active()`` etc. still work.

    def connect_camera(self, setup_id, camera_backend: Optional[str] = None,
                      camera_config: Optional[dict] = None) -> None:
        """Start a camera for one box via the unified Pipeline."""
        try:
            camera_id_text = self._box_camera_id_text(setup_id)
            if not camera_id_text:
                return

            # Resolve the backend: an explicit arg (Camera Connect dialog)
            # wins; otherwise fall back to the camera's saved backend,
            # restored into the pipeline config on project load and keyed
            # by the camera-id string. This lets auto-connect honour
            # Spinnaker / Ximea rigs instead of forcing opencv.
            if camera_backend is None:
                try:
                    cam_cfg = self.pipeline.get_camera_config(camera_id_text)
                    camera_backend = (
                        getattr(cam_cfg, "camera_backend", None) or "opencv")
                except (AttributeError, RuntimeError):
                    camera_backend = "opencv"

            # An OpenCV camera id is an IDENTITY ("fp3557d2de"), resolved to a
            # live index only when the device is opened. A NUMERIC id stays an
            # int: ids are dict keys (box_camera_map, the bus registry), and a
            # mix of 0 and "0" makes those lookups miss.
            camera_id = camera_id_text
            if isinstance(camera_id, str) and camera_id.strip().isdigit():
                camera_id = int(camera_id.strip())
            if camera_backend == "opencv" and not str(camera_id_text).strip():
                raise ValueError("Please enter a camera ID")

            cfg = dict(camera_config) if camera_config else {}
            requested_gray = cfg.get("grayscale", getattr(self, "video_grayscale", False))
            cfg["grayscale"] = self._resolve_grayscale_for_camera(
                requested_gray, setup_id=setup_id)

            self.pipeline.register_box(setup_id)
            ok = self.pipeline.connect_camera(
                camera_id, setup_id,
                grayscale=cfg.get("grayscale", False),
                camera_backend=camera_backend, camera_config=cfg,
            )
            if ok:
                self._box_camera_started_ui(setup_id, camera_id)
                logger.info(
                    "Camera %s (backend=%s) starting for box %s",
                    camera_id, camera_backend, setup_id)
                self._wait_for_camera_streaming(setup_id)
            else:
                self._box_camera_failed_ui(setup_id, "Failed to start camera")
        except Exception as e:
            logger.error("Camera connection error for box %s: %s", setup_id, e)
            try:
                self.showError(f"Camera connection error: {e}")
            except (AttributeError, RuntimeError):
                pass
        finally:
            try:
                self.refresh_ui_state()
            except (AttributeError, RuntimeError):
                pass

    def connect_all_cameras(self):
        """Connect every configured box's camera. Shared by both modes
        (the toolbar bulk action and the auto-connect-on-load hook).

        Boxes are grouped by camera id so each physical camera opens
        exactly ONCE, multi-box rigs share one camera via per-box
        segmentation. Connecting per box re-opened and re-settled the same
        device once per box, blocking the GUI for the whole settle window ×
        N (the symptom: a long freeze with N boxes on one camera). The
        manual Camera Connect dialog has always deduped this way; this
        mirrors it."""
        try:
            groups: dict = {}
            order: list = []
            for setup_id in self._iter_box_ids():
                cid_text = self._box_camera_id_text(setup_id)
                if not cid_text:
                    continue
                if cid_text not in groups:
                    groups[cid_text] = []
                    order.append(cid_text)
                groups[cid_text].append(setup_id)

            # Hardware sync: start secondaries (armed, waiting on their trigger)
            # before the primary begins free-running and strobing, so no opening
            # edges are missed. No-op unless cameras carry a sync_role.
            from source.video.cameras.capture import plan_sync_start_order

            def _role(cid_text):
                cfg = self.pipeline.get_camera_config(cid_text) if self.pipeline else None
                return getattr(cfg, "sync_role", "none") if cfg else "none"

            order = plan_sync_start_order(order, _role)

            vm = getattr(self, "video_manager", None)
            n = 0
            for cid_text in order:
                box_ids = groups[cid_text]
                first = box_ids[0]
                self.connect_camera(first)        # opens the device once
                # The resolved device id (int for opencv) is recorded on the
                # video manager's box→camera map by start_camera. Reuse it to
                # bind the remaining boxes to the SAME bus without re-opening.
                camera_id = (vm.box_camera_map.get(first)
                             if vm is not None else None)
                if camera_id is None:
                    continue                       # first box failed; skip group
                n += 1
                for extra in box_ids[1:]:
                    try:
                        self._bind_box_to_shared_camera(extra, camera_id)
                        n += 1
                    except Exception as e:
                        logger.error("bind box %s to shared camera %s: %s",
                                     extra, camera_id, e)
            self.refresh_ui_state()
            logger.info("Connected %d camera(s)", n)
        except Exception as e:
            logger.error("Failed to connect cameras: %s", e)
            self.showError(f"Failed to connect cameras: {e}")

    def _bind_box_to_shared_camera(self, setup_id, camera_id) -> None:
        """Register a box onto an already-connected camera's FrameBus
        (shared-camera segmentation) WITHOUT re-opening the device. Single
        owner of the non-first-box bind, both auto-connect
        (``connect_all_cameras``) and the Camera Connect dialog route here."""
        if not self.pipeline.bind_box_to_camera(setup_id, camera_id):
            logger.warning("Box %s: camera %s is not open, cannot bind",
                           setup_id, camera_id)
            return
        # Reflect the camera id in the box widget so a shared box doesn't
        # look unconnected (and set_camera_pending doesn't flag it red).
        # notify_camera_starting only updates the status line, so set the
        # id field explicitly (matching the dialog's bind).
        bw = self._setup_widget_for(setup_id)
        if bw is not None and hasattr(bw, "camera_id_edit"):
            try:
                bw.camera_id_edit.setText(str(camera_id))
            except (AttributeError, RuntimeError):
                pass
        self._box_camera_started_ui(setup_id, camera_id)
        # The device is ALREADY streaming for the first box, so there is
        # nothing to wait for, but the readiness hook is what writes
        # "Camera connected" into the box's Live Status log and arms the
        # mode's post-connect UI. Skipping it left every shared box after the
        # first silently statusless, even though its Cam field was filled in
        # above. Fire it directly rather than starting a poll that would
        # succeed on its first tick.
        try:
            self._camera_streaming_ready(setup_id)
        except (AttributeError, RuntimeError) as e:
            logger.debug("streaming-ready hook for shared box %s: %s",
                         setup_id, e)
        logger.info("Box %s bound to shared camera %s", setup_id, camera_id)

    # ------------------------------------------------------------------
    # Post-load auto-connect (cameras + pose init + MCU), shared by both
    # modes. The video pipeline is identical across operant and maze, so
    # this single sequence drives both; neither mode keeps a parallel copy.
    # ------------------------------------------------------------------
    def _auto_flow_after_load(self, cfg) -> None:
        """Central post-load auto-connect sequence.

        Camera auto-connect is gated on the ``auto_connect_cameras_on_load``
        preference (the Camera Connect dialog checkbox). Order, with delays so
        each step settles before the next consumes it:

          1. Cameras, ``connect_all_cameras`` (deduped: each physical
             camera opens once, extra boxes bind as shared; works for the
             one-camera-many-setups case in BOTH modes).
          2. Pose, ``schedule_pose_ready`` prepares every configured box once
             its camera streams. Blob needs no model.
          3. MCU, connect each box's board, preferring the stable USB
             serial (replug-immune), falling back to the COM field.

        Step 2 runs whether or not the cameras were auto-connected: readiness
        follows the CAMERA, so a project opened with auto-connect off still
        prepares its models the moment the operator connects by hand. It used
        to hang off this preference, which left those rigs loading their model
        at the first Record click.
        """
        if not getattr(cfg.meta, "auto_connect_cameras_on_load", False):
            self.schedule_pose_ready(delay_ms=3000)
            return
        box_ids = list(self._iter_box_ids())
        if not box_ids:
            return

        # 1. Cameras, one deduped bulk connect (not per-box re-open).
        logger.info("Auto-connecting cameras (project preference)")
        try:
            self.connect_all_cameras()
        except Exception as e:
            logger.warning("Auto-connect cameras failed: %s", e)

        # 2. Pose, prepare every configured box once the cameras stream.
        self.schedule_pose_ready(delay_ms=3000)

        # 3. MCU, staggered per-box connect after the camera batch.
        for idx, sid in enumerate(box_ids):
            sw = self._setup_widget_for(sid)
            if sw is None:
                continue
            port = (getattr(sw, "_mcu_serial", "") or "").strip()
            if not port and hasattr(sw, "com_id_edit"):
                port = sw.com_id_edit.text().strip()
            if not port:
                continue
            QtCore.QTimer.singleShot(
                1500 + 500 * idx,
                lambda s=sid, p=port: self._auto_connect_mcu(s, p))

    def _auto_connect_mcu(self, setup_id: int, port: str) -> None:
        """Connect one box's MCU by id; no-op if the widget went away."""
        sw = self._setup_widget_for(setup_id)
        if sw is not None and hasattr(sw, "connect_mcu"):
            try:
                sw.connect_mcu(port)
            except Exception as e:
                logger.warning("Box %s: auto MCU connect failed: %s",
                               setup_id, e)

    # ------------------------------------------------------------------
    # Pose readiness, the one place a model is prepared
    # ------------------------------------------------------------------
    #
    # A box that asks for pose is made ready as soon as its camera streams,
    # whatever started the camera: the project's auto-connect, the Camera
    # Connect dialog, or a reconnect. Nothing else loads a model, and a run
    # start only switches inference on. The alternative, which is what this
    # replaced, was to load at the moment the operator clicks Record, with the
    # animal in the box and the clock running.

    def schedule_pose_ready(self, *, delay_ms: int = 1500) -> None:
        """Ask for a readiness pass shortly, coalescing repeat requests.

        Several boxes finishing their camera open in the same second must
        produce ONE pass, not one per box: the model is shared, and a second
        pass while the first is still loading would queue a duplicate load.
        """
        if getattr(self, "_pose_ready_pending", False):
            return
        if not self.pose_configured_boxes():
            return
        self._pose_ready_pending = True
        self._pose_ready_attempts = 0
        QtCore.QTimer.singleShot(max(0, int(delay_ms)), self.ensure_pose_ready)

    def ensure_pose_ready(self, box_ids=None) -> int:
        """Load the pose model for every configured box whose camera is live.

        Loads and warms; does NOT start inference. ``tracking_enabled`` stays
        False and the PoseSink stays disabled for the box until Test Tracking
        or Record asks for it, so opening a project does not quietly begin
        tracking every animal.

        Returns how many boxes are ready afterwards. Boxes with no tracking
        configured are not touched and not counted: they are not misconfigured,
        they are boxes that record video.
        """
        self._pose_ready_pending = False
        vm = getattr(self, "video_manager", None)
        if vm is None:
            return 0
        wanted = [int(b) for b in (box_ids if box_ids is not None
                                   else self.pose_configured_boxes())]
        if not wanted:
            return 0

        streaming, waiting = [], []
        for bid in wanted:
            if (vm.box_camera_map.get(bid) is not None
                    and vm.is_camera_streaming(bid)):
                streaming.append(bid)
            else:
                waiting.append(bid)

        if waiting and not streaming:
            # A cold camera open plus the driver settle (camera_settle_ms,
            # default 5 s) outlasts the first schedule. Retry on a short
            # cadence rather than giving up, else the model never loads on
            # load and the operator has to Init by hand every session.
            attempts = getattr(self, "_pose_ready_attempts", 0) + 1
            self._pose_ready_attempts = attempts
            if attempts <= 12:
                self._pose_ready_pending = True
                logger.info("Pose readiness: cameras not streaming yet "
                            "(attempt %d/12), retrying in 1.5 s", attempts)
                QtCore.QTimer.singleShot(1500, self.ensure_pose_ready)
                return 0
            logger.warning(
                "Pose readiness: cameras for box(es) %s never started "
                "streaming. Their model is not loaded; connect the camera and "
                "the model loads by itself, or use Init in Tracking Config.",
                ", ".join(str(b) for b in waiting))
            return 0

        ready = 0
        for bid in streaming:
            settings = self.pose_settings_for_box(bid)
            path = settings.get("model_path") or ""
            if not path:
                continue
            if not Path(path).exists():
                # Configured, but the model is not where it was saved. Say so
                # once rather than never initialising and never explaining.
                logger.warning(
                    "Box %s: configured %s model is not on disk: %s. Re-pick "
                    "it in Tracking Config.", bid,
                    (settings.get("mode") or "pose").upper(), path)
                continue
            if pose_ready_for(self.pipeline, settings):
                ready += 1
                continue
            # Pose backends need colour frames; the camera may have come up
            # grayscale from a previous session's setting.
            cam = vm.cameras.get(vm.box_camera_map.get(bid))
            if cam is not None and getattr(cam, "grayscale", False):
                cam.grayscale = False
            try:
                if self._configure_pose_for_box(bid, settings):
                    ready += 1
                self.tracking_enabled[bid] = False
            except Exception as e:
                logger.error("Box %s: pose init failed: %s", bid, e)

        if ready:
            logger.info("Pose ready: %d box(es) hold a warm model; a run start "
                        "only has to switch inference on.", ready)
            if hasattr(self, "statusbar"):
                try:
                    self.statusbar.showMessage(
                        f"Pose ready ({ready} box{'es' if ready > 1 else ''})",
                        3000)
                except (AttributeError, RuntimeError):
                    pass
        if waiting:
            # Some boxes streamed and some did not; come back for the rest.
            self.schedule_pose_ready()
        return ready

    def disconnect_all_cameras(self):
        """Disconnect every box's camera via the unified Pipeline. Shared by
        both modes, the toolbar/box-removal bulk action and the Camera
        Connect dialog's 'Disconnect Cameras' button both route here so the
        teardown (sink unsubscribe + bus unregister + device release) is
        identical and the dialog can't bypass the pipeline."""
        try:
            # disconnect_camera refreshes per box; coalesce to one sweep so a
            # multi-box disconnect isn't O(N²).
            self.begin_ui_refresh_batch()
            try:
                n = self._apply_to_all_boxes(self.disconnect_camera)
            finally:
                self.end_ui_refresh_batch()
            logger.info("Disconnected %d camera(s)", n)
        except Exception as e:
            logger.error("Failed to disconnect cameras: %s", e)
            self.showError(f"Failed to disconnect cameras: {e}")

    def disconnect_camera(self, setup_id) -> None:
        """Stop a box's camera via the unified Pipeline."""
        try:
            self.pipeline.disconnect_camera(setup_id)
            # Clear the UI's tracking mirror so a later reconnect doesn't show
            # the box as still tracking (the pipeline already disabled the
            # pose/tracker sinks for this box).
            te = getattr(self, "tracking_enabled", None)
            if isinstance(te, dict):
                te.pop(setup_id, None)
            try:
                self._box_video_clear(setup_id)
            except (AttributeError, RuntimeError):
                pass
            try:
                self._box_camera_disconnected_cleanup(setup_id)
            except (AttributeError, RuntimeError) as e:
                logger.debug("disconnect cleanup hook for %s: %s", setup_id, e)
            logger.info("Camera disconnected for box %s", setup_id)
        except Exception as e:
            logger.error("Camera disconnection error for box %s: %s", setup_id, e)
        finally:
            try:
                self.refresh_ui_state()
            except (AttributeError, RuntimeError):
                pass

    def _wait_for_camera_streaming(self, setup_id,
                                   max_attempts: Optional[int] = None) -> None:
        """Poll until the camera is streaming, then fire the ready hook.

        The poll window must outlast a COLD camera open: the device open
        plus the driver settle (``camera_settle_ms``, default 5 s) can take
        ~6 s. The manual Camera Config dialog rarely hits this (its
        probe/preview already warmed the device); auto-connect-on-load opens
        cold, so the window is derived from the configured settle + headroom
        rather than a fixed timeout, a fixed one fires a false "not
        streaming" warning and skips the ready hook when the camera comes up
        a moment later. Caller may override."""
        if max_attempts is None:
            settle_ms = 5000
            cfg = getattr(self, "_active_config", None)
            try:
                settle_ms = int(cfg.cameras.video_defaults.camera_settle_ms)
            except (AttributeError, TypeError, ValueError):
                pass
            # settle + cold-open/first-frame headroom, polled at 100 ms.
            max_attempts = max(50, (settle_ms + 10000) // 100)
        if not hasattr(self, "_camera_wait_attempts"):
            self._camera_wait_attempts: dict = {}
        self._camera_wait_attempts[setup_id] = 0

        def _tick():
            # Guard against a stale tick after the entry was already
            # popped (the timeout / streaming branch below removes
            # box_id from the dict).  Without this, a re-entry of
            # ``_wait_for_camera_streaming`` (e.g. user reconnects)
            # leaves an outstanding QTimer callback that fires later
            # and hits a KeyError on the missing entry.
            if setup_id not in self._camera_wait_attempts:
                return
            self._camera_wait_attempts[setup_id] += 1
            attempts = self._camera_wait_attempts[setup_id]
            try:
                streaming = self.video_manager.is_camera_streaming(setup_id)
            except Exception:
                streaming = False
            if streaming:
                logger.info(
                    "Camera for box %s streaming after %dms",
                    setup_id, attempts * 100)
                self._camera_wait_attempts.pop(setup_id, None)
                try:
                    self._camera_streaming_ready(setup_id)
                except (AttributeError, RuntimeError) as e:
                    logger.debug("streaming-ready hook for %s: %s", setup_id, e)
                return
            if attempts >= max_attempts:
                logger.warning(
                    "Camera for box %s not streaming after %dms",
                    setup_id, max_attempts * 100)
                self._camera_wait_attempts.pop(setup_id, None)
                try:
                    self._camera_streaming_timeout(setup_id)
                except (AttributeError, RuntimeError) as e:
                    logger.debug("streaming-timeout hook for %s: %s", setup_id, e)
                return
            QtCore.QTimer.singleShot(100, _tick)

        QtCore.QTimer.singleShot(100, _tick)

    # ==================================================================
    # UI state refresh
    # ==================================================================

    def get_all_setup_widgets(self):
        """Return every per-box widget the controller knows about, ascending
        by id. Reads ``_box_widgets`` directly (no per-id method dispatch)."""
        return [w for _, w in sorted(self._box_widgets.items()) if w is not None]

    def running_box_widgets(self):
        """Every box widget whose framework is currently running."""
        return [w for w in self.get_all_setup_widgets()
                if getattr(w, "framework_running", False)]

    def any_box_running(self) -> bool:
        """The ONE 'is any box running?' predicate, a fresh widget scan,
        never a cached ``ui_state`` flag that can lag a just-stopped box.
        Read this everywhere instead of re-implementing the scan. Iterates
        ``_box_widgets`` directly and short-circuits, no list build or sort."""
        return any(getattr(w, "framework_running", False)
                   for w in self._box_widgets.values())

    # ------------------------------------------------------------------
    # Master-button dialog launchers
    # ------------------------------------------------------------------
    #
    # Same eight entry points on both modes. Each opens a Universal*
    # dialog (or, for the analyzer, a subprocess) and then fires the
    # ``_after_universal_dialog_closed`` hook for mode-specific refresh.
    # Subclasses override only the hook.

    def _open_universal_dialog(self, DialogClass) -> None:
        try:
            DialogClass(self).exec()
        except Exception as e:
            logger.error(f"{DialogClass.__name__} error: {e}")
        finally:
            try:
                self._after_universal_dialog_closed()
            except (AttributeError, RuntimeError):
                pass

    def _after_universal_dialog_closed(self) -> None:
        """Hook, UI refresh after a universal dialog closes. Subclasses
        extend for mode extras (operant also resets the main tab)."""
        self.refresh_ui_state()

    def _current_state_name(self, setup_id):
        """This box's current MCU state name (the state machine's
        ``state_text``, e.g. 'ITI'/'choice'), or empty. Matches the
        _video_data.txt 'state' column. Reads the widget's ``task_info``
        panel when it has one; degrades to "" otherwise."""
        sw = self._box_widgets.get(setup_id)
        if not sw or not hasattr(sw, "task_info"):
            return ""
        try:
            return sw.task_info.state_text.text().strip()
        except Exception:
            return 

    def show_connect_dialog(self):
        from source.gui.dialogs import UniversalConnectDialog
        self._open_universal_dialog(UniversalConnectDialog)

    def show_disconnect_dialog(self):
        from source.gui.dialogs import UniversalDisconnectDialog
        self._open_universal_dialog(UniversalDisconnectDialog)

    def show_universal_start_dialog(self):
        """Multi-box Start. Lists ready-to-go boxes (connected + task
        uploaded + not running); sequentially calls each picked
        widget's ``on_record_clicked()``, same flow as the per-box
        Start button on BoxControlWidget.
        """
        from source.gui.dialogs import UniversalStartDialog
        self._open_universal_dialog(UniversalStartDialog)

    def show_universal_stop_dialog(self):
        """Multi-box Stop. Lists running boxes; warns to confirm, then
        sequentially calls each picked widget's ``on_stop_clicked()``."""
        from source.gui.dialogs import UniversalStopDialog
        self._open_universal_dialog(UniversalStopDialog)

    def show_universal_plot_dialog(self):
        """Session Plot, one scrolling window stacking every running box's
        states/events/analog. Shared by both modes (operant has a button,
        both have Ctrl+P)."""
        try:
            running_boxes = self.running_box_widgets()
            if not running_boxes:
                QtWidgets.QMessageBox.information(
                    self, "No Running Boxes",
                    "No boxes are currently running. Start a box first to view plots.")
                return
            if not getattr(self, "universal_plot_window", None):
                from source.gui.plotting import UniversalPlotWindow
                self.universal_plot_window = UniversalPlotWindow(self)
            self.universal_plot_window.update_boxes(running_boxes)
            self.universal_plot_window.show()
            self.universal_plot_window.raise_()
            self.universal_plot_window.activateWindow()
        except Exception as e:
            logger.error("Error showing session plot: %s", e)
            self.showError(f"Failed to show plot dialog: {e}")

    def _navigate_tab(self, delta: int) -> None:
        """Move the main tab widget by ``delta`` (wraps). Bound to Ctrl+Tab /
        Ctrl+Shift+Tab so tabs are keyboard-navigable regardless of focus."""
        tw = getattr(self, "tabWidget", None)
        if tw is None or tw.count() == 0:
            return
        tw.setCurrentIndex((tw.currentIndex() + delta) % tw.count())

    def show_config_dialog(self, setup_id=None):
        """Open the guarded ``UniversalConfigDialog`` (greys out running
        boards). Both the toolbar button and a box widget's own Config
        button route here; ``setup_id`` is accepted (the per-box button
        passes its box number) but ignored, the dialog shows every box."""
        from source.gui.dialogs import UniversalConfigDialog
        self._open_universal_dialog(UniversalConfigDialog)

    def show_upload_dialog(self):
        from source.gui.dialogs import UniversalUploadDialog
        self._open_universal_dialog(UniversalUploadDialog)

    def show_camera_config_dialog(self):
        """Camera Config, pre-check + CameraConnectDialog open."""
        state = self.compute_ui_state()
        if state.get("any_running") or state.get("any_recording_video") \
                or state.get("any_recording_data"):
            QtWidgets.QMessageBox.information(
                self, "Busy",
                "Camera configuration is locked while frameworks are running or recording.")
            return
        if not state.get("has_boxes"):
            QtWidgets.QMessageBox.warning(
                self, "No Boxes",
                "Add at least one box before configuring cameras.")
            return
        from source.gui.dialogs import CameraConnectDialog
        try:
            dlg = CameraConnectDialog(self)
            if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                if getattr(dlg, "connected_count", 0) > 0:
                    self.pipeline.set_capture_defaults(
                        target_fps=getattr(self, "video_target_fps", 30),
                        frame_strategy=getattr(
                            self, "video_frame_strategy", "accept"))
                    # If tracking is enabled AND a model is configured,
                    # auto-init it now. No more "Init Setup" button, the
                    # user gets DLC/SLEAP ready as part of Camera Connect.
                    self.start_tracking_init_background()
        except Exception as e:
            logger.error(f"Camera config error: {e}")
        finally:
            try:
                self._after_universal_dialog_closed()
            except (AttributeError, RuntimeError):
                pass

    def open_offline_analyzer(self):
        """Launch the standalone offline analyzer subprocess (same in both modes)."""
        import subprocess
        import sys as _sys
        project_root = Path(__file__).resolve().parents[2]
        try:
            # CREATE_NO_WINDOW so the child python doesn't flash a console
            # window on Windows.
            _flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            subprocess.Popen(
                [_sys.executable, "-m", "tools.offline_analysis.app"],
                cwd=str(project_root), creationflags=_flags)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Analyzer",
                f"Could not launch the offline analyzer:\n{e}")
            logger.error(f"Failed to launch offline analyzer: {e}")

    # ------------------------------------------------------------------
    # Tracking, single names on the base, mode-specific bodies in
    # _show_tracking_config_impl / _toggle_test_tracking_impl.
    #
    # Operant injects start_tracking/stop_tracking/trigger_event into the
    # UnifiedTrackingDialog and uses tracker_manager directly. Maze fans
    # the dialog config into per-box TrackingConfig + auto-saves zones.
    # The CALLERS only need one name; the implementations stay where
    # their mode-specific helpers live.

    def show_tracking_config_dialog(self):
        """Open the UnifiedTrackingDialog (same name in both modes)."""
        self._show_tracking_config_impl()

    def toggle_test_tracking(self):
        """Toggle test-tracking on/off (same name in both modes)."""
        self._toggle_test_tracking_impl()

    def _show_tracking_config_impl(self):
        """Open the unified tracking-configuration dialog and fan its
        result out to per-box ``TrackingConfig`` + zone storage.

        The flow is identical between operant and maze, only two pieces
        diverge:
          * **Extra kwargs to UnifiedTrackingDialog**  (operant seeds the
            zone-editor canvas with the first connected box).
          * **Post-dialog hook**  (operant refreshes the tracking-toggle
            button + status badges; maze auto-saves zones).

        Subclasses override ``_tracking_dialog_extra_kwargs`` and
        ``_post_tracking_dialog_hook`` for those two seams.
        """
        try:
            if not self._tracking_config_preflight_ok():
                return
            connected_with_cam = self._connected_cam_box_ids()
            if not connected_with_cam:
                QtWidgets.QMessageBox.information(
                    self, "No Cameras",
                    "Connect a camera before configuring tracking.")
                return

            from source.gui.widgets.tracking_panel import UnifiedTrackingDialog

            dialog = UnifiedTrackingDialog(
                parent=self,
                box_ids=connected_with_cam,
                get_frame_callbacks=self._build_get_frame_callbacks(connected_with_cam),
                connected_cameras=dict(self.video_manager.box_camera_map),
                initial_config=self._build_dialog_initial_config(),
                **self._tracking_dialog_extra_kwargs(connected_with_cam),
            )
            dialog.dlc_init_requested.connect(
                lambda cfg, _d=dialog: self._handle_dlc_init_from_dialog(cfg, _d))
            dialog.zones_live_changed.connect(self._on_dialog_zones_live)

            # Transaction boundary: snapshot the state the dialog can mutate as
            # a live preview (zones + dialog globals) so Cancel / Discard / X
            # can roll it back exactly. Nothing the dialog does persists until
            # an explicit apply, except "Save Zones to Project", a zones-only
            # partial commit that also updates this snapshot so the saved
            # zones survive a later Discard.
            txn = self._begin_tracking_dialog_txn()
            dialog.zones_save_requested.connect(
                lambda zmap, _t=txn: self._on_dialog_zones_saved(zmap, _t))
            dialog.exec()

            # Only reconfigure the boxes when the user explicitly applied
            # (Apply & Close). Cancel / X / Escape roll the live preview back,
            # closing the dialog must not silently push this config (or Box-1's
            # zones) onto every box, and must not leave a discarded preview in
            # place either.
            if not dialog.should_apply():
                self._rollback_tracking_dialog_txn(txn)
                logger.info("Tracking dialog dismissed without apply, "
                            "preview rolled back, box config unchanged.")
                return

            # ---- Single commit path -------------------------------------
            # Fan dialog state out into per-box TrackingConfig + global
            # _tracking_dialog_globals; install zones; apply push policy for
            # live boxes; persist once. Everything or nothing.
            full_config = dialog.get_full_config() or {}
            self._commit_tracking_dialog(full_config, dialog, connected_with_cam)
        except Exception as e:
            logger.error("Tracking config error: %s", e)
            import traceback
            traceback.print_exc()
        finally:
            try:
                self.refresh_ui_state()
            except (AttributeError, RuntimeError):
                pass

    # ----- Tracking-dialog transaction (snapshot / rollback / commit) -----

    def _begin_tracking_dialog_txn(self) -> dict:
        """Snapshot everything the tracking dialog can change as a live
        preview, so a non-apply close restores it byte-for-byte.

        The dialog previews zone edits straight into ``tracking_zones`` (for
        the camera-tile overlay) and the "Init DLC" button writes to the
        pipeline ``TrackingConfig`` registry (via ``_apply_push_policy``), so
        the registry is snapshotted too, a discarded Init must not leave zones
        or model state behind in it.
        """
        import copy
        registry = {}
        try:
            pipe = getattr(self, "pipeline", None)
            if pipe is not None:
                registry = {str(bid): tc.to_json()
                            for bid, tc in pipe.all_tracking_configs().items()}
        except Exception as e:
            logger.debug("txn registry snapshot failed: %s", e)
        return {
            "tracking_zones": copy.deepcopy(
                getattr(self, "tracking_zones", {}) or {}),
            "zone_baselines": copy.deepcopy(
                getattr(self, "_zone_baselines", {}) or {}),
            "dialog_globals": copy.deepcopy(
                getattr(self, "_tracking_dialog_globals", {}) or {}),
            "registry": registry,
        }

    def _rollback_tracking_dialog_txn(self, txn: dict) -> None:
        """Restore the open-time snapshot after Cancel / Discard / X.

        Re-installs the snapshotted zones (repainting every affected tile) and
        the dialog globals, so a discarded preview leaves no trace. No autosave
        fires, nothing was ever persisted.
        """
        if not isinstance(txn, dict):
            return
        import copy
        prev_zones = txn.get("tracking_zones", {})
        # Union of before/after box ids so a zone ADDED during the discarded
        # session is cleared, not left behind.
        affected = set(getattr(self, "tracking_zones", {}) or {}) | set(prev_zones)
        self.tracking_zones = copy.deepcopy(prev_zones)
        self._zone_baselines = copy.deepcopy(txn.get("zone_baselines", {}))
        self._tracking_dialog_globals = copy.deepcopy(txn.get("dialog_globals", {}))
        # Restore the pipeline TC registry to its open-time state, undoing any
        # "Init DLC" writes made during the discarded session. install_ only
        # ASSIGNS the snapshotted keys; it does not remove a box that gained a
        # TC mid-session (get_tracking_config auto-vivifies one). Clear first so
        # a discarded Init on a previously-unconfigured box leaves no phantom TC
        # (mirrors the load path, which clears before install).
        try:
            pipe = getattr(self, "pipeline", None)
            reg = txn.get("registry")
            if pipe is not None and reg is not None:
                try:
                    pipe._tracking_configs.clear()
                except (AttributeError, RuntimeError):
                    pass
                pipe.install_tracking_configs(reg)
        except Exception as e:
            logger.debug("txn registry restore failed: %s", e)
        for bid in affected:
            self._invalidate_zone_layer(bid)
            try:
                self._rebuild_zone_manager(bid, self.tracking_zones.get(bid, []))
            except (AttributeError, RuntimeError):
                pass
            try:
                self._redraw_box_overlay(bid)
            except (AttributeError, RuntimeError):
                pass

    def _on_dialog_zones_saved(self, zones_map: dict, txn=None) -> None:
        """Commit zones from the dialog's "Save Zones to Project" button.

        Zones-only partial commit: install per box, persist through the
        project config, and fold the result into the open dialog
        transaction so a later Cancel / Discard keeps the saved zones
        while still rolling tracker settings back. Recording boxes are
        skipped, zone geometry is frozen mid-session.
        """
        import copy
        saved, locked = [], []
        for raw_bid, zones in (zones_map or {}).items():
            try:
                bid = int(raw_bid)
            except (TypeError, ValueError):
                continue
            if self._zone_edit_locked(bid):
                locked.append(bid)
                continue
            if zones:
                self._set_box_zones(bid, list(zones))
            elif (getattr(self, "tracking_zones", {}) or {}).get(bid):
                # Present-empty = deliberate delete-all, same semantics as
                # the project-save reader (_read_tracking).
                self.tracking_zones[bid] = []
                self._invalidate_zone_layer(bid)
            else:
                continue
            with contextlib.suppress(AttributeError, RuntimeError):
                self._rebuild_zone_manager(
                    bid, self.tracking_zones.get(bid) or [])
            with contextlib.suppress(AttributeError, RuntimeError):
                self._redraw_box_overlay(bid)
            saved.append(bid)
        if isinstance(txn, dict):
            txn["tracking_zones"] = copy.deepcopy(
                getattr(self, "tracking_zones", {}) or {})
            txn["zone_baselines"] = copy.deepcopy(
                getattr(self, "_zone_baselines", {}) or {})
        if saved:
            self._project_changed(reason="zones_saved")
        parts = []
        if saved:
            parts.append("Zones saved to project for box(es): "
                         + ", ".join(str(b) for b in sorted(saved)))
        if locked:
            parts.append("NOT saved (recording): "
                         + ", ".join(str(b) for b in sorted(locked)))
        if parts:
            self._status_message(", ".join(parts))

    def _commit_tracking_dialog(self, full_config: dict, dialog,
                                connected_with_cam) -> None:
        """The single commit path for the tracking dialog.

        Fans dialog state into the per-box ``TrackingConfig`` registry,
        installs zones, applies the push policy to any already-tracking box,
        persists once through the project config, and warns when there is no
        project to persist into. Called only on an explicit apply.
        """
        self._apply_dialog_config(full_config)
        # Drop cached push policies so they rebuild from the new config.
        self._reset_tracking_policy()

        # Single source of truth for zones: dialog.get_all_zones().
        # Recording boxes keep their in-session geometry, applying a
        # zone change mid-record would silently redefine the MCU's zone
        # events and the session's _video_data.txt columns.
        skipped_recording = []
        for bid, zones in dialog.get_all_zones().items():
            if zones:
                if self._zone_edit_locked(int(bid)):
                    skipped_recording.append(int(bid))
                    continue
                self._set_box_zones(bid, zones)
        if skipped_recording:
            QtWidgets.QMessageBox.warning(
                self, "Zones Locked",
                "Zone changes were NOT applied to recording box(es): "
                + ", ".join(str(b) for b in sorted(skipped_recording))
                + ".\nStop the recording first, then apply again.")

        # Apply the new policy NOW for any box already tracking, otherwise the
        # user's coord_mapping / triggers wouldn't take effect until the next
        # enable cycle. ``_apply_push_policy`` covers zone-derived + dialog
        # override tables; ``apply_tracking_config`` additionally pushes the new
        # ``push_zones_to_mcu`` flag through the policy's ``push_zone_changed``
        # gate (toggling it mid-session would otherwise wait for Record).
        pipe = getattr(self, "pipeline", None)
        for bid in (self.tracking_enabled or {}):
            if self.tracking_enabled.get(bid):
                # ORDER MATTERS: apply_tracking_config → configure_from_zones
                # ASSIGNS the policy's coord_mapping/triggers from the TC;
                # _apply_push_policy then MERGES the dialog tables on top.
                # The reverse order lets the assign wipe every dialog-authored
                # c.* mapping. The Record path runs assign-then-merge, so the
                # loss would show up only on dialog Apply.
                if pipe is not None:
                    try:
                        pipe.apply_tracking_config(bid)
                    except Exception as e:
                        logger.debug(
                            "apply_tracking_config(%s) after dialog: %s", bid, e)
                self._apply_push_policy(bid)

        self._post_tracking_dialog_hook(dialog, connected_with_cam)
        self._offer_pose_init_after_apply()

        # The registry + zones are committed for this session regardless; but
        # without a loaded project there is nowhere durable to write them, so
        # say so plainly instead of silently losing the work on next launch.
        if not getattr(self, "active_config_path", ""):
            QtWidgets.QMessageBox.information(
                self, "No Project Loaded",
                "Tracking configuration applied for this session.\n\n"
                "Load or create a project to save it, without one, these "
                "settings are lost when the app closes.")

    def _offer_pose_init_after_apply(self) -> None:
        """Closing the dialog is where an init question belongs.

        The operator has just changed the settings, so asking now is fair and
        the answer costs nothing later. Asking at Record instead, which is
        what happened before, put a modal between the experimenter and a
        waiting animal, once per box, on settings they had not changed.

        Nothing is shown when the loaded model already matches, so applying a
        zone edit or a push option is silent. Boxes with no model configured
        are not counted and never asked about.
        """
        stale = []
        for bid in self.pose_configured_boxes():
            settings = self.pose_settings_for_box(bid)
            path = settings.get("model_path") or ""
            if not path or not Path(path).exists():
                continue
            if not pose_ready_for(self.pipeline, settings):
                stale.append(bid)
        if not stale:
            return
        boxes = ", ".join(str(b) for b in stale)
        changed = pose_settings_differ(
            self.pipeline.pose_fingerprint(), self.pose_settings_for_box(stale[0]))
        answer = QtWidgets.QMessageBox.question(
            self, "Initialise tracking model?",
            f"Tracking settings changed for box {boxes} "
            f"({', '.join(changed) or 'model'}).\n\n"
            "Initialise now so the next run starts without waiting?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes)
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            logger.info("Pose init deferred by the operator for box(es) %s; "
                        "the model loads at the next run start.", boxes)
            return
        self.ensure_pose_ready(stale)

    # ----- _show_tracking_config_impl: shared pieces + override hooks -----

    def _tracking_config_preflight_ok(self) -> bool:
        """Block opening the dialog while tracking is already running."""
        if any((self.tracking_enabled or {}).values()):
            QtWidgets.QMessageBox.information(
                self, "Tracking Active",
                "Stop tracking before changing settings.")
            return False
        return True

    def _connected_cam_box_ids(self):
        """Box ids with a non-None camera mapping. Shared across modes."""
        return [bid for bid, cam in self.video_manager.box_camera_map.items()
                if cam is not None]

    def _build_get_frame_callbacks(self, connected_with_cam):
        """Per-box closures returning the latest frame for the zone editor.

        ``video_manager.get_last_frame`` already runs
        ``segment_processor.extract_segment`` internally when a multi-box
        split is configured, the returned frame is ALREADY this box's
        segment. Without segmentation we apply the per-widget ROI on top
        so the dialog sees what the operator sees.
        """
        out = {}
        for bid in connected_with_cam:
            def _make_cb(b):
                def _get():
                    frame = self.video_manager.get_last_frame(b)
                    if frame is None:
                        return None
                    seg = self.video_manager.segment_processor_for_box(b)
                    if seg is not None and b in getattr(seg, 'box_index', {}):
                        return frame  # already segmented by get_last_frame
                    roi = self._get_box_roi(b, frame)
                    if roi:
                        x, y, w, h = (int(v) for v in roi[:4])
                        fh, fw = frame.shape[:2]
                        # A ROI that doesn't fit the current frame means the
                        # geometry no longer matches the resolution. Reject it
                        # (return the full frame with no crop) rather than
                        # CLAMP: a clamped crop is a size the runtime never
                        # produces (extract_segment rejects the same rect), so
                        # zones normalized against it land offset. This is the
                        # single-box path; shared-camera boxes already returned
                        # above via the segment.
                        if (x >= 0 and y >= 0 and w > 0 and h > 0
                                and x + w <= fw and y + h <= fh):
                            return frame[y:y + h, x:x + w]
                        logger.debug(
                            "Box %s: ROI %s does not fit frame %sx%s, "
                            "showing full frame in editor", b, roi, fw, fh)
                    return frame
                return _get
            out[bid] = _make_cb(bid)
        return out

    def _tracking_dialog_extra_kwargs(self, connected_with_cam) -> dict:
        """Mode-specific kwargs to UnifiedTrackingDialog. Default empty,
        maze uses this default; operant overrides to add its in-dialog
        start/stop/trigger callbacks."""
        return {}

    def _post_tracking_dialog_hook(self, dialog, connected_with_cam) -> None:
        """Default: log a one-line summary of the committed config.

        Zones persist through the project config (the commit path calls
        ``_apply_dialog_config`` → ``sync_active_config_tracking`` →
        ``_project_changed``); nothing writes a global side file anymore.

        Mode subclasses MAY override (operant adds button refresh) but MUST
        call ``super()._post_tracking_dialog_hook`` first.
        """
        try:
            for bid, zones in dialog.get_all_zones().items():
                if zones:
                    logger.info(
                        "Stored %d zone(s) for box %s: %s",
                        len(zones), bid, [z.get('name') for z in zones])
            settings = dialog.get_settings()
            method = settings.get("mode", "normal")
            n_boxes = len(settings.get("enabled_boxes", connected_with_cam))
            logger.info(
                "Tracking configured: mode=%s, %d box(es).",
                method, n_boxes)
        except Exception as e:
            logger.debug("_post_tracking_dialog_hook default: %s", e)

    def _on_dialog_zones_live(self, setup_id, zones) -> None:
        """Apply an in-dialog zone/scale edit to the live camera overlay ONLY.

        This is a *preview*, not a commit: ``_set_box_zones`` re-tags + stores
        the zones and invalidates the cached zone layer so the box's tile
        repaints with the new geometry on the next frame, but it does NOT
        persist anything. The tracking dialog is transactional, nothing
        reaches ``experiment_config.json`` until the operator explicitly
        applies (``_commit_tracking_dialog``), and a Cancel / Discard / X
        rolls the preview back to the open-time snapshot
        (``_rollback_tracking_dialog_txn``). Persisting here is exactly what
        made Cancel a lie: zone edits stuck even when the operator discarded
        them, while mode/model did not.
        """
        try:
            bid = int(setup_id)
        except (TypeError, ValueError):
            return
        if self._zone_edit_locked(bid):
            self._zone_lock_notice(bid)
            return
        try:
            self._set_box_zones(bid, list(zones or []))
        except (AttributeError, RuntimeError) as e:
            logger.debug("live zone preview failed for box %s: %s", bid, e)

    # ==================================================================
    # Test Tracking, unified preview-mode tracking (no MCU, no disk)
    # ==================================================================
    #
    # Two scenarios drive live tracking in pyBehaviorLab:
    #
    #   1. TEST TRACKING. User clicks the master "Test Tracking" toolbar
    #      button. Goal: verify zones + tracker tuning before committing
    #      to a real session. No MCU required, no recording happens, no
    #      data hits disk. Every camera-connected, configured box that
    #      isn't currently recording enters preview mode. The per-box
    #      ``online_tracking_enabled`` opt-out is bypassed (force=True)
    #      because this is preview, not persisted output.
    #
    #   2. RECORD WITH TRACKING ENABLED. User clicks per-box Record on a
    #      box whose ``TrackingConfig.online_tracking_enabled=True``.
    #      Tracking auto-starts via ``_on_framework_auto_started``, no
    #      need to press Test Tracking first. Opt-out IS respected
    #      (force=False).
    #
    # The two scenarios are mutually exclusive at the box level. While
    # Test Tracking is active on any box, that box's Record button is
    # disabled (see ``refresh_ui_state``). While ANY box is recording,
    # the master Test Tracking button is disabled, so a fresh preview
    # can't start mid-session.
    #
    # State source of truth: ``self.tracking_enabled[box_id]``, set by
    # ``start_tracking_for_box`` and reset by ``stop_tracking_for_box``.
    # A box is in TEST PREVIEW iff ``tracking_enabled[bid]`` is True AND
    # ``bid not in recording_boxes``.

    def _toggle_test_tracking_impl(self):
        """Toggle preview tracking on/off across configured boxes.

        Unified implementation, operant and maze inherit it directly.
        Mode-specific bringup goes through the
        ``_prepare_test_tracking_for_box`` hook; mode-specific button
        repainting through ``_refresh_test_tracking_button``.
        """
        if self._is_test_tracking_active():
            n = self._stop_test_tracking()
            self._refresh_test_tracking_button(active=False)
            self._status_message(f"Test tracking stopped ({n} box(es))")
            logger.info("Test tracking stopped (%d box(es))", n)
            return
        # Refuse to start while any box is recording, Record's
        # auto-start path already covers the recording-time tracking
        # case (see _on_framework_auto_started → start_tracking_for_box).
        if getattr(self, "recording_setups", None):
            QtWidgets.QMessageBox.information(
                self, "Recording Active",
                "Stop all recordings before starting Test Tracking.\n"
                "Recording boxes already track automatically when their "
                "TrackingConfig has online_tracking_enabled=True.")
            return
        n = self._start_test_tracking()
        if n > 0:
            self._refresh_test_tracking_button(active=True)
            self._status_message(f"Test tracking started ({n} box(es))")
            logger.info("Test tracking started (%d box(es))", n)

    def _start_test_tracking(self) -> int:
        """Enable preview tracking on every connected + configured box
        that isn't currently recording. Returns count started."""
        connected = self._connected_cam_box_ids()
        if not connected:
            QtWidgets.QMessageBox.information(
                self, "No Cameras",
                "Connect a camera before testing tracking.")
            return 0
        recording = set(getattr(self, "recording_setups", set()) or set())
        targets = [bid for bid in connected
                   if self._box_tracking_configured(bid) and bid not in recording]
        if not targets:
            QtWidgets.QMessageBox.information(
                self, "Nothing to Test",
                "No box has both a connected camera and configured "
                "tracking right now. Open Track Config to set one up, "
                "or stop running boxes first.")
            return 0
        # Fresh failure ledger for this attempt (T3).
        self._tracking_start_failures = {}
        # Same preparation as a run start, through the same path: Test Tracking
        # is a preview of the real thing, and it must not have its own way of
        # loading a model. Boxes already ready cost nothing here.
        pose_targets = [b for b in targets if b in set(self.pose_configured_boxes())]
        if pose_targets:
            self.ensure_pose_ready(pose_targets)
        count = 0
        failed: list = []
        for bid in targets:
            try:
                # Mode-specific bringup (operant: blob BG load + auto-init;
                # maze: no-op, handled inside its start_tracking_for_box).
                self._prepare_test_tracking_for_box(bid)
            except Exception as e:
                logger.warning(
                    "Box %s: _prepare_test_tracking_for_box failed: %s",
                    bid, e)
                self._note_tracking_start_failure(bid, f"bringup error: {e}")
                failed.append(bid)
                continue
            try:
                if self.start_tracking_for_box(bid, force=True):
                    count += 1
                else:
                    failed.append(bid)
            except Exception as e:
                logger.warning(
                    "Box %s: start_tracking_for_box(force=True) "
                    "failed: %s", bid, e)
                self._note_tracking_start_failure(bid, f"start error: {e}")
                failed.append(bid)
        if failed:
            logger.warning(
                "Test tracking skipped %d box(es): %s",
                len(failed), ", ".join(str(b) for b in failed))
            # Surface it, a log-only warning meant every target failing read
            # to the operator as "nothing happens" (T3). Name each box + why.
            reasons = getattr(self, "_tracking_start_failures", {}) or {}
            lines = []
            for b in failed:
                why = reasons.get(b, "unknown reason (see log)")
                lines.append(f"  • Box {b}: {why}")
            title = ("Test Tracking could not start"
                     if count == 0 else
                     "Test Tracking started with some failures")
            QtWidgets.QMessageBox.warning(
                self, title,
                ("None of the selected boxes could start tracking:\n\n"
                 if count == 0 else
                 f"{count} box(es) started; these could not:\n\n")
                + "\n".join(lines))
        self.refresh_ui_state()
        return count

    def _stop_test_tracking(self) -> int:
        """Stop preview tracking on every box that's NOT currently
        recording. Recording boxes keep their auto-started tracking
        intact, that's owned by the record lifecycle."""
        recording = set(getattr(self, "recording_setups", set()) or set())
        targets = [bid for bid, on in (self.tracking_enabled or {}).items()
                   if on and bid not in recording]
        for bid in targets:
            try:
                self.stop_tracking_for_box(bid)
            except Exception as e:
                logger.warning(
                    "Box %s: test-tracking stop failed: %s", bid, e)
        self.refresh_ui_state()
        return len(targets)

    def _is_test_tracking_active(self) -> bool:
        """True iff at least one box is in preview mode, i.e. tracking
        is on AND the box is not recording. Differentiates the preview
        state from recording-time auto-tracking."""
        recording = set(getattr(self, "recording_setups", set()) or set())
        return any(on and bid not in recording
                   for bid, on in (self.tracking_enabled or {}).items())

    def _prepare_test_tracking_for_box(self, setup_id) -> None:
        """Mode hook: per-box bringup BEFORE the unified
        ``start_tracking_for_box(force=True)``. Default no-op.

        Operant uses this to load the blob background image + auto-init
        the tracker so the inference path has a baseline frame the
        moment ``pipe.enable_blob_tracking`` activates. Maze's blob
        bringup now runs through the shared ``_start_tracking_for_box_impl``
        (background + enhancer hook + zones + push), so it needs nothing
        here."""
        return

    def _refresh_test_tracking_button(self, *, active: bool) -> None:
        """Reflect the preview state in the toolbar.

        The part every mode needs: **lock Tracking Config while a preview is
        running.** Reconfiguring tracking mid-test reinitialises the very
        thing being previewed, and both modes have the button, but only
        operant locked it, so in maze the operator could open Track Config
        during a preview and reconfigure underneath it.

        Modes override to paint their own toggle button and must call
        ``super()`` to keep the lock.
        """
        cfg_btn = getattr(self, "track_button", None)
        if cfg_btn is None:
            return
        try:
            cfg_btn.setEnabled(not active)
            cfg_btn.setToolTip(
                "Stop Test Tracking before changing settings"
                if active else "Configure tracking settings")
        except RuntimeError:
            pass          # widget already destroyed during teardown

    def _status_message(self, msg: str, timeout_ms: int = 3000) -> None:
        """Tolerant status-bar pulse. Quiet no-op when the host has no
        status bar (e.g. headless tests)."""
        sb = getattr(self, "statusbar", None) or getattr(self, "statusBar", None)
        if sb is None:
            return
        try:
            sb_obj = sb() if callable(sb) else sb
            sb_obj.showMessage(msg, timeout_ms)
        except Exception:
            pass

    def prompt_pose_init_required(self, setup_id: int) -> str:
        """3-button modal: Init now / Disable tracking / Cancel.

        Fired at Record click by ``RunTask._auto_enable_tracking`` when
        the box has DLC configured + a camera connected but the pose
        model is not loaded (or settings changed since last init).

        Returns the operator's choice as a lowercase string:
          * ``"init"``: operator wants to initialize the model now.
          * ``"disable"``: switch ``online_tracking_enabled`` off for
                             this box. ``dlc_model_path`` + other settings
                             STAY in TrackingConfig (user can re-enable
                             later without re-picking the model).
          * ``"cancel"``: abort the Record click.

        Headless / no-Qt fallback: returns ``"cancel"`` so non-GUI tests
        and async callers don't accidentally consume a hidden modal.
        """
        try:
            from PySide6 import QtWidgets
        except Exception:
            return "cancel"
        try:
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle(f"Box {setup_id}: pose not initialised")
            box.setText(
                f"Box {setup_id} has DLC / SLEAP configured but the model "
                f"is not loaded (or settings have changed since the last "
                f"Init).\n\n"
                f"What do you want to do?")
            init_btn   = box.addButton("Init now",         QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            disable_btn = box.addButton("Disable tracking", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
            box.addButton("Cancel",                        QtWidgets.QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(init_btn)
            box.exec()
            clicked = box.clickedButton()
            if clicked is init_btn:
                return "init"
            if clicked is disable_btn:
                return "disable"
            return "cancel"
        except Exception as e:
            logger.warning("prompt_pose_init_required: %s", e)
            return "cancel"

    def begin_ui_refresh_batch(self):
        """Suspend synchronous ``refresh_ui_state`` for a multi-box batch
        (Multi-Start / Multi-Stop). ``compute_ui_state`` sweeps every box
        (O(N)), so calling it once per box during a 16-box start is O(N²).
        While suspended, ``refresh_ui_state`` only records that a refresh is
        due; ``end_ui_refresh_batch`` applies exactly one at the end. Pair
        the two in a try/finally."""
        self._ui_refresh_suspended = True
        self._ui_refresh_pending = False

    def end_ui_refresh_batch(self):
        """Resume ``refresh_ui_state`` and apply one refresh if any box
        requested it while suspended."""
        self._ui_refresh_suspended = False
        if getattr(self, "_ui_refresh_pending", False):
            self._ui_refresh_pending = False
            self.refresh_ui_state()

    def refresh_ui_state(self):
        """Compute and apply UI state.

        Synchronous by design, button enable/disable must reflect the
        current state immediately, not on the next event-loop tick (a
        debounce here races: the Stop button stays enabled briefly after a
        stop, Start stays disabled briefly after the framework stops).
        Duplicate calls within one tick are cheap enough not to matter.

        EXCEPTION: during an explicit multi-box batch (see
        ``begin_ui_refresh_batch``) the O(N) sweep is coalesced to one call
        at batch end, so a 16-box Multi-Start doesn't run it 16× (O(N²)).
        """
        if getattr(self, "_ui_refresh_suspended", False):
            self._ui_refresh_pending = True
            return
        try:
            self._apply_ui_state(self.compute_ui_state())
        except Exception as e:
            logger.error("Failed to refresh UI state: %s", e)

    def compute_ui_state(self):
        """Aggregate per-box state into a single dict shared by both modes.

        Iterates boxes via the ``_iter_box_ids`` / ``_box_widget_for`` hooks
        each subclass implements. Reads per-widget flags through ``getattr``
        so widgets that don't expose every flag still work. Maintains
        ``self.recording_boxes`` set as a side effect (mirrors per-widget
        recording_video flags so iteration code can use either source).
        """
        try:
            box_ids = list(self._iter_box_ids())
        except NotImplementedError:
            box_ids = []
        state = {
            "has_boxes": len(box_ids) > 0,
            "box_states": {},
            "any_connected": False,
            "any_running": False,
            "any_recording_video": False,
            "any_recording_data": False,
            "any_ready_to_start": False,   # connected + task uploaded + record_button enabled
        }
        if not state["has_boxes"]:
            try:
                self.recording_setups.clear()
            except (AttributeError, RuntimeError):
                pass
            return state

        recording_set = getattr(self, "recording_setups", None)
        for setup_id in box_ids:
            try:
                widget = self._setup_widget_for(setup_id)
            except NotImplementedError:
                continue
            if widget is None:
                continue
            recording_video = bool(getattr(widget, "recording_video", False))
            if not recording_video and recording_set is not None:
                # Fall back to the set membership for modes that track
                # recording state in main_window instead of on the widget.
                recording_video = setup_id in recording_set
            box_state = {
                "connected":         bool(getattr(widget, "is_connected", False)
                                          or getattr(widget, "pycboard", None) is not None),
                "framework_running": bool(getattr(widget, "framework_running",
                                                  getattr(widget, "is_running", False))),
                "task_uploaded":     bool(getattr(widget, "task_uploaded", False)),
                "recording_video":   recording_video,
                "recording_data":    bool(getattr(widget, "recording_data", False)),
                "has_camera":        (bool(widget.camera_id_edit.text().strip())
                                      if hasattr(widget, "camera_id_edit") else False),
                # Per-box error alert. Two OR'd sources rendered red on every
                # surface by box_alerts.apply_box_alerts (called in _apply_ui_state):
                #  - TRANSIENT: status kind == "error" (board error, serial/camera
                #    drop), auto-clears when the box recovers (next set_status).
                #  - STICKY: ``_box_alarm`` (recorder/encoder hard failure, e.g. no
                #    NVENC session), survives start/stop/connect; cleared ONLY by
                #    reset_task / re-upload / a subsequent successful record.
                "error":             (getattr(widget, "_status_kind", "") == "error"
                                      or bool(getattr(widget, "_box_alarm", ""))),
                "error_msg":         (str(getattr(widget, "_box_alarm", "") or "")
                                      or str(getattr(widget, "_status_message", "") or "")),
            }
            # "ready to start" = same predicate Multi-Start uses: the
            # per-box record_button is currently enabled (i.e. connected
            # AND task uploaded AND framework idle, gated by the widget's
            # own logic).
            rec_btn = getattr(widget, "record_button", None)
            box_state["ready_to_start"] = (
                rec_btn is not None and rec_btn.isEnabled()
                and not box_state["framework_running"]
            )
            state["box_states"][setup_id] = box_state
            state["any_connected"]        |= box_state["connected"]
            state["any_running"]          |= box_state["framework_running"]
            state["any_recording_video"]  |= box_state["recording_video"]
            state["any_recording_data"]   |= box_state["recording_data"]
            state["any_ready_to_start"]   |= box_state["ready_to_start"]

        # Keep recording_boxes set consistent with per-widget flags.
        if isinstance(recording_set, set):
            recording_set.clear()
            recording_set.update(
                bid for bid, bs in state["box_states"].items()
                if bs["recording_video"] or bs["recording_data"]
            )
        return state

    def _apply_ui_state(self, state):
        """Enable/disable controls, ONE skeleton for both modes:
        common state → metadata buttons → per-box locks + indicator
        surfaces → test-tracking gates → mode buttons. Subclasses implement
        ``_apply_mode_buttons`` (and, for operant, point
        ``_box_indicator_surfaces`` at the video tile holders).

        The mode hook runs LAST because Multi-Start is gated on
        ``any_ready_to_start``, and that is read off the per-box Record
        buttons rather than computed from box facts: Multi-Start clicks
        exactly those buttons, so the widget's own logic is the authority on
        whether one can be pressed. Running the hook first gated it on a
        reading taken before the two steps below that change those buttons,
        and Multi-Start was then a refresh behind, offering to press Record
        on a box whose Record had just been greyed out for test preview."""
        flags = self._apply_common_ui_state(state)

        # Metadata sidebar buttons (Load / Edit / Auto-Populate), flip
        # the run-lock on the start/stop event, not the 1 Hz tick.
        self.update_metadata_button_states()

        # Upload-Task master button, same gate in both modes.
        if hasattr(self, "upload_task_button"):
            self.upload_task_button.setEnabled(flags["any_idle_connected"])

        # Per-box locks + zone-adjust enables + camera-pending flag. Same
        # loop flags boxes whose camera is configured but not connected
        # (red "Camera not connected") so it's clear which still need a
        # Connect.
        vm = getattr(self, "video_manager", None)
        cam_map = getattr(vm, "box_camera_map", {}) if vm is not None else {}
        box_states = state.get("box_states", {})
        for sid, widget in self._box_widgets.items():
            if widget is None or not hasattr(widget, "apply_global_state"):
                continue
            bs = box_states.get(sid, {})
            box_running = bool(bs.get("framework_running"))
            box_recording = bool(bs.get("recording_video")
                                 or bs.get("recording_data"))
            widget.apply_global_state(
                lock_setup=box_running or box_recording,
                # Controls stay enabled mid-run so users can tweak variables.
                lock_controls=False,
            )
            # Empty zones list still counts as "in" the dict, guard so
            # closing the tracking dialog without drawing zones keeps the
            # zone-adjust buttons disabled.
            zones = (getattr(self, "tracking_zones", None) or {}).get(sid) or []
            configured = bool(self._box_camera_id_text(sid))
            connected = cam_map.get(sid) is not None
            for surface in self._box_indicator_surfaces(sid, widget):
                if surface is None:
                    continue
                try:
                    if hasattr(surface, "set_zones_enabled"):
                        # Zones are frozen while the box runs/records,
                        # the adjust buttons grey out with the same gate
                        # the mutation handlers enforce.
                        surface.set_zones_enabled(
                            bool(zones) and not (box_running or box_recording))
                    if hasattr(surface, "set_camera_pending"):
                        surface.set_camera_pending(configured and not connected)
                except RuntimeError:
                    pass  # surface destroyed by a layout rebuild

        # Test Tracking ↔ Record mutual exclusion. After the per-box
        # ``apply_global_state`` loop so the widget's own enable logic
        # doesn't clobber the Record-disable on boxes in test preview.
        self._apply_test_tracking_gates(state)

        # Now the Record buttons are final for this pass, so re-read them
        # before the hook that gates Multi-Start on them.
        self._refresh_ready_to_start(state)
        self._apply_mode_buttons(state, flags)

        # Any duplicated control (operant's Video Stream tab carries a second
        # Camera Config and Test Tracking) follows the originals from here,
        # because this is where every control's state is decided. Hooking it
        # anywhere else means a copy that is right at startup and wrong after
        # the first state change.
        sync = getattr(self, "_sync_mirrored_buttons", None)
        if callable(sync):
            sync()

    def _refresh_ready_to_start(self, state) -> None:
        """Re-read the per-box Record buttons into ``state``.

        ``compute_ui_state`` takes this reading before the pass runs, which
        is right for everything that only needs to know what the rig looked
        like. Multi-Start needs to know what the Record buttons look like
        NOW, because it is about to offer to press them, and both the
        per-box ``apply_global_state`` and the test-preview gate have since
        changed them.
        """
        any_ready = False
        for sid, widget in getattr(self, "_box_widgets", {}).items():
            bs = state.get("box_states", {}).get(sid)
            if bs is None or widget is None:
                continue
            rec_btn = getattr(widget, "record_button", None)
            ready = (rec_btn is not None and rec_btn.isEnabled()
                     and not bs.get("framework_running", False))
            bs["ready_to_start"] = ready
            any_ready |= ready
        state["any_ready_to_start"] = any_ready

    def _apply_mode_buttons(self, state, flags) -> None:
        """Mode hook: enable/disable the buttons only this mode has."""

    def _box_indicator_surfaces(self, setup_id, widget):
        """Widgets carrying this box's zone-adjust buttons + camera-pending
        flag. Default: the box widget itself (maze); operant returns the
        per-box video tile holder instead."""
        return (widget,)

    def _on_framework_auto_started(self, setup_id):
        """Slot for ``BoxControlWidget.framework_started_signal`` /
        ``SetupWidget.framework_started_signal``. Fires AFTER the MCU
        framework boot in EVERY path, record button, multi-start
        dialog, auto-restart. Auto-starts live tracking for that box
        when its ``TrackingConfig.online_tracking_enabled`` is on.

        This is the canonical hook for "Record → auto-track". Doing it in
        the record-click handler instead does NOT work on a normal
        Record click: the recorder open (``open_session_video_recorder``)
        adds the box to ``recording_boxes`` BEFORE the main-window's
        ``start_recording`` lambda runs, so ``_validate_start_recording``
        bails with "already recording". This signal hook fires reliably
        regardless.
        """
        widget = None
        try:
            widget = self._setup_widget_for(setup_id)
        except (AttributeError, NotImplementedError):
            pass
        # Only auto-start when this box has a camera. No camera → no
        # frames → tracking would just spin in the pipeline.
        has_camera = False
        if widget is not None:
            cam_edit = getattr(widget, "camera_id_edit", None)
            if cam_edit is not None:
                try:
                    has_camera = bool(cam_edit.text().strip())
                except Exception:
                    pass
        if not has_camera:
            return
        try:
            self.start_tracking_for_box(setup_id)
        except Exception as e:
            logger.warning(
                "Box %s: auto start_tracking_for_box failed: %s",
                setup_id, e)

    def _register_stats_consumer(self, setup_id, widget) -> None:
        """Register one box with the live MCU statistics canvas on framework
        start, BOTH widgets call this from ``_after_start``; the canvas is
        lazy-created here when a mode hasn't built it yet.

        Gated so an unconfigured, non-auto stats panel pays no per-record
        GUI-thread cost. ``config_mode`` of ``None`` counts as auto, so the
        normal path in both modes still registers. Swallows + logs canvas
        errors so a stats bug never aborts a run start.
        """
        stats_tab = getattr(self, "statisticsTab", None)
        if stats_tab is None:
            # Lazy-create so the consumer is wired on the FIRST framework
            # start and historical events aren't lost (operant builds the
            # tab eagerly, so this fires only in maze / stats-window-less
            # setups).
            try:
                from source.stats import StatsCanvas
                stats_tab = self.statisticsTab = StatsCanvas(self)
            except Exception as e:
                logger.warning(
                    "Box %s: stats canvas lazy-create error: %s", setup_id, e)
                return
        has_config = bool(getattr(stats_tab, "config", None))
        in_auto_mode = getattr(stats_tab, "config_mode", None) in ("auto", None)
        if not (has_config or in_auto_mode):
            return
        try:
            # Only grow the box count when a higher id actually appears,
            # updateBoxCount rebuilds every stats group's table, so calling
            # it on every record start (count unchanged) is wasted work.
            if setup_id > getattr(stats_tab, "box_count", 0):
                stats_tab.updateBoxCount(setup_id)
            stats_tab.onFrameworkStart(setup_id, setup_widget=widget)
        except Exception as e:
            logger.warning(
                "Box %s: stats onFrameworkStart error: %s", setup_id, e)

    def _unregister_stats_consumer(self, setup_id) -> None:
        """Drop one box from the live stats canvas on framework stop.
        Same name + body in both modes. Idempotent, no-op when the canvas
        was never created.
        """
        stats_tab = getattr(self, "statisticsTab", None)
        if stats_tab is None:
            return
        try:
            stats_tab.onFrameworkStop(setup_id)
        except Exception as e:
            logger.warning(
                "Box %s: stats onFrameworkStop error: %s", setup_id, e)

    def _on_metadata_clicked(self, setup_id):
        """Per-box / per-setup Subj** click, shared slot, both modes wire
        ``widget.metadata_clicked`` to it.

        A populated box (subject matching its cohort row) shows the
        ``SubjectCardDialog`` with a "Pick different" button; an empty or
        mismatched one opens the universal ``AssignSubjectsDialog``. Either
        way the cohort is the one already loaded via Load Metadata, no
        re-prompt for the Excel file.

        Operant refreshes its per-box metadata-button colours afterwards;
        maze has no such buttons, so the refresh is guarded.
        """
        self.metadata_manager.show_subject_card_or_assign(self, setup_id)
        refresh = getattr(self, "update_metadata_button_states", None)
        if callable(refresh):
            refresh()

    def _box_in_test_preview(self, setup_id) -> bool:
        """A box is in TEST PREVIEW when its tracking is on AND it is
        NOT currently recording. Mutually exclusive with recording,
        the Record button gets disabled for boxes in this state."""
        recording = getattr(self, "recording_setups", None) or set()
        return (bool((self.tracking_enabled or {}).get(setup_id, False))
                and setup_id not in recording)

    def _apply_test_tracking_gates(self, state) -> None:
        """Apply per-box mutual-exclusion gates between Test Tracking
        and Record.

        **Per-box, never global** (each box runs its own MCU, video,
        tracking, Box A previewing must not lock Box B). Two layers:

          * Per-box widget controls: when this box is in TEST_PREVIEW
            (tracking on AND not recording), disable that box's Record
            + per-widget MCU buttons (Connect / Upload / Camera). Other
            boxes untouched.

          * Master Test Tracking button: enabled iff at least one box
            is eligible for preview (configured + connected camera +
            not recording + not already previewing). Per-mode natural
            updaters compute this in their own _apply_ui_state; this
            method delegates to them rather than gating globally on
            "any box recording".

        Called from each mode's ``_apply_ui_state`` AFTER the per-box
        widget locks (so the per-box Record-disable overlay wins).
        """
        # Per-box locks. Walk every box; if this box is in test
        # preview, disable its widget's record/MCU buttons.
        for bid in state.get("box_states", {}).keys():
            if not self._box_in_test_preview(bid):
                continue
            try:
                widget = self._setup_widget_for(bid)
            except NotImplementedError:
                continue
            if widget is None:
                continue
            # Record / start
            rec_btn = getattr(widget, "record_button", None)
            if rec_btn is not None:
                rec_btn.setEnabled(False)
                rec_btn.setToolTip("Stop Test Tracking before recording")
            # Per-widget MCU controls, locked because tweaking MCU
            # state mid-preview would yank frames out from under the
            # live inference.
            for attr, tip in (
                ("connect_button",   "Stop Test Tracking before changing MCU"),
                ("upload_button",    "Stop Test Tracking before changing MCU"),
                ("disconnect_button","Stop Test Tracking before changing MCU"),
            ):
                btn = getattr(widget, attr, None)
                if btn is not None:
                    btn.setEnabled(False)
                    btn.setToolTip(tip)

        # Master Test Tracking button, restore natural enable state
        # via the operant updater. Maze updates the button inline in
        # its _apply_ui_state, so the base default is a no-op.
        # NOTE: no global "disable because any box recording" branch,
        # per-box independence means recording boxes don't block
        # preview on idle boxes.
        try:
            self._update_tracking_toggle_button()
        except Exception as e:
            logger.debug("_update_tracking_toggle_button failed: %s", e)

    def _update_tracking_toggle_button(self) -> None:
        """Mode hook: refresh the master Test-Tracking button's enabled
        state + tooltip. Operant overrides; maze paints inline in its
        _apply_ui_state. Default no-op."""

    def _apply_common_ui_state(self, state):
        """Enable/disable buttons present in both maze and operant modes.

        Returns the derived flags dict so the subclass can reuse them for
        its mode-specific buttons without re-deriving them:
            ``{"has_boxes", "any_running", "block_run",
               "any_idle_connected", "any_disconnected"}``

        Master buttons gate on "is there anything for this dialog to act
        on", mirroring what each dialog shows after filtering:
          * Connect    → ``any_disconnected`` (a connected box can't be
            re-connected; if every box is connected there's nothing to do).
          * Config/Disconnect → ``any_idle_connected``. Both dialogs show
            running boxes with a DISABLED checkbox, a running MCU can't be
            reconfigured or yanked mid-session, so a rig whose only
            connected boxes are running has nothing for either to act on.
        Per ``feedback-master-buttons-idle-gating``: when some boxes run
        and others are idle, the dialogs filter to the idle ones and the
        buttons stay enabled.
        """
        has_boxes = state.get("has_boxes", False)
        any_running = state.get("any_running", False)
        block_run = any_running
        box_states = state.get("box_states", {}).values()
        any_idle_connected = any(
            bs.get("connected") and not bs.get("framework_running")
            for bs in box_states
        )
        any_disconnected = any(
            not bs.get("connected")
            for bs in box_states
        )

        flags = {
            "has_boxes": has_boxes,
            "any_running": any_running,
            "block_run": block_run,
            "any_idle_connected": any_idle_connected,
            "any_disconnected": any_disconnected,
        }

        # Buttons that exist (with identical semantics) in both modes.
        # getattr-with-None tolerates modes that don't expose a given
        # button; we only call setEnabled when the attribute is present.
        rules = (
            ("add_box_button",          not block_run),
            ("remove_box_button",       has_boxes and not block_run),
            ("save_config_button",      has_boxes and not block_run),
            ("load_config_button",      not block_run),
            ("camera_connect_button",   has_boxes and not block_run),
            ("connect_button",          any_disconnected),
            ("config_button",           any_idle_connected),
            ("disconnect_button",       any_idle_connected),
        )
        for attr, enabled in rules:
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setEnabled(bool(enabled))

        # Propagate per-box ERROR state to ALL surfaces (control widget, live
        # status, stats row, video tile). Single hook for both modes; runs on
        # the same refresh cycle so it appears and auto-clears automatically.
        try:
            from source.gui.box_alerts import apply_box_alerts
            apply_box_alerts(self, state.get("box_states", {}))
        except Exception as e:
            logger.debug("apply_box_alerts failed: %s", e)

        return flags

    # ==================================================================
    # Shared tracking + recording helpers.
    # ==================================================================

    # ------------------------------------------------------------------
    # Unified overlay write, every result callback (pose, blob) lands
    # here.  One write path, one freshness stamp.
    # ------------------------------------------------------------------

    def _on_overlay_update(self, setup_id, **fields):
        """Merge new fields into ``self._overlay[box_id]`` and stamp
        ``last_seen_ns``.  Creates the OverlayState on first write.
        """
        from source.video.framebus.types import OverlayState
        state = self._overlay.get(setup_id)
        if state is None:
            state = OverlayState()
            self._overlay[setup_id] = state
        for k, v in fields.items():
            setattr(state, k, v)
        state.last_seen_ns = time.monotonic_ns()

    def _on_tracking_update(self, setup_id, x, y, w, h, timestamp):
        """TrackerManager-internal callback. The pipeline's TrackerSink
        also publishes via QtBridge → ``_on_box_tracker``, which is the
        canonical writer. We mirror the position here for older code
        paths (Test Tracking) that bypass the pipeline.
        """
        self._on_overlay_update(
            setup_id,
            bbox=(int(x), int(y), int(w), int(h)),
        )

    def _reset_tracking_policy(self, setup_id=None):
        """Drop the cached push policy so it rebuilds from current zones/config."""
        try:
            self.pipeline.reset_push_policy(setup_id)
        except (AttributeError, RuntimeError) as e:
            logger.debug("reset_push_policy(%s) error: %s", setup_id, e)

    def _apply_push_policy(self, setup_id):
        """Push the box's full coord_mapping + triggers into MCUPusher.

        Two layers (in this order):

        1. Zone-derived: ``configure_push_zones`` builds coord_mapping +
           triggers from each zone's ``transmit_mode`` / ``coord_var`` /
           ``event_on_enter`` / ``event_on_exit`` fields.

        2. Dialog overrides: the user's tracking-configure dialog Coord
           Mapping + Triggers tables, stored on ``self._tracking_dialog_globals``.
           Applied via ``configure_push_tracking`` which MERGES, dialog
           values override zone-derived ones on the same coord_name/
           event_name, but additional zone-derived entries survive.

        Called every time tracking enables (blob, pose) AND every time
        the tracking dialog closes, guarantees the policy reflects the
        latest user intent.
        """
        try:
            zones = (getattr(self, "tracking_zones", {}) or {}).get(setup_id, []) or []
            # ALWAYS run the translator, an empty list must CLEAR the
            # zone-derived mappings/triggers. Skipping on empty left the
            # pusher writing c.loc_center for zones the operator deleted
            # (and made behaviour depend on which apply path ran last).
            # Honour the box's push gates + picked body part so the GUI
            # path builds the same policy apply_tracking_config would.
            kw = {}
            try:
                tc = self.pipeline.get_tracking_config(setup_id)
                kw = dict(
                    default_body_part=str(
                        getattr(tc, "zone_change_body_part", "")
                        or "centroid"),
                    push_coords=bool(
                        getattr(tc, "push_coords_to_mcu", True)),
                    push_zone_events=bool(
                        getattr(tc, "push_zones_to_mcu", True)))
            except Exception:
                pass
            self.pipeline.configure_push_zones(setup_id, zones, **kw)
        except (AttributeError, RuntimeError) as e:
            logger.debug("configure_push_zones(%s) error: %s", setup_id, e)
        try:
            ov = getattr(self, "_tracking_dialog_globals", None) or {}
            cm = ov.get("coord_mapping") or {}
            # Authored triggers persist on the box's TrackingConfig (restored
            # from the project on load). The session dialog-globals only hold
            # triggers when the dialog was opened THIS session, so merging only
            # those would wipe project-restored triggers the moment tracking
            # starts (M2), until the next Record click re-applied them via
            # apply_tracking_config. Merge the TC triggers in too; dialog rows
            # win on event_name collision (they come last).
            tc_triggers = []
            try:
                tc = self.pipeline.get_tracking_config(setup_id)
                tc_triggers = list(getattr(tc, "triggers", []) or [])
            except Exception:
                tc_triggers = []
            dialog_tg = list(ov.get("triggers") or [])
            # Dedup only on TRUTHY event names. Display-only triggers (plot /
            # annotation, no MCU event) carry event_name None/"", collapsing
            # those to one key would drop every project-restored display-only
            # trigger the moment the dialog held any display-only row of its
            # own. Keep every tc row whose event_name is falsy.
            dialog_events = {t.get("event_name") for t in dialog_tg
                             if isinstance(t, dict) and t.get("event_name")}
            tg = [t for t in tc_triggers
                  if not (isinstance(t, dict)
                          and t.get("event_name")
                          and t.get("event_name") in dialog_events)] + dialog_tg
            if cm or tg:
                self.pipeline.configure_push_tracking(
                    setup_id, {"coord_mapping": cm, "triggers": tg})
        except Exception as e:
            logger.debug("configure_push_tracking(%s) error: %s", setup_id, e)
        # Mirror the just-applied zones into the central TrackingConfig so
        # the dialog's Save button persists complete state (without this,
        # Save would only carry the DLC model path the dialog explicitly
        # wrote, never the runtime-edited zones). The direct
        # configure_push_zones / configure_push_tracking calls above are
        # still the source of truth at runtime; this sync makes the
        # central registry reflect that state for persistence.
        try:
            zones = (getattr(self, "tracking_zones", {}) or {}).get(setup_id, []) or []
            self.pipeline.update_tracking_config(setup_id, zones=list(zones))
        except (AttributeError, RuntimeError) as e:
            logger.debug("update_tracking_config(%s, zones) error: %s", setup_id, e)

    def _apply_dialog_config(self, full_cfg: dict) -> list:
        """Fan a unified-tracking-dialog ``get_full_config()`` dict out into
        the per-box ``TrackingConfig`` registry.

        The single migration point for dialog → runtime state: every field
        with a TrackingConfig home is written through
        ``pipeline.update_tracking_config``, and what is left over is stashed
        on ``self._tracking_dialog_globals``.

        Returns the box ids that were actually opted in via their checkbox, so
        the caller can run per-box follow-ups (zone install, push policy, UI
        refresh). A rig-level update alone is not a "tracking turned on" event.
        """
        if not isinstance(full_cfg, dict):
            return []
        tracking = full_cfg.get("tracking", full_cfg) or {}
        if not isinstance(tracking, dict):
            return []
        pipe = getattr(self, "pipeline", None)
        if pipe is None:
            return []

        tracker_type = _dialog_tracker_type(tracking)
        is_pose = tracker_type in ("dlc", "sleap")
        body_parts = tracking.get("body_parts") or []
        annotate_saved = tracking.get("annotate_saved") or {}
        smooth = bool(tracking.get("smooth_tracking", False))
        enabled, target_boxes = _dialog_box_ids(tracking)

        configured = []
        for bid in target_boxes:
            online = bid in enabled
            fields = _dialog_shared_fields(tracking, tracker_type, online)
            fields.update(_dialog_pose_fields(tracking, body_parts) if is_pose
                          else _dialog_blob_fields(tracking, smooth))

            bg_path = self._background_path_for_box(bid)
            if bg_path is not None:
                fields["blob_background_path"] = str(bg_path)
            ann = _dialog_annotate_flag(annotate_saved, bid)
            if ann is not None:
                fields["annotation_enabled"] = bool(ann)

            try:
                pipe.update_tracking_config(bid, **fields)
            except (AttributeError, RuntimeError) as e:
                logger.warning("update_tracking_config(%s) failed: %s", bid, e)
                continue
            if online:
                configured.append(bid)

        self._tracking_dialog_globals.update(_dialog_global_overrides(
            full_cfg, tracking, tracker_type, annotate_saved))

        # Refresh the durable tracking copy NOW rather than at the next full
        # read, so an autosave firing while the pipeline is transiently empty
        # cannot drop a just-configured model. See
        # experiment.sync_active_config_tracking.
        try:
            from source.config.experiment import sync_active_config_tracking
            sync_active_config_tracking(self)
        except Exception as e:
            logger.debug("sync_active_config_tracking after dialog: %s", e)
        self._project_changed(reason="tracking_changed")
        return configured

    def _box_tracking_configured(self, setup_id: int) -> bool:
        """True when the box has any tracking-relevant state set,
        either a DLC model path, a blob background, or a non-empty zone
        list.  Used by Test Tracking and refresh_ui_state to decide
        whether the user has configured anything meaningful."""
        try:
            tc = self.pipeline.get_tracking_config(setup_id)
        except Exception:
            return False
        if tc.has_dlc():
            return True
        if tc.has_blob():
            return True
        return False

    def _any_box_tracking_configured(self) -> bool:
        """True if at least one box in the pipeline registry has
        ``_box_tracking_configured`` returning True. Cheap fan-out for the
        toolbar button enable / disable logic."""
        try:
            for tc in self.pipeline.all_tracking_configs().values():
                if tc.has_dlc() or tc.has_blob():
                    return True
        except (AttributeError, RuntimeError):
            pass
        return False

    def _tracking_config_to_settings_dict(self, setup_id: int) -> dict:
        """Build the dict shape the existing ``_enable_pose_for_box`` /
        ``_start_tracking_for_box_impl`` flows still expect.  Bridges the
        per-box ``TrackingConfig`` source-of-truth to the older
        settings-dict callers without forcing a full call-site rewrite.
        Backgrounds + annotate_saved are mirrored back into the dict so
        downstream code sees a single shape regardless of where the
        field lives.
        """
        try:
            tc = self.pipeline.get_tracking_config(setup_id)
        except Exception:
            return {}
        ov = getattr(self, "_tracking_dialog_globals", None) or {}
        is_pose = tc.has_dlc()
        mode = tc.tracker_type if is_pose else "blob"
        settings: dict = {
            "mode":             mode,
            "tracker_type":     mode,
            "model_path":       tc.dlc_model_path or "",
            "body_parts":       list(tc.keypoint_names),
            "dlc_confidence":   float(tc.confidence_threshold),
            "dlc_resize":       float(tc.pose_resize_factor),
            "pose_instances":   int(tc.pose_n_instances),
            "threshold":        int(tc.blob_threshold),
            # NOTE: the pose half of this dict is overwritten below by
            # ``pose_settings_for_box``. What remains here is the blob half
            # plus the fields both trackers share.
            "min_area":         int(tc.blob_min_area),
            "max_area":         int(tc.blob_max_area),
            "detect_dark":      bool(tc.blob_detect_dark),
            "blur_mode":        tc.blob_blur_mode,
            "blur_kernel_size": tc.blob_blur_kernel_size,
            "bg_mode":          tc.blob_bg_mode,
            "open_kernel_size": tc.blob_open_kernel_size,
            "close_kernel_size": tc.blob_close_kernel_size,
            "use_clahe":        bool(tc.blob_use_clahe),
            "clahe_clip_limit": float(tc.blob_clahe_clip_limit),
            "clahe_tile_size":  int(tc.blob_clahe_tile_size),
            "use_adaptive_threshold": bool(tc.blob_use_adaptive_threshold),
            "self_norm_ratio":  float(tc.blob_self_norm_ratio),
            "self_norm_sigma":  float(tc.blob_self_norm_sigma),
            "self_norm_smooth_sigma": float(tc.blob_self_norm_smooth_sigma),
            "self_norm_minsize": int(tc.blob_self_norm_minsize),
            "smooth_tracking":  bool(tc.blob_smooth_tracking),
            "enabled_boxes":    [setup_id],
            "annotate_parts":   list(ov.get("annotate_parts") or []),
            "zone_body_part":   ov.get("zone_body_part") or "",
            "marker_size":      int(ov.get("marker_size", 4) or 4),
            "coord_mapping":    dict(ov.get("coord_mapping") or {}),
            "triggers":         list(ov.get("triggers") or []),
        }
        if is_pose:
            # One builder owns everything a pose model is built from, so this
            # dict cannot be the one that quietly leaves out the SLEAP options
            # or the DLC engine settings.
            settings.update(self.pose_settings_for_box(setup_id))
        return settings

    def _pose_already_loaded_for(self, tc) -> bool:
        """True when the live model is the one ``tc`` asks for, so the dialog
        opens with Init disabled rather than always inviting a re-load.

        Asks the same question Record asks, through the same function, so the
        button and the run agree. They did not before: the button ran a second
        description of the model that omitted the DLC engine options, and so
        reported "not loaded" on every DLC box for ever."""
        try:
            return self.pose_box_is_ready(int(getattr(tc, "setup_id", 0) or 0))
        except Exception as e:
            logger.debug("pose init state for dialog: %s", e)
            return False

    def _build_dialog_initial_config(self) -> dict:
        """Recompose a ``UnifiedTrackingDialog`` ``initial_config`` dict
        from per-box ``TrackingConfig`` + ``_tracking_dialog_globals``.  Used to
        pre-populate the dialog so reopening it shows the user's last
        choices.  When no boxes are configured, returns an empty dict,
        the dialog handles that as "fresh state".
        """
        pipe = getattr(self, "pipeline", None)
        if pipe is None:
            return {}
        all_tcs = {}
        try:
            all_tcs = pipe.all_tracking_configs() or {}
        except Exception:
            return {}
        # Dialog expects dict-shape zone entries, which live on
        # ``tracking_zones`` (TC.zones is the typed Zone dataclass list,
        # not the dialog's wire format). Zones are tracker-agnostic, so
        # they are collected BEFORE the configured-tracker gate below,
        # a box with zones but no tracker choice must still reopen with
        # its zones visible (returning {} here made them look deleted).
        tz = getattr(self, "tracking_zones", {}) or {}
        zones_per_box = {str(bid): list(zs) for bid, zs in tz.items() if zs}
        # Deterministic order, sort by box_id so dialog reopen always
        # picks the same canonical TC even if the dict iterates in a
        # different order between runs.
        configured = sorted([bid for bid, tc in all_tcs.items()
                             if tc.has_dlc() or tc.has_blob()
                             or tc.user_applied])
        if not configured:
            return {"zones": zones_per_box} if zones_per_box else {}
        first = all_tcs[configured[0]]
        ov = getattr(self, "_tracking_dialog_globals", None) or {}
        # The TC's ``tracker_type`` field is the canonical mode the user
        # picked ("dlc" / "sleap" / "blob"). Use it directly rather than
        # ``first.has_dlc()`` so SLEAP mode (no DLC model) reloads correctly.
        tracker_type = (first.tracker_type or "blob").lower()
        mode = tracker_type
        annotate_saved = dict(ov.get("annotate_saved") or {})
        # Seed from the box's own TrackingConfig, then override below.
        #
        # This dict used to be hand-written from end to end, and it listed
        # none of the pose-input, crop, SLEAP or DLC-engine fields, so every
        # one of them came back at its default the next time the dialog was
        # opened, with nothing logged. The config already carries all of them
        # under the SAME names the settings panel saves and restores them by,
        # so passing them through is what closes the hole for good: a field
        # added to the config reaches the dialog without anyone remembering
        # to add a line here.
        #
        # ``zones`` is dropped because zone geometry travels beside this dict
        # (``out["zones"]``), per box; two copies is a chance for them to
        # disagree. Keys the panel does not know are ignored by the replay.
        tracking = {k: v for k, v in first.to_json().items() if k != "zones"}
        tracking.update({
            "mode":             "normal" if mode == "blob" else mode,
            "tracker_type":     mode,
            "model_path":       first.dlc_model_path or "",
            "body_parts":       list(first.keypoint_names),
            "dlc_confidence":   float(first.confidence_threshold),
            "dlc_resize":       float(first.pose_resize_factor),
            "pose_instances":   int(first.pose_n_instances),
            "threshold":        int(first.blob_threshold),
            "min_area":         int(first.blob_min_area),
            "max_area":         int(first.blob_max_area),
            "detect_dark":      bool(first.blob_detect_dark),
            "blur_mode":        first.blob_blur_mode,
            "blur_kernel_size": first.blob_blur_kernel_size,
            "bg_mode":          first.blob_bg_mode,
            "open_kernel_size": first.blob_open_kernel_size,
            "close_kernel_size": first.blob_close_kernel_size,
            # Without these the Simple options row would come back blank on
            # every dialog reopen (the hole SLEAP's options row still has).
            "self_norm_ratio":  float(first.blob_self_norm_ratio),
            "self_norm_sigma":  float(first.blob_self_norm_sigma),
            "self_norm_smooth_sigma": float(first.blob_self_norm_smooth_sigma),
            "self_norm_minsize": int(first.blob_self_norm_minsize),
            "smooth_tracking":  bool(first.blob_smooth_tracking),
            "annotate_saved":   annotate_saved,
            "annotate_parts":   list(ov.get("annotate_parts") or []),
            "zone_body_part":   ov.get("zone_body_part") or "",
            "marker_size":      int(ov.get("marker_size", 4) or 4),
            "enabled_boxes":    list(configured),
            "coord_mapping":    dict(ov.get("coord_mapping") or {}),
            "triggers":         list(ov.get("triggers") or []),
            # MCU push gates + zone_changed body-part picker, reopen
            # the dialog with the saved values still selected.
            "push_zones_to_mcu":     bool(first.push_zones_to_mcu),
            "push_coords_to_mcu":    bool(first.push_coords_to_mcu),
            "push_frame_event":      bool(getattr(first, "push_frame_event", False)),
            "zone_change_body_part": str(first.zone_change_body_part or "centroid"),
            # Tell the dialog whether the live PoseSink is already
            # initialised for this configuration so the Init button
            # comes up DISABLED (not "constantly active"). The dialog
            # re-enables it the moment one of (model_path, resize,
            # confidence, instances) changes.
            "dlc_initialized": self._pose_already_loaded_for(first),
        })
        out = {
            "tracking": tracking,
            "zones": zones_per_box,
            "coord_mapping": dict(ov.get("coord_mapping") or {}),
        }
        scale_map = ov.get("scale") or {}
        if scale_map:
            out["scale"] = dict(scale_map)
        return out

    def _detect_gpu_encoding(self):
        """Probe NVENC / FFmpeg availability in a background thread.

        Sets ``self.gpu_encoding_available`` + ``self.ffmpeg_available``
        once the probe completes. Runs off the GUI thread so startup is
        not blocked.
        """
        import threading
        self.gpu_encoding_available = False
        self.ffmpeg_available = False

        def _detect():
            try:
                from source.video.recording.ffmpeg import EncoderCapabilities
                caps = EncoderCapabilities.get_instance()
                caps.detect_all()
                self.gpu_encoding_available = caps.has_gpu
                self.ffmpeg_available = caps.has_ffmpeg
                if caps.has_gpu:
                    logger.info("GPU encoding (NVENC) available, will use for video recording")
                elif caps.has_ffmpeg:
                    logger.info("FFmpeg available (CPU encoding), will use libx264")
                else:
                    logger.info("FFmpeg not available, will fall back to OpenCV writer")
                # One consolidated GPU-capability line (encode + pose + platform).
                # allow_inference_import=False so this never pays torch/TF's
                # import cost, pose device is logged for real at model init.
                try:
                    from source.video import gpu as _gpu
                    _gpu.log_summary(allow_inference_import=False)
                except Exception:
                    pass
            except Exception as e:
                logger.warning("GPU encoder detection error: %s", e)

        threading.Thread(target=_detect, daemon=True).start()

    def _init_default_data_dir(self):
        """Create the default data directory + open the per-GUI log file.

        ONE log per GUI launch lives at
        ``data/log/<APP>_<YYYY-MM-DD>_pid<N>.log`` and captures
        EVERYTHING the process logs from startup to quit. No per-session
        log handlers, sessions are short-lived; this single per-GUI
        file already carries the full operator narrative including the
        context around every recording.
        """
        try:
            from source.log import set_log_directory
            data_dir = self._data_dir_path()
            data_dir.mkdir(parents=True, exist_ok=True)
            log_dir = data_dir / "log"
            set_log_directory(str(log_dir))
            try:
                if 'dir' in getattr(self, "info_fields", {}):
                    self.info_fields['dir'].setText(str(data_dir))
            except (AttributeError, RuntimeError):
                pass
            logger.info("Default data directory: %s", data_dir)
        except Exception as e:
            logger.error("Error preparing default data directory: %s", e)

    def _init_working_config(self) -> None:
        """Create the in-memory working (draft) Config at launch.

        A draft exists from launch so that edits made while CREATING a
        project, the non-widget prefs (tracking toggle, browsed data dir),
        have something to be written into. Their writers bail on ``None``, so
        with no draft they would silently no-op and be lost. The draft
        gives those a home; ``read_ui_into_config`` seeds from it on Save,
        so everything transfers intact when the project is named.

        Cost: builds ONE dataclass, no timers, threads, or I/O. It does
        NOT touch the record/run hot path. ``_active_project_dir`` /
        ``active_config_path`` stay empty, so autosave and all disk writes
        remain OFF until the user names the project (first Save).
        """
        if getattr(self, "_active_config", None) is not None:
            return
        try:
            from source.config.experiment import new_experiment
            mode = (self._load_mode_name()
                    if hasattr(self, "_load_mode_name") else "operant")
            self._active_config = new_experiment(mode=mode)
        except Exception as e:
            logger.debug("draft working-config init failed: %s", e)
            return
        # Draft SnapshotStore in a temp workspace so HD / DLC sources
        # captured WHILE CREATING (before the project is named) are staged
        # immediately and materialised into the named folder on first Save.
        # __init__ only computes paths + makes one empty temp dir, no hot
        # path, no per-frame work.
        try:
            import tempfile
            from source.config.snapshot_store import SnapshotStore
            self._draft_store_dir = Path(tempfile.mkdtemp(prefix="pblab_draft_"))
            self._snapshot_store = SnapshotStore(
                self._draft_store_dir,
                tracking_enabled=getattr(self._active_config.meta,
                                         "tracking_enabled", True))
            self._is_draft = True
        except Exception as e:
            logger.debug("draft snapshot store init failed: %s", e)
            self._draft_store_dir = None
            self._is_draft = False

    # ==================================================================
    # Zone-button wiring helper (shared by both modes).
    # ==================================================================

    def _wire_zone_buttons(self, widget, setup_id: int) -> None:
        """Wire all 9 per-widget zone-adjust signals to the shared
        ``_shift_zones`` / ``_rotate_zones`` / ``_scale_zones`` / ``_reset_zones_to_baseline``
        handlers in this base class.

        The widget's ``zone_step_value()`` is the typed step value used
        by all three operations: shift = N pixels, rotate = N degrees,
        scale = ±N percent. After wiring, the per-box zone buttons are
        enabled if zones already exist.
        """
        widget.zone_left.connect(
            lambda sid, w=widget: self._shift_zones(sid, -w.zone_step_value(), 0))
        widget.zone_right.connect(
            lambda sid, w=widget: self._shift_zones(sid, w.zone_step_value(), 0))
        widget.zone_up.connect(
            lambda sid, w=widget: self._shift_zones(sid, 0, -w.zone_step_value()))
        widget.zone_down.connect(
            lambda sid, w=widget: self._shift_zones(sid, 0, w.zone_step_value()))
        widget.zone_rotate_cw.connect(
            lambda sid, w=widget: self._rotate_zones(sid, w.zone_step_value()))
        widget.zone_rotate_ccw.connect(
            lambda sid, w=widget: self._rotate_zones(sid, -w.zone_step_value()))
        widget.zone_zoom_in.connect(
            lambda sid, w=widget: self._scale_zones(sid, 1.0 + w.zone_step_value() / 100.0))
        widget.zone_zoom_out.connect(
            lambda sid, w=widget: self._scale_zones(sid, 1.0 - w.zone_step_value() / 100.0))
        widget.zone_home.connect(lambda sid: self._reset_zones_to_baseline(sid))
        widget.marker_size_changed.connect(self._set_marker_size)
        # Show what the project restored, without emitting: this is not an
        # operator edit and must not mark the project dirty on load.
        try:
            row = getattr(widget, "zone_adjust_row", None)
            if row is not None:
                row.set_marker_size(self.marker_size_for(setup_id))
        except Exception as e:
            logger.debug("marker size restore (box=%s): %s", setup_id, e)
        try:
            # Require ZONES TO ACTUALLY EXIST, not just a key entry.
            # Closing the tracking dialog without drawing any zones used
            # to leave ``tracking_zones[bid] = []``, empty list is still
            # ``in dict`` so the buttons turned on with no zones to adjust.
            zones = getattr(self, "tracking_zones", {}).get(setup_id) or []
            widget.set_zones_enabled(bool(zones))
        except Exception:
            pass

    # ---- annotation marker size --------------------------------------
    # Per box, because two boxes can be at different distances from a shared
    # camera and a radius that reads well on one is a blob on the other.
    # Purely how the overlay is DRAWN: it never reaches a model, a sink or
    # the recorder, which is why it is adjustable while a run is going.

    MARKER_SIZE_DEFAULT = MARKER_SIZE_DEFAULT

    def marker_size_for(self, setup_id) -> int:
        """This box's annotation radius in pixels."""
        try:
            store = getattr(self, "_marker_sizes", None) or {}
            return int(store.get(int(setup_id), MARKER_SIZE_DEFAULT))
        except (TypeError, ValueError):
            return MARKER_SIZE_DEFAULT

    def _set_marker_size(self, setup_id, size) -> None:
        """Operator changed the radius: store it, persist it, redraw."""
        try:
            size = max(1, min(20, int(size)))
            if not hasattr(self, "_marker_sizes"):
                self._marker_sizes = {}
            if self._marker_sizes.get(int(setup_id)) == size:
                return
            self._marker_sizes[int(setup_id)] = size
        except (TypeError, ValueError) as e:
            logger.debug("marker size ignored (box=%s, %r): %s", setup_id, size, e)
            return
        # Debounced autosave, the same trigger every other per-box edit uses.
        try:
            self._project_changed(reason="marker_size")
        except Exception as e:
            logger.debug("marker size autosave (box=%s): %s", setup_id, e)

    # ==================================================================
    # Shared pose / DLC lifecycle.
    # ==================================================================
    # State referenced by these methods. Subclasses must set these dicts
    # in __init__ (or rely on the safe ``setdefault`` calls below).
    #   self.pose_configs       box_id -> config dict
    #   self._overlay           box_id -> OverlayState (pose + blob + ts)
    #   self.pose_zone_state    box_id -> {zone_name: bool}
    #   self.tracking_enabled   box_id -> bool
    #   self.tracking_zones     box_id -> [zone, ...]

    # _pose_resolve_model_path, _pose_after_configured,
    # _pose_after_disabled, _configure_pose_for_box, _enable_pose_for_box,
    # _disable_pose_for_box, _shutdown_pose, _handle_dlc_init_from_dialog,
    # _pose_after_dialog_init, all in pose_subsystem.PoseSubsystemMixin.

    # ==================================================================
    # MCU row mirror, installed once per box at record start.
    # ==================================================================

    def _attach_mcu_row_mirror(self, setup_id) -> None:
        """Attach a ``McuRowMirror`` consumer to the box's pyboard so
        every MCU STATE/EVENT message batch is mirrored into the box's
        ``_video_data.txt`` writer (state/events columns).

        Mode-shared. Both modes reach it through ``open_session_mcu_tsv``,
        which installs the mirror before the MCU file open. The mirror
        does NO clock work, per-frame MCU timestamps are read straight
        off ``pycboard.timestamp`` by RecorderSink.
        """
        try:
            widget = self._setup_widget_for(setup_id)
        except (AttributeError, NotImplementedError):
            return
        if widget is None or getattr(widget, "pycboard", None) is None:
            return
        from source.video.recording.mcu_row_mirror import McuRowMirror
        pyc = widget.pycboard
        if not hasattr(pyc, "data_consumers") or pyc.data_consumers is None:
            pyc.data_consumers = []
        # Tear down any previous mirror from an aborted session.
        old = getattr(widget, "_mcu_row_mirror", None)
        if old is not None and old in pyc.data_consumers:
            try:
                pyc.data_consumers.remove(old)
            except ValueError:
                pass
        mirror = McuRowMirror(
            writer_provider=lambda w=widget: getattr(w, "tracking_writer", None),
            # Late-bound so a Reset/Upload mid-session that swaps
            # ``sm_info`` is picked up on the next message batch.
            sm_info_provider=lambda w=widget: (
                getattr(w.pycboard, "sm_info", None)
                if getattr(w, "pycboard", None) is not None else None),
        )
        widget._mcu_row_mirror = mirror
        pyc.data_consumers.append(mirror)

    # ==================================================================
    # Per-box session bookkeeping, entry points only
    # ==================================================================
    #
    # ``start_box_recording`` / ``stop_box_recording`` are the unified
    # entry points both modes call when a record session begins / ends.
    # Pure hooks, no on-disk side effects, no bookkeeping dict
    # ("is box N recording" is answered by ``recording_boxes`` /
    # ``any_box_running()``).

    def start_box_recording(self, setup_id: int) -> None:
        """Mark a recording session as active for ``box_id``. Pure
        bookkeeping, logs that the box is recording so other GUI code can
        branch on it. **No on-disk side effects.** No-ops on a dry run
        (no subject_id set).
        """
        try:
            widget = self._setup_widget_for(setup_id)
        except (AttributeError, NotImplementedError):
            return
        if widget is None:
            return
        subject_id = ""
        sid_edit = getattr(widget, "subject_id_edit", None)
        if sid_edit is not None:
            try:
                subject_id = sid_edit.text().strip()
            except Exception:
                subject_id = ""
        if not subject_id:
            return
        logger.info("Box %s: recording started (subject=%s)", setup_id, subject_id)

    def _on_record_stopped(self, setup_id, data=None):
        """ONE stop-cleanup slot for both modes, user click, duration-end,
        MCU error or serial drop all converge here (via the widget's
        stopped signal / _after_stop). Idempotent.

        Ordering is load-bearing:
          1. ``_pre_record_stopped``: maze mirrors the synthetic
             ``framework_stopped`` event into _video_data.txt while the
             tracking writer is STILL OPEN.
          2. ``stop_tracking_for_box``: live inference off before the
             writer teardown.
          3. ``_stop_recording_for_box``: recorder + tracking-writer
             close + FW-bridge uninstall + history row (the sole closer).
          4. ``_post_record_stopped``: mode extras (maze: close the MCU
             data file; operant: re-arm Record/Stop + live-status line).
          5. ``stop_box_recording``: per-box bookkeeping hook.
          6. ``refresh_ui_state`` in ``finally``: a failure above must
             not leave the UI looking like the run is still going.
        """
        try:
            self._pre_record_stopped(setup_id, data)
            try:
                self.stop_tracking_for_box(setup_id)
            except Exception as e:
                logger.debug("Box %s: stop_tracking_for_box failed: %s",
                             setup_id, e)
            try:
                self._stop_recording_for_box(setup_id)
            except Exception as e:
                logger.debug("Box %s: _stop_recording_for_box failed: %s",
                             setup_id, e)
            self._post_record_stopped(setup_id, data)
            try:
                self.stop_box_recording(setup_id)
            except Exception as e:
                logger.debug("Box %s: stop_box_recording failed: %s",
                             setup_id, e)
        except Exception as e:
            logger.warning("Box %s: stop cleanup error: %s", setup_id, e)
        finally:
            try:
                self.refresh_ui_state()
            except Exception:
                pass

    def _pre_record_stopped(self, setup_id, data) -> None:
        """Mode hook BEFORE any teardown, runs while the tracking writer
        is still open. Default no-op."""

    def _post_record_stopped(self, setup_id, data) -> None:
        """Mode hook after the recorder/writer teardown. Default no-op."""

    def stop_box_recording(self, setup_id: int) -> None:
        """Stop hook (idempotent, no on-disk side effects). The actual
        writer-close paths live in the widget / mode teardown."""

    def _drop_box_state(self, setup_id) -> None:
        """Drop EVERY per-box entry for a removed box, the union of what
        BOTH modes keep, not the subset the current one happens to own.
        Operant does not track zone paths/baselines/pose configs and maze
        does not track tracking_zones/_overlay/tracking_enabled, so a
        per-mode clear leaks the other mode's state. Every pop is a no-op when
        the key is absent."""
        self._box_widgets.pop(setup_id, None)
        for attr in ("tracking_zones", "tracking_zone_paths",
                     "tracking_enabled", "tracking_enhancers",
                     "_zone_baselines", "pose_configs", "pose_zone_state",
                     "_overlay", "_recording_ctx",
                     "video_stream_widget_cache"):
            d = getattr(self, attr, None)
            if isinstance(d, dict):
                d.pop(setup_id, None)
        try:
            self.recording_setups.discard(setup_id)
        except AttributeError:
            pass

    # ==================================================================
    # Metadata loading, mode-shared
    # ==================================================================
    #
    # These base methods are the single source of truth; operant's sidebar
    # buttons + maze's toolbar entry both call into them.

    def load_cohort_metadata(self, *, offer_assign: bool = True) -> bool:
        """Open the Excel/CSV picker and load a cohort into
        ``self.metadata_manager``. Shared by both modes.

        After a successful load, the file is copied into
        ``<project>/metadata/`` and the path is recorded on
        ``cfg.meta.metadata_file`` via ``_project_adopt_metadata_file``.

        ``offer_assign`` (default True) pops the "Open the subject-assignment
        dialog now?" follow-up. Callers already inside an assign flow (the
        in-context "Load Metadata…" prompt) pass False so it isn't asked twice.
        """
        mm = getattr(self, "metadata_manager", None)
        if mm is None:
            logger.warning("load_cohort_metadata: no MetadataManager on host")
            return False
        if not mm.load_metadata(None, self):
            return False
        adopt = getattr(self, "_project_adopt_metadata_file", None)
        if callable(adopt):
            try:
                adopt(mm.metadata_filename)
                self._project_changed(reason="metadata_loaded")
            except Exception as e:
                logger.debug("metadata adopt-to-project failed: %s", e)
        # Mode-specific button gating, if any.
        updater = getattr(self, "update_metadata_button_states", None)
        if callable(updater):
            try:
                updater()
            except Exception:
                pass
        if offer_assign:
            reply = QtWidgets.QMessageBox.question(
                self, "Assign Subjects",
                "Open the subject-assignment dialog now?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.Yes)
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                try:
                    mm.assign_subjects_from_metadata(self)
                except Exception as e:
                    logger.warning("assign_subjects_from_metadata failed: %s", e)
        return True

    def _setup_blob_tracker_for_box(self, setup_id: int, settings: dict,
                                       *, attach_enhancer: bool = False) -> bool:
        """Centralised blob bringup, the SINGLE blob param/BG install for
        BOTH modes. Called by:

          * ``operant._dialog_start_blob_tracking`` (dialog Start path)
          * ``_prepare_test_tracking_for_box`` (test-preview path)
          * ``_start_tracking_for_box_impl`` blob branch (record path) for
            operant AND maze, neither keeps a parallel copy.

        Steps (each best-effort; failures logged but not raised so the
        record / preview can still proceed without blob inference):

          1. Load the blob background, trying in order: the canonical
             folder location (``_background_path_for_box`` →
             ``<project>/background_images/box<N>.png``), then the TC's
             ``blob_background_path``, then, if neither exists, the live
             camera frame as a last-resort reference.
          2. Push the full blob config (threshold / area / detect_dark /
             blur / morph / CLAHE / adaptive-threshold) into
             ``tracker.update_params``.
          3. Optionally attach a Kalman enhancer via the
             ``_attach_kalman_enhancer`` hook (maze implements it; operant
             has none → no-op).
          4. Run a one-shot ``auto_initialize_tracker`` against the latest
             frame so the very next frame already has a tracker ready.

        Returns True if a background was loaded (file or live frame).
        """
        pipe = getattr(self, "pipeline", None)
        if pipe is None:
            return False
        try:
            tc = pipe.get_tracking_config(setup_id)
        except Exception as e:
            logger.debug("Box %s: get_tracking_config failed: %s", setup_id, e)
            return False
        if tc.has_dlc():
            # Pose mode, caller should route to _enable_pose_for_box.
            return False

        tm = getattr(self, "tracker_manager", None)
        if tm is None:
            return False

        import os as _os

        bg_loaded = False
        # 1a. Canonical folder location, both modes save the captured
        # background to <project>/background_images/box<N>.png.
        try:
            scan = self._background_path_for_box(setup_id)
            if scan is not None and _os.path.exists(str(scan)):
                bg_loaded = bool(tm.load_background(setup_id, str(scan)))
                if bg_loaded:
                    logger.info("Box %s: blob BG loaded from %s", setup_id, scan)
        except Exception as e:
            logger.debug("Box %s: BG folder-scan load failed: %s", setup_id, e)
        # 1b. Explicit TC path, background at TrackingConfig.blob_background_path.
        if not bg_loaded:
            bg_path = tc.blob_background_path
            if bg_path and _os.path.exists(bg_path):
                try:
                    import cv2 as _cv2
                    bg_image = _cv2.imread(bg_path)
                    if bg_image is not None:
                        tm.set_background(setup_id, bg_image)
                        bg_loaded = True
                        logger.info("Box %s: blob BG loaded from %s",
                                    setup_id, bg_path)
                except Exception as e:
                    logger.warning("Box %s: BG load failed: %s", setup_id, e)
        # 1c. Live-frame fallback, grab the current camera frame as the
        # reference so a missing BG file doesn't leave the tracker blind.
        if not bg_loaded and getattr(self, "video_manager", None):
            try:
                frame = self.video_manager.get_last_frame(setup_id)
                if frame is not None:
                    tm.set_background(setup_id, frame)
                    bg_loaded = True
                    logger.info("Box %s: using live frame as blob BG", setup_id)
            except Exception as e:
                logger.debug("Box %s: live-frame BG fallback failed: %s",
                             setup_id, e)

        # 1d. Validate the loaded reference against the view it will be
        # subtracted from. A background of a different shape is not merely
        # stale, resizing it mis-registers every arena edge, so the diff
        # lights up along the walls and the tracker locks onto the mismatch
        # instead of the animal. Report it; the caller decides whether to
        # proceed (see _blob_background_problem).
        if getattr(self, "_bg_shape_problem", None) is None:
            self._bg_shape_problem = {}
        self._bg_shape_problem.pop(setup_id, None)
        if bg_loaded and tc.blob_bg_mode != "self_norm":
            try:
                bg_img = getattr(tm, "backgrounds", {}).get(setup_id)
                frame = (self.video_manager.get_last_frame(setup_id)
                         if getattr(self, "video_manager", None) else None)
                if bg_img is not None and frame is not None:
                    code = _bg_mod.status(
                        background=bg_img, frame_shape=frame.shape[:2],
                        captured_at=self._bg_captured_at(setup_id))
                    if code == _bg_mod.MISMATCH:
                        msg, _ = _bg_mod.describe(
                            code, background=bg_img,
                            frame_shape=frame.shape[:2])
                        self._bg_shape_problem[setup_id] = msg
                        logger.warning("Box %s: %s", setup_id, msg)
            except Exception as e:
                logger.debug("Box %s: BG validation failed: %s", setup_id, e)

        tracker = None
        try:
            tracker = tm.get_tracker(setup_id)
            if tracker is None and hasattr(tm, "create_tracker"):
                tracker = tm.create_tracker(setup_id)
        except Exception:
            tracker = None
        if tracker is not None and settings:
            try:
                # Push the FULL saved blob config, not just threshold/area.
                # A background image is compulsory for blob, so the saved
                # bg_mode (default "static" = absdiff vs that reference) MUST
                # reach the tracker; otherwise it silently runs its class
                # default ("running_avg"), which drifts off the captured
                # background and can fade a still animal into it. blur / morph
                # are needed so the live frame is preprocessed identically to
                # the background. (update_params re-prepares background_gray
                # when blur changes, so order vs set_background is safe.)
                tracker.update_params(
                    threshold=settings.get("threshold"),
                    min_area=settings.get("min_area"),
                    max_area=settings.get("max_area"),
                    detect_dark=settings.get("detect_dark"),
                    blur_mode=settings.get("blur_mode"),
                    blur_kernel_size=settings.get("blur_kernel_size"),
                    bg_mode=settings.get("bg_mode"),
                    open_kernel_size=settings.get("open_kernel_size"),
                    close_kernel_size=settings.get("close_kernel_size"),
                    use_clahe=settings.get("use_clahe"),
                    clahe_clip_limit=settings.get("clahe_clip_limit"),
                    clahe_tile_size=settings.get("clahe_tile_size"),
                    use_adaptive_threshold=settings.get("use_adaptive_threshold"),
                    # Restoring a saved ratio pins it, so the tracker does not
                    # re-derive one from the first frame it happens to see.
                    self_norm_ratio=settings.get("self_norm_ratio"),
                    self_norm_sigma=settings.get("self_norm_sigma"),
                    self_norm_smooth_sigma=settings.get("self_norm_smooth_sigma"),
                    self_norm_minsize=settings.get("self_norm_minsize"),
                )
            except Exception as e:
                logger.warning(
                    "Box %s: tracker.update_params failed: %s", setup_id, e,
                )

        if attach_enhancer:
            attach_fn = getattr(self, "_attach_kalman_enhancer", None)
            if callable(attach_fn):
                try:
                    attach_fn(setup_id, settings)
                except Exception as e:
                    logger.debug(
                        "Box %s: _attach_kalman_enhancer failed: %s",
                        setup_id, e)

        # Auto-init off the latest frame, best-effort.
        if tracker is not None and getattr(self, "video_manager", None):
            try:
                frame = self.video_manager.get_last_frame(setup_id)
                if frame is not None:
                    ok = tm.auto_initialize_tracker(setup_id, frame)
                    if ok:
                        logger.info(
                            "Box %s: blob tracker auto-init OK", setup_id,
                        )
                    else:
                        logger.info(
                            "Box %s: blob auto-init no-detect (will "
                            "retry on next frame)", setup_id,
                        )
            except Exception as e:
                logger.debug(
                    "Box %s: auto_init failed: %s", setup_id, e,
                )
        return bg_loaded

    def _ensure_session_dirs(self, setup_id: int):
        """Mode-shared resolver, returns ``(session_dir, mcu_dir, video_dir)``
        from the per-box widget. Both modes inherit
        ``RunTask._ensure_session_dirs`` on their box widget, so there is one
        resolution path regardless of mode."""
        try:
            widget = self._setup_widget_for(setup_id)
        except (AttributeError, NotImplementedError):
            widget = None
        if widget is not None and hasattr(widget, "_ensure_session_dirs"):
            return widget._ensure_session_dirs()
        # Last-resort path used by tests without a real box widget.
        from pathlib import Path
        return Path("."), Path("."), Path(".")

    # ==================================================================
    # Unified tracking start / stop, one entry, one exit
    # ==================================================================
    #
    # All four lifecycle triggers go through these two methods:
    #   * Test Tracking button  → start_tracking_for_box(bid, force=True)
    #   * MCU framework start   → start_tracking_for_box(bid)
    #   * Stop / Stop Test btn  → stop_tracking_for_box(bid)
    #   * Framework auto-stop   → stop_tracking_for_box(bid)
    #
    # Mode-specific blob setup (background load, enhancer, tracking
    # writer) lives in subclass overrides of ``start_tracking_for_box``.

    def start_tracking_for_box(self, setup_id, *, force: bool = False) -> bool:
        """Enable live pose / blob inference for one box, ONE entry for
        BOTH modes (neither overrides this).

        Sets ``_test_tracking_dry_run`` for the duration so per-mode hooks
        (e.g. the recording-path writer open) skip disk writes during the
        Test Tracking preview (``force=True``), then dispatches via
        ``_start_tracking_for_box_impl``.
        """
        self._test_tracking_dry_run = bool(force)
        try:
            return bool(self._start_tracking_for_box_impl(setup_id, force=force))
        finally:
            self._test_tracking_dry_run = False

    def _start_tracking_for_box_impl(self, setup_id, *, force: bool = False) -> bool:
        """Dispatch body for ``start_tracking_for_box``.

        ``force=False`` (default) honours ``TrackingConfig.online_tracking_enabled``,
        skip with a log line when the user opted out; the box still
        records video + the per-frame TXT (timestamps + fw_ms anchor)
        with the pose column empty.

        ``force=True`` bypasses the opt-out check, the Test Tracking
        preview path which is purely transient (no recording).

        Dispatch is ``cfg.has_dlc()`` → ``_enable_pose_for_box``; else the
        blob branch (``enable_blob_tracking`` + the shared
        ``_setup_blob_tracker_for_box`` + zone manager + MCU push policy
        + the optional Kalman-enhancer hook). The per-session tracking
        writer is NOT opened here, the recording path owns it.

        Idempotent, returns immediately when ``tracking_enabled[box_id]``
        is already True so a duplicate click or signal doesn't churn
        the model.
        """
        pipe = getattr(self, "pipeline", None)
        if pipe is None:
            return False
        try:
            tc = pipe.get_tracking_config(setup_id)
        except (AttributeError, RuntimeError) as e:
            logger.debug("get_tracking_config(%s) error: %s", setup_id, e)
            return False
        if not force and not getattr(tc, "online_tracking_enabled", True):
            logger.info(
                "Box %s: online tracking opted out, recording video + "
                "TXT (timestamps + fw_ms only, no pose).", setup_id,
            )
            return False
        if self.tracking_enabled.get(setup_id, False):
            return True

        # Dispatch on the OPERATOR'S CHOICE (tracker_type), not on which
        # artifacts happen to be present. Branching on has_dlc()/has_blob()
        # instead runs BLOB on a box the operator set to DLC whose model path
        # never committed, "I configured DLC, it shows blob". Pose types map
        # to the pose path with their own mode string, so SLEAP is never forced
        # through the DLC backend, and an explicit pose choice with no model is
        # surfaced rather than quietly downgraded.
        tt = (getattr(tc, "tracker_type", "") or "").lower().strip()
        pose_types = ("dlc", "sleap", "sleap-nn", "sleap_nn")

        if tt in pose_types:
            if tc.dlc_model_path:
                mode = "sleap" if tt.startswith("sleap") else "dlc"
                settings = self.pose_settings_for_box(setup_id)
                # Starting a run must not build a model. Readiness has already
                # done that, at project load or when the camera came up, so
                # the normal path here is one enable_pose call and the first
                # frame goes straight through. If the model is NOT the one
                # this box wants, say which settings differ rather than
                # silently paying for a rebuild with the animal in the box.
                if not pose_ready_for(self.pipeline, settings):
                    changed = pose_settings_differ(
                        self.pipeline.pose_fingerprint(), settings)
                    logger.warning(
                        "Box %s: loading the %s model at run start because %s "
                        "changed since it was prepared. This costs seconds on "
                        "the first frame; initialise from Tracking Config to "
                        "avoid it.", setup_id, mode.upper(),
                        ", ".join(changed) or "the model")
                logger.info(
                    "Box %s: starting %s pose tracking "
                    "(model=%s, body_parts=%s, conf=%.2f)",
                    setup_id, mode.upper(), tc.dlc_model_path,
                    list(tc.keypoint_names), float(tc.confidence_threshold),
                )
                ok = bool(self._enable_pose_for_box(setup_id, settings))
                if not ok:
                    self._note_tracking_start_failure(
                        setup_id,
                        f"{mode.upper()} model failed to start "
                        "(see log for details)")
                return ok
            # Pose chosen but no model committed. Fall back to blob ONLY when
            # a real blob BACKGROUND exists, zones alone do not make a usable
            # blob tracker, and the reported bug was exactly a DLC box with
            # zones drawn but the model path lost silently running blob. With
            # no background, surface the misconfiguration loudly.
            has_bg = bool(getattr(tc, "blob_background_path", "") or "")
            if not has_bg:
                logger.warning(
                    "Box %s: %s selected but no model is configured, open "
                    "Track Config and pick a model.", setup_id, tt.upper())
                self._note_tracking_start_failure(
                    setup_id,
                    f"{tt.upper()} selected but no model configured, "
                    "open Track Config")
                return False
            logger.warning(
                "Box %s: %s selected but no model configured, falling back "
                "to the configured blob background.", setup_id, tt.upper())

        # Blob branch: reached for an explicit blob choice, the pose-without-
        # model-but-background fallback, or a legacy TC with no explicit type.
        # In every case the box must actually have blob input (background or
        # zones), a bare "blob" with nothing set would just track noise, so
        # treat it as unconfigured and fall through to the skip below.
        if tc.has_blob():
            try:
                # enable_blob_tracking creates the per-box tracker in the
                # shared TrackerManager; install the saved params + BG into
                # it RIGHT AFTER so the tracker runs the user's bg_mode /
                # blur / morph / CLAHE instead of class defaults (the default
                # "running_avg" bg_mode drifts off the captured background and
                # loses a still animal). attach_enhancer routes to the per-mode
                # Kalman-enhancer hook (maze implements it; operant no-op).
                pipe.enable_blob_tracking(setup_id)
                self._setup_blob_tracker_for_box(
                    setup_id, self._tracking_config_to_settings_dict(setup_id),
                    attach_enhancer=True)
                # A background that does not match the current view cannot be
                # subtracted from it. Refuse rather than track the mismatch
                # along the arena walls and write it into the session as if it
                # were the animal.
                check = getattr(self, "_blob_background_problem", None)
                problem = check(setup_id) if callable(check) else None
                if problem and not force:
                    pipe.disable_blob_tracking(setup_id)
                    self._on_box_health(setup_id, "blob_background_mismatch")
                    self.showError(
                        f"Box {setup_id}: {problem}") if hasattr(
                            self, "showError") else None
                    logger.error("Box %s: blob tracking blocked, %s",
                                 setup_id, problem)
                    return False
                # Zones + MCU push policy, both modes (maze raises
                # _ZONE_MIN_POINTS to 3, so line/scale zones stay out).
                # Without these the tracker runs but per-zone coord/event
                # push to the MCU never fires.
                zones = (getattr(self, "tracking_zones", None) or {}).get(setup_id)
                if zones:
                    try:
                        self._rebuild_zone_manager(setup_id, zones)
                    except Exception as e:
                        logger.debug(
                            "Box %s: zone manager rebuild failed: %s",
                            setup_id, e)
                try:
                    self._apply_push_policy(setup_id)
                except Exception as e:
                    logger.debug(
                        "Box %s: apply push policy failed: %s", setup_id, e)
                self.tracking_enabled[setup_id] = True
                logger.info(
                    "Box %s: starting blob tracking "
                    "(params + BG installed; no pose model, pose column "
                    "stays empty in _video_data.txt)", setup_id,
                )
                return True
            except (AttributeError, RuntimeError) as e:
                logger.warning("Box %s: enable_blob_tracking failed: %s", setup_id, e)
                self._note_tracking_start_failure(
                    setup_id, "blob tracker failed to start")
                return False

        # Legacy TC: a pose model is present but tracker_type was never set
        # to a pose type (old projects). Honour the model.
        if tc.has_dlc():
            logger.info(
                "Box %s: starting pose tracking (legacy TC, no explicit "
                "tracker_type; model=%s)", setup_id, tc.dlc_model_path)
            ok = bool(self._enable_pose_for_box(
                setup_id, self.pose_settings_for_box(setup_id)))
            if not ok:
                self._note_tracking_start_failure(
                    setup_id,
                    f"{self._pose_backend_label(setup_id)} model failed to start")
            return ok

        # No usable tracker input for this box.
        logger.info(
            "Box %s: tracking skipped, no model and no blob "
            "background/zones configured. _video_data.txt pose column "
            "will be empty for this run.", setup_id,
        )
        self._note_tracking_start_failure(
            setup_id, "nothing configured (no model, no blob background/zones)")
        return False

    def _note_tracking_start_failure(self, setup_id, reason: str) -> None:
        """Record why a box failed to start tracking, for the aggregated
        Test-Tracking failure dialog (T3). Cleared at the start of each
        Test-Tracking attempt."""
        store = getattr(self, "_tracking_start_failures", None)
        if store is None:
            store = self._tracking_start_failures = {}
        store[setup_id] = reason

    def stop_tracking_for_box(self, setup_id) -> None:
        """Disable pose + blob inference for one box.  Idempotent.

        Also resets ``self.tracking_enabled[box_id]``; this is the
        single source of truth for the live-tracking-on flag.
        Without the reset, a record-stop would leave the flag stuck
        True and the next Test Tracking toggle would mis-read state
        as "preview already running".

        The model itself stays loaded in ``PoseSink``, the next start
        re-enables instantly.  Overlay clears automatically via the
        renderer's TTL on ``OverlayState.last_seen_ns``.
        """
        pipe = getattr(self, "pipeline", None)
        if pipe is None:
            return
        try:
            self._disable_pose_for_box(setup_id)
        except (AttributeError, RuntimeError) as e:
            logger.debug("_disable_pose_for_box(%s) error: %s", setup_id, e)
        try:
            pipe.disable_blob_tracking(setup_id)
        except (AttributeError, RuntimeError) as e:
            logger.debug("disable_blob_tracking(%s) error: %s", setup_id, e)
        # Clear any per-zone MCU push policy installed at start (symmetric
        # with _apply_push_policy in the start path). Shared by both modes.
        try:
            pipe.reset_push_policy(setup_id)
        except (AttributeError, RuntimeError) as e:
            logger.debug("reset_push_policy(%s) error: %s", setup_id, e)
        # Per-mode Kalman-enhancer teardown (maze implements; operant none).
        detach = getattr(self, "_detach_tracking_enhancer", None)
        if callable(detach):
            try:
                detach(setup_id)
            except Exception as e:
                logger.debug("_detach_tracking_enhancer(%s) error: %s",
                             setup_id, e)
        try:
            self.tracking_enabled[setup_id] = False
        except (AttributeError, TypeError):
            pass

    # ==================================================================
    # Rig init hooks (called by Camera Connect, NOT by a bundled launcher)
    # ==================================================================
    #
    # Each rig-init step is a deliberate user click:
    #   - Connect Boards  → connect pyboards
    #   - Camera Connect  → connect cameras, then (if tracking enabled
    #                        with a configured model) init DLC/SLEAP
    #                        + enable per-box trackers
    # The hook methods below (_rig_init_dlc, _rig_enable_trackers,
    # _rig_tracking_settings, _box_wants_tracking) are called by the Camera
    # Connect success path.

    # Per-box open-run tracking. ``box_id -> run_id`` while a recording
    # is in progress. Populated by ``_open_run`` at Record click, cleared
    # by ``_close_run`` from ``_stop_recording_for_box``.
    # (instance dict; a class-level mutable default would leak run ids
    # across windows and contaminate tests)

    # Project-scoped artifact + change-log store. Instantiated in the
    # project-load chain (after ``_active_project_dir`` is set). None when
    # no project is loaded, every consumer is None-safe.
    _snapshot_store: Optional["SnapshotStore"] = None  # type: ignore[name-defined]
    # Draft state: True between launch and first Save / a project Load. The
    # snapshot store points at a temp workspace; materialised on first Save.
    _is_draft: bool = False
    _draft_store_dir: Optional[Path] = None

    def _materialize_draft_store(self, project_dir) -> None:
        """First Save: move the draft snapshot workspace (``source/``,
        ``_pending/``, ``change_log.jsonl``) into the named project folder
        and re-root the store onto it. Content-addressed ``<djb2>`` files
        merge safely. One-time, on the GUI thread at Save, never on the
        record hot path. No-op when not a draft."""
        if not getattr(self, "_is_draft", False):
            return
        from source.config.snapshot_store import SnapshotStore
        draft_dir = getattr(self, "_draft_store_dir", None)
        pd = Path(project_dir)
        try:
            if draft_dir is not None and Path(draft_dir) != pd:
                _merge_draft_tree(Path(draft_dir), pd)
        except Exception as e:
            logger.warning("draft store materialize (merge) failed: %s", e)
        try:
            cfg = getattr(self, "_active_config", None)
            tk = getattr(getattr(cfg, "meta", None), "tracking_enabled", True)
            self._snapshot_store = SnapshotStore(pd, tracking_enabled=tk)
        except Exception as e:
            logger.warning("draft store re-root failed: %s", e)
        self._is_draft = False
        if draft_dir is not None:
            try:
                import shutil
                shutil.rmtree(draft_dir, ignore_errors=True)
            except Exception:
                pass
            self._draft_store_dir = None

    def _require_saved_project_for_record(self) -> bool:
        """Block recording a REAL run (subject set) until the project is
        named/saved, so its data lands in a tracked ``runs/`` row instead
        of being orphaned. Prompts to save. Returns True when a named
        project exists (or after a successful save), False to abort.

        Discrete (record click), not on the hot path."""
        if (not getattr(self, "_is_draft", False)
                and getattr(self, "_active_project_dir", None) is not None):
            return True
        reply = QtWidgets.QMessageBox.question(
            self, "Save project first",
            "This recording belongs to a project, but the project hasn't "
            "been saved yet.\n\nSave it now so the run is tracked?",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Save,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Save:
            return False
        try:
            self.save_config()
        except Exception as e:
            logger.error("save before record failed: %s", e)
            return False
        return (not getattr(self, "_is_draft", False)
                and getattr(self, "_active_project_dir", None) is not None)

    def _capture_dlc_snapshot(self) -> None:
        """Walk + hash the configured DLC/SLEAP model dir via the
        SnapshotStore; cache the resulting djb2 on cfg.tracking.dlc
        so commit_box_sources reads it with no I/O at Record click.

        Silent no-op when no store, no model path, or tracking disabled.
        """
        store = getattr(self, "_snapshot_store", None)
        cfg   = getattr(self, "_active_config", None)
        if store is None or cfg is None or cfg.tracking is None:
            return
        mode = cfg.tracking.mode
        # DLCConfig / SleapConfig carry the model's filesystem path.
        # Despite the name it holds the filesystem path the user picked, NOT
        # a djb2 hex.
        if mode == "dlc" and cfg.tracking.dlc and cfg.tracking.dlc.model_path:
            sub, path = cfg.tracking.dlc, cfg.tracking.dlc.model_path
        elif mode == "sleap" and cfg.tracking.sleap \
                and cfg.tracking.sleap.model_path:
            sub, path = cfg.tracking.sleap, cfg.tracking.sleap.model_path
        else:
            return
        try:
            manifest_sha = store.capture_dlc(path)
            if manifest_sha:
                # Private cache field, picked up by commit_box_sources at Record.
                setattr(sub, "_cached_manifest_sha", manifest_sha)
        except (AttributeError, RuntimeError) as e:
            logger.warning("snapshot DLC capture failed: %s", e)

    def _is_tracking_configured_with_model(self) -> bool:
        """True iff cfg.tracking is enabled AND a model path is set + file
        exists on disk. Camera Connect uses this to decide whether to
        auto-init DLC/SLEAP after connecting cameras."""
        cfg = self._active_config
        if not (cfg and cfg.tracking and cfg.tracking.enabled):
            return False
        mode = cfg.tracking.mode
        # DLC/SLEAP subconfigs hold the picked filesystem path in
        # ``model_path`` (despite the name, this field holds the path). They
        # have no ``enabled`` field, the rig-level ``cfg.tracking.enabled``
        # is the gate.
        if mode == "dlc":
            return bool(cfg.tracking.dlc and cfg.tracking.dlc.model_path
                        and Path(cfg.tracking.dlc.model_path).exists())
        if mode == "sleap":
            return bool(cfg.tracking.sleap and cfg.tracking.sleap.model_path
                        and Path(cfg.tracking.sleap.model_path).exists())
        return False

    def _rig_init_dlc(self, cfg) -> None:
        """Pre-warm the DLC model on the first tracking-enabled box.

        Mode subclasses may override; default reads the older tracking
        dict via ``_rig_tracking_settings()`` and calls the existing
        ``_configure_pose_for_box`` (which loads + caches the model).
        """
        inner = self._rig_tracking_settings()
        if not inner or inner.get("mode") not in ("dlc", "sleap"):
            return
        for box in cfg.setup_config.boxes:
            if not self._box_wants_tracking(box):
                continue
            if self._configure_pose_for_box(box.setup_number, inner):
                return     # one configure is enough; model is shared

    def _rig_enable_trackers(self, cfg) -> None:
        """Activate per-box tracking via the existing ``_enable_pose_for_box``."""
        inner = self._rig_tracking_settings()
        if not inner:
            return
        for box in cfg.setup_config.boxes:
            if not self._box_wants_tracking(box):
                continue
            try:
                self._enable_pose_for_box(box.setup_number, inner)
            except Exception as e:
                logger.error("Box %s tracker enable failed: %s",
                             box.setup_number, e)

    def start_tracking_init_background(self) -> None:
        """Pre-warm the pose model + enable per-box trackers on a background
        thread (model load is 2-10 s of TensorFlow/PyTorch graph compile).

        No-op unless tracking is configured with an on-disk model. Shared by
        Camera Connect (dialog) and auto-connect-on-load, without it, the
        auto-connect path subscribes the pose sink but never calls
        ``configure_pose_model``, so frames arrive with no model handle and
        pose results stay empty."""
        if not self._is_tracking_configured_with_model():
            return
        cfg = self._active_config
        import threading

        def _bg_init_tracking():
            try:
                self._rig_init_dlc(cfg)
                self._rig_enable_trackers(cfg)
                logger.info("Tracking model initialised + per-box trackers "
                            "enabled (off-GUI-thread).")
                # Snapshot the model manifest once per init (idempotent,
                # same model = same sha → cached lookup on next call).
                try:
                    self._capture_dlc_snapshot()
                except Exception as se:
                    logger.debug("snapshot_capture_dlc: %s", se)
            except Exception as te:
                logger.error("Tracking init (background) failed: %s", te)

        threading.Thread(target=_bg_init_tracking,
                         name="tracking-init", daemon=True).start()

    def _rig_tracking_settings(self) -> dict:
        """Return the inner tracking dict for the launcher.  Default
        implementation reads from per-box ``TrackingConfig``; mode
        subclasses can override if they need a different boxshold."""
        for bid, tc in (self.pipeline.all_tracking_configs() or {}).items():
            if tc.has_dlc() or tc.has_blob():
                return self._tracking_config_to_settings_dict(bid)
        return {}

    def _box_wants_tracking(self, box) -> bool:
        """Decide whether this box should have tracking enabled at launch.

        Default: True (mode subclasses refine).
        """
        return True

    def connect_pyboard_for_box(self, setup_id: int, com_port: str) -> None:
        """Open one box's MCU connection.

        Both modes' box widgets get ``connect_mcu`` from the shared ``RunTask``
        mixin, which is the single implementation that builds the ``Pycboard``,
        so this needs no per-mode fork. It routes to ``connect_mcu`` rather
        than a Connect *button* handler because those toggle: calling
        "connect" on an already-connected box would disconnect it.
        """
        widget = self._setup_widget_for(setup_id)
        if widget is None:
            raise RuntimeError(f"no box widget for box {setup_id}")
        if not hasattr(widget, "connect_mcu"):
            raise RuntimeError(f"box {setup_id} widget has no connect_mcu")
        if getattr(widget, "is_connected", False):
            return                      # already open; connect is idempotent
        widget.connect_mcu(com_port)

    # ==================================================================
    # Run log hooks
    # ==================================================================
    #
    # Record click with subject_id set →
    # ``_open_run(box_id, subject_id)`` appends a row to
    # ``history/<today>.json``. Fires for every non-dry run, regardless
    # of whether video is being saved. Stop converges in
    # ``_stop_recording_for_box``, which calls ``_close_run(box_id)``.
    # All are no-ops when no project is loaded.

    def _open_run(self, setup_id: int, subject_id: str,
                  datetime_now=None) -> Optional[str]:
        """Open a history row for this Record click. Returns the run_id
        or None when no project is loaded / subject_id is empty.

        ``datetime_now`` is the master record-click timestamp. Threading
        it through here aligns the run_id with the MCU TSV / video /
        _video_data.txt filename stem on a single instant, no
        sub-second drift.

        Also pins the project_config snapshot under
        ``<project>/source/configs/<config_djb2>.json`` BEFORE calling
        ``open_run`` so the file exists even if open_run raises
        mid-flight. ``snapshot_config_to_source`` is idempotent, it
        no-ops when the target already exists.
        """
        project_dir = self._active_project_dir
        if self._active_config is None or project_dir is None:
            return None
        try:
            from source.config.experiment import snapshot_config_to_source
            snapshot_config_to_source(self._active_config, project_dir)
        except (AttributeError, RuntimeError) as e:
            logger.warning("snapshot_config_to_source failed: %s", e)
        widget = self._setup_widget_for(int(setup_id))
        pycboard = getattr(widget, "pycboard", None) if widget else None
        run_id = _pw.open_run(project_dir, self._active_config,
                              int(setup_id), subject_id,
                              pycboard=pycboard,
                              datetime_now=datetime_now)
        if run_id is not None:
            self._open_runs[int(setup_id)] = run_id
        return run_id

    def _collect_output_files(self, setup_id: int) -> dict:
        """Return only the paths of the run's output files.
        Hashing on demand (analyzer side) is cheaper than precomputing
        sha256 for every closed run on the recording hot path.

        Returns ``{"mcu_tsv": str|None, "video_mp4": str|None,
        "video_data": str|None}``, entries are None when the file
        wasn't produced.
        """
        widget = self._setup_widget_for(int(setup_id))
        if widget is None:
            return {}
        files: dict = {"mcu_tsv": None, "video_mp4": None, "video_data": None}

        # MCU TSV
        try:
            pyc = getattr(widget, "pycboard", None)
            dl  = getattr(pyc, "data_logger", None) if pyc else None
            if dl is not None:
                for attr in ("full_filename", "filename", "file_path",
                             "tsv_path"):
                    val = getattr(dl, attr, None)
                    if val:
                        files["mcu_tsv"] = str(val)
                        break
        except Exception as e:
            logger.debug("output_files mcu path failed: %s", e)

        # Video recorder. Prefer the live recorder's path; fall back to the
        # path stashed on the widget at encoder-start (``_run_video_path``)
        # so a recorder torn down before stop, e.g. maze's per-stage
        # recorders, doesn't lose the video reference.
        try:
            rec = getattr(widget, "video_recorder", None)
            if rec is not None:
                for attr in ("video_path", "output_path", "filepath"):
                    val = getattr(rec, attr, None)
                    if val:
                        files["video_mp4"] = str(val)
                        break
            if not files.get("video_mp4"):
                stashed = getattr(widget, "_run_video_path", None)
                if stashed:
                    files["video_mp4"] = str(stashed)
        except Exception as e:
            logger.debug("output_files video path failed: %s", e)

        # Tracking writer (_video_data.txt)
        try:
            tw = getattr(widget, "tracking_writer", None)
            if tw is not None:
                for attr in ("path", "filepath", "_filepath", "_path"):
                    val = getattr(tw, attr, None)
                    if val:
                        files["video_data"] = str(val)
                        break
        except Exception as e:
            logger.debug("output_files video_data path failed: %s", e)
        return files

    def _close_run(self, setup_id: int,
                   *, status: str = "completed",
                   files: Optional[dict] = None) -> None:
        """Close the open run for ``box_id`` if any.

        ``files`` is forwarded to ``project_workflow.close_run`` which
        merges it into the history row's ``data_files`` block, the paths
        of every output the run produced (MCU TSV, video, tracking
        outputs), so the analyzer can resolve them without globbing.
        Paths only: see ``_collect_output_files`` for why hashing is left
        to the analyzer rather than done on the recording hot path.
        """
        run_id = self._open_runs.pop(int(setup_id), None)
        if not run_id or self._active_project_dir is None:
            return
        _pw.close_run(self._active_project_dir, run_id,
                      status=status, files=files)
        # NOTE: the history row in history/<date>.json IS the canonical
        # record of run completion (ended_at + duration_s + data_files).
        # We deliberately do NOT mirror run-close to change_log.jsonl,
        # that file is reserved for high-signal events (uploads, commits,
        # tracking-toggle, project saves). Run lifecycle would otherwise
        # produce N noise lines per session for zero added information.

    def _check_unfinished_runs(self) -> None:
        """On project load: prompt about run rows that never recorded an
        MCU TSV path (the run didn't reach close_run, crash or kill).
        Yes stamps ``crashed: true`` on the rows so they stop
        re-prompting on every load."""
        project_dir = self._active_project_dir
        if project_dir is None:
            return
        try:
            unfinished = _pw.scan_unfinished_runs(project_dir)
        except (AttributeError, RuntimeError) as e:
            logger.debug("scan_unfinished_runs failed: %s", e)
            return
        if not unfinished:
            return

        lines = []
        for run_id, row in unfinished:
            subject = row.get("subject_id") or "?"
            task = row.get("task_name") or "?"
            started_at = row.get("started_at") or "?"
            lines.append(
                f"  - {run_id}  (subject={subject}, task={task}, "
                f"started={started_at})")
        msg = (f"{len(unfinished)} previous run(s) in this project never "
               f"recorded their output files, looks like a crash "
               f"mid-recording:\n\n"
               + "\n".join(lines)
               + "\n\nMark them as crashed so they stop appearing here?\n"
               + "(Any data they did write is still on disk, see "
               + "data/temp for dry-run safety copies.)")
        reply = QtWidgets.QMessageBox.question(
            self, "Unfinished runs detected", msg,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            for run_id, _ in unfinished:
                try:
                    _pw.mark_run_crashed(project_dir, run_id)
                except Exception as e:
                    logger.error("mark_run_crashed(%s) failed: %s",
                                 run_id, e)

    def _build_annotate_callback(self, setup_id):
        """Default annotate-callback for Save-with-overlay video mode.

        Returns a ``(frame, capture_ts) -> annotated_frame`` closure that
        the recorder thread invokes per frame. Returns None to keep the
        raw frame. Both maze and operant can opt in via
        ``self._tracking_dialog_globals['annotate_saved'][str(box_id)] = True``
        (set when the user ticks the "Save annotated video" box for that
        box in the unified tracking dialog).
        """
        def _draw(frame, capture_ts):
            try:
                state = self._overlay.get(setup_id)
                if state is None or (not state.has_pose() and not state.has_blob()):
                    return None
                # Staleness, judged against THIS frame rather than against
                # "now". The live tile can ask "now" because it paints the
                # frame it just received; the recorder cannot, because a
                # frame waits in the encoder queue and the answer would then
                # depend on how far behind the encoder is. ``capture_ts`` is
                # the frame's own capture instant on the same monotonic
                # clock as ``last_seen_ns``, and it is the whole reason the
                # callback is handed one.
                #
                # Symmetric, because both directions are wrong: a pose much
                # OLDER than the frame is the stale-overlay case, inference
                # stalled, the model was disabled, or the run ended, and
                # without this the last keypoints stayed burned into every
                # later frame, pinned where the animal no longer was. A pose
                # much NEWER is a queued frame being annotated with a result
                # from well after it, which is the same error mirrored.
                if state.last_seen_ns:
                    age_ns = abs(int(capture_ts * 1e9) - int(state.last_seen_ns))
                    if age_ns > _OVERLAY_MAX_AGE_NS:
                        return None
                out = frame.copy()
                if state.has_pose():
                    conf_thresh = state.confidence_threshold
                    # The SAME colours as the live tile. This drew every
                    # keypoint in one yellow, so the burned-in video could not
                    # be read the way the operator had learnt to read the
                    # screen, and no dot said which part it was.
                    names = list(getattr(state, "body_parts", ()) or ())
                    resolve = getattr(self, "_project_part_colours", None)
                    palette = (resolve(names) if callable(resolve)
                               else part_colour_map(names))
                    fallback = part_colours(len(state.pose) or 1)
                    for i, (x, y, c) in enumerate(state.pose):
                        if c < conf_thresh:
                            continue
                        colour = (palette.get(names[i]) if i < len(names)
                                  else None) or fallback[i % len(fallback)]
                        # round, like every other marker: truncating put
                        # the burned-in dot up and left of the live one
                        # for the same keypoint.
                        centre = (int(round(x)), int(round(y)))
                        cv2.circle(out, centre, 4, colour, -1)
                        cv2.circle(out, centre, 4, (20, 20, 20), 1, cv2.LINE_AA)
                if state.bbox is not None:
                    x, y, w, h = state.bbox
                    cv2.rectangle(out, (int(x), int(y)),
                                  (int(x) + int(w), int(y) + int(h)),
                                  (0, 255, 0), 1)
                    cv2.circle(out, (int(x + w / 2), int(y + h / 2)),
                               5, (0, 0, 255), -1)
                return out
            except Exception:
                return None
        return _draw

    # ==================================================================
    # Shared recording lifecycle
    # ==================================================================
    # The full start path is genuinely mode-specific (maze: stage-based,
    # operant: async framework start with callback). Only the small
    # tail-end pieces are shared, recorder construction args, pipeline
    # registration, async-stop pattern. Both modes call the helpers below
    # rather than reimplementing the boilerplate.

    def open_session_mcu_tsv(self, setup_id, pyboard_dir, subject_id,
                             datetime_now, metadata, *, video_info=None):
        """Open the MCU pyControl TSV for a RECORD run. ONE method, both
        modes (operant's ``_start_recording`` and maze's
        ``start_recording`` both call it).

        Steps, in order:
          1. Install the per-box MCU row mirror BEFORE the file open so the
             first MCU STATE/EVENT batch already lands in the writer (the
             install is idempotent, it tears down any prior one first).
          2. Open the TSV. A failure here PROPAGATES so the shared
             orchestrator aborts the start, the framework never runs on an
             un-recordable session.
          3. Commit the box's staged task/HD source bytes from ``_pending/``
             to ``sources/`` now the TSV exists, never store sources
             for a run the user uploaded but never recorded.
          4. Open the history row, pinned to ``datetime_now`` so the run_id
             matches every artifact filename (MCU TSV, video, frame log).

        ``video_info`` is the no-video header dict the operant TSV embeds
        when no video will record (None when the recorder writes the header
        post-start). Maze leaves it None, pose lives in ``_video_data.txt``,
        not the TSV header.
        """
        widget = self._setup_widget_for(setup_id)
        if widget is None or getattr(widget, "pycboard", None) is None:
            return
        self._attach_mcu_row_mirror(setup_id)
        widget.pycboard.data_logger.open_data_file(
            data_dir=str(pyboard_dir),
            subject_ID=subject_id,
            datetime_now=datetime_now,
            box_ID=setup_id,
            metadata=metadata,
            video_info=video_info,
        )
        logger.info(
            "Box %s: MCU TSV opened for subject '%s'", setup_id, subject_id)
        widget._commit_box_sources_for_run()
        try:
            self._open_run(setup_id, subject_id, datetime_now=datetime_now)
        except Exception as e:
            logger.warning("box %s: _open_run failed: %s", setup_id, e)

    def _video_recorder_geometry(self, setup_id):
        """ONE geometry resolver for both modes' recorders → ``(resolution,
        roi, fps, camera_id, cam_cfg)``.

        Single rule, no per-mode override:
          * frames pre-cropped by a segment processor → the segment size, no
            recorder ROI (avoid a double crop);
          * else, if the box crops its own ROI at the recorder (operant boxes
            that share a camera split by ROI, they expose ``_get_box_roi``),
            the ROI drives the output and resolution is left None so the
            recorder derives it;
          * else the full camera frame (maze: one camera per arena, no ROI).
        """
        vm = self.video_manager
        widget = self._setup_widget_for(setup_id)
        cam_id = vm.box_camera_map.get(setup_id) if hasattr(vm, "box_camera_map") else None
        if cam_id is None:
            cam_text = self._box_camera_id_text(setup_id) or ""
            if str(cam_text).isdigit():
                cam_id = int(cam_text)
        # INVARIANT: exactly one cropper. ``segment_size`` non-None means the
        # FrameBus already cropped this box's slice out of a shared camera, so
        # the recorder must not crop again, hence roi=None on that branch.
        # The recorder's own roi is for the operant path where no segment
        # processor exists and it IS the only cropper; it cannot be deleted in
        # favour of "the bus already crops", because on that branch the bus
        # does not. Pinned by test_recorder_geometry_single_cropper.
        segment_size = self._box_segment_size(setup_id, cam_id)
        if segment_size:
            resolution, roi = segment_size, None
        elif hasattr(widget, "_get_box_roi"):        # operant: ROI-cropped recorder
            resolution, roi = None, self._box_recorder_roi(setup_id)
        else:                                         # maze: full camera frame
            resolution, roi = self._full_camera_wh(cam_id), None
        assert not (segment_size and roi), (
            f"box {setup_id}: bus segment AND recorder roi both set, "
            "the frame would be cropped twice")
        cam_cfg = (vm.resolved_settings_for_box(setup_id)
                   if hasattr(vm, "resolved_settings_for_box") else None)
        fps = self._recording_fps(cam_id, cam_cfg)
        return (resolution, roi, fps, cam_id if cam_id is not None else 0, cam_cfg)

    def _box_segment_size(self, setup_id, cam_id):
        """(w, h) the segment processor delivers for this box, or None when
        the box's frames aren't ROI-segmented."""
        vm = self.video_manager
        seg = (vm.segment_processor_for_box(setup_id)
               if hasattr(vm, "segment_processor_for_box") else None)
        if not seg or setup_id not in getattr(seg, "box_index", {}):
            return None
        shape = None
        try:
            full = vm.get_full_frame(cam_id) if cam_id is not None else None
            if full is not None:
                shape = full.shape
        except Exception:
            shape = None
        return seg.get_segment_size(setup_id, shape) or None

    def _full_camera_wh(self, cam_id):
        """Full (w, h) of the camera feeding this box; (1280, 720) fallback."""
        vm = self.video_manager
        cam = vm.cameras.get(cam_id) if cam_id is not None else None
        if cam is not None:
            return (cam.width, cam.height)
        return (1280, 720)

    def _box_recorder_roi(self, setup_id):
        """The box's recorder-crop ROI (operant) or None (maze arenas have no
        separate ROI, they pre-segment or record the full frame)."""
        widget = self._setup_widget_for(setup_id)
        getter = getattr(widget, "_get_box_roi", None)
        try:
            return getter() if callable(getter) else None
        except Exception:
            return None

    def _tile_fps_facts(self, setup_id, target_fps):
        """``(requested, capable)`` for a box's tile, as two separate facts.

        They were one. "capable" was read from ``selected_fps``, which is what
        the operator PICKED, and then labelled "camera can deliver". A box set
        to 25 fps on a camera that only offers 30 therefore read
        "capable: 25, acquiring: 30", which says the camera is exceeding its
        own capability and tells the reader nothing about which number is
        wrong. The pick and the ceiling are different things.

        "capable" is now what the camera OFFERS at this size, asked of the
        device. It was the trial probe's measurement, and a measurement is the
        camera multiplied by the room: this rig's probe recorded 10 fps at
        1920x1080 on a camera whose descriptors list up to 60, and 5.1 fps on
        one that lists 30. The tile therefore called a 60 fps camera a 10 fps
        camera, and then showed it "acquiring" more than it was "capable" of.
        The measurement survives only for a camera nobody has enumerated.
        """
        requested = capable = 0.0
        try:
            cam_id = self.video_manager.box_camera_map.get(setup_id)
            cfg = (self.pipeline.get_camera_config(cam_id)
                   if cam_id is not None else None)
            if cfg is not None:
                requested = float(cfg.selected_fps or 0.0)
                capable = self._offered_ceiling_for(
                    cam_id, cfg.selected_resolution)
                if capable <= 0:
                    capable = float(
                        cfg.max_fps_for(cfg.selected_resolution) or 0.0)
        except Exception:
            requested = capable = 0.0
        if capable <= 0:
            # Never enumerated and never measured: the pick is the only number
            # there is, and the camera thread's target is the last resort.
            capable = requested or float(target_fps or 0.0)
        return requested, capable

    def _offered_rates_for_box(self, setup_id) -> tuple:
        """Every rate this box's camera reports at its selected size, or ``()``.

        ``()`` means the camera has never been enumerated, which is not the
        same as offering nothing, so the caller must not read it as "the rate
        you picked does not exist".
        """
        try:
            from source.video.cameras import calibration_store as _store
            from source.video.cameras.enumerate_modes import rates_for
            from source.video.cameras.usb_identity import resolve_identity
            cam_id = self.video_manager.box_camera_map.get(setup_id)
            if cam_id is None:
                return ()
            cfg = self.pipeline.get_camera_config(cam_id)
            size = getattr(cfg, "selected_resolution", None) if cfg else None
            if not size:
                return ()
            uid = resolve_identity(cam_id, "opencv").get("unique_id")
            return tuple(rates_for(_store.get_offered(uid), tuple(size)))
        except Exception as e:
            logger.debug("offered rates for box %s: %s", setup_id, e)
            return ()

    def _offered_ceiling_for(self, cam_id, resolution) -> float:
        """Highest rate ``cam_id`` OFFERS at ``resolution``, or ``0.0``.

        ``0.0`` means the camera has never been enumerated, never that it has
        no rates, so the caller falls back rather than showing a zero.

        Memoised against the calibration store's mtime: this runs once a
        second per box from the tile update, so it costs one ``stat`` instead
        of a JSON read, while still seeing a Detect the moment it writes.
        """
        if cam_id is None or not resolution:
            return 0.0
        try:
            from source.video.cameras import calibration_store as _store
            from source.video.cameras.enumerate_modes import rates_for
            from source.video.cameras.usb_identity import resolve_identity
            try:
                stamp = _store.store_path().stat().st_mtime_ns
            except OSError:
                return 0.0
            cache = getattr(self, "_offered_ceiling_cache", None)
            if cache is None or cache.get("stamp") != stamp:
                cache = {"stamp": stamp}
                self._offered_ceiling_cache = cache
            key = (str(cam_id), tuple(resolution))
            if key in cache:
                return cache[key]
            uid = resolve_identity(cam_id, "opencv").get("unique_id")
            rates = rates_for(_store.get_offered(uid), tuple(resolution))
            ceiling = float(max(rates)) if rates else 0.0
            cache[key] = ceiling
            return ceiling
        except Exception as e:
            logger.debug("offered ceiling for camera %s: %s", cam_id, e)
            return 0.0

    def _recording_fps(self, cam_id, cam_cfg):
        """Recording fps: the pipeline's picked capture rate, then the camera
        config target, then the rig default."""
        fps = None
        try:
            if cam_id is not None:
                central = self.pipeline.get_camera_config(cam_id)
                if central and central.selected_fps:
                    fps = float(central.selected_fps)
        except Exception:
            fps = None
        if fps is None:
            fps = (float(cam_cfg.target_fps)
                   if (cam_cfg and getattr(cam_cfg, "target_fps", 0))
                   else float(getattr(self, "video_target_fps", 20) or 20))
        return fps

    def open_session_video_recorder(self, setup_id, video_dir, subject, dt, *,
                                    file_stem=None, tracking_writer=None,
                                    annotate_callback=None, metadata=None):
        """Build + start the per-box mp4 video recorder, register it with the
        frame pipeline, write its video-info block into the open MCU TSV, stash it on
        the box widget, and return it.

        ONE path for BOTH modes, replaces operant ``_start_video_recording`` /
        ``_start_temp_video_recorder`` and maze ``_start_stage_recording`` /
        ``_start_temp_video_recorder``. The only per-mode bit is
        ``_video_recorder_geometry``. ``file_stem`` set → fixed dry-run name.
        ``metadata`` is accepted for caller parity; session metadata lives in
        the MCU TSV header + FrameLog, not the encoder."""
        from source.video.recording.recorder import VideoRecorder
        # First box to start owns the session timer + fresh drop counts.
        if not self._session_started_at:
            self._on_session_started()
        resolution, roi, fps, camera_id, cam_cfg = self._video_recorder_geometry(setup_id)
        # Encoder preference from config: use_gpu + CPU-fallback policy.
        # allow_cpu_fallback keeps a box recording (on CPU) when the NVENC
        # session cap is hit on a many-box rig, instead of losing its video.
        try:
            from source.config import settings as _cfg
            _use_gpu = _cfg.get_setting("video", "use_gpu")
            _allow_cpu = _cfg.get_setting("video", "allow_cpu_fallback")
            _prefer_hevc = _cfg.get_setting("video", "prefer_hevc")
            _crf = _cfg.get_setting("video", "quality_crf")
        except Exception:
            _use_gpu, _allow_cpu, _prefer_hevc, _crf = "auto", True, False, 23
        rec = VideoRecorder(
            camera_id=camera_id if camera_id is not None else 0,
            fps=fps, resolution=resolution or (420, 420), roi=roi,
            use_gpu=_use_gpu if _use_gpu is not None else "auto",
            allow_cpu_fallback=bool(_allow_cpu) if _allow_cpu is not None else True,
            prefer_hevc=bool(_prefer_hevc),
            crf=int(_crf) if _crf is not None else 23,
            frame_strategy=getattr(self, "video_frame_strategy", "accept"))
        logger.info("Box %s: recording at %.2f FPS", setup_id, float(fps))
        if not rec.start_recording(str(video_dir), subject, dt, box_ID=setup_id,
                                   file_stem=file_stem, camera_config=cam_cfg):
            logger.warning("Box %s: video recorder failed to start", setup_id)
            # Sticky red alarm so a box that silently lost video, most often
            # the GPU NVENC session limit on a forced-GPU host, is visible. It
            # persists across start/stop/reconnect; cleared by reset/re-upload or
            # the next successful record (see RunTask.set_box_alarm).
            try:
                from source.video.recording.ffmpeg import EncoderCapabilities
                _gpu = EncoderCapabilities.get_instance().gpu_force_required()
            except Exception:
                _gpu = False
            self._safe_box_call(
                setup_id, "set_box_alarm",
                "No video, GPU encoder session limit? (NVENC); reduce "
                "simultaneous recordings." if _gpu else
                "No video, recorder failed to start.")
            self.refresh_ui_state()
            return None
        # Recording started OK, clear any stale recorder alarm on this box.
        self._safe_box_call(setup_id, "clear_box_alarm")
        # Name the encoder in this box's _video_data.txt header before its first
        # row. header_encoder() is None off a Jetson, so the header there keeps
        # "unknown" exactly as before.
        _enc = rec.header_encoder() if hasattr(rec, "header_encoder") else None
        if (_enc and tracking_writer is not None
                and hasattr(tracking_writer, "write_info")):
            try:
                tracking_writer.write_info("video_encoder", _enc)
            except Exception as e:
                logger.debug("Box %s: encoder header: %s", setup_id, e)
        # Re-arm the one-shot failure modals so a genuine failure in THIS new
        # session pops again (they only ever get .add()-ed otherwise).
        for _attr in ("_encoder_dead_modal_warned", "_pose_failed_modal_warned"):
            warned = getattr(self, _attr, None)
            if warned is not None:
                warned.discard(setup_id)
        # Dropped frames appear inline in _video_data.txt as a #event row.
        if tracking_writer is not None and hasattr(tracking_writer, "on_dropped_frame"):
            rec.drop_callback = tracking_writer.on_dropped_frame
        # Fires only when a Jetson hardware encoder is swapped for libx264.
        if tracking_writer is not None and hasattr(tracking_writer, "note_encoder_change"):
            rec.encoder_changed_callback = tracking_writer.note_encoder_change
        widget = self._setup_widget_for(setup_id)
        self.pipeline.start_recording(
            setup_id, recorder=rec,
            tracking_writer=tracking_writer,
            annotate_callback=annotate_callback,
            pycboard=getattr(widget, "pycboard", None))
        self.recording_setups.add(setup_id)
        path = getattr(rec, "video_path", None)
        # Write the video-info block into this box's open MCU TSV.
        dl = getattr(getattr(widget, "pycboard", None), "data_logger", None)
        if dl is not None and path:
            try:
                from source.video.recording import video_info_from_path
                dl.write_video_info(video_info_from_path(path))
            except Exception as e:
                logger.debug("Box %s: video info write: %s", setup_id, e)
        # Stash on the widget so the shared stop path reaps it.
        if widget is not None:
            widget.video_recorder = rec
            widget.recording_video = True
            widget._run_video_path = str(path or "")
        return rec

    def _tracking_roi(self, setup_id):
        """Per-mode ROI for the tracking-writer header (operant: box ROI;
        maze: arena ROI). Subclasses override; default none."""
        return None

    def open_session_tracking_writer(self, setup_id, video_dir=None,
                                     subject_id=None, dt=None, *, file_path=None,
                                     video_filename=None):
        """Open the per-box ``_video_data.txt`` tracking writer with full
        headers, stash it on the box widget, and return it (or None).

        ONE path for BOTH modes (was operant + maze ``_open_tracking_writer``).
        Idempotent (returns an already-open writer). ``video_dir`` / ``subject_id``
        / ``dt`` are derived from the box widget + session dirs when omitted
        (maze's lazy open-on-pose-config). ``file_path`` forces a fixed path
        (the dry-run temp net). The only per-mode bit is ``_tracking_roi``."""
        from datetime import datetime as _dt
        from source.video.recording.frame_log import (
            build_filepath, open_with_headers, resolve_px_per_m)
        widget = self._setup_widget_for(setup_id)
        existing = getattr(widget, "tracking_writer", None)
        if existing is not None:
            return existing
        if subject_id is None:
            subject_id = (widget.subject_id_edit.text().strip()
                          if widget is not None and hasattr(widget, "subject_id_edit")
                          else "")
        # No subject + not the temp net → no on-disk artifact.
        if not subject_id and file_path is None:
            return None
        if dt is None:
            dt = _dt.now()
        try:
            if file_path is not None:
                # Fixed-path (dry-run) writer: the caller passes the actual
                # temp video name (video_Box<N>.mp4) so the #video_file
                # header doesn't fall back to a derived name that never
                # exists on disk.
                vd_path = file_path
                video_filename = video_filename or ""
            else:
                if video_dir is None:
                    _s, _p, video_dir = self._ensure_session_dirs(setup_id)
                vd_path, video_filename = build_filepath(
                    video_dir, subject_id, setup_id, dt)
            # Tracking mode = the OPERATOR'S CHOICE (tracker_type), matching
            # the start-path dispatch, classifying by which artifacts
            # happened to exist recorded a SLEAP session as "dlc" and a
            # model-less DLC box as "blob".
            tracking_mode = "off"
            try:
                tc = self.pipeline.get_tracking_config(setup_id)
                if tc is not None:
                    tt = str(getattr(tc, "tracker_type", "") or "").lower()
                    if tt in ("dlc", "sleap", "blob"):
                        tracking_mode = tt
                    elif tc.has_dlc():
                        tracking_mode = "dlc"
                    elif tc.has_blob():
                        tracking_mode = "blob"
                    if (tracking_mode != "off"
                            and not getattr(tc, "online_tracking_enabled", True)):
                        tracking_mode = f"{tracking_mode}_offline_only"
            except Exception:
                pass
            wh = self._box_frame_wh(setup_id)
            resolution = ((int(wh[0]), int(wh[1]))
                          if wh and wh[0] > 0 and wh[1] > 0 else None)
            cam_cfg = (self.video_manager.resolved_settings_for_box(setup_id)
                       if hasattr(self.video_manager, "resolved_settings_for_box") else None)
            fps = (float(cam_cfg.target_fps)
                   if (cam_cfg and getattr(cam_cfg, "target_fps", 0) > 0) else None)
            zones = (getattr(self, "tracking_zones", {}) or {}).get(setup_id) or None
            roi = self._tracking_roi(setup_id)
            sm = getattr(getattr(widget, "pycboard", None), "sm_info", None)
            task_name = (widget.task_combo.text()
                         if widget is not None and hasattr(widget, "task_combo") else "")
            px_per_m = resolve_px_per_m(zones, resolution)
            # Feed the same scale to the MCU trigger kinematics (cm/s, mm),
            # the typed TrackingConfig.zones can't carry the calibration length,
            # so this GUI-side px/m is the authoritative source. None ⇒ the
            # extractor stays on body-length units.
            try:
                pipe = getattr(self, "pipeline", None)
                if pipe is not None and hasattr(pipe, "set_features_scale"):
                    pipe.set_features_scale(
                        setup_id, (px_per_m / 1000.0) if px_per_m else None)
            except Exception as e:
                logger.debug("set_features_scale (box=%s): %s", setup_id, e)
            # The names the pose column is actually labelled with, asked of
            # the live sink rather than the dialog, because the sink is what
            # ordered the columns. The skeleton is the operator's / model's
            # drawing edges, which only the config carries.
            bodyparts, skeleton = [], []
            if tracking_mode in ("dlc", "sleap"):
                try:
                    bodyparts = list(self.pipeline.pose_model_info()[0] or ())
                except Exception as e:
                    logger.debug("pose_model_info (box=%s): %s", setup_id, e)
                if not bodyparts and tc is not None:
                    bodyparts = list(getattr(tc, "keypoint_names", ()) or ())
                skeleton = [list(e) for e in
                            (getattr(tc, "skeleton", ()) or ())]
            tw = open_with_headers(
                vd_path, subject_id=subject_id, setup_id=setup_id, start_dt=dt,
                task_name=task_name,
                task_hash=getattr(sm, "task_hash", None) if sm else None,
                hd_name=getattr(sm, "hardware_def_name", "") if sm else "",
                hd_hash=getattr(sm, "hardware_def_hash", None) if sm else None,
                tracking_mode=tracking_mode, resolution=resolution, fps=fps,
                zones=zones, roi=roi, video_filename=video_filename,
                px_per_m=px_per_m, bodyparts=bodyparts, skeleton=skeleton,
                # The drop log is process-wide; only REAL sessions may
                # re-point it (a dry-run rebind sent concurrent boxes'
                # drop rows to data/temp/).
                rebind_drop_log=(file_path is None))
            if widget is not None:
                widget.tracking_writer = tw
            logger.info("Box %s: _video_data.txt opened at %s", setup_id, vd_path)
            return tw
        except Exception as e:
            logger.warning("Box %s: tracking writer open failed: %s", setup_id, e)
            return None

    # Live StopRecorder-* threads; closeEvent joins them so encoder
    # flushes complete before interpreter exit kills the daemons.
    _recorder_stop_threads: ClassVar[list] = []

    @staticmethod
    def _async_stop_recorder(recorder, setup_id) -> None:
        """Stop + close a VideoRecorder on a daemon thread.

        Both modes use this when the user clicks Stop: the recorder owns
        an ffmpeg subprocess that can take ~100-300 ms to flush + close,
        and we don't want the GUI thread blocking on it. Idempotent,
        safe to call with ``None``.
        """
        if recorder is None:
            return

        def _stop():
            try:
                recorder.stop_recording()
                recorder.close()
            except Exception as e:
                logger.error("Box %s: stop_recording error: %s", setup_id, e)

        t = threading.Thread(target=_stop, daemon=True,
                             name=f"StopRecorder-{setup_id}")
        # Registry so closeEvent can JOIN these before the process exits,
        # a daemon reaper killed mid-release() leaves ffmpeg without its
        # stdin close and the mp4 without a moov atom (unplayable).
        reg = MainWindowBase._recorder_stop_threads
        reg[:] = [x for x in reg if x.is_alive()]
        reg.append(t)
        t.start()

    def _stop_recording_for_box(self, setup_id, *,
                                recorder=None,
                                clear_widget_attr: bool = True) -> None:
        """Detach from pipeline + async-stop the recorder. Mode-agnostic.

        Subclasses pass the recorder explicitly (most flows already have
        a reference); when ``recorder`` is None we look at the box widget's
        ``video_recorder`` attribute (operant pattern). Discards the box
        from ``recording_boxes`` and clears the widget attribute when
        ``clear_widget_attr`` is True.
        """
        # Detach from RecorderSink first so no more frames flow into the
        # recorder while we close it. The sink hands back the VideoRecorder it
        # was holding, the authoritative ref to reap, so the ffmpeg
        # child is closed even if widget.video_recorder was already cleared.
        popped_recorder = None
        try:
            popped_recorder = self.pipeline.stop_recording(setup_id)
        except (AttributeError, RuntimeError) as e:
            logger.debug("pipeline.stop_recording(%s) error: %s", setup_id, e)
        try:
            self.recording_setups.discard(setup_id)
        except (AttributeError, RuntimeError):
            pass
        # Hash output files so the history row carries explicit
        # path+sha+size, the analyzer resolves data with zero globbing.
        # Done BEFORE _close_run so the dict can be forwarded.
        out_files = self._collect_output_files(setup_id)
        # Close the history row opened at Record click (no-op if no
        # project was active or this box's run wasn't opened).
        self._close_run(int(setup_id), status="completed", files=out_files)
        try:
            widget = self._setup_widget_for(setup_id)
        except Exception:
            widget = None
        # Reap order: caller's explicit recorder, else the encoder the sink
        # just handed back (authoritative), else the widget attr. Guarantees
        # the ffmpeg child is always closed.
        if recorder is None:
            recorder = popped_recorder
        if recorder is None and widget is not None:
            recorder = getattr(widget, "video_recorder", None)
        self._async_stop_recorder(recorder, setup_id)
        # Close the per-box TrackingWriter + detach the FWAnchor bridge
        # consumer so a re-record on the same box doesn't double-attach.
        # The mirror install/uninstall is owned by the widget (lives
        # with the run lifecycle, not the camera lifecycle).
        if widget is not None:
            try:
                if hasattr(widget, "_uninstall_mcu_row_mirror"):
                    widget._uninstall_mcu_row_mirror()
            except (AttributeError, RuntimeError):
                pass
            try:
                tw = getattr(widget, "tracking_writer", None)
                if tw is not None and hasattr(tw, "close"):
                    tw.close()
                widget.tracking_writer = None
            except Exception:
                pass
        if clear_widget_attr and widget is not None:
            try:
                widget.video_recorder = None
                widget.recording_video = False
                widget._run_video_path = None
            except (AttributeError, RuntimeError):
                pass

    def _apply_to_all_boxes(self, action, *args, **kwargs) -> int:
        """Run ``action(box_id, *args, **kwargs)`` for every registered box.

        Returns the number of boxes the action ran successfully on. Used
        by bulk dialogs (Connect All Cameras, Stop All Recordings, Upload
        All Tasks, etc), both modes implement the same loop today; this
        helper kills that duplication.
        """
        try:
            box_ids = list(self._iter_box_ids())
        except NotImplementedError:
            box_ids = []
        ok_count = 0
        for setup_id in box_ids:
            try:
                action(setup_id, *args, **kwargs)
                ok_count += 1
            except Exception as e:
                logger.error("apply_to_all_boxes: box %s failed: %s",
                             setup_id, e)
        return ok_count

    # ==================================================================
    # Shared sidebar registry + toggle
    # ==================================================================
    # Both modes have the same shape of left-edge sidebars: a vertical
    # rotated-text button on the central widget, a CollapsibleSidebar
    # panel that slides out, and a toggle method that shows/hides the
    # button when the panel collapses/expands. The registry below kills
    # 6 toggle methods (3 maze + 3 operant) + 3 inner RotatedButton
    # class definitions in operant + maze's _setupLeftSidebars button-
    # creation boilerplate. One impl, used by both modes.

    def _ensure_sidebar_registry(self) -> None:
        if not hasattr(self, "_sidebars"):
            self._sidebars = {}

    def _register_sidebar(self, name: str, sidebar, toggle_button) -> None:
        """Wire a CollapsibleSidebar + its toggle button into the registry.

        After registration, ``_toggle_sidebar(name)`` toggles the panel
        and the on_collapse_callback shows the button again. Window
        resize/move automatically syncs the panel's height, no per-
        mode plumbing needed.
        """
        self._ensure_sidebar_registry()
        self._sidebars[name] = (sidebar, toggle_button)

        def _on_collapsed(b=toggle_button):
            try:
                b.show()
            except (AttributeError, RuntimeError):
                pass
        try:
            sidebar.on_collapse_callback = _on_collapsed
        except (AttributeError, RuntimeError):
            pass

    def _toggle_sidebar(self, name: str) -> None:
        """Generic toggle. Replaces _toggle{Info,ErrorLog,Doc}Sidebar."""
        self._ensure_sidebar_registry()
        rec = self._sidebars.get(name)
        if rec is None:
            return
        sidebar, toggle_button = rec
        try:
            sidebar.toggle()
            if not sidebar.is_collapsed:
                toggle_button.hide()
            else:
                toggle_button.show()
        except Exception as e:
            logger.error("toggle_sidebar(%s) error: %s", name, e)

    def _make_rotated_toggle(self, name: str, text: str, *,
                             color: str = "#bd93f9",
                             hover_color: str = "#ff79c6",
                             pressed_color: str = "#8be9fd",
                             y_position: int = 40,
                             height: int = 150) -> "RotatedButton":
        """Construct a RotatedButton sidebar toggle and parent it to the
        central widget. Caller passes the result to ``_register_sidebar``
        once the matching CollapsibleSidebar has been created."""
        from source.gui.widgets import RotatedButton
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        btn = RotatedButton(text, color=color, hover_color=hover_color,
                            pressed_color=pressed_color)
        # Width must clear the rotated text (its height is the button's width).
        btn.setFixedSize(28, int(height))
        btn.setFont(font)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(f"Show {text}")
        btn.setParent(self.centralwidget)
        btn.move(0, int(y_position))
        btn.raise_()
        btn.clicked.connect(lambda _checked=False, n=name: self._toggle_sidebar(n))
        return btn

    def _build_left_sidebars(self) -> None:
        """Create the left-edge overlay sidebars from the per-mode
        ``_sidebar_defs()`` table. One shared create → style → build-content →
        register loop; each mode supplies its own labels, positions, and
        content builders. ``_register_sidebar`` wires the toggle + collapse
        callback, so no per-mode toggle/collapse plumbing is needed.

        Each def is:
            (key, sidebar_attr, toggle_attr, toggle_text,
             (color, hover, pressed), y, height, header, width,
             content_builder)
        ``content_builder`` runs AFTER the sidebar exists (it reads
        ``self.<sidebar_attr>``).
        """
        from source.gui.theme import THEME
        from source.gui.widgets import CollapsibleSidebar
        sidebar_bg = (
            "CollapsibleSidebar {"
            f" background-color: {THEME.palette.surface};"
            f" border-right: 1px solid {THEME.palette.surface_border_strong};"
            "}"
        )
        for (key, sb_attr, tg_attr, tg_text, colors,
             y, height, header, width, build_content) in self._sidebar_defs():
            sidebar = CollapsibleSidebar(
                parent=self.centralwidget, header_text=header,
                expanded_width=width)
            sidebar.setStyleSheet(sidebar_bg)
            setattr(self, sb_attr, sidebar)
            build_content()
            sidebar.move(0, 0)
            sidebar.setFixedHeight(self.centralwidget.height())
            sidebar.hide()
            clr, hover, press = colors
            btn = self._make_rotated_toggle(
                key, tg_text, color=clr, hover_color=hover,
                pressed_color=press, y_position=y, height=height)
            setattr(self, tg_attr, btn)
            self._register_sidebar(key, sidebar, btn)

    def _sync_sidebar_heights(self) -> None:
        """Resize every registered sidebar to match the central widget."""
        try:
            h = self.centralwidget.height()
        except Exception:
            return
        for sidebar, _btn in getattr(self, "_sidebars", {}).values():
            try:
                sidebar.setFixedHeight(h)
            except (AttributeError, RuntimeError):
                pass

    # ==================================================================
    # Resize / move / close, shared lifecycle.
    # ==================================================================

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            self._resize_event_extras(event)
        except (AttributeError, RuntimeError) as e:
            logger.debug("_resize_event_extras error: %s", e)

    def moveEvent(self, event):
        super().moveEvent(event)
        try:
            self._move_event_extras(event)
        except (AttributeError, RuntimeError) as e:
            logger.debug("_move_event_extras error: %s", e)

    def closeEvent(self, event):
        """Shared shutdown: stop recording, shutdown pipeline, close
        detached windows. Subclass extras run via ``_close_event_extras``.

        A confirmation prompt guards against an accidental close, any
        running session would otherwise be stopped silently."""
        reply = QtWidgets.QMessageBox.question(
            self, "Close pyBehaviorLab",
            "Close the application?\nAny running sessions will be stopped.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        try:
            try:
                self._flush_autosave_on_close()
            except (AttributeError, RuntimeError) as e:
                logger.debug("autosave flush on close: %s", e)
            for tname in ("refresh_timer", "process_timer", "display_timer"):
                t = getattr(self, tname, None)
                if t is not None:
                    try:
                        t.stop()
                    except (AttributeError, RuntimeError):
                        pass
            # Timers are stopped, so no new tile jobs can be submitted; drain
            # the ones in flight before the widgets they will paint into go
            # away. Display work is milliseconds, so this cannot hang close.
            pool = getattr(self, "_tile_pool", None)
            if pool is not None:
                try:
                    pool.shutdown(wait=True)
                except Exception as e:
                    logger.debug("tile pool shutdown: %s", e)
                self._tile_pool = None
            # Named explicitly, not probed. ``getattr(self, "stop_recording")``
            # would be ambiguous: the two modes give that name opposite
            # meanings, operant's is a full synchronous teardown, maze has no
            # counterpart, so shutdown would depend on which app is running.
            # ``_stop_recording_for_box`` is on the base and means one thing.
            for setup_id in list(getattr(self, "recording_setups", []) or []):
                try:
                    self._stop_recording_for_box(setup_id)
                except (AttributeError, RuntimeError) as e:
                    logger.debug("stop recording for box %s: %s", setup_id, e)
            # Join the async recorder-stop reapers BEFORE process exit,
            # they are daemon threads, and dying mid-ffmpeg-release leaves
            # an unplayable mp4 (no moov atom) + an orphaned child.
            for t in list(MainWindowBase._recorder_stop_threads):
                if t.is_alive():
                    t.join(timeout=12.0)
                    if t.is_alive():
                        logger.warning(
                            "%s still flushing at app close, video file "
                            "may be truncated", t.name)
            MainWindowBase._recorder_stop_threads.clear()
            # Detach the Qt bridge from the pipeline BEFORE shutdown, the
            # pose backend stops with wait=False, and a late inference
            # result emitting on a dying QObject is the queued-signal
            # crash trap.
            try:
                bridge = getattr(self, "bridge", None)
                if bridge is not None and hasattr(bridge, "shutdown"):
                    bridge.shutdown()
            except (AttributeError, RuntimeError) as e:
                logger.debug("qt bridge shutdown error: %s", e)
            try:
                self.pipeline.shutdown()
            except (AttributeError, RuntimeError) as e:
                logger.debug("pipeline.shutdown error: %s", e)
            # Flush + close the process-wide drop log so trailing drop rows
            # survive (the interval flush only fires on the NEXT drop).
            try:
                from source.video.recording.drop_log import drop_log as _dlog
                _dlog.close()
            except Exception as e:
                logger.debug("drop log close error: %s", e)
            # Detached tabs and the plot window have no Qt parent (so the
            # operator can stack them behind the rig UI), which means Qt will
            # not take them down with us. This read ``self.detached_tabs``,
            # which only operant populates, maze's detached tabs live on the
            # tab widget and were never closed. The registry covers both.
            try:
                n = close_independent_windows()
                if n:
                    logger.debug("Closed %d detached window(s)", n)
            except Exception as e:
                logger.debug("closing detached windows: %s", e)
            try:
                self._close_event_extras(event)
            except (AttributeError, RuntimeError) as e:
                logger.debug("_close_event_extras error: %s", e)
            logger.info("Application closing, cleanup complete")
        except Exception as e:
            logger.error("Cleanup error: %s", e)
        finally:
            super().closeEvent(event)

    # ==================================================================
    # Pipeline bridge slots, receive sink events on the GUI main thread
    # ==================================================================

    def _paint_streaming_cameras_once(self):
        """Pull the latest frame from every streaming camera and paint
        its tile. Called once per ``process_timer`` tick (Qt main thread).

        For every box mapped to a camera, pull the latest frame from
        the CameraThread directly (lock + version compare). Skip when
        nothing changed.

        Optional per-tile rate cap via ``cfg.display.max_fps``, when
        set, the paint fires only if at least ``1 / max_fps`` seconds
        have passed since the previous paint for the same box.
        Capture is unaffected.

        Why polling not signal: an earlier queued display-sink path used a
        ``maxsize=1`` queue + worker thread + queued Qt signal. When
        overlay drawing got slow, the queue silently dropped frames
        upstream of the slot, measured 7 fps display from a 20 fps
        camera. Polling just paints the most recent frame; no hidden
        queue, so display rate ≈ min(camera fps, paint rate).
        Recording / pose / tracker still go through the bus as before.
        """
        try:
            vm = getattr(self, "video_manager", None)
            if vm is None or not getattr(vm, "box_camera_map", None):
                return
            # Snapshot the map, entries can mutate while we iterate
            # (camera connect/disconnect runs on the same thread but a
            # closeEvent could clear it under us).
            jobs = []
            cam_frames: dict = {}      # camera_id -> (frame, version), once per tick
            for setup_id, cam_id in list(vm.box_camera_map.items()):
                cam_thread = vm.cameras.get(cam_id)
                if cam_thread is None:
                    continue
                if not getattr(cam_thread, "connected", False):
                    continue
                # Nobody is looking at this tile, skip the whole pipeline.
                # The video grid lives in a TAB, so while the operator is on
                # Live Status / Session Plot / Logger every tile was still
                # being cropped, resized, converted and painted. Qt reports a
                # widget on a non-current tab as not visible, which makes this
                # the cheapest check available and the largest single saving
                # when the tab is hidden.
                if not self._is_box_tile_visible(setup_id):
                    continue
                # Every box paints here at camera rate, tracking boxes
                # included. The overlay (keypoints / bbox / centroid) is drawn
                # from the per-box OverlayState, kept current by the pose /
                # tracker result callbacks and TTL-gated by the renderer.
                # ONE pull per camera per tick, not one per box. Sixteen boxes
                # on a shared camera were taking the same lock sixteen times
                # to be handed the same image.
                if cam_id in cam_frames:
                    frame, version = cam_frames[cam_id]
                    if frame is None:
                        continue
                else:
                    last_v = min((self._display_last_version.get(b, 0)
                                  for b, c in vm.box_camera_map.items()
                                  if c == cam_id), default=0)
                    payload = cam_thread.get_latest_frame_versioned(last_v)
                    if payload is None:
                        cam_frames[cam_id] = (None, None)
                        continue
                    frame, _capture_host_ns, version = payload
                    cam_frames[cam_id] = (frame, version)
                if self._display_last_version.get(setup_id, 0) >= version:
                    continue        # this tile already shows this frame
                self._display_last_version[setup_id] = version

                # Resolve everything Qt-bound HERE, on the GUI thread, so the
                # per-tile work below can run on the pool: the ROI comes from
                # the box widget and the target size from the tile label.
                try:
                    jobs.append((setup_id, frame,
                                 self._get_box_roi(setup_id, frame),
                                 self._box_video_label_size(setup_id)))
                except Exception as e:
                    logger.debug("display job prep box %s: %s", setup_id, e)

            if not jobs:
                return
            self._paint_prepared_tiles(jobs)
        except Exception as e:
            logger.debug("display poll tick error: %s", e)

    def _paint_prepared_tiles(self, jobs):
        """Prepare every tile in parallel, then paint them all.

        The split is forced by Qt, not chosen: ``QPixmap`` may only be touched
        on the GUI thread, while crop/resize/overlay is numpy and cv2, which
        release the GIL. So the expensive half fans out across the pool and the
        GUI thread is left with ``fromImage`` + ``setPixmap``, measured 2.3 ms
        for sixteen tiles against 50 ms for the whole job done inline.

        Nothing is buffered. Each tick prepares the frame that is current at
        that instant and paints it; a tile that cannot keep up simply misses
        that frame. That is deliberate, an earlier design queued frames for a
        display worker and the queue silently dropped them UPSTREAM of the
        slot, which showed up as 7 fps from a 20 fps camera. Display is the one
        consumer allowed to skip, so it skips here, visibly, rather than
        somewhere a frame accountant cannot see.
        """
        # Group by the work itself, not by box. Boxes sharing a camera with
        # the same ROI and tile size crop-and-resize to identical pixels, so
        # that half runs ONCE per distinct shape. On a 16-box wall fed by one
        # un-segmented camera, every box showing the same view; this is the
        # difference between 123.9 ms and 6.1 ms of resizing per tick.
        mode = getattr(self, "_tile_aspect_mode", "stretch")
        groups: dict = {}
        for sid, frame, roi, target in jobs:
            # Quantise the tile size into the key. A grid of "identical" tiles
            # differs by a pixel or two from layout rounding, which split 16
            # boxes into 4 groups and did the expensive resize 4 times instead
            # of once. Rounding to 16 px collapses them; _render_scaled then
            # scales each pixmap to its exact tile, which it already did.
            size = None
            if target is not None and target.isValid():
                size = (max(16, round(target.width() / 16) * 16),
                        max(16, round(target.height() / 16) * 16))
            key = (id(frame), tuple(roi) if roi else None, size)
            groups.setdefault(key, []).append((sid, frame, roi, target))

        pool = getattr(self, "_tile_pool", None)
        keys = list(groups)

        def _base_for(key):
            _sid, frame, roi, _target = groups[key][0]
            size = key[2]
            target = QtCore.QSize(*size) if size else _target
            try:
                return self._tile_base(frame, roi, target, mode)
            except Exception as e:
                logger.debug("tile base failed: %s", e)
                return None

        if pool is not None and len(keys) > 1:
            bases = dict(zip(keys, pool.map(_base_for, keys)))
        else:
            bases = {k: _base_for(k) for k in keys}

        prepared = []
        for key, members in groups.items():
            got = bases.get(key)
            if got is None:
                continue
            base, sx, sy, ms = got
            # Boxes with nothing to draw produce byte-identical output, so the
            # colour conversion is shared too, the same argument as the
            # resize, one step further down.
            plain = None
            for sid, _f, _r, _t in members:
                try:
                    if self._tile_has_overlay(sid):
                        prepared.append(
                            (sid, self._finish_tile(sid, base, sx, sy, ms)))
                        continue
                    if plain is None:
                        plain = self._finish_tile(sid, base, sx, sy, ms)
                    prepared.append((sid, plain))
                except Exception as e:
                    logger.debug("tile finish box %s: %s", sid, e)
        for setup_id, arr in prepared:
            if arr is None:
                continue
            try:
                pixmap = self._numpy_to_pixmap(arr)
                if pixmap is None or pixmap.isNull():
                    continue
                self._render_scaled(setup_id, pixmap)
                widget = self._setup_widget_for(setup_id)
                if widget is not None:
                    self._update_fps_status(setup_id, widget)
            except Exception as e:
                if "deleted" not in str(e):
                    logger.error("Display update error for box %s: %s",
                                 setup_id, e)

    def _on_box_pose(self, setup_id, cam_frame_id, pose_array, location,
                     speed, zones_by_body_part, raw_pose_dict):
        """PoseSink result callback. Writes pose fields onto the per-box
        OverlayState; renderer reads with TTL freshness check."""
        try:
            self._on_overlay_update(
                setup_id,
                pose=pose_array,
                body_parts=list(self.pipeline.pose_model_info()[0]),
                confidence_threshold=float(self.pipeline.pose_model_info()[1]),
                location=location,
                speed=float(speed),
                cam_frame_id=int(cam_frame_id),
            )
        except Exception as e:
            logger.warning("on_box_pose error (box=%s): %s", setup_id, e)

    def _on_box_triggers(self, setup_id, trigger_frame):
        """TriggerFrame callback (T2/T3): stash the rule states on the box's
        OverlayState for annotation, and feed any open real-time plot."""
        try:
            self._on_overlay_update(
                setup_id,
                triggers=list(getattr(trigger_frame, "states", []) or []),
            )
        except Exception as e:
            logger.debug("on_box_triggers error (box=%s): %s", setup_id, e)
        # Inline Session-Plot trigger lane (the live view's only home).
        upw = getattr(self, "universal_plot_window", None)
        if upw is not None:
            try:
                upw.push_trigger_frame(setup_id, trigger_frame)
            except Exception as e:
                logger.debug("session-plot trigger feed (box=%s): %s", setup_id, e)

    def _on_box_tracker(self, setup_id, cam_frame_id, centroid, location,
                        speed, zones_by_body_part, position):
        """TrackerSink result callback. Writes blob fields onto the
        per-box OverlayState.
        """
        try:
            x, y, w, h = position[:4]
            cent = (int(centroid[0]), int(centroid[1])) if centroid else None
            self._on_overlay_update(
                setup_id,
                bbox=(int(x), int(y), int(w), int(h)),
                centroid=cent,
                location=location,
                speed=float(speed),
                cam_frame_id=int(cam_frame_id),
            )
        except Exception as e:
            logger.debug("on_box_tracker error (box=%s): %s", setup_id, e)

    def _on_box_health(self, setup_id, reason):
        """Pipeline health alarm, log and surface to UI status."""
        logger.warning("Pipeline health (box=%s): %s", setup_id, reason)
        try:
            self._box_status_say(setup_id, f"Pipeline alarm: {reason}")
        except (AttributeError, RuntimeError):
            pass
        # A dead encoder loses all video for the session while tracking rows
        # keep flowing, so it can pass unnoticed, escalate to a one-shot
        # modal per box, like the DLC-stall path.
        if reason == "encoder_dead":
            warned = getattr(self, "_encoder_dead_modal_warned", set())
            if setup_id not in warned:
                warned.add(setup_id)
                self._encoder_dead_modal_warned = warned
                try:
                    QtWidgets.QMessageBox.warning(
                        self, "Video encoder has stopped",
                        f"Box {setup_id}: the video encoder gave up after "
                        "repeated write failures, video recording has "
                        "stopped for this box (tracking data is still being "
                        "logged).\n\nCheck free disk space and the log "
                        "sidebar, then restart recording for this box."
                    )
                except Exception:
                    pass

    def _on_pose_failed(self, setup_id: int, reason: str, streak: int) -> None:
        """Pose silent-failure surfacing.

        Fired when PoseSink has produced N consecutive all-zero-confidence
        results. We log at WARN, raise the per-box status,
        and on the largest streak we also pop a one-shot QMessageBox so a
        user who walked away doesn't lose an experiment to invisible
        DLC death.
        """
        logger.warning(
            "Pose silent failure (box=%s, streak=%s): %s",
            setup_id, streak, reason,
        )
        try:
            self._box_status_say(
                setup_id,
                f"{self._pose_backend_label(setup_id)} stalled "
                f"(streak={streak}): {reason}"
            )
        except (AttributeError, RuntimeError):
            pass
        # Only escalate to a modal at the 100-frame threshold (~3 s of
        # silence) so we don't interrupt the user with every transient
        # blip.  Suppress repeats per box.
        # Escalate only on the SECOND report (ten times the stall
        # threshold), so a modal never precedes the banner.
        from source.video.framebus.pose_sink import PoseSink as _PS
        if streak >= _PS._EMPTY_STALL_FRAMES * 10:
            warned = getattr(self, "_pose_failed_modal_warned", set())
            if setup_id not in warned:
                warned.add(setup_id)
                self._pose_failed_modal_warned = warned
                backend = self._pose_backend_label(setup_id)
                try:
                    QtWidgets.QMessageBox.warning(
                        self, "Pose inference has stalled",
                        f"Box {setup_id} ({backend}): {reason}\n\n"
                        "Open the log sidebar for the full traceback. "
                        "If you'd previously enabled grayscale capture, "
                        f"switch it off and re-Apply the {backend} settings."
                    )
                except Exception:
                    pass

    def _pose_backend_label(self, setup_id=None) -> str:
        """"SLEAP" / "DLC" / "Pose", whichever is actually configured.

        Every stall message used to say "DLC" whatever was running, so a SLEAP
        model that went quiet was reported as a DLC failure and sent the
        operator to look at the wrong log, the wrong model folder and the
        wrong settings. The name has to come from the box's tracking config,
        not from the string the message was written with.
        """
        try:
            pipe = getattr(self, "pipeline", None)
            tc = pipe.get_tracking_config(setup_id) if pipe is not None else None
            tt = str(getattr(tc, "tracker_type", "") or "").lower()
        except Exception:
            tt = ""
        if tt.startswith("sleap"):
            return "SLEAP"
        if tt.startswith("dlc"):
            return "DLC"
        return "Pose"

    @staticmethod
    def _numpy_to_pixmap(frame):
        """BGR / BGRA / Grayscale numpy -> ``QPixmap``. MAIN-THREAD only.

        ONE full-frame copy: the QImage wraps the numpy buffer without
        copying, and ``QPixmap.fromImage`` copies it into the pixmap's own
        storage, so no extra ``.copy()`` is needed (the buffer stays alive
        in this scope until ``fromImage`` has run, even when
        ``ascontiguousarray`` made a temporary).

        A 4-channel BGRA buffer takes the fast path. ``Format_RGB32`` is the
        native 32-bit layout, so ``fromImage`` is essentially a memcpy, where
        ``Format_BGR888`` makes it convert every pixel ON THE GUI THREAD:
        measured 0.19 ms vs 11.12 ms for sixteen 240x180 tiles. The BGR->BGRA
        widening that buys it costs 6.6 ms for the same sixteen, but runs
        wherever the caller prepared the frame, off-thread, in parallel, so
        it is paid on a worker and saved on the thread that cannot afford it.

        Accepted shapes: ``(H, W)`` uint8 grayscale, ``(H, W, 3)`` BGR,
        ``(H, W, 4)`` BGRA. Anything else returns ``None``.
        """
        if frame is None or frame.size == 0:
            return None
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return None
        if frame.ndim == 2:
            bpl = w
            fmt = QtGui.QImage.Format.Format_Grayscale8
        elif frame.ndim == 3 and frame.shape[2] == 4:
            bpl = 4 * w
            fmt = QtGui.QImage.Format.Format_RGB32
        elif frame.ndim == 3 and frame.shape[2] == 3:
            bpl = 3 * w
            fmt = QtGui.QImage.Format.Format_BGR888
        else:
            return None
        qimg = QtGui.QImage(frame.data, w, h, bpl, fmt)
        return QtGui.QPixmap.fromImage(qimg)

    def _get_box_roi(self, setup_id, frame=None):
        """Resolve ROI for a box.

        Resolution: subclass widget (via ``_box_roi_lookup``) ->
        video_segment_config. Returns ``(x, y, w, h)`` or None.
        """
        try:
            try:
                roi = self._box_roi_lookup(setup_id, frame)
            except NotImplementedError:
                roi = None
            if roi:
                return roi

            seg_cfg = getattr(self, "video_segment_config", None)
            if seg_cfg and 'boxes' in seg_cfg:
                for box_config in seg_cfg['boxes']:
                    config_box_id = box_config.get('box_id') or box_config.get('box_number')
                    if config_box_id != setup_id:
                        continue
                    geometry = box_config.get('geometry', {})
                    percent = geometry.get('percent', {})
                    if frame is not None and all(k in percent for k in ('x', 'y', 'width', 'height')):
                        h, w = frame.shape[:2]
                        # round(), NOT int(): ``capture.extract_segment``,
                        # which cuts the slice the POSE coordinates are
                        # measured in, rounds. Truncating here resolved the
                        # same normalized rectangle one pixel up and left
                        # whenever the product's fraction reached .5, so the
                        # displayed image sat a pixel off the keypoints drawn
                        # on it. One cropper, one rounding.
                        return (
                            int(round(float(percent['x']) * w)),
                            int(round(float(percent['y']) * h)),
                            int(round(float(percent['width']) * w)),
                            int(round(float(percent['height']) * h)),
                        )
                    pixel = geometry.get('pixel', {})
                    if all(k in pixel for k in ('x', 'y', 'width', 'height')):
                        return (int(pixel['x']), int(pixel['y']),
                                int(pixel['width']), int(pixel['height']))

            return None
        except Exception as e:
            logger.debug("ROI lookup error for box %s: %s", setup_id, e)
            return None

    def update_box_display(self, setup_id, frame, timestamp=None,
                         already_segmented=True):
        """Display a video frame for one box (BGR888 zero-copy path).

        Called from the polling paint loop (``_paint_streaming_cameras_once``,
        full frame, ``already_segmented=False``). When already segmented by
        the FrameBus (shared camera), ROI cropping is a no-op.

        Pipeline drives display rate; no throttle here.
        """
        # A frame is arriving for this box, so its size is now knowable, run
        # any zone-manager rebuild that was deferred because zones loaded
        # before the first frame (M3).
        pend = getattr(self, "_pending_zone_rebuild", None)
        if pend and setup_id in pend:
            zones = (getattr(self, "tracking_zones", {}) or {}).get(setup_id)
            if zones:
                try:
                    self._rebuild_zone_manager(setup_id, zones)
                except Exception as e:
                    logger.debug("deferred zone rebuild (box %s): %s", setup_id, e)
            else:
                pend.discard(setup_id)
        try:
            if not isinstance(frame, np.ndarray):
                # Fast path: QtBridge worker already did overlay +
                # QImage build off-thread.  Main thread only does
                # QPixmap.fromImage + paint (microseconds).
                if isinstance(frame, QtGui.QImage):
                    pixmap = QtGui.QPixmap.fromImage(frame)
                elif isinstance(frame, QtGui.QPixmap):
                    pixmap = frame
                else:
                    return
                self._render_scaled(setup_id, pixmap)
                widget = self._setup_widget_for(setup_id)
                if widget is not None:
                    self._update_fps_status(setup_id, widget)
                return

            roi = (None if already_segmented
                   else self._get_box_roi(setup_id, frame))
            target = self._box_video_label_size(setup_id)
            self._paint_prepared_tiles([(setup_id, frame, roi, target)])
        except Exception as e:
            if "deleted" not in str(e):
                logger.error("Display update error for box %s: %s", setup_id, e)

    @staticmethod
    def _downscale(frame, out_w: int, out_h: int):
        """Anti-aliased downscale that does not pay full price for the ratio.

        ``INTER_AREA`` averages every source pixel, so shrinking 640x480 to a
        tile costs ~11 ms, the single largest item in a display tick. But
        integer decimation by slicing is free (it is a view), and area
        averaging the remainder gives the same visual result: the aliasing
        INTER_AREA exists to prevent comes from the sub-2x remainder, not from
        dropping whole rows of an already-band-limited image.

        Same reasoning as ``blob.py::_wide_gaussian``, do the expensive
        filter at the resolution where it actually matters.
        """
        h, w = frame.shape[:2]
        fx, fy = w // max(1, out_w), h // max(1, out_h)
        step = max(1, min(fx, fy))
        if step > 1:
            frame = frame[::step, ::step]
        return cv2.resize(frame, (out_w, out_h),
                          interpolation=cv2.INTER_AREA)

    @staticmethod
    def _tile_base(frame, roi, target, aspect_mode):
        """Crop + resize only, no per-box overlay. **Worker-thread safe.**

        Split out because this is the expensive half AND the shareable half.
        Boxes on a shared camera with the same ROI and the same tile size
        produce a byte-identical result here, so on a 16-box CCTV wall it can
        be computed once instead of sixteen times: measured 6.1 ms against
        123.9 ms for the same work repeated per box.

        Returns ``(tile, scale_x, scale_y, marker_scale)``, the scales let
        the caller draw its own overlay at the resized coordinates.
        """
        if roi:
            x, y, w, h = roi
            fh, fw = frame.shape[:2]
            x = max(0, min(int(x), fw - 1))
            y = max(0, min(int(y), fh - 1))
            w = min(int(w), fw - x)
            h = min(int(h), fh - y)
            if w > 0 and h > 0:
                frame = frame[y:y + h, x:x + w]
        scale_x = scale_y = marker_scale = 1.0
        if target is not None and target.isValid():
            tile_w = max(1, int(target.width()))
            tile_h = max(1, int(target.height()))
            src_h, src_w = frame.shape[:2]
            if aspect_mode == "preserve":
                if src_w > 0 and src_h > 0:
                    s = min(tile_w / src_w, tile_h / src_h)
                    if 0 < s < 1.0:
                        fit_w = max(1, int(round(src_w * s)))
                        fit_h = max(1, int(round(src_h * s)))
                        frame = MainWindowBase._downscale(frame, fit_w, fit_h)
                        scale_x, scale_y = fit_w / src_w, fit_h / src_h
                    elif s > 0:
                        marker_scale = 1.0 / s
            elif src_w > 0 and src_h > 0 and (tile_w, tile_h) != (src_w, src_h):
                frame = cv2.resize(frame, (tile_w, tile_h),
                                   interpolation=cv2.INTER_NEAREST)
                scale_x, scale_y = tile_w / src_w, tile_h / src_h
        return frame, scale_x, scale_y, marker_scale

    def _tile_has_overlay(self, setup_id) -> bool:
        """Does this box draw anything of its own onto the tile?"""
        state = self._overlay.get(setup_id)
        return bool(setup_id in getattr(self, "tracking_zones", {})
                    or (state is not None
                        and (state.has_pose() or state.has_blob())))

    def _finish_tile(self, setup_id, base, scale_x, scale_y, marker_scale):
        """Per-box overlay + BGRA on top of a (possibly shared) base tile.

        Copies before drawing, because the base may be shared with other
        boxes and each draws its own keypoints, bbox and centroid onto it.
        The copy is a tile-sized buffer, a few hundred KB, against the
        full resize it saves.
        """
        frame = base
        if self._tile_has_overlay(setup_id):
            frame = self._draw_overlay_on_frame(base.copy(), setup_id,
                                                scale_x=scale_x,
                                                scale_y=scale_y,
                                                draw_zones=False,
                                                marker_scale=marker_scale)
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        return frame

    def _render_scaled(self, setup_id, pixmap):
        """Scale pixmap to the box's video label size, composite the
        cached zone layer onto it, and render via hook.

        Aspect-ratio mode follows ``self._tile_aspect_mode``:

          * ``stretch``: IgnoreAspectRatio.  The numpy frame was
            already cv2-resized to tile size in ``_update_box_display``,
            so this is effectively a no-op aspect-wise.  OPERANT uses this;
            tiles always fully fill their cells.

          * ``preserve``: KeepAspectRatio.  On downscale the frame was
            already cv2-resized to its aspect-fit size in ``update_box_display``
            (so this scale is a no-op that the fast-path skips); on upscale the
            source is kept and Qt does the aspect-preserving upscale.  Black
            letterbox bars appear when the cell aspect doesn't match the camera
            aspect.  MAZE uses this, so an arena keeps its shape at any window
            size.

        The two attributions were the wrong way round here for as long as the
        maze window was silently running in stretch mode.

        Zone overlay is drawn via QPainter from a per-box cached
        transparent QPixmap (``_get_zone_layer``) keyed by
        ``(zone_version, sw, sh)``, so the polygon math runs ONCE per
        zone-mutation or label-resize, not per frame, keeping live-display
        perf at camera-bound rates even with many zones.
        """
        target = self._box_video_label_size(setup_id)
        if target is None or not target.isValid() or target.width() <= 0:
            return
        if getattr(self, "_tile_aspect_mode", "stretch") == "preserve":
            aspect = QtCore.Qt.AspectRatioMode.KeepAspectRatio
        else:
            aspect = QtCore.Qt.AspectRatioMode.IgnoreAspectRatio
        # Skip the QPixmap.scaled allocation when the source pixmap is
        # already sized for the target, saves one full-pixmap copy per
        # frame per box on the GUI thread. Two cases:
        #   * stretch: cv2-resized to exactly (tile_w, tile_h) → both dims
        #     equal target.
        #   * preserve: cv2-resized to the aspect-fit size → one dim equals
        #     the target and the other is ≤ it (the pixmap already touches a
        #     tile edge). The AlignCenter tile label letterboxes the rest,
        #     so Qt's KeepAspectRatio scale would be an identity copy.
        pw, ph, tw, th = (pixmap.width(), pixmap.height(),
                          target.width(), target.height())
        already_fit = (
            (pw == tw and ph <= th) or (ph == th and pw <= tw)
        )
        if already_fit:
            scaled = pixmap
        else:
            scaled = pixmap.scaled(
                target,
                aspect,
                QtCore.Qt.TransformationMode.FastTransformation,
            )

        # Composite cached zone layer onto the scaled pixmap.  The
        # cache is keyed by (zone_version, scaled.width, scaled.height)
        # so resize naturally rebuilds at the new size, no separate
        # resize handler needed.
        if setup_id in getattr(self, "tracking_zones", {}):
            zone_layer = self._get_zone_layer(
                setup_id, scaled.width(), scaled.height()
            )
            if zone_layer is not None and not zone_layer.isNull():
                painter = QtGui.QPainter(scaled)
                try:
                    painter.drawPixmap(0, 0, zone_layer)
                finally:
                    painter.end()

        self._box_render_pixmap(setup_id, scaled)

    def _update_fps_status(self, setup_id, widget):
        """Update the per-box FPS readout (~1 Hz) via ``_box_set_fps_text``."""
        now = time.perf_counter()
        counts = getattr(self, "_display_frame_counts", None)
        if counts is None:
            self._display_frame_counts = {}
            counts = self._display_frame_counts
        counts[setup_id] = counts.get(setup_id, 0) + 1

        times = getattr(self, "_fps_status_times", None)
        if times is None:
            self._fps_status_times = {}
            times = self._fps_status_times

        last = times.get(setup_id, 0.0)
        elapsed = now - last
        if elapsed < 1.0:
            return

        count = counts.get(setup_id, 0)
        display_fps = count / elapsed if elapsed > 0 else 0.0
        try:
            target_fps = self.video_manager.get_fps(setup_id)
        except Exception:
            target_fps = 0.0
        try:
            runtime_fps = self.video_manager.get_runtime_fps(setup_id) \
                if hasattr(self.video_manager, "get_runtime_fps") else 0.0
        except Exception:
            runtime_fps = 0.0
        times[setup_id] = now
        counts[setup_id] = 0

        if target_fps > 0 or runtime_fps > 0 or display_fps > 0:
            # Show display fps / acquiring fps so user sees both rates
            # at a glance, display=rendered to screen, acq=delivered
            # by the camera thread. target_fps is the user-picked rate,
            # runtime_fps the observed delivery.
            acq_fps = runtime_fps if runtime_fps > 0 else target_fps
            # Resolution actually delivered to THIS box (segment crop for a
            # shared camera, full frame otherwise).
            res_txt = ""
            try:
                wh = self._box_frame_wh(setup_id)
                if wh and wh[0] > 0 and wh[1] > 0:
                    res_txt = f" · {int(wh[0])}×{int(wh[1])}"
            except Exception:
                res_txt = ""
            # HEADER = the LIVE delivered ("acquiring") rate + resolution, so
            # the tile always reflects what the camera is actually giving this
            # box. When the camera under-delivers (e.g. 20 fps from a mode
            # configured for 30) the tile shows the real ~20, not a static 30.
            # Fall back to the configured "capable" rate only until the first
            # ~1s measurement window closes (acq_fps==0). Capable + display
            # rates remain in the tooltip for diagnostics.
            # THREE different numbers, and they were two. "capable" was read
            # from ``selected_fps``, which is what the operator PICKED, and
            # then labelled "camera can deliver". A box picked at 25 on a
            # camera that only offers 30 therefore read "capable: 25,
            # acquiring: 30", which says the camera is exceeding its own
            # capability. The pick, the measured ceiling and the live rate are
            # separate facts and each is worth seeing.
            requested_fps, capable_fps = self._tile_fps_facts(setup_id,
                                                              target_fps)
            header_fps = acq_fps if acq_fps > 0 else (requested_fps or capable_fps)
            text = (f"{header_fps:.0f} fps{res_txt}" if header_fps > 0
                    else (res_txt.replace(" · ", "").strip() or ", "))
            tip_lines = [
                f"requested:  {requested_fps:.1f} fps  (picked in Camera Config)",
                f"capable:    {capable_fps:.1f} fps  (offered by the camera at this resolution)",
                f"acquiring:  {runtime_fps:.1f} fps  (camera live delivery)",
                f"display:    {display_fps:.1f} fps  (rendered to tile)",
            ]
            # A UVC camera offers discrete rates and silently rounds to the
            # nearest one it has, reporting the request back unchanged. Say so
            # here rather than leaving two numbers to disagree in silence.
            if (requested_fps > 0 and runtime_fps > 0
                    and abs(runtime_fps - requested_fps)
                    > max(1.0, requested_fps * 0.10)):
                # TWO different faults, and this said the first whichever it
                # was. A camera whose 640x480 MJPEG has exactly one interval,
                # 120, was reported as "does not offer 120 fps" while running
                # at 100: it offers 120 and cannot sustain it, which is a
                # bandwidth problem with a different fix than picking another
                # rate. Which one it is, is decided by whether the requested
                # rate is in what the camera reports for this mode.
                offered = self._offered_rates_for_box(setup_id)
                asked_exists = any(abs(r - requested_fps) < 0.6 for r in offered)
                if offered and asked_exists:
                    tip_lines.append(
                        f"NOTE: this camera reports {requested_fps:.0f} fps at "
                        f"this mode but is only sustaining {runtime_fps:.1f}. "
                        f"That is a link or lighting limit, not a wrong "
                        f"choice of rate. The recording is written at the "
                        f"delivered rate, so it plays back at real speed.")
                else:
                    tip_lines.append(
                        f"NOTE: this camera does not offer {requested_fps:.0f} "
                        f"fps at this mode and is running at {runtime_fps:.1f}. "
                        f"The recording is written at the delivered rate, so "
                        f"it plays back at real speed.")
            if res_txt:
                tip_lines.append(
                    f"resolution:{res_txt.replace(' · ', ' ')}  (delivered to this box)")
            try:
                # Surface measured_fps from the backend for diagnostics.
                cam_thread = self.video_manager.cameras.get(
                    getattr(self.video_manager, "box_camera_map", {}).get(setup_id)
                ) if hasattr(self.video_manager, "cameras") else None
                if cam_thread is not None and hasattr(cam_thread, "get_measured_fps"):
                    measured = float(cam_thread.get_measured_fps())
                    if measured > 0:
                        tip_lines.append(f"measured:  {measured:.1f} fps  (last probe)")
                # Surface "calibrating" while CameraThread's background
                # measurement window is still running, explains why the
                # value may shift over the first ~12s after connect.
                if cam_thread is not None:
                    phase = getattr(cam_thread, "_calib_phase", "done")
                    if phase not in ("done", "idle"):
                        retries = getattr(cam_thread, "_calib_retries", 0)
                        tip_lines.append(
                            f"calibrating: phase={phase} retry={retries}"
                        )
            except Exception:
                pass
            # ── Timing spine: pipeline latency + delivery health ──
            try:
                from source.video.framebus.latency import (
                    format_hud_lines, low_delivery)
                snap = (self.pipeline.latency_snapshot()
                        if hasattr(self.pipeline, "latency_snapshot") else None)
                if snap:
                    tip_lines.extend(format_hud_lines(snap))
                # Delivery-vs-target: flag only when sustained (>3 s) so a
                # transient dip doesn't cry wolf. Reuses perf_counter `now`.
                lows = getattr(self, "_delivery_low_since", None)
                if lows is None:
                    self._delivery_low_since = {}
                    lows = self._delivery_low_since
                if low_delivery(runtime_fps, target_fps):
                    t0 = lows.setdefault(setup_id, now)
                    if now - t0 >= 3.0:
                        tip_lines.append(
                            f"⚠ delivery low: {runtime_fps:.1f} of "
                            f"{target_fps:.0f} fps target (check camera format)")
                        text = f"{text}  ⚠"
                else:
                    lows.pop(setup_id, None)
            except Exception:
                pass
            try:
                self._box_set_fps_text(setup_id, text, "\n".join(tip_lines))
            except NotImplementedError:
                pass
            except Exception as e:
                logger.debug("fps update error for box %s: %s", setup_id, e)

    def _attach_kalman_enhancer(self, setup_id, settings):
        """Attach/clear the optical-flow + Kalman smoothing enhancer for a
        box, per its ``smooth_tracking`` setting. Shared by BOTH modes
        (operant + maze) so the latency-compensated KF forecast runs
        identically in both; owned by one mode, the other's prediction is
        inert. Also registered with PoseSink so a later flip to DLC reuses
        the same KF state for the forecasted zone lookup.
        """
        if not hasattr(self, "tracking_enhancers"):
            self.tracking_enhancers = {}
        smooth = settings.get(
            "smooth_tracking", getattr(self, "smooth_tracking_enabled", False))
        if smooth:
            from source.video.tracking.smoothing import TrackingEnhancer
            enhancer = TrackingEnhancer(setup_id)
            self.tracking_enhancers[setup_id] = enhancer
            self.tracker_manager.set_enhancer(setup_id, enhancer)
            try:
                self.pipeline.set_pose_enhancer(setup_id, enhancer)
            except Exception:
                pass
            logger.info(
                "Box %s: smooth tracking enabled (blob + pose KF lookahead)",
                setup_id)
        else:
            self.tracking_enhancers.pop(setup_id, None)
            self.tracker_manager.set_enhancer(setup_id, None)
            try:
                self.pipeline.set_pose_enhancer(setup_id, None)
            except Exception:
                pass

    def _detach_tracking_enhancer(self, setup_id):
        """Reset + drop the Kalman enhancer for a box (shared by both modes)."""
        if not hasattr(self, "tracking_enhancers"):
            self.tracking_enhancers = {}
        enhancer = self.tracking_enhancers.pop(setup_id, None)
        if enhancer:
            try:
                enhancer.reset()
            except Exception:
                pass
        try:
            self.tracker_manager.set_enhancer(setup_id, None)
        except Exception:
            pass

    def _draw_overlay_on_frame(self, frame, setup_id,
                             scale_x: float = 1.0, scale_y: float = 1.0,
                             draw_zones: bool = True,
                             marker_scale: float = 1.0):
        """Draw zone polygons + pose keypoints + tracking bbox.

        ``scale_x`` / ``scale_y`` map source-cropped-frame pixel coords
        to the (already-resized) display-tile pixel coords.  Caller
        (``update_box_display``) does ``cv2.resize(frame, tile)`` before
        invoking this; we then scale every keypoint/bbox so they land
        at the right spot on the smaller frame.  Zones are stored in
        normalized [0,1] coords and the renderer denormalizes against
        ``frame.shape``, so they automatically follow the resize.

        ``draw_zones=False`` skips the zone burn-in for callers that
        composite a cached zone QPixmap via QPainter instead (the
        live display path).  Pose keypoints + tracking bbox still draw
        because those change every frame.
        """
        try:
            state, fresh, pose_fresh, tracker_fresh = \
                self._fresh_overlay_state(setup_id)
            if not (fresh or pose_fresh or tracker_fresh) and not draw_zones:
                # Live path with a stale/absent overlay: keypoints, bbox,
                # centroid dot, the occupancy fill (needs a fresh centroid)
                # and trigger chips all skip, and zone outlines composite
                # from the cached QPixmap layer, copying the bus's shared
                # buffer here bought nothing, every displayed frame.
                return frame

            frame = frame.copy()          # the bus shares this buffer
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            centroid = None
            if pose_fresh:
                # Read the store directly rather than through the accessor:
                # the overlay has to draw on any host, including the reduced
                # ones the geometry tests use, and a missing size lookup must
                # never be the reason a keypoint is not painted.
                centroid = self._draw_pose_layer(
                    frame, state, scale_x, scale_y, marker_scale,
                    base_radius=(getattr(self, "_marker_sizes", None)
                                 or {}).get(setup_id, MARKER_SIZE_DEFAULT))
            if centroid is None and tracker_fresh:
                centroid = self._bbox_centre(state.bbox, scale_x, scale_y)

            self._draw_zone_layer(frame, setup_id, centroid, draw_zones)
            if tracker_fresh:
                self._draw_bbox(frame, state.bbox, scale_x, scale_y,
                                marker_scale)
            # Red dot in blob mode only: in pose mode the coloured keypoints
            # already mark the animal, and an extra dot on keypoint 0 reads as
            # a separate body part. The centroid is still computed either way,
            # because zone occupancy needs it in both modes.
            if centroid is not None and not pose_fresh:
                cv2.circle(frame, centroid,
                           max(1, int(round(5 * marker_scale))), (0, 0, 255), -1)
            if fresh:
                self._draw_trigger_layer(frame, state, scale_x, scale_y)
            return frame

        except Exception as e:
            logger.warning("Zone overlay error for box %s: %s", setup_id, e)
            return frame

    # ── overlay layers ───────────────────────────────────────────────
    # Called from inside _draw_overlay_on_frame's try, so none of these
    # guards again; a raise here is caught there and the tile still paints.

    @staticmethod
    def _to_tile(v: float, scale: float) -> int:
        """A source-frame coordinate in the coordinates of the resized tile.

        The half-pixel term is the whole point. ``cv2.resize`` maps by pixel
        CENTRES: source pixel ``i`` becomes the destination span
        ``[i*s, (i+1)*s)``, whose centre is ``i*s + (s-1)/2``. Scaling a
        keypoint by ``v * s`` alone lands on that span's top-left CORNER
        instead, so every marker sat ``(s-1)/2`` pixels up and left of the
        thing it marks.

        At ``s == 1`` the term is zero, which is why this was invisible until
        a tile was enlarged, and why the error then grew in proportion to the
        zoom: half a pixel at 2x, one and a half at 4x, measurably off the
        animal on a maximised maze arena.
        """
        return int(round(v * scale + (scale - 1.0) / 2.0))

    def _fresh_overlay_state(self, setup_id):
        """``(state, fresh, pose_fresh, tracker_fresh)``.

        A cached OverlayState older than ``_OVERLAY_MAX_AGE_NS`` is stale,
        the run stopped, the framework errored, or inference stalled. Treating
        it as absent is what stops a dead run leaving its last keypoints
        painted on the tile.
        """
        state = self._overlay.get(setup_id)
        fresh = bool(
            state is not None
            and state.last_seen_ns
            and (time.monotonic_ns() - state.last_seen_ns) <= _OVERLAY_MAX_AGE_NS
        )
        return state, fresh, fresh and state.has_pose(), fresh and state.has_blob()

    @staticmethod
    def _bbox_centre(pos, scale_x, scale_y):
        """Tile-space centre of a tracker box, or ``None``."""
        if not pos:
            return None
        x, y, w, h = pos
        return (MainWindowBase._to_tile(x + w / 2, scale_x),
                MainWindowBase._to_tile(y + h / 2, scale_y))

    def _draw_pose_layer(self, frame, state,
                         scale_x, scale_y, marker_scale, base_radius=None):
        """Draw the keypoints and skeleton; return the centroid they imply.

        Keypoint 0 defines the centroid used for zone occupancy. Marker radius
        scales with ``marker_scale`` so the post-Qt-downscale size in
        preserve-aspect mode matches stretch mode.

        ``base_radius`` is the operator's configured radius for this box. It
        used to be the literal 4 written here, so the marker-size control in
        the tracking dialog was stored, saved into the project and restored on
        load while changing nothing on screen.
        """
        pose = state.pose
        body_parts = state.body_parts
        conf_thresh = state.confidence_threshold
        zone_idx = 0
        if base_radius is None:
            base_radius = self.MARKER_SIZE_DEFAULT
        marker_size = max(1, int(round(float(base_radius) * marker_scale)))

        centroid = None
        try:
            if zone_idx < len(pose):
                row = pose[zone_idx]
                conf = float(row[2]) if len(row) > 2 else 1.0
                if conf >= conf_thresh:
                    centroid = (self._to_tile(float(row[0]), scale_x),
                                self._to_tile(float(row[1]), scale_y))
        except Exception:
            pass

        skeleton = list(getattr(state, "skeleton", ()) or ())
        # NOT ``body_parts``. Passing the name list as its own allow-list made
        # every part pass, so the dialog's per-part Annotate boxes silently did
        # nothing to the live view.
        annotate_parts = list(getattr(state, "annotate_parts", ()) or ())
        if _HAS_CY_DRAWING and hasattr(pose, 'dtype'):
            pose_f = np.asarray(pose, dtype=np.float64)
            if pose_f.ndim == 2 and pose_f.shape[1] >= 2:
                self._draw_keypoints_cython(
                    frame, pose_f, body_parts, zone_idx, conf_thresh,
                    marker_size, scale_x, scale_y, skeleton, annotate_parts)
                return centroid
        self._draw_keypoints_python(frame, pose, body_parts, annotate_parts,
                                    zone_idx, conf_thresh, marker_size,
                                    scale_x, scale_y, skeleton)
        return centroid

    def _draw_keypoints_cython(self, frame, pose_f, body_parts, zone_idx,
                               conf_thresh, marker_size,
                               scale_x, scale_y, skeleton=(),
                               annotate_parts=()):
        """Cython-prepared keypoints, then the figure.

        The Cython side already dropped low-confidence points, so what comes
        back is exactly the set that can be drawn. It works in source-cropped
        space, the display scale is applied here so the compiled module stays
        scale-agnostic and needs no rebuild when the tile size changes.
        """
        kps = _cy_prepare_dlc_keypoints(pose_f, 0.0, 0.0, conf_thresh)
        found = {}
        for idx, px, py, _conf in kps:
            part_name = (body_parts[idx] if idx < len(body_parts)
                         else f"pt{idx}")
            found[part_name] = (self._to_tile(px, scale_x),
                                self._to_tile(py, scale_y))
        self._draw_pose_figure(frame, found, body_parts, skeleton,
                               annotate_parts, zone_idx, marker_size)

    def _draw_zone_layer(self, frame, setup_id, centroid, draw_zones):
        """Zones, by consumer.

        Export burns the outlines into the frame. The live display passes
        ``draw_zones=False`` and composites a cached QPainter layer instead,
        that keeps the per-frame cv2 zone draw off the GUI thread, but the
        "animal is inside this zone" fill still has to update every frame.
        """
        zones = getattr(self, "tracking_zones", {}).get(setup_id)
        if not zones:
            return
        if draw_zones:
            from source.gui.widgets.zone_overlay import render_zones_opencv
            render_zones_opencv(
                frame, zones,
                highlight_inside=[centroid] if centroid else None,
                fast=True)
        elif centroid is not None:
            # No scale arg: resolve_to_px works against frame.shape, which is
            # already the space the centroid sits in.
            self._fill_occupied_zones_live(frame, zones, centroid)

    @staticmethod
    def _draw_bbox(frame, pos, scale_x, scale_y, marker_scale):
        """The tracker's bounding box, scaled into tile coords. Stroke width
        follows ``marker_scale`` so it stays visible after a Qt downscale."""
        if not pos:
            return
        x, y, w, h = pos
        cv2.rectangle(frame,
                      (MainWindowBase._to_tile(x, scale_x),
                       MainWindowBase._to_tile(y, scale_y)),
                      (MainWindowBase._to_tile(x + w, scale_x),
                       MainWindowBase._to_tile(y + h, scale_y)),
                      (0, 255, 0), max(1, int(round(1 * marker_scale))))

    @staticmethod
    def _draw_trigger_layer(frame, state, scale_x, scale_y):
        """Composable-trigger chips + geometry highlight from the box's latest
        TriggerFrame. Shares the pose TTL, so a stale box draws none."""
        if not getattr(state, "triggers", None):
            return
        from source.video.trigger_engine import draw_triggers
        draw_triggers(
            frame, state.triggers,
            chip_corner=getattr(state, "trigger_chip_corner", "top_right"),
            flash_on_fire=getattr(state, "trigger_flash_on_fire", True),
            scale_x=scale_x, scale_y=scale_y)

    def _fill_occupied_zones_live(self, frame, zones, centroid, *,
                                  scale_x: float = 1.0, scale_y: float = 1.0,
                                  alpha: float = 0.2):
        """Alpha-blend a translucent fill into zones containing ``centroid``.

        Live-display companion to the cached QPainter outline layer.
        ``resolve_to_px`` already maps zone coords against ``frame.shape``
        so the polygon lands in the same space the centroid sits in
        (centroid was pre-scaled with the same scale_x / scale_y the
        cv2.resize used). No extra scaling here, that double-applied
        the stretch and offset the fill from the outlines.
        """
        try:
            from source.video.zones.coords import resolve_to_px
            from source.video.zones.geometry import point_inside_px
            from source.gui.widgets.zone_overlay import ZONE_COLORS_BGR

            h, w = frame.shape[:2]
            overlay = None
            cx, cy = float(centroid[0]), float(centroid[1])
            drawn = 0
            for i, zone in enumerate(zones):
                if not isinstance(zone, dict) or not zone.get("enabled", True):
                    continue
                ztype = zone.get("type", "polygon")
                if ztype in ("line", "scale"):
                    continue
                pts_raw = zone.get("points") or []
                if len(pts_raw) < 3:
                    continue
                pts_px = resolve_to_px(pts_raw, w, h, zone.get("coord_space"))
                pts_int = np.array(
                    [[int(round(x)), int(round(y))] for x, y in pts_px],
                    dtype=np.int32,
                )
                if not point_inside_px(pts_int, cx, cy):
                    continue
                if overlay is None:
                    overlay = frame.copy()
                color = ZONE_COLORS_BGR[i % len(ZONE_COLORS_BGR)]
                cv2.fillPoly(overlay, [pts_int], color, lineType=cv2.LINE_AA)
                drawn += 1
            if drawn:
                cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        except Exception as e:
            logger.debug("zone fill highlight (box) error: %s", e)

    def _draw_keypoints_python(self, frame, pose, body_parts, annotate_parts,
                               zone_idx, conf_thresh, marker_size,
                               scale_x: float = 1.0, scale_y: float = 1.0,
                               skeleton=()):
        """Pure-Python keypoint drawing (Cython fallback).

        ``scale_x`` / ``scale_y`` map source-cropped coords to the (resized)
        display tile coords.
        """
        found = {}
        for i, row in enumerate(pose):
            part_name = body_parts[i] if i < len(body_parts) else f"pt{i}"
            try:
                px = self._to_tile(float(row[0]), scale_x)
                py = self._to_tile(float(row[1]), scale_y)
                conf = float(row[2]) if len(row) > 2 else 1.0
            except (IndexError, ValueError):
                continue
            if conf < conf_thresh:
                continue
            found[part_name] = (px, py)
        self._draw_pose_figure(frame, found, body_parts, skeleton,
                               annotate_parts, zone_idx, marker_size)

    def _project_part_colours(self, names) -> dict:
        """This project's ``{part: BGR}``, assigning and storing new parts.

        Stored in the dialog overrides, which travel with the project file, so
        the colour a part is given the first time it is drawn is the colour it
        has in every later session. Every drawing path asks here, so the live
        tile, the saved annotated video and the offline overlay agree.
        """
        store = getattr(self, "_tracking_dialog_globals", None)
        if store is None:
            store = self._tracking_dialog_globals = {}
        saved = store.get("part_colours") or {}
        mapping = part_colour_map(names, saved)
        if len(mapping) != len(saved) or any(
                tuple(saved.get(k, ())) != v for k, v in mapping.items()):
            store["part_colours"] = {k: list(v) for k, v in mapping.items()}
        return mapping

    def _draw_pose_figure(self, frame, found, body_parts, skeleton,
                          annotate_parts, zone_idx, marker_size):
        """Edges, then markers, then labels, for whichever drawer found them.

        The edges are the ones the MODEL declares (or the operator drew),
        looked up by part name. A polyline through the keypoints in list order
        is not a skeleton: for a model whose parts join in a star at the head
        it connects snout to ear to ear, asserting an anatomy nothing in the
        model claims. With no declared
        edges nothing is joined, which is the honest picture.

        Drawing order matters: lines first so the markers sit on top of them,
        and labels last so they are never painted over.
        """
        for a, b in (skeleton or ()):
            pa, pb = found.get(a), found.get(b)
            if pa is None or pb is None:
                continue
            # Dark casing under a light core. A single light-grey line is
            # invisible on an overexposed arena floor, which is most of a
            # top-down rig frame; the casing makes it read on any background
            # without needing a colour that fights the markers.
            cv2.line(frame, pa, pb, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.line(frame, pa, pb, (235, 235, 235), 1, cv2.LINE_AA)

        order = list(body_parts or ()) or list(found)
        # Through the host when it carries a project (so a part keeps its
        # colour across sessions), by name alone otherwise. Drawing must not
        # depend on a project being open: a preview before anything is loaded
        # still has to show the animal.
        resolve = getattr(self, "_project_part_colours", None)
        palette = resolve(order) if callable(resolve) else part_colour_map(order)
        fallback = part_colours(len(order) or 1)
        for name, pt in found.items():
            # By name, so a part keeps its colour when the model's part list
            # changes; by position only for a name the map has never seen.
            colour = palette.get(name)
            if colour is None:
                idx = order.index(name) if name in order else 0
                colour = fallback[idx % len(fallback)]
            # One size for every part. The zone keypoint used to be drawn two
            # pixels larger, which made size carry meaning that colour already
            # carries and left the overlay looking inconsistent between models
            # depending on which part was configured for zones. Neither
            # DeepLabCut nor SLEAP varies marker size by part.
            cv2.circle(frame, pt, marker_size, colour, -1)
            # A thin dark ring: a filled dot alone disappears against fur of
            # its own brightness, which is most of a top-down mouse video.
            cv2.circle(frame, pt, marker_size, (20, 20, 20), 1, cv2.LINE_AA)

        # No part names on the live display, in either mode. A tile is a few
        # hundred pixels across and a mouse occupies a fraction of it, so the
        # labels landed on top of the animal and on each other; the colour
        # already says which part a dot is, and the dialog lists the same
        # colours beside the names. The saved annotated video draws its own
        # markers in ``_build_annotate_callback`` and is unaffected.
        #
        # ``annotate_parts`` therefore no longer changes the live view.
        del annotate_parts

    # ==================================================================
    # Concrete shared behavior, zone persistence
    # ==================================================================

    @staticmethod
    def _ensure_zones_tagged(zones):
        """Defensive helper: ensure every zone dict has ``coord_space`` set.

        Zones reach ``self.tracking_zones`` from many paths, the editor
        (already tagged), ``_load_zones`` (now tagged), the tracking-config
        dialog return dict, and config-file loads.  Any missing tag forces
        the renderer to fall back to a range-vote heuristic that flips to
        ``"pixel"`` when a single shifted point drifts past 1.05, the
        "zones display as a line" symptom.

        This helper runs at every ``tracking_zones[bid] = ...`` assignment
        so the in-memory state is always tagged, which in turn means
        downstream save paths (config files, auto-save, recorder) carry
        the tag automatically.
        """
        from source.video.zones.coords import is_normalized_points
        if not isinstance(zones, list):
            return zones
        out = []
        for z in zones:
            if not isinstance(z, dict):
                continue
            if "coord_space" not in z:
                z = dict(z)  # don't mutate caller's dict
                z["coord_space"] = (
                    "normalized" if is_normalized_points(z.get("points", []))
                    else "pixel"
                )
            out.append(z)
        return out

    def _mirror_zones_to_pipeline(self, setup_id) -> None:
        """Hand the box's current zones to the pipeline.

        Whatever reads zones through the pipeline's ``TrackingConfig``, the
        latency probe's LED window among them, otherwise keeps the set
        captured when the project was loaded. An edit then moves the drawing,
        the overlay and the config file together while the pipeline goes on
        answering with the old geometry, and nothing on screen shows the
        disagreement: the probe reported a window it had been given, sitting
        44 px off a lamp the operator could see it was drawn around.
        """
        pipe = getattr(self, "pipeline", None)
        if pipe is None:
            return
        zones = (getattr(self, "tracking_zones", {}) or {}).get(setup_id) or []
        try:
            pipe.update_tracking_config(setup_id, zones=list(zones))
        except (AttributeError, RuntimeError) as e:
            logger.debug("mirror zones to pipeline (box %s): %s", setup_id, e)

    def _persist_zone_edit(self, setup_id) -> None:
        """Persist an in-place zone adjustment (shift / rotate / scale / Home)
        through the project config.

        Zones live in ``experiment_config.json``; there is no global side
        file anymore. ``_project_changed`` debounces the write (rapid nudges
        coalesce into one save) and reads the current ``tracking_zones`` via
        ``_read_tracking``. When no project is loaded the adjustment stays in
        memory for the session, same as any other unsaved edit, say so once
        (the dialog commit path warns the same way), because there is no side
        file persisting these behind the operator's back.
        """
        if not getattr(self, "active_config_path", ""):
            if not getattr(self, "_warned_zone_edit_no_project", False):
                self._warned_zone_edit_no_project = True
                try:
                    self._status_message(
                        "Zone edit applied for this session only, load or "
                        "create a project to save it.")
                except (AttributeError, RuntimeError):
                    pass
        self._mirror_zones_to_pipeline(setup_id)
        self._project_changed(reason="zones_adjusted")

    # Fields preserved through the disk round-trip.  Skipping any of
    # these silently breaks the live overlay or the zone editor on the
    # next reload.  ``coord_space`` is the load-bearing one, without it
    # the renderer's range-vote heuristic flips to "pixel" the moment a
    # shift / drag pushes any point past 1.05, collapsing zones into the
    # top-left corner ("displays as a line").
    _ZONE_PASSTHROUGH_KEYS = (
        "name", "type", "points", "shape_dim",
        "coord_space",                # critical
        "enabled", "validated",
        "center", "semi_axes", "radius_norm",
        "scale_length", "scale_length_mm", "scale_unit",
        "transmit_mode", "coord_var",
        "event_on_enter", "event_on_exit",
    )

    @classmethod
    def _load_zones(cls, zone_path):
        """Load zones from a JSON file. Returns a list of zone dicts."""
        if not zone_path:
            return []
        try:
            with open(zone_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            zones = []
            for z in data.get("zones", []):
                if not isinstance(z, dict):
                    continue
                pts = z.get("points", [])
                if len(pts) < 2:
                    continue
                entry = {}
                for k in cls._ZONE_PASSTHROUGH_KEYS:
                    if k == "points":
                        entry[k] = [list(p) for p in pts]
                    elif k in z:
                        entry[k] = z[k]
                entry.setdefault("name", "zone")
                entry.setdefault("type", "polygon")
                # If the file pre-dates coord_space tagging, infer once
                # so downstream renderers don't keep range-voting on it.
                if "coord_space" not in entry:
                    from source.video.zones.coords import is_normalized_points
                    entry["coord_space"] = (
                        "normalized" if is_normalized_points(entry["points"])
                        else "pixel"
                    )
                zones.append(entry)
            return zones
        except Exception as e:
            logger.warning("Failed to load zones from %s: %s", zone_path, e)
            return []

    def load_zones_for_box(self, setup_id, zone_path: Optional[str] = None) -> bool:
        """Load a zone-config file into a single box.

        Backs the per-setup "Load Zone Config..." button. If ``zone_path``
        is None, prompts the user for a file. Updates ``tracking_zones``,
        ``tracking_zone_paths``, and rebuilds the ZoneManager for that box.
        Returns True on success.
        """
        if not zone_path:
            zone_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, f"Load Zone Config for Box {setup_id}",
                str(Path(app_paths.tracking_configs_dir)),
                "Zone Config (*.json);;All Files (*.*)",
            )
        if not zone_path:
            return False

        zones = self._load_zones(zone_path)
        if not zones:
            try:
                self.showError(f"No zones found in {zone_path}")
            except (AttributeError, RuntimeError):
                pass
            return False

        if not hasattr(self, "tracking_zones"):
            return False
        # Single boundary: tag + store + snapshot baseline.  After this
        # the Home button can reset back to the just-loaded state.
        try:
            self._set_box_zones(setup_id, zones)
        except Exception:
            logger.error("Cannot store zones for box %s", setup_id)
            return False

        if not hasattr(self, "tracking_zone_paths"):
            self.tracking_zone_paths = {}
        self.tracking_zone_paths[setup_id] = zone_path

        try:
            self._rebuild_zone_manager(setup_id, zones)
        except (AttributeError, NotImplementedError):
            pass
        except Exception as e:
            logger.debug("Rebuild zone manager for box %s: %s", setup_id, e)

        logger.info("Box %s: loaded %d zones from %s", setup_id, len(zones), zone_path)
        return True

    def _box_frame_wh(self, setup_id):
        """Return the box's current cropped-frame (W, H), or None if unknown.

        Used to denormalize stored zone points into the **same pixel
        space the renderer draws against**.  This MUST match
        ``frame.shape`` of whatever the renderer (``render_zones_opencv``)
        sees, otherwise rotating a circle in space-A and drawing it
        normalized into space-B with a different aspect ratio turns the
        circle into a tilted ellipse, exactly the bug the user
        reported.

        Resolution order (most authoritative first):
          1. Last segmented frame from ``video_manager.get_last_frame``,
             this is byte-for-byte the array the renderer touches.
          2. ``video_manager.get_segment_size(box_id, frame_shape)``
             where ``frame_shape`` comes from the latest *raw* camera
             frame, uses the same percent→pixel math FrameBus's
             ``extract_segment`` uses, so size matches even when the
             cached pixel geometry is stale (different camera res).
          3. ``_get_box_roi(box_id)``: config geometry as last resort.
        """
        vm = getattr(self, "video_manager", None) or getattr(self, "video", None)
        if vm is not None:
            try:
                frame = vm.get_last_frame(setup_id)
                if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                    return int(frame.shape[1]), int(frame.shape[0])
            except (AttributeError, RuntimeError):
                pass
            try:
                raw_shape = None
                try:
                    raw = vm.get_last_raw_frame() if hasattr(vm, "get_last_raw_frame") else None
                    if raw is not None:
                        raw_shape = raw.shape
                except Exception:
                    raw_shape = None
                size = vm.get_segment_size(setup_id, frame_shape=raw_shape) \
                    if raw_shape is not None else vm.get_segment_size(setup_id)
                if size and size[0] > 0 and size[1] > 0:
                    return int(size[0]), int(size[1])
            except Exception:
                pass
        try:
            cfg_roi = self._get_box_roi(setup_id)
        except Exception:
            cfg_roi = None
        if cfg_roi and len(cfg_roi) >= 4 and cfg_roi[2] > 0 and cfg_roi[3] > 0:
            return int(cfg_roi[2]), int(cfg_roi[3])
        return None

    # Minimum points for a zone to enter the live manager. Maze raises
    # this to 3: line/scale zones aren't addressable trial-logic objects
    # there.
    _ZONE_MIN_POINTS = 2

    def _rebuild_zone_manager(self, setup_id, zones):
        """Rebuild the ZoneManager for a box (ONE body for both modes;
        the only mode delta is the ``_ZONE_MIN_POINTS`` class attr).

        Zones on disk are normalized [0, 1] of the box's cropped frame.  The
        tracker / pose pipeline emits centroid in *full-camera* pixels for
        the non-segmented path (see ``update_box_display`` and
        ``_draw_overlay_on_frame``) and *cropped-frame* pixels for the
        segmented path.  We denormalize zones to the cropped frame, the
        same space pose results live in for this box.
        """
        zm_owner = getattr(self, "tracker_manager", None)
        if zm_owner is None or not hasattr(zm_owner, "get_zone_manager"):
            return
        zm = zm_owner.get_zone_manager(setup_id)
        if zm is None:
            # Lazy-create the per-box ZoneManager so PoseSink's
            # ``zone_lookup`` is populated; otherwise ``zones_by_body_part``
            # is empty every frame and zone events / coord resolution die.
            try:
                from source.video.zones.triggering import ZoneManager
                zm = ZoneManager()
                if hasattr(zm_owner, "set_zone_manager"):
                    zm_owner.set_zone_manager(setup_id, zm)
                else:
                    zm_owner.zone_managers[setup_id] = zm
            except Exception as e:
                logger.error("Box %s: ZoneManager create failed: %s", setup_id, e)
                return
        try:
            from source.video.zones.triggering import Zone
            from source.video.zones.coords import resolve_to_px
            zm.clear_zones()

            wh = self._box_frame_wh(setup_id)
            if wh is None:
                # No frame size known yet, feeding mis-scaled zones would be
                # worse than deferring. But deferring silently (the old
                # behaviour) meant zones loaded before the first frame NEVER
                # entered the manager: occupancy stayed empty forever and no
                # zone events fired (M3). Register a one-shot retry that fires
                # from the paint loop when the first frame gives us a size.
                pend = getattr(self, "_pending_zone_rebuild", None)
                if pend is None:
                    pend = self._pending_zone_rebuild = set()
                pend.add(setup_id)
                logger.debug(
                    "Box %s: zone manager rebuild deferred to first frame "
                    "(no frame size yet)", setup_id)
                return
            # Frame size known now, clear any pending flag for this box.
            pend = getattr(self, "_pending_zone_rebuild", None)
            if pend:
                pend.discard(setup_id)
            fw, fh = wh
            for zd in zones:
                pts_raw = zd.get("points", [])
                if len(pts_raw) < self._ZONE_MIN_POINTS:
                    continue
                pts_px = resolve_to_px(pts_raw, fw, fh, zd.get("coord_space"))
                # Build from the dict so the zone's own policy survives.
                # Constructing with name/points/zone_type alone dropped
                # enabled, transmit_mode, coord_var and the enter/exit event
                # names, so the live manager ran every zone on the default
                # transmit policy rather than the authored one.
                #
                # Points are substituted before construction, not after: the
                # Shapely cache is built in __post_init__, so assigning points
                # afterwards would leave it holding the unresolved geometry.
                zm.add_zone(Zone.from_dict(
                    {**zd, "points": pts_px, "coord_space": "pixel"}))
        except Exception as e:
            logger.debug("Zone manager rebuild error for box %s: %s", setup_id, e)

    # ==================================================================
    # Per-box zone-adjust handlers (driven by SetupWidget/VideoStreamHolder
    # arrow / rotate / zoom buttons). Operate on self.tracking_zones[box_id]
    # directly; both modes share this code so the operant per-tile buttons
    # behave identically to the maze sidebar buttons.
    # ==================================================================

    # ── Zone baseline (Home/reset support) ────────────────────────
    # Strategy: snapshot the baseline EVERY time zones are loaded into
    # the box (file load, config dialog accept, start-tracking flow).
    # Use ``_set_box_zones`` at every external load site instead of a
    # bare ``self.tracking_zones[bid] = X`` so the baseline always
    # exists, and Home always has something to reset to.
    #
    # Shift / rotate / scale handlers do NOT update the baseline, they
    # mutate the in-memory zones in place, so the baseline keeps its
    # snapshot of the pre-adjustment state.
    #
    # Resetting via Home does NOT clear the baseline either: pressing
    # Home twice in a row gives the same result, and adjusting after a
    # Home click can be undone with another Home click.

    def _set_box_zones(self, setup_id, zones) -> None:
        """Single boundary for "external" zone loads: tag, store, and
        capture a baseline snapshot the Home button can restore from.
        """
        zones = self._ensure_zones_tagged(zones)
        self.tracking_zones[setup_id] = zones
        self._snapshot_zone_baseline(setup_id)
        self._invalidate_zone_layer(setup_id)
        self._check_zone_shape_dim(setup_id, zones)
        self._mirror_zones_to_pipeline(setup_id)

    def _check_zone_shape_dim(self, setup_id, zones) -> None:
        """Warn when zones were drawn on a frame whose aspect ratio differs
        from the box's live crop.

        ``shape_dim`` records the (W, H) each zone was authored against. If the
        ROI or camera resolution changed since, normalized zones re-project
        onto a differently-shaped crop and land offset, the exact silent
        failure Wave 2 targets. This makes it loud (a status chip + one log
        line per box), not fatal: the zones still render, but the operator is
        told to re-check the ROI. Deduped per box so a stable mismatch doesn't
        spam every repaint.
        """
        try:
            dims = [z.get("shape_dim") for z in (zones or [])
                    if isinstance(z, dict) and z.get("shape_dim")]
            if not dims:
                return
            sw, sh = float(dims[0][0]), float(dims[0][1])
            if sw <= 0 or sh <= 0:
                return
            wh = self._box_frame_wh(setup_id)
            if not wh or wh[0] <= 0 or wh[1] <= 0:
                return  # frame size unknown yet, re-checked on next set
            live_aspect = wh[0] / wh[1]
            drawn_aspect = sw / sh
            warned = getattr(self, "_shape_dim_warned", None)
            if warned is None:
                warned = self._shape_dim_warned = {}
            # 2% tolerance absorbs rounding; a real ROI/resolution change is
            # far larger (the reported case was 1.21 vs 1.06 ≈ 14%).
            mismatch = abs(live_aspect - drawn_aspect) / drawn_aspect > 0.02
            if mismatch and warned.get(setup_id) != (sw, sh, wh):
                warned[setup_id] = (sw, sh, wh)
                logger.warning(
                    "Box %s: zones were drawn on %gx%g (aspect %.3f) but the "
                    "live crop is %sx%s (aspect %.3f), re-check the ROI, zones "
                    "may be offset.", setup_id, sw, sh, drawn_aspect,
                    wh[0], wh[1], live_aspect)
                try:
                    self._box_status_say(
                        setup_id,
                        "Zones drawn on a different region, re-check ROI")
                except (AttributeError, RuntimeError):
                    pass
            elif not mismatch:
                warned.pop(setup_id, None)
        except Exception as e:
            logger.debug("shape_dim check (box %s) failed: %s", setup_id, e)

    def _invalidate_zone_layer(self, setup_id) -> None:
        """Drop the cached zone QPixmap for a box.  Call after any
        mutation to ``tracking_zones[box_id]`` so the next paint
        rebuilds the layer with the new geometry.
        """
        try:
            self._zone_version[setup_id] = self._zone_version.get(setup_id, 0) + 1
            self._zone_layer_cache.pop(setup_id, None)
        except (AttributeError, RuntimeError):
            pass

    def _get_zone_layer(self, setup_id, fit_w, fit_h):
        """Return a cached QPixmap of the zones rendered at (fit_w, fit_h),
        or None when the box has no zones.  The layer is transparent
        outside zone shapes so it composites cleanly over the live tile.
        Rebuilt whenever zones mutate (via ``_invalidate_zone_layer``)
        or when the tile size changes."""
        zones = getattr(self, "tracking_zones", {}).get(setup_id)
        if not zones:
            return None
        version = self._zone_version.get(setup_id, 0)
        key = (version, int(fit_w), int(fit_h))
        cached = self._zone_layer_cache.get(setup_id)
        if cached is not None and cached[0] == key and not cached[1].isNull():
            return cached[1]
        # Rebuild: transparent QPixmap, draw zones via QPainter.
        try:
            qpm = QtGui.QPixmap(int(fit_w), int(fit_h))
            qpm.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(qpm)
            try:
                from source.gui.widgets.zone_overlay import render_zones_qpainter
                render_zones_qpainter(painter, zones, int(fit_w), int(fit_h))
            finally:
                painter.end()
            self._zone_layer_cache[setup_id] = (key, qpm)
            return qpm
        except Exception as e:
            logger.debug("zone layer rebuild (box %s) error: %s", setup_id, e)
            return None

    def _snapshot_zone_baseline(self, setup_id) -> None:
        """Take a deep-copy snapshot of the current zones for box_id.

        Called from ``_set_box_zones`` (every external load).  Not called
        from shift/rotate/scale, so the baseline reflects the load-time
        state, not the most-recently-adjusted state.
        """
        if not hasattr(self, "_zone_baselines"):
            self._zone_baselines = {}
        zones = getattr(self, "tracking_zones", {}).get(setup_id)
        if zones is None:
            self._zone_baselines.pop(setup_id, None)
            return
        import copy
        self._zone_baselines[setup_id] = copy.deepcopy(zones)

    def _reset_zones_to_baseline(self, setup_id) -> None:
        """Restore zones to the load-time baseline.

        Idempotent, pressing Home twice gives the same result, and the
        baseline is preserved so a sequence of (adjust, Home, adjust,
        Home) all reset to the same state.  Auto-saves the restored
        zones so the on-disk file matches what the user sees.
        """
        if self._zone_edit_locked(setup_id):
            self._zone_lock_notice(setup_id)
            return
        baseline = getattr(self, "_zone_baselines", {}).get(setup_id)
        if baseline is None:
            # No load-time snapshot, fall back to taking one now (the
            # "baseline" becomes the current state, which means Home has
            # no further effect until the user reloads or restarts).
            self._snapshot_zone_baseline(setup_id)
            try:
                self._box_status_say(setup_id, "Zone baseline now captured")
            except (AttributeError, RuntimeError):
                pass
            return
        import copy
        zones_restored = copy.deepcopy(baseline)
        self.tracking_zones[setup_id] = zones_restored
        self._invalidate_zone_layer(setup_id)
        try:
            self._rebuild_zone_manager(setup_id, zones_restored)
        except (AttributeError, RuntimeError):
            pass
        self._persist_zone_edit(setup_id)
        try:
            self.refresh_ui_state()
        except (AttributeError, RuntimeError):
            pass
        try:
            self._box_status_say(setup_id, "Zones reset to baseline")
        except (AttributeError, RuntimeError):
            pass

    def _zone_edit_locked(self, setup_id) -> bool:
        """True while the box is recording, zone geometry is frozen then.

        A mid-session shift / rotate / redraw silently changes what the
        MCU's zone events and the session's ``_video_data.txt`` columns
        mean, so every zone-mutation entry point (adjust buttons, dialog
        live preview, dialog commit, zones-save) refuses while the box is
        in ``recording_setups``.
        """
        return setup_id in (getattr(self, "recording_setups", None) or set())

    def _zone_lock_notice(self, setup_id) -> None:
        with contextlib.suppress(AttributeError, RuntimeError):
            self._box_status_say(setup_id, "Zones are locked while recording")

    def _redraw_box_overlay(self, setup_id):
        """Force an immediate repaint of the box's video tile with the
        current zone overlay. Pulls the last frame from the pipeline and
        runs ``update_box_display`` directly so the user sees zone
        shift/rotate/scale results without waiting for the next camera
        frame to arrive (no-op when no frame is cached yet).  Also
        invalidates the cached zone QPixmap layer so the rebuilt layer
        reflects the new geometry."""
        self._invalidate_zone_layer(setup_id)
        try:
            vm = getattr(self, "video_manager", None)
            if vm is None:
                return
            frame = vm.get_last_frame(setup_id)
            if frame is None:
                return
            self.update_box_display(setup_id, frame, None)
        except Exception as e:
            logger.debug("redraw_box_overlay(%s) error: %s", setup_id, e)

    def _shift_zones(self, setup_id, dx, dy):
        """Shift all zones for a box by (dx, dy) frame pixels.

        Zone storage is normalized; convert the pixel delta using the box's
        cropped-frame size.  No-op if frame size isn't yet known (avoids
        nudging zones into an arbitrary normalized space).
        """
        if self._zone_edit_locked(setup_id):
            self._zone_lock_notice(setup_id)
            return
        zones = getattr(self, "tracking_zones", {}).get(setup_id)
        if not zones:
            return
        from source.video.zones.coords import is_normalized_points
        wh = self._box_frame_wh(setup_id)
        for zone in zones:
            if zone.get("type") == "scale":
                continue
            pts = zone.get("points", [])
            if is_normalized_points(pts, zone.get("coord_space")):
                if wh is None:
                    continue
                fw, fh = wh
                ddx, ddy = dx / fw, dy / fh
            else:
                ddx, ddy = dx, dy
            for pt in pts:
                pt[0] += ddx
                pt[1] += ddy
            if zone.get("type") in ("circle", "ellipse") and "center" in zone:
                zone["center"][0] += ddx
                zone["center"][1] += ddy
        logger.info("Box %s: shift (%s, %s) px  (frame %s)",
                    setup_id, dx, dy, wh if wh else "?")
        self._rebuild_zone_manager(setup_id, zones)
        self._persist_zone_edit(setup_id)
        self._redraw_box_overlay(setup_id)

    def _rotate_zones(self, setup_id, angle_deg):
        """Rotate all zones around the combined centroid.

        IMPORTANT: rotation must happen in PIXEL space, not normalized
        [0, 1] space.  A "circle" stored as 32 normalized points on a
        non-square frame (e.g. 1280x720) is actually an *anisotropic
        ellipse* in [0, 1]² (because 1/1280 != 1/720).  Rotating that
        ellipse by N degrees in normalized space, then denormalizing
        per-axis, produces a tilted-axis ellipse on screen, the exact
        "circle becomes ellipse" symptom the user sees.

        Fix: denormalize each zone's points to pixel coords using the
        box's frame size, rotate rigidly there, then renormalize.
        Pixel-space rotation is a true rigid transform so circles stay
        circles.
        """
        if self._zone_edit_locked(setup_id):
            self._zone_lock_notice(setup_id)
            return
        zones = getattr(self, "tracking_zones", {}).get(setup_id)
        if not zones:
            logger.warning("Box %s: rotate skipped, no zones in tracking_zones",
                           setup_id)
            try:
                self._box_status_say(setup_id, "No zones to rotate")
            except (AttributeError, RuntimeError):
                pass
            return
        import math
        from source.video.zones.coords import is_normalized_points
        wh = self._box_frame_wh(setup_id)
        if wh is None:
            # Without a frame size we can't denormalize cleanly; skip and
            # tell the user rather than rotate in a broken anisotropic way.
            logger.warning(
                "Box %s: rotate skipped, frame size unknown (camera not "
                "streaming yet?). Reconnect or wait for the first frame.",
                setup_id)
            try:
                self._box_status_say(setup_id,
                                     "Rotate: connect a camera + wait for frames")
            except (AttributeError, RuntimeError):
                pass
            return
        fw, fh = wh
        try:
            angle_deg = float(angle_deg)
        except (TypeError, ValueError):
            angle_deg = 1.0
        logger.info("Box %s: rotate %s°  (frame %dx%d, %d zones)",
                    setup_id, angle_deg, fw, fh, len(zones))

        def _to_px(zone, pt):
            if is_normalized_points(zone.get("points", []),
                                    zone.get("coord_space")):
                return pt[0] * fw, pt[1] * fh
            return pt[0], pt[1]

        def _from_px(zone, x_px, y_px):
            if is_normalized_points(zone.get("points", []),
                                    zone.get("coord_space")):
                return x_px / fw, y_px / fh
            return x_px, y_px

        # Pivot = FRAME CENTRE (display centre) in pixel space, so "Rotate"
        # spins the layout in place rather than arcing it across the screen.
        cx_px = fw / 2.0
        cy_px = fh / 2.0
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        def _rotate_point_in_place(zone, pt):
            x_px, y_px = _to_px(zone, pt)
            dx, dy = x_px - cx_px, y_px - cy_px
            new_x = cx_px + dx * cos_a - dy * sin_a
            new_y = cy_px + dx * sin_a + dy * cos_a
            pt[0], pt[1] = _from_px(zone, new_x, new_y)

        for zone in zones:
            if zone.get("type") == "scale":
                continue
            for pt in zone.get("points", []):
                _rotate_point_in_place(zone, pt)
            if "center" in zone:
                _rotate_point_in_place(zone, zone["center"])
        self._rebuild_zone_manager(setup_id, zones)
        self._persist_zone_edit(setup_id)
        # Force an immediate overlay redraw so the user sees the rotation
        # without waiting for the next camera frame (matters on slow feeds).
        self._redraw_box_overlay(setup_id)

    def _scale_zones(self, setup_id, factor):
        """Scale all zones around the frame centre.

        Like ``_rotate_zones``, this MUST work in pixel space, scaling
        normalized points on a non-square frame would skew the shape
        (a circle becomes an ellipse) because the x and y unit lengths
        differ. Pivot is the frame centre so the layout zooms in/out
        of view rather than drifting toward a corner.
        """
        if self._zone_edit_locked(setup_id):
            self._zone_lock_notice(setup_id)
            return
        zones = getattr(self, "tracking_zones", {}).get(setup_id)
        if not zones:
            return
        from source.video.zones.coords import is_normalized_points
        wh = self._box_frame_wh(setup_id)
        if wh is None:
            logger.warning(
                "Box %s: scale skipped, frame size unknown "
                "(camera not streaming yet?).", setup_id)
            try:
                self._box_status_say(setup_id,
                                     "Scale: connect a camera + wait for frames")
            except (AttributeError, RuntimeError):
                pass
            return
        fw, fh = wh
        cx_px = fw / 2.0
        cy_px = fh / 2.0

        def _to_px(zone, pt):
            if is_normalized_points(zone.get("points", []),
                                    zone.get("coord_space")):
                return pt[0] * fw, pt[1] * fh
            return pt[0], pt[1]

        def _from_px(zone, x_px, y_px):
            if is_normalized_points(zone.get("points", []),
                                    zone.get("coord_space")):
                return x_px / fw, y_px / fh
            return x_px, y_px

        def _scale_point_in_place(zone, pt):
            x_px, y_px = _to_px(zone, pt)
            new_x = cx_px + (x_px - cx_px) * factor
            new_y = cy_px + (y_px - cy_px) * factor
            pt[0], pt[1] = _from_px(zone, new_x, new_y)

        for zone in zones:
            if zone.get("type") == "scale":
                continue
            for pt in zone.get("points", []):
                _scale_point_in_place(zone, pt)
            if "center" in zone:
                _scale_point_in_place(zone, zone["center"])
            if "semi_axes" in zone:
                # semi_axes is stored in the same coord space as points.
                # Pure scalar multiply works in both pixel and normalized
                # space since both axes scale by the same factor.
                zone["semi_axes"][0] *= factor
                zone["semi_axes"][1] *= factor
        logger.info("Box %s: scale x%s  (frame %dx%d, %d zones)",
                    setup_id, factor, fw, fh, len(zones))
        self._rebuild_zone_manager(setup_id, zones)
        self._persist_zone_edit(setup_id)
        self._redraw_box_overlay(setup_id)

    # ==================================================================
    # Config lineage (active path, recent cache, usage history,
    # main-tab badges). Both modes share this surface so loading and
    # session-end logging are identical regardless of mode.
    # ==================================================================

    # active_config_path is set the first time _on_config_loaded fires;
    # initialised here so callers can `getattr(..., "")` safely.
    active_config_path: str = ""
    _active_config: Optional[_cfg_schema.Config] = None
    # The project folder the active config lives in.
    # None means "no project loaded yet", Save will prompt for a name.
    _active_project_dir: Optional[Path] = None
    _session_started_at: float = 0.0

    def _on_config_loaded(self, path: str | None,
                          cfg: _cfg_schema.Config) -> None:
        """Common bookkeeping after a config is loaded from disk.

        Records the active path, bumps the global recent-configs cache
        so the main-tab "Recent" dropdown surfaces this entry, refreshes
        the status badges, and installs the typed ``cfg`` as the active
        config (the sole caller loads project folders, which always
        produce one, a dict round-trip here just built a throwaway).
        """
        if not path:
            return
        self.active_config_path = str(path)

        if not cfg.config_djb2:
            cfg.config_djb2 = _cfg_schema.djb2_hex_from_text(
                _cfg_schema.canonical_json(cfg.to_dict_for_hash()))
        self._active_config = cfg

        # Reset drop counts at the START of a new config session so each
        # session's drop tally starts clean.
        try:
            _drop_log.reset_counts()
        except (AttributeError, RuntimeError):
            pass

        self._refresh_status_badges()

        # Prompt for any runs that crashed before close_run fired on
        # the previous session. No-op for flat-file loads.
        if cfg is not None and self._active_project_dir is not None:
            self._check_unfinished_runs()

    # ==================================================================
    # Save / load, shared scaffolding
    # ==================================================================

    def save_config(self) -> None:
        """Manual Save. Builds a typed Config from the current GUI state
        via ``experiment.read_ui_into_config(self)`` and delegates to
        ``project_workflow.save_project``.
        """
        try:
            from source.config.experiment import read_ui_into_config
            # Skip when there's nothing to save.
            if not self._box_widgets:
                QtWidgets.QMessageBox.warning(
                    self, "Warning", "Nothing to save, no boxes / setups.")
                return
            cfg = read_ui_into_config(self)
            saved = _pw.save_project(self, cfg)
            # First save names the project from the chosen folder and resolves
            # the data dir, push those back into the read-only sidebar fields
            # so they show the real project name / data path (not the empty
            # startup placeholders).
            if saved is not None:
                from source.config.experiment import _apply_meta
                try:
                    _apply_meta(cfg, self)
                except Exception as e:
                    logger.debug("post-save meta refresh failed: %s", e)
        except Exception as e:
            logger.error("save_config failed: %s", e)
            self.showError(f"Failed to save configuration: {e}")

    # ==================================================================
    # load_config, orchestrator + symmetric load_X helpers
    # ==================================================================

    def load_config(self) -> None:
        """Load a project via the central GUI ↔ Config bridge.

        Sequence:
          1. mode-specific clear of current widgets (``_load_clear_existing_state``)
          2. ``_pw.load_project`` picks the folder, returns ``(Config, Path)``
          3. snapshot store + change_log primed for the project
          4. ``_load_create_widgets(cfg.setup_config.boxes)`` (mode hook)
             builds N empty box / setup widgets, one per saved box
          5. ``apply_config_to_ui(cfg, self)`` populates EVERYTHING
             (meta sidebar, video defaults, camera registry, per-box
             ROI/geometry/HD/action/zones, tracking, stats, ui)
          6. ``_set_box_zones`` for any per-box zones still in the
             pipeline (older with maze tracking dialog)
          7. ``refresh_ui_state`` + ``_load_post_restore``
        """
        try:
            if not self._load_clear_existing_state():
                return

            result = _pw.load_project(self, mode=self._load_mode_name())
            if result is None:
                return
            cfg, project_dir = result

            config_path = str(project_dir / _cfg_schema.EXPERIMENT_CONFIG_FILENAME)
            self._active_project_dir = project_dir
            self._on_config_loaded(config_path, cfg)
            self._ensure_project_data_dir(cfg)

            # SnapshotStore, one per project; sweep stale pending duplicates.
            # Loading a real project ends any draft: drop the temp workspace.
            from source.config.snapshot_store import SnapshotStore
            self._is_draft = False
            _draft = getattr(self, "_draft_store_dir", None)
            if _draft is not None:
                try:
                    import shutil
                    shutil.rmtree(_draft, ignore_errors=True)
                except Exception:
                    pass
                self._draft_store_dir = None
            self._snapshot_store = SnapshotStore(
                project_dir,
                tracking_enabled=getattr(cfg.meta, "tracking_enabled", True),
            )
            try:
                self._snapshot_store.sweep_orphan_pending()
            except (AttributeError, RuntimeError) as e:
                logger.debug("sweep_orphan_pending failed: %s", e)

            # Build empty per-box widgets so the bridge has somewhere to
            # populate. Operant and maze override _load_create_widgets to
            # call their own add_box() / _add_setup() per-row.
            self._load_create_widgets(cfg.setup_config.boxes)

            # The single bridge, meta, cameras, tracking, ROI, etc.
            from source.config.experiment import apply_config_to_ui
            apply_config_to_ui(cfg, self)

            # Mirror per-box zones into the maze tracking-dialog overlay via
            # _set_box_zones. Operant tolerates the call as a no-op when
            # there are no zones.
            self._restore_zones_from_config(cfg)

            try:
                self.refresh_ui_state()
            except (AttributeError, RuntimeError):
                pass

            logger.info("Configuration loaded successfully")
            if hasattr(self, "statusbar"):
                try:
                    self.statusbar.showMessage(
                        "Configuration loaded successfully", 3000)
                except (AttributeError, RuntimeError):
                    pass

            self._load_post_restore(cfg)

            # Central post-load auto-connect (cameras + pose init + MCU),
            # gated on the on-load preference. Both modes inherit the same
            # sequence, no mode-specific copy. Pose is warmed only when a
            # real model is configured, so an unconfigured project never
            # triggers a DLC load.
            self._auto_flow_after_load(cfg)
        except Exception as e:
            logger.error("Failed to load configuration: %s", e)
            self.showError(f"Failed to load configuration: {e}")

    def _load_mode_name(self) -> str:
        raise NotImplementedError

    def _load_clear_existing_state(self) -> bool:
        """Clear current widgets in preparation for a fresh load. Return
        False to cancel (operant prompts the user; maze never cancels).
        Default: no-op + continue."""
        return True

    def _clear_project_runtime_state(self) -> None:
        """Tear down per-project pipeline + host state.

        Single source of truth for "project is going away" cleanup,
        used by both ``_teardown_all_boxes`` (operant) and
        ``_load_clear_existing_state`` (maze). Clears the pipeline's
        per-box subscriptions + central registries and the host's zone /
        enable / dialog-override dicts, so a freshly-added box never
        inherits a prior project's tracking / camera state.

        Cleared in this order:
          1. Per-box pipeline subscriptions (each unsubscribe callable
             actually fires, clearing the dict alone would leak the
             bus → sinks edges).
          2. Pipeline central registries (``_tracking_configs``,
             ``_camera_configs``).
          3. Host runtime tracking state (``tracking_zones``,
             ``tracking_enabled``, ``_tracking_dialog_globals``,
             ``video_segment_config``).
        """
        pipe = getattr(self, "pipeline", None)
        if pipe is not None:
            unsubs = getattr(pipe, "_box_unsubs", None)
            if isinstance(unsubs, dict):
                for setup_id, callables in list(unsubs.items()):
                    for fn in list(callables or ()):
                        try:
                            fn()
                        except Exception as e:
                            logger.debug(
                                "unsubscribe error (box %s): %s", setup_id, e,
                            )
                unsubs.clear()
            for attr in ("_tracking_configs", "_camera_configs"):
                reg = getattr(pipe, attr, None)
                if isinstance(reg, dict):
                    reg.clear()
        for attr in ("tracking_zones", "tracking_enabled", "_tracking_dialog_globals"):
            d = getattr(self, attr, None)
            if isinstance(d, dict):
                d.clear()
        if hasattr(self, "video_segment_config"):
            self.video_segment_config = None

    def _load_create_widgets(self, boxes) -> None:
        """Create one per-box widget per saved box via the mode's add_setup(),
        so the GUI-Config bridge has a target to populate."""
        for _ in boxes:
            self.add_setup()

    def _load_post_restore(self, cfg) -> None:
        """Mode-specific finishing (operant: DLC-init prompt + update
        video grid; maze: auto-flow camera/MCU connects). Default no-op.
        Receives the typed v3 Config (not a dict)."""
        return

    def _restore_zones_from_config(self, cfg) -> None:
        """Mirror per-box zones from cfg.setup_config.boxes[].zones into
        the maze tracking-dialog overlay system via ``_set_box_zones``.
        Operant uses pipeline TrackingConfig.zones via a different code
        path; this is a maze convenience that's harmless for operant.
        """
        installer = getattr(self, "_set_box_zones", None)
        if not callable(installer):
            return
        # ONE zone→dict converter (drops None keys, full fidelity) shared with
        # BoxConfig.to_compact / _apply_tracking / maze.apply_mode_extras, so
        # scale / ellipse / event fields survive load (a hand-rolled field
        # subset would silently drop them).
        from source.config.experiment import zone_to_dict
        for box in cfg.setup_config.boxes:
            if not box.zones:
                continue
            try:
                installer(box.setup_number,
                          [zone_to_dict(z) for z in box.zones
                           if hasattr(z, "__dataclass_fields__") or isinstance(z, dict)])
            except Exception as e:
                logger.debug("_set_box_zones for box %s failed: %s",
                             box.setup_number, e)

    def _ensure_project_data_dir(self, cfg) -> None:
        """Eagerly create ``<data_root>/<project>/`` on project load so the
        data tree starts materialising as the experimenter walks through
        setup. The task subfolder is created later, when a task uploads."""
        try:
            from source import paths as _app_paths
            project_name = (cfg.meta.project or cfg.experiment_name or "").strip()
            if not project_name:
                return
            proj_data_dir = Path(_app_paths.top_dir) / "data" / project_name
            proj_data_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Project data dir ready: %s", proj_data_dir)
        except Exception as e:
            logger.warning("Project data dir eager mkdir failed: %s", e)

    def _background_images_dir(self) -> Optional[Path]:
        """Return the project's ``background_images/`` directory.

        Project-scoped only: returns ``None`` when no project is loaded
        (ad-hoc sessions can't persist backgrounds). Created on demand so
        callers can write straight to ``<bg_dir>/box{N}.png``.
        """
        if not self.active_config_path:
            return None
        base = Path(self.active_config_path).parent / "background_images"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _background_path_for_box(self, setup_id) -> Optional[Path]:
        """Return ``<project>/background_images/box{N}.png`` if it exists,
        else None. This is the canonical lookup; there is no JSON dict."""
        bg_dir = self._background_images_dir()
        if bg_dir is None:
            return None
        p = bg_dir / f"box{int(setup_id)}.png"
        return p if p.exists() else None

    def _capture_background_for_box(self, setup_id, get_frame=None) -> bool:
        """Capture, save and stamp this box's reference background.

        THE capture path, the tracking panel's button and the idle refresh
        both come here, so there is one definition of what a background is
        (a median burst, not a grabbed frame) and one place that writes and
        stamps it. ``get_frame`` lets the panel supply its own dialog-scoped
        camera callback; everyone else gets the live camera.
        """
        bg_dir = self._background_images_dir()
        if bg_dir is None:
            logger.info("Box %s: no project loaded, backgrounds live in the "
                        "project's background_images/ folder", setup_id)
            return False
        if get_frame is None:
            vm = getattr(self, "video_manager", None)
            if vm is None:
                return False
            def get_frame():
                return vm.get_last_frame(setup_id)
        try:
            image = _bg_mod.capture_median(
                get_frame, pump=QtWidgets.QApplication.processEvents)
            if image is None:
                image = get_frame()          # better a poor reference than none
            if image is None:
                return False
            cv2.imwrite(str(bg_dir / f"box{int(setup_id)}.png"), image)
            self._mark_bg_captured(setup_id)
            # A fresh reference supersedes the previous one entirely,
            # including any shape mismatch that was blocking this box.
            getattr(self, "_bg_shape_problem", {}).pop(int(setup_id), None)
            logger.info("Box %s: background captured (median burst)", setup_id)
            return True
        except Exception as e:
            logger.error("Box %s: background capture failed: %s", setup_id, e)
            return False

    def _bg_captured_at(self, setup_id) -> Optional[str]:
        """``BoxConfig.bg_captured_at`` for one box, or None."""
        cfg = getattr(self, "_active_config", None)
        if cfg is None:
            return None
        for b in getattr(cfg.setup_config, "boxes", []) or []:
            if b.setup_number == int(setup_id):
                return getattr(b, "bg_captured_at", None)
        return None

    def _blob_background_problem(self, setup_id) -> Optional[str]:
        """A reason this box must not start blob tracking, or None.

        Only the shape mismatch qualifies. Missing and stale backgrounds are
        warned about elsewhere and left to the operator: a rig unchanged since
        yesterday tracks perfectly well on yesterday's reference, and refusing
        to start would be wrong about that far more often than right. A
        background of the wrong shape has no reading under which it is usable.

        No record means no problem. This is a diagnostic the validation step
        fills in, so its absence must read as "nothing found", never as a
        reason to refuse to track.
        """
        return getattr(self, "_bg_shape_problem", {}).get(int(setup_id))

    def _stale_background_boxes(self) -> list:
        """Boxes whose reference background is older than the staleness bound.

        Read from ``BoxConfig.bg_captured_at`` rather than the file's mtime:
        the PNG is rewritten on every retake, but a project copied between
        machines carries the stamp of when the arena was actually shot, which
        is the thing worth warning about.
        """
        cfg = getattr(self, "_active_config", None)
        if cfg is None:
            return []
        stale = []
        for b in getattr(cfg.setup_config, "boxes", []) or []:
            age = _bg_mod.hours_since(getattr(b, "bg_captured_at", None))
            if age is not None and age >= _bg_mod.STALE_AFTER_HOURS:
                stale.append(b.setup_number)
        return stale

    def _mark_bg_captured(self, setup_id) -> None:
        """Stamp ``BoxConfig.bg_captured_at`` with the current ISO
        timestamp and schedule an autosave.

        Called by both BG capture sites (tracking panel's Take BG button
        and the tracker calibration dialog's Apply). The PNG file itself
        is overwritten on every retake, this timestamp tracks WHEN.
        ``bool(box.bg_captured_at)`` then answers "does this box have a
        BG?" without a disk hit.
        """
        if self._active_config is None:
            return
        from datetime import datetime
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for b in self._active_config.setup_config.boxes:
            if b.setup_number == int(setup_id):
                b.bg_captured_at = stamp
                break
        self._project_changed(reason="bg_captured")

    # ==================================================================
    # Autosave, debounced single-flight write of the project file
    # ==================================================================

    _autosave_timer: Optional[QtCore.QTimer] = None
    _change_reason: str = ""
    _autosave_in_flight: bool = False
    _autosave_pending_after: bool = False
    _project_save_lock: Optional[threading.Lock] = None
    AUTOSAVE_DEBOUNCE_MS = 750

    def _init_autosave(self) -> None:
        """Set up the debounced autosave timer. Mode __init__ calls this
        after Qt is ready. Idempotent, safe to call multiple times."""
        if self._autosave_timer is not None:
            return
        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(self.AUTOSAVE_DEBOUNCE_MS)
        self._autosave_timer.timeout.connect(self._autosave_now)
        if self._project_save_lock is None:
            self._project_save_lock = threading.Lock()

    def _project_changed(self, *, reason: str) -> None:
        """Schedule an autosave. Coalesces rapid changes into one save.

        No-op when no project is loaded, manual *Save Config* is still
        required to create a project the first time.
        """
        if not self.active_config_path:
            return
        if self._autosave_timer is None:
            self._init_autosave()
        self._change_reason = reason
        self._autosave_timer.start()

    def _snapshot_cfg_for_autosave(self):
        """Read live UI into a fresh Config. GUI-thread only.

        Returns None when there's nothing meaningful to save (no
        project loaded, no boxes/setup widgets, or the bridge declined).
        Performs the COM-port baseline restore so a mid-repopulation
        combo doesn't wipe saved ports.
        """
        if not self.active_config_path or self._active_project_dir is None:
            return None
        from source.config.experiment import read_ui_into_config
        if not self._box_widgets:
            return None
        cfg = read_ui_into_config(self)
        if cfg is None:
            return None
        base = getattr(self, "_active_config", None)
        if base is not None:
            saved_com = {b.setup_number: b.com_port
                         for b in base.setup_config.boxes
                         if b.com_port}
            for box in cfg.setup_config.boxes:
                if not box.com_port and box.setup_number in saved_com:
                    box.com_port = saved_com[box.setup_number]
        return cfg

    def _autosave_now(self) -> None:
        """Debounce fired, write the current cfg to disk on a worker.

        Snapshots the UI on the GUI thread, then hands the dataclass
        tree to a QThreadPool runnable that does the JSON serialize +
        atomic write. Single-flight: a dirty mark arriving while a
        worker is in flight just sets ``_autosave_pending_after`` so
        the next ``_on_autosave_done`` re-fires once.
        """
        if self._autosave_in_flight:
            self._autosave_pending_after = True
            return
        try:
            cfg = self._snapshot_cfg_for_autosave()
        except Exception as e:
            logger.warning("autosave snapshot failed (%s): %s",
                           self._change_reason, e)
            return
        if cfg is None:
            return
        from source.config.multi_instance import ProjectFileGuard
        _g = getattr(self, "_project_guard", None)
        guard = _g if isinstance(_g, ProjectFileGuard) else None
        if self._project_save_lock is None:
            self._project_save_lock = threading.Lock()
        self._autosave_in_flight = True
        runnable = _AutosaveRunnable(
            cfg=cfg,
            project_dir=self._active_project_dir,
            guard=guard,
            save_lock=self._project_save_lock,
        )
        runnable.signals.done.connect(
            self._on_autosave_done, QtCore.Qt.ConnectionType.QueuedConnection
        )
        QtCore.QThreadPool.globalInstance().start(runnable)

    def _on_autosave_done(self, ok: bool, msg: str, cfg) -> None:
        """Worker finished. Update cached state + fire any pending follow-up."""
        self._autosave_in_flight = False
        if ok:
            self._active_config = cfg
            logger.info("autosave (%s): %s",
                        self._change_reason, self.active_config_path)
            self._status_autosave_pulse()
        elif msg.startswith("deferred:"):
            _, n, paths = msg.split(":", 2)
            logger.warning(
                "autosave deferred, %s field(s) changed on disk: %s",
                n, paths,
            )
            self._status_autosave_pulse(
                msg="Autosave paused, project changed externally"
            )
        else:
            logger.warning("autosave failed (%s): %s",
                           self._change_reason, msg)
        if self._autosave_pending_after:
            self._autosave_pending_after = False
            self._autosave_now()

    def _autosave_now_sync(self) -> None:
        """Synchronous autosave, used by closeEvent flush.

        Mirrors ``_autosave_now`` but runs the disk I/O inline on the
        calling thread so the app doesn't exit before the file lands.
        """
        try:
            cfg = self._snapshot_cfg_for_autosave()
        except Exception as e:
            logger.warning("autosave (close) snapshot failed: %s", e)
            return
        if cfg is None:
            return
        from source.config.experiment import (
            save_experiment, save_template, serialize_for_save,
        )
        from source.config.multi_instance import SaveConflictError, ProjectFileGuard
        from source.gui.project_workflow import _ensure_history_dir
        _g = getattr(self, "_project_guard", None)
        guard = _g if isinstance(_g, ProjectFileGuard) else None
        if self._project_save_lock is None:
            self._project_save_lock = threading.Lock()
        try:
            with self._project_save_lock:
                payload = serialize_for_save(cfg)
                try:
                    save_experiment(
                        cfg, project_dir_path=self._active_project_dir,
                        guard=guard,
                        on_conflict=(lambda _c: "cancel") if guard else None,
                        cached_payload=payload,
                    )
                except SaveConflictError as ce:
                    logger.warning(
                        "autosave (close) deferred, %d conflict(s)",
                        len(ce.conflicts),
                    )
                    return
                save_template(cfg, self._active_project_dir,
                              cached_payload=payload)
                _ensure_history_dir(self._active_project_dir)
            self._active_config = cfg
            logger.info("autosave (close, %s): %s",
                        self._change_reason, self.active_config_path)
        except Exception as e:
            logger.warning("autosave (close) failed: %s", e)

    def _status_autosave_pulse(self, msg: str | None = None) -> None:
        """Tiny status-bar pulse to acknowledge the save (or, with
        ``msg``, to surface a deferred-save reason)."""
        sb = getattr(self, "statusbar", None) or getattr(self, "statusBar", None)
        if sb is None:
            return
        try:
            from datetime import datetime
            text = msg or f"Autosaved {datetime.now().strftime('%H:%M:%S')}"
            sb_obj = sb() if callable(sb) else sb
            sb_obj.showMessage(text, 3000 if msg else 2000)
        except Exception:
            pass

    def _flush_autosave_on_close(self) -> None:
        """Synchronous flush from closeEvent. Pending dirty marks must hit
        disk before the GUI exits.

        Three states to handle:
          1. Timer pending (debounce hasn't fired): cancel it, run a
             synchronous save inline.
          2. Worker already in flight: wait for it (briefly), then if
             a follow-up was queued, run that synchronously too.
          3. Nothing pending: no-op.
        """
        if self._autosave_timer is not None and self._autosave_timer.isActive():
            self._autosave_timer.stop()
            self._autosave_now_sync()
            return
        if self._autosave_in_flight:
            pool = QtCore.QThreadPool.globalInstance()
            # 5s upper bound, autosave is small JSON I/O; if it's not
            # done by then, give up to keep the close path snappy.
            pool.waitForDone(5000)
            self._autosave_in_flight = False
            if self._autosave_pending_after:
                self._autosave_pending_after = False
                self._autosave_now_sync()

    # ==================================================================
    # Metadata file, cohort Excel/CSV becomes a project artefact
    # ==================================================================

    def _project_adopt_metadata_file(self, src_path) -> Optional[Path]:
        """Copy a freshly-loaded cohort Excel/CSV into the project folder
        and record the relative path in ``cfg.meta.metadata_file``.

        Returns the new path inside the project, or None when there's no
        project loaded (ad-hoc sessions keep the file only in the
        in-memory excel cache). Silent overwrite if the project already
        had an adopted cohort file with the same basename.
        """
        if not self.active_config_path or self._active_config is None:
            return None
        src = Path(src_path)
        if not src.exists():
            return None
        pd = Path(self.active_config_path).parent
        meta_dir = pd / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        dst = meta_dir / src.name
        try:
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
        except (OSError, shutil.SameFileError) as e:
            logger.warning("metadata copy failed: %s", e)
            return None
        self._active_config.meta.metadata_file = f"metadata/{src.name}"
        return dst

    def _on_session_started(self) -> None:
        """Called by the per-mode start-recording handler once frames begin
        flowing. Marks the session-start guard (maze runs it once per
        session) and resets drop counts for the new session."""
        self._session_started_at = time.time()
        try:
            _drop_log.reset_counts()
        except (AttributeError, RuntimeError):
            pass

    # ---- Status badges (call _build_status_panel from the mode-specific
    # layout setup, then _refresh_status_badges to update) ----

    def _build_status_panel(self) -> QtWidgets.QWidget:
        """Bottom status bar, project + tracking state.

        Shows two contextual labels:

          - **Project**: name when a project file is loaded; otherwise
            a prompt to save one.
          - **Tracking**: ready / not-configured / actionable hint
            (e.g. "DLC configured, click Init", "Blob: no background
            captured, Take BG").

        The mode-specific main_window calls ``_refresh_status_badges``
        whenever the active config or tracking state changes.
        """
        panel = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(panel)
        h.setContentsMargins(6, 2, 6, 2)
        h.setSpacing(20)

        self._project_status_lbl = QtWidgets.QLabel("")
        self._project_status_lbl.setStyleSheet("font-size: 9pt;")
        h.addWidget(self._project_status_lbl)

        self._tracking_status_lbl = QtWidgets.QLabel("")
        self._tracking_status_lbl.setStyleSheet("font-size: 9pt;")
        h.addWidget(self._tracking_status_lbl)

        h.addStretch()
        self._status_panel = panel
        # Initial paint.
        try:
            self._refresh_status_badges()
        except (AttributeError, RuntimeError):
            pass
        return panel

    def _refresh_status_badges(self) -> None:
        """Recompute and repaint the project + tracking status labels.

        Safe to call any time. Silently no-ops when the panel hasn't
        been built yet (subclass that didn't call
        ``_build_status_panel``).
        """
        # Data-dir Browse… is enabled only while a project is loaded
        # (the data dir is project-scoped meta). Done before the panel
        # guard below so it tracks load/unload even without the badge panel.
        project_loaded = (
            bool(self.active_config_path) and self._active_config is not None)
        for browse_btn in getattr(self, "_data_dir_browse_btns", []):
            try:
                browse_btn.setEnabled(project_loaded)
            except RuntimeError:
                pass  # C++ widget already deleted

        if not hasattr(self, "_project_status_lbl"):
            return

        # ── Project status ──────────────────────────────────────────
        # Subtle muted colours, the status bar is informational, not
        # an attention sink.  Amber/green tints reserved for warnings/ok
        # signals, never at full saturation.
        if not self.active_config_path or self._active_config is None:
            self._project_status_lbl.setText(
                "No project loaded, Save Project to begin"
            )
            self._project_status_lbl.setStyleSheet(
                "color: rgba(251,191,36,0.75); font-size: 9pt;"
            )
        else:
            cfg = self._active_config
            name = cfg.experiment_name or Path(self.active_config_path).stem
            self._project_status_lbl.setText(f"Project: {name}")
            self._project_status_lbl.setStyleSheet(
                "color: rgba(110,231,183,0.85); font-size: 9pt; font-weight: 600;"
            )

        # ── Tracking status ─────────────────────────────────────────
        text, color = self._compute_tracking_status_text()
        self._tracking_status_lbl.setText(text)
        self._tracking_status_lbl.setStyleSheet(f"color: {color}; font-size: 9pt;")

    def _compute_tracking_status_text(self) -> tuple:
        """Return ``(label_text, color_hex)`` describing tracking readiness.

        Examples:
          - ``("Tracking: not configured", "#888")``
          - ``("Tracking: ready", "#22c55e")``
          - ``("Tracking: DLC configured, click Init in Tracking Panel", "#fbbf24")``
          - ``("Tracking: Blob, no background captured (Take BG)", "#fbbf24")``
        """
        # Read configured state straight from per-box TrackingConfig.
        all_tcs = {}
        try:
            all_tcs = self.pipeline.all_tracking_configs() or {}
        except (AttributeError, RuntimeError):
            pass

        # Group by the OPERATOR'S CHOICE (tracker_type), not by which
        # artifacts happen to exist. ``has_dlc()`` is true only once a model
        # path is set and ``has_blob()`` only once a background is saved, so
        # asking them meant a box whose tracker had been chosen but not yet
        # supplied read as "not configured", the panel denied the very
        # setting the operator had just made. This is the same distinction
        # ``_start_tracking_for_box_impl`` already draws for dispatch.
        pose_types = ("dlc", "sleap", "sleap-nn", "sleap_nn")
        chosen: dict = {}
        for bid, tc in all_tcs.items():
            if not getattr(tc, "online_tracking_enabled", True):
                continue          # opted out: records video, tracks nothing
            tt = (getattr(tc, "tracker_type", "") or "").lower().strip()
            if tt:
                chosen.setdefault(tt, []).append(bid)

        if not chosen:
            return "Tracking: not configured", "rgba(148,163,184,0.70)"

        pipeline = getattr(self, "pipeline", None)
        issues, ready = [], []

        # ── pose backends: name the one actually chosen ──────────────
        for tt, boxes in chosen.items():
            if tt not in pose_types:
                continue
            label = "SLEAP" if tt.startswith("sleap") else "DLC"
            missing = [b for b in boxes
                       if not getattr(all_tcs[b], "dlc_model_path", "")]
            if missing:
                ids = ", ".join(str(b) for b in missing)
                issues.append(f"{label} selected, no model set (box {ids})")
                continue
            # "A model is loaded" is not the question; "is the loaded model
            # the one THESE boxes want" is, and it is the same question Record
            # asks, through the same function.
            not_ready = [b for b in boxes if not self.pose_box_is_ready(b)]
            if not not_ready:
                ready.append(f"{label} ready")
            elif len(not_ready) == len(boxes):
                issues.append(f"{label} configured, model not loaded yet")
            else:
                ids = ", ".join(str(b) for b in not_ready)
                issues.append(f"{label} not loaded for box {ids}")

        # ── blob / Simple ────────────────────────────────────────────
        blob_boxes = [b for tt, bs in chosen.items() if tt not in pose_types
                      for b in bs]
        if blob_boxes:
            # Simple (self_norm) is background-FREE, so demanding one would be
            # a permanent false alarm for the mode whose whole point is not
            # needing it.
            needs_bg = [b for b in blob_boxes
                        if getattr(all_tcs[b], "blob_bg_mode", "static")
                        != "self_norm"]
            missing = []
            for b in needs_bg:
                try:
                    tm = pipeline.tracker._tm
                    if (getattr(tm, "backgrounds", {}) or {}).get(b) is None:
                        missing.append(b)
                except Exception:
                    pass
            if missing:
                ids = ", ".join(str(b) for b in missing)
                issues.append(f"Blob, no background (box {ids}); "
                              f"capture it in the Tracking Panel")
            else:
                stale = [b for b in self._stale_background_boxes()
                         if b in needs_bg]
                if stale:
                    ids = ", ".join(str(b) for b in stale)
                    issues.append(
                        f"Blob, background over {int(_bg_mod.STALE_AFTER_HOURS)} h "
                        f"old (box {ids}); recapture if the rig changed")
                else:
                    simple = len(blob_boxes) - len(needs_bg)
                    ready.append(f"Blob ready ({len(blob_boxes)} box"
                                 f"{'es' if len(blob_boxes) != 1 else ''}"
                                 f"{f', {simple} Simple' if simple else ''})")

        if issues:
            return "Tracking: " + "  •  ".join(issues), "rgba(251,191,36,0.75)"
        if ready:
            return "Tracking: " + "  •  ".join(ready), "rgba(110,231,183,0.85)"
        return "Tracking: ready", "rgba(110,231,183,0.85)"
