"""Pipeline types, frames, configs, overlays.

Pure data structures + small math helpers every pipeline sink / consumer
reads. No I/O, no threading, no Qt. Sections: frames (CameraFrame,
BoxFrame), camera config, tracking config, overlay state,
head-direction. All public symbols are re-exported via ``__all__``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

from source.video.zones.schema import Zone


def _num(value, default: float) -> float:
    """Read a saved number, falling back only when it is genuinely absent.

    ``float(raw.get(k, default) or default)`` turns a stored 0 into the
    default, so a knob the operator deliberately zeroed comes back switched
    on. Only ``None`` and unparseable values fall back here.
    """
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


# ============================================================================
# Frame metadata
# ============================================================================
#
# CameraFrame is one capture from one physical camera. BoxFrame is one
# box's view (cropped if CCTV-shared, otherwise the full image). The
# cam_frame_id links a captured frame to its derived BoxFrames + pose.
#
# Color caches: each sink wants a different layout for the same source
# bytes (Recorder/Display→BGR, Pose→RGB, Blob→GRAY). The converted view is
# cached so subsequent sinks skip re-running cv2.cvtColor. CCTV boxes slice
# the parent's whole-frame cache when populated. Treat ``image`` as immutable.
#
# Timestamps:
#   capture_host_ns   source.host_clock.host_ns() at grab. Latency +
#                     ordering, and
#                     the anchor the MCU fw time is derived from.
#   capture_wall      datetime at grab. Human-readable only.
#
# MCU framework time is DERIVED, not stored: each sink maps capture_host_ns
# through the box's clock anchor (pycboard.fw_ms_at) so a row's frame_fw_ms
# is the MCU time of that frame's own capture.


def convert_color(img: Optional[np.ndarray], kind: str) -> Optional[np.ndarray]:
    """BGR source → ``"rgb"`` or ``"gray"`` view.

    Returns ``None`` for ``"rgb"`` when the source is grayscale (caller's
    choice to refuse pose / promote to fake RGB); ``"gray"`` passes a 2D
    source through unchanged. Shared by CameraFrame and BoxFrame.
    """
    if img is None:
        return None
    if kind == "rgb":
        if img.ndim != 3 or img.shape[2] != 3:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@dataclass
class CameraFrame:
    """One frame from one physical camera.

    Created in the Pipeline tick and handed to FrameBus.publish_frame().
    Not frozen
    because the color-conversion cache mutates on first access.
    """

    image: np.ndarray
    cam_frame_id: int
    capture_host_ns: int
    capture_wall: datetime
    camera_id: Any
    is_shared: bool
    box_ids: Tuple[int, ...]
    _color_cache: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def image_rgb(self) -> Optional[np.ndarray]:
        """Whole-frame RGB view. Computed once per CameraFrame.

        Returns ``None`` if the source is grayscale (caller's choice
        to refuse pose / promote to fake RGB).
        """
        cached = self._color_cache.get("rgb")
        if cached is not None:
            return cached
        rgb = convert_color(self.image, "rgb")
        if rgb is None:
            return None
        self._color_cache["rgb"] = rgb
        return rgb

    @property
    def image_gray(self) -> np.ndarray:
        """Whole-frame grayscale view. Computed once per CameraFrame."""
        cached = self._color_cache.get("gray")
        if cached is not None:
            return cached
        gray = convert_color(self.image, "gray")
        self._color_cache["gray"] = gray
        return gray


@dataclass
class BoxFrame:
    """One box's view of a CameraFrame.

    In single-cam mode (one camera per box), ``image`` is the full
    CameraFrame image and ``crop_origin`` / ``crop_size`` are None.
    In CCTV mode, ``image`` is cropped to this box's ROI and the
    origin+size record the pixel bounds in the parent frame.

    ``_parent_camera_frame`` lets ``image_rgb`` / ``image_gray`` slice the
    parent's pre-converted cache when populated (O(1) numpy view);
    otherwise per-box conversion.
    """

    image: np.ndarray
    setup_id: int
    cam_frame_id: int
    camera_id: Any
    capture_host_ns: int
    capture_wall: datetime
    is_shared_camera: bool
    # Host monotonic_ns when the tick thread drained + published this frame.
    # Observational timing spine: (poll - capture) = capture→poll latency;
    # (inference-done - poll) = the inference-stage latency. None until stamped.
    poll_host_ns: Optional[int] = None
    crop_origin: Optional[Tuple[int, int]] = None
    crop_size: Optional[Tuple[int, int]] = None
    _parent_camera_frame: Optional["CameraFrame"] = field(default=None, repr=False)
    _color_cache: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def image_rgb(self) -> Optional[np.ndarray]:
        cache = self._color_cache
        cached = cache.get("rgb")
        if cached is not None:
            return cached
        rgb = self._slice_from_parent("rgb")
        if rgb is None:
            rgb = convert_color(self.image, "rgb")
        if rgb is not None:
            cache["rgb"] = rgb
        return rgb

    @property
    def image_gray(self) -> np.ndarray:
        cache = self._color_cache
        cached = cache.get("gray")
        if cached is not None:
            return cached
        gray = self._slice_from_parent("gray")
        if gray is None:
            gray = convert_color(self.image, "gray")
        cache["gray"] = gray
        return gray

    def _slice_from_parent(self, kind: str) -> Optional[np.ndarray]:
        """Slice the parent CameraFrame's cached buffer when it's
        already populated. If not, return None so the caller falls
        back to per-box conversion."""
        parent = self._parent_camera_frame
        if parent is None:
            return None
        cached_parent = parent._color_cache.get(kind)
        if cached_parent is None:
            # Shared camera: prime the parent's whole-frame conversion once
            # so all sibling boxes slice a view instead of each re-running
            # cvtColor. Dedicated camera: skip, no sibling to share to.
            if not self.is_shared_camera:
                return None
            cached_parent = parent.image_rgb if kind == "rgb" else parent.image_gray
            if cached_parent is None:
                return None
        if self.crop_origin is None or self.crop_size is None:
            return cached_parent
        x, y = self.crop_origin
        w, h = self.crop_size
        H, W = cached_parent.shape[:2]
        x = max(0, min(int(x), W))
        y = max(0, min(int(y), H))
        x2 = max(x, min(x + int(w), W))
        y2 = max(y, min(y + int(h), H))
        return cached_parent[y:y2, x:x2]


# ============================================================================
# CameraConfig
# ============================================================================

# Lower bound for the FPS combo. The hardware can usually go lower but
# behavioral video below 10 fps is rarely useful and clutters the UI.
FPS_MIN = 10

# All valid FPS choices step in this increment. A camera reporting a
# ceiling of 33 fps gets [10, 15, 20, 25, 30] (last entry rounds DOWN,
# never offer a value the camera can't deliver).
FPS_STEP = 5

# The frame rate a behavioural default aims for. Pokes, entries and rears
# resolve fine at 30 fps, and every pose backend the rig ships was trained on
# footage in this range, so buying more temporal resolution costs inference
# time without buying accuracy.
FPS_TARGET_DEFAULT = 30

# …but a mode is not disqualified for missing that target by a little. 20 fps
# is 50 ms of temporal resolution, which still scores entries and pokes, so a
# 720p/25 mode is a better default than a 480p/30 one, more than twice the
# pixels for five frames a second. Below this a mode is only a last resort.
FPS_USABLE_MIN = 20

# Pixel budget for a default mode, in megapixels. 1080p (2.07 MP) passes;
# a webcam's 4K MJPG mode does not. Pose backends letterbox their input down
# to a few hundred pixels a side anyway, so defaulting above this spends USB
# bandwidth and encoder time on detail the tracker throws away. The user can
# still pick any probed mode; this only decides what is selected for them.
DEFAULT_MODE_MAX_MP = 2.2


def default_fps_for(ceiling: float) -> int:
    """The FPS a fresh camera should start at, given its measured ceiling.

    ``FPS_TARGET_DEFAULT`` when the camera can hold it, otherwise the ceiling
    rounded to the nearest ``FPS_STEP``, i.e. as close to the target as the
    hardware allows, never above what it measured.
    """
    try:
        ceiling = float(ceiling or 0.0)
    except (TypeError, ValueError):
        return FPS_MIN
    if ceiling <= 0:
        return FPS_MIN
    top = max(FPS_MIN, round(ceiling / FPS_STEP) * FPS_STEP)
    return min(FPS_TARGET_DEFAULT, top)


def pick_default_mode(modes):
    """Choose the ``(w, h, fps)`` a camera should default to.

    Frame rate gates, resolution decides, not "biggest mode wins", which
    lands a camera on its 4K MJPG mode at 5 fps and drops frames from the
    first second of a session.

    1. A mode is a candidate if it sustains ``FPS_USABLE_MIN``. That is a
       floor, not the target: 720p/25 beats 480p/30, because five frames a
       second is not worth less than half the pixels.
    2. Among candidates, take the largest resolution within
       ``DEFAULT_MODE_MAX_MP``: detail up to the point the tracker stops
       benefiting from it, breaking ties on frame rate.
    3. If the pixel budget excludes every candidate, take the smallest of them:
       a big fast frame still beats a slow one.
    4. If nothing clears the floor, take the fastest mode there is, breaking
       ties on resolution.

    Returns ``None`` for an empty or unparseable list.
    """
    parsed = []
    for m in modes or ():
        try:
            parsed.append((int(m[0]), int(m[1]), float(m[2])))
        except (TypeError, ValueError, IndexError):
            continue
    if not parsed:
        return None

    budget = DEFAULT_MODE_MAX_MP * 1_000_000
    usable = [m for m in parsed if m[2] >= FPS_USABLE_MIN]
    if usable:
        affordable = [m for m in usable if m[0] * m[1] <= budget]
        if affordable:
            return max(affordable, key=lambda m: (m[0] * m[1], m[2]))
        return min(usable, key=lambda m: (m[0] * m[1], -m[2]))
    return max(parsed, key=lambda m: (m[2], m[0] * m[1]))


@dataclass
class TriggerConfig:
    """How a scientific camera is triggered.

    ``mode`` ``freerun`` = the camera runs on its own frame-rate clock (the
    default, today's behaviour); ``hardware`` = each frame starts on an edge of
    an input line (a secondary in a sync group, or an external timer);
    ``software`` = frames issued by an SDK command. ``source`` / ``edge`` /
    ``delay_us`` describe the hardware line and are backend-mapped at apply
    time (a FLIR ``TriggerSource`` entry name, a Ximea ``XI_TRG_*``); they are
    plain strings here so this schema carries no SDK dependency.
    """

    mode: str = "freerun"          # freerun | hardware | software
    source: str = ""               # input line name (backend-specific), e.g. "Line2"
    edge: str = "rising"           # rising | falling
    delay_us: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        return {"mode": self.mode, "source": self.source,
                "edge": self.edge, "delay_us": float(self.delay_us)}

    @classmethod
    def from_json(cls, d: Optional[Dict[str, Any]]) -> "TriggerConfig":
        d = d or {}
        return cls(
            mode=str(d.get("mode", "freerun")),
            source=str(d.get("source") or ""),
            edge=str(d.get("edge", "rising")),
            delay_us=float(d.get("delay_us", 0.0)),
        )


@dataclass
class LineOutputConfig:
    """A camera digital-output line driven from an internal signal.

    The one mechanism behind both features: point ``source`` at the exposure
    window (``exposure_active``) and the line pulses while the sensor
    integrates, wire it to an LED driver for strobe illumination, or into
    other cameras' trigger inputs for hardware sync. ``line`` / ``source`` are
    backend-mapped strings (a FLIR ``LineSelector`` + ``LineSource`` entry, a
    Ximea ``XI_GPO_*`` mode).
    """

    enabled: bool = False
    line: str = ""                 # output line name (backend-specific), e.g. "Line1"
    source: str = "exposure_active"  # exposure_active | frame_active | user_high | user_low
    inverted: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {"enabled": bool(self.enabled), "line": self.line,
                "source": self.source, "inverted": bool(self.inverted)}

    @classmethod
    def from_json(cls, d: Optional[Dict[str, Any]]) -> "LineOutputConfig":
        d = d or {}
        return cls(
            enabled=bool(d.get("enabled", False)),
            line=str(d.get("line") or ""),
            source=str(d.get("source", "exposure_active")),
            inverted=bool(d.get("inverted", False)),
        )


def _trigger_from_json_compat(raw: Dict[str, Any]) -> "TriggerConfig":
    """Read the trigger config, falling back to the legacy top-level
    ``external_trigger`` bool that predates the structured schema (True →
    hardware trigger). New saves carry a ``trigger`` block and win.

    Deliberate exception to the no-legacy-branches rule: this gates
    READING old on-disk camera configs (pinned by test_camera_io_config),
    and write-side always emits the structured block."""
    if raw.get("trigger") is not None:
        return TriggerConfig.from_json(raw.get("trigger"))
    if bool(raw.get("external_trigger", False)):
        return TriggerConfig(mode="hardware")
    return TriggerConfig()


@dataclass
class CameraConfig:
    """The persisted record for one physical camera, what the rig is set to.

    Edited by the camera setup dialog, saved into the project file, and read
    back on load. Distinct from
    ``source.video.cameras.base.ResolvedCameraSettings``, which is the frozen
    outcome of a single ``configure()`` call on the device. The two names are
    kept distinct deliberately: called the same thing, a call site gives the
    reader no way to tell which of them is in hand.

    Two fields the user picks (``selected_resolution``, ``selected_fps``)
    are constrained to values from ``probed_modes`` so the user can
    never request something the camera can't deliver. ``probed_modes``
    is filled by the connect dialog's "Detect Max" probe OR by loading
    a previously-saved config.
    """

    camera_id: str

    selected_resolution: Optional[Tuple[int, int]] = None
    selected_fps: Optional[int] = None

    # Detected modes, each tuple is (width, height, realistic_max_fps)
    # where the FPS is the ceiling actually delivered, not the spec sheet.
    probed_modes: List[Tuple[int, int, float]] = field(default_factory=list)

    # The OS capture backend to open a UVC camera through, by NAME
    # ("dshow"/"msmf"/"v4l2"); "" means walk the platform's preference order.
    #
    # It belongs beside the resolution and the rate because it is the same
    # kind of choice and it constrains the same ceiling: measured on this rig,
    # one camera delivered 10 fps at 720p through DirectShow and 30 through
    # Media Foundation, because only one of them honoured the MJPG request.
    # Picking a mode without picking the door that serves it is picking half
    # the setting.
    capture_backend: str = ""

    # ``probed_modes`` per backend name, what each door can do, from which
    # ``capture_backend`` and ``selected_*`` are chosen. The flat
    # ``probed_modes`` above stays as the modes of the CHOSEN backend, so
    # every existing reader keeps working.
    probed_variants: dict = field(default_factory=dict)

    grayscale: bool = False

    # Geometry correction, applied at CAPTURE, before the bus, so the ROI
    # crop, the tracker, the recorder, the zones and the coordinates pushed to
    # the MCU all describe the same picture. Applying either of these at
    # display time only would leave the operator's left/right agreeing with
    # the screen and disagreeing with every number the session records.
    #
    # ``flip_horizontal`` is the one operators actually need: most webcams
    # mirror their output by default, so the animal's left appears on the
    # right. ``flip_vertical`` covers a camera mounted upside-down.
    flip_horizontal: bool = False
    flip_vertical: bool = False

    # "accept" = write whatever frames the encoder kept up with, no
    # post-processing. "remux" runs a second ffmpeg pass after stop.
    frame_strategy: str = "accept"   # accept | remux
    camera_backend: str = "opencv"  # opencv | spinnaker | ximea
    # UVC capture pixel format (OpenCV backend): "mjpeg" (compressed, keeps
    # full FPS over USB, default) or "yuv" (uncompressed, may cap FPS). None
    # = default (mjpeg). Distinct from the recorder's OUTPUT codec.
    capture_format: Optional[str] = None

    # Scientific-camera controls (Spinnaker/Ximea). Per-camera and
    # persisted with the project so each camera keeps its own, one shared
    # global value cannot serve a mixed rig. ``None`` = driver default /
    # auto.
    exposure_us: Optional[float] = None
    gain_db: Optional[float] = None
    # Triggering + line output (strobe illumination and hardware sync, see
    # TriggerConfig / LineOutputConfig). ``sync_role`` places this camera in a
    # sync group: ``primary`` free-runs and drives its output line;
    # ``secondary`` hardware-triggers off the primary's line (its wired input
    # line is ``trigger.source``). ``none`` = standalone (default).
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    line_output: LineOutputConfig = field(default_factory=LineOutputConfig)
    sync_role: str = "none"          # none | primary | secondary

    # Values for backend-declared CameraFeature descriptors, keyed by feature
    # key. Opaque on purpose: a newly added SDK persists its own settings here
    # without a schema change, and unknown keys survive a save/load round trip.
    features: Dict[str, Any] = field(default_factory=dict)

    # Vendor-specific tunables not modelled above (white-balance, gamma, …).
    # Opaque so adding a new vendor parameter doesn't edit this dataclass.
    extra: Dict[str, Any] = field(default_factory=dict)

    # ``True`` once the user clicks Connect/Apply in the camera_connect
    # dialog. Pipeline-template rows (e.g. "cam0" with a synthetic mode)
    # keep this False so the project save filter omits them.
    user_applied: bool = False

    def max_fps_for(self, resolution: Tuple[int, int]) -> Optional[float]:
        """Realistic FPS ceiling for the given resolution, or None.

        Reads ``probed_modes`` first, then falls back to the per-door
        ``probed_variants``, taking the best door for that size.

        The fallback is not a nicety. Detect records what each OS backend
        measured, per door, and a camera calibrated that way can have an
        EMPTY ``probed_modes``. Without it this returned None, so
        ``fps_options_for`` returned no options and ``clamp_fps_to_options``
        concluded the rate was invalid and dropped it: a camera saved at
        25 fps came back at whatever the row happened to default to, with
        nothing saying the saved value had been discarded.
        """
        if resolution is None:
            return None
        want = tuple(resolution)
        for (w, h, mf) in self.probed_modes:
            if (w, h) == want:
                return float(mf)
        best = None
        for modes in (self.probed_variants or {}).values():
            for m in (modes or []):
                try:
                    if (int(m[0]), int(m[1])) == want:
                        f = float(m[2])
                        if f > 0 and (best is None or f > best):
                            best = f
                except (TypeError, ValueError, IndexError):
                    continue
        return best

    def fps_options_for(self, resolution: Tuple[int, int]) -> List[int]:
        """Return the FPS combo's items for this resolution.

        ``[FPS_MIN, FPS_MIN+FPS_STEP, …, last]`` where ``last`` is the measured
        ceiling ROUNDED to the nearest ``FPS_STEP`` (29.2 → 30, 24.2 → 25),
        so a camera that measures just under a round value still offers it as
        the top option. If the camera can't reach FPS_MIN, the single value
        (the camera's max, as int) is returned so the combo isn't empty.
        """
        mx = self.max_fps_for(resolution)
        if mx is None:
            return []
        if mx < FPS_MIN:
            return [int(mx)] if mx > 0 else []
        last = max(FPS_MIN, int(round(mx / FPS_STEP)) * FPS_STEP)
        return list(range(FPS_MIN, last + 1, FPS_STEP))

    def clamp_fps_to_options(self) -> None:
        """Snap ``selected_fps`` to the highest valid option if no longer
        valid. Drop to None if the resolution itself is gone."""
        if self.selected_resolution is None:
            self.selected_fps = None
            return
        opts = self.fps_options_for(self.selected_resolution)
        if not opts:
            self.selected_fps = None
            return
        if self.selected_fps not in opts:
            self.selected_fps = opts[-1]   # the camera's max

    def to_json(self) -> Dict[str, Any]:
        """JSON-safe dict for settings.json. Tuples become lists."""
        return {
            "camera_id": self.camera_id,
            "selected_resolution": (
                list(self.selected_resolution)
                if self.selected_resolution is not None else None),
            "selected_fps": self.selected_fps,
            "probed_modes": [[w, h, float(mf)] for (w, h, mf) in self.probed_modes],
            "grayscale": bool(self.grayscale),
            "flip_horizontal": bool(self.flip_horizontal),
            "flip_vertical": bool(self.flip_vertical),
            "frame_strategy": str(self.frame_strategy),
            "camera_backend": str(self.camera_backend),
            "capture_format": self.capture_format,
            "capture_backend": str(self.capture_backend or ""),
            # Saved so a project carries its own record of what each
            # backend measured. The machine-level calibration store
            # still wins on a host that has measured for itself: the
            # rate depends on that host's USB controller, so another
            # rig's numbers are a starting point, not a fact.
            "probed_variants": {
                str(k): [[int(m[0]), int(m[1]), float(m[2])]
                         for m in (v or []) if len(m) >= 3]
                for k, v in (self.probed_variants or {}).items()
            },
            "exposure_us": self.exposure_us,
            "gain_db": self.gain_db,
            "trigger": self.trigger.to_json(),
            "line_output": self.line_output.to_json(),
            "sync_role": str(self.sync_role),
            "features": dict(self.features or {}),
            "extra": dict(self.extra or {}),
            "user_applied": bool(self.user_applied),
        }

    @classmethod
    def from_json(cls, raw: Dict[str, Any],
                  camera_id: Optional[str] = None) -> "CameraConfig":
        """Inverse of ``to_json``. Tolerant of missing keys."""
        cam_id = camera_id or raw.get("camera_id") or ""
        sr = raw.get("selected_resolution")
        if sr is not None:
            sr = (int(sr[0]), int(sr[1]))
        modes = []
        for m in raw.get("probed_modes", []) or []:
            try:
                modes.append((int(m[0]), int(m[1]), float(m[2])))
            except (TypeError, ValueError, IndexError):
                continue
        return cls(
            camera_id=str(cam_id),
            selected_resolution=sr,
            selected_fps=int(raw["selected_fps"]) if raw.get("selected_fps") is not None else None,
            probed_modes=modes,
            grayscale=bool(raw.get("grayscale", False)),
            flip_horizontal=bool(raw.get("flip_horizontal", False)),
            flip_vertical=bool(raw.get("flip_vertical", False)),
            frame_strategy=str(raw.get("frame_strategy", "accept")),
            camera_backend=str(raw.get("camera_backend", "opencv")),
            capture_format=(str(raw["capture_format"])
                            if raw.get("capture_format") else None),
            capture_backend=str(raw.get("capture_backend") or ""),
            probed_variants={
                str(k): [(int(m[0]), int(m[1]), float(m[2]))
                         for m in (v or []) if len(m) >= 3]
                for k, v in (raw.get("probed_variants") or {}).items()
            },
            exposure_us=(float(raw["exposure_us"])
                         if raw.get("exposure_us") is not None else None),
            gain_db=(float(raw["gain_db"])
                     if raw.get("gain_db") is not None else None),
            trigger=_trigger_from_json_compat(raw),
            line_output=LineOutputConfig.from_json(raw.get("line_output")),
            sync_role=str(raw.get("sync_role", "none")),
            features=dict(raw.get("features") or {}),
            extra=dict(raw.get("extra") or {}),
            user_applied=bool(raw.get("user_applied", False)),
        )


# ============================================================================
# TrackingConfig
# ============================================================================

DEFAULT_KEYPOINTS = ("head", "center", "tailbase")
DEFAULT_ROTATION_PAIR = ("tailbase", "head")
DEFAULT_CONFIDENCE = 0.5
DEFAULT_BODY_PART = "centroid"


@dataclass
class TrackingConfig:
    """Authoritative tracking config for one box."""

    setup_id: int

    # DLC / SLEAP pose. ``tracker_type`` dispatches between the two
    # backends, both share the rest of the pose fields.
    tracker_type: str = "dlc"
    dlc_model_path: Optional[str] = None
    keypoint_names: Tuple[str, ...] = DEFAULT_KEYPOINTS
    # Connected pairs of keypoint NAMES, for drawing only, inference never
    # reads them. Filled from the model when it declares a skeleton (sleap-nn
    # records one in training_config.yaml; an exported DeepLabCut folder ships
    # none) and editable in the tracking dialog for the models that do not.
    skeleton: Tuple[Tuple[str, str], ...] = ()
    confidence_threshold: float = DEFAULT_CONFIDENCE
    pose_resize_factor: float = 1.0
    pose_n_instances: int = 1

    # SLEAP-specific (ignored for DLC/blob). ``sleap_model_type`` "auto" =
    # detect from the model folder; top-down needs the paired centroid model in
    # ``sleap_centroid_path``. ``sleap_runtime`` picks native torch vs an
    # exported ONNX/TensorRT engine (the Jetson fast path). fp16/device/compile
    # are torch-backend speed knobs. Threaded to the SLEAP tracker via
    # ``derived_sleap_opts`` → PoseSink.configure_model(sleap_opts=…).
    sleap_model_type: str = "auto"
    sleap_centroid_path: Optional[str] = None
    sleap_runtime: str = "auto"        # auto | native | onnx | tensorrt
    sleap_device: str = "auto"         # auto | cuda | cpu
    sleap_fp16: bool = False
    sleap_compile: bool = False
    sleap_peak_threshold: float = 0.2
    # DLC-specific (ignored for SLEAP/blob), mirroring the SLEAP block above.
    # ``dlc_model_type`` selects the engine: ``auto`` takes the fastest the
    # folder and this machine can run, ``onnx`` this repo's exported graph,
    # ``pytorch`` the folder's own .pt snapshot, ``base``/``lite`` DeepLabCut-
    # Live's TensorFlow runners. The default is ``auto`` and not ``base``,
    # because ``base`` names TensorFlow, so it failed with
    # ``ModuleNotFoundError: tensorflow`` on every rig that has none, for a
    # PyTorch model sitting right there. ``dlc_precision`` reaches only the
    # pytorch engine. Resolved by ``source.video.tracking.dlc_engine`` and
    # carried via ``derived_dlc_opts`` → PoseSink.configure_model(dlc_opts=…).
    dlc_model_type: str = "auto"       # auto | onnx | pytorch | base | lite
    dlc_precision: str = "FP32"        # FP32 | FP16
    dlc_device: str = "auto"           # auto | cuda | cuda:N | cpu
    # Channel mode fed to the network. ``auto`` follows what the model's own
    # config declares (ensure_rgb / ensure_grayscale / DLC is RGB); the
    # explicit values override it for a model whose config is silent.
    pose_colour_mode: str = "auto"     # auto | rgb | grayscale

    # How the box's frame is turned into the model's input.
    #
    #   full, hand the frame over as it is
    #   letterbox, scale + pad to one canonical shape, so differently
    #                 cropped boxes share one model (the default, and right for
    #                 a model trained on whole frames)
    #   crop_track, cut a fixed window at NATIVE resolution and follow the
    #                 animal with it; the only correct mode for a model trained
    #                 on crops, because scaling destroys the pixel scale it
    #                 learned. SLEAP single-instance only: top-down models crop
    #                 internally and bottom-up needs the whole frame.
    #   dlc_dynamic, the same idea, done by DeepLabCut-Live itself, which
    #                 restores the crop offsets on output.
    #
    # Absent means ``auto``: the model's own config decides, which is the only
    # setting that gets a crop-trained model right. A model trained on
    # native-scale crops declares an input much smaller than the frame, and
    # letterboxing the whole arena onto it shrinks the animal past anything the
    # network saw, measured at 9-11 px of error against 1.7-2.1 px for the
    # window, on the model's own shipped self-check. A project that states a
    # mode still gets exactly that mode.
    pose_input_mode: str = "auto"
    # The training window, in pixels. 0 = unset: seeded from the model config
    # when it declares one, otherwise the operator must state it, a model can
    # keep its crop in the DATASET rather than the config, and then nothing but
    # the training script knows.
    pose_input_w: int = 0
    pose_input_h: int = 0
    # Steering the window. ``conf_min`` is the confidence at which a keypoint
    # is worth following; ``good_min`` how many are needed before the box counts
    # as tracked; ``reacquire`` whether a lost window sweeps a coarse grid to
    # find the animal again.
    pose_crop_conf_min: float = 0.20
    pose_crop_good_min: int = 3
    pose_crop_reacquire: bool = True
    # How many consecutive frames a keypoint may be PREDICTED across before it
    # is reported missing. 0 turns per-keypoint filtering off entirely and the
    # raw network output is used. 5 is about 170 ms at 30 fps: long enough for
    # a paw hidden by the body, short enough not to invent an animal. The
    # filled parts are named in the session file, never passed off as measured.
    pose_gap_frames: int = 5
    # DeepLabCut-Live's own dynamic cropping: (threshold, margin). Only read
    # when ``pose_input_mode`` is ``dlc_dynamic``.
    dlc_dynamic_threshold: float = 0.5
    dlc_dynamic_margin: int = 10

    # Multi-animal (SLEAP only; blob/DLC stay single). ``n_animals`` 1 = single
    # (the default everywhere). ``identity_method`` none = single/untracked,
    # ``tracker`` = online frame-to-frame association, ``id_model`` = a
    # multi-class model that predicts a fixed identity per animal.
    n_animals: int = 1
    identity_method: str = "none"       # none | tracker | id_model
    identities: Tuple[str, ...] = ()    # fixed identity labels (id_model)

    # Rotation (off by default; user picks which two keypoints).
    rotation_enabled: bool = False
    rotation_keypoints: Tuple[str, str] = DEFAULT_ROTATION_PAIR

    # Blob detection (only used when ``has_dlc()`` is False).
    # Background subtraction + connected-component detection. Each
    # field maps 1:1 onto the blob tracker kwarg of the same name.
    blob_threshold: int = 30
    # Contour area bounds in pixels². A mouse silhouette is 1500–8000 px² at
    # any realistic rig resolution, so an upper bound near 500 rejected the
    # animal itself and left the tracker only specks, shadows, bedding, an
    # IR hotspot, to lock onto. The upper bound exists to reject a
    # whole-arena blob, so it belongs well above the subject, not below it.
    blob_min_area: int = 150
    blob_max_area: int = 20000
    blob_detect_dark: bool = True
    blob_blur_mode: Optional[str] = None        # "gaussian" | "median" | None
    blob_blur_kernel_size: Optional[int] = None
    blob_bg_mode: Optional[str] = None          # static|running_avg|mog2|self_norm|None
    blob_open_kernel_size: Optional[int] = None
    blob_close_kernel_size: Optional[int] = None
    # Contrast / thresholding preprocessing. CLAHE (local histogram
    # equalisation) and Otsu adaptive threshold are user-settable in the
    # tracking dialog and consumed by blob.py; carry them so they reach the
    # live tracker instead of falling back to the tracker's class defaults.
    blob_use_clahe: bool = False
    blob_clahe_clip_limit: float = 3.0
    blob_clahe_tile_size: int = 8
    blob_use_adaptive_threshold: bool = False
    # self_norm ("Simple", background-free) calibration. ratio 0.0 means "not
    # calibrated yet", the tracker auto-estimates from the first frame; a
    # saved non-zero value is restored and pinned so it survives reload.
    # smooth_sigma/minsize: -1 = auto (scaled to frame height), 0 = off;
    # blob.py owns the numbers (SELF_NORM_*) and a test asserts these copies
    # agree with it.
    blob_self_norm_ratio: float = 0.0
    blob_self_norm_sigma: float = 0.0
    blob_self_norm_smooth_sigma: float = -1.0
    blob_self_norm_minsize: int = -1
    # On by default: the dialog checkbox has always been checked, and a
    # stored default of False meant a project saved before the key
    # existed loaded with smoothing off while the UI showed it on.
    # Named `blob_` for history; one enhancer serves blob AND pose.
    blob_smooth_tracking: bool = True
    blob_background_path: Optional[str] = None  # absolute path on disk

    # Zones (geometry + per-zone policy).
    zones: List[Zone] = field(default_factory=list)

    # Recording overlay (burn annotations into recorded video?).
    annotation_enabled: bool = False

    # Online inference opt-out. True: tracker runs alongside recording.
    # False: video + per-frame TXT still produced but the pose column
    # stays empty for offline DLC later.
    online_tracking_enabled: bool = True

    # MCU push toggles.
    push_zones_to_mcu: bool = True
    push_coords_to_mcu: bool = True
    # Per-frame pose event: fire a silent intrinsic ``frame_event`` to the MCU
    # on every pose result (~pose rate), so a task can poll c.* every frame
    # instead of only on zone-change edges. OFF by default, opt-in to A/B it.
    push_frame_event: bool = False

    # Body part that defines the ``zone_changed`` event. MCUPusher diffs
    # only this body_part's zone occupancy between ticks; entry and exit
    # both fire. "centroid" is the synthetic average PoseSink injects for
    # every tracker mode.
    zone_change_body_part: str = DEFAULT_BODY_PART

    # Authored Event-Triggers table (the "Event & Trigger" tab): explicit
    # per-box rules with advanced conditions (speed / rotation / facing /
    # elongation / rearing / zone), each with an event to fire + chip colour
    # + video/plot presentation flags. Merged on top of the per-zone
    # translation (TrackingPushPolicy.configure_from_zones) inside the MCU
    # push policy (the single evaluator). Each entry: ``{condition,
    # body_part, zones, event_name, threshold?, color?, show_on_video?,
    # plot?}``.
    triggers: List[Dict[str, Any]] = field(default_factory=list)

    # User-intent marker. Flipped to True only when the user clicks
    # Apply in the tracking dialog. Pipeline auto-creates a TC for every
    # connected box; those defaults keep ``user_applied=False`` so the
    # project save filter omits them.
    user_applied: bool = False

    def derived_sleap_opts(self) -> Dict[str, Any]:
        """SLEAP tracker kwargs for ``PoseSink.configure_model(sleap_opts=…)``.
        Empty for non-SLEAP backends so DLC/blob never receive them."""
        if str(self.tracker_type).lower() not in ("sleap", "sleap-nn", "sleap_nn"):
            return {}
        return {
            # Namespaced, because both backends spell this option the same way
            # and read it differently. ``filter_options`` resolves the prefixed
            # name to the constructor's ``model_type``; a bare one is dropped as
            # ambiguous, which is how every SLEAP run silently ran on "auto".
            "sleap_model_type": self.sleap_model_type or "auto",
            "centroid_path":  self.sleap_centroid_path or None,
            "runtime":        self.sleap_runtime or "auto",
            "device":         self.sleap_device or "auto",
            "fp16":           bool(self.sleap_fp16),
            "compile":        bool(self.sleap_compile),
            "peak_threshold": float(self.sleap_peak_threshold),
        }

    def derived_dlc_opts(self) -> Dict[str, Any]:
        """DLC tracker kwargs for ``PoseSink.configure_model(dlc_opts=…)``.
        Empty for non-DLC backends so SLEAP/blob never receive them."""
        if str(self.tracker_type).lower() != "dlc":
            return {}
        return {
            # See derived_sleap_opts: the prefix is what carries this through
            # filter_options instead of being dropped as ambiguous.
            "dlc_model_type": self.dlc_model_type or "auto",
            "precision":  self.dlc_precision or "FP32",
            "device":     self.dlc_device or "auto",
            # DeepLabCut-Live's own following window. Ours is written only for
            # SLEAP single-instance, which is the one family with no SDK
            # equivalent, DLC restores the crop offsets itself.
            "dynamic_crop": self.pose_input_mode == "dlc_dynamic",
            "dynamic_threshold": float(self.dlc_dynamic_threshold),
            "dynamic_margin": int(self.dlc_dynamic_margin),
        }

    def has_dlc(self) -> bool:
        return bool(self.dlc_model_path)

    def has_blob(self) -> bool:
        """True when blob detection actually has a configured input, a
        saved background image, or the background-free ``self_norm`` mode.

        Zones deliberately do NOT count: they are tracker-agnostic
        geometry. Counting them made a zones-only box "blob-configured",
        so Test Tracking started blob on boxes whose operator never
        picked a tracker.
        """
        return (bool(self.blob_background_path)
                or str(self.blob_bg_mode or "") == "self_norm")

    def to_json(self) -> Dict[str, Any]:
        """JSON-safe dict for settings.json."""
        return {
            "box_id":               int(self.setup_id),
            "tracker_type":         str(self.tracker_type),
            "dlc_model_path":       self.dlc_model_path,
            "keypoint_names":       list(self.keypoint_names),
            "skeleton":             [list(e) for e in self.skeleton],
            "confidence_threshold": float(self.confidence_threshold),
            "pose_resize_factor":   float(self.pose_resize_factor),
            "pose_n_instances":     int(self.pose_n_instances),
            "sleap_model_type":     str(self.sleap_model_type),
            "sleap_centroid_path":  self.sleap_centroid_path,
            "sleap_runtime":        str(self.sleap_runtime),
            "sleap_device":         str(self.sleap_device),
            "sleap_fp16":           bool(self.sleap_fp16),
            "sleap_compile":        bool(self.sleap_compile),
            "sleap_peak_threshold": float(self.sleap_peak_threshold),
            "dlc_model_type":       str(self.dlc_model_type or "auto"),
            "dlc_precision":        str(self.dlc_precision or "FP32"),
            "dlc_device":           str(self.dlc_device or "auto"),
            "pose_colour_mode":     str(self.pose_colour_mode or "auto"),
            "pose_input_mode":      str(self.pose_input_mode or "letterbox"),
            "pose_input_w":         int(self.pose_input_w),
            "pose_input_h":         int(self.pose_input_h),
            "pose_crop_conf_min":   float(self.pose_crop_conf_min),
            "pose_crop_good_min":   int(self.pose_crop_good_min),
            "pose_crop_reacquire":  bool(self.pose_crop_reacquire),
            "pose_gap_frames":      int(self.pose_gap_frames),
            "dlc_dynamic_threshold": float(self.dlc_dynamic_threshold),
            "dlc_dynamic_margin":   int(self.dlc_dynamic_margin),
            "n_animals":            int(self.n_animals),
            "identity_method":      str(self.identity_method),
            "identities":           list(self.identities),
            "rotation_enabled":     bool(self.rotation_enabled),
            "rotation_keypoints":   list(self.rotation_keypoints),
            "zones": [_zone_to_json(z) for z in self.zones],
            "annotation_enabled":     bool(self.annotation_enabled),
            "online_tracking_enabled": bool(self.online_tracking_enabled),
            "push_zones_to_mcu":      bool(self.push_zones_to_mcu),
            "push_coords_to_mcu":     bool(self.push_coords_to_mcu),
            "push_frame_event":       bool(self.push_frame_event),
            "zone_change_body_part":  str(self.zone_change_body_part),
            "triggers":               [dict(t) for t in self.triggers],
            "blob_threshold":          int(self.blob_threshold),
            "blob_min_area":           int(self.blob_min_area),
            "blob_max_area":           int(self.blob_max_area),
            "blob_detect_dark":        bool(self.blob_detect_dark),
            "blob_blur_mode":          self.blob_blur_mode,
            "blob_blur_kernel_size":   self.blob_blur_kernel_size,
            "blob_bg_mode":            self.blob_bg_mode,
            "blob_open_kernel_size":   self.blob_open_kernel_size,
            "blob_close_kernel_size":  self.blob_close_kernel_size,
            "blob_use_clahe":          bool(self.blob_use_clahe),
            "blob_clahe_clip_limit":   float(self.blob_clahe_clip_limit),
            "blob_clahe_tile_size":    int(self.blob_clahe_tile_size),
            "blob_use_adaptive_threshold": bool(self.blob_use_adaptive_threshold),
            "blob_self_norm_ratio":    float(self.blob_self_norm_ratio),
            "blob_self_norm_sigma":    float(self.blob_self_norm_sigma),
            "blob_self_norm_smooth_sigma": float(self.blob_self_norm_smooth_sigma),
            "blob_self_norm_minsize":  int(self.blob_self_norm_minsize),
            "blob_smooth_tracking":    bool(self.blob_smooth_tracking),
            "blob_background_path":    self.blob_background_path,
            "user_applied":            bool(self.user_applied),
        }

    @classmethod
    def from_json(cls, raw: Dict[str, Any],
                  setup_id: Optional[int] = None) -> "TrackingConfig":
        bid = setup_id if setup_id is not None else int(raw.get("box_id", 0))
        kp = raw.get("keypoint_names") or DEFAULT_KEYPOINTS
        rot_kp = raw.get("rotation_keypoints") or DEFAULT_ROTATION_PAIR
        zones = [_zone_from_json(z) for z in (raw.get("zones") or [])]
        return cls(
            setup_id=bid,
            tracker_type=str(raw.get("tracker_type", "dlc") or "dlc"),
            dlc_model_path=raw.get("dlc_model_path"),
            keypoint_names=tuple(kp),
            skeleton=tuple(
                (str(e[0]), str(e[1]))
                for e in (raw.get("skeleton") or []) if len(e) == 2),
            confidence_threshold=float(raw.get("confidence_threshold", DEFAULT_CONFIDENCE)),
            pose_resize_factor=float(raw.get("pose_resize_factor", 1.0)),
            pose_n_instances=int(raw.get("pose_n_instances", 1)),
            sleap_model_type=str(raw.get("sleap_model_type", "auto") or "auto"),
            sleap_centroid_path=raw.get("sleap_centroid_path"),
            sleap_runtime=str(raw.get("sleap_runtime", "auto") or "auto"),
            sleap_device=str(raw.get("sleap_device", "auto") or "auto"),
            sleap_fp16=bool(raw.get("sleap_fp16", False)),
            sleap_compile=bool(raw.get("sleap_compile", False)),
            sleap_peak_threshold=float(raw.get("sleap_peak_threshold", 0.2)),
            dlc_model_type=str(raw.get("dlc_model_type", "auto") or "auto"),
            dlc_precision=str(raw.get("dlc_precision", "FP32") or "FP32"),
            dlc_device=str(raw.get("dlc_device", "auto") or "auto"),
            pose_colour_mode=str(raw.get("pose_colour_mode", "auto") or "auto"),
            # Absent = letterbox = exactly the behaviour a project saved before
            # these existed already had.
            pose_input_mode=str(raw.get("pose_input_mode") or "letterbox"),
            # ``_num`` and not ``or``: a window the operator deliberately
            # cleared to 0 means "take it from the model", and ``or`` would
            # read that as absent and put the old number back.
            pose_input_w=int(_num(raw.get("pose_input_w"), 0)),
            pose_input_h=int(_num(raw.get("pose_input_h"), 0)),
            pose_crop_conf_min=_num(raw.get("pose_crop_conf_min"), 0.20),
            pose_crop_good_min=int(_num(raw.get("pose_crop_good_min"), 3)),
            pose_crop_reacquire=bool(raw.get("pose_crop_reacquire", True)),
            pose_gap_frames=int(raw.get("pose_gap_frames", 5) or 0),
            dlc_dynamic_threshold=_num(raw.get("dlc_dynamic_threshold"), 0.5),
            dlc_dynamic_margin=int(_num(raw.get("dlc_dynamic_margin"), 10)),
            n_animals=int(raw.get("n_animals", 1) or 1),
            identity_method=str(raw.get("identity_method", "none") or "none"),
            identities=tuple(raw.get("identities", ()) or ()),
            rotation_enabled=bool(raw.get("rotation_enabled", False)),
            rotation_keypoints=(str(rot_kp[0]), str(rot_kp[1])),
            zones=zones,
            annotation_enabled=bool(raw.get("annotation_enabled", False)),
            online_tracking_enabled=bool(raw.get("online_tracking_enabled", True)),
            push_zones_to_mcu=bool(raw.get("push_zones_to_mcu", True)),
            push_coords_to_mcu=bool(raw.get("push_coords_to_mcu", True)),
            push_frame_event=bool(raw.get("push_frame_event", False)),
            zone_change_body_part=str(
                raw.get("zone_change_body_part", DEFAULT_BODY_PART) or DEFAULT_BODY_PART),
            triggers=[dict(t) for t in (raw.get("triggers") or [])
                      if isinstance(t, dict)],
            blob_threshold=int(raw.get("blob_threshold", 30)),
            # Fall back to the dataclass defaults, 50/500 here would
            # resurrect the animal-rejecting bounds the field comment
            # warns about whenever an old save lacks these keys.
            blob_min_area=int(raw.get("blob_min_area", 150)),
            blob_max_area=int(raw.get("blob_max_area", 20000)),
            blob_detect_dark=bool(raw.get("blob_detect_dark", True)),
            blob_blur_mode=raw.get("blob_blur_mode"),
            blob_blur_kernel_size=raw.get("blob_blur_kernel_size"),
            blob_bg_mode=raw.get("blob_bg_mode"),
            blob_open_kernel_size=raw.get("blob_open_kernel_size"),
            blob_close_kernel_size=raw.get("blob_close_kernel_size"),
            blob_use_clahe=bool(raw.get("blob_use_clahe", False)),
            blob_clahe_clip_limit=float(raw.get("blob_clahe_clip_limit", 3.0)),
            blob_clahe_tile_size=int(raw.get("blob_clahe_tile_size", 8)),
            blob_use_adaptive_threshold=bool(
                raw.get("blob_use_adaptive_threshold", False)),
            # Not `or`: a deliberately-zeroed knob must not snap back to its
            # non-zero default on the next round-trip.
            blob_self_norm_ratio=_num(raw.get("blob_self_norm_ratio"), 0.0),
            blob_self_norm_sigma=_num(raw.get("blob_self_norm_sigma"), 0.0),
            blob_self_norm_smooth_sigma=_num(
                raw.get("blob_self_norm_smooth_sigma"), -1.0),
            blob_self_norm_minsize=int(
                _num(raw.get("blob_self_norm_minsize"), -1)),
            blob_smooth_tracking=bool(raw.get("blob_smooth_tracking", True)),
            blob_background_path=raw.get("blob_background_path"),
            user_applied=bool(raw.get("user_applied", False)),
        )


def _zone_to_json(z: Zone) -> Dict[str, Any]:
    """Serialise a Zone. Delegates to the one serialiser on the class.

    The on-disk key is ``"type"``; that is what project configs, zone files
    and tracking configs all carry. A serialiser here that invented its own
    (``"zone_type"``) would round-trip with itself and give every real file's
    zones the default shape, dropping the fields it never knew about.

    A plain dict is passed through. ``zones`` is meant to hold ``Zone``
    objects, but a caller that set it straight from parsed JSON holds dicts,
    and the cost of being strict here is paid at SAVE time, at the end of a
    session, which is the worst moment to lose a configuration over a type.
    """
    if isinstance(z, dict):
        return dict(z)
    return z.to_dict()


def _zone_from_json(raw: Dict[str, Any]) -> Zone:
    return Zone.from_dict(raw)


# ============================================================================
# OverlayState
# ============================================================================
#
# Per-box live-preview overlay. One OverlayState per active box, held in
# MainWindowBase._overlay[bid]. Pose sink and blob tracker sink both update
# fields on the same object via _on_overlay_update; the renderer reads one
# object per box, checks last_seen_ns for freshness, and draws whatever
# fields are populated.


@dataclass
class OverlayState:
    """Latest pose / blob result for one box, plus monotonic freshness."""

    pose: List[List[float]] = field(default_factory=list)
    body_parts: List[str] = field(default_factory=list)
    skeleton: List[tuple] = field(default_factory=list)
    #: Parts whose NAME is drawn beside the marker. Empty = label none.
    annotate_parts: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.5

    # Blob, bounding box ``(x, y, w, h)`` and centroid ``(x, y)`` in
    # source-cropped coords. None when not tracking.
    bbox: Optional[tuple] = None
    centroid: Optional[tuple] = None

    # Common to pose + blob, most recent zone (innermost-wins) and
    # instantaneous speed.
    location: Optional[str] = None
    speed: float = 0.0

    # Lineage, exact camera frame id this result was computed on
    # and the host monotonic ns the GUI received it. Renderer's TTL
    # check compares ``last_seen_ns`` against ``host_clock.host_ns()``.
    cam_frame_id: int = 0
    last_seen_ns: int = 0

    # Composable-trigger annotation (T2): the latest frame's RuleStates + the
    # box's chip options. Drawn by the overlay render path when present.
    triggers: list = field(default_factory=list)
    trigger_chip_corner: str = "top_right"
    trigger_flash_on_fire: bool = True

    def has_pose(self) -> bool:
        return bool(self.pose)

    def has_blob(self) -> bool:
        return self.bbox is not None or self.centroid is not None


# ============================================================================
# Head-direction
# ============================================================================
#
# Robust head/body angle from DLC keypoints. Confidence-gated; smooths
# on the unit circle (cos/sin EMA) to avoid 2π wrap; returns NaN when
# no valid measurement is available. State is per-box; caller owns
# the dict and passes it back each frame.

# Defaults already declared above in the TrackingConfig section
# (DEFAULT_KEYPOINTS, DEFAULT_ROTATION_PAIR). Rotation-specific knobs:
DEFAULT_CONF_CUTOFF = 0.6
DEFAULT_EMA_ALPHA   = 0.2

NAN = float("nan")


def head_direction(pose: dict,
                   kp_pair: Tuple[str, str] = DEFAULT_ROTATION_PAIR,
                   state: Optional[dict] = None,
                   conf_cutoff: float = DEFAULT_CONF_CUTOFF,
                   ema_alpha: float = DEFAULT_EMA_ALPHA) -> Tuple[float, bool]:
    """Compute smoothed head/body direction in radians (image plane).

    Vector points tail→head: ``θ = atan2(head.y-tail.y, head.x-tail.x)``.
    Returns ``(theta_rad, valid)``. ``valid=True`` when this frame
    contributed a new measurement; ``valid=False`` when both keypoints
    were below cutoff (caller gets the previous θ, or NaN if never seen).

    Polarity is unambiguous (defined by anatomy). Wrap-safe: smoothing
    on (cos, sin) independently, so −π → +π doesn't trigger a 2π jump.
    """
    if state is None:
        state = {}

    tail_name, head_name = kp_pair
    tail = pose.get(tail_name)
    head = pose.get(head_name)

    def _conf(pt) -> float:
        if pt is None:
            return 0.0
        try:
            return float(pt[2]) if len(pt) > 2 else 1.0
        except Exception:
            return 0.0

    if _conf(tail) < conf_cutoff or _conf(head) < conf_cutoff:
        return state.get("last_theta", NAN), False

    try:
        tx, ty = float(tail[0]), float(tail[1])
        hx, hy = float(head[0]), float(head[1])
    except Exception:
        return state.get("last_theta", NAN), False

    theta = math.atan2(hy - ty, hx - tx)

    c, s = math.cos(theta), math.sin(theta)
    state["cos_s"] = ema_alpha * c + (1.0 - ema_alpha) * state.get("cos_s", c)
    state["sin_s"] = ema_alpha * s + (1.0 - ema_alpha) * state.get("sin_s", s)
    theta_s = math.atan2(state["sin_s"], state["cos_s"])
    state["last_theta"] = theta_s
    return theta_s, True






__all__ = [
    # frames
    "CameraFrame", "BoxFrame", "convert_color",
    # camera config
    "CameraConfig", "TriggerConfig", "LineOutputConfig", "FPS_MIN", "FPS_STEP",
    # tracking config
    "TrackingConfig",
    "DEFAULT_KEYPOINTS", "DEFAULT_ROTATION_PAIR",
    "DEFAULT_CONFIDENCE", "DEFAULT_BODY_PART",
    # overlay
    "OverlayState",
    # rotation
    "head_direction",
    "DEFAULT_CONF_CUTOFF", "DEFAULT_EMA_ALPHA",
]
