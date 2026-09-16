"""Experiment config schema + canonical hashing.

The "experiment config" is the JSON file the user loads in the main tab.
It is the single source of truth for an experiment definition: which
boxes are wired to which cameras, what HD is the rig default, what
tracking subsystem is enabled, what stats template applies. Created
once per project, reused across many days.

Schema 3.0 layout
=================

    schema_version    = "3.0"
    mode              = "operant" | "maze"  (locked at creation)
    experiment_name   = str
    created_at        = "YYYY-MM-DD-HHMMSS"
    last_modified     = "YYYY-MM-DD-HHMMSS"
    config_djb2       = 8-hex string (self-hash; this field is excluded
                                       from its own input)
    meta              = Meta dataclass
    setup_config      = SetupConfig {boxes: List[BoxConfig]}
                        the per-box hub (com, init_hw_def, camera_id,
                          geometry, roi, save_video, bg_captured_at,
                          tracking_enabled, save_tracking, zones,
                          action_config, api_class)
    cameras           = CamerasConfig {video_defaults, registry}
                        rig-level only
    tracking          = TrackingConfig {enabled, mode, blob, dlc, sleap}
                        rig-level only
    stats_config      = StatsConfig {template_name, template_file}
    ui                = UIState (NOT included in config_djb2)

Identity / hashes
=================

DJB2 is the sole hash function across the project. For file content
(.py task / hardware_definition source) we use the SAME 4-byte
little-endian djb2 pyControl uses on the MCU side, so the value in
``setup_config.boxes[i].init_hw_def.djb2`` matches the
``hardware_def_hash`` line written into the MCU TSV header. For
config-self-hashing and text-blob hashing (DLC manifests etc.) we use
the byte-wise djb2 of the UTF-8 bytes.

Public surface
==============

  load(path)                        - read JSON, return Config
  save(cfg, path)                   - canonicalise + self-hash + write
  new_experiment(mode, project,
                 experimenter)      - fresh schema-3.0 Config
  new_project(name, mode,
              experimenter,
              experiments_dir)      - mkdir + save fresh project
  save_experiment(cfg, project_dir,
                  guard=, on_conflict=)
                                    - atomic save with OCC + auto-merge
  load_experiment(project_dir)      - read experiment_config.json
  load_experiment_with_guard(...)   - read + stamp guard
  save_template(cfg, project_dir)   - save template.json (meta stripped)
  snapshot_config_to_source(cfg,
                            project_dir)
                                    - copy current config into
                                      <project>/source/configs/<djb2>.json
                                      (called at record-start)

No migration: old v2.0 files do not load.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from source.config.multi_instance import ProjectFileGuard

SCHEMA_VERSION = "3.0"


def schema_major_compatible(version) -> bool:
    """True when ``version`` shares this build's MAJOR schema version.

    A mismatch means ``Config.from_dict`` would drop the file's boxes /
    cameras / tracking (load as a blank rig), so the loader must REFUSE such a
    file rather than load-and-autosave-over it. An absent version is treated as
    compatible (legacy same-major files that predate the field)."""
    if version is None:
        return True
    return str(version).split(".")[0] == SCHEMA_VERSION.split(".")[0]


# ============================================================================
# Date format and DJB2 helpers
# ============================================================================


def now_ts() -> str:
    """Project-wide timestamp format: YYYY-MM-DD-HHMMSS."""
    return datetime.now().strftime("%Y-%m-%d-%H%M%S")


from source.config.hashing import (  # noqa: F401  (re-exported for callers)
    djb2_int_from_file as _djb2_int_from_file,
    djb2_hex_from_file,
    djb2_hex_from_bytes,
    djb2_hex_from_text,
)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization used as input to the self-hash.

    Sorted keys, no whitespace, identical bytes for identical content.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


# ============================================================================
# FileRef (empty = {})
# ============================================================================


@dataclass
class FileRef:
    """Reference to a file on disk + identity hash + size.

    The single source-file identity is ``djb2``, the 8-char hex form of
    the pyControl-style 4-byte-LE djb2. ``SnapshotStore`` uses it as the
    on-disk filename inside ``<project>/source/<djb2>.<ext>``.

    An unset FileRef serializes as ``{}``.
    """

    name: str = ""
    path: str = ""
    djb2: str = ""        # 8-hex string; "" means unset (placeholder)
    size_bytes: int = 0

    @classmethod
    def from_dict(cls, d) -> "FileRef":
        if not isinstance(d, dict) or not d:
            return cls()
        return cls(
            name=str(d.get("name") or ""),
            path=str(d.get("path") or ""),
            djb2=str(d.get("djb2") or ""),
            size_bytes=int(d.get("size_bytes", 0)),
        )

    @classmethod
    def from_path(cls, path) -> "FileRef":
        """Build a FileRef from a filesystem path. Empty path → empty ref."""
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls(name=p.name, path=str(p), djb2="", size_bytes=0)
        st = p.stat()
        return cls(
            name=p.name,
            path=str(p),
            djb2=djb2_hex_from_file(p),
            size_bytes=st.st_size,
        )

    def is_set(self) -> bool:
        """A FileRef is "set" iff it has a non-empty djb2 (i.e. content
        was actually captured at least once)."""
        return bool(self.djb2)

    def to_compact(self) -> Dict[str, Any]:
        """Serialize as the full FileRef when set, ``{}`` when unset.

        ``path`` is stored RELATIVE (to top_dir, basename fallback) so the
        saved config never leaks an absolute ``D:/...`` / ``C:/Users/<name>/``
        path. It records which file was used; the snapshot bytes live in source/."""
        if not self.is_set():
            return {}
        return {
            "name": self.name,
            "path": relpath_for_storage(self.path),
            "djb2": self.djb2,
            "size_bytes": self.size_bytes,
        }


# ============================================================================
# Meta
# ============================================================================


@dataclass
class Meta:
    experimenter: str = ""
    project: str = ""
    session_label: str = ""
    data_dir: str = ""
    metadata_file: str = ""              # path under <project>/metadata/
    created_at: str = ""
    tracking_enabled: bool = True        # gates source uploads
    # Loading a project auto-connects every configured camera. On by default:
    # a rig that has cameras configured almost always wants them live, and
    # having to reconnect by hand after every load was pure friction.
    auto_connect_cameras_on_load: bool = True
    # How MCUs are LABELLED for the operator: "hashed" (djb2 of USB serial,
    # OS-independent), "native" (COMn / ttyACMn), "serial" (raw). Empty =
    # inherit the per-machine default (settings.json mcu.display_mode).
    # Binding is always by USB serial; this is display-only.
    mcu_display_mode: str = ""

    @classmethod
    def from_dict(cls, d) -> "Meta":
        if not isinstance(d, dict):
            return cls()
        return cls(
            experimenter=str(d.get("experimenter") or ""),
            project=str(d.get("project") or ""),
            session_label=str(d.get("session_label") or ""),
            data_dir=_resolve_data_dir(d.get("data_dir", "")),
            metadata_file=str(d.get("metadata_file") or ""),
            created_at=str(d.get("created_at") or ""),
            tracking_enabled=bool(d.get("tracking_enabled", True)),
            # A project saved without this field should still auto-connect,
            # so the fallback matches the field default, not False.
            auto_connect_cameras_on_load=bool(
                d.get("auto_connect_cameras_on_load", True)),
            mcu_display_mode=str(d.get("mcu_display_mode") or ""),
        )


# ============================================================================
# BoxVariableSpec, per-task variable behaviour (sidecar, NOT in schema)
# ============================================================================


@dataclass
class BoxVariableSpec:
    """One row of per-task variable behaviour. The persistent flag lives in
    the per-task template ``<task>.variables.json`` and, when a project is
    loaded, the project's own ``persistent_variables.json`` flags (which win);
    edited via the Controls dialog's Standard tab.

    ``persistent``: when True, the final MCU value is captured at Stop and
    restored (by name) at the next Upload. When False (RESET, the default) the
    variable always snaps back to the task-file default on the next Upload
    (the MCU re-imports ``task_file``). The Controls dialog shows a single
    Persist checkbox.
    """

    name: str = ""
    persistent: bool = False      # capture at Stop, restore at Upload (by name)

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict):
            return cls()
        return cls(
            name=str(d.get("name") or ""),
            persistent=bool(d.get("persistent", False)),
        )


# ============================================================================
# Zone (per-box tracking zone, rectangle / polygon / scale / ellipse)
# ============================================================================


# The one Zone class lives in source/video/zones/schema.py and is re-exported
# here because this module is where the config layer's dataclasses are looked
# up. Do not define a second, parallel dataclass here: a config-layer copy
# cannot represent eight of the persisted fields, and a project reload then
# silently loses scale calibration, ellipse geometry and each zone's MCU
# transmit policy.
from source.video.zones.schema import Zone  # noqa: E402,F401


# ============================================================================
# BoxConfig, the per-box hub
# ============================================================================


@dataclass
class BoxGeometry:
    """Crop rectangle of this box within its camera's image (pixels).

    Empty geometry ``{}`` when no camera is assigned. When the box
    has a camera, ``x/y/w/h`` are the pixel coords of its segment.
    """

    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    @classmethod
    def from_dict(cls, d) -> "BoxGeometry":
        if not isinstance(d, dict) or not d:
            return cls()
        return cls(
            x=int(d.get("x", 0)),
            y=int(d.get("y", 0)),
            w=int(d.get("w", 0)),
            h=int(d.get("h", 0)),
        )

    def is_set(self) -> bool:
        return self.w > 0 and self.h > 0

    def to_compact(self) -> Dict[str, Any]:
        if not self.is_set():
            return {}
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class BoxConfig:
    """One box. Carries everything per-box.

    Run-only state (subject_id, task) is not here, that lives in the
    per-day runs JSON. The HD actually used per run is also captured into
    the runs JSON; ``init_hw_def`` here is the project's DEFAULT HD (the
    one the operator wired into the rig at setup).
    """

    setup_number: int = 0
    # USB serial number of the MCU in this box (e.g. "315535563234"). Stable
    # across reboots/replugs/OS port shuffles, the canonical Box ↔ MCU
    # binding. Mapped to a live device path (``/dev/ttyACMn`` / ``COMn``) by
    # ``mcu_ports.device_for_serial`` right before opening the connection.
    mcu_serial: str = ""
    # Fallback device path, used only when ``mcu_serial`` is empty. First
    # successful connect captures the serial and the next autosave fills
    # ``mcu_serial``.
    com_port: str = ""
    init_hw_def: FileRef = field(default_factory=FileRef)

    # Camera assignment
    camera_id: str = ""                                       # "" = no camera
    geometry: BoxGeometry = field(default_factory=BoxGeometry)
    roi_normalized: Optional[List[float]] = None              # [x,y,w,h] in 0..1
    save_video: bool = True
    bg_captured_at: str = ""

    # Tracking
    tracking_enabled: bool = False
    save_tracking: bool = False
    zones: List[Zone] = field(default_factory=list)

    # Behavior code
    action_config: FileRef = field(default_factory=FileRef)
    api_class: FileRef = field(default_factory=FileRef)       # optional

    @classmethod
    def from_dict(cls, d) -> "BoxConfig":
        if not isinstance(d, dict):
            return cls()
        roi = d.get("roi_normalized")
        return cls(
            setup_number=int(d.get("box_number", 0)),
            mcu_serial=str(d.get("mcu_serial", "") or ""),
            com_port=str(d.get("com_port") or ""),
            init_hw_def=FileRef.from_dict(d.get("init_hw_def")),
            camera_id=str(d.get("camera_id") or ""),
            geometry=BoxGeometry.from_dict(d.get("geometry")),
            roi_normalized=[float(x) for x in roi] if roi else None,
            save_video=bool(d.get("save_video", True)),
            bg_captured_at=str(d.get("bg_captured_at") or ""),
            tracking_enabled=bool(d.get("tracking_enabled", False)),
            save_tracking=bool(d.get("save_tracking", False)),
            zones=[Zone.from_dict(z) for z in (d.get("zones") or [])],
            action_config=FileRef.from_dict(d.get("action_config")),
            api_class=FileRef.from_dict(d.get("api_class")),
        )

    def to_compact(self) -> Dict[str, Any]:
        """Per-box dict for the on-disk file.

        Non-FileRef slots are always present (so a reader sees every
        per-box concern); FileRef and BoxGeometry slots collapse to
        ``{}`` when unset.
        """
        return {
            "box_number":       self.setup_number,
            "mcu_serial":       self.mcu_serial,
            "com_port":         self.com_port,
            "init_hw_def":      self.init_hw_def.to_compact(),
            "camera_id":        self.camera_id,
            "geometry":         self.geometry.to_compact(),
            "roi_normalized":   list(self.roi_normalized) if self.roi_normalized else None,
            "save_video":       bool(self.save_video),
            "bg_captured_at":   self.bg_captured_at,
            "tracking_enabled": bool(self.tracking_enabled),
            "save_tracking":    bool(self.save_tracking),
            "zones":            [zone_to_dict(z) for z in self.zones],
            "action_config":    self.action_config.to_compact(),
            "api_class":        self.api_class.to_compact(),
        }


@dataclass
class SetupConfig:
    """Top-level per-box hub."""

    boxes: List[BoxConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d) -> "SetupConfig":
        if not isinstance(d, dict):
            return cls()
        return cls(
            boxes=[BoxConfig.from_dict(b) for b in (d.get("boxes") or [])],
        )

    def to_compact(self) -> Dict[str, Any]:
        return {"boxes": [b.to_compact() for b in self.boxes]}


# ============================================================================
# Cameras
# ============================================================================


@dataclass
class CameraDefaults:
    """Rig-level video defaults. ``target_fps`` is the ONLY fps that
    drives capture. Recorder reads it verbatim."""

    target_fps: int = 30
    frame_strategy: str = "accept"       # "accept" | "remux"
    grayscale: bool = False
    camera_backend: str = "opencv"
    camera_settle_ms: int = 5000

    @classmethod
    def from_dict(cls, d) -> "CameraDefaults":
        if not isinstance(d, dict):
            return cls()
        return cls(
            target_fps=int(d.get("target_fps", 30)),
            frame_strategy=str(d.get("frame_strategy", "accept")),
            grayscale=bool(d.get("grayscale", False)),
            camera_backend=str(d.get("camera_backend", "opencv")),
            camera_settle_ms=int(d.get("camera_settle_ms", 5000)),
        )


@dataclass
class CameraPreference:
    """User's selected camera settings, what the recorder uses.

    Distinct from ``CameraCapabilities``: capabilities lists ALL options
    the camera advertised; preference holds the operator's choice.

    All ``null`` (None in Python) means "not yet picked", recorder must
    refuse to start.
    """

    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None                # the picked rate; None = not set
    codec: Optional[str] = None
    exposure_us: Optional[int] = None        # None = auto
    gain_db: Optional[float] = None
    external_trigger: bool = False
    # OS capture backend by NAME ("dshow"/"msmf"/"v4l2"); "" = let the
    # pipeline pick the fastest measured one for the selected mode.
    #
    # The CHOICE travels with the project; the measurements behind it do not.
    # Achievable rate is a property of the host's USB controller and port, so
    # it lives in the machine-level calibration store, carrying it in the
    # project would mean believing one rig's numbers on another.
    capture_backend: str = ""

    # How the operator asked the image to be presented. These were absent,
    # so ``CameraConfig.to_json()`` carried them and the save dropped them:
    # set a flip, save the project, reopen it, and the recording came back
    # the other way round with nothing reporting a change. They are the
    # operator's choice about this camera, so they travel with the project.
    flip_horizontal: bool = False
    flip_vertical: bool = False
    grayscale: bool = False
    #: What the bus does when a consumer is behind ("latest"/"queue"...).
    frame_strategy: str = ""

    # Whatever the camera reported and the operator changed on the Camera
    # options tab, keyed by the canonical feature name (exposure_auto,
    # pixel_format, trigger_mode...). Free-form because the key set is the
    # camera's, not ours.
    features: Dict[str, Any] = field(default_factory=dict)
    # Structured scientific-camera I/O (FLIR/Ximea Advanced tab) so the portable
    # project config is self-contained. Stored as the pipeline
    # TriggerConfig/LineOutputConfig JSON dicts; None = driver default/free-run.
    trigger: Optional[Dict[str, Any]] = None
    line_output: Optional[Dict[str, Any]] = None
    sync_role: str = "none"                   # none | primary | secondary

    @classmethod
    def from_dict(cls, d) -> "CameraPreference":
        if not isinstance(d, dict):
            return cls()
        return cls(
            width=None if d.get("width") is None else int(d["width"]),
            height=None if d.get("height") is None else int(d["height"]),
            fps=None if d.get("fps") is None else int(d["fps"]),
            codec=None if d.get("codec") is None else str(d["codec"]),
            exposure_us=None if d.get("exposure_us") is None else int(d["exposure_us"]),
            gain_db=None if d.get("gain_db") is None else float(d["gain_db"]),
            external_trigger=bool(d.get("external_trigger", False)),
            flip_horizontal=bool(d.get("flip_horizontal", False)),
            flip_vertical=bool(d.get("flip_vertical", False)),
            grayscale=bool(d.get("grayscale", False)),
            frame_strategy=str(d.get("frame_strategy") or ""),
            trigger=(dict(d["trigger"]) if isinstance(d.get("trigger"), dict) else None),
            line_output=(dict(d["line_output"])
                         if isinstance(d.get("line_output"), dict) else None),
            sync_role=str(d.get("sync_role", "none") or "none"),
            features=(dict(d["features"])
                      if isinstance(d.get("features"), dict) else {}),
        )


@dataclass
class CameraResolutionEntry:
    width: int = 0
    height: int = 0
    fps_options: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d) -> "CameraResolutionEntry":
        if not isinstance(d, dict):
            return cls()
        return cls(
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
            fps_options=[int(x) for x in (d.get("fps_options") or [])],
        )


@dataclass
class CameraCapabilities:
    """Full Detect-Max result.

    Empty lists / null ranges are valid, they mean "not scanned
    yet" OR "scanner doesn't probe this dimension". The GUI populates
    dropdowns from whatever is here; missing fields just yield empty
    pickers.
    """

    scanned_at: str = ""                  # "" = never scanned
    resolutions: List[CameraResolutionEntry] = field(default_factory=list)
    # ``{backend_name: [(w, h, measured_fps), ...]}``, what each OS capture
    # backend actually delivered. ``resolutions`` above is the flat view of
    # the chosen one; this is the evidence the choice was made from, kept so
    # a saved project can explain itself on another day.
    probed_variants: Dict[str, List[Tuple[int, int, float]]] = field(
        default_factory=dict)

    @classmethod
    def from_dict(cls, d) -> "CameraCapabilities":
        if not isinstance(d, dict):
            return cls()
        variants: Dict[str, List[Tuple[int, int, float]]] = {}
        for name, modes in (d.get("probed_variants") or {}).items():
            clean = []
            for m in (modes or []):
                try:
                    clean.append((int(m[0]), int(m[1]), float(m[2])))
                except (TypeError, ValueError, IndexError):
                    continue
            if clean:
                variants[str(name)] = clean
        return cls(
            scanned_at=str(d.get("scanned_at") or ""),
            resolutions=[CameraResolutionEntry.from_dict(r)
                         for r in (d.get("resolutions") or [])],
            probed_variants=variants,
        )


@dataclass
class CameraEntry:
    """One physical camera in the rig registry."""

    camera_id: str = ""                  # opaque stable string
    label: str = ""                      # user-editable display name
    backend: str = "opencv"
    unique_id: str = ""                  # OS device id (cv2 index, vendor guid)
    preference: CameraPreference = field(default_factory=CameraPreference)
    capabilities: CameraCapabilities = field(default_factory=CameraCapabilities)

    @classmethod
    def from_dict(cls, d) -> "CameraEntry":
        if not isinstance(d, dict):
            return cls()
        return cls(
            camera_id=str(d.get("camera_id") or ""),
            label=str(d.get("label") or ""),
            backend=str(d.get("backend", "opencv")),
            unique_id=str(d.get("unique_id") or ""),
            preference=CameraPreference.from_dict(d.get("preference")),
            capabilities=CameraCapabilities.from_dict(d.get("capabilities")),
        )


@dataclass
class CamerasConfig:
    """Top-level rig camera config. Per-box camera assignment +
    geometry live on ``BoxConfig``.
    """

    video_defaults: CameraDefaults = field(default_factory=CameraDefaults)
    registry: List[CameraEntry] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d) -> "CamerasConfig":
        if not isinstance(d, dict):
            return cls()
        return cls(
            video_defaults=CameraDefaults.from_dict(d.get("video_defaults")),
            registry=[CameraEntry.from_dict(c) for c in (d.get("registry") or [])],
        )


# ============================================================================
# Tracking (placeholder-when-disabled)
# ============================================================================


def _num(value, default: float) -> float:
    """Read a saved number, falling back only when it is genuinely absent.

    ``float(d.get(k, default) or default)`` reads fine but turns a stored 0
    into the default, so a setting the operator deliberately zeroed comes back
    switched on. Only ``None`` and unparseable text fall back here.
    """
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@dataclass
class BlobConfig:
    threshold: int = 25
    min_area: int = 100
    max_area: int = 50000
    detect_dark: bool = True
    blur_mode: str = "gaussian"
    blur_kernel_size: int = 5
    bg_mode: str = "static"
    open_kernel_size: int = 3
    close_kernel_size: int = 7
    use_clahe: bool = False
    clahe_clip_limit: float = 3.0
    clahe_tile_size: int = 8
    use_adaptive_threshold: bool = False
    # self_norm ("Simple") calibration. ratio 0.0 = not yet calibrated, so the
    # tracker auto-estimates it from the first frame; a saved non-zero value is
    # what makes "calibrate once, reload forever" work. sigma 0 = auto
    # (crop_height/20). smooth_sigma/minsize: -1 = auto (scaled to frame
    # height), 0 = off, blob.py owns those numbers, see SELF_NORM_* there.
    self_norm_ratio: float = 0.0
    self_norm_sigma: float = 0.0
    self_norm_smooth_sigma: float = -1.0
    self_norm_minsize: int = -1
    # Per-box smoothing for blob tracker output (Kalman/EMA on the centroid
    # trajectory). Round-tripped via blob_smooth_tracking in the framebus TC.
    smooth_tracking: bool = True

    @classmethod
    def from_dict(cls, d) -> "BlobConfig":
        if not isinstance(d, dict):
            return cls()
        return cls(
            threshold=int(d.get("threshold", 25)),
            min_area=int(d.get("min_area", 100)),
            max_area=int(d.get("max_area", 50000)),
            detect_dark=bool(d.get("detect_dark", True)),
            blur_mode=str(d.get("blur_mode", "gaussian")),
            blur_kernel_size=int(d.get("blur_kernel_size", 5)),
            bg_mode=str(d.get("bg_mode", "static")),
            open_kernel_size=int(d.get("open_kernel_size", 3)),
            close_kernel_size=int(d.get("close_kernel_size", 7)),
            use_clahe=bool(d.get("use_clahe", False)),
            clahe_clip_limit=float(d.get("clahe_clip_limit", 3.0)),
            clahe_tile_size=int(d.get("clahe_tile_size", 8)),
            use_adaptive_threshold=bool(d.get("use_adaptive_threshold", False)),
            # `or` would be wrong here: an operator who deliberately turned
            # the texture blur off saves 0.0, and `0.0 or -1.0` would quietly
            # hand it back as auto.
            self_norm_ratio=_num(d.get("self_norm_ratio"), 0.0),
            self_norm_sigma=_num(d.get("self_norm_sigma"), 0.0),
            self_norm_smooth_sigma=_num(d.get("self_norm_smooth_sigma"), -1.0),
            self_norm_minsize=int(_num(d.get("self_norm_minsize"), -1)),
            smooth_tracking=bool(d.get("smooth_tracking", True)),
        )


@dataclass
class DLCConfig:
    """DLC subtree. Empty ``{}`` placeholder when ``tracking.mode != "dlc"``.

    ``model_path`` is the model's filesystem path (stored relative to
    top_dir on disk, resolved back on load); model lineage (djb2 of the
    captured files) is SnapshotStore's job, not this field's.
    """

    model_path: str = ""
    body_parts: List[str] = field(default_factory=list)
    confidence: float = 0.5
    marker_size: int = 4
    zone_body_part: str = ""
    # Frame resize factor fed into the pose backend (0.5 = half-res
    # inference). Round-tripped via pose_resize_factor in the framebus TC.
    resize: float = 1.0
    # Number of parallel inference workers. Round-tripped via
    # pose_n_instances in the framebus TC.
    instances: int = 1

    @classmethod
    def from_dict(cls, d) -> "DLCConfig":
        if not isinstance(d, dict):
            return cls()
        return cls(
            model_path=_pose_model_from_dict(d),
            body_parts=[str(p) for p in (d.get("body_parts") or [])],
            confidence=float(d.get("confidence", 0.5)),
            marker_size=int(d.get("marker_size", 4)),
            zone_body_part=str(d.get("zone_body_part") or ""),
            resize=float(d.get("resize", 1.0)),
            instances=int(d.get("instances", 1)),
        )


@dataclass
class SleapConfig:
    model_path: str = ""
    body_parts: List[str] = field(default_factory=list)
    confidence: float = 0.75
    resize: float = 1.0
    instances: int = 1

    @classmethod
    def from_dict(cls, d) -> "SleapConfig":
        if not isinstance(d, dict):
            return cls()
        return cls(
            model_path=_pose_model_from_dict(d),
            body_parts=[str(p) for p in (d.get("body_parts") or [])],
            confidence=float(d.get("confidence", 0.75)),
            resize=float(d.get("resize", 1.0)),
            instances=int(d.get("instances", 1)),
        )


@dataclass
class TrackingConfig:
    """Rig-level tracking config. Per-box ``tracking_enabled`` /
    ``zones`` / ``save_tracking`` live on ``BoxConfig``.

    Placeholder rule: the ``blob`` / ``dlc`` / ``sleap`` subtree
    that matches ``mode`` is populated; the other two are emitted as
    ``{}``.
    """

    enabled: bool = False
    mode: str = "none"                   # "none" | "blob" | "dlc" | "sleap"
    # Pixel↔physical scale is NOT stored here. It lives in the per-box scale
    # zone (type "scale") and is derived on demand by
    # ``zones/coords.py::scale_zone_to_px_per_mm``, the single source of
    # truth. A duplicate scalar here only ever went stale.
    blob: BlobConfig = field(default_factory=BlobConfig)
    dlc: DLCConfig = field(default_factory=DLCConfig)
    sleap: SleapConfig = field(default_factory=SleapConfig)
    # MCU push gates + zone-change body-part picker. Rig-level, dialog
    # applies the same value to every selected box. Persisted under
    # ``cfg.tracking`` so the project YAML round-trips them.
    push_zones_to_mcu: bool = True
    push_coords_to_mcu: bool = True
    # Per-frame pose event (silent ``frame_event`` each pose result). OFF by
    # default, opt-in toggle to A/B against pure zone-change driving.
    push_frame_event: bool = False
    zone_change_body_part: str = "centroid"

    @classmethod
    def from_dict(cls, d) -> "TrackingConfig":
        if not isinstance(d, dict):
            return cls()
        return cls(
            enabled=bool(d.get("enabled", False)),
            mode=str(d.get("mode", "none")),
            blob=BlobConfig.from_dict(d.get("blob")) if isinstance(d.get("blob"), dict) else BlobConfig(),
            dlc=DLCConfig.from_dict(d.get("dlc")) if isinstance(d.get("dlc"), dict) else DLCConfig(),
            sleap=SleapConfig.from_dict(d.get("sleap")) if isinstance(d.get("sleap"), dict) else SleapConfig(),
            push_zones_to_mcu=bool(d.get("push_zones_to_mcu", True)),
            push_coords_to_mcu=bool(d.get("push_coords_to_mcu", True)),
            push_frame_event=bool(d.get("push_frame_event", False)),
            zone_change_body_part=str(
                d.get("zone_change_body_part", "centroid") or "centroid"),
        )

    def to_compact(self) -> Dict[str, Any]:
        """Only the active mode's subtree is populated;
        the other two are emitted as ``{}``.
        """
        out: Dict[str, Any] = {
            "enabled":               bool(self.enabled),
            "mode":                  self.mode,
            "blob":                  {},
            "dlc":                   {},
            "sleap":                 {},
            "push_zones_to_mcu":     bool(self.push_zones_to_mcu),
            "push_coords_to_mcu":    bool(self.push_coords_to_mcu),
            "push_frame_event":      bool(self.push_frame_event),
            "zone_change_body_part": str(self.zone_change_body_part or "centroid"),
        }
        # Persist the active mode's subtree whenever it's actually configured
        #, gated on real content (a model path for pose), NOT on ``enabled``.
        # This lets the operator DISABLE tracking without losing the model /
        # blob setup, so re-enabling later restores it. A pose mode with no
        # model still collapses to ``{}`` (no empty shells on disk).
        if self.mode == "blob":
            out["blob"] = asdict(self.blob)
        elif self.mode == "dlc" and self.dlc.model_path:
            dlc_d = asdict(self.dlc)
            # Stored functionally relative to top_dir (like data_dir) so a
            # shared project/template never leaks ``D:/...`` /
            # ``C:/Users/<name>/...``. Resolved back on load in from_dict.
            dlc_d["model_path"] = _relpath_data_dir(self.dlc.model_path)
            out["dlc"] = dlc_d
        elif self.mode == "sleap" and self.sleap.model_path:
            sleap_d = asdict(self.sleap)
            sleap_d["model_path"] = _relpath_data_dir(self.sleap.model_path)
            out["sleap"] = sleap_d
        return out


# ============================================================================
# Stats + UI
# ============================================================================


@dataclass
class StatsConfig:
    template_name: str = ""               # "" = none selected
    template_file: FileRef = field(default_factory=FileRef)

    @classmethod
    def from_dict(cls, d) -> "StatsConfig":
        if not isinstance(d, dict):
            return cls()
        return cls(
            template_name=str(d.get("template_name") or ""),
            template_file=FileRef.from_dict(d.get("template_file")),
        )

    def to_compact(self) -> Dict[str, Any]:
        return {
            "template_name": self.template_name,
            "template_file": self.template_file.to_compact(),
        }


@dataclass
class UIState:
    dialog_overrides: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d) -> "UIState":
        if not isinstance(d, dict):
            return cls()
        return cls(
            dialog_overrides=dict(d.get("dialog_overrides", {})),
        )


# ============================================================================
# Change-log row schema (IO in source/config/history.py)
# ============================================================================


# ============================================================================
# Config, top-level schema 3.0 dataclass
# ============================================================================


@dataclass
class Config:
    """In-memory mirror of an experiment-config JSON file (schema 3.0)."""

    schema_version: str = SCHEMA_VERSION
    mode: str = "operant"                # locked at creation
    experiment_name: str = ""
    created_at: str = ""
    last_modified: str = ""
    config_djb2: str = ""                # 8-hex self-hash; excluded from its own input

    meta: Meta = field(default_factory=Meta)
    setup_config: SetupConfig = field(default_factory=SetupConfig)
    cameras: CamerasConfig = field(default_factory=CamerasConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    stats_config: StatsConfig = field(default_factory=StatsConfig)
    ui: UIState = field(default_factory=UIState)

    # ---- serialization ----------------------------------------------------

    def to_dict_for_hash(self) -> Dict[str, Any]:
        """Hash input. Excludes self-hash, volatile timestamp, and ui."""
        d = self.to_dict()
        d.pop("config_djb2", None)
        d.pop("last_modified", None)
        d.pop("ui", None)
        return d

    def to_dict(self) -> Dict[str, Any]:
        """Full dict for the on-disk file. All top-level keys present;
        placeholder rules apply inside sub-blocks (FileRef → {},
        BoxGeometry → {}, tracking subtrees → {}).
        """
        meta_d = asdict(self.meta)
        # Store data_dir RELATIVE (under top_dir) so the shared config never
        # leaks an absolute D:/ path; resolved back on load in Meta.from_dict.
        meta_d["data_dir"] = _relpath_data_dir(meta_d.get("data_dir", ""))
        return {
            "schema_version":  self.schema_version,
            "mode":            self.mode,
            "experiment_name": self.experiment_name,
            "created_at":      self.created_at,
            "last_modified":   self.last_modified,
            "config_djb2":     self.config_djb2,
            "meta":            meta_d,
            "setup_config":    self.setup_config.to_compact(),
            "cameras":         asdict(self.cameras),
            "tracking":        self.tracking.to_compact(),
            "stats_config":    self.stats_config.to_compact(),
            "ui":              asdict(self.ui),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Parse a v3.0 experiment_config.json dict.

        Does NOT accept v2.0 / v1.0 files. A wrong-shape file falls back to
        defaults where keys are missing (loads as a blank rig). Warn loudly
        on a major-version mismatch so the content loss is visible.
        """
        if not isinstance(d, dict):
            d = {}
        raw_ver = d.get("schema_version")
        if (raw_ver is not None
                and str(raw_ver).split(".")[0] != SCHEMA_VERSION.split(".")[0]):
            logger.warning(
                "experiment_config schema_version=%s is incompatible with the "
                "expected %s, this file is from an older/newer version and "
                "will load as an EMPTY rig (its boxes / cameras / tracking are "
                "dropped). Re-create the config in this app to migrate.",
                raw_ver, SCHEMA_VERSION)
        return cls(
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
            mode=str(d.get("mode", "operant")),
            experiment_name=str(d.get("experiment_name") or ""),
            created_at=str(d.get("created_at") or ""),
            last_modified=str(d.get("last_modified") or ""),
            config_djb2=str(d.get("config_djb2") or ""),
            meta=Meta.from_dict(d.get("meta")),
            setup_config=SetupConfig.from_dict(d.get("setup_config")),
            cameras=CamerasConfig.from_dict(d.get("cameras")),
            tracking=TrackingConfig.from_dict(d.get("tracking")),
            stats_config=StatsConfig.from_dict(d.get("stats_config")),
            ui=UIState.from_dict(d.get("ui")),
        )


# ============================================================================
# Load / save (low-level)
# ============================================================================


def load(path: str | Path) -> Config:
    """Read a v3 experiment_config.json. Does NOT verify the self-hash."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return Config.from_dict(data)


def load_from_bytes(payload: bytes) -> Config:
    """Parse v3 experiment_config.json from in-memory bytes. Avoids a
    second file read when the caller already has the payload."""
    return Config.from_dict(json.loads(payload.decode("utf-8")))


# Keys excluded from the canonical hash input: self-hash, volatile
# timestamp, and ui state. Mirrors Config.to_dict_for_hash().
_HASH_EXCLUDE_TOP = ("config_djb2", "last_modified", "ui")


def serialize_for_save(cfg: Config) -> dict:
    """Refresh cfg.last_modified + cfg.config_djb2 and return the
    JSON-ready dict. SINGLE to_dict() walk per save, the returned
    dict is what callers should write to disk and may reuse for the
    template."""
    cfg.last_modified = now_ts()
    payload = cfg.to_dict()
    for_hash = {k: v for k, v in payload.items() if k not in _HASH_EXCLUDE_TOP}
    cfg.config_djb2 = djb2_hex_from_text(canonical_json(for_hash))
    payload["config_djb2"] = cfg.config_djb2
    payload["last_modified"] = cfg.last_modified
    return payload


def save(cfg: Config, path: str | Path) -> None:
    """Atomically write the config. Updates last_modified + config_djb2."""
    from source.config.multi_instance import write_atomic
    payload = serialize_for_save(cfg)
    write_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False))


# ============================================================================
# Project folder IO
# ============================================================================
#
# Layout:
#
#   <experiments_dir>/projects/<project_name>/
#       experiment_config.json
#       template.json
#       change_log.jsonl
#       source/
#           index.jsonl
#           <djb2>.py            # task / hw_def / api_class
#           <djb2>.json          # action_config / stats_template
#           configs/<djb2>.json  # project_config snapshots (record-start)
#           dlc/<djb2>.json      # DLC model manifests
#       runs/<task_family>/<YYYY-MM-DD>.json
#       metadata/                # cohort excel, captured bg images
#
# `experiments_dir` defaults to Path("experiments").


PROJECTS_DIR_NAME = "projects"
EXPERIMENT_CONFIG_FILENAME = "experiment_config.json"
TEMPLATE_FILENAME = "template.json"
SOURCE_DIR_NAME = "source"
SOURCE_CONFIGS_DIR_NAME = "configs"
SOURCE_DLC_DIR_NAME = "dlc"
SOURCE_HD_DIR_NAME = "hd"
SOURCE_DEVICES_DIR_NAME = "devices"
RUNS_DIR_NAME = "runs"
METADATA_DIR_NAME = "metadata"
DEFAULT_EXPERIMENTS_DIR = Path("experiments")


def projects_root(experiments_dir: str | Path | None = None) -> Path:
    """Return <experiments_dir>/projects/."""
    base = Path(experiments_dir) if experiments_dir else DEFAULT_EXPERIMENTS_DIR
    return base / PROJECTS_DIR_NAME


def project_dir(name: str, experiments_dir: str | Path | None = None) -> Path:
    """Return the path of a named project folder. Does not create it."""
    if not name:
        raise ValueError("project name required")
    return projects_root(experiments_dir) / name


def experiment_config_path(project_dir_path: str | Path) -> Path:
    return Path(project_dir_path) / EXPERIMENT_CONFIG_FILENAME


def template_path(project_dir_path: str | Path) -> Path:
    return Path(project_dir_path) / TEMPLATE_FILENAME


def source_dir(project_dir_path: str | Path) -> Path:
    return Path(project_dir_path) / SOURCE_DIR_NAME


def source_configs_dir(project_dir_path: str | Path) -> Path:
    return source_dir(project_dir_path) / SOURCE_CONFIGS_DIR_NAME


def source_dlc_dir(project_dir_path: str | Path) -> Path:
    return source_dir(project_dir_path) / SOURCE_DLC_DIR_NAME


def source_hd_dir(project_dir_path: str | Path) -> Path:
    """Hardware-definition snapshots live in their own subfolder
    (``<project>/source/hd/<djb2>.py``), parallel to ``source/dlc/``,
    keeps task .py files in source/ uncluttered by HD .py files."""
    return source_dir(project_dir_path) / SOURCE_HD_DIR_NAME


def source_devices_dir(project_dir_path: str | Path) -> Path:
    """Device-driver snapshots live in their own subfolder
    (``<project>/source/devices/<djb2>.py``), parallel to ``source/hd/``,
    one HD pulls many drivers, kept out of the flat source/ task tree."""
    return source_dir(project_dir_path) / SOURCE_DEVICES_DIR_NAME


def relpath_for_storage(path, project_dir=None) -> str:
    """Convert a path to a storage-safe RELATIVE form for saved metadata
    (experiment_config.json, change_log.jsonl, snapshots) so we never persist
    absolute paths like ``D:/Ulm_Data/...`` or ``C:/Users/<name>/...`` (a
    privacy/security leak when a project or its logs are shared).

    Made relative to the project dir if the path lives under it, else the app
    ``top_dir`` (e.g. ``data/...`` / ``tasks/...`` / ``experiments/...``),
    else falls back to just the basename. Empty or already-relative paths
    pass through unchanged (normalised to forward slashes).
    """
    s = str(path or "").strip()
    if not s:
        return ""
    p = Path(s)
    if not p.is_absolute():
        return s.replace("\\", "/")
    bases = []
    if project_dir:
        bases.append(Path(project_dir))
    try:
        from source import paths as _paths
        bases.append(Path(_paths.top_dir))
    except Exception:
        pass
    for base in bases:
        try:
            return p.relative_to(base).as_posix()
        except (ValueError, TypeError):
            continue
    return p.name


def _relpath_data_dir(path) -> str:
    """Relativize a FUNCTIONAL data dir for storage: relative to ``top_dir``
    when it lives under it (``data/<project>`` form), else kept ABSOLUTE
    (an external drive / network share must still resolve, accept that those
    rare cases keep their path). Differs from ``relpath_for_storage`` which
    basenames anything outside the tree (fine for identifying a file, not for a path
    that has to resolve back)."""
    s = str(path or "").strip()
    if not s:
        return ""
    p = Path(s)
    if not p.is_absolute():
        return s.replace("\\", "/")
    try:
        from source import paths as _paths
        return p.relative_to(Path(_paths.top_dir)).as_posix()
    except (ValueError, TypeError):
        return s.replace("\\", "/")  # external, keep absolute (functional)


def _resolve_data_dir(stored) -> str:
    """Inverse of :func:`_relpath_data_dir` for load: a relative stored
    ``data_dir`` resolves to an absolute path under ``top_dir`` so every
    in-memory consumer keeps working unchanged. Empty / already-absolute
    pass through."""
    s = str(stored or "").strip()
    if not s or Path(s).is_absolute():
        return s
    try:
        from source import paths as _paths
        return str(Path(_paths.top_dir) / s)
    except Exception:
        return s


def _looks_absolute(s: str) -> bool:
    """Cross-platform absoluteness test: a leading ``/`` or ``\\``, or a
    ``X:`` drive prefix. Unlike ``Path.is_absolute()`` this recognises a
    POSIX path on Windows (and a drive path on Linux), so a model path saved
    on one OS isn't mistaken for relative and mangled on another."""
    return (s.startswith("/") or s.startswith("\\")
            or (len(s) > 1 and s[1] == ":"))


def _resolve_model_path(stored) -> str:
    """Inverse of the ``_relpath_data_dir`` applied to a pose model path.

    A pure-relative stored path (model kept under ``top_dir``) resolves back
    to absolute; any already-absolute path (external model dir, any OS) passes
    through unchanged so ``Path(model).exists()`` works and cross-platform
    projects don't get their absolute paths rewritten."""
    s = str(stored or "").strip()
    if not s or _looks_absolute(s):
        return s
    try:
        from source import paths as _paths
        return str(Path(_paths.top_dir) / s)
    except Exception:
        return s


def _pose_model_from_dict(d) -> str:
    """Read + normalize a pose model path: coerce a literal ``"None"`` to
    ``""`` and resolve a stored relative path back to absolute. Shared by
    DLCConfig/SleapConfig.from_dict. Also reads the retired ``model_djb2``
    key (older saves stored the path under that misnomer)."""
    mp = str(d.get("model_path", "") or d.get("model_djb2", "") or "")
    if mp == "None":
        mp = ""
    return _resolve_model_path(mp) if mp else mp


def list_projects(experiments_dir: str | Path | None = None) -> list[Path]:
    """Return project folders that contain an experiment_config.json."""
    root = projects_root(experiments_dir)
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and experiment_config_path(p).exists()
    )


# ----------------------------------------------------------------------------
# Constructors
# ----------------------------------------------------------------------------


def _default_data_dir_for(project: str) -> str:
    """Canonical data root for a named project, ``<top_dir>/data/<project>``.

    Falls back to ``"data/<project>"`` (relative) if ``config.paths`` isn't
    importable (e.g. unit tests running without the app environment).
    """
    if not project:
        return ""
    try:
        from source import paths as _paths
        return str(Path(_paths.top_dir) / "data" / project)
    except Exception:
        return str(Path("data") / project)


def new_experiment(mode: str = "operant",
                   project: str = "",
                   experimenter: str = "") -> Config:
    """Build a fresh schema-3.0 Config for a new experiment.

    ``mode`` is pinned at creation. ``project`` and ``experimenter``
    populate both the top-level fields and ``meta``. ``meta.data_dir``
    defaults to the canonical ``<top_dir>/data/<project>`` so the
    sidebar Data Dir field shows the right path immediately, the user
    can still override before save.
    """
    now = now_ts()
    cfg = Config(
        schema_version=SCHEMA_VERSION,
        mode=mode,
        experiment_name=project,
        created_at=now,
        last_modified=now,
        meta=Meta(
            experimenter=experimenter,
            project=project,
            created_at=now,
            data_dir=_default_data_dir_for(project),
        ),
    )
    cfg.config_djb2 = djb2_hex_from_text(canonical_json(cfg.to_dict_for_hash()))
    return cfg


def new_project(name: str,
                mode: str = "operant",
                experimenter: str = "",
                experiments_dir: str | Path | None = None
                ) -> Tuple[Path, Config]:
    """Create a new project folder and write a fresh experiment_config.json."""
    pd = project_dir(name, experiments_dir)
    cfg_path = experiment_config_path(pd)
    if cfg_path.exists():
        raise FileExistsError(f"project already exists: {cfg_path}")
    pd.mkdir(parents=True, exist_ok=True)
    (pd / METADATA_DIR_NAME).mkdir(parents=True, exist_ok=True)
    cfg = new_experiment(mode=mode, project=name, experimenter=experimenter)
    save(cfg, cfg_path)
    return pd, cfg


# ----------------------------------------------------------------------------
# Project save / load with OCC guard
# ----------------------------------------------------------------------------


def save_experiment(cfg: Config,
                    project_dir_path: str | Path | None = None,
                    experiments_dir: str | Path | None = None,
                    *,
                    guard: Optional["ProjectFileGuard"] = None,
                    on_conflict: Optional[Callable[[list], str]] = None,
                    cached_payload: Optional[dict] = None,
                    ) -> Path:
    """Save Config to ``<project_dir>/experiment_config.json`` atomically.

    Multi-instance safety: when ``guard`` is provided, the write goes
    through the guard's OCC + auto-merge path.

    ``cached_payload`` may be passed by callers that already built the
    dict via ``serialize_for_save(cfg)`` and intend to reuse it (e.g.
    for ``save_template``). Avoids a second cfg.to_dict() walk.
    """
    if project_dir_path is None:
        if not cfg.meta.project:
            raise ValueError(
                "save_experiment: cfg.meta.project is empty and no "
                "project_dir_path given"
            )
        project_dir_path = project_dir(cfg.meta.project, experiments_dir)
    pd = Path(project_dir_path)
    pd.mkdir(parents=True, exist_ok=True)
    cfg_path = experiment_config_path(pd)

    payload = cached_payload if cached_payload is not None else serialize_for_save(cfg)

    if guard is None:
        from source.config.multi_instance import write_atomic
        write_atomic(cfg_path, json.dumps(payload, indent=2, ensure_ascii=False))
        return cfg_path

    from source.config.multi_instance import SaveConflictError
    written, conflicts = guard.save_json(
        payload,
        on_conflict=on_conflict,
        volatile_paths=(
            ("last_modified",),
            ("config_djb2",),
        ),
    )
    if not written:
        raise SaveConflictError(conflicts)
    return cfg_path


def load_experiment(project_dir_path: str | Path) -> Config:
    """Load Config from ``<project_dir>/experiment_config.json``."""
    cfg_path = experiment_config_path(project_dir_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"no experiment_config.json at {cfg_path}")
    return load(cfg_path)


def load_experiment_with_guard(project_dir_path: str | Path
                               ) -> tuple[Config, "ProjectFileGuard"]:
    """Load Config + stamp a ProjectFileGuard for later OCC saves.

    Reads the file ONCE, the bytes feed both Config.from_dict and the
    guard's snapshot. Saves ~50-200 ms per load on slow storage (Jetson eMMC).
    """
    from source.config.multi_instance import ProjectFileGuard
    cfg_path = experiment_config_path(project_dir_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"no experiment_config.json at {cfg_path}")
    payload = cfg_path.read_bytes()
    cfg = load_from_bytes(payload)
    guard = ProjectFileGuard.stamp_from_bytes(cfg_path, payload)
    return cfg, guard


# ----------------------------------------------------------------------------
# Template, config minus identifying meta
# ----------------------------------------------------------------------------


# Top-level keys stripped from template.json
_TEMPLATE_STRIP_TOP = (
    "experiment_name",
    "created_at",
    "last_modified",
    "config_djb2",
)

# Meta keys stripped (everything that identifies a specific instance)
_TEMPLATE_STRIP_META = (
    "experimenter",
    "project",
    "session_label",
    "data_dir",
    "metadata_file",
    "created_at",
)


def save_template(cfg: Config,
                  project_dir_path: str | Path,
                  *,
                  cached_payload: Optional[dict] = None,
                  ) -> Path:
    """Save ``<project>/template.json``, Config minus identifying meta.

    ``cached_payload`` lets callers reuse a dict already produced by
    ``serialize_for_save(cfg)`` (e.g. from the matching
    ``save_experiment`` call). Avoids a second cfg.to_dict() walk,
    important on Jetson where each walk costs ~200-400 ms.
    """
    from source.config.multi_instance import write_atomic
    if cached_payload is not None:
        # Shallow-copy at top level and at meta, we mutate both.
        payload = dict(cached_payload)
        meta = payload.get("meta")
        if isinstance(meta, dict):
            payload["meta"] = dict(meta)
    else:
        payload = cfg.to_dict()
    for k in _TEMPLATE_STRIP_TOP:
        payload.pop(k, None)
    meta = payload.get("meta")
    if isinstance(meta, dict):
        for k in _TEMPLATE_STRIP_META:
            meta[k] = ""
    pd = Path(project_dir_path)
    pd.mkdir(parents=True, exist_ok=True)
    path = template_path(pd)
    write_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False))
    return path


# ----------------------------------------------------------------------------
# Config snapshot, pinned at record-start
# ----------------------------------------------------------------------------


def snapshot_config_to_source(cfg: Config,
                              project_dir_path: str | Path) -> str:
    """Copy the current Config into ``<project>/source/configs/<djb2>.json``.

    Idempotent. At record-start the caller runs:

        djb2 = snapshot_config_to_source(cfg, project_dir)

    The returned djb2 is the per-run ``config_djb2`` that goes into the
    per-day runs JSON. The on-disk file resolves cleanly any time the
    user wants to know which exact config drove a given run.

    Returns the 8-hex djb2 of the snapshot.
    """
    from source.config.multi_instance import write_atomic
    if not cfg.config_djb2:
        # Recompute defensively, caller may have mutated cfg without resaving.
        cfg.config_djb2 = djb2_hex_from_text(canonical_json(cfg.to_dict_for_hash()))
    target_dir = source_configs_dir(project_dir_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{cfg.config_djb2}.json"
    if target.exists():
        return cfg.config_djb2
    write_atomic(target, json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
    return cfg.config_djb2


# ============================================================================
# GUI ↔ Config bridge: ONE module owns create/load/save AND the decomposer
# that drives widgets from Config / reads widgets into Config.
#
# Operant and maze main_windows expose the same minimum host contract:
#
#   host.info_fields:                    Dict[str, QLineEdit/QCheckBox]
#       keys: "experimenter", "project", "session", "dir",
#             "metadata" or "metadata_file", "tracking_enabled"
#
#   host.iter_box_widgets() -> Iterable[(box_number: int, box_widget)]
#       box_widget exposes (best-effort, attribute names duck-typed):
#         .serial_combo or .com_id_edit   (com_port source)
#         .camera_id_edit
#         .subject_id_edit
#         .save_video_enabled
#         .roi_normalized
#         .roi_segment                    (operant only; pixel form)
#         .init_hw_def, .action_config, .api_class  (FileRef attrs)
#         ._hw_def_path                   (maze only; path string)
#         .bg_captured_at, .zones, .tracking_enabled, .save_tracking
#
#   host.pipeline:                       camera + tracking registries
#       host.pipeline.all_camera_configs()   -> Dict[str, pipeline.CameraConfig]
#       host.pipeline.install_camera_configs(dict)
#       host.pipeline.all_tracking_configs() -> Dict[int, pipeline.TrackingConfig]
#       host.pipeline.install_tracking_configs(dict)
#
#   host.video_target_fps / .video_frame_strategy / .video_grayscale /
#   ._camera_backend                     attrs (or absent)
#
#   host.video_segment_config            runtime in-memory mirror dict
#                                        (the bridge keeps this synced as
#                                         a convenience for downstream
#                                         camera/recorder code)
#
#   host._tracking_dialog_globals               dict
#
#   host.statisticsTab (optional)        for stats_config bridge
#
#   host.apply_mode_extras(cfg) / .read_mode_extras(cfg)   (optional)
#       Mode-specific knobs that don't fit the common shape (maze's
#       tracking_zones / tracking_enabled main-window dicts, maze's
#       _hw_def_path SetupWidget attribute, top-level maze_zones).
# ============================================================================


def _widget_text(w) -> str:
    """Read text from QLineEdit / QComboBox / QTextEdit. ``""`` when unset."""
    if w is None:
        return ""
    for attr in ("text", "currentText", "toPlainText"):
        fn = getattr(w, attr, None)
        if callable(fn):
            try:
                v = fn()
                return str(v) if v is not None else ""
            except Exception:
                return ""
    return ""


def _set_widget_text(w, value: str) -> None:
    if w is None:
        return
    for attr in ("setText", "setCurrentText", "setPlainText"):
        fn = getattr(w, attr, None)
        if callable(fn):
            try:
                fn(str(value or ""))
                return
            except Exception:
                return


def _is_real_user_value(s: str) -> bool:
    """Strip placeholders like ``"--- Select COM ---"`` that the combo
    boxes show when no real value is picked."""
    return bool(s) and not s.startswith("---")


def _set_checkable(w, checked: bool) -> None:
    if w is None:
        return
    fn = getattr(w, "setChecked", None)
    if not callable(fn):
        return
    # Block signals during the programmatic set so loading a config never
    # fires a user-action handler mid-load (e.g. the tracking-enabled
    # checkbox's toggled→_on_tracking_toggle would pop a modal and dirty
    # the just-loaded project). SnapshotStore is already synced from
    # cfg.meta.tracking_enabled at load time.
    blocker = getattr(w, "blockSignals", None)
    was_blocked = w.signalsBlocked() if hasattr(w, "signalsBlocked") else False
    try:
        if callable(blocker):
            w.blockSignals(True)
        fn(bool(checked))
    except Exception:
        pass
    finally:
        if callable(blocker):
            try:
                w.blockSignals(was_blocked)
            except Exception:
                pass


def _is_checked(w, default: bool = False) -> bool:
    if w is None:
        return default
    fn = getattr(w, "isChecked", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return default
    return default


# -------- meta --------

def _apply_meta(cfg: Config, host) -> None:
    info = getattr(host, "info_fields", None) or {}
    _set_widget_text(info.get("experimenter"), cfg.meta.experimenter)
    _set_widget_text(info.get("project"),      cfg.meta.project)
    _set_widget_text(info.get("session"),      cfg.meta.session_label)
    # If the config has a blank data_dir but a project name, fall back
    # to the canonical default. Doesn't modify cfg, only the widget.
    data_dir = cfg.meta.data_dir or _default_data_dir_for(cfg.meta.project)
    _set_widget_text(info.get("dir"),          data_dir)
    meta_w = info.get("metadata") or info.get("metadata_file")
    _set_widget_text(meta_w, cfg.meta.metadata_file)
    _set_checkable(info.get("tracking_enabled"), cfg.meta.tracking_enabled)

    # MCU label mode: the project's pinned mode wins; empty inherits the
    # per-machine default. Set the process-wide mode so per-box COM fields
    # (label_for) render in the right scheme on load.
    try:
        from source.communication import mcu_ports
        from source.config.settings import get_mcu_display_mode
        mcu_ports.set_display_mode(cfg.meta.mcu_display_mode or get_mcu_display_mode())
    except Exception as e:
        logger.debug("apply mcu display mode: %s", e)


def _read_meta(host, cfg: Config) -> None:
    info = getattr(host, "info_fields", None) or {}
    base_meta = getattr(getattr(host, "_active_config", None), "meta", None)

    # Seed cfg.meta from the active config so fields WITHOUT an authoritative
    # sidebar widget (created_at, metadata_file, and the read-only project +
    # data_dir) survive a fresh read_ui_into_config instead of resetting to
    # Meta() defaults. deepcopy carries any future Meta field forward too.
    if base_meta is not None:
        import copy as _copy
        cfg.meta = _copy.deepcopy(base_meta)

    # --- genuinely user-editable sidebar widgets override the seed ---
    exp_w = info.get("experimenter")
    if exp_w is not None:
        cfg.meta.experimenter = _widget_text(exp_w).strip()
    ses_w = info.get("session")
    if ses_w is not None:
        cfg.meta.session_label = _widget_text(ses_w).strip()
    tk = info.get("tracking_enabled")
    if tk is not None:
        cfg.meta.tracking_enabled = _is_checked(tk, default=True)
    meta_w = info.get("metadata") or info.get("metadata_file")
    if meta_w is not None:
        cfg.meta.metadata_file = _widget_text(meta_w).strip()

    # --- display-only fields: the read-only Project / Data Directory widgets
    # only MIRROR cfg.meta. Read them solely on the VERY FIRST save (no active
    # config yet); save_project then fills the project name from the chosen
    # folder, and the data dir defaults to <top>/data/<project>. A custom data
    # dir can only be set via Browse…, which requires a loaded project (so
    # base_meta is non-None by then and the seeded value carries through). ---
    if base_meta is None:
        cfg.meta.project = _widget_text(info.get("project")).strip()
        cfg.meta.data_dir = ""

    cfg.experiment_name = cfg.meta.project


# -------- cameras: video_defaults --------

def _apply_video_defaults(cfg: Config, host) -> None:
    vd = cfg.cameras.video_defaults
    host.video_target_fps     = int(vd.target_fps)
    host.video_frame_strategy = str(vd.frame_strategy)
    host.video_grayscale      = bool(vd.grayscale)
    host._camera_backend      = str(vd.camera_backend)


def _read_video_defaults(host, cfg: Config) -> None:
    vd = cfg.cameras.video_defaults
    # Guard against attributes present-but-None: getattr's default only
    # fires when the attribute is MISSING; a present None would become the
    # string "None" via str().
    fps = getattr(host, "video_target_fps", None)
    if fps is not None:
        try:
            vd.target_fps = int(fps)
        except (TypeError, ValueError):
            pass
    fs = getattr(host, "video_frame_strategy", None)
    if fs:
        vd.frame_strategy = str(fs)
    gs = getattr(host, "video_grayscale", None)
    if gs is not None:
        vd.grayscale = bool(gs)
    cb = getattr(host, "_camera_backend", None)
    if cb:
        vd.camera_backend = str(cb)


# -------- cameras: registry (pipeline.CameraConfig ↔ CameraEntry) --------

def _camera_json_to_entry(cam_id: str, raw: Dict[str, Any]) -> "CameraEntry":
    """pipeline.CameraConfig.to_json() shape -> CameraEntry."""
    sel_res = raw.get("selected_resolution") or (None, None)
    try:
        w, h = int(sel_res[0]), int(sel_res[1])
    except (TypeError, ValueError, IndexError):
        w, h = None, None
    sel_fps = raw.get("selected_fps")
    try:
        sel_fps = int(sel_fps) if sel_fps is not None else None
    except (TypeError, ValueError):
        sel_fps = None
    extra = raw.get("extra") or {}

    # exposure/gain are TOP-LEVEL in CameraConfig.to_json(); fall back to extra
    # only for legacy blobs that nested them there.
    def _pick(key):
        v = raw.get(key)
        return extra.get(key) if v is None else v

    _exp, _gain = _pick("exposure_us"), _pick("gain_db")
    pref = CameraPreference(
        width=w, height=h, fps=sel_fps,
        codec=str(raw["capture_format"]) if raw.get("capture_format") else None,
        capture_backend=str(raw.get("capture_backend") or ""),
        exposure_us=int(_exp) if _exp is not None else None,
        gain_db=float(_gain) if _gain is not None else None,
        external_trigger=bool(extra.get("external_trigger", False)),
        flip_horizontal=bool(raw.get("flip_horizontal", False)),
        flip_vertical=bool(raw.get("flip_vertical", False)),
        grayscale=bool(raw.get("grayscale", False)),
        frame_strategy=str(raw.get("frame_strategy") or ""),
        # Structured scientific I/O, kept so the project config is self-contained.
        trigger=(dict(raw["trigger"]) if isinstance(raw.get("trigger"), dict) else None),
        line_output=(dict(raw["line_output"])
                     if isinstance(raw.get("line_output"), dict) else None),
        sync_role=str(raw.get("sync_role", "none") or "none"),
        features=(dict(raw["features"])
                  if isinstance(raw.get("features"), dict) else {}),
    )
    # Probed modes -> resolutions list (group fps_options per (w,h)).
    by_res: Dict[Tuple[int, int], List[int]] = {}
    for m in raw.get("probed_modes") or []:
        try:
            rw, rh, rf = int(m[0]), int(m[1]), int(round(float(m[2])))
        except (TypeError, ValueError, IndexError):
            continue
        by_res.setdefault((rw, rh), []).append(rf)
    resolutions = [
        CameraResolutionEntry(width=rw, height=rh,
                              fps_options=sorted(set(fps_list)))
        for (rw, rh), fps_list in by_res.items()
    ]
    # ``or ""`` / ``or "opencv"`` after the get so a key-present-with-None
    # cell in the pipeline TC's to_json() output doesn't produce a literal
    # "None" string in the saved CameraEntry.
    variants: Dict[str, List[Tuple[int, int, float]]] = {}
    for name, modes in (raw.get("probed_variants") or {}).items():
        clean = []
        for m in (modes or []):
            try:
                clean.append((int(m[0]), int(m[1]), float(m[2])))
            except (TypeError, ValueError, IndexError):
                continue
        if clean:
            variants[str(name)] = clean
    caps = CameraCapabilities(
        scanned_at=str(extra.get("scanned_at") or ""),
        resolutions=resolutions,
        probed_variants=variants,
    )
    return CameraEntry(
        camera_id=str(cam_id),
        label=str(extra.get("label") or ""),
        backend=str(raw.get("camera_backend") or "opencv"),
        unique_id=str(extra.get("unique_id") or ""),
        preference=pref,
        capabilities=caps,
    )


def _entry_to_camera_json(entry: "CameraEntry") -> Dict[str, Any]:
    """CameraEntry -> pipeline.CameraConfig.from_json() shape."""
    sel_res = None
    if entry.preference.width is not None and entry.preference.height is not None:
        sel_res = [entry.preference.width, entry.preference.height]
    probed = [[res.width, res.height, fps]
              for res in entry.capabilities.resolutions
              for fps in (res.fps_options or [0])]
    extra: Dict[str, Any] = {
        "label":            entry.label,
        "unique_id":        entry.unique_id,
        "external_trigger": entry.preference.external_trigger,
        "scanned_at":       entry.capabilities.scanned_at,
    }
    extra = {k: v for k, v in extra.items() if v is not None and v != ""}
    # exposure/gain/trigger/line_output/sync_role go TOP-LEVEL, that's where
    # CameraConfig.from_json() reads them (nesting them in extra silently drops
    # them on load).
    out: Dict[str, Any] = {
        "camera_id":           entry.camera_id,
        "camera_backend":      entry.backend,
        "selected_resolution": sel_res,
        "selected_fps":        entry.preference.fps,
        "probed_modes":        probed,
        "capture_format":      entry.preference.codec,
        "capture_backend":     entry.preference.capture_backend,
        "probed_variants":     dict(entry.capabilities.probed_variants or {}),
        "exposure_us":         entry.preference.exposure_us,
        "gain_db":             entry.preference.gain_db,
        # Top-level, where CameraConfig.from_json reads them; nesting these
        # in ``extra`` is exactly how they were being lost.
        "flip_horizontal":     entry.preference.flip_horizontal,
        "flip_vertical":       entry.preference.flip_vertical,
        "grayscale":           entry.preference.grayscale,
        "frame_strategy":      entry.preference.frame_strategy,
        # external_trigger ALSO goes top-level: the pipeline's legacy-compat
        # trigger reader looks for it there when no ``trigger`` block exists;
        # the copy in ``extra`` only round-trips back into CameraPreference.
        "external_trigger":    bool(entry.preference.external_trigger),
        "sync_role":           entry.preference.sync_role or "none",
        "features":            dict(entry.preference.features or {}),
        "extra":               extra,
        "user_applied":        bool(entry.preference.width and entry.preference.height),
    }
    if entry.preference.trigger is not None:
        out["trigger"] = dict(entry.preference.trigger)
    if entry.preference.line_output is not None:
        out["line_output"] = dict(entry.preference.line_output)
    return out


def _apply_camera_registry(cfg: Config, host) -> None:
    pipe = getattr(host, "pipeline", None)
    if pipe is None or not hasattr(pipe, "install_camera_configs"):
        return
    # Clear existing then install fresh (pipeline does the actual replace).
    try:
        if hasattr(pipe, "_camera_configs"):
            pipe._camera_configs.clear()
    except Exception:
        pass
    payload = {e.camera_id: _entry_to_camera_json(e) for e in cfg.cameras.registry}
    try:
        pipe.install_camera_configs(payload)
    except Exception as e:
        logger.warning("install_camera_configs failed: %s", e)
        return
    # A camera that saved its own rate keeps it. install_camera_configs clamps
    # a config with no selected_fps to the camera's max, so the rig-wide
    # video_defaults.target_fps fills in only for those, applying it to every
    # camera would flatten a mixed-rate rig back to one rate on each load.
    target_fps = int(getattr(host, "video_target_fps", 0) or 0)
    if target_fps > 0:
        for cam in pipe.all_camera_configs().values():
            if not getattr(cam, "selected_fps", None):
                cam.selected_fps = target_fps


def _read_camera_registry(host, cfg: Config) -> None:
    pipe = getattr(host, "pipeline", None)
    if pipe is None or not hasattr(pipe, "all_camera_configs"):
        return
    try:
        live = pipe.all_camera_configs() or {}
    except Exception:
        live = {}
    # Empty pipeline → keep the seeded registry (deepcopied from
    # _active_config in read_ui_into_config) rather than wiping it. The pipe
    # is transiently empty when it's rebuilt on a camera reconnect; overwriting
    # here would drop every saved CameraEntry. Same guard as _read_tracking.
    if not live:
        return
    registry: List[CameraEntry] = []
    seen: set = set()
    for cid, cam in live.items():
        try:
            raw = cam.to_json() if hasattr(cam, "to_json") else dict(cam)
        except Exception:
            continue
        key = str(cid)
        # An index is a location, not an identity: it changes with whatever
        # else is plugged in. Saving one as a registry key produced projects
        # carrying '0', '1' AND 'fp755b78f6' for a single physical camera,
        # with contradictory resolutions, and settings written against the
        # index entry never reached the box, which binds the identity.
        if key.isdigit() or key.endswith("-opencv"):
            logger.warning(
                "camera registry: entry keyed by index %r dropped. An index "
                "is not an identity, it moves when cameras are plugged in or "
                "removed. Re-pick this camera in the camera dialog so it is "
                "stored by its own id.", key)
            continue
        if key in seen:
            continue          # one entry per camera, first wins
        seen.add(key)
        registry.append(_camera_json_to_entry(key, raw))
    cfg.cameras.registry = registry


# -------- per-box (operant BoxControlWidget + maze SetupWidget) --------

def _box_com_widget(bw):
    """Return the widget that holds the canonical com_port value, the
    visible ``com_id_edit`` in both modes. (Operant also exposes a hidden
    ``serial_combo`` fallback, but com_id_edit is the value source.)"""
    return getattr(bw, "com_id_edit", None) or getattr(bw, "serial_combo", None)


def _apply_per_box(cfg: Config, host) -> None:
    """Walk cfg.setup_config.boxes; for each, find the host's widget and
    populate it. The orchestrator does NOT create widgets, callers are
    expected to have called add_box() / _add_setup() N times before
    apply (load_config does this via _load_clear_existing_state +
    add_box-per-row).
    """
    boxes_by_num = {b.setup_number: b for b in cfg.setup_config.boxes}
    iter_fn = getattr(host, "iter_box_widgets", None)
    if iter_fn is None:
        return
    for bn, bw in iter_fn():
        box = boxes_by_num.get(int(bn))
        if box is None:
            continue
        # Restore the MCU serial onto the widget. Prefer the saved
        # ``mcu_serial``; otherwise upgrade ``com_port`` by looking up the
        # live serial for that device path so the next autosave persists it.
        bw._mcu_serial = box.mcu_serial or ""
        if not bw._mcu_serial and box.com_port:
            try:
                from source.communication.mcu_ports import serial_for_device
                upgraded = serial_for_device(box.com_port) or ""
                if upgraded:
                    bw._mcu_serial = upgraded
                    # Also patch cfg so the very next autosave persists.
                    box.mcu_serial = upgraded
            except Exception:
                pass
        # The com field shows the operator-facing label (COMn on Windows,
        # serial on Linux) resolved from the stable serial; falls back to
        # the saved com_port when no serial is known.
        from source.communication.mcu_ports import label_for
        _set_widget_text(_box_com_widget(bw),
                         label_for(bw._mcu_serial) or box.com_port)
        _set_widget_text(getattr(bw, "camera_id_edit", None), box.camera_id)
        # save_video lands on .save_video_enabled (attribute, not widget).
        if hasattr(bw, "save_video_enabled"):
            bw.save_video_enabled = bool(box.save_video)
        # ROI / geometry, both forms restored.
        if box.roi_normalized and len(box.roi_normalized) == 4:
            try:
                bw.roi_normalized = tuple(float(x) for x in box.roi_normalized)
            except (TypeError, ValueError):
                pass
        if box.geometry.is_set():
            try:
                bw.roi_segment = (
                    int(box.geometry.x), int(box.geometry.y),
                    int(box.geometry.w), int(box.geometry.h),
                )
            except Exception:
                pass
        # FileRef attrs (operant write-back from snapshot capture).
        if box.init_hw_def.is_set():
            bw.init_hw_def = box.init_hw_def
            # maze SetupWidget separately tracks the path string.
            if hasattr(bw, "_hw_def_path"):
                bw._hw_def_path = box.init_hw_def.path
        if box.action_config.is_set():
            bw.action_config = box.action_config
            # operant BoxControlWidget mirrors as string for the
            # ConfigSelectionDialog quick-status.
            if hasattr(bw, "action_config_path"):
                bw.action_config_path = box.action_config.path
        if box.api_class.is_set():
            bw.api_class = box.api_class
        if box.bg_captured_at and hasattr(bw, "bg_captured_at"):
            bw.bg_captured_at = box.bg_captured_at


def _looks_like_port(text: str) -> bool:
    """True for a real OS port name: ``COM7`` or ``/dev/ttyACM1``."""
    t = (text or "").strip()
    return bool(t) and (t.upper().startswith("COM") or t.startswith("/dev/"))


def _box_port_value(bw, base: BoxConfig) -> str:
    """The device path to save for this box.

    The per-box COM field shows a LABEL, and under the default display mode
    that label is an 8-hex hash of the serial (``7dcefce1``), or the raw serial
    under "serial" mode. Saving the field verbatim put those into ``com_port``,
    where they are worthless: nothing can open them, on any machine, and
    ``com_port`` is precisely the fallback for a box whose serial is unknown.
    So it holds a real port or nothing: the live path of the bound board when
    one is known, else the port saved last time.
    """
    text = _real_or_blank(_widget_text(_box_com_widget(bw)))
    if _looks_like_port(text):
        return text
    serial = str(getattr(bw, "_mcu_serial", "") or base.mcu_serial or "")
    if serial:
        try:
            from source.communication.mcu_ports import device_for_serial
            live = device_for_serial(serial)
        except Exception as e:
            logger.debug("com_port lookup for %s: %s", serial, e)
            live = None
        if live:
            return live
    return base.com_port or ""


def _read_per_box(host, cfg: Config) -> None:
    iter_fn = getattr(host, "iter_box_widgets", None)
    if iter_fn is None:
        return
    # Per-box fallbacks for state not always live on the widget: mcu_serial
    # before the box has connected, and geometry / roi_normalized while a
    # camera is transiently detached. ``cfg`` is freshly built here, so the
    # last-saved boxes come from ``_active_config``, not ``cfg`` (which is
    # always empty, which had silently disabled the fallback entirely).
    base_cfg = getattr(host, "_active_config", None)
    prev = ({b.setup_number: b for b in base_cfg.setup_config.boxes}
            if base_cfg is not None and getattr(base_cfg, "setup_config", None)
            and base_cfg.setup_config.boxes else {})
    boxes: List[BoxConfig] = []
    for bn, bw in iter_fn():
        setup_number = int(bn)
        base = prev.get(setup_number, BoxConfig(setup_number=setup_number))
        box = BoxConfig(
            setup_number=setup_number,
            # mcu_serial is the stable USB serial number, captured by
            # ``connect_mcu`` and stashed on the widget as ``_mcu_serial``.
            # Falls back to the saved value when the widget hasn't been
            # connected yet (e.g. autosave fires before first connect).
            mcu_serial=str(getattr(bw, "_mcu_serial", "")
                           or base.mcu_serial or ""),
            com_port=_box_port_value(bw, base),
            init_hw_def=getattr(bw, "init_hw_def", FileRef()) or FileRef(),
            camera_id=_real_or_blank(_widget_text(getattr(bw, "camera_id_edit", None))),
            geometry=base.geometry,
            roi_normalized=base.roi_normalized,
            save_video=bool(getattr(bw, "save_video_enabled", True)),
            bg_captured_at=str(getattr(bw, "bg_captured_at", "") or ""),
            tracking_enabled=bool(getattr(bw, "tracking_enabled", False)),
            save_tracking=bool(getattr(bw, "save_tracking", False)),
            # Normalise to typed Zone instances, the widget may hold
            # dict-shape zones (dialog wire format). Without this, dicts
            # leak into BoxConfig.zones and to_compact's asdict(z) raises
            # "asdict() should be called on dataclass instances" at save
            # for any box _read_tracking doesn't later overwrite.
            #
            # No widget actually exposes a ``.zones`` attribute, the runtime
            # store is ``host.tracking_zones``, read by ``_read_tracking``.
            # So this always fell back, and the fallback MUST be the previous
            # config's zones (carry-forward), never ``[]``: an autosave firing
            # while ``tracking_zones`` is transiently empty would otherwise
            # write an empty list here and permanently wipe the box's zones
            # (the "read_ui carry-forward" bug class).
            zones=[z if isinstance(z, Zone) else Zone.from_dict(z)
                   for z in (getattr(bw, "zones", None) or base.zones or [])],
            action_config=getattr(bw, "action_config", FileRef()) or FileRef(),
            api_class=getattr(bw, "api_class", FileRef()) or FileRef(),
        )
        # ROI, ``roi_normalized`` is the single canonical stored form
        # (resolution-independent; the runtime crop derives pixels from it).
        # The widget's ``roi_segment`` is a per-session PIXEL cache at whatever
        # resolution it was last drawn, persisting it as ``geometry`` created
        # a second source of truth that went stale and disagreed with
        # ``roi_normalized`` across a resolution change. So we no longer
        # write pixel geometry from it; ``box.geometry`` carries forward from
        # the previous config, and the video-segment mirror rebuilds any pixel
        # form it needs from the normalized rect at the live resolution.
        roi_n = getattr(bw, "roi_normalized", None)
        if roi_n and len(roi_n) == 4:
            try:
                box.roi_normalized = [float(x) for x in roi_n]
            except (TypeError, ValueError):
                pass
        # Maze SetupWidget stashes HD as a path string only; build FileRef.
        hd_path = getattr(bw, "_hw_def_path", None)
        if (not box.init_hw_def.is_set()) and hd_path:
            box.init_hw_def = FileRef.from_path(hd_path)
        # Operant mirrors action_config as a path string too.
        ac_path = getattr(bw, "action_config_path", None)
        if (not box.action_config.is_set()) and ac_path:
            box.action_config = FileRef.from_path(ac_path)
        boxes.append(box)
    cfg.setup_config.boxes = boxes


def _real_or_blank(s: str) -> str:
    """Strip placeholders like ``"--- Select COM ---"`` from combo text."""
    return s if _is_real_user_value(s) else ""


# -------- tracking (pipeline.TrackingConfig ↔ cfg.tracking + per-box) --------

# The blob-tracker fields carried between cfg.tracking.blob and the per-box
# pipeline TC (as ``blob_<key>``). ONE source so _apply_tracking (write) and
# _read_tracking (read) can't drift.
_BLOB_KEYS = (
    "threshold", "min_area", "max_area", "detect_dark", "blur_mode",
    "blur_kernel_size", "bg_mode", "open_kernel_size", "close_kernel_size",
    "use_clahe", "clahe_clip_limit", "clahe_tile_size", "use_adaptive_threshold",
    "self_norm_ratio", "self_norm_sigma", "self_norm_smooth_sigma",
    "self_norm_minsize",
)


def _apply_tracking(cfg: Config, host) -> None:
    pipe = getattr(host, "pipeline", None)
    if pipe is None or not hasattr(pipe, "install_tracking_configs"):
        return
    try:
        if hasattr(pipe, "_tracking_configs"):
            pipe._tracking_configs.clear()
    except Exception:
        pass
    # Build per-box TrackingConfig dicts from cfg.tracking (rig-level) +
    # per-box fields. When the rig has tracking configured
    # (``cfg.tracking.enabled``), install a TC for EVERY box so the
    # dialog round-trips the model_path/mode even if no box is currently
    # ticked. When not, install only for boxes that have per-box state
    # (zones / save_tracking flag) so we don't pollute fresh boxes.
    rig_dlc = cfg.tracking.dlc
    rig_blob = cfg.tracking.blob
    rig_sleap = cfg.tracking.sleap
    payload: Dict[str, Dict[str, Any]] = {}
    # "Configured" requires a real model for pose modes, not just the
    # enabled flag + mode string, else a box with no model subscribes the
    # pose sink and trips the "no model handle" warning. Blob needs no model.
    rig_configured = bool(cfg.tracking.enabled and (
        cfg.tracking.mode == "blob"
        or (cfg.tracking.mode == "dlc" and rig_dlc.model_path)
        or (cfg.tracking.mode == "sleap" and rig_sleap.model_path)
    ))
    for box in cfg.setup_config.boxes:
        if not (rig_configured or box.tracking_enabled or box.zones or box.save_tracking):
            continue
        raw: Dict[str, Any] = {
            "box_id":                  box.setup_number,
            # Saved mode even when the rig gate is off, a disabled rig's
            # boxes must not claim "dlc" when the operator never picked a
            # pose backend (placeholder-then-fill: gate flag + real state).
            "tracker_type":            cfg.tracking.mode or "none",
            "online_tracking_enabled": bool(box.tracking_enabled),
            "annotation_enabled":      bool(box.save_tracking),
            # ``user_applied`` = the operator made a model/mode choice.
            # Rig-level config counts even if this box's checkbox is off,
            # so the dialog reopen path isn't blank after project reload.
            "user_applied":            bool(box.tracking_enabled or rig_configured),
            "zones":                   [zone_to_dict(z) for z in box.zones],
            # Rig-level MCU push gates + zone-change body-part picker; same
            # value on every box. Pass them through so the framebus TC's
            # from_json doesn't fall back to defaults on project reload.
            "push_zones_to_mcu":       bool(cfg.tracking.push_zones_to_mcu),
            "push_coords_to_mcu":      bool(cfg.tracking.push_coords_to_mcu),
            "push_frame_event":        bool(cfg.tracking.push_frame_event),
            "zone_change_body_part":   str(cfg.tracking.zone_change_body_part or "centroid"),
        }
        if cfg.tracking.mode == "dlc":
            raw.update({
                "dlc_model_path":       rig_dlc.model_path or "",
                "keypoint_names":       list(rig_dlc.body_parts),
                "confidence_threshold": rig_dlc.confidence,
                "pose_resize_factor":   float(rig_dlc.resize),
                "pose_n_instances":     int(rig_dlc.instances),
            })
        elif cfg.tracking.mode == "sleap":
            raw.update({
                "dlc_model_path":       rig_sleap.model_path or "",
                "keypoint_names":       list(rig_sleap.body_parts),
                "confidence_threshold": rig_sleap.confidence,
                "pose_resize_factor":   float(rig_sleap.resize),
                "pose_n_instances":     int(rig_sleap.instances),
                "tracker_type":         "sleap",
            })
        elif cfg.tracking.mode == "blob":
            for k in _BLOB_KEYS:
                v = getattr(rig_blob, k, None)
                if v is not None:
                    raw[f"blob_{k}"] = v
            raw["blob_smooth_tracking"] = bool(rig_blob.smooth_tracking)
        payload[str(box.setup_number)] = raw
    try:
        pipe.install_tracking_configs(payload)
    except Exception as e:
        logger.warning("install_tracking_configs failed: %s", e)


def _read_tracking(host, cfg: Config) -> None:
    by_num = {b.setup_number: b for b in cfg.setup_config.boxes}

    # Per-box zones: the runtime source of truth is host.tracking_zones
    # (shared by renderer / Home button / zone editor). The pipeline TC.zones
    # list only syncs for tracking-enabled boxes, so reading it would lose
    # zones drawn while tracking was off.
    runtime_zones = getattr(host, "tracking_zones", None) or {}
    for raw_bid, zones in runtime_zones.items():
        try:
            bn = int(raw_bid)
        except (TypeError, ValueError):
            continue
        box = by_num.get(bn)
        if box is None:
            continue
        # A box KEY present in tracking_zones is authoritative, including an
        # empty list, which means the operator deleted every zone and must
        # persist as empty. Skipping empties here would combine with
        # _read_per_box's carry-forward seed to resurrect deleted zones on the
        # next reload. Carry-forward covers only boxes ABSENT from
        # tracking_zones (the transient-empty case _read_per_box guards).
        if not zones:
            box.zones = []
            continue
        # tracking_zones normally holds dialog-wire dicts, but be defensive:
        # accept an already-typed Zone too so a stray dataclass isn't dropped
        # (which would wipe the box's zones on the next autosave).
        box.zones = [
            z if isinstance(z, Zone) else Zone.from_dict(z)
            for z in zones
            if isinstance(z, (dict, Zone))
        ]

    # Tracking-mode + per-box enable / save flags still come from the
    # pipeline TC registry (populated by _apply_dialog_config).
    pipe = getattr(host, "pipeline", None)
    if pipe is None or not hasattr(pipe, "all_tracking_configs"):
        return
    try:
        live = pipe.all_tracking_configs() or {}
    except Exception:
        live = {}
    if not live:
        return
    rig_set = False
    for bid, tc in live.items():
        try:
            raw = tc.to_json() if hasattr(tc, "to_json") else dict(tc)
        except Exception:
            continue
        bn = int(bid)
        box = by_num.get(bn)
        if box is not None:
            box.tracking_enabled = bool(raw.get("online_tracking_enabled",
                                                raw.get("user_applied", False)))
            box.save_tracking    = bool(raw.get("annotation_enabled", False))
            # Zones already read from host.tracking_zones above.
        mode = str(raw.get("tracker_type", "none"))
        # Normalise the model path: a missing key, None, "", or the literal
        # string "None" (the str(None) bug class) all mean "no model".
        _mp_raw = raw.get("dlc_model_path")
        model_path = (
            "" if _mp_raw is None or str(_mp_raw).strip().lower() in ("", "none")
            else str(_mp_raw).strip())
        # "Configured" means the operator actually picked a pose model
        # (dlc/sleap) or set up blob, not merely that a per-box online-
        # tracking checkbox is on. Guards against saving an enabled pose
        # mode with an empty model that trips the pose-sink warning on load.
        #
        # A pose model with a real filesystem path is unambiguously a real
        # config, pipeline-default TCs have ``dlc_model_path=None``, so
        # persist it whether or not ``user_applied`` is set. The model may
        # have been installed by auto-init / snapshot-recovery on load (which
        # don't stamp user_applied), not only by a dialog Apply; requiring
        # user_applied here silently dropped DLC/SLEAP from every save.
        # Blob still requires user_applied: ``mode == "blob"`` alone (no
        # model) can be a bare default TC we shouldn't persist.
        pose_configured = mode in ("dlc", "sleap") and bool(model_path)
        blob_configured = mode == "blob" and bool(raw.get("user_applied"))
        if not rig_set and (pose_configured or blob_configured):
            cfg.tracking.enabled = True
            cfg.tracking.mode = mode
            # Capture the rig-level MCU push gates + body-part picker from
            # the first user-applied TC (dialog wrote the same value to all).
            cfg.tracking.push_zones_to_mcu = bool(
                raw.get("push_zones_to_mcu", True))
            cfg.tracking.push_coords_to_mcu = bool(
                raw.get("push_coords_to_mcu", True))
            cfg.tracking.push_frame_event = bool(
                raw.get("push_frame_event", False))
            cfg.tracking.zone_change_body_part = str(
                raw.get("zone_change_body_part", "centroid") or "centroid")
            # The pipeline TC's ``dlc_model_path`` is the filesystem path
            # the user picked. model_path is already normalised above so an
            # empty pick saves as "" rather than the literal "None".
            if cfg.tracking.mode == "dlc":
                cfg.tracking.dlc.model_path = model_path
                cfg.tracking.dlc.body_parts = list(raw.get("keypoint_names", []))
                cfg.tracking.dlc.confidence = float(raw.get("confidence_threshold", 0.5))
                # Capture the operator-edited resize + n_instances so they
                # survive project reload.
                cfg.tracking.dlc.resize = float(raw.get("pose_resize_factor", 1.0))
                cfg.tracking.dlc.instances = int(raw.get("pose_n_instances", 1))
            elif cfg.tracking.mode == "sleap":
                cfg.tracking.sleap.model_path = model_path
                cfg.tracking.sleap.body_parts = list(raw.get("keypoint_names", []))
                cfg.tracking.sleap.confidence = float(raw.get("confidence_threshold", 0.75))
                cfg.tracking.sleap.resize = float(raw.get("pose_resize_factor", 1.0))
                cfg.tracking.sleap.instances = int(raw.get("pose_n_instances", 1))
            elif cfg.tracking.mode == "blob":
                for k in _BLOB_KEYS:
                    v = raw.get(f"blob_{k}")
                    if v is not None and hasattr(cfg.tracking.blob, k):
                        setattr(cfg.tracking.blob, k, v)
                # Persist the smoothing toggle (per-box TC blob_smooth_tracking
                # mirrored onto cfg.tracking.blob.smooth_tracking).
                cfg.tracking.blob.smooth_tracking = bool(
                    raw.get("blob_smooth_tracking", True))
            rig_set = True


def sync_active_config_tracking(host) -> None:
    """Mirror the live pipeline's tracking config into
    ``host._active_config.tracking`` right when the user applies it.

    ``read_ui_into_config`` seeds each save's ``cfg.tracking`` from
    ``_active_config.tracking`` and ``_read_tracking`` only *overwrites* it
    when the live pipeline holds a configured pose/blob TC. If the pipeline
    is transiently empty at autosave time (e.g. rebuilt on a camera
    reconnect), the seed is what survives, so the durable copy must be
    refreshed the moment the dialog applies a model, not only on the next
    full read. Without this, DLC configured mid-session could vanish from a
    later save and never come back on reload.

    Reads the pipeline into a throwaway Config (seeded from the current
    durable tracking) and copies only the ``.tracking`` subtree back, so
    per-box state on ``_active_config`` is left untouched.
    """
    base = getattr(host, "_active_config", None)
    if base is None or getattr(base, "tracking", None) is None:
        return
    import copy as _copy
    try:
        scratch_cfg = Config()
        scratch_cfg.setup_config = _copy.deepcopy(base.setup_config)
        scratch_cfg.tracking = _copy.deepcopy(base.tracking)
        _read_tracking(host, scratch_cfg)
        base.tracking = scratch_cfg.tracking
    except Exception as e:
        logger.warning("sync_active_config_tracking failed: %s", e)


def zone_to_dict(z) -> Dict[str, Any]:
    """Serialize a Zone (dataclass) or any dict-like zone to dict.

    Delegates to ``Zone.to_dict``, which drops ``None`` values so a rectangle
    is not padded with the scale/ellipse fields that only apply to other zone
    types. Not ``asdict()``: the field is named ``zone_type`` while the on-disk
    key is ``type``, and ``asdict`` would also try to deep-copy the Shapely
    cache and fail at JSON-encode time.
    """
    if hasattr(z, "to_dict"):
        return z.to_dict()
    if isinstance(z, dict):
        return {k: v for k, v in z.items() if v is not None}
    return {}


# -------- stats --------

def _apply_stats(cfg: Config, host) -> None:
    tab = getattr(host, "statisticsTab", None)
    if tab is None:
        return
    path = cfg.stats_config.template_file.path
    if path:
        try:
            tab.setConfigPath(path)
        except Exception as e:
            logger.debug("statisticsTab.setConfigPath failed: %s", e)


def _read_stats(host, cfg: Config) -> None:
    tab = getattr(host, "statisticsTab", None)
    if tab is None:
        return
    try:
        path = tab.getConfigPath() if hasattr(tab, "getConfigPath") else ""
    except Exception:
        path = ""
    if path:
        cfg.stats_config.template_file = FileRef.from_path(str(path))
        cfg.stats_config.template_name = Path(str(path)).stem


# -------- ui --------

#: Per-box annotation marker radius, inside ``ui.dialog_overrides`` beside the
#: other per-box maps (``annotate_saved``, ``scale``). Keyed by box id as a
#: string, because JSON object keys are strings and a round trip would
#: otherwise turn 1 into "1" and lose the entry.
MARKER_SIZE_KEY = "marker_size_by_box"


def _apply_ui(cfg: Config, host) -> None:
    host._tracking_dialog_globals = dict(cfg.ui.dialog_overrides)
    # Seed the per-box marker sizes the overlay draws with. Read here rather
    # than at first draw so a box whose zone-adjust row is built before any
    # frame arrives still shows the size the project was saved with.
    sizes = {}
    for k, v in (cfg.ui.dialog_overrides.get(MARKER_SIZE_KEY) or {}).items():
        try:
            sizes[int(k)] = max(1, min(20, int(v)))
        except (TypeError, ValueError):
            continue
    host._marker_sizes = sizes


def _read_ui(host, cfg: Config) -> None:
    cfg.ui.dialog_overrides = dict(getattr(host, "_tracking_dialog_globals", {}) or {})
    sizes = getattr(host, "_marker_sizes", None) or {}
    if sizes:
        cfg.ui.dialog_overrides[MARKER_SIZE_KEY] = {
            str(k): int(v) for k, v in sizes.items()}
    else:
        # Nothing customised: drop the key rather than writing an empty map,
        # so a project file only carries what someone actually chose.
        cfg.ui.dialog_overrides.pop(MARKER_SIZE_KEY, None)


# -------- video_segment_config in-memory mirror --------

def _apply_video_segment_mirror(cfg: Config, host) -> None:
    """Rebuild ``host.video_segment_config`` from box geometry + ROI so
    downstream camera/recorder code that still reads the older dict
    sees a consistent picture. Boxes with neither ROI nor geometry are
    skipped, they'd contribute an empty entry.
    """
    boxes_out = []
    for b in cfg.setup_config.boxes:
        has_pixel = b.geometry.is_set()
        has_percent = bool(b.roi_normalized) and len(b.roi_normalized) == 4
        if not (has_pixel or has_percent):
            continue
        pixel = {}
        if has_pixel:
            pixel = {"x": b.geometry.x, "y": b.geometry.y,
                     "width": b.geometry.w, "height": b.geometry.h}
        percent = {}
        if has_percent:
            x, y, w, h = b.roi_normalized
            percent = {"x": x, "y": y, "width": w, "height": h}
        boxes_out.append({
            "box_id":     b.setup_number,
            "box_number": b.setup_number,
            "geometry": {
                "pixel":             pixel,
                "percent":           percent,
                "camera_resolution": {},
            },
        })
    host.video_segment_config = {"boxes": boxes_out} if boxes_out else None


# -------- top-level orchestrators --------

def _recover_tracking_from_snapshots(cfg: Config, host) -> None:
    """Heal a project whose ``cfg.tracking`` is empty but a snapshot still
    holds a populated tracking block.

    Walks ``<project>/source/configs/*.json`` most-recent-first and adopts
    the first snapshot whose ``tracking.enabled`` is True, so the operator
    doesn't have to re-pick model_path + body parts. Next autosave persists
    it; logs a WARNING so the heal is visible in the per-session .log.
    """
    # Only attempt recovery when cfg.tracking really looks wiped, not
    # for fresh projects that never had tracking configured.
    if cfg.tracking is None:
        return
    if cfg.tracking.enabled or cfg.tracking.mode not in ("none", ""):
        return  # cfg.tracking is healthy; nothing to recover
    # Need a project dir to find snapshots.
    project_dir = getattr(host, "_active_project_dir", None)
    if project_dir is None:
        return
    snap_dir = Path(project_dir) / "source" / "configs"
    if not snap_dir.is_dir():
        return

    # Walk snapshots most-recent-first by mtime; use the first one
    # whose tracking block is populated.
    import json as _json
    candidates = sorted(snap_dir.glob("*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for snap_path in candidates:
        try:
            with snap_path.open("r", encoding="utf-8") as f:
                data = _json.load(f)
        except Exception:
            continue
        snap_t = data.get("tracking") if isinstance(data, dict) else None
        if not isinstance(snap_t, dict):
            continue
        if not snap_t.get("enabled"):
            continue
        # Found a populated snapshot, adopt its tracking config.
        try:
            cfg.tracking = TrackingConfig.from_dict(snap_t)
        except Exception as e:
            logger.warning(
                "tracking recovery: snapshot %s parse failed: %s",
                snap_path.name, e)
            continue
        logger.warning(
            "tracking recovery: cfg.tracking was empty; restored "
            "mode=%s, dlc.model=%r, body_parts=%s from snapshot %s. "
            "Next autosave will persist the recovered config.",
            cfg.tracking.mode,
            cfg.tracking.dlc.model_path if cfg.tracking.mode == "dlc" else "",
            (cfg.tracking.dlc.body_parts if cfg.tracking.mode == "dlc"
             else cfg.tracking.sleap.body_parts if cfg.tracking.mode == "sleap"
             else []),
            snap_path.name,
        )
        return
    # No populated snapshot found, leave cfg.tracking as-is (empty).


def apply_config_to_ui(cfg: Config, host) -> None:
    """Drive every host widget from the typed v3 Config.

    Use this on project load (after the caller has cleared existing
    boxes and added one widget per cfg.setup_config.boxes entry).
    Mode-specific extras are handled via ``host.apply_mode_extras(cfg)``
    if defined.
    """
    # Heal cfg.tracking from a source/configs/ snapshot before applying, so
    # an empty tracking block is restored without re-picking model + parts.
    _recover_tracking_from_snapshots(cfg, host)
    _apply_meta(cfg, host)
    _apply_video_defaults(cfg, host)
    _apply_camera_registry(cfg, host)
    _apply_per_box(cfg, host)
    _apply_tracking(cfg, host)
    _apply_stats(cfg, host)
    _apply_ui(cfg, host)
    _apply_video_segment_mirror(cfg, host)
    extras = getattr(host, "apply_mode_extras", None)
    if callable(extras):
        try:
            extras(cfg)
        except Exception as e:
            logger.warning("apply_mode_extras failed: %s", e)


def read_ui_into_config(host) -> Config:
    """Build a fresh v3 Config from the host's current widget state.

    Preserves invariants the GUI can't reproduce (created_at, schema
    version, project mode) from ``host._active_config`` when available.
    """
    import copy as _copy
    base = getattr(host, "_active_config", None)
    cfg = Config(
        schema_version=SCHEMA_VERSION,
        mode=str(getattr(base, "mode", None) or
                 (host._load_mode_name() if hasattr(host, "_load_mode_name")
                  else "operant")),
        created_at=str(getattr(base, "created_at", "") or now_ts()),
    )
    # Seed the sections that are NOT fully reconstructable from live widgets
    # from the last-loaded config, so an autosave firing while a subsystem is
    # transiently empty (pipeline rebuilt on a camera reconnect, a tab absent,
    # a non-widget-backed field like ``camera_settle_ms``)
    # doesn't blow away the saved value. Each ``_read_*`` below overwrites its
    # seed when it holds authoritative live state; otherwise the last-saved
    # value is kept instead of resetting to defaults. Generalises the
    # long-standing tracking seed to cameras / stats / ui too.
    if base is not None:
        for _section in ("tracking", "cameras", "stats_config", "ui"):
            src = getattr(base, _section, None)
            if src is not None:
                try:
                    setattr(cfg, _section, _copy.deepcopy(src))
                except Exception:
                    pass
    _read_meta(host, cfg)
    _read_video_defaults(host, cfg)
    _read_per_box(host, cfg)
    _read_camera_registry(host, cfg)
    _read_tracking(host, cfg)
    _read_stats(host, cfg)
    _read_ui(host, cfg)
    extras = getattr(host, "read_mode_extras", None)
    if callable(extras):
        try:
            extras(cfg)
        except Exception as e:
            logger.warning("read_mode_extras failed: %s", e)
    return cfg
