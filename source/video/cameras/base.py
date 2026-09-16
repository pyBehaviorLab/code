"""
GenericCamera base class - defines the interface all camera backends must implement.

``target_fps`` is the ONLY fps that drives capture. The recorder runs at
``target_fps`` verbatim (20 → 20, 30 → 30). There is no effective/requested/
constrained fps in the schema.

``measured_fps`` is a runtime observation (not part of ResolvedCameraSettings).
The calibration loop in CameraThread sets it; the GUI reads it for
display in the "Measured FPS" column; the per-day runs JSON pulls it
at record-stop. It NEVER feeds back into capture decisions.

ResolvedCameraSettings is the resolved configuration after configure() runs:

    cfg = cam.configure(target_fps=30, exposure_us=5000)
    print(cfg.target_fps)   # 30, exactly what the user picked
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedCameraSettings:
    """What the DEVICE ended up running, after ``configure()`` applied it.

    Frozen, small, and per-open: it is the outcome of a configure call, not a
    stored preference. Do not confuse it with
    ``source.video.framebus.types.CameraConfig``, which is the persisted
    per-camera record the setup dialog edits and the project file saves
    (selected mode, probed modes, backend, trigger wiring, …). Both were once
    called ``CameraConfig`` and both were reachable from ``gui/base.py``, so a
    reader could not tell from a call site which one was in hand.

    ``target_fps`` is the single source of truth for capture rate. Recorder,
    writer and GUI all read it. No capping, no constraint flags.
    """

    target_fps: float
    #: What the device was OBSERVED delivering just after it opened, or None
    #: when it was not measured. ``target_fps`` is what was asked for, and a
    #: UVC driver may accept a rate it cannot hold: asked for 10 fps this rig
    #: reported 10 and delivered 15. The recorder stamps the file with this
    #: when it is present, because a file labelled with the request rather
    #: than the truth plays back at the wrong speed and every timestamp
    #: derived from its nominal rate is wrong.
    delivered_fps: Optional[float] = None
    #: ``"hw"`` when the device stamps each frame itself, ``"host"`` when the
    #: time is taken as the frame is handed over and therefore carries the
    #: whole transport delay. A UVC camera through OpenCV is always the
    #: second; Spinnaker and XIMEA are the first. An analysis aligning video
    #: to controller time is holding a different quantity in each case.
    timestamp_source: str = ""
    exposure_us: Optional[float] = None
    gain_db: Optional[float] = None
    width: int = 0
    height: int = 0
    roi: Optional[tuple[int, int, int, int]] = None


class GenericCamera:
    """Base camera interface. All backends implement this.

    The core contract is:
      1. configure(...) sets fps/exposure/gain/roi (no capping; whatever
         the user picked is what self.config.target_fps becomes)
      2. begin_capturing() starts frame acquisition
      3. get_available_images() drains all buffered frames (non-blocking)
      4. stop_capturing() stops acquisition

    Runtime observation:
      * ``self.measured_fps``: observed delivery rate, populated by
        CameraThread's calibration. Not part of ResolvedCameraSettings.
      * ``self.capable_fps``: the SDK's achievable-rate reading under the
        current exposure/bandwidth/ROI, refreshed by configure(). A
        capability, not a measurement: it informs diagnostics, never drives
        capture and never substitutes for ``measured_fps``.

    get_available_images() returns a dict:
        {
            "images": list[numpy.ndarray],
            "timestamps": list[int],        # nanoseconds
            "dropped_frames": int,
            "timestamp_source": str,        # "hw" or "host"
        }
    """

    #: Whose clock the frame times come from. ``"host"`` means the time is
    #: taken as the frame is handed over and carries the whole transport
    #: delay; ``"hw"`` means the device stamped it. Declared per backend
    #: because it is a property of the backend, not of a session, and it is
    #: recorded in every session header so a reader knows which quantity the
    #: frame times in that file are.
    TIMESTAMP_SOURCE: str = "host"

    def __init__(self):
        self.serial_number: str | None = None
        self.device_model: str = "GenericCamera"
        self.unique_id: str = ""
        self.BUFFER_SIZE: int = 100
        #: Resolved configuration, set by configure(), read by everyone else.
        self.config: Optional[ResolvedCameraSettings] = None
        #: Runtime observation, populated by CameraThread calibration.
        #: 0.0 means "not yet measured".
        self.measured_fps: float = 0.0
        #: Capability, the SDK's achievable-rate reading, refreshed by
        #: configure(). 0.0 means "no reading available". Kept separate from
        #: measured_fps: capable informs, measured observes, target drives.
        self.capable_fps: float = 0.0
        #: When True, configure() skips the frame-rate request, the rate
        #: is driven by the external trigger line instead.
        self._external_trigger: bool = False

    # --- Configuration (single entry point) ----------------------------------

    def configure(
        self,
        fps: Optional[float] = None,
        exposure_us: Optional[float] = None,
        gain_db: Optional[float] = None,
        roi: Optional[tuple[int, int, int, int]] = None,
    ) -> ResolvedCameraSettings:
        """Apply settings and store the target_fps.

        Template shared by the SDK backends: apply ROI / exposure / gain
        through the per-backend ``_apply_*`` hooks (these may physically
        limit fps in the SDK), then apply the requested fps, then store
        ``ResolvedCameraSettings`` with ``target_fps = fps`` verbatim. No clamping,
        no negotiation. ``_read_diagnostic_fps`` feeds ``capable_fps``
        only, a capability, never a cap and never ``measured_fps``.

        Omitted arguments carry forward from the previous config.
        Returns the new ``self.config``.
        """
        prev = self.config
        target_fps = fps if fps is not None else (prev.target_fps if prev else 30.0)
        eff_exposure = (exposure_us if exposure_us is not None
                        else (prev.exposure_us if prev else None))
        eff_gain = gain_db if gain_db is not None else (prev.gain_db if prev else None)

        # 1. ROI first, changes the achievable max rate.
        if roi is not None:
            self._apply_roi(roi)

        # 2. Exposure (caps the rate).
        if eff_exposure is not None:
            self._apply_exposure(eff_exposure)

        # 3. Gain (independent but apply for consistency).
        if eff_gain is not None:
            self._apply_gain(eff_gain)

        # 4. Frame rate request.
        if not self._external_trigger:
            self._apply_frame_rate(target_fps)

        # 5. Diagnostic achievable-rate read (published via capable_fps;
        #    never overrides target_fps, and never measured_fps; that stays
        #    reserved for the calibration loop's genuine observation).
        diagnostic = self._read_diagnostic_fps()
        if diagnostic > 0:
            self.capable_fps = diagnostic

        cfg = ResolvedCameraSettings(
            target_fps=float(target_fps),
            exposure_us=eff_exposure,
            gain_db=eff_gain,
            width=self.get_width(),
            height=self.get_height(),
            roi=roi if roi is not None else (prev.roi if prev else None),
            timestamp_source=self.TIMESTAMP_SOURCE,
        )
        self.config = cfg
        return cfg

    # --- Backend hooks used by the configure() template ----------------------

    def _apply_roi(self, roi: tuple[int, int, int, int]) -> None:
        pass

    def _apply_exposure(self, us: float) -> None:
        pass

    def _apply_gain(self, db: float) -> None:
        pass

    def _apply_frame_rate(self, fps: float) -> None:
        pass

    def _read_diagnostic_fps(self) -> float:
        """Achievable-rate read published via ``capable_fps``.
        0.0 means no reading available."""
        return 0.0

    # --- Frame dimensions ----------------------------------------------------

    def get_width(self) -> int:
        raise NotImplementedError

    def get_height(self) -> int:
        raise NotImplementedError

    def apply_measured_fps(self, measured_fps: float) -> None:
        """Record a runtime FPS observation. Does NOT change target_fps.

        Measurements are observation only. They surface in the
        GUI as a "Measured FPS" column and in the per-day runs JSON;
        they never feed back into capture decisions.
        """
        self.measured_fps = float(measured_fps)

    # --- Acquisition ---------------------------------------------------------

    def begin_capturing(self) -> None:
        raise NotImplementedError

    def stop_capturing(self) -> None:
        raise NotImplementedError

    def get_available_images(self) -> dict | None:
        raise NotImplementedError


# --- Shared helpers (backend enumerators + drain loops) -----------------------

def camera_info(backend: str, serial: str, model: str) -> dict:
    """Enumeration result shape shared by every backend's camera lister."""
    return {
        "unique_id": f"{serial}-{backend}",
        "backend": backend,
        "model": model,
        "serial": serial,
    }


# ── Realistic-FPS probe: one timing policy for every backend ──────────
#
# What a camera *reports* is the rate it was asked for; what it delivers is
# something else, so every backend measures. They each grew their own copy of
# the same settle-then-count loop with DIFFERENT hardcoded windows, OpenCV
# settled 2.0 s and counted over 3.0 s, the SDK backends settled 0.5 s and
# counted over 2.0 s, so the numbers the three produced were not comparable,
# and a camera slower to settle than the window simply read low.
#
# The windows below are the policy. They are deliberately different per family
# and that difference is now visible and named rather than buried three times:
# a USB UVC camera renegotiates its mode and re-converges auto-exposure on
# every open, which takes seconds; an industrial SDK camera does not.
UVC_FPS_DRAIN_S = 2.0
UVC_FPS_MEASURE_S = 3.0
SDK_FPS_DRAIN_S = 0.5
SDK_FPS_MEASURE_S = 2.0

# Below this many frames the window is too short to divide by.
FPS_MIN_FRAMES = 2


def measure_delivered_fps(drain_once, count_once, *,
                          drain_s: float, measure_s: float) -> float:
    """Settle, then count delivered frames over a fixed window.

    ``drain_once()`` is called repeatedly during the settle phase and its
    result ignored; ``count_once()`` is called repeatedly during the measure
    phase and must return how many frames it obtained (0 is fine). Either may
    raise to abort early, whatever was counted so far still stands, because a
    camera that stops delivering mid-window has told us something real.

    Returns frames/second, or ``0.0`` when the window produced too few frames
    to be meaningful.

    Timed on ``perf_counter``: ``monotonic`` ticks every ~15.6 ms on Windows,
    so a camera that aborts inside the first tick measures as zero elapsed and
    its real frames are thrown away, which is the one case this promises to
    keep.
    """
    import time as _t

    t_end = _t.perf_counter() + float(drain_s)
    while _t.perf_counter() < t_end:
        try:
            drain_once()
        except Exception:
            break

    n = 0
    t0 = _t.perf_counter()
    while _t.perf_counter() - t0 < float(measure_s):
        try:
            n += int(count_once() or 0)
        except Exception:
            break
    elapsed = _t.perf_counter() - t0

    if elapsed <= 0 or n < FPS_MIN_FRAMES:
        return 0.0
    return n / elapsed


def normalize_channels(raw, grayscale: bool, copy_bgr: bool = False):
    """Grayscale↔BGR decision shared by the SDK drain loops.

    ``grayscale=True`` → single-channel HxW; ``False`` → 3-channel BGR.
    Mono passthroughs are copied because SDK backends recycle the source
    buffer after the drain call returns. ``copy_bgr`` forces ownership of
    a BGR passthrough too (Ximea recycles its image object between reads).
    """
    if grayscale:
        if raw.ndim == 2:
            return raw.copy()
        if raw.shape[2] == 1:
            return raw[:, :, 0].copy()
        return cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    if raw.ndim == 2 or raw.shape[2] == 1:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    return raw.copy() if copy_bgr else raw


#: Fraction of the requested rate below which the delivery is called short.
#: Not 1.0: a camera negotiating 29.97 against a requested 30 is fine, and a
#: measurement window always costs a little. 0.8 is far enough below the noise
#: to mean something and far enough above the 3x and 6x shortfalls that
#: prompted this to catch them with room to spare.
SHORTFALL_RATIO = 0.8

#: Formats that carry a high-resolution stream through USB. Anything else at
#: 720p or above is bandwidth-limited whatever the driver claims.
COMPRESSED_FORMATS = ("MJPG", "MJPEG", "H264", "H265")
@dataclass(frozen=True)
class Shortfall:
    """A camera is not delivering what it was asked for.

    Carries the numbers rather than a pre-baked sentence so a caller can log
    it, banner it, or put it in a session header without re-deriving anything.
    """

    camera_id: str
    requested_fps: float
    measured_fps: float
    requested_wh: Optional[tuple] = None
    actual_wh: Optional[tuple] = None
    pixel_format: str = ""
    backend: str = ""
    #: The SDK's own achievable-rate reading under the CURRENT exposure, ROI
    #: and link budget, ``AcquisitionResultingFrameRate`` on Spinnaker,
    #: ``XI_PRM_FRAMERATE`` on Ximea. Authoritative and available immediately,
    #: where a UVC camera has to be measured for ten seconds to learn the same
    #: thing. 0.0 when the camera has no such reading (every UVC device).
    capable_fps: float = 0.0
    #: The driver's own complaint at open time, if it made one. Carried
    #: separately because it is available IMMEDIATELY, whereas the measured
    #: rate is not: the background calibration treats a low first reading as
    #: "still settling" and retries twice at ~12 s each, so a confirmed
    #: shortfall arrives about fifty seconds after Connect, long after an
    #: operator would have started the session. The format complaint is the
    #: early half of the same truth.
    format_warning: str = ""

    @property
    def size_differs(self) -> bool:
        # A camera opening, closing, or briefly reconfiguring reports 0x0.
        # That is "not known yet", not a mismatch, treating it as one
        # banners a shutdown as a fault.
        if not (self.requested_wh and self.actual_wh):
            return False
        if int(self.actual_wh[0]) <= 0 or int(self.actual_wh[1]) <= 0:
            return False
        return tuple(self.requested_wh) != tuple(self.actual_wh)

    @property
    def rate_differs(self) -> bool:
        return bool(self.requested_fps > 0 and self.measured_fps > 0
                    and self.measured_fps < self.requested_fps * SHORTFALL_RATIO)

    @property
    def capability_short(self) -> bool:
        """The SDK says up front it cannot reach the requested rate.

        On a machine-vision camera this is knowable the moment it is
        configured, because the reading already accounts for exposure, ROI and
        link bandwidth. FLIR documents the usual cause plainly: if the
        resulting rate is below the requested one, the exposure time is longer
        than the frame time.
        """
        return bool(self.requested_fps > 0 and self.capable_fps > 0
                    and self.capable_fps < self.requested_fps * SHORTFALL_RATIO)

    @property
    def uncompressed_at_high_res(self) -> bool:
        """A likely cause, not a certainty, worth naming because it is the
        one the operator can act on by picking a different mode."""
        wh = self.actual_wh or self.requested_wh
        if not wh or not self.pixel_format:
            return False
        return (int(wh[0]) * int(wh[1]) > 640 * 480
                and self.pixel_format.upper() not in COMPRESSED_FORMATS)

    def message(self) -> str:
        """One line for the operator, naming the cause when it is knowable."""
        parts = []
        # Reported even with no rate measurement yet: the driver already knows
        # it could not give what was asked for, and saying so at Connect beats
        # saying it after the session has started.
        if self.format_warning and not (self.rate_differs or self.size_differs):
            return self.format_warning
        if self.size_differs:
            parts.append(
                f"asked for {self.requested_wh[0]}x{self.requested_wh[1]}, "
                f"getting {self.actual_wh[0]}x{self.actual_wh[1]}")
        if self.rate_differs:
            parts.append(
                f"asked for {self.requested_fps:.0f} fps, "
                f"getting {self.measured_fps:.1f}")
        elif self.capability_short:
            parts.append(
                f"asked for {self.requested_fps:.0f} fps; the camera reports "
                f"it can only reach {self.capable_fps:.1f} with the current "
                f"exposure, ROI and bandwidth settings")
        if not parts:
            return ""
        text = f"Camera {self.camera_id}: " + "; ".join(parts) + "."
        if self.uncompressed_at_high_res:
            text += (f" The stream is {self.pixel_format}, which is "
                     f"uncompressed, USB bandwidth caps the rate at this "
                     f"resolution. A compressed (MJPG) mode, a smaller "
                     f"resolution, or a different capture backend would lift "
                     f"it.")
        return text
