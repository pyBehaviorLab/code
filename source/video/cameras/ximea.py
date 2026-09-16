"""
Ximea camera backend - wraps xiapi SDK to match GenericCamera interface.

Adapted from pyMultiVideo's camera_api/ximea.py. Key features:
- 100-frame hardware buffer with non-blocking batch drain
- Frame drop detection via frame number tracking
- External trigger support
- configure() resolves the achievable framerate via XI_PRM_FRAMERATE
  with the XI_PRM_INFO_MAX modifier (after exposure / bandwidth are
  applied), no recording-time probe.

Family handling for the FPS report path:
- MQ / MD: XI_PRM_FRAMERATE reports the current rate directly.
- CB / MC / MT / MX: XI_PRM_FRAMERATE reports last-set, not actual; we
  query INFO_MAX for the achievable upper bound and trust it.

Requires: ximea Python package (Ximea SDK)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .base import (
    SDK_FPS_DRAIN_S,
    SDK_FPS_MEASURE_S,
    GenericCamera,
    camera_info,
    measure_delivered_fps,
    normalize_channels,
)

logger = logging.getLogger(__name__)

try:
    from ximea import xiapi
    XIMEA_AVAILABLE = True
except ImportError:
    XIMEA_AVAILABLE = False
    xiapi = None

#: Camera families whose XI_PRM_FRAMERATE reports current/achievable rate
#: directly (no need to query INFO_MAX). Other families need INFO_MAX.
_DIRECT_FAMILIES = ("MQ", "MD")


class XimeaCamera(GenericCamera):
    #: This device stamps every frame itself, so the frame
    #: times in a session recorded from it are the device's.
    TIMESTAMP_SOURCE = "hw"

    """Ximea camera via xiapi SDK.

    Args:
        unique_id: Camera identifier in "SERIAL-ximea" format.
        fps: Requested frame rate (initial, refine later via configure()).
        exposure_us: Exposure time in microseconds.
        gain_db: Gain in dB.
        external_trigger: Enable external trigger mode.
    """

    BUFFER_SIZE = 100

    def __init__(self, unique_id: str, fps: float = 30, exposure_us: float = 5000,
                 gain_db: float = 0, external_trigger: bool = False,
                 grayscale: bool = False,
                 trigger: Optional[dict] = None,
                 line_output: Optional[dict] = None):
        super().__init__()
        if not XIMEA_AVAILABLE:
            raise RuntimeError("ximea package not available. Install Ximea SDK.")

        self.unique_id = unique_id
        # rsplit, not split: a vendor serial may contain a dash, and an
        # unpack of three parts raises where the backend suffix is all we want
        # to drop.
        self.serial_number = unique_id.rsplit("-", 1)[0]
        self.BUFFER_SIZE = 100

        # Structured trigger + strobe (see SpinnakerCamera for the shape).
        self._trigger = dict(trigger or {})
        self._line_output = dict(line_output or {})
        self._external_trigger = external_trigger or (
            self._trigger.get("mode") == "hardware")
        self._grayscale = grayscale
        self._previous_frame_number = 0
        self._streaming = False

        # Open camera
        self._cam = xiapi.Camera()
        self._cam.open_device_by_SN(self.serial_number)
        self.device_model = str(self._cam.get_device_model_id())
        self._family = self._detect_family(self.device_model)

        # Manual control + framerate timing mode
        self._cam.disable_aeag()
        self._cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
        self._cam.set_acq_buffer_size(self.BUFFER_SIZE)

        # Resolve initial config (sets self.config)
        self.configure(fps=fps, exposure_us=exposure_us, gain_db=gain_db)

        logger.info(
            f"XimeaCamera initialized: {self.device_model} ({self.serial_number}) "
            f"family={self._family} target_fps={self.config.target_fps:.2f} "
            f"capable_fps={self.capable_fps:.2f}"
        )

    @staticmethod
    def _detect_family(model_id: str) -> str:
        """Return the 2-letter family prefix (MQ, MD, CB, MC, MT, MX, ...).

        Ximea model IDs look like "MQ022CG-CM" or "MC050CG-SY". The first
        two characters identify the family.
        """
        m = (model_id or "").upper().strip()
        return m[:2] if len(m) >= 2 else ""

    # --- Configuration --------------------------------------------------------
    # configure() itself is the base-class template; only the hooks below
    # are Ximea-specific.

    def _read_diagnostic_fps(self) -> float:
        """Family-aware achievable-rate read. Diagnostic only.

        MQ/MD: ``get_framerate()`` reports the current rate (post-exposure).
        CB/MC/MT/MX: ``XI_PRM_FRAMERATE`` with the ``XI_PRM_INFO_MAX`` modifier.
        Returns 0.0 when no reading can be made.
        """
        if self._family in _DIRECT_FAMILIES:
            try:
                v = float(self._cam.get_framerate())
                if v > 0:
                    return v
            except Exception as e:
                logger.debug(f"XIMEA get_framerate failed: {e}")

        # Try INFO_MAX modifier (works on all families per xiAPI docs).
        try:
            v = self._get_param_max("framerate")
            if v and v > 0:
                return float(v)
        except Exception as e:
            logger.debug(f"XIMEA framerate INFO_MAX failed: {e}")

        return 0.0

    def _get_param_max(self, name: str) -> Optional[float]:
        """Query XI_PRM_<name>:max via the xiapi private parameter modifier.

        The Python xiapi wrapper doesn't expose INFO_MAX directly on every
        binding version, so we go through ``get_param`` with the ":max"
        suffix that xiAPI accepts on the C-level parameter string.
        """
        try:
            return float(self._cam.get_param(f"{name}:max"))
        except Exception:
            try:
                # Some bindings expose dedicated maximum getters.
                getter = getattr(self._cam, f"get_{name}_maximum", None)
                if getter:
                    return float(getter())
            except Exception:
                pass
        return None

    def _apply_frame_rate(self, fps: float) -> None:
        try:
            self._cam.set_framerate(float(fps))
        except Exception as e:
            logger.debug(f"XimeaCamera set_framerate({fps}) failed: {e}")

    def _apply_exposure(self, us: float) -> None:
        try:
            self._cam.set_exposure(float(us))
        except Exception as e:
            logger.debug(f"XimeaCamera set_exposure({us}) failed: {e}")

    def _apply_gain(self, db: float) -> None:
        try:
            self._cam.set_gain(float(db))
        except Exception as e:
            logger.debug(f"XimeaCamera set_gain({db}) failed: {e}")

    def _apply_roi(self, roi: tuple[int, int, int, int]) -> None:
        x, y, w, h = roi
        try:
            self._cam.set_width(int(w))
            self._cam.set_height(int(h))
            self._cam.set_offsetX(int(x))
            self._cam.set_offsetY(int(y))
        except Exception as e:
            logger.warning(f"XimeaCamera ROI apply failed: {e}")

    # --- Frame dimensions ----------------------------------------------------

    def get_width(self) -> int:
        return self._cam.get_width()

    def get_height(self) -> int:
        return self._cam.get_height()

    # --- Acquisition mode ----------------------------------------------------

    def _apply_trigger(self, trig: dict) -> None:
        """Hardware/free-run triggering via the verified XI_* params. Stops
        acquisition to reconfigure if live, then resumes."""
        was_streaming = self._streaming
        if was_streaming:
            self._cam.stop_acquisition()

        mode = (trig or {}).get("mode", "freerun")
        if mode == "hardware":
            edge = ("XI_TRG_EDGE_FALLING" if trig.get("edge") == "falling"
                    else "XI_TRG_EDGE_RISING")
            try:
                self._cam.set_gpi_selector("XI_GPI_PORT1")
                self._cam.set_gpi_mode("XI_GPI_TRIGGER")
            except Exception as e:
                logger.debug("Ximea GPI selector/mode: %s", e)
            self._cam.set_trigger_source(edge)
            self._cam.set_trigger_selector("XI_TRG_SEL_FRAME_START")
            self._cam.set_acq_frame_burst_count(1)
            try:
                if trig.get("delay_us"):
                    self._cam.set_trigger_delay(int(trig["delay_us"]))
            except Exception as e:
                logger.debug("Ximea trigger delay unsupported: %s", e)
        else:
            self._cam.set_trigger_source("XI_TRG_OFF")

        if was_streaming:
            self._cam.start_acquisition()
        # Only a hardware trigger may raise the flag: a camera constructed
        # with the legacy ``external_trigger=True`` bool carries a default/
        # freerun trigger dict yet is still externally timed, so a
        # non-hardware dict must not clobber the flag back into an internal
        # frame-rate request (same guard as SpinnakerCamera).
        if mode == "hardware":
            self._external_trigger = True

    def _apply_line_output(self, lo: dict) -> None:
        """Drive a GPO high during exposure (strobe / sync source)."""
        if not (lo or {}).get("enabled"):
            return
        mode = ("XI_GPO_EXPOSURE_ACTIVE_NEG" if lo.get("inverted")
                else "XI_GPO_EXPOSURE_ACTIVE")
        try:
            self._cam.set_gpo_selector(lo.get("line") or "XI_GPO_PORT1")
            self._cam.set_gpo_mode(mode)
        except Exception as e:
            logger.debug("Ximea GPO strobe config failed: %s", e)

    # --- Acquisition control -------------------------------------------------

    def begin_capturing(self) -> None:
        if not self._cam.CAM_OPEN:
            self._cam.open_device_by_SN(self.serial_number)
            self._previous_frame_number = 0

        self._apply_trigger(self._trigger)
        self._apply_line_output(self._line_output)

        if not self._streaming:
            try:
                self._cam.start_acquisition()
                self._streaming = True
            except Exception as e:
                logger.error(f"XimeaCamera {self.serial_number}: start error: {e}")
                # Re-raise: the capture layer treats begin_capturing() raising
                # as the failed-connection signal, swallowing this left the
                # thread polling a non-streaming device forever.
                raise

        # Re-apply settings now that acquisition is live (Ximea requirement).
        if self.config is not None:
            try:
                self._apply_frame_rate(self.config.target_fps)
                if self.config.gain_db is not None:
                    self._apply_gain(self.config.gain_db)
                if self.config.exposure_us is not None:
                    self._apply_exposure(self.config.exposure_us)
            except Exception as e:
                logger.debug(f"XimeaCamera re-apply on start failed: {e}")

        logger.info(f"XimeaCamera {self.serial_number}: acquisition started")

    def stop_capturing(self) -> None:
        if self._streaming:
            self._cam.stop_acquisition()
            self._streaming = False
        if self._cam.CAM_OPEN:
            self._cam.close_device()
        logger.info(f"XimeaCamera {self.serial_number}: acquisition stopped")

    def get_available_images(self) -> dict | None:
        """Drain all buffered frames using non-blocking get_image(timeout=0)."""
        img_buffer = []
        timestamps_buffer = []
        dropped_frames = 0

        try:
            while True:
                next_image = xiapi.Image()
                self._cam.get_image(next_image, timeout=0)  # Throws if empty

                raw = np.frombuffer(next_image.get_image_data_raw(), dtype=np.uint8)

                h = next_image.height
                w = next_image.width
                if raw.size == h * w:
                    img_buffer.append(
                        normalize_channels(raw.reshape((h, w)), self._grayscale))
                elif raw.size == h * w * 3:
                    # copy_bgr: own the buffer (SDK recycles next_image).
                    img_buffer.append(normalize_channels(
                        raw.reshape((h, w, 3)), self._grayscale, copy_bgr=True))
                elif self._grayscale:
                    img_buffer.append(raw.reshape((h, w)).copy())
                else:
                    img_buffer.append(
                        cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_BayerRG2BGR))

                # Timestamp (convert to nanoseconds)
                ts_ns = next_image.tsSec * 1_000_000_000 + next_image.tsUSec * 1_000
                timestamps_buffer.append(ts_ns)

                # Frame drop detection
                if self._previous_frame_number != (next_image.acq_nframe - 1):
                    dropped_frames += next_image.acq_nframe - self._previous_frame_number - 1
                self._previous_frame_number = next_image.acq_nframe

        except Exception:
            pass

        if not img_buffer:
            return None

        return {
            "images": img_buffer,
            "timestamps": timestamps_buffer,
            "dropped_frames": dropped_frames,
            "timestamp_source": "hw",
        }

    # --- Calibration probe (best-effort; used by camera_connect) -------------
    #
    # rig-verify: width:max/height:max param reads + the short-acquisition
    # FPS count below are validated against xiAPI docs but not yet run on
    # Ximea hardware. Confirm parameter names on a real camera before
    # trusting the numbers.

    _STANDARD_MODES = (
        (1920, 1200), (1920, 1080), (1440, 1080), (1280, 1024),
        (1280, 720), (1024, 768), (800, 600), (640, 480),
    )

    @classmethod
    def _open_serial(cls, unique_id: str):
        """Return an opened xiapi.Camera for ``unique_id``, or ``None``.

        Caller MUST close_device() the returned camera.
        """
        if not XIMEA_AVAILABLE:
            return None
        serial = str(unique_id).split("-")[0]
        try:
            cam = xiapi.Camera()
            cam.open_device_by_SN(serial)
            return cam
        except Exception as e:
            logger.debug(f"Ximea _open_serial({serial}) failed: {e}")
            return None

    @classmethod
    def probe_supported_resolutions(cls, unique_id: str, report=None):
        """Best-effort ``([(w, h), ...], max_mode)`` from width:max/height:max.

        Returns ``([], None)`` when the SDK is missing or the camera can't be
        opened.
        """
        if not XIMEA_AVAILABLE:
            return ([], None)
        cam = cls._open_serial(unique_id)
        if cam is None:
            return ([], None)
        modes: list[tuple[int, int]] = []
        try:
            w_max = int(float(cam.get_param("width:max")))
            h_max = int(float(cam.get_param("height:max")))
            if w_max > 0 and h_max > 0:
                modes.append((w_max, h_max))
                for (w, h) in cls._STANDARD_MODES:
                    if w <= w_max and h <= h_max and (w, h) not in modes:
                        modes.append((w, h))
                # One emit per mode, INCLUDING the sensor-max appended first.
                if callable(report):
                    for (w, h) in modes:
                        try:
                            report("resolution_found", w=w, h=h)
                        except Exception:
                            pass
        except Exception as e:
            logger.debug(f"Ximea probe_supported_resolutions failed: {e}")
        finally:
            try:
                cam.close_device()
            except Exception:
                pass
        return (modes, modes[0] if modes else None)

    @classmethod
    def measure_fps_at(cls, unique_id: str, w: int, h: int,
                       target_fps: float = 30.0, report=None) -> float:
        """Best-effort realistic FPS at ``(w, h)``.

        Counts frames over a short start_acquisition/get_image window.
        Returns ``0.0`` on failure, no ``get_framerate()`` fallback, because
        on most families that reports the last-SET (target) rate, not a
        measurement (see the module header).
        """
        if not XIMEA_AVAILABLE:
            return 0.0
        cam = cls._open_serial(unique_id)
        if cam is None:
            return 0.0
        measured = 0.0
        acquiring = False
        try:
            try:
                cam.set_width(int(w))
                cam.set_height(int(h))
            except Exception:
                pass  # rig-verify: some models clamp ROI to steps.
            cam.start_acquisition()
            acquiring = True
            img = xiapi.Image()

            def _drain():
                cam.get_image(img, timeout=500)

            def _count():
                cam.get_image(img, timeout=500)
                return 1

            # Shared settle-then-count policy, one window across all
            # backends, so the rates the three produce are comparable.
            measured = measure_delivered_fps(
                _drain, _count,
                drain_s=SDK_FPS_DRAIN_S, measure_s=SDK_FPS_MEASURE_S)
        except Exception as e:
            logger.debug(f"Ximea measure_fps_at({w}x{h}) failed: {e}")
        finally:
            # Stop acquisition here (guarded on it having started, so a stop
            # on a never-started acquisition can't mask the original error),
            # closing a still-streaming device leaves it claimed.
            if acquiring:
                try:
                    cam.stop_acquisition()
                except Exception as stop_err:
                    logger.debug(
                        f"Ximea measure_fps_at stop_acquisition: {stop_err}")
            try:
                cam.close_device()
            except Exception:
                pass
        if callable(report):
            try:
                report("fps_measured", w=int(w), h=int(h), fps=measured)
            except Exception:
                pass
        return measured


# --- Module-level functions --------------------------------------------------

def list_available_cameras() -> list[dict]:
    """Detect all Ximea cameras."""
    if not XIMEA_AVAILABLE:
        return []

    cameras = []
    try:
        cam = xiapi.Camera()
        num_devices = cam.get_number_devices()

        for idx in range(num_devices):
            try:
                dev = xiapi.Camera(dev_id=idx)
                serial = dev.get_device_info_string("device_sn").decode("utf-8")
                model = str(dev.get_device_model_id())
                cameras.append(camera_info("ximea", serial, model))
                if dev.CAM_OPEN:
                    dev.close_device()
            except Exception as e:
                logger.debug(f"Error reading Ximea camera {idx}: {e}")
    except Exception as e:
        logger.debug(f"Error enumerating Ximea cameras: {e}")

    return cameras
