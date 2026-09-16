"""
FLIR/Spinnaker camera backend - wraps PySpin SDK to match GenericCamera interface.

Key features:
- 100-frame hardware FIFO buffer with non-blocking batch drain
- Chunk data timestamps (nanoseconds from camera clock)
- Frame drop detection via timestamp interval analysis
- External trigger support
- configure() reads AcquisitionResultingFrameRate to learn what the camera
  will actually deliver under current exposure/bandwidth/ROI constraints

Requires: PySpin SDK (FLIR Spinnaker SDK Python bindings)
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import (
    SDK_FPS_DRAIN_S,
    SDK_FPS_MEASURE_S,
    ResolvedCameraSettings,
    GenericCamera,
    camera_info,
    measure_delivered_fps,
    normalize_channels,
)

logger = logging.getLogger(__name__)

try:
    import PySpin
    SPINNAKER_AVAILABLE = True
except ImportError:
    try:
        import pyspin as PySpin
        SPINNAKER_AVAILABLE = True
    except ImportError:
        SPINNAKER_AVAILABLE = False
        PySpin = None

# Lazy singleton: only create the PySpin system when first camera is instantiated
_pyspin_system = None


def _get_pyspin_system():
    global _pyspin_system
    if _pyspin_system is None and SPINNAKER_AVAILABLE:
        _pyspin_system = PySpin.System.GetInstance()
    return _pyspin_system


class SpinnakerCamera(GenericCamera):
    #: This device stamps every frame itself, so the frame
    #: times in a session recorded from it are the device's.
    TIMESTAMP_SOURCE = "hw"

    """FLIR camera via PySpin/Spinnaker SDK.

    Args:
        unique_id: Camera identifier in "SERIAL-spinnaker" format.
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
        if not SPINNAKER_AVAILABLE:
            raise RuntimeError("PySpin SDK not available. Install FLIR Spinnaker SDK.")

        self.unique_id = unique_id
        # rsplit, not split: a vendor serial may contain a dash, and an
        # unpack of three parts raises where the backend suffix is all we want
        # to drop.
        self.serial_number = unique_id.rsplit("-", 1)[0]
        self.BUFFER_SIZE = 100

        # Structured trigger + strobe (TriggerConfig / LineOutputConfig .to_json()
        # dicts from the pipeline). A ``hardware`` trigger implies external
        # timing (skip the internal frame-rate request); the legacy
        # ``external_trigger`` bool maps to the same thing.
        self._trigger = dict(trigger or {})
        self._line_output = dict(line_output or {})
        self._external_trigger = external_trigger or (
            self._trigger.get("mode") == "hardware")
        self._grayscale = grayscale

        # Internal state
        self._frame_timestamp = None
        self._previous_frame_number = 0
        self._inter_frame_interval = int(1e9 // max(int(fps), 1))

        # Initialize camera
        system = _get_pyspin_system()
        cam_list = system.GetCameras()
        self._cam = None
        for cam in cam_list:
            if cam.TLDevice.DeviceSerialNumber.GetValue() == self.serial_number:
                self._cam = cam
                break
        cam_list.Clear()

        if self._cam is None:
            raise RuntimeError(f"Spinnaker camera {self.serial_number} not found")

        self._cam.Init()
        self.device_model = self._cam.TLDevice.DeviceModelName.GetValue()[:20]
        self._is_chameleon = self.device_model.startswith("Chameleon3")
        self._nodemap = self._cam.GetNodeMap()
        self._stream_nodemap = self._cam.GetTLStreamNodeMap()

        # Configure buffer handling: OldestFirst, 100 frames
        bh_node = PySpin.CEnumerationPtr(self._stream_nodemap.GetNode("StreamBufferHandlingMode"))
        bh_node.SetIntValue(bh_node.GetEntryByName("OldestFirst").GetValue())
        sbc_node = PySpin.CIntegerPtr(self._stream_nodemap.GetNode("StreamBufferCountManual"))
        sbc_node.SetValue(self.BUFFER_SIZE)

        # Configure chunk data (timestamp + frame ID)
        chunk_selector = PySpin.CEnumerationPtr(self._nodemap.GetNode("ChunkSelector"))
        if self._is_chameleon:
            chunk_selector.SetIntValue(chunk_selector.GetEntryByName("FrameCounter").GetValue())
        else:
            chunk_selector.SetIntValue(chunk_selector.GetEntryByName("FrameID").GetValue())
        self._cam.ChunkEnable.SetValue(True)

        chunk_selector.SetIntValue(chunk_selector.GetEntryByName("Timestamp").GetValue())
        self._cam.ChunkEnable.SetValue(True)
        self._cam.ChunkModeActive.SetValue(True)

        # Continuous acquisition mode
        acq_mode = PySpin.CEnumerationPtr(self._nodemap.GetNode("AcquisitionMode"))
        acq_mode.SetIntValue(acq_mode.GetEntryByName("Continuous").GetValue())

        # Manual exposure and gain (lock auto so configure() readings are stable)
        exc_node = PySpin.CEnumerationPtr(self._nodemap.GetNode("ExposureAuto"))
        exc_node.SetIntValue(PySpin.ExposureAuto_Off)
        gnc_node = PySpin.CEnumerationPtr(self._nodemap.GetNode("GainAuto"))
        gnc_node.SetIntValue(PySpin.GainAuto_Off)

        # Trigger + strobe/line output, all set BEFORE BeginAcquisition.
        self._apply_trigger(self._trigger)
        self._apply_line_output(self._line_output)

        # Resolve initial config (sets self.config)
        self.configure(fps=fps, exposure_us=exposure_us, gain_db=gain_db)

        logger.info(
            f"SpinnakerCamera initialized: {self.device_model} ({self.serial_number}) "
            f"target_fps={self.config.target_fps:.2f} capable_fps={self.capable_fps:.2f}"
        )

    # --- Configuration --------------------------------------------------------

    def configure(
        self,
        fps: Optional[float] = None,
        exposure_us: Optional[float] = None,
        gain_db: Optional[float] = None,
        roi: Optional[tuple[int, int, int, int]] = None,
    ) -> ResolvedCameraSettings:
        """Apply settings via the base template; refresh the drop-detection
        interval from the resolved target_fps. We never cap: the SDK is told
        what the user wants and ``self.measured_fps`` reports what the
        hardware actually delivered."""
        cfg = super().configure(fps=fps, exposure_us=exposure_us,
                                gain_db=gain_db, roi=roi)
        self._inter_frame_interval = int(1e9 // max(int(cfg.target_fps), 1))
        return cfg

    def _read_diagnostic_fps(self) -> float:
        """AcquisitionResultingFrameRate; 0.0 when unavailable.

        A capability (what the camera can deliver under current exposure/
        bandwidth/ROI), published via ``self.capable_fps``, never used to
        modify ``self.config.target_fps`` and never a stand-in for the
        calibration-loop ``measured_fps``.
        """
        try:
            node = PySpin.CFloatPtr(self._nodemap.GetNode("AcquisitionResultingFrameRate"))
            if node.IsValid() and PySpin.IsReadable(node):
                v = node.GetValue()
                if v > 0:
                    return float(v)
        except Exception as e:
            logger.debug(f"AcquisitionResultingFrameRate unavailable: {e}")
        return 0.0

    def _apply_frame_rate(self, fps: float) -> None:
        if self._is_chameleon:
            frc = PySpin.CBooleanPtr(self._nodemap.GetNode("AcquisitionFrameRateEnabled"))
            frc.SetValue(True)
            fra = PySpin.CEnumerationPtr(self._nodemap.GetNode("AcquisitionFrameRateAuto"))
            fra.SetIntValue(fra.GetEntryByName("Off").GetValue())
        else:
            try:
                fra = PySpin.CBooleanPtr(self._nodemap.GetNode("AcquisitionFrameRateEnable"))
                fra.SetValue(True)
            except PySpin.SpinnakerException:
                pass

        # Clamp to the node's current range (exposure/ROI may have shrunk it).
        node = PySpin.CFloatPtr(self._nodemap.GetNode("AcquisitionFrameRate"))
        try:
            lo, hi = node.GetMin(), node.GetMax()
            fps = max(lo, min(float(fps), hi))
        except Exception:
            pass
        node.SetValue(float(fps))

    def _apply_exposure(self, us: float) -> None:
        node = PySpin.CFloatPtr(self._nodemap.GetNode("ExposureTime"))
        try:
            lo, hi = node.GetMin(), node.GetMax()
            us = max(lo, min(float(us), hi))
        except Exception:
            pass
        node.SetValue(float(us))

    def _apply_gain(self, db: float) -> None:
        node = PySpin.CFloatPtr(self._nodemap.GetNode("Gain"))
        try:
            lo, hi = node.GetMin(), node.GetMax()
            db = max(lo, min(float(db), hi))
        except Exception:
            pass
        node.SetValue(float(db))

    def _apply_roi(self, roi: tuple[int, int, int, int]) -> None:
        x, y, w, h = roi
        # Order matters: width/height before offsets (max offsets depend on dims).
        try:
            PySpin.CIntegerPtr(self._nodemap.GetNode("Width")).SetValue(int(w))
            PySpin.CIntegerPtr(self._nodemap.GetNode("Height")).SetValue(int(h))
            PySpin.CIntegerPtr(self._nodemap.GetNode("OffsetX")).SetValue(int(x))
            PySpin.CIntegerPtr(self._nodemap.GetNode("OffsetY")).SetValue(int(y))
        except Exception as e:
            logger.warning(f"SpinnakerCamera ROI apply failed: {e}")

    # --- Frame dimensions ----------------------------------------------------

    def get_width(self) -> int:
        return PySpin.CIntegerPtr(self._nodemap.GetNode("Width")).GetValue()

    def get_height(self) -> int:
        return PySpin.CIntegerPtr(self._nodemap.GetNode("Height")).GetValue()

    # --- Acquisition mode ----------------------------------------------------

    # GenICam node setters, thin wrappers so the trigger/strobe recipes below
    # read as the FLIR app-note steps and are unit-testable without PySpin
    # (tests patch these three; the recipe logic is what's verified headless,
    # the actual node writes are rig-verified).

    def _set_enum(self, node: str, entry: str) -> None:
        n = PySpin.CEnumerationPtr(self._nodemap.GetNode(node))
        n.SetIntValue(n.GetEntryByName(entry).GetValue())

    def _set_bool(self, node: str, value: bool) -> None:
        PySpin.CBooleanPtr(self._nodemap.GetNode(node)).SetValue(bool(value))

    def _set_float(self, node: str, value: float) -> None:
        PySpin.CFloatPtr(self._nodemap.GetNode(node)).SetValue(float(value))

    #: pipeline line-source name → FLIR ``LineSource`` enum entry.
    _LINE_SOURCE = {
        "exposure_active": "ExposureActive",
        "frame_active": "FrameActive",
        "user_high": "UserOutput0",
        "user_low": "UserOutput0",
    }
    #: default strobe output line by family (BFS uses Line1; older CM3/FL3/GS3/
    #: FFY-DL/ORX use Line2). Overridden by the config's explicit ``line``.
    _DEFAULT_OUT_LINE = "Line1"

    def _apply_trigger(self, trig: dict) -> None:
        """Configure hardware/software/free-run triggering per the verified FLIR
        order: TriggerMode Off → params → On last. ``ReadOut`` overlap is set so
        a secondary sustains the primary's full frame rate."""
        mode = (trig or {}).get("mode", "freerun")
        # Always disable first so params are editable.
        self._set_enum("TriggerMode", "Off")
        if mode == "hardware":
            source = trig.get("source") or "Line3"    # BFS opto/GPIO input
            edge = "FallingEdge" if trig.get("edge") == "falling" else "RisingEdge"
            self._set_enum("TriggerSelector", "FrameStart")
            self._set_enum("TriggerSource", source)
            self._set_enum("TriggerActivation", edge)
            try:
                self._set_float("TriggerDelay", float(trig.get("delay_us", 0.0)))
            except Exception as e:
                logger.debug("TriggerDelay unsupported: %s", e)
            try:
                self._set_enum("TriggerOverlap", "ReadOut")  # sustain full rate
            except Exception as e:
                logger.debug("TriggerOverlap unsupported: %s", e)
            self._set_enum("TriggerMode", "On")
        elif mode == "software":
            self._set_enum("TriggerSelector", "FrameStart")
            self._set_enum("TriggerSource", "Software")
            self._set_enum("TriggerMode", "On")
        # freerun → leave TriggerMode Off. Only a hardware trigger may raise
        # the flag: a camera constructed with the legacy ``external_trigger=
        # True`` bool carries a default/freerun trigger dict yet is still
        # externally timed, so a non-hardware dict must not clobber the flag
        # back into an internal frame-rate request.
        if mode == "hardware":
            self._external_trigger = True

    def _apply_line_output(self, lo: dict) -> None:
        """Drive a digital output line from the exposure window (strobe / sync
        source). No-op unless enabled."""
        if not (lo or {}).get("enabled"):
            return
        line = lo.get("line") or self._DEFAULT_OUT_LINE
        source = self._LINE_SOURCE.get(lo.get("source", "exposure_active"),
                                       "ExposureActive")
        self._set_enum("LineSelector", line)
        self._set_enum("LineMode", "Output")
        self._set_enum("LineSource", source)
        try:
            self._set_bool("LineInverter", bool(lo.get("inverted", False)))
        except Exception as e:
            logger.debug("LineInverter unsupported: %s", e)
        # Non-isolated output lines (e.g. Line2) must source their own current;
        # opto-isolated lines ignore this. Best-effort, absent on some models.
        try:
            self._set_bool("V3_3Enable", True)
        except Exception as e:
            logger.debug("V3_3Enable unsupported on %s: %s", line, e)

    # --- Acquisition control -------------------------------------------------

    def begin_capturing(self) -> None:
        if not self._cam.IsInitialized():
            self._cam.Init()
        if not self._cam.IsStreaming():
            self._cam.BeginAcquisition()
        self._frame_timestamp = None
        self._previous_frame_number = 0
        logger.info(f"SpinnakerCamera {self.serial_number}: acquisition started")

    def stop_capturing(self) -> None:
        if self._cam.IsStreaming():
            self._cam.EndAcquisition()
        if self._cam.IsInitialized():
            self._cam.DeInit()
        logger.info(f"SpinnakerCamera {self.serial_number}: acquisition stopped")

    def get_available_images(self) -> dict | None:
        """Drain all buffered frames using non-blocking GetNextImage(0)."""
        img_buffer = []
        timestamps_buffer = []
        dropped_frames = 0

        try:
            while True:
                next_image = self._cam.GetNextImage(0)  # Non-blocking; throws if empty
                raw = next_image.GetNDArray()
                # copy_bgr: own the buffer (Release() below recycles it).
                img_buffer.append(
                    normalize_channels(raw, self._grayscale, copy_bgr=True))

                # Chunk timestamp (nanoseconds from camera clock)
                chunk_data = next_image.GetChunkData()
                ts = chunk_data.GetTimestamp()
                timestamps_buffer.append(ts)

                # Detect dropped frames via timestamp intervals
                if self._frame_timestamp is not None:
                    elapsed_frames = round(
                        (ts - self._frame_timestamp) / self._inter_frame_interval
                    )
                    dropped_frames += max(0, elapsed_frames - 1)
                self._frame_timestamp = ts

                next_image.Release()

        except PySpin.SpinnakerException:
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
    # rig-verify: WidthMax/HeightMax node reads + the short-acquisition FPS
    # count below are validated against SDK docs but not yet run on FLIR
    # hardware. Confirm node names on a real camera before trusting the
    # numbers.

    #: Common modes offered in addition to the sensor max, filtered to those
    #: that fit within WidthMax/HeightMax.
    _STANDARD_MODES = (
        (1920, 1200), (1920, 1080), (1440, 1080), (1280, 1024),
        (1280, 720), (1024, 768), (800, 600), (640, 480),
    )

    @classmethod
    def _open_serial(cls, unique_id: str):
        """Return an Init'd PySpin camera for ``unique_id``, or ``None``.

        Caller MUST DeInit + release the returned camera. Serial is the
        leading token of the ``SERIAL-spinnaker`` unique_id.
        """
        if not SPINNAKER_AVAILABLE:
            return None
        serial = str(unique_id).split("-")[0]
        system = _get_pyspin_system()
        cam_list = system.GetCameras()
        found = None
        try:
            for cam in cam_list:
                if cam.TLDevice.DeviceSerialNumber.GetValue() == serial:
                    found = cam
                    break
        except Exception as e:
            logger.debug(f"Spinnaker _open_serial enumerate failed: {e}")
        if found is not None:
            try:
                found.Init()
            except Exception as e:
                logger.debug(f"Spinnaker Init({serial}) failed: {e}")
                found = None
        cam_list.Clear()
        return found

    @classmethod
    def probe_supported_resolutions(cls, unique_id: str, report=None):
        """Best-effort ``([(w, h), ...], max_mode)`` from WidthMax/HeightMax.

        Returns ``([], None)`` when the SDK is missing or the camera can't
        be opened.
        """
        if not SPINNAKER_AVAILABLE:
            return ([], None)
        cam = cls._open_serial(unique_id)
        if cam is None:
            return ([], None)
        modes: list[tuple[int, int]] = []
        try:
            nodemap = cam.GetNodeMap()
            w_max = int(PySpin.CIntegerPtr(nodemap.GetNode("WidthMax")).GetValue())
            h_max = int(PySpin.CIntegerPtr(nodemap.GetNode("HeightMax")).GetValue())
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
            logger.debug(f"Spinnaker probe_supported_resolutions failed: {e}")
        finally:
            try:
                cam.DeInit()
            except Exception:
                pass
            del cam
        return (modes, modes[0] if modes else None)

    @classmethod
    def measure_fps_at(cls, unique_id: str, w: int, h: int,
                       target_fps: float = 30.0, report=None) -> float:
        """Best-effort realistic FPS at ``(w, h)``.

        Counts frames over a short BeginAcquisition/GetNextImage window.
        Returns ``0.0`` on failure, no AcquisitionResultingFrameRate
        fallback, because that node is a capability, not a measurement.
        """
        if not SPINNAKER_AVAILABLE:
            return 0.0
        cam = cls._open_serial(unique_id)
        if cam is None:
            return 0.0
        measured = 0.0
        acquiring = False
        try:
            nodemap = cam.GetNodeMap()
            try:
                PySpin.CIntegerPtr(nodemap.GetNode("Width")).SetValue(int(w))
                PySpin.CIntegerPtr(nodemap.GetNode("Height")).SetValue(int(h))
            except Exception:
                pass  # rig-verify: some models clamp ROI to steps.
            cam.BeginAcquisition()
            acquiring = True

            def _drain():
                cam.GetNextImage(200).Release()

            def _count():
                cam.GetNextImage(500).Release()
                return 1

            # Shared settle-then-count policy, one window across all
            # backends, so the rates the three produce are comparable.
            measured = measure_delivered_fps(
                _drain, _count,
                drain_s=SDK_FPS_DRAIN_S, measure_s=SDK_FPS_MEASURE_S)
        except Exception as e:
            logger.debug(f"Spinnaker measure_fps_at({w}x{h}) failed: {e}")
        finally:
            # Stop acquisition here (guarded on it having started, so a stop
            # on a never-started acquisition can't mask the original error),
            # DeInit on a still-streaming camera leaves it claimed.
            if acquiring:
                try:
                    cam.EndAcquisition()
                except Exception as stop_err:
                    logger.debug(
                        f"Spinnaker measure_fps_at EndAcquisition: {stop_err}")
            try:
                cam.DeInit()
            except Exception:
                pass
            del cam
        if callable(report):
            try:
                report("fps_measured", w=int(w), h=int(h), fps=measured)
            except Exception:
                pass
        return measured


# --- Module-level functions --------------------------------------------------

def list_available_cameras() -> list[dict]:
    """Detect all FLIR/Spinnaker cameras."""
    if not SPINNAKER_AVAILABLE:
        return []

    cameras = []
    try:
        system = _get_pyspin_system()
        cam_list = system.GetCameras()

        for cam in cam_list:
            try:
                serial = cam.TLDevice.DeviceSerialNumber.GetValue()
                model = cam.TLDevice.DeviceModelName.GetValue()
                cameras.append(camera_info("spinnaker", serial, model))
            except Exception as e:
                logger.debug(f"Error reading Spinnaker camera info: {e}")

        cam_list.Clear()
    except Exception as e:
        logger.debug(f"Error enumerating Spinnaker cameras: {e}")

    return cameras
