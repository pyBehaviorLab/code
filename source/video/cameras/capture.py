"""Video capture and management for behavioral experiments.

Manages camera capture, frame processing, and synchronized recording
across multiple boxes, with per-box ROI extraction for shared cameras.

Classes:
- CameraThread: Camera capture in a separate thread.
- VideoSegmentProcessor: Per-box ROI segmentation for a shared camera.
- VideoManager: Coordinates camera capture across boxes.
"""

import cv2
from datetime import datetime
import time
from source import host_clock
import threading
from collections import deque
from typing import Optional
import os

# Set ``PYBL_VIDEO_DEBUG=1`` before launch to surface per-batch
# calibration counts, recorder queue depth, and other inner-loop
# diagnostics. Off by default, zero perf cost when not set.
_VIDEO_DEBUG = bool(os.environ.get("PYBL_VIDEO_DEBUG"))
from source.video.cameras.base import Shortfall
from source.log import get_logger
from source.video.recording.drop_log import drop_log as _drop_log
from source.video.cameras import GenericCamera, CameraFactory
from source.video.cameras.opencv import OpenCVCamera
# CameraFrame imported lazily (bound once on first use) to avoid a circular
# import: source.video.framebus.types triggers source.video.framebus package
# init → controller → back here, while this module is still partially loaded.
_CameraFrame = None

# Optional Cython acceleration for batch BGR->Gray conversion using a
# fused kernel with ITU-R BT.601 luminance weights. Falls back to
# per-frame cv2.cvtColor() in the grayscale branch below.
_HAS_CYTHON_BATCH_GRAY = False
try:
    from source.cython._image_ops import batch_bgr_to_gray as _cy_batch_bgr_to_gray
    _HAS_CYTHON_BATCH_GRAY = True
except ImportError:
    pass

# Session clock reference: set once at first camera start so all cameras share
# the same monotonic → wall-clock mapping.
_session_start_host_ns: int = 0
_session_start_wall: float = 0.0  # time.time() at session start

logger = get_logger()


def _wall_time_at(mono_ns: int) -> datetime:
    """Map a host-monotonic timestamp onto the shared session wall clock.

    All cameras share one monotonic → wall anchor (set at first camera
    start), so batch wall stamps stay consistent across cameras without a
    ``datetime.now()`` syscall per drained batch.
    """
    global _session_start_host_ns, _session_start_wall
    if _session_start_host_ns == 0:
        _session_start_host_ns = host_clock.host_ns()
        _session_start_wall = time.time()
    return datetime.fromtimestamp(
        _session_start_wall + (mono_ns - _session_start_host_ns) / 1e9)


class CameraThread(threading.Thread):
    """Thread for handling camera capture via GenericCamera batch drain.

    Uses GenericCamera.get_available_images() for batch frame acquisition,
    which supports both USB webcams (1-5 frames/batch) and scientific cameras
    (up to 100 frames/batch from hardware FIFO).

    Timestamps come from ``source.host_clock``, which every other instant they
    are compared against also uses, the recorder's start, the sinks, and the
    ``pycboard`` anchor that maps them onto MCU framework time. Not
    ``time.monotonic_ns()``: on Windows that is ``GetTickCount64()`` at
    **15.625 ms**, coarser than a third of a frame interval at 20 fps, which
    quantises ``frame_ts_ms`` and every latency figure to a tick.

    A frame version counter supports latest-frame-wins polling: callers pass
    their last-seen version and get None back when the frame hasn't changed.
    """

    def __init__(self, camera_id, width=1280, height=720, target_fps=30,
                 camera: GenericCamera = None,
                 camera_config: dict = None):
        """Initialize camera thread.

        Args:
            camera_id: Camera identifier (int for OpenCV, or unique_id string).
            width: Requested capture width (OpenCV only).
            height: Requested capture height (OpenCV only).
            target_fps: Target FPS for capture.
            camera: Pre-created GenericCamera instance (overrides camera_id).
            camera_config: Config dict for CameraFactory.create_camera().
        """
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.running = True
        self.connected = False
        self.connection_checked = threading.Event()
        self.lock = threading.Lock()
        self._pending_fps = None  # Deferred FPS change (applied in run loop)
        # Deferred live feature writes, {key: value}, drained by the run loop
        # so the GUI thread never touches the SDK. Same contract as _pending_fps.
        self._pending_features: dict = {}
        # Backend id for registry feature calls. Derived from the unique_id
        # suffix when one was given, else the plain-index OpenCV case.
        self._feature_backend = (
            str(camera_id).rsplit("-", 1)[-1]
            if isinstance(camera_id, str) and "-" in camera_id else "opencv")
        self.last_frame = None
        self.last_timestamp_ns = 0
        self.last_version = 0
        self.last_frame_lock = threading.Lock()

        # Camera instance (created in run() if not provided)
        self._camera = camera
        self._camera_config = camera_config or {}

        # Grayscale mode: convert frames to grayscale at capture time.
        # Default False, the display uses Qt BGR888, so no conversion needed.
        self.grayscale = self._camera_config.get("grayscale", False)

        # Geometry correction, applied here at capture so that every consumer
        #, ROI crop, tracker, recorder, zones, MCU coordinates, describes
        # the same picture. Most webcams mirror their output, which is why the
        # operator's left shows up on the right.
        self.flip_horizontal = bool(
            self._camera_config.get("flip_horizontal", False))
        self.flip_vertical = bool(
            self._camera_config.get("flip_vertical", False))

        # Capture-side FPS tracking: (timestamp_ns, cumulative_frame_count) pairs
        self._capture_samples = deque(maxlen=30)
        self._total_captured = 0

        # Background FPS calibration state. Phases:
        #   "drain"   → first ``CALIB_PRE_DRAIN_S`` of streaming, no count
        #   "measure" → next ``CALIB_MEASURE_S`` of streaming, count batches
        #   "done"    → final result applied to camera.config via
        #               camera.apply_measured_fps(...)
        # Purely passive: counts ``len(batch["images"])`` per run-loop
        # iteration, so frames keep flowing to the display sink throughout.
        # No extra ``cap.grab`` calls.
        self.CALIB_PRE_DRAIN_S = 2.0
        self.CALIB_MEASURE_S = 10.0
        # If the first measurement is implausibly low (camera still settling
        # auto-exposure / cold-start ramp), restart the drain+measure window.
        # Caps at CALIB_MAX_RETRIES retries before accepting the result so a
        # genuinely slow camera doesn't loop forever.
        self.CALIB_LOW_THRESHOLD_FPS = 18.0
        self.CALIB_MAX_RETRIES = 2
        self._calib_phase = "idle"
        self._calib_phase_start_ns = 0
        self._calib_count = 0
        self._calib_retries = 0

        # Per-camera monotonic frame ID counter, never resets, tags each
        # captured frame with a globally unique identity for the
        # CameraFrame -> BoxFrame join used by the strict pose-pairing
        # pipeline (framebus/types.py).
        self._cam_frame_counter = 0
        # Last emitted capture_host_ns, enforces a strictly increasing
        # capture clock across drain batches (see _buffer_camera_frames).
        self._last_capture_ns = 0
        # Bounded buffer of CameraFrame instances drained by the Pipeline
        # tick. Sized for ~10 s at 30 FPS so a GUI hiccup doesn't drop
        # frames silently.
        self._recording_buffer: deque = deque(maxlen=300)

        # ── Liveness contract ────────────────────────────────────────────
        # ``connected`` is the device-open lifecycle flag (set once the
        # backend opens, cleared on hard failure). ``_liveness`` is the
        # live frame-FLOW state, driven by frame age, so a brief unplug is
        # visible and recovered from instead of silently freezing on the
        # last frame. States: starting → streaming → stalled → reconnecting
        # → (streaming on recovery). The Pipeline watchdog reads ``_liveness``
        # each tick and fans a per-box health banner on any transition.
        self._liveness = "starting"
        self._streaming = False           # frames currently flowing
        self._last_good_ns = 0            # monotonic ns of the last real frame
        self._stall_after_s = 0.75       # this long with no frame ⇒ STALLED
        # OpenCV auto-reconnect backoff: once stalled, retry begin_capturing
        # every _reconnect_interval_s, FOREVER, so a camera that comes back
        # recovers instead of the thread dying. Scientific-camera backends
        # surface the stall but don't auto-reconnect.
        self._reconnect_interval_s = 2.0
        self._next_reconnect_ns = 0
        self._reconnect_attempts = 0

        # Track camera backend for capability queries
        self._camera_backend = "opencv"
        if camera:
            self._camera_backend = camera.unique_id.rsplit("-", 1)[-1] if "-" in camera.unique_id else "opencv"

        logger.debug(f"Initializing camera thread for camera {camera_id} "
                     f"(backend={self._camera_backend}) with target FPS: {target_fps}")

    def run(self):
        """Main camera capture loop, orchestrator only.

        Phases: create camera → begin capturing → main poll loop →
        stop capturing. Per-iteration work delegated to ``_process_batch``
        / ``_handle_empty_batch``.

        Wrapped in a broad except so any unhandled exception inside the
        worker thread is logged and the connection_checked event is set
        (otherwise the GUI thread waits forever for an event that never
        fires).
        """
        logger.info(
            f"CameraThread.run(): camera_id={self.camera_id} "
            f"width={self.width} height={self.height} "
            f"target_fps={self.target_fps} entering"
        )
        try:
            if not self._initialize_camera():
                logger.warning(
                    f"CameraThread.run(): camera_id={self.camera_id} "
                    f"_initialize_camera returned False, exiting"
                )
                return
            if not self._begin_camera_capture():
                logger.warning(
                    f"CameraThread.run(): camera_id={self.camera_id} "
                    f"_begin_camera_capture returned False, exiting"
                )
                return
            self._mark_connected()
            self._reset_calibration_state()

            while self.running:
                self._apply_pending_fps()
                self._apply_pending_features()
                batch = self._safe_get_batch()
                if batch and batch.get("images"):
                    # A per-frame processing error (cv2 conversion, a malformed
                    # frame) must NOT kill the capture thread, drop this batch
                    # and keep streaming. Only a truly unrecoverable error (the
                    # outer BaseException) ends the thread + flags the camera.
                    try:
                        self._process_batch(batch)
                    except Exception as e:
                        self._process_err_count = (
                            getattr(self, "_process_err_count", 0) + 1)
                        if (self._process_err_count <= 3
                                or self._process_err_count % 100 == 0):
                            logger.error(
                                "CameraThread(%s): _process_batch error #%d: %s",
                                self.camera_id, self._process_err_count, e)
                else:
                    # Stall detection + reconnect live in the liveness contract;
                    # the loop exits only when self.running goes False.
                    self._handle_empty_batch()

            self._stop_camera_safely()
        except BaseException as e:
            # BaseException catches everything including KeyboardInterrupt
            # and SystemExit so the GUI thread isn't left hanging on
            # connection_checked.wait(timeout=8.0).
            logger.exception(
                f"CameraThread.run(): camera_id={self.camera_id} "
                f"UNCAUGHT exception type={type(e).__name__}: {e}"
            )
            try:
                self._mark_failed_connection()
            except Exception:
                pass
            try:
                self._stop_camera_safely()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # run() helpers
    # ------------------------------------------------------------------
    def _initialize_camera(self) -> bool:
        """Create ``self._camera`` if not already provided. Returns False
        and emits a failed connection signal on construction failure."""
        if self._camera is not None:
            return True
        capture_format = (self._camera_config or {}).get("capture_format")
        # The chosen OS capture backend, by name. Both direct-construction
        # branches below bypass CameraFactory, so the translation has to
        # happen here too or a pinned backend silently reverts to auto.
        from source.video.cameras.opencv import cv_backend_for
        cv_backend = cv_backend_for(
            (self._camera_config or {}).get("capture_backend"))
        try:
            if (isinstance(self.camera_id, int)
                    or (isinstance(self.camera_id, str)
                        and self.camera_id.isdigit())):
                # integer camera_id -> OpenCVCamera
                cam_id = (int(self.camera_id)
                          if isinstance(self.camera_id, str)
                          else self.camera_id)
                self._camera = OpenCVCamera(
                    camera_id=cam_id, width=self.width,
                    height=self.height, fps=self.target_fps,
                    capture_format=capture_format,
                    cv_backend=cv_backend,
                )
            elif isinstance(self.camera_id, str) and "-" in self.camera_id:
                # unique_id format: "ID-backend"
                self._camera = CameraFactory.create_camera(
                    self.camera_id,
                    {**self._camera_config, "width": self.width,
                     "height": self.height, "fps": self.target_fps},
                )
            else:
                self._camera = OpenCVCamera(
                    camera_id=self.camera_id, width=self.width,
                    height=self.height, fps=self.target_fps,
                    capture_format=capture_format,
                    cv_backend=cv_backend,
                )
            self._camera_backend = (
                self._camera.unique_id.rsplit("-", 1)[-1]
                if "-" in self._camera.unique_id else "opencv"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create camera {self.camera_id}: {e}")
            self._mark_failed_connection()
            return False

    def _begin_camera_capture(self) -> bool:
        """Call ``begin_capturing`` on the backend. Returns False and
        emits a failed connection signal on failure."""
        try:
            self._camera.begin_capturing()
            return True
        except Exception as e:
            logger.error(f"Camera {self.camera_id} failed to start: {e}")
            self._mark_failed_connection()
            return False

    def _mark_failed_connection(self) -> None:
        self.connected = False
        self.connection_checked.set()

    def _mark_connected(self) -> None:
        self.connected = True
        self.connection_checked.set()
        logger.info(
            f"Camera {self.camera_id} started successfully "
            f"(backend={self._camera_backend})")

    def _reset_calibration_state(self) -> None:
        """Restart the background FPS calibration drain phase. Called at
        startup and whenever the camera is reconfigured."""
        self._calib_phase = "drain"
        self._calib_phase_start_ns = host_clock.host_ns()
        self._calib_count = 0

    def _apply_pending_fps(self) -> None:
        """Apply a deferred FPS change queued via ``set_target_fps``.

        Restarts the background calibration so the new requested rate gets
        a fresh measurement while frames keep flowing, no synchronous probe
        that would freeze the run loop and starve the display.
        """
        with self.lock:
            pending = self._pending_fps
            self._pending_fps = None
        if pending is None or not self._camera:
            return
        try:
            cfg = self._camera.configure(fps=pending)
            self._reset_calibration_state()
            self._calib_retries = 0
            logger.info(
                "Camera %s reconfigured: target_fps=%.2f",
                self.camera_id, cfg.target_fps,
            )
        except Exception as e:
            logger.debug(f"Camera {self.camera_id} configure error: {e}")

    def _safe_get_batch(self):
        """Call ``get_available_images`` with exception protection."""
        try:
            return self._camera.get_available_images()
        except Exception as e:
            logger.error(
                f"Camera {self.camera_id} get_available_images error: {e}")
            return None

    def _set_liveness(self, state: str) -> None:
        """Transition the frame-flow state and log the edge. Idempotent."""
        if state == self._liveness:
            return
        prev = self._liveness
        self._liveness = state
        if state == "streaming":
            self._streaming = True
            self._reconnect_attempts = 0
            if prev in ("stalled", "reconnecting"):
                logger.info("Camera %s recovered, frames flowing again",
                            self.camera_id)
        else:
            self._streaming = False
            if prev == "streaming":
                logger.warning("Camera %s stopped delivering frames (%s)",
                               self.camera_id, state)

    def _process_batch(self, batch) -> None:
        """Handle one non-empty batch from the camera: convert grayscale
        if needed, advance calibration, buffer CameraFrames + publish the
        latest frame, emit signals. One ``host_clock.host_ns()`` read is
        threaded through the per-batch helpers."""
        now_ns = host_clock.host_ns()
        # Frames are flowing, refresh liveness (recovers from a stall).
        self._last_good_ns = now_ns
        if self._liveness != "streaming":
            self._set_liveness("streaming")
        images = batch["images"]
        timestamps = batch["timestamps"]

        # Geometry before format: flip first so everything downstream, the
        # ROI crop, the tracker, the recorder, the zone tests and the
        # coordinates pushed to the MCU, sees one consistent picture.
        if self.flip_horizontal or self.flip_vertical:
            images = self._flip_batch(images)
            batch["images"] = images

        if self.grayscale:
            images = self._convert_batch_to_grayscale(images)
            batch["images"] = images

        self._update_capture_counters(len(images), now_ns)
        self._calibration_tick(len(images), now_ns)
        # Buffers CameraFrames AND publishes the latest frame (post
        # grayscale conversion) in one last_frame_lock critical section.
        self._buffer_camera_frames(images, timestamps, now_ns)

    def _flip_batch(self, images):
        """Mirror / invert every frame in the batch.

        ``cv2.flip`` codes: 1 = horizontal (mirror), 0 = vertical, -1 = both.
        It returns a fresh array, which is what the rest of the capture path
        expects, backends hand out newly allocated frames per batch and
        downstream stores the reference without copying, so flipping in place
        would corrupt a frame another thread is already reading.
        """
        code = (-1 if (self.flip_horizontal and self.flip_vertical)
                else 1 if self.flip_horizontal else 0)
        out = []
        for f in images:
            try:
                out.append(cv2.flip(f, code))
            except Exception as e:
                logger.debug("Camera %s flip failed: %s", self.camera_id, e)
                out.append(f)
        return out

    def _convert_batch_to_grayscale(self, images):
        """True single-channel (H×W) conversion; no BGR round-trip."""
        if _HAS_CYTHON_BATCH_GRAY:
            return _cy_batch_bgr_to_gray(images)
        converted = []
        for f in images:
            if f.ndim == 3 and f.shape[2] == 3:
                converted.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            else:
                converted.append(f)  # Already grayscale
        return converted

    def _update_capture_counters(self, n_new_frames: int, now_ns: int) -> None:
        self._total_captured += n_new_frames
        self._capture_samples.append((now_ns, self._total_captured))

    def _calibration_tick(self, n_new_frames: int, now_ns: int) -> None:
        """One step of the background FPS calibration state machine.

        Passive, counts batches already produced by
        ``get_available_images``, no extra camera I/O. Phases:
        ``drain`` → ``measure`` → ``done``.
        """
        if self._calib_phase in ("done", "idle"):
            return
        elapsed_ns = now_ns - self._calib_phase_start_ns
        if self._calib_phase == "drain":
            if elapsed_ns >= int(self.CALIB_PRE_DRAIN_S * 1e9):
                self._calib_phase = "measure"
                self._calib_phase_start_ns = now_ns
                self._calib_count = 0
            return
        if self._calib_phase == "measure":
            self._calib_count += n_new_frames
            if _VIDEO_DEBUG:
                logger.info(
                    "Camera %s calib batch: +%d (total %d, "
                    "elapsed %.2fs/%.1fs, retries=%d)",
                    self.camera_id, n_new_frames, self._calib_count,
                    elapsed_ns / 1e9, self.CALIB_MEASURE_S,
                    self._calib_retries,
                )
            if elapsed_ns >= int(self.CALIB_MEASURE_S * 1e9):
                self._finalize_calibration_measurement(now_ns, elapsed_ns)

    def _finalize_calibration_measurement(self, now_ns: int,
                                          elapsed_ns: int) -> None:
        """Apply the measurement: discard implausible / retry low / accept.

        Sanity bound: a healthy USB / vendor camera never delivers
        >500 fps. A count that high means the camera streamed garbage
        (e.g. post-reconfigure flood from the cap buffer); discard the
        reading so the config keeps the previous value.
        """
        measured_fps = self._calib_count / (elapsed_ns / 1e9)
        if measured_fps > 500.0:
            logger.warning(
                "Camera %s background calibration: implausibly high rate "
                "%.0f fps (%d frames over %.2fs), discarding and retrying",
                self.camera_id, measured_fps,
                self._calib_count, elapsed_ns / 1e9,
            )
            self._restart_calibration_drain(now_ns)
            return

        too_low_and_retryable = (
            measured_fps < self.CALIB_LOW_THRESHOLD_FPS
            and self._calib_retries < self.CALIB_MAX_RETRIES
            and self.target_fps
            and float(self.target_fps) >= self.CALIB_LOW_THRESHOLD_FPS
        )
        if too_low_and_retryable:
            self._calib_retries += 1
            logger.warning(
                "Camera %s background calibration: %.2f fps measured but "
                "%.1f fps requested, camera may still be settling. "
                "Retry %d/%d (drain %.0fs + measure %.0fs)",
                self.camera_id, measured_fps, float(self.target_fps),
                self._calib_retries, self.CALIB_MAX_RETRIES,
                self.CALIB_PRE_DRAIN_S, self.CALIB_MEASURE_S,
            )
            self._restart_calibration_drain(now_ns)
            return

        try:
            self._camera.apply_measured_fps(measured_fps)
        except Exception as e:
            logger.debug(
                "Camera %s apply_measured_fps failed: %s",
                self.camera_id, e)
        self._calib_phase = "done"
        logger.info(
            "Camera %s background calibration: %d frames over %.2fs → "
            "%.2f fps (retries %d/%d)",
            self.camera_id, self._calib_count, elapsed_ns / 1e9,
            measured_fps, self._calib_retries, self.CALIB_MAX_RETRIES,
        )

    def _restart_calibration_drain(self, now_ns: int) -> None:
        self._calib_phase = "drain"
        self._calib_phase_start_ns = now_ns
        self._calib_count = 0

    def _buffer_camera_frames(self, images, timestamps, now_ns: int) -> None:
        """Build ``CameraFrame`` entries for the pose-pairing pipeline, push
        them to ``self._recording_buffer``, and publish the batch's latest
        frame for display, one ``last_frame_lock`` acquisition per batch.
        Each accepted frame gets a monotonic ``cam_frame_id``; downstream
        BoxFrame derivation joins on this id. The image reference is stored
        as-is, camera backends allocate fresh numpy arrays per frame so no
        copy is needed.

        Records a ``camera_ring`` drop if the bounded deque would evict
        older frames silently (controller tick fell behind).
        """
        global _CameraFrame
        if _CameraFrame is None:
            from source.video.framebus.types import CameraFrame
            _CameraFrame = CameraFrame
        # Wall stamp derived from the shared session anchor, no
        # datetime.now() syscall per batch.
        wall_now = _wall_time_at(now_ns)
        first_cf = None
        n_new = 0
        with self.last_frame_lock:
            buffer = self._recording_buffer
            free = buffer.maxlen - len(buffer)
            for img, ts in zip(images, timestamps):
                # Enforce a strictly increasing capture clock. OpenCV back-stamps
                # each drain batch on an idealized period grid anchored at the
                # drain instant; because consecutive drains aren't spaced exactly
                # n*period apart, the oldest frame of a new batch can land BEFORE
                # the newest of the previous one. cam_frame_id is gap-free and
                # ordered, so clamp capture_host_ns to agree with it. This is the
                # single source of monotonicity for the elapsed column AND (via
                # pycboard.fw_ms_at, which is affine in capture_host_ns) the
                # frame_fw_ms column.
                ts = int(ts)
                if ts <= self._last_capture_ns:
                    ts = self._last_capture_ns + 1
                self._last_capture_ns = ts
                cf = _CameraFrame(
                    image=img,
                    cam_frame_id=self._cam_frame_counter,
                    capture_host_ns=ts,
                    capture_wall=wall_now,
                    camera_id=self.camera_id,
                    is_shared=False,   # set by controller per box_camera_map
                    box_ids=(),        # filled in at controller layer
                )
                self._cam_frame_counter += 1
                if first_cf is None:
                    first_cf = cf
                buffer.append(cf)
                n_new += 1
            # Latest-frame publish shares this critical section (one lock
            # acquisition per batch).
            self.last_frame = images[-1]
            self.last_timestamp_ns = timestamps[-1]
            self.last_version += 1
        overflow = max(0, n_new - free)
        if overflow:
            _drop_log.record(
                "camera_ring",
                frame_idx=first_cf.cam_frame_id,
                capture_ts=first_cf.capture_host_ns / 1e9,
                reason=f"ring_full_cam{self.camera_id}_n{overflow}",
            )

    def _handle_empty_batch(self) -> None:
        """Handle a poll that returned no frames.

        A brief gap is normal; once the last real frame is older than
        ``_stall_after_s`` the camera is marked STALLED (the Pipeline watchdog
        then banners every box on it). For OpenCV we retry ``reconnect()`` on a
        backoff schedule, forever, so a camera that returns recovers on the
        next good batch. Frame loss never kills the thread; the run loop exits
        only when ``self.running`` goes False.
        """
        now_ns = host_clock.host_ns()
        age_s = (now_ns - self._last_good_ns) / 1e9 if self._last_good_ns else 0.0

        # Mark the stall as soon as frames stop.
        if self._streaming and self._last_good_ns and age_s >= self._stall_after_s:
            self._set_liveness("stalled")
            self._next_reconnect_ns = now_ns   # try to recover promptly

        # OpenCV auto-reconnect on a backoff schedule while stalled.
        # Not after stop has been asked for. ``reconnect()`` REOPENS the
        # device, which takes seconds in DirectShow and does not check
        # ``running`` while it is inside the driver. Starting one here while
        # the thread is being torn down is how a disconnect left a daemon
        # thread inside a driver call: the process then exited past it, the
        # handle was never released, and the camera stayed claimed until the
        # USB device was replugged.
        if (self.running
                and self._liveness in ("stalled", "reconnecting")
                and isinstance(self._camera, OpenCVCamera)
                and now_ns >= self._next_reconnect_ns):
            self._set_liveness("reconnecting")
            self._reconnect_attempts += 1
            logger.warning("Camera %s: reconnect attempt #%d…",
                           self.camera_id, self._reconnect_attempts)
            ok = False
            try:
                ok = bool(self._camera.reconnect())
            except Exception as e:
                logger.debug("Camera %s reconnect raised: %s", self.camera_id, e)
            if ok:
                # liveness flips to "streaming" on the next real batch
                logger.info("Camera %s device reopened, awaiting frames",
                            self.camera_id)
            else:
                self._next_reconnect_ns = (
                    now_ns + int(self._reconnect_interval_s * 1e9))
        # 10 ms backoff on empty batch, frame intervals are 16–33 ms for
        # 30–60 fps cameras, so this keeps frame-arrival latency negligible
        # while cutting wakeups. Drivers buffer frames during the sleep.
        time.sleep(0.010)

    def _stop_camera_safely(self) -> None:
        try:
            self._camera.stop_capturing()
        except Exception as e:
            logger.debug(f"Camera {self.camera_id} stop error: {e}")
        logger.debug(f"Camera {self.camera_id} thread stopped")

    def queue_feature(self, key: str, value) -> None:
        """Queue a live feature write, applied on the next run-loop pass.

        The GUI thread must never call into a camera SDK directly, so writes
        land in ``_pending_features`` and the capture thread drains them,
        exactly the contract ``set_target_fps`` already uses.
        """
        with self.lock:
            self._pending_features[str(key)] = value

    def _apply_pending_features(self) -> None:
        """Push queued feature writes to the camera via its backend."""
        with self.lock:
            if not self._pending_features:
                return
            pending = dict(self._pending_features)
            self._pending_features.clear()
        if not self._camera:
            return
        from . import registry
        for key, value in pending.items():
            if registry.set_feature(self._feature_backend, self._camera,
                                    key, value):
                logger.debug("Camera %s feature %s -> %r",
                             self.camera_id, key, value)
            else:
                logger.info("Camera %s could not apply %s=%r live",
                            self.camera_id, key, value)

    def describe_features(self) -> list:
        """Feature descriptors for the open camera; empty when unsupported."""
        if not self._camera:
            return []
        from . import registry
        return registry.describe_features(self._feature_backend, self._camera)

    def set_target_fps(self, fps):
        """Update target FPS dynamically.

        Queues a configure() call onto the camera's run loop (thread-safe).
        Returns ``None`` (the resolved ResolvedCameraSettings is available via
        ``get_target_fps`` / ``camera.config`` after the run loop applies
        it on the next iteration).
        """
        with self.lock:
            self.target_fps = fps
            self._pending_fps = fps
            logger.debug(f"Camera {self.camera_id} target FPS queued to {fps}")
        return None

    def get_target_fps(self) -> float:
        """target_fps is the single source of truth for capture rate."""
        if self._camera is not None and self._camera.config is not None:
            return self._camera.config.target_fps
        return float(self.target_fps)

    def get_measured_fps(self) -> float:
        """Observation only: the camera's most recent measured rate."""
        if self._camera is not None:
            return float(getattr(self._camera, "measured_fps", 0.0))
        return 0.0

    def shortfall(self, camera_id=None) -> "Optional[Shortfall]":
        """Compare what a running camera delivers against what was asked.

        Returns ``None`` while there is nothing to say, including before the
        background rate measurement has settled, because an unmeasured camera is
        not a slow one and warning early would train the operator to ignore this.

        Reads only; safe to call from the pipeline tick.
        """
        try:
            requested_fps = float(self.get_target_fps() or 0.0)
            measured_fps = float(self.get_measured_fps() or 0.0)
        except Exception:
            return None
        cam = self._camera
        fmt = ""
        actual_wh = None
        try:
            reader = getattr(cam, "_read_fourcc_str", None)
            if callable(reader):
                fmt = reader() or ""
        except Exception:
            fmt = ""
        try:
            actual_wh = (int(cam.get_width()), int(cam.get_height()))
        except Exception:
            actual_wh = None
        requested_wh = None
        try:
            rw = getattr(cam, "_requested_width", None)
            rh = getattr(cam, "_requested_height", None)
            if rw and rh:
                requested_wh = (int(rw), int(rh))
        except Exception:
            requested_wh = None

        # The SDK's own answer, where there is one. Read from the camera object
        # rather than the thread: it is refreshed on every configure(), so it
        # tracks exposure and ROI changes instead of going stale.
        capable = 0.0
        try:
            capable = float(getattr(cam, "capable_fps", 0.0) or 0.0)
        except (TypeError, ValueError):
            capable = 0.0

        fmt_warning = ""
        try:
            fmt_warning = str(getattr(cam, "format_warning", "") or "")
        except Exception:
            fmt_warning = ""

        short = Shortfall(
            camera_id=str(camera_id if camera_id is not None
                          else self.camera_id),
            requested_fps=requested_fps,
            measured_fps=measured_fps,
            requested_wh=requested_wh,
            actual_wh=actual_wh,
            pixel_format=fmt,
            backend=str(getattr(cam, "_open_backend", "") or ""),
            format_warning=fmt_warning,
            capable_fps=capable,
        )
        # A rate of zero means the background calibration has not settled; the
        # rate comparison is meaningless then, but the size and the driver's own
        # complaint are already known and worth saying.
        if (measured_fps <= 0.0 and not short.size_differs
                and not fmt_warning and not short.capability_short):
            return None
        return short if short.message() else None

    def get_capture_fps(self) -> float:
        """Live measured fps from the rolling ``_capture_samples`` window.

        Returns the realised delivery rate over the most recent samples
        (deque max 30, ~1 s at 30 fps). ``0.0`` until at least two
        samples have arrived or if the time span is non-positive.
        """
        samples = self._capture_samples
        if len(samples) < 2:
            return 0.0
        first_ts, first_count = samples[0]
        last_ts, last_count = samples[-1]
        span_ns = last_ts - first_ts
        if span_ns <= 0:
            return 0.0
        delivered = last_count - first_count
        if delivered <= 0:
            return 0.0
        return delivered / (span_ns / 1_000_000_000.0)

    def drain_recording_buffer(self):
        """Return and clear all CameraFrames buffered since last drain.

        Returns
        -------
        list[CameraFrame]
            Frames in capture order, oldest first. Empty list if no
            new frames. Drained on the controller's main-thread tick so
            the camera capture loop never blocks waiting on GUI work.
        """
        with self.last_frame_lock:
            if not self._recording_buffer:
                return []
            frames = list(self._recording_buffer)
            self._recording_buffer.clear()
            return frames

    def get_last_frame(self):
        """Return the most recent captured frame, or None.

        Returns the actual ndarray reference, not a copy: CPython
        ndarray reference assignment is atomic under the GIL and the
        camera backend allocates a fresh ndarray per ``cap.read()`` /
        ``get_available_images()``, so a published frame is never
        overwritten in place. Downstream consumers (cv2.resize, GPU
        upload) allocate their own buffers; no defensive copy needed.
        """
        with self.last_frame_lock:
            return self.last_frame

    def publish_corrected_frame(self, image) -> None:
        """Replace the published display frame with the lens-corrected one.

        Called by the FrameBus after undistortion so every
        ``get_last_frame`` / ``get_latest_frame_versioned`` consumer
        (display tile, ROI helpers, pose probe frame) shares the bus's
        corrected coordinate space instead of the raw driver frame.
        The version is deliberately NOT bumped, this corrects the same
        capture; a bump would repaint pollers twice per frame.
        """
        with self.last_frame_lock:
            self.last_frame = image

    def get_latest_frame_versioned(self, last_version=0):
        """Latest-frame-wins: return new frame only if version changed.

        Returns ``(frame, timestamp_ns, version)`` when the camera has
        produced a frame newer than ``last_version``, else ``None``.
        Frame is the live ndarray reference, see ``get_last_frame``
        for why no copy is needed.
        """
        with self.last_frame_lock:
            if self.last_version > last_version and self.last_frame is not None:
                return (self.last_frame, self.last_timestamp_ns, self.last_version)
        return None

    def stop(self):
        """Stop camera capture"""
        self.running = False
        logger.debug(f"Stopping camera {self.camera_id}")

    # ------------------------------------------------------------------
    # QThread-API aliases so callers using isRunning()/wait() still work
    # against this threading.Thread.
    # ------------------------------------------------------------------
    def isRunning(self) -> bool:
        """QThread alias for threading.Thread.is_alive()."""
        return self.is_alive()

    def wait(self, timeout=None):
        """QThread alias for threading.Thread.join().

        QThread.wait returns bool (True if thread finished, False on timeout);
        threading.Thread.join returns None. Mimic the QThread shape.
        """
        self.join(timeout)
        return not self.is_alive()


class VideoSegmentProcessor:
    """Handles video segmentation for multiple boxes sharing one camera."""

    def __init__(self, config):
        self.config = config
        self.validate_config()
        self._build_box_index()
        logger.debug("Initializing video segment processor")

    def _build_box_index(self):
        """Build index mapping for faster box lookups"""
        self.box_index = {}
        if 'boxes' in self.config:
            for idx, box in enumerate(self.config['boxes']):
                # Support both 'box_id' and 'box_number' keys
                setup_id = box.get('box_id') or box.get('box_number')
                if setup_id is not None:
                    self.box_index[setup_id] = idx
        # Cache for pre-computed pixel coordinates: (box_id, frame_h, frame_w) -> (x, y, w, h)
        self._coord_cache = {}

    def validate_config(self):
        """Validate video segmentation configuration.

        Accepts either ``percent`` (resolution-independent, preferred) or
        ``pixel`` geometry per box.
        """
        try:
            if not self.config or 'boxes' not in self.config:
                logger.error("Invalid video segmentation config")
                raise ValueError("Invalid video segmentation config")

            required = ('x', 'y', 'width', 'height')
            for box in self.config['boxes']:
                geom = box.get('geometry')
                if not isinstance(geom, dict):
                    logger.error("Invalid box geometry configuration")
                    raise ValueError("Invalid box geometry configuration")

                percent = geom.get('percent')
                pixel = geom.get('pixel')
                if percent and all(k in percent for k in required):
                    continue
                if pixel and all(k in pixel for k in required):
                    continue
                logger.error("Box geometry must include 'percent' or 'pixel' "
                             "with x/y/width/height")
                raise ValueError("Box geometry missing percent and pixel")

            logger.info("Video segmentation configuration validated successfully")

        except Exception as e:
            logger.error(f"Configuration validation error: {str(e)}")
            raise

    def get_segment_size(self, setup_id, frame_shape=None):
        """Get the size of a specific box segment.

        If frame_shape is provided and percent geometry exists, compute size
        from the actual frame dimensions to adapt to any camera resolution.
        """
        try:
            if setup_id not in self.box_index:
                return None

            box_config = self.config['boxes'][self.box_index[setup_id]]
            geom = box_config.get('geometry', {})
            percent = geom.get('percent')
            pixel = geom.get('pixel', {})

            if frame_shape is not None and percent:
                frame_h, frame_w = frame_shape[:2]
                w = int(round(percent.get('width', 0) * frame_w))
                h = int(round(percent.get('height', 0) * frame_h))
                if w > 0 and h > 0:
                    return (w, h)

            if pixel:
                return (int(pixel.get('width', 0)), int(pixel.get('height', 0)))

            return None
        except Exception as e:
            logger.error(f"Error getting segment size for box {setup_id}: {str(e)}")
            return None

    def extract_segment(self, frame, setup_id):
        """Extract video segment for a specific box.

        Prefer percent-based geometry when available so segmentation adapts to
        the actual camera resolution.  Pixel coordinates are computed once per
        (box_id, frame_shape) and cached for subsequent calls.
        """
        try:
            if setup_id not in self.box_index:
                logger.warning(f"Box {setup_id} not found in segmentation config")
                return None

            frame_h, frame_w = frame.shape[:2]
            cache_key = (setup_id, frame_h, frame_w)

            coords = self._coord_cache.get(cache_key)
            if coords is None:
                box_config = self.config['boxes'][self.box_index[setup_id]]
                geom = box_config.get('geometry', {})
                percent = geom.get('percent')
                pixel = geom.get('pixel', {})

                if percent:
                    x = int(round(percent.get('x', 0) * frame_w))
                    y = int(round(percent.get('y', 0) * frame_h))
                    w = int(round(percent.get('width', 0) * frame_w))
                    h = int(round(percent.get('height', 0) * frame_h))
                else:
                    x = int(pixel.get('x', 0))
                    y = int(pixel.get('y', 0))
                    w = int(pixel.get('width', 0))
                    h = int(pixel.get('height', 0))

                if w <= 0 or h <= 0:
                    logger.warning(f"Invalid segment size for box {setup_id}: w={w}, h={h}")
                    return None

                if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
                    logger.warning(
                        f"Invalid segment coordinates for box {setup_id}: "
                        f"(x={x}, y={y}, w={w}, h={h}) vs frame {frame.shape}"
                    )
                    return None

                coords = (x, y, w, h)
                self._coord_cache[cache_key] = coords

            x, y, w, h = coords
            # Return a view (no copy), callers that need persistence (e.g.
            # add_frame) already copy.  This avoids a redundant ~2.7 MB copy
            # per frame at 1280×720 BGR.
            return frame[y:y + h, x:x + w]

        except Exception as e:
            logger.error(f"Segment extraction error for Box {setup_id}: {str(e)}")
            return None

    def segment_rect(self, setup_id, frame_h, frame_w):
        """Cached ``(x, y, w, h)`` pixel crop for ``setup_id`` at this frame
        size, or ``None``. Populated by ``extract_segment`` (percent OR pixel
        geometry), call that first; this just reads the cache so publishers
        don't recompute the rect per frame.
        """
        return self._coord_cache.get((setup_id, frame_h, frame_w))

def plan_sync_start_order(camera_ids, role_of):
    """Order camera starts for hardware sync.

    A secondary camera must be armed and waiting on its trigger BEFORE the
    primary begins free-running and driving the shared line, otherwise the
    first strobe edges are missed. Returns ``camera_ids`` reordered so every
    ``secondary`` starts first, standalone (``none``) cameras next, and every
    ``primary`` last; order within each group is preserved. ``role_of(cid)``
    yields ``"primary" | "secondary" | "none"``.
    """
    sec, standalone, pri = [], [], []
    for cid in camera_ids:
        role = (role_of(cid) or "none")
        if role == "secondary":
            sec.append(cid)
        elif role == "primary":
            pri.append(cid)
        else:
            standalone.append(cid)
    return sec + standalone + pri


class VideoManager:
    """Manages video streaming and recording operations with dynamic FPS control."""

    def __init__(self):
        self.cameras = {}
        self.box_camera_map = {}
        # Per-camera ROI segmenters keyed by camera_id, so each camera crops
        # its OWN boxes and a second camera's segmentation never overwrites
        # the first's.
        self._segment_processors: dict = {}

        # Default FPS applied when a caller doesn't specify one.
        self.target_fps = 30

        # Capture resolution chosen by the user in the camera-connect dialog.
        # ``None`` means "no choice yet", start_camera() refuses to open the
        # camera rather than picking a default, which would misalign ROIs.
        self.target_resolution: Optional[tuple[int, int]] = None

        # Frame loss strategy: "accept" (default) or "remux"
        self.frame_strategy = "accept"

        logger.debug("Initializing VideoManager")

        # Detect encoder capabilities in background (avoids delay on first
        # recording). EncoderCapabilities is a process-wide singleton, so
        # skip if already populated, avoids a redundant ffmpeg-probe
        # subprocess each time a VideoManager is reconstructed (Jetson CPU).
        from source.video.recording.ffmpeg import EncoderCapabilities
        if not EncoderCapabilities.get_instance()._detection_done:
            threading.Thread(
                target=self._detect_encoders,
                name="VideoManager-encoder-detect",
                daemon=True,
            ).start()

    def _detect_encoders(self):
        """Detect available video encoders in background thread.

        Idempotent and re-entrant via ``EncoderCapabilities.detect_all``'s
        internal lock, if another caller is already running detection,
        this returns immediately after the lock acquire.
        """
        try:
            from source.video.recording.ffmpeg import EncoderCapabilities
            caps = EncoderCapabilities.get_instance()
            caps.detect_all()
        except Exception as e:
            logger.warning(f"Encoder detection failed: {e}")

    def start_camera(self, camera_id, setup_id, segment_config=None,
                     camera_backend="opencv", camera_config=None):
        """Start camera for a box.

        Args:
            camera_id: Camera identifier (int for OpenCV, unique_id string for others).
            setup_id: Box number.
            segment_config: Optional ROI segmentation config dict.
            camera_backend: Backend name ("opencv", "spinnaker", "ximea").
            camera_config: Optional config dict for scientific cameras
                (exposure_us, gain_db, external_trigger, etc.).
        """
        global _session_start_host_ns, _session_start_wall
        try:
            if segment_config:
                self._segment_processors[camera_id] = VideoSegmentProcessor(segment_config)

            # Shared-camera guard, AFTER this call's own segment config is
            # installed (checking before it rejected the very call that
            # carried the config). If another box already streams this
            # camera and no ROI segmentation exists, every box would see
            # the full frame.
            already_serving = any(
                str(cid) == str(camera_id) and sid != setup_id
                for sid, cid in self.box_camera_map.items())
            if already_serving and camera_id not in self._segment_processors:
                logger.error(
                    "Video segmentation config required for shared camera")
                return False

            # Initialize session clock reference (once)
            if _session_start_host_ns == 0:
                _session_start_host_ns = host_clock.host_ns()
                _session_start_wall = time.time()

            # Add box to map first to get correct active count
            self.box_camera_map[setup_id] = camera_id

            # Per-camera target FPS: the caller (controller) puts this camera's
            # own selected_fps in camera_config so two cameras can run at
            # different rates. Falls back to the manager default only when the
            # caller didn't specify one.
            target_fps = int((camera_config or {}).get("target_fps")
                             or self.target_fps)

            # Clean up a dead CameraThread BEFORE the membership check. A
            # thread that exited without connecting (busy device / wedged
            # driver) stays in ``self.cameras`` and blocks future connects
            # for the same camera_id via the ``not in self.cameras`` branch
            # below. Remove the stale entry so the next connect tries again.
            existing_thread = self.cameras.get(camera_id)
            if existing_thread is not None and not existing_thread.is_alive():
                logger.info(
                    f"VideoManager: removing dead CameraThread for "
                    f"camera {camera_id} (was alive=False, connected="
                    f"{getattr(existing_thread, 'connected', '?')}) "
                    f"so the next connect can spawn a fresh thread."
                )
                # Best-effort cleanup of the dead thread's camera handle.
                try:
                    existing_thread.stop()
                except Exception:
                    pass
                del self.cameras[camera_id]

            if camera_id not in self.cameras:
                # Resolution priority: explicit camera_config > VideoManager
                # target_resolution. The segment_processor schema is
                # deliberately resolution-AGNOSTIC (see camera_connect.py
                # :: buildSegmentConfig docstring) so it is not read for
                # camera resolution.
                cfg = dict(camera_config or {})
                if "width" not in cfg or "height" not in cfg:
                    if self.target_resolution is not None:
                        cfg["width"], cfg["height"] = self.target_resolution
                    else:
                        raise ValueError(
                            "Camera resolution not configured. Set "
                            "VideoManager.target_resolution or pass width/height "
                            "in camera_config before calling start_camera()."
                        )

                # Build unique_id for camera creation if needed
                if camera_backend != "opencv" and isinstance(camera_id, str) and "-" in camera_id:
                    unique_id = camera_id
                else:
                    unique_id = f"{camera_id}-{camera_backend}"

                thread = CameraThread(
                    camera_id=unique_id if camera_backend != "opencv" else camera_id,
                    width=int(cfg["width"]),
                    height=int(cfg["height"]),
                    target_fps=target_fps,
                    camera_config=cfg,
                )
                thread.start()

                self.cameras[camera_id] = thread
            elif already_serving:
                # Shared camera: another box is live on this handle, so
                # applying this box's rate would silently retune ITS capture
                # (the exact anti-pattern set_target_fps warns about). Keep
                # the established rate.
                logger.debug(
                    f"Camera {camera_id} already streaming for another box; "
                    f"keeping its current rate (box {setup_id} requested "
                    f"{target_fps})")
            else:
                # Re-attach to a camera only this box uses (e.g. reconnect):
                # safe to apply the requested rate.
                self.cameras[camera_id].set_target_fps(target_fps)

            logger.info(f"Camera started for box {setup_id} (backend={camera_backend}) "
                        f"with target FPS: {target_fps}")
            return True

        except Exception as e:
            # Surface the original exception type + message so the GUI
            # error log is more than "Failed to start camera".
            import traceback as _tb
            logger.error(
                f"Failed to start camera (box {setup_id}, camera_id={camera_id}, "
                f"backend={camera_backend}): {type(e).__name__}: {e}"
            )
            logger.debug("start_camera traceback: %s", _tb.format_exc())
            if setup_id in self.box_camera_map:
                del self.box_camera_map[setup_id]
            return False

    def stop_camera(self, setup_id):
        """Stop camera for a box and recalculate FPS for remaining cameras."""
        try:
            if setup_id not in self.box_camera_map:
                return

            camera_id = self.box_camera_map[setup_id]
            del self.box_camera_map[setup_id]

            if not any(cam_id == camera_id for cam_id in self.box_camera_map.values()):
                # Last box on this camera → drop its segmenter too.
                self._segment_processors.pop(camera_id, None)
                if camera_id in self.cameras:
                    thread = self.cameras.pop(camera_id)
                    thread.stop()
                    self._reap_camera_thread(camera_id, thread)

            logger.info(f"Camera stopped for box {setup_id}")

        except Exception as e:
            logger.error(f"Failed to stop camera: {str(e)}")

    def _reap_camera_thread(self, camera_id, thread) -> None:
        """Wait out a stopped camera thread off the GUI thread.

        ``stop_camera`` runs on the GUI thread (Disconnect button). A camera
        parked in a blocking ``grab()``/``read()``, or mid-``reconnect()``
        with its retry sleeps, can take seconds to notice ``running=False``,
        joining here froze the UI for up to 3 s per camera. The thread is a
        daemon that releases the device as it unwinds, so the join only needs
        to happen somewhere; it does not need to happen before the click
        returns. Logged on overrun so a genuinely wedged device is visible
        rather than silently abandoned.
        """
        def _join():
            thread.wait(timeout=10.0)
            if thread.is_alive():
                logger.warning(
                    "Camera %s thread still alive 10 s after stop, the "
                    "device may stay claimed until it unwinds", camera_id)
            else:
                logger.debug("Camera %s thread reaped", camera_id)

        threading.Thread(target=_join, name=f"camera-reap-{camera_id}",
                         daemon=True).start()

    def get_fps(self, setup_id):
        """target_fps for the box (single source of truth).

        Returns ``camera.config.target_fps``, the user-picked rate.
        For runtime delivery health (FPS label tooltip, dip detection)
        call ``get_runtime_fps`` instead.
        """
        try:
            if setup_id in self.box_camera_map:
                camera_id = self.box_camera_map[setup_id]
                cam_thread = self.cameras.get(camera_id)
                if cam_thread is not None:
                    return float(cam_thread.get_target_fps())
            return float(self.target_fps)
        except Exception as e:
            logger.error(f"Error reading target FPS for box {setup_id}: {str(e)}")
            return 0.0

    def get_runtime_fps(self, setup_id):
        """Live measured delivery rate from the camera-thread sample deque."""
        try:
            if setup_id in self.box_camera_map:
                camera_id = self.box_camera_map[setup_id]
                cam_thread = self.cameras.get(camera_id)
                if cam_thread is not None:
                    fps = cam_thread.get_capture_fps()
                    if fps > 0:
                        return fps
            return 0.0
        except Exception as e:
            logger.error(f"Error reading runtime FPS for box {setup_id}: {str(e)}")
            return 0.0

    def resolved_settings_for_box(self, setup_id):
        """The resolved ResolvedCameraSettings for a box (or None if not yet configured)."""
        if setup_id in self.box_camera_map:
            camera_id = self.box_camera_map[setup_id]
            cam_thread = self.cameras.get(camera_id)
            if cam_thread is not None and cam_thread._camera is not None:
                return cam_thread._camera.config
        return None

    def cleanup(self):
        """Clean up all resources."""
        try:
            # Signal every camera first, then join, stopping and joining one
            # at a time serialised each camera's unwind behind the previous
            # one. Bounded so a wedged device can't hang app shutdown.
            cams = list(self.cameras.values())
            for camera in cams:
                camera.stop()
            for camera in cams:
                if not camera.wait(timeout=5.0):
                    logger.warning(
                        "Camera %s did not stop within 5 s during cleanup",
                        getattr(camera, "camera_id", "?"))

            self.cameras.clear()
            self.box_camera_map.clear()
            self._segment_processors.clear()

            logger.info("VideoManager cleanup completed successfully")

        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")

    def set_target_fps(self, fps, camera_id=None):
        """Set the target FPS.

        ``camera_id=None`` sets the manager DEFAULT applied to the next
        camera open. It does NOT retune already-running cameras: connecting a
        second camera at a different rate would then silently retune every
        other one. Pass a ``camera_id`` to retune exactly that one running
        camera (e.g. a per-camera runtime settings change).
        """
        if not (fps and isinstance(fps, (int, float)) and fps > 0):
            return
        fps = int(fps)
        if camera_id is None:
            self.target_fps = fps
            logger.info(f"VideoManager default target FPS set to {fps}")
            return
        thread = self.cameras.get(camera_id)
        if thread is not None and hasattr(thread, "set_target_fps"):
            thread.set_target_fps(fps)
            logger.info(f"Camera {camera_id} target FPS set to {fps}")

    def describe_camera_features(self, camera_id) -> list:
        """Feature descriptors for one running camera; empty if it is not open.

        Descriptors are read from the live device, so ranges and enum entries
        are the sensor's own rather than a guess.
        """
        thread = self.cameras.get(camera_id)
        if thread is None or not hasattr(thread, "describe_features"):
            return []
        return thread.describe_features()

    def set_camera_feature(self, camera_id, key: str, value) -> bool:
        """Queue a live feature write for one running camera.

        Returns False when the camera is not open, the caller should then
        store the value and let it apply at the next connect.
        """
        thread = self.cameras.get(camera_id)
        if thread is None or not hasattr(thread, "queue_feature"):
            return False
        thread.queue_feature(key, value)
        return True

    def get_target_resolution(self) -> Optional[tuple[int, int]]:
        """Return the (width, height) the next camera open will request, or None."""
        return self.target_resolution

    def set_target_resolution(self, width: int, height: int) -> None:
        """Set the capture resolution applied at the next camera open.

        Resolution can't be changed mid-stream on OpenCV without releasing
        and reopening the device, so this only affects future ``start_camera``
        calls. Already-running cameras keep their current resolution until
        they are stopped and restarted (the runtime settings dialog has a
        Reconnect button for that).
        """
        if not (isinstance(width, int) and isinstance(height, int)
                and width > 0 and height > 0):
            raise ValueError(
                f"Resolution must be positive integers, got {width!r}x{height!r}"
            )
        self.target_resolution = (int(width), int(height))
        logger.info(
            f"VideoManager target resolution set to "
            f"{self.target_resolution[0]}x{self.target_resolution[1]} "
            f"(applies on next camera connect)"
        )

    def set_frame_strategy(self, strategy):
        """Set frame loss strategy: 'accept' (default) or 'remux'."""
        if strategy in ("accept", "remux"):
            self.frame_strategy = strategy
            logger.info(f"VideoManager frame strategy set to '{strategy}'")

    def is_camera_streaming(self, setup_id):
        """Check if camera for a box is actually streaming (has received frames).

        Args:
            setup_id: Box identifier

        Returns:
            bool: True if camera is connected AND has captured at least one frame
        """
        try:
            camera_id = self.box_camera_map.get(setup_id)
            if camera_id is None:
                return False

            camera_thread = self.cameras.get(camera_id)
            if camera_thread is None:
                return False

            # Connected (device open) AND liveness says frames are flowing
            # (``_streaming`` flips False only once a stall is DETECTED, ~0.75s
            # of silence, not on every inter-frame gap) AND a frame exists.
            # A briefly-stalled camera therefore reads "not streaming" rather
            # than reporting OK on its stale last frame.
            if not camera_thread.connected:
                return False
            if not getattr(camera_thread, "_streaming", True):
                return False
            frame = camera_thread.get_last_frame()
            return frame is not None

        except Exception as e:
            logger.error(f"Error checking camera streaming for box {setup_id}: {e}")
            return False

    def get_last_frame(self, setup_id):
        """Get last captured frame for a box.

        Args:
            box_id: Box identifier

        Returns:
            numpy array of frame (full or segmented based on config), or None
        """
        try:
            camera_id = self.box_camera_map.get(setup_id)
            if camera_id is None:
                return None

            camera_thread = self.cameras.get(camera_id)
            if camera_thread is None:
                return None

            frame = camera_thread.get_last_frame()
            if frame is None:
                return None

            # Apply THIS camera's segmentation if configured.
            seg = self._segment_processors.get(camera_id)
            if seg:
                if setup_id in getattr(seg, "box_index", {}):
                    # This box is a ROI segment of a shared camera. If
                    # extraction fails (ROI out of bounds for the current
                    # resolution) the correct answer is "no frame", NOT the
                    # full, uncropped frame. Returning the whole frame is what
                    # made the zone editor normalize zones against the wrong
                    # reference and land them offset at replay. Matches the
                    # FrameBus blackout semantics for the same failure.
                    return seg.extract_segment(frame, setup_id)
                # Box isn't segmented on this camera → the full frame IS its
                # view (single-box-per-camera).
                return frame

            return frame

        except Exception as e:
            logger.error(f"Error getting last frame for box {setup_id}: {str(e)}")
            return None

    def segment_processor_for(self, camera_id):
        """Return the ROI segmenter for ``camera_id`` (or None)."""
        return self._segment_processors.get(camera_id)

    def segment_processor_for_box(self, setup_id):
        """Return the ROI segmenter for the camera that ``setup_id`` is on."""
        return self._segment_processors.get(self.box_camera_map.get(setup_id))

    def get_full_frame(self, camera_id):
        """Get last captured full frame from a camera (without segmentation).

        Args:
            camera_id: Camera identifier

        Returns:
            numpy array of full frame, or None
        """
        try:
            camera_thread = self.cameras.get(camera_id)
            if camera_thread:
                return camera_thread.get_last_frame()
            return None
        except Exception as e:
            logger.error(f"Error getting full frame from camera {camera_id}: {str(e)}")
            return None

