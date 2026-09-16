"""Threaded video recording for behavioural experiments.

VideoRecorder encodes frames and does file I/O in a background thread so
the GUI never blocks. It manages file creation, frame writing, and
cleanup on stop via OpenCV's VideoWriter (or FFmpeg/NVENC), with
configurable codec, resolution, and frame rate, and synchronises
per-frame timestamps to pyControl MCU time.

Usage:
    recorder = VideoRecorder(camera_id=0, fps=20, resolution=(240, 240))
    recorder.start_recording("recordings/box1")
    recorder.stop_recording()
"""

import os
import cv2
import time
from source import host_clock
import threading
import subprocess
import numpy as np
from collections import deque
from datetime import datetime
from source.log import get_logger
from source.video.recording.ffmpeg import EncoderCapabilities, VideoWriterFactory
from source.video.recording.drop_log import drop_log as _drop_log

logger = get_logger()

#: How far the delivered rate may sit from the request and still be called
#: honoured. Wider than measurement noise, far narrower than the 10-asked /
#: 15-delivered this exists to catch.
_FPS_HONOURED_FRAC = 0.10

# Standard video resolutions for scaling (based on larger dimension)
STANDARD_SIZES = [240, 360, 480, 540, 720]


def get_standard_resolution(width, height, min_size=240, max_size=720):
    """Calculate standard output resolution preserving aspect ratio.

    Args:
        width: Input width (e.g., ROI width)
        height: Input height (e.g., ROI height)
        min_size: Minimum output size for larger dimension
        max_size: Maximum output size for larger dimension

    Returns:
        (output_width, output_height) scaled to closest standard size
    """
    if width <= 0 or height <= 0:
        return (480, 480)  # Default fallback

    # Find the larger dimension
    larger_dim = max(width, height)
    aspect_ratio = width / height

    # Below min scales up to min; above max scales down to max; else
    # snap to the closest standard size.
    if larger_dim < min_size:
        target_size = min_size
    elif larger_dim > max_size:
        target_size = max_size
    else:
        target_size = min(STANDARD_SIZES, key=lambda x: abs(x - larger_dim))

    # Calculate output dimensions preserving aspect ratio
    if width >= height:
        # Width is larger
        out_width = target_size
        out_height = int(target_size / aspect_ratio)
    else:
        # Height is larger
        out_height = target_size
        out_width = int(target_size * aspect_ratio)

    # Ensure even dimensions (required for video codecs)
    out_width = out_width if out_width % 2 == 0 else out_width + 1
    out_height = out_height if out_height % 2 == 0 else out_height + 1

    # Ensure minimum size of 2x2
    out_width = max(2, out_width)
    out_height = max(2, out_height)

    return (out_width, out_height)


class VideoRecorder:
    """Record video with timestamps synchronized to pyControl data.

    Frames are fed in via ``add_frame()`` from the shared-camera pipeline
    (``RecorderSink``); a background thread encodes and writes them so the
    GUI never blocks. Per-frame metadata is owned by the pipeline's
    ``FrameLog``/``TrackingWriter``, not by this recorder.
    """

    def __init__(
        self,
        camera_id=0,
        fps=20,
        resolution=(420, 420),
        codec='MJPEG',
        roi=None,
        use_gpu='auto',
        frame_strategy="accept",
        allow_cpu_fallback=True,
        prefer_hevc=False,
        crf=23,
    ):
        """Initialize video recorder.

        Args:
            camera_id: Camera device ID (int) or path (str), for logging/naming
            fps: Frames per second for recording
            resolution: Video resolution as (width, height)
            codec: Video codec - 'MJPEG' (lightest, default), 'H264', or 'XVID'
                   MJPEG: Fastest encoding, lightest CPU, larger file, easy to convert
                   H264: Better compression, heavier CPU, smaller file
                   XVID: Medium compression, medium CPU
            use_gpu: GPU encoding preference - 'auto' (detect), True (force), False (disable)
                     When GPU is available, uses H264 NVENC for efficient encoding
        """
        self.camera_id = camera_id
        self.fps = fps
        self.resolution = resolution
        self.codec = codec.upper() if codec else 'MJPEG'
        if self.codec not in ('MJPEG', 'MJPG', 'H264', 'XVID'):
            self.codec = 'MJPEG'
        # GPU encoding settings
        self.use_gpu = use_gpu  # 'auto', True, or False
        # When a hardware encoder is present but its writer won't open
        # (usually the NVENC concurrent-session cap on a many-box rig),
        # allow a CPU fallback so the box still records rather than losing
        # video. False = strict hardware-only (abort on failure).
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        # Output codec family + quality. ``prefer_hevc`` picks H.265 (smaller
        # files) where a HEVC encoder exists; ``crf`` is the constant-quality
        # factor (0-51, lower = better/larger) passed to the writer. Both feed
        # the existing VideoWriterFactory params, no new encoder path.
        self._prefer_hevc = bool(prefer_hevc)
        self._crf = max(0, min(51, int(crf))) if crf is not None else 23
        self._using_gpu = False  # Track if GPU encoding is active
        self._actual_encoder = None  # Track which encoder is being used
        self.frame_strategy = frame_strategy if frame_strategy in ("accept", "remux") else "accept"
        self.writer = None
        self.recording = False
        self.thread = None
        self.frame_count = 0
        # Consecutive encode-write errors; reset on any successful write.
        # After _WRITE_ERROR_FATAL in a row the recorder gives up and
        # signals ``encoder_write_failed`` rather than spinning forever.
        self._write_errors = 0
        self._WRITE_ERROR_FATAL = 150   # ~5 s at 30 fps of solid failure
        # Set True only when the encoder gives up (fatal write-error streak),
        # distinguishing a dead encoder from an intentional stop_recording().
        # RecorderSink polls this to raise a per-box health alarm even when a
        # tracking writer keeps the box's rows flowing.
        self.encoder_failed = False
        # One switch from a dead h264_nvmpi writer to libx264 per session.
        self._hw_fallback_done = False
        # Wall-clock at recording start (for human-readable headers only).
        self.start_time = None
        # Monotonic ns at recording start. All per-frame timing math is
        # in monotonic ns: actual_capture_ms = (capture_host_ns -
        # rec_start_host_ns) // 1_000_000.
        self.rec_start_host_ns = None
        # Latest in-queue frame's monotonic_ns. Used for final mp4
        # duration via remux: end - rec_start_host_ns.
        self.last_capture_host_ns = None
        self.video_path = None
        self.roi = roi  # Optional (x, y, w, h) crop area
        self._roi_valid = False
        # Frame queue, buffers the external feed against variable timing.
        self.frame_queue = deque(maxlen=60)
        # Drop-frame callback, set by main_window after construction:
        #   recorder.drop_callback = lambda count, reason: writer.on_dropped_frame(count, reason)
        # Called when add_frame() finds the queue already at maxlen (frame
        # would be evicted). Optional, None means drops happen silently.
        self.drop_callback = None
        # Called as (encoder, replaced, lost_frames) when a dead h264_nvmpi
        # writer is swapped for libx264, so the session file can say so.
        self.encoder_changed_callback = None
        # Per-frame overlay hook for annotated-record mode, set by
        # RecorderSink. Called as annotate_callback(frame, capture_ts)
        # -> annotated_frame.
        self.annotate_callback = None
        # Single-channel recording: a grayscale camera delivers true 2D frames
        # (capture converts once), so the encoder is fed ``-pix_fmt gray`` and
        # the writer chain stays 1 byte/pixel end to end, a third of the pipe
        # bandwidth and NVENC load vs bgr24. Set per session from the source
        # CameraConfig in start_recording; the write loop coerces each frame to
        # match so the FFmpeg pixel format and the data never disagree.
        self._grayscale = False
        # Lock for thread-safe frame queue access
        self._queue_lock = threading.Lock()
        # Last line of the constructor, read by ``__del__``: a recorder that
        # never finished being built has nothing to close, and trying anyway
        # raises from inside garbage collection where the message is
        # unattributable and hides whatever actually failed.
        self._fully_constructed = True

    def _apply_roi_crop(self, frame):
        """Apply ROI crop to a frame. Returns cropped frame or original on failure."""
        if not self._roi_valid or not self.roi:
            return frame
        try:
            x, y, w, h = map(int, self.roi)
            frame_h, frame_w = frame.shape[:2]
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = max(1, min(w, frame_w - x))
            h = max(1, min(h, frame_h - y))
            return frame[y:y + h, x:x + w]
        except Exception as e:
            logger.warning(f"ROI crop failed, writing full frame: {e}")
            self._roi_valid = False
            return frame

    @staticmethod
    def _is_valid_frame(frame):
        """Check if a frame is non-None and non-empty."""
        return frame is not None and frame.size > 0 and frame.shape[0] > 0 and frame.shape[1] > 0

    @staticmethod
    def _coerce_channels(frame, grayscale):
        """Make ``frame``'s channel count agree with the encoder's pixel
        format. A grayscale session (``-pix_fmt gray``) wants 2D, an annotate
        overlay may have promoted it to BGR; a colour session wants 3-channel,
        a stray 2D frame would desync the encoder's byte count.

        Deliberately NOT ``cameras.base.normalize_channels``, which makes the
        same channel decision but *copies* a mono passthrough because the SDK
        drain loops hand it a buffer the driver recycles. Here the frame is
        already ours and this runs per recorded frame, so a copy would be a
        pure per-frame allocation on the recording hot path. Same shape,
        different ownership contract; that is why there are two.
        """
        if grayscale:
            if frame.ndim == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

    def start_recording(self, data_dir, subject_id, datetime_now=None, box_ID=None,
                        camera_config=None, file_stem=None):
        """Start video recording.

        Args:
            data_dir: Base data directory (video goes to video_data subfolder)
            subject_id: Subject ID for file naming
            datetime_now: Optional datetime for file naming
            box_ID: Optional Box ID for file naming
            camera_config: Optional CameraConfig from the source camera. When
                provided, ``target_fps`` is the rate written to the encoder
                and the .txt header (used verbatim).
            file_stem: Optional explicit file stem (dry-run fixed name).
        """
        if self.recording:
            logger.warning("Recording already in progress")
            return False

        self._adopt_camera_settings(camera_config)
        if datetime_now is None:
            datetime_now = datetime.now()
        self.video_path = self._build_output_path(
            data_dir, subject_id, datetime_now, box_ID, file_stem)
        self._target_size = self._decide_output_size(box_ID)

        if not self._open_writer(self._target_size):
            return False

        self._arm_session()
        logger.info(f"Started video recording: {self.video_path}")
        return True

    # ── start_recording steps ────────────────────────────────────────────

    def _adopt_camera_settings(self, camera_config) -> None:
        """Take the rate and colour mode from the camera, and drop any frames
        left over from a previous session."""
        # The rate the camera is OBSERVED delivering, when it was measured,
        # and the requested rate only as a fallback. Using the request
        # verbatim wrote a file whose header disagreed with its own contents:
        # a session asked for 10 fps on a camera delivering 15 produced 1.5x
        # of real time, played back slow, and every timestamp derived from the
        # nominal rate was wrong. A driver accepts a rate it cannot hold and
        # reports it back unchanged, so the request is not evidence.
        want = float(getattr(camera_config, "target_fps", 0) or 0)
        delivered = float(getattr(camera_config, "delivered_fps", 0) or 0)
        if delivered > 0 and want > 0 and abs(delivered - want) <= max(
                1.0, want * _FPS_HONOURED_FRAC):
            # Honoured. The REQUEST is the right header: it is the round,
            # exact number, and a measurement carries about a frame of noise,
            # so stamping 29.87 on a camera that really runs at 30 would be
            # less accurate, not more.
            self.fps = want
        elif delivered > 0:
            # Not honoured. The measurement is the only truthful header.
            self.fps = delivered
            if want > 0:
                logger.warning(
                    "Camera %s: recording at the delivered %.2f fps, not the "
                    "requested %.0f fps. The file plays back at real speed; "
                    "the camera does not hold the rate that was picked.",
                    self.camera_id, self.fps, want)
        elif want > 0:
            # Never measured. The request is all there is.
            self.fps = want
        # A grayscale source records single-channel, the encoder is told
        # ``gray`` and the frames stay 2D (see the write loop's channel coerce).
        self._grayscale = bool(getattr(camera_config, "grayscale", False))
        #: Recorded in the session header so a reader knows whether the frame
        #: times in it are the device's or the host's.
        self.timestamp_source = str(
            getattr(camera_config, "timestamp_source", "") or "")
        self.requested_fps = float(getattr(camera_config, "target_fps", 0) or 0)
        with self._queue_lock:
            self.frame_queue.clear()

    def _build_output_path(self, data_dir, subject_id, datetime_now,
                           box_ID, file_stem) -> str:
        """Full path for the video, creating its directory.

        The stem matches the data_logger pattern (ID-Box_ID-rest) because
        ``build_session_stem`` owns that format for every file of a session,
        MCU TSV, video and frame log, so none of them can drift apart.
        """
        # exist_ok so an existing session directory is never clobbered.
        os.makedirs(data_dir, exist_ok=True)
        from source.video.recording import build_session_stem
        if file_stem:
            # Explicit stem (dry run: a fixed "video_Box<N>" overwritten each
            # run); the container/ext still matches a real recording.
            stem = file_stem
        elif box_ID:
            stem = build_session_stem(subject_id, box_ID, datetime_now)
        else:
            stem = subject_id + datetime_now.strftime("-%Y-%m-%d-%H%M%S")
        return os.path.join(data_dir, stem + ".avi")

    # Output videos are normalised to roughly this on the long edge: small
    # enough to stay cheap to encode and review, large enough for pose.
    ROI_TARGET_SIZE = 420

    def _decide_output_size(self, box_ID) -> tuple:
        """Encoded frame size: an ROI crop scaled to ~420 on its long edge,
        else the frame normalised to a standard resolution.

        An ROI that does not fit inside the frame is ignored rather than
        clamped, a crop the operator did not draw would silently record the
        wrong part of the arena.
        """
        width, height = self.resolution
        logger.info(f"External feed mode: using resolution {width}x{height}")
        if not self.roi:
            return get_standard_resolution(width, height)

        try:
            x, y, w, h = map(int, self.roi)
        except Exception as e:
            logger.warning(f"Failed to parse ROI {self.roi}: {e}")
            return width, height
        if not (x >= 0 and y >= 0 and w > 0 and h > 0
                and x + w <= width and y + h <= height):
            logger.warning(f"Ignoring invalid ROI for Box {box_ID}: "
                           f"{self.roi} (frame {width}x{height})")
            return width, height

        aspect = w / h
        if aspect >= 1:                       # wider than tall
            tw, th = self.ROI_TARGET_SIZE, int(self.ROI_TARGET_SIZE / aspect)
        else:
            tw, th = int(self.ROI_TARGET_SIZE * aspect), self.ROI_TARGET_SIZE
        # Codecs want even dimensions.
        tw += tw % 2
        th += th % 2
        self._roi_valid = True
        logger.info(f"Recording with ROI crop for Box {box_ID}: "
                    f"x={x}, y={y}, w={w}, h={h} -> output {tw}x{th}")
        return tw, th

    def _open_writer(self, target_size) -> bool:
        """Open an encoder for ``target_size``. False means do not record.

        Order: hardware encoder first when the host has one, then the FFmpeg
        CPU chain, then a last-resort OpenCV writer. The one refusal is strict
        hardware-only mode, a hardware encoder exists, its writer would not
        open, and the operator disabled CPU fallback, where recording on CPU
        anyway would misreport what the session was.
        """
        self.writer = None
        self._using_gpu = False
        self._actual_encoder = None
        caps = EncoderCapabilities.get_instance()
        gpu_forced = (self.use_gpu is not False) and caps.gpu_force_required()

        # ffmpeg is tried whenever available, use_gpu=False must still
        # reach the libx264 CPU chain (skipping ffmpeg entirely silently
        # recorded MJPG .avi instead of H.264 mp4).
        if caps.has_ffmpeg:
            self._try_ffmpeg_writer(target_size, gpu_forced)

        if self.writer is None and gpu_forced and not self.allow_cpu_fallback:
            # create_writer already logged the strict-refusal reason,
            # don't log a near-identical second error line.
            return False

        if self.writer is None:
            return self._open_opencv_writer(target_size)
        return True

    def _try_ffmpeg_writer(self, target_size, gpu_forced) -> None:
        """Ask the factory for an encoder; leave ``self.writer`` None if none
        opened, so the caller can fall through to the OpenCV chain."""
        try:
            self.writer = VideoWriterFactory.create_writer(
                self.video_path,            # factory changes .avi → .mp4
                self.fps,
                target_size,
                use_gpu=(self.use_gpu is not False),
                crf=self._crf,
                prefer_hevc=self._prefer_hevc,
                grayscale=self._grayscale,  # gray sources → -pix_fmt gray
                allow_cpu_fallback=self.allow_cpu_fallback,
            )
        except Exception as e:
            logger.warning(f"FFmpeg encoder setup failed: {e}")
            self.writer = None
            return
        if not (self.writer and self.writer.isOpened()):
            self.writer = None
            return
        # Read the encoder name back from the writer so the log and
        # _using_gpu reflect what was ACTUALLY selected (h264_nvenc desktop,
        # h264_v4l2m2m Jetson, or a libx264/HEVC CPU fallback after an NVENC
        # session limit).
        # Only the cv2 XVID last-resort writer lacks .encoder, labelling
        # it by the gpu_forced guess recorded "h264_nvenc" for an XVID avi.
        self._actual_encoder = getattr(self.writer, 'encoder', 'XVID')
        # Every fixed-function encoder counts, not just NVIDIA's: a session
        # encoded on Quick Sync reported "CPU" here, which is what the session
        # metadata and the operator would then have believed.
        self._using_gpu = self._actual_encoder in (
            'h264_nvenc', 'hevc_nvenc', 'h264_nvmpi', 'h264_v4l2m2m',
            'h264_qsv', 'h264_amf', 'h264_mf')
        real_path = getattr(self.writer, 'path', None)
        if real_path is not None:
            # FFmpegVideoWriter knows its actual output file.
            self.video_path = str(real_path)
        elif getattr(self.writer, 'encoder', None) is not None:
            # ffmpeg-like writer without a path attribute, the factory
            # normalises the container to .mp4.
            self.video_path = self.video_path.rsplit('.', 1)[0] + '.mp4'
        else:
            # cv2 XVID last resort, the factory wrote <stem>.avi. The old
            # unconditional .mp4 rename advertised a file that never
            # existed into the MCU TSV header and the runs index.
            self.video_path = self.video_path.rsplit('.', 1)[0] + '.avi'
        logger.info(f"Using {self._actual_encoder} for video recording "
                    f"-> {self.video_path}")

    def _open_opencv_writer(self, target_size) -> bool:
        """Last-resort writer on a host with no usable FFmpeg encoder."""
        fourcc_name = {'MJPEG': 'MJPG', 'MJPG': 'MJPG',
                       'XVID': 'XVID', 'H264': 'H264'}.get(self.codec, 'MJPG')
        self.writer = cv2.VideoWriter(
            self.video_path,
            cv2.VideoWriter_fourcc(*fourcc_name),
            self.fps,
            target_size,
            isColor=not self._grayscale,
        )
        self._actual_encoder = self.codec
        if not self.writer.isOpened():
            logger.error(f"Failed to create video writer for {self.video_path}")
            return False
        logger.info(f"Using OpenCV {self.codec} for video recording "
                    f"-> {self.video_path}")
        return True

    #: How long after recording starts a dead h264_nvmpi writer is replaced by
    #: libx264 in the same file. Later than this the file holds real minutes of
    #: video that reopening would overwrite, so a late failure keeps the
    #: existing encoder-dead alarm instead.
    _NVMPI_FALLBACK_WINDOW_S = 10.0

    def _recover_dead_nvmpi_writer(self, rejected_frame: bool = True) -> bool:
        """Swap a Jetson hardware writer that died at start-up for libx264.

        ``h264_nvmpi`` exists only on a Jetson, so no other machine takes this
        path. The file is reopened at the same path, so the MCU TSV and the
        runs index still name the right video; the frames the dead encoder had
        taken are counted as dropped, with the reason, in ``_drops.tsv``.
        ``rejected_frame`` says whether the frame in hand was refused (and so
        is lost too) or has not been written yet.
        """
        if (self._actual_encoder != 'h264_nvmpi' or self._hw_fallback_done
                or not self.allow_cpu_fallback or self.writer is None):
            return False
        started = self.rec_start_host_ns
        if started is None:
            return False
        elapsed = (host_clock.host_ns() - started) / 1e9
        if elapsed > self._NVMPI_FALLBACK_WINDOW_S:
            return False
        # Dead either way: the writer says so, or its FFmpeg has exited. The
        # second is the one that matters. Frames go into a 100 MB pipe buffer,
        # so a writer keeps "accepting" them for many seconds after the process
        # behind it has gone, longer than the window above at 240x240.
        healthy = getattr(self.writer, "is_healthy", None)
        proc = getattr(self.writer, "_process", None)
        exited = proc is not None and proc.poll() is not None
        if not exited and (not callable(healthy) or healthy()):
            return False
        self._hw_fallback_done = True
        lost = self.frame_count + (1 if rejected_frame else 0)
        logger.error(
            "Box %s: the Jetson hardware encoder (h264_nvmpi) stopped %.1f s "
            "into the recording; continuing on CPU libx264 in the same file. "
            "The %d frame(s) it had taken are lost.",
            self.camera_id, elapsed, lost)
        try:
            self.writer.release()
        except Exception as e:
            logger.debug("releasing the dead h264_nvmpi writer: %s", e)
        from pathlib import Path
        writer = VideoWriterFactory._try_ffmpeg(
            Path(self.video_path), self.fps, self._target_size, 'libx264',
            self._crf, self._grayscale,
            'CPU H.264 (libx264), after h264_nvmpi stopped')
        if writer is None:
            logger.error("Box %s: libx264 would not open either; this box's "
                         "video stops here.", self.camera_id)
            return False
        self.writer = writer
        self._actual_encoder = 'libx264'
        self._using_gpu = False
        self.frame_count = 0
        self._write_errors = 0
        _drop_log.record(
            "recorder", frame_idx=0, capture_ts=host_clock.host_ns() / 1e9,
            reason=f"nvmpi_stopped_fallback_libx264_lost_x{lost}")
        if self.drop_callback:
            try:
                self.drop_callback(lost, "nvmpi_stopped_fallback_libx264")
            except Exception:
                pass
        if self.encoder_changed_callback:
            try:
                self.encoder_changed_callback('libx264', 'h264_nvmpi', lost)
            except Exception:
                pass
        return True

    def header_encoder(self):
        """The encoder to name in the session's ``_video_data.txt`` header.

        Jetson only; ``None`` everywhere else, where the header keeps its
        "unknown" and the files written there do not change.
        """
        try:
            if not EncoderCapabilities.get_instance().is_jetson():
                return None
        except Exception:
            return None
        return self._actual_encoder

    def _arm_session(self) -> None:
        """Reset per-session counters and start the write thread.

        Per-frame metadata belongs to the pipeline's RecorderSink →
        FrameLog/TrackingWriter; this recorder writes only the encoded video.
        """
        self.frame_count = 0
        # Clear failure state so a reused recorder object never starts a new
        # session already tripped (encoder_failed / _write_errors are set on
        # the fatal path and otherwise only reset in __init__).
        self.encoder_failed = False
        self._write_errors = 0
        self._hw_fallback_done = False
        # Two anchors at recording start:
        #   start_time, wall-clock seconds, only for human-readable
        #                    headers / file naming.
        #   rec_start_host_ns, monotonic_ns, used for all per-frame math
        #                    (mp4 duration, frame_ms in tsv).
        self.start_time = time.time()
        self.rec_start_host_ns = host_clock.host_ns()
        self.recording = True
        self.thread = threading.Thread(target=self._external_record_loop,
                                       daemon=True)
        self.thread.start()

    def add_frame(self, frame, capture_time_ns=None):
        """Queue a frame for the recording thread.

        The caller (source.video.framebus RecorderSink) hands the buffer in by
        reference. Display copies before any in-place overlay draw, so
        no defensive copy here. If `annotate_callback` is set, it runs
        first and produces a fresh annotated frame to save.

        Timebase: ``capture_time_ns`` is canonical monotonic_ns at
        capture (from CameraThread.capture_host_ns). The queue stores
        monotonic_ns; conversion to relative-ms happens at write time.
        Wall-clock is only for human-readable headers.

        Returns:
            True  if the frame was accepted.
            False if recording is off, the frame is bad, or the queue is
                  full. RecorderSink turns False into an
                  ``encoder_backpressure`` health alarm.
        """
        if not self.recording:
            return False
        if not self._is_valid_frame(frame):
            # Account it, LOSSLESS means every captured frame is written
            # or shows up in _drops.tsv; a bare False mislabelled this as
            # encoder backpressure with no drop row.
            _drop_log.record(
                "recorder",
                frame_idx=self.frame_count,
                capture_ts=(capture_time_ns or host_clock.host_ns()) / 1e9,
                reason="invalid_frame",
            )
            return False

        # Canonical capture timestamp: monotonic ns (one clock source, so
        # mp4 durations and tsv timestamps stay consistent). Fall back to
        # the current monotonic reading if the caller didn't supply one.
        if capture_time_ns is None:
            capture_time_ns = host_clock.host_ns()

        if self.annotate_callback is not None:
            # Annotate callback expects seconds, provide
            # monotonic-relative seconds.
            ts_sec_for_annotate = capture_time_ns / 1e9
            annotated = self.annotate_callback(frame, ts_sec_for_annotate)
            if annotated is not None:
                frame = annotated

        with self._queue_lock:
            full = len(self.frame_queue) >= self.frame_queue.maxlen
            if full:
                # Caller (RecorderSink) sees False → fires the
                # encoder_backpressure health alarm.
                _drop_log.record(
                    "recorder",
                    frame_idx=self.frame_count,
                    capture_ts=capture_time_ns / 1e9,
                    reason="queue_full",
                )
                if self.drop_callback is not None:
                    try:
                        self.drop_callback(1, "queue_full")
                    except Exception:
                        pass
                return False
            self.frame_queue.append((frame, capture_time_ns))

        return True

    def _external_record_loop(self):
        """Recording loop, writes frames from the add_frame() queue."""
        _first_frame_logged = False

        logger.info(f"External feed recording loop started: effective FPS={self.fps:.2f}")

        while True:
            frame_data = None

            # Check for frame in queue
            with self._queue_lock:
                if self.frame_queue:
                    frame_data = self.frame_queue.popleft()

            # Exit ONLY once stopped AND drained, stop_recording used
            # to flip ``recording`` and this loop exited with up to
            # maxlen frames still queued: silently lost, no drop row.
            if frame_data is None and not self.recording:
                break

            if frame_data:
                # Guard the whole encode step: a single bad frame or a
                # writer/ffmpeg hiccup must NOT kill the encode thread and
                # leave ``recording`` stuck True (silent video loss). One
                # error is logged + skipped; only sustained failure gives up
                # (with a drop_callback health signal so the box can alarm).
                try:
                    frame, capture_time_ns = frame_data

                    if not self._is_valid_frame(frame):
                        continue

                    frame = self._apply_roi_crop(frame)

                    if not self._is_valid_frame(frame):
                        continue

                    # Resize if needed to match target size. cv2.resize
                    # allocates a fresh contiguous array; when the ROI crop
                    # is already at target size it stays a non-contiguous
                    # view, make it contiguous ONCE here so the writer's
                    # contiguity fallback never fires on the hot path.
                    target_w, target_h = self._target_size
                    if frame.shape[1] != target_w or frame.shape[0] != target_h:
                        frame = cv2.resize(frame, (target_w, target_h))
                    elif not frame.flags['C_CONTIGUOUS']:
                        frame = np.ascontiguousarray(frame)

                    frame = self._coerce_channels(frame, self._grayscale)

                    if (self._actual_encoder == 'h264_nvmpi'
                            and not self._hw_fallback_done):
                        # Before each write, not only after a refused one: see
                        # _recover_dead_nvmpi_writer. No other encoder checks.
                        self._recover_dead_nvmpi_writer(rejected_frame=False)

                    if self.writer.write(frame) is False:
                        # FFmpegVideoWriter.write() returns False (never raises)
                        # for TWO reasons; treat them differently so a slow
                        # disk doesn't end the session:
                        #   - dead writer thread / ffmpeg crash (unhealthy) →
                        #     fatal: raise so the write-error streak trips the
                        #     encoder-dead alarm instead of losing video
                        #     silently.
                        #   - transient full queue (writer still healthy) →
                        #     drop this frame and keep recording, as before.
                        # Writers returning None on success (cv2) never hit
                        # this branch.
                        healthy = getattr(self.writer, "is_healthy", None)
                        if callable(healthy) and not healthy():
                            raise RuntimeError(
                                "writer rejected frame (ffmpeg pipe dead / "
                                "ffmpeg crashed)")
                        # Transient backpressure, account the drop, continue.
                        if self.drop_callback:
                            try:
                                self.drop_callback(1, "queue_full")
                            except Exception:
                                pass
                        continue

                    if not _first_frame_logged:
                        logger.info(f"First frame written: {frame.shape[1]}x{frame.shape[0]} "
                                    f"target={target_w}x{target_h} writer={type(self.writer).__name__}")
                        _first_frame_logged = True

                    self.last_capture_host_ns = capture_time_ns
                    self.frame_count += 1
                    self._write_errors = 0
                except Exception as e:
                    if self._recover_dead_nvmpi_writer():
                        continue
                    self._write_errors += 1
                    if self._write_errors <= 3 or self._write_errors % 100 == 0:
                        logger.error("Box %s: video write error #%d: %s",
                                     self.camera_id, self._write_errors, e)
                    if self._write_errors >= self._WRITE_ERROR_FATAL:
                        logger.error(
                            "Box %s: video encoder failing persistently "
                            "(%d consecutive errors), stopping recorder.",
                            self.camera_id, self._write_errors)
                        if self.drop_callback:
                            try:
                                self.drop_callback(self._write_errors,
                                                   "encoder_write_failed")
                            except Exception:
                                pass
                        self.encoder_failed = True
                        self.recording = False
                        # A dead writer can't consume the drained tail,
                        # discard the remaining queue WITH accounting and
                        # stop, instead of re-failing per frame.
                        with self._queue_lock:
                            n_left = len(self.frame_queue)
                            self.frame_queue.clear()
                        if n_left:
                            _drop_log.record(
                                "recorder", frame_idx=self.frame_count,
                                capture_ts=host_clock.host_ns() / 1e9,
                                reason=f"encoder_failed_discard_x{n_left}")
                        break
                    continue
            else:
                # No frame available, sleep briefly
                time.sleep(0.01)

        logger.info(f"External feed recording loop ended: {self.frame_count} frames written")

    def stop_recording(self):
        """Stop video recording and close files."""
        if not self.recording:
            return

        logger.info(f"Stopping video recording ({self.frame_count} frames so far)...")
        self.recording = False

        # The loop drains the remaining queue after ``recording`` flips,
        # give it room for up to maxlen frames at a slow encode before
        # abandoning (abandonment is logged; frames past it are lost).
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10.0)
            if self.thread.is_alive():
                logger.warning("Video recording thread did not finish within timeout; continuing shutdown.")

        # Close files
        if self.writer:
            self.writer.release()
            self.writer = None

        logger.info(f"Stopped video recording. {self.frame_count} frames captured.")

        if self.frame_strategy == "remux":
            self._postprocess_remux_async()

    def _postprocess_remux_async(self):
        """Run remux in a background thread to avoid blocking the GUI."""
        def _run():
            try:
                self._postprocess_remux()
            except Exception as e:
                logger.warning(f"Remux failed: {e}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        logger.info("Remux scheduled in background")

    def _postprocess_remux(self):
        """Post-process video to align duration with actual capture time.

        Uses a monotonic_ns delta (last - start) so the duration math
        stays on one clock source.
        """
        if not self.video_path or not os.path.exists(self.video_path):
            return
        if not self.rec_start_host_ns or not self.last_capture_host_ns:
            return

        actual_duration = max(0.0,
            (self.last_capture_host_ns - self.rec_start_host_ns) / 1e9)
        if actual_duration <= 0 or self.frame_count < 2:
            return

        actual_fps = self.frame_count / actual_duration
        if actual_fps <= 0:
            return

        if abs(actual_fps - self.fps) < 0.5:
            return

        caps = EncoderCapabilities.get_instance()
        if not caps.has_ffmpeg:
            logger.warning("Remux requested but FFmpeg not available; video duration may be short.")
            return

        base, ext = os.path.splitext(self.video_path)
        temp_path = f"{base}.remux{ext}"
        encoder = caps.get_best_encoder(prefer_gpu=True) or "libx264"

        cmd = [
            "ffmpeg",
            "-y",
            "-r", f"{actual_fps:.3f}",
            "-i", self.video_path,
            "-r", f"{self.fps:.3f}",
            "-vsync", "cfr",
        ]

        # Honour the session's configured quality, a hardcoded 23 here
        # re-encoded a quality_crf=28 recording at a different quality.
        crf = str(int(self._crf))
        if encoder == "h264_nvenc":
            cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", crf, "-pix_fmt", "yuv420p"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", crf, "-pix_fmt", "yuv420p"]

        cmd.append(temp_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
            if result.returncode != 0:
                err = result.stderr.decode(errors="ignore")
                logger.warning(f"Remux failed (ffmpeg): {err[:200]}")
                return
            os.replace(temp_path, self.video_path)
            logger.info(
                f"Remuxed video to target FPS {self.fps} "
                f"(actual {actual_fps:.2f} fps, duration {actual_duration:.2f}s)"
            )
        except Exception as e:
            logger.warning(f"Remux failed: {e}")

    def close(self):
        """Release resources."""
        self.stop_recording()
        logger.info("Video recorder closed")

    def __del__(self):
        """Cleanup on deletion.

        A recorder whose ``__init__`` raised part-way is still collected, and
        it reaches here without the attributes ``close`` reads. The traceback
        that produces is printed by the interpreter from inside garbage
        collection, where it cannot be caught by the caller and does not say
        which recorder it came from: it hides the REAL error, which is
        whatever stopped the constructor.
        """
        if not getattr(self, "_fully_constructed", False):
            return
        try:
            self.close()
        except Exception:
            # Nothing useful can be done during collection, and raising here
            # only produces a second, more confusing report.
            pass
