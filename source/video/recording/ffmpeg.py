"""
FFmpeg Video Writer with GPU (NVENC) Support

This module provides GPU-accelerated H264 video encoding using FFmpeg subprocess
with NVIDIA NVENC, with automatic fallback to CPU (libx264) or OpenCV when GPU
is unavailable.

Classes:
    EncoderCapabilities: Singleton for detecting FFmpeg and NVENC availability
    FFmpegVideoWriter: Video writer using FFmpeg subprocess (mirrors cv2.VideoWriter)
    VideoWriterFactory: Factory for creating appropriate video writer with fallback

Usage:
    writer = VideoWriterFactory.create_writer(
        path="output.mp4",
        fps=30,
        size=(640, 480),
        use_gpu=True,
    )
    writer.write(frame)
    writer.release()
"""

import collections
import subprocess
import threading
import time
import queue
import logging
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from source.video.recording.drop_log import drop_log as _drop_log

logger = logging.getLogger(__name__)


# ── Hardware-encoder session tracking ────────────────────────────────────
# Consumer GeForce drivers cap concurrent NVENC sessions, which a many-box rig
# can exceed. This counter tracks how many hardware-encoder writers are
# currently open so we can (a) log it and
# (b) surface it on the sticky "No video, NVENC session limit" alarm. The
# real admission control is reactive: ``create_writer`` tries the hardware
# encoder and, when it won't open (session limit hit), falls through to CPU
# so the box still records, see ``allow_cpu_fallback``.
_hw_session_lock = threading.Lock()
_hw_sessions = 0


def hw_encoder_sessions() -> int:
    """Number of hardware-encoder (NVENC / v4l2m2m) writers currently open."""
    with _hw_session_lock:
        return _hw_sessions


def _inc_hw_sessions() -> int:
    global _hw_sessions
    with _hw_session_lock:
        _hw_sessions += 1
        return _hw_sessions


def _dec_hw_sessions() -> int:
    global _hw_sessions
    with _hw_session_lock:
        if _hw_sessions > 0:
            _hw_sessions -= 1
        return _hw_sessions


class EncoderCapabilities:
    """Singleton class to detect and cache FFmpeg/NVENC capabilities."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._ffmpeg_available = None
        self._nvenc_available = None
        # Intel Quick Sync and AMD AMF: the vendor-neutral hardware encoders
        # every non-NVIDIA machine actually has. Probed like the rest, a
        # codec compiled into ffmpeg but with no silicon behind it fails the
        # probe, so presence in ``-encoders`` is never taken as availability.
        self._qsv_available = None
        self._amf_available = None
        self._mf_available = None
        self._libx264_available = None
        self._hevc_nvenc_available = None
        self._libx265_available = None
        # Jetson H.264 hardware encoder (NVIDIA's V4L2 M2M). On JetPack the
        # GPU is exposed as a V4L2 mem-to-mem encoder, not the desktop
        # ``h264_nvenc`` codec; stock FFmpeg ships ``h264_v4l2m2m`` for it.
        self._v4l2m2m_available = None
        # Jetson H.264 through NVIDIA's Multimedia API (``h264_nvmpi``, from
        # the jetson-ffmpeg patch). On JetPack 6 the encoder is not a standard
        # V4L2 M2M device, so ``h264_v4l2m2m`` is compiled in but reports
        # "Could not find a valid device"; this is the codec that reaches it.
        self._nvmpi_available = None
        self._is_jetson = None
        self._detection_done = False
        # Serialize concurrent detect_all() callers (boot prewarm,
        # VideoManager init, MainWindowBase _detect_gpu_encoding) so they
        # don't each race to spawn an ffmpeg probe per codec. Later callers
        # wait, then see ``_detection_done`` and return immediately.
        self._detect_all_lock = threading.RLock()
        logger.debug("EncoderCapabilities singleton created")

    @classmethod
    def get_instance(cls) -> 'EncoderCapabilities':
        """Get singleton instance."""
        return cls()

    def detect_all(self):
        """Run all detection checks (call on startup).

        Re-entrant + thread-safe: concurrent callers from different
        threads serialize on ``_detect_all_lock``. The fast double-check
        (outside-then-inside lock) means callers after the first see
        ``_detection_done == True`` and return immediately without
        re-running any subprocesses.
        """
        # Fast-path: skip lock acquisition if detection already done.
        if self._detection_done:
            return
        with self._detect_all_lock:
            # Re-check under the lock, another thread may have
            # completed detection between our two checks.
            if self._detection_done:
                return
            self.detect_ffmpeg()
            if self._ffmpeg_available:
                self.detect_nvenc()
                self.detect_qsv()
                self.detect_amf()
                self.detect_mf()
                self.detect_libx264()
                self.detect_hevc_nvenc()
                self.detect_libx265()
                # Jetson hardware H.264 (only probe on Linux + actual Jetson
                #, saves a 1s subprocess on every other host).
                if self.is_jetson():
                    if not self.detect_nvmpi():
                        self.detect_v4l2m2m()
            self._detection_done = True
            logger.info(
                "Encoder capabilities: FFmpeg=%s H.264[NVENC=%s QSV=%s "
                "AMF=%s MF=%s nvmpi=%s v4l2m2m=%s libx264=%s] "
                "HEVC[NVENC=%s libx265=%s] Jetson=%s",
                self._ffmpeg_available,
                self._nvenc_available, self._qsv_available,
                self._amf_available, self._mf_available,
                self._nvmpi_available, self._v4l2m2m_available,
                self._libx264_available, self._hevc_nvenc_available,
                self._libx265_available, self._is_jetson,
            )

    def is_jetson(self) -> bool:
        """True if running on an NVIDIA Jetson (Nano / Xavier / Orin / etc.).

        Reads ``/proc/device-tree/model`` which Jetson kernels populate
        with strings like ``"NVIDIA Jetson Orin Nano Developer Kit"``.
        Non-Jetson Linux hosts don't expose this node. Cached after first
        probe.
        """
        if self._is_jetson is not None:
            return self._is_jetson
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip('\x00').strip().lower()
            self._is_jetson = ('jetson' in model or 'tegra' in model
                               or 'nvidia' in model)
        except (OSError, IOError):
            self._is_jetson = False
        return self._is_jetson

    def detect_v4l2m2m(self) -> bool:
        """Check if ``h264_v4l2m2m`` (Jetson hardware H.264) is available.

        Stock FFmpeg on JetPack 5.x / 6.x ships with this encoder, which
        uses Jetson's NVIDIA Multimedia API via the V4L2 mem-to-mem
        interface. Throughput at 1080p: ~60 fps on Orin Nano vs libx264
        at ~2-3 fps. Only meaningful on Linux + Jetson hardware.
        """
        if self._v4l2m2m_available is not None:
            return self._v4l2m2m_available
        if not self.detect_ffmpeg():
            self._v4l2m2m_available = False
            return False
        # Only probe on Jetson, h264_v4l2m2m on non-Jetson Linux can
        # exist (some embedded systems) but our writer config (bitrate
        # 5M, no CRF, no preset) is tuned for the Jetson encoder.
        if not self.is_jetson():
            self._v4l2m2m_available = False
            return False
        self._v4l2m2m_available = self._probe_encoder('h264_v4l2m2m')
        if self._v4l2m2m_available:
            logger.info(
                "Jetson hardware H.264 (h264_v4l2m2m) encoder available"
            )
        else:
            logger.warning(
                "Jetson detected but neither h264_nvmpi nor h264_v4l2m2m "
                "works in the FFmpeg on PATH, so recording falls back to "
                "CPU libx264. On JetPack 6 h264_v4l2m2m cannot reach the "
                "encoder: install the FFmpeg with h264_nvmpi (docs: Install "
                "on NVIDIA Jetson, step 10) and start the application from "
                "that environment."
            )
        return self._v4l2m2m_available

    #: Smallest frame side the Jetson encoder takes. Measured on an AGX Orin
    #: (JetPack 6.2): 160x120, 176x144 and 240x96 encode; 64x64, 128x128,
    #: 144x144 and 96x240 are rejected, and the jetson-ffmpeg wrapper then
    #: spins forever instead of failing. So neither the probe nor a writer
    #: may hand it anything smaller.
    NVMPI_MIN_SIDE = 160

    def detect_nvmpi(self) -> bool:
        """Check if ``h264_nvmpi`` (Jetson hardware H.264) is available.

        Comes from the jetson-ffmpeg patch (github.com/Keylost/jetson-ffmpeg)
        and drives the encoder through NVIDIA's Multimedia API, which is what
        reaches the hardware on JetPack 6. Probed rather than listed, like
        every other hardware codec here. Only meaningful on a Jetson.
        """
        if self._nvmpi_available is not None:
            return self._nvmpi_available
        if not self.detect_ffmpeg() or not self.is_jetson():
            self._nvmpi_available = False
            return False
        # Not the shared 64x64 probe: below NVMPI_MIN_SIDE the encoder hangs,
        # the 10 s probe timeout kills it, and a working encoder reads as
        # absent.
        self._nvmpi_available = self._probe_encoder('h264_nvmpi', size='320x240')
        if self._nvmpi_available:
            logger.info("Jetson hardware H.264 (h264_nvmpi) encoder available")
        return self._nvmpi_available

    def _run_encoder_probe(self, codec_name, size='64x64'):
        """Run a tiny ffmpeg encode to verify a codec works on this machine.

        Returns (ok, stderr_text), stderr lets callers sniff WHY a probe
        failed (e.g. NVENC's "no GPU/driver" vs a generic error).
        """
        if not self.detect_ffmpeg():
            return False, ''
        try:
            result = subprocess.run(
                ['ffmpeg', '-f', 'lavfi', '-i', f'nullsrc=s={size}:d=0.1',
                 '-c:v', codec_name, '-f', 'null', '-'],
                capture_output=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            return result.returncode == 0, result.stderr.decode(errors='ignore')
        except subprocess.TimeoutExpired:
            logger.warning(f"{codec_name} detection timed out")
            return False, ''
        except Exception as e:
            logger.debug(f"{codec_name} detection error: {e}")
            return False, ''

    def _probe_encoder(self, codec_name, size='64x64'):
        """Run a tiny ffmpeg encode to verify a codec works on this machine."""
        ok, _ = self._run_encoder_probe(codec_name, size)
        return ok

    def detect_ffmpeg(self) -> bool:
        """Check if FFmpeg is available in PATH."""
        if self._ffmpeg_available is not None:
            return self._ffmpeg_available

        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            self._ffmpeg_available = result.returncode == 0
            if self._ffmpeg_available:
                # Extract version info
                version_line = result.stdout.decode().split('\n')[0]
                logger.info(f"FFmpeg detected: {version_line}")
            else:
                logger.warning("FFmpeg not found or returned error")
        except FileNotFoundError:
            self._ffmpeg_available = False
            logger.warning("FFmpeg not found in PATH")
        except subprocess.TimeoutExpired:
            self._ffmpeg_available = False
            logger.warning("FFmpeg detection timed out")
        except Exception as e:
            self._ffmpeg_available = False
            logger.warning(f"FFmpeg detection error: {e}")

        return self._ffmpeg_available

    def detect_nvenc(self) -> bool:
        """Check if NVIDIA NVENC encoder is available."""
        if self._nvenc_available is not None:
            return self._nvenc_available

        ok, stderr = self._run_encoder_probe('h264_nvenc')
        self._nvenc_available = ok
        if ok:
            logger.info("NVIDIA NVENC encoder available")
        elif 'Cannot load' in stderr or 'not found' in stderr.lower():
            logger.info("NVENC not available (no NVIDIA GPU or driver)")
        elif stderr:
            logger.debug(f"NVENC test failed: {stderr[:200]}")

        return self._nvenc_available

    def detect_qsv(self) -> bool:
        """Check if Intel Quick Sync (``h264_qsv``) works on this machine.

        Present on every Intel CPU with integrated graphics, which is most
        lab machines. Fixed-function silicon: it encodes without touching the
        cores the GUI and the tracker are using.
        """
        if self._qsv_available is not None:
            return self._qsv_available
        ok, stderr = self._run_encoder_probe('h264_qsv')
        self._qsv_available = ok
        if ok:
            logger.info("Intel Quick Sync (h264_qsv) encoder available")
        elif stderr:
            logger.debug("QSV probe failed: %s", stderr[-200:])
        return self._qsv_available

    def detect_amf(self) -> bool:
        """Check if AMD AMF (``h264_amf``) works on this machine."""
        if self._amf_available is not None:
            return self._amf_available
        ok, stderr = self._run_encoder_probe('h264_amf')
        self._amf_available = ok
        if ok:
            logger.info("AMD AMF (h264_amf) encoder available")
        elif stderr:
            logger.debug("AMF probe failed: %s", stderr[-200:])
        return self._amf_available

    def detect_mf(self) -> bool:
        """Check if Windows MediaFoundation (``h264_mf``) works.

        The vendor-neutral last hardware resort on Windows: it binds whatever
        the OS exposes when neither QSV nor AMF probe clean.
        """
        if self._mf_available is not None:
            return self._mf_available
        if os.name != 'nt':
            self._mf_available = False
            return False
        ok, stderr = self._run_encoder_probe('h264_mf')
        self._mf_available = ok
        if ok:
            logger.info("Windows MediaFoundation (h264_mf) encoder available")
        elif stderr:
            logger.debug("MF probe failed: %s", stderr[-200:])
        return self._mf_available

    def detect_libx264(self) -> bool:
        """Check if libx264 encoder is available."""
        if self._libx264_available is not None:
            return self._libx264_available

        self._libx264_available = self._probe_encoder('libx264')
        if self._libx264_available:
            logger.info("libx264 encoder available")

        return self._libx264_available

    def detect_hevc_nvenc(self) -> bool:
        """Check if NVIDIA HEVC NVENC encoder is available (H.265)."""
        if self._hevc_nvenc_available is not None:
            return self._hevc_nvenc_available

        self._hevc_nvenc_available = self._probe_encoder('hevc_nvenc')
        if self._hevc_nvenc_available:
            logger.info("NVIDIA HEVC NVENC encoder available")

        return self._hevc_nvenc_available

    def detect_libx265(self) -> bool:
        """Check if libx265 (CPU HEVC) encoder is available."""
        if self._libx265_available is not None:
            return self._libx265_available

        self._libx265_available = self._probe_encoder('libx265')
        if self._libx265_available:
            logger.info("libx265 encoder available")

        return self._libx265_available

    def get_best_encoder(self, prefer_gpu: bool = True, prefer_hevc: bool = False) -> Optional[str]:
        """Get the best available encoder.

        Args:
            prefer_gpu: If True, prefer hardware encoders over libx264

        Returns:
            Encoder name ('h264_nvenc' / 'h264_v4l2m2m' / 'libx264', or None)
        """
        if not self.detect_ffmpeg():
            return None

        # HEVC path (smaller files at same quality, slower decode)
        if prefer_hevc:
            if prefer_gpu and self.detect_hevc_nvenc():
                return 'hevc_nvenc'
            if self.detect_libx265():
                return 'libx265'
            # Fall through to H.264 if HEVC not available

        # H.264 path (default, broadest compatibility). Hardware first, in
        # order of how well the encoder is understood on this project's rigs:
        # NVENC (desktop NVIDIA) → QSV (Intel iGPU) → AMF (AMD) → MediaFoundation
        # (Windows, vendor-neutral) → v4l2m2m (Jetson). Anything here beats
        # libx264, which on a 15 W laptop CPU stalls the capture pipe for
        # seconds and drops a third of the session.
        if prefer_gpu and self.detect_nvenc():
            return 'h264_nvenc'
        if prefer_gpu and self.detect_qsv():
            return 'h264_qsv'
        if prefer_gpu and self.detect_amf():
            return 'h264_amf'
        if prefer_gpu and self.detect_mf():
            return 'h264_mf'
        if prefer_gpu and self.detect_nvmpi():
            return 'h264_nvmpi'
        if prefer_gpu and self.detect_v4l2m2m():
            return 'h264_v4l2m2m'

        if self.detect_libx264():
            return 'libx264'

        return None

    @property
    def has_gpu(self) -> bool:
        """Any hardware encoder at all, NVIDIA, Intel, AMD, Windows MF or Jetson."""
        return (self.detect_nvenc() or self.detect_qsv() or self.detect_amf()
                or self.detect_mf() or self.detect_nvmpi()
                or self.detect_v4l2m2m())

    @property
    def has_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        return self.detect_ffmpeg()

    def gpu_force_required(self) -> bool:
        """True when ANY hardware encoder is present on this host.

        The pipeline policy: when a hardware H.264 encoder is available
        (desktop NVENC or Jetson v4l2m2m) it is tried first. Whether a failed
        hardware open falls back to CPU (libx264 / OpenCV) is governed by
        ``allow_cpu_fallback`` in ``VideoWriterFactory.create_writer`` (default
        True) and the per-recorder picker in ``recorder.py``. Use this to detect
        that a hardware encoder is present; CPU-only hosts have none.
        """
        return (self.detect_nvenc() or self.detect_hevc_nvenc()
                or self.detect_qsv() or self.detect_amf() or self.detect_mf()
                or self.detect_nvmpi() or self.detect_v4l2m2m())


class FFmpegVideoWriter:
    """Video writer using FFmpeg subprocess.

    Mirrors cv2.VideoWriter interface for drop-in replacement.
    Pipes raw BGR frames to FFmpeg for encoding.
    """

    def __init__(self, path: str, fps: float, size: Tuple[int, int],
                 encoder: str = 'h264_nvenc', crf: int = 23, preset: str = 'fast',
                 grayscale: bool = False):
        """Initialize FFmpeg video writer.

        Args:
            path: Output video file path (.mp4)
            fps: Frame rate
            size: Frame size as (width, height)
            encoder: FFmpeg encoder ('h264_nvenc' or 'libx264')
            crf: Constant rate factor (0-51, lower = better quality)
            preset: Encoding preset ('fast', 'medium', 'slow' for libx264;
                   'p1'-'p7' for NVENC, p4 is balanced)
            grayscale: If True, expect single-channel input (pix_fmt gray)
        """
        self.path = path
        self.fps = fps
        self.width, self.height = size
        self.encoder = encoder
        self.crf = crf
        self.preset = preset
        self.grayscale = grayscale

        # Ensure even dimensions (required for H264)
        self.width = self.width if self.width % 2 == 0 else self.width + 1
        self.height = self.height if self.height % 2 == 0 else self.height + 1

        self._process = None
        # Deep enough to absorb a hardware encoder's first-write stall.
        # Measured on this rig at 720x700: h264_qsv blocks 1.5 s on its first
        # frame while the session initialises, h264_amf 0.4 s. A 30-frame
        # queue is 1.5 s at 20 fps, no margin at all, so the very first
        # seconds of every hardware-encoded recording dropped frames. 90
        # frames matches the recorder sink's own depth.
        self._frame_queue = queue.Queue(maxsize=90)
        self._writer_thread = None
        self._stop_event = threading.Event()
        self._opened = False
        self._error = None
        self._frames_written = 0
        # Rate limit for the queue-depth warning above.
        self._last_depth_warn = 0.0

        self._start_ffmpeg()

    def _build_command(self) -> list:
        """Build FFmpeg command line."""
        # Per-encoder CPU thread cap (half the cores, min 1) so parallel
        # software encodes don't each grab all cores. GPU encoders are
        # session-bound, not thread-bound, so they don't get this.
        cpu_threads = max(1, (os.cpu_count() or 4) // 2)
        input_pix_fmt = 'gray' if self.grayscale else 'bgr24'
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', input_pix_fmt,
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', 'pipe:0',  # Read from stdin
        ]

        if self.encoder == 'h264_nvenc':
            cmd.extend([
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',
                '-cq', str(self.crf),
                '-pix_fmt', 'yuv420p',
            ])
        elif self.encoder == 'h264_qsv':
            # Intel Quick Sync. global_quality is QSV's CRF analogue; the
            # low-power path is deliberately not requested; it is unavailable
            # on several iGPU generations and makes the open fail rather than
            # degrade.
            cmd.extend([
                '-c:v', 'h264_qsv',
                '-preset', 'veryfast',
                '-global_quality', str(self.crf),
                '-pix_fmt', 'nv12',
            ])
        elif self.encoder == 'h264_amf':
            # AMD AMF. Constant-QP rather than CQVBR: on a capture pipe the
            # rate controller must never stall waiting to hit a bitrate.
            cmd.extend([
                '-c:v', 'h264_amf',
                '-quality', 'speed',
                '-rc', 'cqp',
                '-qp_i', str(self.crf), '-qp_p', str(self.crf),
                '-pix_fmt', 'yuv420p',
            ])
        elif self.encoder == 'h264_mf':
            # Windows MediaFoundation, whatever hardware the OS exposes.
            cmd.extend([
                '-c:v', 'h264_mf',
                '-rate_control', 'quality',
                '-quality', str(self.crf),
                '-pix_fmt', 'yuv420p',
            ])
        elif self.encoder == 'h264_nvmpi':
            # Jetson hardware H.264 via NVIDIA's Multimedia API (jetson-ffmpeg).
            # No CRF, so a 5 Mbps target like the v4l2m2m path below, plus a
            # quantiser floor. Without it one very noisy frame (measured: grey
            # +/-48 sensor noise at 240x240) fails inside the Orin's encoder
            # driver and the wrapper then blocks for good, so that box stops
            # recording. QP 20 prevents it, and on 60 s of real box video it
            # matched this rig's libx264 settings (SSIM 0.976 vs 0.968,
            # 1.29 vs 1.19 MB).
            cmd.extend([
                '-c:v', 'h264_nvmpi',
                '-b:v', '5M',
                '-qmin', '20',
                '-qmax', '51',
                '-pix_fmt', 'yuv420p',
            ])
        elif self.encoder == 'h264_v4l2m2m':
            # Jetson hardware H.264 via V4L2 mem-to-mem. Uses NVIDIA's
            # Multimedia API kernel driver. Doesn't expose CRF, uses
            # constant-bitrate instead. 5 Mbps targets ~CRF 23 quality
            # at 1080p. Pix-fmt MUST be yuv420p (the v4l2m2m kernel
            # driver only accepts NV12/I420 input).
            cmd.extend([
                '-c:v', 'h264_v4l2m2m',
                '-b:v', '5M',
                '-pix_fmt', 'yuv420p',
            ])
        elif self.encoder == 'hevc_nvenc':
            # NVIDIA HEVC (H.265) NVENC, ~40% smaller than H.264 at same quality
            cmd.extend([
                '-c:v', 'hevc_nvenc',
                '-preset', 'p4',
                '-cq', str(self.crf),
                '-pix_fmt', 'yuv420p',
                '-tag:v', 'hvc1',  # Mac/QuickTime compatibility
            ])
        elif self.encoder == 'libx265':
            # CPU HEVC, smaller files, slower encode
            cmd.extend([
                '-c:v', 'libx265',
                '-preset', self.preset,
                '-crf', str(self.crf),
                '-threads', str(cpu_threads),
                '-pix_fmt', 'yuv420p',
                '-tag:v', 'hvc1',
            ])
        else:
            # libx264 default
            # A capture pipe is not a transcode: the encoder must never be
            # the reason a frame is dropped. ultrafast + zerolatency + no
            # B-frames trades file size for the ability to keep up, and a
            # 2-second GOP keeps the file seekable. Measured on this rig,
            # -preset fast stalled the pipe for up to 7.9 s and lost 33 % of
            # the session; the frames matter and the bytes do not.
            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-bf', '0',
                '-g', str(max(1, int(round(self.fps * 2)))),
                '-crf', str(self.crf),
                '-threads', str(cpu_threads),
                '-pix_fmt', 'yuv420p',
            ])

        cmd.append(self.path)
        return cmd

    def _start_ffmpeg(self):
        """Start FFmpeg subprocess."""
        try:
            cmd = self._build_command()
            logger.debug(f"Starting FFmpeg: {' '.join(cmd)}")

            # Start FFmpeg process
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                # DEVNULL, not PIPE: nothing drains stdout (output goes to a
                # file), so a PIPE would be a latent deadlock if ffmpeg ever
                # wrote to it, and one OS pipe handle per box.
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=10**8,  # Large buffer for performance
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            # Background stderr drainer. FFmpeg writes diagnostics to stderr
            # continuously; if nothing reads it the OS pipe buffer fills and
            # the process deadlocks. Drain into a bounded ring so release()
            # can log the tail on non-zero exit.
            self._stderr_tail = collections.deque(maxlen=128)  # last 128 lines
            self._stderr_drainer = threading.Thread(
                target=self._drain_stderr, daemon=True,
                name=f"ffmpeg-stderr-{Path(self.path).stem}",
            )
            self._stderr_drainer.start()

            # Start writer thread
            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer_thread.start()

            self._opened = True
            logger.info(f"FFmpeg started: {self.encoder} -> {self.path}")

        except Exception as e:
            self._error = str(e)
            self._opened = False
            logger.error(f"Failed to start FFmpeg: {e}")

    def _drain_stderr(self):
        """Read ffmpeg stderr line-by-line into a bounded ring buffer;
        exits when stderr EOFs (ffmpeg has exited). Keeps the OS pipe from
        filling and deadlocking the encode.
        """
        try:
            stderr = self._process.stderr
            if stderr is None:
                return
            for raw in iter(stderr.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    self._stderr_tail.append(line)
        except Exception as e:
            logger.debug(f"ffmpeg stderr drainer ended: {e}")

    def _writer_loop(self):
        """Background thread to write frames to FFmpeg.

        Queue items are either a numpy array (one frame) or None (poison
        pill to stop).
        """
        while not self._stop_event.is_set():
            try:
                item = self._frame_queue.get(timeout=0.1)
                if item is None:  # Poison pill
                    break

                frame = item
                # Safety net for non-recorder feeds: VideoRecorder already
                # delivers target-size contiguous frames, so neither branch
                # fires on the recording hot path.
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)
                # Write via buffer protocol (zero-copy). Time the call so a
                # stalled pipe (disk full, slow storage, ffmpeg starved)
                # surfaces a warning instead of hanging silently.
                _t0 = time.monotonic()
                self._process.stdin.write(frame.data)
                _dt = time.monotonic() - _t0
                if _dt > 1.0:
                    logger.error(
                        "FFmpeg write took %.2f s, pipe likely stalled "
                        "(disk full / slow remote storage / ffmpeg "
                        "starved). Recording will stutter until the "
                        "blockage clears.", _dt)
                self._frames_written += 1

            except queue.Empty:
                continue
            except BrokenPipeError:
                logger.error("FFmpeg pipe broken")
                self._error = "Pipe broken"
                break
            except Exception as e:
                logger.error(f"Writer thread error: {e}")
                self._error = str(e)
                break

    def write(self, frame: np.ndarray) -> bool:
        """Write a frame to the video.

        Args:
            frame: BGR image as numpy array

        Returns:
            True if frame was queued successfully
        """
        if not self._opened or self._process is None:
            return False

        # Writer thread died (broken pipe, ffmpeg crash, disk full), refuse
        # frames so the caller sees sustained failure and can alarm, instead
        # of silently filling a queue that will never drain.
        if not self.is_healthy():
            return False

        try:
            self._frame_queue.put(frame, timeout=0.5)
            # Warn while the queue is merely FILLING, not once it has already
            # overflowed. By the time frames are dropped the session has lost
            # data; a run that is quietly climbing toward the ceiling is still
            # recoverable, lower the resolution, or accept the drops
            # knowingly. Rate-limited so a struggling encoder cannot spam.
            depth = self._frame_queue.qsize()
            cap = self._frame_queue.maxsize
            if cap and depth * 2 >= cap:
                now = time.time()
                if now - self._last_depth_warn > 5.0:
                    self._last_depth_warn = now
                    logger.warning(
                        "Encoder queue %d/%d full for %s, %s is not keeping "
                        "up with %.1f fps. Frames will be dropped if this "
                        "continues.", depth, cap, Path(self.path).name,
                        self.encoder, self.fps)
            return True
        except queue.Full:
            # Account for it: this was the one drop path in the encoder with no
            # drop-log entry, so a session that lost frames here still produced
            # a clean <session>/_drops.tsv. Distinct stream name from the
            # recorder's own "queue_full" so the two are told apart.
            logger.warning("Frame queue full, dropping frame")
            try:
                _drop_log.record("encoder", reason="write_queue_full")
            except Exception as e:
                logger.debug("drop-log record failed: %s", e)
            return False

    def is_healthy(self) -> bool:
        """True while the writer can still accept frames, opened, no recorded
        error, and the writer thread alive. Lets the recorder tell a transient
        full queue (drop-and-continue) from a dead pipe (fatal)."""
        return bool(
            self._opened
            and self._process is not None
            and self._error is None
            and self._writer_thread is not None
            and self._writer_thread.is_alive())

    def release(self):
        """Release resources and finish encoding."""
        if not self._opened:
            return

        self._opened = False

        # Decrement the process-wide hardware-session counter exactly once
        # for a hardware writer, so admission-control accounting stays right
        # even if release() is somehow called twice.
        if getattr(self, "_is_hardware", False) and not getattr(self, "_hw_released", False):
            self._hw_released = True
            _dec_hw_sessions()

        try:
            # Signal writer thread to stop
            self._stop_event.set()
            self._frame_queue.put(None)  # Poison pill

            # Wait for writer thread
            if self._writer_thread and self._writer_thread.is_alive():
                self._writer_thread.join(timeout=5)

            # Close FFmpeg stdin and wait
            if self._process:
                if self._process.stdin:
                    try:
                        self._process.stdin.close()
                    except Exception:
                        pass

                # Wait for FFmpeg to finish; escalate terminate → kill.
                # The second wait can also time out, without the kill a
                # hung ffmpeg survived release() as a zombie holding the
                # output file open.
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("FFmpeg didn't finish in time, terminating")
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.error("FFmpeg ignored terminate, killing")
                        self._process.kill()
                        self._process.wait(timeout=5)

                # Drainer thread reads stderr line-by-line into
                # ``self._stderr_tail`` (ring buffer of last 128 lines).
                # Once ffmpeg has exited the pipe EOFs and the drainer
                # returns, give it a short window to flush the tail.
                drainer = getattr(self, "_stderr_drainer", None)
                if drainer is not None and drainer.is_alive():
                    drainer.join(timeout=2)

                # Check for errors
                if self._process.returncode != 0:
                    tail = "\n".join(getattr(self, "_stderr_tail", []) or [])
                    logger.warning(
                        f"FFmpeg exited with code {self._process.returncode} "
                        f"({self._frames_written} frames written). "
                        f"Last stderr lines:\n{tail[-2000:]}"
                    )
                else:
                    logger.info(f"FFmpeg finished: {self._frames_written} frames written to {self.path}")

        except Exception as e:
            logger.error(f"Error releasing FFmpeg writer: {e}")

    def isOpened(self) -> bool:
        """Check if writer is opened and ready."""
        return self._opened and self._process is not None and self._process.poll() is None

#: How long a Jetson test encode may take before it counts as a failure.
_NVMPI_PROBE_TIMEOUT_S = 8.0


def _nvmpi_encodes_at(size: Tuple[int, int]) -> bool:
    """Whether ``h264_nvmpi`` encodes a short clip at ``size`` right now.

    A Jetson hardware encoder that cannot start usually says so at once: the
    library is missing, the encoder is out of sessions, or it refuses the size.
    Inside a recording that only shows after the first frames are piped in, and
    the box then has no video. A 0.2 s encode at the box's own size, before the
    real writer opens, turns that into a choice of encoder instead.

    Killed rather than asked to stop on timeout: the jetson-ffmpeg wrapper
    ignores SIGTERM once the encoder has failed. A hung test counts as failed.
    """
    w = int(size[0]) + int(size[0]) % 2
    h = int(size[1]) + int(size[1]) % 2
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-v', 'error',
             '-f', 'lavfi', '-i', f'nullsrc=s={w}x{h}:d=0.2',
             '-pix_fmt', 'yuv420p', '-c:v', 'h264_nvmpi',
             '-b:v', '5M', '-qmin', '20', '-qmax', '51',
             '-f', 'null', '-'],
            stdin=subprocess.DEVNULL, capture_output=True,
            timeout=_NVMPI_PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        logger.warning("h264_nvmpi did not finish a %dx%d test encode within "
                       "%.0f s", w, h, _NVMPI_PROBE_TIMEOUT_S)
        return False
    except Exception as e:
        logger.warning("h264_nvmpi test encode could not run: %s", e)
        return False
    if result.returncode != 0:
        tail = (result.stderr or b"").decode(errors="ignore").strip()
        logger.warning("h264_nvmpi failed a %dx%d test encode (exit %s): %s",
                       w, h, result.returncode,
                       " | ".join(tail.splitlines()[-3:]))
        return False
    return True


class VideoWriterFactory:
    """Factory for creating video writers with automatic fallback."""

    @staticmethod
    def _try_ffmpeg(path_obj: Path, fps: float, size: Tuple[int, int],
                    encoder: str, crf: int, grayscale: bool,
                    label: str, is_hardware: bool = False) -> Optional[FFmpegVideoWriter]:
        """Build + open an FFmpegVideoWriter; return it on success, None
        on failure (releases the half-open writer for the caller).

        ``is_hardware`` marks NVENC / v4l2m2m writers so the process-wide
        hardware-session counter is incremented on open and decremented on
        release (see ``hw_encoder_sessions``)."""
        try:
            mp4_path = path_obj.with_suffix('.mp4')
            # Jetson only: h264_nvmpi exists nowhere else, so no other encoder
            # pays for, or is changed by, this check.
            if encoder == 'h264_nvmpi' and not _nvmpi_encodes_at(size):
                logger.warning("Not using %s for %s: its test encode failed, "
                               "so the next encoder is tried.", label, mp4_path)
                return None
            writer = FFmpegVideoWriter(
                str(mp4_path), fps, size,
                encoder=encoder, crf=crf, grayscale=grayscale,
            )
            if writer.isOpened():
                if is_hardware:
                    writer._is_hardware = True
                    n = _inc_hw_sessions()
                    logger.info("Using %s for %s (%d hardware encode session[s] open)",
                                label, mp4_path, n)
                else:
                    logger.info(f"Using {label} for {mp4_path}")
                return writer
            writer.release()
        except Exception as e:
            logger.warning(f"{encoder} failed: {e}")
        return None

    @staticmethod
    def create_writer(path: str, fps: float, size: Tuple[int, int],
                     use_gpu: bool = True,
                     crf: int = 23, grayscale: bool = False,
                     prefer_hevc: bool = False,
                     allow_cpu_fallback: bool = True) -> Optional[object]:
        """Create a video writer.

        GPU policy: when a hardware H.264 encoder is detected (desktop
        NVENC or Jetson v4l2m2m) it is tried FIRST. If it won't open, the
        common cause on a many-box rig is the consumer-GPU NVENC concurrent-
        session cap (~12) being exceeded by boxes 13-16, behaviour depends
        on ``allow_cpu_fallback``:

          * ``True`` (default, "rock solid"): fall through to CPU libx264 so
            the box STILL records. A loud warning is logged so the operator
            knows this box is on CPU. Losing a fast encoder beats losing the
            video entirely.
          * ``False`` (strict): refuse and return ``None``, for operators who
            require every box on the hardware encoder.

        Fallback chain when no hardware encoder is present, or after a
        hardware failure with ``allow_cpu_fallback``:
            prefer_hevc=True  : libx265 -> libx264 -> XVID
            prefer_hevc=False : libx264 -> XVID
        """
        caps = EncoderCapabilities.get_instance()
        path_obj = Path(path)
        gpu_forced = bool(use_gpu) and caps.gpu_force_required()

        # ── HARDWARE ENCODER PATH ─────────────────────────────────
        if gpu_forced:
            if prefer_hevc and caps.detect_hevc_nvenc():
                writer = VideoWriterFactory._try_ffmpeg(
                    path_obj, fps, size, 'hevc_nvenc', crf, grayscale,
                    'GPU HEVC (hevc_nvenc)', is_hardware=True)
                if writer is not None:
                    return writer
            if caps.detect_nvenc():
                writer = VideoWriterFactory._try_ffmpeg(
                    path_obj, fps, size, 'h264_nvenc', crf, grayscale,
                    'GPU H.264 (h264_nvenc)', is_hardware=True)
                if writer is not None:
                    return writer
            if caps.detect_qsv():
                writer = VideoWriterFactory._try_ffmpeg(
                    path_obj, fps, size, 'h264_qsv', crf, grayscale,
                    'Intel Quick Sync (h264_qsv)', is_hardware=True)
                if writer is not None:
                    return writer
            if caps.detect_amf():
                writer = VideoWriterFactory._try_ffmpeg(
                    path_obj, fps, size, 'h264_amf', crf, grayscale,
                    'AMD AMF (h264_amf)', is_hardware=True)
                if writer is not None:
                    return writer
            if caps.detect_mf():
                writer = VideoWriterFactory._try_ffmpeg(
                    path_obj, fps, size, 'h264_mf', crf, grayscale,
                    'Windows MediaFoundation (h264_mf)', is_hardware=True)
                if writer is not None:
                    return writer
            if (caps.detect_nvmpi()
                    and min(size) >= EncoderCapabilities.NVMPI_MIN_SIDE):
                # Jetson hardware encoder via NVIDIA's Multimedia API. A box
                # smaller than the encoder's minimum records on the CPU
                # instead: given such a frame, the encoder never returns.
                writer = VideoWriterFactory._try_ffmpeg(
                    path_obj, fps, size, 'h264_nvmpi', crf, grayscale,
                    'Jetson HW H.264 (h264_nvmpi)', is_hardware=True)
                if writer is not None:
                    return writer
            if caps.detect_v4l2m2m():
                # Jetson hardware encoder via V4L2 M2M.
                writer = VideoWriterFactory._try_ffmpeg(
                    path_obj, fps, size, 'h264_v4l2m2m', crf, grayscale,
                    'Jetson HW H.264 (h264_v4l2m2m)', is_hardware=True)
                if writer is not None:
                    return writer
            if not allow_cpu_fallback:
                logger.error(
                    "Hardware encoder required (NVENC / QSV / AMF / MF / "
                    "v4l2m2m detected) but no writer would open, and CPU "
                    "fallback is disabled. Check the GPU driver, the encoder "
                    "session limit, or codec session conflicts.")
                return None
            # "Rock solid" admission control: HW encoder busy/failed (most
            # often the NVENC session cap at %d open sessions), record on
            # CPU so this box is not lost. Falls through to the CPU chain.
            logger.warning(
                "Hardware encoder would not open (%d HW session[s] already "
                "open, likely the NVENC concurrent-session cap). Falling "
                "back to CPU libx264 for %s so this box still records.",
                hw_encoder_sessions(), path_obj.name)

        # ── CPU PATH (no GPU on this host) ────────────────────────
        if prefer_hevc and caps.detect_libx265():
            writer = VideoWriterFactory._try_ffmpeg(
                path_obj, fps, size, 'libx265', crf, grayscale,
                'CPU HEVC (libx265)')
            if writer is not None:
                return writer
        if caps.detect_libx264():
            writer = VideoWriterFactory._try_ffmpeg(
                path_obj, fps, size, 'libx264', crf, grayscale,
                'CPU H.264 (libx264)')
            if writer is not None:
                return writer
        try:
            avi_path = path_obj.with_suffix('.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(str(avi_path), fourcc, fps, size,
                                    isColor=not grayscale)
            if writer.isOpened():
                logger.info(f"Using OpenCV (XVID) encoder for {avi_path}")
                return writer
        except Exception as e:
            logger.error(f"OpenCV VideoWriter failed: {e}")

        logger.error("No video encoder available")
        return None
