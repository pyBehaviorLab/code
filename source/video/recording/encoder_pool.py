"""Encoder pre-warmup, eliminates the 100–300 ms first-write penalty.

Profiling shows that the FIRST ``cv2.VideoWriter`` / FFmpeg NVENC writer
created in a Python process pays a 100–300 ms cost while Windows loads
the codec DLLs (MediaFoundation, nvEncodeAPI64.dll). Subsequent writers
in the same process are ~10–20 ms each.

When 8–16 setups click Record at once, the first per-box codec init
stacks on the GUI thread and produces the visible spike. Calling
``warm_video_encoders()`` once at project-load forces those DLLs in
before any box ever clicks Record. After warmup, per-box writer init
is fast enough to run on a worker thread and the visible spike
disappears.

This module is intentionally tiny, no pool of *reusable* writers
(they own their target path and can't be rebound), just a one-shot
DLL primer.

Usage::

    from source.video.recording.encoder_pool import warm_video_encoders
    warm_video_encoders()          # idempotent; safe to call repeatedly
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from typing import Optional, Tuple

import numpy as np

from source.video.recording.ffmpeg import (
    EncoderCapabilities,
    VideoWriterFactory,
)

logger = logging.getLogger(__name__)


_WARM_LOCK = threading.Lock()
_WARM_DONE: bool = False


def warm_video_encoders(
    *,
    size: Tuple[int, int] = (640, 480),
    fps: float = 30.0,
    use_gpu: Optional[bool] = None,
) -> bool:
    """Force codec DLL load by opening a throwaway 1-frame writer.

    Idempotent: returns immediately on subsequent calls.

    Args
    ----
    size:
        Frame size for the throwaway writer. Not tiny: a hardware encoder
        allocates its session on the first real frame, and that stall was
        measured at 1.5 s (QSV) / 0.4 s (AMF) on this rig. Paying it here,
        at a realistic size, is the difference between a clean start and a
        burst of dropped frames in the first seconds of every recording.
    fps:
        Throwaway target fps; irrelevant beyond writer init.
    use_gpu:
        ``None`` (default) → auto-pick (NVENC if available, else libx264
        / MJPEG). ``True`` / ``False`` force the path that the rig will
        actually use, so the warmup primes the right DLL.

    Returns
    -------
    bool
        True on success, False on any failure. A failure here does not
        crash the app, recording will still work, just without the
        DLL-load amortisation.
    """
    global _WARM_DONE
    with _WARM_LOCK:
        if _WARM_DONE:
            return True
        t0 = time.monotonic()

        # Touch the EncoderCapabilities singleton; cheap subprocess probes
        # of ffmpeg / nvidia-smi run here once.
        try:
            EncoderCapabilities.get_instance().detect_all()
        except Exception as e:
            logger.debug("EncoderCapabilities.detect_all() failed: %s", e)

        # Auto-pick: try GPU first if available, else CPU. ``has_gpu``
        # covers Jetson's h264_v4l2m2m too, reading the private desktop
        # NVENC flag primed libx264 on Jetson while every real recording
        # opened the v4l2m2m encoder, paying the first-write cost anyway.
        caps = EncoderCapabilities.get_instance()
        if use_gpu is None:
            try:
                use_gpu = bool(caps.has_gpu)
            except Exception:
                use_gpu = False

        tmp_dir = tempfile.gettempdir()
        # Use a process-unique filename so concurrent app instances don't
        # collide while warming up.
        throwaway = os.path.join(
            tmp_dir, f"_pybehaviorlab_warmup_pid{os.getpid()}.mp4"
        )
        encoder_name = "unknown"
        ok = False
        error = None
        try:
            writer = VideoWriterFactory.create_writer(
                throwaway, fps, size, use_gpu=bool(use_gpu),
                crf=23,
            )
            if writer is not None and getattr(writer, "isOpened", lambda: False)():
                encoder_name = getattr(writer, "encoder", "opencv")
                # Write a single black frame, then close, that completes
                # codec init end-to-end (alloc + first encode + flush).
                try:
                    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                    writer.write(frame)
                except Exception:
                    pass
                try:
                    writer.release()
                except Exception:
                    pass
                ok = True
        except Exception as e:
            error = repr(e)
            logger.debug("warm_video_encoders: writer init raised %s", error)
        finally:
            # Best-effort cleanup of the throwaway file.
            try:
                if os.path.exists(throwaway):
                    os.remove(throwaway)
            except Exception:
                pass

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        # Latch on ANY completed attempt, warming is best-effort, and
        # latching only success made every later call re-run the full
        # probe + throwaway-writer sequence on a host with no encoder.
        _WARM_DONE = True
        if ok:
            logger.info(
                "Video encoder warmup ok: encoder=%s gpu=%s elapsed=%.0f ms",
                encoder_name, bool(use_gpu), elapsed_ms,
            )
        else:
            logger.warning(
                "Video encoder warmup failed (elapsed=%.0f ms, err=%s), "
                "first-box record will pay the DLL-load cost on the GUI "
                "thread.", elapsed_ms, error,
            )
        return ok
