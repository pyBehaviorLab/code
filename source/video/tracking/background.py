"""Reference-background capture and validation for differencing trackers.

Background subtraction is only as good as the image it subtracts, so the
reference is treated here as an asset with a lifecycle, captured, stamped,
checked and refreshed, rather than a file someone remembered to make.

**Why the median.** A single grabbed frame captures whatever noise, flicker
and animal happened to be in it, and the animal then subtracts itself away
wherever it was standing. The per-pixel median over a short burst captures the
*arena*: the animal is somewhere different in most samples, so it falls out,
while every wall, shelf and shadow survives. The animal being present during
capture is therefore expected, not a problem; there is no "take the mouse out
first" ceremony.

Measured on a 17.8-minute operant session, this is what makes differencing
work: static subtraction against a median reference tracked the animal in
96 % / 81 % of frames across two windows, where the background-free detector
managed 47 % / 20 % at four times the CPU.

The caveat that remains: the median only erases the animal if the animal MOVES
during the burst. One that sits still for the whole window is captured into its
own background and then cannot be seen. ``looks_occupied`` exists to catch the
worst of that, and the idle-refresh policy is what repairs it in practice.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from source.log import get_logger

logger = get_logger()

# Burst geometry. Long enough that a mouse ambling at a few cm/s has vacated
# most of the pixels it started on, short enough that an operator does not
# think the app has hung. Sixteen samples is well past the point where extra
# ones change the median.
CAPTURE_SECONDS = 8.0
CAPTURE_SAMPLES = 16

# A reference older than this is probably not this session's arena any more:
# feeders get moved, lights drift, cameras get nudged. Warn, never block,
# only the operator knows whether the rig actually changed.
STALE_AFTER_HOURS = 24.0

# Status codes. ``OK`` and ``STALE`` are usable; ``MISSING`` and ``MISMATCH``
# are not, and only MISMATCH is worth blocking a start over (see ``describe``).
OK = "ok"
MISSING = "missing"
STALE = "stale"
MISMATCH = "shape_mismatch"


def capture_median(get_frame: Callable[[], Optional[np.ndarray]],
                   *,
                   seconds: float = CAPTURE_SECONDS,
                   samples: int = CAPTURE_SAMPLES,
                   pump: Optional[Callable[[], None]] = None,
                   sleep: Optional[Callable[[float], None]] = None
                   ) -> Optional[np.ndarray]:
    """Per-pixel median of ``samples`` frames taken over ``seconds``.

    ``get_frame`` is polled; ``None`` results are skipped rather than aborting,
    because a camera can miss a grab without being broken. ``pump`` is called
    between samples so a GUI caller can keep its event loop alive, without it
    an 8-second capture looks like a freeze.

    Returns None if fewer than three usable samples arrived: a "median" of one
    or two frames is just a frame, and would silently reintroduce the
    single-frame failure this function exists to avoid.
    """
    import time as _time
    sleep = sleep or _time.sleep
    interval = max(0.0, float(seconds) / max(1, int(samples)))
    frames: List[np.ndarray] = []
    for i in range(int(samples)):
        try:
            f = get_frame()
        except Exception as e:
            logger.debug("background capture: frame grab failed: %s", e)
            f = None
        if f is not None and getattr(f, "size", 0):
            if frames and f.shape != frames[0].shape:
                # Resolution changed mid-burst: the samples no longer describe
                # one view, so start again from the new one.
                logger.info("background capture: frame shape changed "
                            "%s -> %s, restarting burst",
                            frames[0].shape, f.shape)
                frames = []
            frames.append(f.copy())
        if i < int(samples) - 1:
            if pump is not None:
                try:
                    pump()
                except Exception:
                    pass
            sleep(interval)
    if len(frames) < 3:
        logger.warning("background capture: only %d usable frame(s), "
                       "not enough for a median", len(frames))
        return None
    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def looks_occupied(background: np.ndarray, frame: np.ndarray,
                   *, min_fraction: float = 0.35) -> bool:
    """Does ``frame`` differ from ``background`` over most of the view?

    A sanity check on a freshly captured reference. If a live frame differs
    almost everywhere, the capture caught something transient, the light
    changed, the lid was open, the camera moved, and subtracting it would
    make the whole arena foreground.

    It deliberately cannot detect the case it would be most useful for, an
    animal that never moved during the burst: that animal is IN the reference,
    so a later frame agrees with it. Only motion, or the operator, resolves
    that one.
    """
    if background is None or frame is None:
        return False
    if background.shape != frame.shape:
        return True
    bg = _gray(background)
    fr = _gray(frame)
    diff = cv2.absdiff(fr, bg)
    changed = float((diff > 25).mean())
    return changed >= float(min_fraction)


def shape_of(background: Optional[np.ndarray]) -> Optional[Tuple[int, int]]:
    """``(h, w)`` of a background image, or None."""
    if background is None:
        return None
    return tuple(background.shape[:2])


def hours_since(stamp: Optional[str]) -> Optional[float]:
    """Hours since an ISO ``YYYY-MM-DD HH:MM:SS`` stamp, or None if unusable."""
    if not stamp:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            then = datetime.strptime(stamp, fmt)
        except ValueError:
            continue
        return max(0.0, (datetime.now() - then).total_seconds() / 3600.0)
    return None


def status(*, background: Optional[np.ndarray],
           frame_shape: Optional[Tuple[int, int]],
           captured_at: Optional[str],
           stale_after_hours: float = STALE_AFTER_HOURS) -> str:
    """Classify a box's reference background.

    Order matters: a background of the wrong shape is reported as MISMATCH
    even when it is also stale, because that is the finding that has to be
    acted on, a stretched reference mis-registers every arena edge, and the
    tracker then locks onto the mismatch rather than the animal.
    """
    if background is None:
        return MISSING
    if frame_shape is not None and shape_of(background) != tuple(frame_shape):
        return MISMATCH
    age = hours_since(captured_at)
    if age is not None and age >= float(stale_after_hours):
        return STALE
    return OK


def describe(code: str, *, captured_at: Optional[str] = None,
             background: Optional[np.ndarray] = None,
             frame_shape: Optional[Tuple[int, int]] = None) -> Tuple[str, bool]:
    """``(message, blocks_start)`` for a status code.

    Only MISMATCH blocks. Missing and stale are the operator's call, a rig
    that has not changed since yesterday is fine on yesterday's background,
    and refusing to start would be wrong about that more often than right.
    """
    if code == OK:
        return "", False
    if code == MISSING:
        return ("No background captured for this box, blob tracking needs "
                "one. Capture background in the Tracking panel."), False
    if code == STALE:
        age = hours_since(captured_at)
        when = f"{age:.0f} h ago" if age is not None else "a while ago"
        return (f"Background was captured {when}. If the arena, lighting or "
                f"camera has changed since, recapture it."), False
    if code == MISMATCH:
        have = shape_of(background)
        want = tuple(frame_shape) if frame_shape else None
        return (f"Background is {have[1]}x{have[0]} but the camera is now "
                f"{want[1]}x{want[0]}. A background from a different view "
                f"cannot be subtracted, recapture it before starting."
                if have and want else
                "Background does not match the current camera view, "
                "recapture it before starting."), True
    return "", False


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
