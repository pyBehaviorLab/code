"""Centroid → speed helper shared by PoseSink and TrackerSink.

Speed is the Euclidean distance between previous and current centroid
divided by the time delta. Returns 0.0 on the first frame or when dt <= 0.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple


def compute_speed(prev: Optional[Tuple[float, float, int]],
                  cx: float, cy: float, now_ns: int) -> float:
    """Compute speed (px/s) from a previous centroid + timestamp and a new centroid.

    Args:
        prev: ``(prev_x, prev_y, prev_mono_ns)`` or ``None`` on the first observation.
        cx, cy: current centroid (pixel space).
        now_ns: ``host_clock.host_ns()`` at observation time.

    Returns:
        Pixels per second. 0.0 when no prior centroid or ``dt <= 0``.
    """
    if prev is None:
        return 0.0
    px, py, pt_ns = prev
    dt = (now_ns - pt_ns) / 1e9
    if dt <= 0:
        return 0.0
    return math.hypot(cx - px, cy - py) / dt


__all__ = ["compute_speed"]
