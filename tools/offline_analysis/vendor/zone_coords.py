"""Coordinate-space helpers, single source of truth for the video pipeline.

Three coord spaces exist in the system; this module is the only place that
converts between them:

- **Camera space**: full sensor frame.  ROIs (which carve up the camera into
  per-box regions) live here, normalized to the camera resolution.
- **Box space**: an ROI-cropped frame for one arena.  Zones, scale-zone
  endpoints, and any tracking-result point that's stored on disk lives here,
  normalized to the cropped frame's (Wb, Hb).
- **Display space**: widget pixels.  Computed at draw-time, never stored.

Every conversion in the GUI routes through ``to_norm`` / ``from_norm``.  Any
``pts * 3`` or ``pts - roi_x`` math elsewhere is a bug.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

# Tolerance for the normalized-vs-pixel range vote.  Absorbs small overshoots
# from interactive transforms (a corner at 1.02 is still normalized); larger
# overshoots fall through to pixel.
NORM_TOL = 0.05


def is_normalized_points(pts: Sequence, coord_space_tag: Optional[str] = None) -> bool:
    """Decide whether ``pts`` are in normalized [0, 1] space.

    Resolution order:
      1. ``coord_space_tag == "normalized"`` and points look normalized → True.
      2. ``coord_space_tag == "pixel"`` but every point is in [-NORM_TOL,
         1+NORM_TOL] → tag is untrustworthy; trust the range vote.
      3. No tag → use the range vote.

    Some project files stamp ``coord_space:"pixel"`` on points that are
    actually in [0, 1]; the range vote guards against that tagging slip so
    zones don't render invisibly at the origin or vanish off-frame.
    """
    if not pts:
        return False
    try:
        all_in_range = all(
            -NORM_TOL <= float(p[0]) <= 1.0 + NORM_TOL
            and -NORM_TOL <= float(p[1]) <= 1.0 + NORM_TOL
            for p in pts
        )
    except (TypeError, IndexError):
        return False

    if coord_space_tag == "normalized":
        return True
    return all_in_range


def to_norm(pts_px: Iterable[Sequence[float]], w: int, h: int) -> List[List[float]]:
    """Pixel coords → normalized [0, 1].  ``w``/``h`` are the reference frame size."""
    if w <= 0 or h <= 0:
        raise ValueError(f"to_norm: invalid frame size ({w}, {h})")
    return [[float(p[0]) / w, float(p[1]) / h] for p in pts_px]


def from_norm(pts_norm: Iterable[Sequence[float]], w: int, h: int) -> List[List[float]]:
    """Normalized [0, 1] → pixel coords for a frame of size (w, h)."""
    return [[float(p[0]) * w, float(p[1]) * h] for p in pts_norm]


def resolve_to_px(
    pts: Sequence,
    w: int,
    h: int,
    coord_space_tag: Optional[str] = None,
) -> List[List[float]]:
    """Return ``pts`` in pixel coords for a frame of size (w, h).

    Routes through ``is_normalized_points`` so callers need not know whether
    the points are normalized or pixel-tagged.  Used at every draw / hit-test site.
    """
    if is_normalized_points(pts, coord_space_tag):
        return from_norm(pts, w, h)
    return [[float(p[0]), float(p[1])] for p in pts]


def scale_zone_to_px_per_mm(
    scale_zone: dict,
    frame_w: int,
    frame_h: int,
) -> Optional[float]:
    """Compute pixels-per-mm from a calibration zone, for a given frame size.

    Stored form: a zone dict with ``type:"scale"``, two endpoint ``points``
    (normalized box-space), and ``scale_length_mm`` (real number, mm).  We
    normalize-to-pixel each endpoint here, take the Euclidean distance, and
    divide.  Resolution-independent, a zone calibrated at 640x480 still gives
    the right px/mm at 1280x720.
    """
    if not isinstance(scale_zone, dict):
        return None
    pts_raw = scale_zone.get("points") or []
    if len(pts_raw) < 2:
        return None
    length_mm = scale_zone.get("scale_length_mm")
    if length_mm is None:
        # Fallback scale_length + scale_unit; convert to mm so callers never see units.
        length = scale_zone.get("scale_length")
        unit = (scale_zone.get("scale_unit") or "mm").lower()
        if length is None:
            return None
        try:
            length_mm = float(length) * {
                "mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4,
            }.get(unit, 1.0)
        except (TypeError, ValueError):
            return None
    try:
        length_mm = float(length_mm)
    except (TypeError, ValueError):
        return None
    if length_mm <= 0:
        return None

    pts_px = resolve_to_px(pts_raw, frame_w, frame_h, scale_zone.get("coord_space"))
    (x1, y1), (x2, y2) = pts_px[0], pts_px[1]
    length_px = math.hypot(x2 - x1, y2 - y1)
    if length_px <= 0:
        return None
    return length_px / length_mm


__all__ = [
    "NORM_TOL",
    "is_normalized_points",
    "to_norm", "from_norm",
    "resolve_to_px",
    "scale_zone_to_px_per_mm",
]
