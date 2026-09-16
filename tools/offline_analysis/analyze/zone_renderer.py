"""
Unified zone rendering, single implementation for both OpenCV and Qt.

Ensures zones look identical everywhere: main display, arena widget,
zone editor, analysis tools.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Zone palette, chosen to be visually distinct from the green/red
# tracker-box and ROI lines drawn by the GUI. No pure green or pure red
# so the user never confuses a zone boundary with a tracker overlay.
# BGR tuples (OpenCV native), converted to RGB for Qt where needed.
ZONE_COLORS_BGR: List[Tuple[int, int, int]] = [
    (0, 215, 255),     # gold / yellow
    (255, 0, 255),     # magenta
    (255, 255, 0),     # cyan
    (0, 140, 255),     # orange
    (180, 105, 255),   # hot pink
    (255, 200, 80),    # sky blue
]

ZONE_ALPHA = 0.25
ZONE_LINE_THICKNESS = 2
ZONE_FONT_SCALE = 0.5
ZONE_FONT_THICKNESS = 1


# Coord-space resolution. All three zone renderers (OpenCV, QPainter,
# arena_widget's own inline one) share this helper so they can't drift
# apart again.
#
# Rationale: shipped projects carry zones with ``coord_space:"pixel"``
# but points in the 0..1 range, the old zone editor stamped the tag
# wrong. Trusting the tag blindly made zones invisible (0.015 pixels
# at the origin). Trusting the heuristic blindly made zones vanish on
# a +/- scale that nudged a corner slightly over 1.0.
#
# Resolution order:
#   1. If ``coord_space`` is set AND it matches the point-range
#      evidence → trust it. Definitive.
#   2. If ``coord_space`` is set but DISAGREES with the evidence
#      (tag says "pixel" but every point is in [-0.05, 1.05]) → the
#      tag is wrong (known legacy bug); fall back to the range vote.
#   3. If no tag → use the range vote.
#
# Range vote says "normalized" iff every sampled coord is in
# ``[-NORM_TOL, 1 + NORM_TOL]``. The tolerance (0.05) absorbs small
# overshoots from transforms; larger overshoots fall through to pixel.
NORM_TOL = 0.05


def is_normalized_points(pts, coord_space_tag=None):
    """Decide if ``pts`` are in normalized [0, 1] space.

    ``pts`` is an iterable of (x, y) pairs; ``coord_space_tag`` is
    the value of the zone's ``coord_space`` field, if any.
    """
    if not pts:
        return False
    all_in_range = all(
        -NORM_TOL <= float(p[0]) <= 1.0 + NORM_TOL
        and -NORM_TOL <= float(p[1]) <= 1.0 + NORM_TOL
        for p in pts
    )
    if coord_space_tag == "normalized":
        return True
    if coord_space_tag == "pixel":
        # Legacy projects lie about this, if the points all look
        # normalized, override the tag. Prints nothing because this
        # path runs per frame at ~30 Hz.
        return all_in_range
    return all_in_range


def render_zones_opencv(
    frame: np.ndarray,
    zones: List[dict],
    highlight_zone: Optional[str] = None,
) -> np.ndarray:
    """Draw zone overlays on a BGR frame.  Returns the modified frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR or grayscale frame.  If grayscale, converted to BGR for drawing.
    zones : list of dict
        Each dict must have: "name", "points", "type".
        Optional: "enabled" (default True).
    highlight_zone : str, optional
        Zone name to highlight with thicker border.

    Returns
    -------
    Frame with zone overlays drawn.
    """
    if not zones:
        return frame

    # Ensure BGR for colored drawing
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    overlay = frame.copy()
    h, w = frame.shape[:2]

    for i, zone in enumerate(zones):
        if not zone.get("enabled", True):
            continue
        ztype = zone.get("type", "polygon")
        if ztype == "scale":
            continue

        name = zone.get("name", "")
        pts_raw = zone.get("points", [])
        if not pts_raw:
            continue

        # Coord-space resolution via shared helper so every renderer
        # makes the same call. Crucially, the tag is a *hint* now,
        # legacy projects carry ``coord_space:"pixel"`` with
        # normalized point values; the helper sees through that.
        if is_normalized_points(pts_raw, zone.get("coord_space")):
            pts = [(int(max(0.0, min(1.0, p[0])) * w),
                    int(max(0.0, min(1.0, p[1])) * h)) for p in pts_raw]
        else:
            pts = [(int(p[0]), int(p[1])) for p in pts_raw]

        color = ZONE_COLORS_BGR[i % len(ZONE_COLORS_BGR)]
        pts_np = np.array(pts, dtype=np.int32)

        if ztype == "line":
            if len(pts) >= 2:
                cv2.line(overlay, pts[0], pts[1], color, ZONE_LINE_THICKNESS)
        else:
            # Filled polygon with alpha
            cv2.fillPoly(overlay, [pts_np], color)

        # Outline
        thickness = ZONE_LINE_THICKNESS * 2 if name == highlight_zone else ZONE_LINE_THICKNESS
        if ztype != "line":
            cv2.polylines(overlay, [pts_np], True, color, thickness)

        # Zone name labels removed, boundaries only

    # Blend overlay with original
    cv2.addWeighted(overlay, ZONE_ALPHA, frame, 1 - ZONE_ALPHA, 0, frame)
    return frame


def render_zones_qpainter(
    painter,
    zones: List[dict],
    target_rect,
    highlight_zone: Optional[str] = None,
):
    """Draw zone overlays using QPainter.

    Parameters
    ----------
    painter : QPainter
        Active painter on the target widget/pixmap.
    zones : list of dict
        Each dict must have: "name", "points", "type".
    target_rect : QRect
        The rectangle to scale normalized coords into.
    highlight_zone : str, optional
        Zone name to highlight.
    """
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QColor, QPen, QPolygonF

    if not zones:
        return

    tw, th = target_rect.width(), target_rect.height()

    for i, zone in enumerate(zones):
        if not zone.get("enabled", True):
            continue
        ztype = zone.get("type", "polygon")
        if ztype == "scale":
            continue

        name = zone.get("name", "")
        pts_raw = zone.get("points", [])
        if not pts_raw:
            continue

        # Coord-space resolution via shared helper, see the helper's
        # docstring for why the tag alone is not trusted.
        if is_normalized_points(pts_raw, zone.get("coord_space")):
            pts = [(max(0.0, min(1.0, p[0])) * tw,
                    max(0.0, min(1.0, p[1])) * th) for p in pts_raw]
        else:
            pts = [(float(p[0]), float(p[1])) for p in pts_raw]

        bgr = ZONE_COLORS_BGR[i % len(ZONE_COLORS_BGR)]
        color = QColor(bgr[2], bgr[1], bgr[0])  # BGR → RGB

        # Fill
        fill_color = QColor(color)
        fill_color.setAlpha(int(255 * ZONE_ALPHA))

        pen_width = ZONE_LINE_THICKNESS * 2 if name == highlight_zone else ZONE_LINE_THICKNESS
        pen = QPen(color, pen_width)
        painter.setPen(pen)
        painter.setBrush(fill_color)

        polygon = QPolygonF([QPointF(x, y) for x, y in pts])

        if ztype == "line" and len(pts) >= 2:
            painter.drawLine(QPointF(*pts[0]), QPointF(*pts[1]))
        else:
            painter.drawPolygon(polygon)

        # Zone name labels removed, boundaries only
