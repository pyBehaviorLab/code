"""Unified zone rendering, single implementation for OpenCV and Qt.

Zones look identical everywhere: live tracking display, zone editor,
analysis tools.  Both renderers share the ``coord_space.is_normalized_points``
resolver so they stay in sync.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tools.offline_analysis.vendor.zone_coords import resolve_to_px
from tools.offline_analysis.vendor.zone_geometry import point_inside_px

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None  # cv2 may be missing on headless test boxes; render is a no-op then.


# Zone palette, chosen to be visually distinct from the green/red
# tracker-box and ROI lines drawn elsewhere.  No pure green or pure red so
# the user never confuses a zone boundary with a tracker overlay.
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

# Sub-pixel resolution for ``cv2`` draws via the ``shift`` parameter.
# 4 bits → coordinates scaled by 2^4 = 16. With ``cv2.LINE_AA`` this gives
# the smoothest anti-aliased line cv2 produces; without ``shift`` the AA
# looks integer-pixel-quantized.
_CV_SHIFT = 4
_CV_SHIFT_SCALE = 1 << _CV_SHIFT


def render_zones_opencv(
    frame: np.ndarray,
    zones: Iterable[dict],
    *,
    highlight_zone: Optional[str] = None,
    highlight_inside: Optional[Sequence[Tuple[int, int]]] = None,
    fast: bool = False,
) -> np.ndarray:
    """Draw zone overlays on a BGR frame.  Returns the modified frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR or grayscale frame.  If grayscale, converted to BGR.
    zones : iterable of dict
        Each dict must have ``name``, ``points``, ``type``.  Optional:
        ``enabled`` (default True), ``coord_space`` ("normalized" /
        "pixel"; resolved via the helper if absent).
    highlight_zone : str, optional
        Zone name to highlight with thicker border (e.g. selected in UI).
    highlight_inside : sequence of (x, y), optional
        Centroid pixel positions; any zone containing one gets a brighter
        fill (used by live tracking display to show occupancy).
    fast : bool
        If True, use ``cv2.LINE_8`` (no anti-aliasing, no sub-pixel
        shift, no occupancy alpha-blend pass), about 2-3× faster than
        the LINE_AA path. Use for the live display (small, Qt-scaled
        anyway). Keep False for high-quality export / analysis
        screenshots where AA matters.
    """
    if cv2 is None or frame is None or not zones:
        return frame

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Two integer-coord representations per zone:
    #   pts_int_subpx, float pixel × 16, for shift=4 sub-pixel AA draws
    #   pts_int, rounded ints, for pointPolygonTest + fillPoly
    drawables = []
    for i, zone in enumerate(zones):
        if not isinstance(zone, dict) or not zone.get("enabled", True):
            continue
        ztype = zone.get("type", "polygon")
        if ztype == "scale":
            continue  # scale zones are calibration data, never drawn live

        pts_raw = zone.get("points") or []
        if not pts_raw:
            continue

        pts_px = resolve_to_px(pts_raw, w, h, zone.get("coord_space"))
        pts_int = np.array(
            [[int(round(x)), int(round(y))] for x, y in pts_px], dtype=np.int32
        )
        pts_int_subpx = np.array(
            [[int(round(x * _CV_SHIFT_SCALE)),
              int(round(y * _CV_SHIFT_SCALE))] for x, y in pts_px],
            dtype=np.int32,
        )

        base_color = ZONE_COLORS_BGR[i % len(ZONE_COLORS_BGR)]
        is_highlight = highlight_zone is not None and zone.get("name") == highlight_zone

        # Occupancy highlight, brighten color if any centroid is inside.
        occupied = False
        if highlight_inside and ztype not in ("line",) and len(pts_int) >= 3:
            occupied = any(point_inside_px(pts_int, cx, cy)
                           for cx, cy in highlight_inside)

        color = (
            tuple(min(255, int(c * 1.4)) for c in base_color) if occupied else base_color
        )
        # Thickness floor of 2, single-pixel AA looks ragged on diagonal
        # segments after the live frame is resized to the widget.
        # Highlight = thicker still.
        if is_highlight:
            thickness = ZONE_LINE_THICKNESS * 2
        elif occupied:
            thickness = ZONE_LINE_THICKNESS + 1
        else:
            thickness = ZONE_LINE_THICKNESS
        drawables.append((ztype, pts_int, pts_int_subpx, color, thickness, occupied))

    if fast:
        # Fast path: skip the alpha-blend overlay copy + LINE_AA sub-pixel
        # path; just outline each zone with LINE_8 in one pass.
        for ztype, pts_int, _pts_subpx, color, thickness, _occupied in drawables:
            if ztype == "line" and len(pts_int) >= 2:
                cv2.line(frame, tuple(pts_int[0]), tuple(pts_int[1]),
                         color, thickness, cv2.LINE_8)
            elif len(pts_int) >= 2:
                cv2.polylines(frame, [pts_int], True, color, thickness,
                              cv2.LINE_8)
        return frame

    # Pass 1, fills onto the alpha-blend layer (only when occupied).
    # fillPoly does NOT support the shift parameter, use rounded ints.
    for ztype, pts_int, _pts_subpx, color, _thickness, occupied in drawables:
        if not occupied:
            continue
        if ztype == "line" or len(pts_int) < 3:
            continue
        cv2.fillPoly(overlay, [pts_int], color, lineType=cv2.LINE_AA)

    if drawables:
        cv2.addWeighted(overlay, ZONE_ALPHA, frame, 1 - ZONE_ALPHA, 0, frame)

    # Pass 2, outlines drawn directly on `frame` (full opacity, solid line).
    # Use sub-pixel coords + shift=4 + LINE_AA for the smoothest anti-
    # aliased lines OpenCV can produce.
    for ztype, _pts_int, pts_subpx, color, thickness, _occupied in drawables:
        if ztype == "line" and len(pts_subpx) >= 2:
            cv2.line(frame, tuple(pts_subpx[0]), tuple(pts_subpx[1]),
                     color, thickness, cv2.LINE_AA, _CV_SHIFT)
        else:
            cv2.polylines(frame, [pts_subpx], True, color, thickness,
                          cv2.LINE_AA, _CV_SHIFT)
    return frame


def render_zones_qpainter(
    painter,
    zones: Iterable[dict],
    target_w: int,
    target_h: int,
    *,
    highlight_zone: Optional[str] = None,
):
    """Draw zone overlays via QPainter.

    Parameters
    ----------
    painter : QPainter
        Active painter on the target widget/pixmap.
    zones : iterable of dict
        Same schema as ``render_zones_opencv``.
    target_w, target_h : int
        Pixel dimensions of the frame the painter is drawing onto.  Used
        to denormalize zone points.
    highlight_zone : str, optional
        Zone name to highlight.
    """
    from PySide6 import QtCore
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF

    if not zones:
        return

    # Smooth diagonals.  Caller may have set this already; setting it
    # again is cheap and idempotent.
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    for i, zone in enumerate(zones):
        if not isinstance(zone, dict) or not zone.get("enabled", True):
            continue
        ztype = zone.get("type", "polygon")
        if ztype == "scale":
            continue

        pts_raw = zone.get("points") or []
        if not pts_raw:
            continue

        pts_px = resolve_to_px(pts_raw, target_w, target_h, zone.get("coord_space"))

        bgr = ZONE_COLORS_BGR[i % len(ZONE_COLORS_BGR)]
        base_color = QColor(bgr[2], bgr[1], bgr[0])

        is_highlight = highlight_zone is not None and zone.get("name") == highlight_zone
        # Match the cv2 path: brighten color when the centroid is inside,
        # bump pen width to make the active zone obvious.
        if is_highlight:
            color = QColor(min(255, int(base_color.red() * 1.4)),
                           min(255, int(base_color.green() * 1.4)),
                           min(255, int(base_color.blue() * 1.4)))
            pen_width = ZONE_LINE_THICKNESS + 2
            fill = QColor(color)
            fill.setAlpha(int(255 * ZONE_ALPHA))
        else:
            color = base_color
            pen_width = ZONE_LINE_THICKNESS
            # No fill when not highlighted, outline-only for a clean look.
            fill = QtCore.Qt.BrushStyle.NoBrush

        # Cosmetic pen so the line width is in DEVICE pixels and stays
        # exactly ``pen_width`` regardless of any painter transform,
        # which matters because the painter is drawing onto a pixmap
        # that may itself be scaled when blitted.
        pen = QPen(color, pen_width)
        pen.setCosmetic(True)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(fill)

        polygon = QPolygonF([QPointF(x, y) for x, y in pts_px])
        if ztype == "line" and len(pts_px) >= 2:
            painter.drawLine(QPointF(*pts_px[0]), QPointF(*pts_px[1]))
        else:
            # Polygon, rectangle (4-point: TL/TR/BR/BL), ellipse-as-polygon.
            painter.drawPolygon(polygon)
