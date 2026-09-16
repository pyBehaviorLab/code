"""Geometric primitives for the zones library.

``point_in_polygon``: a ray-casting hit-test with no Shapely dependency.
Tries the Cython accelerator in source.cython.zone_math first and falls back to
the pure-Python implementation below when Cython isn't compiled.

``point_inside_px``: the overlay's containment test against the integer pixel
polygon it already built for drawing.

There are three containment tests in the codebase and that is deliberate; each
one's docstring says which question it answers and how its boundary handling
differs. The one that decides behaviour is ``Zone.contains`` in
source.video.zones.schema (Shapely-cached, boundary-exclusive), reach for that
unless you have a specific reason not to.
"""

from __future__ import annotations

from typing import Sequence

import cv2

# ---------------------------------------------------------------------------
# Optional Cython accelerator
# ---------------------------------------------------------------------------
_HAS_CYTHON = False
_cy_point_in_polygon = None
try:
    from source.cython.zone_math import point_in_polygon as _cy_point_in_polygon  # type: ignore
    _HAS_CYTHON = True
except ImportError:
    pass


def point_in_polygon(px: float, py: float, polygon_pts: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon test (works for any 2-D polygon).

    Pure-Python implementation, no Shapely required. Use the Shapely-cached
    `Zone.contains()` API in source.video.zones.schema when you have many
    queries against the same polygon; this function is for one-shot checks
    where building a Shapely Polygon would be wasteful.
    """
    if _HAS_CYTHON:
        return bool(_cy_point_in_polygon(px, py, polygon_pts))
    n = len(polygon_pts)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon_pts[i][0], polygon_pts[i][1]
        xj, yj = polygon_pts[j][0], polygon_pts[j][1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def point_inside_px(pts_int, x: float, y: float) -> bool:
    """Containment against an integer pixel polygon, for the live overlay.

    A THIRD containment test exists on purpose, and the difference matters:

    * ``Zone.contains`` (Shapely) is what the tracker decides on. Shapely's
      ``contains`` is boundary-EXCLUSIVE, a point exactly on an edge is out.
    * this one wraps ``cv2.pointPolygonTest(...) >= 0``, which is boundary-
      INCLUSIVE, and runs on the rounded int polygon the overlay already built
      to draw with.

    So for a centroid sitting exactly on a zone edge the highlight can say
    "inside" while the tracker says "outside". Sub-pixel in practice, but it is
    a real disagreement between what the operator sees and what fires an MCU
    event, and it is written down here rather than left to be rediscovered.

    Reusing the drawing polygon is why this is cv2 and not
    :func:`point_in_polygon`: the overlay runs per frame per zone, and
    converting back to float points to re-ray-cast would cost more than the
    test. Returns False rather than raising on a degenerate polygon.
    """
    if pts_int is None or len(pts_int) < 3:
        return False
    try:
        return cv2.pointPolygonTest(pts_int, (float(x), float(y)), False) >= 0
    except cv2.error:
        return False


__all__ = ["point_in_polygon", "point_inside_px"]
