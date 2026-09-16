# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython-accelerated zone geometry math for pyOperant tracking pipeline.

Functions:
    cross_line_side: Cross product sign to determine which side of a line
        a point is on. Used by ZoneManager._check_cross_line_trigger().

    closest_point_on_segment: Project a point onto a line segment, clamped
        to [0,1]. Used by ZoneManager._check_facing_line_trigger().

    calculate_rotation: Head-tail axis angle in degrees.
        Used by ZoneManager._calculate_rotation().

    point_in_polygon: Ray-casting point-in-polygon test. Used by
        zones.geometry.point_in_polygon when Shapely is unavailable
        (e.g. the Jetson / embedded deployment target).
"""

from libc.math cimport atan2, sqrt, acos, M_PI


def point_in_polygon(double px, double py, polygon_pts):
    """Ray-casting point-in-polygon test for any 2-D polygon.

    Args:
        px, py:       Query point.
        polygon_pts:  Sequence of (x, y) vertex pairs.

    Returns:
        True if (px, py) is inside the polygon, else False.
    """
    cdef Py_ssize_t n = len(polygon_pts)
    if n < 3:
        return False

    cdef bint inside = False
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = n - 1
    cdef double xi, yi, xj, yj

    for i in range(n):
        xi = polygon_pts[i][0]
        yi = polygon_pts[i][1]
        xj = polygon_pts[j][0]
        yj = polygon_pts[j][1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def cross_line_side(
    double px, double py,
    double x1, double y1,
    double x2, double y2
):
    """Determine which side of line (x1,y1)-(x2,y2) point (px,py) is on.

    Returns:
        1 if left, -1 if right, 0 if on the line.
    """
    cdef double cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if cross > 0.0:
        return 1
    elif cross < 0.0:
        return -1
    return 0


def closest_point_on_segment(
    double px, double py,
    double x1, double y1,
    double x2, double y2
):
    """Project point (px,py) onto segment (x1,y1)-(x2,y2), clamped to endpoints.

    Returns:
        (cx, cy): closest point on the segment.
    """
    cdef double dx = x2 - x1
    cdef double dy = y2 - y1
    cdef double length_sq = dx * dx + dy * dy
    cdef double t, cx, cy

    if length_sq < 1e-12:
        return (x1, y1)

    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    cx = x1 + t * dx
    cy = y1 + t * dy
    return (cx, cy)


def calculate_rotation(
    double head_x, double head_y,
    double tail_x, double tail_y
):
    """Angle of head-tail axis relative to horizontal, in degrees (-180..180)."""
    cdef double dx = head_x - tail_x
    cdef double dy = head_y - tail_y
    return atan2(dy, dx) * (180.0 / M_PI)


def facing_angle(
    double hx, double hy,
    double bx, double by,
    double cpx, double cpy
):
    """Angle between heading (body->head) and target (head->closest point).

    Returns angle in degrees (0 = facing directly, 180 = facing away).
    Returns -1.0 if vectors are degenerate (zero length).
    """
    cdef double heading_x = hx - bx
    cdef double heading_y = hy - by
    cdef double heading_len = sqrt(heading_x * heading_x + heading_y * heading_y)

    cdef double target_x = cpx - hx
    cdef double target_y = cpy - hy
    cdef double target_len = sqrt(target_x * target_x + target_y * target_y)

    cdef double dot, angle_rad

    if heading_len < 1e-6:
        return -1.0

    if target_len < 1e-6:
        return 0.0  # Head is on the target

    dot = (heading_x * target_x + heading_y * target_y) / (heading_len * target_len)
    if dot < -1.0:
        dot = -1.0
    elif dot > 1.0:
        dot = 1.0

    angle_rad = acos(dot)
    return angle_rad * (180.0 / M_PI)
