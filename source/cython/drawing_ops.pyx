# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython-accelerated drawing preparation for per-frame video overlay.

Called by the video overlay path on every displayed frame. A pure-Python
fallback (_draw_keypoints_python) exists in source/gui/base.py when this
module is unavailable.

Functions:
    prepare_dlc_keypoints: vectorized DLC pose extraction with confidence filter
"""

import numpy as np


def prepare_dlc_keypoints(double[:, :] pose, double roi_x, double roi_y,
                           double conf_thresh):
    """Extract valid DLC keypoints from pose array, applying ROI offset and confidence filter.

    Replaces a Python for-loop with per-row try/except and float() conversions
    in main_window._draw_zones_on_frame(). Avoids Python overhead per keypoint.

    Args:
        pose:        DLC pose array (N x 3) with columns [x, y, confidence].
                     Also accepts (N x 2); confidence defaults to 1.0.
        roi_x:       X offset to subtract (ROI crop origin).
        roi_y:       Y offset to subtract (ROI crop origin).
        conf_thresh: Minimum confidence to include a keypoint (0.0-1.0).

    Returns:
        List of (index, px, py, conf) tuples for keypoints above threshold.
        - index: keypoint index (for color lookup)
        - px, py: integer pixel coords with ROI offset applied
        - conf: original confidence value
    """
    cdef Py_ssize_t n = pose.shape[0]
    cdef Py_ssize_t ncols = pose.shape[1]
    cdef Py_ssize_t i
    cdef double x, y, conf
    cdef list result = []

    for i in range(n):
        x = pose[i, 0]
        y = pose[i, 1]
        # If pose has only 2 columns (no confidence), assume conf = 1.0
        if ncols > 2:
            conf = pose[i, 2]
        else:
            conf = 1.0

        if conf >= conf_thresh:
            result.append((
                <int>i,
                <int>(x - roi_x),
                <int>(y - roi_y),
                conf
            ))

    return result
