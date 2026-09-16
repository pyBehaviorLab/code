# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython-accelerated image operations for pyOperant tracking pipeline.

Functions:
    fused_illumination_normalize: Single-pass illumination normalization
        replacing the 4-allocation NumPy pipeline in tracker.py._prepare_gray().

    batch_bgr_to_gray: C-level typed loop calling cv2.cvtColor for
        scientific camera batches (up to 100 frames), reducing per-iteration
        Python dispatch overhead.
"""

import numpy as np
cimport numpy as np
import cv2

np.import_array()

ctypedef np.uint8_t uint8_t
ctypedef np.float32_t float32_t


def fused_illumination_normalize(
    np.ndarray[uint8_t, ndim=2] src,
    np.ndarray[float32_t, ndim=2] blurred_bg,
    np.ndarray[uint8_t, ndim=2] dst
):
    """Single-pass illumination normalization.

    Replaces the 4-allocation pipeline:
        blurred_bg = GaussianBlur(result, (51,51), 0).astype(float32)
        blurred_bg[blurred_bg < 1] = 1
        normalized = (result.astype(float32) / blurred_bg) * 128
        result = np.clip(normalized, 0, 255).astype(uint8)

    Args:
        src: Input grayscale frame (H x W, uint8). Read-only.
        blurred_bg: Pre-blurred background (H x W, float32).
            Caller should compute: cv2.GaussianBlur(src, (51,51), 0).astype(np.float32)
        dst: Pre-allocated output buffer (H x W, uint8). Written in-place.

    Returns:
        dst (same array, for convenience).
    """
    cdef int h = src.shape[0]
    cdef int w = src.shape[1]
    cdef int i, j
    cdef float bg_val, val

    for i in range(h):
        for j in range(w):
            bg_val = blurred_bg[i, j]
            if bg_val < 1.0:
                bg_val = 1.0
            val = (<float>src[i, j] / bg_val) * 128.0
            if val < 0.0:
                val = 0.0
            elif val > 255.0:
                val = 255.0
            dst[i, j] = <uint8_t>val

    return dst


def batch_bgr_to_gray(list frames):
    """Convert a batch of BGR frames to grayscale using cv2.cvtColor.

    Reduces Python dispatch overhead by using a typed loop.

    Args:
        frames: List of numpy arrays (BGR uint8, H x W x 3).

    Returns:
        List of grayscale numpy arrays (H x W, uint8).
    """
    cdef int n = len(frames)
    cdef int i
    cdef list result = []

    for i in range(n):
        f = frames[i]
        if f.ndim == 3 and f.shape[2] == 3:
            result.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        else:
            result.append(f)

    return result
