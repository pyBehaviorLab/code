"""WS9 consolidation: background→gray derivation is a single helper.

set_background, initialize, and the reprocess path all route through
_recompute_background_gray, producing the same background_gray for a given
background regardless of which entry point set it.
"""
from __future__ import annotations

import numpy as np

from source.video.tracking.blob import BlobTracker


def test_set_background_bgr_populates_gray():
    t = BlobTracker(setup_id=1)
    bgr = np.full((30, 40, 3), 120, np.uint8)
    t.set_background(bgr)
    assert t.background_gray is not None
    assert t.background_gray.shape == (30, 40)  # single channel


def test_initialize_and_set_background_agree():
    bgr = np.random.randint(0, 255, (30, 40, 3), np.uint8)

    a = BlobTracker(setup_id=1)
    a.set_background(bgr)

    b = BlobTracker(setup_id=2)
    b.initialize(bgr)  # sets background via the same helper

    assert a.background_gray is not None and b.background_gray is not None
    assert np.array_equal(a.background_gray, b.background_gray)


def test_grayscale_background_is_accepted():
    t = BlobTracker(setup_id=1)
    gray = np.full((30, 40), 90, np.uint8)
    t.set_background(gray)
    assert t.background_gray is not None
    assert t.background_gray.shape == (30, 40)
