"""Blob-tracker robustness tests.

Spatial gating: the tracker prefers the largest contour near the
velocity-predicted position, so a larger but distant distractor can't teleport
the centroid. Temporal debounce: an out-of-gate blob must persist a few frames
(and be size-plausible) before the track jumps to it. Both gated behind
`use_spatial_gating` (default True); False restores pure-largest behavior.
"""
import numpy as np

from source.video.tracking.blob import BlobTracker

_BG = np.full((240, 320), 200, np.uint8)


def _frame(blobs):
    f = _BG.copy()
    for (x0, y0, x1, y1) in blobs:
        f[y0:y1, x0:x1] = 40          # dark animal/distractor on bright bg
    return f


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _tracker(gating=True):
    t = BlobTracker(setup_id=1)
    t.set_background(_BG)
    t.update_params(threshold=25, min_area=20, max_area=50000, detect_dark=True,
                    bg_mode="static")
    # Code-level default, no dialog widget, set the attribute directly.
    t.use_spatial_gating = gating
    t.initialize(_BG)
    return t


def _establish_track(t, n=5, step=8):
    """Track a small animal moving rightward to seed velocity + gate."""
    for i in range(n):
        x = 60 + step * i
        t.update(_frame([(x, 55, x + 20, 75)]), timestamp=i * 33)
    return 60 + step * n            # next animal x


def test_larger_far_distractor_does_not_steal_track():
    t = _tracker(gating=True)
    ax = _establish_track(t)
    # animal (20x20) still present + a BIGGER distractor (40x40) far away
    f = _frame([(ax, 55, ax + 20, 75), (250, 170, 290, 210)])
    t.update(f, timestamp=5 * 33)
    assert _dist(t.last_centroid, (ax + 10, 65)) < 25          # stayed on animal
    assert _dist(t.last_centroid, (270, 190)) > 100            # not the distractor


def test_without_gating_largest_wins():
    """With gating off the larger distractor takes the track."""
    t = _tracker(gating=False)
    ax = _establish_track(t)
    f = _frame([(ax, 55, ax + 20, 75), (250, 170, 290, 210)])
    t.update(f, timestamp=5 * 33)
    assert _dist(t.last_centroid, (270, 190)) < 30             # jumped to largest


def test_new_blob_debounced_then_acquired():
    t = _tracker(gating=True)
    _establish_track(t, n=4, step=6)
    far = [(40, 170, 60, 190)]                                 # new blob, far away
    ok1, _ = t.update(_frame(far), timestamp=4 * 33)           # 1st appearance
    ok2, _ = t.update(_frame(far), timestamp=5 * 33)           # persists → commit
    assert ok1 is False                                        # debounced frame 1
    assert ok2 is True
    assert _dist(t.last_centroid, (50, 180)) < 20              # re-acquired at new blob


def test_one_frame_transient_ignored():
    t = _tracker(gating=True)
    ax = _establish_track(t, n=4, step=6)
    # animal in-gate + a transient flicker elsewhere, present this frame only
    t.update(_frame([(ax, 55, ax + 20, 75), (40, 170, 70, 200)]), timestamp=4 * 33)
    assert _dist(t.last_centroid, (ax + 10, 65)) < 25          # stayed on animal


def test_first_frame_acquires_largest():
    """No prediction yet, so the largest candidate is acquired."""
    t = _tracker(gating=True)
    ok, pos = t.update(_frame([(150, 110, 175, 135)]), timestamp=0)
    assert ok and pos is not None
    assert t.last_centroid is not None
