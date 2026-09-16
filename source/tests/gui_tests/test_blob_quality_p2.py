"""Blob-tracker quality + efficiency (double-blur removal, pooled bg buffer).

The default preprocessing ran medianBlur(5) THEN GaussianBlur back-to-back
(nothing between), over-smoothing the animal's edges and costing a full-frame
op every frame. P2 skips the redundant second blur when the median pre-filter
already ran and no contrast stage (CLAHE/illum) sat between them, sharper
edges (better centroid/area) + one fewer per-frame pass. These tests lock the
behavior and confirm a synthetic dark-animal-on-light track stays accurate.
"""
import numpy as np

import source.video.tracking.blob as blobmod
from source.video.tracking.blob import BlobTracker


def _count_blurs(monkeypatch):
    calls = {"median": 0, "gauss": 0}
    real_median = blobmod.cv2.medianBlur
    real_gauss = blobmod.cv2.GaussianBlur

    def med(img, k, *a, **kw):
        calls["median"] += 1
        return real_median(img, k, *a, **kw)

    def gau(img, k, s, *a, **kw):
        calls["gauss"] += 1
        return real_gauss(img, k, s, *a, **kw)

    monkeypatch.setattr(blobmod.cv2, "medianBlur", med)
    monkeypatch.setattr(blobmod.cv2, "GaussianBlur", gau)
    return calls


def test_default_config_runs_single_blur(monkeypatch):
    # median prefilter on, CLAHE off, illum off → only the median runs.
    t = BlobTracker(setup_id=1)
    assert t.use_median_prefilter and not t.use_clahe and not t.use_illumination_norm
    calls = _count_blurs(monkeypatch)
    t._prepare_gray(np.full((60, 80), 100, np.uint8))
    assert calls["median"] == 1
    assert calls["gauss"] == 0            # redundant second blur skipped


def test_contrast_stage_keeps_post_blur(monkeypatch):
    # With CLAHE on, the post-blur is kept (CLAHE amplifies noise).
    t = BlobTracker(setup_id=1)
    t.use_clahe = True
    calls = _count_blurs(monkeypatch)
    t._prepare_gray(np.full((60, 80), 100, np.uint8))
    assert calls["median"] == 1           # prefilter
    assert calls["gauss"] == 1            # post-blur kept


def test_no_prefilter_keeps_post_blur(monkeypatch):
    t = BlobTracker(setup_id=1)
    t.use_median_prefilter = False
    calls = _count_blurs(monkeypatch)
    t._prepare_gray(np.full((60, 80), 100, np.uint8))
    assert calls["median"] == 0
    assert calls["gauss"] == 1            # the only blur


def _disc(frame, cx, cy, r=9, val=40):
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    frame[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = val
    return frame


def test_synthetic_track_follows_dark_animal():
    # Light arena, dark disc moving left→right; the moments centroid must
    # track the disc within a few px each frame.
    H, W = 120, 200
    bg = np.full((H, W), 200, np.uint8)
    t = BlobTracker(setup_id=1)
    t.min_area = 50
    t.subject_min_area = 50
    t.initialize(bg)                       # sets background + activates

    errs = []
    for i in range(20):
        cx = 30 + i * 7
        cy = 60
        frame = _disc(np.full((H, W), 200, np.uint8), cx, cy)
        ok, _pos = t.update(frame, timestamp=float(i))
        assert ok, f"frame {i}: tracker missed the disc"
        got = t.last_centroid
        assert got is not None
        errs.append(((got[0] - cx) ** 2 + (got[1] - cy) ** 2) ** 0.5)
    # AnyMaze-class: centroid error small and stable (no teleport to noise).
    assert max(errs) < 6.0, f"max centroid error {max(errs):.1f}px too high"
    assert sum(errs) / len(errs) < 3.0


def test_pooled_bg_buffer_reused_across_frames():
    # running_avg mode overwrites background_gray in the pooled buffer
    # instead of allocating a new array each frame.
    H, W = 80, 100
    bg = np.full((H, W), 200, np.uint8)
    t = BlobTracker(setup_id=1)
    # Explicit: the tracker defaults to "static" (the captured reference
    # frame stays the reference), and only running_avg pools this buffer.
    t.update_params(bg_mode="running_avg")
    t.min_area = 50
    t.subject_min_area = 50
    t.initialize(bg)
    t.update(_disc(np.full((H, W), 200, np.uint8), 40, 40), timestamp=0.0)
    ref1 = t.background_gray
    t.update(_disc(np.full((H, W), 200, np.uint8), 47, 40), timestamp=1.0)
    ref2 = t.background_gray
    assert ref1 is ref2                    # same pooled buffer object
    assert t._bg_gray_buf is ref2
