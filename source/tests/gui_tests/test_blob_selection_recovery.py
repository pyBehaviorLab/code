"""Blob tracker: the animal must win over a distractor, and a track that has
latched onto one must be able to come back.

The failure these pin down is the one operators actually hit, the mask shows
the mouse, the bounding box sits on a bedding clump, and nothing recovers for
the rest of the session.
"""
import numpy as np

from source.video.tracking.blob import BlobTracker

H, W = 240, 320


def _arena(level=200):
    return np.full((H, W), level, np.uint8)


def _blob(frame, cx, cy, r, level=40):
    """Paint a filled dark disc; returns the frame for chaining."""
    h, w = frame.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    frame[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = level
    return frame


def _tracker():
    t = BlobTracker(setup_id=1)
    t.initialize(_arena())
    return t


def test_shipped_defaults_find_a_mouse_sized_animal():
    """A max_area default below the animal's own contour leaves only noise
    surviving the filter."""
    t = _tracker()
    frame = _blob(_arena(), 160, 120, r=25)      # ~1960 px2, a mouse
    ok, pos = t.update(frame, timestamp=0.0)
    assert ok, "shipped defaults rejected a mouse-sized blob"
    bx, by, bw, bh = pos
    assert abs((bx + bw / 2) - 160) < 8
    assert abs((by + bh / 2) - 120) < 8


def test_animal_beats_a_smaller_distractor():
    t = _tracker()
    frame = _arena()
    _blob(frame, 60, 40, r=10)                   # bedding clump, ~314 px2
    _blob(frame, 160, 120, r=25)                 # the animal
    ok, pos = t.update(frame, timestamp=0.0)
    assert ok
    bx, _by, bw, _bh = pos
    assert abs((bx + bw / 2) - 160) < 10, "locked onto the distractor"


def test_track_recovers_after_latching_onto_a_distractor():
    """A stationary distractor sits inside its own gate forever, so a hard
    spatial gate could never let the animal back in. Selection now scores
    proximity against size plausibility, and a sustained size mismatch drops
    the motion model entirely."""
    t = _tracker()
    # Build a normal track on the animal so the median area is the animal's.
    for i in range(12):
        ok, _ = t.update(_blob(_arena(), 150 + i, 120, r=25), timestamp=i * 0.1)
        assert ok
    med = t._median_recent_area()
    assert med > 1000

    # Force the track onto a small stationary clump near the last position.
    t._predicted_centroid = (60.0, 40.0)
    t._velocity = (0.0, 0.0)

    # The animal is present every frame, well away from the bogus prediction.
    recovered_at = None
    for i in range(40):
        frame = _arena()
        _blob(frame, 60, 40, r=10)               # the clump, stationary
        _blob(frame, 200, 160, r=25)             # the animal, stationary
        ok, pos = t.update(frame, timestamp=2.0 + i * 0.1)
        if ok and pos is not None:
            bx, by, bw, bh = pos
            if abs((bx + bw / 2) - 200) < 15 and abs((by + bh / 2) - 160) < 15:
                recovered_at = i
                break
    assert recovered_at is not None, (
        "track never came back to the animal, the gate is still absorbing")


def test_oversize_rejections_are_reported():
    """A Max Area below the animal has to be visible, not silent."""
    t = _tracker()
    t.update_params(max_area=500)
    t.update(_blob(_arena(), 160, 120, r=25), timestamp=0.0)
    m = t.get_quality_metrics()
    assert m["rejected_too_large"] >= 1
    assert m["largest_rejected_area"] > 1000


def test_mismatched_background_warns_once(caplog):
    """Stretching a wrong-sized reference mis-registers every arena edge; it
    keeps the box alive but must not do so silently."""
    import logging
    t = _tracker()
    caplog.set_level(logging.WARNING)
    small = np.full((H // 2, W // 2), 200, np.uint8)
    t.update(_blob(small.copy(), 80, 60, r=12), timestamp=0.0)
    warnings = [r for r in caplog.records if "re-capture" in r.getMessage()]
    assert len(warnings) == 1
    # Still warned only once after a second mismatched frame.
    t.update(_blob(small.copy(), 82, 60, r=12), timestamp=0.1)
    warnings = [r for r in caplog.records if "re-capture" in r.getMessage()]
    assert len(warnings) == 1


def test_challenger_size_check_uses_the_median_not_the_last_blob():
    """Anchoring on _last_area meant that once the track went wrong, the wrong
    blob's area became the reference and the animal was rejected for being the
    wrong size."""
    t = _tracker()
    for i in range(10):
        t.update(_blob(_arena(), 150, 120, r=25), timestamp=i * 0.1)
    med = t._median_recent_area()
    # Pretend the last committed blob was a tiny speck.
    t._last_area = 60.0
    animal = (med, (0, 0, 50, 50), (200.0, 160.0), None)
    assert t._challenger_confirmed(animal) or t._challenger_streak >= 1, (
        "an animal-sized challenger was rejected against a speck")
