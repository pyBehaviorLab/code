"""A pose must be un-mapped with the window ITS OWN frame was cut from.

Inference finishes on an executor thread, so by the time a result comes back
the box has usually dispatched another frame. The crop window is stored per box
and moves with the animal, so a single shared slot holds the NEWER window by
then, and every keypoint is un-mapped by the wrong offset.

Under a letterbox this is invisible: the transform never changes, so the stale
one is the right one. Under a following window it is a small, constant-looking
lag in the overlay, a few pixels behind the animal, growing with speed. That
is what was reported, and it is what this pins.
"""
import pytest

from source.video.framebus.pose_sink import PoseSink
from source.video.tracking.crop_tracker import CropWindow


@pytest.fixture
def sink():
    s = PoseSink()
    s._body_parts = ["centroid"]
    s._confidence = 0.5
    s._input_mode = "crop_track"
    s._crop_wh = (224, 176)
    return s


def _pose_at(x, y):
    """A pose in CROP pixels."""
    return {"centroid": [float(x), float(y), 0.9]}


def test_the_transform_the_caller_supplies_wins(sink):
    """The dispatch hands down the window it cut with; a newer one that has
    since landed in the shared slot must not be used instead."""
    seen = []
    sink.on_result(lambda bid, cfid, arr, loc, spd, zones, raw, **kw:
                   seen.append(raw))

    cut_with = CropWindow(x0=100, y0=50, w=224, h=176)
    # A newer frame has already replaced the slot by the time the result lands.
    sink._input_transform[1] = CropWindow(x0=140, y0=90, w=224, h=176)

    sink._on_pose_done(1, 1, _pose_at(10, 10), transform=cut_with)

    assert seen, "no result was delivered"
    x, y, _c = seen[0]["centroid"]
    assert (x, y) == (110.0, 60.0), (
        f"un-mapped with the wrong window: expected the frame's own "
        f"(100, 50) offset, got ({x}, {y})")


def test_without_a_supplied_transform_it_falls_back_to_the_slot(sink):
    """The direct-call path, used by tests and by any caller that has not
    been threaded through, must keep working."""
    seen = []
    sink.on_result(lambda bid, cfid, arr, loc, spd, zones, raw, **kw:
                   seen.append(raw))
    sink._input_transform[1] = CropWindow(x0=7, y0=3, w=224, h=176)

    sink._on_pose_done(1, 1, _pose_at(1, 2))

    x, y, _c = seen[0]["centroid"]
    assert (x, y) == (8.0, 5.0)


def test_a_none_transform_is_respected_and_not_treated_as_absent(sink):
    """``full`` mode has NO transform, and None is that answer, not "the
    caller forgot". A sentinel is the difference; without one, a box in full
    mode would silently pick up whatever the slot last held."""
    seen = []
    sink.on_result(lambda bid, cfid, arr, loc, spd, zones, raw, **kw:
                   seen.append(raw))
    sink._input_mode = "full"
    sink._input_transform[1] = CropWindow(x0=999, y0=999, w=10, h=10)

    sink._on_pose_done(1, 1, _pose_at(4, 6), transform=None)

    x, y, _c = seen[0]["centroid"]
    assert (x, y) == (4.0, 6.0), "a None transform must leave coordinates alone"


def test_the_dispatch_carries_one_transform_per_box(sink):
    """Two boxes cut at different places must each get their own window back,
    not whichever was written to the slot last."""
    seen = {}
    sink.on_result(lambda bid, cfid, arr, loc, spd, zones, raw, **kw:
                   seen.__setitem__(bid, raw))

    windows = {1: CropWindow(x0=10, y0=20, w=224, h=176),
               2: CropWindow(x0=300, y0=200, w=224, h=176)}
    for bid in (1, 2):
        sink._on_pose_done(bid, 1, _pose_at(5, 5), transform=windows[bid])

    assert seen[1]["centroid"][:2] == [15.0, 25.0]
    assert seen[2]["centroid"][:2] == [305.0, 205.0]


def test_a_moving_window_would_have_shown_the_bug(sink):
    """The regression, stated as the symptom: with the window advancing every
    frame, using the latest one puts each keypoint short by exactly how far
    the window moved."""
    seen = []
    sink.on_result(lambda bid, cfid, arr, loc, spd, zones, raw, **kw:
                   seen.append(raw["centroid"][:2]))

    # The animal moves 6 px per frame; the window follows it.
    for i in range(5):
        cut_with = CropWindow(x0=i * 6, y0=0, w=224, h=176)
        sink._input_transform[1] = CropWindow(x0=(i + 1) * 6, y0=0,
                                              w=224, h=176)  # newer frame
        sink._on_pose_done(1, i, _pose_at(112, 88), transform=cut_with)

    got = [x for x, _y in seen]
    assert got == [112.0, 118.0, 124.0, 130.0, 136.0], (
        f"each result must use its own window's offset, got {got}")
