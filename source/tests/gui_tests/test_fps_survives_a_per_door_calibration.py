"""A saved frame rate must survive being loaded again.

Detect records what each OS backend measured, per door, in
``probed_variants``. A camera calibrated that way can have an EMPTY
``probed_modes``, and the rate ladder was built from ``probed_modes`` alone.

So on load the ladder came out empty, ``clamp_fps_to_options`` concluded the
saved rate was no longer valid, and dropped it. A camera saved at 25 fps came
back at whatever the row happened to default to, silently. Nothing said the
saved value had been discarded, and the next session ran at the wrong rate.
"""
from __future__ import annotations

import pytest

from source.video.framebus.types import CameraConfig

TWO_DOOR = {
    "dshow": [[1920, 1080, 5.0], [1280, 720, 10.0], [640, 480, 29.9]],
    "msmf": [[1920, 1080, 30.4], [1280, 720, 29.9], [640, 480, 29.9]],
}


def _per_door_camera(fps=25):
    cfg = CameraConfig(camera_id="camA")
    cfg.probed_variants = {k: [tuple(m) for m in v] for k, v in TWO_DOOR.items()}
    cfg.selected_resolution = (1280, 720)
    cfg.selected_fps = fps
    cfg.capture_backend = "msmf"
    return cfg


def test_the_ladder_is_built_from_what_the_doors_measured():
    cfg = _per_door_camera()
    assert cfg.probed_modes == [], "this is the case that was unhandled"
    assert cfg.max_fps_for((1280, 720)) == 29.9, (
        "the ceiling is the best door's, and both doors measured this size")
    assert 25 in cfg.fps_options_for((1280, 720))


def test_a_saved_rate_is_not_dropped_on_load():
    cfg = _per_door_camera(25)
    cfg.clamp_fps_to_options()
    assert cfg.selected_fps == 25, (
        "the rate the operator chose was discarded because the ladder came "
        "out empty")


def test_a_rate_the_camera_cannot_reach_is_still_corrected():
    """The clamp still does its job; it just has the facts now."""
    cfg = _per_door_camera(60)
    cfg.clamp_fps_to_options()
    assert cfg.selected_fps == 30, cfg.selected_fps


def test_probed_modes_still_wins_when_it_is_present():
    """The flat list is the older, chosen-door record and stays authoritative."""
    cfg = _per_door_camera()
    cfg.probed_modes = [(1280, 720, 10.0)]
    assert cfg.max_fps_for((1280, 720)) == 10.0


def test_a_size_no_door_measured_has_no_ceiling():
    cfg = _per_door_camera()
    assert cfg.max_fps_for((3840, 2160)) is None
    assert cfg.fps_options_for((3840, 2160)) == []


@pytest.mark.parametrize("size,expect", [((1920, 1080), 30.4),
                                         ((1280, 720), 29.9),
                                         ((640, 480), 29.9)])
def test_the_best_door_sets_the_ceiling(size, expect):
    """1080p is 30.4 fps through one door and 5.0 through the other."""
    assert _per_door_camera().max_fps_for(size) == expect
