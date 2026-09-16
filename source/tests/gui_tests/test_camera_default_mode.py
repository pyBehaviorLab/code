"""A fresh camera must default to a mode it can actually sustain.

"Biggest resolution wins" makes a webcam's 4K MJPG mode at 5 fps the default
for a behavioural session. The recorder and tracker sinks are lossless, so
every frame the camera cannot deliver on time becomes a drop-log entry rather
than a silently skipped frame.

The rule these tests pin is frame rate as a floor and resolution as the
decider, within a pixel budget, see ``pick_default_mode``. Its two consumers,
the picker combos and the headless defaults path, are covered next door in
``test_camera_default_selection``.
"""
from __future__ import annotations

import pytest

from source.video.framebus.types import (
    DEFAULT_MODE_MAX_MP,
    FPS_MIN,
    FPS_STEP,
    FPS_TARGET_DEFAULT,
    FPS_USABLE_MIN,
    default_fps_for,
    pick_default_mode,
)

pytestmark = pytest.mark.unit


# --- pick_default_mode ------------------------------------------------------

def test_a_slow_huge_mode_never_wins_over_a_fast_smaller_one():
    """The regression this whole rule exists for."""
    modes = [(3840, 2160, 5.0), (1920, 1080, 30.0), (640, 480, 30.0)]
    assert pick_default_mode(modes) == (1920, 1080, 30.0)


def test_largest_resolution_wins_among_modes_that_hold_the_target():
    modes = [(640, 480, 30.0), (1280, 720, 30.0), (1920, 1080, 30.0)]
    assert pick_default_mode(modes) == (1920, 1080, 30.0)


def test_resolution_is_capped_at_the_pixel_budget():
    """4K at a full 30 fps is still not a default, the tracker downscales it."""
    modes = [(3840, 2160, 30.0), (1920, 1080, 30.0)]
    w, h, _ = pick_default_mode(modes)
    assert w * h <= DEFAULT_MODE_MAX_MP * 1_000_000
    assert (w, h) == (1920, 1080)


def test_1080p_is_inside_the_budget():
    """The budget must not be so tight that the common rig mode is excluded."""
    assert DEFAULT_MODE_MAX_MP * 1_000_000 >= 1920 * 1080


def test_a_few_fps_are_not_worth_half_the_pixels():
    """The frame-rate bar is a floor, not the target.

    720p/25 against 480p/30: an exact-target rule picks the small one and
    throws away 2.25x the pixels to gain five frames a second.
    """
    assert pick_default_mode(
        [(1280, 720, 25.0), (640, 480, 30.0)]) == (1280, 720, 25.0)


def test_but_a_genuinely_slow_mode_still_loses():
    """The floor has to bite somewhere, or 4K/5 comes straight back."""
    assert pick_default_mode(
        [(1280, 720, 8.0), (640, 480, 30.0)]) == (640, 480, 30.0)


def test_the_floor_leaves_headroom_below_the_target():
    """If these were equal the rule would be "exact target" again."""
    assert FPS_USABLE_MIN < FPS_TARGET_DEFAULT


def test_when_every_fast_mode_is_over_budget_take_the_smallest_fast_one():
    """Better a big-but-fast frame than a slow one: drops are the worse failure."""
    modes = [(4096, 2160, 60.0), (3840, 2160, 30.0), (1920, 1080, 12.0)]
    assert pick_default_mode(modes) == (3840, 2160, 30.0)


def test_when_nothing_reaches_the_target_take_the_fastest():
    modes = [(1920, 1080, 8.0), (640, 480, 20.0), (320, 240, 20.0)]
    # 20 fps beats 8; the two 20 fps modes tie-break on resolution.
    assert pick_default_mode(modes) == (640, 480, 20.0)


def test_unparseable_and_empty_input():
    assert pick_default_mode([]) is None
    assert pick_default_mode(None) is None
    assert pick_default_mode([("wide", "tall", "fast")]) is None
    # A bad entry alongside good ones is skipped, not fatal.
    assert pick_default_mode([(None, 2, 3), (640, 480, 30.0)]) == (640, 480, 30.0)


def test_string_modes_are_accepted():
    """probed_modes round-trips through JSON, so ints can arrive as strings."""
    assert pick_default_mode([("1280", "720", "30")]) == (1280, 720, 30.0)


# --- default_fps_for --------------------------------------------------------

def test_a_fast_camera_defaults_to_the_target_not_its_ceiling():
    assert default_fps_for(120.0) == FPS_TARGET_DEFAULT
    assert default_fps_for(60.0) == FPS_TARGET_DEFAULT


def test_a_slow_camera_defaults_to_its_own_rounded_ceiling():
    assert default_fps_for(24.2) == 25      # nearest step, not floor
    assert default_fps_for(20.0) == 20


def test_the_default_never_exceeds_what_was_measured():
    for ceiling in (11.0, 17.0, 23.0, 29.2, 31.0, 59.0):
        got = default_fps_for(ceiling)
        assert got <= round(ceiling / FPS_STEP) * FPS_STEP


def test_degenerate_ceilings_fall_back_to_the_floor():
    assert default_fps_for(0.0) == FPS_MIN
    assert default_fps_for(None) == FPS_MIN
    assert default_fps_for("nonsense") == FPS_MIN
    assert default_fps_for(2.0) == FPS_MIN   # below FPS_MIN clamps up
