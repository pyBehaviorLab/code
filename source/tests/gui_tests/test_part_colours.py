"""Marker colours: distinct, never red, and the same next session.

Three faults, one function:

* the sweep ran hue 0 to 170, and both ends are red, so on a six-part model
  the snout and the tail base came out near enough the same dot;
* red is the interface's alarm colour (error borders, "camera not connected",
  stop), so a keypoint drawn in it reads as a fault across a wall of tiles;
* the colour came from the part's POSITION, so adding one keypoint repainted
  every other part and an operator who had learnt the overlay had to learn it
  again.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from source.gui.base import _HUE_MAX, _HUE_MIN, part_colour_map, part_colours

SIX = ["Snout", "Head", "Left_Ear", "Right_Ear", "center", "Tail_Base"]


def _hue(bgr) -> float:
    px = np.uint8([[list(bgr)]])
    return float(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0, 0])


def _is_red(bgr) -> bool:
    """OpenCV hue wraps at 180, so red sits at both ends of the scale."""
    h = _hue(bgr)
    return h < _HUE_MIN or h > _HUE_MAX


@pytest.mark.parametrize("n", range(1, 21))
def test_no_two_parts_share_a_colour(n):
    colours = part_colours(n)
    assert len(colours) == n
    assert len(set(colours)) == n


@pytest.mark.parametrize("n", range(1, 21))
def test_no_part_is_drawn_in_red(n):
    """Red belongs to the alarm states, not to an animal's nose."""
    assert not any(_is_red(c) for c in part_colours(n))


def test_the_first_and_last_part_are_far_apart():
    """The wrap. Hue 0 and hue 170 are both red, which is what made these two
    the same dot at six parts."""
    first, last = part_colours(6)[0], part_colours(6)[-1]
    assert abs(_hue(first) - _hue(last)) > 60


# ── stability by name ─────────────────────────────────────────────────


def test_a_saved_map_is_returned_unchanged():
    saved = {n: list(c) for n, c in zip(SIX, part_colours(6))}
    again = part_colour_map(SIX, saved)
    assert {n: list(c) for n, c in again.items()} == saved


def test_adding_a_part_leaves_the_others_where_they_were():
    """The whole point of keying by name. Indexing by position repainted every
    part whenever the model's list changed."""
    first = part_colour_map(SIX)
    second = part_colour_map(SIX + ["Left_Hind"],
                             {k: list(v) for k, v in first.items()})
    for name in SIX:
        assert second[name] == first[name]
    assert "Left_Hind" in second


def test_a_new_part_is_not_red_and_not_a_duplicate():
    first = part_colour_map(SIX)
    second = part_colour_map(SIX + ["Left_Hind"],
                             {k: list(v) for k, v in first.items()})
    assert not _is_red(second["Left_Hind"])
    assert len(set(second.values())) == len(second)


def test_a_new_part_takes_the_widest_free_gap():
    """Furthest from every colour in use, so a late addition is distinct from
    its neighbours rather than landing on one."""
    saved = {"a": list(part_colours(2)[0]), "b": list(part_colours(2)[1])}
    out = part_colour_map(["a", "b", "c"], saved)
    hues = sorted(_hue(out[k]) for k in ("a", "b"))
    new = _hue(out["c"])
    assert min(abs(new - h) for h in hues) > 20


def test_a_dropped_part_does_not_disturb_the_rest():
    first = part_colour_map(SIX)
    fewer = part_colour_map(SIX[:-1], {k: list(v) for k, v in first.items()})
    for name in SIX[:-1]:
        assert fewer[name] == first[name]
    assert "Tail_Base" not in fewer
