"""What each Condition actually does when a frame arrives.

The trigger list offers fifteen conditions and the UI gave no way to tell which
of them genuinely fire. Two answers, both pinned here, drive the panel's help
text and its blob gating:

* ``in_zone`` and ``enter_zone`` are the same rule under two names, and so are
  ``not_in_zone`` and ``exit_zone``: every condition is edge-fired in
  ``TrackingPushPolicy.evaluate``, so a "level" reading of ``in_zone`` was never
  what happened. Both names stay (``in_zone`` is the default and sits in saved
  configs); the panel now says they are equivalent rather than implying a
  difference that does not exist.

* Orientation and posture rules need a head-tail axis, two distinct keypoints.
  Under blob tracking there is one centroid, the heading is ``None``, and the
  rule can never fire. That is why the panel disables them under blob instead of
  offering a control that silently does nothing.

Driven through the real Pipeline → MCUPusher → board chain, same as
``test_trigger_e2e``.
"""
from __future__ import annotations

import pytest

from source.video.framebus.controller import Pipeline


class _FakeBoard:
    framework_running = True

    def __init__(self):
        self.fired = []

    def queue_trigger_event(self, name):
        self.fired.append(name)

    def queue_set_coordinates(self, name, val):
        pass

    def queue_trigger_intrinsic_event(self, name):
        pass


def _install(pipe, setup_id, triggers):
    board = _FakeBoard()
    pipe.push.register_box(setup_id, board)
    pipe.push._policies[setup_id].update(triggers=triggers)
    return board


def _pose(pipe, setup_id, zones, coords=None, speed=0.0):
    """One pose frame. ``zones`` is the per-body-part zone membership the
    segmenter would have produced."""
    pipe.push.on_pose_result(
        setup_id, cam_frame_id=1, pose_array=[], location=None, speed=speed,
        zones_by_body_part=zones,
        raw_pose_dict=coords or {"centroid": (0, 0, 0.9)})


def _run(triggers, sequence):
    """Feed ``sequence`` (a list of zone-membership dicts) through one box and
    return the events the board received."""
    pipe = Pipeline(target_fps=30)
    try:
        board = _install(pipe, 1, triggers)
        for zones in sequence:
            _pose(pipe, 1, zones)
        return list(board.fired)
    finally:
        pipe.shutdown()


# Zone membership is per body part, per zone: {body_part: {zone: bool}}.
IN = {"centroid": {"ZoneA": True}}
OUT = {"centroid": {"ZoneA": False}}


# --- the two zone namesakes -------------------------------------------------

@pytest.mark.parametrize("cond", ["in_zone", "enter_zone"])
def test_entry_conditions_fire_on_the_entering_frame_only(cond):
    trig = {"condition": cond, "zones": ["ZoneA"],
            "body_part": "centroid", "event_name": "hit"}
    # out, in, still-in, out, in  →  two entries, so two events.
    fired = _run([trig], [OUT, IN, IN, OUT, IN])
    assert fired == ["hit", "hit"], (
        f"{cond} must fire once per entry; it is an edge, not a level")


def test_in_zone_and_enter_zone_are_the_same_rule():
    """If these ever diverge the panel's help ("Identical to enter_zone") lies."""
    seq = [OUT, IN, IN, OUT, OUT, IN, OUT]
    a = _run([{"condition": "in_zone", "zones": ["ZoneA"],
               "body_part": "centroid", "event_name": "e"}], seq)
    b = _run([{"condition": "enter_zone", "zones": ["ZoneA"],
               "body_part": "centroid", "event_name": "e"}], seq)
    assert a == b == ["e", "e"]


def test_not_in_zone_and_exit_zone_are_the_same_rule():
    seq = [IN, OUT, OUT, IN, OUT]
    a = _run([{"condition": "not_in_zone", "zones": ["ZoneA"],
               "body_part": "centroid", "event_name": "e"}], seq)
    b = _run([{"condition": "exit_zone", "zones": ["ZoneA"],
               "body_part": "centroid", "event_name": "e"}], seq)
    assert a == b == ["e", "e"]


def test_a_zone_the_rule_does_not_name_is_ignored():
    fired = _run([{"condition": "in_zone", "zones": ["ZoneA"],
                   "body_part": "centroid", "event_name": "hit"}],
                 [OUT,
                  {"centroid": {"ZoneA": False, "ZoneB": True}},
                  {"centroid": {"ZoneA": True, "ZoneB": True}}])
    assert fired == ["hit"], "only the frame that entered ZoneA counts"


def test_the_rule_follows_its_own_body_part():
    """A rule watching the nose must ignore the tail's zone membership."""
    fired = _run([{"condition": "in_zone", "zones": ["ZoneA"],
                   "body_part": "nose", "event_name": "hit"}],
                 [{"nose": {"ZoneA": False}, "tail": {"ZoneA": True}},
                  {"nose": {"ZoneA": False}, "tail": {"ZoneA": True}},
                  {"nose": {"ZoneA": True}, "tail": {"ZoneA": False}}])
    assert fired == ["hit"]


# --- orientation under blob -------------------------------------------------

@pytest.mark.parametrize("cond", ["rotation_gt", "rotation_lt", "elongation_gt",
                                  "rearing", "head_angle_gt", "head_angle_lt"])
def test_axis_conditions_never_fire_from_a_single_centroid(cond):
    """The reason the panel greys these out under blob.

    A blob tracker delivers one point. Heading is undefined, so however the
    centroid moves these rules stay silent, configuring one under blob would
    look like a working trigger and produce nothing.
    """
    pipe = Pipeline(target_fps=30)
    try:
        board = _install(pipe, 1, [
            {"condition": cond, "threshold": 0.0,
             "body_part": "centroid", "part_b": "centroid",
             "event_name": "turned"}])
        # Sweep the centroid through a quarter circle: plenty of motion, but
        # no second keypoint to measure an orientation against.
        for x, y in [(0, 0), (10, 3), (20, 12), (26, 25), (28, 40)]:
            _pose(pipe, 1, {}, {"centroid": (x, y, 0.95)}, speed=5.0)
        assert board.fired == [], (
            f"{cond} fired with one keypoint, either the evaluator changed or "
            f"the blob gating in the panel is now wrong")
    finally:
        pipe.shutdown()


def test_rotation_does_fire_once_a_second_keypoint_exists():
    """The other half of the claim: with a head-tail axis the rule is real, so
    the gating is about the tracker's output and not a dead condition."""
    pipe = Pipeline(target_fps=30)
    try:
        board = _install(pipe, 1, [
            {"condition": "rotation_gt", "threshold": 1.0,
             "body_part": "head", "event_name": "turned"}])
        pipe.push.set_features_context(1, axis=("tail", "head"))
        # Rotate the head-tail axis a quarter turn about a fixed tail.
        for hx, hy in [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0)]:
            _pose(pipe, 1, {},
                  {"head": (hx, hy, 0.95), "tail": (0.0, 0.0, 0.95)})
        assert board.fired, (
            "rotation_gt produced nothing even with two keypoints, the "
            "condition would be dead everywhere, not just under blob")
    finally:
        pipe.shutdown()
