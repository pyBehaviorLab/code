"""The four conditions nothing exercised: speed_lt, facing_line, distance_gt/lt.

The dialog offers sixteen conditions and the evaluator answers all sixteen, but
a coverage check found four that no test had ever fired. A condition nobody has
seen fire is a condition nobody knows fires, and the failure mode when one does
not is silence, not an error: the rule sits in the saved config, the session
runs, and the behaviour simply never happens.

Driven through the real Pipeline → MCUPusher → board chain, the same way
``test_trigger_condition_semantics`` does, so what is measured is what runs.
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


def _run(trigger, frames):
    """Feed ``frames``, each ``(speed, pose_dict)``, through one box."""
    pipe = Pipeline(target_fps=30)
    try:
        board = _FakeBoard()
        pipe.push.register_box(1, board)
        pipe.push._policies[1].update(triggers=[trigger])
        for speed, pose in frames:
            pipe.push.on_pose_result(
                1, cam_frame_id=1, pose_array=[], location=None, speed=speed,
                zones_by_body_part={}, raw_pose_dict=pose)
        return list(board.fired)
    finally:
        pipe.shutdown()


def _body(head=(100.0, 100.0), tail=(60.0, 100.0), extra=None):
    """A pose with a head-tail axis, which the orientation rules need."""
    pose = {"Head": (head[0], head[1], 0.95),
            "Tail_Base": (tail[0], tail[1], 0.95),
            "centroid": ((head[0] + tail[0]) / 2,
                         (head[1] + tail[1]) / 2, 0.95)}
    pose.update(extra or {})
    return pose


def _moving(xs):
    """Frames whose CENTROID walks through ``xs``.

    The speed rules read ``FeatureTracker.speed_in``, which the tracker derives
    from successive centroid positions, not the ``speed`` argument the sink
    also carries. Feeding that argument and leaving the pose still is what an
    obvious-looking test does, and it measures nothing.
    """
    return [(0.0, _body(head=(x + 20.0, 100.0), tail=(x, 100.0))) for x in xs]


class TestSpeedLessThan:
    """The counterpart of ``speed_gt``, and the one an immobility rule uses."""

    def test_it_fires_when_the_animal_slows_below_the_threshold(self):
        trig = {"condition": "speed_lt", "threshold": 5.0,
                "body_part": "centroid", "event_name": "slow", "unit": "px"}
        # Big steps, then a stop.
        fired = _run(trig, _moving([0, 200, 400, 600, 600, 600]))
        assert fired == ["slow"]

    def test_it_is_an_edge_not_a_level(self):
        """Twenty still frames must produce ONE event, not twenty.

        Stated as a single long stop rather than two crossings on purpose.
        ``speed_in`` is an EMA and the frames here are fed in a tight loop, so
        ``dt`` is near zero and a moving frame's speed is enormous; how many
        still frames it then takes to decay back under a threshold is a
        property of the filter and the loop timing, not of the rule. The rule's
        own contract, one event per crossing, does not depend on either.
        """
        trig = {"condition": "speed_lt", "threshold": 5.0,
                "body_part": "centroid", "event_name": "slow", "unit": "px"}
        fired = _run(trig, _moving([0, 300, 600] + [600] * 20))
        assert fired == ["slow"], (
            f"one event per crossing, not one per slow frame, got {fired}")

    def test_an_unset_threshold_does_not_raise(self):
        """``value < None`` is a TypeError, and it would land in the middle of
        a recording."""
        trig = {"condition": "speed_lt", "body_part": "centroid",
                "event_name": "slow", "unit": "px"}
        assert _run(trig, _moving([0, 300, 300, 300])) == []


class TestDistanceBetweenTwoParts:
    """``distance_gt`` / ``distance_lt`` measure Body Part to Part B."""

    def _trig(self, cond, threshold):
        # ``part_a``, not ``body_part``: for the distance conditions the panel
        # copies the Body Part column into ``part_a`` and the Part B column
        # into ``part_b``, and the evaluator reads that pair. A trigger dict
        # carrying only ``body_part`` measures nothing.
        return {"condition": cond, "threshold": threshold,
                "part_a": "Head", "part_b": "Tail_Base",
                "body_part": "Head", "event_name": "d", "unit": "px"}

    def test_distance_gt_fires_when_the_parts_separate(self):
        near = _body(head=(70.0, 100.0), tail=(60.0, 100.0))     # 10 px
        far = _body(head=(160.0, 100.0), tail=(60.0, 100.0))     # 100 px
        assert _run(self._trig("distance_gt", 50.0),
                    [(0.0, near), (0.0, far)]) == ["d"]

    def test_distance_lt_fires_when_they_close(self):
        near = _body(head=(70.0, 100.0), tail=(60.0, 100.0))
        far = _body(head=(160.0, 100.0), tail=(60.0, 100.0))
        assert _run(self._trig("distance_lt", 50.0),
                    [(0.0, far), (0.0, near)]) == ["d"]

    def test_a_missing_part_b_fires_nothing_rather_than_raising(self):
        """Part B is a separate dropdown; a rule saved before it was chosen,
        or naming a keypoint this model does not have, must be inert and not
        an exception on the inference path."""
        trig = {"condition": "distance_gt", "threshold": 5.0,
                "part_a": "Head", "part_b": "NoSuchPart",
                "event_name": "d", "unit": "px"}
        assert _run(trig, [(0.0, _body()), (0.0, _body(head=(300.0, 100.0)))]) == []

    def test_a_low_confidence_part_does_not_fire(self):
        """A keypoint the model is unsure about is not a measurement."""
        trig = self._trig("distance_gt", 50.0)
        weak = {"Head": (300.0, 100.0, 0.01), "Tail_Base": (60.0, 100.0, 0.95)}
        assert _run(trig, [(0.0, _body()), (0.0, weak)]) == []


class TestFacingLine:
    """Whether the animal's heading points across a configured line.

    Needs a head-tail axis: with one point the heading is None and the rule
    can never fire, which is why the panel disables it under blob tracking.
    """

    def test_it_does_not_raise_without_a_line_configured(self):
        trig = {"condition": "facing_line", "body_part": "centroid",
                "event_name": "faced"}
        assert _run(trig, [(0.0, _body()), (0.0, _body(head=(60.0, 140.0)))]) == []

    def test_it_cannot_fire_without_an_axis(self):
        """One keypoint is a centroid, not a heading. The rule must be inert,
        and inert quietly; this is the documented blob-tracking case."""
        trig = {"condition": "facing_line", "body_part": "centroid",
                "event_name": "faced", "zones": ["Line"]}
        blob = {"centroid": (100.0, 100.0, 0.9)}
        assert _run(trig, [(0.0, blob), (0.0, blob)]) == []


@pytest.mark.parametrize("cond", ["speed_lt", "facing_line",
                                  "distance_gt", "distance_lt"])
def test_the_condition_survives_a_frame_with_no_pose_at_all(cond):
    """Between detections the pose dict is empty. Every rule has to tolerate
    it: an exception here stops the push loop for the whole box."""
    trig = {"condition": cond, "threshold": 1.0, "body_part": "centroid",
            "part_a": "Head", "part_b": "Tail_Base",
            "event_name": "e", "unit": "px"}
    assert _run(trig, [(0.0, {}), (0.0, {}), (5.0, _body())]) is not None
