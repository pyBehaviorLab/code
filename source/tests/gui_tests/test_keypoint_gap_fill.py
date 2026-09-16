"""Missing keypoints are carried across, briefly, and never passed off as real.

Before this, "Kalman and optical flow" did nothing for keypoints. The filter
followed ONE point, the centroid, and its output went to a single consumer: the
forecast that keeps the board's zone lookup from being late. Optical flow was
switched off on the pose path outright (``frame_gray`` passed as ``None``), the
smoothed value ``update()`` returned was discarded, and no body part was ever
smoothed, held or filled. Every dropout the network had reached the overlay,
the recorded row and the triggers exactly as it came out.

Two rules the fill has to obey. It is SHORT, because predicting a keypoint
forward is honest for the fraction of a second a paw is hidden by the body and
invents an animal if carried further. And a filled point is never presented as
a measured one.
"""
from __future__ import annotations

import pytest

from source.video.tracking.smoothing import (DEFAULT_MAX_GAP_FRAMES,
                                             KeypointFilter, PoseGapFiller)

DT = 1 / 30.0


def _run(filt, seq):
    return [filt.update(*pt, DT) for pt in seq]


class TestOneKeypoint:
    def test_a_measured_point_is_not_marked_filled(self):
        f = KeypointFilter(max_gap=5)
        x, y, c, filled = f.update(10.0, 20.0, 0.9, DT)
        assert (x, y) == (10.0, 20.0)
        assert filled is False

    def test_a_short_gap_is_carried(self):
        f = KeypointFilter(max_gap=5)
        f.update(10.0, 10.0, 0.9, DT)
        f.update(12.0, 10.0, 0.9, DT)
        x, y, c, filled = f.update(None, None, 0.0, DT)
        assert filled is True
        assert x is not None and y is not None

    def test_a_long_gap_reports_missing(self):
        """The honest answer for a part that is simply not visible."""
        f = KeypointFilter(max_gap=3)
        f.update(10.0, 10.0, 0.9, DT)
        for _ in range(3):
            assert f.update(None, None, 0.0, DT)[3] is True
        x, y, c, filled = f.update(None, None, 0.0, DT)
        assert (x, y, filled) == (None, None, False)

    def test_confidence_decays_across_the_gap(self):
        """So a consumer that thresholds on confidence drops a filled point
        before the gap even runs out, without knowing about any of this."""
        f = KeypointFilter(max_gap=5)
        f.update(10.0, 10.0, 0.9, DT)
        confs = [f.update(None, None, 0.0, DT)[2] for _ in range(4)]
        assert confs == sorted(confs, reverse=True)
        assert confs[-1] < 0.9

    def test_a_measurement_ends_the_gap(self):
        f = KeypointFilter(max_gap=5)
        f.update(10.0, 10.0, 0.9, DT)
        f.update(None, None, 0.0, DT)
        x, y, c, filled = f.update(11.0, 10.0, 0.8, DT)
        assert filled is False
        assert c == pytest.approx(0.8)

    def test_a_below_threshold_detection_counts_as_missing(self):
        """A keypoint the network is not confident about is noise, not a
        position. Treating it as a measurement is how the marker ends up
        somewhere the animal is not."""
        f = KeypointFilter(max_gap=5, min_conf=0.3)
        f.update(10.0, 10.0, 0.9, DT)
        assert f.update(500.0, 500.0, 0.05, DT)[3] is True

    def test_nothing_is_invented_before_the_first_detection(self):
        f = KeypointFilter(max_gap=5)
        assert f.update(None, None, 0.0, DT) == (None, None, 0.0, False)

    def test_a_moving_point_is_carried_along_its_motion(self):
        """Holding it still would put the marker where the animal was, which
        is the same error the fill exists to remove."""
        f = KeypointFilter(max_gap=5)
        for x in (10.0, 20.0, 30.0, 40.0):
            f.update(x, 10.0, 0.9, DT)
        x, _, _, filled = f.update(None, None, 0.0, DT)
        assert filled is True
        assert x > 40.0


class TestAWholePose:
    def test_only_the_missing_parts_are_named(self):
        g = PoseGapFiller(max_gap=5)
        g.apply({"Snout": [10, 10, 0.9], "Tail": [30, 10, 0.9]}, DT)
        out, filled = g.apply({"Snout": [11, 10, 0.9], "Tail": None}, DT)
        assert filled == ["Tail"]
        assert out["Tail"] is not None
        assert out["Snout"] is not None

    def test_the_callers_dict_is_not_mutated(self):
        """The raw result is still what anyone asking "what did the network
        actually say" has to be able to read."""
        g = PoseGapFiller(max_gap=5)
        raw = {"Snout": [10, 10, 0.9]}
        g.apply(raw, DT)
        assert raw == {"Snout": [10, 10, 0.9]}

    def test_metadata_keys_pass_through_untouched(self):
        g = PoseGapFiller(max_gap=5)
        out, _ = g.apply({"Snout": [10, 10, 0.9], "_rotation_rad": 1.2}, DT)
        assert out["_rotation_rad"] == 1.2

    def test_a_part_the_model_never_reports_costs_nothing(self):
        g = PoseGapFiller(max_gap=5)
        out, filled = g.apply({"Snout": None}, DT)
        assert out["Snout"] is None
        assert filled == []

    def test_parts_are_filled_independently(self):
        g = PoseGapFiller(max_gap=2)
        g.apply({"A": [0, 0, 0.9], "B": [5, 5, 0.9]}, DT)
        for _ in range(3):
            out, filled = g.apply({"A": [1, 1, 0.9], "B": None}, DT)
        assert filled == []            # B's gap outlasted max_gap
        assert out["A"] is not None
        assert out["B"] is None


def test_the_default_gap_is_about_a_sixth_of_a_second():
    """Short on purpose. Named here so a future change to the constant has to
    be a decision rather than a drift."""
    assert DEFAULT_MAX_GAP_FRAMES == 5
    assert 0.1 < DEFAULT_MAX_GAP_FRAMES / 30.0 < 0.25


def test_the_sink_can_turn_it_off():
    """An operator who wants the raw network output gets exactly that."""
    from source.video.framebus.pose_sink import PoseSink

    sink = PoseSink()
    sink.set_gap_fill(1, 5)
    assert 1 in sink._gap_fillers
    sink.set_gap_fill(1, 0)
    assert 1 not in sink._gap_fillers


def test_the_sink_marks_filled_parts_in_the_pose_dict():
    """Under the underscore-prefixed metadata convention every consumer
    already skips, so the fact reaches the recorder without changing a single
    callback signature."""
    from source.video.framebus.pose_sink import PoseSink

    sink = PoseSink()
    sink.set_gap_fill(1, 5)
    sink._fill_pose_gaps(1, {"Snout": [10, 10, 0.9]}, 1_000_000_000)
    out = sink._fill_pose_gaps(1, {"Snout": None}, 1_033_000_000)
    assert out["_filled_parts"] == ["Snout"]


def test_no_filler_means_the_pose_is_returned_untouched():
    from source.video.framebus.pose_sink import PoseSink

    sink = PoseSink()
    raw = {"Snout": None}
    assert sink._fill_pose_gaps(1, raw, 1_000_000_000) is raw
