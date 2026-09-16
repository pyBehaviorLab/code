"""Per-box kinematic FeatureExtractor, the efficient online formulas."""
from __future__ import annotations

import math

import pytest

from source.video.features import FeatureExtractor


def test_body_length_running_median():
    fx = FeatureExtractor(axis=("tail", "head"))
    for d in (10, 12, 100, 11, 13):   # 100 is an outlier
        fx.update({"tail": (0, 0), "head": (d, 0)}, (d / 2, 0), speed_px_s=0, t_s=0)
    assert fx.body_length_px() == 12    # median ignores the 100 outlier


def test_speed_units_scale_and_bodylen():
    fx = FeatureExtractor(axis=("tail", "head"), px_per_mm=10.0)  # 10 px = 1 mm
    fx.update({"tail": (0, 0), "head": (40, 0)}, (20, 0), speed_px_s=200.0, t_s=0)
    # px/s
    assert fx.speed_in("px") == pytest.approx(200.0)
    # mm/s: 200 px/s ÷ 10 px/mm = 20 mm/s
    assert fx.speed_in("mm") == pytest.approx(20.0)
    # cm/s: ÷ 100 px/cm = 2 cm/s
    assert fx.speed_in("cm") == pytest.approx(2.0)
    # bodylen/s: body length = 40 px → 200/40 = 5 bl/s
    assert fx.speed_in("bodylen") == pytest.approx(5.0)


def test_speed_falls_back_when_no_scale():
    fx = FeatureExtractor(axis=("tail", "head"))  # no scale, no L yet
    fx.update({}, None, speed_px_s=123.0, t_s=0)
    assert fx.speed_in("cm") == pytest.approx(123.0)     # no scale → px passthrough
    assert fx.speed_in("bodylen") == pytest.approx(123.0)  # no L → px passthrough


def test_turning_rate_sign_and_magnitude():
    fx = FeatureExtractor(axis=("tail", "head"), ema_alpha=1.0)
    # heading along +x at t=0
    fx.update({"tail": (0, 0), "head": (10, 0)}, (5, 0), 0, t_s=0.0)
    # rotate heading +90° (to +y) over 0.5 s → +180 deg/s (CCW in math coords)
    fx.update({"tail": (0, 0), "head": (0, 10)}, (0, 5), 0, t_s=0.5)
    assert fx.turning_deg_s() == pytest.approx(180.0, abs=1e-6)


def test_turning_unwraps_across_180():
    fx = FeatureExtractor(axis=("tail", "head"))
    fx.update({"tail": (0, 0), "head": (-10, -1)}, None, 0, t_s=0.0)   # ~181°
    fx.update({"tail": (0, 0), "head": (-10, 1)}, None, 0, t_s=1.0)    # ~179°
    # a small step across the ±180 boundary → small unwrapped delta, NOT ~-348
    assert abs(fx.turning_deg_s()) < 30


def test_head_body_angle_straight_is_zero():
    fx = FeatureExtractor(axis=("tail", "nose"))
    fx.update({"tail": (0, 0), "neck": (10, 0), "nose": (20, 0)}, None, 0, t_s=0)
    assert fx.head_body_angle_deg("neck") == pytest.approx(0.0)


def test_head_body_angle_turned_left():
    fx = FeatureExtractor(axis=("tail", "nose"))
    # body along +x, head turned to +y at the neck → +90°
    fx.update({"tail": (0, 0), "neck": (10, 0), "nose": (10, 10)}, None, 0, t_s=0)
    assert fx.head_body_angle_deg("neck") == pytest.approx(90.0)


def test_facing_target():
    fx = FeatureExtractor(axis=("tail", "nose"))
    fx.update({"tail": (0, 0), "nose": (10, 0)}, None, 0, t_s=0)
    assert fx.facing_deg((100, 0)) == pytest.approx(0.0)     # dead ahead
    assert abs(fx.facing_deg((10, 100))) == pytest.approx(90.0)  # 90° off


def test_elongation():
    fx = FeatureExtractor(axis=("tail", "nose"))
    for _ in range(5):   # establish L = 40
        fx.update({"tail": (0, 0), "nose": (40, 0)}, None, 0, t_s=0)
    assert fx.elongation() == pytest.approx(1.0)
    fx.update({"tail": (0, 0), "nose": (20, 0)}, None, 0, t_s=0)  # foreshortened
    assert fx.elongation() == pytest.approx(20 / fx.body_length_px())
    assert fx.elongation() < 0.7


def test_distance_units():
    fx = FeatureExtractor(px_per_mm=2.0)
    fx.update({"a": (0, 0), "b": (20, 0)}, None, 0, t_s=0)
    assert fx.distance_in("a", "b", "px") == pytest.approx(20.0)
    assert fx.distance_in("a", "b", "mm") == pytest.approx(10.0)
    assert fx.distance_in("a", "c", "px") is None    # missing keypoint


def test_missing_axis_gives_none():
    fx = FeatureExtractor(axis=("tail", "head"))
    fx.update({"tail": (0, 0)}, None, 0, t_s=0)   # no head
    assert fx.turning_deg_s() is None
    assert fx.elongation() is None
    assert fx.heading_deg() is None
