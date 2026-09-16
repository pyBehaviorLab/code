"""Smoke tests for the shared compute_speed helper."""
from source.video.tracking.speed import compute_speed


def test_returns_zero_when_no_prev():
    assert compute_speed(None, 10.0, 20.0, 1_000_000_000) == 0.0


def test_returns_zero_when_dt_zero():
    prev = (10.0, 20.0, 1_000_000_000)
    assert compute_speed(prev, 11.0, 22.0, 1_000_000_000) == 0.0


def test_returns_zero_when_dt_negative():
    prev = (10.0, 20.0, 2_000_000_000)
    # now < prev (clock went backward)
    assert compute_speed(prev, 11.0, 22.0, 1_000_000_000) == 0.0


def test_simple_horizontal_speed():
    # 100 px traveled in 1 second
    prev = (0.0, 0.0, 0)
    speed = compute_speed(prev, 100.0, 0.0, 1_000_000_000)
    assert abs(speed - 100.0) < 1e-9


def test_diagonal_uses_euclidean():
    # 3-4-5 triangle in 1 second → 5 px/s
    prev = (0.0, 0.0, 0)
    speed = compute_speed(prev, 3.0, 4.0, 1_000_000_000)
    assert abs(speed - 5.0) < 1e-9


def test_subsecond_dt_scales():
    # 50 px in 0.5 s = 100 px/s
    prev = (0.0, 0.0, 0)
    speed = compute_speed(prev, 50.0, 0.0, 500_000_000)
    assert abs(speed - 100.0) < 1e-9
