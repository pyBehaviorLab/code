"""TrackingEnhancer constant-acceleration model + confidence-scaled noise.

Covers the dlc-live-parity upgrade: the forward predictor accounts for
acceleration (not just velocity), and measurement noise scales with detection
confidence.
"""
import numpy as np

from source.video.tracking.smoothing import (
    TrackingEnhancer, _CONF_R_MAX_SCALE,
)


def _feed(enh, positions, dt_s=1.0 / 30.0):
    """Feed (x, y) positions at a fixed dt; return the last result tuple."""
    ts = 0
    out = None
    for (x, y) in positions:
        ts += int(dt_s * 1e9)
        out = enh.update(x, y, 1.0, None, ts, detected=True)
    return out


def test_state_vector_is_six_dim():
    enh = TrackingEnhancer(setup_id=1, use_optical_flow=False)
    _feed(enh, [(0, 0), (1, 0), (2, 0)])
    assert enh._kf.statePost.flatten().shape[0] == 6


def test_forecast_accounts_for_acceleration():
    """Under accelerating motion the CA forecast reaches further than a pure
    constant-velocity (v·dt) extrapolation would."""
    enh = TrackingEnhancer(setup_id=1, use_optical_flow=False)
    # x accelerates: 0, 1, 3, 6, 10 (Δ grows each step) → positive ax.
    _feed(enh, [(0, 0), (1, 0), (3, 0), (6, 0), (10, 0)])
    state = enh._kf.statePost.flatten()
    x, vx, ax = state[0], state[2], state[4]
    assert ax > 0.0  # acceleration was estimated
    horizon_ms = 60.0
    dt_s = horizon_ms / 1000.0
    fx, _fy = enh.predict_ahead(horizon_ms)
    cv_only = x + vx * dt_s               # constant-velocity extrapolation
    assert fx > cv_only                    # CA term pushes it further ahead


def test_predict_ahead_zero_horizon_is_current():
    enh = TrackingEnhancer(setup_id=1, use_optical_flow=False)
    _feed(enh, [(5, 7), (5, 7)])
    fx, fy = enh.predict_ahead(0.0)
    assert (round(fx), round(fy)) == (5, 7)


def test_scaled_R_inverse_with_confidence():
    enh = TrackingEnhancer(setup_id=1, use_optical_flow=False)
    base = enh._base_R
    full = enh._scaled_R(1.0)
    low = enh._scaled_R(0.25)
    floored = enh._scaled_R(0.0)          # clamped, not divide-by-zero
    assert np.allclose(full, base)
    assert low[0, 0] > base[0, 0]         # low confidence → more noise
    assert np.allclose(floored, base * _CONF_R_MAX_SCALE)  # clamped at max


def test_nonfinite_confidence_does_not_corrupt_state():
    """A NaN/inf confidence must be treated as full confidence, never allowed
    to write a NaN measurement noise that permanently poisons the filter."""
    enh = TrackingEnhancer(setup_id=1, use_optical_flow=False)
    # Verify _scaled_R stays finite for pathological confidences.
    for bad in (float("nan"), float("inf"), float("-inf")):
        R = enh._scaled_R(bad)
        assert np.isfinite(R).all()
    # Drive a few frames with NaN confidence; state must stay finite.
    ts = 0
    dt = int((1 / 30.0) * 1e9)
    for x in (10.0, 12.0, 14.0):
        ts += dt
        enh.update(x, 20.0, float("nan"), None, ts, detected=True)
    assert np.isfinite(enh._kf.statePost).all()
    fx, fy = enh.predict_ahead(50.0)
    assert np.isfinite(fx) and np.isfinite(fy)


def test_low_confidence_detection_moves_state_less():
    """A jump seen at low confidence pulls the estimate less than the same
    jump seen at full confidence."""
    def settle_then_jump(conf):
        enh = TrackingEnhancer(setup_id=1, use_optical_flow=False)
        ts = 0
        dt = int((1 / 30.0) * 1e9)
        for _ in range(10):
            ts += dt
            enh.update(100.0, 100.0, 1.0, None, ts, detected=True)
        ts += dt
        enh.update(160.0, 100.0, conf, None, ts, detected=True)
        return enh._kf.statePost.flatten()[0]

    x_high = settle_then_jump(1.0)
    x_low = settle_then_jump(0.15)
    assert x_high > x_low                  # full-confidence tracks the jump more
