"""``pycboard.fw_ms_at``, map a frame's host capture instant to MCU fw ms.

``get_timestamp`` evaluated at a specified ``time.monotonic_ns()`` instant
(the frame's ``capture_host_ns``) instead of "now", so each video frame
carries the MCU time of its own capture. Returns ``None`` until the run's
first MCU message anchors the clock (callers stamp ``na``).
"""
from __future__ import annotations

from source.communication.pycboard import Pycboard


def _bare_board(timestamp_ms, anchor_mono_s, anchored):
    """A Pycboard with just the clock fields set, no serial port."""
    b = object.__new__(Pycboard)
    b.timestamp = timestamp_ms
    b._last_message_mono = anchor_mono_s
    b._fw_anchored = anchored
    return b


def test_fw_ms_at_returns_none_until_anchored():
    b = _bare_board(timestamp_ms=0, anchor_mono_s=100.0, anchored=False)
    # No MCU message yet this run → no valid mapping.
    assert b.fw_ms_at(int(100.5 * 1e9)) is None


def test_fw_ms_at_maps_capture_instant_to_fw_ms():
    # Anchor: MCU was at 5000 ms when host monotonic read 100.0 s.
    b = _bare_board(timestamp_ms=5000, anchor_mono_s=100.0, anchored=True)
    # A frame captured 250 ms after the anchor → fw 5250 ms.
    host_ns = int((100.0 + 0.250) * 1e9)
    assert b.fw_ms_at(host_ns) == 5250
    # A frame captured before the anchor (a pose's older frame) maps back.
    host_ns_earlier = int((100.0 - 0.080) * 1e9)
    assert b.fw_ms_at(host_ns_earlier) == 4920


def test_fw_ms_at_is_monotonic_in_capture_instant():
    b = _bare_board(timestamp_ms=1000, anchor_mono_s=50.0, anchored=True)
    # Affine mapping: increasing capture instants → non-decreasing fw ms.
    prev = None
    for k in range(0, 500, 7):
        fw = b.fw_ms_at(int((50.0 + k / 1000.0) * 1e9))
        if prev is not None:
            assert fw >= prev
        prev = fw


def test_get_timestamp_shares_the_mapping():
    # get_timestamp() is fw_ms_at(now) without the anchored gate, one
    # shared formula, so they can't drift apart.
    b = _bare_board(timestamp_ms=2000, anchor_mono_s=10.0, anchored=False)
    assert b._fw_ms_for_mono(10.0) == 2000
    assert b._fw_ms_for_mono(10.3) == 2300
