"""The _video_data.txt timing columns must never be poisoned by an unstamped
frame.

``frame_fw_ms`` and ``pose_fw_ms`` are MCU framework times mapped from host
capture instants, and the pose age an analysis reads is their difference. The
mapping is an affine offset from the last-message anchor:

    fw_ms = board.timestamp + 1000 * (host_s - last_message_mono)

so mapping a host instant of 0 does not yield 0, it yields
``timestamp - 1000 * last_message_mono``, a huge negative number on any
machine that has been up for a while. One such row makes the staleness column
meaningless, and it is silent: the file still parses.

The pose side already guards this. These tests hold both sides to it.
"""
from __future__ import annotations

import threading

from source.video.framebus.recorder_sink import RecorderSink


class _Board:
    """A pycboard whose clock is anchored, with a realistic monotonic base."""

    # ~7 days of uptime, the scale that makes an unguarded map obviously wrong
    _last_message_mono = 604_800.0
    timestamp = 12_345          # MCU framework ms since run start
    _fw_anchored = True

    def fw_ms_at(self, host_ns):
        from source.communication.pycboard import Pycboard
        return Pycboard.fw_ms_at(self, host_ns)

    def _fw_ms_for_mono(self, mono_s):
        from source.communication.pycboard import Pycboard
        return Pycboard._fw_ms_for_mono(self, mono_s)


def _sink_with_board():
    s = RecorderSink.__new__(RecorderSink)
    s._lock = threading.RLock()
    s._pycboards = {1: _Board()}
    s._last_pose = {}
    return s


# ── the hazard, stated directly ──────────────────────────────────────────

def test_mapping_an_unstamped_instant_is_wildly_wrong():
    """Not a style point: the value is ~600 million ms out."""
    board = _Board()
    assert board.fw_ms_at(0) < -600_000_000
    # A properly stamped instant maps sanely.
    assert abs(board.fw_ms_at(int(604_800.0 * 1e9)) - board.timestamp) < 5


# ── pose side (already guarded, pin it) ─────────────────────────────────

def test_pose_fw_ms_is_none_for_an_unstamped_frame():
    s = _sink_with_board()
    s.on_pose_result(1, cam_frame_id=1, pose_array=[], location=None,
                     speed=0.0, zones_by_body_part={}, raw_pose_dict={},
                     capture_host_ns=0)
    assert s._last_pose[1][4] is None


def test_pose_fw_ms_is_mapped_for_a_stamped_frame():
    s = _sink_with_board()
    s.on_pose_result(1, cam_frame_id=1, pose_array=[], location=None,
                     speed=0.0, zones_by_body_part={}, raw_pose_dict={},
                     capture_host_ns=int(604_800.0 * 1e9))
    assert abs(s._last_pose[1][4] - 12_345) < 5


# ── frame side (the gap) ─────────────────────────────────────────────────

def test_frame_fw_ms_is_none_for_an_unstamped_frame():
    """The same guard the pose side has. Without it one unstamped frame
    writes a ~-600,000,000 ms value into the column and the staleness
    difference becomes nonsense for that row."""
    s = _sink_with_board()
    assert s._frame_fw_ms(1, 0) is None


def test_frame_fw_ms_is_mapped_for_a_stamped_frame():
    s = _sink_with_board()
    assert abs(s._frame_fw_ms(1, int(604_800.0 * 1e9)) - 12_345) < 5


def test_frame_fw_ms_is_none_without_a_board():
    s = _sink_with_board()
    assert s._frame_fw_ms(99, int(604_800.0 * 1e9)) is None


def test_frame_fw_ms_is_none_before_the_clock_is_anchored():
    """Until an MCU message has been seen there is no anchor to map through."""
    s = _sink_with_board()
    s._pycboards[1]._fw_anchored = False
    assert s._frame_fw_ms(1, int(604_800.0 * 1e9)) is None


# ── the two columns agree ────────────────────────────────────────────────

def test_staleness_is_a_small_positive_number_for_normal_frames():
    """frame_fw_ms - pose_fw_ms is the pose age; a pose inferred 40 ms before
    the current frame must read as ~40 ms, not as a huge value."""
    s = _sink_with_board()
    base = int(604_800.0 * 1e9)
    s.on_pose_result(1, cam_frame_id=1, pose_array=[], location=None,
                     speed=0.0, zones_by_body_part={}, raw_pose_dict={},
                     capture_host_ns=base)
    pose_fw = s._last_pose[1][4]
    frame_fw = s._frame_fw_ms(1, base + 40_000_000)      # 40 ms later
    assert 35 <= (frame_fw - pose_fw) <= 45


def test_blob_tracker_side_uses_the_same_guard():
    """Blob tracking writes the same column through the same mapper, so an
    unstamped result must not poison it either."""
    s = _sink_with_board()
    s.on_tracker_result(1, cam_frame_id=1, centroid=(0.0, 0.0),
                        location="left", speed=1.0, capture_host_ns=0)
    assert s._last_pose[1][4] is None

    s.on_tracker_result(1, cam_frame_id=2, centroid=(0.0, 0.0),
                        location="left", speed=1.0,
                        capture_host_ns=int(604_800.0 * 1e9))
    assert abs(s._last_pose[1][4] - 12_345) < 5


def test_a_board_that_raises_does_not_break_the_row():
    """A clock that errors mid-session yields ``na``, not an exception that
    would drop the whole _video_data row."""
    s = _sink_with_board()

    def _boom(_ns):
        raise RuntimeError("anchor gone")
    s._pycboards[1].fw_ms_at = _boom
    assert s._frame_fw_ms(1, int(604_800.0 * 1e9)) is None
