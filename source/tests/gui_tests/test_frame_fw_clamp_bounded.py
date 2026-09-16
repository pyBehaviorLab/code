"""The MCU-time column must survive one bad sample.

``frame_fw_ms`` is what an offline analysis uses to line video up with the
controller's own rows, so it must never run backwards. It was therefore
clamped forward, and the clamp was unbounded.

Measured on the 2026-09-10 inference session: the FIRST frame was stamped
163119 ms, the board's uptime, while every later frame carried run time. Each
real value was smaller than the bad one, so each was clamped away, and the
column sat frozen for 163 SECONDS of a 600 second session, on all four boxes.
27% of that session has no usable MCU time at all.

The rule is a threshold, not a direction: a small backward step is anchor
jitter and the new value is wrong; a large one means the value already held is
wrong, and holding it corrupts everything behind it.
"""
from __future__ import annotations

import pytest

from source.video.recording.frame_log import FrameLog


@pytest.fixture
def log(tmp_path):
    fl = FrameLog(tmp_path / "s_video_data.txt", rebind_drop_log=False)
    try:
        yield fl
    finally:
        fl.close()


def _feed(fl, values):
    """Write a frame per value; return the column as written."""
    for i, fw in enumerate(values):
        fl.write_frame(frame_number=i + 1,
                       timestamp_ms=int(i * 1000 / 30),
                       frame_fw_ms=fw)
    fl.close()
    # Read the column's index off the header rather than hard-coding it.
    # Hard-coded, this broke the day a column was inserted before it, and the
    # failure read as a clamp bug rather than as a test reading the wrong
    # field.
    text = (fl._path).read_text(encoding="utf-8")
    header = [ln for ln in text.splitlines() if ln.startswith("#columns")]
    assert header, "the file must declare its columns"
    idx = header[0].split(None, 1)[1].split().index("frame_fw_ms")
    out = []
    for line in text.splitlines():
        if not line[:1].isdigit() or "\t" not in line:
            continue
        parts = line.split("\t")
        if len(parts) > idx and parts[idx] not in ("", "na"):
            out.append(float(parts[idx]))
    return out


def test_small_backward_steps_are_clamped(log):
    """Anchor jitter: 7 steps, worst -72 ms, on a real session."""
    got = _feed(log, [1000.0, 1033.0, 1010.0, 1100.0])
    assert got == [1000.0, 1033.0, 1033.0, 1100.0], got


def test_a_huge_backward_step_re_seats_instead_of_freezing(log):
    """One bad first sample must cost one row, not the next three minutes."""
    got = _feed(log, [163119.0, 114.0, 147.0, 180.0])
    assert got[0] == 163119.0, "the bad sample itself is written as seen"
    assert got[1:] == [114.0, 147.0, 180.0], (
        f"the column stayed stuck behind the bad sample: {got}")


def test_the_session_recovers_immediately_not_after_163_seconds(log):
    """The exact shape of the rig failure, in miniature."""
    frames = [163119.0] + [float(33 * i) for i in range(1, 200)]
    got = _feed(log, frames)
    frozen = sum(1 for a, b in zip(got, got[1:]) if a == b)
    assert frozen == 0, (
        f"{frozen} frames carry a repeated stamp; on the rig this was 4,974 "
        f"of 18,004")
    assert got[1] == 33.0 and got[-1] == frames[-1]


def test_re_seats_are_counted_and_clamps_stay_separate(log):
    _feed(log, [1000.0, 990.0, 500000.0, 1200.0])
    assert log._fw_clamps == 1, "the 10 ms step is jitter"
    assert log._fw_reseats == 1, "the 500 s step is a wrong anchor"


def test_a_forward_only_column_is_untouched(log):
    frames = [float(33 * i) for i in range(1, 50)]
    assert _feed(log, frames) == frames
    assert log._fw_clamps == 0 and log._fw_reseats == 0
