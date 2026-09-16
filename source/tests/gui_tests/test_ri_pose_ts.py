"""How old the pose on a row is, stated outright.

The file used to carry ``pose_fw_ms``, the board time at the capture of the
frame the pose was inferred on, and a reader worked out the pose's age by
subtracting it from ``frame_fw_ms``. That is now one column, ``pose_lag_ms``:
milliseconds from the frame being captured to its pose existing, measured by
the pose sink rather than reconstructed by the reader.

``filter_ms`` sits next to it and says how much of that the Kalman and
optical-flow smoothing took, so a frame that ran late says which stage spent
the time. Both are ``na`` on a row that carries no pose.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime

from source.video.recording.frame_log import open_with_headers


def _rows(path):
    txt = open(path, encoding="utf-8").read().splitlines()
    cols = next(l for l in txt if l.startswith("#columns")).split(None, 1)[1].split()
    data = [l.split("\t") for l in txt if l and not l.startswith("#") and "\t" in l]
    return cols, data


def _session():
    td = tempfile.mkdtemp()
    path = os.path.join(td, "T_video_data.txt")
    log = open_with_headers(path, subject_id="T", setup_id=1,
                            start_dt=datetime.now(), task_name="t",
                            task_hash=1, hd_name="h", hd_hash=2,
                            tracking_mode="dlc", resolution=(640, 480), fps=30)
    # A frame captured at board time 1500, whose pose existed 41.7 ms after
    # the capture, 0.42 ms of which was the smoother.
    log.write_frame(frame_number=1, timestamp_ms=500, location="z",
                    pose_array=[[1.0, 2.0, 0.9]], frame_fw_ms=1500,
                    pose_lag_ms=41.7, filter_ms=0.42)
    log.write_frame(frame_number=2, timestamp_ms=550, frame_fw_ms=1550)
    log.close()
    return path


def test_the_two_latencies_are_written_next_to_the_board_time():
    cols, _data = _rows(_session())
    assert cols.index("pose_lag_ms") == cols.index("frame_fw_ms") + 1
    assert cols.index("filter_ms") == cols.index("pose_lag_ms") + 1
    assert "pose_fw_ms" not in cols, (
        "the pose's age is stated as pose_lag_ms now, not left as a subtraction")
    assert "host_mono_ns" not in cols


def test_the_age_is_the_value_the_sink_measured():
    cols, data = _rows(_session())
    row = data[0]
    assert row[cols.index("pose_lag_ms")] == "41.7"
    assert row[cols.index("filter_ms")] == "0.42"


def test_a_row_with_no_pose_has_no_latencies():
    """Both describe how a pose was arrived at, so a row without one has
    nothing to say about them."""
    cols, data = _rows(_session())
    row = data[1]
    assert row[cols.index("pose")] == "na"
    assert row[cols.index("pose_lag_ms")] == "na"
    assert row[cols.index("filter_ms")] == "na"


def test_the_board_time_is_still_that_frames_own():
    cols, data = _rows(_session())
    assert data[0][cols.index("frame_fw_ms")] == "1500"
    assert data[1][cols.index("frame_fw_ms")] == "1550"
