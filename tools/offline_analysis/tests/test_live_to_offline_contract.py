"""What the rig records, the analyser must be able to read.

These are the only tests that drive BOTH halves of the pipeline: the live
recorder writes the file and the offline analyser reads it, with nothing
in between standing in for either. Everything else in the suite tests one
side against a fixture, and a fixture is written by whoever wrote the test,
which is how the two halves came to disagree without a single test failing.

They disagreed on the pose column. The recorder writes it positionally,
``[[x,y,c], …]``, with the names declared once in the header; the analyser
understood only the named form and returned ``None`` for anything else. A
session recorded with pose therefore read back as a session with no pose,
silently, and every measure over it came out empty. No recording had yet been
made with online pose enabled, so nothing had caught it.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.offline_analysis import video_data_schema as vds
from tools.offline_analysis.engine import offline_analysis as oa

# The analyser is meant to be liftable out of this repository on its own, so
# these, the only tests that need the rig, skip rather than fail when it is
# not there. See `test_no_live_rig_imports.py` for the rule and why tests are
# exempt from it.
fl = pytest.importorskip("source.video.recording.frame_log",
                         reason="the live recorder is not present, so the "
                                "recorder/reader contract cannot be checked")

PARTS = ["Snout", "Head", "center"]


def record(path, n=60, parts=PARTS, pose=True, states=True):
    """A session written by the real recorder, not by a fixture."""
    log = fl.open_with_headers(path, subject_id="282", setup_id=1,
                               bodyparts=parts, fps=20.0)
    for i in range(n):
        if states and i in (0, n // 2):
            log.note_state("init_trial" if i == 0 else "choice_state")
        log.write_frame(
            frame_number=i, timestamp_ms=i * 50, speed=1.5, location="left",
            pose_array=[[10.0 + i, 20.0, 0.9], [12.0 + i, 22.0, 0.8],
                        [11.0 + i, 21.0, 0.95]][:len(parts)] if pose else None,
            frame_fw_ms=1000 + i * 50,
            # How old this row's pose is, and the smoother's share of it.
            pose_lag_ms=8.0 if pose else None,
            filter_ms=0.5 if pose else None)
    log.close()
    return path


@pytest.fixture
def recorded(tmp_path):
    return record(str(tmp_path / "live_video_data.txt"))


# ── the column contract ──────────────────────────────────────────────────

def test_every_column_the_recorder_writes_is_readable(recorded):
    header = vds.VideoDataHeader.parse(recorded)
    rows = list(vds.iter_rows(recorded, header))
    assert rows, "the analyser read no rows from a file the recorder wrote"
    for row in rows:
        assert vds.parse_int(row.get("frame_number"), -1) >= 0
        for column in ("frame_ts_ms", "mcu_ts_ms"):
            value = vds.parse_elapsed(row.get(column))
            assert value == value, f"{column} did not survive the round trip"


def test_the_pose_survives_the_round_trip(recorded):
    """Positional in, named out."""
    header = vds.VideoDataHeader.parse(recorded)
    for row in vds.iter_rows(recorded, header):
        pose = vds.parse_pose(row.get("pose_array"))
        assert pose, "a recorded pose read back as no pose at all"
        assert sorted(pose) == sorted(PARTS)
        assert len(pose["Snout"]) >= 3


def test_the_analyser_recovers_every_keypoint(recorded):
    sess = oa.parse_txt(recorded)
    assert sorted(sess.kp) == sorted(PARTS)
    for part in PARTS:
        x = np.asarray(sess.kp[part]["x"], float)
        assert np.isfinite(x).all(), f"{part} came back with holes in it"


def test_a_recorded_session_measures(recorded):
    """The end of it: a file off the rig produces a trajectory."""
    track = oa.build_track(oa.parse_txt(recorded), dict(oa.DEFAULTS))
    assert np.isfinite(track.cx).mean() > 0.95


def test_the_task_state_survives(recorded):
    sess = oa.parse_txt(recorded)
    assert {"init_trial", "choice_state"} <= set(sess.stage_series)


def test_the_poses_age_is_readable(recorded):
    """How stale the pose on a row was, which the session states outright.

    It used to be a subtraction between two board-time columns. It is one
    number now, written by the sink that measured it, and the analyser has to
    read the column that exists rather than the one that used to.
    """
    header = vds.VideoDataHeader.parse(recorded)
    rows = list(vds.iter_rows(recorded, header))
    ages = [float(r.get("pose_lag_ms")) for r in rows if r.get("pose_lag_ms") not in
            (None, "", "na")]
    assert len(ages) == len(rows), "some rows lost their inference latency"
    assert all(abs(age - 8.0) < 1e-6 for age in ages)


def test_the_smoothers_share_is_readable(recorded):
    header = vds.VideoDataHeader.parse(recorded)
    shares = [float(r.get("filter_ms")) for r in vds.iter_rows(recorded, header)
              if r.get("filter_ms") not in (None, "", "na")]
    assert shares and all(abs(v - 0.5) < 1e-6 for v in shares)


# ── the failure modes, out loud ──────────────────────────────────────────

def test_a_positional_pose_without_names_warns(caplog):
    """It cannot be read, and that has to be said. Returning None quietly is
    what let a whole session look untracked."""
    with caplog.at_level("WARNING"):
        assert vds.parse_pose("[[1.0,2.0,0.9]]", ()) is None
    assert any("body_parts" in r.message for r in caplog.records)


def test_a_count_mismatch_warns(caplog):
    with caplog.at_level("WARNING"):
        pose = vds.parse_pose("[[1.0,2.0,0.9],[3.0,4.0,0.8]]", ["only_one"])
    assert pose == {"only_one": [1.0, 2.0, 0.9]}
    assert any("body parts" in r.message for r in caplog.records)


def test_the_named_form_still_works():
    """The re-tracker's spelling must not have been broken by teaching the
    reader the recorder's."""
    named = '{"Snout":[1.0,2.0,0.9]}'
    assert vds.parse_pose(named) == {"Snout": [1.0, 2.0, 0.9]}


def test_an_untracked_frame_is_absent_not_empty(tmp_path):
    """``na`` means the tracker was off, which is not a frame where it ran and
    found nothing. Measures treat the two differently."""
    path = record(str(tmp_path / "np_video_data.txt"), n=5, pose=False)
    header = vds.VideoDataHeader.parse(path)
    for row in vds.iter_rows(path, header):
        assert vds.parse_pose(row.get("pose_array")) is None
