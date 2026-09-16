"""The arena recorder's _video_data.txt reads, and reads CORRECTLY.

Written from a real file's header and rows. Each test names a way the main
reader failed on this shape, so a regression is legible rather than just red:

  * tab-separated ``info`` rows were consumed as data
  * the ``# frame_number ...`` header was filed as an info key, so the column
    names were never learned
  * ``pose_array`` is a JSON object; ``_parse_pose`` returns None for a dict,
    which emptied every centroid, head position and zone
  * ``actual_ts`` is milliseconds, and ``_elapsed_to_ms`` reads a bare number
    as seconds, so the naive route makes time bins 1000x too long

The last one is checked against real numbers rather than a shape, because it
is the failure that produces a plausible-looking but wrong analysis.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tools.offline_analysis import legacy_arena
from tools.offline_analysis.analysis_input import canonicalize_session
from tools.offline_analysis.video_data_parser import parse_video_data

HEADER = "\n".join([
    "time\ttype\tsubtype\tcontent",
    "0.000\tinfo\tvideo_file\tF609_PreInjection_270826105716.mp4",
    "0.000\tinfo\tsubject_id\tF609",
    "0.000\tinfo\tbox_id\t4",
    "0.000\tinfo\tstage\tPreInjection",
    "0.000\tinfo\tstart_time\t2026-08-27T10:57:16.509",
    "0.000\tinfo\ttarget_fps\t30",
    "0.000\tinfo\tresolution\t360x202",
    "0.000\tinfo\tcodec\tlibx264",
    '0.000\tinfo\tmetadata\t{"arena_id":4,"subject_id":"F609",'
    '"stage":"PreInjection"}',
    'Zones\t[{"name":"Outer","points":[[0.01,0.02],[0.98,0.02],[0.98,0.98],'
    '[0.01,0.98]],"type":"rectangle","enabled":true},'
    '{"name":"Center","points":[[0.17,0.23],[0.80,0.23],[0.80,0.67],'
    '[0.17,0.67]],"type":"rectangle","enabled":true}]',
    '0.000\tinfo\tbody_parts\t["Head","Center","TailBase"]',
    "# frame_number\tcam_frame_id\tactual_ts\tpose_cam_frame_id\t"
    "pose_lat_frames\tpose_age_ms\tspeed\tlocation\tpose_array\tstage",
])

POSE = ('{"Head":[600.6,238.0,0.999],"Center":[564.9,259.0,0.999],'
        '"TailBase":[524.5,265.0,0.999]}')

ROWS = [
    f"0\t134558\t0\t134558\t0\t-1\t0\tnone\t{POSE}\tPreInjection",
    "1\t134559\t31\t134557\t2\t-1\t0\tnone\tnone\tPreInjection",
    "2\t134560\t62\t134557\t3\t-1\t0\tOuter\tnone\tPreInjection",
    f"3\t134561\t93\t134557\t4\t-1\t1.5\tCenter\t{POSE}\tPreInjection",
    "4\t134562\t140\t134557\t5\t-1\t0\tnone\tnone\tPreInjection",
]


@pytest.fixture
def arena_file(tmp_path):
    p = tmp_path / "F609_PreInjection_270826105716_video_data.txt"
    p.write_text(HEADER + "\n" + "\n".join(ROWS) + "\n", encoding="utf-8")
    return str(p)


def test_it_is_recognised_by_its_header_not_its_name(arena_file, tmp_path):
    """Content decides. A file renamed to anything must still read."""
    assert legacy_arena.is_arena_file(arena_file)
    renamed = tmp_path / "totally_different_name.txt"
    renamed.write_text(open(arena_file, encoding="utf-8").read(),
                       encoding="utf-8")
    assert legacy_arena.is_arena_file(str(renamed))


def test_the_main_parser_routes_to_it(arena_file):
    s = parse_video_data(arena_file)
    assert len(s.frames_df) == len(ROWS), "info rows leaked in as data"


def test_info_rows_are_header_not_data(arena_file):
    s = parse_video_data(arena_file)
    assert s.info["subject_id"] == "F609"
    assert s.info["video_file"].endswith(".mp4")
    assert s.info["box_id"] == "4"


def test_the_column_header_is_learned(arena_file):
    """The real bug: ``# frame_number`` was stored as an info KEY."""
    s = parse_video_data(arena_file)
    assert "frame_number" not in s.info
    assert s.frames_df["frame_num"].tolist() == [0, 1, 2, 3, 4]


def test_pose_dict_becomes_an_ordered_list(arena_file):
    """A dict makes _parse_pose return None and empties every position."""
    s = parse_video_data(arena_file)
    first = json.loads(s.frames_df["pose"].iloc[0])
    assert isinstance(first, list) and len(first) == 3
    # body_parts order: Head, Center, TailBase
    assert first[0][:2] == [600.6, 238.0]
    assert first[2][:2] == [524.5, 265.0]
    assert s.frames_df["pose"].iloc[1] == "na"


def test_time_bins_are_milliseconds_not_seconds(arena_file):
    """actual_ts is ms; _elapsed_to_ms reads a bare number as SECONDS.

    Route it naively and every bin is 1000x too long, which looks like a
    plausible session rather than an error.
    """
    s = parse_video_data(arena_file)
    ai = canonicalize_session(s, head_idx=0, body_idx=1)
    t = np.asarray(ai.times_buf, dtype=float)
    assert t[0] == pytest.approx(0.0, abs=1e-6)
    assert t[1] == pytest.approx(31.0, abs=0.5)
    assert t[4] == pytest.approx(140.0, abs=0.5)
    # A 5-frame 30 fps clip is ~0.14 s, never 140 s.
    assert t[-1] - t[0] < 1000.0


def test_no_mcu_clock_is_invented(arena_file):
    """This format has no MCU time. An invented one would be indistinguishable
    from a real one downstream."""
    s = parse_video_data(arena_file)
    assert s.frames_df["mcu_fw_capture_ms"].isna().all()


def test_positions_and_zones_reach_the_analysis(arena_file):
    s = parse_video_data(arena_file)
    ai = canonicalize_session(s, head_idx=0, body_idx=1)
    assert ai.centers_buf[0] is not None, "centroid lost"
    assert ai.head_buf[0] is not None, "head position lost"
    assert ai.centers_buf[1] is None, "a 'none' pose must stay empty"
    assert any(z for z in ai.zone_buf), "no zone resolved on any frame"


def test_header_metadata_reaches_sessiondata(arena_file):
    s = parse_video_data(arena_file)
    assert s.resolution == (360, 202)
    assert s.fps == pytest.approx(30.0)
    assert len(s.zones) == 2
    assert {z["name"] for z in s.zones} == {"Outer", "Center"}


def test_speed_and_stage_survive(arena_file):
    s = parse_video_data(arena_file)
    assert s.frames_df["speed_pxs"].iloc[3] == pytest.approx(1.5)
    ai = canonicalize_session(s, head_idx=0, body_idx=1)
    assert ai.state_buf[0] == "PreInjection"


def test_current_format_is_untouched(tmp_path):
    """The main pipeline's v3 file must not be captured by the sniffer."""
    p = tmp_path / "S1-Box1-2026-09-14-145732_video_data.txt"
    p.write_text(
        "#version           3\n"
        "#session_start     2026-09-14 14:57:32.000\n"
        "#columns           frame elapsed capture_host_ns frame_fw_ms "
        "pose_lag_ms filter_ms pose zone state\n"
        "# ====\n"
        "1\t00:00.033\t35742119033\t16\t11.4\t0.4\tna\tna\t-\n",
        encoding="utf-8")
    assert not legacy_arena.is_arena_file(str(p))
    s = parse_video_data(str(p))
    assert len(s.frames_df) == 1
    assert s.frames_df["capture_host_ns"].iloc[0] == 35742119033
