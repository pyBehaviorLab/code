"""v0 _video_data support in the offline analyzer parser.

v0 files (from the v0 GUI) use ``B``/``M``/``V`` header lines + one JSON
object per row, with frame counters that reset between segments and no
MCU-fw clock. They must parse into the same SessionData shape as v1/v2
so every downstream view works unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

# Package imports, the analyzer modules use relative imports
# (``from . import legacy_v0``), so they must be loaded as a package.
from tools.offline_analysis import video_data_parser as vp  # noqa: E402
from tools.offline_analysis import legacy_v0  # noqa: E402


def _write_v0(path: Path) -> None:
    B = {"Experiment": "cKet", "Experimenter": "SK", "Subject ID": "D860",
         "Group": "", "Start time": "2025/07/30 11:34:26"}
    M = {"zones": {"Center": {"name": "Center", "type": "polygon",
                              "values": [[1, 2], [3, 4], [5, 6]], "points": None}},
         "scale": {"length": "22", "values": [10, 20]},
         "arena": {"arena": {"name": "arena", "type": "polygon",
                             "values": [[0, 0], [9, 9]]}}}
    # 3-point pose_arrays, mirroring real D860 (head/center/tail).
    rows = [
        {"frames": 1, "mcu_frame": 0, "state_name": None, "timestamps": 89,
         "speed": 0, "location": "Center",
         "pose_array": [[100.0, 200.0, 0.99], [101.0, 210.0, 0.98], [102.0, 220.0, 0.97]]},
        {"frames": 2, "mcu_frame": 0, "state_name": None, "timestamps": 120,
         "speed": 1.5, "location": "Center",
         "pose_array": [[101.0, 201.0, 0.99], [102.0, 211.0, 0.98], [103.0, 221.0, 0.97]]},
        # segment reset: frames counter drops back to 1
        {"frames": 1, "mcu_frame": 1, "state_name": "State: habituation",
         "timestamps": 50, "speed": 2.0, "location": "Arm",
         "pose_array": [[110.0, 210.0, 0.98], [111.0, 220.0, 0.97], [112.0, 230.0, 0.96]]},
        {"frames": 2, "mcu_frame": 2, "state_name": "State: habituation",
         "timestamps": 90, "speed": 3.0, "location": "Arm",
         "pose_array": [[111.0, 211.0, 0.98], [112.0, 221.0, 0.97], [113.0, 231.0, 0.96]]},
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("B " + json.dumps(B) + "\n\n")
        f.write("M " + json.dumps(M) + "\n\n")
        f.write("V [70, 510, 70, 620]\n\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_v0_parses_to_sessiondata(tmp_path):
    p = tmp_path / "D860_video_data_-2025-07-30-113426.txt"
    _write_v0(p)
    s = vp.parse_video_data(str(p))

    assert s.info.get("version") == "0"
    assert s.info["sync"]["source"] == "v0_host_ms"
    assert s.subject_id == "D860"
    assert s.info["pycontrol"]["task"] == "cKet"

    # behavioural zones: M block, values copied to points for the editor
    behavioural = [z for z in s.zones if z.get("type") != "scale"]
    assert len(behavioural) == 1
    z = behavioural[0]
    assert z["name"] == "Center"
    assert z["points"] == [[1, 2], [3, 4], [5, 6]]
    assert s.scale.get("length") == "22"

    # scale injected as a type:"scale" zone so _detect_scale resolves it.
    # values [10, 20] (flat-2) -> horizontal bar points [[10,0],[20,0]],
    # length 22 cm -> 10 px / 22 cm.
    scale_zones = [z for z in s.zones if z.get("type") == "scale"]
    assert len(scale_zones) == 1
    sz = scale_zones[0]
    assert sz["points"] == [[10.0, 0.0], [20.0, 0.0]]
    assert sz["scale_length"] == 22.0
    assert sz["scale_unit"] == "cm"

    # frames + normalized columns
    assert s.n_frames == 4
    for col in ("frame_num", "mcu_fw_capture_ms", "pose", "x", "y",
                "zone", "speed_pxs", "state", "location", "segment"):
        assert col in s.frames_df, col
    assert float(s.frames_df.iloc[0].x) == 100.0
    assert float(s.frames_df.iloc[0].y) == 200.0
    assert s.frames_df.iloc[0].location == "Center"

    # segment reset detected (frames 1,2 then 1,2 -> two segments)
    assert sorted(s.frames_df["segment"].unique().tolist()) == [0, 1]

    # state-change event derived once on transition into habituation
    state_events = [e for e in s.events if e.get("name") == "habituation"]
    assert len(state_events) == 1
    assert state_events[0]["kind"] == "S"


def test_v0_discovered_by_finder(tmp_path):
    p = tmp_path / "D860_video_data_-2025-07-30-113426.txt"
    _write_v0(p)
    found = vp.find_video_data_files(str(tmp_path))
    assert str(p) in found


def test_v1_v2_not_misrouted_as_v0(tmp_path):
    """A ``#``-header file must still take the v1/v2 path, not v0."""
    p = tmp_path / "x_video_data.txt"
    p.write_text(
        "# ===\n#version           2\n"
        "#columns           frame elapsed fw_ms pose zone speed state events\n"
        "# ===\n"
        "1\t00:00.033\t10\tna\tCenter\t0.0\t-\tpoke_left\n",
        encoding="utf-8",
    )
    assert not legacy_v0.is_v0_file(str(p))
    s = vp.parse_video_data(str(p))
    assert s.info.get("version") != "0"
    assert s.n_frames == 1


def test_v0_video_pairing(tmp_path):
    """v0 video is ``<subject>-<timestamp>.mp4`` (the ``_video_data_-`` chunk
    dropped); the txt-named or substring file must still be found."""
    txt = tmp_path / "D860_video_data_-2025-07-30-113426.txt"
    _write_v0(txt)
    # The real video drops the _video_data_- chunk.
    vid = tmp_path / "D860-2025-07-30-113426.mp4"
    vid.write_bytes(b"\x00")
    # Direct module call.
    assert legacy_v0.find_video_v0(str(txt)) == str(vid)
    # And via the shared finder, which must dispatch to v0. view_video pulls
    # heavy GUI deps, so import lazily and skip if unavailable.
    import pytest
    view_video = pytest.importorskip("tools.offline_analysis.view_video")
    assert view_video.find_video_for_txt(str(txt)) == str(vid)


def test_v0_video_pairing_timestamp_fallback(tmp_path):
    """When no exact/substring match exists, fall back to the timestamp."""
    txt = tmp_path / "D860_video_data_-2025-07-30-113426.txt"
    _write_v0(txt)
    vid = tmp_path / "recording_2025-07-30-113426_cam0.avi"
    vid.write_bytes(b"\x00")
    assert legacy_v0.find_video_v0(str(txt)) == str(vid)


def test_v0_no_video_returns_empty(tmp_path):
    txt = tmp_path / "D860_video_data_-2025-07-30-113426.txt"
    _write_v0(txt)
    assert legacy_v0.find_video_v0(str(txt)) == ""


def test_v0_view_loader_scale_stage_bodyparts(tmp_path):
    """The main analysis view's loader shape: pose-as-dict (named), a
    ``stage`` column from state_name, a scale zone, and body parts."""
    txt = tmp_path / "D860_video_data_-2025-07-30-113426.txt"
    _write_v0(txt)
    legacy_v0.set_body_parts(str(txt), ["head", "center", "tail"])
    header, zones, df, bps = legacy_v0.load_frames_for_view(str(txt))

    assert bps == ["head", "center", "tail"]
    assert "stage" in df.columns
    assert "habituation" in set(df["stage"])
    # pose converted to a name-keyed dict
    pose = df["pose"].iloc[1]
    assert isinstance(pose, dict) and set(pose) == {"head", "center", "tail"}
    # scale zone present so the view's px/cm resolver finds it
    scale = [z for z in zones if z.get("type") == "scale"]
    assert len(scale) == 1 and scale[0]["points"] == [[10.0, 0.0], [20.0, 0.0]]


def test_v0_peek_pose_len(tmp_path):
    txt = tmp_path / "D860_video_data_-2025-07-30-113426.txt"
    _write_v0(txt)
    assert legacy_v0.peek_pose_len(str(txt)) == 3
