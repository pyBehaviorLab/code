"""Tests for the v2 ``_video_data.txt`` row format (one row per camera
frame, with state/event names folded into ``state`` / ``events``
columns).

Covers:
  * Header carries ``#version 2``, ``#units``, and the 8-col ``#columns``.
  * Each ``write_frame`` writes ONE row (no duplicates per frame_num).
  * ``note_state`` / ``note_event`` between two ``write_frame`` calls
    fold into the NEXT frame's ``state`` / ``events`` columns,
    pipe-joined when multiple landed in the same interval.
  * ``zone_changed`` event names are suppressed at note_event time.
  * Speed conversion: m/s when ``px_per_m`` is set, px/s otherwise.
  * Hashes rendered as 8-char zero-padded hex (cross-ref with TSV +
    snapshot store filenames).
  * ``resolve_px_per_m`` helper picks the scale zone correctly.
  * Offline parser reads the v2 file back and surfaces both the frame
    DataFrame AND an events list reconstructed from the folded columns.
"""
from __future__ import annotations

import os
import tempfile

from source.video.recording.frame_log import (
    open_with_headers,
    resolve_px_per_m,
)


SCALE_ZONE = {
    "name": "scale",
    "type": "scale",
    "points": [[0.0, 0.5], [0.5, 0.5]],
    "coord_space": "normalized",
    "scale_length_mm": 250.0,  # 0.25 m
}
RECT_ZONE = {"name": "R1", "type": "rectangle"}


# ─── resolve_px_per_m ────────────────────────────────────────────────


def test_resolve_px_per_m_with_scale_zone():
    # 0.5 of 1000 px = 500 px ; 250 mm → 2 px/mm → 2000 px/m
    assert resolve_px_per_m([SCALE_ZONE, RECT_ZONE], (1000, 1000)) == 2000.0


def test_resolve_px_per_m_no_scale_returns_none():
    assert resolve_px_per_m([RECT_ZONE], (640, 480)) is None


def test_resolve_px_per_m_invalid_resolution_returns_none():
    assert resolve_px_per_m([SCALE_ZONE], (0, 0)) is None
    assert resolve_px_per_m([SCALE_ZONE], None) is None


def test_resolve_px_per_m_empty_zones_returns_none():
    assert resolve_px_per_m([], (640, 480)) is None
    assert resolve_px_per_m(None, (640, 480)) is None


# ─── session writes ─────────────────────────────────────────────────


def _write_session(tmp_dir, *, px_per_m=None):
    """Mimic a real session: 5 frames with state/event interleavings
    that fold into the appropriate frames' state/events columns.

    Timeline:
      F1 (capture), no MCU msgs yet
      F2 (capture), no MCU msgs
      ↓ MCU: poke_3, choice_state, poke_4, poke2_reward,
              inter_trial_interval (all within ~5 ms, pyControl
              cascading transitions)
      F3 (capture), flushes F2 with all the MCU msgs in F2's
                            interval; F3 starts pending
      ↓ MCU: zone_changed (suppressed), reward_pulse
      F4 (capture), flushes F3 with reward_pulse only
                            (zone_changed dropped); F4 starts pending
      F5 (capture), flushes F4 with no msgs
      close, flushes F5 with no msgs
    """
    path = os.path.join(tmp_dir, "_video_data.txt")
    log = open_with_headers(
        path,
        subject_id="M01", setup_id=2,
        task_name="Demo/Stage1",
        task_hash=564403420, hd_name="rig.py", hd_hash=987654321,
        tracking_mode="dlc",
        resolution=(640, 480), fps=30.0,
        zones=[SCALE_ZONE, RECT_ZONE],
        px_per_m=px_per_m,
    )
    log.write_frame(frame_number=1, timestamp_ms=33, speed=400.0)
    log.write_frame(frame_number=2, timestamp_ms=66, speed=400.0)
    # MCU cascade, all land in F2's interval (between F2 and F3)
    log.note_event(name="poke_3")
    log.note_state(name="choice_state")
    log.note_event(name="poke_4")
    log.note_state(name="poke2_reward")
    log.note_state(name="inter_trial_interval")
    log.write_frame(frame_number=3, timestamp_ms=99, speed=300.0)
    # zone_changed must be suppressed, reward_pulse must land in F3's interval
    log.note_event(name="zone_changed")
    log.note_event(name="reward_pulse")
    log.write_frame(frame_number=4, timestamp_ms=132, speed=200.0)
    log.write_frame(frame_number=5, timestamp_ms=165, speed=100.0)
    log.close()
    return path


def _data_rows(body):
    """Return non-comment, non-zone-block, tab-delimited rows."""
    rows = []
    in_zone = False
    for ln in body.splitlines():
        if ln.startswith("#zone_config_begin"):
            in_zone = True; continue
        if ln.startswith("#zone_config_end"):
            in_zone = False; continue
        if in_zone or ln.startswith("#") or "\t" not in ln:
            continue
        rows.append(ln)
    return rows


# ─── header ─────────────────────────────────────────────────────────


def test_header_v2_columns_and_lineage():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        path = _write_session(td, px_per_m=2000.0)
        body = open(path, encoding="utf-8").read()
    assert "#version           3" in body
    assert ('#columns           frame elapsed capture_host_ns frame_fw_ms '
            'pose_lag_ms filter_ms pose zone state') in body
    # task_hash=564403420 → 0x21a41cdc ; hd_hash=987654321 → 0x3ade68b1
    assert '"task_hash":"21a41cdc"' in body
    assert '"hd_name":"rig.py"' in body
    assert '"hd_hash":"3ade68b1"' in body


def test_header_units_m_per_s_when_calibrated():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        path = _write_session(td, px_per_m=2000.0)
        body = open(path, encoding="utf-8").read()
    assert '#units             {"speed":"m/s"}' in body


def test_header_units_px_per_s_when_not_calibrated():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        path = _write_session(td, px_per_m=None)
        body = open(path, encoding="utf-8").read()
    assert '#units             {"speed":"px/s"}' in body


# ─── row layout, one row per frame ────────────────────────────────


def test_one_row_per_frame_no_duplicates():
    """The v2 format writes one row per cam_frame_id, no duplicate
    frame numbers."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        path = _write_session(td, px_per_m=2000.0)
        body = open(path, encoding="utf-8").read()
    rows = _data_rows(body)
    frames = [r.split("\t")[0] for r in rows]
    # 5 frames written → 5 rows, no repeats
    assert frames == ["1", "2", "3", "4", "5"], frames


def test_a_row_has_one_field_per_declared_column():
    # Counted against the header rather than against a number written here:
    # a column declared and never written is invisible to a fixed count, and
    # that is exactly how ``filled_parts`` shipped declared but empty.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        path = _write_session(td, px_per_m=2000.0)
        body = open(path, encoding="utf-8").read()
    assert "capture_host_ns" in body, (
        "the #columns header must declare capture_host_ns")
    assert "frame_fw_ms" in body, "the #columns header must declare frame_fw_ms"
    assert "pose_lag_ms" in body, "the #columns header must declare pose_lag_ms"
    assert "filter_ms" in body, "the #columns header must declare filter_ms"
    # Against the COLUMNS line, not the whole file: ``#units`` still states
    # the pixels-per-metre calibration, which is a session fact the analyser
    # needs to work out speed for itself now that the column is gone.
    declared = next(ln for ln in body.splitlines()
                    if ln.startswith("#columns"))
    for gone in ("pose_fw_ms", "host_mono_ns", "speed", "events", "track",
                 "part_zones", "filled_parts"):
        assert gone not in declared, (
            f"{gone} was retired and the header still declares it")
    declared = next(ln for ln in body.splitlines()
                    if ln.startswith("#columns")).split(None, 1)[1].split()
    rows = _data_rows(body)
    for row in rows:
        fields = row.split("\t")
        assert len(fields) == len(declared), (
            f"header declares {len(declared)} columns {declared}, "
            f"row has {len(fields)}: {fields}")



def _column(body, name):
    """Index of a column, by name. Positions move; names do not."""
    header = next(ln for ln in body.splitlines() if ln.startswith("#columns"))
    return header.split(None, 1)[1].split().index(name)


def _by_frame(body):
    return {r.split("\t")[0]: r.split("\t") for r in _data_rows(body)}


def test_states_are_folded_into_the_owning_frame_row():
    """State names that landed between F2 (timestamp_ms=66) and F3
    (timestamp_ms=99) belong to F2's row, pipe-joined.

    Events are no longer a column here: the MCU TSV records them against the
    board's own clock, which is a better account of them than a bucket by
    frame interval.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        path = _write_session(td, px_per_m=2000.0)
        body = open(path, encoding="utf-8").read()
    at = _column(body, "state")
    assert _by_frame(body)["2"][at] == (
        "choice_state|poke2_reward|inter_trial_interval")


def test_a_frame_with_no_state_change_uses_a_dash():
    """F1, F4 and F5 had no state transition in their intervals."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        path = _write_session(td, px_per_m=2000.0)
        body = open(path, encoding="utf-8").read()
    at = _column(body, "state")
    by_frame = _by_frame(body)
    for fid in ("1", "4", "5"):
        assert by_frame[fid][at] == "-", f'F{fid} state should be "-"'


# ─── speed unit conversion ─────────────────────────────────────────




def test_state_event_before_first_frame_absorbed_into_F1():
    """Edge: a state noted BEFORE any ``write_frame`` must not create a
    phantom frame=0 row; it belongs to the first real frame."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        path = os.path.join(td, "_video_data.txt")
        log = open_with_headers(path, subject_id="X", setup_id=1,
                                task_name="T", resolution=(640, 480),
                                fps=30.0, px_per_m=None)
        log.note_event(name="setup_evt")
        log.note_state(name="initial")
        log.write_frame(frame_number=1, timestamp_ms=33)
        log.write_frame(frame_number=2, timestamp_ms=66)
        log.close()
        body = open(path, encoding="utf-8").read()
    rows = _data_rows(body)
    frames = [r.split("\t")[0] for r in rows]
    # NO phantom frame=0 row, only 1 and 2.
    assert frames == ["1", "2"], frames
    assert _by_frame(body)["1"][_column(body, "state")] == "initial"



def test_close_with_pending_state_event_flushes_final_row():
    """Edge: a state noted after the last ``write_frame``, then close. The
    final pending row must be flushed rather than dropped."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        path = os.path.join(td, "_video_data.txt")
        log = open_with_headers(path, subject_id="X", setup_id=1,
                                task_name="T", resolution=(640, 480),
                                fps=30.0, px_per_m=None)
        log.write_frame(frame_number=1, timestamp_ms=33)
        log.note_state(name="final_state")
        log.note_event(name="final_event")
        log.close()
        body = open(path, encoding="utf-8").read()
    rows = _data_rows(body)
    assert len(rows) == 1
    assert rows[0].split("\t")[_column(body, "state")] == "final_state"


def test_close_empty_session_writes_no_rows():
    """Edge: open, close, no frames or events, file has header
    only, no data rows."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        path = os.path.join(td, "_video_data.txt")
        log = open_with_headers(path, subject_id="X", setup_id=1,
                                task_name="T", resolution=(640, 480),
                                fps=30.0, px_per_m=None)
        log.close()
        body = open(path, encoding="utf-8").read()
    assert _data_rows(body) == []


def test_the_offline_parser_reads_a_file_this_writer_wrote():
    """One frame per data row, and the states rebuilt from the folded column.

    Only states: the events column is gone from this file, and the parser has
    to read what the writer writes rather than what it used to.
    """
    from tools.offline_analysis.video_data_parser import parse_video_data
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        path = _write_session(td, px_per_m=2000.0)
        sess = parse_video_data(path)
    assert sess.n_frames == 5
    # The cascade on F2: three state transitions, no events.
    assert [e["kind"] for e in sess.events] == ["S", "S", "S"]
    assert sorted(e["name"] for e in sess.events) == [
        "choice_state", "inter_trial_interval", "poke2_reward",
    ]
    assert all(e["frame"] == 2 for e in sess.events)
    assert sess.info.get("units") == {"speed": "m/s"}
    assert sess.info.get("version") == "3"


# ── integrity: the columns a reader aligns on ────────────────────────

def test_frame_fw_ms_never_goes_backwards(tmp_path):
    """The MCU clock column must be monotonic.

    A real session (2026-08-16) stepped backwards 7 times, worst -72 ms,
    because the host↔MCU anchor is re-taken on every board message and a
    later frame can map through an earlier one. A reader aligning video to
    MCU time cannot recover from that, so the column is clamped forward.
    """
    from source.video.recording.frame_log import FrameLog

    log = FrameLog(str(tmp_path / "s_video_data.txt"))
    for i, fw in enumerate([1000.0, 1050.0, 978.0, 1100.0, 1090.0, 1200.0], start=1):
        log.write_frame(frame_number=i, timestamp_ms=i * 50, frame_fw_ms=fw)
    log.close()

    # Data rows start with the frame number; the header carries an
    # un-commented JSON zone block, so "not a #" is not enough to find them.
    rows = [ln.split("\t") for ln in
            (tmp_path / "s_video_data.txt").read_text().splitlines()
            if ln[:1].isdigit()]
    got = [float(r[2]) for r in rows if r[2] not in ("na", "")]
    assert got == sorted(got), f"frame_fw_ms went backwards: {got}"
    assert log._fw_clamps == 2, "both backward steps should be counted"


def test_the_header_records_the_geometry_that_was_written(tmp_path):
    """``resolution`` is what readers scale normalised coordinates by.

    So it must come from the frame that was actually written, not from
    config: the two disagree in practice, 853x830 against 720x700, and
    315x176 against 360x202, in real sessions.
    """
    from source.video.recording.frame_log import FrameLog

    log = FrameLog(str(tmp_path / "g_video_data.txt"))
    log.write_info("resolution", [853, 830])      # what config believed
    log.write_info("resolution", [720, 700])      # what the first frame was
    log.write_frame(frame_number=1, timestamp_ms=0)
    log.close()

    header = [ln for ln in (tmp_path / "g_video_data.txt").read_text().splitlines()
              if ln.startswith("#")]
    res_lines = [ln for ln in header if "resolution" in ln or "camera" in ln]
    assert any("720" in ln and "700" in ln for ln in res_lines), res_lines
    assert not any("853" in ln for ln in res_lines), (
        "the stale config geometry reached the header")
