"""The analyser reads v3's ``capture_host_ns``, written by the real writer.

The parser resolves columns by name from the ``#columns`` header, so inserting
a column should move nothing. These tests hold it to that, using FrameLog to
write the file rather than a hand-built fixture: a fixture would keep passing
after the writer changed, which is the failure this guards against.
"""
from __future__ import annotations

from source.video.recording.frame_log import FrameLog
from tools.offline_analysis.video_data_parser import parse_video_data


def _write(tmp_path, rows):
    p = tmp_path / "s_video_data.txt"
    log = FrameLog(str(p))
    for n, (ns, fw) in enumerate(rows, start=1):
        log.write_frame(frame_number=n, timestamp_ms=n * 33,
                        capture_host_ns=ns, frame_fw_ms=fw)
    log.close()
    return p


def test_the_raw_instant_survives_the_round_trip(tmp_path):
    base = 1_234_567_890_123_456
    rows = [(base + i * 33_333_100, 1000.0 + i * 33) for i in range(5)]
    sess = parse_video_data(str(_write(tmp_path, rows)))
    got = sess.frames_df["capture_host_ns"].tolist()
    assert got == [r[0] for r in rows], "the nanoseconds did not come back"


def test_it_keeps_nanosecond_resolution(tmp_path):
    """Two frames 2.1 ms apart, the spacing a paired UVC delivery shows.

    ``elapsed`` is written to a millisecond, so from it the gap reads as 2 or
    3 ms. This column has to give 2.1.
    """
    base = 900_000_000_000
    sess = parse_video_data(str(_write(
        tmp_path, [(base, 1000.0), (base + 2_100_000, 1002.0)])))
    ns = sess.frames_df["capture_host_ns"].tolist()
    assert ns[1] - ns[0] == 2_100_000


def test_the_older_spelling_still_lands_in_the_same_field(tmp_path):
    """Files written before v3 named it ``host_mono_ns``.

    A caller should not have to know which era its file came from, so both
    spellings resolve to the same field.
    """
    p = tmp_path / "old_video_data.txt"
    p.write_text(
        "#version           2\n"
        "#session_start     2026-01-01 00:00:00.000\n"
        "#columns           frame elapsed host_mono_ns frame_fw_ms pose zone state\n"
        "# ====\n"
        "1\t00:00.033\t900000000000\t16\tna\tna\t-\n"
        "2\t00:00.066\t900033333100\t49\tna\tna\t-\n",
        encoding="utf-8")
    sess = parse_video_data(str(p))
    assert sess.frames_df["capture_host_ns"].tolist() == [900000000000,
                                                          900033333100]


def test_a_file_without_it_reads_without_error(tmp_path):
    """Every v1 and v2 recording already on disk has no such column."""
    p = tmp_path / "v2_video_data.txt"
    p.write_text(
        "#version           2\n"
        "#session_start     2026-01-01 00:00:00.000\n"
        "#columns           frame elapsed frame_fw_ms pose zone state\n"
        "# ====\n"
        "1\t00:00.033\t16\tna\tna\t-\n",
        encoding="utf-8")
    sess = parse_video_data(str(p))
    assert len(sess.frames_df) == 1
    assert sess.frames_df["capture_host_ns"].isna().all()


def test_inserting_the_column_did_not_move_the_others(tmp_path):
    """The point of resolving by name: frame_fw_ms is still frame_fw_ms.

    Read positionally, v3's third field is the host instant where v2's was the
    board time, and a reader that assumed position would now be off by one.
    """
    base = 700_000_000_000
    sess = parse_video_data(str(_write(
        tmp_path, [(base, 1000.0), (base + 33_000_000, 1033.0)])))
    df = sess.frames_df
    assert df["mcu_fw_capture_ms"].tolist() == [1000.0, 1033.0]
    assert df["frame_num"].tolist() == [1, 2]
