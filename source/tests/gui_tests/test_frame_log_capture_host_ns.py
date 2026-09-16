"""``capture_host_ns``, the raw instant every other clock on the row came from.

``elapsed`` rounds it to a millisecond and ``frame_fw_ms`` rounds it AND clamps
it forward to stay monotonic. Both are lossy and neither is reversible, so
without the raw value a recording cannot be checked: the frame interval cannot
be recovered below 1 ms, a clamped ``frame_fw_ms`` cannot be told from a real
one, and the host-to-MCU mapping cannot be recomputed offline.

This matters most on a fast camera. A 120 fps UVC stream through OpenCV is
delivered in PAIRS roughly 2 ms apart, so millisecond rounding loses half the
structure of the interval distribution, and the ``#timestamp_source`` header
says ``host``: the value is taken when ``read()`` returned and is not the
exposure instant.
"""
from __future__ import annotations

from source.video.recording.frame_log import FrameLog


def _rows(path):
    """Data rows only.

    Not "every line that is not a comment": the zone-config block between
    ``#zone_config_begin`` and ``#zone_config_end`` is pretty-printed JSON
    whose lines start with neither. A data row starts with the frame number
    and a tab.
    """
    return [ln.split("\t") for ln in path.read_text(encoding="utf-8").splitlines()
            if ln[:1].isdigit() and "\t" in ln]


def test_the_raw_host_instant_is_written_exactly(tmp_path):
    """Not rounded, not scaled: the integer handed in is the integer stored.

    A float would silently lose the low digits at a perf_counter epoch of a
    few days, which is the range this actually runs at.
    """
    p = tmp_path / "a_video_data.txt"
    log = FrameLog(str(p))
    big = 1_234_567_890_123_456          # ~14 days of uptime, in ns
    log.write_frame(frame_number=1, timestamp_ms=0, capture_host_ns=big)
    log.write_frame(frame_number=2, timestamp_ms=8, capture_host_ns=big + 8_333_100)
    log.close()

    rows = _rows(p)
    assert len(rows) == 2
    assert rows[0][2] == str(big)
    assert rows[1][2] == str(big + 8_333_100)


def test_it_recovers_an_interval_the_elapsed_column_cannot(tmp_path):
    """Two frames 2.1 ms apart, the spacing a paired UVC delivery shows.

    ``elapsed`` is written to 1 ms, so from it alone the gap reads as 2 ms or
    3 ms depending on where the rounding falls. The raw column gives 2.1 ms.
    """
    p = tmp_path / "b_video_data.txt"
    log = FrameLog(str(p))
    t0 = 900_000_000_000
    log.write_frame(frame_number=1, timestamp_ms=0, capture_host_ns=t0)
    log.write_frame(frame_number=2, timestamp_ms=2,
                    capture_host_ns=t0 + 2_100_000)
    log.close()

    rows = _rows(p)
    gap_ns = int(rows[1][2]) - int(rows[0][2])
    assert gap_ns == 2_100_000
    assert abs(gap_ns / 1e6 - 2.1) < 1e-9


def test_a_frame_without_one_says_na_rather_than_zero(tmp_path):
    """``0`` is a valid instant on a freshly based clock. ``na`` is not, and
    the difference decides whether a reader treats the row as usable."""
    p = tmp_path / "c_video_data.txt"
    log = FrameLog(str(p))
    log.write_frame(frame_number=1, timestamp_ms=0)
    log.close()

    assert _rows(p)[0][2] == "na"


def test_the_header_declares_it_and_the_column_order_matches(tmp_path):
    """A reader indexes on the ``#columns`` line, so the two must agree."""
    p = tmp_path / "d_video_data.txt"
    log = FrameLog(str(p))
    log.write_frame(frame_number=1, timestamp_ms=0, capture_host_ns=42)
    log.close()

    text = p.read_text(encoding="utf-8")
    cols = next(ln for ln in text.splitlines() if ln.startswith("#columns"))
    names = cols.split(None, 1)[1].split()
    assert names.index("capture_host_ns") == 2
    assert _rows(p)[0][2] == "42"
    assert len(_rows(p)[0]) == len(names), (
        "a row must have exactly as many fields as the header declares")


def test_the_clamp_on_frame_fw_ms_stays_visible_in_the_raw_column(tmp_path):
    """The point of keeping it.

    ``frame_fw_ms`` is held forward when the host-to-MCU anchor jitters
    backwards, so two rows can carry the SAME board time. The raw host
    instants still differ, which is how a reader tells a clamped row from a
    genuinely simultaneous one.
    """
    p = tmp_path / "e_video_data.txt"
    log = FrameLog(str(p))
    t0 = 700_000_000_000
    log.write_frame(frame_number=1, timestamp_ms=0, capture_host_ns=t0,
                    frame_fw_ms=1000.0)
    # anchor jitter: a later frame maps through an earlier anchor
    log.write_frame(frame_number=2, timestamp_ms=33, capture_host_ns=t0 + 33_000_000,
                    frame_fw_ms=995.0)
    log.close()

    rows = _rows(p)
    assert rows[0][3] == rows[1][3], "frame_fw_ms should have been clamped"
    assert int(rows[1][2]) > int(rows[0][2]), (
        "the raw host instants must still separate the two rows")
