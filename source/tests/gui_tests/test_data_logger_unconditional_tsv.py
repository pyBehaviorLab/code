"""TSV save contract: subject_id gates dry-run vs real run.

- subject_id == ""  -> dry run; caller opens no TSV (gated at the call site,
  not in data_logger). Framework still starts for hardware sanity-checks.
- subject_id set    -> TSV is opened, independent of video. Path separators in
  subject_ID are sanitised so a stray "zfs/z" can't cause a missing-folder error.
"""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import Mock


from source.communication.data_logger import Data_logger


def _board_stub():
    """Minimal pycboard stand-in for header-line writes.

    ``task_hash`` / ``hardware_def_hash`` are 32-bit ints in real ``sm_info``;
    ``Data_logger`` renders them as 8-char zero-padded hex in the TSV header, so
    the stub passes ints matching that convention.
    """
    board = Mock()
    board.sm_info.name = "testtask"
    board.sm_info.task_hash = 0xdeadbeef
    board.sm_info.hardware_def_name = "test_hw"
    board.sm_info.hardware_def_hash = 0xabc123
    board.sm_info.devices = {"poke.py": 0x1234abcd, "audio_board.py": 0x5678ef01}
    board.sm_info.framework_version = "1.0"
    board.sm_info.micropython_version = "1.20"
    board.sm_info.analog_inputs = {}
    board.timestamp = 0
    return board


def test_open_data_file_with_real_subject(tmp_path):
    dl = Data_logger(_board_stub())
    dt = datetime(2026, 5, 13, 9, 30, 0)
    dl.open_data_file(
        data_dir=str(tmp_path),
        subject_ID="M001",
        datetime_now=dt,
        box_ID=2,
    )
    expected = tmp_path / "M001-Box2-2026-05-13-093000.tsv"
    assert expected.exists()
    assert dl.subject_ID == "M001"
    dl.close_files()


def test_subject_id_with_slash_sanitised(tmp_path):
    """A subject_id like ``zfs/z`` must not become a missing-folder lookup.
    Slash becomes underscore in the filename; the header keeps the literal."""
    dl = Data_logger(_board_stub())
    dt = datetime(2026, 5, 13, 9, 30, 0)
    dl.open_data_file(
        data_dir=str(tmp_path),
        subject_ID="zfs/z",
        datetime_now=dt,
        box_ID=1,
    )
    expected = tmp_path / "zfs_z-Box1-2026-05-13-093000.tsv"
    assert expected.exists()
    # NO subfolder was created.
    assert not (tmp_path / "zfs").exists()
    dl.close_files()


def test_subject_id_with_backslash_sanitised(tmp_path):
    """Same rule for backslash (Windows path separator)."""
    dl = Data_logger(_board_stub())
    dt = datetime(2026, 5, 13, 9, 30, 0)
    dl.open_data_file(
        data_dir=str(tmp_path),
        subject_ID="zfs\\z",
        datetime_now=dt,
        box_ID=1,
    )
    assert (tmp_path / "zfs_z-Box1-2026-05-13-093000.tsv").exists()
    dl.close_files()


def test_tsv_header_columns_present(tmp_path):
    dl = Data_logger(_board_stub())
    dt = datetime(2026, 5, 13, 9, 30, 0)
    dl.open_data_file(
        data_dir=str(tmp_path),
        subject_ID="M001",
        datetime_now=dt,
        box_ID=1,
    )
    dl.close_files()
    f = tmp_path / "M001-Box1-2026-05-13-093000.tsv"
    body = f.read_text(encoding="utf-8")
    first_line = body.splitlines()[0]
    assert first_line == "time\ttype\tsubtype\tcontent"
    # The devices lineage line carries a JSON {name: 8-char-hex} dict, same
    # hex rendering as task/HD hashes so it cross-references source/devices/.
    dev_line = next(l for l in body.splitlines() if "\tdevices\t" in l)
    assert json.loads(dev_line.split("\t", 3)[3]) == {
        "poke.py": "1234abcd", "audio_board.py": "5678ef01"}
