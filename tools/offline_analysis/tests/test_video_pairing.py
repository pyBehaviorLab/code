"""Finding the video for a session, by header and by name.

Three failures this pins, all of which made a session show up with no video
(and so no frame for the zone editor to draw on):

  * ``_VID_NAME_RE`` excluded underscores and was used with ``search``, so the
    subject was taken from wherever the last underscore-free run began
  * pairing compared the MCU file's ``subject`` (which stops before
    ``-Box<n>-``) against a video's full stem, so it could never match
  * ``find_video_for_txt`` guessed from the filename and never read the
    ``video_file`` the recorder had written into the header
"""
from __future__ import annotations

import pytest

from tools.offline_analysis.engine.session_bundle import (
    find_video_for_txt, video_name_in_header)
from tools.offline_analysis.session_catalog import _VID_NAME_RE, _video_stem

CURRENT = "Validation_4box_cctv_latency_box1-Box1-2026-09-14-145732"
ARENA = "F645_PreInjection_280826105833"
V0 = "D860"


@pytest.mark.parametrize("stem", [CURRENT, ARENA, "39-Box2-2026-09-07-132547"])
def test_the_whole_stem_survives_underscores(stem):
    m = _VID_NAME_RE.match(stem + "_video_data.txt")
    assert m and m.group("subject") == stem


def test_v0_midname_form_still_yields_the_subject():
    m = _VID_NAME_RE.match(f"{V0}_video_data_-2025-07-30-113426.txt")
    assert m and m.group("subject") == V0


def test_video_stem_matches_the_mcu_stem():
    """What pairing compares. These must be equal or nothing ever pairs."""
    assert _video_stem(CURRENT + "_video_data.txt") == CURRENT
    assert _video_stem(ARENA + "_video_data.txt") == ARENA


def _write(tmp_path, txt_name, header, video_name=None, sub=""):
    d = tmp_path / sub if sub else tmp_path
    d.mkdir(parents=True, exist_ok=True)
    txt = tmp_path / txt_name
    txt.write_text(header, encoding="utf-8")
    if video_name:
        (d / video_name).write_bytes(b"\x00")
    return str(txt)


def test_header_names_the_video_current_format(tmp_path):
    p = _write(tmp_path, CURRENT + "_video_data.txt",
               f"#video_file        {CURRENT}.mp4\n"
               "#columns           frame elapsed pose zone state\n",
               CURRENT + ".mp4")
    assert video_name_in_header(p) == CURRENT + ".mp4"
    assert find_video_for_txt(p).endswith(CURRENT + ".mp4")


def test_header_names_the_video_arena_format(tmp_path):
    p = _write(tmp_path, ARENA + "_video_data.txt",
               "time\ttype\tsubtype\tcontent\n"
               f"0.000\tinfo\tvideo_file\t{ARENA}.mp4\n"
               "# frame_number\tcam_frame_id\tactual_ts\n",
               ARENA + ".mp4")
    assert video_name_in_header(p) == ARENA + ".mp4"
    assert find_video_for_txt(p).endswith(ARENA + ".mp4")


def test_header_wins_when_the_names_disagree(tmp_path):
    """The stem guess is wrong here; the header is right. This is the case
    that leaves the zone editor with no frame."""
    p = _write(tmp_path, ARENA + "_video_data.txt",
               "time\ttype\tsubtype\tcontent\n"
               "0.000\tinfo\tvideo_file\tF594_PostInjection_Day1_280826101148.mp4\n"
               "# frame_number\tcam_frame_id\tactual_ts\n",
               "F594_PostInjection_Day1_280826101148.mp4")
    got = find_video_for_txt(p)
    assert got.endswith("F594_PostInjection_Day1_280826101148.mp4")


def test_it_still_finds_by_stem_when_the_header_is_silent(tmp_path):
    p = _write(tmp_path, CURRENT + "_video_data.txt",
               "#columns frame elapsed pose zone state\n",
               CURRENT + ".mp4")
    assert video_name_in_header(p) == ""
    assert find_video_for_txt(p).endswith(CURRENT + ".mp4")


def test_video_in_a_sibling_video_folder(tmp_path):
    p = _write(tmp_path, ARENA + "_video_data.txt",
               "time\ttype\tsubtype\tcontent\n"
               f"0.000\tinfo\tvideo_file\t{ARENA}.mp4\n",
               ARENA + ".mp4", sub="video")
    assert find_video_for_txt(p).endswith(ARENA + ".mp4")


def test_missing_video_returns_empty_not_a_wrong_file(tmp_path):
    """No video must stay 'no video'. Adopting a stray file put a 64x64
    warm-up clip behind the zones once already."""
    p = _write(tmp_path, ARENA + "_video_data.txt",
               "time\ttype\tsubtype\tcontent\n"
               f"0.000\tinfo\tvideo_file\t{ARENA}.mp4\n")
    (tmp_path / "_encoder_warmup_14976.mp4").write_bytes(b"\x00")
    assert find_video_for_txt(p) == ""


def test_header_scan_stops_at_the_data(tmp_path):
    """A 100 MB session must not be scanned for a key that is not there."""
    body = "\n".join(f"{i}\t1\t{i * 33}\tna" for i in range(5000))
    p = _write(tmp_path, ARENA + "_video_data.txt",
               "time\ttype\tsubtype\tcontent\n"
               "# frame_number\tcam_frame_id\tactual_ts\tpose_array\n" + body)
    assert video_name_in_header(p) == ""
