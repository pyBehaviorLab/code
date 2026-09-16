"""The recording names its own keypoints.

The pose column in ``_video_data.txt`` is positional: ``[[x,y,conf], ...]`` in
whatever order the tracker emitted. Nothing recorded which part each position
belonged to, so the only route from a coordinate back to a name was the model
folder - which may have been retrained, moved, or never existed on the machine
doing the analysis. Downstream the gap showed up as an analyser carrying a
hardcoded table of guessed part names and a guessed anatomy.

``frame_log`` already had a ``bodyparts`` key in its ``#tracker`` block and
nothing ever set it. Now the writer records both the names and the skeleton,
and this pins the round trip through the analyser's own parser - which is the
reader that has to agree, and is allowed here precisely as a cross-check (see
``test_no_live_rig_imports.test_the_rig_does_not_import_the_analyser``).
"""
from __future__ import annotations

import os
from datetime import datetime

import pytest

from source.video.recording.frame_log import open_with_headers

PARTS = ["Snout", "Head", "Left_Ear", "Right_Ear", "center", "Tail_Base"]
SKELETON = [("Snout", "Head"), ("Left_Ear", "Head"), ("Right_Ear", "Head"),
            ("Head", "center"), ("center", "Tail_Base")]


def _write(tmp_path, **kw):
    path = os.path.join(str(tmp_path), "s_video_data.txt")
    log = open_with_headers(
        path, subject_id="282", setup_id=1, start_dt=datetime.now(),
        tracking_mode=kw.pop("tracking_mode", "sleap"),
        resolution=(360, 202), fps=20.0, rebind_drop_log=False, **kw)
    log._ensure_header_written()
    log.close()
    return path


def _tracker_line(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#tracker"):
                return line
    return ""


class TestTheWriterRecordsThem:
    def test_the_part_names_are_written(self, tmp_path):
        assert '"bodyparts":["Snout"' in _tracker_line(
            _write(tmp_path, bodyparts=PARTS, skeleton=SKELETON))

    def test_the_skeleton_is_written(self, tmp_path):
        assert '"skeleton":[["Snout","Head"]' in _tracker_line(
            _write(tmp_path, bodyparts=PARTS, skeleton=SKELETON))

    def test_a_session_without_pose_writes_neither(self, tmp_path):
        """A blob run has no keypoints; empty keys would be noise."""
        line = _tracker_line(_write(tmp_path, tracking_mode="blob"))
        assert "bodyparts" not in line and "skeleton" not in line

    def test_parts_without_a_skeleton_are_fine(self, tmp_path):
        """An exported DeepLabCut folder declares no skeleton, and the
        operator need not draw one."""
        line = _tracker_line(_write(tmp_path, bodyparts=["Head", "Tailbase"]))
        assert '"bodyparts":["Head","Tailbase"]' in line
        assert "skeleton" not in line


class TestTheAnalyserReadsThemBack:
    """The reader on the other side of the file has to agree."""

    @staticmethod
    def _header(path):
        from tools.offline_analysis.video_data_schema import VideoDataHeader
        return VideoDataHeader.parse(path)

    def test_the_names_survive_the_round_trip(self, tmp_path):
        header = self._header(_write(tmp_path, bodyparts=PARTS,
                                     skeleton=SKELETON))
        assert header.body_parts == PARTS

    def test_the_skeleton_survives_the_round_trip(self, tmp_path):
        header = self._header(_write(tmp_path, bodyparts=PARTS,
                                     skeleton=SKELETON))
        assert header.skeleton == SKELETON

    def test_an_older_recording_declares_neither(self, tmp_path):
        """A file that declares neither must read as "none declared" rather
        than raise, those fall back to sniffing the rows."""
        header = self._header(_write(tmp_path, tracking_mode="blob"))
        assert header.body_parts == []
        assert header.skeleton == []

    def test_an_edge_naming_an_unrecorded_part_is_dropped(self, tmp_path):
        """It cannot be drawn, so keeping it is a silent no-op."""
        path = _write(tmp_path, bodyparts=["Head", "Tailbase"],
                      skeleton=[("Head", "Tailbase"), ("Head", "ghost")])
        assert self._header(path).skeleton == [("Head", "Tailbase")]


@pytest.mark.skipif(
    not os.path.isfile("data/282-Box1-2026-05-27-110433_video_data.txt"),
    reason="the reference recording is not on this machine")
def test_a_real_older_recording_still_reads():
    """The recordings already on disk predate this and must not break."""
    from tools.offline_analysis.video_data_schema import VideoDataHeader
    header = VideoDataHeader.parse(
        "data/282-Box1-2026-05-27-110433_video_data.txt")
    assert header.skeleton == []
