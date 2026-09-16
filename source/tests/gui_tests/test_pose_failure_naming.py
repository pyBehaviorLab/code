"""What the operator is told when pose inference stops, on either backend.

The stall modal was titled "DLC tracking has stalled" whatever was running, so
a SLEAP model that went quiet was reported as a DLC failure and sent the
operator to the wrong log, the wrong model folder and the wrong settings. The
title is now the thing that actually happened, pose inference stalled, and the
backend is named in the body, from the box's own tracking config rather than
from the string the message happened to be written with.

``_pose_backend_label`` already existed for this; the modal simply did not use
it. These drive the real handler, not the helper, because that gap is the bug.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6 import QtWidgets

from source.gui.base import MainWindowBase


class _Host:
    """Just the two collaborators the stall path touches."""

    _on_pose_failed = MainWindowBase._on_pose_failed
    _pose_backend_label = MainWindowBase._pose_backend_label

    def __init__(self, tracker_type):
        self.pipeline = SimpleNamespace(
            get_tracking_config=lambda _sid: SimpleNamespace(
                tracker_type=tracker_type))
        self.said = []

    def _box_status_say(self, setup_id, msg):
        self.said.append(msg)


@pytest.fixture
def shown(monkeypatch):
    """Capture the modal instead of blocking on it."""
    seen = {}

    def _warn(_parent, title, text, *a, **k):
        seen["title"] = title
        seen["text"] = text
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(_warn))
    return seen


#: Well past ``_EMPTY_STALL_FRAMES * 10``, which is what escalates to a modal.
_LOUD = 100_000


def test_the_title_names_what_happened_not_a_backend(shown):
    host = _Host("sleap_topdown")
    host._on_pose_failed(1, "all confidences zero", _LOUD)
    assert shown["title"] == "Pose inference has stalled"
    assert "DLC" not in shown["title"]


@pytest.mark.parametrize("tracker_type,expected", [
    ("sleap_topdown", "SLEAP"),
    ("dlc_live", "DLC"),
    ("", "Pose"),                      # configured with nothing recognisable
])
def test_the_body_names_the_backend_that_actually_stalled(
        shown, tracker_type, expected):
    host = _Host(tracker_type)
    host._on_pose_failed(3, "all confidences zero", _LOUD)
    assert expected in shown["text"], shown["text"]
    if expected != "DLC":
        assert "DLC" not in shown["text"], (
            f"a {expected} stall still mentions DLC: {shown['text']}")


def test_a_sleap_box_is_never_told_to_re_apply_dlc_settings(shown):
    """The remedy sentence named DLC too, which is the part that actually
    sends someone to the wrong dialog."""
    host = _Host("sleap_topdown")
    host._on_pose_failed(2, "all confidences zero", _LOUD)
    assert "re-Apply the SLEAP settings" in shown["text"], shown["text"]


def test_the_box_status_line_agrees_with_the_modal(shown):
    """Two surfaces, one answer: the banner and the modal must not name
    different backends for the same box."""
    host = _Host("sleap_topdown")
    host._on_pose_failed(4, "all confidences zero", _LOUD)
    assert host.said and "SLEAP" in host.said[0]
    assert "SLEAP" in shown["text"]


def test_a_short_streak_does_not_interrupt_the_operator(shown):
    """The modal is the escalation, not the first word: a transient blip
    raises the status line only."""
    host = _Host("dlc_live")
    host._on_pose_failed(5, "one empty batch", 3)
    assert "title" not in shown, "a blip popped a modal"
    assert host.said, "and it should still have said something"


def test_the_modal_fires_once_per_box(shown):
    """A stalled box reports continuously; a modal per report is unusable."""
    host = _Host("dlc_live")
    host._on_pose_failed(6, "all confidences zero", _LOUD)
    shown.clear()
    host._on_pose_failed(6, "all confidences zero", _LOUD + 1)
    assert "title" not in shown, "the same box popped a second modal"
