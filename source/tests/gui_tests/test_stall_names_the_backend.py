"""A stall must name the backend that stalled.

Reported from a real session: a SLEAP model loaded correctly and one second
later the banner read

    DLC stalled (streak=3): ... model failed to initialise ... check log for
    DLCLiveTracker traceback

Three things wrong at once; it was not DLC, it had not failed to initialise,
and there is no DLCLiveTracker in a SLEAP run. An operator following that goes
to the wrong log, the wrong model folder and the wrong settings page.

The name is not decoration: it is the first thing the message is used for.
Every layer that can say it is checked here, because fixing the sink's text
left the GUI's own banner still saying "DLC".
"""
from __future__ import annotations

import pytest

pytest.importorskip("cv2")

from source.gui.base import MainWindowBase  # noqa: E402
from source.video.framebus.pose_sink import PoseSink  # noqa: E402

LABEL = MainWindowBase._pose_backend_label
ON_FAILED = MainWindowBase._on_pose_failed


class _TC:
    def __init__(self, tracker_type):
        self.tracker_type = tracker_type


class _Pipe:
    def __init__(self, tracker_type):
        self._tc = _TC(tracker_type)

    def get_tracking_config(self, _setup_id=None):
        return self._tc


class _Host:
    """The surface the stall path touches."""

    _pose_backend_label = LABEL
    _on_pose_failed = ON_FAILED

    def __init__(self, tracker_type):
        self.pipeline = _Pipe(tracker_type)
        self.said = []

    def _box_status_say(self, setup_id, text):
        self.said.append(text)


# ── the label itself ─────────────────────────────────────────────────────

@pytest.mark.parametrize("tracker_type,expected", [
    ("sleap", "SLEAP"),
    ("SLEAP", "SLEAP"),
    ("sleap_nn", "SLEAP"),
    ("dlc", "DLC"),
    ("DLC", "DLC"),
    ("blob", "Pose"),
    ("", "Pose"),
])
def test_the_label_follows_the_configured_backend(tracker_type, expected):
    assert _Host(tracker_type)._pose_backend_label(1) == expected


def test_a_host_with_no_pipeline_does_not_raise():
    """This runs inside a warning path. A raise here loses the warning."""
    class _Bare:
        _pose_backend_label = LABEL
    assert _Bare()._pose_backend_label(1) == "Pose"


# ── the banner the operator actually reads ───────────────────────────────

def test_a_sleap_stall_never_says_dlc():
    """The exact reported bug."""
    host = _Host("sleap")
    host._on_pose_failed(1, "pose has returned nothing for 300 frames", 300)
    assert host.said, "no banner raised"
    banner = host.said[0]
    assert banner.startswith("SLEAP stalled"), banner
    assert "DLC" not in banner, banner


def test_a_dlc_stall_still_says_dlc():
    """The other half, a correct name is not "never say DLC"."""
    host = _Host("dlc")
    host._on_pose_failed(1, "pose has returned nothing for 300 frames", 300)
    assert host.said[0].startswith("DLC stalled"), host.said[0]


# ── the sink's half of the sentence ──────────────────────────────────────

def _run_empty(tracker_type, n=None):
    sink = PoseSink()
    sink._tracker_type = tracker_type
    sink._body_parts = ["centroid"]
    sink._confidence = 0.5
    said = []
    sink.on_failed(lambda bid, reason, streak, **kw: said.append(reason))
    dead = {"centroid": [0.0, 0.0, 0.0]}
    for _ in range(n or sink._EMPTY_STALL_FRAMES):
        sink._on_pose_done(1, 1, dict(dead))
    sink.shutdown()
    return said


def test_the_sink_message_names_sleap_and_never_dlclivetracker():
    said = _run_empty("sleap")
    assert said, "the sink raised no failure"
    assert "sleap" in said[0].lower()
    assert "DLCLiveTracker" not in said[0]
    assert "DLC" not in said[0]


def test_the_sink_message_names_dlc_for_dlc():
    said = _run_empty("dlc")
    assert said and "dlc" in said[0].lower()


def test_neither_layer_claims_the_model_failed_to_initialise():
    """It said "model failed to initialise" one second AFTER the model had
    logged a successful init with all six body parts."""
    said = _run_empty("sleap")
    assert "failed to initialise" not in said[0].lower()


def test_the_banner_and_the_sink_agree_on_the_backend():
    """Two layers write this sentence between them; if they disagree the
    operator sees "DLC stalled: sleap pose has returned nothing"."""
    reason = _run_empty("sleap")[0]
    host = _Host("sleap")
    host._on_pose_failed(1, reason, 300)
    banner = host.said[0]
    assert banner.count("SLEAP") >= 1
    assert "DLC" not in banner, banner
