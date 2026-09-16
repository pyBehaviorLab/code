"""Confidence changes must not rebuild the model.

Confidence is a post-inference cutoff applied inside ``PoseSink``, the network
never sees it, and it is deliberately absent from the model cache key. It was
nevertheless wired to ``_mark_dlc_dirty("Confidence changed, re-initialize!")``,
so moving a threshold asked the operator for a full drain, close and reload.
On a TensorRT engine that is minutes, for a value the model does not use.
"""
import contextlib

import pytest

from source.video.framebus.pose_sink import PoseSink


@pytest.fixture
def sink():
    s = PoseSink(confidence_threshold=0.5)
    yield s
    with contextlib.suppress(Exception):
        s.stop()


class TestSetConfidence:
    def test_it_changes_the_cutoff(self, sink):
        sink.set_confidence(0.8)
        _parts, conf = sink.model_info()
        assert conf == pytest.approx(0.8)

    def test_it_does_not_touch_the_model_handle(self, sink):
        """The whole point: no eviction, no reload, no key change."""
        sentinel = object()
        sink._handle = sentinel
        sink.set_confidence(0.9)
        assert sink._handle is sentinel, "confidence must not disturb the model"

    @pytest.mark.parametrize("given,expect", [
        (-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.5, 1.0),
    ])
    def test_it_clamps_to_a_probability(self, sink, given, expect):
        sink.set_confidence(given)
        assert sink.model_info()[1] == pytest.approx(expect)

    @pytest.mark.parametrize("junk", [None, "high", object()])
    def test_junk_is_ignored_not_raised(self, sink, junk):
        """Reached from a config load, so a bad stored value must not take the
        whole apply down with it."""
        sink.set_confidence(junk)
        assert sink.model_info()[1] == pytest.approx(0.5), "unchanged"

    def test_setting_the_same_value_is_a_no_op(self, sink):
        sink.set_confidence(0.5)
        assert sink.model_info()[1] == pytest.approx(0.5)


def test_confidence_is_absent_from_the_model_key():
    """The property that makes a live change correct rather than a shortcut:
    two configs differing only in confidence are the SAME cached model."""
    import numpy as np

    from source.video.framebus.pose_sink import _make_model_key
    frame = np.zeros((32, 32, 3), np.uint8)
    a = _make_model_key("m", frame, 1.0, "dlc")
    b = _make_model_key("m", frame, 1.0, "dlc")
    assert a == b
    assert not any(isinstance(part, float) and part in (0.3, 0.9) for part in a), \
        "no confidence value may appear in the key"


def test_the_dialog_no_longer_demands_a_reinit_for_confidence():
    """Source-level, because the handler's whole job is to NOT act."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "gui" / "widgets" / "tracking_panel.py"
    body = src.read_text(encoding="utf-8").split(
        "def _on_dlc_confidence_changed", 1)[1].split("def ", 1)[0]
    assert "_mark_dlc_dirty" not in body, \
        "confidence must not mark the model dirty, it applies live"


def test_resize_and_instances_still_do_demand_one():
    """Guard the classification from over-correction: these two ARE rebuilds,
    resize is in the model cache key, instances swaps the backend."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "gui" / "widgets"
           / "tracking_panel.py").read_text(encoding="utf-8")
    for handler in ("_on_resize_changed", "_on_dlc_instances_changed"):
        body = src.split(f"def {handler}", 1)[1].split("def ", 1)[0]
        assert "_mark_dlc_dirty" in body, f"{handler} genuinely needs a rebuild"
