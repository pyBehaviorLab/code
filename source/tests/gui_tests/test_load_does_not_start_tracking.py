"""Opening a project prepares tracking; it does not start it.

Loading a project called ``_enable_pose_for_box``, which configures the model
AND calls ``pipeline.enable_pose`` - so the PoseSink began inferring on every
box the moment a project was opened. The line after it set
``tracking_enabled = False``, so the buttons showed "not tracking" while
inference ran: GPU busy, frames consumed, and the operator with no way to tell
from the interface.

What should happen on load is the expensive half only. The model is loaded and
warmed so the first Record is not cold, and nothing infers until the
experimenter presses Test Tracking or starts a run.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from source.gui.pose_subsystem import PoseSubsystemMixin


class _Pipeline:
    """Records which half of the two-step was called."""

    def __init__(self):
        self.enabled = []
        self.configured = []

    def enable_pose(self, setup_id, zone_lookup=None):
        self.enabled.append(setup_id)

    def disable_pose(self, setup_id):
        pass

    def has_pose_model(self):
        return True


class _Host(PoseSubsystemMixin):
    def __init__(self):
        self.pipeline = _Pipeline()
        self.tracking_enabled = {}
        self.configured = []

    def _configure_pose_for_box(self, setup_id, cfg):
        self.configured.append(setup_id)
        self.pipeline.configured.append(setup_id)
        return True


def test_configure_alone_does_not_start_inference():
    host = _Host()
    host._configure_pose_for_box(1, {})
    assert host.pipeline.configured == [1]
    assert host.pipeline.enabled == [], (
        "configuring loaded the model AND started the sink")


def test_enable_starts_inference():
    """The other half must still work, or Test Tracking and Record break."""
    host = _Host()
    assert host._enable_pose_for_box(2, {}) is True
    assert host.pipeline.enabled == [2]
    assert host.tracking_enabled[2] is True


def _code_only(func) -> str:
    """Source with comments stripped.

    The comment above the call names ``_enable_pose_for_box`` to say why it is
    NOT used, so a plain substring search over the source finds it and reports
    the opposite of the truth. Assert on code, not on prose about the code.
    """
    import inspect
    import io
    import tokenize

    src = inspect.getsource(func)
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            out.append(tok.string)
    return " ".join(out)


def test_the_load_path_configures_without_enabling():
    """The readiness pass must call the configure half, not the enable half."""
    from source.gui.base import MainWindowBase

    code = _code_only(MainWindowBase.ensure_pose_ready)
    assert "_configure_pose_for_box" in code, (
        "the readiness pass must still load the model")
    assert "_enable_pose_for_box" not in code, (
        "the readiness pass must not start the PoseSink; Test Tracking and "
        "Record are what turn tracking on")


def test_the_load_path_leaves_every_box_marked_not_tracking():
    import inspect

    from source.gui.base import MainWindowBase

    src = inspect.getsource(MainWindowBase.ensure_pose_ready)
    assert "self.tracking_enabled[bid] = False" in src
