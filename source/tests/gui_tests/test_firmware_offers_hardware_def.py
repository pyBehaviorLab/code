"""Loading the framework offers to send the hardware definition after it.

A framework load leaves the board without its pin map, so the hardware
definition has to go up again. Nothing said so, and a board left in that state
runs a task whose inputs are wired to nothing: the events never arrive, and the
failure is silent because a missing pin map is not an error, it is simply no
input.

The prompt is a question, not an action. A board being re-flashed before its
wiring is decided has no hardware definition to send yet, so declining has to
leave the board exactly as the framework load left it.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PySide6")

from source.gui.dialogs import mcu as mcu_mod


def _config_dialog_cls():
    for obj in vars(mcu_mod).values():
        if isinstance(obj, type) and hasattr(obj, "uploadFramework"):
            return obj
    raise AssertionError("no dialog class exposes uploadFramework")


def test_the_framework_upload_offers_the_hardware_definition():
    src = inspect.getsource(_config_dialog_cls().uploadFramework)
    assert "all_finished=" in src, (
        "the offer must run once at the end of the batch, not per board")
    assert "loadHardwareDefinition" in src, (
        "the prompt must lead to the existing HD loader rather than a "
        "second implementation of it")


def test_the_offer_runs_after_every_board_not_during():
    """``post_action`` fires per box while the batch is still going; a modal
    question there would interrupt the upload once per board."""
    src = inspect.getsource(_config_dialog_cls().uploadFramework)
    post = src.split("post_action=")[1].split("\n")[0]
    assert "loadHardwareDefinition" not in post
    assert "MessageBox" not in post


def test_declining_uploads_nothing():
    """The only action behind the prompt is the HD loader, and it is reached
    solely through the Yes branch."""
    src = inspect.getsource(_config_dialog_cls().uploadFramework)
    yes_branch = src.split("StandardButton.Yes:")[-1]
    assert "self.loadHardwareDefinition()" in yes_branch
    # One call site only: nothing outside the branch triggers an upload.
    assert src.count("self.loadHardwareDefinition()") == 1


def test_the_hd_loader_still_asks_for_a_file():
    """The prompt reuses the normal loader, so the operator still chooses the
    file. Sending whatever was last used would be a different, worse feature.
    """
    src = inspect.getsource(_config_dialog_cls().loadHardwareDefinition)
    assert "getOpenFileName" in src
