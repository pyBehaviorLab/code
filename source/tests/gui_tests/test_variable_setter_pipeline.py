"""End-to-end tests for the variable-setter pipeline.

Covers the full path from a controls-dialog ``Set`` click down to the
``Pycboard.set_variable`` call, including:

  - the live read-through ``VariableSetter.board`` property;
  - the central-MCU registry consistency: ``box_widget.pycboard``,
    ``main_window.mcu[box_id]``, and the dialog all see the same
    instance;
  - re-pull on showEvent: ``refresh_enabled_states`` rebinds
    ``StandardControlsTab.board`` to the live pycboard so set/get/reload
    talk to the current board after a reconnect or a mid-session task
    re-upload;
  - defensive guards: set/get/reload do not raise when the board is
    None or sm_info is missing a variable name.
"""

from __future__ import annotations

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from unittest.mock import MagicMock

import pytest
from PySide6 import QtWidgets

from source.communication.controller import MCUController
from source.communication.pycboard import Pycboard
from source.gui.dialogs.controls import StandardControlsTab, VariableSetter
from source.tests.qt_dispose import WidgetBin

_BIN = WidgetBin()


@pytest.fixture(autouse=True)
def _drain_widget_bin():
    """Destroy the widgets this file's helpers built (see qt_dispose)."""
    yield
    _BIN.drain()



# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _make_pycboard(running: bool = False, variables: dict | None = None):
    """Mock that quacks like a Pycboard, only the attributes the
    dialog reads."""
    b = MagicMock(spec=Pycboard)
    b.framework_running = running
    b.timestamp = 0
    b.sm_info = MagicMock()
    b.sm_info.variables = variables if variables is not None else {
        "trial_count": 0,
        "reward_volume": 5.0,
        "session_dur": 1200,
    }
    b.sm_info.events = {"reward": 1, "zone_changed": 2}
    return b


def _make_box_widget(qapp, setup_number: int = 1, pycboard=None):
    """Bare object that has the two attributes the dialog reads:
    ``box_number`` and ``pycboard``. We don't construct the real
    ``BoxControlWidget`` because that pulls Qt + the entire app stack."""
    bw = MagicMock()
    bw.setup_number = setup_number
    bw.pycboard = pycboard
    return bw


# VariableSetter reads through parent_tab.board (no snapshot)


class TestLiveBoardReadthrough:
    def test_set_uses_current_board_after_reconnect(self, qapp):
        """The user disconnects + reconnects mid-session; the
        VariableSetter must talk to the NEW board, not the one present
        when the dialog was created."""
        first = _make_pycboard(running=True)
        bw = _make_box_widget(qapp, pycboard=first)
        tab = _BIN.add(StandardControlsTab(setup_widget=bw))
        # Find the row for trial_count
        setters = tab.findChildren(VariableSetter)
        target = next(s for s in setters if s.v_name == "trial_count")

        # Simulate reconnect: box widget gets a new pycboard.
        second = _make_pycboard(running=True)
        bw.pycboard = second
        tab.refresh_enabled_states()  # what showEvent triggers

        # User types "42" + clicks Set
        target.value_str.setText("42")
        target.set()

        # The new board got the call; the stale first one did not.
        second.set_variable.assert_called_once_with("trial_count", 42)
        first.set_variable.assert_not_called()

    def test_get_uses_current_board_after_reconnect(self, qapp):
        first = _make_pycboard(running=True)
        bw = _make_box_widget(qapp, pycboard=first)
        tab = _BIN.add(StandardControlsTab(setup_widget=bw))
        target = next(s for s in tab.findChildren(VariableSetter)
                      if s.v_name == "reward_volume")

        second = _make_pycboard(running=True)
        bw.pycboard = second
        tab.refresh_enabled_states()

        target.get()
        second.get_variable.assert_called_once_with("reward_volume")
        first.get_variable.assert_not_called()


# None-board defensive guards


class TestDefensiveGuards:
    def test_set_with_none_board_does_not_raise(self, qapp):
        bw = _make_box_widget(qapp, pycboard=_make_pycboard())
        tab = _BIN.add(StandardControlsTab(setup_widget=bw))
        target = next(s for s in tab.findChildren(VariableSetter)
                      if s.v_name == "trial_count")

        # Clear the board.
        bw.pycboard = None
        tab.refresh_enabled_states()

        target.value_str.setText("99")
        target.set()  # must not raise
        assert target.value_str.text() == "Not connected"

    def test_get_with_none_board_does_not_raise(self, qapp):
        bw = _make_box_widget(qapp, pycboard=_make_pycboard())
        tab = _BIN.add(StandardControlsTab(setup_widget=bw))
        target = next(s for s in tab.findChildren(VariableSetter)
                      if s.v_name == "trial_count")

        bw.pycboard = None
        tab.refresh_enabled_states()

        target.get()  # must not raise
        assert target.value_str.text() == "Not connected"

    def test_reload_with_missing_variable_does_not_raise(self, qapp):
        b = _make_pycboard(running=True)
        bw = _make_box_widget(qapp, pycboard=b)
        tab = _BIN.add(StandardControlsTab(setup_widget=bw))
        target = next(s for s in tab.findChildren(VariableSetter)
                      if s.v_name == "trial_count")

        # Simulate the variable disappearing from sm_info (e.g. after
        # task re-upload removed the variable). Reload must not raise.
        b.sm_info.variables = {}
        target.reload()  # must not raise


# PyboardError catch (extends BaseException, not Exception)


class TestPyboardErrorCatch:
    def test_set_swallows_pyboard_error(self, qapp):
        from source.communication.pycboard import PyboardError
        b = _make_pycboard(running=True)
        b.set_variable.side_effect = PyboardError("simulated MCU error")
        bw = _make_box_widget(qapp, pycboard=b)
        tab = _BIN.add(StandardControlsTab(setup_widget=bw))
        target = next(s for s in tab.findChildren(VariableSetter)
                      if s.v_name == "trial_count")

        target.value_str.setText("7")
        target.set()  # must not raise, PyboardError extends BaseException

        assert target.value_str.text() == "Set error"

    def test_get_swallows_pyboard_error(self, qapp):
        from source.communication.pycboard import PyboardError
        b = _make_pycboard(running=True)
        b.get_variable.side_effect = PyboardError("simulated")
        bw = _make_box_widget(qapp, pycboard=b)
        tab = _BIN.add(StandardControlsTab(setup_widget=bw))
        target = next(s for s in tab.findChildren(VariableSetter)
                      if s.v_name == "trial_count")
        target.get()  # must not raise
        assert target.value_str.text() == "Get error"


# Sync vs async path semantics


class TestSyncAsyncPaths:
    def test_async_path_when_framework_running(self, qapp):
        """Framework running -> set_variable returns None (async),
        ``setting..`` placeholder, 200ms reload scheduled."""
        b = _make_pycboard(running=True)
        b.set_variable.return_value = None
        bw = _make_box_widget(qapp, pycboard=b)
        tab = _BIN.add(StandardControlsTab(setup_widget=bw))
        target = next(s for s in tab.findChildren(VariableSetter)
                      if s.v_name == "trial_count")
        target.value_str.setText("42")
        target.set()
        assert target.value_str.text() == "setting.."
        b.set_variable.assert_called_once_with("trial_count", 42)

    def test_sync_path_returns_set_failed_on_false(self, qapp):
        """Framework idle -> set_variable returns True/False
        immediately; False -> ``Set failed``."""
        b = _make_pycboard(running=False)
        b.set_variable.return_value = False
        bw = _make_box_widget(qapp, pycboard=b)
        tab = _BIN.add(StandardControlsTab(setup_widget=bw))
        target = next(s for s in tab.findChildren(VariableSetter)
                      if s.v_name == "trial_count")
        target.value_str.setText("42")
        target.set()
        assert target.value_str.text() == "Set failed"

    def test_sync_path_clears_colour_on_success(self, qapp):
        b = _make_pycboard(running=False)
        b.set_variable.return_value = True
        bw = _make_box_widget(qapp, pycboard=b)
        tab = _BIN.add(StandardControlsTab(setup_widget=bw))
        target = next(s for s in tab.findChildren(VariableSetter)
                      if s.v_name == "trial_count")
        target.value_str.setText("42")
        target.set()
        # Success path keeps the typed value in the field.
        assert target.value_str.text() == "42"


# Cross-pipeline: dialog board === main_window.mcu[box_id]


class TestCrossPipelineConsistency:
    """The user's invariant: ``box_widget.pycboard`` and
    ``main_window.mcu[box_id]`` must be the same instance, so the
    dialog (which reads ``box_widget.pycboard``) and the central
    MCUController see the same board for any per-box op."""

    def test_dialog_and_mcu_see_same_board(self, qapp):
        b = _make_pycboard(running=True)
        bw = _make_box_widget(qapp, pycboard=b)
        # Simulate what BoxWidgetMixin.mcu_connect does.
        mcu = MCUController()
        mcu.register(bw.setup_number, b)
        # Now the dialog…
        tab = _BIN.add(StandardControlsTab(setup_widget=bw))
        target = next(s for s in tab.findChildren(VariableSetter)
                      if s.v_name == "trial_count")

        # The dialog's live board === the central registry's board.
        assert target.board is mcu[bw.setup_number]

    def test_disconnect_path_propagates_to_dialog(self, qapp):
        """After mcu_disconnect (which clears box_widget.pycboard AND
        unregisters from MCUController), the dialog must observe a
        None board on its next refresh."""
        b = _make_pycboard(running=False)
        bw = _make_box_widget(qapp, pycboard=b)
        mcu = MCUController()
        mcu.register(bw.setup_number, b)
        tab = _BIN.add(StandardControlsTab(setup_widget=bw))
        target = next(s for s in tab.findChildren(VariableSetter)
                      if s.v_name == "trial_count")
        assert target.board is b

        # Simulate disconnect.
        bw.pycboard = None
        mcu.unregister(bw.setup_number)
        tab.refresh_enabled_states()
        assert target.board is None


# Refresh on showEvent rebuilds variables when sm_info changes


class TestRefreshSemantics:
    def test_refresh_picks_up_new_variables(self, qapp):
        """User uploads a task with one variable, opens dialog,
        (closes), uploads a different task with new variables, reopens,
        the dialog rows reflect the new task."""
        b = _make_pycboard(variables={"trial_count": 0})
        bw = _make_box_widget(qapp, pycboard=b)
        tab = _BIN.add(StandardControlsTab(setup_widget=bw))
        names1 = sorted(s.v_name for s in tab.findChildren(VariableSetter))
        assert names1 == ["trial_count"]

        # New task uploaded with different variables.
        b.sm_info.variables = {"reward_volume": 5.0, "iti_ms": 1000}
        tab.refresh_enabled_states()  # what showEvent triggers
        names2 = sorted(s.v_name for s in tab.findChildren(VariableSetter))
        assert names2 == ["iti_ms", "reward_volume"]

    def test_refresh_filters_pyControl_internal_vars(self, qapp):
        """Internal-name variables (``___`` suffix), and the magic
        ``custom_controls_dialog`` / ``api_class`` keys are hidden from
        the Standard tab. ``hw_*`` variables ARE shown (highlighted) so
        the user can configure their reset/persist behaviour from the
        same per-setup grid as the rest."""
        b = _make_pycboard(variables={
            "trial_count": 0,
            "hw_solenoid_pin": "X1",       # hw_ var, shown (highlighted)
            "internal___": 99,             # filtered
            "custom_controls_dialog": "x", # filtered
            "api_class": None,             # filtered
        })
        bw = _make_box_widget(qapp, pycboard=b)
        tab = _BIN.add(StandardControlsTab(setup_widget=bw))
        names = sorted(s.v_name for s in tab.findChildren(VariableSetter))
        assert names == ["hw_solenoid_pin", "trial_count"], (
            f"internal/magic vars leaked or hw_ filtered: {names}"
        )
