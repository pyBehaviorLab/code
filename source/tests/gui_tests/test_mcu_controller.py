"""Tests for ``source.communication.controller.MCUController``.

The controller is intentionally thin, a dict-like registry of
``box_id -> Pycboard``. These tests cover:

    1. Registry semantics (register / unregister / get / __getitem__ /
       __contains__ / __iter__ / __len__ / keys / values / items).
    2. Re-registration is idempotent (same box_id replaces).
    3. Pycboard methods are not wrapped, callers reach methods like
       ``set_variable`` and ``framework_running`` directly via
       ``mcu[box_id].method()``; indexing returns the exact registered
       object.
    4. ``stop_all`` calls ``stop_framework`` only on running boards and
       swallows per-board exceptions without aborting the loop.
    5. ``close_all`` stops + closes every board and clears the registry.

We mock ``Pycboard`` because it owns a real serial port; the controller
itself has no business with pyserial.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from source.communication.controller import MCUController


def _make_board(framework_running: bool = False) -> MagicMock:
    """Return a ``MagicMock`` that quacks like a ``Pycboard`` for the
    handful of attributes the controller's bulk helpers touch."""
    b = MagicMock(name="Pycboard")
    b.framework_running = framework_running
    return b


# ── Registry semantics ──────────────────────────────────────────────


class TestRegistry:
    def test_empty_on_construction(self):
        mcu = MCUController()
        assert len(mcu) == 0
        assert list(mcu) == []
        assert list(mcu.keys()) == []
        assert list(mcu.values()) == []
        assert list(mcu.items()) == []
        assert mcu.boards == {}

    def test_register_and_index(self):
        mcu = MCUController()
        b = _make_board()
        mcu.register(1, b)
        assert mcu[1] is b
        assert 1 in mcu
        assert len(mcu) == 1
        assert list(mcu) == [1]

    def test_register_accepts_int_str_box_id(self):
        """``box_id`` is coerced to ``int`` so callers can pass either."""
        mcu = MCUController()
        b = _make_board()
        mcu.register("3", b)
        assert mcu[3] is b
        assert mcu["3"] is b
        assert 3 in mcu
        assert "3" in mcu  # __contains__ also coerces

    def test_get_returns_default_when_missing(self):
        mcu = MCUController()
        assert mcu.get(99) is None
        sentinel = object()
        assert mcu.get(99, sentinel) is sentinel

    def test_unregister_returns_board(self):
        mcu = MCUController()
        b = _make_board()
        mcu.register(2, b)
        popped = mcu.unregister(2)
        assert popped is b
        assert 2 not in mcu
        assert len(mcu) == 0

    def test_unregister_missing_returns_none(self):
        mcu = MCUController()
        assert mcu.unregister(404) is None

    def test_register_replaces_existing(self):
        """Re-registering same ``box_id`` swaps the entry, caller is
        responsible for closing the old one."""
        mcu = MCUController()
        a, b = _make_board(), _make_board()
        mcu.register(7, a)
        mcu.register(7, b)
        assert mcu[7] is b
        assert len(mcu) == 1

    def test_iteration_after_multiple_registrations(self):
        mcu = MCUController()
        b1, b2, b3 = _make_board(), _make_board(), _make_board()
        mcu.register(1, b1)
        mcu.register(2, b2)
        mcu.register(3, b3)
        assert sorted(mcu.keys()) == [1, 2, 3]
        assert set(mcu.values()) == {b1, b2, b3}
        assert dict(mcu.items()) == {1: b1, 2: b2, 3: b3}

    def test_indexing_missing_raises_keyerror(self):
        mcu = MCUController()
        with pytest.raises(KeyError):
            _ = mcu[42]


# ── No method wrapping, callers use Pycboard names directly ────────


class TestPycboardPassthrough:
    """Pycboard's API is reachable via ``mcu[box_id]`` with the original
    method names, guards against re-introducing wrappers."""

    def test_set_variable_via_index(self):
        mcu = MCUController()
        b = _make_board(framework_running=True)
        b.set_variable.return_value = None
        mcu.register(1, b)

        result = mcu[1].set_variable("trial_count", 42, source="a")
        assert result is None
        b.set_variable.assert_called_once_with("trial_count", 42, source="a")

    def test_framework_running_attribute_via_index(self):
        mcu = MCUController()
        b = _make_board(framework_running=True)
        mcu.register(1, b)
        assert mcu[1].framework_running is True
        b.framework_running = False
        assert mcu[1].framework_running is False

    def test_trigger_event_via_values_iteration(self):
        mcu = MCUController()
        b1, b2 = _make_board(framework_running=True), _make_board(framework_running=True)
        mcu.register(1, b1)
        mcu.register(2, b2)
        for board in mcu.values():
            board.trigger_event("zone_changed", source="t")
        b1.trigger_event.assert_called_once_with("zone_changed", source="t")
        b2.trigger_event.assert_called_once_with("zone_changed", source="t")


# ── stop_all / close_all bulk helpers ───────────────────────────────


class TestStopAll:
    def test_stops_only_running_boards(self):
        mcu = MCUController()
        running = _make_board(framework_running=True)
        idle = _make_board(framework_running=False)
        mcu.register(1, running)
        mcu.register(2, idle)
        mcu.stop_all()
        running.stop_framework.assert_called_once()
        idle.stop_framework.assert_not_called()

    def test_swallows_per_board_exceptions(self):
        """One failing stop_framework must not block the rest."""
        mcu = MCUController()
        good = _make_board(framework_running=True)
        bad = _make_board(framework_running=True)
        bad.stop_framework.side_effect = RuntimeError("simulated")
        mcu.register(1, bad)
        mcu.register(2, good)
        mcu.stop_all()  # must not raise
        good.stop_framework.assert_called_once()
        bad.stop_framework.assert_called_once()

    def test_swallows_pyboarderror_baseexception(self):
        """``PyboardError`` extends ``BaseException`` (not ``Exception``).
        The except clause must catch ``BaseException`` or these escape."""
        from source.communication.pycboard import PyboardError
        mcu = MCUController()
        b = _make_board(framework_running=True)
        b.stop_framework.side_effect = PyboardError("pyboard simulated")
        mcu.register(1, b)
        mcu.stop_all()  # must not raise
        b.stop_framework.assert_called_once()

    def test_does_not_remove_from_registry(self):
        """``stop_all`` only stops; it does not unregister."""
        mcu = MCUController()
        b = _make_board(framework_running=True)
        mcu.register(1, b)
        mcu.stop_all()
        assert 1 in mcu


class TestCloseAll:
    def test_stops_then_closes_each(self):
        mcu = MCUController()
        b = _make_board(framework_running=True)
        mcu.register(1, b)
        mcu.close_all()
        b.stop_framework.assert_called_once()
        b.close.assert_called_once()
        # Verify ordering: stop must come before close.
        order = [c[0] for c in b.method_calls]
        assert order.index("stop_framework") < order.index("close")

    def test_closes_idle_boards_too(self):
        mcu = MCUController()
        b = _make_board(framework_running=False)
        mcu.register(1, b)
        mcu.close_all()
        b.stop_framework.assert_not_called()  # not running -> skip stop
        b.close.assert_called_once()

    def test_clears_registry(self):
        mcu = MCUController()
        mcu.register(1, _make_board())
        mcu.register(2, _make_board())
        mcu.close_all()
        assert len(mcu) == 0
        assert mcu.boards == {}

    def test_close_failure_still_clears_other_boards(self):
        mcu = MCUController()
        bad = _make_board()
        good = _make_board()
        bad.close.side_effect = RuntimeError("close failed")
        mcu.register(1, bad)
        mcu.register(2, good)
        mcu.close_all()  # must not raise
        good.close.assert_called_once()
        # Registry is cleared regardless of per-board errors.
        assert len(mcu) == 0


# ── Public API contract ─────────────────────────────────────────────


class TestPublicSurface:
    """Pin the public name set so re-introducing fat wrappers
    (set_variable, trigger_event, ...) gets caught here."""

    EXPECTED_PUBLIC = {
        # registry / dict-like
        "boards", "get", "items", "keys", "register", "unregister", "values",
        # bulk shutdown
        "close_all", "stop_all",
    }

    def test_only_thin_registry_methods_exposed(self):
        mcu = MCUController()
        public = {n for n in dir(mcu) if not n.startswith("_")}
        assert public == self.EXPECTED_PUBLIC, (
            f"Public surface drifted from thin registry. "
            f"Added: {public - self.EXPECTED_PUBLIC}, "
            f"Removed: {self.EXPECTED_PUBLIC - public}"
        )
