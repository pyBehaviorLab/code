"""Unit test for RunTask hardware-error path.

``PyboardError`` subclasses ``BaseException``, so ``tick_active`` must trap
``BaseException`` to catch it. This test exercises ``tick_active`` directly:
with a pycboard mock whose ``process_data`` raises ``PyboardError``, the call
must terminate the run (``framework_running == False``, sticky red
``Error: ...`` status) so the central process_timer drops the box.
"""
from __future__ import annotations

from unittest.mock import MagicMock


from source.communication.pyboard import PyboardError
from source.gui.widgets.run_task import RunTask


class _FakeStatusWidget:
    """Stand-in for a QLineEdit that records the last setText call."""

    def __init__(self) -> None:
        self.text_value = ""
        self.style_sheet = ""

    def setText(self, text: str) -> None:
        self.text_value = text

    def setStyleSheet(self, css: str) -> None:
        self.style_sheet = css


class _RunTaskHarness(RunTask):
    """Concrete RunTask without Qt parentage so init_run_task is enough."""

    def __init__(self) -> None:
        self.init_run_task(setup_id=1, main_window=None)
        # Pretend _build_ui already ran:
        self._status_widget = _FakeStatusWidget()
        self.framework_running = True

    def _log_func(self, msg: str) -> None:  # quiet logger
        pass


def test_tick_active_pyboarderror_triggers_mcu_stop():
    """PyboardError during process_data → mcu_stop runs →
    framework_running flips False, sticky 'Error: ...' status appears.
    """
    rt = _RunTaskHarness()
    pyc = MagicMock()
    pyc.framework_running = True
    pyc.process_data.side_effect = PyboardError("device serial error")
    rt.pycboard = pyc

    # Must not raise, the trap catches BaseException-derived PyboardError.
    rt.tick_active()

    assert rt.framework_running is False, \
        "framework_running must flip False after error stop"
    assert rt._status_widget.text_value.startswith("Error:"), \
        f"status must show sticky Error: prefix, got: {rt._status_widget.text_value!r}"


def test_tick_active_generic_exception_does_not_stop():
    """Non-board exceptions are logged + skipped (next tick may succeed),
    they must NOT trigger mcu_stop.
    """
    rt = _RunTaskHarness()
    pyc = MagicMock()
    pyc.framework_running = True
    pyc.process_data.side_effect = ValueError("transient parse hiccup")
    rt.pycboard = pyc

    rt.tick_active()

    assert rt.framework_running is True, \
        "transient non-board errors must not stop the framework"
