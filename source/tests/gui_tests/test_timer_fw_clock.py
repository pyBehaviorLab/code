"""Unit tests for the FW-clock session timer (June 2026).

The session HH:MM:SS clock is driven by the MCU framework time
(``pycboard.get_timestamp()`` / ``pycboard.timestamp``), the SAME value
written into the TSV, not the host wall clock. These tests exercise the
three changed code paths directly:

  1. ``format_run_clock``: the single formatter.
  2. ``MainWindowBase._on_process_tick``: reads ``pycboard.get_timestamp()``
     per running box and skips a box that auto-stopped mid-tick.
  3. ``RunTask.mcu_stop``: freezes the label at the exact final
     ``pycboard.timestamp`` (no interpolation, matches the last TSV row).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from source.datetime_formats import format_run_clock
from source.gui.base import MainWindowBase
from source.gui.widgets.run_task import RunTask


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------
class _FakeLabel:
    def __init__(self, text: str = "") -> None:
        self._t = text

    def text(self) -> str:
        return self._t

    def setText(self, t: str) -> None:
        self._t = t


class _FakeStatusWidget:
    def __init__(self) -> None:
        self.text_value = ""

    def setText(self, text: str) -> None:
        self.text_value = text

    def setStyleSheet(self, css: str) -> None:
        pass


class _FakePyc:
    """Minimal pycboard stand-in: get_timestamp() returns fw ms."""

    def __init__(self, ms: int) -> None:
        self.timestamp = ms

    def get_timestamp(self) -> int:
        return self.timestamp


class _FakeBox:
    def __init__(self, ms: int, running: bool = True) -> None:
        self.framework_running = running
        self.pycboard = _FakePyc(ms)
        self._timer_label = _FakeLabel()
        self.setup_number = 1
        self.ticked = False

    def tick_active(self) -> None:
        self.ticked = True


class _AutoStopBox(_FakeBox):
    """Simulates a box whose MCU run ended during tick_active()."""

    def tick_active(self) -> None:
        self.ticked = True
        self.framework_running = False


class _StubWindow:
    """Just enough surface for MainWindowBase._on_process_tick."""

    # Exercise the REAL single fan-out path (box card + Live Status + the
    # mode-specific _stamp_box_timer hook), which the tick now delegates to.
    _apply_box_timer = MainWindowBase._apply_box_timer

    def __init__(self, boxes) -> None:
        self._boxes = boxes
        self._live_status_by_box = None
        self.paints = 0
        self.syncs = 0

    def get_all_setup_widgets(self):
        return self._boxes

    def _paint_streaming_cameras_once(self) -> None:
        self.paints += 1

    def _sync_timer_mode(self) -> None:
        self.syncs += 1


# --------------------------------------------------------------------------
# 1. formatter
# --------------------------------------------------------------------------
def test_format_run_clock_basic():
    assert format_run_clock(0) == "00:00:00"
    assert format_run_clock(999) == "00:00:00"          # sub-second floors
    assert format_run_clock(1000) == "00:00:01"
    assert format_run_clock(3_599_999) == "00:59:59"
    assert format_run_clock(3_600_000) == "01:00:00"    # exact hour
    assert format_run_clock(3_661_000) == "01:01:01"
    assert format_run_clock(36_000_000) == "10:00:00"   # double-digit hours


def test_format_run_clock_clamps_negative():
    # A pre-first-message read can momentarily go negative; must clamp.
    assert format_run_clock(-5) == "00:00:00"


# --------------------------------------------------------------------------
# 2. central tick
# --------------------------------------------------------------------------
def test_process_tick_writes_fw_clock_to_label():
    box = _FakeBox(ms=3_661_000)            # 1h 1m 1s of fw time
    win = _StubWindow([box])

    MainWindowBase._on_process_tick(win)

    assert box.ticked, "tick_active must be called for a running box"
    assert box._timer_label.text() == "01:01:01", \
        f"label must show fw time, got {box._timer_label.text()!r}"
    # Painting is NOT this tick's job; it has its own timer, so that a slow
    # repaint cannot delay the MCU drain this tick exists for.
    assert win.paints == 0, "the realtime tick must not paint tiles"
    assert win.syncs == 1


def test_process_tick_skips_box_that_autostopped_in_tick():
    """If tick_active() auto-stops the box, the central tick must NOT write
    one more (extrapolated) value, mcu_stop already froze the exact final
    time. Guard: ``if not framework_running: continue`` after tick_active."""
    box = _AutoStopBox(ms=120_000)
    box._timer_label.setText("00:02:00")   # pretend mcu_stop froze it here
    win = _StubWindow([box])

    MainWindowBase._on_process_tick(win)

    assert box.ticked
    assert box._timer_label.text() == "00:02:00", \
        "frozen label must not be overwritten after auto-stop"


def test_process_tick_skips_box_without_pycboard():
    box = _FakeBox(ms=5000)
    box.pycboard = None                    # disconnected mid-run
    win = _StubWindow([box])

    MainWindowBase._on_process_tick(win)    # must not raise

    assert box._timer_label.text() == ""    # nothing written


# --------------------------------------------------------------------------
# 3. mcu_stop freeze
# --------------------------------------------------------------------------
class _RunTaskHarness(RunTask):
    """Concrete RunTask without Qt parentage (same pattern as the
    error-path test)."""

    def __init__(self) -> None:
        self.init_run_task(setup_id=1, main_window=None)
        self._status_widget = _FakeStatusWidget()
        self._timer_label = _FakeLabel()
        self.framework_running = True

    def _log_func(self, msg: str) -> None:
        pass


def test_mcu_stop_freezes_timer_at_final_fw_timestamp():
    """After the final drain, mcu_stop sets the label to the RAW final
    ``pycboard.timestamp`` so the frozen value equals the last TSV row."""
    rt = _RunTaskHarness()
    pyc = MagicMock()
    pyc.framework_running = False
    pyc.timestamp = 3_600_000              # exactly 1 hour
    rt.pycboard = pyc

    rt.mcu_stop("auto")

    assert rt.framework_running is False
    assert rt._timer_label.text() == "01:00:00", \
        f"timer must freeze at final fw time, got {rt._timer_label.text()!r}"
