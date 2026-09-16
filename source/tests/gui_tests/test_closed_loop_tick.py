"""The realtime tick period IS the delay on every queued host→MCU write.

Queued events (coords, triggers, the probe's per-frame ``frame_in``) leave on
``process_timer``'s tick, so its period is a uniform 0–period delay on every
one of them, measured at 16 ms of ping/pong round trip with a 10 ms tick.
A box that is RUNNING is in a closed loop and gets the short tick; a camera
streaming on its own only needs the frame pull and keeps the cheap one.
"""
from types import SimpleNamespace

import pytest
from PySide6 import QtCore

from source.gui.base import MainWindowBase


class _Timer:
    """Stands in for the QTimer: records what interval it was started with."""

    def __init__(self):
        self.started_with = None
        self.stopped = False

    def start(self, ms):
        self.started_with = ms
        self.stopped = False

    def stop(self):
        self.stopped = True

    def interval(self):
        return self.started_with or 0

    def isActive(self):
        return self.started_with is not None and not self.stopped


def _window(running, streaming):
    w = MainWindowBase.__new__(MainWindowBase)
    w.process_timer = _Timer()
    w.display_timer = _Timer()
    w._timer_mode = None
    w._display_mode = None
    w._any_box_running = lambda: running
    w._any_camera_streaming = lambda: streaming
    w._display_interval_ms = lambda: 33
    return w


@pytest.mark.parametrize("running,streaming,expect", [
    (True, True, MainWindowBase.CLOSED_LOOP_INTERVAL_MS),
    (True, False, MainWindowBase.CLOSED_LOOP_INTERVAL_MS),
    (False, True, MainWindowBase.PROCESS_INTERVAL_MS),
])
def test_a_running_box_gets_the_short_tick(running, streaming, expect):
    w = _window(running, streaming)
    w._sync_timer_mode()
    assert w.process_timer.started_with == expect


def test_the_short_tick_is_actually_shorter():
    assert (MainWindowBase.CLOSED_LOOP_INTERVAL_MS
            < MainWindowBase.PROCESS_INTERVAL_MS), (
        "the closed-loop tick must beat the default or it buys nothing")


def test_the_tick_switches_when_a_box_starts_running():
    """The interval has to be re-applied on transition, not only on the
    idle→active edge, the timer is already active when a box starts."""
    w = _window(running=False, streaming=True)
    w._sync_timer_mode()
    assert w.process_timer.started_with == MainWindowBase.PROCESS_INTERVAL_MS
    w._any_box_running = lambda: True
    w._sync_timer_mode()
    assert w.process_timer.started_with == MainWindowBase.CLOSED_LOOP_INTERVAL_MS


def test_nothing_to_do_stops_the_tick():
    w = _window(running=False, streaming=False)
    w._sync_timer_mode()
    assert w.process_timer.stopped
