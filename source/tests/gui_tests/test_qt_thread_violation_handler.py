"""The Qt message handler that names a cross-thread violation.

Qt reports "Timers cannot be started from another thread" on stderr and
carries on; the process can then die later with no Python traceback, which is
what a Session Plot crash looked like on 2026-09-09. The handler exists to
turn that unactionable warning into a file and a line, so it has to be proven
to capture the stack of the thread that provoked it, not merely to log.
"""
from __future__ import annotations

import logging
import threading

import pytest
from PySide6 import QtCore, QtWidgets

from source.app_bootstrap import install_qt_message_handler


@pytest.fixture
def handler_installed():
    previous = QtCore.qInstallMessageHandler(None)
    install_qt_message_handler()
    try:
        yield
    finally:
        QtCore.qInstallMessageHandler(previous)


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_a_thread_violation_is_logged_with_a_stack(handler_installed, caplog):
    with caplog.at_level(logging.ERROR, logger="qt"):
        QtCore.qWarning("QBasicTimer::start: Timers cannot be started "
                        "from another thread")
    rec = [r for r in caplog.records if r.name == "qt"]
    assert rec, "the violation was not logged at all"
    text = rec[0].getMessage()
    assert "THREAD VIOLATION" in text
    # The whole point: the offending call site, not just Qt's sentence.
    assert "test_qt_thread_violation_handler.py" in text, (
        "no Python stack captured, so the warning still names no call site")


def test_the_stack_names_the_thread_that_provoked_it(handler_installed,
                                                     caplog):
    """The handler runs on the offending thread, which is what makes the
    captured stack the useful one."""
    done = threading.Event()

    def worker():
        QtCore.qWarning("QObject::startTimer: Timers cannot be started "
                        "from another thread")
        done.set()

    t = threading.Thread(target=worker, name="pose-worker-1")
    with caplog.at_level(logging.ERROR, logger="qt"):
        t.start()
        assert done.wait(5), "worker never ran"
        t.join(5)

    text = "\n".join(r.getMessage() for r in caplog.records if r.name == "qt")
    assert "pose-worker-1" in text, (
        "the log must name the thread, or a four-box run cannot be told apart")


def test_an_ordinary_qt_warning_is_logged_without_a_stack(handler_installed,
                                                          caplog):
    """Only violations pay for a formatted stack; everything else stays a
    one-liner, or a noisy rig log buries the finding."""
    with caplog.at_level(logging.DEBUG, logger="qt"):
        QtCore.qWarning("QPainter::begin: Paint device returned engine == 0")
    rec = [r for r in caplog.records if r.name == "qt"]
    assert rec
    assert "THREAD VIOLATION" not in rec[0].getMessage()
    assert "Traceback" not in rec[0].getMessage()


@pytest.mark.parametrize("message", [
    "QBasicTimer::start: Timers cannot be started from another thread",
    "QObject::killTimer: Timers cannot be stopped from another thread",
    "QObject: Cannot create children for a parent that is in a different thread",
    "QCoreApplication::sendEvent: Cannot send events to objects owned by a "
    "different thread",
])
def test_every_marker_is_recognised(handler_installed, caplog, message):
    """One spelling per Qt call, and all of them mean the same fault."""
    with caplog.at_level(logging.ERROR, logger="qt"):
        QtCore.qWarning(message)
    text = "\n".join(r.getMessage() for r in caplog.records if r.name == "qt")
    assert "THREAD VIOLATION" in text, f"not recognised: {message}"


def test_installing_does_not_swallow_later_handlers(qapp):
    """The suite installs and removes handlers; ours must restore cleanly."""
    seen = []
    install_qt_message_handler()
    previous = QtCore.qInstallMessageHandler(
        lambda _m, _c, msg: seen.append(msg))
    try:
        QtCore.qWarning("after ours")
        assert seen == ["after ours"]
    finally:
        QtCore.qInstallMessageHandler(previous)
