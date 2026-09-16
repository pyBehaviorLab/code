"""Truly-parallel multi-box recording stop.

Drives N boxes' MCU-stop + persistent-var-capture work concurrently:

  Phase 1 (GUI thread, fast)
    For every selected box: flip ``framework_running`` off, re-sync
    the central tick mode, and swap ``pycboard.print`` with a
    thread-safe deque buffer so the serial-side logger doesn't try to
    touch Qt from a worker thread.

  Phase 2 (worker threads, TRUE PARALLEL)
    For every box, on its own ``QThreadPool`` worker, do the slow work:

      * ``pycboard.stop_framework()`` (serial roundtrip).
      * Drain final messages.
      * Read persistent variables off the MCU + merge-write
        ``<project>/<task>/persistent_variables.json``.

    These steps release the GIL during ``serial.read``, so N workers
    truly progress in parallel, wall-clock ≈ slowest single box rather
    than N × per-box.

  Phase 3 (GUI thread, fast)
    For every completed worker, on the GUI thread:

      * Restore ``pycboard.print`` and replay any buffered lines into
        the box's status widget.
      * Run subclass ``_after_stop`` hook (close data file, stop video
        recorder, notify task_plot / statistics, refresh main window).
      * Update status text + button states.

The START side uses a serial for-loop (UniversalStartDialog), per-box
start is fast enough that parallelism gains nothing.

If a box can't be prepared (no pycboard, not running), it is silently
skipped, the coordinator only dispatches workers for boxes that can
actually stop.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6 import QtCore

from source.log import get_logger

logger = get_logger()


class ThreadSafePrintBuffer:
    """Captures ``pycboard.print(text)`` calls from a worker thread and
    replays them onto the GUI-thread print callback later.

    Pycboard's ``print`` slot is typically wired to a Qt widget
    (QPlainTextEdit.append). Calling that from a worker thread races
    Qt's paint engine. This buffer is the safe substitute: same
    ``__call__(text)`` API as the original print callable, but writes
    go into an in-memory deque under a lock.
    """

    def __init__(self) -> None:
        self._buf: "collections.deque[str]" = collections.deque()
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs) -> None:
        # Mirror pycboard.print's signature; join multi-arg lines with " ".
        text = " ".join(str(a) for a in args)
        with self._lock:
            self._buf.append(text)

    def drain(self) -> List[str]:
        with self._lock:
            out = list(self._buf)
            self._buf.clear()
        return out


class _WorkerSignals(QtCore.QObject):
    """Cross-thread signal carrier. The signal MUST be on a QObject."""

    finished = QtCore.Signal(object, bool, str)  # widget, ok, error_msg


class _BoxActionRunnable(QtCore.QRunnable):
    """One worker per box. Calls the named worker method off the GUI
    thread; emits ``finished(widget, ok, error_msg)`` on completion
    (signal delivers on the GUI thread by default via
    QueuedConnection)."""

    def __init__(self, widget, worker_name: str, signals: _WorkerSignals) -> None:
        super().__init__()
        self._widget = widget
        self._worker_name = worker_name
        self._signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        ok = False
        err = ""
        t0 = time.monotonic()
        try:
            method = getattr(self._widget, self._worker_name, None)
            if method is None:
                err = f"widget has no {self._worker_name}()"
            else:
                ok = bool(method())
        except Exception as e:
            err = repr(e)
            logger.error(
                "Box %s %s raised: %s",
                getattr(self._widget, "setup_id", "?"),
                self._worker_name, e,
            )
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            try:
                self._signals.finished.emit(self._widget, ok, err)
            except Exception:
                pass
            logger.info(
                "ParallelStopCoord: box %s done in %.0f ms ok=%s",
                getattr(self._widget, "setup_id", "?"), elapsed_ms, ok,
            )


class ParallelStopCoordinator(QtCore.QObject):
    """Drive N boxes' record-stop work truly in parallel.

    Use::

        coord = ParallelStopCoordinator(widgets, on_finished=cb)
        coord.run()

    The dialog returns from the user-clicked Stop handler immediately;
    workers fire in the background; ``cb(ok_count, total)`` runs on the
    GUI thread after the last worker completes.
    """

    # Hard ceiling on concurrent serial-port operations (max rig = 8-16 boxes).
    MAX_CONCURRENT_WORKERS = 16

    _PREP_NAME   = "prepare_record_stop"
    _WORKER_NAME = "run_record_stop_worker"
    _FINAL_NAME  = "finalize_record_stop"

    def __init__(
        self,
        widgets: List[QtCore.QObject],
        *,
        on_progress: Optional[Callable[[int, int, QtCore.QObject, bool], None]] = None,
        on_finished: Optional[Callable[[int, int], None]] = None,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._widgets = list(widgets)
        self._on_progress = on_progress
        self._on_finished = on_finished
        self._signals = _WorkerSignals()
        self._signals.finished.connect(
            self._on_worker_finished,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        # Per-widget bookkeeping. Keyed by id(widget) (widget itself is
        # not hashable when it's a Qt widget without __eq__/__hash__).
        self._inflight: Dict[int, QtCore.QObject] = {}
        self._buffers: Dict[int, Tuple[Any, ThreadSafePrintBuffer]] = {}
        self._ok_count = 0
        self._done_count = 0
        # Set by run(); None means the pool count was never raised, so
        # _fire_finished (reachable via the empty-widgets early-out) has
        # nothing to restore.
        self._prev_max_threads: Optional[int] = None

    # ---- public API -----------------------------------------------------

    def run(self) -> None:
        """Snapshot widget state, swap pycboard.print, dispatch workers."""
        if not self._widgets:
            self._fire_finished()
            return

        pool = QtCore.QThreadPool.globalInstance()
        # Raise the pool thread count so all boxes run concurrently; restored
        # in _fire_finished.
        self._prev_max_threads = pool.maxThreadCount()
        pool.setMaxThreadCount(max(self._prev_max_threads, self.MAX_CONCURRENT_WORKERS))

        for widget in self._widgets:
            if not self._prepare_widget_on_gui_thread(widget):
                # Skipped boxes still count as finished (ok=False) so the
                # batch progresses.
                self._record_skipped(widget)
                continue
            self._inflight[id(widget)] = widget
            runnable = _BoxActionRunnable(
                widget, self._WORKER_NAME, self._signals)
            pool.start(runnable)

        # If every widget was skipped, fire finished immediately.
        if not self._inflight:
            self._fire_finished()

    # ---- phase 1: snapshot + swap pycboard.print -----------------------

    def _prepare_widget_on_gui_thread(self, widget) -> bool:
        """Return True if the worker should run for this widget.

        Calls ``prepare_record_stop`` and swaps the pycboard's print
        callback with a thread-safe buffer so the worker can run safely
        off the GUI thread.
        """
        bid = getattr(widget, "setup_id", "?")
        t0 = time.monotonic()
        try:
            prep = getattr(widget, self._PREP_NAME, None)
            if prep is None:
                return False
            ready = bool(prep())
            if not ready:
                return False
        except Exception as e:
            logger.error("Box %s: %s raised: %s",
                         bid, self._PREP_NAME, e)
            return False
        finally:
            logger.info(
                "ParallelStopCoord: box %s prepare=%.0fms",
                bid, (time.monotonic() - t0) * 1000.0,
            )

        pyc = getattr(widget, "pycboard", None)
        if pyc is not None:
            saved_print = getattr(pyc, "print", None)
            buf = ThreadSafePrintBuffer()
            try:
                pyc.print = buf
            except Exception:
                # print not assignable, skip off-thread mode for this box.
                return False
            self._buffers[id(widget)] = (saved_print, buf)
        return True

    # ---- phase 3: finalize on GUI thread -------------------------------

    def _on_worker_finished(self, widget, ok: bool, err: str) -> None:
        """Slot, runs on the GUI thread (Qt::QueuedConnection)."""
        bid = getattr(widget, "setup_id", "?")
        # Restore pycboard.print + replay buffered lines.
        saved_and_buf = self._buffers.pop(id(widget), None)
        if saved_and_buf is not None:
            saved_print, buf = saved_and_buf
            pyc = getattr(widget, "pycboard", None)
            try:
                lines = buf.drain()
            except Exception:
                lines = []
            if pyc is not None and saved_print is not None:
                try:
                    pyc.print = saved_print
                except Exception:
                    pass
                for line in lines:
                    try:
                        saved_print(line)
                    except Exception:
                        pass

        # Subclass finaliser.
        t0 = time.monotonic()
        try:
            fin = getattr(widget, self._FINAL_NAME, None)
            if fin is not None:
                fin(ok=ok, error=err)
        except Exception as e:
            logger.error("Box %s: %s raised: %s",
                         bid, self._FINAL_NAME, e)
        finally:
            logger.info(
                "ParallelStopCoord: box %s finalize=%.0fms ok=%s",
                bid, (time.monotonic() - t0) * 1000.0, ok,
            )

        # Bookkeeping.
        self._inflight.pop(id(widget), None)
        self._done_count += 1
        if ok:
            self._ok_count += 1
        if self._on_progress is not None:
            try:
                self._on_progress(self._done_count,
                                  len(self._widgets), widget, ok)
            except Exception:
                pass
        if not self._inflight:
            self._fire_finished()

    def _record_skipped(self, widget) -> None:
        """Widget didn't pass prepare_record_stop, count it done."""
        self._done_count += 1
        if self._on_progress is not None:
            try:
                self._on_progress(self._done_count,
                                  len(self._widgets), widget, False)
            except Exception:
                pass

    def _fire_finished(self) -> None:
        if self._prev_max_threads is not None:
            try:
                pool = QtCore.QThreadPool.globalInstance()
                pool.setMaxThreadCount(self._prev_max_threads)
            except Exception:
                pass
        if self._on_finished is not None:
            try:
                self._on_finished(self._ok_count, len(self._widgets))
            except Exception:
                pass
