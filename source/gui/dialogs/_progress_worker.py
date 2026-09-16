"""Shared worker harness for multi-box pycboard actions.

Runs a per-box pycboard call concurrently on a QThreadPool, swapping each
pycboard's ``print`` for a signal-emit proxy so Qt widget writes stay on the
main thread, fanning results back through signals, advancing a progress bar,
and auto-closing when the last worker reports.

Callers supply only what's unique: ``worker_body`` (the per-box pycboard
call), ``on_success`` and ``on_failure`` (main-thread UI hooks), and
optionally ``on_progress`` (each progress emit from the proxy).

Public surface:
    BoxWorkerSignals, progress + result, payload is a plain dict
    BoxWorker, QRunnable that calls worker_body and emits result
    run_box_actions_in_parallel(...),
                        orchestrator; returns once every worker reports
"""
from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Optional

from PySide6 import QtCore, QtWidgets

from source.log import get_logger

logger = get_logger()


def _int_env(name: str, default: int, *, min_value: int, max_value: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        val = default
    else:
        try:
            val = int(str(raw).strip())
        except Exception:
            val = default
    return max(min_value, min(max_value, val))


class BoxWorkerSignals(QtCore.QObject):
    """Signals for background per-box work.

    Payload is a plain dict to avoid fragile overload signatures.
    ``progress`` carries ``{"box", "msg", "end"}``; ``result`` carries
    ``{"box", "ok", "err"}``.
    """

    progress = QtCore.Signal(object)
    result   = QtCore.Signal(object)


class BoxWorker(QtCore.QRunnable):
    """Single worker: call ``fn(box_widget, signals)`` and emit result.

    ``PyboardError`` subclasses ``BaseException``, so we catch BaseException
    (re-raising KeyboardInterrupt / SystemExit).
    """

    def __init__(self, setup_widget, fn: Callable[[Any, BoxWorkerSignals], None],
                 *, signals: BoxWorkerSignals):
        super().__init__()
        self.setup_widget = setup_widget
        self.fn = fn
        self.signals = signals

    def run(self) -> None:
        setup_number = getattr(self.setup_widget, "setup_number", None)
        try:
            self.fn(self.setup_widget, self.signals)
            self.signals.result.emit({"box": setup_number, "ok": True, "err": ""})
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            self.signals.result.emit(
                {"box": setup_number, "ok": False, "err": str(e)})


def build_progress_dialog(parent, title, total, *, header=None,
                          with_log=False, with_detail=False,
                          width=260, log_min_height=160,
                          show=True):
    """One modal progress popup for every batch flow (MCU connect /
    config actions / task upload / camera calibration), each used to
    hand-roll its own. Returns the QDialog with ``bar`` and, when
    requested, ``detail`` (single status QLabel) / ``log_view``
    (scrolling QTextEdit) attached as attributes."""
    from PySide6 import QtWidgets

    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setModal(True)
    dlg.setFixedWidth(width)
    lay = QtWidgets.QVBoxLayout(dlg)
    lay.setContentsMargins(12, 12, 12, 12)
    lay.setSpacing(8)

    head = QtWidgets.QLabel(header or title)
    head.setStyleSheet("font-weight: bold; color: #e5e7eb;")
    lay.addWidget(head)

    dlg.detail = None
    if with_detail:
        dlg.detail = QtWidgets.QLabel("")
        dlg.detail.setStyleSheet("color: #e5e7eb;")
        lay.addWidget(dlg.detail)

    dlg.log_view = None
    if with_log:
        dlg.log_view = QtWidgets.QTextEdit()
        dlg.log_view.setReadOnly(True)
        dlg.log_view.setMinimumHeight(log_min_height)
        dlg.log_view.setStyleSheet(
            "QTextEdit {"
            " background-color: #111; color: #e0e0e0;"
            " border: 1px solid #333; border-radius: 4px;"
            " font-family: Consolas, monospace; font-size: 9pt;"
            "}"
        )
        lay.addWidget(dlg.log_view)

    dlg.bar = QtWidgets.QProgressBar()
    dlg.bar.setRange(0, total)
    dlg.bar.setValue(0)
    dlg.bar.setStyleSheet("QProgressBar { background: #111; color: #e5e7eb; }")
    lay.addWidget(dlg.bar)

    dlg.setStyleSheet("QDialog { background-color: #0f172a; color: #e5e7eb; }")
    if show:
        dlg.show()
        QtWidgets.QApplication.processEvents()
    return dlg


def _default_route_progress(setup_widget, msg: str, end: str) -> None:
    """Default progress sink: append to the box widget's live status log.

    Tries ``append_status`` first (maze SetupWidget), falls back to
    ``print_to_log`` (operant BoxControlWidget via LiveStatusWidget).
    Tolerates both ``fn(msg, end)`` and ``fn(msg)`` signatures.
    """
    fn = getattr(setup_widget, "append_status", None) or getattr(
        setup_widget, "print_to_log", None)
    if not callable(fn):
        return
    try:
        fn(str(msg), end=end)
    except TypeError:
        # Signature without end=.
        try:
            fn(str(msg))
        except Exception:
            pass
    except Exception:
        pass


def wrap_with_print_proxy(
    worker_body: Callable[[Any, BoxWorkerSignals], None],
) -> Callable[[Any, BoxWorkerSignals], None]:
    """Wrap a worker body so pycboard.print emits progress signals.

    pycboard's ``self.print`` writes Qt widgets directly, which worker threads
    must not touch. This swaps ``pycboard.print`` for a signal-emit proxy for
    the duration of the body, then restores the original. Raises
    ``RuntimeError`` if the box has no pycboard connection.
    """

    def _wrapped(setup_widget, signals: BoxWorkerSignals) -> None:
        if not hasattr(setup_widget, "pycboard") or not setup_widget.pycboard:
            raise RuntimeError("No pycboard connection")
        pyc = setup_widget.pycboard
        setup_number = getattr(setup_widget, "setup_number", None)
        original_print = pyc.print

        def _print_proxy(text, end="\n"):
            try:
                signals.progress.emit(
                    {"box": setup_number, "msg": str(text), "end": end})
            except Exception:
                pass

        pyc.print = _print_proxy
        try:
            worker_body(setup_widget, signals)
        finally:
            pyc.print = original_print

    return _wrapped


def run_box_actions_in_parallel(
    *,
    setup_widgets: Iterable[Any],
    worker_body: Callable[[Any, BoxWorkerSignals], None],
    on_success: Callable[[Any], None],
    on_failure: Callable[[Any, str], None],
    progress_dialog: QtWidgets.QDialog,
    progress_bar: QtWidgets.QProgressBar,
    detail_label: Optional[QtWidgets.QLabel] = None,
    on_progress: Optional[Callable[[Any, str, str], None]] = None,
    on_queued: Optional[Callable[[Any], None]] = None,
    wrap_print: bool = True,
    max_workers_env: str = "PYBEHAVIORLAB_MCU_PARALLELISM",
    default_max_workers: int = 4,
) -> None:
    """Run ``worker_body`` for each box concurrently on the global QThreadPool.

    Per-box success / failure dispatch and progress bar advancement
    happen on the main thread via signals. The function blocks on
    ``progress_dialog.exec()`` and returns once every worker has reported
    a result (the dialog auto-accepts after a 400 ms grace tick).

    Parameters
    ----------
    box_widgets       : list of per-box widgets to run against.
    worker_body       : ``(box_widget, signals) -> None``. Raises on failure.
    on_success        : main-thread callback; ``on_success(box_widget)``.
    on_failure        : main-thread callback; ``on_failure(box_widget, err_text)``.
    progress_dialog   : the QDialog whose .exec() will block the call.
    progress_bar      : QProgressBar to tick once per result.
    detail_label      : optional QLabel; gets "Box N" updates per result.
    on_progress       : optional ``(box_widget, msg, end) -> None``; defaults
                        to routing the message to the box's status log.
    on_queued         : optional ``(box_widget) -> None`` for "queued" UI hint.
    wrap_print        : if True (default), wrap worker_body with the pycboard.print
                        signal-proxy so worker-side prints surface as progress.
    max_workers_env   : env var name overriding the pool size cap.
    default_max_workers : default cap if the env var is unset.
    """
    setup_widgets = list(setup_widgets)
    if not setup_widgets:
        return

    pool = QtCore.QThreadPool.globalInstance()
    pool.setMaxThreadCount(
        _int_env(max_workers_env, default_max_workers,
                 min_value=1, max_value=16))

    widgets_by_box = {
        getattr(w, "setup_number", None): w for w in setup_widgets
    }
    total = len(setup_widgets)
    remaining = {"n": total}

    if on_progress is None:
        on_progress = _default_route_progress

    def _route_progress(payload: object) -> None:
        try:
            box = (payload or {}).get("box", None)
            msg = (payload or {}).get("msg", "")
            end = (payload or {}).get("end", "\n")
            if not msg:
                return
            bw = widgets_by_box.get(box)
            if bw is None:
                return
            try:
                on_progress(bw, str(msg), end)
            except Exception as e:
                logger.debug("on_progress error for Box %s: %s", box, e)
        except Exception as e:
            logger.debug("progress payload error: %s", e)

    def _route_result(payload: object) -> None:
        box = (payload or {}).get("box", None)
        ok  = bool((payload or {}).get("ok", False))
        err = str((payload or {}).get("err", "") or "")
        bw = widgets_by_box.get(box)

        try:
            if ok and bw is not None:
                on_success(bw)
            elif bw is not None:
                on_failure(bw, err)
        except Exception as e:
            logger.error("post-result hook error for Box %s: %s", box, e)

        if detail_label is not None:
            try:
                detail_label.setText(f"Box {box}")
            except Exception:
                pass
        try:
            progress_bar.setValue(progress_bar.value() + 1)
        except Exception:
            pass
        QtWidgets.QApplication.processEvents()

        remaining["n"] -= 1
        if remaining["n"] <= 0:
            try:
                progress_bar.setValue(progress_bar.maximum())
            except Exception:
                pass
            QtWidgets.QApplication.processEvents()
            QtCore.QTimer.singleShot(400, progress_dialog.accept)

    body = wrap_with_print_proxy(worker_body) if wrap_print else worker_body
    # Keep every BoxWorkerSignals alive until this call returns. Each signals
    # object is otherwise owned only by its autoDelete QRunnable worker; the
    # pool frees the worker (and the signals QObject) the instant run()
    # returns, which can free the QObject while a queued emit is still pending
    # on the main thread → use-after-free. This list holds the references.
    signal_refs = []
    for setup_widget in setup_widgets:
        if on_queued is not None:
            try:
                on_queued(setup_widget)
            except Exception:
                pass
        signals = BoxWorkerSignals()
        signal_refs.append(signals)
        signals.progress.connect(_route_progress)
        signals.result.connect(_route_result)
        worker = BoxWorker(setup_widget, body, signals=signals)
        pool.start(worker)

    progress_dialog.exec()
