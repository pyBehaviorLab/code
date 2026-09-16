"""Application-level Python logging configuration.

Qt-free, importable from any non-GUI layer without dragging PySide6
into the import graph. The Qt-aware widget handler
lives in source.gui.logging_handler and is wired up at runtime by
``initialize_logger(log_widget=...)`` via a deferred import.

Public API:
    set_app_name(name), entry-point sets this BEFORE other imports.
    set_log_directory(path), open the per-GUI log file in ``path``.
    get_logger(name=None), module-level logger factory.
    initialize_logger(log_widget=None),
                                  wire up GUI handler (lazy-imports Qt).
    set_debug_mode(enabled), toggle DEBUG level globally.

On-disk layout
--------------
ONE log file per GUI launch:

    <data_dir>/log/<APP_NAME>_<YYYY-MM-DD>_pid<PID>.log

Created by ``set_log_directory`` (called once at app startup by
``MainWindowBase._init_default_data_dir``). Every log line for the
lifetime of that GUI process lands in this single file. No per-session
log handlers, sessions are short-lived; the per-GUI log captures
everything the operator did from launch through quit, including the
context around each recording.
"""

import logging
import os
from pathlib import Path
from datetime import datetime


LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


class _PathScrubFormatter(logging.Formatter):
    """Formatter that strips absolute filesystem paths out of every log line
    so logs never leak ``D:/Ulm_Data/.../code/...`` (→ relative) or the home
    dir ``C:/Users/<name>`` (→ ``~``). Applied to all handlers, so individual
    ``logger.info(f"...{abs_path}...")`` call sites need no changes."""

    _subs = None  # lazily-built list of (needle, replacement), longest first

    @classmethod
    def _build_subs(cls):
        subs = []

        def _variants(base):
            b = str(base).rstrip("/\\")
            return {b, b.replace("/", os.sep), b.replace(os.sep, "/")}

        try:
            from source import paths as _paths
            for v in _variants(_paths.top_dir):
                # top_dir prefix → "" so "<top>/data/X" reads as "data/X".
                subs.append((v + os.sep, ""))
                subs.append((v + "/", ""))
                subs.append((v, ""))
        except Exception:
            pass
        try:
            for v in _variants(Path.home()):
                subs.append((v, "~"))
        except Exception:
            pass
        # Longest needle first so the more specific prefix wins.
        subs.sort(key=lambda t: len(t[0]), reverse=True)
        return subs

    def format(self, record):
        s = super().format(record)
        if _PathScrubFormatter._subs is None:
            _PathScrubFormatter._subs = self._build_subs()
        for needle, repl in _PathScrubFormatter._subs:
            if needle and needle in s:
                s = s.replace(needle, repl)
        return s

_file_handler = None  # set by set_log_directory()

# Reference to the Qt GUI handler (if attached). Held here so we can swap it
# when initialize_logger() is called more than once.
_gui_handler = None

# App name used when get_logger() is called with no argument and as the
# log-file prefix. Entry points (pyOperant.py / pyMaze.py) call
# set_app_name() *before* importing anything from source/ so every logger
# created downstream picks up the right name.
_APP_NAME = 'pyBehaviorLab'


def set_app_name(name: str) -> None:
    """Set the application name used by ``get_logger()`` and the log filename.

    Should be called as early as possible by the entry-point script
    (pyOperant: 'pyBLOperant', pyMaze: 'pyBLmaze')."""
    global _APP_NAME
    if name:
        _APP_NAME = str(name)


def set_log_directory(log_dir):
    """Open the per-GUI log file in ``log_dir``.

    Filename includes the PID so concurrent GUI instances (e.g. two
    pyOperant launches on one workstation) each get their own file
    instead of interleaving lines. Date prefix sorts lexicographically.

    Idempotent across calls, if a file handler is already attached it
    is closed and replaced.
    """
    global _file_handler
    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        new_log_file = log_dir / (
            f'{_APP_NAME}_{datetime.now().strftime("%Y-%m-%d")}'
            f'_pid{os.getpid()}.log'
        )

        root_logger = logging.getLogger()
        if _file_handler:
            root_logger.removeHandler(_file_handler)
            try:
                _file_handler.close()
            except Exception:
                pass

        _file_handler = logging.FileHandler(new_log_file, encoding='utf-8')
        formatter = _PathScrubFormatter(LOG_FORMAT, DATE_FORMAT)
        _file_handler.setFormatter(formatter)
        # File follows the active level (INFO by default; set
        # PYBEHAVIORLAB_LOGLEVEL=DEBUG for the full firehose when diagnosing).
        _file_handler.setLevel(_display_level)
        root_logger.addHandler(_file_handler)

        root_logger.info(f"Per-GUI log file: {new_log_file}")
    except Exception as e:
        logging.getLogger().error(f"Failed to set log directory: {e}")


# Env var that sets the active log level on the rig without a code edit
# (e.g. PYBEHAVIORLAB_LOGLEVEL=DEBUG). Default INFO: per-frame
# ``logger.debug(...)`` in hot paths short-circuits at the ROOT level
# (no format, no disk write); DEBUG turns the full firehose back on.
LEVEL_ENV = "PYBEHAVIORLAB_LOGLEVEL"
_display_level = logging.INFO   # the ONE active level: root + file + GUI all follow it

# The ROOT level gates record creation, so it (not just per-handler levels)
# follows the active level for debug() to be free when off. NO console/stdout
# handler, GUI app. Handlers are added by set_log_directory() (file) /
# initialize_logger() (the in-app log widget).
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[]
)


def _level_from_name(name, default=logging.INFO):
    """Map a level NAME ('DEBUG'/'info'/…) to its logging int; default if bad."""
    lvl = getattr(logging, str(name).strip().upper(), None)
    return lvl if isinstance(lvl, int) else default


def _apply_level(level):
    """Apply ``level`` to the root logger AND both handlers. The ROOT level
    makes ``debug()`` free when off: it gates record creation, so a per-frame
    ``logger.debug(...)`` returns immediately whenever the active level is
    above DEBUG."""
    global _display_level
    _display_level = level
    logging.getLogger().setLevel(level)
    if _file_handler is not None:
        _file_handler.setLevel(level)
    if _gui_handler is not None:
        _gui_handler.setLevel(level)


def configure_logging(display_level=None):
    """Set the ONE active log level (root + file + GUI). NO console handler.

    Level = ``display_level`` if given, else ``PYBEHAVIORLAB_LOGLEVEL``, else
    INFO. INFO default = only important lines and ``debug()`` is free. Idempotent,
    called once at startup by ``initialise_app_logger``."""
    if display_level is None:
        env = os.environ.get(LEVEL_ENV, "").strip()
        display_level = _level_from_name(env) if env else logging.INFO
    _apply_level(display_level)


def set_log_level(level):
    """Set the active log level at runtime (root + file + GUI). Accepts a logging
    int or a name ('DEBUG'). Above DEBUG, ``debug()`` short-circuits for free."""
    _apply_level(level if isinstance(level, int) else _level_from_name(level))


def get_logger(name=None):
    """Get a logger instance. Lazy-resolves the app name so loggers created
    after ``set_app_name`` pick up the new label."""
    if name is None:
        name = _APP_NAME
    return logging.getLogger(name)


def initialize_logger(log_widget=None, log_to_file=True):
    """Initialize the logging system, optionally wiring a Qt log widget.

    Lazy-imports source.gui.logging_handler so that runtime/non-Qt callers
    can still import this module without pulling PySide6.

    ``log_to_file`` is unused; file logging is opened by
    ``set_log_directory``.
    """
    global _gui_handler

    root_logger = logging.getLogger()

    if log_widget is not None:
        # Defer the Qt import to here so this module remains Qt-free at
        # import time.
        from source.gui.logging_handler import QTextBrowserHandler

        if _gui_handler is not None:
            root_logger.removeHandler(_gui_handler)

        _gui_handler = QTextBrowserHandler(log_widget)
        # Follow the current display level (env / set_log_level), not a
        # hardcoded INFO, so DEBUG mode actually surfaces in the widget.
        _gui_handler.setLevel(_display_level)
        formatter = _PathScrubFormatter(LOG_FORMAT, DATE_FORMAT)
        _gui_handler.setFormatter(formatter)
        root_logger.addHandler(_gui_handler)


def set_debug_mode(enabled):
    """Toggle DEBUG. On → full firehose (root + file + GUI at DEBUG). Off → INFO,
    so only important lines run and per-frame ``debug()`` is free."""
    set_log_level(logging.DEBUG if enabled else logging.INFO)


# Default logger so ``from .log import logger`` patterns work.
logger = get_logger()


__all__ = [
    "set_app_name",
    "set_log_directory",
    "get_logger", "initialize_logger",
    "configure_logging", "set_log_level", "set_debug_mode",
    "LEVEL_ENV", "LOG_FORMAT", "DATE_FORMAT",
    "logger",
]
