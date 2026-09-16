"""Main window for the standalone offline analyzer.

Layout (matches the main rig GUI's sidebar pattern exactly):

    ┌──────────────────────────────────────────────────────────────────┐
    │ ┌─┐  Top selector  [Folder] [Files] [Project]  chip          [Clear]│
    │ │L│  ────────────────────────────────────────────────────────────│
    │ │o│                                                                │
    │ │g│  …the Analyze view…                                            │
    │ └─┘                                                                │
    └──────────────────────────────────────────────────────────────────┘
       ▲ floating RotatedButton "Log" (22×150 px, top-left of canvas)

Clicking the rotated Log button **slides a CollapsibleSidebar** out
from the left edge (300 ms cubic ease-out); the button hides while
the sidebar is expanded and reappears when the user clicks outside
or hits the close ◀ button. Exact same pattern as the rig GUI.

One view, not a tab bar: this analyser analyses recordings, and a tab
bar with a single tab is a control that cannot be operated.

No File / Help menubar. No QDockWidget.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .session_selector_panel import SessionSelectorPanel
from .sidebar_log import CollapsibleSidebar, LogPanel, RotatedButton


logger = logging.getLogger(__name__)


class OfflineAnalyzerMainWindow(QtWidgets.QMainWindow):
    """The Analyze view + slide-out Log sidebar (matches rig GUI)."""

    # Floating button position, matches the rig GUI's first sidebar
    # button anchor (source/gui/operant.py:780, y_position=50).
    _LOG_BTN_Y = 50
    _LOG_BTN_W = 22
    _LOG_BTN_H = 150
    _LOG_SIDEBAR_W = 380

    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyBehaviorLab, Offline Analyzer")
        self._size_to_screen()

        # ── Central widget = top selector + the view (no left strip) ─
        # Left margin reserves a 26 px column for the floating
        # RotatedButton (22 px + 4 px gap) so it never overlaps the
        # top selector or the view. Same trick as the rig GUI
        # (source/gui/operant.py:277 ``setContentsMargins(26, 4, 4, 0)``).
        central = QtWidgets.QWidget(self)
        central.setObjectName("mwCentral")
        cl = QtWidgets.QVBoxLayout(central)
        cl.setContentsMargins(26, 4, 4, 0)
        cl.setSpacing(0)

        self.selector = SessionSelectorPanel(self)
        self.selector.projectLoaded.connect(self._on_project_loaded)
        self.selector.sessionsPushed.connect(self._on_sessions_pushed)
        self.selector.videosPicked.connect(self._on_videos_picked)
        cl.addWidget(self.selector)

        # The view IS pyBehaveTrack's Analyze tab, ported whole and 2D
        # only. Not a reimplementation of it: same columns, same controls, same
        # plan strip, reading this rig's recordings through the ported engine.
        from .analyze.analyze_tab import AnalyzeTab
        self.vt_view = AnalyzeTab(self)
        cl.addWidget(self.vt_view, 1)
        # The view has no log of its own; there is one Log in this window and
        # it is the sidebar. When what the view has to say IS the log, it asks
        # for it rather than growing a second one.
        self.vt_view.logRequested.connect(self._show_log)
        self.setCentralWidget(central)
        self._central = central

        # ── Floating RotatedButton + slide-out CollapsibleSidebar ───
        # Button is parented to the central widget so it overlays the
        # tabs at a fixed (x=0, y=8), same as main GUI's pattern.
        self.log_btn = RotatedButton("LOG", parent=central)
        self.log_btn.setFixedSize(self._LOG_BTN_W, self._LOG_BTN_H)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.log_btn.setFont(font)
        self.log_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.log_btn.setToolTip("Show Log")
        self.log_btn.move(0, self._LOG_BTN_Y)
        self.log_btn.raise_()
        self.log_btn.clicked.connect(self._toggle_log)

        # Sidebar is also parented to the central widget so it overlays
        # the canvas (rather than pushing content sideways).
        self.log_sidebar = CollapsibleSidebar(
            parent=central, header_text="LOG",
            expanded_width=self._LOG_SIDEBAR_W)
        self.log_panel = LogPanel(self.log_sidebar)
        self.log_sidebar.content_layout.addWidget(self.log_panel)
        self.log_sidebar.on_collapse_callback = self._on_sidebar_collapsed
        self.log_sidebar.setFixedHeight(central.height())
        self.log_sidebar.move(0, 0)
        self.log_sidebar.hide()       # collapsed_width=0 + hidden

        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _size_to_screen(self):
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            self.resize(1400, 900)
            return
        a = screen.availableGeometry()
        w, h = int(a.width() * 0.9), int(a.height() * 0.9)
        self.resize(w, h)
        self.move(a.x() + (a.width() - w) // 2,
                  a.y() + (a.height() - h) // 2)

    def _default_data_dir(self) -> str:
        try:
            project_root = Path(__file__).resolve().parents[2]
            data_dir = project_root / "data"
            if data_dir.exists():
                return str(data_dir)
        except Exception:
            pass
        import os
        return os.getcwd()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # Keep the slide-out sidebar full-height as the window resizes.
        try:
            self.log_sidebar.setFixedHeight(self._central.height())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Log toggle (matches main GUI _toggle_sidebar pattern)
    # ------------------------------------------------------------------

    def _show_log(self):
        """Slide the Log out, or leave it out if it already is."""
        if self.log_sidebar.is_collapsed:
            self._toggle_log()

    def _toggle_log(self):
        if self.log_sidebar.is_collapsed:
            self.log_sidebar.show()
            self.log_sidebar.expand()
            self.log_btn.hide()
        else:
            self.log_sidebar.collapse()
            self.log_btn.show()

    def _on_sidebar_collapsed(self):
        """Sidebar closed via close button or outside-click, show the
        floating Log button again."""
        self.log_btn.show()
        self.log_btn.raise_()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_project_loaded(self, ctx):
        try:
            n = len(ctx.sessions)
            self.statusBar().showMessage(
                f"Project '{ctx.project_name}': {n} sessions", 5000)
            self.log_panel.set_project_dir(ctx.project_dir)
        except Exception as e:
            logger.warning("project load handler failed: %s", e)
        if hasattr(self.vt_view, "set_project_context"):
            try:
                self.vt_view.set_project_context(ctx)
            except Exception as e:
                logger.warning("set_project_context failed: %s", e)

    def _on_videos_picked(self, paths):
        """Bare videos, nothing tracked yet, often no data file at all."""
        if not paths:
            return
        try:
            self.vt_view.add_videos(list(paths))
        except Exception as e:
            logger.error("adding videos failed: %s", e)
            return
        self.statusBar().showMessage(f"{len(paths)} video(s) added", 5000)

    def _on_sessions_pushed(self, sessions):
        if hasattr(self.vt_view, "set_top_sessions"):
            try:
                self.vt_view.set_top_sessions(sessions)
            except Exception as e:
                logger.error("set_top_sessions failed: %s", e)
        self.statusBar().showMessage(
            f"{len(sessions)} session(s) loaded", 5000)
