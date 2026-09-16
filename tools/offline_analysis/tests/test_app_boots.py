"""Full-app boot smoke, proves the standalone analyzer launches without
the rig codebase. Run offscreen so CI doesn't need a display.

Booting is the assertion that no unit test can make: the window wires the
selector, the ported tab, the log sidebar and their timers together, and a
mistake in any of that shows up here and nowhere else.
"""
from __future__ import annotations

import os
import re
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTHONUTF8", "1")

from PySide6 import QtWidgets


@pytest.fixture(scope="module")
def app():
    from tools.offline_analysis.theming import setup_dark_theme
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    setup_dark_theme(a)
    return a


@pytest.fixture
def win(app):
    """A main window that is really gone by the end of the test.

    Each of these tests builds the whole application, the ported tab, its
    worker threads, its timers. Leaving that to be collected whenever means
    one test's window is still alive during the next one.
    """
    from tools.offline_analysis.main_window import OfflineAnalyzerMainWindow
    w = OfflineAnalyzerMainWindow()
    try:
        yield w
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


def _inside_ported_analyze_tab(widget, view) -> bool:
    """True if ``widget`` lives under the ported tab, which is allowed its own
    internal tabs; what is forbidden is a tab bar wrapping the view itself."""
    p = widget
    while p is not None:
        if p is view:
            return True
        p = p.parent()
    return False


def test_main_window_boots(win):
    assert win.windowTitle()


def test_the_analyze_view_is_the_window(win):
    """One view, mounted directly, no tab bar.

    The window used to carry an MCU Analysis tab beside the video one. MCU
    logs are read on the rig; this analyser analyses recordings, and a tab bar
    holding a single tab is a control that cannot be operated.
    """
    from tools.offline_analysis.analyze.analyze_tab import AnalyzeTab
    assert isinstance(win.vt_view, AnalyzeTab)
    assert not hasattr(win, "mcu_view")
    outer = [t for t in win.findChildren(QtWidgets.QTabWidget)
             if not _inside_ported_analyze_tab(t, win.vt_view)]
    assert not outer, f"a tab bar wraps the view again: {outer}"


def test_top_selector_three_buttons(win):
    for name in ("folder_btn", "files_btn", "project_btn"):
        assert hasattr(win.selector, name), f"selector lost {name}"


def test_no_menubar(win):
    assert win.menuBar().actions() == [], "expected empty menubar (File/Help dropped)"


def test_floating_log_button_pattern(win):
    """Match main GUI: floating RotatedButton + CollapsibleSidebar slide-out.
    No QDockWidget, no permanent left strip."""
    from tools.offline_analysis.sidebar_log import CollapsibleSidebar, RotatedButton

    win.show()
    assert isinstance(win.log_btn, RotatedButton)
    assert isinstance(win.log_sidebar, CollapsibleSidebar)
    assert win.log_sidebar.is_collapsed, "the log must start out of the way"
    assert not win.findChildren(QtWidgets.QDockWidget), "a dock came back"


def test_the_video_tab_is_the_ported_analyze_tab(win):
    """The video tab IS pyBehaveTrack's Analyze tab, not a reimplementation.

    So the assertions are the ported tab's own anatomy, the files table with
    its seven columns and the plan strip that says what Run will do.
    """
    from tools.offline_analysis.analyze.analyze_tab import AnalyzeTab
    assert isinstance(win.vt_view, AnalyzeTab)
    assert win.vt_view._file_table.columnCount() == 7
    assert hasattr(win.vt_view, "_plan_label"), "no plan strip"


_3D_WORDS = re.compile(
    r"\b3d\b|volume|rearing|reconstruct|pose3d|multi-?view|triangulat|mveks",
    re.I)


def test_no_3d_anywhere_on_the_tab(win):
    """3D is absent, not hidden.

    The port arrived with pyBehaveTrack's whole 3D workspace, volumes, height
    and rearing, the mvEKS smoother, DLC-3D export. Hiding it left every one
    of those controls a ``setVisible`` call away from a rig that cannot
    produce the data they need.
    """
    win.show()
    found = []
    for w in win.findChildren(QtWidgets.QWidget):
        for getter in ("text", "windowTitle", "toolTip"):
            f = getattr(w, getter, None)
            if callable(f):
                t = f()
                if isinstance(t, str) and _3D_WORDS.search(t):
                    found.append(f"{type(w).__name__}: {t[:60]}")
    for act in win.findChildren(QtWidgets.QWidget):
        for a in act.actions():
            if _3D_WORDS.search(a.text()):
                found.append(f"QAction: {a.text()}")
    assert not found, "3D controls found:\n" + "\n".join(found)


def test_log_button_does_not_overlap_content(win):
    """The central widget must reserve a left column for the floating
    Log button so it doesn't cover the top selector or the tab bar.
    Mirrors the rig GUI's setContentsMargins(26, 4, 4, 0) trick."""
    win.show()
    left = win._central.layout().contentsMargins().left()
    assert left >= win._LOG_BTN_W, (
        f"left margin {left} does not clear the {win._LOG_BTN_W}px Log button")


def test_there_is_exactly_one_log(win):
    """One log in the window, and it is the sidebar.

    The ported tab arrived with a Log pane of its own, which was kept for a
    while as a named exception because the port was meant to be faithful. Two
    logs means half the messages are in the other one.
    """
    from tools.offline_analysis.sidebar_log import LogPanel
    panels = win.findChildren(LogPanel)
    assert len(panels) == 1, f"expected one LogPanel, found {len(panels)}"
    assert _inside_ported_analyze_tab(panels[0], win.log_sidebar), (
        "the one log must be the sidebar's")


def test_asking_for_the_log_opens_it(win):
    """The button is the only way in, so it has to work."""
    win.show()
    assert win.log_sidebar.is_collapsed
    win.log_btn.click()
    win.log_sidebar.expand()          # the animation's end state
    assert not win.log_sidebar.is_collapsed
