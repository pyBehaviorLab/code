"""
source/gui/utility.py - Minimal GUI utilities (ported from original GUI).
"""

import os
import re
from PySide6 import QtGui, QtCore, QtWidgets
from source.communication.message import MsgType


_COM_NATURAL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def com_sort_key(name: str):
    """Natural-sort key for COM port names: ``COM2`` sorts before
    ``COM10``. Falls back to ``(name, 0)`` for anything that doesn't
    look like ``<letters><digits>``.
    """
    s = str(name or "")
    m = _COM_NATURAL_RE.match(s)
    if m:
        return (m.group(1).upper(), int(m.group(2)))
    return (s.upper(), 0)


class NoWheelComboBox(QtWidgets.QComboBox):
    """QComboBox that ignores mouse-wheel events so scrolling over the
    widget can't silently change its selection (a common accidental
    re-assignment, e.g. picking the wrong COM port). The wheel event is
    passed up to the parent so the surrounding view still scrolls.
    """

    def wheelEvent(self, event):  # noqa: N802 (Qt override)
        event.ignore()


def install_quiet_qt_warnings():
    """Replace ``QMessageBox.warning`` globally with a variant that uses
    ``Icon.NoIcon`` so the OS doesn't play a system-error beep every time
    we surface a non-critical warning. Idempotent, safe to call once
    per process from each main_window's ``__init__``."""
    def _quiet(parent, title, text,
               buttons=QtWidgets.QMessageBox.StandardButton.Ok,
               defaultButton=QtWidgets.QMessageBox.StandardButton.NoButton):
        box = QtWidgets.QMessageBox(parent)
        box.setWindowTitle(title)
        box.setText(text)
        box.setIcon(QtWidgets.QMessageBox.Icon.NoIcon)
        box.setStandardButtons(buttons)
        box.setDefaultButton(
            defaultButton if defaultButton != QtWidgets.QMessageBox.StandardButton.NoButton
            else buttons)
        return box.exec()
    QtWidgets.QMessageBox.warning = staticmethod(_quiet)

# Constants used for eval of expressions in controls dialogs.
variable_constants = {
    "ms": 1,
    "second": 1000,
    "minute": 60000,
    "hour": 3600000,
}


class TableCheckbox(QtWidgets.QWidget):
    """Checkbox centered in a table cell."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.checkbox = QtWidgets.QCheckBox(parent=parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.checkbox)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)

    def isChecked(self):
        return self.checkbox.isChecked()

    def setChecked(self, state):
        self.checkbox.setChecked(state)


def init_keyboard_shortcuts(widget, shortcut_dict):
    """Initialize keyboard shortcuts on a widget."""
    for key_str, action in shortcut_dict.items():
        key_seq = QtGui.QKeySequence(key_str)
        QtGui.QShortcut(key_seq, widget, action)


class NestedMenu(QtWidgets.QPushButton):
    """QPushButton with a nested QMenu reflecting a folder tree of task files.

    Acts as a drop-in replacement for QComboBox task selectors.
    Call update_menu(root_folder) to populate, set_callback(fn) to receive selection.
    The selected item text is available via inherited .text().
    """

    def __init__(self, default_text="--- Select Task ---", file_filter=".py", parent=None):
        super().__init__(default_text, parent)
        self._default_text = default_text
        self._file_filter = file_filter
        self._callback = None
        self._menu = QtWidgets.QMenu(self)
        self.setMenu(self._menu)
        # Visually a combobox, every plain QComboBox in the app uses
        # the same recipe via style_builders.combobox_style(), and
        # both classes point at the same chevron SVG so the closed
        # state is pixel-identical.
        from source.gui.theme import THEME as _T
        from source.gui.style_builders import _CHEVRON_URL as _CHEV
        p = _T.palette
        self.setStyleSheet(
            "QPushButton {"
            f" background: {p.surface_elev};"
            f" color: {p.text};"
            f" border: 1px solid {p.surface_border_strong};"
            f" border-top: 1px solid rgba(255,255,255,0.22);"
            f" border-radius: {_T.radius.md}px;"
            # Tight vertical padding so the menu-button respects the
            # caller's setFixedHeight.
            " padding: 2px 30px 2px 12px;"
            " text-align: left;"
            f" font: {_T.font.body_pt}pt '{_T.font.family}';"
            "}"
            f"QPushButton:hover    {{ border-color: {p.text_muted}; }}"
            f"QPushButton:focus    {{ border-color: {p.focus}; }}"
            f"QPushButton:disabled {{ color: {p.text_dim};"
            " background: rgba(255,255,255,0.02); }"
            "QPushButton::menu-indicator {"
            f" image: url({_CHEV});"
            "  width: 12px; height: 8px;"
            "  subcontrol-position: right center;"
            "  subcontrol-origin: padding;"
            "  right: 10px;"
            "}"
        )
        # Dark menu popup, same palette as QComboBox dropdown.
        self._menu.setStyleSheet(
            "QMenu {"
            f" background-color: {p.surface};"
            f" color: {p.text};"
            f" border: 1px solid {p.surface_border_strong};"
            f" border-radius: {_T.radius.md}px;"
            " padding: 4px;"
            "}"
            "QMenu::item {"
            "  padding: 6px 18px 6px 12px;"
            f"  border-radius: {_T.radius.sm}px;"
            "  min-height: 22px;"
            "}"
            "QMenu::item:selected {"
            "  background: rgba(37,99,235,0.30);"
            f"  color: {p.text};"
            "}"
            f"QMenu::separator {{ height: 1px; background: {p.surface_border}; margin: 4px 8px; }}"
        )

    def set_callback(self, fn):
        """Set the callback invoked when a task is selected: fn(task_relative_path)."""
        self._callback = fn

    def update_menu(self, root_folder):
        """Rebuild the menu tree from root_folder, showing files matching file_filter."""
        self._menu.clear()
        if not root_folder or not os.path.isdir(root_folder):
            action = self._menu.addAction("No tasks found")
            action.setEnabled(False)
            return
        self._build_menu(self._menu, root_folder, root_folder)

    def _build_menu(self, parent_menu, current_dir, root_dir):
        """Recursively build menu items for directories and matching files."""
        try:
            entries = sorted(os.listdir(current_dir), key=str.lower)
        except OSError:
            return

        # Directories first
        for entry in entries:
            full_path = os.path.join(current_dir, entry)
            if os.path.isdir(full_path) and not entry.startswith((".", "__")):
                sub_menu = parent_menu.addMenu(entry)
                self._build_menu(sub_menu, full_path, root_dir)
                # Remove empty submenus
                if sub_menu.isEmpty():
                    parent_menu.removeAction(sub_menu.menuAction())

        # Then files
        for entry in entries:
            full_path = os.path.join(current_dir, entry)
            if os.path.isfile(full_path) and entry.endswith(self._file_filter):
                # Relative path from root, without extension
                rel = os.path.relpath(full_path, root_dir)
                display = os.path.splitext(entry)[0]
                parent_menu.addAction(
                    self.create_action(display, rel, parent_menu))

    def create_action(self, display_text, task_rel_path, owner=None):
        """A QAction that sets the button text and fires the callback.

        Parented to the MENU that will show it, never to this button.
        ``QMenu.clear()`` only deletes actions the menu itself owns, so an
        action parented to the button survived every rebuild as a hidden
        child, and because its ``triggered`` lambda captures ``self``, each
        leaked action formed a ``self -> action -> lambda -> self`` cycle that
        only the cyclic GC could break. Collecting one whose C++ side was
        still referenced double-freed it and corrupted the heap.
        """
        action = QtGui.QAction(display_text, owner if owner is not None
                               else self._menu)
        action.triggered.connect(
            lambda checked=False, p=task_rel_path: self._on_selected(p))
        return action

    def _on_selected(self, task_rel_path):
        """Handle menu item selection."""
        # Remove .py extension if present for display
        display = os.path.splitext(task_rel_path)[0]
        self.setText(display)
        if self._callback:
            self._callback(display)

    # Compatibility shims for code that used QComboBox API
    def currentText(self):
        """Compatibility: return current text like QComboBox."""
        return self.text()

    def setEnabled(self, enabled):
        """Override to also enable/disable the menu."""
        super().setEnabled(enabled)


class TaskInfo:
    """State/Event/Print display labels for LiveStatusWidget.

    Scans incoming data tuples for the latest state, event, print, and warning.
    Warnings shown in orange.
    """

    def __init__(self):
        # Optional callback fired on every state CHANGE with
        # ``(state_name, fw_time_ms_or_None)``, lets a host stamp when the
        # current state was entered (e.g. maze's per-state duration read-out).
        self.state_change_cb = None
        label_style = "font-weight: bold; font-size: 8pt; color: #aaa;"
        text_style = "font-size: 8pt; color: #f8f8f2; background: transparent; border: none;"
        # State label gets the same 10pt size as the state text, with an
        # accent colour so it stands out in the info bar.
        state_label_style = "font-weight: bold; font-size: 10pt; color: #8be9fd;"

        self.state_label = QtWidgets.QLabel("State:")
        self.state_label.setStyleSheet(state_label_style)
        # State-name text, bigger font, left aligned, persisted across
        # process_data / update_state reapplications.
        self.state_text = QtWidgets.QLineEdit()
        self.state_text.setReadOnly(True)
        self.state_text.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.state_text.setStyleSheet(self._STATE_NORMAL_STYLE)
        self.state_text.setFixedWidth(140)

        self.event_label = QtWidgets.QLabel("Event:")
        self.event_label.setStyleSheet(label_style)
        self.event_text = QtWidgets.QLineEdit()
        self.event_text.setReadOnly(True)
        self.event_text.setStyleSheet(text_style)
        self.event_text.setFixedWidth(100)

        self.print_label = QtWidgets.QLabel("Print:")
        self.print_label.setStyleSheet(label_style)
        self.print_text = QtWidgets.QLineEdit()
        self.print_text.setReadOnly(True)
        self.print_text.setStyleSheet(text_style)
        self.print_text.setFixedWidth(160)

    def set_state_machine(self, sm_info):
        """Attach State_machine_info so process_data can resolve IDs to names.
        Mirrors pyControl_v0 utility.py:437. Call after setup_state_machine().
        """
        self.sm_info = sm_info
        self.state_text.setText("")
        self.event_text.setText("")
        self.print_text.setText("")

    def process_data(self, new_data):
        """Scan new_data (list of Datatuples) for latest state/event/print/warning.

        Resolves state/event IDs to names via sm_info.ID2name when available
        (mirrors pyControl_v0). Falls back to the integer ID if sm_info is
        not yet attached (pre-upload datatuples).
        """
        sm_info = getattr(self, 'sm_info', None)
        id2name = getattr(sm_info, 'ID2name', None) if sm_info else None
        normal = self._NORMAL_STYLE
        warn = self._WARN_STYLE
        state_done = False
        for datum in reversed(new_data):
            if datum.type == MsgType.STATE:
                # reversed(new_data) is newest-first; take the newest STATE
                # only so the field shows the CURRENT state (not the oldest
                # in the batch) and the duration stamps the right entry time.
                if state_done:
                    continue
                state_done = True
                name = id2name.get(datum.content, str(datum.content)) if id2name else str(datum.content)
                self.state_text.setText(name)
                self.state_text.setStyleSheet(self._STATE_NORMAL_STYLE)
                if callable(self.state_change_cb):
                    try:
                        self.state_change_cb(name, getattr(datum, "time", None))
                    except Exception:
                        pass
            elif datum.type == MsgType.EVENT:
                name = id2name.get(datum.content, str(datum.content)) if id2name else str(datum.content)
                self.event_text.setText(name)
                self.event_text.setStyleSheet(normal)
            elif datum.type == MsgType.PRINT:
                self.print_text.setText(str(datum.content))
                self.print_text.setStyleSheet(normal)
            elif datum.type == MsgType.WARNG:
                self.print_text.setText("! " + str(datum.content))
                self.print_text.setStyleSheet(warn)

    # â”€â”€ Typed setters for IPC-driven flows (maze) â”€â”€
    # When pycboard runs in an engine subprocess, the GUI receives typed events
    # rather than raw datatuples; these setters let the same TaskInfo widget be
    # driven from those event handlers.

    _NORMAL_STYLE = "font-size: 8pt; color: #f8f8f2; background: transparent; border: none;"
    _WARN_STYLE = "font-size: 8pt; color: orange; background: transparent; border: none;"
    # State field gets a bigger font for the maze info bar, re-applied on
    # every state change so process_data updates don't shrink it back.
    _STATE_NORMAL_STYLE = "font-size: 10pt; font-weight: bold; color: #f8f8f2; background: transparent; border: none; padding-left: 2px;"
    _STATE_WARN_STYLE   = "font-size: 10pt; font-weight: bold; color: orange; background: transparent; border: none; padding-left: 2px;"

    def update_state(self, name: str):
        self.state_text.setText(str(name))
        self.state_text.setStyleSheet(self._STATE_NORMAL_STYLE)
        self.state_text.home(False)
        # No firmware time in the typed/IPC path, host stamps its own clock.
        if callable(self.state_change_cb):
            try:
                self.state_change_cb(str(name), None)
            except Exception:
                pass

    def clear(self):
        """Clear all fields."""
        for field in (self.state_text, self.event_text, self.print_text):
            field.setText("")


