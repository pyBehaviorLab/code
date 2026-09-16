"""MCU / board dialogs, single module for everything that talks to the
pyboard or its UI surface.

Contains:
  * UniversalConnectDialog, multi-box Connect
  * UniversalDisconnectDialog, multi-box Disconnect
  * UniversalConfigDialog, multi-box FW / HD / flash / DFU actions
  * UniversalUploadDialog, multi-box task upload

All multi-box dialogs enumerate via ``main_window.get_all_setup_widgets()``
and use the unified widget surface (``box_number``, ``is_connected``,
``com_port``, ``connect_mcu(port)``, ``disconnect_mcu()``) so operant
boxes and maze setups are treated identically.
"""
import os
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from serial import SerialException

from source.communication.pyboard import PyboardError
from source.log import get_logger
from source.gui.utility import TableCheckbox, NoWheelComboBox
from source.gui.style_builders import dialog_theme_qss
from source.gui.dialogs._progress_worker import (
    run_box_actions_in_parallel,
)

logger = get_logger()


def _flag_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in ("0", "false", "no", "off", "")


def _update_live_status(main_window, setup_number, message) -> None:
    """Post a message to a box's LiveStatus log. Best-effort + processEvents
    so the line shows immediately during a blocking serial action."""
    try:
        for w in main_window.get_all_setup_widgets():
            if getattr(w, "setup_number", None) == setup_number:
                if hasattr(w, "print_to_log"):
                    w.print_to_log(message)
                break
        QtWidgets.QApplication.processEvents()
    except Exception as e:
        logger.debug(f"Could not update LiveStatus: {e}")


def _snapshot_capture_hd_source(main_window, hw_file: str, setup_id):
    """Stage an HD .py through the SnapshotStore and
    write the resulting FileRef onto cfg.setup_config.boxes[i].init_hw_def.

    No-op when no project is loaded / no SnapshotStore present.
    """
    if main_window is None:
        return
    store = getattr(main_window, "_snapshot_store", None)
    cfg   = getattr(main_window, "_active_config", None)
    if store is None or cfg is None or setup_id is None:
        return
    box_cfg = next(
        (b for b in cfg.setup_config.boxes
         if b.setup_number == int(setup_id)),
        None,
    )
    if box_cfg is None:
        return
    try:
        ref = store.capture_source(hw_file,
                                   kind="hw_def",
                                   setup_id=int(setup_id),
                                   label=Path(hw_file).name)
        if ref is not None:
            box_cfg.init_hw_def = ref
            # Stage the HD into source/<djb2>.py now so the project becomes
            # self-contained immediately. Idempotent (no-op if already
            # committed); off the hot path.
            try:
                store.commit_box_sources(cfg, int(setup_id))
            except Exception as e:
                logger.debug("immediate HD commit (box %s): %s", setup_id, e)
    except Exception as e:
        logger.warning("snapshot HD capture (box %s): %s", setup_id, e)


def _hd_carry_ref_to_widget(cfg, bw, hw_file, box_no):
    """Mirror the captured HD FileRef from ``cfg.setup_config.boxes[i]`` onto
    the box widget (``init_hw_def`` / ``_hw_def_path``).

    ``read_ui_into_config`` rebuilds the config from widgets on Save, so
    without this the snapshot capture (which only touches cfg) would be lost
    on the next manual Save. No Qt, unit-testable."""
    try:
        bc = None
        if cfg is not None and box_no is not None:
            bc = next((b for b in cfg.setup_config.boxes
                       if b.setup_number == int(box_no)), None)
        if bc is not None and bc.init_hw_def.is_set():
            bw.init_hw_def = bc.init_hw_def
        if hasattr(bw, "_hw_def_path"):
            bw._hw_def_path = hw_file
    except Exception as e:
        logger.debug("HD widget-ref carry (box %s): %s", box_no, e)


def _hd_autosave_is_draft(mw) -> bool:
    """True when the project is still an unsaved draft, so the HD-load
    completion must NOT force a Save (which would pop a name dialog mid
    hardware-setup). The HD ref rides on the widget + snapshot store and
    transfers on the user's first real Save."""
    return (getattr(mw, "_is_draft", False)
            or getattr(mw, "_active_project_dir", None) is None)


# Worker harness (BoxWorker / BoxWorkerSignals / pycboard.print proxy)
# lives in source/gui/dialogs/_progress_worker.py.


# Standard MCU dialog QSS so every dialog renders identically.
_DIALOG_STYLE = dialog_theme_qss()


def _dlg_title(text: str) -> "QtWidgets.QLabel":
    """Big white dialog title, matches the mockup's 22 px bold heading."""
    from PySide6 import QtWidgets as _Qw
    from source.gui.theme import THEME as _T
    lbl = _Qw.QLabel(text)
    lbl.setStyleSheet(
        f"QLabel {{"
        f" color: {_T.palette.text};"
        f" font: 700 16pt '{_T.font.family}';"
        f" padding: 2px 0 6px 0;"
        f"}}"
    )
    return lbl


def _dlg_column_header(text: str) -> "QtWidgets.QLabel":
    """Uppercase muted small-caps column header (BOX / COM PORT / ALL)."""
    from PySide6 import QtWidgets as _Qw
    from source.gui.theme import THEME as _T
    lbl = _Qw.QLabel(text.upper())
    lbl.setStyleSheet(
        f"QLabel {{"
        f" color: {_T.palette.text_muted};"
        f" font: 700 {_T.font.caption_pt}pt '{_T.font.family}';"
        f" letter-spacing: 0.6px;"
        f"}}"
    )
    return lbl


# Icons live at source/gui/icons/, not source/gui/dialogs/icons/. One
# module-level constant used by every callsite.
_ICON_DIR = Path(__file__).parent.parent / 'icons'


def _is_running(widget) -> bool:
    """True when the box's pyControl framework is live (a task is running).

    Uses the per-widget predicate ``_is_box_running()`` (on both operant
    and maze widgets) with a ``framework_running`` fallback. A running box
    is excluded from Disconnect/Config: its row is shown but its checkbox is
    disabled, so the action is never offered for it."""
    try:
        fn = getattr(widget, "_is_box_running", None)
        if callable(fn):
            return bool(fn())
    except Exception:
        pass
    return bool(getattr(widget, "framework_running", False))


def _sort_by_box_number(widgets):
    """Order box widgets by integer box_number (1, 2, 3 ... 10, 11),
    falling back to string compare for non-numeric ids so a stray
    string-id box doesn't crash the dialog list."""
    def _key(w):
        bn = getattr(w, "setup_number", None)
        try:
            return (0, int(bn))
        except (TypeError, ValueError):
            return (1, str(bn))
    return sorted(widgets, key=_key)


def _fit_table_to_rows(table: "QtWidgets.QTableWidget", row_height: int = 28,
                       max_rows: int = 12, min_visible_rows: int = 3) -> None:
    """Size the table to show all its rows (up to ``max_rows``) without
    scrolling. Reserves ``min_visible_rows`` of vertical space so an empty
    table still looks like a table area. Beyond ``max_rows`` it scrolls.
    """
    rows = min(table.rowCount(), max_rows)
    visible_rows = max(rows, min_visible_rows)
    border = 2 * table.frameWidth()
    table.setFixedHeight(visible_rows * row_height + border)
    table.setVerticalScrollBarPolicy(
        QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        if table.rowCount() > max_rows
        else QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    table.setHorizontalScrollBarPolicy(
        QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )


def _show_empty_table_placeholder(table: "QtWidgets.QTableWidget",
                                  message: str) -> None:
    """Insert a single greyed placeholder row spanning all columns when the
    table has no real data, keeping the selector area recognisable."""
    if table.rowCount() != 0:
        return
    table.setRowCount(1)
    cols = max(1, table.columnCount())
    item = QtWidgets.QTableWidgetItem(message)
    item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)  # not selectable, not editable
    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    font = item.font()
    font.setItalic(True)
    item.setFont(font)
    item.setForeground(QtGui.QColor("#9ca3af"))
    table.setItem(0, 0, item)
    if cols > 1:
        table.setSpan(0, 0, 1, cols)
    table.setRowHeight(0, 28)


# =============================================================================
#  UniversalConnectDialog, multi-box Connect
# =============================================================================


class _BoxTableDialogBase(QtWidgets.QDialog):
    """Shared shell for the multi-box dialogs (Connect / Disconnect /
    Start / Stop / Config / Upload): window scaffolding, the
    Box | <middle> | All header with its master toggle, the 3-column
    table, row build and checkbox collection.

    Subclasses set the class attrs, implement ``_eligible_widgets`` and
    the action, and shape their middle column via ``_status_for`` (plain
    text) or ``_middle_cell`` (custom item/widget).
    """

    WINDOW_TITLE = ""
    TITLE_TEXT = None            # title label; default WINDOW_TITLE
    ACTION_LABEL = ""
    COLOR_KEY = "info"
    ICON_NAME = ""
    EMPTY_TEXT = ""
    MIDDLE_HEADER = "Status"
    SELECT_HEADER = None         # extra label above the header row
    FIXED_WIDTH = None           # int → setFixedWidth; None → MIN_WIDTH
    MIN_WIDTH = 280
    COL0_WIDTH = 50
    COL2_WIDTH = 24
    ROW_HEIGHT = 28

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setWindowTitle(self.WINDOW_TITLE)
        if self.FIXED_WIDTH:
            self.setFixedWidth(self.FIXED_WIDTH)
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
        else:
            self.setMinimumWidth(self.MIN_WIDTH)
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Preferred,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
        if self.ICON_NAME:
            icon_path = _ICON_DIR / self.ICON_NAME
            if icon_path.exists():
                self.setWindowIcon(QtGui.QIcon(str(icon_path)))

        self._setup_ui()
        self._populate_rows()
        QtCore.QTimer.singleShot(0, self.adjustSize)

    # ---- subclass hooks --------------------------------------------------

    def _eligible_widgets(self):
        return []

    def _status_for(self, widget) -> str:
        return ""

    def _middle_cell(self, row, widget):
        """Column-1 content: a QTableWidgetItem (default: centred
        read-only ``_status_for`` text) or any QWidget."""
        item = QtWidgets.QTableWidgetItem(self._status_for(widget))
        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        return item

    def _row_checkbox_state(self, widget):
        """``(checked, enabled, tooltip)`` for the row's checkbox."""
        return True, True, ""

    def _prepare_rows(self, widgets) -> None:
        """Called once per (re)populate, before the row loop."""

    def _build_extra_top(self, layout) -> None:
        """Rows between the title and the header (e.g. Connect's
        display-mode picker)."""

    def _build_buttons(self, layout) -> None:
        """Bottom controls; default is one styled action button."""
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(8)
        button_layout.addStretch(1)
        self.action_btn = QtWidgets.QPushButton(self.ACTION_LABEL)
        if self.ICON_NAME:
            icon_path = _ICON_DIR / self.ICON_NAME
            if icon_path.exists():
                self.action_btn.setIcon(QtGui.QIcon(str(icon_path)))
                self.action_btn.setIconSize(QtCore.QSize(16, 16))
        self.action_btn.clicked.connect(self._on_action)
        self.action_btn.setDefault(True)
        from source.gui.styles import BUTTON_STYLE as _BS, COLORS as _C
        self.action_btn.setStyleSheet(_BS.format(
            color=_C[self.COLOR_KEY],
            hover_color=_C[self.COLOR_KEY + "_hover"]))
        self.action_btn.setMinimumHeight(32)
        self.action_btn.setMinimumWidth(130)
        button_layout.addWidget(self.action_btn)
        layout.addLayout(button_layout)

    def _confirm(self, selections) -> bool:
        """Override to insert a confirmation step before the loop."""
        return True

    def _on_action(self):
        raise NotImplementedError(
            f"{type(self).__name__} must override _on_action()")

    # ---- shared UI -------------------------------------------------------

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        self.setStyleSheet(_DIALOG_STYLE)
        layout.addWidget(_dlg_title(self.TITLE_TEXT or self.WINDOW_TITLE))
        self._build_extra_top(layout)
        if self.SELECT_HEADER:
            layout.addWidget(_dlg_column_header(self.SELECT_HEADER))

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setSpacing(8)
        header_layout.addWidget(_dlg_column_header("Box"))
        header_layout.addStretch()
        header_layout.addWidget(_dlg_column_header(self.MIDDLE_HEADER))
        header_layout.addStretch()
        header_layout.addWidget(_dlg_column_header("All"))
        self.master_checkbox = QtWidgets.QCheckBox()
        self.master_checkbox.setChecked(True)
        self.master_checkbox.stateChanged.connect(self._toggle_all)
        header_layout.addWidget(self.master_checkbox)
        layout.addLayout(header_layout)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, self.COL0_WIDTH)
        self.table.setColumnWidth(2, self.COL2_WIDTH)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        layout.addWidget(self.table)

        self._build_buttons(layout)

    def _populate_rows(self):
        widgets = _sort_by_box_number(self._eligible_widgets())
        self.table.setRowCount(len(widgets))

        if not widgets:
            btn = getattr(self, "action_btn", None)
            if btn is not None:
                btn.setEnabled(False)
                btn.setText(self.EMPTY_TEXT or "Nothing to do")
            _show_empty_table_placeholder(self.table, self.EMPTY_TEXT)
            _fit_table_to_rows(self.table, row_height=self.ROW_HEIGHT)
            self.adjustSize()
            return

        self._prepare_rows(widgets)
        for row, widget in enumerate(widgets):
            id_item = QtWidgets.QTableWidgetItem(f"Box {widget.setup_number}")
            id_item.setFlags(id_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            id_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            id_item.setData(QtCore.Qt.ItemDataRole.UserRole, widget)
            self.table.setItem(row, 0, id_item)

            middle = self._middle_cell(row, widget)
            if isinstance(middle, QtWidgets.QTableWidgetItem):
                self.table.setItem(row, 1, middle)
            elif middle is not None:
                self.table.setCellWidget(row, 1, middle)

            checked, enabled, tip = self._row_checkbox_state(widget)
            cb_widget = TableCheckbox()
            cb_widget.setChecked(checked)
            cb_widget.checkbox.setEnabled(enabled)
            if tip:
                cb_widget.checkbox.setToolTip(tip)
            self.table.setCellWidget(row, 2, cb_widget)
            self.table.setRowHeight(row, self.ROW_HEIGHT)
        _fit_table_to_rows(self.table, row_height=self.ROW_HEIGHT)
        self.adjustSize()

    def _toggle_all(self, state):
        checked = (state == QtCore.Qt.CheckState.Checked.value)
        for row in range(self.table.rowCount()):
            cb_container = self.table.cellWidget(row, 2)
            if cb_container:
                cb = cb_container.findChild(QtWidgets.QCheckBox)
                if cb and cb.isEnabled():
                    cb.setChecked(checked)

    def _selected_rows(self):
        """``[(row, widget)]`` for every checked AND enabled row."""
        out = []
        for row in range(self.table.rowCount()):
            cb_container = self.table.cellWidget(row, 2)
            cb = cb_container.findChild(QtWidgets.QCheckBox) if cb_container else None
            if not cb or not cb.isChecked() or not cb.isEnabled():
                continue
            id_item = self.table.item(row, 0)
            widget = id_item.data(QtCore.Qt.ItemDataRole.UserRole) if id_item else None
            if widget is not None:
                out.append((row, widget))
        return out

    def _collect_selections(self):
        return [w for _, w in self._selected_rows()]


class UniversalConnectDialog(_BoxTableDialogBase):
    """Dialog for connecting multiple boxes to COM ports."""

    WINDOW_TITLE = "Connect"
    ACTION_LABEL = "Connect"
    COLOR_KEY = "info"
    ICON_NAME = "connect.svg"
    EMPTY_TEXT = "No boxes"
    MIDDLE_HEADER = "COM Port"

    def _build_extra_top(self, layout):
        # MCU label mode, how boards are shown (display only; binding is
        # always by USB serial). Default "hashed" is OS-independent.
        from source.communication import mcu_ports
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(8)
        mode_row.addWidget(_dlg_column_header("Display MCUs as"))
        self.mode_combo = NoWheelComboBox()
        self.mode_combo.addItem("Hashed ID", "hashed")
        self.mode_combo.addItem("Native port", "native")
        self.mode_combo.addItem("Raw serial", "serial")
        self.mode_combo.setToolTip(
            "Hashed ID: djb2 of the USB serial, same on every PC/OS.\n"
            "Native port: COMn / /dev/ttyACMn (OS- and replug-dependent).\n"
            "Raw serial: the board's USB serial number.")
        _cur_mode = self._resolve_display_mode()
        mcu_ports.set_display_mode(_cur_mode)
        _i = self.mode_combo.findData(_cur_mode)
        if _i >= 0:
            self.mode_combo.setCurrentIndex(_i)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

    def _build_buttons(self, layout):
        # Default action button + a ports-refresh tool button next to it.
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(8)
        button_layout.addStretch(1)
        self.action_btn = QtWidgets.QPushButton(self.ACTION_LABEL)
        connect_icon = _ICON_DIR / 'connect.svg'
        if connect_icon.exists():
            self.action_btn.setIcon(QtGui.QIcon(str(connect_icon)))
            self.action_btn.setIconSize(QtCore.QSize(16, 16))
        self.action_btn.clicked.connect(self._on_action)
        self.action_btn.setDefault(True)
        from source.gui.styles import BUTTON_STYLE as _BS, COLORS as _C
        self.action_btn.setStyleSheet(_BS.format(
            color=_C['info'], hover_color=_C['info_hover']))
        self.action_btn.setMinimumHeight(32)
        self.action_btn.setMinimumWidth(120)
        button_layout.addWidget(self.action_btn)

        self.refresh_btn = QtWidgets.QToolButton()
        refresh_icon = _ICON_DIR / 'refresh.svg'
        if refresh_icon.exists():
            self.refresh_btn.setIcon(QtGui.QIcon(str(refresh_icon)))
            self.refresh_btn.setIconSize(QtCore.QSize(16, 16))
        else:
            # Fallback glyph if the asset is missing in a deployed bundle.
            self.refresh_btn.setText("↻")
        self.refresh_btn.setToolTip("Refresh ports")
        self.refresh_btn.clicked.connect(self._populate_rows)
        self.refresh_btn.setFixedHeight(24)
        self.refresh_btn.setFixedWidth(32)
        button_layout.addWidget(self.refresh_btn)

        layout.addLayout(button_layout)

    @staticmethod
    def _list_all_mcu_choices():
        """Return ``[(serial, device), ...]`` for every pickable MCU,
        cross-platform. Serial'd pyboards from ``list_mcu_serials`` come first;
        then ANY pyboard-like port (MicroPython VID 0xF055, or a
        Pyboard/USB-Serial description) not already listed is appended with
        its serial if it has one, else an empty serial. This is what keeps a
        board visible after Disable Flash Drive, in VCP-only mode its PID
        leaves ``list_mcu_serials``' allowed set, so it must be picked up here
        by description/VID rather than dropped.
        """
        from source.communication.mcu_ports import list_mcu_serials
        out = list(list_mcu_serials())
        have = {dev for _, dev in out}
        try:
            from serial.tools import list_ports
            for p in list_ports.comports():
                desc = p.description or ""
                is_pyb = (p.vid == 0xF055
                          or "Pyboard" in desc or "USB Serial Device" in desc)
                if is_pyb and p.device not in have:
                    out.append(((p.serial_number or "").strip(), p.device))
                    have.add(p.device)
        except Exception:
            pass
        return out

    def _resolve_display_mode(self):
        """Project's pinned mcu_display_mode if set, else the per-machine
        default (settings.json)."""
        from source.config.settings import get_mcu_display_mode
        cfg = getattr(self.main_window, "_active_config", None)
        proj = (getattr(getattr(cfg, "meta", None), "mcu_display_mode", "") or "")
        return proj or get_mcu_display_mode()

    def _on_mode_changed(self):
        """Apply the picked label mode (process-wide + per-machine default +
        pinned into the project), then repopulate the combos and refresh the
        per-box COM fields live."""
        from source.communication import mcu_ports
        from source.config.settings import set_mcu_display_mode
        mode = self.mode_combo.currentData() or "hashed"
        mcu_ports.set_display_mode(mode)
        set_mcu_display_mode(mode)
        cfg = getattr(self.main_window, "_active_config", None)
        if cfg is not None and getattr(cfg, "meta", None) is not None:
            cfg.meta.mcu_display_mode = mode
            changed = getattr(self.main_window, "_project_changed", None)
            if callable(changed):
                try:
                    changed(reason="mcu_display_mode")
                except Exception:
                    pass
        self._populate_rows()
        # Live-refresh per-box COM labels (they're set on connect/load).
        try:
            for w in self.main_window.get_all_setup_widgets():
                sn = (getattr(w, "_mcu_serial", "")
                      or getattr(w, "mcu_serial", "") or "")
                if sn and hasattr(w, "com_id_edit"):
                    w.com_id_edit.setText(mcu_ports.label_for(sn))
        except Exception as e:
            logger.debug("per-box COM relabel: %s", e)

    def _eligible_widgets(self):
        if not self.main_window:
            return []
        try:
            return list(self.main_window.get_all_setup_widgets())
        except Exception:
            return []

    def _prepare_rows(self, widgets):
        # One cross-platform enumeration per populate. ``choices`` =
        # [(serial, device)]; serial "" means a no-serial board (connect by
        # device path). The combo displays the mode-dependent label but
        # stores the connect value in itemData, so the label can be a hash
        # without breaking connect.
        self._choices = self._list_all_mcu_choices()
        self._device_by_serial = {sn: dev for sn, dev in self._choices if sn}
        # Serials claimed by connected boxes are taken, so two boxes can't
        # grab the same MCU in one dialog round.
        self._used = {(getattr(w, "_mcu_serial", "")
                       or getattr(w, "mcu_serial", "") or "")
                      for w in widgets if w.is_connected}
        self._used.discard("")

    def _middle_cell(self, row, widget):
        from source.communication import mcu_ports
        ConnectRole = QtCore.Qt.ItemDataRole.UserRole
        TipRole = QtCore.Qt.ItemDataRole.ToolTipRole
        com_combo = NoWheelComboBox()
        com_combo.setEditable(False)
        current_serial = (getattr(widget, "_mcu_serial", "")
                          or getattr(widget, "mcu_serial", "") or "")

        available = [(sn, dev) for sn, dev in self._choices
                     if not sn or sn not in self._used or sn == current_serial]
        if not current_serial:
            com_combo.addItem("--- Select MCU ---", "")
        for sn, dev in available:
            if sn:
                com_combo.addItem(mcu_ports.label_for(sn, dev), sn)
                com_combo.setItemData(com_combo.count() - 1,
                                      mcu_ports.tooltip_for(sn, dev), TipRole)
            else:
                # No-serial board: label + connect-value are the device.
                com_combo.addItem(dev, dev)
                com_combo.setItemData(com_combo.count() - 1,
                                      f"{dev}  (no USB serial, bind not stable)",
                                      TipRole)

        # Pre-select this box's saved board by its connect value (serial).
        sel = next((i for i in range(com_combo.count())
                    if current_serial and com_combo.itemData(i, ConnectRole) == current_serial),
                   -1)
        if sel >= 0:
            com_combo.setCurrentIndex(sel)
        elif current_serial:
            # Saved board isn't plugged in, greyed entry so the operator
            # sees "this box wants <id> but it's unplugged".
            com_combo.addItem(
                mcu_ports.label_for(current_serial) + "  (not connected)",
                current_serial)
            com_combo.setItemData(com_combo.count() - 1,
                                  mcu_ports.tooltip_for(current_serial), TipRole)
            com_combo.setCurrentIndex(com_combo.count() - 1)
        if current_serial:
            com_combo.setToolTip(
                mcu_ports.tooltip_for(current_serial,
                                      self._device_by_serial.get(current_serial)))
        return com_combo

    def _row_checkbox_state(self, widget):
        is_connected = widget.is_connected
        return (not is_connected, not is_connected, "")

    def _on_action(self):
        try:
            selections = [(widget, self.table.cellWidget(row, 1))
                          for row, widget in self._selected_rows()]

            if not selections:
                QtWidgets.QMessageBox.information(
                    self, "No connections made",
                    "Select boxes and valid COM ports first.")
                return

            from source.gui.dialogs._progress_worker import build_progress_dialog
            progress = build_progress_dialog(
                self, "Connecting boards", len(selections),
                header="Connecting selected boards...", with_detail=True)
            bar, detail = progress.bar, progress.detail

            connected_count = 0
            for idx, (widget, com_combo) in enumerate(selections, start=1):
                QtWidgets.QApplication.processEvents()
                # The connect value lives in itemData (serial, or device path
                # for a no-serial board), not the visible text, which in
                # hashed/native mode isn't a connectable identifier.
                serial = (com_combo.currentData(QtCore.Qt.ItemDataRole.UserRole)
                          if com_combo else "") or ""
                if not serial:
                    detail.setText(f"Box {widget.setup_number}: no MCU selected")
                    bar.setValue(idx)
                    continue
                try:
                    widget.connect_mcu(serial)
                    if widget.is_connected:
                        connected_count += 1
                        detail.setText(
                            f"Box {widget.setup_number}: connected (MCU {serial})")
                    else:
                        detail.setText(f"Box {widget.setup_number}: connect failed")
                except Exception as e:
                    logger.error(f"Box {widget.setup_number}: connect error: {e}")
                    detail.setText(f"Box {widget.setup_number}: error")
                bar.setValue(idx)

            QtCore.QTimer.singleShot(400, progress.accept)
            progress.exec()

            if connected_count:
                logger.info(f"Successfully connected {connected_count} box(es)")
            self.accept()

        except Exception as e:
            logger.error(f"Error connecting boxes: {e}")
            self.reject()



# =============================================================================
#  UniversalDisconnectDialog, multi-box Disconnect
# =============================================================================


class UniversalDisconnectDialog(_BoxTableDialogBase):
    """Dialog for disconnecting selected boards."""

    WINDOW_TITLE = "Disconnect"
    ACTION_LABEL = "Disconnect"
    # Warning (not danger) since disconnect is recoverable.
    COLOR_KEY = "warning"
    ICON_NAME = "disconnect.svg"
    EMPTY_TEXT = "No boards connected"
    # Fixed width, the Box label + COM port + checkbox row is short.
    FIXED_WIDTH = 182

    def _eligible_widgets(self):
        if not self.main_window:
            return []
        try:
            return [w for w in self.main_window.get_all_setup_widgets()
                    if getattr(w, "is_connected", False)]
        except Exception:
            return []

    def _middle_cell(self, row, widget):
        running = _is_running(widget)
        port = widget.com_port or ""
        # A running box stays visible; its status reads "(running)" to
        # explain why its checkbox is greyed.
        status_text = (f"{port}  (running)".strip() if port else "running") \
            if running else port
        item = QtWidgets.QTableWidgetItem(status_text)
        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        if running:
            item.setForeground(QtGui.QColor("#9ca3af"))
        return item

    def _row_checkbox_state(self, widget):
        # Show the row but disable the checkbox for running boxes:
        # disconnecting one would yank its MCU mid-session, so the
        # option is simply not offered.
        running = _is_running(widget)
        return (not running, not running, "")

    def _on_action(self):
        try:
            # Running boxes have a disabled, unchecked checkbox; the shared
            # collection's isEnabled() guard keeps them out.
            selections = self._collect_selections()

            if not selections:
                QtWidgets.QMessageBox.information(
                    self, "No Selection",
                    "Select at least one board to disconnect.")
                return

            disconnected_count = 0
            for widget in selections:
                try:
                    widget.disconnect_mcu()
                    disconnected_count += 1
                    logger.info(f"Disconnected box {widget.setup_number}")
                except Exception as e:
                    logger.error(f"Box {widget.setup_number}: disconnect error: {e}")

            if disconnected_count and self.main_window:
                try:
                    if hasattr(self.main_window, "statusbar"):
                        self.main_window.statusbar.showMessage(
                            f"Disconnected {disconnected_count} board(s)", 3000)
                    self.main_window.refresh_ui_state()
                except Exception:
                    pass

            self.accept()

        except Exception as e:
            logger.error(f"Error disconnecting boxes: {e}")
            self.reject()


# =============================================================================
#  UniversalStartDialog / UniversalStopDialog, multi-box Start/Stop
# =============================================================================
#
# List the eligible boxes (Start = ready, Stop = running), let the user pick
# a subset via per-row checkboxes, then call the per-box handler on each.
# The Stop dialog requires a confirm-warning step first.


class UniversalStartDialog(_BoxTableDialogBase):
    """Multi-box Start.

    Eligibility = box is ready to start, which means:
      * MCU connected
      * task uploaded
      * the per-box record_button is currently enabled (i.e. it's in
        its Start / Record state, not greyed out because the box is
        already running).

    Action = click each selected box's record_button. Whatever that
    button does at runtime fires.
    """

    WINDOW_TITLE = "Multi-Start"
    ACTION_LABEL = "Start"
    COLOR_KEY = "success"
    ICON_NAME = "play.svg"
    EMPTY_TEXT = "No boxes ready to start."

    def _eligible_widgets(self):
        if not self.main_window:
            return []
        try:
            widgets = self.main_window.get_all_setup_widgets()
        except Exception:
            return []
        out = []
        for w in widgets:
            try:
                if not w._is_box_connected():
                    continue
                if not w._is_task_uploaded():
                    continue
                btn = getattr(w, "record_button", None)
                if btn is None or not btn.isEnabled():
                    continue
                out.append(w)
            except Exception:
                continue
        return out

    def _status_for(self, widget) -> str:
        try:
            subject = (widget.subject_id_edit.text() or "").strip()
        except Exception:
            subject = ""
        try:
            task = ""
            combo = getattr(widget, "task_combo", None)
            if combo is not None and hasattr(combo, "text"):
                task = (combo.text() or "").strip()
        except Exception:
            task = ""
        bits = []
        if subject:
            bits.append(subject)
        else:
            bits.append("(no subject)")
        if task and task not in ("--- Select Task ---", "No tasks found"):
            bits.append(task)
        return " · ".join(bits)

    def _on_action(self):
        """Serial for-loop over selected boxes. Each per-box Start is
        ~50 ms (just ``start_framework``), so serial across 16 boxes is
        well under a second.
        """
        try:
            selections = self._collect_selections()
            if not selections:
                QtWidgets.QMessageBox.information(
                    self, "No Selection",
                    "Select at least one box.")
                return
            if not self._confirm(selections):
                return

            from source.gui.project_workflow import (
                flush_pending_runs,
                _BATCH_DEFER,
            )

            # Defer the per-box runs-JSON flush for one disk write at the
            # end of the batch; flush_pending_runs below finalises. Also
            # coalesce the O(N) refresh_ui_state sweep to one call at the
            # end (else 16 boxes → O(N²) button-state sweeps mid-start).
            try:
                _BATCH_DEFER.active = True
            except Exception:
                pass
            mw = self.main_window
            if mw is not None and hasattr(mw, "begin_ui_refresh_batch"):
                try:
                    mw.begin_ui_refresh_batch()
                except Exception:
                    pass

            ok_count = 0
            try:
                for widget in selections:
                    try:
                        # _invoke clicks the record_button synchronously on
                        # the GUI thread, same as a manual per-box click.
                        if self._invoke(widget):
                            ok_count += 1
                    except Exception as e:
                        logger.warning(
                            "Universal Start: box %s start failed: %s",
                            getattr(widget, "setup_number", "?"), e)
                    QtWidgets.QApplication.processEvents()
            finally:
                try:
                    _BATCH_DEFER.active = False
                except Exception:
                    pass
                if mw is not None and hasattr(mw, "end_ui_refresh_batch"):
                    try:
                        mw.end_ui_refresh_batch()
                    except Exception:
                        pass

            if ok_count and self.main_window:
                try:
                    if hasattr(self.main_window, "statusbar"):
                        self.main_window.statusbar.showMessage(
                            f"Start: {ok_count} box(es)", 3000)
                    self.main_window.refresh_ui_state()
                except Exception:
                    pass

            try:
                pd = getattr(self.main_window, "_active_project_dir", None)
                if pd is not None:
                    flush_pending_runs(pd)
            except Exception as e:
                logger.warning("runs: batched flush failed: %s", e)
            self.accept()
        except Exception as e:
            logger.error("Universal Start (serial) error: %s", e)
            self.reject()

    def _invoke(self, widget) -> bool:
        """Click the per-box record_button (synchronous on the GUI thread);
        returns True iff it fired (button was enabled)."""
        btn = getattr(widget, "record_button", None)
        if btn is None or not btn.isEnabled():
            return False
        btn.click()
        return True


class UniversalStopDialog(_BoxTableDialogBase):
    """Multi-box Stop (sequentially call ``widget.on_stop_clicked()``).

    Requires explicit confirmation, stopping a running session is
    destructive in the sense that you can't undo it from this dialog.
    """

    WINDOW_TITLE = "Multi-Stop"
    ACTION_LABEL = "Stop"
    COLOR_KEY = "warning"
    ICON_NAME = "stop.svg"
    EMPTY_TEXT = "No running sessions."

    def _eligible_widgets(self):
        if not self.main_window:
            return []
        try:
            widgets = self.main_window.get_all_setup_widgets()
        except Exception:
            return []
        out = []
        for w in widgets:
            try:
                if w._is_box_running():
                    out.append(w)
            except Exception:
                continue
        return out

    def _status_for(self, widget) -> str:
        try:
            subject = (widget.subject_id_edit.text() or "").strip()
        except Exception:
            subject = ""
        return subject or "running"

    def _confirm(self, selections) -> bool:
        names = ", ".join(f"Box {w.setup_number}" for w in selections)
        reply = QtWidgets.QMessageBox.warning(
            self, "Stop selected boxes?",
            f"Stop the following {len(selections)} session(s)?\n\n{names}\n\n"
            "Recording and the pyControl framework will halt for these "
            "boxes. This cannot be undone, any further data must be a "
            "new run.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    def _on_action(self):
        """Drive the parallel coordinator (action='stop'): per-box mcu_stop
        + writer-close run concurrently."""
        try:
            selections = self._collect_selections()
            if not selections:
                QtWidgets.QMessageBox.information(
                    self, "No Selection",
                    "Select at least one running box.")
                return
            if not self._confirm(selections):
                return

            from source.gui.dialogs.parallel_stop import (
                ParallelStopCoordinator,
            )
            from source.gui.project_workflow import (
                flush_pending_runs,
                _BATCH_DEFER,
            )

            def _on_finished(ok_count: int, total: int) -> None:
                try:
                    _BATCH_DEFER.active = False
                except Exception:
                    pass
                if ok_count and self.main_window:
                    try:
                        if hasattr(self.main_window, "statusbar"):
                            self.main_window.statusbar.showMessage(
                                f"Stop: {ok_count} box(es)", 3000)
                        self.main_window.refresh_ui_state()
                    except Exception:
                        pass
                try:
                    pd = getattr(self.main_window, "_active_project_dir", None)
                    if pd is not None:
                        flush_pending_runs(pd)
                except Exception as e:
                    logger.warning("runs: batched flush failed: %s", e)
                self.accept()

            try:
                _BATCH_DEFER.active = True
            except Exception:
                pass

            self._coordinator = ParallelStopCoordinator(
                widgets=selections,
                on_finished=_on_finished,
                parent=self,
            )
            self._coordinator.run()
        except Exception as e:
            logger.error("Universal Stop (parallel) error: %s", e)
            self.reject()

# =============================================================================
#  UniversalConfigDialog, multi-box FW / HD / flash drive / DFU
# =============================================================================


class UniversalConfigDialog(_BoxTableDialogBase):
    """Dialog for configuring multiple boards with direct action buttons."""

    WINDOW_TITLE = "Config boards"
    TITLE_TEXT = "Global Configuration"
    SELECT_HEADER = "Select Boards"
    MIDDLE_HEADER = "COM Port"
    ICON_NAME = "upload.svg"
    EMPTY_TEXT = "No connected boards, connect first."
    # Fixed width, the long "Load Hardware Definition" label is shrunk
    # to 9 pt below so it still fits.
    FIXED_WIDTH = 252
    COL0_WIDTH = 60
    COL2_WIDTH = 40

    def keyPressEvent(self, event):
        # Enter closes the Config dialog; it must NOT trigger a destructive
        # hardware action (Load Framework / HW def / DFU / flash). Other keys
        # behave normally.
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self.accept()
            event.accept()
            return
        super().keyPressEvent(event)

    def _build_buttons(self, layout):
        # Parallel toggle right under the selector, applies to the
        # selection, not the action.
        self.parallel_checkbox = QtWidgets.QCheckBox("Parallel (faster)")
        self.parallel_checkbox.setChecked(_flag_env("PYBEHAVIORLAB_MCU_PARALLEL", True))
        self.parallel_checkbox.setToolTip(
            "Run selected boards concurrently (per-board worker threads). "
            "Disable if your USB/serial hub is unstable."
        )
        layout.addWidget(self.parallel_checkbox)

        # Thin divider to visually separate selection from actions.
        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        divider.setStyleSheet("color: #3d3d3d; background: #3d3d3d; max-height: 1px;")
        layout.addWidget(divider)

        # ---- Action buttons --------------------------------------------
        # Order: framework + hardware-def (MCU prep), then the flash-drive
        # pair with DFU between them to separate Enable from Disable.
        button_column = QtWidgets.QVBoxLayout()
        button_column.setSpacing(6)

        # Each tuple = (label, icon, callback, COLORS-key).
        from source.gui.styles import BUTTON_STYLE as _BS, COLORS as _C
        buttons_data = [
            ("Load Framework",           "upload.svg",  self.uploadFramework,        "info"),
            ("Load Hardware Definition", "upload.svg",  self.loadHardwareDefinition, "warning"),
            ("Enable Flash Drive",       "enable.svg",  self.enableFlashDrive,       "success"),
            ("DFU Mode",                 "wrench.svg",  self.dfuMode,                "info"),
            ("Disable Flash Drive",      "disable.svg", self.disableFlashDrive,      "primary"),
        ]

        for text, icon_name, callback, color_key in buttons_data:
            btn = QtWidgets.QPushButton(text)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                              QtWidgets.QSizePolicy.Policy.Fixed)
            btn.setFixedHeight(48)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

            icon_path = _ICON_DIR / icon_name
            if icon_path.exists():
                btn.setIcon(QtGui.QIcon(str(icon_path)))
                btn.setIconSize(QtCore.QSize(20, 20))

            # Gradient + glass top-edge from BUTTON_STYLE. Force 9 pt so the
            # longest label ("Load Hardware Definition") fits the 252 px
            # dialog without clipping.
            btn.setStyleSheet(_BS.format(
                color=_C[color_key],
                hover_color=_C[color_key + '_hover'],
            ) + " QPushButton { font-size: 9pt; padding: 0 4px; }")
            btn.setIconSize(QtCore.QSize(16, 16))

            btn.clicked.connect(callback)
            button_column.addWidget(btn)

        layout.addLayout(button_column)

    def _eligible_widgets(self):
        # Every connected box, including running ones; a running box is
        # excluded from configuration (row visible, checkbox disabled).
        return [w for w in self.main_window.get_all_setup_widgets()
                if w.is_connected]

    def _middle_cell(self, row, widget):
        running = _is_running(widget)
        port = widget.com_port or ""
        com_text = (f"{port}  (running)".strip() if port else "running") \
            if running else port
        com_display = QtWidgets.QLineEdit(com_text)
        com_display.setReadOnly(True)
        com_display.setStyleSheet("background:#0b1220; color:#e5e7eb; border:1px solid #1f2937; padding:4px;")
        return com_display

    def _row_checkbox_state(self, widget):
        # Disabled + unchecked for running boxes: configuring
        # (FW/HD/flash/DFU) a live session would corrupt it.
        running = _is_running(widget)
        return (not running, not running,
                "Box is running, stop it before configuring." if running else "")

    def getSelectedBoxes(self):
        # Widgets ride in each row's UserRole, so the selection can't
        # desync from a stale re-enumeration. Running boxes stay out via
        # the disabled-checkbox guard.
        return self._collect_selections()

    def _build_progress_dialog(self, action_name, total_steps):
        from source.gui.dialogs._progress_worker import build_progress_dialog
        dlg = build_progress_dialog(
            self, f"{action_name} progress", total_steps,
            header=f"{action_name} - Live updates",
            with_log=True, width=360)
        return dlg, dlg.log_view, dlg.bar

    def _run_action_with_progress(self, action_name, selected_boxes, action_fn,
                                  post_action=None,
                                  all_finished=None,
                                  status_in_progress=None,
                                  status_ok=None,
                                  status_error=None):
        """Run a board action with a live progress window.

        ``all_finished`` (optional, no args) fires once at the end of the
        batch, after every box's ``action_fn`` + ``post_action`` complete.
        Used e.g. by the HD-load flow to save the project once.
        """
        if status_in_progress is None:
            status_in_progress = f"{action_name}..."
        if status_ok is None:
            status_ok = f"{action_name} OK"
        if status_error is None:
            status_error = f"{action_name} error"

        if not selected_boxes:
            logger.warning(f"No boards selected for {action_name.lower()}")
            return

        if getattr(self, "parallel_checkbox", None) is not None and self.parallel_checkbox.isChecked():
            return self._run_action_with_progress_parallel(
                action_name, selected_boxes, action_fn,
                post_action=post_action,
                all_finished=all_finished,
                status_in_progress=status_in_progress,
                status_ok=status_ok,
                status_error=status_error,
            )

        progress_dialog, log_view, progress_bar = self._build_progress_dialog(
            action_name, len(selected_boxes))
        log_view.append(f"Starting {action_name.lower()} for {len(selected_boxes)} board(s)...")
        QtWidgets.QApplication.processEvents()

        for setup_widget in selected_boxes:
            setup_number = getattr(setup_widget, 'setup_number', '?')
            box_label = f"Box {setup_number}"
            log_view.append(f"{box_label}: executing {action_name.lower()}...")

            self._updateLiveStatus(setup_number, f"{action_name}...")
            self._setBoxStatusLine(setup_widget, status_in_progress, "neutral")

            QtWidgets.QApplication.processEvents()
            # PyboardError subclasses BaseException, catch it so
            # framework/HD/DFU failures all surface here.
            try:
                action_fn(setup_widget)
                if post_action is not None:
                    try:
                        post_action(setup_widget)
                    except BaseException as e:
                        logger.debug("post_action error for %s: %s", box_label, e)
                log_view.append(f"{box_label}: {action_name} completed")
                self._updateLiveStatus(setup_number, f"{action_name} completed")
                self._setBoxStatusLine(setup_widget, status_ok, "ready")
            except BaseException as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                err_text = str(e)
                if hasattr(e, "args") and e.args:
                    first = e.args[0]
                    if isinstance(first, str) and first:
                        err_text = first
                logger.error(f"Error during {action_name} for {box_label}: {err_text}")
                log_view.append(f"{box_label}: failed - {err_text}")
                self._updateLiveStatus(setup_number, f"{action_name} failed: {err_text[:40]}")
                self._setBoxStatusLine(setup_widget,
                                       f"{status_error}: {err_text[:40]}",
                                       "error")
            progress_bar.setValue(progress_bar.value() + 1)
            QtWidgets.QApplication.processEvents()

        log_view.append("All operations finished.")
        QtWidgets.QApplication.processEvents()
        progress_bar.setValue(progress_bar.maximum())

        # Whole-batch finaliser (e.g. save_project after HD load).
        if all_finished is not None:
            try:
                all_finished()
            except Exception as e:
                logger.warning("all_finished hook raised: %s", e)

        QtCore.QTimer.singleShot(400, progress_dialog.accept)
        progress_dialog.exec()
        self._finish_and_close()

    def _run_action_with_progress_parallel(self, action_name, selected_boxes, action_fn,
                                          post_action=None,
                                          all_finished=None,
                                          status_in_progress=None,
                                          status_ok=None,
                                          status_error=None):
        """Parallel variant. Threading lives in
        ``dialogs/_progress_worker.run_box_actions_in_parallel``; this method
        supplies the per-action UI hooks (status lines, log append,
        post_action)."""
        if status_in_progress is None:
            status_in_progress = f"{action_name}..."
        if status_ok is None:
            status_ok = f"{action_name} OK"
        if status_error is None:
            status_error = f"{action_name} error"

        progress_dialog, log_view, progress_bar = self._build_progress_dialog(
            action_name, len(selected_boxes))
        log_view.append(
            f"Starting {action_name.lower()} for {len(selected_boxes)} board(s) in parallel..."
        )
        QtWidgets.QApplication.processEvents()

        # Pre-queue UI: log line + per-box status. The helper advances the
        # progress bar + accepts the dialog when the last worker reports.
        def _on_queued(bw):
            n = getattr(bw, 'setup_number', '?')
            log_view.append(f"Box {n}: queued {action_name.lower()}...")
            self._updateLiveStatus(n, f"{action_name} queued")
            self._setBoxStatusLine(bw, status_in_progress, "neutral")

        def _on_progress(bw, msg, end):
            n = getattr(bw, 'setup_number', '?')
            log_view.append(f"Box {n}: {msg}")
            fn = getattr(bw, "append_status", None) or getattr(bw, "print_to_log", None)
            if callable(fn):
                try:
                    fn(str(msg), end=end)
                except TypeError:
                    try:
                        fn(str(msg))
                    except Exception:
                        pass
                except Exception:
                    pass

        def _on_success(bw):
            n = getattr(bw, 'setup_number', '?')
            log_view.append(f"Box {n}: {action_name} completed")
            self._updateLiveStatus(n, f"{action_name} completed")
            self._setBoxStatusLine(bw, status_ok, "ready")
            if post_action is not None:
                try:
                    post_action(bw)
                except BaseException as e:
                    if isinstance(e, (KeyboardInterrupt, SystemExit)):
                        raise
                    logger.debug("post_action error for Box %s: %s", n, e)

        def _on_failure(bw, err_text):
            n = getattr(bw, 'setup_number', '?')
            logger.error("Error during %s for Box %s: %s", action_name, n, err_text)
            log_view.append(f"Box {n}: failed - {err_text}")
            self._updateLiveStatus(n, f"{action_name} failed: {err_text[:40]}")
            self._setBoxStatusLine(bw, f"{status_error}: {err_text[:40]}", "error")

        def _worker_body(bw, signals):
            # Log "started" before running action_fn so per-box progress
            # shows even for fast actions.
            signals.progress.emit(
                {"box": getattr(bw, "setup_number", None),
                 "msg": f"executing {action_name.lower()}...",
                 "end": "\n"})
            action_fn(bw)

        run_box_actions_in_parallel(
            setup_widgets=selected_boxes,
            worker_body=_worker_body,
            on_success=_on_success,
            on_failure=_on_failure,
            on_progress=_on_progress,
            on_queued=_on_queued,
            progress_dialog=progress_dialog,
            progress_bar=progress_bar,
        )
        log_view.append("All operations finished.")

        # Whole-batch finaliser (e.g. save_project after HD load).
        if all_finished is not None:
            try:
                all_finished()
            except Exception as e:
                logger.warning("all_finished hook raised: %s", e)

        self._finish_and_close()

    def _updateLiveStatus(self, setup_number, message):
        _update_live_status(self.main_window, setup_number, message)

    def _setBoxStatusLine(self, setup_widget, message, kind):
        try:
            if hasattr(setup_widget, "set_status"):
                setup_widget.set_status(message, kind)
        except Exception as e:
            logger.debug(f"Could not set status line: {e}")

    def _finish_and_close(self):
        self.accept()

    def _invalidate_uploaded_task(self, setup_widget, *, close_pycboard=False):
        """Clear cached upload state after FW/HD/Flash/DFU actions."""
        try:
            setup_widget.task_uploaded = False
            for attr in ("task_file_hash", "last_task_hash"):
                try:
                    setattr(setup_widget, attr, None)
                except AttributeError:
                    pass
            for attr in ("_mcu_variables", "_mcu_events", "_mcu_coordinates"):
                try:
                    val = getattr(setup_widget, attr, None)
                    if isinstance(val, dict):
                        val.clear()
                    elif isinstance(val, list):
                        val.clear()
                except (AttributeError, TypeError):
                    pass
            btn = getattr(setup_widget, "upload_button", None)
            if btn is not None:
                try:
                    btn.setText("Upload")
                except Exception:
                    pass
            if close_pycboard:
                pyc = getattr(setup_widget, "pycboard", None)
                if pyc is not None:
                    # PyboardError subclasses BaseException; catch it so we
                    # don't leak a half-closed serial port.
                    try:
                        pyc.close()
                    except BaseException as e:
                        if isinstance(e, (KeyboardInterrupt, SystemExit)):
                            raise
                        logger.debug("pyc.close(box %s): %s",
                                     getattr(setup_widget, "setup_number", "?"), e)
                setup_widget.pycboard = None
                setup_widget.framework_running = False
                # Unregisters from the MCU registry AND detaches the
                # pipeline's pycboard reference, the camera worker must
                # not push events to the closed port.
                try:
                    setup_widget._unregister_from_main_window()
                except Exception:
                    pass
            try:
                setup_widget._update_button_states()
            except Exception:
                pass
        except Exception as e:
            logger.debug("invalidate_uploaded_task(box %s): %s",
                         getattr(setup_widget, "setup_number", "?"), e)

    def uploadFramework(self):
        selected_boxes = self.getSelectedBoxes()
        logger.info(f"Uploading framework to {len(selected_boxes)} board(s)...")

        def _do_upload(setup_widget):
            if hasattr(setup_widget, 'pycboard') and setup_widget.pycboard:
                logger.info(f"Loading framework to Box {setup_widget.setup_number}")
                setup_widget.pycboard.load_framework()
            else:
                raise RuntimeError("No pycboard connection")

        def _offer_hardware_definition():
            """Loading the framework leaves the board without its pin map, so
            offer to send it straight after rather than leaving the operator to
            remember. Declining is a normal answer: a board being re-flashed
            before its wiring is decided has no hardware definition to send
            yet."""
            n = len(selected_boxes)
            reply = QtWidgets.QMessageBox.question(
                self, "Load hardware definition?",
                f"Framework loaded to {n} board(s).\n\n"
                f"The hardware definition has to be sent again after a "
                f"framework load. Select and upload it now?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.Yes)
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                self.loadHardwareDefinition()

        self._run_action_with_progress(
            "Upload Framework", selected_boxes, _do_upload,
            post_action=lambda bw: self._invalidate_uploaded_task(bw),
            all_finished=_offer_hardware_definition,
            status_in_progress="FW loading...",
            status_ok="FW loaded",
            status_error="FW load error",
        )

    def enableFlashDrive(self):
        selected_boxes = self.getSelectedBoxes()
        logger.info(f"Enabling flash drive on {len(selected_boxes)} board(s)...")

        def _do_enable(setup_widget):
            if hasattr(setup_widget, 'pycboard') and setup_widget.pycboard:
                logger.info(f"Enabling flash drive on Box {setup_widget.setup_number}")
                setup_widget.pycboard.enable_mass_storage()
            else:
                raise RuntimeError("No pycboard connection")

        self._run_action_with_progress(
            "Enable Flash Drive", selected_boxes, _do_enable,
            post_action=lambda bw: self._invalidate_uploaded_task(
                bw, close_pycboard=True),
            status_in_progress="Flash ON...",
            status_ok="Flash drive ON",
            status_error="Flash ON error",
        )

    def disableFlashDrive(self):
        selected_boxes = self.getSelectedBoxes()
        logger.info(f"Disabling flash drive on {len(selected_boxes)} board(s)...")

        def _do_disable(setup_widget):
            if hasattr(setup_widget, 'pycboard') and setup_widget.pycboard:
                logger.info(f"Disabling flash drive on Box {setup_widget.setup_number}")
                setup_widget.pycboard.disable_mass_storage()
            else:
                raise RuntimeError("No pycboard connection")

        self._run_action_with_progress(
            "Disable Flash Drive", selected_boxes, _do_disable,
            post_action=lambda bw: self._invalidate_uploaded_task(
                bw, close_pycboard=True),
            status_in_progress="Flash OFF...",
            status_ok="Flash drive OFF",
            status_error="Flash OFF error",
        )

    def _snapshot_capture_hd(self, hw_file, setup_id):
        """Delegate to the module-level helper."""
        _snapshot_capture_hd_source(self.main_window, hw_file, setup_id)

    def loadHardwareDefinition(self):
        selected_boxes = self.getSelectedBoxes()
        if not selected_boxes:
            logger.warning("No boards selected for hardware definition load")
            return

        hw_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Hardware Definition",
            str(Path('hardware_definitions')), "Python Files (*.py)"
        )
        if not hw_file:
            return

        logger.info(f"Loading hardware definition to {len(selected_boxes)} board(s)...")

        def _do_load(setup_widget):
            if hasattr(setup_widget, 'pycboard') and setup_widget.pycboard:
                logger.info(f"Loading hardware definition to Box {setup_widget.setup_number}")
                setup_widget.pycboard.load_hardware_definition(hw_file)
            else:
                raise RuntimeError("No pycboard connection")

        def _post(bw):
            self._invalidate_uploaded_task(bw)
            # Snapshot capture: HD content → sources store. Runs once per
            # box on the main thread (post_action fires after the worker
            # completes), off the upload critical path.
            # _snapshot_capture_hd populates
            # ``cfg.setup_config.boxes[i].init_hw_def`` on _active_config;
            # we mirror that ref onto the box widget (init_hw_def /
            # _hw_def_path) so a later manual Save (which rebuilds cfg from
            # widgets) persists the HD link.
            box_no = getattr(bw, "setup_number", None)
            try:
                self._snapshot_capture_hd(hw_file, box_no)
            except Exception as e:
                logger.warning("universal HD snapshot capture (box %s): %s",
                               box_no, e)
            cfg = getattr(self.main_window, "_active_config", None)
            _hd_carry_ref_to_widget(cfg, bw, hw_file, box_no)

        def _all_finished():
            # The HD FileRef rides on each box widget (set in _post) and the
            # .py is committed to the snapshot store, so it transfers into
            # experiment_config.json on the next Save. Silent-overwrite only
            # when a named project already exists; skip the save on a draft
            # so we don't force a first-save dialog mid hardware-setup.
            mw = self.main_window
            cfg = getattr(mw, "_active_config", None) if mw is not None else None
            if mw is None or cfg is None:
                return
            if _hd_autosave_is_draft(mw):
                if hasattr(mw, "statusbar"):
                    try:
                        mw.statusbar.showMessage(
                            f"HD loaded ({len(selected_boxes)} box(es)), "
                            "save the project to keep it", 3000)
                    except Exception:
                        pass
                return
            try:
                from source.gui.project_workflow import save_project
                save_project(mw, cfg)
                logger.info(
                    "Project config auto-saved after HD load "
                    "(%d box(es))", len(selected_boxes))
                if hasattr(mw, "statusbar"):
                    try:
                        mw.statusbar.showMessage(
                            f"HD loaded + project saved ({len(selected_boxes)} box(es))",
                            3000)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("Auto-save after HD load failed: %s", e)

        self._run_action_with_progress(
            "Load Hardware Definition", selected_boxes, _do_load,
            post_action=_post,
            all_finished=_all_finished,
            status_in_progress="HD loading...",
            status_ok="HD loaded",
            status_error="HD load error",
        )

    def dfuMode(self):
        selected_boxes = self.getSelectedBoxes()
        logger.info(f"Entering DFU mode on {len(selected_boxes)} board(s)...")

        def _do_dfu(setup_widget):
            if hasattr(setup_widget, 'pycboard') and setup_widget.pycboard:
                logger.info(f"Entering DFU mode on Box {setup_widget.setup_number}")
                setup_widget.pycboard.DFU_mode()
            else:
                raise RuntimeError("No pycboard connection")

        self._run_action_with_progress(
            "DFU Mode", selected_boxes, _do_dfu,
            post_action=lambda bw: self._invalidate_uploaded_task(
                bw, close_pycboard=True),
            status_in_progress="DFU mode...",
            status_ok="DFU OK",
            status_error="DFU error",
        )


# =============================================================================
#  UniversalUploadDialog, multi-box task upload
# =============================================================================


class UniversalUploadDialog(_BoxTableDialogBase):
    """Dialog for selecting boxes and uploading a task with progress."""

    WINDOW_TITLE = "Upload Task"
    MIDDLE_HEADER = "COM Port"
    EMPTY_TEXT = "No connected boards, connect first."
    # Fixed width so the parallel-checkbox row and the task picker
    # cannot widen the dialog past its intended footprint.
    FIXED_WIDTH = 224
    COL0_WIDTH = 60
    COL2_WIDTH = 40
    ROW_HEIGHT = 26

    def _build_buttons(self, layout):
        self.selected_task_path = None
        self.selected_task_name = None

        self.parallel_checkbox = QtWidgets.QCheckBox("Parallel (faster)")
        self.parallel_checkbox.setChecked(_flag_env("PYBEHAVIORLAB_MCU_PARALLEL", True))
        self.parallel_checkbox.setToolTip(
            "Run selected boxes concurrently. Uses worker threads for per-box serial IO "
            "and updates the GUI from the main thread."
        )
        layout.addWidget(self.parallel_checkbox)

        upload_btn = QtWidgets.QPushButton("Select && Upload Task")
        from source.gui.styles import BUTTON_STYLE as _BS, COLORS as _C
        upload_btn.setStyleSheet(_BS.format(
            color=_C['info'], hover_color=_C['info_hover']))
        upload_btn.setMinimumHeight(40)
        upload_btn.clicked.connect(self._on_action)
        upload_btn.setDefault(True)        # Enter → Select & Upload Task
        upload_btn.setAutoDefault(True)
        layout.addWidget(upload_btn)

    def _eligible_widgets(self):
        return [w for w in self.main_window.get_all_setup_widgets()
                if w.is_connected and not getattr(w, 'framework_running', False)]

    def _middle_cell(self, row, widget):
        com_display = QtWidgets.QLineEdit(widget.com_port)
        com_display.setReadOnly(True)
        com_display.setStyleSheet("QLineEdit { padding: 2px; background:#0f172a; color:#e5e7eb; border:1px solid #1f2937; } QLineEdit:disabled { background:#d5d5d5; color:#888888; border:1px solid #c0c0c0; }")
        return com_display

    def _on_action(self):
        try:
            selections = [(widget, self.table.cellWidget(row, 1))
                          for row, widget in self._selected_rows()]
            if not selections:
                logger.warning("No boxes selected for upload")
                QtWidgets.QMessageBox.warning(self, "No Selection",
                                              "Please select at least one box.")
                return

            tasks_dir = Path("tasks")
            start_dir = str(tasks_dir.absolute()) if tasks_dir.exists() else ""

            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select Task File", start_dir,
                "Python Files (*.py);;All Files (*)")

            if not file_path:
                return

            file_path = Path(file_path)

            try:
                if tasks_dir.exists():
                    rel_path = file_path.relative_to(tasks_dir.absolute())
                    rel_task = str(rel_path.with_suffix("")).replace("\\", "/")
                else:
                    rel_task = file_path.stem
            except ValueError:
                rel_task = file_path.stem
                logger.warning(f"Task file is outside tasks/ directory: {file_path}")

            logger.info(f"Selected task file: {rel_task} ({file_path})")
            self._uploadToBoxes(selections, rel_task)

        except Exception as e:
            logger.error(f"Error in select and upload: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "Error",
                                           f"Failed to upload task: {str(e)}")

    def _uploadToBoxes(self, selections, rel_task):
        from source.gui.dialogs._progress_worker import build_progress_dialog
        progress_dialog = build_progress_dialog(
            self, "Upload Task Progress", len(selections),
            header="Uploading task...", with_detail=True)
        detail, bar = progress_dialog.detail, progress_dialog.bar

        task_display = rel_task.split('/')[-1] if '/' in rel_task else rel_task

        if getattr(self, "parallel_checkbox", None) is not None and self.parallel_checkbox.isChecked():
            return self._uploadToBoxes_parallel(selections, rel_task, task_display, progress_dialog, detail, bar)

        for idx, (setup_widget, _) in enumerate(selections, start=1):
            QtWidgets.QApplication.processEvents()
            self._updateLiveStatus(setup_widget.setup_number,
                                   f"Uploading task '{task_display}'...")
            try:
                setup_widget.upload_task(rel_task)
                self._updateLiveStatus(setup_widget.setup_number,
                                       f"Task '{task_display}' uploaded")
            except (PyboardError, SerialException) as e:
                err = str(e)[:120]
                self._updateLiveStatus(setup_widget.setup_number,
                                       f"Upload failed (board error): {err}")
                logger.error(f"Box {setup_widget.setup_number}: upload board error: {e}")
            except Exception as e:
                err = str(e)[:120]
                self._updateLiveStatus(setup_widget.setup_number,
                                       f"Upload failed: {err}")
                logger.error(f"Box {setup_widget.setup_number}: upload error: {e}")

            detail.setText(f"Box {setup_widget.setup_number}")
            bar.setValue(idx)
            QtWidgets.QApplication.processEvents()

        bar.setValue(len(selections))
        QtCore.QTimer.singleShot(400, progress_dialog.accept)
        progress_dialog.exec()
        self.accept()

    def _uploadToBoxes_parallel(self, selections, rel_task, task_display,
                                progress_dialog, detail, bar):
        """Parallel task-upload. Threading lives in
        ``dialogs/_progress_worker.run_box_actions_in_parallel``; this method
        wires the upload-specific UI hooks (status flips, per-box
        ``_after_mcu_upload_success`` / ``initialise_API`` /
        ``_after_upload_clicked`` chain)."""
        logger.info("Parallel task upload: %d board(s)", len(selections))

        # The helper takes a flat list of box widgets; ``selections`` here
        # is a list of ``(box_widget, _)`` tuples.
        setup_widgets = [bw for bw, _ in selections]

        def _on_queued(bw):
            n = getattr(bw, "setup_number", "?")
            self._updateLiveStatus(n, f"Uploading task '{task_display}'...")
            try:
                bw.set_status(f"Uploading: {task_display}", "neutral")
            except Exception:
                pass

        def _on_success(bw):
            n = getattr(bw, "setup_number", "?")
            self._updateLiveStatus(n, f"Task '{task_display}' uploaded")
            # Mirror the task name into the per-box task selector so the box
            # looks the same as picking the task from the per-box menu.
            try:
                combo = getattr(bw, "task_combo", None)
                if combo is not None and hasattr(combo, "setText"):
                    combo.setText(rel_task)
                bw.task_uploaded = True
            except Exception:
                pass
            # The worker uploaded with run_ui_hooks/initialise_api off (no Qt
            # from worker threads), run those hooks here, then the same
            # success tail the per-box Upload button uses.
            try:
                sm_name = Path(rel_task).name
                task_path = (Path("tasks") / Path(rel_task)).with_suffix(".py")
                bw._after_mcu_upload_success(sm_name, task_path)
            except Exception:
                pass
            try:
                bw.initialise_API()
            except Exception:
                pass
            try:
                bw._post_upload_success_ui(rel_task, rel_task)
            except Exception as e:
                logger.warning(
                    "Box %s: post-upload UI tail error: %s",
                    getattr(bw, "setup_number", "?"), e)

        def _on_failure(bw, err_text):
            n = getattr(bw, "setup_number", "?")
            self._updateLiveStatus(n, f"Upload failed: {err_text[:120]}")
            try:
                bw.set_status(f"Upload error: {err_text[:40]}", "error")
            except Exception:
                pass

        def _worker_body(bw, signals):
            # pycboard.setup_state_machine prints progress via self.print;
            # the helper's wrap_with_print_proxy already swapped it for a
            # main-thread-safe signals emit. Call the upload routine and
            # surface its in-band errors list.
            #
            # Reset task_uploaded first so setup_state_machine doesn't reuse
            # a cached state machine and push the wrong task on a second
            # universal upload (the single-box path does this via
            # on_task_changed, which the parallel worker bypasses).
            bw.task_uploaded = False
            errors: list[str] = []
            ok = bw.mcu_upload_task(
                rel_task,
                log=lambda m: errors.append(str(m)),
                run_ui_hooks=False,
                initialise_api=False,
            )
            if not ok:
                raise RuntimeError(errors[-1] if errors else "upload failed")

        run_box_actions_in_parallel(
            setup_widgets=setup_widgets,
            worker_body=_worker_body,
            on_success=_on_success,
            on_failure=_on_failure,
            on_queued=_on_queued,
            progress_dialog=progress_dialog,
            progress_bar=bar,
            detail_label=detail,
        )
        self.accept()

    def _updateLiveStatus(self, setup_number, message):
        _update_live_status(self.main_window, setup_number, message)
