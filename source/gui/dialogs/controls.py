"""Task controls dialogs, single module.

Contains:
  * ConfigSelectionDialog, pick a saved camera/box config to load.
  * ControlsDialog, top-level controls dialog (Standard + Trigger events).
  * StandardControlsTab, per-task variable editor (value / get / set / persist) + note-to-log.
  * VariableSetter, one variable row inside StandardControlsTab.
  * EventsTab, trigger pyControl events.
"""

from PySide6 import QtCore, QtGui, QtWidgets

from source.config.settings import get_setting
from source.gui.styles import COLORS, BUTTON_STYLE
from source.gui.utility import variable_constants
from source.log import get_logger

logger = get_logger()


# Token-driven QSS for the variable-row QLineEdits.
def _dark_line_edit_style() -> str:
    """High-contrast line edit for the Variables / Controls dialog cells.

    White text at weight 600 on opaque dark slate, so values stay legible
    on HDR / contrast-shifted display profiles.
    """
    from source.gui.theme import THEME as _T
    p = _T.palette
    return f"""
        QLineEdit {{
            padding: 1px 6px;
            border: 1px solid {p.surface_border_strong};
            border-radius: {_T.radius.sm}px;
            background-color: rgba(15,23,42,0.95);
            color: #ffffff;
            font: 600 9pt '{_T.font.family}';
            min-height: 18px;
        }}
        QLineEdit:focus {{
            border: 2px solid {p.focus};
            background-color: rgba(15,23,42,1.0);
            color: #ffffff;
        }}
        QLineEdit:hover {{ border-color: {p.text_muted}; }}
        QLineEdit:disabled {{
            color: {p.text_muted};
            background-color: rgba(15,23,42,0.55);
            border-color: {p.surface_border};
        }}
    """


_DARK_LINE_EDIT_STYLE = _dark_line_edit_style()

_GROUPBOX_TITLE_STYLE = """
    QGroupBox {
        border: none;
        margin-top: 14px;
        padding-top: 10px;
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 0px;
        padding: 0 2px;
    }
"""


# =============================================================================
#  config_selection
# =============================================================================

class ConfigSelectionDialog(QtWidgets.QDialog):
    """Dialog for selecting between existing config or creating new"""

    def __init__(self, camera_id, num_boxes, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.num_boxes = num_boxes
        self.selected_action = None
        self.setWindowTitle("Camera Configuration")
        self.setFixedWidth(320)
        self.setModal(True)
        self._build_ui()
        self.applyDarkTheme()

    def applyDarkTheme(self):
        """Shared dark-dialog QSS."""
        from source.gui.style_builders import apply_dialog_theme
        apply_dialog_theme(self)

    def _build_ui(self):
        """Setup the dialog UI"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 16, 20, 16)

        # Message
        from source.gui.theme import THEME as _T
        message = QtWidgets.QLabel(
            f"Camera {self.camera_id} is shared by {self.num_boxes} boxes.\n\n"
            "Do you want to load an existing configuration or create a new one?"
        )
        message.setWordWrap(True)
        message.setStyleSheet(f"font-size: 9pt; padding: 8px; color: {_T.palette.text};")
        layout.addWidget(message)

        # Buttons: info / success gradients; cancel slate secondary.
        _BS, _C = BUTTON_STYLE, COLORS
        from source.gui.style_builders import button_style as _btn_st
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(8)

        select_btn = QtWidgets.QPushButton("Load Existing Config")
        select_btn.setStyleSheet(_BS.format(
            color=_C['info'], hover_color=_C['info_hover']))
        select_btn.setMinimumHeight(36)
        select_btn.clicked.connect(lambda: self.selectAction("select"))
        button_layout.addWidget(select_btn)

        create_btn = QtWidgets.QPushButton("Create New Config")
        create_btn.setStyleSheet(_BS.format(
            color=_C['success'], hover_color=_C['success_hover']))
        create_btn.setMinimumHeight(36)
        create_btn.clicked.connect(lambda: self.selectAction("create"))
        button_layout.addWidget(create_btn)

        layout.addLayout(button_layout)

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setStyleSheet(_btn_st("secondary", height=32))
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)

    def selectAction(self, action):
        """Set selected action and accept dialog"""
        self.selected_action = action
        self.accept()




# =============================================================================
#  controls
# =============================================================================

class ControlsDialog(QtWidgets.QDialog):
    """Controls dialog with Standard and Custom tabs."""

    def __init__(self, setup_widget, parent=None):
        super().__init__(parent)
        self.setup_widget = setup_widget
        task = getattr(setup_widget, "_uploaded_task_rel", None) or "no task"
        self.setWindowTitle(
            f"Box {setup_widget.setup_number}, Controls, task: {task}"
        )
        # No fixed size / scroll, dialog grows to fit its content.

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.standard_tab = StandardControlsTab(setup_widget, parent=self)
        self.events_tab = EventsTab(setup_widget, parent=self)

        self.tab_widget.addTab(self.standard_tab, "Standard")
        self.tab_widget.addTab(self.events_tab, "Trigger events")

        layout.addWidget(self.tab_widget)

        # No bottom Close button, Escape / window-frame ✕ close the dialog.

        # Cap the dialog at the screen's work area before the first show so
        # the initial geometry already fits (no clamp warning / post-show
        # resize). The variables panel scrolls inside its own capped viewport.
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            self.setMaximumSize(max(self.minimumWidth(), avail.width() - 40),
                                max(self.minimumHeight(), avail.height() - 80))
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        # adjustSize after tabs paint once; bounded by the maximumSize above.
        QtCore.QTimer.singleShot(0, self.adjustSize)

        # Refresh enabled-states whenever the framework starts/stops so the
        # Events tab toggles in real time while the dialog is open. The
        # signals live on BoxControlWidget; maze SetupWidget lacks them, so
        # guard.
        for sig_name in ("framework_started_signal", "framework_stopped_signal"):
            sig = getattr(setup_widget, sig_name, None)
            if sig is not None:
                try:
                    sig.connect(self._refresh_all)
                except Exception:
                    pass

    def keyPressEvent(self, event):
        """Never let a bare Enter/Return close the dialog.

        A focused value field / note box consumes Return first (firing its
        Set / Add-note); this only catches Return when focus is elsewhere
        and swallows it instead of falling through to the default-button →
        close behaviour.
        """
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    def _refresh_all(self, *_args):
        """Re-evaluate enabled state across all tabs."""
        try:
            self.standard_tab.refresh_enabled_states()
            self.events_tab.refresh_enabled_states()
            board = getattr(self.setup_widget, "pycboard", None)
            events_ready = bool(
                board and board.framework_running and getattr(board, "sm_info", None)
                and board.sm_info.events
            )
            self.tab_widget.setTabEnabled(1, events_ready)
        except Exception as e:
            logger.warning("ControlsDialog refresh failed: %s", e)

    def showEvent(self, event):
        super().showEvent(event)
        # Guarantee a single legend bar renders, whatever the runtime path.
        legends = self.findChildren(QtWidgets.QWidget, "vars_legend_bar")
        for extra in legends[1:]:
            extra.setParent(None)
            extra.deleteLater()
        self.standard_tab.refresh_enabled_states()
        self.events_tab.refresh_enabled_states()
        # Disable Events tab when framework not running.
        board = getattr(self.setup_widget, "pycboard", None)
        events_ready = bool(board and board.framework_running and getattr(board, "sm_info", None) and board.sm_info.events)
        self.tab_widget.setTabEnabled(1, events_ready)


# Dark glass surface for the Variables panel; colours derive from
# the theme palette so it matches the dark-only app.
def _vars_panel_style() -> str:
    from source.gui.theme import THEME as _T
    p = _T.palette
    return f"""
        QGroupBox#variables_panel {{
            background-color: {p.surface};
            border: 1px solid {p.surface_border_strong};
            border-radius: {_T.radius.md}px;
            margin-top: 14px;
            padding-top: 14px;
            color: {p.text};
            font-weight: 700;
        }}
        QGroupBox#variables_panel::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            padding: 0 8px;
            background: {p.bg};
            color: {p.text_muted};
            font: 700 {_T.font.caption_pt}pt '{_T.font.family}';
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }}
        QGroupBox#variables_panel QLabel {{
            color: {p.text};
            background: transparent;
            border: none;
        }}
        QGroupBox#variables_panel QLineEdit {{
            background-color: rgba(255,255,255,0.04);
            color: {p.text};
            border: 1px solid {p.surface_border_strong};
            border-radius: {_T.radius.sm}px;
            padding: 4px 8px;
            min-height: 20px;
            font-size: 9pt;
        }}
        QGroupBox#variables_panel QLineEdit:focus {{
            border: 2px solid {p.focus};
            background-color: rgba(255,255,255,0.06);
        }}
        QGroupBox#variables_panel QCheckBox {{
            background: transparent;
            color: {p.text};
        }}
        QGroupBox#variables_panel QScrollArea {{
            background: transparent;
            border: none;
        }}
    """


_VARS_PANEL_STYLE = _vars_panel_style()

# Banner between the groupbox title and the grid: mint (fresh task, all
# reset by default) or amber (sidecar present, hash differs).
_VARS_BANNER_FRESH = (
    "background: rgba(16,185,129,0.18); border-left: 3px solid #34d399;"
    " color: #6ee7b7;"
    " padding: 6px 10px; font-size: 9pt; border-radius: 3px;")
_VARS_BANNER_INFO = (
    "background: rgba(245,158,11,0.18); border-left: 3px solid #fbbf24;"
    " color: #fbbf24;"
    " padding: 6px 10px; font-size: 9pt; border-radius: 3px;")

_VARS_HEADER_STYLE = (
    "color: #cbd5e1; background: transparent; border: none;"
    " font-weight: 700; font-size: 8pt;"
    " border-bottom: 1px solid rgba(255,255,255,0.10); padding-bottom: 4px;")


class StandardControlsTab(QtWidgets.QWidget):
    """Standard controls: notes + per-task variables.

    Trigger events live in their own tab (``EventsTab``); this tab is just
    Note-to-log + Variables.
    """

    def __init__(self, setup_widget, parent=None):
        super().__init__(parent)
        self.setup_widget = setup_widget
        self.board = getattr(setup_widget, "pycboard", None)
        # Effective spec list + autosave coalescer; populated by
        # ``_load_specs_for_current_task`` on re-bind. Project scope decides
        # whether a Persist toggle writes the project store or the sidecar.
        self._specs: list = []
        self._task_py_path = None
        self._project_dir = None
        self._task_family = ""
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(750)
        self._save_timer.timeout.connect(self._flush_save)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self._line_edit_style = _DARK_LINE_EDIT_STYLE

        # Notes row (single line + button). The note is written via
        # ``data_logger.print_message`` which only goes anywhere when the
        # data file is open, i.e. while a recording is in progress,
        # so flag that to the user with a small hint label.
        self.notes_groupbox = QtWidgets.QGroupBox("Add note to log")
        self.notes_groupbox.setStyleSheet(_GROUPBOX_TITLE_STYLE)
        notes_box_layout = QtWidgets.QVBoxLayout(self.notes_groupbox)
        notes_box_layout.setContentsMargins(6, 6, 6, 6)
        notes_box_layout.setSpacing(2)
        notes_layout = QtWidgets.QHBoxLayout()
        self.notes_textbox = QtWidgets.QLineEdit()
        self.notes_textbox.setFont(QtGui.QFont("Courier New", get_setting("GUI", "log_font_size")))
        self.notes_textbox.setStyleSheet(self._line_edit_style)
        self.notes_textbox.setPlaceholderText("Note text…")
        self.notes_textbox.setMinimumHeight(34)   # taller, easier to type into
        self.notes_textbox.returnPressed.connect(self.add_note)
        # Gray Add-note button; the only coloured buttons in this panel are
        # Get / Set on the variable rows.
        self.note_button = QtWidgets.QPushButton("Add note")
        self.note_button.setFixedWidth(110)
        self.note_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        # Secondary slate; height matches the note line edit so the row
        # reads as one band.
        from source.gui.style_builders import button_style as _btn_note
        self.note_button.setStyleSheet(_btn_note("secondary", height=34))
        self.note_button.setFixedHeight(34)
        self.note_button.clicked.connect(self.add_note)
        notes_layout.addWidget(self.notes_textbox)
        notes_layout.addWidget(self.note_button)
        notes_box_layout.addLayout(notes_layout)
        from source.gui.theme import THEME as _T_nh
        self.notes_hint_label = QtWidgets.QLabel("Note is only logged while a recording is running")
        self.notes_hint_label.setStyleSheet(
            f"font-size: 8pt; font-style: italic; color: {_T_nh.palette.text_dim};"
            " border: none; background-color: transparent;")
        notes_box_layout.addWidget(self.notes_hint_label)

        # Variables groupbox (scrollable). Five columns:
        # 0 Name  1 Value  2 Get  3 Set  4 Persist (chk)
        # Row 0 is the optional banner; row 1 is the column header;
        # variables start at row 2.
        self.variables_groupbox = QtWidgets.QGroupBox("Variables (per-task config)")
        self.variables_groupbox.setObjectName("variables_panel")
        self.variables_groupbox.setStyleSheet(_VARS_PANEL_STYLE)
        outer = QtWidgets.QVBoxLayout(self.variables_groupbox)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)
        # Banner row, hidden by default; populated by _set_banner.
        self.vars_banner = QtWidgets.QLabel("")
        self.vars_banner.setWordWrap(True)
        self.vars_banner.setVisible(False)
        outer.addWidget(self.vars_banner)
        # Grid lives inside its own container so it sits on the groupbox
        # background cleanly.
        grid_host = QtWidgets.QWidget()
        grid_host.setStyleSheet("background: transparent;")
        self.grid_layout = QtWidgets.QGridLayout(grid_host)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setHorizontalSpacing(6)
        self.grid_layout.setVerticalSpacing(1)
        # Only the Value column grows; Name sizes to content, Get/Set/Persist
        # stay at their button/checkbox width. The value field (Expanding,
        # below) fills this column so there's no dead gap around it.
        for c in range(5):
            self.grid_layout.setColumnStretch(c, 1 if c == 1 else 0)
        # Column headers, row 1 (row 0 reserved for the banner).
        for col_idx, txt in enumerate(
                ("Name", "Value (live)", "Get", "Set", "Persist")):
            hl = QtWidgets.QLabel(txt)
            hl.setStyleSheet(_VARS_HEADER_STYLE)
            if col_idx >= 2:
                hl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.grid_layout.addWidget(hl, 1, col_idx)
        # Trailing stretch row keeps the rows packed at the top when a task
        # has few variables (so the scroll viewport doesn't space them out).
        self.grid_layout.setRowStretch(9999, 1)
        # Scroll the variable list inside a capped viewport so a many-variable
        # task doesn't grow the dialog taller than the screen.
        self.vars_scroll = QtWidgets.QScrollArea(self.variables_groupbox)
        self.vars_scroll.setWidget(grid_host)
        self.vars_scroll.setWidgetResizable(True)
        self.vars_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.vars_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Default cap; ControlsDialog shrinks it on a short screen so the whole
        # dialog fits the work area (see ControlsDialog.__init__).
        self.vars_scroll.setMaximumHeight(320)
        self.vars_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }")
        outer.addWidget(self.vars_scroll)

        # Bottom legend bar, small reminders the user can glance at:
        # "hw_* hardware variable (highlighted)" with a peach hw_* chip
        # and a Persist hint.
        self.legend_bar = QtWidgets.QWidget()
        self.legend_bar.setObjectName("vars_legend_bar")
        legend_layout = QtWidgets.QHBoxLayout(self.legend_bar)
        legend_layout.setContentsMargins(4, 2, 4, 2)
        legend_layout.setSpacing(12)
        hw_chip = QtWidgets.QLabel("hw_*")
        hw_chip.setStyleSheet(
            "color: #b45309; background: #fde68a;"
            " border-radius: 3px; padding: 1px 6px;"
            " font-weight: 600; font-size: 8pt;"
        )
        # Both legend labels use an Ignored horizontal size policy so they CLIP
        # on one line when the dialog is narrowed, they neither force a wide
        # minimum width nor wrap into extra lines (wrapping inflated the dialog's
        # minimum HEIGHT past the screen cap → geometry warning + squish). Full
        # text stays available as a tooltip.
        _ELIDE = QtWidgets.QSizePolicy.Policy.Ignored
        hw_text = QtWidgets.QLabel("hardware variable (highlighted)")
        hw_text.setToolTip(hw_text.text())
        hw_text.setSizePolicy(_ELIDE, QtWidgets.QSizePolicy.Policy.Preferred)
        hw_text.setStyleSheet(
            "color: #6b7280; background: transparent;"
            " border: none; font-size: 8pt;"
        )
        persist_text = QtWidgets.QLabel(
            "Persist ✓, saves final value across sessions; "
            "unticked resets to the task default each run"
        )
        persist_text.setToolTip(persist_text.text())
        persist_text.setSizePolicy(_ELIDE, QtWidgets.QSizePolicy.Policy.Preferred)
        persist_text.setStyleSheet(
            "color: #6b7280; background: transparent;"
            " border: none; font-size: 8pt;"
        )
        legend_layout.addWidget(hw_chip)
        legend_layout.addWidget(hw_text)
        legend_layout.addSpacing(12)
        legend_layout.addWidget(persist_text)
        legend_layout.addStretch(1)

        layout.addWidget(self.notes_groupbox)
        layout.addWidget(self.variables_groupbox)
        layout.addWidget(self.legend_bar)
        # No bottom stretch, let the panel close around its content.

        self.refresh_variables()

    # ---- banner helpers --------------------------------------------------

    def _set_banner(self, kind: str, message: str) -> None:
        """Populate the banner. ``kind`` is 'fresh' (green) / 'info'
        (amber) / 'none' (hide)."""
        if kind == "none" or not message:
            self.vars_banner.setVisible(False)
            self.vars_banner.setText("")
            return
        style = _VARS_BANNER_FRESH if kind == "fresh" else _VARS_BANNER_INFO
        self.vars_banner.setStyleSheet(style)
        self.vars_banner.setText(message)
        self.vars_banner.setVisible(True)

    def refresh_variables(self):
        # Compute the target variable list first so we can skip the expensive
        # destroy+rebuild when nothing changed (``refresh_variables`` runs on
        # every showEvent). Only rebuild when the variable SET changes (task
        # uploaded / re-uploaded, board (dis)connects); otherwise just refresh
        # the live values on the existing rows.
        board = self.board
        if board and getattr(board, "sm_info", None):
            variables = board.sm_info.variables
            # Sort: hw_* first, then alphabetical, matches the mockup.
            names = sorted(
                (n for n in variables
                 if not n.endswith("___")
                 and n != "custom_controls_dialog"
                 and n != "api_class"),
                key=lambda n: (0 if n.startswith("hw_") else 1, n),
            )
        else:
            names = None  # no board / no task

        rows = self.findChildren(VariableSetter)
        if names == getattr(self, "_built_var_names", "__unset__") and rows:
            # Unchanged set, keep the widgets, just refresh displayed values.
            if names:
                for r in rows:
                    if r.v_name in variables:
                        r.value_str.setText(repr(variables[r.v_name]))
            return

        # The set changed, drop the old rows (rows 2..N; banner + header stay).
        for setter in rows:
            setter.setParent(None)
            setter.deleteLater()
        items_to_drop = []
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if item is None:
                continue
            row, col, *_ = self.grid_layout.getItemPosition(i)
            if row >= 2:
                items_to_drop.append(item.widget())
        for w in items_to_drop:
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._built_var_names = names

        if not board or not getattr(board, "sm_info", None):
            self._set_banner("none", "")
            return

        # Resolve the task .py path for the uploaded task and load specs
        # from its sidecar JSON. Absent sidecar → empty list → every
        # variable renders with Persist unticked (MCU task_file is the
        # source of truth).
        self._load_specs_for_current_task()

        control_row = 2   # row 0 = banner host (outer), row 1 = header
        for v_name in names:
            VariableSetter(
                v_name, variables[v_name], control_row, self,
                specs=self._specs, on_change=self._schedule_save,
            )
            control_row += 1

    def _load_specs_for_current_task(self) -> None:
        """Resolve the task .py path and the box's EFFECTIVE persistent specs:
        the per-task template ``<task>.variables.json`` overlaid by the loaded
        project's own flags (project wins). Caches the project scope so a
        Persist toggle routes to the right store. Sets the banner too."""
        from source.config import task_variables as _tvf
        bw = self.setup_widget
        task_rel = getattr(bw, "_uploaded_task_rel", None)
        # Project scope for layered flag read/write: project-specific when a
        # project is loaded, else the per-task sidecar template.
        mw = getattr(bw, "main_window", None)
        self._project_dir = getattr(mw, "_active_project_dir", None) if mw else None
        try:
            self._task_family = bw._persistent_task_folder() if bw else ""
        except Exception:
            self._task_family = ""
        if not task_rel:
            # No task uploaded yet (smoke-test path), show nothing.
            self._specs = []
            self._task_py_path = None
            self._set_banner("none", "")
            return
        self._task_py_path = _tvf.task_py_path_from_relname(task_rel)
        hash_ok = _tvf.hash_matches(self._task_py_path)
        self._specs = _tvf.resolve_specs(
            self._task_py_path, self._project_dir, self._task_family)
        scoped = "this project" if (self._project_dir and self._task_family) \
            else f"{self._task_py_path.with_suffix('').name}.variables.json"
        if hash_ok is None:
            self._set_banner(
                "fresh",
                "Fresh task, every variable resets to its task default each "
                "run. Tick Persist to carry a variable's final value across "
                f"runs. Saves to {scoped}.")
        elif hash_ok is False:
            self._set_banner(
                "info",
                "Task file changed since variables were last saved, "
                "config loaded best-effort; stale rows ignored. "
                f"Saves to {scoped}.")
        else:
            self._set_banner(
                "info", f"Edits autosave to {scoped}.")

    # ---- sidecar autosave (debounced) -----------------------------------

    def _schedule_save(self) -> None:
        """Debounced per-task sidecar write (draft / no-project path);
        coalesces a burst of Persist toggles into one write."""
        if self._task_py_path is None:
            return
        self._save_timer.start()

    def _flush_save(self) -> None:
        if self._task_py_path is None:
            return
        from source.config import task_variables as _tvf
        try:
            _tvf.save_specs(
                self._task_py_path, self._specs,
                task_relname=getattr(self.setup_widget,
                                     "_uploaded_task_rel", None))
        except Exception as e:
            logger.error("variables sidecar save failed: %s", e)

    def write_persist_flag(self, name: str, checked: bool) -> None:
        """Layered write of one variable's Persist flag: to the project store
        when a project is loaded (project-specific), else the per-task sidecar
        template (debounced). Called by each VariableSetter row on toggle."""
        # Keep the in-memory spec list consistent so a re-render shows it.
        spec = None
        for s in self._specs:
            if s.name == name:
                spec = s
                break
        if spec is None:
            from source.config.experiment import BoxVariableSpec
            spec = BoxVariableSpec(name=name)
            self._specs.append(spec)
        spec.persistent = checked
        if self._project_dir and self._task_family:
            from source.config import task_variables as _tvf
            try:
                _tvf.write_project_flags(
                    self._project_dir, self._task_family, {name: checked})
            except Exception as e:
                logger.error("project persist-flag write failed (%s): %s",
                             name, e)
        else:
            self._schedule_save()   # draft / no project → sidecar template

    def refresh_enabled_states(self):
        # Re-pull the live pycboard from the box widget every time so
        # set/get/reload talk to the current board (the user may have
        # reconnected or uploaded a different task since dialog construction).
        self.board = getattr(self.setup_widget, "pycboard", None)
        # Repopulate variable rows when sm_info shape may have changed
        # (task uploaded / re-uploaded between dialog opens).
        self.refresh_variables()
        board = self.board
        if not board or not getattr(board, "sm_info", None):
            self.variables_groupbox.setEnabled(False)
            return
        # Variables are editable whenever a task is uploaded: pre-run, during
        # run (async via V message), and between runs (sync via REPL).
        # VariableSetter.set/get pick the path from ``board.framework_running``.
        self.variables_groupbox.setEnabled(bool(board.sm_info.variables))

    def add_note(self):
        if not self.board:
            return
        note_text = self.notes_textbox.text()
        self.notes_textbox.clear()
        self.board.data_logger.print_message(note_text, source="u")


# Get / Set buttons for the Variables panel: blue (info) and mint
# (success) gradients with glass cap.
_VAR_GET_BUTTON_STYLE = BUTTON_STYLE.format(
    color=COLORS['info'], hover_color=COLORS['info_hover'])
_VAR_SET_BUTTON_STYLE = BUTTON_STYLE.format(
    color=COLORS['success'], hover_color=COLORS['success_hover'])


class VariableSetter(QtWidgets.QWidget):
    """For setting and getting a single variable.

    Reads the live pycboard via ``parent_tab.board`` on every operation so
    a board reconnect (or mid-session task re-upload that swaps ``sm_info``)
    is picked up automatically.
    """

    def __init__(self, v_name, v_value, row, parent_tab,
                 specs=None, on_change=None):
        super().__init__(parent_tab)
        # Keep a parent_tab reference so set/get/reload always read the
        # current board attached to the StandardControlsTab.
        self._parent_tab = parent_tab
        self.v_name = v_name
        self._specs = specs       # list[BoxVariableSpec] OR None
        self._on_change = on_change   # callable() OR None
        # Variable rows pinned to 18 px so more variables fit without
        # scrolling. Checkboxes are also fixed to ROW_H so font metrics
        # don't inflate the rendered row.
        from source.gui.theme import THEME as _T_vr
        ROW_H = 18
        self.label = QtWidgets.QLabel(v_name)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label.setFixedHeight(ROW_H)
        # hw_* variables get a warm highlight so the user can spot them.
        if v_name.startswith("hw_"):
            self.label.setStyleSheet(
                "color: #b45309; background: #fde68a;"
                " border-radius: 3px; padding: 1px 4px;"
                " font-weight: 600;")
        else:
            # palette.text for readable contrast on the dark surface.
            self.label.setStyleSheet(
                f"color: {_T_vr.palette.text}; background: transparent;"
                " border: none; padding-left: 2px; font-weight: 600;")
        self.get_button = QtWidgets.QPushButton("Get")
        self.set_button = QtWidgets.QPushButton("Set")
        self.get_button.setFixedSize(36, ROW_H)
        self.set_button.setFixedSize(36, ROW_H)
        self.get_button.setStyleSheet(_VAR_GET_BUTTON_STYLE)
        self.set_button.setStyleSheet(_VAR_SET_BUTTON_STYLE)
        # The value field fills the (only) stretch column. Min 90 px keeps
        # a 9-char value like "100000000" readable.
        self.value_str = QtWidgets.QLineEdit(repr(v_value))
        self.value_str.setFixedHeight(ROW_H)
        self.value_str.setMinimumWidth(90)
        self.value_str.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.value_text_colour("gray")

        self.get_button.clicked.connect(self.get)
        self.set_button.clicked.connect(self.set)
        self.value_str.textChanged.connect(lambda x: self.value_text_colour("black"))
        self.value_str.returnPressed.connect(self.set)
        self.get_button.setDefault(False)
        self.get_button.setAutoDefault(False)
        self.set_button.setDefault(False)
        self.set_button.setAutoDefault(False)

        # Per-row Persist checkbox. Ticked → the variable's final value is
        # captured at Stop and restored at the next Upload for the same
        # subject. Unticked (default) → resets to the task-file default each
        # run. No host-side push at Upload unless the user opts in.
        spec = self._resolve_spec()
        self.persist_chk = self._mk_chk(bool(spec.persistent))
        # Pin to ROW_H so font metrics don't inflate the row.
        # Indicator stays 16x16 from QSS; the widget bounds set the floor.
        self.persist_chk.setFixedHeight(ROW_H)
        self.persist_chk.toggled.connect(self._on_persist_toggled)

        parent_tab.grid_layout.addWidget(self.label, row, 0)
        parent_tab.grid_layout.addWidget(self.value_str, row, 1)
        parent_tab.grid_layout.addWidget(self.get_button, row, 2)
        parent_tab.grid_layout.addWidget(self.set_button, row, 3)
        parent_tab.grid_layout.addWidget(self.persist_chk, row, 4,
                                        QtCore.Qt.AlignmentFlag.AlignCenter)

    @staticmethod
    def _mk_chk(checked: bool) -> "QtWidgets.QCheckBox":
        cb = QtWidgets.QCheckBox()
        cb.setChecked(bool(checked))
        cb.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        return cb

    def _resolve_spec(self):
        """Return the effective BoxVariableSpec for this variable (from the
        parent tab's resolved spec list), or a fresh RESET default."""
        from source.config.experiment import BoxVariableSpec
        if self._specs is not None:
            for s in self._specs:
                if s.name == self.v_name:
                    return s
        return BoxVariableSpec(name=self.v_name)

    def _on_persist_toggled(self, checked: bool) -> None:
        """User toggled Persist, route to the parent tab's layered writer
        (project store when a project is loaded, else the sidecar template)."""
        self._parent_tab.write_persist_flag(self.v_name, checked)
        if callable(self._on_change):
            self._on_change()

    def value_text_colour(self, color="gray"):
        # Dark-only UI: "black" → pure white (committed value); "gray" →
        # muted #cbd5e1 for un-modified / placeholder cells.
        resolved = "#ffffff" if color == "black" else "#cbd5e1"
        self.value_str.setStyleSheet(
            f"color: {resolved}; font-weight: 600;")

    @property
    def board(self):
        """Live read-through to the parent tab's current board reference;
        None when no board is connected (methods treat None as
        "operation skipped")."""
        return getattr(self._parent_tab, "board", None)

    def get(self):
        board = self.board
        if not board:
            self.value_str.setText("Not connected")
            return
        try:
            if board.framework_running:
                board.get_variable(self.v_name)
                self.value_str.setText("getting..")
                QtCore.QTimer.singleShot(200, self.reload)
            else:
                self.value_text_colour("black")
                self.value_str.setText(repr(board.get_variable(self.v_name)))
                QtCore.QTimer.singleShot(1000, self.value_text_colour)
        except BaseException as e:
            # PyboardError extends BaseException, catch here or it escapes
            # through the Qt event loop and locks the dialog.
            logger.error("VariableSetter.get(%s) failed: %s",
                         self.v_name, e)
            self.value_str.setText("Get error")

    def set(self):
        board = self.board
        if not board:
            self.value_str.setText("Not connected")
            return
        try:
            v_value = eval(self.value_str.text(), variable_constants)
        except Exception:
            self.value_str.setText("Invalid value")
            return
        try:
            if board.framework_running:
                # Async path: MCU echoes the value back as a V message
                # which updates sm_info.variables; the 200 ms reload
                # then reads the echoed value back into the line edit.
                board.set_variable(self.v_name, v_value)
                self.value_str.setText("setting..")
                QtCore.QTimer.singleShot(200, self.reload)
            else:
                # Sync path via REPL: returns True/False immediately.
                if board.set_variable(self.v_name, v_value):
                    self.value_text_colour("gray")
                else:
                    self.value_str.setText("Set failed")
        except BaseException as e:
            logger.error("VariableSetter.set(%s, %r) failed: %s",
                         self.v_name, v_value, e)
            self.value_str.setText("Set error")

    def reload(self):
        board = self.board
        if not board or not getattr(board, "sm_info", None):
            return
        # ``.get`` so a variable renamed mid-session doesn't KeyError.
        v = board.sm_info.variables.get(self.v_name)
        self.value_text_colour("black")
        self.value_str.setText(repr(v))
        QtCore.QTimer.singleShot(1000, self.value_text_colour)


class EventsTab(QtWidgets.QWidget):
    """Trigger any event quickly without a combobox."""

    def __init__(self, setup_widget, parent=None):
        super().__init__(parent)
        self.setup_widget = setup_widget
        self.board = getattr(setup_widget, "pycboard", None)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        self.events_groupbox = QtWidgets.QGroupBox("Trigger events")
        self.events_groupbox.setStyleSheet(_GROUPBOX_TITLE_STYLE)
        group_layout = QtWidgets.QVBoxLayout(self.events_groupbox)

        self.events_table = QtWidgets.QTableWidget()
        self.events_table.setColumnCount(2)
        self.events_table.setHorizontalHeaderLabels(["Event", "Trigger"])
        self.events_table.verticalHeader().setVisible(False)
        self.events_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.events_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.events_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.events_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.events_table.setColumnWidth(1, 64)
        self.events_table.verticalHeader().setDefaultSectionSize(30)
        group_layout.addWidget(self.events_table)

        layout.addWidget(self.events_groupbox)
        layout.addStretch()
        self.refresh_events()

    def refresh_events(self):
        self.events_table.setRowCount(0)
        board = self.board
        if not board or not getattr(board, "sm_info", None):
            return

        events = list(board.sm_info.events)
        self.events_table.setRowCount(len(events))
        for row, event_name in enumerate(events):
            item = QtWidgets.QTableWidgetItem(event_name)
            self.events_table.setItem(row, 0, item)
            btn = QtWidgets.QPushButton("Trigger")
            btn.setFixedWidth(56)
            btn.setDefault(False)
            btn.setAutoDefault(False)
            btn.setStyleSheet(BUTTON_STYLE.format(
                color=COLORS["warning"],
                hover_color=COLORS["warning_hover"],
            ))
            btn.clicked.connect(lambda _=False, en=event_name: self.trigger_event(en))
            self.events_table.setCellWidget(row, 1, btn)

    def refresh_enabled_states(self):
        # Re-pull the live pycboard like StandardControlsTab does, the
        # user may have reconnected since dialog construction, and a
        # frozen reference would drive the dead board.
        self.board = getattr(self.setup_widget, "pycboard", None)
        self.refresh_events()
        board = self.board
        if not board or not getattr(board, "sm_info", None):
            self.events_groupbox.setEnabled(False)
            return
        self.events_groupbox.setEnabled(board.framework_running and bool(board.sm_info.events))

    def trigger_event(self, event_name):
        if self.board and self.board.framework_running:
            # BaseException: PyboardError (serial/board failure) is NOT an
            # Exception subclass; an unknown event name raises KeyError.
            # Either escaping a Qt slot takes the dialog down.
            try:
                self.board.trigger_event(event_name)
            except BaseException as e:
                logger.warning("Trigger event %r failed: %s", event_name, e)


