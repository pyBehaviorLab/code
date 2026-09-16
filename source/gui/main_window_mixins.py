"""
source/gui/main_window_mixins.py - Shared MainWindow helpers.

Lightweight mixin holding utilities that BOTH main_windows want and
that don't tangle with mode-specific UI naming (theme buttons,
sidebar widgets, error-log shapes, etc.). Anything mode-specific
stays on the subclass.

Usage:
    class MainWindow(QtWidgets.QMainWindow, MainWindowUtilsMixin):
        ...

The two hooks subclasses may override:
    _data_dir_path() -> Path     where _openDataFolder opens
    _tasks_dir_path() -> Path    where _openTasksFolder opens
Defaults resolve relative to the source tree root.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)


class MainWindowUtilsMixin:
    """Mode-agnostic helpers for both pyOperant and pyMaze main_windows."""

    # ------------------------------------------------------------------
    # File-system shortcuts (Ctrl+D / Ctrl+T)
    # ------------------------------------------------------------------

    def _data_dir_path(self) -> Path:
        """Return the data directory path. Override per mode if needed."""
        return Path(__file__).resolve().parents[2] / "data"

    def _tasks_dir_path(self) -> Path:
        """Return the tasks directory path. Override per mode if needed."""
        return Path(__file__).resolve().parents[2] / "tasks"

    def _openDataFolder(self):
        """Open the data directory in the system file manager."""
        try:
            data_dir = self._data_dir_path()
            data_dir.mkdir(parents=True, exist_ok=True)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(data_dir)))
        except Exception as e:
            logger.error("Error opening data folder: %s", e)

    def _openTasksFolder(self):
        """Open the tasks directory in the system file manager."""
        try:
            tasks_dir = self._tasks_dir_path()
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(tasks_dir)))
        except Exception as e:
            logger.error("Error opening tasks folder: %s", e)

    def _register_data_dir_browse_btn(self, btn) -> None:
        """Track a data-dir Browse button so ``_refresh_status_badges``
        can enable it only while a project is loaded. Both the inline
        Browse… (operant + maze) and maze's extra "Browse Directory"
        button register here."""
        btns = getattr(self, "_data_dir_browse_btns", None)
        if btns is None:
            btns = []
            self._data_dir_browse_btns = btns
        btns.append(btn)

    def _browseDirectory(self):
        """Pick a data directory; writes into self.info_fields['dir'] if present."""
        fields = getattr(self, "info_fields", None)
        start = ""
        if fields and "dir" in fields:
            try:
                start = fields["dir"].text() or ""
            except Exception:
                start = ""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Data Directory", start)
        if dir_path and fields and "dir" in fields:
            try:
                fields["dir"].setText(dir_path)
            except Exception:
                pass
            # Push value into the live cfg so get_session_dirs picks it
            # up immediately, before the next autosave.
            cfg = getattr(self, "_active_config", None)
            if cfg is not None and cfg.meta is not None:
                cfg.meta.data_dir = dir_path
                marker = getattr(self, "_project_changed", None)
                if callable(marker):
                    try:
                        marker(reason="data_dir_browse")
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Help-button factory (operant uses this widely; maze can adopt)
    # ------------------------------------------------------------------

    def createHelpButton(self, tooltip_text: str) -> QtWidgets.QPushButton:
        """Circular help button with tooltip. Mode-agnostic."""
        btn = QtWidgets.QPushButton("?")
        btn.setFixedSize(20, 20)
        btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #8e8e8e;"
            "  color: white;"
            "  border-radius: 10px;"
            "  border: none;"
            "  font-weight: bold;"
            "  font-size: 11px;"
            "  padding: 0px;"
            "}"
            "QPushButton:hover { background-color: #6e6e6e; }"
            "QPushButton:pressed { background-color: #5e5e5e; }"
        )
        btn.setToolTip(tooltip_text)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        return btn

    # ------------------------------------------------------------------
    # Control-button sections (shared by operant + maze control rows)
    # ------------------------------------------------------------------

    def _make_control_section_container(self, title: str, help_text=None, *,
                                        group_qss: str = None,
                                        group_spacing: int = 8):
        """Header row (uppercase title + optional help "?") above a group box
        for a control-button section. Returns ``(container, group_layout)``,
        the caller drops buttons into ``group_layout``. ``help_text=None``
        omits the help button (maze's control rows); pass text to include it
        (operant's)."""
        from source.gui.theme import THEME
        if group_qss is None:
            from source.gui.styles import BOX_CONTROLS_SUB_GROUP_STYLE
            group_qss = BOX_CONTROLS_SUB_GROUP_STYLE

        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setSpacing(0)
        container_layout.setContentsMargins(0, 0, 0, 0)

        header_row = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header_row)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)
        header_label = QtWidgets.QLabel(title)
        header_label.setStyleSheet(
            "QLabel {"
            " font-weight: 700;"
            " font-size: 9pt;"
            f" color: {THEME.palette.text_muted};"
            " padding: 0px 4px;"
            " margin: 0px;"
            " background-color: transparent;"
            " text-transform: uppercase;"
            " letter-spacing: 0.5px;"
            "}"
        )
        header_layout.addWidget(header_label)
        if help_text:
            header_layout.addWidget(self.createHelpButton(help_text))
        header_layout.addStretch()
        container_layout.addWidget(header_row)

        group = QtWidgets.QGroupBox("")
        group_layout = QtWidgets.QHBoxLayout(group)
        group_layout.setSpacing(group_spacing)
        group.setStyleSheet(group_qss)
        container_layout.addWidget(group)
        return container, group_layout

    def _make_control_section_button(self, text: str, icon_file: str,
                                     callback, color_key: str,
                                     *, extra_qss: str = ""):
        """``make_button`` with the control-row size policy / icon size /
        click wiring applied. ``extra_qss`` is appended (the master row bumps
        font-size)."""
        from source.gui.widgets.common import make_button
        icon_dir = Path(__file__).parent / 'icons'
        btn = make_button(
            text, color=color_key, icon=icon_dir / icon_file, height=36,
        )
        btn.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        btn.setIconSize(QtCore.QSize(16, 16))
        if extra_qss:
            btn.setStyleSheet(btn.styleSheet() + extra_qss)
        btn.clicked.connect(callback)
        return btn

    def _apply_window_icon(self, icon_name: str) -> None:
        """Set a multi-size window + taskbar icon from ``icons/<icon_name>`` so
        the title bar / taskbar / Alt-Tab switcher render crisply at every size
        Windows requests. No-op if the file is missing."""
        icon_path = Path(__file__).parent / "icons" / icon_name
        if not icon_path.exists():
            return
        icon = QtGui.QIcon()
        for size in (16, 24, 32, 48, 64, 128, 256):
            icon.addFile(str(icon_path), QtCore.QSize(size, size))
        self.setWindowIcon(icon)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)

    def _confirm(self, title: str, body: str) -> bool:
        """Modal Yes/No prompt (default No). Returns True on Yes."""
        reply = QtWidgets.QMessageBox.question(
            self, title, body,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------------
    # Debug-mode toggle (works if subclass exposes self.debugToggleButton)
    # ------------------------------------------------------------------

    def _toggleDebugMode(self):
        btn = getattr(self, "debugToggleButton", None)
        if btn is None:
            return
        checked = btn.isChecked()
        try:
            btn.setText(f"Debug: {'ON' if checked else 'OFF'}")
        except Exception:
            pass
        # Lower the GUI log-widget handler to DEBUG (ON) or back to INFO (OFF).
        # Goes through the log module so the per-GUI log FILE keeps its DEBUG
        # firehose either way, toggling only changes what the window shows.
        from source.log import set_debug_mode
        set_debug_mode(checked)

    # ------------------------------------------------------------------
    # Error-log helpers
    # ------------------------------------------------------------------

    def _appendErrorLog(self, msg: str):
        """Append a timestamped line to the error-log browser and list."""
        ts_short = datetime.now().strftime("%H:%M:%S")
        browser = getattr(self, "errorLogBrowser", None)
        if browser is not None:
            try:
                browser.append(f"[{ts_short}] {msg}")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Thread-safe error dialog (uses _appendErrorLog above)
    # ------------------------------------------------------------------

    def showError(self, message: str):
        """Thread-safe error reporter: dialog on main thread + log entry."""
        QtCore.QMetaObject.invokeMethod(
            self,
            "_showErrorImpl",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, message),
        )

    @QtCore.Slot(str)
    def _showErrorImpl(self, message: str):
        """Main-thread implementation of showError. Subclasses may override."""
        try:
            QtWidgets.QMessageBox.critical(self, "Error", message)
        except Exception:
            pass
        try:
            self._appendErrorLog(message)
        except Exception:
            pass
        logger.error(message)

    def populate_experiment_info_fields(self, sidebar) -> None:
        """Build the four QLineEdit rows (experimenter / project / session
        / data dir) into ``sidebar`` and register them in
        ``self.info_fields`` so save/load helpers find them by key.

        Also adds the "Track HD / task snapshots" checkbox (default ON),
        toggling it sets ``cfg.meta.tracking_enabled`` and updates the
        live SnapshotStore. When OFF, upload events skip hashing + content
        capture; only an audit line is appended to change_log.jsonl.
        """
        rows = [
            ("Experimenter",   "experimenter", "Enter experimenter name"),
            ("Project",        "project",      "Enter project name"),
            ("Session",        "session",      "Enter session ID"),
            ("Data Directory", "dir",          "Select data directory"),
        ]
        # Experiment-info values: small, clean blue, regular weight. A light
        # blue (not a dark navy) so the text stays legible on the dark grey
        # field background.
        edit_style = (
            "QLineEdit { background-color: #44475a;"
            " border: 1px solid #6272a4; border-radius: 4px;"
            " padding: 5px 8px; color: #60a5fa; font-size: 12px; }"
            "QLineEdit:focus { border: 1px solid #bd93f9; }")
        # Locked fields (project name, data dir) keep the darker background
        # so it's clear they aren't free-text, but share the same small
        # blue value text, project name is fixed at creation, and the
        # data dir is changed only via Browse….
        readonly_style = (
            "QLineEdit { background-color: #383a4a;"
            " border: 1px solid #44475a; border-radius: 4px;"
            " padding: 5px 8px; color: #60a5fa; font-size: 12px; }")
        # ``project`` is set when the project is created (folder basename),
        # ``dir`` is set via the Browse… button, neither is typed here.
        locked_keys = ("project", "dir")
        for label, key, placeholder in rows:
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.setFixedHeight(32)
            if key in locked_keys:
                edit.setReadOnly(True)
                edit.setStyleSheet(readonly_style)
                edit.setToolTip(
                    "Set when the project is created, not editable here."
                    if key == "project"
                    else "Change with the Browse… button (project must be loaded).")
            else:
                edit.setStyleSheet(edit_style)
            self.info_fields[key] = edit
            if key == "dir":
                # Pair the line edit with a Browse… button so the user can
                # pick where THIS project's data lands; blank = default
                # <top>/data/<project>.
                row = QtWidgets.QWidget()
                hl = QtWidgets.QHBoxLayout(row)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.setSpacing(6)
                hl.addWidget(edit, 1)
                browse = QtWidgets.QPushButton("Browse…")
                browse.setFixedHeight(32)
                browse.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
                browse.setStyleSheet(
                    "QPushButton { background-color: #6272a4; color: #f8f8f2;"
                    " border: 1px solid #6272a4; border-radius: 4px;"
                    " padding: 0 12px; font-size: 11px; }"
                    "QPushButton:hover { background-color: #7282b4; }")
                browse.clicked.connect(self._browseDirectory)
                # Data dir is project-scoped, only meaningful once a
                # project is loaded. Start disabled; _refresh_status_
                # badges() flips it on/off as projects load/unload.
                browse.setEnabled(False)
                self._register_data_dir_browse_btn(browse)
                hl.addWidget(browse, 0)
                sidebar.addField(label, row)
            else:
                sidebar.addField(label, edit)
        sidebar.content_layout.addSpacing(6)
        # Tracking toggle.
        track_cb = QtWidgets.QCheckBox("Track HD / task snapshots (recommended)")
        track_cb.setChecked(True)
        track_cb.setToolTip(
            "When ON (default), every upload captures the source .py "
            "into <project>/source/ for full trace-back. When OFF, "
            "uploads skip staging, change_log still records that tracking "
            "was off so the gap is auditable.")
        track_cb.setStyleSheet("QCheckBox { color: #f8f8f2; padding: 4px; }")
        track_cb.toggled.connect(self._on_tracking_toggle)
        self.info_fields["tracking_enabled"] = track_cb
        sidebar.content_layout.addWidget(track_cb)
        sidebar.content_layout.addSpacing(10)

    # ------------------------------------------------------------------
    # Metadata buttons in the Experiment-Info sidebar (shared by both modes)
    # ------------------------------------------------------------------

    def create_sidebar_help_button(self, tooltip_text):
        """Circular '?' help button for sidebar rows (token-driven)."""
        from source.gui.theme import THEME
        btn = QtWidgets.QPushButton("?")
        btn.setFixedSize(22, 22)
        btn.setStyleSheet(
            "QPushButton {"
            f" background-color: {THEME.palette.surface_border_strong};"
            f" color: {THEME.palette.text};"
            " border-radius: 11px; border: none;"
            " font-weight: bold; font-size: 12px; padding: 0px;"
            "}"
            f"QPushButton:hover {{ background-color: {THEME.palette.surface_border}; }}"
        )
        btn.setToolTip(tooltip_text)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        return btn

    def build_metadata_buttons(self, sidebar):
        """Add the cohort-metadata buttons to ``sidebar``, Load / Edit, each
        with a help dot. Single owner for both modes: operant and maze build an
        identical Experiment-Info metadata block. Sets ``self.load_meta_button``
        / ``edit_meta_button``. (Per-box subject assignment + loading a new
        cohort happen in the per-box subject picker, so there is no
        Auto-Populate button.)
        """
        from source.gui.style_builders import button_style as _btn_style

        specs = [
            ("Load Metadata",   "info",    self.load_cohort_metadata,
             "load_meta_button", "Load subject metadata from an Excel/CSV cohort file"),
            ("Edit Metadata",   "warning", self.edit_all_metadata,
             "edit_meta_button", "View and edit the loaded cohort for all boxes"),
        ]
        for text, color, callback, attr, help_text in specs:
            container = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(5)
            btn = QtWidgets.QPushButton(text)
            btn.setFixedHeight(32)
            btn.setStyleSheet(_btn_style(color, height=32))
            btn.clicked.connect(callback)
            setattr(self, attr, btn)
            row.addWidget(btn)
            row.addWidget(self.create_sidebar_help_button(help_text))
            sidebar.content_layout.addWidget(container)
            sidebar.content_layout.addSpacing(8)
        sidebar.content_layout.addStretch()
        # Initial gating (Edit / Auto-Populate disabled until a cohort loads).
        try:
            self.update_metadata_button_states()
        except Exception as e:
            logger.debug("update_metadata_button_states (initial): %s", e)

    def edit_all_metadata(self):
        """Edit the loaded cohort DataFrame in place. Offers an in-context
        'Load Metadata…' if none is loaded (shared by both modes)."""
        if not self.metadata_manager.prompt_load_if_missing(self):
            return
        from source.gui.dialogs import MetadataEditorDialog
        dlg = MetadataEditorDialog(self.metadata_manager, self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.show_temporary_message("Cohort updated.")
            self.update_metadata_button_states()

    def show_temporary_message(self, message, timeout=3000):
        """Transient status-bar message (both modes have ``statusbar``)."""
        try:
            self.statusbar.showMessage(message, timeout)
        except (AttributeError, RuntimeError):
            pass

    def update_metadata_button_states(self):
        """Gate the metadata sidebar buttons by cohort-loaded + box-IDs +
        running state. Single owner for both modes; every widget is
        hasattr-guarded so a mode without a given button (e.g. maze has no
        Clear-Meta) simply skips it. Clear/Edit lock while any box runs,
        changing the cohort mid-run would orphan an open MCU TSV.
        """
        mm = getattr(self, "metadata_manager", None)
        has_cohort = mm is not None and mm.metadata_df is not None
        try:
            has_ids = any(edit.text().strip()
                          for _, edit in self.iter_box_subject_widgets())
        except Exception:
            has_ids = False
        any_running = self.any_box_running()

        if hasattr(self, "clear_metadata_button"):
            self.clear_metadata_button.setEnabled(has_ids and not any_running)
            self.clear_metadata_button.setToolTip(
                "Stop running sessions before clearing Subject IDs" if any_running
                else "Clear Subject IDs (cohort stays loaded)" if has_ids
                else "No Subject IDs to clear")
        if hasattr(self, "edit_meta_button"):
            self.edit_meta_button.setEnabled(has_cohort and not any_running)
            self.edit_meta_button.setToolTip(
                "Edit the loaded cohort" if has_cohort else "Load a cohort first")
        if hasattr(self, "load_meta_button"):
            self.load_meta_button.setEnabled(not any_running)

    def _on_tracking_toggle(self, checked: bool) -> None:
        """Push the checkbox state into cfg.meta + the live SnapshotStore.

        When turning tracking OFF, confirm with the user, every run
        during the OFF window will have an untraceable HD/task snapshot.
        Re-enabling does not prompt (safe direction).
        """
        # Turning tracking OFF, confirm with a modal warning first.
        if not checked:
            cb = (self.info_fields.get("tracking_enabled")
                  if hasattr(self, "info_fields") else None)
            reply = QtWidgets.QMessageBox.warning(
                self,
                "Disable source tracking?",
                "Disabling tracking means uploads will NOT be saved into "
                "<project>/source/. You will not be able to trace "
                "back which task / HD was used for runs during this "
                "period, only a 'tracking off' window will be logged "
                "in change_log.jsonl.\n\n"
                "Are you sure you want to disable tracking?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                # Revert the checkbox WITHOUT re-firing toggled.
                if cb is not None:
                    try:
                        cb.blockSignals(True)
                        cb.setChecked(True)
                    finally:
                        cb.blockSignals(False)
                return

        cfg = getattr(self, "_active_config", None)
        if cfg is not None and cfg.meta is not None:
            cfg.meta.tracking_enabled = bool(checked)
        store = getattr(self, "_snapshot_store", None)
        if store is not None:
            try:
                store.set_tracking_enabled(bool(checked))
            except Exception as e:
                logger.debug("set_tracking_enabled failed: %s", e)
        # Mark project dirty so autosave persists the toggle.
        marker = getattr(self, "_project_changed", None)
        if callable(marker):
            try:
                marker(reason="tracking_toggle")
            except Exception:
                pass

    def _exportErrorLog(self):
        """Save the current error-log browser to a .txt file."""
        browser = getattr(self, "errorLogBrowser", None)
        if browser is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Error Log", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(browser.toPlainText())
            sb = getattr(self, "statusbar", None)
            if sb is not None:
                sb.showMessage(f"Error log exported to {path}", 3000)
        except Exception as e:
            logger.error("Failed to export error log: %s", e)
