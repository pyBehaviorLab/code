"""The maze mode's ``MainWindow``, one arena per tab, one camera each.

Half of the two-line fork described in ``docs/dev/module-map.md``: ``pyMaze.py``
and ``pyOperant.py`` load the same ``source/`` package and differ only in the
entry file and this subclass. Everything shared, the Pipeline, the camera
lifecycle, tracking config, recording, zones, the display tiles, lives in
:class:`~source.gui.base.MainWindowBase`; what is here is only what a maze is
and an operant box is not.

Which is, concretely:

* **one camera per arena**, so no ROI split, where operant divides one
  camera between boxes, a maze tab owns its whole frame;
* **aspect-preserving tiles** (``_tile_aspect_mode``), so an arena keeps its
  shape as the window is resized, a stretched arena makes a circular field
  look elliptical and a mouse look the wrong shape, and the operator has no
  way to tell that from a genuinely distorted lens;
* **polygon zones only** (``_ZONE_MIN_POINTS = 3``): a line or a scale bar is
  a measurement here, not an addressable trial-logic object the way it is in
  operant;
* **stage-based recording**, against operant's framework-start-with-callback.

Anything that reads "and operant does the same" belongs in ``base.py``, not
here: the two modes are meant to diverge in behaviour, never in plumbing.
"""

from pathlib import Path

from PySide6 import QtGui, QtWidgets

from source import paths as app_paths

from source.gui.styles import (
    STATUS_BAR_STYLE, HRULE_QSS,
)
from source.gui.theme import THEME
from source.gui.widgets import (
    DetachableTabWidget, SetupWidget, ErrorLogPanel, MarkdownView,
)
from source.gui.metadata_manager import MetadataManager
from source.log import get_logger

logger = get_logger()


# Suppress system beep on QMessageBox.warning dialogs (no-icon variant).
from source.gui.utility import install_quiet_qt_warnings
install_quiet_qt_warnings()


# DetachableTabWindow lives in widgets.py (used by both maze and operant).
from source.gui.widgets import DetachableTabWindow  # noqa: F401  (re-export for callers)



# Shown in the Documentation sidebar only if GUI_DOCUMENTATION_MAZE.md is
# missing/unreadable (the real guide lives in that file at the repo root).
_MAZE_DOC_FALLBACK = """
<html><body style="color:#f8f8f2;">
<h1 style="color:#ff79c6;">pyMaze</h1>
<p>One maze per tab, each owns its MCU, camera, tracking, doors and run.</p>
<ol>
<li>Experimental Info (Ctrl+E): set experimenter, project, data dir; Save.</li>
<li>Box Setup: add a maze tab; set Subject ID + task.</li>
<li>Camera Config: Calibrate All Cameras, then pick Resolution + Target FPS.</li>
<li>Tracking Config: mode + zones + background; Apply &amp; Close.</li>
<li>Bottom bar: Connect boards → Config → Upload Task → Start.</li>
</ol>
<p>See GUI_DOCUMENTATION_MAZE.md for the full guide.</p>
</body></html>
"""


# ============================================================================
# MAIN WINDOW
# ============================================================================

from .base import MainWindowBase


class MainWindow(MainWindowBase):
    """pyMaze - Per-setup tab architecture matching GUI2.JPG.

    Each tab represents one maze setup with video display and controls.
    Left sidebars: Experimental Info, Error Log, Documentation.
    Top: Box Setup + Camera Control. Bottom: Setup Control.
    """

    # Line/scale zones aren't addressable trial-logic objects in maze,
    # only polygons (3+ points) enter the live zone manager.
    _ZONE_MIN_POINTS = 3

    # Keep the camera's aspect ratio as the window resizes, letterboxing when
    # the cell's shape differs from the camera's, the same treatment operant
    # already uses. The alternative, stretching to fill the cell, distorts the
    # arena: a circular field reads as elliptical and the distortion changes
    # with the window size, which is indistinguishable from a lens that needs
    # calibrating.
    _tile_aspect_mode = "preserve"

    def __init__(self):
        super().__init__()
        logger.info("Initializing MainWindow - pyMaze (single-process, pycboard in widget)")

        # Setup management
        self.setup_count = 0

        # ── Unified pipeline (camera → bus → sinks → MCU). ──
        # Constructs self.pipeline + self.bridge + self.video_manager alias
        # and wires bridge signals to the base slots.
        self._setup_pipeline(target_fps=30)

        # Tracker overlay state (set by base._on_box_tracker via QtBridge).
        # tracker_manager is exposed by the pipeline for the few maze call
        # sites that need it (zone manager registry, etc).
        self.tracker_manager = self.pipeline.tracker_manager
        self.tracker_manager.register_callback(self._on_tracking_update)
        self.tracking_enabled = {}         # box_id -> bool (UI mirror)
        self.tracking_enhancers = {}       # box_id -> TrackingEnhancer
        self.smooth_tracking_enabled = True

        # Recording state. The pipeline owns the actual recording lifecycle
        # via RecorderSink; these dicts hold mode-specific bookkeeping
        # (per-arena recorder references, tracking writers).
        self.recording_setups = set()
        # Per-stage recorder lives on the SetupWidget (widget.video_recorder).
        # _recording_ctx holds the arena's session-start context, set up
        # before the recorder is created.
        self._recording_ctx = {}
        # Per-box TrackingWriter is stored on the SetupWidget
        # (``sw.tracking_writer``), single source for base consumers and
        # the sole closer (_stop_recording_for_box).
        self.video_target_fps = 30
        self.video_camera_resolution = None  # (w, h) from Camera Connect Dialog
        self.video_camera_realistic_fps = None  # measured ceiling at chosen resolution
        self.video_grayscale = False
        self.video_frame_strategy = "accept"  # single file, no post-record remux

        # Config
        self.metadata_manager = MetadataManager()
        self.info_fields = {}

        # Pose-related UI state (overlay rendering reads self._overlay
        # which is populated by base._on_box_pose).  The actual model +
        # inference lifetimes are owned by self.pipeline.pose.
        self.pose_configs = {}             # box_id → config dict
        self.pose_zone_state = {}          # box_id → {zone_name: bool}

        # Display-side accounting
        self._display_frame_counts = {}

        # Zone state (per-box). The pipeline reads zones via the per-box
        # zone_lookup the maze passes into pipeline.enable_pose; the live
        # overlay reads them from this dict.
        self.tracking_zones = {}
        self.tracking_zone_paths = {}      # box_id -> filepath for auto-save

        # Per-box tracking state lives on ``self.pipeline._tracking_configs``;
        # global dialog-only fields on ``self._tracking_dialog_globals`` (base).

        # UI state
        self.gpu_encoding_available = False
        self.ffmpeg_available = False

        self._build_ui()
        self._init_default_data_dir()
        # Draft working config so create-phase edits accumulate in memory
        # and transfer on the first Save.
        self._init_working_config()

        # GPU detection
        self._detect_gpu_encoding()

        # Shared timer pipeline: refresh_timer (1 Hz, always-on folder rescan
        # + per-box hash poll) + process_timer (10 ms MCU drain + plot +
        # display + clock, started only when active via _sync_timer_mode).
        self._start_refresh_timer()

        # Keyboard shortcuts
        from source.gui.utility import init_keyboard_shortcuts
        shortcuts = {
            "Ctrl+D": self._openDataFolder,
            "Ctrl+T": self._openTasksFolder,
            "Ctrl+E": lambda: self._toggle_sidebar("info"),       # Experimental info
            "Ctrl+L": lambda: self._toggle_sidebar("errorlog"),   # Logger
            "Ctrl+M": self.load_cohort_metadata,                  # Load Meta
            "Ctrl+O": self.load_config,                           # Load project config
            "Ctrl+P": self.show_universal_plot_dialog,            # Session plots (uniform w/ operant)
            "Ctrl+=": self.add_setup,                            # Add setup (also Ctrl++)
            "Ctrl++": self.add_setup,
            "Ctrl+-": self.remove_setup,                         # Remove setup
            "Ctrl+Tab": lambda: self._navigate_tab(1),           # Next maze tab
            "Ctrl+Shift+Tab": lambda: self._navigate_tab(-1),    # Previous maze tab
        }
        init_keyboard_shortcuts(self, shortcuts)

        logger.info("MainWindow initialization complete")

    # ========================================================================
    # UI SETUP
    # ========================================================================

    def _build_ui(self):
        self.setObjectName("MainWindow")
        self.resize(1000, 700)
        self.setMinimumSize(800, 500)
        self.setWindowTitle("pyBehaviorLab - Maze")

        self._apply_window_icon('pymaze.svg')

        # Central widget
        self.centralwidget = QtWidgets.QWidget()
        self.centralwidget.setObjectName("mwCentral")
        # Glassy white outline.
        self.centralwidget.setStyleSheet(
            "QWidget#mwCentral {"
            f" background-color: {THEME.palette.bg};"
            " border: 1px solid rgba(255,255,255,0.18);"
            "}"
        )
        self.setCentralWidget(self.centralwidget)
        main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(26, 4, 4, 0)  # left margin for sidebar buttons

        # -- Top bar: Box Setup + Camera Control --
        self._build_top_control_buttons(main_layout)

        # -- Separator between top controls and setup tabs --
        # Soft gradient rule (transparent → white-glow → transparent) so
        # the divider reads as part of the dark theme.
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setFixedHeight(1)
        separator.setStyleSheet(HRULE_QSS)
        main_layout.addWidget(separator)

        # -- Setup tabs (per-maze) --
        self.tabWidget = DetachableTabWidget()
        self.tabWidget.setTabsClosable(False)
        main_layout.addWidget(self.tabWidget, stretch=1)

        # Statistics window is opened on demand from the top-bar Statistics
        # button, see _open_statistics(). Not a main tab in maze mode; main
        # tabs are reserved for per-setup arena widgets.
        self.statisticsTab = None
        self._statistics_window = None

        # -- BOTTOM BAR: Setup Control --
        self._build_bottom_control_buttons(main_layout)

        # -- LEFT sidebars (overlay, rotated toggle buttons) --
        self._build_left_sidebars()

        # -- Status bar --
        self._build_status_bar()

        self._title_bar = None
        self.setWindowTitle("pyBehaviorLab - Maze")

        # Initial UI state (disable buttons that need boxes/connections).
        self.refresh_ui_state()

    def _build_top_control_buttons(self, parent_layout):
        """Top control buttons: Box Setup + Experiment + Camera Control.

        Uses the shared ``_make_control_section_*`` helpers with no help
        button (maze's control rows are help-less). Cohort metadata lives in
        the Experimental Info sidebar; Session Plot is on the master row.
        """
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(10)
        button_layout.setContentsMargins(0, 0, 0, 0)

        sections = [
            ("Box Setup", [
                ("Add Maze",    "add.svg",    self.add_setup,    "primary", "add_box_button"),
                ("Remove Maze", "remove.svg", self.remove_setup, "danger",  "remove_box_button"),
            ]),
            ("Experiment", [
                ("Save", "save.svg",   self.save_config, "success", "save_config_button"),
                ("Load", "folder.svg", self.load_config, "warning", "load_config_button"),
            ]),
            ("Camera Control", [
                ("Camera Config",   "camera.svg", self.show_camera_config_dialog,  "primary", "camera_connect_button"),
                ("Tracking Config", "chip.svg",   self.show_tracking_config_dialog, "info",   "track_button"),
                ("Test Tracking",   "video.svg",  self.toggle_test_tracking,        "info",   "test_tracking_button"),
            ]),
        ]
        for title, buttons in sections:
            container, group_layout = self._make_control_section_container(title)
            for text, icon_file, callback, color_key, attr in buttons:
                btn = self._make_control_section_button(
                    text, icon_file, callback, color_key)
                setattr(self, attr, btn)
                group_layout.addWidget(btn)
            button_layout.addWidget(container)

        parent_layout.addLayout(button_layout)

    def _open_statistics(self):
        """Open the live-MCU Statistics widget in a top-level window.

        StatsCanvas reads pyControl events/states from the MCU log
        stream, identical for both operant and maze sessions. Lazy-created
        on first click; subsequent clicks raise the existing window.
        """
        try:
            if self.statisticsTab is None:
                from source.stats import StatsCanvas
                self.statisticsTab = StatsCanvas(self)
            if self._statistics_window is None:
                win = QtWidgets.QWidget()
                win.setWindowTitle("Live MCU Statistics")
                screen = QtWidgets.QApplication.primaryScreen()
                if screen:
                    avail = screen.availableGeometry()
                    w, h = int(avail.width() * 0.85), int(avail.height() * 0.85)
                    win.resize(w, h)
                    win.move(avail.x() + (avail.width() - w) // 2,
                             avail.y() + (avail.height() - h) // 2)
                layout = QtWidgets.QVBoxLayout(win)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(self.statisticsTab)
                self._statistics_window = win
            self._statistics_window.show()
            self._statistics_window.raise_()
            self._statistics_window.activateWindow()
        except Exception as e:
            logger.error(f"Failed to open Statistics window: {e}")

    def _build_bottom_control_buttons(self, parent_layout):
        """Bottom control buttons: the Setup Control master row."""
        # Same soft glow rule as the top separator.
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setFixedHeight(1)
        separator.setStyleSheet(HRULE_QSS)
        parent_layout.addWidget(separator, 0)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(10)
        button_layout.setContentsMargins(0, 0, 0, 0)

        # Master row: bordered card group, tighter spacing, ~10% larger text.
        from source.gui.style_builders import master_group_qss
        container, group_layout = self._make_control_section_container(
            "Setup Control", group_qss=master_group_qss(), group_spacing=6)
        master_buttons = [
            ("Connect boards", "connect.svg",    self.show_connect_dialog,        "primary", "connect_button"),
            ("Session Plot",   "bar-graph.svg",  self.show_universal_plot_dialog, "info",    "plot_button"),
            ("Config boards",  "settings.svg",   self.show_config_dialog,         "primary", "config_button"),
            ("Upload Task",    "upload.svg",     self.show_upload_dialog,          "info",    "upload_task_button"),
            ("Disconnect",     "disconnect.svg", self.show_disconnect_dialog,     "warning", "disconnect_button"),
            ("Analysis",       "bar-graph.svg",  self.open_offline_analyzer,      "success", "analyzer_button"),
            ("Statistics",     "bar-graph.svg",  self._open_statistics,           "success", "stats_button"),
        ]
        for text, icon_file, callback, color_key, attr in master_buttons:
            btn = self._make_control_section_button(
                text, icon_file, callback, color_key,
                extra_qss=" QPushButton { font-size: 10pt; }")
            setattr(self, attr, btn)
            group_layout.addWidget(btn)
        button_layout.addWidget(container)

        parent_layout.addLayout(button_layout)

    def _build_status_bar(self):
        self.statusbar = QtWidgets.QStatusBar()
        self.statusbar.setMinimumHeight(26)
        self.setStatusBar(self.statusbar)

        # Status badges -- project + tracking state.
        self.statusbar.addPermanentWidget(self._build_status_panel())

        self.statusbar.setStyleSheet(STATUS_BAR_STYLE)

    # -- LEFT SIDEBARS --

    def _sidebar_defs(self):
        """Per-mode left-sidebar table for the shared
        ``MainWindowBase._build_left_sidebars`` loop. Each row:
        (key, sidebar_attr, toggle_attr, toggle_text, (clr, hover, press),
         y, height, header, width, content_builder)."""
        return [
            ("info", "infoSidebar", "info_toggle_btn", "Experimental info",
             ("#bd93f9", "#ff79c6", "#8be9fd"), 40, 150,
             "EXPERIMENTAL INFO", 460, self._build_experiment_info_content),
            ("errorlog", "errorLogSidebar", "errorlog_toggle_btn", "Logger",
             ("#ff6b6b", "#ff79c6", "#ff5555"), 200, 120,
             "LOGGER", 500, self._build_error_log_content),
            ("doc", "docSidebar", "doc_toggle_btn", "Documentation",
             ("#0f3460", "#1e88e5", "#1565c0"), 330, 150,
             "DOCUMENTATION", 550, self._build_doc_content),
        ]

    def _build_experiment_info_content(self):
        self.populate_experiment_info_fields(self.infoSidebar)
        # Cohort-metadata buttons (Load / Edit) come from the shared builder
        # (mirrors operant).
        self.build_metadata_buttons(self.infoSidebar)

    def _build_error_log_content(self):
        panel = ErrorLogPanel(
            style="dark",
            on_debug_toggled=self._toggleDebugMode,
            on_export=self._exportErrorLog,
        )
        self.errorLogBrowser = panel.browser
        self.debugToggleButton = panel.debug_button
        self.errorLogSidebar.content_layout.addWidget(panel, stretch=1)

    def _build_doc_content(self):
        self.docBrowser = MarkdownView()
        # Resolve relative to the repo root so it loads regardless of CWD.
        doc_path = Path(__file__).resolve().parents[2] / "GUI_DOCUMENTATION_MAZE.md"
        self.docBrowser.load_md_file(doc_path, fallback_html=_MAZE_DOC_FALLBACK)
        self.docSidebar.content_layout.addWidget(self.docBrowser, stretch=1)

    # ========================================================================
    # SETUP TAB MANAGEMENT
    # ========================================================================

    def add_setup(self):
        # Next free setup id = smallest positive integer not already a box, so
        # numbering stays compact after removals (add→remove→add reuses the gap
        # instead of climbing a monotonic counter). setup_count tracks the live
        # count only (nothing else reads it).
        existing = set(self._box_widgets.keys())
        sid = 1
        while sid in existing:
            sid += 1
        widget = SetupWidget(setup_id=sid, main_window=self)

        # Connect signals. Record/Stop are wired on the widget itself to the
        # shared RunTask.on_record_clicked / on_stop_clicked.
        widget.pause_clicked.connect(self._on_pause_setup)
        widget.next_clicked.connect(self._on_next_stage)
        widget.prev_clicked.connect(self._on_prev_stage)
        widget.metadata_clicked.connect(self._on_metadata_clicked)
        widget.doors_clicked.connect(self._on_doors_clicked)
        widget.controls_clicked.connect(self._on_controls_clicked)
        widget.detach_clicked.connect(self._on_detach_setup)
        # All 9 zone-adjust signals routed through the shared helper.
        self._wire_zone_buttons(widget, getattr(widget, "setup_number", None))
        # Recording/tracking teardown runs in SetupWidget._after_stop (the
        # shared mcu_stop convergence), covering every stop path.
        # Statistics register/unregister now runs through the SHARED
        # _after_start / _after_stop widget hooks (same as operant), no
        # per-mode framework_started/stopped_signal slots.
        # Auto-start tracking when the framework boots. Canonical hook
        # shared with operant via base._on_framework_auto_started.
        widget.framework_started_signal.connect(self._on_framework_auto_started)
        # Subject-id changes are handled by the shared on_subject_id_changed
        # (wired on subject_id_edit.textChanged in SetupWidget, same as
        # operant), no per-mode editingFinished slot.
        # Tab title updates (subject ID change, state change)
        widget.tab_title_changed.connect(self._on_tab_title_changed)

        self._box_widgets[sid] = widget
        self.setup_count = len(self._box_widgets)
        self.tabWidget.addTab(widget, f"Setup{sid}")
        self.tabWidget.setCurrentWidget(widget)

        # Apply per-tab accent color (colored underline via stylesheet)
        tab_idx = self.tabWidget.indexOf(widget)
        self._apply_tab_color(tab_idx, widget.tab_color)

        logger.info(f"Added Setup{sid}")
        self.statusbar.showMessage(f"Setup{sid} added", 3000)
        self.refresh_ui_state()
        self._project_changed(reason="boxes_changed")

    def remove_setup(self):
        idx = self.tabWidget.currentIndex()
        if idx < 0:
            return
        tab_text = self.tabWidget.tabText(idx)
        widget = self.tabWidget.widget(idx)

        if isinstance(widget, SetupWidget) and widget.is_running:
            QtWidgets.QMessageBox.warning(
                self, "Cannot Remove",
                f"{tab_text} is running. Stop it first.")
            return

        reply = QtWidgets.QMessageBox.question(
            self, "Remove Setup",
            f"Remove {tab_text}?",
            QtWidgets.QMessageBox.StandardButton.Yes |
            QtWidgets.QMessageBox.StandardButton.No)
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.tabWidget.removeTab(idx)
            if isinstance(widget, SetupWidget):
                self._drop_box_state(widget.setup_number)
                self.setup_count = len(self._box_widgets)
            widget.deleteLater()
            logger.info(f"Removed {tab_text}")
            self.refresh_ui_state()
            self._project_changed(reason="boxes_changed")

    # ------------------------------------------------------------------
    # Tracking-init hooks (maze-specific overrides)
    # Called by Camera Connect when tracking is enabled + model configured.
    # ------------------------------------------------------------------

    def _box_wants_tracking(self, box) -> bool:
        """Maze: only boxes with ``tracking_enabled`` flag set after the
        Test Tracking flow."""
        return bool(self.tracking_enabled.get(box.setup_number, False))

    # save_config / load_config inherited from MainWindowBase. The bridge in
    # experiment.apply_config_to_ui / read_ui_into_config owns persistence,
    # plus maze.apply_mode_extras / read_mode_extras for the tracking_zones
    # / tracking_enabled dicts (defined further down).

    def _load_mode_name(self):
        return "maze"

    def _load_clear_existing_state(self):
        """Maze clears silently (no confirmation prompt). Tears down the
        tabWidget contents and resets per-box dicts before the new load.

        Pipeline + host runtime state goes through the shared
        ``_clear_project_runtime_state`` helper so maze + operant clear
        the same things in the same order.
        """
        while self.tabWidget.count():
            w = self.tabWidget.widget(0)
            self.tabWidget.removeTab(0)
            if isinstance(w, SetupWidget):
                self._box_widgets.pop(w.setup_number, None)
            w.deleteLater()
        self.setup_count = 0
        self._clear_project_runtime_state()
        return True

    # Per-setup state is restored by the bridge in
    # experiment.apply_config_to_ui + maze.apply_mode_extras. _hw_def_path
    # comes from BoxConfig.init_hw_def.path; _hw_config is parsed lazily
    # when the door dialog opens.

    def _load_post_restore(self, cfg):
        """Status summary only. The post-load auto-connect sequence
        (cameras + pose init + MCU) lives in ``MainWindowBase``
        (``_auto_flow_after_load``), shared with operant, maze keeps no
        parallel copy."""
        n = len(cfg.setup_config.boxes)
        self.statusbar.showMessage(
            f"Config loaded: {n} setup(s), "
            f"{len(self.tracking_zones)} zone set(s), "
            f"{sum(1 for v in self.tracking_enabled.values() if v)} tracking",
            3000)

    # ========================================================================
    # BOTTOM BAR ACTIONS (MCU + Video)
    # ========================================================================

    # connect_camera, disconnect_camera, _wait_for_camera_streaming inherited
    # from MainWindowBase; the hooks below shape the maze-specific UI.

    # Per-box widget access goes through the box-widget protocol SetupWidget
    # exposes (shared in MainWindowBase). Only the override below remains,
    # maze keeps per-box display caches that need clearing on disconnect.

    # Zone auto-save + log summary now in MainWindowBase default. Maze
    # adds the "Press Test Tracking to start" hint unique to maze UX.
    def _post_tracking_dialog_hook(self, dialog, connected_with_cam):
        super()._post_tracking_dialog_hook(dialog, connected_with_cam)
        try:
            settings = dialog.get_settings()
            method = settings.get("mode", "normal")
            n_boxes = len(settings.get("enabled_boxes", connected_with_cam))
            logger.info(
                "Maze: tracking configured (%s, %d box(es)). "
                "Press 'Test Tracking' to start.",
                method, n_boxes)
        except Exception as e:
            logger.debug(f"maze _post_tracking_dialog_hook tail: {e}")

    # Test Tracking is implemented in MainWindowBase
    # (``_toggle_test_tracking_impl``); ``start_tracking_for_box(force=True)``
    # arms the dry-run flag itself. Maze only paints the button:

    def _refresh_test_tracking_button(self, *, active: bool) -> None:
        """Maze override: paint the test-tracking toolbar button. The
        Track-Config lock is the base's, both modes need it."""
        super()._refresh_test_tracking_button(active=active)
        btn = getattr(self, "test_tracking_button", None)
        if btn is None:
            return
        btn.setText("Stop Tracking" if active else "Test Tracking")
        if not active:
            # Clear the dry-run flag so the next real recording opens
            # the writer normally.
            self._test_tracking_dry_run = False

    # ========================================================================
    # SETUP BUTTON HANDLERS
    # ========================================================================

    # Record/Stop drive the shared RunTask.on_record_clicked / on_stop_clicked
    # path. The RECORD branch (open MCU TSV + video + tracking writer) lives
    # in SetupWidget._start_recording → start_recording; the
    # DRY branch in SetupWidget._begin_temp_safety_run; post-start
    # bookkeeping in SetupWidget._post_record_start.

    def _tracking_roi(self, setup_id):
        """Maze ROI for the tracking-writer header, the arena ROI drawn on
        this box's camera (widget ``roi_segment`` / segment-config pixels,
        via the shared resolver)."""
        roi = self._get_box_roi(setup_id)
        return list(roi[:4]) if roi and len(roi) >= 4 else None

    # _video_recorder_geometry is shared on MainWindowBase (one resolver for
    # both modes; maze has no _get_box_roi so it takes the full-frame branch).

    def _on_pause_setup(self, setup_id):
        """Toggle pause/resume for this setup, based on the widget's
        is_paused flag."""
        sw = self._box_widgets.get(setup_id)
        if not sw or not sw.is_running:
            return
        if sw.is_paused:
            sw.resume()
        else:
            sw.pause()

    def _on_next_stage(self, setup_id):
        sw = self._box_widgets.get(setup_id)
        if not sw or not sw.is_running:
            return
        if not self._confirm(f"Next stage on Setup {setup_id}?",
                             f"Advance Setup {setup_id} to the next stage?"):
            return
        sw.next_stage()

    def _on_prev_stage(self, setup_id):
        sw = self._box_widgets.get(setup_id)
        if not sw or not sw.is_running:
            return
        if not self._confirm(f"Previous stage on Setup {setup_id}?",
                             f"Move Setup {setup_id} back to the previous stage?"):
            return
        sw.prev_stage()

    def _on_tab_title_changed(self, setup_id, title):
        """Update tab text when setup state changes."""
        sw = self._box_widgets.get(setup_id)
        if sw:
            idx = self.tabWidget.indexOf(sw)
            if idx >= 0:
                self.tabWidget.setTabText(idx, title)

    def _apply_tab_color(self, tab_index, color):
        """Apply accent color to a tab via icon indicator."""
        if tab_index < 0:
            return
        # Use a small colored square as tab icon
        pixmap = QtGui.QPixmap(10, 10)
        pixmap.fill(QtGui.QColor(color))
        self.tabWidget.setTabIcon(tab_index, QtGui.QIcon(pixmap))


    # _on_metadata_clicked is the shared slot on MainWindowBase.

    # iter_box_subject_widgets + iter_box_widgets inherited from
    # MainWindowBase (over _box_widgets).

    def apply_mode_extras(self, cfg):
        """Maze-specific load step: rebuild ``self.tracking_zones`` and
        ``self.tracking_enabled`` from per-box config + the optional
        ``ui.dialog_overrides["maze_zones"]`` blob.
        """
        self.tracking_zones.clear()
        self.tracking_enabled.clear()
        # Per-box zones land in setup_config.boxes[].zones; mirror into the
        # main-window dicts maze reads from. tracking_zones is the dialog/
        # renderer/zone-editor wire format: DICTS (drop-None, full fidelity)
        # via the ONE shared zone_to_dict converter, same result base's
        # _restore_zones_from_config produces, so the two agree.
        from source.config.experiment import zone_to_dict
        for b in cfg.setup_config.boxes:
            if b.zones:
                self.tracking_zones[int(b.setup_number)] = [
                    zone_to_dict(z) for z in b.zones]
            self.tracking_enabled[int(b.setup_number)] = bool(b.tracking_enabled)
        # Top-level "zones" dict preserved under ui.dialog_overrides["maze_zones"]
        # for round-trip parity.
        maze_zones = (cfg.ui.dialog_overrides or {}).get("maze_zones")
        if isinstance(maze_zones, dict):
            for k, v in maze_zones.items():
                try:
                    self.tracking_zones.setdefault(int(k), v)
                except (TypeError, ValueError):
                    continue

    def read_mode_extras(self, cfg):
        """Maze-specific save step: fold main_window-level dicts
        (``tracking_zones``, ``tracking_enabled``) back into the typed
        Config so the next load reproduces the maze state.
        """
        by_num = {b.setup_number: b for b in cfg.setup_config.boxes}
        # Per-box zones (tracking_zones → box.zones) are folded by the shared
        # experiment._read_tracking, which runs just before this hook in
        # read_ui_into_config, same source (host.tracking_zones), same
        # Zone.from_dict conversion. Maze only needs to mirror the per-box
        # online-tracking flags (its source of truth, since pose is deferred).
        for bn, on in (self.tracking_enabled or {}).items():
            box = by_num.get(int(bn))
            if box is not None:
                box.tracking_enabled = bool(on)
        # Preserve the top-level "zones" dict under ui.dialog_overrides.
        if self.tracking_zones:
            cfg.ui.dialog_overrides["maze_zones"] = {
                str(k): v for k, v in self.tracking_zones.items()
            }

    def _on_doors_clicked(self, setup_id):
        """Open door control dialog for this setup.

        Pre-checks only the hard blockers: MCU connection + framework idle.
        Door / hardware-definition presence is left to the dialog itself --
        it reads doors from the uploaded HD file host-side (cached _hw_config
        -> cached _hw_def_path -> project init_hw_def / board _loaded_hwd_path
        -> in-dialog 'Load HW Def' file picker). No MCU introspection.
        """
        sw = self._box_widgets.get(setup_id)
        if not sw:
            return
        if not sw.is_connected:
            QtWidgets.QMessageBox.warning(
                self, "Not Connected",
                "MCU is not connected for this setup.")
            return
        if sw.is_running:
            QtWidgets.QMessageBox.warning(
                self, "Framework Running",
                "Stop the framework before controlling doors manually.")
            return
        try:
            from source.gui.dialogs import DoorControlDialog
            dlg = DoorControlDialog(self, setup_id, parent=self)
            dlg.exec()
        except Exception as e:
            logger.error(f"Door control dialog error: {e}", exc_info=True)
            QtWidgets.QMessageBox.warning(
                self, "Door Control Error",
                f"Could not open door control:\n{e}")

    def _on_controls_clicked(self, setup_id):
        """Open Controls dialog for this setup."""
        try:
            from source.gui.dialogs import ControlsDialog
            sw = self._box_widgets.get(setup_id)
            if not sw:
                return

            dlg = ControlsDialog(setup_widget=sw, parent=self)
            sw._controls_dialog = dlg
            dlg.finished.connect(lambda: setattr(sw, '_controls_dialog', None))
            dlg.show()
        except Exception as e:
            logger.error(f"Controls dialog error: {e}")

    def _on_detach_setup(self, setup_id):
        """Detach a setup tab into its own window."""
        for i in range(self.tabWidget.count()):
            w = self.tabWidget.widget(i)
            if isinstance(w, SetupWidget) and w.setup_number == setup_id:
                self.tabWidget.detach_tab(i)
                break

    # ========================================================================
    # Tracking push policy management (the actual push is event-driven now,
    # wired Pose/TrackerSink result → MCUPusher inside the Pipeline).
    # _reset_tracking_policy lives in MainWindowBase.
    # ========================================================================

    # _on_record_stopped (below) is the maze recording/tracking teardown,
    # called from SetupWidget._after_stop.

    def _pre_record_stopped(self, setup_id, data):
        """Footer line into _video_data.txt while the tracking writer is
        still open, the shared teardown that follows is the sole closer
        of ``sw.tracking_writer``."""
        self._mirror_mcu_event_to_writer(
            setup_id, "framework_stopped",
            fw_ms=data.get("fw_ms") if isinstance(data, dict) else None,
        )

    def _post_record_stopped(self, setup_id, data):
        """Maze extra after the shared teardown: close the MCU pyControl
        data file (opened by start_recording)."""
        if setup_id in self._recording_ctx:
            sw = self._box_widgets.get(setup_id)
            if sw and sw.pycboard:
                try:
                    sw.pycboard.data_logger.close_files()
                except Exception as e:
                    logger.debug(f"Setup{setup_id}: close_data_file failed: {e}")
            self._recording_ctx.pop(setup_id, None)

    def _mirror_mcu_event_to_writer(self, setup_id, event_name, fw_ms=None):
        """Mirror a synthetic MCU-side event into the active
        ``_video_data.txt`` writer's ``events`` column.

        No-op if no tracking writer is open for this box. pyControl's own
        log remains authoritative; this is a convenience in-band copy for
        synthetic events such as ``framework_stopped``. ``fw_ms`` is accepted
        for caller compatibility but unused, per-row MCU timestamps come
        from ``pycboard.timestamp``.
        """
        try:
            sw = self._box_widgets.get(setup_id)
            writer = getattr(sw, "tracking_writer", None)
            if writer is None:
                return
            writer.note_event(name=str(event_name))
        except Exception as e:
            logger.debug(f"_mirror_mcu_event_to_writer failed ({event_name}): {e}")

    # ========================================================================
    # VIDEO FRAME POLLING & DISPLAY
    # ========================================================================

    # Frame polling lives in source/video/controller.py (Pipeline._tick).
    # Display, ROI lookup, BGR888 paint, and zone overlay are inherited from
    # MainWindowBase. Per-box rendering flows through the box-widget protocol
    # SetupWidget exposes (update_frame / clear_video / video_size /
    # set_fps_text / get_roi).

    # ========================================================================
    # RECORDING (one video per arena per session)
    # ========================================================================

    # _ensure_session_dirs lives on MainWindowBase (resolves via the box
    # widget's inherited RunTask._ensure_session_dirs), same path as operant.

    # Camera frame size / recording geometry now resolved centrally by
    # MainWindowBase._video_recorder_geometry (+ _box_segment_size /
    # _full_camera_wh), one path for both modes.

    # start_recording is the shared RunTask record path: SetupWidget.
    # _start_recording (the widget hook) opens the MCU TSV + tracking writer
    # + video recorder via the same central base.open_session_* calls operant
    # uses. No maze-specific record orchestrator on the MainWindow.

    # _build_annotate_callback lives in MainWindowBase.

    # ========================================================================
    # TRACKING
    # ========================================================================

    # _on_tracking_update lives in MainWindowBase.

    # ========================================================================
    # POSE ESTIMATION PIPELINE (DLC / SLEAP)
    # ========================================================================

    # _handle_dlc_init_from_dialog, _enable_pose_for_box,
    # _disable_pose_for_box, _shutdown_pose, _build_annotate_callback,
    # _configure_pose_for_box all live in MainWindowBase.
    # Maze hook below adds the Kalman-enhancer teardown for the pose path.
    # The per-session tracking writer is opened by the recording path
    # (start_recording), not here.


    def _pose_after_dialog_init(self, cfg, dialog, success_count):
        """Maze defers per-frame inference to Test Tracking.  After the
        dialog init loads the model, flip every per-box flag back to
        False so PoseSink stays idle until the user clicks Test Tracking."""
        for bid in list(self.pose_configs):
            self.tracking_enabled[bid] = False
            try:
                self.pipeline.disable_pose(bid)
            except Exception:
                pass

    def _close_tracking_writer(self, setup_id):
        """Close the tracking writer for a box if it's still open.

        Normally base._stop_recording_for_box is the sole closer. This
        handles the off-stop case (e.g. tracking disabled mid-run): close
        if still present, then clear."""
        sw = self._box_widgets.get(setup_id)
        writer = getattr(sw, "tracking_writer", None)
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
            sw.tracking_writer = None

    # ========================================================================
    # ZONE ADJUSTMENT (shift / rotate from SetupWidget buttons)
    # ========================================================================

    # _shift_zones / _rotate_zones / _scale_zones live in MainWindowBase now,
    # shared with operant. Both modes wire their per-box buttons to them.

    # _persist_zone_edit inherited from MainWindowBase (persists zone nudges
    # through the project config; no global side file).


    # ``self.video_segment_config`` is rebuilt from cfg.setup_config.boxes
    # geometry + roi_normalized on every load by the bridge in
    # source.config.experiment._apply_video_segment_mirror.

    # ========================================================================
    # UTILITY
    # ========================================================================

    # compute_ui_state lives in MainWindowBase; _iter_box_ids (inherited,
    # sorted over _box_widgets) feeds it.

    def _apply_mode_buttons(self, state, flags):
        """Maze-only buttons; the skeleton (locks, indicators, gates)
        lives in MainWindowBase._apply_ui_state."""
        has_boxes = flags["has_boxes"]
        any_running = flags["any_running"]
        block_run = flags["block_run"]

        if hasattr(self, "track_button"):
            # ``not _is_test_tracking_active()`` is what makes the preview
            # lock durable. Without it this 1 Hz refresh re-enabled Tracking
            # Config a second after the preview locked it, so the operator
            # could reconfigure tracking underneath a running test.
            self.track_button.setEnabled(
                has_boxes and not block_run
                and not self._is_test_tracking_active())
        if hasattr(self, "test_tracking_button"):
            has_tracking_cfg = self._any_box_tracking_configured()
            any_camera = bool(self.video_manager.box_camera_map)
            self.test_tracking_button.setEnabled(
                has_boxes and any_camera and has_tracking_cfg)
        if hasattr(self, "analyzer_button"):
            # Analysis disabled while ANY setup is running (per user spec).
            self.analyzer_button.setEnabled(not any_running)
        if hasattr(self, "stats_button"):
            self.stats_button.setEnabled(True)

    # _detect_gpu_encoding + _init_default_data_dir live in MainWindowBase.
    # _openDataFolder, _openTasksFolder, _browseDirectory, _toggleDebugMode,
    # _exportErrorLog, _appendErrorLog inherited from MainWindowUtilsMixin.

    def _data_dir_path(self):
        return Path(app_paths.data_dir)

    def _tasks_dir_path(self):
        return Path(app_paths.tasks_dir)

    # showError + _showErrorImpl inherited from MainWindowUtilsMixin.

    # ========================================================================
    # RESIZE / CLOSE
    # ========================================================================

    # resizeEvent + moveEvent + closeEvent live in MainWindowBase.
    # Sidebar height sync is automatic, _resize_event_extras default
    # iterates the registered sidebars. Maze-specific shutdown (tracking
    # writers, pose, MCU disconnect) runs via _close_event_extras.

    def _close_event_extras(self, event):
        # Per-arena: stop any still-active recorders, close tracking
        # writers, disable pose (frees GPU), disconnect pycboards.
        for setup_id, sw in list(self._box_widgets.items()):
            if getattr(sw, "video_recorder", None) is not None:
                try:
                    # The recorder teardown, not the bookkeeping hook.
                    self._stop_recording_for_box(setup_id)
                except Exception:
                    pass
        for setup_id in list(self._box_widgets.keys()):
            try:
                self._close_tracking_writer(setup_id)
            except Exception:
                pass
        try:
            self._shutdown_pose()
        except Exception:
            pass
        for sw in self._box_widgets.values():
            try:
                sw.disconnect_mcu()
            except Exception:
                pass
