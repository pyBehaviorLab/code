"""The operant mode's ``MainWindow``, many boxes, often one shared camera.

The other half of the fork described in ``docs/dev/module-map.md``:
``pyOperant.py`` and ``pyMaze.py`` load the same ``source/`` package and differ
only in the entry file and this subclass. The Pipeline, camera lifecycle,
tracking config, recording, zones and display tiles are all
:class:`~source.gui.base.MainWindowBase`; this file holds what an operant rig
is and a maze is not.

Which is, concretely:

* **several boxes on one camera**, split by ROI, the segment table drives the
  bus, and each box's coordinates are measured from its own slice;
* **fill-the-cell tiles** (``_tile_aspect_mode``), so a crowded grid spends its
  height on the animals rather than on letterbox bars, maze preserves aspect
  instead, because its picture is read against drawn zones;
* **a pyControl board per box**, with the task upload, variable dialogs and
  the async framework start that maze has no equivalent of;
* **per-box Record**, gated on the box's own subject and framework state.

Concurrency is per box throughout: each owns its MCU and its tracking, gates
apply per box, and the master buttons fan out over eligibility predicates so an
idle box stays clickable while another records.
"""

from datetime import datetime
from PySide6 import QtCore, QtGui, QtWidgets
from pathlib import Path

try:
    from serial.tools import list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from source.gui.styles import (
    COLORS, BUTTON_STYLE, COMBOBOX_STYLE, STATUS_BAR_STYLE,
    BOX_CONTROLS_GROUP_STYLE, HRULE_QSS,
)
from source.gui.theme import THEME
from source.gui.widgets import (
    DetachableTabWidget, BoxControlWidget, LiveStatusWidget,
    VideoStreamHolder, ErrorLogPanel, MarkdownView,
)
from source.gui.metadata_manager import MetadataManager
from source.log import get_logger

logger = get_logger()

# Suppress system beep on QMessageBox.warning dialogs (no-icon variant).
from source.gui.utility import install_quiet_qt_warnings
install_quiet_qt_warnings()


# DetachableTabWindow lives in widgets.py (used by both maze and operant).
from source.gui.widgets import DetachableTabWindow  # noqa: F401  (re-export for callers)


from .base import MainWindowBase


_OPERANT_DOC_FALLBACK = """
<html><body style="color: #f8f8f2; background-color: #1e1e1e;">
<h1 style="color: #ff79c6;">OperantBox Control</h1>
<h2 style="color: #bd93f9;">Session Workflow</h2>
<h3 style="color: #8be9fd;">At a Glance</h3>
<pre style="background:#111;padding:10px;border-radius:6px;color:#f8f8f2;line-height:1.4;">
Main Tab ── add boxes / subjects / tasks
     │
     ├── MCU: Connect → Upload/Config → Start
     ├── Live Status: Monitor stream + prints
     ├── Statistics: Metrics ▸ Export ▸ Detach
     └── Video (opt.): Connect ▸ Stream/Record
</pre>
<h3 style="color: #8be9fd;">Before You Start</h3>
<ul>
<li>Open <b>Experimental Info</b> sidebar → set experimenter, project, session, data folder.</li>
<li>Add boxes; pick COM ports, subject IDs, and tasks. Save config if you'll reuse it.</li>
</ul>
<h3 style="color: #8be9fd;">Run a Session</h3>
<ol>
<li><b>MCU:</b> In Setup Control, <i>Connect</i> boards → <i>Upload/Config</i> tasks → <i>Start</i> framework.</li>
<li><b>Video (optional):</b> Connect cameras, load segmentation, start stream/record.</li>
<li><b>Monitor:</b> Use <i>Live Status</i> for events/prints; detach if you want a floating view.</li>
<li><b>Stats:</b> Open <i>Statistics</i> for live metrics; <i>Export</i> TSV, <i>Detach</i> for a separate window.</li>
</ol>
<h3 style="color: #8be9fd;">Wrap Up</h3>
<ul>
<li>Stop framework, disconnect MCUs, stop video.</li>
<li>Export stats/logs if needed; use the <b>Logger</b> sidebar for issues (enable Debug for detail).</li>
</ul>
<h3 style="color: #8be9fd;">Keyboard Shortcuts</h3>
<pre style="background:#111;padding:10px;border-radius:6px;color:#f8f8f2;line-height:1.5;">
Ctrl+E   Experimental Info sidebar      Ctrl+O   Load project config
Ctrl+L   Logger sidebar                 Ctrl+M   Load metadata (cohort file)
Ctrl+D   Open data folder               Ctrl+P   Session plots
Ctrl+T   Open tasks folder              Ctrl++   Add box     Ctrl+-  Remove box
Ctrl+Tab Next tab   Ctrl+Shift+Tab  Previous tab
</pre>
</body></html>
"""


class MainWindow(MainWindowBase):
    """Main application window - GUI Process main thread."""

    def __init__(self):
        super().__init__()
        logger.info("Initializing MainWindow (single-process, pycboard in widget)")

        self.box_count = 0
        self.detached_tabs = {}
        self.info_fields = {}
        # BoxControlWidget exposes everything (.status_edit, .record_button)
        # directly.
        # Unified pipeline (camera → bus → sinks → MCU). Constructs
        # self.pipeline + self.bridge + self.video_manager alias and wires
        # bridge signals to base slots.
        self._setup_pipeline(target_fps=30)
        self.tracker_manager = self.pipeline.tracker_manager
        self.tracker_manager.register_callback(self._on_tracking_update)
        self.tracking_enabled = {}
        self.recording_setups = set()
        # Operant tiles FILL their cell: the picture is scaled to the whole
        # tile whatever the cell's shape. A grid of sixteen chambers is read by
        # glancing across it, and letterboxing every cell spends the scarce
        # thing - screen height - on bars rather than on the animal.
        #
        # The distortion this admits is acceptable here and not in maze mode.
        # An operant tile is watched to see what the animal is doing; a maze
        # arena is watched against drawn zones, where a stretched picture would
        # put the zone outline somewhere the animal is not. Maze therefore sets
        # "preserve" (see gui/maze.py).
        #
        # Overlay geometry is unaffected either way: keypoints and zones are
        # transformed through the same mapping as the image, so they stay on
        # the pixels they were computed from at any tile shape.
        self._tile_aspect_mode = "stretch"
        self.video_target_fps = 30  # Default FPS, configurable in Camera Connect Dialog
        self.video_camera_resolution = None  # (w, h) chosen in Camera Connect Dialog
        self.video_camera_realistic_fps = None  # measured ceiling at chosen resolution
        self.video_frame_strategy = "accept"  # single file, no remux

        self.metadata_manager = MetadataManager()

        # Pose runtime state lives on ``pose_configs`` + ``pose_zone_state``
        # (shared with maze via MainWindowBase pose lifecycle methods).
        self.pose_configs = {}
        self.pose_zone_state = {}
        # ``self._overlay`` (per-box OverlayState) lives on MainWindowBase.
        self.tracking_zone_paths = {}  # box_id -> filepath for auto-save
        self.dlc_global_model_path = None
        # Inference is owned by self.pipeline.pose (PoseSink wrapping
        # ThreadInferenceBackend); single-worker because pose models are
        # not thread-safe.

        # Per-box state owned by main_window (UI-side). The pipeline reads
        # zones via the per-box zone_lookup we pass into pipeline.enable_pose;
        # the live overlay reads them directly.
        self.tracking_zones = {}
        # Per-box tracking state lives on ``self.pipeline._tracking_configs``;
        # global dialog-only fields on ``self._tracking_dialog_globals`` (base).

        # Display-side accounting, the polling paint loop shows the latest
        # frame per box (optional cap via cfg.display.max_fps).
        self._display_frame_counts = {}

        # Cache video stream widgets for fast lookup (avoids loop on every frame)
        self.video_stream_widget_cache = {}

        self._build_ui()
        self._init_default_data_dir()
        # Draft working config so create-phase edits accumulate in memory
        # and transfer on the first Save.
        self._init_working_config()

        # Initialize button states (disabled when no boxes)
        self.refresh_ui_state()

        # Detect GPU encoding capabilities on startup
        self._detect_gpu_encoding()

        # Shared timer pipeline: refresh_timer (1 Hz, always-on folder rescan
        # + per-box hash poll) + process_timer (10 ms MCU drain + plot +
        # display + clock, started only when active via _sync_timer_mode).
        self._start_refresh_timer()

        # Register keyboard shortcuts
        from source.gui.utility import init_keyboard_shortcuts
        shortcuts = {
            "Ctrl+D": self._openDataFolder,
            "Ctrl+T": self._openTasksFolder,
            "Ctrl+E": lambda: self._toggle_sidebar("info"),       # Experimental info
            "Ctrl+L": lambda: self._toggle_sidebar("errorlog"),   # Logger
            "Ctrl+M": self.load_cohort_metadata,                  # Load Meta
            "Ctrl+O": self.load_config,                           # Load project config
            "Ctrl+P": self.show_universal_plot_dialog,            # Session plots
            "Ctrl+=": self.add_setup,                             # Add setup (also Ctrl++)
            "Ctrl++": self.add_setup,
            "Ctrl+-": self.remove_setup,                       # Remove setup
            "Ctrl+Tab": lambda: self._navigate_tab(1),           # Next tab
            "Ctrl+Shift+Tab": lambda: self._navigate_tab(-1),    # Previous tab
        }
        init_keyboard_shortcuts(self, shortcuts)

        logger.debug("MainWindow initialization complete")

    # _detect_gpu_encoding lives in MainWindowBase.

    def _build_ui(self):
        """Initialize the main UI"""
        self.setObjectName("MainWindow")
        # Wide enough that the per-box row (Subject + Task + Upload/Start/Stop/
        # Timer + Controls + status) fits without truncating task names.
        self.resize(1000, 600)
        self.setMinimumSize(920, 500)
        self.setWindowTitle("pyBehaviorLab - Operant")

        self._apply_window_icon('logo.svg')

        self.centralwidget = QtWidgets.QWidget()
        self.centralwidget.setObjectName("mwCentral")
        # Glassy white outline so the window edge reads against a dark desktop.
        # Applied directly here because the global QSS hook via
        # `QMainWindow > QWidget#mwCentral` doesn't always render.
        self.centralwidget.setStyleSheet(
            "QWidget#mwCentral {"
            f" background-color: {THEME.palette.bg};"
            " border: 1px solid rgba(255,255,255,0.18);"
            "}"
        )
        self.setCentralWidget(self.centralwidget)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(4)
        # Left margin = sidebar toggle-button width (22 px) + 4 px gap so
        # the Main Control tab starts after the Experimental Info button.
        self.verticalLayout.setContentsMargins(26, 4, 4, 0)

        self.tabWidget = DetachableTabWidget()
        self.tabWidget.setStyleSheet(
            "QTabBar::tab {"
            f" font: 700 11pt '{THEME.font.family}';"
            " padding: 5px 16px;"
            " min-width: 92px;"
            "}"
        )
        self.verticalLayout.addWidget(self.tabWidget)

        self.mainTab = QtWidgets.QWidget()
        self.liveStatusTab = QtWidgets.QWidget()
        self.statisticsTab = None  # Will be created on first use
        self.videoStreamTab = QtWidgets.QWidget()

        self.tabWidget.addTab(self.mainTab, "Main Control")
        self.tabWidget.addTab(self.liveStatusTab, "Live Status")
        # Statistics tab will be added dynamically
        self.tabWidget.addTab(self.videoStreamTab, "Video Stream")

        self._build_main_tab_ui()
        self._build_live_status_tab_ui()
        self._build_statistics_tab_ui()
        self._build_video_stream_tab_ui()
        # Offline analysis lives in tools/offline_analysis/ launched as a
        # subprocess from the master toolbar's "Analyzer" button. No tab.

        self.statusbar = QtWidgets.QStatusBar()
        self.statusbar.setMinimumHeight(26)
        self.setStatusBar(self.statusbar)

        # Status badges -- project + tracking state.
        self.statusbar.addPermanentWidget(self._build_status_panel())

        self.statusbar.setStyleSheet(STATUS_BAR_STYLE)

        self._title_bar = None
        self.setWindowTitle("pyBehaviorLab - Operant")

        logger.debug("Main UI setup complete")

    # show_temporary_message comes from MainWindowUtilsMixin (shared).

    # createHelpButton, showError, _showErrorImpl inherited from
    # MainWindowUtilsMixin. The mixin's _showErrorImpl appends to
    # self.errorLogBrowser when present.

    # get_all_setup_widgets inherited from MainWindowBase (reads the
    # base-owned _box_widgets registry).

    # get_setup_widget removed, it duplicated MainWindowBase._setup_widget_for
    # and, being operant-only, made shared code that probed for it silently
    # skip maze entirely (see box_alerts).

    def _after_universal_dialog_closed(self) -> None:
        """Operant extra on top of the base refresh: stay on the main tab."""
        if hasattr(self, 'tabWidget'):
            self.tabWidget.setCurrentIndex(0)
        super()._after_universal_dialog_closed()

    # =========================================================================
    # Tracking Methods (Blob Detection Object Tracking)
    # =========================================================================

    # _on_tracking_update + _reset_tracking_policy live in MainWindowBase.
    # Operant default for push policy: nothing pushed. User opts in by
    # configuring triggers in Zone Config; the pipeline's MCUPusher then
    # auto-pushes on each Pose/TrackerSink result (event-driven).

    # show_universal_plot_dialog moved to MainWindowBase (shared by both modes).

    def _build_main_tab_ui(self):
        """Setup the main tab UI"""
        try:
            main_tab_layout = QtWidgets.QVBoxLayout(self.mainTab)
            main_tab_layout.setSpacing(4)
            # Zero left margin here, the outer verticalLayout already
            # carries the 26 px gutter that clears the sidebar toggle button.
            main_tab_layout.setContentsMargins(0, 0, 0, 0)

            # Thin glow rule between the tab bar and the content area,
            # the "top line that was there before" per the latest feedback.
            # Same gradient recipe as the two rules inside the groupbox so
            # all three read as one design language.
            top_rule = QtWidgets.QFrame()
            top_rule.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            top_rule.setFixedHeight(1)
            top_rule.setStyleSheet(HRULE_QSS)
            main_tab_layout.addWidget(top_rule, 0)

            self.mainSplitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            main_tab_layout.addWidget(self.mainSplitter)

            # Create Box Controls groupbox as placeholder/separator
            self.controlGroupBox = QtWidgets.QGroupBox("")  # No title
            self.controlLayout = QtWidgets.QVBoxLayout(self.controlGroupBox)
            self.controlLayout.setSpacing(2)
            self.controlLayout.setContentsMargins(5, 2, 5, 2)
            self.controlGroupBox.setStyleSheet(BOX_CONTROLS_GROUP_STYLE)
            self._build_box_controls()

            self.mainSplitter.addWidget(self.controlGroupBox)

            # Left-edge overlay sidebars (Experimental Info / Error Log /
            # Documentation) via the shared base loop over _sidebar_defs().
            self._build_left_sidebars()

            logger.debug("Main tab UI setup complete")

        except Exception as e:
            logger.error(f"Error setting up main tab UI: {str(e)}")
            self.showError(f"Failed to setup main tab: {str(e)}")

    # resizeEvent / moveEvent + sidebar height-sync live in MainWindowBase
    # (auto-iterates self._sidebars).

    # ------------------------------------------------------------------
    # Tracking-init hooks (operant-specific overrides)
    # Called by Camera Connect when tracking is enabled + model configured.
    # ------------------------------------------------------------------

    def _box_wants_tracking(self, box) -> bool:
        """Operant: tracking is on for every box once any TC is configured."""
        return self._any_box_tracking_configured()

    # save_config / load_config inherited from MainWindowBase; the bridge in
    # experiment.apply_config_to_ui / read_ui_into_config owns persistence.

    def _load_mode_name(self):
        return "operant"

    def _load_clear_existing_state(self):
        """Confirm before clearing existing boxes (operant prompts; maze
        clears silently). Return False to cancel."""
        if self.dynamicBoxes.count() > 0:
            if not self._confirm(
                'Confirm Load',
                'Current boxes will be removed. Continue?'):
                return False
            self._teardown_all_boxes()      # this also clears runtime state
        else:
            # No boxes to tear down, but the pipeline registries and the
            # host's zone/enable dicts can still hold the previous project's
            # entries, a box removed by hand leaves them behind. Maze clears
            # unconditionally; operant reached the clear only through
            # _teardown_all_boxes, which early-returns on an empty rig, so
            # loading a second project could inherit the first one's state.
            self._clear_project_runtime_state()
        return True

    # Per-box state (including ROI in pixel and percent forms) is restored by
    # the bridge in experiment.apply_config_to_ui. Operant's only mode-specific
    # load step is the DLC-init flag flip in _load_post_restore.

    def _load_post_restore(self, cfg):
        """Operant post-load: refresh live-status / video grid / tracking
        toggle button, then prompt about DLC init when relevant.

        Receives the typed v3 ``Config`` directly, no dict round-trip.
        """
        # No init flag. Whether a box can track is answered by asking the
        # loaded model, through ``pose_box_is_ready``, and readiness itself is
        # driven by the camera coming up. A flag set here and cleared somewhere
        # else was a second account of the same fact, and the two disagreed.
        self.update_live_status()
        self.update_video_streams()
        self._update_test_tracking_button()

    def add_setup(self):
        """Add a new experimental box to the interface"""
        try:
            self.box_count += 1
            current_box_id = self.box_count
            setup_widget = BoxControlWidget(setup_number=current_box_id, main_window=self)
            # Wire the box-number button → per-subject metadata dialog
            # (same dialog maze opens from its Meta button).
            setup_widget.metadata_clicked.connect(self._on_metadata_clicked)

            # BoxControlWidget exposes every per-box control as a direct
            # attribute (record_button, status_edit, camera_id_edit, ...).
            self._box_widgets[current_box_id] = setup_widget
            # Each BoxControlWidget paints its own card surface; the parent
            # layout's spacing provides the gap between rows.
            self.dynamicBoxes.addWidget(setup_widget)

            # Record-button click → operant.start_recording (bookkeeping). The
            # widget's on_record_clicked does the per-box recording setup
            # (data_logger, video_recorder, framework start);
            # _start_framework_backend short-circuits with (True, "") when
            # framework_running is already True, so the lambda runs the
            # operant-side success bookkeeping.
            #
            # Stop-button click → box_widget.on_stop_clicked (single handler):
            # confirm (RECORD) → _on_stop_confirmed UX hook → mcu_stop runs the
            # teardown synchronously. framework_stopped_signal →
            # _on_record_stopped finalises the operant-side cleanup.
            # Record button is connected ONCE, to the widget's on_record_clicked
            # (in BoxControlWidget). Operant-side bookkeeping lives in the shared
            # _post_record_start hook, no second connection, no double start.
            # framework_stopped_signal is emitted from RunTask.mcu_stop for
            # every exit path (user / auto / error), so this one slot covers
            # all of them, including auto-stop, which never touches the Stop
            # button.
            setup_widget.framework_stopped_signal.connect(self._on_record_stopped)
            # Canonical auto-start-tracking hook: fires the moment the MCU
            # framework boots on any path (record button, multi-start, restart).
            setup_widget.framework_started_signal.connect(self._on_framework_auto_started)

            self.update_live_status()
            self.update_video_streams()

            # Update statistics tab with new box count
            if self.statisticsTab:
                self.statisticsTab.updateBoxCount(self.box_count)

            # Populate task menu from folder tree
            tasks_dir = str(Path(__file__).resolve().parents[2] / "tasks")
            setup_widget.task_combo.update_menu(tasks_dir)

            # Populate COM ports
            self.populate_com_ports(setup_widget.serial_combo)

            self.refresh_ui_state()
            self._project_changed(reason="boxes_changed")

            logger.info(f"Added new box with ID {current_box_id}")

        except Exception as e:
            logger.error(f"Error adding new box: {str(e)}")
            self.showError(f"Failed to add new box: {str(e)}")

    def _teardown_all_boxes(self):
        """Tear down every box: stop recordings, disconnect cameras + MCU
        boards, clear pipeline + host runtime state, remove the widgets.

        No confirmation prompt; this is the internal clear used by the Load
        flow (``_load_clear_existing_state`` asks once itself). There is no
        user-facing 'remove all' button; single-box Remove is the only UI
        action, matching maze.
        """
        try:
            if self.dynamicBoxes.count() == 0:
                return

            # Stop all recordings first
            for setup_id in list(self.recording_setups):
                self.stop_recording(setup_id)

            # Disconnect all cameras
            logger.info("Disconnecting all cameras before removing boxes")
            self.disconnect_all_cameras()

            # Disconnect all MCU boards
            logger.info("Disconnecting all boards before removing boxes")
            self.disconnect_all_boards()

            # Through the pipeline, which owns the manager. Calling
            # video_manager.cleanup() directly tore the cameras down behind
            # the Pipeline's back, leaving its buses and sink subscriptions
            # pointing at threads that no longer exist. Not shutdown(), the
            # Pipeline has to survive; only its cameras go.
            self.pipeline.release_all_cameras()

            # Clear pipeline + host runtime state so the pipeline TC /
            # camera-config registries and the host's tracking_zones /
            # tracking_enabled / _tracking_dialog_globals dicts don't keep stale
            # entries for box numbers that no longer exist.
            self._clear_project_runtime_state()

            # Remove all box widgets
            while self.dynamicBoxes.count():
                item = self.dynamicBoxes.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            self.box_count = 0
            self.video_stream_widget_cache.clear()
            # Reset the box index so update_live_status rebuilds from empty.
            self._box_widgets.clear()

            self.update_live_status()
            self.update_video_streams()

            # Update statistics tab with new box count
            if self.statisticsTab:
                self.statisticsTab.updateBoxCount(self.box_count)

            self.refresh_ui_state()
            self._project_changed(reason="boxes_changed")

            logger.info("All boxes removed")

        except Exception as e:
            logger.error(f"Error removing all boxes: {str(e)}")
            self.showError(f"Failed to remove all boxes: {str(e)}")

    # _on_metadata_clicked is the shared slot on MainWindowBase (it calls
    # this operant-only update_metadata_button_states refresh via getattr).

    def remove_setup(self):
        """Remove the last added box from the interface (asks to confirm)."""
        try:
            if self.dynamicBoxes.count() == 0:
                self.statusbar.showMessage("No boxes to remove", 3000)
                return

            # Get the last box
            last_index = self.dynamicBoxes.count() - 1
            item = self.dynamicBoxes.itemAt(last_index)
            if not (item and item.widget()):
                return
            widget_to_remove = item.widget()
            last_box_number = getattr(widget_to_remove, "setup_number", "?")

            # Confirmation, default No so an accidental click of the small
            # Remove Box button doesn't destroy state.
            if not self._confirm(
                "Confirm Remove Box",
                f"Remove Box {last_box_number}?\n\n"
                f"This drops the box's widget, its camera assignment, "
                f"and any in-memory state. The on-disk project file "
                f"is unchanged until the next save / autosave.",
            ):
                logger.info("Remove Box cancelled by user")
                return

            # Get the authoritative box_id from the widget itself.
            last_box_id = getattr(widget_to_remove, "setup_number", None)

            # Stop recording if this box is recording
            if last_box_id and last_box_id in self.recording_setups:
                self.stop_recording(last_box_id)

            # Drop ALL box-keyed state, regardless of whether a camera
            # was ever connected (the union clear, shared with maze).
            if last_box_id is not None:
                self._drop_box_state(last_box_id)

            # Detach + queue deletion synchronously.
            self.dynamicBoxes.takeAt(last_index)
            widget_to_remove.setParent(None)
            widget_to_remove.deleteLater()

            self.box_count -= 1

            self.update_live_status()
            self.update_video_streams()

            # Update statistics tab with new box count
            if self.statisticsTab:
                self.statisticsTab.updateBoxCount(self.box_count)

            self.refresh_ui_state()

            logger.info(f"Box {last_box_id} removed")
            self.statusbar.showMessage("Box removed", 3000)

        except Exception as e:
            logger.error(f"Error removing last box: {str(e)}")
            self.showError(f"Failed to remove last box: {str(e)}")

    def _sidebar_defs(self):
        """Per-mode left-sidebar table for the shared
        ``MainWindowBase._build_left_sidebars`` loop. Each row:
        (key, sidebar_attr, toggle_attr, toggle_text, (clr, hover, press),
         y, height, header, width, content_builder)."""
        return [
            ("info", "infoSidebar", "info_toggle_btn", "Experimental Info",
             ("#bd93f9", "#ff79c6", "#8be9fd"), 50, 150,
             "EXPERIMENT INFO", 460, self._build_experiment_info_sidebar),
            ("errorlog", "errorLogSidebar", "errorlog_toggle_btn", "Logger",
             ("#ff6b6b", "#ff79c6", "#ff5555"), 210, 120,
             "ERROR LOG", 500, self._build_error_log_sidebar),
            ("doc", "docSidebar", "doc_toggle_btn", "Documentation",
             ("#0f3460", "#1e88e5", "#1565c0"), 340, 150,
             "DOCUMENTATION", 550, self._build_documentation_sidebar),
        ]

    def _build_experiment_info_sidebar(self):
        """Setup experiment info fields + metadata buttons in the sidebar.

        Metadata buttons (Load / Edit / Auto-Populate) come from the shared
        ``build_metadata_buttons`` so operant and maze mirror each other."""
        try:
            self.populate_experiment_info_fields(self.infoSidebar)
            self.build_metadata_buttons(self.infoSidebar)
            logger.debug("Experiment info sidebar setup complete")
        except Exception as e:
            logger.error(f"Error setting up experiment info sidebar: {str(e)}")
            self.showError(f"Failed to setup experiment info sidebar: {str(e)}")

    # create_sidebar_help_button comes from MainWindowUtilsMixin (shared).

    def _build_error_log_sidebar(self):
        """Embed the shared ErrorLogPanel in the error-log sidebar."""
        try:
            panel = ErrorLogPanel(
                style="dark",
                on_debug_toggled=self._toggleDebugMode,
                on_export=self._exportErrorLog,
            )
            self.errorLogBrowser = panel.browser
            self.debugToggleButton = panel.debug_button
            self.errorLogSidebar.content_layout.addWidget(panel, stretch=1)
        except Exception as e:
            logger.error("Error setting up error log sidebar: %s", e)

    def _build_documentation_sidebar(self):
        """Embed the shared MarkdownView in the documentation sidebar."""
        try:
            self.docBrowser = MarkdownView()
            # Resolve relative to the repo root so it loads regardless of CWD.
            doc_path = Path(__file__).resolve().parents[2] / "GUI_DOCUMENTATION.md"
            self.docBrowser.load_md_file(doc_path, fallback_html=_OPERANT_DOC_FALLBACK)
            self.docSidebar.content_layout.addWidget(self.docBrowser, stretch=1)
        except Exception as e:
            logger.error("Error setting up documentation sidebar: %s", e)

    # markdownToHtml + _processInlineMarkdown extracted to
    # source/gui/widgets/markdown_view.py (MarkdownView).

    # _openDataFolder, _openTasksFolder inherited from MainWindowUtilsMixin

    def _build_box_controls(self):
        """Setup box controls section"""
        try:
            # Add Setup Control and Camera Control buttons at top
            self._build_top_control_buttons()

            # Divider between the action row and the box list.
            separator = QtWidgets.QFrame()
            separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            separator.setFixedHeight(1)
            separator.setStyleSheet(HRULE_QSS)
            self.controlLayout.addWidget(separator, 0)

            # Add scroll area with boxes, token-driven thin scrollbars
            # come from the global stylesheet (scroll_area_style).
            self.scrollArea = QtWidgets.QScrollArea()
            self.scrollArea.setWidgetResizable(True)
            self.scrollArea.setContentsMargins(0, 0, 0, 0)
            self.scrollArea.setStyleSheet("QScrollArea { border: none; }")
            self.scrollAreaWidgetContents = QtWidgets.QWidget()
            self.boxLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
            # Gap between box rows so cards read as separate cards.
            self.boxLayout.setSpacing(3)
            self.boxLayout.setContentsMargins(0, 0, 0, 0)
            self.boxLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
            self.dynamicBoxes = QtWidgets.QVBoxLayout()
            self.dynamicBoxes.setSpacing(4)  # gap between setup rows
            self.dynamicBoxes.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
            self.boxLayout.addLayout(self.dynamicBoxes)
            self.scrollArea.setWidget(self.scrollAreaWidgetContents)
            self.controlLayout.addWidget(self.scrollArea)

            # No rule above SETUP CONTROL, the section header gives enough
            # visual separation on its own.

            # Now add Setup Control (MCU control) at the bottom
            self._build_bottom_control_buttons()

            logger.debug("Box controls setup complete")

        except Exception as e:
            logger.error(f"Error setting up box controls: {str(e)}")
            self.showError(f"Failed to setup box controls: {str(e)}")

    # ------------------------------------------------------------------
    # Top control buttons
    # ------------------------------------------------------------------
    def _build_top_control_buttons(self):
        """Setup control buttons at top (Box Setup + Camera Control).

        Orchestrator only, each section is its own builder.
        """
        try:
            button_layout = QtWidgets.QHBoxLayout()
            button_layout.setSpacing(4)   # gap between Box Setup / Experiment / Camera Control
            button_layout.setContentsMargins(0, 0, 0, 0)
            button_layout.addWidget(self._build_box_setup_section())
            button_layout.addWidget(self._build_experiment_section())
            button_layout.addWidget(self._build_camera_control_section())
            self.controlLayout.addLayout(button_layout, 0)
            logger.debug("Top control buttons setup complete")
        except Exception as e:
            logger.error(f"Error setting up top control buttons: {str(e)}")
            self.showError(f"Failed to setup top control buttons: {str(e)}")

    def _build_box_setup_section(self):
        """Box Setup group: Add / Remove / Remove-All."""
        container, group_layout = self._make_control_section_container(
            "Box Setup",
            "Box Setup Functions:\n"
            "- Add Box: Add a new box control row\n"
            "- Remove Box: Remove the last added box\n"
            "- Remove All: Remove all box controls",
        )
        buttons_config = [
            ("Add Box",    "add.svg",    self.add_setup,       "primary", "add_box_button"),
            ("Remove Box", "remove.svg", self.remove_setup, "danger",  "remove_box_button"),
        ]
        for text, icon_file, callback, color_key, attr in buttons_config:
            btn = self._make_control_section_button(text, icon_file, callback, color_key)
            setattr(self, attr, btn)
            group_layout.addWidget(btn)
        return container

    def _build_experiment_section(self):
        """Experiment group: Save / Load config, its own group so config
        persistence reads separately from Box Setup."""
        container, group_layout = self._make_control_section_container(
            "Experiment",
            "Experiment Functions:\n"
            "- Save Config: Save current configuration to file\n"
            "- Load Config: Load configuration from file",
        )
        buttons_config = [
            ("Save", "save.svg",   self.save_config, "success", "save_config_button"),
            ("Load", "folder.svg", self.load_config, "warning", "load_config_button"),
        ]
        for text, icon_file, callback, color_key, attr in buttons_config:
            btn = self._make_control_section_button(text, icon_file, callback, color_key)
            setattr(self, attr, btn)
            group_layout.addWidget(btn)
        return container

    def _build_camera_control_section(self):
        """Camera Control group: Camera Config + Tracking Config + Test Tracking."""
        container, group_layout = self._make_control_section_container(
            "Camera Control",
            "Camera Control Functions:\n"
            "- Connect Camera: Connect cameras to boxes with automatic segmentation",
        )
        buttons_config = [
            ("Camera Config",   "camera.svg", self.show_camera_config_dialog,  "primary", "camera_connect_button"),
            ("Tracking Config", "chip.svg",   self.show_tracking_config_dialog, "info",   "track_button"),
            ("Test Tracking",   "video.svg",  self.toggle_test_tracking,        "info",   "test_tracking_button"),
        ]
        for text, icon_file, callback, color_key, attr in buttons_config:
            btn = self._make_control_section_button(text, icon_file, callback, color_key)
            setattr(self, attr, btn)
            group_layout.addWidget(btn)
        # Gated buttons, disabled until prerequisites are met.
        self.track_button.setEnabled(False)
        self.track_button.setToolTip("Connect camera first")
        self.test_tracking_button.setEnabled(False)
        self.test_tracking_button.setToolTip(
            "Configure tracking first (use Tracking Config)"
        )
        return container

    # ------------------------------------------------------------------
    # Bottom control buttons
    # ------------------------------------------------------------------
    def _build_bottom_control_buttons(self):
        """Setup control buttons at bottom (Setup Control row)."""
        try:
            button_layout = QtWidgets.QHBoxLayout()
            button_layout.setSpacing(10)
            button_layout.setContentsMargins(0, 0, 0, 0)
            button_layout.addWidget(self._build_setup_control_section())
            self.controlLayout.addLayout(button_layout, 0)
            logger.debug("Bottom control buttons setup complete")
        except Exception as e:
            logger.error(f"Error setting up bottom control buttons: {str(e)}")
            self.showError(f"Failed to setup bottom control buttons: {str(e)}")

    def _build_setup_control_section(self):
        """Setup Control group: Meta, Init Setup, Boards, Config, Upload,
        Disconnect, Analysis. Labels and colors align to
        maze.py:_build_bottom_control_buttons so a user moving between
        modes sees one consistent action language.
        """
        from source.gui.style_builders import master_group_qss
        container, group_layout = self._make_control_section_container(
            "Setup Control",
            "Setup Control Functions:\n"
            "- Meta: Clear loaded metadata\n"
            "- Setup: Launch the experiment using the active project config\n"
            "- Boards (plug icon): Connect boxes to COM ports\n"
            "- Session plot: Show real-time plots for all running boxes\n"
            "- Boards (gear icon): Configure selected boards with multiple actions\n"
            "- Task: Upload a task to selected boxes\n"
            "- Disconnect: Disconnect MCU boards\n"
            "- Test Tracking: Quick preview of camera tracking on configured boxes "
            "(no recording, for verifying setup before clicking Record)\n"
            "- Analyzer: Open the standalone offline analyzer in a separate "
            "process (works on saved sessions, not the live one)",
            group_qss=master_group_qss(),
            group_spacing=6,
        )
        # Tuple shape: (label, icon, callback, color, attr_name, post_spacing)
        master_buttons_config = [
            ("Clear Meta",     "trash.svg",      self.clear_metadata,                "warning",   "clear_metadata_button",  6),
            ("Connect boards", "connect.svg",    self.show_connect_dialog,           "primary",   "connect_button",          0),
            ("Session Plot",   "bar-graph.svg",  self.show_universal_plot_dialog,    "info",      "plot_button",             0),
            ("Config boards",  "settings.svg",   self.show_config_dialog,            "primary",   "config_button",           0),
            ("Multi-Start",    "play.svg",       self.show_universal_start_dialog,   "success",   "start_button",            0),
            ("Multi-Stop",     "stop.svg",       self.show_universal_stop_dialog,    "warning",   "stop_button",             0),
            ("Upload Task",    "upload.svg",     self.show_upload_dialog,            "info",      "upload_task_button",    8),
            ("Disconnect",     "disconnect.svg", self.show_disconnect_dialog,        "warning",   "disconnect_button",      12),
            ("Analysis",       "bar-graph.svg",  self.open_offline_analyzer,         "success",   "analyzer_button",         0),
        ]
        # Master row uses slightly larger text than top sections.
        master_btn_extra_qss = " QPushButton { font-size: 10pt; }"
        for text, icon_file, callback, color_key, attr, post_spacing in master_buttons_config:
            btn = self._make_control_section_button(
                text, icon_file, callback, color_key,
                extra_qss=master_btn_extra_qss,
            )
            setattr(self, attr, btn)
            group_layout.addWidget(btn)
            if post_spacing:
                group_layout.addSpacing(post_spacing)

        # Per-button gating + tooltips (kept outside the table so the
        # table stays a pure declaration).
        self.clear_metadata_button.setEnabled(False)
        self.clear_metadata_button.setToolTip("No metadata loaded")
        self.connect_button.setToolTip("Connect boards to COM ports")
        self.config_button.setToolTip(
            "Configure boards (hardware definition, settings)"
        )
        self.upload_task_button.setToolTip(
            "Upload task to selected boards"
        )
        self.start_button.setToolTip(
            "Multi-Start: pick from boxes that are ready "
            "(connected + uploaded task + Start button armed); "
            "clicks each selected box's Start button"
        )
        self.stop_button.setToolTip(
            "Click each selected box's Stop button, asks to confirm"
        )
        self.analyzer_button.setToolTip(
            "Open the standalone offline analyzer in a separate "
            "process, works on saved sessions, not the live one"
        )
        return container

    # ------------------------------------------------------------------
    # Shared control-section helpers (used by top + bottom builders)
    # ------------------------------------------------------------------
    def _build_live_status_tab_ui(self):
        """Setup the live status tab UI"""
        try:
            live_status_layout = QtWidgets.QVBoxLayout(self.liveStatusTab)
            # Zero left margin, the outer verticalLayout already
            # carries the 26 px sidebar-toggle gutter.
            live_status_layout.setContentsMargins(0, 5, 5, 5)
            live_status_layout.setSpacing(5)

            self.liveStatusGroupBox = QtWidgets.QGroupBox("Live Status")
            live_status_layout.addWidget(self.liveStatusGroupBox)

            self.liveStatusLayout = QtWidgets.QGridLayout(self.liveStatusGroupBox)
            self.liveStatusLayout.setSpacing(5)

            self.detachLiveStatusButton = QtWidgets.QPushButton("Detach Live Status")
            self.detachLiveStatusButton.clicked.connect(lambda: self.detach_tab("Live Status"))
            self.detachLiveStatusButton.setStyleSheet(BUTTON_STYLE.format(
                color=COLORS['primary'],
                hover_color=COLORS['primary_hover']
            ))
            live_status_layout.addWidget(self.detachLiveStatusButton, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

            logger.debug("Live status tab setup complete")

        except Exception as e:
            logger.error(f"Error setting up live status tab: {str(e)}")
            self.showError(f"Failed to setup live status tab: {str(e)}")

    def _build_statistics_tab_ui(self):
        """Setup the statistics tab UI"""
        try:
            from source.stats import StatsCanvas

            # Create statistics tab widget
            self.statisticsTab = StatsCanvas(self)
            # Per-row Controls button → open that box's Controls dialog.
            self.statisticsTab.box_controls_requested.connect(
                self._open_box_controls_from_stats)

            # Insert after Live Status tab (index 2)
            self.tabWidget.insertTab(2, self.statisticsTab, "Statistics")

            # Add detach button at the top
            self.detachStatisticsButton = QtWidgets.QPushButton("Detach")
            self.detachStatisticsButton.clicked.connect(lambda: self.detach_tab("Statistics"))
            self.detachStatisticsButton.setStyleSheet(BUTTON_STYLE.format(
                color=COLORS['primary'],
                hover_color=COLORS['primary_hover']
            ))
            self.detachStatisticsButton.setFixedSize(110, 24)

            # Add detach button to statistics tab header layout
            if hasattr(self.statisticsTab, 'header_layout'):
                self.statisticsTab.header_layout.addWidget(
                    self.detachStatisticsButton
                )

            logger.debug("Statistics tab setup complete")

        except Exception as e:
            logger.error(f"Error setting up statistics tab: {str(e)}")
            self.showError(f"Failed to setup statistics tab: {str(e)}")

    def _open_box_controls_from_stats(self, setup_id):
        """Open a box's Controls dialog from its stats-table row button.
        Mirrors the real Controls button: only acts when that button is
        enabled (connected, not locked), checked at click time, no polling."""
        bw = self._setup_widget_for(setup_id)
        if bw is None:
            return
        btn = getattr(bw, "controls_button", None)
        if btn is not None and btn.isEnabled() and hasattr(bw, "onControlsClicked"):
            bw.onControlsClicked()

    def _build_video_stream_tab_ui(self):
        """Setup the video stream tab UI"""
        try:
            video_stream_layout = QtWidgets.QVBoxLayout(self.videoStreamTab)
            # Zero left margin, the outer verticalLayout owns the
            # sidebar-toggle gutter.
            video_stream_layout.setContentsMargins(0, 4, 4, 4)
            video_stream_layout.setSpacing(4)

            # Bare QWidget container -- no QGroupBox title bar / frame ridge
            # eating the perimeter; the tab bar already labels "Video Stream".
            self.videoStreamGroupBox = QtWidgets.QWidget()
            self.videoStreamGroupBox.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            video_stream_layout.addWidget(self.videoStreamGroupBox, stretch=1)

            self.videoStreamLayout = QtWidgets.QGridLayout(self.videoStreamGroupBox)
            self.videoStreamLayout.setContentsMargins(0, 0, 0, 0)
            self.videoStreamLayout.setSpacing(4)

            # Top controls row: detach + grid selector on one line
            controls_row = QtWidgets.QHBoxLayout()
            controls_row.setContentsMargins(0, 0, 0, 0)
            controls_row.setSpacing(8)
            controls_row.addStretch()

            # Camera Config and Test Tracking, duplicated from the Main
            # Control group. The same callbacks and the same look: this is a
            # second place to reach them, not a second implementation. They
            # sit INSIDE the tab, so detaching the tab carries them into the
            # detached window, which is the whole point of having them here.
            self.videoTabCameraButton = self._make_control_section_button(
                "Camera Config", "camera.svg",
                self.show_camera_config_dialog, "primary")
            self.videoTabTestTrackingButton = self._make_control_section_button(
                "Test Tracking", "video.svg",
                self.toggle_test_tracking, "info")
            for _b in (self.videoTabCameraButton,
                       self.videoTabTestTrackingButton):
                _b.setMinimumHeight(24)
                controls_row.addWidget(_b)
            # One owner for their state: the Main Control buttons decide, and
            # these follow. Two buttons that gate themselves independently
            # drift, and the operator is then told two different things about
            # the same rig.
            self._mirrored_buttons = [
                ("camera_connect_button", self.videoTabCameraButton),
                ("test_tracking_button", self.videoTabTestTrackingButton),
            ]
            # Gated off at birth. This tab can be built BEFORE the Main
            # Control group that owns the originals, so the twins start in the
            # safe state and _apply_ui_state brings them into line from
            # then on; it is the one place that decides every control's state.
            self.videoTabTestTrackingButton.setEnabled(False)
            self.videoTabCameraButton.setEnabled(False)

            self.detachVideoStreamButton = QtWidgets.QPushButton("Detach Video Stream")
            self.detachVideoStreamButton.clicked.connect(lambda: self.detach_tab("Video Stream"))
            self.detachVideoStreamButton.setMinimumHeight(24)
            self.detachVideoStreamButton.setStyleSheet(BUTTON_STYLE.format(
                color=COLORS['primary'],
                hover_color=COLORS['primary_hover']
            ))
            controls_row.addWidget(self.detachVideoStreamButton)

            self.videoGridDropdown = QtWidgets.QComboBox()
            # "Auto" computes cols = ceil(sqrt(N_boxes)); manual options
            # override it.
            self.videoGridDropdown.addItems(["Auto", "4x4", "3x3", "2x2"])
            self.videoGridDropdown.setCurrentText("Auto")
            self.videoGridDropdown.setStyleSheet(COMBOBOX_STYLE)
            self.videoGridDropdown.setFixedSize(100, 24)
            self.videoGridDropdown.currentIndexChanged.connect(self.update_video_streams)
            controls_row.addWidget(self.videoGridDropdown)

            video_stream_layout.addLayout(controls_row)

            logger.debug("Video stream tab setup complete")

        except Exception as e:
            logger.error(f"Error setting up video stream tab: {str(e)}")
            self.showError(f"Failed to setup video stream tab: {str(e)}")

    # toggleDebugMode / clearErrorLog / exportErrorLog inherited from
    # MainWindowUtilsMixin (_toggleDebugMode + _exportErrorLog). Clear is
    # wired directly to ErrorLogPanel's button.

    # connect_camera, disconnect_camera, _wait_for_camera_streaming inherited
    # from MainWindowBase; the hooks below shape the operant-specific UI.

    # Per-box widget access goes through the box-widget protocol
    # BoxControlWidget exposes (shared bodies in MainWindowBase). Only the
    # overrides below remain, each adds MainWindow-side state the widget
    # can't reach.

    def _box_camera_started_ui(self, setup_id, camera_id):
        super()._box_camera_started_ui(setup_id, camera_id)
        # Operant renders frames through a per-box VideoStreamHolder; create
        # it now if missing so the next frame has somewhere to render.
        try:
            if not self._get_cached_stream_widget(setup_id):
                self.update_video_streams()
        except Exception as e:
            logger.debug("update_video_streams after camera start: %s", e)

    def _stamp_box_timer(self, bw, txt):
        """Operant: fan the per-box elapsed clock out to the stats-table
        Timer column + the camera tile header (so it's visible from any tab).
        Same value the box card / Live Status already show."""
        setup_id = getattr(bw, "setup_number", None)
        if setup_id is None:
            return
        st = getattr(self, "statisticsTab", None)
        if st is not None and hasattr(st, "update_box_timer"):
            st.update_box_timer(setup_id, txt)
        holder = self._get_cached_stream_widget(setup_id)
        if holder is not None and hasattr(holder, "set_timer"):
            holder.set_timer(txt)

    def _refresh_box_id_labels(self, setup_id=None):
        """Mirror each box's Subject ID into its Live Status card + camera
        tile headers (shown only when assigned). Single reader of the box
        widget's ``subject_id_edit``; called on subject change and after a
        Live Status / Video Stream rebuild. ``setup_id`` limits the refresh
        to one box (used by the per-box change path)."""
        ls_by_box = getattr(self, "_live_status_by_box", None) or {}
        for bw in self.get_all_setup_widgets():
            sid = getattr(bw, "setup_number", None)
            if sid is None or (setup_id is not None and sid != setup_id):
                continue
            edit = getattr(bw, "subject_id_edit", None)
            subj = edit.text().strip() if edit is not None else ""
            ls = ls_by_box.get(sid)
            if ls is not None and hasattr(ls, "set_subject"):
                ls.set_subject(subj)
            holder = self._get_cached_stream_widget(sid)
            if holder is not None and hasattr(holder, "set_subject"):
                holder.set_subject(subj)

    def _camera_streaming_ready(self, setup_id):
        super()._camera_streaming_ready(setup_id)
        track_btn = getattr(self, "track_button", None)
        if track_btn is not None:
            try:
                track_btn.setEnabled(True)
                track_btn.setToolTip("Configure tracking")
            except Exception:
                pass

    def _camera_streaming_timeout(self, setup_id):
        super()._camera_streaming_timeout(setup_id)
        track_btn = getattr(self, "track_button", None)
        if track_btn is not None:
            try:
                track_btn.setEnabled(True)
                track_btn.setToolTip("Camera may still be initializing")
            except Exception:
                pass

    def _box_camera_disconnected_cleanup(self, setup_id):
        super()._box_camera_disconnected_cleanup(setup_id)
        # Operant-only main-window state: global track button gate,
        # recording teardown, fps cache.
        track_btn = getattr(self, "track_button", None)
        if track_btn is not None:
            try:
                has_cameras = any(
                    cam is not None
                    for cam in self.video_manager.box_camera_map.values())
                if not has_cameras:
                    track_btn.setEnabled(False)
                    track_btn.setToolTip("Connect camera first")
            except Exception:
                pass
        if setup_id in self.recording_setups:
            try:
                self.stop_recording(setup_id)
            except Exception:
                pass
        # _fps_status_times is base state and is cleared by super() now, so
        # maze no longer leaks it.

    def _tracking_roi(self, setup_id):
        """Operant ROI for the tracking-writer header, the per-box box ROI in
        full-camera pixels.

        Normalized-first: the canonical ``roi_normalized`` scaled to the live
        full-camera frame, so the recorded ROI metadata matches the runtime
        crop (which is percent-driven) rather than a stale pixel cache from a
        previous resolution. Falls back to the widget's pixel accessor
        when the normalized form or a frame is unavailable.
        """
        bw = self._setup_widget_for(setup_id)
        if bw is None:
            return None
        try:
            norm = getattr(bw, "roi_normalized", None)
            if norm and len(norm) == 4:
                cam_id = self.video_manager.box_camera_map.get(setup_id)
                full = (self.video_manager.get_full_frame(cam_id)
                        if cam_id is not None else None)
                if full is not None and full.shape[0] > 0 and full.shape[1] > 0:
                    fh, fw = full.shape[:2]
                    return [int(round(float(norm[0]) * fw)),
                            int(round(float(norm[1]) * fh)),
                            int(round(float(norm[2]) * fw)),
                            int(round(float(norm[3]) * fh))]
            roi = bw._get_box_roi()
            return list(roi[:4]) if roi and len(roi) >= 4 else None
        except Exception:
            return None

    # _video_recorder_geometry is shared on MainWindowBase (one resolver for
    # both modes; operant's box-ROI crop is the `_get_box_roi` branch).

    # Operant records on the same single path as maze: the widget's
    # on_record_clicked starts the framework; per-box bookkeeping lives in
    # BoxControlWidget._post_record_start.

    def stop_recording(self, setup_id):
        """Programmatic stop entry point, full synchronous teardown.

        Called from cleanup flows where the caller MUST observe the
        framework actually stopped before continuing (``remove_box``,
        ``_teardown_all_boxes``, ``_box_camera_disconnected_cleanup``).
        For the **Stop-button click path**, see the widget's
        ``on_stop_clicked`` (single handler): the multi-box Universal Stop
        dialog dispatches through ParallelStopCoordinator instead of running
        this synchronously on the GUI thread.
        """
        try:
            if setup_id not in self.recording_setups:
                return

            setup_widget = self._setup_widget_for(setup_id)
            if setup_widget is None:
                logger.warning(f"Box widget not found for box {setup_id}")
                return

            # Stop the pyControl framework first.
            setup_widget.stop_framework()

            # Stop live inference BEFORE tearing the recorder down so no
            # stale pose result lands on the next box's first frame.
            self.stop_tracking_for_box(setup_id)

            # Detach from pipeline + async-stop the recorder + clear the
            # widget attribute.
            self._stop_recording_for_box(setup_id)
            self.recording_setups.discard(setup_id)
            setup_widget.record_button.setEnabled(True)
            setup_widget.stop_button.setEnabled(False)
            # ``timer_label`` and ``status_edit`` are owned by RunTask: the
            # timer freezes at the last elapsed value, status reads "Stopped"
            # (or "Error: <msg>"). Do NOT overwrite.

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.update_live_status_for_box(setup_id, f"Stopped at {timestamp}", timestamp)
            logger.info(f"Stopped for box {setup_id}")

        except Exception as e:
            logger.error(f"Failed to stop for box {setup_id}: {str(e)}")
            self.showError(f"Failed to stop: {str(e)}")
        finally:
            try:
                self.refresh_ui_state()
            except Exception:
                pass

    # The Stop button is single-handled by the widget's on_stop_clicked; its
    # immediate UX lives in BoxControlWidget._on_stop_confirmed.

    def _post_record_stopped(self, setup_id, data):
        """Operant extras after the shared teardown: re-arm the per-box
        Record/Stop buttons + drop a live-status line.
        (``recording_setups`` is already discarded inside
        ``_stop_recording_for_box``.)"""
        setup_widget = self._setup_widget_for(setup_id)
        if setup_widget is None:
            return
        setup_widget.record_button.setEnabled(True)
        setup_widget.stop_button.setEnabled(False)
        # ``timer_label`` and ``status_edit`` are owned by RunTask: the
        # widget froze the timer and set the sticky status before this
        # slot ran. Do NOT overwrite.
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.update_live_status_for_box(setup_id, f"Stopped at {timestamp}", timestamp)
        logger.info(f"Box {setup_id}: Framework auto-stopped")

    # connect_all_cameras is inherited from MainWindowBase.

    def disconnect_all_boards(self):
        """Disconnect all MCU boards (pyControl boards)"""
        try:
            # Confirm with user since this is destructive
            if not self._confirm(
                "Disconnect All Boards",
                "Are you sure you want to disconnect all boards?"):
                logger.info("Disconnect all cancelled by user")
                return

            disconnected_count = 0
            for setup_widget in self.get_all_setup_widgets():
                if hasattr(setup_widget, 'pycboard') and setup_widget.pycboard:
                    # Stop framework if running
                    if hasattr(setup_widget, 'framework_running') and setup_widget.framework_running:
                        setup_widget.stop_framework()
                    # Close the board connection
                    setup_widget.pycboard.close()
                    setup_widget.pycboard = None
                    # Drop from the central MCU registry to keep the
                    # controller's view consistent with the widget.
                    try:
                        if hasattr(self, "mcu"):
                            self.mcu.unregister(setup_widget.setup_number)
                    except Exception:
                        pass
                    # Reset per-box state/flags
                    setup_widget.framework_running = False
                    setup_widget.task_uploaded = False
                    if hasattr(setup_widget, 'upload_button'):
                        setup_widget.upload_button.setText("Upload")
                    setup_widget.connect_button.setText("Connect")
                    setup_widget.connect_button.setStyleSheet(BUTTON_STYLE.format(
                        color=COLORS['success'],
                        hover_color=COLORS['success_hover']
                    ))
                    # Route through RunTask.set_status so styling matches the
                    # palette and any sticky "Error: ..." is overwritten.
                    setup_widget.set_status("Disconnected", "neutral")
                    try:
                        setup_widget._update_button_states()
                    except Exception:
                        pass
                    disconnected_count += 1
                    logger.info(f"Disconnected MCU board for Box {setup_widget.setup_number}")

            if disconnected_count > 0:
                logger.info(f"Disconnected {disconnected_count} MCU board(s)")
                self.statusbar.showMessage(f"Disconnected {disconnected_count} board(s)", 3000)
            else:
                logger.info("No boards were connected")
                self.statusbar.showMessage("No boards to disconnect", 3000)

            # Refresh control buttons to reflect no connected boards.
            self.refresh_ui_state()

        except Exception as e:
            logger.error(f"Failed to disconnect boards: {str(e)}")
            self.showError(f"Failed to disconnect boards: {str(e)}")

    # disconnect_all_cameras lives in MainWindowBase (shared by both modes).

    # Frame polling lives in source/video/controller.py (Pipeline._tick).
    # Display, ROI lookup, BGR888 paint, and zone overlay are inherited from
    # MainWindowBase (the QtBridge slot _on_box_frame calls update_box_display).

    # update_box_display + _numpy_to_pixmap inherited from MainWindowBase.

    def _get_cached_stream_widget(self, setup_id):
        """Widget lookup with cache. Returns VideoStreamHolder or None."""
        if setup_id in self.video_stream_widget_cache:
            widget = self.video_stream_widget_cache[setup_id]
            try:
                if widget and not widget.isHidden():
                    return widget
            except (RuntimeError, AttributeError):
                pass
            self.video_stream_widget_cache.pop(setup_id, None)

        for i in range(self.videoStreamLayout.count()):
            w = self.videoStreamLayout.itemAt(i).widget()
            if isinstance(w, VideoStreamHolder) and w.setup_number == setup_id:
                self.video_stream_widget_cache[setup_id] = w
                return w
        return None

    # _update_fps_status inherited from MainWindowBase, writes through the
    # box-widget protocol (BoxControlWidget.set_fps_text → VideoStreamHolder).

    # The sidebar Load button wires to the shared base.load_cohort_metadata
    # (load + project-adopt + assign offer).

    def clear_metadata(self):
        """Clear per-box Subject IDs only. The loaded cohort DataFrame
        is untouched, re-open the assign dialog to repopulate.
        """
        cleared = 0
        for bw in self._box_widgets.values():
            if bw.subject_id_edit.text().strip():
                bw.subject_id_edit.clear()
                cleared += 1
        self.update_metadata_button_states()
        logger.info("Cleared %d Subject ID(s) (cohort preserved)", cleared)
        self.show_temporary_message(f"Cleared {cleared} Subject ID(s)")

    # update_metadata_button_states comes from MainWindowUtilsMixin (shared,
    # hasattr-guarded, operant's Clear-Meta branch applies on top).

    def _idle_button_refresh(self) -> None:
        """1 Hz idle-tick backup for the metadata buttons. The PRIMARY
        driver is the start/stop event via ``_apply_ui_state``
        (``update_metadata_button_states``); this keeps a cheap, idempotent
        safety net for any path that doesn't refresh the UI state."""
        self.update_metadata_button_states()

    # iter_box_subject_widgets + iter_box_widgets inherited from
    # MainWindowBase (over _box_widgets).

    # _init_default_data_dir lives in MainWindowBase. Operant uses the
    # mixin default data path (project_root/data); maze overrides via
    # _data_dir_path() to point at app_paths.data_dir.

    # edit_all_metadata lives in MainWindowUtilsMixin (shared by operant + maze).

    def populate_com_ports(self, combo_box):
        """Populate a combo box with available Pyboard/USB serial COM ports"""
        try:
            combo_box.clear()
            combo_box.addItem("--- Select COM ---")

            if SERIAL_AVAILABLE:
                # Filter for Pyboard or USB Serial Device.
                ports = list_ports.comports()
                filtered_ports = []

                from source.gui.utility import com_sort_key as _com_key
                for port in sorted(ports, key=lambda p: _com_key(p.device)):
                    if "Pyboard" in port.description or "USB Serial Device" in port.description:
                        filtered_ports.append(port)
                        display_text = port.device
                        combo_box.addItem(display_text, port.device)
                        # Set tooltip with full description
                        index = combo_box.count() - 1
                        tooltip = f"{port.device}: {port.description}"
                        if port.manufacturer:
                            tooltip += f" ({port.manufacturer})"
                        combo_box.setItemData(index, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)

                if not filtered_ports:
                    combo_box.addItem("No Pyboard/USB serial devices")
                    logger.info("No Pyboard or USB Serial Device found")
                else:
                    logger.debug(f"Found {len(filtered_ports)} Pyboard/USB serial ports: {[p.device for p in filtered_ports]}")
            else:
                combo_box.addItem("No serial module")
                logger.warning("PySerial not available - COM ports cannot be listed")

        except Exception as e:
            logger.error(f"Error populating COM ports: {str(e)}")
            combo_box.addItem("Error loading ports")

    def update_live_status(self):
        """Update the Live Status grid incrementally.

        Reuses existing ``LiveStatusWidget`` cards, creates cards only
        for newly-added boxes and removes only the gone ones (the same
        pattern ``update_video_streams`` uses). A full teardown/rebuild
        on every add/remove was O(boxes²) widget churn and destroyed the
        cards' scrolling logs.
        """
        try:
            # box_number → LiveStatusWidget, the ONE container (per-box
            # lookup for logs/alerts + the elapsed-clock mirror in
            # MainWindowBase._on_process_tick).
            if not hasattr(self, "_live_status_by_box"):
                self._live_status_by_box = {}

            # Freeze paints on the container while the grid shuffles.
            self.liveStatusGroupBox.setUpdatesEnabled(False)
            try:
                widgets = list(self.get_all_setup_widgets())
                desired = {w.setup_number for w in widgets}

                # Drop cards for boxes that no longer exist.
                for bid in list(self._live_status_by_box.keys()):
                    if bid in desired:
                        continue
                    card = self._live_status_by_box.pop(bid)
                    try:
                        self.liveStatusLayout.removeWidget(card)
                    except Exception:
                        pass
                    card.setParent(None)
                    card.deleteLater()

                # Reuse-or-create + place each box at its grid slot.
                cols = 4
                for i, setup_widget in enumerate(widgets):
                    bid = setup_widget.setup_number
                    row, col = divmod(i, cols)
                    card = self._live_status_by_box.get(bid)
                    if card is None:
                        card = LiveStatusWidget(bid, setup_widget=setup_widget)
                        self._live_status_by_box[bid] = card
                    else:
                        # Detach so addWidget places it at the (possibly
                        # new) (row, col) after earlier removals.
                        try:
                            self.liveStatusLayout.removeWidget(card)
                        except Exception:
                            pass
                    self.liveStatusLayout.addWidget(card, row, col)
            finally:
                self.liveStatusGroupBox.setUpdatesEnabled(True)

            # Seed/refresh the cards' Subject ID labels.
            self._refresh_box_id_labels()
            logger.debug("Live status display updated")

        except Exception as e:
            logger.error(f"Error updating live status: {str(e)}")

    def update_video_streams(self):
        """Update the video stream display incrementally.

        Reuses existing ``VideoStreamHolder`` instances, only creates new
        ones for newly-added box_ids and removes ones for boxes that are
        gone. Reparents existing widgets to the new grid cell (layout-only,
        no GL-context teardown) to avoid native-window recreation on Windows.
        """
        try:
            grid_option = self.videoGridDropdown.currentText()
            if grid_option == "Auto":
                # Adaptive grid: cols = ceil(sqrt(N)).
                import math
                n = max(1, self.box_count)
                cols = max(1, int(math.ceil(math.sqrt(n))))
                rows = max(1, int(math.ceil(n / cols)))
            else:
                rows, cols = map(int, grid_option.split('x'))

            if not hasattr(self, "_video_holders"):
                self._video_holders = {}

            # Freeze paint events on the container while we shuffle the
            # grid, eliminates intermediate paint frames that show partial
            # layouts during the rearrange.
            self.videoStreamGroupBox.setUpdatesEnabled(False)
            try:
                desired_bids = set(range(1, self.box_count + 1))

                # Drop holders for boxes that no longer exist.
                for bid in list(self._video_holders.keys()):
                    if bid in desired_bids:
                        continue
                    holder = self._video_holders.pop(bid)
                    try:
                        self.videoStreamLayout.removeWidget(holder)
                    except Exception:
                        pass
                    holder.setParent(None)
                    holder.deleteLater()
                    self.video_stream_widget_cache.pop(bid, None)

                # Reuse-or-create + place each box at its new grid slot.
                for i in range(self.box_count):
                    bid = i + 1
                    row, col = divmod(i, cols)
                    holder = self._video_holders.get(bid)
                    if holder is None:
                        holder = VideoStreamHolder(bid)
                        self._wire_zone_buttons(holder, bid)
                        self._video_holders[bid] = holder
                    else:
                        # Detach from current grid cell so addWidget below
                        # places it at the (possibly new) (row, col).
                        try:
                            self.videoStreamLayout.removeWidget(holder)
                        except Exception:
                            pass
                    self.videoStreamLayout.addWidget(holder, row, col)
                    # Refresh tracking-zone-button enable state (frozen
                    # while the box records, same gate as the handlers).
                    _zones = (self.tracking_zones or {}).get(bid) or []
                    holder.set_zones_enabled(
                        bool(_zones) and not self._zone_edit_locked(bid))
                    # Keep the cache in sync so _get_cached_stream_widget is
                    # a hashmap hit instead of a layout walk.
                    self.video_stream_widget_cache[bid] = holder

                # Reset row/col stretch so collapsed cells stop holding
                # space, then equal-stretch the active rows + cols.
                for r in range(self.videoStreamLayout.rowCount()):
                    self.videoStreamLayout.setRowStretch(r, 0)
                for c in range(self.videoStreamLayout.columnCount()):
                    self.videoStreamLayout.setColumnStretch(c, 0)
                for r in range(rows):
                    self.videoStreamLayout.setRowStretch(r, 1)
                for c in range(cols):
                    self.videoStreamLayout.setColumnStretch(c, 1)
            finally:
                self.videoStreamGroupBox.setUpdatesEnabled(True)

            # Freshly (re)built tiles, seed their Subject ID labels.
            self._refresh_box_id_labels()
            logger.debug(
                "Video streams updated to %s (incremental, %d holders)",
                grid_option, len(self._video_holders))

        except Exception as e:
            logger.error(f"Error updating video streams: {e}")

    # ------------------------------------------------------------------
    # Centralised UI state management
    # ------------------------------------------------------------------
    # compute_ui_state + refresh_ui_state live in MainWindowBase.

    def _apply_mode_buttons(self, state, flags):
        """Operant-only buttons; the skeleton (locks, indicators, gates)
        lives in MainWindowBase._apply_ui_state."""
        has_boxes = flags["has_boxes"]
        any_running = flags["any_running"]
        block_run = flags["block_run"]

        if hasattr(self, "videoGridDropdown"):
            self.videoGridDropdown.setEnabled(has_boxes and not block_run)

        # Detach buttons stay enabled at all times (users detach views anytime).
        for btn_attr in ("detachLiveStatusButton", "detachStatisticsButton",
                         "detachVideoStreamButton"):
            btn = getattr(self, btn_attr, None)
            if btn:
                btn.setEnabled(True)

        if hasattr(self, "plot_button"):
            self.plot_button.setEnabled(any_running)
        # Multi-Start / Multi-Stop gating, disable the toolbar buttons
        # when there's nothing to act on. Same predicates the dialogs
        # use internally (UniversalStartDialog._eligible_widgets +
        # UniversalStopDialog._eligible_widgets).
        any_ready = bool(state.get("any_ready_to_start", False))
        if hasattr(self, "start_button"):
            self.start_button.setEnabled(any_ready)
            self.start_button.setToolTip(
                "Multi-Start: pick from boxes that are ready "
                "(connected + uploaded task + Start button armed); "
                "clicks each selected box's Start button"
                if any_ready
                else "No boxes ready to start "
                     "(need connected + uploaded task)")
        if hasattr(self, "stop_button"):
            self.stop_button.setEnabled(any_running)
            self.stop_button.setToolTip(
                "Click each selected box's Stop button, asks to confirm"
                if any_running
                else "No running sessions to stop")

    def _box_indicator_surfaces(self, setup_id, widget):
        """Operant: zone-adjust buttons + camera-pending flag live on the
        per-box video tile holder, not the box control widget."""
        return (getattr(self, "_video_holders", {}).get(setup_id),)

    def update_live_status_for_box(self, setup_id, status_text=None, timestamp=None):
        """Append a one-shot lifecycle line to the box's Live Status
        scrolling log (e.g. ``Framework started at HH:MM:SS``).

        The scrolling edit is owned by ``BoxControlWidget.print_to_log``
        which streams every state / event / print line from the board,
        we only ``appendStatus`` here, never ``setText``, so the running
        log isn't wiped. No FPS / frame / time header line: the user
        explicitly does not want that in the panel.
        """
        if not status_text:
            return
        try:
            if not timestamp:
                timestamp = datetime.now().strftime("%H:%M:%S")
            widget = self._live_status_by_box.get(setup_id)
            if widget is not None:
                widget.appendStatus(f"[{timestamp}] {status_text}")
        except Exception as e:
            logger.error(f"Error updating live status for Box {setup_id}: {str(e)}")

    def detach_tab(self, tab_name):
        """Detach a tab into a separate window"""
        try:
            if tab_name == "Live Status":
                tab_widget = self.liveStatusTab
                detach_button = self.detachLiveStatusButton
            elif tab_name == "Statistics":
                tab_widget = self.statisticsTab
                detach_button = self.detachStatisticsButton
            elif tab_name == "Video Stream":
                tab_widget = self.videoStreamTab
                detach_button = self.detachVideoStreamButton
            else:
                return

            if tab_widget.parent() is None:
                return

            self.tabWidget.removeTab(self.tabWidget.indexOf(tab_widget))

            detached_window = DetachableTabWindow(tab_widget, tab_name, self)
            detached_window.show()
            self.detached_tabs[tab_name] = detached_window

            detach_button.setEnabled(False)
            logger.info(f"Tab detached: {tab_name}")

        except Exception as e:
            logger.error(f"Error detaching tab {tab_name}: {str(e)}")

    def attach_tab(self, tab_name, tab_widget):
        """Attach a previously detached tab"""
        try:
            if tab_name == "Live Status":
                index = 1
                detach_button = self.detachLiveStatusButton
            elif tab_name == "Statistics":
                index = 2
                detach_button = self.detachStatisticsButton
            elif tab_name == "Video Stream":
                index = 3
                detach_button = self.detachVideoStreamButton
            else:
                return

            self.tabWidget.insertTab(index, tab_widget, tab_name)
            self.tabWidget.setCurrentIndex(index)

            detach_button.setEnabled(True)

            del self.detached_tabs[tab_name]
            logger.info(f"Tab reattached: {tab_name}")

        except Exception as e:
            logger.error(f"Error reattaching tab {tab_name}: {str(e)}")

    # resizeEvent / moveEvent + sidebar height-sync live in MainWindowBase
    # (auto-iterates self._sidebars registered via _register_sidebar).

    # =====================================================================
    # DLC (DeepLabCut) Integration Methods
    # =====================================================================

    # _show_tracking_config_impl moved to MainWindowBase. Operant hooks
    # the two seams below:
    #   * extra dialog kwargs (seed box for the zone-editor canvas)
    #   * post-dialog work (UI badge / button refresh)

    def _tracking_dialog_extra_kwargs(self, connected_with_cam):
        """Operant-only: seed the dialog's zone-editor canvas with the
        first connected box (also pre-warns on non-streaming cameras)."""
        seed_box = connected_with_cam[0]
        if not self.video_manager.is_camera_streaming(seed_box):
            logger.info(
                "Camera for box %s connected but not yet streaming frames",
                seed_box)
        return dict(setup_id=seed_box)

    def _post_tracking_dialog_hook(self, dialog, connected_with_cam):
        """Operant: chain to base (zone auto-save + log) THEN do
        operant-only extras: refresh the master tracking-toggle button
        and the status badges (BGs may have just changed)."""
        super()._post_tracking_dialog_hook(dialog, connected_with_cam)
        try:
            self._update_test_tracking_button()
        except Exception:
            pass
        try:
            self._refresh_status_badges()
        except Exception:
            pass

    def _sync_mirrored_buttons(self) -> None:
        """Copy each Main Control button's state onto its Video Stream twin.

        Enabled state, label, tooltip, icon and style, because Test Tracking
        changes all of them when a preview starts ("Stop Test", red) and a
        twin that kept saying "Test Tracking" would be a second, wrong answer
        to the same question.
        """
        for attr, clone in getattr(self, "_mirrored_buttons", ()):
            primary = getattr(self, attr, None)
            if primary is None or clone is None:
                continue
            try:
                clone.setEnabled(primary.isEnabled())
                clone.setText(primary.text())
                clone.setToolTip(primary.toolTip())
                clone.setStyleSheet(primary.styleSheet())
                clone.setIcon(primary.icon())
            except RuntimeError:
                continue          # the tab was closed out from under us

    def _update_test_tracking_button(self):
        """Enable/disable tracking toggle button based on configuration."""
        try:
            if not hasattr(self, 'test_tracking_button'):
                return

            is_configured = self._any_box_tracking_configured()
            # No init gate. A configured box prepares its model as soon as its
            # camera streams, and Test Tracking prepares any that are not ready
            # through the same path before it starts. Disabling the button
            # until the operator visits a dialog was a gate on a flag, not on
            # whether anything could actually run.

            self.test_tracking_button.setEnabled(is_configured)
            if is_configured:
                self.test_tracking_button.setToolTip(
                    "Test Tracking, quick preview of pose / blob tracking "
                    "without recording. Use to verify your setup before "
                    "clicking Record. Live tracking during a real run is "
                    "started automatically when Record starts the framework."
                )
            else:
                self.test_tracking_button.setToolTip("Configure tracking first (use Track Config)")
            self._sync_mirrored_buttons()

        except Exception as e:
            logger.error(f"Error updating tracking toggle button: {e}")

    # Test Tracking is implemented in MainWindowBase
    # (``_toggle_test_tracking_impl`` + ``_start_test_tracking`` +
    # ``_stop_test_tracking``). Operant only overrides two hooks:
    #   * ``_prepare_test_tracking_for_box``, blob bringup (BG load +
    #     tracker auto-init) so ``pipe.enable_blob_tracking`` finds a
    #     ready tracker. The DLC / SLEAP path is fully handled by the
    #     unified ``start_tracking_for_box(force=True)``.
    #   * ``_refresh_test_tracking_button``, flip the toolbar
    #     toggle's label / icon / colour between idle and active.

    def _prepare_test_tracking_for_box(self, setup_id):
        """Operant override: load blob background + auto-init tracker
        before the unified ``start_tracking_for_box(force=True)`` runs.
        Delegates to the shared base helper, no operant-specific
        bringup left."""
        settings = self._tracking_config_to_settings_dict(setup_id)
        self._setup_blob_tracker_for_box(
            setup_id, settings, attach_enhancer=False)

    def _refresh_test_tracking_button(self, *, active: bool) -> None:
        """Operant override: paint the master ``test_tracking_button``
        based on the current preview state. The Track-Config lock is the
        base's, both modes need it."""
        super()._refresh_test_tracking_button(active=active)
        btn = getattr(self, "test_tracking_button", None)
        if btn is None:
            return
        icon_dir = Path(__file__).parent / "icons"
        if active:
            btn.setText("Stop Test")
            btn.setStyleSheet(BUTTON_STYLE.format(
                color=COLORS['danger'], hover_color=COLORS['danger_hover']))
            stop_icon = icon_dir / "stop.svg"
            if stop_icon.exists():
                btn.setIcon(QtGui.QIcon(str(stop_icon)))
        else:
            btn.setText("Test Tracking")
            btn.setStyleSheet(BUTTON_STYLE.format(
                color=COLORS['warning'], hover_color=COLORS['warning_hover']))
            play_icon = icon_dir / "play.svg"
            if play_icon.exists():
                btn.setIcon(QtGui.QIcon(str(play_icon)))
        self._sync_mirrored_buttons()

    # _handle_dlc_init_from_dialog, _configure_pose_for_box,
    # _enable_pose_for_box, _disable_pose_for_box, _build_annotate_callback
    # all live in MainWindowBase. Operant overrides below add the
    # mode-specific extras (yml→dir model path).

    def _pose_resolve_model_path(self, cfg):
        """Operant: cfg path > cached global path > fallback resolver.
        Also handles .yml → parent-directory resolution that DLCLive needs.

        The config being applied wins over the session-cached global path:
        re-picking a different model in the dialog must load THAT model, not
        keep resurrecting the one cached at first init until a restart (T5).
        The cache is only a fallback for boxes that carry no explicit path.
        """
        mp = (cfg.get("model_path")
              or cfg.get("dlc_model_path")
              or self.dlc_global_model_path
              or self._resolve_dlc_model_path())
        if not mp:
            return None
        try:
            obj = Path(mp)
            if not obj.exists():
                return None
            # DLCLive wants the EXPORTED model DIR; if user picked a yml,
            # use its parent directory.
            if obj.is_file() and obj.suffix.lower() in (".yml", ".yaml"):
                return str(obj.parent)
            return str(obj)
        except Exception:
            return None

    def _pose_after_configured(self, setup_id, cfg, handle):
        """Operant extra: remember the resolved model folder.

        The fallback for a box whose own config carries no path, which is how
        a rig configured through the dialog before any box was saved still
        finds its model."""
        try:
            mp = self._pose_resolve_model_path(cfg)
            if mp:
                self.dlc_global_model_path = mp
        except Exception:
            pass

    def _pose_after_dialog_init(self, cfg, dialog, success_count):
        """Operant: refresh the Tracking-toggle button after init."""
        if success_count > 0:
            try:
                self._update_test_tracking_button()
            except Exception:
                pass

    # _disable_pose_for_box also covers operant, pose_configs.pop +
    # pipeline.disable_pose are already in the base method. The
    # pose_zone_state cleanup happens in this disabled hook.

    def _resolve_dlc_model_path(self):
        """Resolve DLCLive model path from environment or config."""
        try:
            import os
            env_path = os.environ.get("DLC_MODEL_PATH")
            if env_path and Path(env_path).exists():
                return env_path
            logger.warning("DLC_MODEL_PATH not set or missing; DLC-live disabled until provided.")
            return None
        except Exception:
            return None

    # _load_zones inherited from MainWindowBase.

    # Real-time zone triggers flow through pipeline.push.configure_zones →
    # MCUPusher (edge-detected at the sink), not through main_window.

    # closeEvent lives in MainWindowBase; it stops every recording_boxes
    # entry, calls pipeline.shutdown, closes detached_tabs, and runs
    # _close_event_extras for any subclass extras (operant has none).
