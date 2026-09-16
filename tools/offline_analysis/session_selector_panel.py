"""Top selector, three buttons for the three input sources.

User mental model (after several rounds of feedback):
    * Project is *just* a fast filtered file index, open a project to
      pick from hundreds of sessions by subject / date / task / box.
    * Folder is an ad-hoc browse, point at a directory, scan files.
    * Files is the explicit pick, choose individual files.

This widget is the only place to set the input. Tabs (MCU / V+T) auto-
consume the committed selection via ``sessionsPushed``. No per-tab Source
radio, no separate Cohort or "Push to active tab" buttons; those were
removed because they duplicated this one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PySide6 import QtCore, QtWidgets

from .session_catalog import (
    INTEGRITY_UNLISTED, ProjectContext, Session, load_project_context,
)
from .session_picker_dialog import SessionPickerDialog
from .theming import style_button, style_chip

logger = logging.getLogger(__name__)


class SessionSelectorPanel(QtWidgets.QFrame):
    """Thin top bar: ``[Folder] [Files] [Project]`` + selection chip + Clear."""

    projectLoaded    = QtCore.Signal(object)   # ProjectContext
    sessionsPushed   = QtCore.Signal(list)     # list[Session] committed
    sessionsSelected = QtCore.Signal(list)     # alias (live)
    videosPicked     = QtCore.Signal(list)     # list[str] of video paths

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setMaximumHeight(40)
        self._ctx: Optional[ProjectContext] = None
        self._selection: List[Session] = []
        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(10, 4, 10, 4)
        row.setSpacing(8)

        # ── The source buttons ───────────────────────────────────────
        #
        # Every way of getting recordings in is here, and only here: the
        # Analyze view used to carry a "+ Load" menu offering Folder, Files
        # and Project all over again, one row below these.
        #
        # Ordered by how much you are opening at once, one file, a folder of
        # them, a whole project, with videos last because that is the way in
        # for a recording that has not been tracked at all.
        #
        # The tooltips are long on purpose. Which of these four to press is
        # the first decision anyone makes here, the difference between them is
        # not obvious from four words, and getting it wrong means a dialog
        # that shows no files and no explanation.
        self.files_btn = QtWidgets.QPushButton("📄 Files")
        self.files_btn.setToolTip(
            "<b>Pick individual recordings.</b><br>"
            "Choose one or more <code>*_video_data.txt</code> files by hand. "
            "No filtering and no scanning, you get exactly what you "
            "select.<br><br>"
            "Each file's video is looked for beside it, by name, so a "
            "recording arrives complete without picking the video too.<br><br>"
            "<i>Use this when you know the handful of sessions you want.</i>")
        style_button(self.files_btn, "info")
        self.files_btn.clicked.connect(self._on_open_files)
        row.addWidget(self.files_btn)

        self.folder_btn = QtWidgets.QPushButton("📁 Folder")
        self.folder_btn.setToolTip(
            "<b>Scan a folder for recordings.</b><br>"
            "Searches the folder and everything under it for "
            "<code>*_video_data.txt</code>, then offers the list to filter "
            "and multi-select before anything loads.<br><br>"
            "Videos are matched to their data files automatically; a video "
            "with no data file is left for the Videos button.<br><br>"
            "<i>Use this for a day's recordings, or a box's, when they share "
            "a folder.</i>")
        style_button(self.folder_btn, "info")
        self.folder_btn.clicked.connect(self._on_open_folder)
        row.addWidget(self.folder_btn)

        self.project_btn = QtWidgets.QPushButton("📋 Project")
        self.project_btn.setToolTip(
            "<b>Open a rig project.</b><br>"
            "Reads the project's <code>experiment_config.json</code> and "
            "lists every session it knows about, filterable by subject, "
            "date, task and box.<br><br>"
            "The project also carries what the rig recorded ALONGSIDE the "
            "video, so the sessions arrive with their task and subject "
            "already named, rather than guessed from filenames.<br><br>"
            "<i>Use this for a cohort or a study, across hundreds of "
            "sessions.</i>")
        style_button(self.project_btn, "primary")
        self.project_btn.clicked.connect(self._on_open_project)
        row.addWidget(self.project_btn)

        self.videos_btn = QtWidgets.QPushButton("🎬 Videos")
        self.videos_btn.setToolTip(
            "<b>Open videos that were never tracked.</b><br>"
            "For a session recorded with tracking off, or a video with no "
            "data file at all. Nothing is written beside your data: the "
            "video becomes a row whose Plan column says what it still "
            "needs.<br><br>"
            "If a matching <code>*_video_data.txt</code> sits beside it, that "
            "is picked up too, so an untracked session keeps its timestamps "
            "and zones.<br><br>"
            "From there, <b>Set up tracker</b> produces the poses, the blob "
            "tracker needs no model and no GPU.<br><br>"
            "<i>Use this when there is footage but no tracking yet.</i>")
        style_button(self.videos_btn, "info")
        self.videos_btn.clicked.connect(self._on_open_videos)
        row.addWidget(self.videos_btn)

        # ── Status chip ──────────────────────────────────────────────
        row.addSpacing(12)
        self.sel_chip = QtWidgets.QLabel("no sessions selected")
        style_chip(self.sel_chip, "neutral")
        row.addWidget(self.sel_chip)

        # ── Project name pill (only visible when a project is loaded) ─
        self.proj_pill = QtWidgets.QLabel("")
        style_chip(self.proj_pill, "info")
        self.proj_pill.setVisible(False)
        row.addWidget(self.proj_pill)

        row.addStretch(1)

        # ── Clear ───────────────────────────────────────────────────
        self.clear_btn = QtWidgets.QPushButton("Clear")
        style_button(self.clear_btn, "secondary")
        self.clear_btn.clicked.connect(self._on_clear)
        row.addWidget(self.clear_btn)

    # ------------------------------------------------------------------ Slots

    def _on_open_folder(self):
        """Scan a folder for session-like files and open the picker."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Pick a folder of sessions", self._default_start_dir())
        if not folder:
            return
        sessions = self._build_sessions_from_folder(Path(folder))
        if not sessions:
            QtWidgets.QMessageBox.information(
                self, "Folder",
                "No session files found in that folder.\n"
                "(Looking for *_video_data.txt or *-Box*-*.tsv files.)")
            return
        self._open_picker(sessions, title=f"Sessions in {Path(folder).name}")

    def _on_open_files(self):
        """Direct file pick, no filter, push immediately."""
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Pick session files", self._default_start_dir(),
            "Sessions (*_video_data.txt *_video_data_*.txt *.tsv);;All (*)")
        if not files:
            return
        sessions = []
        for fp in files:
            p = Path(fp)
            s = Session(project_dir=p.parent, date="", subject=p.stem,
                        integrity=INTEGRITY_UNLISTED)
            # Naive classification by extension
            if p.suffix.lower() == ".tsv":
                s.mcu_path = p
            else:
                s.video_path = p
            sessions.append(s)
        self._selection = sessions
        self._update_chip()
        self.sessionsPushed.emit(sessions)

    def _on_open_videos(self):
        """Bare videos, straight to whatever is mounted below."""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Pick video(s) with no tracking yet", self._default_start_dir(),
            "Video files (*.mp4 *.avi *.mov *.mkv *.m4v);;All files (*)")
        if paths:
            self.videosPicked.emit([str(p) for p in paths])

    def _on_open_project(self):
        """Load a project, then open the filter picker."""
        start = self._default_start_dir(prefer_projects=True)
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open project (experiment_config.json)",
            start, "Project Config (experiment_config.json);;All (*)")
        if not fp:
            return
        try:
            ctx = load_project_context(fp)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Open Project", f"Failed to load: {e}")
            return
        self.set_context(ctx)
        if not ctx.sessions:
            QtWidgets.QMessageBox.information(
                self, "Project",
                f"Loaded '{ctx.project_name}' but no sessions found yet.")
            return
        self._open_picker(ctx.sessions,
                          title=f"Sessions in {ctx.project_name}",
                          cohort_df=ctx.cohort_df)

    def _on_clear(self):
        self._ctx = None
        self._selection = []
        self.proj_pill.setVisible(False)
        self.proj_pill.setText("")
        self._update_chip()
        # Push empty list so downstream tabs clear their state.
        self.sessionsPushed.emit([])

    # ------------------------------------------------------------------ Helpers

    def _open_picker(self, sessions: List[Session], *,
                     title: str = "Select sessions",
                     cohort_df=None):
        dlg = SessionPickerDialog(
            self, sessions,
            cohort_df=cohort_df,
            preselected=self._selection,
            title=title,
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        sel = dlg.selected_sessions()
        if not sel:
            return
        self._selection = sel
        self._update_chip()
        self.sessionsPushed.emit(sel)

    def _build_sessions_from_folder(self, folder: Path) -> List[Session]:
        """Walk a folder and build a Session list from any files that
        look like sessions (MCU TSVs or _video_data.txt)."""
        sessions: dict = {}
        for p in folder.rglob("*"):
            if not p.is_file():
                continue
            name = p.name.lower()
            if "_video_data" in name and name.endswith(".txt"):
                # subject is the part before ``_video_data`` (handles both
                # ``<subj>_video_data.txt`` and v0 ``<subj>_video_data_-<date>.txt``).
                subj = p.stem.split("_video_data")[0]
                key = (subj, "")
                s = sessions.get(key) or Session(
                    project_dir=folder, date="", subject=subj,
                    integrity=INTEGRITY_UNLISTED)
                s.video_path = p
                sessions[key] = s
            elif name.endswith(".tsv") and "-box" in name.lower():
                # naming: <subject>-Box<n>-YYYY-MM-DD-HHMMSS.tsv
                import re
                m = re.match(
                    r"^(?P<sub>.+?)-Box(?P<box>\d+)-"
                    r"(?P<date>\d{4}-\d{2}-\d{2})-\d{6}\.tsv$", p.name,
                    re.IGNORECASE)
                if m:
                    key = (m.group("sub"), m.group("date"))
                    s = sessions.get(key) or Session(
                        project_dir=folder, date=m.group("date"),
                        subject=m.group("sub"), box=int(m.group("box")),
                        integrity=INTEGRITY_UNLISTED)
                    s.mcu_path = p
                    sessions[key] = s
        return list(sessions.values())

    def _update_chip(self):
        n = len(self._selection)
        if n == 0:
            self.sel_chip.setText("no sessions selected")
            return
        subs = len({s.subject for s in self._selection if s.subject})
        dates = len({s.date for s in self._selection if s.date})
        self.sel_chip.setText(f"{n} session(s) · {subs} subj · {dates} date(s)")

    def _default_start_dir(self, *, prefer_projects: bool = False) -> str:
        try:
            here = Path(__file__).resolve().parents[2]
        except Exception:
            return ""
        if prefer_projects:
            cand = here / "experiments" / "projects"
            if cand.exists():
                return str(cand)
        cand = here / "data"
        if cand.exists():
            return str(cand)
        return str(here)

    # ------------------------------------------------------------------ Public API

    def set_context(self, ctx: ProjectContext):
        self._ctx = ctx
        self.proj_pill.setVisible(True)
        self.proj_pill.setText(f"📋 {ctx.project_name}")
        self.projectLoaded.emit(ctx)

    def context(self) -> Optional[ProjectContext]:
        return self._ctx

    def checked_sessions(self) -> List[Session]:
        return list(self._selection)
