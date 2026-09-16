"""Camera-connect dialog. Single dialog class per file; importable directly
or via the gui.dialogs package.
"""
import contextlib
import os
import threading
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from source.gui.theme import THEME as _T_cc
from source.log import get_logger

from .controls import ConfigSelectionDialog
from .video_roi import ROISegmentationDialog

logger = get_logger()


class _CamItem:
    """Per-camera work item for the calibration batch runner.

    ``setup_number`` is what ``run_box_actions_in_parallel`` keys on (it maps
    each item to its result via that attribute), so it carries the camera_id.
    ``result`` is filled by the worker body with a :class:`ProbeResult`.
    """

    def __init__(self, spec: dict):
        self.setup_number = spec["camera_id"]
        self.camera_id = spec["camera_id"]
        self.backend = spec.get("backend", "opencv")
        self.target_fps = spec.get("target_fps", 30.0)
        self.unique_id = spec.get("unique_id", "")
        self.identity = spec.get("identity") or {}
        self.weak = bool(spec.get("weak", False))
        self.label = spec.get("label", "")
        # The format the OPERATOR picked. Detect measures for this format and
        # no other: the rate a size reaches depends on it, so a measurement
        # taken in a different one describes a camera they are not running.
        self.pixel_format = spec.get("pixel_format", "mjpeg")
        self.door = spec.get("door", "")
        self.result = None


def _format_probe_report(kind: str, data: dict) -> str:
    """Turn a probe ``report(kind, **data)`` event into one log line."""
    if kind == "resolutions":
        return f"Found {len(data.get('modes') or [])} resolution(s)"
    if kind == "measuring":
        return (f"Measuring {data.get('w')}x{data.get('h')} "
                f"({data.get('index')}/{data.get('total')})...")
    if kind == "mode_done":
        try:
            fps = float(data.get("fps", 0.0) or 0.0)
        except (TypeError, ValueError):
            fps = 0.0
        fmt = str(data.get("fourcc") or "").strip()
        # The format is quoted with the rate because the same mode delivers
        # roughly twice as many frames compressed as it does raw.
        note = "" if fmt in ("", "?", "MJPG") else f"  [{fmt} - uncompressed, "
        note += "" if not note else "USB-capped]"
        return f"  {data.get('w')}x{data.get('h')} -> {fps:.1f} fps{note}"
    if kind == "resolution_found":
        return f"  confirmed {data.get('w')}x{data.get('h')}"
    if kind == "fps_measured":
        return ""  # summarised by mode_done; skip the duplicate line
    return kind


class _CameraIdCombo(QtWidgets.QComboBox):
    """Editable Camera-ID picker: type any id, or open the dropdown to pick
    from cameras detected on this machine. Detection runs lazily on first
    popup (opening a device is slow) so the dialog itself opens instantly."""

    def __init__(self, populate_cb, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self._populate_cb = populate_cb
        self._populated = False
        # Dropdown items show a friendly "id, model" label but carry the bare
        # id in itemData. ``activated`` fires AFTER Qt copies the item's label
        # into the line edit, so writing the bare id back here wins, the edit
        # text (and thus the parsed camera id) stays clean.
        self.activated.connect(self._pick_bare_id)

    def _pick_bare_id(self, index):
        data = self.itemData(index)
        if data:
            self.setEditText(str(data))

    def setEditText(self, text):
        """Set the text and scroll it back to the START.

        A line edit leaves the cursor at the end, and when the id is a hair
        wider than the field that scrolls the first character out of view:
        ``cam3cbb52d3`` rendered as ``:am3cbb52d3``. The leading characters
        are the ones that say which camera this is, so losing them is worse
        than losing the tail, and it looked like a corrupted id rather than a
        scrolled one.
        """
        super().setEditText(text)
        le = self.lineEdit()
        if le is not None:
            le.setCursorPosition(0)

    def showPopup(self):
        if not self._populated:
            self._populated = True
            try:
                self._populate_cb(self)
            except Exception:
                pass
        super().showPopup()


class _ToggleSwitch(QtWidgets.QCheckBox):
    """A pill on/off switch, hand-painted. It IS a QCheckBox, so isChecked /
    setChecked / the toggled signal all work, a drop-in for a checkbox."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(46, 24)

    def sizeHint(self):
        return QtCore.QSize(46, 24)

    def hitButton(self, pos):
        return self.rect().contains(pos)

    def paintEvent(self, _e):
        from PySide6 import QtGui
        on = self.isChecked()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        r = QtCore.QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor("#5fcf8a") if on else QtGui.QColor("#3a4250"))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        d = r.height() - 4
        x = (r.right() - d - 2) if on else (r.left() + 2)
        p.setBrush(QtGui.QColor("#ffffff"))
        p.drawEllipse(QtCore.QRectF(x, r.top() + 2, d, d))
        p.end()


#: Backend names as an operator would recognise them. An unknown name falls
#: through unchanged, a backend nobody has named is still one the camera
#: measured, and hiding it would make the row unexplainable.
_BACKEND_LABELS = {
    "dshow": "DirectShow",
    "msmf": "Media Foundation",
    "v4l2": "V4L2",
    "any": "GStreamer/FFmpeg",
    "avfoundation": "AVFoundation",
}

#: The same doors, short enough for a table cell. Only shown when a camera
#: has more than one, so the ordinary single-door case stays uncluttered.
_BACKEND_SHORT = {
    "dshow": "DS",
    "msmf": "MF",
    "v4l2": "V4L2",
    "any": "GST",
    "avfoundation": "AVF",
}

#: What a mode nobody measured is allowed to ask for. Such a mode exists so a
#: backend whose probe FAILED can still be forced, which is the one time an
#: operator needs to choose a door by hand. It has no measured ceiling, so the
#: ladder is capped at a rate no UVC webcam exceeds and opens on the
#: behavioural target rather than the top of the list.
#: Separates the rate a mode is SET to from the rate it DELIVERS in a
#: resolution cell, e.g. "640x480 (120->100)".
ARROW = "→"

_UNMEASURED_FPS_CEILING = 60.0
_UNMEASURED_FPS_DEFAULT = 30


class CameraConnectDialog(QtWidgets.QDialog):
    """Dialog for connecting cameras to boxes with camera ID input."""

    # Raised on the worker that walks the USB bus; delivered on the GUI thread
    # so the pickers fill themselves the moment the list exists, rather than
    # waiting for someone to open a dropdown.
    camera_ids_ready = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setWindowTitle("Connect Camera")
        # Resizable (was a fixed 520 that cramped the table); wide enough for
        # the identity table + Capture Settings card without a scrollbar.
        self.setMinimumSize(760, 560)
        self.resize(870, 660)
        self.setModal(True)
        self.camera_ids = {}  # {box_number: camera_id}
        self.connected_count = 0

        # Detect available camera backends
        self._available_backends = ["OpenCV"]
        try:
            from source.video.cameras import CameraFactory
            backends = CameraFactory.get_available_backends()
            if "spinnaker" in backends:
                self._available_backends.append("Spinnaker (FLIR)")
            if "ximea" in backends:
                self._available_backends.append("Ximea")
        except Exception:
            pass

        self._build_ui()
        self.applyDarkTheme()
        # The active camera's Resolution/FPS/Format are restored from its saved
        # CameraConfig by populateBoxes()->_load_active_camera_settings(); no
        # separate per-session resolution replay is needed.

    #: Backend names as an operator would recognise them. Unknown names fall
    #: through unchanged rather than being hidden, a backend nobody has named
    #: is still a backend the camera measured.
    def _populateResolutionCombo(self, modes, preselect_wh=None,
                                 res_combo=None, fps_combo=None,
                                 compact=False, preselect_backend=None,
                                 cam_id=None):
        """Fill ``res_combo`` from a list of ``(w, h, realistic_fps)`` tuples.

        Each entry's display is ``"W x H @ N.N fps"`` and itemData is a
        ``{"width", "height", "fps"}`` dict so downstream code can read both
        the resolution and the measured ceiling without re-parsing strings.
        ``res_combo`` / ``fps_combo`` default to the shared handles (the
        currently-targeted table row).

        ``compact`` is the same list sized for a table cell: the label is just
        ``"W×H"`` and only the fastest door to each size is listed. The per-row
        columns began as a SECOND implementation of this and of the rate ladder
        below it, and every way the two drifted became a bug the operator hit,
        a rate list built for the wrong resolution, a saved size that never
        re-selected, MJPEG defaulting differently. One populator, called twice.
        """
        res_combo = res_combo if res_combo is not None else self.resolution_combo
        if res_combo is None:
            return
        # RESTORED, not forced off. ``blockSignals`` is a flag, not a
        # counter, so unblocking here re-armed a combo a caller had blocked
        # for its own critical section: ``_fill_row_capture`` blocked both
        # combos, called this, and its next ``setCurrentIndex`` then fired
        # ``_on_row_capture_changed`` re-entrantly, which wrote the config
        # from a selection this method had just replaced. That is how
        # choosing the DirectShow door stored Media Foundation.
        was_blocked = res_combo.signalsBlocked()
        res_combo.blockSignals(True)
        try:
            res_combo.clear()
            if not modes:
                res_combo.addItem("(no modes detected)", None)
                return
            # One row per measured mode PER BACKEND, best first.
            #
            # The mode and the backend that serves it are a single choice, so
            # they are a single list: this camera reaches 1920x1080 at 30 fps
            # through Media Foundation and 5 through DirectShow, and offering
            # only the merged best hid which door delivered it while a separate
            # backend picker asked the operator to make the same decision
            # twice. Sorted by pixels then rate, so the first row is the
            # largest picture at the highest rate anything achieved, the
            # default, unless a saved pick says otherwise.
            entries = []
            for mode in modes:
                try:
                    w, h, fps = int(mode[0]), int(mode[1]), float(mode[2])
                except (TypeError, ValueError, IndexError):
                    continue
                backend = str(mode[3]) if len(mode) > 3 and mode[3] else ""
                entries.append((w, h, fps, backend))
            entries.sort(key=lambda e: (-(e[0] * e[1]), -e[2]))

            # The DEFAULT is not simply the first row. Frame rate gates and
            # resolution decides, ``pick_default_mode``'s rule, and it is
            # right: taking the head of a largest-first list lands a webcam on
            # 1920x1080 at 5 fps, which for lossless recorder and tracker
            # sinks does not degrade gracefully, it fills _drops.tsv from the
            # first second. Among the rows for the chosen size, the fastest
            # wins, which is where the backend gets decided.
            from source.video.framebus.types import pick_default_mode
            chosen = pick_default_mode([(w, h, f) for w, h, f, _b in entries])
            default_wh = (chosen[0], chosen[1]) if chosen else None

            if compact:
                # One entry per (size, DOOR), for every door that was
                # MEASURED. Both are real capability and they differ: on this
                # rig 1920x1080 is 18.3 fps through Media Foundation and 1.0
                # through DirectShow, and which one the camera opens with is
                # the operator's to choose.
                #
                # This was briefly collapsed to one row per size, and that was
                # a fix aimed at the wrong thing: the list had doubled because
                # every size carried a SPECULATIVE "(try)" row for a door that
                # had never been probed. Those are the noise, and they are now
                # one row per door rather than one per size. A measured pair is
                # not noise.
                seen_mode, kept = set(), []
                for e in entries:
                    key = (e[0], e[1], e[3])
                    if key not in seen_mode:
                        seen_mode.add(key)
                        kept.append(e)
                entries = kept
            # Named only when there is a choice to make, and an unmeasured
            # door is not one: it already says "try" and it is one row. Naming
            # every measured row because a single untried row exists put "DS"
            # on all nine sizes of a camera that has only ever had one door.
            multi_backend = len({e[3] for e in entries if e[3] and e[2] > 0}) > 1

            saved_idx = default_idx = saved_fallback_idx = None
            # The ceiling printed beside a size has to come from the SAME
            # place the rate ladder beside it comes from, or the row states
            # two different numbers. Once the ladder moved to the enumeration
            # and the label stayed on the measurement, a row read
            # "640x480 DS (24.6)" while the ladder next to it offered 20 and
            # 30: 24.6 is what the camera was once seen delivering in this
            # room, it is not a rate the camera has, and it appeared nowhere
            # else in the application. Computed once per size, not once per
            # row, because the store is read to answer it.
            # THE MEASUREMENT, which is what Detect put in this list.
            #
            # The label preferred the enumeration and the list beside it came
            # from the measurement, so the two disagreed by construction: a
            # row read "640x480 (120)" for a mode Detect had measured at 99.8.
            # The enumeration is what the camera CLAIMS; the measurement is
            # what it DELIVERS, and the cell that names a size has to say what
            # choosing that size will actually get you. The claim is still
            # what the FPS cell offers, because those are the only intervals
            # the driver can be set to.
            #
            # This is only honest now that a mode is measured at its OWN
            # advertised rate rather than at a fixed 30 (see
            # probe._rate_targets); before that, the measurement was "what 30
            # delivered here" and meant nothing as a ceiling.
            offered_ceiling: dict = {}

            def _offered(w, h):
                key = (w, h)
                if key not in offered_ceiling:
                    rates = self._offered_rates_for({"width": w, "height": h},
                                                    cam_id=cam_id)
                    offered_ceiling[key] = max(rates) if rates else 0.0
                return offered_ceiling[key]

            def _ceiling(w, h, measured):
                """What this size is worth, as ``(offered, delivered)``.

                BOTH, because they are different facts and the operator needs
                each. The offered value is the interval the driver accepts and
                is what the FPS cell lists; the delivered value is what Detect
                measured coming out, and is what the recorder writes.

                Showing only the claim printed "640x480 (120)" for a mode
                Detect had measured at 100. Showing only the measurement put a
                number in this cell that appeared in no other, which is the
                "(24.6) beside a ladder of 20 and 30" defect. So the cell says
                "120->100" when they differ and "30" when they agree, and the
                ladder's top value is always in it.
                """
                claim = _offered(w, h)
                got = float(measured or 0.0)
                if claim <= 0:
                    return (got, 0.0)          # never enumerated
                if got <= 0:
                    return (claim, 0.0)        # never measured
                if abs(got - claim) <= max(1.0, claim * 0.05):
                    return (claim, 0.0)        # they agree, one number
                return (claim, got)

            for w, h, fps, backend in entries:
                pretty = _BACKEND_LABELS.get(backend, backend)
                if compact:
                    # The ceiling is what decides whether a size is usable at
                    # all, so it belongs beside the size, as it is on the
                    # shared card. Trailing ".0" dropped: "30" reads as a rate
                    # where "30.0" reads as a measurement in a cell this
                    # narrow. A size nothing measured shows no rate rather
                    # than a "0" nobody observed.
                    claim, got = _ceiling(w, h, fps)
                    # WHOLE numbers, because that is what the FPS cell beside
                    # this one offers: the camera reports 120.101 and the rate
                    # combo labels it "120", so a ".1" here would put two
                    # numbers on screen for one rate.
                    rate = (f"{round(claim)}{ARROW}{round(got)}" if got > 0
                            else (f"{round(claim)}" if claim > 0 else ""))
                    # An untried row ALWAYS names its door: the door is the
                    # whole of what it offers, and "(try)" alone says nothing
                    # about what would be tried.
                    door = (f" {_BACKEND_SHORT.get(backend, backend)}"
                            if backend and (multi_backend or fps <= 0) else "")
                    label = (f"{w}×{h}{door} ({rate})" if rate
                             else f"{w}×{h}{door} (try)")
                elif fps > 0:
                    label = (f"{w} x {h} @ {fps:.1f} fps"
                             + (f"  ({pretty})" if pretty else ""))
                else:
                    # Never measured, say so rather than print a rate nobody
                    # observed. Picking it is how an operator tries a backend
                    # whose probe failed.
                    label = f"{w} x {h}  ({pretty}, not measured, try it)"
                idx = res_combo.count()
                res_combo.addItem(label, {"width": w, "height": h,
                                          "fps": fps, "backend": backend})
                # Size AND door. Matching on size alone re-selected the
                # fastest door to that size, so a deliberate pick of the
                # slower one was undone the moment the row redrew.
                if preselect_wh and (w, h) == preselect_wh:
                    if preselect_backend and backend == preselect_backend:
                        saved_idx = idx
                    elif saved_idx is None and not preselect_backend:
                        saved_idx = idx
                    elif saved_fallback_idx is None:
                        saved_fallback_idx = idx
                # Entries are sorted fastest-first within a size, so the first
                # match is the fastest door to it.
                if default_idx is None and default_wh and (w, h) == default_wh:
                    default_idx = idx
            if saved_idx is None:
                saved_idx = saved_fallback_idx
            if saved_idx is not None:
                res_combo.setCurrentIndex(saved_idx)
            elif default_idx is not None:
                res_combo.setCurrentIndex(default_idx)
            else:
                res_combo.setCurrentIndex(0)
        finally:
            res_combo.blockSignals(was_blocked)
        self._repopulate_fps_combo(res_combo=res_combo, fps_combo=fps_combo,
                                   cam_id=cam_id)

    def applyDarkTheme(self):
        """Apply the shared dark-dialog QSS, plus the chip rules every Ready
        cell inherits instead of styling itself."""
        from source.gui.style_builders import apply_dialog_theme
        apply_dialog_theme(self)
        self.setStyleSheet((self.styleSheet() or "") + self.CHIP_QSS)

    def _build_ui(self):
        """Compose the camera-setup dialog as one tabbed walk-through.

        Tabs, in the order the job is actually done:

            Setups & cameras, the arrangement plus the per-box table: which
                                camera, what it records, its pixel format and
                                its detected resolution / rate
            ROI, the one place regions are drawn
            Camera options, whatever the selected camera reports
            Lens, lens calibration, the only "calibration" left
            Connect, record selection, review, and the connection

        Each tab carries its own completion mark, so the strip doubles as a
        checklist. Nothing here gates anything.
        """
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)
        self._group_qss = self._make_group_qss()

        # Two steps. "Camera options" was a third, holding a single shared
        # Resolution / rate / Format / Orientation card that edited whichever
        # row was selected. Those are columns on the table now, one control
        # per camera, so the tab had nothing left that the table did not say
        # better. What it uniquely owned, the encoder quality and the sensor
        # readout, moved to the tab that decides what gets recorded.
        self._STEPS = ("Setups & cameras", "Regions, lens & connect")
        self._step = 0
        self._syncing_tab = False
        # Shown only after a Connect attempt with nothing assigned.
        self._warn_assign = False

        # Typing a camera id fires one signal per character, and a full
        # refresh resolves USB identities and re-reads the lens store. Coalesce
        # them: the last keystroke in a burst is the only one that matters.
        self._ready_cache = {}
        self._lens_cache = {}
        self._marks_timer = QtCore.QTimer(self)
        self._marks_timer.setSingleShot(True)
        self._marks_timer.setInterval(120)
        self._marks_timer.timeout.connect(self._refresh_marks)
        # Describing a camera's features talks to the SDK, so it must not run
        # once per character either.
        self._features_timer = QtCore.QTimer(self)
        self._features_timer.setSingleShot(True)
        self._features_timer.setInterval(150)
        self._features_timer.timeout.connect(self._refresh_feature_panel)

        self._prewarm_camera_ids()

        # Every control exists before any layout, so the arrangement row can
        # host the shared-camera picker and the tabs can host the capture
        # controls without depending on assembly order.
        self._create_camera_controls()
        self._create_capture_controls()
        for w in (self.sel_record_check, self.record_state_label,
                  self.active_cam_combo, self.shared_cam_combo):
            w.setParent(self)
            w.setVisible(False)

        root.addWidget(self._build_inspector(), 1)

        self.populateBoxes()
        self._load_camera_section()

        root.addLayout(self._build_footer())
        self._goto_step(0)
        self.updateButtonStates()
        self._refresh_marks()

    # ----- inspector tabs + completion marks -----

    def _build_inspector(self):
        """The tabbed walk-through. The table lives on the first tab, where
        the boxes and their cameras are set up; every later tab acts on the
        row selected there."""
        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_box_panel(), self._STEPS[0])
        # Built, deliberately, and not added as a tab. The identity labels and
        # the shared Capture widgets it creates are read and written from ~85
        # places in this dialog; constructing them keeps every one of those
        # working, and they now mirror the per-row columns rather than being
        # the only way to reach a camera. The two parts an operator still
        # needs, encoder quality and the sensor readout, are re-parented onto
        # the Connect tab below, which is why this page is built FIRST.
        self._options_page = self._page_options()
        self._options_page.setVisible(False)
        # AFTER the page exists, because it constructs these and claims them
        # into its own layout as it goes.
        self._adopt_shared_capture_row()
        tabs.addTab(self._page_lens_and_connect(), self._STEPS[1])
        tabs.setStyleSheet(self._tab_qss())
        tabs.currentChanged.connect(self._goto_step)
        self._tabs = tabs
        return tabs

    def _tab_qss(self):
        p = _T_cc.palette
        return (
            f"QTabWidget::pane {{ border:1px solid {p.surface_border_strong};"
            f" border-radius:{_T_cc.radius.md}px; background:{p.surface}; }}"
            f"QTabBar::tab {{ color:{p.text_dim}; background:transparent;"
            " padding:7px 13px; border:0;"
            " border-bottom:2px solid transparent; }"
            f"QTabBar::tab:selected {{ color:{p.text}; font-weight:600;"
            f" border-bottom:2px solid {p.accent}; }}"
            f"QTabBar::tab:hover {{ color:{p.text}; }}")

    def _queue_marks(self):
        """Ask for a refresh soon. Safe to call on every keystroke."""
        timer = getattr(self, "_marks_timer", None)
        if timer is None:
            self._refresh_marks()
        else:
            timer.start()

    def _refresh_marks(self):
        """Restate completion in both places it belongs: a mark on each tab,
        and a chip in each table row. These report; they never block."""
        tabs = getattr(self, "_tabs", None)
        if tabs is None:
            return
        for i, mark in enumerate(self._tab_marks()):
            # "&" in a tab label is a mnemonic unless doubled.
            name = self._STEPS[i].replace("&", "&&")
            tabs.setTabText(i, f"{name}  {mark}" if mark else name)
        self._refresh_table_cells()
        if hasattr(self, "_footer_hint"):
            self._footer_hint.setText(self._footer_hint_text())

    def _tab_marks(self):
        """One short state mark per tab, in tab order."""
        lens, connect = self._lens_status_mark(), self._connect_status_mark()
        # The last tab carries three concerns; show regions and lens only when
        # they apply, so a dedicated-camera rig with no calibration reads as
        # just the connect count rather than a row of "n/a".
        roi = self._roi_status_mark()
        extra = [f"regions {roi}"] if roi != "n/a" else []
        if lens:
            extra.append(f"lens {lens}")
        sep = " \u00b7 "
        tail = sep.join([connect, *extra]) if extra else connect
        return (self._assign_status_mark(), tail)

    def _assign_status_mark(self):
        rows = self.table.rowCount() if hasattr(self, "table") else 0
        assigned = sum(1 for r in range(rows) if self._camera_id_text(r))
        return f"{assigned} of {rows}" if rows else ""

    def _roi_status_mark(self):
        # Required only where one camera is split between boxes. A dedicated
        # camera reads "whole frame", which is a state, not a shortfall.
        rows = self.table.rowCount() if hasattr(self, "table") else 0
        drawn = sum(1 for r in range(rows) if self._roi_for_row(r) is not None)
        if not self._is_cctv_mode() and not drawn:
            return "whole frame"
        return f"{drawn} of {rows}"

    def _lens_status_mark(self):
        cams = self._assigned_camera_ids()
        if not cams:
            return ""
        done = sum(1 for c in cams if self._lens_entry_for(c) is not None)
        return f"{done} of {len(cams)}" if done else "not done"

    def _connect_status_mark(self):
        rows = self.table.rowCount() if hasattr(self, "table") else 0
        ready = sum(1 for r in range(rows) if self._camera_id_text(r))
        return f"{ready} of {rows}" if rows else ""

    def _footer_hint_text(self):
        """What still stands between here and a working session."""
        rows = self.table.rowCount() if hasattr(self, "table") else 0
        missing = [r + 1 for r in range(rows) if not self._camera_id_text(r)]
        if missing:
            return (f"Box {missing[0]} still needs a camera"
                    if len(missing) == 1
                    else f"{len(missing)} boxes still need a camera")
        # Only a shared camera has to be split, so only that is unfinished
        # business. A dedicated camera at whole frame is ready to record.
        undrawn = sum(1 for r in range(rows) if self._roi_for_row(r) is None)
        if undrawn and self._is_cctv_mode():
            return f"{undrawn} region(s) still to draw"
        n = len(self._assigned_camera_ids())
        return f"{n} camera(s) ready" if n else "no cameras assigned yet"

    def _assigned_camera_ids(self):
        """Distinct camera ids assigned to a box, in row order."""
        cams = [c for c in self._allCameraIdsInTable()
                if c not in (None, "")]
        return list(dict.fromkeys(cams))

    def _roi_for_row(self, row):
        """``roi_normalized`` of ``row``'s box widget, or ``None``."""
        widgets = self._row_box_widgets()
        if row >= len(widgets):
            return None
        return getattr(widgets[row], "roi_normalized", None) or None

    def _lens_store(self):
        """The machine's lens-calibration store, read once per dialog and
        dropped after a calibration so a fresh profile is seen."""
        store = getattr(self, "_lens_store_obj", None)
        if store is None:
            from source.video.cameras.lens import LensCalibrationStore
            store = self._lens_store_obj = LensCalibrationStore()
        return store

    def _lens_entry_for(self, cam_id, box_number=None):
        """This camera's stored lens calibration, or ``None``.

        Memoised: resolving the USB identity enumerates devices and the store
        read touches disk, and this is consulted on every refresh.
        """
        ck = (str(cam_id), box_number)
        cache = getattr(self, "_lens_cache", None)
        if cache is None:
            cache = self._lens_cache = {}
        if ck not in cache:
            try:
                cache[ck] = self._lens_store().get(
                    self._lens_key_for(cam_id, box_number))
            except Exception:
                cache[ck] = None
        return cache[ck]

    def _lens_key_for(self, cam_id, box_number=None):
        """Storage key for a lens profile.

        A camera keeps one profile, except a shared/CCTV camera, where each
        box looks through a different part of the frame, so the profile is
        stored per box as well as per camera.
        """
        backend = self._backendForCameraId(cam_id) or "opencv"
        ident = self._resolve_identity_cached(cam_id, backend)
        key = ident.get("unique_id") or f"{cam_id}-{backend}"
        if box_number is not None and self._camera_is_shared(cam_id):
            return f"{key}#box{box_number}"
        return key

    def _lens_crop_rect(self, cam_id, box_number):
        """Normalised ``(x0, y0, x1, y1)`` this box occupies, or ``None``.

        ``None`` means calibrate the whole frame: a dedicated camera, or a
        shared one whose box has no region drawn yet.
        """
        if box_number is None or not self._camera_is_shared(cam_id):
            return None
        widgets = self._row_box_widgets()
        for r, w in enumerate(widgets):
            if getattr(w, "setup_number", r + 1) != box_number:
                continue
            roi = self._roi_for_row(r)
            if roi and len(roi) == 4:
                x, y, rw, rh = (float(v) for v in roi)
                if rw > 0 and rh > 0:
                    return (x, y, x + rw, y + rh)
            return None
        return None

    def _camera_is_shared(self, cam_id):
        """True when more than one box looks through this camera."""
        rows = self.table.rowCount() if hasattr(self, "table") else 0
        seen = sum(1 for r in range(rows)
                   if self._camera_id_text(r) == str(cam_id))
        return seen > 1

    def _invalidate_lens_cache(self):
        self._lens_store_obj = None
        self._lens_cache = {}

    # One stylesheet for every chip, applied once on the dialog. Styling each
    # label individually cost ~7 ms a call, and a rig with eight boxes builds
    # sixteen of them before the dialog is even on screen.
    CHIP_QSS = (
        "QLabel[chip] { border-radius:7px; padding:1px 7px;"
        " font-size:10px; font-weight:600; }"
        "QLabel[chip=\"ok\"] { color:#8fe0a4;"
        " background:rgba(63,185,80,0.14); }"
        "QLabel[chip=\"warn\"] { color:#e8c06a;"
        " background:rgba(210,153,34,0.14); }"
        "QLabel[chip=\"na\"] { color:#8b93a1;"
        " background:rgba(255,255,255,0.05); }"
    )

    def _make_chip(self, text, kind):
        """A small state pill. ``kind`` is ok / warn / na, and selects the
        colours from :attr:`CHIP_QSS` rather than a per-label stylesheet."""
        lbl = QtWidgets.QLabel(text)
        lbl.setProperty("chip", kind if kind in ("ok", "warn", "na") else "na")
        return lbl

    def _camera_config_for(self, cam_id):
        pipe = (getattr(self.main_window, "pipeline", None)
                if self.main_window else None)
        if pipe is None:
            return None
        try:
            return pipe.all_camera_configs().get(str(cam_id))
        except Exception:
            return None

    def _is_cctv_mode(self):
        btn = getattr(self, "mode_cctv_btn", None)
        return bool(btn is not None and btn.isChecked())

    def _goto_step(self, i):
        """Reveal tab ``i``. Always permitted, no gating, no fixed order."""
        if getattr(self, "_syncing_tab", False):
            return                      # re-entry from our own setCurrentIndex
        i = max(0, min(len(self._STEPS) - 1, int(i)))
        self._step = i
        tabs = getattr(self, "_tabs", None)
        if tabs is not None and tabs.currentIndex() != i:
            self._syncing_tab = True
            try:
                tabs.setCurrentIndex(i)
            finally:
                self._syncing_tab = False
        if hasattr(self, "assign_warn"):
            self.assign_warn.setVisible(
                not self._assign_ready() and self._warn_assign)
        if i == 1:
            self._refresh_feature_panel()
        elif i == 2:
            self._refresh_regions_note()
            self._refresh_lens_list()
            self._refresh_record_summary()
            self._refresh_review()
        self._refresh_marks()

    def _assign_ready(self):
        """At least one box has a camera. Reported, never enforced."""
        return bool(self._allCameraIdsInTable())

    def _build_footer(self):
        """Bottom bar: the preset buttons, what is still missing, then the
        actions. Connect is reachable from every tab."""
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self._build_presets_button())
        self._footer_hint = QtWidgets.QLabel("")
        self._footer_hint.setStyleSheet(
            "color:" + _T_cc.palette.text_muted + "; font-size:11px;")
        row.addWidget(self._footer_hint)
        row.addStretch(1)
        row.addLayout(self._build_action_row())
        return row

    # ----- sections -----

    def _build_box_panel(self):
        """Left side, the spine: how the cameras are arranged, the per-box
        table, and the batch detect. It never disappears, so every per-box
        value stays visible for every box at once."""
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        v.addWidget(self._build_arrangement_step())
        v.addWidget(self._build_box_table(), 1)
        v.addLayout(self._build_detect_all_row())
        self.assign_warn = QtWidgets.QLabel(
            "Assign a camera to at least one box before connecting.")
        self.assign_warn.setStyleSheet(
            "color: #f0b45c; background: rgba(224,169,92,0.12);"
            " border: 1px solid rgba(224,169,92,0.35); border-radius: 6px;"
            " padding: 6px 10px;")
        self.assign_warn.setVisible(False)
        v.addWidget(self.assign_warn)
        return panel

    def _page_roi(self):
        """The one place regions are drawn."""
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)
        self._regions_note = QtWidgets.QLabel()
        self._regions_note.setWordWrap(True)
        self._regions_note.setStyleSheet("color:" + _T_cc.palette.text_dim + ";")
        v.addWidget(self._regions_note)
        self._roi_chip_row = QtWidgets.QHBoxLayout()
        self._roi_chip_row.setSpacing(6)
        v.addLayout(self._roi_chip_row)
        self.define_rois_btn2 = QtWidgets.QPushButton("Draw regions\u2026")
        self.define_rois_btn2.clicked.connect(self._onDefineCctvRois)
        v.addWidget(self.define_rois_btn2, 0,
                    QtCore.Qt.AlignmentFlag.AlignLeft)
        v.addStretch(1)
        self._refresh_regions_note()
        return page

    def _refresh_regions_note(self):
        """Explain the arrangement, then show one chip per box: drawn, still
        to draw, or not applicable."""
        if not hasattr(self, "_regions_note"):
            return
        cctv = self._is_cctv_mode()
        # A shared camera MUST be split, so its regions are required. A
        # dedicated camera already frames its box, so its region defaults to
        # the whole frame and drawing one is a choice: worth making when the
        # box is a small part of the picture, because the region fixes the
        # pixels the tracker and the pose model see.
        self._regions_note.setText(
            "One camera is shared, so each box needs its own region of the "
            "frame. The first region drawn fixes the size; every later box "
            "gets that exact size so one pose model fits them all."
            if cctv else
            "Each box has its own camera, so the whole frame is its region "
            "unless one is drawn. Draw one to crop tighter: the region fixes "
            "the pixels the tracker and the pose model see.")
        self.define_rois_btn2.setEnabled(True)
        row = getattr(self, "_roi_chip_row", None)
        if row is None:
            return
        self._clear_layout(row)
        for r in range(self.table.rowCount() if hasattr(self, "table") else 0):
            if self._roi_for_row(r) is not None:
                row.addWidget(self._make_chip(f"Box {r + 1} drawn", "ok"))
            elif cctv:
                row.addWidget(self._make_chip(f"Box {r + 1} to draw", "warn"))
            else:
                # A state, not a shortfall: a dedicated camera works as drawn.
                row.addWidget(self._make_chip(f"Box {r + 1} whole frame", "ok"))
        row.addStretch(1)

    def _build_flip_row(self):
        """Mirror / upside-down correction for the selected camera.

        Applied at capture, so the ROI crop, the tracker, the recorder, the
        zone tests and the coordinates pushed to the MCU all describe the same
        picture. Correcting this at display time only would make the screen
        agree with the operator and every recorded number disagree.

        Built once. It has two callers now, the CCTV shared-capture row and
        the options page kept for its widgets, and a second call would rebind
        ``flip_h_check`` to a fresh pair, leaving whichever row was built
        first holding orphans that toggle nothing.
        """
        existing = getattr(self, "_flip_row_widget", None)
        if existing is not None:
            return existing
        w = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        self.flip_h_check = QtWidgets.QCheckBox("Mirror (left ↔ right)")
        self.flip_h_check.setToolTip(
            "Tick when the animal's left appears on the right.\n"
            "Most webcams mirror their output by default.\n\n"
            "Applied at capture, so the recording, the tracker and the\n"
            "coordinates sent to the board all match what you see.")
        self.flip_v_check = QtWidgets.QCheckBox("Flip (top ↔ bottom)")
        self.flip_v_check.setToolTip("Tick for a camera mounted upside-down.")
        for cb in (self.flip_h_check, self.flip_v_check):
            cb.toggled.connect(self._onFlipToggled)
            row.addWidget(cb)
        row.addStretch(1)
        self._flip_row_widget = w
        return w

    def _onFlipToggled(self, *_):
        """Persist the flip and apply it to the live camera at once."""
        if getattr(self, "_loading_capture", False):
            return
        cam_id = self._active_capture_cam()
        if cam_id is None:
            return
        pipe = getattr(self.main_window, "pipeline", None) if self.main_window else None
        if pipe is None:
            return
        try:
            pipe.set_camera_flip(
                cam_id,
                horizontal=self.flip_h_check.isChecked(),
                vertical=self.flip_v_check.isChecked())
        except Exception as e:
            logger.debug("set_camera_flip(%s) failed: %s", cam_id, e)

    def _page_options(self):
        """Camera options, identity, acquisition, and whatever the selected
        camera reports about itself.

        The acquisition controls live here rather than in the table because a
        webcam offers three of them and a FLIR offers sixty; the table shows
        the resulting values as read-only echoes.
        """
        page = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)

        inner = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(inner)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(10)

        ident = QtWidgets.QGroupBox("Identity")
        ident.setStyleSheet(self._group_qss)
        f1 = self._make_form_layout()
        f1.setContentsMargins(10, 15, 10, 10)
        ident.setLayout(f1)
        self._options_model = QtWidgets.QLabel("\u2014")
        self._options_model.setStyleSheet("color:" + _T_cc.palette.text + ";")
        f1.addRow("Model:", self._options_model)
        f1.addRow("Backend:", self.sel_backend_combo)
        v.addWidget(ident)

        acq = QtWidgets.QGroupBox("Acquisition")
        acq.setStyleSheet(self._group_qss)
        f2 = self._make_form_layout()
        f2.setContentsMargins(10, 15, 10, 10)
        acq.setLayout(f2)
        rr = QtWidgets.QHBoxLayout()
        rr.setSpacing(6)
        rr.addWidget(self.resolution_combo, 1)
        rr.addWidget(self.fps_combo)
        rr.addWidget(self.redetect_btn)
        rrw = QtWidgets.QWidget()
        rrw.setLayout(rr)
        f2.addRow("Resolution / rate:", rrw)
        f2.addRow("Pixel format:", self.format_combo)
        f2.addRow("Orientation:", self._build_flip_row())
        # Output quality is a connection setting, it decides how the frames
        # this tab configures get encoded, so it belongs here rather than on
        # the final step.
        # Output quality decides how the frames get ENCODED, so it belongs
        # with the recording decision rather than with the camera's own
        # settings. Kept on the instance so the Connect tab can adopt it.
        self._video_output_combo_w = self._build_video_output_combo()
        f2.addRow("Output quality:", self._video_output_combo_w)
        v.addWidget(acq)

        # Whatever this camera reports, read from the sensor itself. Kept on
        # the instance because the Connect tab re-parents it: this page is
        # built for its widgets, not to be shown.
        grp = self._build_sci_settings_group()
        grp.setTitle("Reported by this camera")
        self._sci_group = grp
        v.addWidget(grp)
        self._options_host = v
        v.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        scroll.setStyleSheet(
            "QScrollArea, QScrollArea > QWidget > QWidget"
            " { background: transparent; }")
        scroll.viewport().setAutoFillBackground(False)
        self._detail_scroll = scroll
        outer.addWidget(scroll)
        return page

    def _page_lens_and_connect(self):
        """The last step, in two columns.

        Left is everything that describes the camera's view of the arena,
        the regions each box occupies, then the lens that distorts them. Both
        are optional per-rig and both are per-camera geometry, so they stack
        in one column and read top-down in the order you'd do them. Right is
        the decision: what to record, and Connect.

        Splitters throughout: an operator who never calibrates can collapse
        the lens box away, and Connect grows with the box count.
        """
        geometry = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        geometry.setChildrenCollapsible(True)
        geometry.addWidget(self._boxed("Regions", self._page_roi()))
        geometry.addWidget(self._boxed("Lens calibration", self._page_lens()))
        # Re-parented from the removed Camera options tab. The sensor readout
        # describes the camera on the selected row and is per camera, like the
        # geometry above it, so it stacks in the same column and follows the
        # selection exactly as it did before.
        sci = getattr(self, "_sci_group", None)
        if sci is not None:
            sci.setParent(None)
            geometry.addWidget(sci)
            geometry.setStretchFactor(2, 2)
        geometry.setStretchFactor(0, 3)
        geometry.setStretchFactor(1, 2)
        geometry.setSizes([300, 220, 200])
        self._geometry_split = geometry

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(True)
        split.addWidget(geometry)
        split.addWidget(self._page_review())
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([300, 380])
        self._lens_connect_split = split
        return split

    def _boxed(self, title, inner):
        """Wrap a page in a titled group so stacked pages stay distinguishable."""
        group = QtWidgets.QGroupBox(title)
        group.setStyleSheet(self._group_qss)
        lay = QtWidgets.QVBoxLayout(group)
        lay.setContentsMargins(8, 14, 8, 8)
        lay.setSpacing(0)
        lay.addWidget(inner)
        return group

    def _page_lens(self):
        """Lens calibration, the only thing here still called calibration.

        Detecting resolution and frame rate is not lens work; that is the
        Detect button in the table, beside the values it fills in.
        """
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)
        head = QtWidgets.QLabel(
            "Lens calibration solves barrel / wide-angle distortion from a "
            "printed checkerboard and stores the result against the camera's "
            "USB identity. It is needed only for metric distances or a "
            "straightened image.")
        head.setWordWrap(True)
        head.setStyleSheet("color:" + _T_cc.palette.text_dim + ";")
        v.addWidget(head)
        from source.gui.style_builders import button_style as _btn
        self.calibrate_lens_btn = QtWidgets.QPushButton(
            "Calibrate every camera\u2026")
        self.calibrate_lens_btn.setStyleSheet(_btn("info", height=28))
        self.calibrate_lens_btn.setToolTip(
            "Walk the checkerboard wizard once per target, in order.")
        self.calibrate_lens_btn.clicked.connect(self._onCalibrateAllLenses)

        # The wizard needs a printed board, so the board comes from here too.
        # Sourcing it from a web image is how a calibration ends up solved
        # against squares that are not the size the operator entered.
        self.make_board_btn = QtWidgets.QPushButton("Print a checkerboard…")
        self.make_board_btn.setStyleSheet(_btn("secondary", height=28))
        self.make_board_btn.setToolTip(
            "Save a checkerboard at true scale for the paper you print on.")
        self.make_board_btn.clicked.connect(self._onGenerateBoard)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self.calibrate_lens_btn)
        row.addWidget(self.make_board_btn)
        row.addStretch(1)
        v.addLayout(row)

        self._lens_list = QtWidgets.QVBoxLayout()
        self._lens_list.setSpacing(4)
        v.addLayout(self._lens_list)
        v.addStretch(1)
        return page

    def _onGenerateBoard(self):
        """Save a printable checkerboard at true scale.

        The geometry offered is the one the wizard defaults to, so the board
        that comes out of here is the board the solver expects. Anything the
        chosen paper cannot hold at true scale is refused with the size it
        would need, rather than being shrunk to fit and quietly invalidating
        the square size printed on it.
        """
        from source.video.cameras.lens import BoardSpec, PAPER_MM, render_board

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Print a checkerboard")
        form = QtWidgets.QFormLayout(dlg)
        cols = QtWidgets.QSpinBox(); cols.setRange(3, 30); cols.setValue(9)
        rows = QtWidgets.QSpinBox(); rows.setRange(3, 30); rows.setValue(6)
        square = QtWidgets.QDoubleSpinBox()
        square.setRange(5.0, 80.0); square.setValue(25.0); square.setSuffix(" mm")
        paper = QtWidgets.QComboBox(); paper.addItems(sorted(PAPER_MM))
        paper.setCurrentText("A4")
        dpi = QtWidgets.QSpinBox(); dpi.setRange(150, 1200); dpi.setValue(300)
        form.addRow("Inner corners across", cols)
        form.addRow("Inner corners down", rows)
        form.addRow("Square size", square)
        form.addRow("Paper", paper)
        form.addRow("Resolution", dpi)
        note = QtWidgets.QLabel(
            "Inner corners are where four squares meet, so a 9 x 6 board "
            "prints as 10 x 7 squares. Print at 100% with no scaling, and "
            "mount it on something flat.")
        note.setWordWrap(True)
        note.setStyleSheet("color:" + _T_cc.palette.text_dim + ";")
        form.addRow(note)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        spec = BoardSpec(cols=cols.value(), rows=rows.value(),
                         square_mm=float(square.value()))
        try:
            image = render_board(spec, paper=paper.currentText(),
                                 dpi=dpi.value())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Print a checkerboard", str(exc))
            return

        suggested = (f"checkerboard_{spec.cols}x{spec.rows}_"
                     f"{spec.square_mm:g}mm_{paper.currentText()}.png")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save checkerboard", suggested, "PNG image (*.png)")
        if not path:
            return
        try:
            import cv2 as _cv2
            ok = _cv2.imwrite(path, image)
            if not ok:
                raise OSError("OpenCV declined to write the file")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Print a checkerboard", f"Could not save:\n{exc}")
            return
        QtWidgets.QMessageBox.information(
            self, "Print a checkerboard",
            f"Saved {Path(path).name}.\n\n"
            f"Print at 100% (no 'fit to page'), then measure one square: it "
            f"should be {spec.square_mm:g} mm. Enter {spec.cols} x {spec.rows} "
            f"corners and {spec.square_mm:g} mm in the calibration wizard.")

    def _lens_targets(self):
        """What can be calibrated, as ``(camera_id, box_number)``.

        One entry per camera normally. Under a shared camera the frame each
        box sees is a different crop, so each box is its own target.
        """
        rows = self.table.rowCount() if hasattr(self, "table") else 0
        targets, seen = [], set()
        for r in range(rows):
            # _cameraIdForRow, not _camera_id_text: the FrameBus is keyed by
            # the value connect_camera used, which is an int for OpenCV. A
            # string "0" would miss a bus stored under 0 and the wizard would
            # claim the camera is not connected.
            cam = self._cameraIdForRow(r)
            if cam in (None, ""):
                continue
            box = r + 1
            widgets = self._row_box_widgets()
            if r < len(widgets):
                box = getattr(widgets[r], "setup_number", box)
            if self._camera_is_shared(cam):
                targets.append((cam, box))
            elif cam not in seen:
                seen.add(cam)
                targets.append((cam, None))
        return targets

    def _refresh_lens_list(self):
        """One row per target: what it is, its state, and its own button."""
        if not hasattr(self, "_lens_list"):
            return
        self._clear_layout(self._lens_list)
        targets = self._lens_targets()
        if not targets:
            self._lens_list.addWidget(
                QtWidgets.QLabel("Assign a camera in the table first."))
            return
        from source.gui.style_builders import button_style as _btn
        for cam, box in targets:
            entry = self._lens_entry_for(cam, box)
            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)
            name = (f"camera {cam}   \u00b7   Box {box}" if box is not None
                    else f"camera {cam}")
            lbl = QtWidgets.QLabel(name)
            lbl.setStyleSheet(f"color: {_T_cc.palette.text};")
            h.addWidget(lbl)
            when = (getattr(entry, "calibrated_at", "") or "")[:10] \
                if entry else ""
            h.addWidget(self._make_chip(when or "not calibrated",
                                        "ok" if entry else "warn"))
            h.addStretch(1)
            btn = QtWidgets.QPushButton(
                "Re-calibrate\u2026" if entry else "Calibrate\u2026")
            btn.setStyleSheet(_btn("secondary", height=26))
            btn.clicked.connect(
                lambda _=False, c=cam, b=box: self._onCalibrateLens(c, b))
            h.addWidget(btn)
            self._lens_list.addWidget(row)

    def _onCalibrateAllLenses(self):
        """Run the wizard for every target in turn. Cancelling one stops the
        run rather than marching through the rest."""
        targets = self._lens_targets()
        if not targets:
            QtWidgets.QMessageBox.information(
                self, "Calibrate lens", "Assign a camera in the table first.")
            return
        for cam, box in targets:
            if self._onCalibrateLens(cam, box) is False:
                break
        self._refresh_lens_list()

    def _page_review(self):
        """Recording options and the connect action, together: both are about
        starting a session rather than describing the rig."""
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(10)
        rec = QtWidgets.QGroupBox("Record selection")
        rec.setStyleSheet(self._group_qss)
        self._record_summary = QtWidgets.QVBoxLayout(rec)
        self._record_summary.setContentsMargins(12, 16, 12, 10)
        self._record_summary.setSpacing(4)
        v.addWidget(rec)
        # Re-parented from the removed Camera options tab. It decides how the
        # frames are ENCODED, which is a property of the recording rather than
        # of any one camera, so it sits with Record selection.
        combo = getattr(self, "_video_output_combo_w", None)
        if combo is not None:
            combo.setParent(None)
            quality = QtWidgets.QHBoxLayout()
            quality.setContentsMargins(0, 0, 0, 0)
            quality.setSpacing(8)
            label = QtWidgets.QLabel("Output quality:")
            label.setStyleSheet(f"color:{_T_cc.palette.text_muted};")
            quality.addWidget(label)
            quality.addWidget(combo, 1)
            v.addLayout(quality)
        v.addLayout(self._build_auto_connect_row())
        grp = QtWidgets.QGroupBox("Review")
        grp.setStyleSheet(self._group_qss)
        self._review_box = QtWidgets.QVBoxLayout(grp)
        self._review_box.setContentsMargins(12, 16, 12, 10)
        self._review_box.setSpacing(5)
        v.addWidget(grp, 1)
        return page

    @staticmethod
    def _clear_layout(lay):
        """Empty ``lay``, unparenting as it goes.

        ``deleteLater`` alone leaves each widget a child of the page until the
        event loop drains, and a child with no layout keeps its default
        640x480 geometry at the origin, which paints over everything.
        """
        while lay.count():
            it = lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _refresh_record_summary(self):
        if not hasattr(self, "_record_summary"):
            return
        self._clear_layout(self._record_summary)
        sel = self.getSelectedBoxesWithCameras()
        if not sel:
            self._record_summary.addWidget(
                QtWidgets.QLabel("No cameras assigned yet."))
            return
        for item in sel:
            on = bool(item.get("save_video", True))
            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(9)
            dot = QtWidgets.QLabel("●" if on else "○")
            dot.setStyleSheet(f"color: {'#ff9b95' if on else '#69707d'};")
            h.addWidget(dot)
            h.addWidget(QtWidgets.QLabel(
                f"Box {item['box_number']}   ·   camera {item['camera_id']}"))
            h.addStretch(1)
            state = QtWidgets.QLabel(
                "Record selected" if on else "Not selected")
            state.setStyleSheet(
                "color: %s;" % (_T_cc.palette.success if on
                                else _T_cc.palette.text_muted))
            h.addWidget(state)
            self._record_summary.addWidget(row)

    def _refresh_review(self):
        if not hasattr(self, "_review_box"):
            return
        self._clear_layout(self._review_box)
        cctv = bool(getattr(self, "mode_cctv_btn", None)
                    and self.mode_cctv_btn.isChecked())
        head = QtWidgets.QLabel(
            ("Shared / CCTV, one camera split into per-box regions."
             if cctv else "One camera per box.")
            + "   Review below, then Connect.")
        head.setStyleSheet("color: #98a1b0;")
        self._review_box.addWidget(head)
        sel = self.getSelectedBoxesWithCameras()
        if not sel:
            self._review_box.addWidget(QtWidgets.QLabel(
                "No cameras assigned, pick one per box in the table."))
            return
        pipe = getattr(self.main_window, "pipeline", None) if self.main_window else None
        for item in sel:
            cid = item["camera_id"]
            res = None
            if pipe is not None:
                try:
                    cfg = pipe.all_camera_configs().get(str(cid))
                    res = getattr(cfg, "selected_resolution", None)
                except Exception:
                    res = None
            resx = f"{res[0]}×{res[1]}" if res else "not detected"
            rec = "REC" if item.get("save_video", True) else "no rec"
            lbl = QtWidgets.QLabel(
                f"Box {item['box_number']}   ·   camera {cid}   ·   "
                f"{resx}   ·   {rec}")
            lbl.setStyleSheet("color: #e6eaf0;")
            self._review_box.addWidget(lbl)
        self._review_box.addStretch(1)

    # ----- segmented control -----

    def _segment_qss(self):
        p = _T_cc.palette
        return (
            "QPushButton {"
            f" color: {p.text_muted}; background: {p.surface};"
            f" border: 1px solid {p.surface_border_strong};"
            " padding: 8px 18px; font-weight: 600;"
            "}"
            f"QPushButton:checked {{ background: {p.accent}; color: #0f1216;"
            f" border-color: {p.accent}; }}"
        )

    def _build_arrangement_step(self):
        """Step 1, one clear choice: a camera per box, or one shared/CCTV
        camera. Drives the same Individual/CCTV logic as before."""
        group = QtWidgets.QGroupBox()
        group.setStyleSheet(self._group_qss)
        v = QtWidgets.QVBoxLayout(group)
        v.setSpacing(8)
        lbl = QtWidgets.QLabel("How are your cameras arranged?")
        lbl.setStyleSheet("font-weight: 700; font-size: 13px; color: #e6eaf0;")
        v.addWidget(lbl)

        seg = QtWidgets.QHBoxLayout()
        seg.setSpacing(0)
        self.mode_individual_btn = QtWidgets.QPushButton("One camera per box")
        self.mode_cctv_btn = QtWidgets.QPushButton("Shared camera (CCTV)")
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for b in (self.mode_individual_btn, self.mode_cctv_btn):
            b.setCheckable(True)
            b.setStyleSheet(self._segment_qss())
            self._mode_group.addButton(b)
            b.toggled.connect(self._onCameraModeToggled)
        self.mode_individual_btn.setChecked(True)
        seg.addWidget(self.mode_individual_btn)
        seg.addWidget(self.mode_cctv_btn)
        seg.addStretch(1)
        v.addLayout(seg)

        # CCTV drives every box from one picker; individual mode uses the
        # per-row Camera cells instead, so this row only exists for CCTV.
        self._shared_cam_row = QtWidgets.QWidget()
        sr = QtWidgets.QHBoxLayout(self._shared_cam_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.setSpacing(8)
        sr_lbl = QtWidgets.QLabel("Camera for all boxes:")
        sr_lbl.setStyleSheet("color: #98a1b0;")
        sr.addWidget(sr_lbl)
        sr.addWidget(self.sel_device_combo, 1)
        self._shared_cam_row.setVisible(False)
        v.addWidget(self._shared_cam_row)

        # The same settings, ONCE, for the one camera every box shares. In
        # CCTV the per-row Resolution / FPS / Format / Flip / Detect columns
        # are four copies of a value that can only ever be one, and four
        # chances for them to disagree, exactly the reason Camera and Backend
        # are already hidden in this mode. These are the original shared
        # widgets, which is what they were built to be.
        # Two lines, not one: five controls and a label on a single row give
        # each combo a few characters of width, and the resolution strings are
        # the widest thing in the dialog. The split follows the question each
        # line answers, what the camera captures, then which way up it is.
        self._shared_capture_row = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(self._shared_capture_row)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(4)
        self._shared_capture_lines = []
        for text in ("Mode for all boxes:", "Orientation:"):
            line = QtWidgets.QHBoxLayout()
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(8)
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet("color: #98a1b0;")
            lbl.setMinimumWidth(118)
            line.addWidget(lbl)
            cv.addLayout(line)
            self._shared_capture_lines.append(line)
        # The controls are adopted later, by ``_adopt_shared_capture_row``.
        # They are created by ``_page_options``, which runs AFTER this panel
        # and re-parents them into its own form as it builds: filling the row
        # here left it holding a label and nothing else.
        self._shared_capture_row.setVisible(False)
        v.addWidget(self._shared_capture_row)

        self.arrangement_help = QtWidgets.QLabel()
        self.arrangement_help.setWordWrap(True)
        self.arrangement_help.setStyleSheet("color: #98a1b0; font-size: 11px;")
        v.addWidget(self.arrangement_help)
        return group

    def _build_detect_all_row(self):
        """Device-list actions under the table.

        No batch "Detect all cameras" here: a camera is measured per row
        by its own Detect button, and any camera still missing a
        resolution/rate is probed automatically on connect, so a batch
        button would only offer a third way to do what
        already happens.
        """
        from source.gui.style_builders import button_style as _btn
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        row.setContentsMargins(0, 0, 0, 0)
        self.refresh_ids_btn = QtWidgets.QPushButton("Refresh camera IDs")
        self.refresh_ids_btn.setToolTip(
            "Re-scan for connected cameras. Needed after plugging one in, the "
            "device list is walked once and cached, because the scan opens "
            "every index in turn.")
        self.refresh_ids_btn.setStyleSheet(_btn("secondary", height=28))
        self.refresh_ids_btn.clicked.connect(self._onRefreshCameraIds)
        row.addWidget(self.refresh_ids_btn)
        row.addStretch(1)
        return row

    def _camera_id_combos(self):
        """Every editable Camera-ID combo in the dialog: one per table row plus
        the two right-pane pickers."""
        combos = []
        if hasattr(self, "table"):
            for r in range(self.table.rowCount()):
                c = self.table.cellWidget(r, self.COL_CAMERA_ID)
                if c is not None:
                    combos.append(c)
        for name in ("sel_device_combo", "shared_cam_combo"):
            c = getattr(self, name, None)
            if c is not None:
                combos.append(c)
        return combos

    def _onRefreshCameraIds(self):
        """Re-scan the machine for cameras and refill every Camera-ID picker.

        The device walk opens each index in turn with a blocking read, so it is
        cached process-wide and each combo fills once. That makes a camera
        plugged in after launch invisible until something drops both caches,
        which is this button. Clearing only one of them silently does nothing.
        """
        from source.video.cameras.factory import invalidate_camera_enumeration

        invalidate_camera_enumeration()
        self._avail_cam_ids_cache = None
        self._identity_cache = {}
        self._invalidate_lens_cache()

        combos = self._camera_id_combos()
        for combo in combos:
            # Without this the combo believes it is already filled and
            # showPopup() never calls back.
            combo._populated = False

        # The scan opens each index in turn with a blocking read, 3-20 s on
        # Windows, so it runs on a worker while the GUI keeps painting. The
        # operator has to see the result, though, so we wait here rather than
        # returning early and leaving the stale list on screen.
        prev = self.refresh_ids_btn.text()
        self.refresh_ids_btn.setEnabled(False)
        self.refresh_ids_btn.setText("Scanning…")
        done = threading.Event()

        def _scan():
            try:
                self._available_camera_ids()
            finally:
                done.set()

        threading.Thread(target=_scan, name="camera-enum-refresh",
                         daemon=True).start()
        try:
            finished = self._wait_event_pumping(done, 40.0)
            found = self._avail_cam_ids_cache or []
            self._fill_all_camera_id_combos()
        finally:
            self.refresh_ids_btn.setText(prev)
            self.refresh_ids_btn.setEnabled(True)

        self._refresh_marks()
        logger.info("Camera ID refresh: %d camera(s) detected", len(found))
        if not finished:
            QtWidgets.QMessageBox.warning(
                self, "Refresh camera IDs",
                "The scan is taking unusually long and was left running in the "
                "background. A camera or its driver may be unresponsive; try "
                "again in a moment.")
        elif not found:
            QtWidgets.QMessageBox.information(
                self, "Refresh camera IDs",
                "No cameras detected. Check the cable and that no other "
                "application is holding the device.")

    def _create_camera_controls(self):
        """Create + wire the per-box Device/Backend/Record widgets (proxy to the
        hidden table cells). Plus a hidden shared-camera handle used by the CCTV
        stamping code."""
        self._loading_camera_section = False
        self.sel_device_combo = _CameraIdCombo(self._populate_camera_id_combo)
        self.sel_device_combo.setMinimumWidth(220)
        self.sel_device_combo.lineEdit().setPlaceholderText("camera id / device")
        self.sel_device_combo.currentTextChanged.connect(self._on_sel_device_changed)

        self.sel_backend_combo = QtWidgets.QComboBox()
        for b in self._available_backends:
            self.sel_backend_combo.addItem(b)
        self.sel_backend_combo.currentTextChanged.connect(self._on_sel_backend_changed)

        self.sel_record_check = _ToggleSwitch()
        self.sel_record_check.toggled.connect(self._on_sel_record_changed)
        self.record_state_label = QtWidgets.QLabel("On")
        self.record_state_label.setStyleSheet("color: #e6eaf0;")

        # CCTV drives every box from the one device picker; this hidden handle
        # keeps the existing shared-camera code paths working.
        self.shared_cam_combo = _CameraIdCombo(self._populate_camera_id_combo)
        self.shared_cam_combo.setVisible(False)
        self.shared_cam_combo.currentTextChanged.connect(self._onSharedCameraChanged)

    def _create_capture_controls(self):
        """Create + wire Resolution / FPS / Format + the (hidden) active-camera
        handle the capture load/save logic keys on."""
        self._loading_capture = False
        self.active_cam_combo = QtWidgets.QComboBox()
        self.active_cam_combo.setVisible(False)
        self.active_cam_combo.currentIndexChanged.connect(self._on_active_cam_changed)

        self.resolution_combo = QtWidgets.QComboBox()
        self.resolution_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.resolution_combo.setMinimumContentsLength(14)
        self.resolution_combo.setMinimumWidth(130)
        self.resolution_combo.addItem("Detect to populate", None)
        self.resolution_combo.currentIndexChanged.connect(self._onCaptureResolutionChanged)

        self.fps_combo = QtWidgets.QComboBox()
        self.fps_combo.setMinimumWidth(88)
        self.fps_combo.setEnabled(False)
        self.fps_combo.currentIndexChanged.connect(self._onCaptureFpsChanged)

        from source.gui.style_builders import button_style as _btn
        self.redetect_btn = QtWidgets.QPushButton("Detect")
        self.redetect_btn.setToolTip("Measure this camera's resolutions + frame rates.")
        self.redetect_btn.setStyleSheet(_btn("secondary", height=28))
        self.redetect_btn.clicked.connect(self._onRedetectActive)

        self.format_combo = QtWidgets.QComboBox()
        # These labels are long enough to force the whole inspector wider than
        # its share of the splitter; elide instead of dictating the layout.
        self.format_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.format_combo.setMinimumContentsLength(16)
        self.format_combo.addItem("MJPEG, compressed, full FPS (recommended)", "mjpeg")
        self.format_combo.addItem("YUV, uncompressed (may cap FPS)", "yuv")
        self.format_combo.currentIndexChanged.connect(self._onCaptureFormatChanged)

    #: What the CCTV "Mode for all boxes" row must offer. Named so the test
    #: and the builder cannot drift: a missing control here is a setting the
    #: operator cannot reach at all in CCTV, because the per-row columns that
    #: would otherwise carry it are hidden in that mode.
    #: ``(attribute, line, stretch)``. Line 0 is what the camera captures,
    #: line 1 is which way up it is.
    _SHARED_CAPTURE_WIDGETS = (("resolution_combo", 0, 1),
                               ("fps_combo", 0, 0),
                               ("format_combo", 0, 0),
                               ("redetect_btn", 0, 0),
                               ("_flip_row_widget", 1, 1))

    def _adopt_shared_capture_row(self):
        """Move the shared capture controls into the CCTV row.

        Late, and on purpose. ``_page_options`` builds these and adds them to
        its own form, so anything moved before that call is taken straight
        back: the row ended up holding its label and nothing else, and in
        CCTV, where the per-row columns are hidden, that left no way to set
        resolution, rate, format, orientation or to re-detect.
        """
        row = getattr(self, "_shared_capture_row", None)
        lines = getattr(self, "_shared_capture_lines", None)
        if row is None or not lines:
            return
        missing = []
        for attr, line, stretch in self._SHARED_CAPTURE_WIDGETS:
            w = getattr(self, attr, None)
            if w is None:
                missing.append(attr)
                continue
            w.setParent(None)
            lines[min(line, len(lines) - 1)].addWidget(w, stretch)
        for line in lines:
            line.addStretch(1)
        if missing:
            # Never silent: in CCTV these are the ONLY way to reach the
            # setting, so one that failed to arrive is unreachable.
            logger.error("camera dialog: the shared-camera row is missing %s, "
                         "so those settings cannot be changed in CCTV mode.",
                         ", ".join(missing))

    def _set_shared_camera_row_visible(self, on):
        """Swap between per-box pickers and one master picker.

        With a shared camera every box uses the same device and backend, so
        the per-row Camera and Backend columns would be four controls that can
        only ever hold one value, and four chances for them to disagree. The
        columns are hidden and the master picker above the table drives them
        all; Record and the ROI stay per box.
        """
        on = bool(on)
        for attr in ("_shared_cam_row", "_shared_capture_row"):
            row = getattr(self, attr, None)
            if row is not None:
                row.setVisible(on)
        if hasattr(self, "table"):
            self.table.setColumnHidden(self.COL_CAMERA_ID, on)
            self.table.setColumnHidden(self.COL_BACKEND, on)
            # Same argument as Camera and Backend: with one shared camera
            # these describe that camera, not the box, so per-row copies read
            # as four independent settings when there is only one.
            for col in (self.COL_RES, self.COL_FPS, self.COL_FORMAT,
                        self.COL_FLIP, self.COL_DETECT):
                self.table.setColumnHidden(col, on)

    # ----- rig-wide video output preset + presets menu -----

    _VOUT_PRESETS = (
        ("Balanced (recommended)", (False, 23, True)),
        ("Smaller files (H.265)", (True, 26, True)),
        ("Best quality (largest)", (False, 18, True)),
    )

    def _current_vout_preset_index(self):
        from source.config import settings as _cfg
        hevc = bool(_cfg.get_setting("video", "prefer_hevc"))
        crf = _cfg.get_setting("video", "quality_crf")
        crf = int(crf) if crf is not None else 23
        best, bi = 1e9, 0
        for i, (_, (h, c, _a)) in enumerate(self._VOUT_PRESETS):
            d = (0 if h == hevc else 5) + abs(c - crf)
            if d < best:
                best, bi = d, i
        return bi

    def _onVideoOutputPresetChanged(self, idx):
        from source.config import settings as _cfg
        if not (0 <= idx < len(self._VOUT_PRESETS)):
            return
        hevc, crf, allow = self._VOUT_PRESETS[idx][1]
        try:
            _cfg.set_setting("video", "prefer_hevc", hevc)
            _cfg.set_setting("video", "quality_crf", crf)
            _cfg.set_setting("video", "allow_cpu_fallback", allow)
        except Exception as e:
            logger.debug("video output preset persist: %s", e)

    def _build_video_output_combo(self):
        """Rig-wide output-quality preset (codec + CRF + GPU folded into one
        friendly choice). Returned bare, it sits in the Camera options form,
        which supplies the label."""
        self.video_output_combo = QtWidgets.QComboBox()
        for label, _cfg_vals in self._VOUT_PRESETS:
            self.video_output_combo.addItem(label)
        self.video_output_combo.setCurrentIndex(
            self._current_vout_preset_index())
        self.video_output_combo.setToolTip(
            "How recorded video is encoded (applies to all boxes). Balanced is "
            "visually lossless; Smaller files uses H.265; Best is largest.")
        self.video_output_combo.currentIndexChanged.connect(
            self._onVideoOutputPresetChanged)
        return self.video_output_combo

    def _build_presets_button(self):
        """Save/Load camera presets, as four visible buttons.

        These were collapsed into a ``Presets ▾`` menu, which bought nothing:
        four short actions cost one row laid out flat and take two clicks each
        when hidden behind a menu.
        """
        from source.gui.style_builders import button_style as _btn
        holder = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        qss = _btn("secondary", height=30)
        self.preset_buttons = {}
        for label, slot, key in (
                ("Save camera", self._onSaveCameraConfig, "save_one"),
                ("Load camera", self._onLoadCameraConfig, "load_one"),
                ("Save all", self._onSaveAllCameras, "save_all"),
                ("Load all", self._onLoadAllCameras, "load_all")):
            btn = QtWidgets.QPushButton(label)
            btn.setStyleSheet(qss)
            btn.clicked.connect(slot)
            row.addWidget(btn)
            self.preset_buttons[key] = btn
        return holder

    # ----- shared groupbox QSS used by every section -----

    def _make_group_qss(self):
        """Token-driven QSS that every section's QGroupBox shares so they
        read as one family. Title sits on a coloured underline; body uses
        the surface_elev card."""
        p_cc = _T_cc.palette
        return (
            "QGroupBox {"
            f" font: 700 11pt '{_T_cc.font.family}';"
            f" color: {p_cc.text};"
            f" border: 1px solid {p_cc.surface_border_strong};"
            f" border-radius: {_T_cc.radius.md}px;"
            f" margin-top: 14px;"
            f" padding: 14px 12px 10px;"
            f" background-color: {p_cc.surface_elev};"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 12px;"
            "  padding: 0 6px;"
            f" background-color: {p_cc.surface};"
            "}"
        )

    @staticmethod
    def _make_form_layout():
        """Standard QFormLayout with this dialog's spacing/alignment.
        Parentless so the caller controls placement."""
        f = QtWidgets.QFormLayout()
        f.setSpacing(8)
        f.setContentsMargins(12, 10, 12, 10)
        f.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                            | QtCore.Qt.AlignmentFlag.AlignVCenter)
        return f

    # ----- header row (bulk Save Video / Select toggles) -----

    def _onDefineCctvRois(self):
        """Draw per-box ROIs on the one shared CCTV camera, reusing the connect
        flow's ROI editor (which opens the camera transiently when it isn't
        streaming). Restores any existing ROIs for editing."""
        selected = self.getSelectedBoxesWithCameras()
        if not selected:
            QtWidgets.QMessageBox.information(
                self, "Define ROIs",
                "Enter the shared camera id first, then draw the box regions.")
            return
        camera_id_map: dict = {}
        for item in selected:
            camera_id_map.setdefault(item["camera_id"], []).append(item)
        # One region size across EVERY camera, not one per camera. All boxes
        # feed the same pose model and the model takes a single input shape:
        # ``canonical_shape`` uses the widest width and tallest height and
        # letterboxes the rest, so unequal ROIs make every box pay the largest
        # box's inference cost while the smaller ones lose input resolution.
        # Within one camera the first ROI already locks the others; this is
        # the same rule carried from one camera to the next.
        locked = self._existing_roi_size()
        refused = []
        for camera_id, items in camera_id_map.items():
            seg = ROISegmentationDialog(camera_id, items, self.main_window,
                                        locked_size=locked)
            if seg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            self.applySegments(items, seg.segments,
                               camera_resolution=seg.frame_size)
            if getattr(seg, "lock_rejected", False):
                refused.append((camera_id, seg.frame_size))
            locked = seg.locked_size() or locked
        # Never silent: a box that kept its own size gets letterboxed against
        # the others, the thing the shared size exists to avoid.
        self._warn_roi_sizes_differ(refused, locked)
        self.updateButtonStates()

    def _onCameraModeToggled(self, *_):
        """Enter/leave CCTV. In CCTV the one device picker drives every box (the
        Box-regions row appears); Individual is a camera per box."""
        cctv = bool(getattr(self, "mode_cctv_btn", None) and self.mode_cctv_btn.isChecked())
        if not hasattr(self, "table"):
            return   # initial setChecked fires before the table is built
        if hasattr(self, "arrangement_help"):
            self.arrangement_help.setText(
                "One shared/overhead camera covers all boxes, each box is a "
                "region you draw on its frame, on the ROI tab."
                if cctv else
                "Each box has its own camera. Pick a device in each row's "
                "Camera cell.")
        self._set_shared_camera_row_visible(cctv)
        self._set_row_camera_ids_editable(not cctv)
        self._refresh_regions_note()
        if cctv:
            cid = self.shared_cam_combo.currentText().strip()
            if not cid:
                first = self._firstCameraIdInTable()
                cid = str(first) if first is not None else ""
            if cid:
                self.shared_cam_combo.blockSignals(True)
                self.shared_cam_combo.setEditText(cid)
                self.shared_cam_combo.blockSignals(False)
                self._apply_shared_camera(cid)

    def _onSharedCameraChanged(self, *_):
        if getattr(self, "mode_cctv_btn", None) and self.mode_cctv_btn.isChecked():
            self._apply_shared_camera(self.shared_cam_combo.currentText().strip())

    def _apply_shared_camera(self, cam_id):
        """Stamp ``cam_id`` into every box row's Camera ID cell (CCTV). Signals
        are blocked during the fan-out so per-row handlers don't storm; the
        active-camera picker + button states are refreshed once at the end."""
        if not cam_id:
            return
        for r in range(self.table.rowCount()):
            edit = self.table.cellWidget(r, self.COL_CAMERA_ID)
            if edit is None:
                continue
            edit.blockSignals(True)
            try:
                edit.setEditText(str(cam_id))
            finally:
                edit.blockSignals(False)
        self._refresh_active_cam_combo()
        self._load_active_camera_settings()
        self._load_camera_section()
        self._refresh_status_dots()
        self.updateButtonStates()

    def _set_row_camera_ids_editable(self, editable):
        """Enable/disable per-row Camera ID editing (disabled in CCTV mode,
        where the shared picker drives every box). Disabled combos still report
        their text to the connect flow. The right-pane device combo follows."""
        for r in range(self.table.rowCount()):
            edit = self.table.cellWidget(r, self.COL_CAMERA_ID)
            if edit is not None:
                edit.setEnabled(bool(editable))
        # The right-pane device picker stays enabled in both modes, in CCTV it
        # is the shared-camera picker that drives every box.

    def _detect_camera_mode(self):
        """Pick the initial mode from the current rows: CCTV when 2+ boxes
        already point at the one same camera (the shape the shared-camera ROI
        path produces), else Individual."""
        if not hasattr(self, "mode_cctv_btn"):
            return
        ids = self._allCameraIdsInTable()
        nonempty = sum(1 for r in range(self.table.rowCount())
                       if self._camera_id_text(r))
        cctv = len(ids) == 1 and nonempty >= 2
        btn = self.mode_cctv_btn if cctv else self.mode_individual_btn
        for b in (self.mode_individual_btn, self.mode_cctv_btn):
            b.blockSignals(True)
        btn.setChecked(True)
        for b in (self.mode_individual_btn, self.mode_cctv_btn):
            b.blockSignals(False)
        if cctv:
            self.shared_cam_combo.blockSignals(True)
            self.shared_cam_combo.setEditText(str(ids[0]))
            self.shared_cam_combo.blockSignals(False)
        self._set_shared_camera_row_visible(cctv)
        self._onCameraModeToggled()   # sync help text + editability + stamping

    def _connected_box_numbers(self):
        """Box numbers currently streaming a camera, from the video manager's
        box→camera map (empty set when nothing is connected)."""
        vm = getattr(self.main_window, "video_manager", None) if self.main_window else None
        bcm = getattr(vm, "box_camera_map", None) if vm is not None else None
        return set(bcm.keys()) if isinstance(bcm, dict) else set()

    def _row_box_widgets(self):
        """Box widgets in table-row order (mirrors populateBoxes' iteration)."""
        mw = self.main_window
        if mw is None or not hasattr(mw, "get_all_setup_widgets"):
            return []
        return list(mw.get_all_setup_widgets())

    def _refresh_status_dots(self):
        """Box column: a filled dot for a streaming camera, hollow for idle.
        Pure display, the rest of the row carries the box's settings."""
        connected = self._connected_box_numbers()
        widgets = self._row_box_widgets()
        for row in range(self.table.rowCount()):
            n = (getattr(widgets[row], "setup_number", row + 1)
                 if row < len(widgets) else row + 1)
            item = self.table.item(row, self.COL_BOX)
            if item is None:
                item = QtWidgets.QTableWidgetItem()
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled
                              | QtCore.Qt.ItemFlag.ItemIsSelectable)
                self.table.setItem(row, self.COL_BOX, item)
            live = n in connected
            item.setText(("\u25cf " if live else "\u25cb ") + f"Box {n}")
            item.setForeground(QtGui.QColor("#8fe0a4" if live else "#dfe4ec"))
            item.setToolTip("streaming" if live else "idle")
        self._refresh_table_cells()

    # ----- box table -----

    # Column indices, the single source of truth for every cell read/write.
    # Backend stays a column so existing readers keep working, but it is
    # hidden: picking the device already implies its backend.
    # Resolution, rate, format and orientation are columns of their OWN, and
    # each row edits its own camera. They used to be read-only echoes of a
    # single shared card that showed whichever row was selected, so with one
    # camera per box there was no way to give each camera its own resolution
    # and rate: you set them, picked another row, and set the same widgets
    # again for a different camera.
    COL_BOX = 0
    COL_CAMERA_ID = 1
    COL_BACKEND = 2
    # Backend, then FORMAT, then resolution, then rate: that is the order the
    # decisions actually depend on each other in. The backend decides which
    # formats exist, the format decides which sizes exist, and the size
    # decides which rates exist. Format used to sit to the RIGHT of the two
    # columns it governs, so the operator set a resolution and a rate and then
    # changed the thing that determined both.
    COL_FORMAT = 3
    COL_RES = 4
    COL_FPS = 5
    COL_SAVE = 6
    # Detect fills in the three columns to its left, and Record is the
    # decision about the session, so both come before Flip: orientation is a
    # correction you set once and rarely return to.
    COL_DETECT = 7
    COL_FLIP = 8
    COL_READY = 9

    def _build_box_table(self):
        """The per-box table: every value you set for a box, visible for every
        box at once. Editing a cell changes only that box; the tabs on the
        right act on whichever row is selected."""
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels(
            ["Box", "Camera", "Backend", "Format", "Resolution", "FPS",
             "Record", "", "Flip", "Ready"])
        hdr = self.table.horizontalHeader()
        hdr.setStretchLastSection(True)
        rm = QtWidgets.QHeaderView.ResizeMode
        for col, mode, width in (
                (self.COL_BOX, rm.Fixed, 68),
                # Widths measured against what each cell actually renders,
                # not guessed: a combo elides rather than shrinks, so a column
                # a few pixels short shows "M" for "MJPEG", or "cbb52d3" for
                # a camera id, over a config that is entirely correct. With
                # the cell chrome trimmed (_CELL_COMBO_QSS) these are the
                # narrowest widths that still draw the longest real value.
                # Editable, so the text sits in a line edit narrower than the
                # cell; this leaves real headroom rather than the 3 px that
                # still clipped the id's last character.
                (self.COL_CAMERA_ID, rm.Fixed, 162),
                (self.COL_BACKEND, rm.Fixed, 129),
                (self.COL_FORMAT, rm.Fixed, 108),
                # Wide enough for "1920×1080 MF (30.4)": the ceiling rides
                # beside the size so the rate a size can reach is visible
                # while choosing it, not only afterwards, and the door is
                # named whenever a camera offers more than one. Measured on
                # the real display by tools/check_cell_widths.py; 176 fitted
                # the label before the door was added and clipped it after.
                (self.COL_RES, rm.Fixed, 196),
                (self.COL_FPS, rm.Fixed, 107),
                (self.COL_SAVE, rm.Fixed, 56),
                (self.COL_FLIP, rm.Fixed, 74),
                (self.COL_DETECT, rm.Fixed, 70),
                (self.COL_READY, rm.Stretch, 0)):
            hdr.setSectionResizeMode(col, mode)
            if width:
                self.table.setColumnWidth(col, width)
        hdr.setMinimumSectionSize(46)
        self.table.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        # Record reads as a decision about the session, so it sits after the
        # values that describe the camera rather than between them.
        hdr.moveSection(hdr.visualIndex(self.COL_SAVE), self.COL_DETECT)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(False)
        p = _T_cc.palette
        self.table.setStyleSheet(
            "QTableWidget {"
            f" background: {p.surface}; border: 1px solid {p.surface_border_strong};"
            f" border-radius: {_T_cc.radius.md}px; outline: none;"
            " selection-background-color: #16323d;"
            f" selection-color: {p.text};"
            "}"
            "QTableWidget::item { border: none; padding: 3px 6px; }"
            "QHeaderView::section {"
            f" background: transparent; color: {p.text_muted};"
            f" border: 0; border-bottom: 1px solid {p.surface_border_strong};"
            " padding: 5px 6px; font-size: 10.5px; font-weight: 700; }"
        )
        self.table.itemSelectionChanged.connect(self._onTableRowSelected)
        return self.table

    def _refresh_table_cells(self):
        """Fill the read-only columns from each row's camera config, and the
        Ready column with what that box still needs.

        Resolution, rate and pixel format are echoes; they are chosen in the
        Camera options tab, but showing them per row is the whole point of
        having a table.
        """
        if not hasattr(self, "table"):
            return
        cctv = self._is_cctv_mode()
        for row in range(self.table.rowCount()):
            cam = self._camera_id_text(row)
            cfg = self._camera_config_for(cam) if cam else None
            res = getattr(cfg, "selected_resolution", None) if cfg else None
            # These are editable controls now, not echoes of a card elsewhere,
            # so the refresh fills them in rather than printing their values.
            self._fill_row_capture(row)
            btn = self.table.cellWidget(row, self.COL_DETECT)
            if btn is not None:
                btn.setEnabled(bool(cam))
            chips = self._ready_chips(row, cam, res, cctv)
            if self._ready_cache.get(row) == chips:
                continue                    # nothing this row says has changed
            self._ready_cache[row] = chips
            stale = self.table.cellWidget(row, self.COL_READY)
            if stale is not None:
                stale.setParent(None)       # else it paints over row 1
                stale.deleteLater()
            self.table.setCellWidget(row, self.COL_READY,
                                     self._ready_cell(chips))

    def _set_plain_cell(self, row, col, text, dim=False):
        item = self.table.item(row, col)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled
                          | QtCore.Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, col, item)
        item.setText(text)
        item.setForeground(QtGui.QColor("#7d858f" if dim else "#dfe4ec"))

    def _ready_cell(self, chips):
        """Render one row's chips. ``chips`` is the tuple _ready_chips built,
        which is also the cache key that decides whether this runs at all."""
        w = QtWidgets.QWidget()
        # No stylesheet here: an unqualified rule set on a cell widget is
        # inherited by its children, and would override the chip colours the
        # dialog defines. A plain QWidget already paints no background.
        w.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(4, 0, 4, 0)
        h.setSpacing(5)
        for text, kind in chips:
            h.addWidget(self._make_chip(text, kind))
        h.addStretch(1)
        return w

    def _ready_chips(self, row, cam, res, cctv):
        """What this row still needs, as ``((text, kind), ...)``.

        Pure and cheap, so it doubles as the cache key: identical output means
        the rendered cell would be identical and can be left alone.
        """
        chips = []
        if cam:
            chips.append(("camera", "ok"))
            if not res:
                chips.append(("detect", "warn"))
        else:
            chips.append(("no camera", "warn"))
        # A region is required in both arrangements, a dedicated camera still
        # needs one, because it fixes the frame the tracker and the pose model
        # actually see.
        roi = self._roi_for_row(row)
        if roi is not None:
            size = ""
            if res and isinstance(roi, (tuple, list)) and len(roi) == 4:
                rw = max(1, round(roi[2] * res[0]))
                rh = max(1, round(roi[3] * res[1]))
                size = f" {rw}\u00d7{rh}"
            chips.append(("ROI" + size, "ok"))
        else:
            chips.append(("ROI to draw", "warn"))
        return tuple(chips)

    # ----- right-pane Camera section (device + backend + record) -----

    def _selected_row(self):
        r = self.table.currentRow()
        if r is not None and r >= 0:
            return r
        return 0 if self.table.rowCount() else -1

    def _load_camera_section(self):
        """Fill the right Camera section from the selected row's hidden cells."""
        if not hasattr(self, "sel_device_combo"):
            return
        row = self._selected_row()
        if row < 0:
            return
        self._loading_camera_section = True
        try:
            self.sel_device_combo.setEditText(self._camera_id_text(row))
            bc = self.table.cellWidget(row, self.COL_BACKEND)
            if bc is not None:
                i = self.sel_backend_combo.findText(bc.currentText())
                self.sel_backend_combo.setCurrentIndex(i if i >= 0 else 0)
            if row < len(self.save_video_checkboxes):
                on = bool(self.save_video_checkboxes[row].isChecked())
                self.sel_record_check.setChecked(on)
                self.record_state_label.setText("On" if on else "Off")
        finally:
            self._loading_camera_section = False

    def _on_sel_device_changed(self, text):
        if getattr(self, "_loading_camera_section", False):
            return
        # CCTV: the one device picker drives every box.
        if getattr(self, "mode_cctv_btn", None) and self.mode_cctv_btn.isChecked():
            self.shared_cam_combo.blockSignals(True)
            self.shared_cam_combo.setEditText(text)
            self.shared_cam_combo.blockSignals(False)
            self._apply_shared_camera(text)
            return
        row = self._selected_row()
        if row < 0:
            return
        edit = self.table.cellWidget(row, self.COL_CAMERA_ID)
        if edit is not None and edit.currentText() != text:
            edit.setEditText(text)          # fires _on_row_camera_id_changed
        self._refresh_status_dots()

    def _on_sel_backend_changed(self, text):
        if getattr(self, "_loading_camera_section", False):
            return
        row = self._selected_row()
        if row < 0:
            return
        bc = self.table.cellWidget(row, self.COL_BACKEND)
        if bc is not None:
            i = bc.findText(text)
            if i >= 0 and i != bc.currentIndex():
                bc.setCurrentIndex(i)       # fires _onBackendChanged

    def _on_sel_record_changed(self, checked):
        if hasattr(self, "record_state_label"):
            self.record_state_label.setText("On" if checked else "Off")
        if getattr(self, "_loading_camera_section", False):
            return
        row = self._selected_row()
        if 0 <= row < len(self.save_video_checkboxes):
            self.save_video_checkboxes[row].setChecked(bool(checked))
        self._refresh_status_dots()

    # ----- per-row capture settings (Resolution + Target FPS in the table) --

    def _cameraIdForRow(self, row):
        """Camera id in ``row``'s Camera ID cell (int when numeric, str
        otherwise), or ``None`` when the cell is empty."""
        edit = self.table.cellWidget(row, self.COL_CAMERA_ID)
        if edit is None:
            return None
        # Editable combo (current form) exposes currentText; a plain line-edit
        # exposes text, tolerate both so old call sites keep working.
        if hasattr(edit, "currentText"):
            text = edit.currentText().strip()
        elif hasattr(edit, "text"):
            text = edit.text().strip()
        else:
            text = ""
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return text

    def _camera_id_text(self, row):
        """Stripped text of ``row``'s Camera ID cell, tolerating the editable
        combo (currentText) or a plain line-edit (text)."""
        edit = self.table.cellWidget(row, self.COL_CAMERA_ID)
        if edit is None:
            return ""
        if hasattr(edit, "currentText"):
            return edit.currentText().strip()
        if hasattr(edit, "text"):
            return edit.text().strip()
        return ""

    def _resolve_identity_cached(self, cam_id, backend):
        """``resolve_identity`` memoised per (camera_id, backend); this runs on
        every Camera-ID keystroke, and identity resolution enumerates USB
        devices, so it must not repeat needlessly."""
        cache = getattr(self, "_identity_cache", None)
        if cache is None:
            cache = self._identity_cache = {}
        k = (str(cam_id), str(backend))
        if k not in cache:
            try:
                from source.video.cameras.usb_identity import resolve_identity
                cache[k] = resolve_identity(cam_id, backend)
            except Exception:
                cache[k] = {}
        return cache[k]

    def _machine_variants_for_format(self, cam_id, cfg=None) -> dict:
        """Modes measured in the format this row is set to, or ``{}``."""
        try:
            from source.video.cameras import calibration_store as _store
            fmt = self._selected_format_for(cam_id)
            if not fmt and cfg is not None:
                fmt = str(getattr(cfg, "capture_format", "") or "").lower() or None
            if not fmt or fmt == "auto":
                return {}
            ident = self._resolve_identity_cached(
                cam_id, self._backendForCameraId(cam_id) or "opencv")
            return _store.get_variants_for_format(ident.get("unique_id"), fmt)
        except Exception as e:
            logger.debug("per-format modes for %s: %s", cam_id, e)
            return {}

    def _per_backend_modes(self, cam_id, cfg=None):
        """``[(w, h, fps, backend), ...]``, every mode every backend reached.

        One row per backend rather than the merged best, because the mode and
        the door that serves it are one choice. Falls back to the flat merged
        list (backend "") for a camera measured before per-backend data
        existed, so an old project still populates.
        """
        # The format the operator picked decides which sizes and rates
        # exist, so the table is read for THAT format. Falling back to the
        # format-blind table is only for a camera measured before Detect
        # recorded the format, and it is what made a YUY2 measurement show up
        # under MJPEG.
        variants = self._machine_variants_for_format(cam_id, cfg)
        if not variants:
            variants = self._machine_cached_variants(cam_id)
        if not variants and cfg is not None:
            variants = dict(getattr(cfg, "probed_variants", None) or {})
        rows = []
        for backend, modes in (variants or {}).items():
            for m in (modes or []):
                try:
                    rows.append((int(m[0]), int(m[1]), float(m[2]),
                                 str(backend)))
                except (TypeError, ValueError, IndexError):
                    continue
        if rows:
            # Which doors were TRIED, not which produced modes. A door that
            # was measured and came back empty cannot serve this camera, and
            # must stop being offered as something to try.
            return rows + self._untried_backend_rows(
                rows, tried=set(variants or {}))
        flat = self._machine_cached_modes(cam_id)
        if not flat and cfg is not None:
            flat = list(getattr(cfg, "probed_modes", None) or [])
        return [(int(m[0]), int(m[1]), float(m[2]), "")
                for m in (flat or []) if len(m) >= 3]

    @staticmethod
    def _untried_backend_rows(measured, tried=None):
        """Offer the backends this platform has but the probe never measured.

        A backend that fails to probe reports no modes, so it has no rows, so
        it cannot be chosen, and the one time an operator needs to force a
        backend by hand is exactly when its probe failed. Measured here:
        Media Foundation could not open the camera during Detect, DirectShow
        measured three modes, and the result was a list offering only the
        backend that caps this camera at 5 fps at 1080p.

        So each unmeasured backend gets ONE row, marked as untried rather than
        given a number nobody measured, at the size the list defaults to.
        Rate ``0.0`` sorts it below everything real.

        One row, not one per size. A row per size doubled the list: a camera
        with nine measured modes offered eighteen, half of them speculative,
        and the everyday job of picking a size got buried under a recovery
        action needed once. The cost of the short list is small, because
        ``begin_capturing`` already falls back through every backend when the
        pinned one will not open. Choosing an untried row PINS that door;
        it was never the only way to reach it.
        """
        from source.video.cameras.opencv import _uvc_backends
        try:
            available = set(_uvc_backends())
        except Exception:
            return []
        # ``tried`` counts a door that was probed and found nothing; without
        # it only doors that PRODUCED modes counted as tried, so a door that
        # had already failed kept being offered, every time, for ever.
        seen = {b for _w, _h, _f, b in measured if b}
        untried = available - (set(tried) if tried is not None else seen)
        if not untried or not seen:
            return []
        from source.video.framebus.types import pick_default_mode
        best = pick_default_mode([(w, h, f) for w, h, f, _b in measured])
        if best is None:
            return []
        return [(int(best[0]), int(best[1]), 0.0, backend)
                for backend in sorted(untried)]

    def _machine_cached_variants(self, cam_id):
        """Per-backend measurements for this camera from the per-PC store."""
        if cam_id is None:
            return {}
        backend = self._backendForCameraId(cam_id) or "opencv"
        try:
            from source.video.cameras import calibration_store as _cs
            ident = self._resolve_identity_cached(cam_id, backend)
            return dict(_cs.get_variants(ident.get("unique_id"),
                                         ident.get("identity")) or {})
        except Exception:
            return {}

    def _machine_cached_modes(self, cam_id):
        """``[(w, h, fps), ...]`` for this camera from the per-PC machine cache
        (the single source of truth for FPS/resolution calibration), or ``[]``.
        Keyed by the camera's USB unique_id, auto-loaded so a previously
        calibrated camera repopulates without pressing Calibrate."""
        if cam_id is None:
            return []
        backend = self._backendForCameraId(cam_id) or "opencv"
        try:
            from source.video.cameras import calibration_store as _cs
            ident = self._resolve_identity_cached(cam_id, backend)
            return list(_cs.get(ident.get("unique_id"), ident.get("identity")) or [])
        except Exception:
            return []

    @staticmethod
    def _canonical_camera_id(cam_id):
        """The identity of the camera this id refers to, when that is known.

        A project written before identities existed stores an index (``0``),
        while the dropdown offers identities (``fp3557d2de``). Left alone the
        two name the SAME camera differently: one physical device then appears
        twice in the connect selection and CCTV grouping splits it in half.

        Translation uses only an already-built live table, walking the bus
        takes seconds and no GUI path may block on it, so an id stays as it
        is until the cameras have been enumerated at least once.
        """
        s = str(cam_id).strip()
        if not s.isdigit():
            return cam_id
        try:
            from source.video.cameras.identity import live_cameras
            for entry in live_cameras(cached_only=True):
                if int(entry["index"]) == int(s):
                    return entry["id"]
        except Exception:
            pass
        return cam_id

    def _canonicalise_row_camera_ids(self):
        """Rewrite legacy numeric row ids to identities, once they are known.

        Called after the camera list has been built (the dropdown popup), so
        the rows stop disagreeing with the list the operator is picking from.
        """
        for r in range(self.table.rowCount()):
            edit = self.table.cellWidget(r, self.COL_CAMERA_ID)
            if edit is None:
                continue
            raw = edit.currentText().strip()
            canon = str(self._canonical_camera_id(raw))
            if canon != raw:
                edit.blockSignals(True)
                try:
                    edit.setEditText(canon)
                finally:
                    edit.blockSignals(False)
                logger.info("camera id %s in row %s is camera %s, using the "
                            "identity so it survives re-indexing", raw, r, canon)

    def _available_camera_ids(self):
        """Identifiers of cameras detected on this machine, cached for the
        dialog's lifetime. The identifier is the part before the ``-backend``
        suffix of each ``unique_id`` (e.g. ``"0"`` from ``"0-opencv"``, or a
        vendor serial), which is what the Camera ID cell stores."""
        cached = getattr(self, "_avail_cam_ids_cache", None)
        if cached is not None:
            return cached
        # Walking the bus takes ~10 s. On the GUI thread that is a freeze, so
        # the answer is "nothing yet" and a worker is started; when it lands,
        # camera_ids_ready refills every picker. Only a worker ever waits.
        import threading as _th
        if _th.current_thread() is _th.main_thread():
            if not getattr(self, "_enum_worker_running", False):
                self._enum_worker_running = True

                def _scan_off_thread():
                    try:
                        self._available_camera_ids()
                    finally:
                        self._enum_worker_running = False
                        try:
                            self.camera_ids_ready.emit()
                        except RuntimeError:
                            pass      # dialog closed while scanning
                _th.Thread(target=_scan_off_thread, name="camera-enum-lazy",
                           daemon=True).start()
            return []
        ids: list = []
        if not hasattr(self, "_cam_family"):
            self._cam_family = {}
        try:
            # Serialised and cached process-wide inside the factory, so the
            # prewarm thread and a dropdown click can't walk the bus at once.
            from source.video.cameras import CameraFactory
            for cam in CameraFactory.list_available_cameras():
                uid = str(cam.get("unique_id") or "").strip()
                if not uid:
                    continue
                ident = uid.rsplit("-", 1)[0] if "-" in uid else uid
                model = str(cam.get("model") or "").strip()
                # The index is where the camera is NOW, not what it is called.
                # Shown because an identity alone ("fp3557d2de") tells nobody
                # which of two physical cameras they are picking.
                idx = cam.get("index")
                if idx is not None:
                    model = f"{model} (index {idx})" if model else f"index {idx}"
                ids.append((ident, model))
                # The family travels in the unique_id's suffix and is the only
                # place it is knowable. The Backend cell used to carry it and
                # now carries the DOOR, so it has to be recorded here or a
                # scientific camera becomes indistinguishable from a UVC one.
                fam = uid.rsplit("-", 1)[1] if "-" in uid else "opencv"
                self._cam_family[ident] = fam
        except Exception as e:
            logger.debug("list_available_cameras failed: %s", e)
        # De-dup by identifier, preserve order.
        seen = set()
        out = []
        for ident, model in ids:
            if ident not in seen:
                seen.add(ident)
                out.append((ident, model))
        self._avail_cam_ids_cache = out
        return out

    def _prewarm_camera_ids(self):
        """Enumerate cameras on a worker thread at dialog open.

        Enumeration opens up to 8 device indices with a blocking read each,
        3–20 s on Windows. Running it here means the first Camera-ID dropdown
        click reads a warm cache instead of freezing the GUI. Touches no Qt
        objects; the result is a plain list on ``self``.
        """
        try:
            self.camera_ids_ready.connect(self._fill_all_camera_id_combos,
                                          QtCore.Qt.ConnectionType.UniqueConnection)
        except (TypeError, RuntimeError):
            pass          # already connected
        if getattr(self, "_avail_cam_ids_cache", None) is not None:
            self.camera_ids_ready.emit()
            return

        def _scan():
            try:
                self._available_camera_ids()
            finally:
                # Emitting from the worker is safe: a queued connection hands
                # the slot to the GUI thread, which is the only thread allowed
                # to touch the combos.
                self.camera_ids_ready.emit()

        threading.Thread(target=_scan, name="camera-enum-prewarm",
                         daemon=True).start()

    def _fill_all_camera_id_combos(self):
        """Fill every Camera-ID picker from the detected list, once it exists.

        Filled eagerly, not lazily on first popup: a dialog that has been open
        for a minute would otherwise still show an empty dropdown until it is
        clicked, with the rows holding whatever id the project happened to
        store.
        """
        for combo in self._camera_id_combos():
            try:
                combo._populated = True
                self._populate_camera_id_combo(combo)
            except RuntimeError:
                pass          # widget went away while the scan was running

    def _populate_camera_id_combo(self, combo):
        """Fill an editable Camera-ID combo with detected cameras, preserving
        whatever the user has already typed. The visible item text is the bare
        identifier (so ``currentText()`` stays parseable as the camera id); the
        camera model, if any, goes in the item tooltip."""
        current = combo.currentText()
        combo.blockSignals(True)
        try:
            combo.clear()
            found = self._available_camera_ids()
            if not found and getattr(self, "_enum_worker_running", False):
                # Walking the USB bus takes about 5 s per camera, 13 s for
                # three on this rig, and it runs off the GUI thread so the
                # dialog stays alive. Until it lands this list is EMPTY, and
                # an empty dropdown with no explanation reads as a broken
                # dialog rather than a slow one. Say which it is.
                combo.addItem("Scanning for cameras…", None)
                item = combo.model().item(0)
                if item is not None:
                    item.setEnabled(False)
                combo.setEditText(current)
                return
            for ident, model in found:
                # Display a friendly "id, model"; keep the bare id in itemData
                # so _CameraIdCombo can restore it as the (parseable) edit text.
                label = f"{ident}, {model}" if model else ident
                combo.addItem(label, ident)
                if model:
                    combo.setItemData(combo.count() - 1, label,
                                      QtCore.Qt.ItemDataRole.ToolTipRole)
            combo.setEditText(current)
        finally:
            combo.blockSignals(False)
        # The list is now known, so legacy numeric rows can be named the same
        # way the list names them.
        try:
            self._canonicalise_row_camera_ids()
        except Exception as e:
            logger.debug("canonicalise row camera ids: %s", e)

    def _rowForCameraId(self, camera_id):
        """Row index whose Camera ID matches ``camera_id``, or ``None``."""
        target = str(camera_id)
        for row in range(self.table.rowCount()):
            if str(self._cameraIdForRow(row)) == target:
                return row
        return None

    # ----- Capture Settings card (per active camera) -----------------------

    def _active_capture_cam(self):
        """Camera id the Capture / Scientific cards target (the Active-camera
        picker), int when numeric, else str, or None."""
        combo = getattr(self, "active_cam_combo", None)
        txt = combo.currentText().strip() if combo is not None else ""
        if not txt:
            return None
        try:
            return int(txt)
        except ValueError:
            return txt

    def _refresh_active_cam_combo(self):
        """Repopulate the Active-camera picker from the table's camera ids,
        preserving the current pick. Guarded so the fill never fires reloads.

        Returns True when the pick could NOT be preserved and the picker
        therefore now points at a different camera, the caller must reload
        the Capture card, or it will keep displaying the old camera while
        edits are written to the new one.
        """
        combo = getattr(self, "active_cam_combo", None)
        if combo is None:
            return
        ids = [str(c) for c in self._allCameraIdsInTable()]
        prev = combo.currentText().strip()
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItems(ids)
            if prev in ids:
                combo.setCurrentIndex(ids.index(prev))
            elif ids:
                # Falling back to index 0 silently re-points the Capture card
                # at a DIFFERENT box's camera. The picker is hidden, so nothing
                # on screen shows the switch, and the next Resolution/FPS edit
                # is then written onto that camera. Report the change instead
                # of swallowing it, so the caller can reload the card.
                combo.setCurrentIndex(0)
        finally:
            combo.blockSignals(False)
        return combo.currentText().strip() != prev

    def _onTableRowSelected(self):
        """Selecting a box loads its camera into the right-pane Camera section
        and makes it the active camera (reloads Capture + Advanced)."""
        self._load_camera_section()
        row = self.table.currentRow()
        cid = self._cameraIdForRow(row) if row is not None and row >= 0 else None
        combo = getattr(self, "active_cam_combo", None)
        if cid is not None and combo is not None:
            i = combo.findText(str(cid))
            if i >= 0 and i != combo.currentIndex():
                combo.setCurrentIndex(i)   # fires _on_active_cam_changed
                return
        self._load_active_camera_settings()

    # ── per-row capture controls ─────────────────────────────────────────
    # Resolution, rate, format and orientation are properties of a CAMERA,
    # and ``update_camera_config`` has always keyed them that way. What was
    # missing was a way to reach more than one camera at a time: the shared
    # card only ever edited whichever row was selected, so on a rig with one
    # camera per box every camera had to be selected in turn, and nothing on
    # screen showed what the others were set to.
    #
    # These write through exactly the same per-camera call, so nothing new
    # reaches the project file; the values already had a home in
    # ``cameras.registry[].preference``.

    def _row_capture_widgets(self, row):
        return (self.table.cellWidget(row, self.COL_RES),
                self.table.cellWidget(row, self.COL_FPS),
                self.table.cellWidget(row, self.COL_FORMAT),
                self.table.cellWidget(row, self.COL_FLIP))

    def _rows_sharing_camera(self, cam_id):
        """Every row bound to this camera. On a CCTV rig four boxes share one
        camera, so a change made on one row has to show on the others: they
        are one camera, not four."""
        if not cam_id:
            return []
        return [r for r in range(self.table.rowCount())
                if self._camera_id_text(r) == str(cam_id)]

    #: The dialog theme pads a combo ``2px 30px 2px 12px`` around a 24 px
    #: drop-down, 68 px of chrome before a character is drawn. That is right
    #: for a form, but in a table cell it left 20 px for the text and Qt
    #: ELIDES rather than shrinks: "MJPEG" painted as "M" and a camera id as
    #: "cbb52d3", each over a perfectly correct config. Scoped to the cells,
    #: so every other dialog keeps the roomier form styling.
    _CELL_COMBO_QSS = (
        "QComboBox { padding: 2px 22px 2px 6px; }"
        "QComboBox::drop-down { width: 18px; }"
        "QComboBox::down-arrow { margin-right: 5px; }")

    def _build_row_capture_cells(self, row):
        """Resolution / FPS / Format / Flip for one row."""
        res = QtWidgets.QComboBox()
        res.setToolTip("Resolution for this row's camera.")
        res.currentIndexChanged.connect(
            lambda _i, r=row: self._on_row_resolution_changed(r))
        self.table.setCellWidget(row, self.COL_RES, res)

        fps = QtWidgets.QComboBox()
        fps.setToolTip("Frame rate this camera really delivers at the "
                       "selected resolution.")
        fps.currentIndexChanged.connect(
            lambda _i, r=row: self._on_row_capture_changed(r))
        self.table.setCellWidget(row, self.COL_FPS, fps)

        fmt = QtWidgets.QComboBox()
        # "Any" stored "", which every reader treats as unset and replaces
        # with mjpeg, here and in the capture backend both, so the entry
        # asked for the driver's own choice and delivered MJPEG. It is now
        # "auto" and means what it says: the FOURCC is left alone. That is
        # the escape hatch for a camera whose driver refuses a forced format.
        for label, data in (("MJPEG", "mjpeg"), ("YUY2", "yuy2"),
                            ("H264", "h264"), ("Auto", "auto")):
            fmt.addItem(label, data)
        fmt.setToolTip("Transport format. MJPEG is what lets a USB camera "
                       "reach its full rate at higher resolutions. Auto "
                       "leaves the driver's own choice untouched.")
        fmt.currentIndexChanged.connect(
            lambda _i, r=row: self._on_row_capture_changed(r))
        self.table.setCellWidget(row, self.COL_FORMAT, fmt)

        for w in (res, fps, fmt):
            w.setStyleSheet(self._CELL_COMBO_QSS)

        flip = QtWidgets.QWidget()
        fl = QtWidgets.QHBoxLayout(flip)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(2)
        fl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        for name, tip in (("H", "Flip horizontally"),
                          ("V", "Flip vertically")):
            b = QtWidgets.QToolButton()
            b.setText(name)
            b.setCheckable(True)
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            b.setToolTip(f"{tip}. Applies to the recording, not just the "
                         "preview.")
            b.setStyleSheet(
                "QToolButton { color:#cfd6e2; background:rgba(255,255,255,0.05);"
                " border:1px solid #39404d; border-radius:4px;"
                " padding:1px 6px; font-size:10.5px; }"
                "QToolButton:checked { color:#0d1117; background:#6fb6cd;"
                " border-color:#6fb6cd; font-weight:700; }")
            b.toggled.connect(lambda _c, r=row: self._on_row_capture_changed(r))
            fl.addWidget(b)
            setattr(flip, f"flip_{name.lower()}_btn", b)
        self.table.setCellWidget(row, self.COL_FLIP, flip)

    def _fill_row_capture(self, row):
        """Show what this row's camera is set to, from its own config.

        Called from ``_refresh_table_cells``, which a 120 ms timer fires after
        anything the operator touches. Clearing and repopulating the combos on
        every one of those made them unusable: the list was rebuilt under the
        open dropdown, and a selection made a moment earlier was replaced by
        whatever the config still said. So this rebuilds only when the OPTIONS
        actually changed, and never while a dropdown is open.
        """
        res, fps, fmt, flip = self._row_capture_widgets(row)
        if res is None:
            return
        for w in (res, fps, fmt):
            view = w.view() if w is not None else None
            if view is not None and view.isVisible():
                return          # the operator is choosing from this list
        # Held for the WHOLE body. This method only displays; nothing it does
        # to a widget may be mistaken for the operator doing it. Blocking the
        # combos is not enough on its own, because the populators this calls
        # manage their own blocking.
        was_loading = getattr(self, "_loading_capture", False)
        self._loading_capture = True
        try:
            self._fill_row_capture_inner(row, res, fps, fmt, flip)
        finally:
            self._loading_capture = was_loading

    def _fill_row_capture_inner(self, row, res, fps, fmt, flip):
        """The body of ``_fill_row_capture``, under its re-entrancy latch."""
        cam = self._camera_id_text(row)
        cfg = self._camera_config_for(cam) if cam else None

        # The DOOR first, because everything below it is filtered by it. It
        # comes from the project when the project names one, and from the
        # platform default when it does not: an older project saved before
        # the door was a choice still has to land on something, and that
        # something is written back the next time the project is saved.
        door_combo = self.table.cellWidget(row, self.COL_BACKEND)
        if door_combo is not None:
            saved_door = str(getattr(cfg, "capture_backend", "") or "")
            self._fill_door_combo(door_combo, cam_id=cam, selected=saved_door)
            # A project saved before the door was a choice carries none. It is
            # given the platform default AND that default is written into the
            # config, so the project records which door its numbers were
            # measured through the next time it is saved. Displaying it
            # without storing it left the project silent about the one setting
            # that decides what the camera delivers.
            if cam and cfg is not None and not saved_door:
                picked = self._doorForRow(row)
                if picked:
                    self._store_backfilled_door(cam, picked)
        door = self._doorForRow(row)

        modes = self._per_backend_modes(cam, cfg) if cam else []
        # Only the modes this door measured. A row that lists another door's
        # sizes is offering a capability the camera does not have through the
        # door it is about to be opened with.
        if door and modes:
            through = [m for m in modes if not m[3] or str(m[3]) == door]
            if through:
                modes = through

        # Everything this row would display. Unchanged means there is nothing
        # to redraw, and redrawing anyway is what stole the selection.
        sig = (str(cam or ""), tuple(tuple(m) for m in (modes or ())),
               tuple(getattr(cfg, "selected_resolution", None) or ()),
               # The resolution SHOWN, not only the one stored: the rate list
               # is built for it, so leaving it out let the guard skip the
               # rebuild after a resolution change and the FPS column kept
               # the previous resolution's ladder.
               self._row_size(res),
               # The door, both stored and shown. Two entries can carry the
               # same size, so without this the guard skipped the rebuild
               # after a backend change and the row kept the other door's
               # rate ladder.
               str(getattr(cfg, "capture_backend", "") or ""),
               door,
               self._row_backend(res),
               getattr(cfg, "selected_fps", None),
               str(getattr(cfg, "capture_format", "") or ""),
               bool(getattr(cfg, "flip_horizontal", False)),
               bool(getattr(cfg, "flip_vertical", False)))
        if not hasattr(self, "_row_capture_sig"):
            self._row_capture_sig = {}
        if self._row_capture_sig.get(row) == sig:
            return
        self._row_capture_sig[row] = sig

        for w in (res, fps, fmt):
            w.blockSignals(True)
        try:
            wh = getattr(cfg, "selected_resolution", None) if cfg else None
            want_wh = (int(wh[0]), int(wh[1])) if wh else None
            if modes:
                # The SAME populator the shared card uses, pointed at this
                # row's two combos. It fills the resolution list, selects the
                # saved size, and then builds the rate ladder for whatever
                # ended up selected, so the row and the card cannot disagree.
                self._populateResolutionCombo(
                    modes, preselect_wh=want_wh, res_combo=res, fps_combo=fps,
                    compact=True,
                    preselect_backend=str(
                        getattr(cfg, "capture_backend", "") or "") or None,
                    cam_id=cam)
            else:
                res.clear()
                res.addItem("Detect →", None)
                fps.clear()
                fps.addItem("-", None)
                fps.setEnabled(False)
            res.setEnabled(bool(cam) and bool(modes))

            # Prefer the saved rate when the selected resolution still reaches
            # it; otherwise keep the ladder's own default. Leaving the combo
            # wherever it landed while the config said something else is what
            # let a row read 5 fps over a camera configured for 30.
            saved_fps = getattr(cfg, "selected_fps", None) if cfg else None
            offered = {int(fps.itemData(i)): i for i in range(fps.count())
                       if fps.itemData(i) is not None}
            want = int(saved_fps) if saved_fps is not None else None
            if want in offered:
                fps.setCurrentIndex(offered[want])
            elif offered:
                want = int(fps.currentData()) if fps.currentData() else None
            if (want is not None and saved_fps is not None
                    and int(saved_fps) != want and cam):
                # The stored rate is not one this resolution can deliver, so
                # the row had to show a different number. Correct the config
                # rather than leave the two disagreeing: a row reading 5 fps
                # over a camera configured for 30 is the contradiction, and
                # whichever the operator believed, one of them was wrong.
                self._correct_stored_fps(cam, want, int(saved_fps))

            # ``or "mjpeg"``, the same default the shared card uses. Falling
            # back to "" instead selected the "Any" entry, so a camera with
            # no format saved opened on Any here and on MJPEG in CCTV: the
            # two surfaces disagreed about the same camera, and MJPEG is what
            # lets a USB camera reach its rate at the higher resolutions.
            saved_fmt = str(getattr(cfg, "capture_format", None) or "mjpeg").lower()
            i = fmt.findData(saved_fmt)
            fmt.setCurrentIndex(i if i >= 0 else 0)
            fmt.setEnabled(bool(cam))
        finally:
            for w in (res, fps, fmt):
                w.blockSignals(False)

        for attr, field in (("flip_h_btn", "flip_horizontal"),
                            ("flip_v_btn", "flip_vertical")):
            b = getattr(flip, attr, None)
            if b is None:
                continue
            b.blockSignals(True)
            try:
                b.setChecked(bool(getattr(cfg, field, False)) if cfg else False)
                b.setEnabled(bool(cam))
            finally:
                b.blockSignals(False)

    def _row_capture_fields(self, row, cam=None):
        """This row's picks as CameraConfig fields, or ``None`` if it has none.

        One reader for two callers. The row is the per-box control, so it is
        what an edit writes AND what Connect persists; when only the edit read
        it, a camera whose row the operator never touched reached Connect with
        ``selected_resolution=None`` while its row plainly showed a size.
        """
        res, fps, fmt, flip = self._row_capture_widgets(row)
        if res is None:
            return None
        wh = self._row_size(res)
        if not wh:
            return None
        cam = cam if cam is not None else self._camera_id_text(row)
        # The rate list belongs to a RESOLUTION, and the backend travels in
        # the resolution's own item data, so both are read from the selection
        # that is on screen. Writing a rate the previous resolution offered is
        # how 9 fps (the 1080p ceiling) got stored against a 720p mode the
        # camera runs at 21; a resolution change rebuilds the ladder first.
        d = res.currentData()
        rate = fps.currentData() if fps is not None else None
        fields = {
            "selected_resolution": (int(wh[0]), int(wh[1])),
            "selected_fps": (int(rate) if rate is not None else None),
            # THE OPERATOR'S DOOR, read from THIS ROW's own cell. It used
            # to be read out of whichever resolution row happened to be
            # selected, so the door was a side effect of picking a size
            # rather than a choice of its own. Looking it up by camera id
            # would be wrong too: a shared CCTV camera is on several rows, so
            # every row would answer with the first row's door and editing
            # the second one would appear to do nothing.
            "capture_backend": (self._doorForRow(row)
                                or str((d or {}).get("backend", "") or "")),
            "capture_format": (fmt.currentData() if fmt is not None else None),
            "camera_backend": self._backendForCameraId(cam) or "opencv",
            "user_applied": True,
        }
        for attr, field in (("flip_h_btn", "flip_horizontal"),
                            ("flip_v_btn", "flip_vertical")):
            b = getattr(flip, attr, None) if flip is not None else None
            if b is not None:
                fields[field] = bool(b.isChecked())
        return fields

    @staticmethod
    def _row_backend(res_combo):
        """The backend the row's resolution combo is showing, or ``""``."""
        d = res_combo.currentData() if res_combo is not None else None
        return str((d or {}).get("backend", "") or "") if isinstance(d, dict) else ""

    @staticmethod
    def _row_size(res_combo):
        """The ``(w, h)`` a row's resolution combo is showing, or ``()``.

        Item data is the same ``{"width", "height", "fps", "backend"}`` dict
        the shared card stores, so this is the one place that unpacks it.
        """
        d = res_combo.currentData() if res_combo is not None else None
        if not isinstance(d, dict):
            return ()
        try:
            return (int(d["width"]), int(d["height"]))
        except (KeyError, TypeError, ValueError):
            return ()

    def _correct_stored_fps(self, cam, want, was):
        """Write back a rate the selected resolution can actually deliver.

        The per-PC machine cache of real measurements outranks whatever a
        config carries, so a project saved when a camera was on a different
        mode, or measured on another machine, can hold a rate this resolution
        never reaches. Showing the honest number and leaving the stale one
        stored means the row and the rig disagree until something else
        happens to write.
        """
        if self.main_window is None:
            return
        pipe = getattr(self.main_window, "pipeline", None)
        if pipe is None:
            return
        try:
            pipe.update_camera_config(str(cam), selected_fps=int(want))
        except Exception as e:
            logger.debug("camera %s: could not correct the stored rate: %s",
                         cam, e)
            return
        logger.info("camera %s: stored rate %s fps is not offered at the "
                    "selected resolution; corrected to %s fps, which is what "
                    "it measured.", cam, was, want)

    def _on_row_resolution_changed(self, row):
        """A new resolution means a new rate ceiling, so the ladder is rebuilt
        before anything reads it, exactly as ``_onCaptureResolutionChanged``
        does for the shared card."""
        if getattr(self, "_loading_capture", False):
            return
        res, fps, _fmt, _flip = self._row_capture_widgets(row)
        self._repopulate_fps_combo(res_combo=res, fps_combo=fps,
                                   cam_id=self._camera_id_text(row))
        self._on_row_capture_changed(row)

    def _on_row_capture_changed(self, row):
        """Write this row's choices onto ITS camera, then mirror the change
        onto any other row bound to the same camera."""
        if getattr(self, "_loading_capture", False):
            return
        cam = self._camera_id_text(row)
        if not cam or self.main_window is None:
            return
        pipe = getattr(self.main_window, "pipeline", None)
        if pipe is None:
            return
        fields = self._row_capture_fields(row, cam)
        if fields is None:
            return
        try:
            pipe.update_camera_config(str(cam), **fields)
        except Exception as e:
            logger.warning("camera %s: could not apply the row's capture "
                           "settings: %s", cam, e)
            return
        self._loading_capture = True
        try:
            for r in self._rows_sharing_camera(cam):
                if r != row:
                    self._fill_row_capture(r)
        finally:
            self._loading_capture = False
        # The rate list depends on the resolution, so refresh this row too.
        self._fill_row_capture(row)
        self._refresh_marks()
        with contextlib.suppress(Exception):
            self.main_window._project_changed(reason="camera_config_changed")

    def _onDetectRow(self, row):
        """Probe one row's camera. Selecting the row first keeps the Camera
        options tab pointed at the camera being measured."""
        if row < 0 or row >= self.table.rowCount():
            return
        self.table.setCurrentCell(row, self.COL_BOX)
        self._onRedetectActive()
        self._refresh_marks()

    def _on_row_record_toggled(self, _checked=False):
        """A Record toggle changed: keep the hidden proxy and the Connect
        tab's record summary in step.

        The toggle itself is the model, so only the proxy needs syncing, and
        only when the row that changed is the one the proxy mirrors.
        """
        row = self._selected_row()
        sender = self.sender()
        if (sender in self.save_video_checkboxes
                and self.save_video_checkboxes.index(sender) != row):
            self._refresh_record_summary()
            self._queue_marks()
            return
        if 0 <= row < len(self.save_video_checkboxes) \
                and hasattr(self, "sel_record_check"):
            on = bool(self.save_video_checkboxes[row].isChecked())
            self.sel_record_check.blockSignals(True)
            try:
                self.sel_record_check.setChecked(on)
            finally:
                self.sel_record_check.blockSignals(False)
            self.record_state_label.setText("On" if on else "Off")
        self._refresh_record_summary()
        self._queue_marks()

    def _on_row_camera_id_changed(self, row):
        """A Camera-ID cell changed: keep the Active-camera picker in sync, and
        reload the card if the edited row is the active camera.

        Fires once per keystroke on an editable combo, including the empty
        string you pass through when clearing the cell to retype. Both facts
        matter below.
        """
        edited = self._cameraIdForRow(row)
        active_changed = self._refresh_active_cam_combo()
        active = self._active_capture_cam()

        # Mid-edit the cell is empty, so both ids are None and a plain
        # ``str(a) == str(b)`` compares "None" to "None" and passes, which
        # reloaded the Capture card for "no camera" and blanked the operator's
        # resolution/FPS picks while they were still typing. An absent id
        # matches nothing.
        same_camera = (edited is not None and active is not None
                       and str(edited) == str(active))

        # Reload when this row IS the active camera, and also when the picker
        # silently moved to a different one, otherwise the card keeps showing
        # the old camera's settings while writes land on the new one.
        if same_camera or active_changed:
            self._load_active_camera_settings()

        # Changing a connected box's camera makes Connect relevant again.
        # Without this the button keeps whatever state it had when the box
        # was connected, i.e. disabled, and the edit can never be applied.
        self.updateButtonStates()
        self._write_camera_id_to_box_widget(row)
        self._queue_marks()

    def _write_camera_id_to_box_widget(self, row):
        """Mirror the row's Camera ID onto its box widget, and mark dirty.

        The table cell and ``setup_widget.camera_id_edit`` are two records of
        the same fact, and the widget is the authoritative one: it is what
        ``connect_camera`` reads and what the project save persists.
        Writing it only on a successful Connect would silently discard an
        edited ID on close or Disconnect, leaving the old id in the
        project file.
        """
        widgets = self._row_box_widgets()
        if row >= len(widgets):
            return
        edit = getattr(widgets[row], "camera_id_edit", None)
        if edit is None:
            return
        text = self._camera_id_text(row)
        try:
            if edit.text().strip() == text:
                return
            edit.setText(text)
        except (AttributeError, RuntimeError) as e:
            logger.debug("camera id write-back for row %s: %s", row, e)
            return
        mark = getattr(self.main_window, "_mark_project_dirty", None)
        if callable(mark):
            try:
                mark(reason="camera_id")
            except Exception as e:
                logger.debug("mark dirty after camera id change: %s", e)

    def _on_active_cam_changed(self, *_):
        self._load_active_camera_settings()

    def _load_active_camera_settings(self):
        """Load the active camera's Capture + Scientific settings into the cards
        and reveal the Advanced group iff the active camera is scientific."""
        self._load_capture_for(self._active_capture_cam())
        self._loadSciForRow()
        # The feature query is the expensive part and it decides the group's
        # visibility, so both are deferred together.
        self._queue_feature_panel()

    def _load_capture_for(self, cam_id):
        """Fill the Capture card from ``cam_id``'s modes (per-PC machine cache
        first, else the project config) and its saved selection. The machine
        cache is the source of truth; a project's ``probed_modes`` is only a
        transitional display fallback for cameras never calibrated on this PC."""
        res_combo = getattr(self, "resolution_combo", None)
        fps_combo = getattr(self, "fps_combo", None)
        if res_combo is None or fps_combo is None:
            return
        self._loading_capture = True
        try:
            cfg = None
            if cam_id is not None and self.main_window is not None:
                pipe = getattr(self.main_window, "pipeline", None)
                if pipe is not None:
                    try:
                        cfg = pipe.all_camera_configs().get(str(cam_id))
                    except Exception:
                        cfg = None
            modes = self._per_backend_modes(cam_id, cfg)
            if modes:
                self._populateResolutionCombo(
                    modes, preselect_wh=getattr(cfg, "selected_resolution", None),
                    cam_id=cam_id)
                saved_fps = getattr(cfg, "selected_fps", None)
                if saved_fps is not None:
                    i = fps_combo.findData(int(saved_fps))
                    if i >= 0:
                        fps_combo.setCurrentIndex(i)
            else:
                res_combo.blockSignals(True)
                try:
                    res_combo.clear()
                    res_combo.addItem("Detect to populate", None)
                finally:
                    res_combo.blockSignals(False)
                fps_combo.blockSignals(True)
                try:
                    fps_combo.clear()
                    fps_combo.setEnabled(False)
                finally:
                    fps_combo.blockSignals(False)
            fmt = getattr(self, "format_combo", None)
            if fmt is not None:
                saved_fmt = (getattr(cfg, "capture_format", None) or "mjpeg")
                i = fmt.findData(str(saved_fmt).lower())
                fmt.blockSignals(True)
                try:
                    fmt.setCurrentIndex(i if i >= 0 else 0)
                finally:
                    fmt.blockSignals(False)
            # Orientation is per camera, so switching rows has to show THIS
            # camera's setting rather than leave the previous one ticked.
            for attr, field in (("flip_h_check", "flip_horizontal"),
                                ("flip_v_check", "flip_vertical")):
                cb = getattr(self, attr, None)
                if cb is None:
                    continue
                cb.blockSignals(True)
                try:
                    cb.setChecked(bool(getattr(cfg, field, False)))
                finally:
                    cb.blockSignals(False)
        finally:
            self._loading_capture = False

    def _write_capture_to(self, cam_id):
        """Persist the Capture card's current Resolution / FPS / Format +
        probed modes onto ``cam_id``'s CameraConfig. No-op until a real
        resolution is picked."""
        if cam_id is None or self.main_window is None:
            return
        pipe = getattr(self.main_window, "pipeline", None)
        if pipe is None:
            return
        res = self.getSelectedResolution()
        if res is None:
            return
        # The backend travels with the mode the operator picked, same row,
        # same decision.
        picked = (self.resolution_combo.currentData()
                  if getattr(self, "resolution_combo", None) is not None else None)
        chosen_backend = str((picked or {}).get("backend", "") or "")
        fmt = (self.format_combo.currentData()
               if getattr(self, "format_combo", None) is not None else None)
        try:
            pipe.update_camera_config(
                str(cam_id),
                selected_resolution=res,
                selected_fps=self.getSelectedFPS() or None,
                probed_modes=self._captureProbedModesFromCombo(),
                frame_strategy="accept",
                camera_backend=self._backendForCameraId(cam_id) or "opencv",
                capture_format=fmt,
                capture_backend=chosen_backend,
            )
        except Exception as e:
            logger.debug("write capture for %s: %s", cam_id, e)

    def _ensure_capture_defaults(self, cam_id):
        """Make sure ``cam_id``'s config has a resolution + FPS derived from its
        OWN modes (top resolution + rounded ceiling), WITHOUT touching the
        card, so a camera the user never opened in the picker still connects."""
        if cam_id is None or self.main_window is None:
            return
        pipe = getattr(self.main_window, "pipeline", None)
        if pipe is None:
            return
        cfg = None
        try:
            cfg = pipe.all_camera_configs().get(str(cam_id))
        except Exception:
            cfg = None
        modes = self._machine_cached_modes(cam_id)
        if not modes and cfg:
            modes = list(getattr(cfg, "probed_modes", None) or [])
        if not modes:
            return
        from source.video.framebus.types import default_fps_for, pick_default_mode
        sel = getattr(cfg, "selected_resolution", None) if cfg else None
        valid = sel and any(
            (int(m[0]), int(m[1])) == tuple(sel) for m in modes)
        if not valid:
            best = pick_default_mode(modes)
            if best is None:
                return
            sel = (best[0], best[1])
        ceiling = next((float(m[2]) for m in modes
                        if (int(m[0]), int(m[1])) == tuple(sel)), 0.0)
        fps = getattr(cfg, "selected_fps", None) if cfg else None
        if not fps and ceiling > 0:
            fps = default_fps_for(ceiling)
        try:
            pipe.update_camera_config(
                str(cam_id), probed_modes=modes,
                selected_resolution=sel, selected_fps=fps or None)
        except Exception as e:
            logger.debug("ensure capture defaults %s: %s", cam_id, e)

    def _onCaptureResolutionChanged(self, *_):
        if getattr(self, "_loading_capture", False):
            return
        self._repopulate_fps_combo()
        self._write_capture_to(self._active_capture_cam())

    def _onCaptureFpsChanged(self, *_):
        if getattr(self, "_loading_capture", False):
            return
        self._write_capture_to(self._active_capture_cam())

    def _onCaptureFormatChanged(self, *_):
        if getattr(self, "_loading_capture", False):
            return
        self._write_capture_to(self._active_capture_cam())

    def _onRedetectActive(self):
        """Re-probe the active camera's resolutions/FPS, then reload the card."""
        cid = self._active_capture_cam()
        if cid is None:
            return
        self._probe_cameras([cid])
        self._load_capture_for(cid)

    # ----- persistence row (Save/Load per-camera + Save All/Load All bulk) -----

    def _connect_row_for_calibration(self, row) -> bool:
        """Connect ONLY the camera in ``row`` through the normal Connect
        sub-steps (push selections → preflight → ROI → segment → connect), but
        WITHOUT the final accept() so this dialog stays open. Used by Calibrate
        Lens so the user doesn't have to press Connect (which closes the dialog)
        first. Returns True when the camera ends up streaming."""
        cid = self._cameraIdForRow(row)
        if cid is None:
            return False
        sel = self.getSelectedBoxesWithCameras()
        items = [s for s in sel if str(s.get('camera_id')) == str(cid)]
        if not items:
            return False
        camera_id_map = {cid: items}
        try:
            self._push_dialog_selections_to_pipeline(camera_id_map)
            if not self._preflight_resolution_and_fps(camera_id_map):
                return False
            if not self._prompt_roi_for_pending_cameras(camera_id_map):
                return False
            self._build_and_publish_segment_config(camera_id_map)
            successful, _failed = self._connect_each_camera(camera_id_map)
        except Exception as e:
            logger.error("auto-connect for lens calibration failed (%s): %s", cid, e)
            return False
        return successful > 0

    def _onCalibrateLens(self, camera_id=None, box_number=None):
        """Open the checkerboard wizard for one target.

        Streams raw frames from the camera's live FrameBus with correction
        toggled off, so the wizard sees the true distorted image, and
        activates the saved profile on close. Auto-connects the camera first
        when it isn't already streaming.

        ``camera_id`` defaults to the selected row's camera. Returns ``False``
        when the run did not happen (nothing to calibrate, or cancelled), so
        the calibrate-everything loop can stop.
        """
        from source.gui.dialogs.lens_calibration import LensCalibrationDialog
        from source.video.cameras.lens import LensCalibrationStore
        from source.video.cameras.usb_identity import resolve_identity

        pipe = getattr(self.main_window, "pipeline", None) if self.main_window else None
        row = self.table.currentRow()
        if row < 0 and self.table.rowCount() == 1:
            row = 0
        cid = camera_id if camera_id not in (None, "") else (
            self._cameraIdForRow(row) if row >= 0 else None)
        if pipe is None or cid is None:
            QtWidgets.QMessageBox.information(
                self, "Calibrate lens", "Select a camera row first.")
            return False
        target_row = self._rowForCameraId(cid)
        bus = pipe.get_bus(cid)
        if bus is None:
            # Auto-connect this camera first (same flow as Connect) so the
            # wizard has a live stream, without closing this dialog.
            if target_row is None or \
                    not self._connect_row_for_calibration(target_row):
                QtWidgets.QMessageBox.information(
                    self, "Calibrate lens",
                    "Couldn't connect this camera automatically. Check its "
                    "Camera ID / resolution, then try again.")
                return False
            bus = pipe.get_bus(cid)
        if bus is None:
            QtWidgets.QMessageBox.information(
                self, "Calibrate lens",
                "This camera isn't streaming yet, connect it, then calibrate.")
            return False

        backend = self._backendForCameraId(cid) or "opencv"
        ident = resolve_identity(cid, backend)
        key = self._lens_key_for(cid, box_number)
        if ident.get("weak"):
            warned = getattr(self, "_weak_lens_warned", set())
            if cid not in warned:
                QtWidgets.QMessageBox.warning(
                    self, "Calibrate lens",
                    "This camera has no stable USB identity. Its calibration is "
                    "saved under a per-PC fallback key (the camera index); if you "
                    "move it to a different USB port you may need to re-calibrate.")
                warned.add(cid)
                self._weak_lens_warned = warned
        friendly = (ident.get("identity") or {}).get("name") or f"Camera {cid}"
        if box_number is not None:
            friendly = f"{friendly} \u00b7 Box {box_number}"

        latest = {"frame": None}
        # A shared camera's boxes each look through a different crop, and a
        # crop has its own principal point and field of view, solving the
        # checkerboard on the whole frame and filing it under a per-box key
        # would store N profiles that fit no box. Feed the wizard the box's
        # own region so the intrinsics it solves are the ones that region
        # actually needs.
        crop = self._lens_crop_rect(cid, box_number)
        if crop is None:
            def _take(cf):
                latest["frame"] = cf.image
        else:
            def _take(cf, _r=crop):
                img = cf.image
                if img is None:
                    return
                h, w = img.shape[:2]
                x0, y0, x1, y1 = (int(round(_r[0] * w)), int(round(_r[1] * h)),
                                  int(round(_r[2] * w)), int(round(_r[3] * h)))
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                latest["frame"] = img[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else img
        unsub = bus.on_camera_frame(_take)
        # See raw frames while calibrating, even if a profile is already active.
        pipe.set_lens_correction_enabled(False)
        try:
            dlg = LensCalibrationDialog(
                lambda: latest["frame"], key, friendly,
                LensCalibrationStore(), parent=self)
            dlg.exec()
        finally:
            try:
                unsub()
            except Exception:
                pass
            pipe.set_lens_correction_enabled(True)
            pipe.refresh_lens_correction(cid)  # activate a just-saved profile
            self._invalidate_lens_cache()      # re-read the just-saved profile
            self._refresh_lens_list()
            self._refresh_marks()
        return True

    # ── Multi-camera calibration ──────────────────────────────────────

    def _apply_calibration_modes(self, camera_id, modes,
                                 by_backend: dict | None = None) -> None:
        """Install probed/loaded modes into the pipeline + refresh that
        camera's table row.

        Writes only capabilities (``probed_modes``), never the user's target
        FPS. Repopulates the matching row's Resolution/FPS combos and mirrors
        into the per-session cache.
        """
        if not modes or self.main_window is None:
            return
        pipe = getattr(self.main_window, "pipeline", None)
        if pipe is not None:
            try:
                fields = {"probed_modes": list(modes)}
                if by_backend:
                    fields["probed_variants"] = dict(by_backend)
                pipe.update_camera_config(str(camera_id), **fields)
            except Exception as e:
                logger.debug("apply calibration modes for %s: %s", camera_id, e)
        # Seed a usable default selection for this camera, and if it's the one
        # shown in the Capture card, reload the card to show the fresh modes.
        self._ensure_capture_defaults(camera_id)
        if str(camera_id) == str(self._active_capture_cam()):
            self._load_capture_for(camera_id)

    def _build_calibration_progress_dialog(self, total_steps):
        """The shared batch-progress popup, calibration-sized."""
        from source.gui.dialogs._progress_worker import build_progress_dialog
        dlg = build_progress_dialog(
            self, "Calibrate Cameras progress", total_steps,
            header="Camera calibration - Live updates",
            with_log=True, width=380, log_min_height=180)
        return dlg, dlg.log_view, dlg.bar

    @staticmethod
    def _append_cal_log(log_view, text) -> None:
        if not text:
            return
        try:
            log_view.append(str(text))
        except Exception:
            pass

    def _run_calibration_probes(self, items, log_view, dlg, bar) -> None:
        """Run the probe batch off the GUI thread via the shared worker harness.

        Worker bodies only touch device I/O + ``signals.progress.emit``; all
        widget mutation (combo repopulate, log append) happens in the
        main-thread success/failure/progress slots the harness dispatches.
        """
        from source.gui.dialogs._progress_worker import run_box_actions_in_parallel
        from source.video.cameras import calibration_store as _store
        from source.video.cameras import probe as _probe

        def _probe_body(item, signals):
            def report(kind, **data):
                try:
                    signals.progress.emit({
                        "box": item.camera_id,
                        "msg": _format_probe_report(kind, data),
                        "end": "\n",
                    })
                except Exception:
                    pass
            # Every capture backend, not just the platform's preferred one:
            # what a camera can deliver depends on the door, and measuring one
            # door presents its limits as the camera's.
            variants = _probe.probe_all(item.camera_id, item.backend,
                                        item.target_fps, report=report,
                                        pixel_format=item.pixel_format,
                                        doors=[item.door] if item.door else None)
            item.variants = variants
            item.result = _probe.merged_modes(variants)
            if not item.result:
                raise RuntimeError("; ".join(
                    f"{v.backend}: {v.error}" for v in variants if v.error)
                    or "no supported modes detected")

        def _on_success(item):
            modes = list(getattr(item, "result", None) or [])
            variants = list(getattr(item, "variants", None) or [])
            # Doors that measured nothing are recorded as EMPTY rather than
            # left out. Left out, a door that was tried and cannot serve this
            # camera is indistinguishable from one never tried, so the row
            # kept inviting the operator to "try" a door that had already
            # failed, every time, with nothing said.
            by_backend = {v.backend: list(v.modes or []) for v in variants}
            self._apply_calibration_modes(item.camera_id, modes, by_backend)
            try:
                # Detect REPLACES. The operator pressed it because the
                # stored numbers were not trusted, so merging the new reading
                # with the old one is the one thing it must not do.
                _store.put(item.unique_id, item.backend, item.label,
                           modes, item.identity, weak=item.weak,
                           variants=by_backend, replace=True,
                           pixel_format=item.pixel_format)
            except Exception as e:
                logger.debug("calibration_store.put(%s) failed: %s",
                             item.unique_id, e)
            suffix = ("  [" + ", ".join(
                f"{v.backend} {v.best_fps():.0f}fps" for v in variants
                if v.modes) + "]") if len(by_backend) > 1 else ""
            self._append_cal_log(
                log_view, f"[cam {item.camera_id}] done: {len(modes)} mode(s){suffix}")

        def _on_failure(item, err):
            self._append_cal_log(log_view, f"[cam {item.camera_id}] FAILED: {err}")

        def _on_progress(item, msg, _end):
            self._append_cal_log(log_view, f"[cam {item.camera_id}] {msg}")

        def _on_queued(item):
            self._append_cal_log(log_view, f"[cam {item.camera_id}] queued...")

        run_box_actions_in_parallel(
            setup_widgets=items,
            worker_body=_probe_body,
            on_success=_on_success,
            on_failure=_on_failure,
            progress_dialog=dlg,
            progress_bar=bar,
            on_progress=_on_progress,
            on_queued=_on_queued,
            wrap_print=False,
            max_workers_env="PYBEHAVIORLAB_CAMERA_PARALLELISM",
            default_max_workers=4,
        )

    # ----- frame loss strategy (combo + soft info panel) -----

    def _build_sci_settings_group(self):
        """Per-camera scientific-camera I/O (Spinnaker / Ximea): exposure,
        gain, triggering, strobe/illumination, and sync role. The controls act
        on the currently-selected table row and persist to that camera's
        CameraConfig (loaded on row-select, saved on change), so each camera
        keeps its own settings. Only shown when a scientific backend exists."""
        from source.gui.style_builders import spinbox_style as _spin_st_cc

        # Titled for what it holds: whatever the selected camera reports.
        # A webcam shows a few rows, a FLIR shows its whole node map.
        self.sci_settings_group = QtWidgets.QGroupBox("Camera options")
        self.sci_settings_group.setStyleSheet(self._group_qss)
        self.sci_settings_group.setVisible(len(self._available_backends) > 1)
        sci = QtWidgets.QFormLayout(self.sci_settings_group)
        sci.setSpacing(8)
        sci.setContentsMargins(10, 15, 10, 10)

        spin_qss = _spin_st_cc()
        # Guard so programmatic loads (_loadSciForRow) don't fire the change
        # handler and write the row back to itself.
        self._loading_sci = False

        self.exposure_spin = QtWidgets.QSpinBox()
        self.exposure_spin.setRange(7, 100000)
        self.exposure_spin.setValue(5000)
        self.exposure_spin.setSuffix(" us")
        self.exposure_spin.setStyleSheet(spin_qss)
        self.exposure_spin.setMinimumWidth(100)
        sci.addRow("Exposure Time:", self.exposure_spin)

        self.gain_spin = QtWidgets.QDoubleSpinBox()
        self.gain_spin.setRange(0, 48)
        self.gain_spin.setValue(0)
        self.gain_spin.setSuffix(" dB")
        self.gain_spin.setStyleSheet(spin_qss)
        self.gain_spin.setMinimumWidth(100)
        sci.addRow("Gain:", self.gain_spin)

        # ── Trigger ──────────────────────────────────────────────────────
        self.trigger_mode_combo = QtWidgets.QComboBox()
        self.trigger_mode_combo.addItems(["Free-run", "Hardware", "Software"])
        sci.addRow("Trigger:", self.trigger_mode_combo)

        self.trigger_source_edit = QtWidgets.QLineEdit()
        self.trigger_source_edit.setPlaceholderText("input line, e.g. Line3")
        sci.addRow("  Source line:", self.trigger_source_edit)

        self.trigger_edge_combo = QtWidgets.QComboBox()
        self.trigger_edge_combo.addItems(["Rising", "Falling"])
        sci.addRow("  Edge:", self.trigger_edge_combo)

        self.trigger_delay_spin = QtWidgets.QSpinBox()
        self.trigger_delay_spin.setRange(0, 1_000_000)
        self.trigger_delay_spin.setSuffix(" us")
        self.trigger_delay_spin.setStyleSheet(spin_qss)
        sci.addRow("  Delay:", self.trigger_delay_spin)

        # ── Illumination / strobe out ────────────────────────────────────
        self.strobe_check = QtWidgets.QCheckBox("Drive output line from exposure")
        self.strobe_check.setStyleSheet("QCheckBox { color: #e0e0e0; }")
        sci.addRow("Strobe:", self.strobe_check)

        self.strobe_line_edit = QtWidgets.QLineEdit()
        self.strobe_line_edit.setPlaceholderText("output line, e.g. Line1 (BFS)")
        sci.addRow("  Output line:", self.strobe_line_edit)

        self.strobe_source_combo = QtWidgets.QComboBox()
        self.strobe_source_combo.addItems(["Exposure active", "Frame active"])
        sci.addRow("  Source:", self.strobe_source_combo)

        self.strobe_invert_check = QtWidgets.QCheckBox("Invert polarity")
        self.strobe_invert_check.setStyleSheet("QCheckBox { color: #e0e0e0; }")
        sci.addRow("", self.strobe_invert_check)

        # ── Sync role ────────────────────────────────────────────────────
        self.sync_role_combo = QtWidgets.QComboBox()
        self.sync_role_combo.addItems(["None", "Primary", "Secondary"])
        self.sync_role_combo.setToolTip(
            "Primary free-runs and drives its output line; a secondary "
            "hardware-triggers off that line (set its trigger source). "
            "Secondaries are started (armed) before the primary.")
        sci.addRow("Sync role:", self.sync_role_combo)

        # Persist on any change to the selected row's camera config.
        for w in (self.exposure_spin, self.gain_spin, self.trigger_delay_spin):
            w.valueChanged.connect(self._saveSciForRow)
        for w in (self.trigger_mode_combo, self.trigger_edge_combo,
                  self.strobe_source_combo, self.sync_role_combo):
            w.currentIndexChanged.connect(self._saveSciForRow)
        for w in (self.strobe_check, self.strobe_invert_check):
            w.toggled.connect(self._saveSciForRow)
        for w in (self.trigger_source_edit, self.strobe_line_edit):
            w.editingFinished.connect(self._saveSciForRow)
        self.trigger_mode_combo.currentIndexChanged.connect(self._syncSciEnabled)
        self.strobe_check.toggled.connect(self._syncSciEnabled)
        self._syncSciEnabled()

        # Descriptor-driven panel. When the camera is open its backend reports
        # what it actually has, ranges, enum entries and write access come
        # from the sensor, and that replaces the fixed form above, which is
        # only a fallback for a camera that is not connected yet.
        from source.gui.widgets.camera_feature_panel import CameraFeaturePanel

        self._fixed_sci_rows = [sci.itemAt(i).widget()
                                for i in range(sci.count())
                                if sci.itemAt(i) is not None
                                and sci.itemAt(i).widget() is not None]
        self.feature_panel = CameraFeaturePanel()
        self.feature_panel.featureChanged.connect(self._on_feature_changed)
        # A FLIR reports far more rows than fit, so the panel scrolls inside a
        # fixed area rather than stretching the dialog. Without an explicit
        # minimum the form layout squashes it to nothing.
        # The detail column already scrolls, so the panel keeps its natural
        # height here, nesting a second scroll area only crushes the rows.
        self.feature_scroll = self.feature_panel
        self.feature_panel.setVisible(False)
        sci.addRow(self.feature_panel)

        self.all_features_btn = QtWidgets.QPushButton("All features…")
        self.all_features_btn.setToolTip(
            "Every feature this camera reports, searchable, with its SDK node "
            "name, nothing is hidden.")
        self.all_features_btn.clicked.connect(self._show_all_features)
        self.all_features_btn.setVisible(False)
        sci.addRow("", self.all_features_btn)
        return self.sci_settings_group

    # ── descriptor-driven camera options ─────────────────────────────────

    def _feature_camera_id(self):
        """Camera id whose features the panel is showing, or None."""
        cid = self._active_capture_cam()
        return str(cid) if cid not in (None, "") else None

    def _describe_active_features(self):
        pipe = getattr(self.main_window, "pipeline", None) if self.main_window else None
        cid = self._feature_camera_id()
        if pipe is None or cid is None:
            return []
        try:
            return pipe.describe_camera_features(cid)
        except Exception as exc:
            logger.debug("describe_camera_features(%s) failed: %s", cid, exc)
            return []

    def _queue_feature_panel(self):
        """Ask the camera what it supports soon, not on every keystroke."""
        timer = getattr(self, "_features_timer", None)
        if timer is None:
            self._refresh_feature_panel()
        else:
            timer.start()

    def _refresh_feature_panel(self):
        """Show camera-reported options when available, else the fixed form."""
        if not hasattr(self, "feature_panel"):
            return
        features = self._describe_active_features()
        curated = [f for f in features if f.tier == "curated"]
        self._all_features_cache = features

        pipe = getattr(self.main_window, "pipeline", None) if self.main_window else None
        cid = self._feature_camera_id()
        values = {}
        if pipe is not None and cid is not None:
            try:
                cfg = pipe.all_camera_configs().get(cid)
                values = dict(getattr(cfg, "features", None) or {})
            except Exception:
                values = {}

        has_desc = bool(curated)
        self.feature_panel.set_features(curated, values)
        self.feature_scroll.setVisible(has_desc)
        self.all_features_btn.setVisible(bool(features))
        # The hand-written exposure/gain/trigger rows are a stand-in for a
        # camera that cannot describe itself; hide them once it can.
        for widget in getattr(self, "_fixed_sci_rows", []):
            try:
                widget.setVisible(not has_desc)
            except RuntimeError:
                continue
        # Depends on _all_features_cache, which this method just filled.
        self._updateSciSettingsVisibility()

    def _on_feature_changed(self, key, value):
        """Apply live when streaming; always persist to the camera config."""
        pipe = getattr(self.main_window, "pipeline", None) if self.main_window else None
        cid = self._feature_camera_id()
        if pipe is None or cid is None:
            return
        applied = pipe.set_camera_feature(cid, key, value)
        logger.debug("camera %s feature %s=%r (%s)", cid, key, value,
                     "live" if applied else "queued for next connect")
        marker = getattr(self.main_window, "_project_changed", None)
        if callable(marker):
            marker(reason="camera_feature_changed")

    def _show_all_features(self):
        """Open the full feature tree for the active camera."""
        from source.gui.widgets.camera_feature_panel import CameraFeatureTree

        features = getattr(self, "_all_features_cache", None) or \
            self._describe_active_features()
        if not features:
            QtWidgets.QMessageBox.information(
                self, "All features",
                "Connect this camera to read the features it supports.")
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"All features, camera {self._feature_camera_id()}")
        dlg.resize(720, 560)
        lay = QtWidgets.QVBoxLayout(dlg)
        tree = CameraFeatureTree()
        pipe = getattr(self.main_window, "pipeline", None) if self.main_window else None
        values = {}
        if pipe is not None:
            try:
                cfg = pipe.all_camera_configs().get(self._feature_camera_id())
                values = dict(getattr(cfg, "features", None) or {})
            except Exception:
                values = {}
        tree.set_features(features, values)
        lay.addWidget(tree)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        lay.addWidget(buttons)
        dlg.exec()

    # ── per-camera sci-settings load / save ───────────────────────────────

    _TRIG_MODES = ("freerun", "hardware", "software")
    _STROBE_SRC = ("exposure_active", "frame_active")

    def _selected_sci_camera_id(self):
        """Active camera id when it's a scientific backend, else ``None`` (the
        sci controls only apply to Spinnaker/Ximea). Keyed off the same
        Active-camera picker as the Capture card so both stay in step."""
        cid = self._active_capture_cam()
        if cid is None:
            return None
        backend = self._backendForCameraId(cid) or "opencv"
        if backend == "opencv":
            return None
        return cid

    def _syncSciEnabled(self):
        """Grey out trigger/strobe detail rows that don't apply to the current
        selections (hardware-trigger rows, strobe rows)."""
        hw = self.trigger_mode_combo.currentText() == "Hardware"
        for w in (self.trigger_source_edit, self.trigger_edge_combo,
                  self.trigger_delay_spin):
            w.setEnabled(hw)
        on = self.strobe_check.isChecked()
        for w in (self.strobe_line_edit, self.strobe_source_combo,
                  self.strobe_invert_check):
            w.setEnabled(on)

    def _loadSciForRow(self):
        """Populate the sci controls from the selected camera's CameraConfig."""
        if not hasattr(self, "trigger_mode_combo"):
            return
        cid = self._selected_sci_camera_id()
        pipe = getattr(self.main_window, "pipeline", None) if self.main_window else None
        cfg = None
        if cid is not None and pipe is not None:
            try:
                cfg = pipe.all_camera_configs().get(str(cid))
            except Exception:
                cfg = None
        self._loading_sci = True
        try:
            if cfg is None:
                return
            if cfg.exposure_us is not None:
                self.exposure_spin.setValue(int(cfg.exposure_us))
            if cfg.gain_db is not None:
                self.gain_spin.setValue(float(cfg.gain_db))
            t = cfg.trigger
            self.trigger_mode_combo.setCurrentIndex(
                self._TRIG_MODES.index(t.mode) if t.mode in self._TRIG_MODES else 0)
            self.trigger_source_edit.setText(t.source)
            self.trigger_edge_combo.setCurrentIndex(1 if t.edge == "falling" else 0)
            self.trigger_delay_spin.setValue(int(t.delay_us))
            lo = cfg.line_output
            self.strobe_check.setChecked(bool(lo.enabled))
            self.strobe_line_edit.setText(lo.line)
            self.strobe_source_combo.setCurrentIndex(
                self._STROBE_SRC.index(lo.source) if lo.source in self._STROBE_SRC else 0)
            self.strobe_invert_check.setChecked(bool(lo.inverted))
            role = (cfg.sync_role or "none").capitalize()
            i = self.sync_role_combo.findText(role)
            self.sync_role_combo.setCurrentIndex(i if i >= 0 else 0)
        finally:
            self._loading_sci = False
            self._syncSciEnabled()

    def _saveSciForRow(self, *_):
        """Write the sci controls back to the selected camera's CameraConfig."""
        if getattr(self, "_loading_sci", False):
            return
        cid = self._selected_sci_camera_id()
        pipe = getattr(self.main_window, "pipeline", None) if self.main_window else None
        if cid is None or pipe is None:
            return
        from source.video.framebus.types import LineOutputConfig, TriggerConfig
        trig = TriggerConfig(
            mode=self._TRIG_MODES[self.trigger_mode_combo.currentIndex()],
            source=self.trigger_source_edit.text().strip(),
            edge="falling" if self.trigger_edge_combo.currentIndex() == 1 else "rising",
            delay_us=float(self.trigger_delay_spin.value()),
        )
        lo = LineOutputConfig(
            enabled=self.strobe_check.isChecked(),
            line=self.strobe_line_edit.text().strip(),
            source=self._STROBE_SRC[self.strobe_source_combo.currentIndex()],
            inverted=self.strobe_invert_check.isChecked(),
        )
        try:
            pipe.update_camera_config(
                str(cid),
                exposure_us=float(self.exposure_spin.value()),
                gain_db=float(self.gain_spin.value()),
                trigger=trig, line_output=lo,
                sync_role=self.sync_role_combo.currentText().lower(),
            )
        except Exception as e:
            logger.debug("save sci settings for %s failed: %s", cid, e)

    # ----- auto-connect-on-load project preference -----

    def _build_auto_connect_row(self):
        """Project preference: when ticked, loading this project auto-connects
        every configured camera. Saved on ``cfg.meta.auto_connect_cameras_on_load``;
        read back here from the active config so the box reflects the saved
        state, and saved the instant the user toggles it (no Connect needed)."""
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.auto_connect_check = QtWidgets.QCheckBox(
            "Auto-connect cameras when this project loads")
        self.auto_connect_check.setStyleSheet("QCheckBox { color: #e0e0e0; }")
        self.auto_connect_check.setToolTip(
            "Saved in the project config. When on, opening this project "
            "connects every configured camera automatically.")
        meta = getattr(getattr(self.main_window, "_active_config", None),
                       "meta", None)
        self.auto_connect_check.setChecked(
            True if meta is None
            else bool(getattr(meta, "auto_connect_cameras_on_load", True)))
        # Connect AFTER setChecked so the initial state restore doesn't fire it.
        self.auto_connect_check.toggled.connect(self._on_auto_connect_toggled)
        row.addWidget(self.auto_connect_check)
        row.addStretch(1)
        return row

    def _on_auto_connect_toggled(self, checked: bool) -> None:
        """Write the pref onto the active project config + mark it dirty so
        autosave persists it."""
        meta = getattr(getattr(self.main_window, "_active_config", None),
                       "meta", None)
        if meta is None:
            return
        meta.auto_connect_cameras_on_load = bool(checked)
        try:
            self.main_window._project_changed(reason="auto_connect_pref_changed")
        except Exception:
            pass

    # ----- action row (Connect / Clear ROIs / Disconnect) -----

    def _build_action_row(self):
        """Action buttons: Connect = success (primary), Clear = warning,
        Disconnect = danger, all in one row."""
        from source.gui.styles import BUTTON_STYLE as _BS
        from source.gui.styles import COLORS as _C
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        row.setContentsMargins(0, 4, 0, 0)

        # (label, color_key, slot, stretch, attr_name)
        action_specs = (
            ("Connect Cameras",    "success",  self.connect_cameras,    2, "connect_btn"),
            ("Clear All ROIs",     "warning",  self.clearAllROIs,       1, None),
            ("Disconnect Cameras", "danger",   self.disconnect_cameras, 1, "disconnect_btn"),
        )
        for label, key, slot, stretch, attr in action_specs:
            b = QtWidgets.QPushButton(label)
            b.setStyleSheet(_BS.format(color=_C[key], hover_color=_C[key + '_hover']))
            b.setMinimumHeight(38)
            b.setAutoDefault(False)   # don't let Enter trigger Disconnect/Clear
            b.clicked.connect(slot)
            if attr is not None:
                setattr(self, attr, b)
            row.addWidget(b, stretch=stretch)
        # Enter → Connect Cameras (the primary, non-destructive action).
        self.connect_btn.setDefault(True)
        self.connect_btn.setAutoDefault(True)
        return row

    def _connected_camera_boxes(self):
        """``{camera_id: [box ids]}`` for what is actually streaming.

        Read from ``box_camera_map`` rather than the table: the table holds
        what the operator *typed*, which after an edit is not necessarily what
        is open. Disconnect has to act on the open devices.
        """
        vm = getattr(self.main_window, "video_manager", None) \
            if self.main_window else None
        bcm = getattr(vm, "box_camera_map", None) if vm is not None else None
        out: dict = {}
        if isinstance(bcm, dict):
            for setup_id, cam_id in bcm.items():
                out.setdefault(cam_id, []).append(setup_id)
        return out

    def _choose_cameras_to_disconnect(self, by_camera):
        """Ask which cameras to release. Returns the chosen ids, or ``None``
        if the operator cancelled.

        Skipped when only one camera is open; there is nothing to choose.
        """
        if len(by_camera) <= 1:
            return list(by_camera)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Disconnect cameras")
        dlg.setModal(True)
        v = QtWidgets.QVBoxLayout(dlg)
        v.setSpacing(8)
        head = QtWidgets.QLabel(
            "Several cameras are streaming. Choose which to disconnect, the "
            "others keep running.")
        head.setWordWrap(True)
        v.addWidget(head)
        checks = {}
        for cam_id, boxes in sorted(by_camera.items(), key=lambda kv: str(kv[0])):
            names = ", ".join(f"Box {b}" for b in sorted(boxes))
            cb = QtWidgets.QCheckBox(f"camera {cam_id}   ·   {names}")
            cb.setChecked(True)
            checks[cam_id] = cb
            v.addWidget(cb)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        v.addWidget(buttons)
        from source.gui.style_builders import apply_dialog_theme
        apply_dialog_theme(dlg)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return [cam for cam, cb in checks.items() if cb.isChecked()]

    def disconnect_cameras(self):
        """Release streaming cameras, asking which when more than one is open.

        Each box is torn down through ``main_window.disconnect_camera``, the
        single pipeline path (sink unsubscribe, bus unregister, device
        release), so disconnecting one camera leaves the others streaming.
        The typed camera ids stay in the table: disconnect releases the
        capture, it does not un-configure the rig.
        """
        try:
            if not self.main_window:
                return
            by_camera = self._connected_camera_boxes()
            if not by_camera:
                QtWidgets.QMessageBox.information(
                    self, "No cameras", "No cameras are connected.")
                return

            chosen = self._choose_cameras_to_disconnect(by_camera)
            if chosen is None:
                return                      # cancelled
            if not chosen:
                QtWidgets.QMessageBox.information(
                    self, "Nothing selected",
                    "No camera was selected, so nothing was disconnected.")
                return

            released, failed = [], []
            for cam_id in chosen:
                for setup_id in by_camera.get(cam_id, []):
                    try:
                        self.main_window.disconnect_camera(setup_id)
                    except Exception as e:
                        failed.append(f"Box {setup_id}: {e}")
                        logger.error("disconnect box %s failed: %s",
                                     setup_id, e)
                released.append(cam_id)

            # The segment map only describes cameras that are still open.
            if not self._connected_camera_boxes() and \
                    hasattr(self.main_window, "video_segment_config"):
                self.main_window.video_segment_config = None
            self.connected_count = len(self._connected_camera_boxes())

            self._refresh_status_dots()
            self._refresh_marks()

            names = ", ".join(f"camera {c}" for c in released)
            if failed:
                QtWidgets.QMessageBox.warning(
                    self, "Disconnected with errors",
                    f"Released {names}.\n\nProblems:\n" + "\n".join(failed))
            else:
                logger.info("Disconnected %s", names)
                QtWidgets.QMessageBox.information(
                    self, "Cameras disconnected", f"Released {names}.")
        except Exception as e:
            logger.error(f"Error disconnecting cameras: {e!s}")
            QtWidgets.QMessageBox.warning(
                self,
                "Error",
                f"Failed to disconnect cameras: {e!s}"
            )
        finally:
            self.updateButtonStates()

    def updateButtonStates(self):
        """Update Connect/Disconnect button states.

        Connect stays enabled whenever a box with a Camera ID is NOT yet
        connected, so a partial multi-camera setup (some boxes up, others
        still to configure) isn't trapped by a disabled Connect button. It
        only disables once every box that has a camera id is connected.
        """
        try:
            any_connected = False
            bcm = {}
            if self.main_window and hasattr(self.main_window, 'video_manager'):
                bcm = self.main_window.video_manager.box_camera_map or {}
                any_connected = len(bcm) > 0

            selected = self.getSelectedBoxesWithCameras()
            # "Pending" means the box is not already connected TO THE CAMERA IT
            # NOW NAMES. Testing box membership alone left Connect disabled
            # forever once a box was connected, so changing that box's Camera ID
            # could never be applied, the operator had to Disconnect first.
            pending = any(
                str(bcm.get(item.get('box_number'))) != str(item.get('camera_id'))
                for item in selected
            ) if selected else True

            self.connect_btn.setEnabled(pending)
            self.disconnect_btn.setEnabled(any_connected)
        except Exception as e:
            logger.error(f"Error updating button states: {e!s}")

    def populateBoxes(self):
        """Populate table with connected boxes"""
        self.save_video_checkboxes = []
        self.backend_combos = []

        if not self.main_window or not hasattr(self.main_window, 'get_all_setup_widgets'):
            return

        connected_boxes = list(self.main_window.get_all_setup_widgets())

        self.table.setRowCount(len(connected_boxes))
        # Row N is a different box now, so what row N last displayed says
        # nothing about what it should display next.
        self._row_capture_sig = {}

        for row, setup_widget in enumerate(connected_boxes):
            # Box number
            # Empty item, the styled cell widget (set by _refresh_status_dots)
            # provides the row's display; a non-empty item text would bleed
            # through the transparent widget as a duplicate "Box N".
            box_item = QtWidgets.QTableWidgetItem("")
            self.table.setItem(row, 0, box_item)

            # Camera ID, editable combo: type any id, or pick from cameras
            # detected on this machine (populated lazily on first dropdown).
            camera_id_edit = _CameraIdCombo(self._populate_camera_id_combo)
            camera_id_edit.lineEdit().setPlaceholderText("choose…")
            # Get existing camera ID if available
            if hasattr(setup_widget, 'camera_id_edit') and setup_widget.camera_id_edit.text():
                camera_id_edit.setEditText(setup_widget.camera_id_edit.text())
            # A changed Camera ID re-syncs the active-camera picker (and, when
            # this row is the active one, reloads the Capture Settings card).
            camera_id_edit.currentTextChanged.connect(
                lambda _t, r=row: self._on_row_camera_id_changed(r)
            )
            camera_id_edit.setStyleSheet(self._CELL_COMBO_QSS)
            self.table.setCellWidget(row, self.COL_CAMERA_ID, camera_id_edit)

            # Backend combo: WHICH DOOR the camera is opened through.
            #
            # It used to name the camera FAMILY (OpenCV / Spinnaker / XIMEA),
            # which on a UVC rig reads "OpenCV" on every row and tells the
            # operator nothing. The door is the choice that changes what the
            # camera delivers: measured on this rig, one camera gave 19.9 fps
            # through DirectShow and 29.9 through Media Foundation at the same
            # mode, while another was the other way round. It cannot be
            # guessed per platform, so it is chosen here and Detect measures
            # it. The family is read from the enumeration instead.
            backend_combo = QtWidgets.QComboBox()
            backend_combo.setStyleSheet(self._CELL_COMBO_QSS)
            self._fill_door_combo(backend_combo, cam_id=None)
            backend_combo.currentIndexChanged.connect(self._onBackendChanged)
            self.table.setCellWidget(row, self.COL_BACKEND, backend_combo)
            self.backend_combos.append(backend_combo)

            # Record toggle, a decision about the session, per box.
            save_widget = QtWidgets.QWidget()
            save_layout = QtWidgets.QHBoxLayout(save_widget)
            save_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            save_layout.setContentsMargins(0, 0, 0, 0)
            save_checkbox = _ToggleSwitch()
            current_save_pref = getattr(setup_widget, "save_video_enabled", True)
            save_checkbox.setChecked(current_save_pref if current_save_pref is not None else True)
            save_checkbox.toggled.connect(self._on_row_record_toggled)
            save_layout.addWidget(save_checkbox)
            self.table.setCellWidget(row, self.COL_SAVE, save_widget)
            self.save_video_checkboxes.append(save_checkbox)

            # Per-row mode probe, beside the values it fills in.
            detect_btn = QtWidgets.QToolButton()
            detect_btn.setText("Detect")
            detect_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            detect_btn.setToolTip(
                "Measure this camera's resolutions and the frame rate each "
                "one really delivers.")
            detect_btn.setStyleSheet(
                "QToolButton { color:#cfd6e2; background:rgba(255,255,255,0.05);"
                " border:1px solid #39404d; border-radius:4px;"
                " padding:2px 8px; font-size:10.5px; }"
                "QToolButton:hover { background:rgba(255,255,255,0.10); }"
                "QToolButton:disabled { color:#5b626e; border-color:#2b313b; }")
            detect_btn.clicked.connect(
                lambda _=False, r=row: self._onDetectRow(r))
            self.table.setCellWidget(row, self.COL_DETECT, detect_btn)

            # Resolution / FPS / Format / Flip, per row, so a rig with one
            # camera per box can set each camera without selecting it first.
            self._build_row_capture_cells(row)

        # Populate the active-camera picker from the table, then load the first
        # camera's Capture Settings. Guarded so this initial fill doesn't fire
        # the change handler mid-build.
        self._refresh_active_cam_combo()
        # Select the first row that has a Camera ID so the table selection and
        # the active-camera picker agree. Falls back to row 0.
        first_nonempty = next(
            (r for r in range(self.table.rowCount())
             if self._cameraIdForRow(r) is not None), 0)
        if self.table.rowCount() > 0:
            self.table.setCurrentCell(first_nonempty, self.COL_BOX)
        # Load the first camera's Capture + Scientific settings.
        self._load_active_camera_settings()
        # Update scientific settings visibility
        self._updateSciSettingsVisibility()
        # Start in CCTV mode when the rows already share one camera.
        self._detect_camera_mode()
        # Fill the Box column and every read-only cell that follows it.
        self._refresh_status_dots()

    def _onBackendChanged(self, *_):
        """The operator picked a different DOOR for one row.

        The door decides what the camera can do, so everything below it in the
        chain is rebuilt: the sizes and rates on offer are the ones measured
        through THIS door, not the one that was showing a moment ago.
        """
        if getattr(self, "_loading_capture", False):
            return
        combo = self.sender()
        row = (self.backend_combos.index(combo)
               if combo in self.backend_combos else self._selected_row())
        if 0 <= row < self.table.rowCount():
            self._refill_row_after_door_change(row)
        self._load_active_camera_settings()

    def _refill_row_after_door_change(self, row) -> None:
        """Rebuild one row's Resolution and FPS from the door it now names."""
        cam = self._camera_id_text(row)
        if not cam:
            return
        # The row's signature is what suppresses a redundant rebuild; the door
        # is part of what it describes, so it has to be dropped or the rebuild
        # below is skipped as "nothing changed".
        if hasattr(self, "_row_capture_sig"):
            self._row_capture_sig.pop(row, None)
        # WRITE FIRST, then redraw. ``_fill_row_capture`` restores the door
        # from the stored config, so redrawing before the new choice is stored
        # re-read the old one and snapped the cell straight back.
        self._on_row_capture_changed(row)
        self._fill_row_capture(row)

    def _updateSciSettingsVisibility(self):
        """Reveal the scientific-camera block the instant the SELECTED camera is
        FLIR/Ximea (and auto-open the Advanced section so it's seen); hidden for
        plain webcams so the panel stays simple."""
        from source.video.cameras.features import TIER_CURATED
        # Shown when the camera reports something the Acquisition card above
        # does NOT already control, i.e. a CURATED feature.
        #
        # "Reports anything at all" was the old rule, and for a webcam that
        # meant a second copy of the card: rate, transport format, width and
        # height, in different widgets, with the rate reading 1 fps beside a
        # picker reading 30. Everything a webcam reports is now full-tier, so
        # it stays reachable under "All features" and stops being a duplicate
        # section. A FLIR still fills this group, because its exposure, gain
        # and trigger nodes genuinely are not on the card.
        curated = [f for f in (getattr(self, "_all_features_cache", None) or [])
                   if getattr(f, "tier", TIER_CURATED) == TIER_CURATED]
        show = self._selected_sci_camera_id() is not None or bool(curated)
        if hasattr(self, "sci_settings_group"):
            self.sci_settings_group.setVisible(show)
        model = getattr(self, "_options_model", None)
        if model is not None:
            cam = self._active_capture_cam()
            name = ""
            if cam is not None:
                backend = self._backendForCameraId(cam) or "opencv"
                ident = self._resolve_identity_cached(cam, backend)
                name = (ident.get("identity") or {}).get("name") or ""
            model.setText(name or (f"camera {cam}" if cam is not None
                                   else "\u2014"))

    #: The OS capture doors, per platform, as (display name, stored id).
    #: The ids are the ones already used everywhere else in the store and the
    #: probe ("dshow"/"msmf"/"v4l2"/"any"), so a door chosen here matches the
    #: door a measurement was filed under without translation.
    @staticmethod
    def _available_doors() -> list:
        import sys as _sys
        if os.name == "nt":
            return [("DirectShow", "dshow"), ("Media Foundation", "msmf")]
        if _sys.platform == "darwin":
            return [("AVFoundation", "avfoundation")]
        return [("V4L2", "v4l2"), ("GStreamer", "any")]

    @staticmethod
    def _default_door() -> str:
        """The door a camera starts on when a project does not name one.

        DirectShow on Windows, measured: it opens in ~6 s where Media
        Foundation took 50-73 s on two of the three cameras on this rig, and
        it reports its pixel format, which Media Foundation does not. Media
        Foundation is still the better door for some cameras (one reached
        29.9 fps where DirectShow managed 19.9), which is exactly why this is
        a choice in the UI rather than a constant.
        """
        import sys as _sys
        if os.name == "nt":
            return "dshow"
        if _sys.platform == "darwin":
            return "avfoundation"
        return "v4l2"

    def _fill_door_combo(self, combo, cam_id=None, selected=None) -> None:
        """Fill one row's Backend cell for the camera it holds.

        A UVC camera is offered this platform's doors. A scientific camera has
        no door to choose: it is reached through its SDK, so the cell names
        that and is disabled rather than offering choices that do not apply.
        """
        family = self._family_for_camera(cam_id) if cam_id else "opencv"
        was = combo.signalsBlocked()
        combo.blockSignals(True)
        try:
            combo.clear()
            if family != "opencv":
                combo.addItem(_BACKEND_LABELS.get(family, family.title()), "")
                combo.setCurrentIndex(0)
                combo.setEnabled(False)
                return
            combo.setEnabled(True)
            for label, door in self._available_doors():
                combo.addItem(label, door)
            want = str(selected or "") or self._default_door()
            i = combo.findData(want)
            combo.setCurrentIndex(i if i >= 0 else 0)
        finally:
            combo.blockSignals(was)

    def _family_for_camera(self, camera_id) -> str:
        """``opencv`` / ``spinnaker`` / ``ximea`` for this camera.

        Read from the enumeration, NOT from the Backend cell: that cell now
        chooses the OS door, which a scientific camera does not have.
        """
        fam = getattr(self, "_cam_family", {}).get(str(camera_id))
        return str(fam or "opencv")

    def _store_backfilled_door(self, cam, door) -> None:
        """Write a defaulted door into the camera's config, once."""
        if self.main_window is None:
            return
        pipe = getattr(self.main_window, "pipeline", None)
        if pipe is None:
            return
        try:
            pipe.update_camera_config(str(cam), capture_backend=str(door))
        except Exception as e:
            logger.debug("backfilling door for %s: %s", cam, e)
            return
        logger.info("camera %s: this project named no capture backend; "
                    "defaulted to %s and recorded it.", cam, door)
        mark = getattr(self.main_window, "_mark_project_dirty", None)
        if callable(mark):
            try:
                mark(reason="capture_backend")
            except Exception:
                pass

    def _doorForRow(self, row) -> str:
        """The door named by ``row``'s own Backend cell, or ``""``."""
        if row is None or row < 0 or row >= self.table.rowCount():
            return ""
        combo = self.table.cellWidget(row, self.COL_BACKEND)
        return str((combo.currentData() if combo is not None else None) or "")

    def _doorForCameraId(self, camera_id):
        """The OS door the operator picked for this camera, or ``""``.

        ``""`` for a scientific camera, which is reached through its SDK and
        has no door to choose.
        """
        target = str(camera_id)
        for row in range(self.table.rowCount()):
            if str(self._cameraIdForRow(row)) != target:
                continue
            combo = self.table.cellWidget(row, self.COL_BACKEND)
            data = combo.currentData() if combo is not None else None
            return str(data or "")
        return ""

    def _backend_name_to_id(self, display_name):
        """Convert display name to backend id."""
        mapping = {
            "OpenCV": "opencv",
            "Spinnaker (FLIR)": "spinnaker",
            "Ximea": "ximea",
        }
        return mapping.get(display_name, "opencv")

    def getSelectedFPS(self):
        """Return the selected FPS as int, or 0 if nothing's picked yet
        (combo is empty until a resolution is chosen). Callers treat 0 as
        "unknown"; we never invent a value here."""
        if getattr(self, "fps_combo", None) is None:
            return 0
        try:
            data = self.fps_combo.currentData()
            if data is not None:
                return int(data)
            text = self.fps_combo.currentText().strip()
            return int(text) if text else 0
        except (ValueError, TypeError):
            return 0

    def getSelectedResolution(self) -> Optional[tuple]:
        """Return the chosen ``(width, height)`` or ``None`` if not set yet.

        Combo data is now a ``{"width", "height", "fps"}`` dict (set by
        ``_populateResolutionCombo``), so unpack the resolution from it.
        Returns ``None`` for the placeholder / "no modes" entries that
        carry ``None`` as itemData.
        """
        if getattr(self, "resolution_combo", None) is None:
            return None
        data = self.resolution_combo.currentData()
        if data is None:
            return None
        if isinstance(data, dict):
            try:
                return (int(data["width"]), int(data["height"]))
            except (KeyError, TypeError, ValueError):
                return None
        # 2-tuple form, in case a caller inserts that shape.
        try:
            return (int(data[0]), int(data[1]))
        except (TypeError, ValueError, IndexError):
            return None

    # ── Central CameraConfig integration ──────────────────────────────

    def _activeCameraId(self):
        """Return the camera_id of the currently-selected table row, falling
        back to the first non-empty row. Int when the cell is numeric, str
        otherwise (matches ``_firstCameraIdInTable``)."""
        row = self.table.currentRow()
        if row is not None and row >= 0:
            cid = self._cameraIdForRow(row)
            if cid is not None:
                return cid
        return self._firstCameraIdInTable()

    def _allCameraIdsInTable(self):
        """Return the deduplicated list of camera_ids currently in the
        table (skips empty rows), preserving table order. Used to populate
        the active-camera combo and to iterate Save All targets."""
        seen = []
        for row in range(self.table.rowCount()):
            text = self._camera_id_text(row)
            if not text:
                continue
            try:
                cid = int(text)
            except ValueError:
                cid = text
            if cid not in seen:
                seen.append(cid)
        return seen

    def _repopulate_fps_combo(self, res_combo=None, fps_combo=None,
                              cam_id=None):
        """Rebuild ``fps_combo`` from the realistic ceiling at ``res_combo``'s
        currently-selected resolution.

        The ceiling comes from the resolution combo's itemData (set by
        ``_populateResolutionCombo`` from the (w, h, fps) tuples calibration
        measured). Options step in ``FPS_STEP`` from ``FPS_MIN`` up to the
        ceiling ROUNDED to the nearest ``FPS_STEP``, so a camera that measures
        29.2 offers 30 as its top value and one that measures 24.2 offers 25,
        rather than flooring to the step below. Defaults to that top value;
        Defaults to ``default_fps_for(ceiling)``, the behavioural target when
        the camera can hold it, its rounded ceiling when it cannot, rather
        than always the top option. ``_load_capture_for`` re-applies any saved
        pick.
        """
        from source.video.framebus.types import FPS_MIN, FPS_STEP, default_fps_for
        res_combo = res_combo if res_combo is not None else self.resolution_combo
        fps_combo = fps_combo if fps_combo is not None else self.fps_combo
        if res_combo is None or fps_combo is None:
            return
        data = res_combo.currentData()
        fps_was_blocked = fps_combo.signalsBlocked()
        fps_combo.blockSignals(True)
        try:
            fps_combo.clear()
            if not isinstance(data, dict):
                fps_combo.setEnabled(False)
                return
            try:
                ceiling = float(data.get("fps", 0.0) or 0.0)
            except (TypeError, ValueError):
                ceiling = 0.0
            # WHAT THE CAMERA OFFERS, when it has been enumerated.
            #
            # The ladder below is an arithmetic sequence up to a MEASURED
            # ceiling, and a measured ceiling is the camera multiplied by the
            # room: the same mode read 30.13 fps and then 24.88 fps minutes
            # apart because auto-exposure lengthens integration in dimmer
            # light. It also has no idea where the camera STARTS, so it
            # offered 10 and 5 fps to a camera whose slowest interval is 15.
            #
            # The enumeration answers both. See
            # docs/dev/camera-modes-and-rates.md.
            offered = self._offered_rates_for(data, cam_id=cam_id)
            if offered:
                for fps in offered:
                    fps_combo.addItem(str(int(round(fps))), int(round(fps)))
                fps_combo.setEnabled(True)
                # A list of one is the camera's answer, not a broken control,
                # and it does not look like one. Some devices report a single
                # discrete interval per (size, format): this rig has one whose
                # 640x480 MJPEG is 120 fps and nothing else, so a rate combo
                # with one entry is correct and unexplained. Say what it means
                # and where the usual rate does exist.
                fps_combo.setToolTip(
                    self._one_rate_explanation(data, cam_id, offered)
                    if len(offered) == 1 else
                    "The frame intervals this camera accepts at this size and "
                    "format. The number beside the resolution is what Detect "
                    "measured it DELIVERING, which is what gets recorded.")
                want = default_fps_for(max(offered))
                exact = [i for i in range(fps_combo.count())
                         if fps_combo.itemData(i) == want]
                fps_combo.setCurrentIndex(
                    exact[0] if exact else fps_combo.count() - 1)
                return

            # A mode nobody measured carries rate 0.0. Refusing it a ladder
            # left the combo empty AND disabled, so choosing that mode stored
            # selected_fps=None and the camera opened at no rate at all. The
            # row exists to say "try this door", so give it something to try.
            unmeasured = ceiling <= 0
            if unmeasured:
                ceiling = _UNMEASURED_FPS_CEILING
            if ceiling < FPS_MIN:
                # Rounded, not truncated: a camera measuring 4.9875 is a 5 fps
                # camera, and int() called it 4 while the resolution beside it
                # was labelled "(5)". Same nearest-value rule as the ladder
                # below, so the two can no longer contradict each other.
                opts = [round(ceiling)] if round(ceiling) > 0 else []
            else:
                # Round to the NEAREST step (29.2 -> 30, 24.2 -> 25), not floor.
                last = max(FPS_MIN, round(ceiling / FPS_STEP) * FPS_STEP)
                opts = list(range(FPS_MIN, last + 1, FPS_STEP))
            # A UVC camera has a handful of discrete rates and rounds
            # silently to the nearest one it has. The ladder above is an
            # arithmetic superset of them, so a rate the camera has been SEEN
            # not to offer is dropped: this rig measures 29.9 at 640x480, the
            # ladder offered 10/15/20/25/30, and only 30/20/15 exist. Picking
            # 25 always delivered 30.
            rejected = self._rejected_rates_for(data, cam_id=cam_id)
            if rejected:
                kept = [f for f in opts if f not in rejected]
                if kept:
                    opts = kept
            for fps in opts:
                fps_combo.addItem(str(fps), fps)
            fps_combo.setEnabled(bool(opts))
            if opts:
                want = (_UNMEASURED_FPS_DEFAULT if unmeasured
                        else default_fps_for(ceiling))
                fps_combo.setCurrentIndex(
                    opts.index(want) if want in opts else len(opts) - 1)
        finally:
            fps_combo.blockSignals(fps_was_blocked)

    def _one_rate_explanation(self, mode_data, cam_id, offered) -> str:
        """Why this mode has a single rate, and where to find the usual one."""
        from source.video.framebus.types import FPS_TARGET_DEFAULT
        rate = round(float(offered[0]))
        size = (mode_data.get("width"), mode_data.get("height"))
        fmt = self._selected_format_for(cam_id) or ""
        lines = [f"This camera reports exactly one frame rate at "
                 f"{size[0]}x{size[1]}"
                 + (f" in {fmt.upper()}" if fmt else "") + f": {rate} fps.",
                 "It is a discrete-interval device, so asking for any other "
                 "rate here makes the driver round to this one."]
        # The interval the driver ACCEPTS and the rate the camera DELIVERS are
        # different numbers, and the two cells show one each: 120 in this cell
        # beside "640x480 (100)" in the one before it reads as a contradiction
        # unless it is said out loud.
        got = mode_data.get("fps")
        try:
            got = float(got or 0.0)
        except (TypeError, ValueError):
            got = 0.0
        if got > 0 and abs(got - rate) > max(1.0, rate * 0.05):
            lines.append(
                f"Detect measured it delivering {got:.0f} fps at this "
                f"setting. The recording is written at the delivered rate, so "
                f"it plays back at real speed.")
        # Only worth saying when this mode does NOT already give the target.
        elsewhere = ([] if abs(rate - FPS_TARGET_DEFAULT) < 0.6
                     else self._modes_reaching(cam_id, FPS_TARGET_DEFAULT))
        if elsewhere:
            lines.append(f"\n{FPS_TARGET_DEFAULT} fps is available at: "
                         + ", ".join(elsewhere) + ".")
        return "\n".join(lines)

    def _modes_reaching(self, cam_id, want_fps, limit=6) -> list:
        """``["1920x1080 MJPEG", ...]`` for modes offering ``want_fps``."""
        try:
            from source.video.cameras import calibration_store as _store
            ident = self._resolve_identity_cached(
                cam_id, self._backendForCameraId(cam_id) or "opencv")
            offered = _store.get_offered(ident.get("unique_id"))
            hits = [f"{m.width}x{m.height} {m.pixel_format.upper()}"
                    for m in sorted(offered,
                                    key=lambda m: -(m.width * m.height))
                    if any(abs(r - want_fps) < 0.6 for r in (m.rates or ()))]
            return hits[:limit]
        except Exception as e:
            logger.debug("modes reaching %s for %s: %s", want_fps, cam_id, e)
            return []

    def _offered_rates_for(self, mode_data, cam_id=None) -> tuple:
        """The rates ``cam_id`` OFFERS at this mode, or ``()`` if unknown.

        ``cam_id`` is REQUIRED to be passed by every per-row caller. Without
        it this fell back to the Active-camera picker, so in a multi-camera
        rig every row's rate list was built from whichever camera happened to
        be active: row 2 and row 3 were shown row 1's rates. Detect read each
        camera correctly and the combo beside it then listed another camera's
        modes. Falling back is still right for the shared Capture card, whose
        subject IS the active camera.

        Enumerated from the camera rather than measured, so it does not move
        with the room, and narrowed by the rates this camera has been seen to
        round away. On Linux the enumeration is the camera's exact interval
        list; on Windows DirectShow reports only bounds, so the list inside
        them is still confirmed by opening. That is why the rejections are
        applied here too.

        ``()`` means the camera has never been enumerated, so the caller falls
        back to the ladder. It must not be read as "this mode has no rates".
        """
        if not isinstance(mode_data, dict):
            return ()
        size = (mode_data.get("width"), mode_data.get("height"))
        if not all(size):
            return ()
        cam = None
        try:
            from source.video.cameras import calibration_store as _store
            from source.video.cameras.enumerate_modes import rates_for
            cam = cam_id if cam_id not in (None, "") else (
                self._active_capture_cam() or self._firstCameraIdInTable())
            if cam is None:
                return ()
            ident = self._resolve_identity_cached(
                cam, self._backendForCameraId(cam) or "opencv")
            offered = _store.get_offered(ident.get("unique_id"))
            if not offered:
                return ()
            fmt = self._selected_format_for(cam)
            rates = rates_for(offered, size, fmt) or rates_for(offered, size)
            if not rates:
                return ()
            rejected = self._rejected_rates_for(mode_data, cam_id=cam)
            kept = tuple(r for r in rates if int(round(r)) not in rejected)
            return kept or rates
        except Exception as e:
            logger.debug("offered rates for %s: %s", cam, e)
            return ()

    def _selected_format_for(self, cam_id):
        """The transport format this camera is set to, or ``None``.

        The rate a size can reach depends on it: the same 4K mode is 30 fps
        compressed and 1 fps uncompressed on this rig's camera.
        """
        try:
            cfg = self._camera_config_for(cam_id)
            fmt = str(getattr(cfg, "capture_format", "") or "").lower()
            return fmt if fmt and fmt != "auto" else None
        except Exception:
            return None

    def _rejected_rates_for(self, mode_data, cam_id=None) -> set:
        """Rates ``cam_id`` has been seen not to offer at this mode.

        Same rule as ``_offered_rates_for``: a per-row caller passes the
        row's camera, and only the shared card falls back to the active one.
        """
        if not isinstance(mode_data, dict):
            return set()
        # Every lookup inside the guard. This is a display filter: when it
        # cannot answer it must offer the unfiltered ladder, never raise into
        # the paint path.
        cam = None
        try:
            from source.video.cameras import calibration_store as _store
            cam = cam_id if cam_id not in (None, "") else (
                self._active_capture_cam() or self._firstCameraIdInTable())
            if cam is None:
                return set()
            ident = self._resolve_identity_cached(
                cam, self._backendForCameraId(cam) or "opencv")
            return _store.get_rejected_rates(
                ident.get("unique_id"), mode_data.get("backend", ""),
                (mode_data.get("width"), mode_data.get("height")),
                pixel_format=self._selected_format_for(cam))
        except Exception as e:
            logger.debug("rejected rates for %s: %s", cam, e)
            return set()

    def _captureProbedModesFromCombo(self):
        """Read the (w, h, fps) tuples currently in the resolution combo,
        the dialog's source of truth between Detect Max and Save."""
        modes = []
        if getattr(self, "resolution_combo", None) is None:
            return modes
        for i in range(self.resolution_combo.count()):
            d = self.resolution_combo.itemData(i)
            if not isinstance(d, dict):
                continue
            try:
                modes.append((
                    int(d["width"]),
                    int(d["height"]),
                    float(d.get("fps", 0.0) or 0.0),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return modes

    def _writeDialogSelectionsTo(self, camera_id):
        """Persist ``camera_id``'s capture selections into its central
        CameraConfig before Save / Connect.

        The Capture card holds only the ACTIVE camera's picks. For the active
        camera we write those explicit picks; for any other camera we leave its
        own persisted Resolution/FPS/Format untouched (they were saved when it
        was active or loaded), only refreshing backend + strategy and seeding a
        default selection so it is connectable even if never opened."""
        if self.main_window is None:
            return
        pipe = getattr(self.main_window, "pipeline", None)
        if pipe is None:
            return
        backend = self._backendForCameraId(camera_id) or "opencv"
        # The ROW first, for every camera that has one. This used to write the
        # ACTIVE camera from the Camera options card and merely seed defaults
        # for the rest, so on a per-box rig a camera whose row was never
        # touched connected with nothing set while its row showed a size and
        # a rate. The row is the per-box control; it is the answer.
        row = self._rowForCameraId(camera_id)
        if row is not None:
            fields = self._row_capture_fields(row, camera_id)
            if fields:
                try:
                    pipe.update_camera_config(str(camera_id), **fields)
                    return
                except Exception as e:
                    logger.debug("persist row %s for %s: %s", row, camera_id, e)
        is_active = str(camera_id) == str(self._active_capture_cam())
        if is_active and self.getSelectedResolution() is not None:
            self._write_capture_to(camera_id)
            return
        # Non-active (or active-but-unpicked): keep this camera's own capture
        # values; ensure it has a usable default, then stamp backend + strategy.
        self._ensure_capture_defaults(camera_id)
        try:
            pipe.update_camera_config(
                str(camera_id),
                frame_strategy="accept",
                camera_backend=backend,
            )
        except Exception as e:
            logger.debug("write selections (non-active) %s: %s", camera_id, e)

    def _backendForCameraId(self, camera_id):
        """The camera FAMILY (``opencv`` / ``spinnaker`` / ``ximea``).

        Read from the enumeration, not from the Backend cell. That cell now
        chooses the OS door, which only a UVC camera has; a scientific camera
        is reached through its SDK and would have answered "dshow" here, which
        would file its calibration and its lens profile under a wrong
        identity.
        """
        if camera_id in (None, ""):
            return None
        return self._family_for_camera(camera_id)

    def _defaultCameraConfigDir(self):
        """Default starting folder for camera-config Save / Load,
        ``experiments/`` so saved presets sit alongside the project
        folders the user already navigates to."""
        from source import paths as app_paths
        d = Path(app_paths.experiments_dir)
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return d

    def _onSaveCameraConfig(self):
        """Save the active camera's config to a JSON file the user picks.

        Default folder is ``experiments/``. Writes the central CameraConfig
        as JSON (probed modes, selected resolution + FPS, vendor extras,
        backend); Load replays it so the user can skip the Detect Max probe.
        """
        cam_id = self._activeCameraId()
        if cam_id is None:
            QtWidgets.QMessageBox.warning(
                self, "No camera ID",
                "Enter a camera ID in the table first."
            )
            return
        if self.main_window is None or not hasattr(self.main_window, "pipeline"):
            return
        modes = self._captureProbedModesFromCombo()
        if not modes:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to save",
                "Click Detect Max first; there are no probed modes "
                "to persist yet."
            )
            return
        self._writeDialogSelectionsTo(cam_id)
        cfg = self.main_window.pipeline.get_camera_config(cam_id)

        suggested = self._defaultCameraConfigDir() / f"camera_{cam_id}.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Camera Config", str(suggested),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg.to_json(), f, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Save failed",
                f"Could not write camera config:\n{e}"
            )
            return
        logger.info("Camera %s config saved to %s", cam_id, path)
        QtWidgets.QMessageBox.information(
            self, "Saved",
            f"Saved camera {cam_id} ({len(modes)} probed mode(s)) "
            f"to:\n{path}"
        )

    def _onLoadCameraConfig(self):
        """Load a saved CameraConfig JSON and replay it into the combos.

        Default folder is ``experiments/`` (same as Save). Picks any
        JSON file the user navigates to, parses it via
        ``CameraConfig.from_json``, installs it into the central
        registry under the active camera's id, and re-fills the
        resolution + FPS combos.
        """
        cam_id = self._activeCameraId()
        if cam_id is None:
            QtWidgets.QMessageBox.warning(
                self, "No camera ID",
                "Enter a camera ID in the table first."
            )
            return
        if self.main_window is None or not hasattr(self.main_window, "pipeline"):
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Camera Config", str(self._defaultCameraConfigDir()),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            import json
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Load failed", f"Could not read file:\n{e}"
            )
            return
        # Install through the pipeline's own loader; it does the parse, the
        # fps clamp and the registry write, so this dialog doesn't need to
        # know how camera configs are stored.
        pipe = self.main_window.pipeline
        if not pipe.install_camera_configs({str(cam_id): raw}):
            QtWidgets.QMessageBox.critical(
                self, "Invalid file",
                "File is not a valid camera config."
            )
            return
        cfg = pipe.get_camera_config(cam_id)
        if not cfg.probed_modes:
            QtWidgets.QMessageBox.warning(
                self, "Empty config",
                "Loaded file has no probed modes. Run Calibrate All Cameras."
            )
            return
        # Reflect the freshly-installed config: if this is the active camera,
        # reload the Capture card; either way seed its default selection.
        self._ensure_capture_defaults(cam_id)
        if str(cam_id) == str(self._active_capture_cam()):
            self._load_capture_for(cam_id)
        QtWidgets.QMessageBox.information(
            self, "Loaded",
            f"Loaded camera {cam_id} ({len(cfg.probed_modes)} probed "
            f"mode(s)) from:\n{path}"
        )

    def _onSaveAllCameras(self):
        """Bundle every known camera's CameraConfig into one JSON file.

        Writes ``{"cameras": {camera_id: CameraConfig.to_json(), ...}}``,
        the same shape ``_onLoadAllCameras`` consumes. Captures the active
        camera's dialog state before serialising so pre-save edits aren't
        lost; other cameras come from the central registry as-is.
        """
        if self.main_window is None or not hasattr(self.main_window, "pipeline"):
            return
        pipe = self.main_window.pipeline
        active = self._activeCameraId()
        if active is not None:
            self._writeDialogSelectionsTo(active)
        all_cfgs = pipe.all_camera_configs()
        if not all_cfgs:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to save",
                "No camera configs in the registry yet. Configure at "
                "least one camera (Detect Max + pick FPS) before Save All."
            )
            return
        suggested = self._defaultCameraConfigDir() / "cameras_all.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save All Cameras", str(suggested),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            import json
            bundle = {
                "cameras": {
                    str(cid): cfg.to_json() for cid, cfg in all_cfgs.items()
                }
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(bundle, f, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Save failed",
                f"Could not write camera bundle:\n{e}"
            )
            return
        logger.info("Saved %d camera config(s) to %s", len(all_cfgs), path)
        QtWidgets.QMessageBox.information(
            self, "Saved",
            f"Saved {len(all_cfgs)} camera config(s) to:\n{path}"
        )

    def _onLoadAllCameras(self):
        """Load a multi-camera bundle and install every entry into the
        central registry under its original camera_id.

        Skips malformed entries with a warning rather than failing the
        whole load. Re-syncs the visible combos to the active camera so
        the user sees the freshly-loaded state immediately.
        """
        if self.main_window is None or not hasattr(self.main_window, "pipeline"):
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load All Cameras", str(self._defaultCameraConfigDir()),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            import json
            with open(path, encoding="utf-8") as f:
                bundle = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Load failed", f"Could not read file:\n{e}"
            )
            return
        cams = bundle.get("cameras") if isinstance(bundle, dict) else None
        if not isinstance(cams, dict) or not cams:
            QtWidgets.QMessageBox.warning(
                self, "Empty bundle",
                "File has no 'cameras' block, nothing to load."
            )
            return
        pipe = self.main_window.pipeline
        n = pipe.install_camera_configs(cams)
        # Reload the active camera's Capture card from the freshly-loaded config.
        self._load_active_camera_settings()
        QtWidgets.QMessageBox.information(
            self, "Loaded",
            f"Loaded {n} camera config(s) from:\n{path}"
        )

    def _firstCameraIdInTable(self):
        """Return the first non-empty integer/string camera ID from the table."""
        for row in range(self.table.rowCount()):
            text = self._camera_id_text(row)
            if not text:
                continue
            try:
                return int(text)
            except ValueError:
                return text
        return None

    def _build_probe_item(self, camera_id):
        """Build a ``_CamItem`` probe spec for ``camera_id`` (backend from its
        row, USB identity resolved), or ``None`` for a blank id."""
        if camera_id is None:
            return None
        from source.video.cameras.usb_identity import resolve_identity
        backend = self._backendForCameraId(camera_id) or "opencv"
        ident = resolve_identity(camera_id, backend)
        return _CamItem({
            "camera_id": camera_id,
            "backend": backend,
            "target_fps": 30.0,
            "pixel_format": self._selected_format_for(camera_id) or "mjpeg",
            # The door the operator picked. Detect measures THAT door, because
            # a measurement taken through another one describes a camera they
            # are not going to open.
            "door": self._doorForCameraId(camera_id) or self._default_door(),
            "unique_id": ident.get("unique_id", f"{camera_id}-{backend}"),
            "identity": ident.get("identity") or {},
            "weak": bool(ident.get("weak", False)),
            "label": f"Camera {camera_id}",
        })

    def _probe_cameras(self, camera_ids) -> set:
        """Probe every id in ``camera_ids`` in ONE background batch + progress
        popup (not one modal per camera). ``_apply_calibration_modes``
        populates each camera's row on completion. Returns the set of ids that
        reported usable modes.

        Used by the connect preflight to calibrate cameras the user hasn't
        calibrated yet; there is no per-camera Detect Max button anymore.
        """
        items = []
        for cid in camera_ids:
            it = self._build_probe_item(cid)
            if it is not None:
                items.append((cid, it))
        if not items:
            return set()
        # A camera that is OPEN cannot be probed. Windows hands a UVC device
        # to one client, so every backend reports "could not open" and the
        # probe returns nothing for all of them, which is indistinguishable
        # from a camera that has no modes. Pressing Detect on a connected
        # camera therefore did nothing at all, said nothing, and left the
        # unmeasured door reading "try" however many times it was pressed.
        released = self._release_for_probe([cid for cid, _it in items])
        # The format decides which sizes and rates exist at all, so it is
        # confirmed with the CAMERA before anything is measured in it. A UVC
        # driver accepts any FOURCC, reports it back unchanged, and then
        # streams whatever it likes; unchecked, every rate on this rig was
        # measured in YUY2 while the dialog said MJPEG.
        items = self._confirm_probe_formats(items)
        if not items:
            return set()
        items = self._confirm_probe_doors(items)
        if not items:
            return set()
        dlg, log_view, bar = self._build_calibration_progress_dialog(len(items))
        for cid, _it in items:
            self._append_cal_log(log_view, f"[cam {cid}] queued...")
        # ENUMERATE FIRST, before the trial probe opens anything. About a
        # second per camera, and it answers the question the trial probe
        # cannot: what the camera OFFERS, as opposed to what it delivered in
        # this room at this moment. See docs/dev/camera-modes-and-rates.md.
        for cid, _it in items:
            self._enumerate_offered(cid, log_view)
        self._run_calibration_probes([it for _c, it in items], log_view, dlg, bar)

        if released:
            self._append_cal_log(
                log_view,
                "released " + ", ".join(f"camera {c}" for c in sorted(released))
                + " to measure it; press Connect again when finished.")
        ok = set()
        for cid, it in items:
            # ``result`` is the merged [(w, h, fps)] list across every backend
            # that was measured, not a ProbeResult. Truthiness IS the test:
            # a camera with no measurable mode is one that was not calibrated.
            if getattr(it, "result", None):
                ok.add(cid)
        if ok:
            marker = getattr(self.main_window, "_project_changed", None) \
                if self.main_window else None
            if callable(marker):
                try:
                    marker(reason="camera_calibrated")
                except Exception:
                    pass
        return ok

    def _confirm_probe_doors(self, items) -> list:
        """Check each item's chosen door can open the camera, and say so.

        Same rule as the format: a door that cannot serve this camera is not
        swapped behind the operator's back. They are told which camera, which
        door, and offered the fallback, because the door changes what the
        camera delivers and is theirs to choose. Measured on this rig, one
        camera reached 29.9 fps through Media Foundation and 19.9 through
        DirectShow, and another was the reverse.
        """
        import cv2

        from source.video.cameras import identity as _ident
        fallback = self._default_door()
        codes = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
                 "v4l2": getattr(cv2, "CAP_V4L2", 0),
                 "any": cv2.CAP_ANY,
                 "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", 0)}
        keep = []
        for cid, item in items:
            door = str(getattr(item, "door", "") or "")
            # A scientific camera has no door; its SDK is the path.
            if not door or self._family_for_camera(cid) != "opencv":
                keep.append((cid, item))
                continue
            if self._door_opens(cid, codes.get(door), _ident):
                keep.append((cid, item))
                continue
            if door == fallback or not self._door_opens(
                    cid, codes.get(fallback), _ident):
                QtWidgets.QMessageBox.warning(
                    self, "Backend cannot open this camera",
                    f"Camera {cid} could not be opened through "
                    f"{self._door_label(door)}."
                    + "\n\n" +
                    "The fallback could not open it either, so this camera "
                    "was not measured. Another process may be holding it.")
                continue
            answer = QtWidgets.QMessageBox.question(
                self, "Backend cannot open this camera",
                f"Camera {cid} could not be opened through "
                f"{self._door_label(door)}."
                + "\n\n" +
                f"Detect through {self._door_label(fallback)} instead?"
                + "\n\n" +
                "Measuring through a door the camera will not open reports "
                "nothing at all, which reads as a camera with no modes.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes)
            if answer != QtWidgets.QMessageBox.Yes:
                continue
            item.door = fallback
            self._write_door_to_row(cid, fallback)
            keep.append((cid, item))
        return keep

    @staticmethod
    def _door_opens(camera_id, code, _ident) -> bool:
        """Can this camera be opened through ``code``? Never raises."""
        import cv2
        if not code:
            return False
        cap = None
        try:
            idx = _ident.address_of(camera_id)
            if idx is None:
                return False
            cap = cv2.VideoCapture(int(idx), int(code))
            return bool(cap.isOpened() and cap.read()[0])
        except Exception as e:
            logger.debug("door test %s on %s: %s", code, camera_id, e)
            return False
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _door_label(self, door) -> str:
        for label, value in self._available_doors():
            if value == door:
                return label
        return str(door or "?")

    def _write_door_to_row(self, camera_id, door) -> None:
        """Point the Backend cell at the door Detect is about to use."""
        row = self._rowForCameraId(camera_id)
        if row is None:
            return
        combo = self.table.cellWidget(row, self.COL_BACKEND)
        if combo is None:
            return
        i = combo.findData(door)
        if i < 0:
            return
        was = combo.signalsBlocked()
        combo.blockSignals(True)
        try:
            combo.setCurrentIndex(i)
        finally:
            combo.blockSignals(was)
        self._on_row_capture_changed(row)

    def _confirm_probe_formats(self, items) -> list:
        """Check each item's chosen format against the camera, and say so.

        Returns the items to go on and probe, with ``pixel_format`` set to
        something the camera actually delivers. A camera that cannot serve the
        chosen format is NOT silently switched: the operator is told which
        camera, which format, and what it delivered instead, and chooses.
        """
        from source.video.cameras.opencv import OpenCVCamera

        keep = []
        for cid, item in items:
            want = str(getattr(item, "pixel_format", "") or "mjpeg").lower()
            try:
                ok, got = OpenCVCamera.supports_format(cid, want)
            except Exception as e:
                logger.warning("camera %s: could not test %s: %s", cid, want, e)
                keep.append((cid, item))          # do not block on a bad test
                continue
            if ok:
                keep.append((cid, item))
                continue
            other = next((f for f in OpenCVCamera.PROBE_FORMATS
                          if f != want), "yuy2")
            try:
                other_ok, _ = OpenCVCamera.supports_format(cid, other)
            except Exception:
                other_ok = False
            if not other_ok:
                QtWidgets.QMessageBox.warning(
                    self, "Format not supported",
                    f"Camera {cid} does not deliver {want.upper()}"
                    + (f" (it returned {got})." if got else ".")
                    + f"\n\n{other.upper()} was tried as well and "
                      "also could not be confirmed, so this camera was "
                      "not measured.")
                continue
            answer = QtWidgets.QMessageBox.question(
                self, "Format not supported",
                f"Camera {cid} does not deliver {want.upper()}"
                + (f" (it returned {got})." if got else ".")
                + f"\n\nDetect in {other.upper()} instead?\n\n"
                  "Measuring in a format the camera will not serve reports "
                  "rates it cannot hold.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes)
            if answer != QtWidgets.QMessageBox.Yes:
                continue
            item.pixel_format = other
            self._write_format_to_row(cid, other)
            keep.append((cid, item))
        return keep

    def _write_format_to_row(self, camera_id, pixel_format) -> None:
        """Point the Format cell at what Detect is about to measure, so the
        table never disagrees with the numbers beside it."""
        row = self._rowForCameraId(camera_id)
        if row is None:
            return
        try:
            _res, _fps, fmt, _flip = self._row_capture_widgets(row)
        except Exception:
            return
        if fmt is None:
            return
        i = fmt.findData(pixel_format)
        if i < 0:
            return
        was = fmt.signalsBlocked()
        fmt.blockSignals(True)
        try:
            fmt.setCurrentIndex(i)
        finally:
            fmt.blockSignals(was)
        self._on_row_capture_changed(row)

    def _enumerate_offered(self, camera_id, log_view=None) -> int:
        """Ask the platform what ``camera_id`` offers and record it.

        Returns how many (size, format) combinations were found; 0 means the
        question could not be ASKED, in which case nothing is written and the
        trial probe still runs.
        """
        try:
            from source.video.cameras import calibration_store as _store
            from source.video.cameras import identity as _ident
            from source.video.cameras.enumerate_modes import enumerate_modes
            from source.video.cameras.usb_identity import resolve_identity

            devices = list(_ident.list_dshow_devices() or ())
            index = None
            for i in range(len(devices)):
                if str(_ident.device_id_for_index(i)) == str(camera_id):
                    index = i
                    break
            if index is None:
                # Not every id is a device-path id. A camera the OS will not
                # describe carries a capability fingerprint (``fp<hash>#<n>``)
                # instead, and a project saved by an older version may still
                # hold a bare index. Both name a real camera and both can be
                # enumerated, so the id is resolved to an index the way an
                # open resolves it. Handing an id ffmpeg has never heard of
                # to ffmpeg as a device name, which is what this did, could
                # only ever fail, and it failed silently: no modes, nothing
                # written, and a Detect that appeared to run.
                index = _ident.address_of(camera_id)
            device = None
            if index is not None and 0 <= int(index) < len(devices):
                # The PATH, not the friendly name. Two cameras of the same
                # model report the same name (this rig has "HD USB Camera"
                # twice), so a name picks whichever ffmpeg lists first and one
                # camera's modes get stored under the other camera's id. The
                # ``@device_pnp_`` path is unique per device.
                entry = devices[int(index)]
                device = entry.get("path") or entry.get("name")
            if (not device and index is not None
                    and os.path.exists(f"/dev/video{int(index)}")):
                # Linux has no DirectShow list, so the lookup above always came
                # back empty there and no camera was ever enumerated. The FPS
                # list then offered rates the camera does not have: 30 fps at a
                # 640x480 mode that is 120 fps only. On Linux the index IS the
                # V4L2 node, and enumerate_modes reads its exact intervals with
                # v4l2-ctl. Windows lists its devices and has no /dev/video
                # nodes, so it never reaches this.
                device = f"/dev/video{int(index)}"
            if not device:
                logger.warning(
                    "camera %s could not be matched to a video device, so its "
                    "offered modes were not read.", camera_id)
                if log_view is not None:
                    self._append_cal_log(
                        log_view,
                        f"[cam {camera_id}] no matching video device; offered "
                        f"modes NOT read, measuring only")
                return 0
            modes = enumerate_modes(device)
            if not modes:
                logger.warning(
                    "camera %s (%s) reported no offered modes.", camera_id,
                    device)
                if log_view is not None:
                    self._append_cal_log(
                        log_view,
                        f"[cam {camera_id}] device found but reported no "
                        f"modes; measuring only")
                return 0
            ident = resolve_identity(camera_id, "opencv")
            unique_id = ident.get("unique_id")
            if not _store.put_offered(unique_id, modes):
                # The modes were read and then not kept. Nothing downstream can
                # tell that apart from a camera that was never enumerated, so
                # it is said out loud rather than returned as a success.
                logger.warning(
                    "camera %s: read %d offered mode(s) but could not store "
                    "them under %s.", camera_id, len(modes), unique_id)
                if log_view is not None:
                    self._append_cal_log(
                        log_view,
                        f"[cam {camera_id}] offered modes could NOT be saved")
                return 0
            sizes = len({(m.width, m.height) for m in modes})
            if log_view is not None:
                self._append_cal_log(
                    log_view,
                    f"[cam {camera_id}] offers {sizes} size(s) in "
                    f"{len({m.pixel_format for m in modes})} format(s)")
            return len(modes)
        except Exception as e:
            logger.warning("enumerating camera %s failed: %s", camera_id, e)
            if log_view is not None:
                self._append_cal_log(
                    log_view, f"[cam {camera_id}] enumeration failed: {e}")
            return 0

    def _release_for_probe(self, camera_ids) -> set:
        """Disconnect any camera in ``camera_ids`` that is currently open.

        Returns the ids released, so the operator can be told: a probe that
        silently disconnects a running camera is worse than one that refuses.
        """
        if self.main_window is None:
            return set()
        try:
            bcm = dict(self.main_window.video_manager.box_camera_map or {})
        except Exception:
            return set()
        wanted = {str(c) for c in camera_ids}
        released = set()
        for setup_id, cam_id in bcm.items():
            if str(cam_id) not in wanted:
                continue
            try:
                self.main_window.disconnect_camera(setup_id)
                released.add(str(cam_id))
            except Exception as e:
                logger.warning("could not release camera %s on box %s before "
                               "probing: %s", cam_id, setup_id, e)
        return released

    def getSelectedBoxesWithCameras(self):
        """Get list of selected boxes with their camera IDs and backend info"""
        selected = []

        if not self.main_window or not hasattr(self.main_window, 'get_all_setup_widgets'):
            return selected

        connected_boxes = list(self.main_window.get_all_setup_widgets())

        # Connection is gated by Camera ID presence, every row with a
        # non-empty Camera ID participates.
        for i in range(self.table.rowCount()):
            if i >= len(connected_boxes):
                continue
            camera_id_text = self._camera_id_text(i)
            if not camera_id_text:
                continue

            # Get backend selection
            backend_combo = self.table.cellWidget(i, 2)
            backend_display = backend_combo.currentText() if backend_combo else "OpenCV"
            backend_id = self._backend_name_to_id(backend_display)

            # An identity ("fp3557d2de") is resolved to a live index at open
            # time, rejecting anything non-numeric here silently dropped every
            # enumerated camera from the connect. Numeric ids stay ints because
            # they are used as dict keys downstream.
            if not str(camera_id_text).strip():
                logger.warning("Row %s has no camera ID", i)
                continue
            camera_id = camera_id_text
            if isinstance(camera_id, str) and camera_id.strip().isdigit():
                camera_id = int(camera_id.strip())

            save_checkbox = (self.save_video_checkboxes[i]
                             if i < len(self.save_video_checkboxes) else None)

            # Scientific-camera I/O comes from THIS camera's persisted
            # CameraConfig (exposure/gain/trigger/strobe are per-camera now,
            # edited in the sci-settings group and saved on change), not a
            # single global widget value shared across cameras.
            camera_config = None
            if backend_id != "opencv":
                pipe = getattr(self.main_window, "pipeline", None)
                cfg = (pipe.all_camera_configs().get(str(camera_id))
                       if pipe is not None else None)
                if cfg is not None:
                    camera_config = {
                        "exposure_us": cfg.exposure_us,
                        "gain_db": cfg.gain_db,
                        "trigger": cfg.trigger.to_json(),
                        "line_output": cfg.line_output.to_json(),
                    }

            selected.append({
                'box_widget': connected_boxes[i],
                'box_number': connected_boxes[i].setup_number,
                'camera_id': camera_id,
                'save_video': bool(save_checkbox.isChecked()) if save_checkbox else True,
                'camera_backend': backend_id,
                'camera_config': camera_config,
            })

        return selected

    def connect_cameras(self):
        """Handle camera connection.

        Orchestrator only, each phase delegates to a named helper so the
        flow (validate -> preflight -> push -> ROI prompt -> segment config
        -> connect -> finalize) stays debuggable from one screen.
        """
        camera_id_map = self._collect_and_dedupe_selection()
        if camera_id_map is None:
            return
        # Persist the active camera's picks first, THEN validate every camera.
        self._push_dialog_selections_to_pipeline(camera_id_map)
        if not self._preflight_resolution_and_fps(camera_id_map):
            return
        if not self._prompt_roi_for_pending_cameras(camera_id_map):
            return
        self._build_and_publish_segment_config(camera_id_map)
        successful, failed = self._connect_each_camera(camera_id_map)
        self._finalize_connection_results(camera_id_map, successful, failed)

    # ------------------------------------------------------------------
    # connect_cameras helpers
    # ------------------------------------------------------------------
    def _collect_and_dedupe_selection(self):
        """Return camera_id -> [items] map, or None if nothing valid was picked."""
        selected = self.getSelectedBoxesWithCameras()
        if not selected:
            # Point at the section that fixes it rather than only complaining.
            self._warn_assign = True
            self._goto_step(0)
            QtWidgets.QMessageBox.warning(
                self,
                "No Selection",
                "Please select at least one box and enter a valid camera ID."
            )
            return None
        self._warn_assign = False
        camera_id_map: dict = {}
        for item in selected:
            # Keyed by identity: a row still holding a legacy index and a row
            # holding the identity of that same camera must land in ONE group,
            # or the shared-camera ROI split silently becomes two cameras.
            key = self._canonical_camera_id(item['camera_id'])
            item['camera_id'] = key
            camera_id_map.setdefault(key, []).append(item)
        return camera_id_map

    def _preflight_resolution_and_fps(self, camera_id_map: dict) -> bool:
        """Verify EVERY camera in the batch has its own resolution + FPS.

        Each camera's Resolution/FPS come from its OWN table row. A camera
        with no probed modes yet is calibrated on the fly (which fills its
        row); the user then can't proceed until each camera has both set.
        """
        pipe = (getattr(self.main_window, "pipeline", None)
                if self.main_window is not None else None)

        # Persist each row's current picks into its CameraConfig first.
        for cam_id in camera_id_map:
            try:
                self._writeDialogSelectionsTo(cam_id)
            except Exception as e:
                logger.debug("persist row for %s: %s", cam_id, e)

        def _has_res_fps(cid):
            cfg = pipe.get_camera_config(cid) if pipe is not None else None
            return (getattr(cfg, "selected_resolution", None)
                    and getattr(cfg, "selected_fps", None))

        # Calibrate every uncalibrated camera in ONE batch probe (not one
        # modal per camera), then persist their freshly-filled rows.
        uncalibrated = [cid for cid in camera_id_map if not _has_res_fps(cid)]
        if uncalibrated:
            self._probe_cameras(uncalibrated)
            for cam_id in uncalibrated:
                try:
                    self._ensure_capture_defaults(cam_id)
                    self._writeDialogSelectionsTo(cam_id)
                except Exception as e:
                    logger.debug("persist row for %s: %s", cam_id, e)
            # Reflect any freshly-probed modes in the active Capture card.
            self._load_capture_for(self._active_capture_cam())

        for cam_id in camera_id_map:
            if not _has_res_fps(cam_id):
                QtWidgets.QMessageBox.critical(
                    self, "Camera not configured",
                    f"Camera {cam_id} has no resolution / FPS set.\n\n"
                    "Click 'Calibrate All Cameras', then pick a Resolution "
                    "and Target FPS in that camera's row before connecting."
                )
                return False

        # Manager default + ROI-preview hint come from the first camera (the
        # ROI prompt sets each camera's own resolution anyway).
        first_id = next(iter(camera_id_map), None)
        first_cfg = (pipe.get_camera_config(first_id)
                     if pipe is not None and first_id is not None else None)
        resolution = getattr(first_cfg, "selected_resolution", None) if first_cfg else None
        self._target_fps = getattr(first_cfg, "selected_fps", None) if first_cfg else None
        if self.main_window is not None:
            if resolution is not None:
                self.main_window.video_camera_resolution = resolution
            self.main_window.video_camera_realistic_fps = (
                self._realistic_fps_for(first_cfg, resolution))
            pipe = getattr(self.main_window, "pipeline", None)
            if resolution is not None and pipe is not None:
                pipe.set_capture_defaults(target_resolution=resolution)
        return True

    @staticmethod
    def _realistic_fps_for(cfg, resolution):
        """Measured ceiling FPS for ``resolution`` from ``cfg.probed_modes``,
        or ``None`` when unknown."""
        if cfg is None or resolution is None:
            return None
        for mode in getattr(cfg, "probed_modes", None) or []:
            try:
                if (int(mode[0]), int(mode[1])) == tuple(resolution):
                    fps = float(mode[2])
                    return fps if fps > 0 else None
            except (TypeError, ValueError, IndexError):
                continue
        return None

    def _push_dialog_selections_to_pipeline(self, camera_id_map: dict) -> None:
        """Write each camera's own row selections into its central
        CameraConfig BEFORE camera start, so recorder PTS /
        apply_tracking_config / etc. read the values the user confirmed.
        Every camera is independent, no active-camera concept."""
        if self.main_window is None or not hasattr(self.main_window, "pipeline"):
            return
        for cam_id in camera_id_map:
            try:
                self._writeDialogSelectionsTo(cam_id)
            except Exception as e:
                logger.debug("push selections for %s: %s", cam_id, e)

    def _prompt_roi_for_pending_cameras(self, camera_id_map: dict) -> bool:
        """Prompt ROI for any camera lacking a roi_segment.

        Returns False iff the user cancelled (caller should bail). Every
        camera gets an ROI prompt on first connect, not just shared ones.
        """
        cameras_needing_config = {
            cid: items
            for cid, items in camera_id_map.items()
            if not all(
                hasattr(item['box_widget'], 'roi_segment')
                and item['box_widget'].roi_segment is not None
                for item in items
            )
        }
        if not cameras_needing_config:
            return True

        # A camera ONE box looks through needs no region: the frame already
        # is that box, so the whole frame is the honest default and demanding
        # a drawing before Connect blocks a rig that has nothing to split.
        # It still gets a region, because buildSegmentConfig drops a box that
        # has none, so the default is written rather than left absent.
        # Regions can be drawn later from "Define regions" whenever framing
        # the box more tightly is worth it.
        dedicated = [cid for cid, items in cameras_needing_config.items()
                     if len(items) == 1]
        for cid in dedicated:
            items = cameras_needing_config.pop(cid)
            for item in items:
                w = item['box_widget']
                w.roi_normalized = (0.0, 0.0, 1.0, 1.0)
                w.roi_segment = None
                logger.info(
                    "Camera %s serves box %s alone, so its region defaults to "
                    "the whole frame. Draw one from Define regions to crop "
                    "tighter.", cid, item['box_number'])
        if dedicated:
            self._roi_state_changed()
        if not cameras_needing_config:
            return True

        # What is left is SHARED, and a shared camera genuinely needs the
        # split: without it cameras 2..N connect with no region, their boxes
        # are dropped by buildSegmentConfig, and the disabled Connect button
        # traps the user.
        # One region size across every camera, carried from each editor to
        # the next. Without the chaining each camera locked only to its own
        # first ROI, so the boxes ended up different sizes and the pose model
        # letterboxed all but the largest.
        locked = self._existing_roi_size()
        refused = []
        for camera_id, items in cameras_needing_config.items():
            logger.info(
                f"Camera {camera_id} ({len(items)} box"
                f"{'es' if len(items) > 1 else ''}) - ROI required"
            )
            # Point the ROI preview at THIS camera's own resolution (not the
            # active camera's) so the drawn ROIs match the frame it delivers.
            pipe = (getattr(self.main_window, "pipeline", None)
                    if self.main_window is not None else None)
            if pipe is not None:
                try:
                    ccfg = pipe.get_camera_config(camera_id)
                    if getattr(ccfg, "selected_resolution", None):
                        self.main_window.video_camera_resolution = ccfg.selected_resolution
                except Exception:
                    pass
            config_dialog = ConfigSelectionDialog(camera_id, len(items), self)
            if config_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return False  # user cancelled → bail the whole batch
            action = config_dialog.selected_action
            if action == "create":
                seg_dialog = ROISegmentationDialog(camera_id, items,
                                                   self.main_window,
                                                   locked_size=locked)
                if seg_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                    return False
                self.applySegments(
                    items, seg_dialog.segments,
                    camera_resolution=seg_dialog.frame_size,
                )
                if getattr(seg_dialog, "lock_rejected", False):
                    refused.append((camera_id, seg_dialog.frame_size))
                locked = seg_dialog.locked_size() or locked
            elif action == "select":
                if not self.loadExistingConfig(items):
                    return False  # cancelled picker / load failed → bail batch
        self._warn_roi_sizes_differ(refused, locked)
        return True

    def _existing_roi_size(self):
        """The region size already drawn on this rig, in pixels, or None.

        A camera added to a rig that already has ROIs must match them, not
        start a second size of its own.
        """
        try:
            for item in self.getSelectedBoxesWithCameras() or ():
                seg = getattr(item.get("box_widget"), "roi_segment", None)
                if seg and len(seg) == 4 and seg[2] and seg[3]:
                    return (int(seg[2]), int(seg[3]))
        except Exception as e:
            logger.debug("existing ROI size lookup: %s", e)
        return None

    def _warn_roi_sizes_differ(self, refused, locked):
        """Say so when a camera could not take the shared region size."""
        if not refused or not locked:
            return
        listing = "\n".join(f"  camera {c}: {s[0]}x{s[1]}" for c, s in refused)
        QtWidgets.QMessageBox.warning(
            self, "Region size could not be shared",
            f"The region size fixed on the first camera "
            f"({locked[0]}x{locked[1]} px) does not fit:\n\n{listing}\n\n"
            "Those boxes keep their own size, so the tracker pads every box "
            "to the largest region: each box pays the largest box's "
            "inference cost and the padded ones lose resolution.\n\n"
            "Either raise those cameras' resolution, or Reset All and draw a "
            "smaller first region.")

    def _build_and_publish_segment_config(self, camera_id_map: dict) -> None:
        """Merge per-camera segment configs into one dict on main_window.

        Every camera with at least one ROI'd box needs cropping, including
        single-box dedicated cameras.
        """
        merged_boxes: list = []
        for camera_id, items in camera_id_map.items():
            cam_cfg = self.buildSegmentConfig(camera_id, items)
            if cam_cfg.get('boxes'):
                merged_boxes.extend(cam_cfg['boxes'])
                logger.info(
                    f"Built segment config for camera {camera_id} with "
                    f"{len(cam_cfg['boxes'])} box(es)"
                )
        if self.main_window:
            self.main_window.video_segment_config = (
                {'boxes': merged_boxes} if merged_boxes else None
            )

    def _connect_each_camera(self, camera_id_map: dict) -> tuple:
        """Connect every camera. For each camera_id:
            - dedicated (1 box): connectSingleCamera
            - shared (N boxes): connectSingleCamera + register extras so they
              share the same FrameBus + per-box segmentation.
        Returns (successful, failed) counts.
        """
        successful = 0
        failed = 0
        for camera_id, items in camera_id_map.items():
            first_item = items[0]
            if not self.connectSingleCamera(first_item):
                failed += len(items)
                continue
            successful += 1
            for item in items[1:]:
                # Guard each shared-camera bind so one failure doesn't abort
                # the whole Connect batch or get miscounted as a success.
                try:
                    self._bind_extra_box_to_shared_camera(camera_id, item)
                    successful += 1
                except Exception as e:
                    logger.error(
                        "bind extra box %s to shared camera %s: %s",
                        item.get('box_number'), camera_id, e)
                    failed += 1
        self.connected_count = successful
        return successful, failed

    def _bind_extra_box_to_shared_camera(self, camera_id, item) -> None:
        """Wire a non-first box of a shared camera into the pipeline.

        Thin wrapper over ``MainWindowBase._bind_box_to_shared_camera`` so
        the dialog and auto-connect share one bind implementation
        (box_camera_map + register_box + bus register + sink subscribe +
        widget id update)."""
        self.main_window._bind_box_to_shared_camera(item['box_number'], camera_id)

    def _finalize_connection_results(self, camera_id_map: dict,
                                     successful: int, failed: int) -> None:
        """Show success/failure dialog, persist video settings, stamp
        user_applied on each camera's config, mark project dirty, accept.
        """
        if successful <= 0:
            logger.error("All camera connections failed")
            self.updateButtonStates()
            QtWidgets.QMessageBox.warning(
                self,
                "Connection Failed",
                "Failed to connect any cameras. Please check camera IDs and ROI segments."
            )
            return

        self._save_video_settings_to_main_window()
        message = f"Successfully connected {successful} camera(s)."
        if failed > 0:
            message += f"\n{failed} camera(s) failed to connect."
        logger.info(
            f"Camera connections complete: {successful} succeeded, {failed} failed"
        )
        self.updateButtonStates()
        QtWidgets.QMessageBox.information(self, "Cameras Connected", message)
        self._stamp_user_applied(camera_id_map)
        self._persist_auto_connect_pref()
        try:
            self.main_window._project_changed(reason="camera_config_changed")
        except Exception:
            pass
        self.accept()

    def _persist_auto_connect_pref(self) -> None:
        """Write the auto-connect checkbox onto the active project config's
        meta; the next autosave persists it."""
        meta = getattr(getattr(self.main_window, "_active_config", None),
                       "meta", None)
        chk = getattr(self, "auto_connect_check", None)
        if meta is not None and chk is not None:
            meta.auto_connect_cameras_on_load = bool(chk.isChecked())

    def _save_video_settings_to_main_window(self) -> None:
        """Persist the rig-level target FPS after a successful connect.
        Everything per-camera (backend, resolution, format) lives on the
        per-camera CameraConfig, no first-row-wins globals. Strategy is
        always "accept" (the window default) and grayscale is owned by
        the Tracking Config dialog.
        """
        if not self.main_window:
            return
        self.main_window.video_target_fps = getattr(
            self, "_target_fps", self.getSelectedFPS()
        )
        logger.info(
            f"Video settings: FPS={self.main_window.video_target_fps}")

    def _stamp_user_applied(self, camera_id_map: dict) -> None:
        """Mark every successfully-connected camera's CameraConfig as
        user_applied so the project save filter persists only cameras the
        user actually confirmed (not pipeline-template placeholders).
        """
        try:
            pipe = self.main_window.pipeline
            for cam_id in camera_id_map:
                cc = pipe.get_camera_config(cam_id)
                if cc is not None:
                    cc.user_applied = True
        except Exception as e:
            logger.debug("user_applied stamp failed: %s", e)

    def buildSegmentConfig(self, camera_id, items):
        """Build video segment configuration for a shared camera.

        Single source of truth: ``box_widget.roi_normalized`` (percent of
        the frame the ROI was drawn against). ``extract_segment``
        denormalizes against the camera's current frame.shape at runtime,
        so the segment adapts to whatever resolution the camera delivers.

        ``camera_resolution`` and pixel ROI are not stored on disk, percent
        is camera-agnostic and avoids stale-resolution coupling.
        """
        boxes = []
        for item in items:
            setup_widget = item['box_widget']
            setup_number = item['box_number']

            roi_norm = getattr(setup_widget, 'roi_normalized', None)
            if not roi_norm or len(roi_norm) != 4:
                # No normalized ROI; skip the box if no pixel ROI either.
                seg = getattr(setup_widget, 'roi_segment', None)
                if not seg:
                    logger.warning(f"Box {setup_number}: no ROI defined, skipping")
                    continue
                # Can't normalize a pixel ROI without a camera frame
                # reference; roi_normalized should already be set.
                logger.error(
                    f"Box {setup_number}: roi_segment set but roi_normalized "
                    f"missing, frame size unknown.  Re-draw ROI."
                )
                continue

            px, py, pw, ph = roi_norm
            box_config = {
                'box_id': setup_number,
                'box_number': setup_number,
                'geometry': {
                    'percent': {
                        'x': float(px), 'y': float(py),
                        'width': float(pw), 'height': float(ph),
                    },
                },
            }
            boxes.append(box_config)
            logger.debug(
                f"Box {setup_number}: percent=({px:.3f}, {py:.3f}, "
                f"{pw:.3f}, {ph:.3f})"
            )

        config = {'boxes': boxes}
        logger.info(f"Built segment config for camera {camera_id}: {len(boxes)} boxes (normalized)")
        return config

    @staticmethod
    def _wait_event_pumping(event, timeout=8.0):
        """Wait for a threading.Event up to ``timeout`` s while keeping the Qt
        event loop alive, so a per-camera connect wait doesn't freeze the GUI.
        Returns True if the event was set before the deadline."""
        import time as _t
        deadline = _t.monotonic() + float(timeout)
        while not event.is_set() and _t.monotonic() < deadline:
            # Process pending events for up to ~50 ms, then re-check.
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
            if not event.is_set():
                _t.sleep(0.02)
        return event.is_set()

    def connectSingleCamera(self, item):
        """Connect a single camera to a box. Returns True if successful."""
        try:
            setup_widget = item['box_widget']
            camera_id = item['camera_id']
            setup_number = item['box_number']
            camera_backend = item.get('camera_backend', 'opencv')
            camera_config = item.get('camera_config', None)

            # Persist save-video preference on the box widget
            if hasattr(setup_widget, "save_video_enabled"):
                setup_widget.save_video_enabled = item.get("save_video", True)

            # Persist backend info on box widget for later reconnection
            setup_widget._camera_backend = next(
                (name for name in self._available_backends
                 if self._backend_name_to_id(name) == camera_backend),
                "OpenCV"
            )
            setup_widget._camera_config = camera_config

            # Update camera ID in box widget
            if hasattr(setup_widget, 'camera_id_edit'):
                setup_widget.camera_id_edit.setText(str(camera_id))

            # Connect via the main window, then verify through the
            # Pipeline's PUBLIC connect-state read (up to 8 s for driver
            # retries), the dialog keeps only messaging, no thread
            # internals.
            if hasattr(self.main_window, 'connect_camera'):
                self.main_window.connect_camera(
                    setup_number,
                    camera_backend=camera_backend,
                    camera_config=camera_config,
                )
                pipe = getattr(self.main_window, "pipeline", None)
                if pipe is None:
                    return False
                # Pump events while polling instead of a blocking wait
                # (which would stall the UI up to 8 s × N cameras).
                import time as _t
                deadline = _t.monotonic() + 8.0
                state = pipe.camera_connect_state(setup_number)
                while state == "connecting" and _t.monotonic() < deadline:
                    QtWidgets.QApplication.processEvents(
                        QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
                    _t.sleep(0.02)
                    state = pipe.camera_connect_state(setup_number)

                if state == "connected":
                    logger.info(f"Successfully connected camera {camera_id} "
                                f"(backend={camera_backend}) to Box {setup_number}")
                    return True
                if state == "connecting":
                    # Timed out while ``begin_capturing`` is still retrying
                    # (common on slow opens). Keep the box→camera mapping so
                    # ``_wait_for_camera_streaming`` picks up late frames.
                    logger.info(
                        f"Camera {camera_id} still opening for Box {setup_number} "
                        f"after 8s, leaving mapping in place so frames "
                        "route correctly once the driver finishes."
                    )
                    return True
                if state == "failed":
                    logger.warning(
                        f"Camera {camera_id} thread exited without connecting "
                        f"for Box {setup_number}")
                    pipe.unbind_box_camera(setup_number)
                    return False
                logger.warning(
                    "Failed to connect camera %s to Box %s",
                    camera_id, setup_number)
                return False

            return False
        except Exception as e:
            logger.error(f"Error connecting camera to box {item.get('box_number', '?')}: {e!s}")
            return False

    def applySegments(self, items, segments, camera_resolution=None):
        """Apply ROI segments to boxes, also stores roi_normalized for
        resolution-independent ROI (matches maze's applySegments behavior)."""
        for item in items:
            setup_number = item['box_number']
            if setup_number in segments:
                segment = segments[setup_number]

                # Ensure segment is in tuple format (x, y, width, height)
                if isinstance(segment, dict):
                    segment = (
                        segment.get('x', 0),
                        segment.get('y', 0),
                        segment.get('width', 0),
                        segment.get('height', 0)
                    )
                elif isinstance(segment, list):
                    segment = tuple(segment)

                logger.info(f"Applied ROI segment to Box {setup_number}: {segment}")
                setup_widget = item['box_widget']
                setup_widget.roi_segment = segment
                # Compute normalized ROI from the dialog's frame size.
                # ``roi_normalized`` is canonical; pixel ``roi_segment``
                # is a transient hint only.
                if camera_resolution and len(camera_resolution) == 2:
                    cam_w, cam_h = camera_resolution
                    if cam_w > 0 and cam_h > 0:
                        setup_widget.roi_normalized = (
                            segment[0] / cam_w,
                            segment[1] / cam_h,
                            segment[2] / cam_w,
                            segment[3] / cam_h,
                        )
                        logger.info("Box %s roi_normalized: %s",
                                    setup_number,
                                    setup_widget.roi_normalized)
        self._roi_state_changed()

    def _roi_state_changed(self):
        """Repaint everything that reports ROI state.

        Called from the single writer (``applySegments``) and the single
        eraser (``clearAllROIs``), so every route, the Draw regions button,
        the connect-time ROI step, Clear All ROIs, lands here.

        Without it the per-box chips only refreshed when the operator
        *switched to* the regions tab, so drawing a region while already on
        that tab left it reading "to draw" in warning yellow until they
        navigated away and back.
        """
        for name in ("_refresh_regions_note", "_refresh_review",
                     "_refresh_marks"):
            fn = getattr(self, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception as e:
                    logger.debug("%s after ROI change failed: %s", name, e)

    def loadExistingConfig(self, items) -> bool:
        """Load existing camera configuration.

        Returns True when a config was loaded, False when the user cancelled
        the file picker or the load failed, so the connect-time ROI loop can
        bail the whole batch instead of silently advancing to the next camera.
        """
        # Open file dialog to select config
        base_dir = self._defaultCameraConfigDir()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Camera Configuration",
            str(base_dir) if base_dir else "",
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return False  # user cancelled the picker

        try:
            import json
            with open(file_path) as f:
                config = json.load(f)

            # Read percent-only schema (the on-disk format). Configs
            # carrying ``pixel`` + ``camera_resolution`` are handled
            # below; percent is camera-agnostic.
            applied = 0
            for box_num_str, segment in config.get('segments', {}).items():
                try:
                    bn = int(box_num_str)
                except (TypeError, ValueError):
                    continue
                item = next(
                    (it for it in items if it['box_number'] == bn),
                    None,
                )
                if item is None:
                    continue
                bw = item['box_widget']
                if isinstance(segment, dict) and 'percent' in segment:
                    pc = segment['percent']
                    bw.roi_normalized = (
                        float(pc.get('x', 0)),
                        float(pc.get('y', 0)),
                        float(pc.get('width', 0)),
                        float(pc.get('height', 0)),
                    )
                    # roi_segment (pixel) gets re-derived at extract time
                    # from the live camera; clear any stale value.
                    bw.roi_segment = None
                    applied += 1
                elif isinstance(segment, dict) and 'pixel' in segment:
                    # Pixel-only file with camera_resolution.
                    cr = config.get('camera_resolution') or {}
                    cw = cr.get('width') or 0
                    ch = cr.get('height') or 0
                    if cw > 0 and ch > 0:
                        px = segment['pixel']
                        bw.roi_normalized = (
                            px['x'] / cw, px['y'] / ch,
                            px['width'] / cw, px['height'] / ch,
                        )
                        bw.roi_segment = None
                        applied += 1

            if applied == 0:
                # File parsed but no segment matched a pending box → those
                # boxes still have no ROI. Treat as "not configured" so the
                # connect batch doesn't advance them silently.
                QtWidgets.QMessageBox.warning(
                    self, "No matching ROIs",
                    "The selected file has no ROI for the camera(s) being "
                    "configured. Draw ROIs, or pick a matching file."
                )
                return False

            logger.info(
                f"Loaded camera configuration from {file_path} "
                f"({applied} ROI(s))")
            return True

        except Exception as e:
            logger.error(f"Error loading config: {e!s}")
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load configuration: {e!s}"
            )
            return False

    def clearAllROIs(self):
        """Clear all ROIs for every box.

        Wipes the canonical ``roi_normalized`` percent storage AND the
        transient ``roi_segment`` pixel cache on every box widget, plus
        the per-camera ``video_segment_config``. ``roi_normalized`` must be
        nulled or the next reload would restore the ROI from it.
        """
        try:
            reply = QtWidgets.QMessageBox.question(
                self,
                'Confirm Clear',
                'Are you sure you want to clear all ROIs? This will remove ROI data for every box.',
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No
            )

            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                if not self.main_window or not hasattr(self.main_window, 'get_all_setup_widgets'):
                    return

                cleared_count = 0
                for setup_widget in self.main_window.get_all_setup_widgets():
                    touched = False
                    # Canonical percent storage, must be wiped or the
                    # next reload restores the ROI.
                    if hasattr(setup_widget, 'roi_normalized') and setup_widget.roi_normalized:
                        setup_widget.roi_normalized = None
                        touched = True
                    # Pixel cache, derived from roi_normalized at extract
                    # time but may be set transiently.
                    if hasattr(setup_widget, 'roi_segment') and setup_widget.roi_segment:
                        setup_widget.roi_segment = None
                        touched = True
                    if touched:
                        cleared_count += 1

                # Wipe per-camera segment config (which boxes share which
                # camera + percent ROIs).
                if hasattr(self.main_window, 'video_segment_config'):
                    self.main_window.video_segment_config = None

                logger.info("Cleared ROIs for %d box(es)", cleared_count)
                # Back to "to draw" yellow before the confirmation box, so the
                # chips have already changed when the operator dismisses it.
                self._roi_state_changed()
                QtWidgets.QMessageBox.information(
                    self,
                    "ROIs Cleared",
                    f"ROI data cleared for {cleared_count} box(es)."
                )

        except Exception as e:
            logger.error(f"Error clearing ROIs: {e!s}")
            QtWidgets.QMessageBox.critical(
                self,
                "Clear Error",
                f"Failed to clear ROIs: {e!s}"
            )


