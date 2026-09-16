"""Look at the tracking. On the video. Frame by frame.

Every silent-wrong-number defect this pipeline had would have been obvious in
five seconds here: a timestamp of "row number in milliseconds" shows up as a
``dt`` of 1 ms, a lost zone shows as ``location: none`` on every frame, a lost
calibration shows as a speed in px/s, and body parts called ``bp_0`` show up
as body parts called ``bp_0``.

It reads a pose stream and a video and draws one on the other. It computes
nothing, every number shown is read from the file being verified, which is
the point: if the number is wrong here, it is wrong in the workbook.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,
                               QSizePolicy,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

import logging

logger = logging.getLogger(__name__)

#: The palette, the skeleton and the drawing itself now live in
#: `engine.annotate`, which has no Qt in it, so the batch pipeline can write
#: an annotated video without a widget. Imported rather than copied: this
#: view's export must show exactly what the preview showed, and the surest way
#: to keep two renderers agreeing is not to have two.
# ``_BP_COLORS`` is a deliberate RE-EXPORT, not a leftover: it is unused inside
# this module, so a blanket "remove unused imports" pass deletes it and the two
# renderers quietly stop sharing a palette. Pinned by
# ``test_annotated_video.py::TestOnlyOneRenderer``.
from tools.offline_analysis.engine.annotate import (  # noqa: E402,F401
    BP_COLORS as _BP_COLORS,
    bp_colour as _bp_colour,
    skeleton_pairs as _skeleton_pairs,
)


#: Parts that stand in for the head, best first. The facing angle is measured
#: FROM this point, so a recording that names no head part draws no cone --
#: guessing one would invent an orientation the workbook never scored.
_HEAD_NAMES = ("snout", "nose", "head", "neck")

#: The interaction overlay, in the rig's own colours (BGR): yellow is the
#: outward edge of the object's zone, orange the animal's heading and its line
#: to that edge, green a frame that counts as investigation.
_IA_EDGE = (0, 255, 255)
_IA_RAY = (0, 140, 255)
_IA_HIT = (0, 255, 0)


def _head_point(pose: dict) -> Optional[Tuple[float, float]]:
    """The point the animal looks out of, or None if it named no head part."""
    low = {str(bp).lower(): v for bp, v in pose.items()}
    for want in _HEAD_NAMES:
        v = low.get(want)
        if v and len(v) >= 2 and v[0] is not None:
            return float(v[0]), float(v[1])
    return None


class VerifyView(QWidget):
    """Scrub a recording with its poses and zones drawn on it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path = ""
        self._cap = None
        self._rows: List[dict] = []
        self._states: dict = {}       # frame -> task state, joined from the .tsv
        self._header = None
        self._zones: List[dict] = []
        #: The part order the colours and the legend follow, worked out once
        #: per recording, see `_part_order`.
        self._part_order_cache: List[str] = []
        #: The frame the POSES are in, not always the video's. See
        #: `_recover_pose_frame`.
        self._pose_frame = None
        #: Interaction scoring for this recording, worked out once on load --
        #: see `_build_interaction`. ``{zone: {points, edge, in_zone, facing}}``.
        self._ia: Dict[str, dict] = {}
        self._ia_thresh = 45.0
        self._frame_of_row: List[int] = []
        self._build()

    # ── construction ─────────────────────────────────────────────────

    def _build(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(6)
        self._picker = QComboBox()
        self._picker.setMinimumWidth(240)
        self._picker.currentIndexChanged.connect(self._on_pick)
        top.addWidget(self._picker)
        for label, delta in (("◀◀", -25), ("◀", -1), ("▶", 1), ("▶▶", 25)):
            b = QPushButton(label)
            b.setFixedWidth(38)
            b.clicked.connect(lambda _=False, d=delta: self._step(d))
            top.addWidget(b)
        self._chk_zones = QCheckBox("zones")
        self._chk_zones.setChecked(True)
        self._chk_zones.stateChanged.connect(lambda _=0: self._render())
        top.addWidget(self._chk_zones)
        self._chk_pose = QCheckBox("pose")
        self._chk_pose.setChecked(True)
        self._chk_pose.stateChanged.connect(lambda _=0: self._render())
        top.addWidget(self._chk_pose)
        self._chk_trail = QCheckBox("trail")
        self._chk_trail.setChecked(True)
        self._chk_trail.stateChanged.connect(lambda _=0: self._render())
        top.addWidget(self._chk_trail)
        self._chk_int = QCheckBox("interaction")
        self._chk_int.setChecked(True)
        self._chk_int.setToolTip(
            "Draw what the investigation measure sees: the object edge the "
            "animal has to face (yellow), the line from its head to that edge "
            "and its own heading (orange), and INT on every frame the scoring "
            "counts as investigation (green)." + chr(10)
            + "Recordings without interaction zones have nothing to draw.")
        self._chk_int.stateChanged.connect(lambda _=0: self._render())
        top.addWidget(self._chk_int)

        # A top-down arena is a pale floor under flat lighting: the whole
        # frame lands in a narrow band of greys (mean 132 of 255 on these
        # recordings) and the animal is hard to see against it. This stretches
        # what is there, it changes the PICTURE and nothing else, so no
        # measurement moves by a pixel.
        self._chk_contrast = QCheckBox("contrast")
        self._chk_contrast.setChecked(True)
        self._chk_contrast.setToolTip(
            "Stretch the picture so the animal stands out from the floor."
            + chr(10) + "A VIEW setting: it changes what you see here and in "
            "a saved video, never the poses, the zones or any measure.")
        self._chk_contrast.stateChanged.connect(lambda _=0: self._render())
        top.addWidget(self._chk_contrast)
        top.addStretch()
        self._btn_save = QPushButton("Save video…")
        self._btn_save.setToolTip(
            "Write this recording out with the overlays burned in, exactly "
            "what is on screen, at the recording's own frame rate."
            + chr(10) + "The zones / pose / trail switches decide what is drawn.")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._save_annotated)
        top.addWidget(self._btn_save)
        v.addLayout(top)

        self._canvas = QLabel("Load a recording and press Run, then pick it here.")
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # The frame SCALES to whatever height it is given; it does not
        # demand one. A 240 px floor plus the scrubber and two readout lines
        # made the view taller than the panel could be with the settings
        # sidebar open, and the readout was pushed off the bottom, the one
        # part that says which frame you are looking at.
        self._canvas.setMinimumHeight(120)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Ignored,
                                   QSizePolicy.Policy.Ignored)
        self._canvas.setStyleSheet("background:#0d1117;border:1px solid #30363d;")
        # See `eventFilter`: the frame is re-scaled from the canvas's own
        # resize, which is a layout pass later than this widget's.
        self._canvas.installEventFilter(self)
        v.addWidget(self._canvas, 1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(lambda _=0: self._render())
        v.addWidget(self._slider)

        self._readout = QLabel("")
        self._readout.setStyleSheet(
            "color:#c9d1d9;font-family:Consolas,monospace;font-size:8pt;")
        self._readout.setTextFormat(Qt.TextFormat.RichText)
        # Wraps rather than truncating: this line is the whole point of the
        # view, which frame, at what time, where, in which zone, and the
        # last field was being cut to "spee…" at panel widths that are
        # perfectly normal with the settings sidebar open.
        self._readout.setWordWrap(True)
        v.addWidget(self._readout)

        self._summary = QLabel("")
        self._summary.setStyleSheet("color:#f0c988;font-size:8pt;")
        self._summary.setWordWrap(True)
        v.addWidget(self._summary)

    # ── loading ──────────────────────────────────────────────────────

    def set_recordings(self, entries: List[Tuple[str, str, str]]) -> None:
        """``entries`` = ``(label, pose_txt_path, video_path)``."""
        self._picker.blockSignals(True)
        self._picker.clear()
        for label, txt, video in entries:
            self._picker.addItem(label, (txt, video))
        self._picker.blockSignals(False)
        if entries:
            self._picker.setCurrentIndex(0)
            self._on_pick(0)

    def _on_pick(self, index: int) -> None:
        data = self._picker.itemData(index)
        if not data:
            return
        txt, video = data
        self._load(txt, video)

    def _states_by_frame(self, txt_path: str) -> dict:
        """The task state per frame, from the MCU log when the rows lack one.

        Verify read ``row["stage"]`` and showed a dash when it was blank, which
        on this rig is most of the corpus: 527 of the 1,001 recordings under
        one project carry no ``stage`` column at all and keep the state in the
        session ``.tsv`` beside them. Stepping through frames without knowing
        which state they were in is the one thing this view exists to avoid.
        """
        if not txt_path or not os.path.exists(txt_path):
            return {}
        from tools.offline_analysis.engine import mcu_states

        try:
            return mcu_states.merge_by_frame(txt_path,
                                             video_path=self._video_path,
                                             header=self._header)
        except Exception as e:                # never fatal: it is a caption
            logger.debug("no MCU states for %s: %s", txt_path, e)
            return {}

    def _load(self, txt_path: str, video_path: str) -> None:
        from tools.offline_analysis import video_data_schema as vds

        self._close_cap()
        self._rows = []
        self._zones = []
        self._header = None
        self._video_path = video_path or ""

        if txt_path and os.path.exists(txt_path):
            self._header = vds.VideoDataHeader.parse(txt_path)
            self._rows = list(vds.iter_rows(txt_path, self._header))
            self._zones = vds.zones_as_list(self._header.zones)
        self._frame_of_row = [vds.parse_int(r.get("frame_number"), i)
                              for i, r in enumerate(self._rows)]
        self._states = self._states_by_frame(txt_path)
        self._pose_frame = self._recover_pose_frame()
        self._ia = self._build_interaction()

        if self._video_path and os.path.exists(self._video_path):
            try:
                import cv2
                self._cap = cv2.VideoCapture(self._video_path)
                if not self._cap.isOpened():
                    self._cap = None
            except Exception as e:                        # pragma: no cover
                logger.warning("verify: cannot open %s: %s", self._video_path, e)
                self._cap = None

        n = max(len(self._rows) - 1, 0)
        self._btn_save.setEnabled(bool(self._rows) and self._cap is not None)
        self._slider.setEnabled(n > 0)
        self._slider.setRange(0, n)
        self._slider.setValue(0)
        self._summarise()
        self._render()

    def _recover_pose_frame(self):
        """The frame the POSES are in, which is not always the video's.

        A rig that reads the camera at one size and encodes the video smaller
        writes poses in camera pixels; drawn straight onto the video every
        keypoint lands at camera/video times its true position, so the
        markers sit beside the animal instead of on it. Nothing in an older
        file says this, so it is recovered from the file's own `location`
        column -- see `space.infer_pose_frame`.
        """
        if not self._rows or self._header is None:
            return None
        from tools.offline_analysis import video_data_schema as vds

        xs, ys, locs = [], [], []
        for r in self._rows:
            v = _primary(vds.parse_pose(r.get("pose_array")) or {})
            xs.append(v[0] if v else float("nan"))
            ys.append(v[1] if v else float("nan"))
            locs.append(r.get("location") or "")
        try:
            from tools.offline_analysis.engine.space import infer_pose_frame
            return infer_pose_frame(self._header, (xs, ys), locs, self._zones)
        except Exception as e:                            # pragma: no cover
            logger.debug("verify: pose frame not recovered: %s", e)
            return None

    def _build_interaction(self) -> Dict[str, dict]:
        """Score every interaction zone ONCE per recording, the engine's way.

        An overlay that disagrees with the workbook is worse than no overlay,
        so the frames marked INT here are the frames `space.rezone` counts as
        investigation -- the same outward edge, the same facing cone, the same
        dwell confirmation -- rather than a second rule that looks about right.
        Nothing here is drawn from a number this view invented.
        """
        if not self._rows or self._header is None or not self._zones:
            return {}
        try:
            from tools.offline_analysis import video_data_schema as vds
            from tools.offline_analysis.engine import space as sp

            model = sp.space_from_header(self._header, self._pose_frame or None)
            if not model.interaction_zones:
                return {}

            n = len(self._rows)
            bx = np.full(n, np.nan)
            by = np.full(n, np.nan)
            hx = np.full(n, np.nan)
            hy = np.full(n, np.nan)
            ts = np.full(n, np.nan)
            for i, r in enumerate(self._rows):
                pose = vds.parse_pose(r.get("pose_array")) or {}
                b = _primary(pose)
                h = _head_point(pose)
                if b:
                    bx[i], by[i] = b
                if h:
                    hx[i], hy[i] = h
                ts[i] = vds.parse_float(r.get("frame_ts_ms"))
            if not np.isfinite(hx).any():
                return {}                  # no head part: no orientation to draw

            dt = np.diff(ts, prepend=ts[0] if n else 0.0) / 1000.0
            dt[~np.isfinite(dt)] = 0.0
            dt[dt < 0] = 0.0
            if not np.any(dt > 0):         # no usable clock -- fall back to fps
                dt = np.full(n, 1.0 / max(self.true_fps(), 1e-6))

            res = sp.rezone(model, body_xy=(bx, by), head_xy=(hx, hy), dt=dt)
            centre = sp.arena_centre(model)
            self._ia_thresh = float(
                (model.interaction or {}).get("angle_threshold_deg", 45.0))

            out: Dict[str, dict] = {}
            for z in model.zones:
                ia = res.interaction.get(z.get("name"))
                if ia is None:
                    continue
                pts = [(float(p[0]), float(p[1])) for p in (z.get("points") or [])]
                out[str(z.get("name"))] = {
                    "points": pts,
                    "edge": sp.outer_edge(pts, centre),
                    "in_zone": ia.in_zone,
                    "facing": ia.facing,
                    "angle": ia.angle_deg,
                }
            return out
        except Exception as e:                            # pragma: no cover
            logger.debug("verify: interaction overlay unavailable: %s", e)
            return {}

    def _pose_scale(self, frame) -> float:
        """Pose pixels -> this frame's pixels. 1.0 when they are the same."""
        pf = getattr(self, "_pose_frame", None)
        if not pf or not pf[0]:
            return 1.0
        return float(frame.shape[1]) / float(pf[0])

    def _close_cap(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def closeEvent(self, event):                          # pragma: no cover
        self._close_cap()
        super().closeEvent(event)

    # ── the honest summary ───────────────────────────────────────────

    def _summarise(self) -> None:
        """What is wrong with this recording, stated up front."""
        from tools.offline_analysis import video_data_schema as vds

        if not self._rows:
            self._summary.setText("No pose rows in this file.")
            return
        notes: List[str] = []
        ts = np.array([vds.parse_float(r.get("frame_ts_ms"))
                       for r in self._rows], float)
        good = ts[np.isfinite(ts)]
        if len(good) > 1:
            dt = np.diff(good)
            dt = dt[dt > 0]
            med = float(np.median(dt)) if dt.size else math.nan
            if not math.isnan(med):
                notes.append(f"median dt {med:.1f} ms ({1000 / med:.1f} fps)")
                if med < 5:
                    notes.append("⚠ that is not a plausible frame interval, "
                                 "the timestamps are probably not milliseconds")
        n_pose = sum(1 for r in self._rows if vds.parse_pose(r.get("pose_array")))
        pct = 100.0 * n_pose / len(self._rows)
        notes.append(f"{n_pose}/{len(self._rows)} frames with a pose ({pct:.0f}%)")
        if pct < 50:
            notes.append("⚠ over half the frames have no detection")
        # ...of the rows there ARE. A retrack that was stopped early writes a
        # short pose file, and every count above it is then 100% of a fraction
        # of the recording, which is exactly how a two-percent retrack gets
        # exported as an hour of un-annotated video.
        n_src = self._source_frames()
        if n_src and len(self._rows) < n_src * 0.99:
            cov = 100.0 * len(self._rows) / n_src
            notes.append(f"⚠ these poses cover {len(self._rows)} of {n_src} "
                         f"video frames ({cov:.0f}%), the retrack did not "
                         f"finish the recording")
        located = sum(1 for r in self._rows
                      if r.get("location") and not vds.is_na(r.get("location")))
        if self._zones and located == 0:
            notes.append("⚠ zones exist but no frame is in one")
        if self._header is not None and not self._header.px_per_cm:
            notes.append("⚠ no scale, speeds and distances are in PIXELS")
        if self._cap is None and self._video_path:
            notes.append("video could not be opened")
        elif not self._video_path:
            notes.append("no video, poses shown without their frames")
        notes.extend(self._tracker_notes())
        self._summary.setText(" · ".join(notes))

    def _source_frames(self) -> int:
        """How many pictures the video has, or 0 when there is no video."""
        if self._cap is None:
            return 0
        try:
            import cv2

            return max(0, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        except Exception:                                 # pragma: no cover
            return 0

    def _tracker_notes(self) -> List[str]:
        """What produced these poses, when the recording says.

        Two sessions that differ only by the engine or the input mode look
        identical everywhere else, same columns, same shape, plausible
        numbers. This is the one place that says which of them you are
        looking at, so it is worth the line.
        """
        if self._header is None:
            return []
        tr = getattr(self._header, "tracker", None) or {}
        if not isinstance(tr, dict) or not tr:
            return []
        bits = []
        engine = tr.get("engine") or tr.get("backend")
        if engine:
            bits.append(str(engine))
        mode = tr.get("input_mode")
        if mode and mode != "full":
            w, h = tr.get("input_w"), tr.get("input_h")
            bits.append(f"{mode} {w}x{h}" if w and h else str(mode))
        for key in ("precision", "colour_mode"):
            if tr.get(key):
                bits.append(str(tr[key]))
        return [" / ".join(bits)] if bits else []

    # ── rendering ────────────────────────────────────────────────────

    def _step(self, delta: int) -> None:
        self._slider.setValue(
            max(0, min(self._slider.maximum(), self._slider.value() + delta)))

    def eventFilter(self, obj, event):
        """Re-scale the frame whenever the CANVAS changes size.

        The pixmap was scaled once, to whatever the canvas measured at the
        moment the frame was drawn, and never again: a later resize left the
        frame at the old scale, 268px of image in a 261px canvas, so the
        bottom of the arena and the lowest keypoint label were cut off, and
        after a widening the frame stopped filling the space it was given.

        Watching the canvas rather than this widget matters for the FIRST
        render too: the canvas gets its final geometry one layout pass after
        the view does, so a handler on the view alone is always one step
        behind on the frame the user actually opens on.
        """
        if obj is self._canvas and event.type() == QEvent.Type.Resize:
            if self._rows:
                self._render()
        return super().eventFilter(obj, event)

    def _render(self) -> None:

        if not self._rows:
            self._canvas.setText("Nothing to verify.")
            self._readout.setText("")
            return
        i = int(self._slider.value())
        i = max(0, min(i, len(self._rows) - 1))
        row = self._rows[i]

        frame = self._frame_for(i)
        if frame is None:
            size = self._header.resolution if self._header else (0, 0)
            w, h = (size if size[0] else (480, 360))
            frame = np.full((int(h), int(w), 3), 24, np.uint8)

        frame, pose = self._annotate(frame, i)
        self._canvas.setPixmap(_to_pixmap(frame, self._canvas.size()))
        self._readout.setText(self._readout_html(i, row, pose))

    def true_fps(self) -> float:
        """Frames per second from the TIMESTAMPS, not from the container.

        A recording written at a nominal 30 whose frames arrived at 30.3 is
        played back 1% fast if you trust the container, and a recording whose
        writer stamped a wrong rate can be out by much more. The timestamps
        are what the rest of the analysis measures time with, so they are
        what an exported video is paced by too.
        """
        from tools.offline_analysis import video_data_schema as vds

        ts = np.array([vds.parse_float(r.get("frame_ts_ms"))
                       for r in self._rows], float)
        good = ts[np.isfinite(ts)]
        if len(good) > 1:
            dt = np.diff(good)
            dt = dt[dt > 0]
            if dt.size:
                med = float(np.median(dt))
                if 1.0 <= med <= 1000.0:
                    return 1000.0 / med
        if self._cap is not None:
            try:
                import cv2
                fps = float(self._cap.get(cv2.CAP_PROP_FPS))
                if 1.0 <= fps <= 1000.0:
                    return fps
            except Exception:                             # pragma: no cover
                pass
        return 30.0

    def _save_annotated(self) -> None:
        """Write the overlaid video to a file the user picks."""
        from PySide6.QtWidgets import (QApplication, QFileDialog,
                                       QMessageBox, QProgressDialog)

        if not self._rows or self._cap is None:
            QMessageBox.information(
                self, "Nothing to save",
                "This needs both the recording's video and its data file.")
            return
        stem = self._picker.currentText() or "annotated"
        start = os.path.join(os.path.dirname(self._video_path or ""),
                             f"{stem}_annotated.mp4")
        out, _ = QFileDialog.getSaveFileName(
            self, "Save annotated video", start, "MP4 video (*.mp4)")
        if not out:
            return

        import cv2

        # A partial retrack would otherwise be written out in full: a few
        # annotated frames buried in an hour of untouched video, which reads
        # as "the overlay is broken" rather than "the retrack stopped early".
        span_only = False
        frames = [int(fn) for fn in self._frame_of_row] or [0]
        first_f, last_f = min(frames), max(frames)
        n_src = self._source_frames()
        if n_src and (last_f - first_f + 1) < n_src * 0.9:
            cov = 100.0 * (last_f - first_f + 1) / n_src
            answer = QMessageBox.question(
                self, "Only part of this recording is tracked",
                f"The poses cover frames {first_f}–{last_f} of {n_src} "
                f"({cov:.0f}%).{chr(10)}{chr(10)}"
                f"Write only that part? “No” writes the whole recording, with "
                f"the untracked frames carrying no overlay.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes)
            if answer == QMessageBox.StandardButton.Cancel:
                return
            span_only = answer == QMessageBox.StandardButton.Yes

        fps = self.true_fps()
        # Read the video straight through rather than seeking per row: a seek
        # per frame turns a 9,000-frame session into minutes of thrashing.
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            QMessageBox.warning(self, "Cannot read", "The video did not open.")
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # The written frame carries the legend band, so it is TALLER than the
        # source. A writer opened at the video's own size accepts the first
        # frame and silently drops every one that does not match it, which is
        # every one, leaving a file that plays for no time at all.
        probe, _pose = self._annotate(np.zeros((h, w, 3), np.uint8), 0)
        size = (int(probe.shape[1]), int(probe.shape[0]))
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, size)
        if not writer.isOpened():
            cap.release()
            QMessageBox.warning(
                self, "Cannot write",
                "No encoder available for that file. Try an .avi name.")
            return

        # Which row belongs to which SOURCE frame. A recording that dropped
        # frames has fewer rows than the video has pictures, and pairing them
        # by position would slide the overlay out of step part way through.
        row_of_frame = {}
        for i, fn in enumerate(self._frame_of_row):
            row_of_frame.setdefault(int(fn), i)

        prog = QProgressDialog("Writing annotated video…", "Cancel",
                               0, len(self._rows), self)
        prog.setWindowTitle("Save annotated video")
        prog.setMinimumDuration(0)
        prog.setValue(0)

        written = skipped = 0
        f = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if span_only and f < first_f:
                    f += 1
                    continue
                if span_only and f > last_f:
                    break
                i = row_of_frame.get(f)
                if i is None:
                    # A picture with no pose row still gets the band, or it
                    # would be the one frame in the file with a different
                    # size, and the writer would drop it.
                    skipped += 1
                    frame = self._draw_header(frame, f, None)
                else:
                    frame, _pose = self._annotate(frame, i)
                    if written % 25 == 0:
                        prog.setValue(min(written, len(self._rows)))
                        QApplication.processEvents()
                        if prog.wasCanceled():
                            break
                writer.write(frame)
                written += 1
                f += 1
        finally:
            writer.release()
            cap.release()
            prog.setValue(len(self._rows))

        if prog.wasCanceled():
            QMessageBox.information(
                self, "Stopped",
                f"Stopped after {written} frames. The part written so far is "
                f"in {os.path.basename(out)}.")
            return
        note = (f"{written} frames at {fps:.2f} fps "
                f"({written / fps:.1f} s).")
        if skipped:
            note += (f"{chr(10)}{skipped} frame(s) had no pose row and were "
                     f"written without an overlay.")
        QMessageBox.information(self, "Saved", f"{os.path.basename(out)}"
                                f"{chr(10)}{note}")

    def _annotate(self, frame, i: int):
        """Draw row `i` onto `frame`. Returns ``(frame, pose)``.

        The export writes exactly what the preview shows because it is this
        function, not a second copy of it: a saved video that disagreed with
        the frame the user checked would be worse than no export at all.
        """
        from tools.offline_analysis import video_data_schema as vds

        row = self._rows[i]
        frame = frame.copy()
        if self._chk_contrast.isChecked():
            frame = _stretch(frame)
        if self._chk_zones.isChecked() and self._zones:
            try:
                from tools.offline_analysis.analyze.zone_renderer import render_zones_opencv
                frame = render_zones_opencv(
                    frame, self._zones,
                    highlight_zone=(row.get("location") or None))
            except Exception as e:                        # pragma: no cover
                logger.debug("verify: zone overlay failed: %s", e)
        if self._chk_trail.isChecked():
            self._draw_trail(frame, i)
        pose = vds.parse_pose(row.get("pose_array"))
        if self._chk_pose.isChecked() and pose:
            self._draw_pose(frame, pose)
        if self._chk_int.isChecked() and pose:
            self._draw_interaction(frame, pose, i)
        if self._chk_zones.isChecked():
            self._draw_zone_label(frame, row)
        return self._draw_header(frame, i, pose), pose

    def _frame_for(self, i: int):
        if self._cap is None:
            return None
        import cv2

        target = self._frame_of_row[i] if i < len(self._frame_of_row) else i
        try:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, target)))
            ok, frame = self._cap.read()
            return frame if ok else None
        except Exception:                                 # pragma: no cover
            return None

    def _recorded_skeleton(self):
        """The edges this recording declares, or none.

        Recordings written before the rig recorded a skeleton carry nothing
        here, and those fall back to the name-matched guess below.
        """
        header = getattr(self, "_header", None)
        try:
            return list(getattr(header, "skeleton", ()) or ())
        except Exception:
            return []

    def _draw_pose(self, frame, pose: dict) -> None:
        """Keypoints, joined, in the colours the legend names.

        The names go in the legend above the picture, where they are written
        once. Printed beside every point, six labels on a 360x202 frame
        overlap each other and the animal, which is most of what makes a saved
        video unreadable.
        """
        import cv2

        k = self._pose_scale(frame)
        order = self._part_order(pose)
        drawn = {}
        for bp, v in pose.items():
            if not v or len(v) < 2 or v[0] is None:
                continue
            drawn[bp] = (int(round(float(v[0]) * k)),
                         int(round(float(v[1]) * k)))

        # Skeleton first, so the joints sit on top of the bones. The
        # recording's own edges win: newer files carry what the model
        # declared, and a declared skeleton beats a guess from part names.
        declared = [(a, b) for a, b in self._recorded_skeleton()
                    if a in drawn and b in drawn]
        for a, b in (declared or _skeleton_pairs(list(drawn))):
            # Dark casing under a light core, so a bone still reads against
            # an overexposed arena floor.
            cv2.line(frame, drawn[a], drawn[b], (20, 20, 20), 3, cv2.LINE_AA)
            cv2.line(frame, drawn[a], drawn[b], (235, 235, 235), 1, cv2.LINE_AA)

        radius = 3 if min(frame.shape[:2]) < 400 else 4
        for bp, (x, y) in drawn.items():
            colour = _bp_colour(bp, order)
            cv2.circle(frame, (x, y), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius, colour, -1, cv2.LINE_AA)

    def _draw_interaction(self, frame, pose: dict, i: int) -> None:
        """The interaction geometry, drawn the way the rig's own clips draw it.

        Yellow: the outward edge of each interaction zone -- the object face
        the animal is scored against. Orange: its heading, the cone that
        counts as facing, and the line to the edge while its head is inside.
        Green ring + INT: this frame is investigation in the workbook too.
        """
        import cv2

        if not self._ia:
            return
        head = _head_point(pose)
        body = _primary(pose)
        if head is None or body is None:
            return
        from tools.offline_analysis.engine import space as sp

        k = self._pose_scale(frame)

        def P(p):
            return (int(round(float(p[0]) * k)), int(round(float(p[1]) * k)))

        big = min(frame.shape[:2]) >= 400
        thin = 2 if big else 1
        hp = P(head)
        facing_now = False

        for ia in self._ia.values():
            edge = ia.get("edge")
            if edge:
                # Drawn a touch past both ends: an interaction zone can be a
                # few pixels wide on a 360-px frame, and a two-pixel yellow
                # stub reads as an artefact rather than as the object face.
                (ax, ay), (bx, by) = edge
                ex, ey = (bx - ax) * 0.15, (by - ay) * 0.15
                cv2.line(frame, P((ax - ex, ay - ey)), P((bx + ex, by + ey)),
                         _IA_EDGE, thin + 1, cv2.LINE_AA)
            inside = i < len(ia["in_zone"]) and bool(ia["in_zone"][i])
            hit = i < len(ia["facing"]) and bool(ia["facing"][i])
            if not inside:
                continue
            target = (sp.nearest_point_on_segment(edge[0], edge[1], head[0], head[1])
                      if edge else
                      sp.nearest_point_on_polygon(ia["points"], head[0], head[1]))
            cv2.line(frame, hp, P(target), _IA_HIT if hit else _IA_RAY,
                     thin, cv2.LINE_AA)
            facing_now = facing_now or hit

        # The cone itself: where the animal is pointing, and how far off that
        # a target may sit and still count. Without it a rejected frame looks
        # like a bug rather than a mouse looking somewhere else.
        vx, vy = head[0] - body[0], head[1] - body[1]
        norm = math.hypot(vx, vy)
        if norm > 1e-6:
            reach = max(frame.shape[:2]) * 0.18
            for sign in (-1.0, 1.0):
                a = math.radians(self._ia_thresh) * sign
                ca, sa = math.cos(a), math.sin(a)
                dx, dy = (vx * ca - vy * sa) / norm, (vx * sa + vy * ca) / norm
                cv2.line(frame, hp,
                         (int(round(hp[0] + dx * reach)),
                          int(round(hp[1] + dy * reach))),
                         _IA_RAY, 1, cv2.LINE_AA)

        if facing_now:
            # Sized to the animal, not to the frame: one fixed radius is a
            # collar on a 1080-px recording and a halo on a 360-px one.
            span = math.hypot(*(a - b for a, b in zip(head, body))) * k
            r = int(round(min(max(span * 1.5, 10.0), min(frame.shape[:2]) * 0.25)))
            cv2.circle(frame, hp, r, _IA_HIT, thin + 1, cv2.LINE_AA)
            f = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.7 if big else 0.4
            (tw, _th), _ = cv2.getTextSize("INT", f, scale, thin)
            org = (hp[0] - tw // 2, hp[1] - r - (6 if big else 3))
            cv2.putText(frame, "INT", org, f, scale, (0, 0, 0), thin + 2, cv2.LINE_AA)
            cv2.putText(frame, "INT", org, f, scale, _IA_HIT, thin, cv2.LINE_AA)

    def _draw_zone_label(self, frame, row: dict) -> None:
        """Which zone the file says the animal is in, big, along the bottom.

        The readout line under the picture says the same thing, but it is not
        in a saved video and it is not in a figure panel -- and the zone is
        the one thing a reader of either has to be able to check by eye.
        """
        import cv2
        from tools.offline_analysis import video_data_schema as vds

        name = row.get("location")
        if not name or vds.is_na(name):
            return
        h, w = frame.shape[:2]
        big = min(h, w) >= 400
        f = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.9 if big else 0.45
        thick = 2 if big else 1
        (tw, th), _ = cv2.getTextSize(str(name), f, scale, thick)
        org = (max(4, (w - tw) // 2), h - max(6, th // 2))
        cv2.putText(frame, str(name), org, f, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(frame, str(name), org, f, scale, (255, 255, 255), thick, cv2.LINE_AA)

    def _part_order(self, pose: dict) -> List[str]:
        """The part order the colours and the legend both follow.

        The recording's declared order when it has one; that is the order the
        model itself uses, otherwise first-seen. Never the sorted order: a
        frame missing one part re-sorted the rest and every colour shifted.
        """
        if self._part_order_cache:
            return self._part_order_cache
        declared = list(getattr(self._header, "body_parts", ()) or ())
        self._part_order_cache = declared or list(pose)
        return self._part_order_cache

    def _draw_header(self, frame, i: int, pose: Optional[dict]):
        """A band above the picture: what this is, and which colour is what.

        Modelled on the annotated clips the rig itself produces, because those
        are what people compare against, the legend across the top, one dot
        per part in the part's own colour, and a short line saying which frame
        and how many keypoints were found.
        """
        import cv2

        order = self._part_order(pose or {})
        state = self._state_of(self._rows[i], i) if i < len(self._rows) else ""
        if not order and not state:
            return frame
        # NOT `if not order`. A band that exists only to caption a pose gives
        # a recording with no pose no band at all, and therefore no frame
        # number and no state, on exactly the recordings where the state is the
        # only thing there is to show. A frame with a state and no keypoints is
        # still worth labelling.

        width = frame.shape[1]
        scale = max(0.28, min(0.42, width / 1400.0))
        line_h = 15 if width < 700 else 18
        band_h = line_h * 2 + 6
        band = np.zeros((band_h, width, 3), dtype=frame.dtype)

        found = sum(1 for v in (pose or {}).values()
                    if v and len(v) >= 2 and v[0] is not None)
        stem = os.path.basename(self._video_path or "")[:34]
        cv2.putText(band, stem, (6, line_h - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (120, 220, 140), 1, cv2.LINE_AA)
        # The SOURCE frame number, not the row index. They agree on a
        # recording that dropped nothing and diverge on one that did, and a
        # band labelled "f" that shows a row number is the sort of thing
        # someone checks a stopwatch against.
        frame_no = (self._frame_of_row[i] if i < len(self._frame_of_row) else i)
        label = f"f{int(frame_no):06d}   {found}/{len(order)} kp"
        cv2.putText(band, label, (6, line_h * 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (190, 190, 190), 1, cv2.LINE_AA)

        # The task state, from the row or from the MCU log joined to it. An
        # operant clip is unreadable without it: the animal sitting still for
        # ninety seconds looks identical in `init_trial` and the inter-trial
        # interval, and this export had no way to tell them apart.
        if state:
            x = 6 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                    scale, 1)[0][0] + 12
            cv2.putText(band, str(state), (x, line_h * 2),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (120, 255, 160), 1,
                        cv2.LINE_AA)

        # The legend, right-aligned, wrapping onto the second line when the
        # frame is too narrow for one row of names.
        entries = []
        for name in order:
            w = cv2.getTextSize(str(name), cv2.FONT_HERSHEY_SIMPLEX,
                                scale, 1)[0][0]
            entries.append((name, w + 16))
        total = sum(w for _n, w in entries)
        rows = 1 if total <= width - 120 else 2
        per_row = (len(entries) + rows - 1) // rows
        for r in range(rows):
            chunk = entries[r * per_row:(r + 1) * per_row]
            x = width - sum(w for _n, w in chunk) - 6
            y = line_h * (r + 1) - 4
            for name, w in chunk:
                colour = _bp_colour(name, order)
                cv2.circle(band, (x + 4, y - 4), 3, colour, -1, cv2.LINE_AA)
                cv2.putText(band, str(name), (x + 11, y),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, (225, 225, 225),
                            1, cv2.LINE_AA)
                x += w
        return np.vstack([band, frame])

    def _draw_trail(self, frame, i: int, span: int = 60) -> None:
        import cv2
        from tools.offline_analysis import video_data_schema as vds

        k = self._pose_scale(frame)
        pts = []
        for j in range(max(0, i - span), i + 1):
            p = vds.parse_pose(self._rows[j].get("pose_array"))
            if not p:
                continue
            v = _primary(p)
            if v is not None:
                pts.append((int(round(v[0] * k)), int(round(v[1] * k))))
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, (90, 190, 255), 1, cv2.LINE_AA)

    def _readout_html(self, i: int, row: dict, pose: Optional[dict]) -> str:
        from tools.offline_analysis import video_data_schema as vds

        t = vds.parse_float(row.get("frame_ts_ms"))
        dt = math.nan
        if i > 0:
            prev = vds.parse_float(self._rows[i - 1].get("frame_ts_ms"))
            if not (math.isnan(t) or math.isnan(prev)):
                dt = t - prev
        unit = "cm/s" if (self._header and self._header.px_per_cm) else "px/s"
        speed = row.get("speed")
        loc = row.get("location")
        loc_txt = ("<span style='color:#8bd4a8'>%s</span>" % loc
                   if loc and not vds.is_na(loc)
                   else "<span style='color:#6e7681'>none</span>")
        dt_txt = ", " if math.isnan(dt) else f"{dt:.0f} ms"
        if not math.isnan(dt) and dt < 5:
            dt_txt = f"<span style='color:#e46a6a'>{dt:.0f} ms</span>"
        centre = _primary(pose or {})
        pos = (", " if centre is None
               else f"({centre[0]:.1f}, {centre[1]:.1f})")
        return (f"frame <b>{self._frame_of_row[i] if i < len(self._frame_of_row) else i}</b>"
                f" &nbsp; row {i + 1}/{len(self._rows)} &nbsp;|&nbsp; "
                f"t = <b>{', ' if math.isnan(t) else f'{t:.0f}'}</b> ms &nbsp; "
                f"dt = {dt_txt} &nbsp;|&nbsp; "
                f"pos {pos} &nbsp;|&nbsp; "
                f"zone {loc_txt} &nbsp;|&nbsp; "
                f"speed {speed} {unit} &nbsp;|&nbsp; "
                f"stage <b>{self._state_of(row, i) or ', '}</b>")

    def _state_of(self, row: dict, i: int) -> str:
        """The row's own state, or the one joined from the MCU log."""
        from tools.offline_analysis.engine.mcu_states import state_of_cell

        own = state_of_cell(row.get("stage"))
        if own or not self._states:
            return own
        frame = (self._frame_of_row[i] if i < len(self._frame_of_row) else i)
        return self._states.get(frame, "")


def _primary(pose: dict) -> Optional[Tuple[float, float]]:
    """The point a single-animal measure follows, centre if there is one."""
    for want in ("center", "centre", "body", "middle"):
        for bp, v in pose.items():
            if bp.lower() == want and v and len(v) >= 2 and v[0] is not None:
                return float(v[0]), float(v[1])
    for _bp, v in sorted(pose.items()):
        if v and len(v) >= 2 and v[0] is not None:
            return float(v[0]), float(v[1])
    return None


#: How much of the stretch to actually apply. Full strength blows a pale
#: arena floor to paper white; this is the point where the animal separates
#: from the floor and the bedding texture is still there.
_STRENGTH = 0.75


def _stretch(frame: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0):
    """Spread the frame's own range of greys across the full one.

    A top-down arena under flat lighting occupies a narrow band, on these
    recordings the mean is 132 of 255 and almost everything sits within about
    forty levels of it, so a dark animal on a pale floor reads as two
    similar greys. Anchoring on the 1st and 99th percentiles rather than the
    extremes means one specular highlight or one black corner cannot flatten
    everything else back down.

    Applied to the LUMINANCE only, with the colour left where it is: pushing
    the channels independently would tint the floor, and the zone overlays are
    drawn in colours the user has to be able to tell apart.
    """
    import cv2

    if frame is None or frame.size == 0:
        return frame
    if frame.ndim == 2:
        lo, hi = np.percentile(frame, [lo_pct, hi_pct])
        if hi - lo < 1:
            return frame
        out = (frame.astype(np.float32) - lo) * (255.0 / (hi - lo))
        return np.clip(out, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    lo, hi = np.percentile(L, [lo_pct, hi_pct])
    if hi - lo < 1:                      # a flat frame has nothing to stretch
        return frame
    # Into [8, 247] rather than the full range, and mixed back at STRENGTH.
    # Taken all the way, a pale arena floor goes to paper white and the
    # bedding texture that tells you where the animal is disappears into it,
    # the picture looks more contrasty and shows less.
    stretched = (L.astype(np.float32) - lo) * ((247.0 - 8.0) / (hi - lo)) + 8.0
    mixed = (1.0 - _STRENGTH) * L.astype(np.float32) + _STRENGTH * stretched
    lab[:, :, 0] = np.clip(mixed, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _to_pixmap(frame: np.ndarray, target) -> QPixmap:
    h, w = frame.shape[:2]
    if frame.ndim == 2:
        img = QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        img = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_BGR888)
    pm = QPixmap.fromImage(img.copy())
    tw = max(1, target.width() - 4)
    th = max(1, target.height() - 4)
    return pm.scaled(tw, th, Qt.AspectRatioMode.KeepAspectRatio,
                     Qt.TransformationMode.SmoothTransformation)
