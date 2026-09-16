"""Zone editing as a place you stay, not a dialog you dismiss.

Drawing zones was a modal: open it, edit one recording, press OK, and it was
gone. Checking your work against the next recording meant closing it,
highlighting a different row and opening it again, and because it was modal,
the readiness table and the ZONES & SCALE readout it changes were both hidden
behind it the whole time.

This is the same editor with the two things that were missing: it navigates
the recordings itself, and it is a panel, so everything it affects stays on
screen beside it.

The editing surface is the RIG's zone editor, vendored, the same widget the
tracking dialog uses to draw zones before a session, so drawing them
afterwards follows exactly the same rules and produces exactly the same
stored shape. It already owns the zone list, renaming, undo/redo and the
scale line; nothing here re-implements any of that. This supplies the
recording to edit and decides who the result is written to.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

logger = logging.getLogger(__name__)

#: What an edit is written to. Per-recording is the default because the panel
#: now makes moving between recordings cheap, the reason "apply to all" used
#: to be the only sane option was that reopening the modal was expensive.


class ZoneWorkbench(QWidget):
    """Draw zones across a set of recordings without leaving the view.

    The host supplies three callables and gets one signal back:

    ``frame_for(key)``   first frame of that recording, or None
    ``zones_for(key)``   its current zone dicts
    ``apply_to(zones, keys, frame_size)``  write them, returning how many took
    ``size_for(key)``    the frame size it declares, for when its video is gone
    ``zoned()``          which loaded recordings currently carry zones
    ``write_files(keys)``  write those zones into the recordings' own files

    Edits are committed on navigation and on Done, never discarded silently:
    moving to the next recording with unsaved zones was the one way this could
    quietly lose work.
    """

    finished = Signal()
    #: Emitted after a commit so the host can refresh whatever displays zones.
    applied = Signal(int)

    def __init__(self, frame_for: Callable, zones_for: Callable,
                 apply_to: Callable, scale_for: Optional[Callable] = None,
                 size_for: Optional[Callable] = None,
                 zoned: Optional[Callable] = None,
                 write_files: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        self._frame_for = frame_for
        self._zones_for = zones_for
        self._apply_to = apply_to
        # Zones are held normalized and drawn in pixels, so the frame size is
        # part of their meaning. Usually it comes from the frame itself; when
        # the video is missing, the recording still declares one, and without
        # it every zone would be scaled against whatever was on screen last.
        self._size_for = size_for or (lambda _key: (0, 0))
        # Writing into the recording's own file is optional wiring: without
        # it the button is simply absent, rather than present and inert.
        self._zoned = zoned
        self._write_files = write_files
        # A recording can be calibrated without carrying a scale LINE, most
        # are, because the rig records px/cm in the header. Judging the scale
        # from the zone list alone told every one of them "NO SCALE".
        self._scale_for = scale_for or (lambda _key: 0.0)
        self._recs: List[Tuple[str, str]] = []      # (label, key)
        self._index = 0
        self._frame_size: Tuple[int, int] = (0, 0)
        self._dirty = False
        #: Set when there is no frame to draw on. Held separately because
        #: `_describe` rewrites the state line on every edit and would
        #: otherwise wipe the one warning the user most needs to see.
        self._frame_note = ""
        self._build()

    # ── construction ─────────────────────────────────────────────

    def _build(self):
        from tools.offline_analysis.vendor.zone_editor import ZoneEditorWidget

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        nav = QHBoxLayout()
        nav.setSpacing(4)
        self.btn_done = QPushButton("‹  Back to recordings")
        self.btn_done.setObjectName("setup")
        self.btn_done.setToolTip("Commit the current zones and return to the "
                                 "recordings table.")
        self.btn_done.clicked.connect(self._finish)
        nav.addWidget(self.btn_done)

        nav.addSpacing(12)
        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        for b, tip in ((self.btn_prev, "Previous recording"),
                       (self.btn_next, "Next recording")):
            b.setObjectName("quiet")
            b.setFixedWidth(30)
            b.setToolTip(tip + ", the zones you have drawn are kept.")
        self.btn_prev.clicked.connect(lambda: self._go(-1))
        self.btn_next.clicked.connect(lambda: self._go(1))
        nav.addWidget(self.btn_prev)

        self.combo = QComboBox()
        self.combo.setMinimumWidth(200)
        self.combo.setMaximumWidth(340)
        self.combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setToolTip("Jump straight to a recording.")
        self.combo.currentIndexChanged.connect(self._on_combo)
        nav.addWidget(self.combo)
        nav.addWidget(self.btn_next)
        nav.addSpacing(8)

        self.lbl_pos = QLabel("")
        self.lbl_pos.setObjectName("field")
        nav.addWidget(self.lbl_pos)
        nav.addStretch(1)

        # Up here, with the recording it acts on and the count it applies to,
        # rather than in the footer under the canvas: copying one arena's
        # zones to a whole folder is the common case, and it should be the
        # first thing in reach, not something found after scrolling past the
        # drawing.
        self.btn_all = QPushButton("Copy to all")
        self.btn_all.setObjectName("setup")
        self.btn_all.setToolTip(
            "Write these zones and this scale to EVERY recording loaded, "
            "for a folder recorded on one rig with the camera untouched, "
            "which is the usual case.")
        self.btn_all.clicked.connect(self._copy_to_all)
        nav.addWidget(self.btn_all)

        # Zones live beside the recording by default. This puts them INSIDE
        # it, which is what "so they are not lost" actually means.
        if self._zoned is not None and self._write_files is not None:
            self.btn_write = QPushButton("Save into file")
            self.btn_write.setObjectName("setup")
            self.btn_write.setToolTip(
                "Write the zones into each recording's OWN data file, in the "
                "same header block the rig writes.<br><br>"
                "Until you do this they are kept beside the recording, in its "
                "analysis folder, which is safe, but stays behind if you "
                "move or send the recording.<br><br>"
                "Only the zone block changes; every recorded row is copied "
                "through untouched.")
            self.btn_write.clicked.connect(self._write_into_files)
            nav.addWidget(self.btn_write)

            # Beside it, because copying to all and then saving is one
            # intention and the two-step order was easy to get wrong.
            self.btn_write_all = QPushButton("Save into all files")
            self.btn_write_all.setObjectName("setup")
            self.btn_write_all.setToolTip(
                "Put these zones and this scale onto EVERY recording loaded, "
                "then write them into every one of those recordings' own "
                "data files, in a single step.<br><br>"
                "The same as pressing \"Copy to all\" and then "
                "\"Save into file\", which is the usual intention for a "
                "folder recorded on one rig with the camera untouched.")
            self.btn_write_all.clicked.connect(self._write_into_all_files)
            nav.addWidget(self.btn_write_all)
        v.addLayout(nav)

        # THE rig's zone editor, vendored, not a second one that behaves
        # almost the same. Drawing a zone offline has to follow the same rules
        # as drawing it in the tracking dialog before a session: same shapes,
        # same scale line, same rotation and snapping, and above all the same
        # storage, normalized [0, 1] points with the frame size beside them,
        # so a zone does not move when the resolution does.
        self.editor = ZoneEditorWidget(parent=self)
        self.editor.zones_changed.connect(self._on_edited)
        self._compact(self.editor)
        v.addWidget(self.editor, 1)

        # The footer carries the sentence describing what is about to be
        # written: how many zones, whether there is a scale, and whether
        # anything is unsaved.
        foot = QHBoxLayout()
        foot.setSpacing(6)
        self.lbl_state = QLabel("")
        self.lbl_state.setObjectName("field")
        self.lbl_state.setWordWrap(True)
        foot.addWidget(self.lbl_state, 1)

        v.addLayout(foot)

    # ── the set of recordings being worked through ───────────────

    def set_recordings(self, recs: Sequence[Tuple[str, str]], index: int = 0):
        """``recs`` is [(label, key)] in the order the table shows them."""
        self._recs = list(recs)
        self._index = max(0, min(int(index), len(self._recs) - 1)) \
            if self._recs else 0
        self.combo.blockSignals(True)
        self.combo.clear()
        for label, _key in self._recs:
            self.combo.addItem(label)
        self.combo.setCurrentIndex(self._index)
        self.combo.blockSignals(False)
        self._load()

    @property
    def current_key(self) -> Optional[str]:
        if not self._recs:
            return None
        return self._recs[self._index][1]

    # ── navigation ───────────────────────────────────────────────

    def _on_combo(self, i: int):
        if i == self._index or not (0 <= i < len(self._recs)):
            return
        self._commit()
        self._index = i
        self._load()

    def _go(self, delta: int):
        i = self._index + delta
        if not (0 <= i < len(self._recs)):
            return
        self._commit()
        self._index = i
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(i)
        self.combo.blockSignals(False)
        self._load()

    def _load(self):
        """Show the current recording's own frame and its own zones."""
        key = self.current_key
        self._sync_nav()
        if key is None:
            self.editor.set_zones([])
            self._frame_note = "No recordings to draw on."
            self._describe()
            return

        frame = None
        try:
            frame = self._frame_for(key)
        except Exception as e:                       # never strand the panel
            logger.warning("zone workbench frame for %s: %s", key, e)
        if frame is None:
            # A recording whose video is gone can still be re-zoned from
            # another one's geometry, so this is a warning, not a dead end.
            self.editor.canvas.set_frame_callback(None)
            declared = tuple(self._size_for(key) or (0, 0))
            if len(declared) == 2 and all(declared):
                self._frame_size = (int(declared[0]), int(declared[1]))
                self.editor.canvas.frame_w = self._frame_size[0]
                self.editor.canvas.frame_h = self._frame_size[1]
            self._frame_note = (
                "NO VIDEO for this recording, draw on one from the same rig "
                "and use Copy to all.")
        else:
            self._frame_size = (int(frame.shape[1]), int(frame.shape[0]))
            self._frame_note = ""
            self.editor.set_frame_callback(lambda f=frame: f)
            self.editor.canvas.refresh_background()

        try:
            self.editor.set_zones(self._to_editor(self._zones_for(key) or []))
        except Exception as e:
            logger.warning("zone workbench zones for %s: %s", key, e)
            self.editor.set_zones([])
        self._dirty = False
        self._describe()

    @staticmethod
    def _compact(editor) -> None:
        """Give the zone NAMES the room the buttons were taking.

        The editor asks for 25 px buttons; this analyser's stylesheet sets a
        `min-height` on QPushButton, and in Qt a stylesheet minimum overrides
        `setFixedHeight`. Every control in the panel came out at 40–42 px, and
        the zone list, the thing you actually read while drawing, was left
        107 px tall, about four names.

        Undone here rather than in the vendored copy: the copy has to stay
        identical to the rig's editor for the drift guard to mean anything,
        and it is this panel, not the editor, that is short of room.
        """
        editor.setStyleSheet(
            "QPushButton{min-height:22px;max-height:24px;padding:2px 6px;}"
            "QComboBox,QSpinBox,QDoubleSpinBox,QLineEdit"
            "{min-height:20px;max-height:22px;padding:1px 4px;}")
        zone_list = getattr(editor, "zone_list", None)
        if zone_list is not None:
            zone_list.setMinimumHeight(240)

    def _to_editor(self, pixel_zones: List[dict]) -> List[dict]:
        """Bundle zones (pixels) → what the editor stores (normalized).

        The bundle keeps zones in frame pixels because that is what re-zoning
        compares poses against. The editor keeps them normalized because that
        is what survives a resolution change. Neither is wrong; the conversion
        belongs here, at the one boundary between them, rather than in either
        of them.
        """
        from tools.offline_analysis.vendor.zone_coords import to_norm

        w, h = self._frame_size or (0, 0)
        if not (w and h):
            return [dict(z) for z in pixel_zones]
        out: List[dict] = []
        for z in pixel_zones:
            z = dict(z)
            pts = z.get("points") or []
            if pts:
                z["points"] = to_norm(pts, w, h)
            # An ellipse is its outline AND its centre and semi-axes, and the
            # editor draws the SHAPE from the first and the RESIZE HANDLES
            # from the other two. Converting only the outline and then
            # stamping the zone "normalized" made the editor denormalise a
            # centre that was already in pixels, putting every handle far
            # outside the frame, visible as a zone you could move but not
            # reshape.
            for key in ("center", "semi_axes"):
                pair = z.get(key)
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    z[key] = to_norm([pair], w, h)[0]
            z["coord_space"] = "normalized"
            z["shape_dim"] = [w, h]
            out.append(z)
        return out

    def _sync_nav(self):
        n = len(self._recs)
        self.btn_prev.setEnabled(self._index > 0)
        self.btn_next.setEnabled(self._index < n - 1)
        self.lbl_pos.setText(f"{self._index + 1} of {n}" if n else "")
        # The count is ON the button: "copy to all" is a different promise
        # with two recordings loaded than with forty.
        self.btn_all.setEnabled(n > 1)
        self.btn_all.setText(f"Copy to all {n}" if n > 1 else "Copy to all")

    # ── committing ───────────────────────────────────────────────

    def _on_edited(self):
        self._dirty = True
        self._describe()

    def _targets(self) -> List[str]:
        """What an ordinary save writes to: the recording on screen.

        Every other recording is reached through Copy to all, which is an act
        the operator performs rather than a mode they leave switched on and
        then forget.
        """
        key = self.current_key
        return [key] if key else []

    def _commit(self, targets: Optional[List[str]] = None,
                *, force: bool = False) -> int:
        """Write the current zones out. Called on every navigation, so that
        moving to the next recording cannot quietly discard an edit."""
        if not (self._dirty or force):
            return 0
        targets = self._targets() if targets is None else targets
        if not targets:
            return 0
        try:
            n = int(self._apply_to(self.editor.get_zones(), targets,
                                   self._frame_size) or 0)
        except Exception as e:
            logger.error("zone workbench apply: %s", e)
            self.lbl_state.setText(f"Could not apply: {e}")
            return 0
        self._dirty = False
        self.applied.emit(n)
        return n

    def _copy_to_all(self):
        """Put what is on screen onto every loaded recording.

        Deliberately not gated on ``_dirty``: the operator may have drawn on
        one recording, saved it, and only then decided the rest of the folder
        should match. Refusing because nothing had changed since the last save
        would be refusing the thing they just asked for.
        """
        keys = [k for _l, k in self._recs]
        if not keys:
            return
        n = self._commit(keys, force=True)
        if n:
            self.lbl_state.setText(
                f"Written to {n} recording(s), every one of them now carries "
                f"these zones and this scale.")

    def _write_into_files(self):
        """Put the zones into the recordings' own files, after asking.

        The copy beside a recording is safe and invisible; it is also easy to
        leave behind. Someone who has drawn zones for a folder and then moves
        the recordings, or sends them on, expects the zones to travel with
        them, and nothing about this panel said they would not.

        This is the one action here that changes the operator's data, so it
        says how many files, what it will do to them, and what the situation
        was until now.
        """
        self._commit()                      # never write a stale drawing
        self._write_keys(copied=0)

    def _write_into_all_files(self):
        """Put this drawing onto every recording, then into every file.

        Copying to all and writing the files are two halves of one intention,
        and doing only the first leaves the zones in the analysis folders,
        where a recording that is moved or forwarded loses them. Split across
        two buttons it also meant knowing to press them in the right order.
        """
        keys = [k for _label, k in self._recs]
        if not keys:
            return
        self._write_keys(copied=self._commit(keys, force=True))

    def _write_keys(self, *, copied: int):
        """Ask, then write the zones into every recording that carries them.

        ``copied`` is how many recordings the drawing was just applied to, so
        the question can say that it was, and is 0 when only the recording on
        screen was touched.
        """
        from PySide6.QtWidgets import QMessageBox

        keys = [k for k in self._zoned() if k]
        if not keys:
            self.lbl_state.setText(
                "Nothing to write yet: no loaded recording has zones.")
            return
        ask = QMessageBox(self)
        ask.setWindowTitle("Write zones into the data files")
        ask.setIcon(QMessageBox.Icon.Question)
        ask.setText(f"Write these zones into {len(keys)} recording file(s)?")
        ask.setInformativeText(
            (f"These zones and this scale were first copied onto all "
             f"{copied} loaded recording(s).\n\n" if copied else "")
            + "The zones go into each recording's own header block, where the "
            "rig writes them, so they travel with the file to another "
            "machine or another person.\n\n"
            "Only that block changes: every recorded row is copied through "
            "untouched, and the file is replaced only once it has been "
            "written in full.\n\n"
            "Until now these zones lived only beside the recording, in its "
            "analysis folder, safe, but left behind when the recording "
            "moves.")
        ask.setStandardButtons(QMessageBox.StandardButton.Yes
                               | QMessageBox.StandardButton.Cancel)
        ask.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if ask.exec() != QMessageBox.StandardButton.Yes:
            return

        try:
            written, failures = self._write_files(keys)
        except Exception as e:                       # never strand the panel
            logger.exception("writing zones into the files failed")
            self.lbl_state.setText(f"Could not write: {e}")
            return
        if failures and not written:
            self.lbl_state.setText("Could not write: " + failures[0])
            QMessageBox.warning(self, "Nothing was written",
                                "\n".join(failures[:8]))
            return
        note = f"Written into {written} recording file(s)."
        if failures:
            note += f" {len(failures)} could not be written."
            QMessageBox.warning(self, "Some files were not written",
                                "\n".join(failures[:8]))
        self.lbl_state.setText(note)
        self.applied.emit(written)

    def _finish(self):
        self._commit()
        self.finished.emit()

    # ── what the panel says about itself ─────────────────────────

    def _describe(self):
        zones = self.editor.get_zones() or []
        named = [z.get("name") for z in zones
                 if z.get("name") and z.get("type") != "scale"]
        line = next((z for z in zones if z.get("type") == "scale"), None)
        bits = [f"{len(named)} zone(s)" + (": " + ", ".join(named[:5])
                                           if named else "")]
        try:
            ppc = float(self._scale_for(self.current_key) or 0)
        except Exception:
            ppc = 0.0
        if line is not None:
            bits.append("scale line drawn")
        elif ppc > 0:
            bits.append(f"{ppc:.1f} px/cm, from the recording")
        else:
            bits.append("NO SCALE, distances stay in pixels")
        if self._dirty:
            bits.append("unsaved, will be written to this recording")
        if self._frame_note:
            bits.insert(0, self._frame_note)
        self.lbl_state.setText("   ·   ".join(bits))


def _field(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("field")
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight
                     | Qt.AlignmentFlag.AlignVCenter)
    return lbl
