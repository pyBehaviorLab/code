"""Choose a detector for an OFFLINE re-track, and nothing else.

The recording pipeline's :class:`TrackingConfigDialog` is 1,700 lines and two
tabs: a zone editor, event triggers, coordinate mapping and MCU-push gates.
None of that applies to re-tracking a file recorded weeks ago; there is no
MCU to push to and no trigger to fire, and opening it to pick a detector put
a zone editor on screen in answer to the question "which tracker?".

So this asks the one question, in the vocabulary of the thing it configures:
which detector, which model, and the few parameters that detector actually
has. It emits the same config dict the live dialog does, so
``_spec_from_tracking_config`` remains the single translation.

The live dialog is untouched. This is a second, smaller door to the same
place, not a replacement.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSpinBox,
                               QStackedWidget, QVBoxLayout, QWidget)

#: (config key, label), in the order the chooser offers them.
BACKENDS = (("background_subtraction", "Blob - background subtraction"),
            ("deeplabcut", "DeepLabCut"),
            ("sleap", "SLEAP"))

_KEYS = [k for k, _label in BACKENDS]


class OfflineTrackerDialog(QDialog):
    """Which detector to re-run over the video, and its parameters."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, parent=None,
                 theme: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Choose a tracking method")
        if theme:
            self.setStyleSheet(theme)
        self.resize(560, 430)
        self._cfg = dict(config or {})
        self._ok = None
        self._build()
        self._load(self._cfg)

    # -- construction ------------------------------------------------

    def _build(self):
        v = QVBoxLayout(self)
        v.setSpacing(8)

        lead = QLabel(
            "The detector to run over the video again. Zones, the scale and "
            "the measures are set in the panel behind this; none of them "
            "change what the tracker does.")
        lead.setWordWrap(True)
        lead.setObjectName("field")
        v.addWidget(lead)

        row = QHBoxLayout()
        row.addWidget(QLabel("Method"))
        self._method = QComboBox()
        for key, label in BACKENDS:
            self._method.addItem(label, key)
        self._method.currentIndexChanged.connect(self._on_method)
        row.addWidget(self._method, 1)
        v.addLayout(row)

        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_blob())
        self._pages.addWidget(self._build_model("deeplabcut"))
        self._pages.addWidget(self._build_model("sleap"))
        v.addWidget(self._pages, 1)

        self._note = QLabel("")
        self._note.setObjectName("note")
        self._note.setWordWrap(True)
        v.addWidget(self._note)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        self._ok = bb.button(QDialogButtonBox.StandardButton.Ok)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _build_blob(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._min_area = QSpinBox()
        self._min_area.setRange(1, 100000)
        self._min_area.setValue(40)
        self._min_area.setToolTip(
            "Blobs smaller than this are noise, not an animal.")
        f.addRow("Minimum area", self._min_area)

        self._max_area = QSpinBox()
        self._max_area.setRange(0, 1000000)
        self._max_area.setValue(0)
        self._max_area.setSpecialValueText("no limit")
        f.addRow("Maximum area", self._max_area)

        self._bg_mode = QComboBox()
        self._bg_mode.addItem("median of the video (best)", "median")
        self._bg_mode.addItem("simple, no background", "simple")
        self._bg_mode.setToolTip(
            "A median background costs one pass over the video and separates "
            "the animal far better than none.")
        f.addRow("Background", self._bg_mode)

        self._detect_dark = QCheckBox("The animal is darker than the arena")
        self._detect_dark.setChecked(True)
        f.addRow("", self._detect_dark)
        return w

    def _build_model(self, kind: str) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        row = QHBoxLayout()
        lbl = QLabel("none chosen")
        lbl.setWordWrap(True)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda _=False, k=kind: self._pick_model(k))
        row.addWidget(lbl, 1)
        row.addWidget(btn)
        host = QWidget()
        host.setLayout(row)
        f.addRow("Model folder", host)

        conf = QDoubleSpinBox()
        conf.setRange(0.0, 1.0)
        conf.setSingleStep(0.05)
        conf.setDecimals(2)
        conf.setValue(0.55)
        conf.setToolTip(
            "A keypoint below this confidence is treated as MISSING for that "
            "frame rather than as a position.")
        f.addRow("Confidence at least", conf)

        n = QSpinBox()
        n.setRange(1, 8)
        n.setValue(1)
        n.setToolTip("How many animals the model should find per frame.")
        n.valueChanged.connect(lambda *_: self._refresh_model_card(kind))
        n.valueChanged.connect(self._on_method)
        f.addRow("Animals", n)

        # Names, because with a crop window they are not decoration: each named
        # animal gets its OWN window, and the columns are numbered in this
        # order (`nose` is the first name, `nose#2` the second). Unnamed
        # animals cannot have per-identity windows at all.
        ids = QLineEdit()
        ids.setPlaceholderText("male, female")
        ids.setToolTip(
            "One name per animal, comma-separated. With a tracking crop each "
            "name gets its own window; the order also fixes which animal is "
            "which column.")
        ids.textChanged.connect(lambda *_: self._refresh_model_card(kind))
        ids.textChanged.connect(self._on_method)
        row = f.rowCount()
        f.addRow("Names", ids)
        setattr(self, "_%s_ids" % kind, ids)
        setattr(self, "_%s_ids_row" % kind, row)

        # What the model says it is. Read from the file beside the weights, so
        # it is answerable before anything is loaded, and on a host with no
        # inference stack, which is where a plan is usually made.
        card = QLabel("")
        card.setWordWrap(True)
        card.setStyleSheet("color:#9aa3b4; font-size:11px;")
        f.addRow("", card)

        # How the frame becomes the model's input, and which engine runs it.
        # The SAME widget the live dialog uses: a retrack that cannot express
        # what the live run did cannot reproduce it.
        from tools.offline_analysis.analyze.pose_engine_panel import PoseEnginePanel

        panel = PoseEnginePanel(
            backend="dlc" if kind == "deeplabcut" else "sleap", compact=True)
        panel.changed.connect(self._on_method)
        f.addRow("", panel)
        setattr(self, "_%s_panel" % kind, panel)

        setattr(self, "_%s_card" % kind, card)
        setattr(self, "_%s_model_lbl" % kind, lbl)
        setattr(self, "_%s_conf" % kind, conf)
        setattr(self, "_%s_n" % kind, n)
        setattr(self, "_%s_form" % kind, f)
        setattr(self, "_%s_model" % kind, "")
        if kind == "deeplabcut":
            self._dlc_resize = QDoubleSpinBox()
            self._dlc_resize.setRange(0.1, 1.0)
            self._dlc_resize.setSingleStep(0.1)
            self._dlc_resize.setValue(1.0)
            self._dlc_resize.setToolTip(
                "Downscale each frame before inference: faster, and less "
                "accurate on a small animal.")
            f.addRow("Resize", self._dlc_resize)
        return w

    # -- behaviour ---------------------------------------------------

    def _pick_model(self, kind: str):
        start = getattr(self, "_%s_model" % kind, "") or ""
        d = QFileDialog.getExistingDirectory(self, "%s model folder" % kind,
                                             start)
        if not d:
            return
        setattr(self, "_%s_model" % kind, d)
        getattr(self, "_%s_model_lbl" % kind).setText(os.path.basename(d) or d)
        self._refresh_model_card(kind)
        self._on_method()

    def _refresh_model_card(self, kind: str):
        """Say what the chosen model IS, and where the run disagrees with it.

        The dialog asks for a body-part count and an animal count that the
        model file usually already states. Where the two differ, the model is
        right about itself and the project is only right about what was
        wanted, so the difference is shown rather than resolved silently.
        """
        card = getattr(self, "_%s_card" % kind, None)
        if card is None:
            return
        from tools.offline_analysis.engine.trackers import ModelInfo

        path = getattr(self, "_%s_model" % kind, "") or ""
        panel = getattr(self, "_%s_panel" % kind, None)
        n = getattr(self, "_%s_n" % kind, None)
        if panel is not None:
            panel.set_model_path(path)
            panel.set_n_animals(int(n.value()) if n is not None else 1,
                                self._identities(kind))
        info = ModelInfo.read(path)
        if not path:
            card.setText("")
            return
        if not info.ok:
            card.setText("no config found beside this model, its body parts, "
                         "family and input size are unknown")
            return
        n = getattr(self, "_%s_n" % kind, None)
        clashes = info.disagreements(
            n_animals=int(n.value()) if n is not None else None)
        lines = [info.summary()]
        lines += ["! " + w for w in info.warnings]
        lines += ["! " + c for c in clashes]
        text = "\n".join(lines)
        card.setText(text)

    def _identities(self, kind=None) -> list:
        """The names typed for this backend, in the order they were typed."""
        edit = getattr(self, "_%s_ids" % (kind or self.method), None)
        if edit is None:
            return []
        return [part.strip() for part in edit.text().split(",") if part.strip()]

    def _sync_identity_row(self):
        """Ask for names only where they change something.

        One animal needs no names; it is the single-animal path either way,
        so the row would be a question with no consequence.
        """
        for kind in ("deeplabcut", "sleap"):
            edit = getattr(self, "_%s_ids" % kind, None)
            form = getattr(self, "_%s_form" % kind, None)
            spin = getattr(self, "_%s_n" % kind, None)
            if edit is None or form is None:
                continue
            show = spin is not None and int(spin.value()) > 1
            edit.setVisible(show)
            label = form.labelForField(edit)
            if label is not None:
                label.setVisible(show)

    def _on_method(self, *_):
        self._sync_identity_row()
        self._pages.setCurrentIndex(_KEYS.index(self.method))
        problems = self.problems()
        self._note.setText(problems[0] if problems else "")
        if self._ok is not None:
            self._ok.setEnabled(not problems)

    def problems(self):
        """Why this choice could not run, worst first.

        Asked of :class:`TrackSpec` rather than re-implemented here, so the
        dialog cannot disagree with the runner about what is usable.
        """
        from tools.offline_analysis.engine.retrack2d import TrackSpec

        model = ("" if self.method == "background_subtraction"
                 else getattr(self, "_%s_model" % self.method, ""))
        # Built from what the dialog currently holds, so a mode that cannot run
        # on this model is refused HERE, at the moment it is chosen, rather
        # than at minute one of a six-hour retrack.
        panel = getattr(self, "_%s_panel" % self.method, None)
        params: Dict[str, Any] = {}
        n = 1
        if panel is not None:
            vals = panel.values()
            params = {"input_mode": vals["pose_input_mode"],
                      "input_w": vals["pose_input_w"],
                      "input_h": vals["pose_input_h"],
                      "runtime": vals["runtime"], "device": vals["device"],
                      "precision": vals["precision"],
                      "identities": self._identities()}
            spin = getattr(self, "_%s_n" % self.method, None)
            n = int(spin.value()) if spin is not None else 1
        return TrackSpec(backend=self.method, model_path=model,
                         n_instances=n, params=params).validate()

    @property
    def method(self) -> str:
        return self._method.currentData() or "background_subtraction"

    # -- the contract with the caller --------------------------------

    def _load(self, cfg: Dict[str, Any]):
        i = self._method.findData(cfg.get("method", "background_subtraction"))
        self._method.setCurrentIndex(max(0, i))
        for kind, key in (("deeplabcut", "dlc"), ("sleap", "sleap")):
            sub = cfg.get(key) or {}
            if sub.get("model_path"):
                setattr(self, "_%s_model" % kind, sub["model_path"])
                getattr(self, "_%s_model_lbl" % kind).setText(
                    os.path.basename(sub["model_path"]))
            getattr(self, "_%s_conf" % kind).setValue(
                float(sub.get("confidence", 0.55) or 0.55))
            # How many animals, and who they are. Both were only ever WRITTEN
            # before, so a reopened dialog offered to retrack one anonymous
            # animal however the project was configured.
            getattr(self, "_%s_n" % kind).setValue(
                max(1, int(cfg.get("n_animals", 1) or 1)))
            getattr(self, "_%s_ids" % kind).setText(
                ", ".join(str(x) for x in (cfg.get("identities") or [])))
            panel = getattr(self, "_%s_panel" % kind, None)
            if panel is not None:
                panel.load(cfg, sub)
            # A model restored from the project gets the same card as one just
            # chosen, otherwise the disagreements only appear for whoever
            # happens to re-browse.
            self._refresh_model_card(kind)
        self._min_area.setValue(int(cfg.get("min_area", 40) or 40))
        j = self._bg_mode.findData(cfg.get("bg_mode", "median"))
        self._bg_mode.setCurrentIndex(max(0, j))
        self._on_method()

    def get_tracking_config(self) -> Dict[str, Any]:
        """The same shape the live dialog emits.

        Carrying the incoming config forward means a recording that already
        names its body parts keeps them; only the keys this dialog actually
        asks about are overwritten.
        """
        out: Dict[str, Any] = dict(self._cfg)
        out["method"] = self.method
        out["n_animals"] = (1 if self.method == "background_subtraction"
                            else int(getattr(self, "_%s_n" % self.method).value()))
        out["min_area"] = int(self._min_area.value())
        if self._max_area.value():
            out["max_area"] = int(self._max_area.value())
        out["bg_mode"] = self._bg_mode.currentData()
        out["detect_dark"] = bool(self._detect_dark.isChecked())
        for kind, key in (("deeplabcut", "dlc"), ("sleap", "sleap")):
            sub = dict(out.get(key) or {})
            sub["model_path"] = getattr(self, "_%s_model" % kind, "")
            sub["confidence"] = float(getattr(self, "_%s_conf" % kind).value())
            if kind == self.method:
                out["identities"] = self._identities(kind)
            if kind == "deeplabcut":
                sub["resize"] = float(self._dlc_resize.value())
            panel = getattr(self, "_%s_panel" % kind, None)
            if panel is not None:
                vals = panel.values()
                # The input mode belongs to the RUN (one arena, one frame
                # source), the engine to the MODEL, the same split the live
                # config uses, so one translation serves both.
                if kind == self.method:
                    for k in ("pose_input_mode", "pose_input_w", "pose_input_h",
                              "pose_crop_conf_min", "pose_crop_good_min",
                              "pose_crop_reacquire"):
                        out[k] = vals[k]
                for k in ("runtime", "device", "precision"):
                    sub[k] = vals[k]
                if kind == "sleap" and vals["centroid_model_path"]:
                    sub["centroid_model_path"] = vals["centroid_model_path"]
                if kind == "deeplabcut":
                    sub["model_type"] = vals["runtime"]
            out[key] = sub
        return out
