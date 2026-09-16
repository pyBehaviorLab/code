"""Renders :class:`CameraFeature` descriptors into live controls.

The camera setup dialog no longer hardcodes a settings form per vendor. A
backend reports what its camera has (see ``source/video/cameras/features.py``)
and this widget draws it:

* a UVC webcam reports three features and gets three rows,
* a FLIR reports its node map and gets the full grouped panel,
* a newly added SDK gets a working panel with no changes here.

Two views share one renderer. :class:`CameraFeaturePanel` shows the curated
tier grouped by function; :class:`CameraFeatureTree` shows every feature the
camera reported, searchable, with its SDK node name, so no option is ever
unreachable.

Ranges, enum entries and write access all come from the descriptor, which the
backend read from the sensor. Nothing here invents a limit.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from source.gui.theme import THEME
from source.log import get_logger
from source.video.cameras.features import (
    ACCESS_RO,
    KIND_BOOL,
    KIND_COMMAND,
    KIND_ENUM,
    KIND_FLOAT,
    KIND_INT,
    TIER_CURATED,
    group_features,
)

logger = get_logger()

_P = THEME.palette

_GROUP_TITLE_QSS = (
    f"color:{_P.text_dim}; font-size:10px; font-weight:700;"
    " letter-spacing:0.6px; padding:2px 0;"
)
_HELP_QSS = f"color:{_P.text_muted}; font-size:10px;"
_RANGE_QSS = f"color:{_P.text_muted}; font-size:10px;"
_BADGE_LIVE = (
    f"color:{_P.success}; border:1px solid {_P.success}; border-radius:8px;"
    " padding:0 6px; font-size:9px; font-weight:700;"
)
_BADGE_RECONNECT = (
    f"color:{_P.warning}; border:1px solid {_P.warning}; border-radius:8px;"
    " padding:0 6px; font-size:9px; font-weight:700;"
)


def _decimals_for(feature) -> int:
    """Editor precision, taken from the camera's own step and range.

    A sensor that steps exposure in whole microseconds should not show
    ``4800.000``: the extra digits imply a precision the device does not have.
    """
    inc = feature.increment
    if inc:
        inc = float(inc)
        if inc >= 1:
            return 0
        for decimals in range(1, 7):
            if round(inc, decimals) == inc:
                return decimals
        return 6
    lo, hi = feature.minimum, feature.maximum
    if lo is not None and hi is not None and (float(hi) - float(lo)) >= 100:
        return 0
    return 2


def _fmt(value, feature) -> str:
    if value is None:
        return ", "
    if feature.kind == KIND_FLOAT:
        text = f"{float(value):.6g}"
    elif feature.kind == KIND_INT:
        text = str(int(value))
    else:
        text = str(value)
    return f"{text} {feature.unit}".strip()


class _FeatureRow(QtWidgets.QWidget):
    """One descriptor rendered as label + editor (+ range hint)."""

    changed = QtCore.Signal(str, object)

    def __init__(self, feature, value=None, parent=None):
        super().__init__(parent)
        self.feature = feature
        self._editor = None
        self._emitting = False

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(8)

        label = QtWidgets.QLabel(feature.label)
        label.setMinimumWidth(132)
        label.setMaximumWidth(132)
        label.setWordWrap(True)
        if feature.help:
            label.setToolTip(feature.help)
        lay.addWidget(label)

        editor = self._build_editor(value)
        self._editor = editor
        lay.addWidget(editor, 1)

        hint = self._range_hint()
        if hint:
            hint_label = QtWidgets.QLabel(hint)
            hint_label.setStyleSheet(_RANGE_QSS)
            hint_label.setMinimumWidth(96)
            lay.addWidget(hint_label)

    # -- editors ---------------------------------------------------------
    def _build_editor(self, value):
        f = self.feature
        if f.access == ACCESS_RO or not f.writable:
            w = QtWidgets.QLabel(_fmt(value, f))
            w.setStyleSheet(f"color:{_P.text};")
            return w
        if f.kind == KIND_BOOL:
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(lambda v: self._emit(bool(v)))
            return w
        if f.kind == KIND_ENUM:
            w = QtWidgets.QComboBox()
            for opt_value, opt_label in f.options:
                w.addItem(str(opt_label), opt_value)
            if value is not None:
                idx = w.findData(value)
                if idx >= 0:
                    w.setCurrentIndex(idx)
            w.currentIndexChanged.connect(
                lambda _i, c=w: self._emit(c.currentData()))
            return w
        if f.kind == KIND_COMMAND:
            w = QtWidgets.QPushButton(f.label)
            w.clicked.connect(lambda: self._emit(True))
            return w
        if f.kind == KIND_INT:
            w = QtWidgets.QSpinBox()
            w.setRange(int(f.minimum) if f.minimum is not None else -10**9,
                       int(f.maximum) if f.maximum is not None else 10**9)
            if f.increment:
                w.setSingleStep(max(1, int(f.increment)))
            if f.unit:
                w.setSuffix(f" {f.unit}")
            if value is not None:
                w.setValue(int(value))
            w.valueChanged.connect(lambda v: self._emit(int(v)))
            return w
        w = QtWidgets.QDoubleSpinBox()
        w.setDecimals(_decimals_for(f))
        w.setRange(float(f.minimum) if f.minimum is not None else -1e9,
                   float(f.maximum) if f.maximum is not None else 1e9)
        if f.increment:
            w.setSingleStep(float(f.increment))
        if f.unit:
            w.setSuffix(f" {f.unit}")
        if value is not None:
            w.setValue(float(value))
        w.valueChanged.connect(lambda v: self._emit(float(v)))
        return w

    def _range_hint(self) -> str:
        f = self.feature
        if f.kind not in (KIND_FLOAT, KIND_INT):
            return ""
        if f.minimum is None and f.maximum is None:
            return ""
        if f.minimum == f.maximum:
            return ""
        lo = "" if f.minimum is None else f"{f.minimum:g}"
        hi = "" if f.maximum is None else f"{f.maximum:g}"
        return f"{lo}–{hi} {f.unit}".strip()  # noqa: RUF001 (en dash reads as a range)

    # -- value -----------------------------------------------------------
    def _emit(self, value):
        if self._emitting:
            return
        self.changed.emit(self.feature.key, self.feature.clamp(value))

    def set_enabled_state(self, enabled: bool):
        if self._editor is not None:
            self._editor.setEnabled(bool(enabled))


class CameraFeaturePanel(QtWidgets.QWidget):
    """Curated features, grouped, with a live/reconnect badge per group."""

    featureChanged = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: dict[str, _FeatureRow] = {}
        self._features: list = []
        self._values: dict = {}

        self._lay = QtWidgets.QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(4)

        self._empty = QtWidgets.QLabel(
            "Connect this camera to read the options it supports.")
        self._empty.setStyleSheet(_HELP_QSS)
        self._empty.setWordWrap(True)
        self._lay.addWidget(self._empty)
        self._lay.addStretch(1)

    # -- population -------------------------------------------------------
    def set_features(self, features, values=None, *, tier=TIER_CURATED):
        """Rebuild the panel from descriptors. ``values`` seeds the editors."""
        self._clear()
        self._features = [f for f in (features or [])
                          if tier is None or f.tier == tier]
        self._values = dict(values or {})

        if not self._features:
            self._empty.setVisible(True)
            return
        self._empty.setVisible(False)

        insert_at = self._lay.count() - 1        # keep the trailing stretch
        for group_name, group_feats in group_features(self._features):
            header = self._build_group_header(group_name, group_feats)
            self._lay.insertWidget(insert_at, header)
            insert_at += 1
            for feature in group_feats:
                row = _FeatureRow(feature, self._values.get(feature.key))
                row.changed.connect(self._on_row_changed)
                self._rows[feature.key] = row
                self._lay.insertWidget(insert_at, row)
                insert_at += 1
        self._refresh_dependencies()

    def _build_group_header(self, name, group_feats):
        box = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(box)
        lay.setContentsMargins(0, 8, 0, 2)
        lay.setSpacing(6)
        title = QtWidgets.QLabel(name.upper())
        title.setStyleSheet(_GROUP_TITLE_QSS)
        lay.addWidget(title)
        writable = [f for f in group_feats if f.writable]
        if writable:
            all_live = all(f.live for f in writable)
            badge = QtWidgets.QLabel("LIVE" if all_live else "NEEDS RECONNECT")
            badge.setStyleSheet(_BADGE_LIVE if all_live else _BADGE_RECONNECT)
            badge.setToolTip(
                "Applies to the running camera immediately." if all_live else
                "Stored now; applied the next time this camera connects.")
            lay.addWidget(badge)
        lay.addStretch(1)
        return box

    def _clear(self):
        self._rows.clear()
        while self._lay.count() > 2:                    # empty label + stretch
            item = self._lay.takeAt(1)
            w = item.widget()
            if w is not None and w is not self._empty:
                w.deleteLater()

    # -- interaction ------------------------------------------------------
    def _on_row_changed(self, key, value):
        self._values[key] = value
        self._refresh_dependencies()
        self.featureChanged.emit(key, value)

    def _refresh_dependencies(self):
        for feature in self._features:
            row = self._rows.get(feature.key)
            if row is not None:
                row.set_enabled_state(feature.is_enabled_by(self._values))

    def values(self) -> dict:
        return dict(self._values)


class CameraFeatureTree(QtWidgets.QWidget):
    """Every feature the camera reported, searchable, with SDK node names."""

    featureChanged = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._features: list = []
        self._values: dict = {}

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search features…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)
        lay.addWidget(self.search)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Feature", "Value", "SDK node"])
        self.tree.setRootIsDecorated(True)
        # Alternating rows fight the dark palette, the second colour lands
        # near the background and the text disappears. Explicit colours instead.
        self.tree.setAlternatingRowColors(False)
        self.tree.setStyleSheet(
            f"QTreeWidget {{ background:{_P.surface}; color:{_P.text};"
            f" border:1px solid {_P.surface_border}; }}"
            f"QTreeWidget::item {{ padding:2px 0; }}"
            f"QTreeWidget::item:selected {{ background:{_P.accent};"
            f" color:{_P.bg}; }}"
            f"QHeaderView::section {{ background:{_P.surface_2};"
            f" color:{_P.text_dim}; border:0; border-bottom:1px solid"
            f" {_P.surface_border}; padding:3px 6px; font-size:10px; }}"
        )
        self.tree.header().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.tree, 1)

        self.count_label = QtWidgets.QLabel("")
        self.count_label.setStyleSheet(_HELP_QSS)
        lay.addWidget(self.count_label)

    def set_features(self, features, values=None):
        self._features = list(features or [])
        self._values = dict(values or {})
        self.tree.clear()

        by_category: dict[str, list] = {}
        for f in self._features:
            by_category.setdefault(f.category or f.group or "Other", []).append(f)

        header_font = self.font()
        header_font.setBold(True)
        header_brush = QtGui.QBrush(QtGui.QColor(_P.accent))
        for category in sorted(by_category):
            parent = QtWidgets.QTreeWidgetItem(self.tree, [category, "", ""])
            parent.setFirstColumnSpanned(True)
            parent.setFont(0, header_font)
            parent.setForeground(0, header_brush)
            for f in sorted(by_category[category], key=lambda x: x.label):
                value = self._values.get(f.key)
                item = QtWidgets.QTreeWidgetItem(
                    parent, [f.label, _fmt(value, f), f.sdk_node or f.key])
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, f.key)
                tip = f.help or ""
                if not f.writable:
                    tip = (tip + "\n\nRead-only on this camera.").strip()
                if tip:
                    for col in range(3):
                        item.setToolTip(col, tip)
            parent.setExpanded(True)

        writable = sum(1 for f in self._features if f.writable)
        self.count_label.setText(
            f"{len(self._features)} features reported · {writable} writable")

    def _apply_filter(self, text):
        needle = (text or "").strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            visible_children = 0
            for j in range(parent.childCount()):
                child = parent.child(j)
                hit = (not needle
                       or needle in child.text(0).lower()
                       or needle in child.text(2).lower())
                child.setHidden(not hit)
                visible_children += int(hit)
            parent.setHidden(visible_children == 0)
            if needle and visible_children:
                parent.setExpanded(True)
