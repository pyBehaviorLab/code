"""
Real-time statistics visualization with table and multi-subplot layout.

This module provides:
- TemplateLoader: Load statistics templates from external JSON files
- StatisticsAutoDetector: Auto-detects statistics patterns from task prints
- StatisticsTableWidget: Table for displaying statistics summary
- CircularProgressWidget: Circular progress bar for percentage metrics
- StatisticsPlotWidget: Multi-subplot widget for box-based statistics
- BoxStatisticsCalculator: Calculate statistics from event counts
- StatisticsDataConsumer: Data consumer wrapper for pyControl integration
- StatsCanvas: Main tab widget combining all components
"""

import json
import time
import re
import statistics as _statistics
import numpy as np
# pandas is NOT imported at module level, its import is slow and the canvas
# isn't shown until opened. The plot moving-average uses numpy
# (``_rolling_mean`` below); pandas is imported lazily only in the Excel-export
# path.
from datetime import datetime
from collections import defaultdict, deque, Counter
from pathlib import Path
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
from source.log import get_logger
from source.gui.theme import THEME as _THEME
from source.communication.message import MsgType
from source.stats.format import to_canvas_config

logger = get_logger()


def _rolling_mean(y, window):
    """Trailing moving average (pure numpy).

    Each output point is the mean of up to ``window`` preceding samples
    (fewer near the start). Vectorised via a cumulative sum.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y
    csum = np.cumsum(np.insert(y, 0, 0.0))
    idx = np.arange(n)
    lo = np.maximum(0, idx - window + 1)
    counts = idx - lo + 1
    return (csum[idx + 1] - csum[lo]) / counts


# ─── Plot palette ────────────────────────────────────────────────────────────
# Per-box colors used by line + bar plots. Chosen from a colour-blind-safe
# qualitative palette so series are distinguishable. Order: cycle by box_id.
_BOX_COLORS = (
    "#38bdf8",  # 1 sky
    "#34d399",  # 2 mint
    "#fbbf24",  # 3 amber
    "#f87171",  # 4 rose
    "#a855f7",  # 5 violet
    "#ec4899",  # 6 pink
    "#14b8a6",  # 7 teal
    "#f97316",  # 8 orange
    "#60a5fa",  # 9 blue
    "#84cc16",  # 10 lime
)


# pyqtgraph global defaults, dark canvas + light foreground so the few
# bits we don't explicitly theme below also read right.
pg.setConfigOption("background", _THEME.palette.surface)
pg.setConfigOption("foreground", _THEME.palette.text_muted)
pg.setConfigOption("antialias", True)


def _box_color(setup_id: int) -> str:
    """Stable per-box hex colour for plots, cycles the qualitative palette."""
    return _BOX_COLORS[(int(setup_id) - 1) % len(_BOX_COLORS)]


# Pre-built QColor for the muted axis pen, pyqtgraph's mkPen rejects
# CSS-style rgba(...) strings, so we synthesise the QColor here.
_AXIS_PEN_COLOR = QtGui.QColor(255, 255, 255, 46)   # ~0.18 alpha
_GRID_PEN_COLOR = QtGui.QColor(255, 255, 255, 26)   # ~0.10 alpha


def _theme_pg_plot(plot_widget) -> None:
    """Apply the dark statistics theme to a pyqtgraph ``PlotWidget``.

    Sets canvas background, axis pen/text/font, and grid alpha so
    circular/line/bar all render in the same dark vocabulary.
    """
    p = _THEME.palette
    plot_widget.setBackground(p.surface)
    pen = pg.mkPen(color=_AXIS_PEN_COLOR, width=1)
    txt = QtGui.QColor(p.text_muted)
    label_font = QtGui.QFont(_THEME.font.family, 9)
    for ax_name in ("left", "bottom"):
        ax = plot_widget.getAxis(ax_name)
        ax.setPen(pen)
        ax.setTextPen(txt)
        ax.setTickFont(label_font)
    plot_widget.showGrid(x=False, y=True, alpha=0.18)


# =============================================================================
# SAFE STATISTICAL FUNCTIONS FOR FORMULA EVALUATION
# =============================================================================

def _safe_mean(x):
    """Arithmetic mean; returns 0 for empty/scalar inputs."""
    if isinstance(x, (int, float)):
        return float(x)
    if not x:
        return 0.0
    return float(_statistics.mean(x))

def _safe_stdev(x):
    """Sample standard deviation; returns 0 for empty/short inputs."""
    if isinstance(x, (int, float)):
        return 0.0
    if not x or len(x) < 2:
        return 0.0
    return float(_statistics.stdev(x))

def _safe_cv(x):
    """Coefficient of variation (stdev/mean); returns 0 when undefined."""
    m = _safe_mean(x)
    if m == 0:
        return 0.0
    return _safe_stdev(x) / m

def _safe_median(x):
    """Median; returns 0 for empty inputs."""
    if isinstance(x, (int, float)):
        return float(x)
    if not x:
        return 0.0
    return float(_statistics.median(x))

def _safe_min(x):
    """Minimum; returns 0 for empty inputs."""
    if isinstance(x, (int, float)):
        return float(x)
    if not x:
        return 0.0
    return float(min(x))

def _safe_max(x):
    """Maximum; returns 0 for empty inputs."""
    if isinstance(x, (int, float)):
        return float(x)
    if not x:
        return 0.0
    return float(max(x))

def _safe_len(x):
    """Length; returns 0 for non-iterable inputs."""
    if isinstance(x, (int, float)):
        return 0
    if not x:
        return 0
    return len(x)

def _safe_sum(x):
    """Sum; returns 0 for empty inputs."""
    if isinstance(x, (int, float)):
        return float(x)
    if not x:
        return 0.0
    return float(sum(x))


# Globals dict for safe formula evaluation
FORMULA_GLOBALS = {
    "__builtins__": {},
    "np": np,
    "mean": _safe_mean,
    "stdev": _safe_stdev,
    "cv": _safe_cv,
    "median": _safe_median,
    "min": _safe_min,
    "max": _safe_max,
    "len": _safe_len,
    "sum": _safe_sum,
    "abs": abs,
    "round": round,
}


class _CountersDict(dict):
    """Eval-locals dict for stat formulas.

    Missing keys default to ``0`` so a formula referencing an unseen
    counter doesn't crash. Names in ``FORMULA_GLOBALS`` (``mean``, ``sum``,
    ``cv``, ``len``, etc.) instead raise ``KeyError`` so Python's name
    lookup falls through to the globals dict (a plain ``defaultdict(int)``
    would shadow ``mean`` with ``0`` and break ``mean(...)`` calls).
    """

    _GLOBALS_NAMES = frozenset(FORMULA_GLOBALS) - {"__builtins__"}

    def __missing__(self, key):
        if key in _CountersDict._GLOBALS_NAMES:
            raise KeyError(key)
        return 0


# =============================================================================
# TEMPLATE LOADER
# =============================================================================

class TemplateLoader:
    """Load statistics templates from external JSON files."""
    # Single source of truth, paths.stats_templates_dir points at
    # ``experiments/config/stats_templates/``.
    from source import paths as _paths
    TEMPLATES_DIR = Path(_paths.stats_templates_dir)
    _cache = {}

    @classmethod
    def load_all(cls) -> dict:
        """Load all *.json from templates directory into cache."""
        cls._cache.clear()
        templates_dir = cls.TEMPLATES_DIR
        if not templates_dir.exists():
            logger.warning(f"Templates directory not found: {templates_dir}")
            return cls._cache
        for json_file in sorted(templates_dir.glob("*.json")):
            try:
                with open(json_file, 'r') as f:
                    template = json.load(f)
                name = json_file.stem  # filename without extension
                # Templates may be authored in the new flat schema; the
                # matcher and apply_template consume the legacy 4-section
                # shape (to_canvas_config also synthesises the
                # auto_detect_keywords the matcher needs). Idempotent for
                # already-legacy files.
                cls._cache[name] = to_canvas_config(template)
                logger.debug(f"Loaded template: {name} from {json_file.name}")
            except Exception as e:
                logger.error(f"Error loading template {json_file.name}: {e}")
        logger.info(f"Loaded {len(cls._cache)} templates from {templates_dir}")
        return cls._cache

    @classmethod
    def get(cls, name) -> dict | None:
        """Get template by name (lazy-loads on first call)."""
        if not cls._cache:
            cls.load_all()
        return cls._cache.get(name)

    @classmethod
    def get_all(cls) -> dict:
        """Get all templates (lazy-loads)."""
        if not cls._cache:
            cls.load_all()
        return cls._cache


# =============================================================================
# STATISTICS AUTO-DETECTION
# =============================================================================

class StatisticsAutoDetector:
    """Generic auto-detector: counts prints, matches templates, generates fallback configs.

    Contains ZERO task-specific knowledge.  All domain knowledge lives in
    external JSON files (stats_templates/ and tasks/<folder>/stats.json|config.json).
    """

    # Generic regex for "key: value" pairs in any print string
    _VALUE_PATTERN = re.compile(r"(\w+):\s*([0-9]*\.?[0-9]+)")

    def __init__(self):
        self.print_counts = defaultdict(int)

    def analyze_print(self, print_string):
        """Record a print statement."""
        if not isinstance(print_string, str):
            print_string = str(print_string)
        self.print_counts[print_string] += 1

    def detect_task_type(self):
        """Delegate to match_template(), all knowledge is in external JSON."""
        if not self.print_counts:
            return "unknown"
        return match_template(self.print_counts)

    def generate_default_config(self):
        """Build a purely generic config from the most frequent prints.

        Creates one counter per frequent print (exact match, agg=count),
        one column per counter, and one bar plot per counter.
        No assumptions about what the prints mean.
        """
        top_prints = self.get_top_prints(n=15)

        counters = {}
        calculations = {}
        columns = [
            {"key": "box_id", "label": "Box", "format": "{}"},
            {"key": "subject_id", "label": "Subject", "format": "{}"},
            {"key": "task", "label": "Task", "format": "{}"},
        ]
        plots = []

        for print_str, _count in top_prints:
            # Derive a safe key from the print string
            key = re.sub(r"[^a-zA-Z0-9_]", "_", print_str).strip("_").lower()
            if not key:
                continue

            counters[key] = {"type": "print", "match": print_str}
            calculations[key] = key   # identity: expose the count directly

            # Use the original print string as the column label
            label = print_str[:20]  # truncate long strings
            columns.append({"key": key, "label": label, "format": "{:.0f}"})

            # Add a bar plot for the first 4 counters
            if len(plots) < 4:
                plots.append({
                    "name": key,
                    "type": "bar",
                    "metric": key,
                    "title": label,
                    "color": ["#2ecc71", "#3498db", "#e74c3c", "#e67e22"][len(plots)],
                    "ylabel": "Count",
                })

        config = {
            "mode": "auto",
            "update_interval_ms": 1000,
            "counters": counters,
            "calculations": calculations,
            "columns": columns,
            "plots": plots,
        }

        logger.info(f"Generated generic auto-config: {len(counters)} counters")
        return config

    def get_top_prints(self, n=10):
        """Get the n most common print statements."""
        return Counter(self.print_counts).most_common(n)

    def has_sufficient_data(self, min_samples=5):
        """Check if we have enough data for auto-detection."""
        return len(self.print_counts) >= min_samples

    def reset(self):
        """Reset all collected data."""
        self.print_counts.clear()


# =============================================================================
# TEMPLATE-BASED MATCHING AND APPLICATION
# =============================================================================


def match_template(print_counts):
    """Auto-select best template based on detected prints."""
    if not print_counts:
        return "generic"

    templates = TemplateLoader.get_all()
    if not templates:
        return "generic"

    # Convert prints to lowercase for matching
    lower_prints = {p.lower() for p in print_counts}

    scores = {}
    for template_name, template in templates.items():
        if template_name == "generic":
            scores[template_name] = 0  # Fallback option
            continue

        keywords = template.get("auto_detect_keywords", [])
        if not keywords:
            continue

        # Calculate match score. Keywords must be lowercased like the
        # prints, or any capitalised keyword can never match.
        matches = sum(1 for kw in keywords
                      if any(str(kw).lower() in p for p in lower_prints))
        score = matches / len(keywords) if keywords else 0
        scores[template_name] = score

    if not scores:
        return "generic"

    # Return template with highest score (must be > 0.5 to be confident)
    best_template = max(scores, key=scores.get)
    best_score = scores[best_template]

    if best_score >= 0.5:
        logger.info(f"Matched template '{best_template}' with confidence {best_score:.2f}")
        return best_template
    else:
        logger.info(f"No good template match (best: {best_template} = {best_score:.2f}), using generic")
        return "generic"


def apply_template(template_name, print_counts=None):
    """Apply a template and customize based on actual prints if provided."""
    template = TemplateLoader.get(template_name)
    if template is None:
        logger.warning(f"Unknown template '{template_name}', using generic")
        template = TemplateLoader.get("generic") or {}

    logger.info(f"Applied template: {template_name} - {template.get('description', '')}")
    return {
        "mode": "template",
        "template": template_name,
        "update_interval_ms": template.get("update_interval_ms", 1000),
        "counters": template.get("counters", {}),
        "calculations": template.get("calculations", {}),
        "columns": template.get("columns", []),
        "plots": template.get("plots", [])
    }


# =============================================================================
# STATISTICS WIDGETS
# =============================================================================


class StatisticsTableWidget(QtWidgets.QTableWidget):
    """Table widget for displaying statistics summary"""

    # Emitted with a setup_id when a row's Controls button is clicked.
    controls_clicked = QtCore.Signal(int)

    # Width of the per-row Controls button / its column.
    CONTROLS_BTN_W = 30

    # Timer ("elapsed") column styling. Colour matches the Live Status widget
    # timer (theme palette.focus); larger monospace font for readability. The
    # column is pinned to a FIXED width so the table doesn't reshuffle each
    # second as the clock ticks.
    TIMER_COLOR = "#a855f7"      # == THEME.palette.focus (Live Status timer)
    TIMER_FONT_PT = 13           # bigger than the 9pt table body

    @classmethod
    def _timer_font(cls):
        return QtGui.QFont("Cascadia Mono", cls.TIMER_FONT_PT,
                           QtGui.QFont.Weight.Bold)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = None

        # Styling - theme-adaptive
        self.setStyleSheet("""
            QTableWidget {
                gridline-color: palette(mid);
                border: 1px solid palette(mid);
                border-radius: 4px;
                font-size: 9pt;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background-color: #3498db;
                color: white;
                padding: 6px;
                border: none;
                font-weight: bold;
                font-size: 9pt;
            }
            QTableWidget::item:selected {
                background-color: palette(highlight);
            }
        """)

        self.horizontalHeader().setStretchLastSection(False)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        # Display-only table, no row selection (otherwise clicking the Controls
        # button highlighted the whole row dark green).
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)

    @staticmethod
    def _expand_sequence_columns(columns):
        """Fan any sequence column into ``count`` per-index columns.

        A column authored as ``{"key":"trials","sequence":true,"count":14}``
        becomes 14 columns keyed ``trials#1``..``trials#14`` labelled 1..14.
        The canvas fills each cell from the box's captured token sequence.
        Non-sequence columns pass through untouched.
        """
        out = []
        for c in (columns or []):
            if c.get("sequence"):
                n = int(c.get("count", 0) or 0)
                key = c.get("key", "seq")
                for i in range(1, n + 1):
                    out.append({"key": f"{key}#{i}", "label": str(i),
                                "format": "{}"})
            else:
                out.append(c)
        return out

    @staticmethod
    def _ensure_timer_column(columns):
        """Prepend the synthetic ``elapsed`` (Timer) + ``controls`` columns.

        ``elapsed`` is the display-only session clock (HH:MM:SS from MCU time),
        filled by the central process tick; ``controls`` holds a per-row button
        that opens that box's Controls dialog. Both are injected for every table
        rather than authored in a template. No-op when there are no stats
        columns."""
        cols = list(columns or [])
        if not cols or any(c.get("key") == "elapsed" for c in cols):
            return cols
        return [{"key": "elapsed", "label": "Timer", "format": "{}"},
                {"key": "controls", "label": ""}] + cols

    def _make_controls_button(self, setup_id):
        """Per-row button mirroring the box's Controls button. Event-driven,
        emits ``controls_clicked`` only on click, no ongoing cost."""
        btn = QtWidgets.QPushButton("⚙")  # gear
        btn.setFixedSize(self.CONTROLS_BTN_W, 18)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        btn.setToolTip(f"Open Controls for Box {setup_id}")
        # Flat, subtle button so it reads as an inline action, not a raised
        # OS-chrome button stamped into the cell.
        btn.setStyleSheet(
            "QPushButton {"
            " border: 1px solid rgba(255,255,255,0.18);"
            " border-radius: 4px; background: rgba(255,255,255,0.06);"
            " color: #c8cde0; font-size: 11pt; padding: 0px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.14);"
            " border-color: rgba(255,255,255,0.30); }"
            "QPushButton:pressed { background: rgba(255,255,255,0.22); }")
        btn.clicked.connect(lambda _=False, sid=setup_id: self.controls_clicked.emit(sid))
        return btn

    def setupTable(self, columns, box_count):
        """Setup table based on column config, DATA-PRESERVING.

        Called at task UPLOAD (loadConfigFromTask). Per-box reset is owned by
        ``onFrameworkStart`` (run start), so here we only (re)build structure
        when columns or box_count change and never overwrite a cell that
        already has data.
        """
        # Previously-applied column keys come from _col_index_by_key, not
        # self.config, loadConfig overwrites self.config with the new config
        # before calling us, so self.config would compare against itself.
        columns = self._ensure_timer_column(self._expand_sequence_columns(columns))
        prev_keys = list(getattr(self, '_col_index_by_key', {}).keys()) or None
        new_keys = [c.get('key') for c in columns]

        self.config = {'columns': columns}
        self.box_count = box_count
        # updateCell uses this for O(1) key→column lookup instead of
        # rescanning self.config['columns'] on every cell update.
        self._col_index_by_key = {c.get('key'): i for i, c in enumerate(columns)}

        if not columns:
            return

        # Structure unchanged (same column keys + same dimensions) → keep
        # every existing cell as is (the common re-upload case), leaving
        # other boxes' stats untouched.
        if (prev_keys == new_keys
                and self.rowCount() == box_count
                and self.columnCount() == len(columns)):
            return

        self.setRowCount(box_count)
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels([col['label'] for col in columns])
        self.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        # Pin the Timer column to a FIXED width (stat columns keep
        # ResizeToContents) so the table doesn't reshuffle as the clock ticks.
        # Width = the timer font's advance for "00:00:00" + cell padding.
        elapsed_idx = self._col_index_by_key.get('elapsed')
        if elapsed_idx is not None:
            self.horizontalHeader().setSectionResizeMode(
                elapsed_idx, QtWidgets.QHeaderView.ResizeMode.Fixed)
            tw = QtGui.QFontMetrics(self._timer_font()).horizontalAdvance("00:00:00")
            self.setColumnWidth(elapsed_idx, tw + 28)
        # Pin the Controls button column to a fixed width too.
        controls_idx = self._col_index_by_key.get('controls')
        if controls_idx is not None:
            self.horizontalHeader().setSectionResizeMode(
                controls_idx, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.setColumnWidth(controls_idx, self.CONTROLS_BTN_W + 12)

        # Create only MISSING cells, preserve any cell that already holds
        # a value (e.g. a box still running with old structure). Fresh
        # cells start empty; onFrameworkStart clears a box's row at start.
        for row in range(box_count):
            for col_idx, col_config in enumerate(columns):
                key = col_config.get('key')
                if key == 'controls':
                    # Create the per-row button once (a cell widget has no item).
                    if self.cellWidget(row, col_idx) is None:
                        self.setCellWidget(row, col_idx,
                                           self._make_controls_button(row + 1))
                    continue
                if self.item(row, col_idx) is not None:
                    continue
                # The elapsed/Timer cell starts at 00:00:00 (the central tick
                # overwrites it for running boxes); other cells start blank.
                is_timer = key == 'elapsed'
                item = QtWidgets.QTableWidgetItem("00:00:00" if is_timer else "")
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                # Box-id cells share the Timer's font + colour (setText preserves
                # both, so the value fill keeps the look).
                if is_timer or key == 'box_id':
                    item.setForeground(QtGui.QColor(self.TIMER_COLOR))
                    item.setFont(self._timer_font())
                if 'color' in col_config:
                    item.setBackground(QtGui.QColor(col_config['color']))
                self.setItem(row, col_idx, item)

    def updateCell(self, setup_id, column_key, value):
        """Update a specific cell"""
        col_idx = getattr(self, '_col_index_by_key', {}).get(column_key)
        if col_idx is None:
            return

        row = setup_id - 1
        if row < 0 or row >= self.rowCount():
            return

        item = self.item(row, col_idx)
        if item is None:
            return
        col_config = self.config['columns'][col_idx]
        format_str = col_config.get('format', '{:.2f}')
        if isinstance(value, (int, float)):
            item.setText(format_str.format(value))
        else:
            item.setText(str(value))

    def updateRow(self, setup_id, data_dict):
        """Update entire row for a box"""
        for key, value in data_dict.items():
            self.updateCell(setup_id, key, value)

    def setRowError(self, setup_id, on):
        """Tint this box's row red (text) when in error; reset on clear.
        Driven by StatsCanvas.set_box_error -> box_alerts.apply_box_alerts."""
        row = setup_id - 1
        if row < 0 or row >= self.rowCount():
            return
        idx = getattr(self, '_col_index_by_key', {})
        timer_cols = {idx.get('elapsed'), idx.get('box_id')}
        for col in range(self.columnCount()):
            item = self.item(row, col)
            if item is None:
                continue
            if on:
                item.setForeground(QtGui.QColor("#ff6b6b"))
            elif col in timer_cols:
                item.setForeground(QtGui.QColor(self.TIMER_COLOR))  # keep Timer/Box colour
            else:
                item.setForeground(QtGui.QBrush())  # reset to default


class CircularProgressWidget(QtWidgets.QWidget):
    """Donut-style progress indicator for percentage metrics.

    Visual recipe (matches the rest of the dark app):
    - Track: 8 px ``palette.surface_elev_2`` ring (faint, just enough to read).
    - Arc: 8 px conic gradient from the metric colour to a 30 % lighter
      shade, drawn with rounded caps so it reads as a polished bar.
    - Glass cap: 1 px translucent white slice along the top edge,
      same trick as button_style().
    - Number: tabular-numeric, bold, centred in the metric colour.
    - Caption: subject id above, box id below in muted text.
    """

    def __init__(self, setup_id, metric_name, subject_id="", color="#34d399", parent=None):
        super().__init__(parent)
        self.setup_id = setup_id
        self.metric_name = metric_name
        self.subject_id = subject_id
        self.value = 0.0
        self.arc_color = color
        self.setMinimumSize(130, 130)
        self.setMaximumSize(170, 170)

    def setValue(self, value):
        """Set the value displayed in the donut.

        Accepts either a percentage (0–100) or a fraction (0–1) and
        auto-scales fractions up to 0–100 so the ring fills
        correctly.  Heuristic: values ≤ 1.0 are treated as fractions
        (matches conventional ``moving_average``, accuracy as ratio,
        etc.).  Anything >1 is treated as already-percent.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = 0.0
        if 0 <= v <= 1.0:
            v *= 100.0
        self.value = max(0.0, min(100.0, v))
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        width, height = self.width(), self.height()
        side = min(width, height)
        painter.translate(width / 2, height / 2)

        p = _THEME.palette
        track_color = QtGui.QColor(p.surface_elev_2)
        muted = QtGui.QColor(p.text_muted)

        arc_qcolor = QtGui.QColor(self.arc_color or p.focus)
        # Lighter shade for the gradient highlight along the arc.
        arc_light = QtGui.QColor(arc_qcolor)
        arc_light = arc_light.lighter(135)

        ring_thickness = 9
        radius = side / 2.55
        rect = QtCore.QRectF(-radius, -radius, radius * 2, radius * 2)

        # Track ring
        pen = QtGui.QPen(track_color, ring_thickness)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawEllipse(rect)

        # Progress arc with a vertical gradient → reads as a 3D-ish ring.
        if self.value > 0:
            grad = QtGui.QLinearGradient(QtCore.QPointF(0, -radius),
                                         QtCore.QPointF(0,  radius))
            grad.setColorAt(0.0, arc_light)
            grad.setColorAt(1.0, arc_qcolor)
            arc_pen = QtGui.QPen(QtGui.QBrush(grad), ring_thickness)
            arc_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            painter.setPen(arc_pen)
            angle_span = int(self.value * 3.6 * 16)
            painter.drawArc(rect, 90 * 16, -angle_span)

        # Glass-cap highlight on the very top, 1 px translucent white sliver.
        cap_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 60), 1.2)
        cap_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(cap_pen)
        painter.drawArc(rect, 85 * 16, -10 * 16)

        # Subject id above the value (caption-style)
        if self.subject_id:
            painter.setPen(muted)
            f = QtGui.QFont(_THEME.font.family, 9, QtGui.QFont.Weight.Medium)
            painter.setFont(f)
            txt_rect = QtCore.QRectF(-radius, -radius * 0.55, radius * 2, 20)
            painter.drawText(
                txt_rect,
                QtCore.Qt.AlignmentFlag.AlignCenter,
                str(self.subject_id),
            )

        # Big tabular-numeric value in the metric colour
        painter.setPen(arc_qcolor)
        f = QtGui.QFont(_THEME.font.family, 22, QtGui.QFont.Weight.Bold)
        f.setStyleStrategy(QtGui.QFont.StyleStrategy.PreferAntialias)
        painter.setFont(f)
        val_rect = QtCore.QRectF(-radius, -radius * 0.2, radius * 2, radius * 0.55)
        painter.drawText(
            val_rect,
            QtCore.Qt.AlignmentFlag.AlignCenter,
            f"{self.value:.1f}",
        )

        # "%" suffix, smaller, muted, beside the number band
        painter.setPen(muted)
        f = QtGui.QFont(_THEME.font.family, 10, QtGui.QFont.Weight.DemiBold)
        painter.setFont(f)
        pct_rect = QtCore.QRectF(-radius, radius * 0.18, radius * 2, 16)
        painter.drawText(
            pct_rect,
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "%",
        )

        # Box label at the bottom, dim
        painter.setPen(muted)
        f = QtGui.QFont(_THEME.font.family, 9, QtGui.QFont.Weight.Bold)
        f.setLetterSpacing(QtGui.QFont.SpacingType.PercentageSpacing, 105)
        painter.setFont(f)
        box_rect = QtCore.QRectF(-radius, radius * 0.45, radius * 2, 18)
        painter.drawText(
            box_rect,
            QtCore.Qt.AlignmentFlag.AlignCenter,
            f"BOX {self.setup_id}",
        )

        painter.end()


class _PageScrollPlotWidget(pg.PlotWidget):
    """pg.PlotWidget whose mouse wheel scrolls the surrounding page instead
    of zooming the X axis. Wheeling over a plot now moves the view up/down
    like the rest of the tab; drag (left-button) still pans for inspection."""

    def wheelEvent(self, ev):
        # Ignore so the event bubbles to the parent QScrollArea, which then
        # scrolls vertically. Without this, pyqtgraph's ViewBox eats the
        # wheel and rescales the X axis.
        ev.ignore()


class StatisticsPlotWidget(QtWidgets.QWidget):
    """Widget containing multiple subplots for box-based statistics"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = None
        self.box_count = 0
        self.plot_widgets = {}
        self.plot_data = defaultdict(lambda: defaultdict(float))
        self.plot_sd = defaultdict(lambda: defaultdict(float))  # Store SD data
        # Line-plot history, bounded: a multi-hour session otherwise grows
        # these without limit and every refresh re-runs the rolling mean +
        # setData over the whole series, display degrades linearly. 5000
        # points is far beyond what a plot pixel-resolves.
        self.plot_history = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=5000)))
        self.plot_timestamps = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=5000)))
        self.plot_last_trial_count = defaultdict(lambda: defaultdict(int))  # Track trial count per box/plot
        self.subject_ids = {}  # Store subject IDs for each box
        self.box_metadata = {}  # Optional metadata map injected by parent (subject_id, etc.)
        self._dirty_plots = set()  # Track which plots need refreshing

        self._build_ui()

    def _build_ui(self):
        """Setup the plot area"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # Scroll area for plots
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_widget = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QVBoxLayout(scroll_widget)
        self.plot_layout.setSpacing(12)
        self.plot_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        # Placeholder
        self.placeholder_label = QtWidgets.QLabel("No plots configured")
        from source.gui.theme import THEME as _T_st
        self.placeholder_label.setStyleSheet(
            "QLabel {"
            f" color: {_T_st.palette.text_dim};"
            " font-size: 12pt;"
            " font-style: italic;"
            " padding: 40px;"
            "}"
        )
        self.placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.plot_layout.addWidget(self.placeholder_label)

    def _plots_signature(self, config, box_count):
        """A cheap identity for the plot layout, same signature means the
        donuts/plots are structurally identical and must NOT be rebuilt
        (rebuilding would delete every box's live plot)."""
        plots = (config or {}).get('plots') or []
        try:
            sig = tuple((p.get('name'), p.get('type')) for p in plots)
        except Exception:
            sig = None
        return (int(box_count), sig)

    def setupPlots(self, config, box_count):
        """Setup plots based on JSON config, DATA-PRESERVING.

        Called at task UPLOAD (loadConfig). Re-applying the same config must
        NOT delete the existing per-box plot widgets; per-box reset is owned
        by ``onFrameworkStart`` (run start). Only rebuild when the plot
        structure or box_count changes."""
        new_sig = self._plots_signature(config, box_count)
        if getattr(self, '_plots_sig', None) == new_sig and self.plot_widgets:
            # Structure unchanged → keep existing plots + their data.
            self.config = config
            self.box_count = box_count
            return
        self._plots_sig = new_sig
        self.config = config
        self.box_count = box_count

        # Clear existing plots
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.plot_widgets.clear()

        if not config or 'plots' not in config:
            self.placeholder_label = QtWidgets.QLabel(
                "Waiting for task data, plots appear when "
                "the session starts producing events."
            )
            self.placeholder_label.setStyleSheet(
                "QLabel {"
                f" color: {_THEME.palette.text_dim};"
                " font-size: 12pt;"
                " font-style: italic;"
                " padding: 40px;"
                " background: transparent;"
                "}"
            )
            self.placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.plot_layout.addWidget(self.placeholder_label)
            return

        # Create each subplot
        for plot_config in config['plots']:
            self._create_subplot(plot_config)

    def _create_subplot(self, plot_config):
        """Dispatch one plot_config dict to the type-specific builder.

        Three flavors: circular (donut card grid), line (pg.PlotWidget
        with legend + pan/zoom), bar/scatter (fixed-axis pg.PlotWidget).
        """
        plot_type = plot_config.get('type', 'bar')
        if plot_type == 'circular':
            self._create_circular_subplot(plot_config)
        elif plot_type == 'line':
            self._create_line_subplot(plot_config)
        else:
            self._create_static_subplot(plot_config)

    # ----- per-type subplot builders -----

    def _create_circular_subplot(self, plot_config):
        """Donut card, surface_elev container with a grid of per-box
        CircularProgressWidget instances."""
        plot_name = plot_config['name']
        plot_title = plot_config.get('title', plot_name)
        color = plot_config.get('color', '#4a90d9')

        p = _THEME.palette
        container = QtWidgets.QWidget()
        container.setObjectName("DonutCard")
        container.setStyleSheet(
            "QWidget#DonutCard {"
            f" background: {p.surface_elev};"
            f" border: 1px solid {p.surface_border_strong};"
            f" border-radius: {_THEME.radius.md}px;"
            "}"
        )
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(12, 10, 12, 12)
        vbox.setSpacing(6)

        title_label = QtWidgets.QLabel(plot_title)
        title_label.setStyleSheet(
            "QLabel {"
            f" color: {p.text};"
            " font-size: 11pt; font-weight: 700;"
            " letter-spacing: 0.4px; padding: 4px 0;"
            " background: transparent;"
            "}"
        )
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(title_label)

        grid_widget = QtWidgets.QWidget()
        grid_widget.setStyleSheet("QWidget { background: transparent; }")
        grid_layout = QtWidgets.QGridLayout(grid_widget)
        grid_layout.setSpacing(18)
        grid_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        circular_widgets = {}
        cols = min(5, self.box_count) if self.box_count else 1
        for setup_id in range(1, self.box_count + 1):
            row, col = (setup_id - 1) // cols, (setup_id - 1) % cols
            # Per-box color from qualitative palette unless the template
            # hard-codes a specific arc color.
            box_arc_color = color if color and color != '#4a90d9' else _box_color(setup_id)
            circ = CircularProgressWidget(setup_id, plot_title, color=box_arc_color)
            subj = self._get_subject_id(setup_id) if hasattr(self, "_get_subject_id") else ""
            if subj:
                circ.subject_id = subj
            grid_layout.addWidget(circ, row, col)
            circular_widgets[setup_id] = circ

        vbox.addWidget(grid_widget)
        self.plot_widgets[plot_name] = {
            'widget': container,
            'config': plot_config,
            'type': 'circular',
            'color': color,
            'circular_widgets': circular_widgets,
        }
        self.plot_layout.addWidget(container)

    def _create_line_subplot(self, plot_config):
        """Dark-themed line plot, pg.PlotWidget styled via
        _theme_pg_plot() so colours come from the theme tokens. Mouse pan/zoom
        on X axis only; legend in top-left."""
        plot_name = plot_config['name']
        plot_title = plot_config.get('title', plot_name)
        ylabel = plot_config.get('ylabel', '')
        ylim = plot_config.get('ylim', None)
        color = plot_config.get('color', '#4a90d9')

        p = _THEME.palette
        plot_widget = self._make_styled_plot_widget(plot_title, ylabel, p)

        x_axis_type = plot_config.get('x-axis', 'trials')
        x_label = 'Time (s)' if x_axis_type == 'time' else 'Trials'
        plot_widget.setLabel('bottom', x_label,
                             **{'color': p.text_muted, 'font-size': '10pt'})
        # Minimum keeps a plot readable; no maximum so the user can drag the
        # table/plot splitter and grow the plot area to whatever height they
        # want (overflow scrolls in the inner area).
        plot_widget.setMinimumHeight(180)
        plot_widget.setMouseEnabled(x=True, y=False)
        plot_widget.setMenuEnabled(False)

        if ylim and len(ylim) == 2:
            plot_widget.setYRange(ylim[0], ylim[1], padding=0)
            plot_widget.setLimits(yMin=ylim[0], yMax=ylim[1])

        # Dark surface + light text + glass cap on the legend.
        legend = plot_widget.addLegend(offset=(10, 10))
        try:
            legend.setBrush(pg.mkBrush(QtGui.QColor(p.surface_elev_2)))
            legend.setPen(pg.mkPen(color=_AXIS_PEN_COLOR, width=1))
            legend.setLabelTextColor(p.text)
        except Exception:
            pass

        self.plot_widgets[plot_name] = {
            'widget': plot_widget,
            'config': plot_config,
            'type': 'line',
            'color': color,
            'legend': legend,
            'window': plot_config.get('window', 10),
            'x_axis': x_axis_type,
            'line_items': {},  # PlotDataItem per box
            'start_time': time.time(),  # for time-based x-axis
        }
        self.plot_layout.addWidget(plot_widget)

    def _create_static_subplot(self, plot_config):
        """Bar or scatter plot, fixed x-range to box positions, no autoscale,
        no mouse interaction. Y-range honours config['ylim'] when set."""
        plot_name = plot_config['name']
        plot_title = plot_config.get('title', plot_name)
        ylabel = plot_config.get('ylabel', '')
        ylim = plot_config.get('ylim', None)
        plot_type = plot_config.get('type', 'bar')
        color = plot_config.get('color', '#4a90d9')

        p = _THEME.palette
        plot_widget = self._make_styled_plot_widget(plot_title, ylabel, p)
        plot_widget.setLabel('bottom', 'Box',
                             **{'color': p.text_muted, 'font-size': '10pt'})
        # No maximum height, let the splitter drive the plot-area size.
        plot_widget.setMinimumHeight(160)
        plot_widget.setMouseEnabled(x=False, y=False)
        plot_widget.setMenuEnabled(False)
        plot_widget.enableAutoRange(axis='x', enable=False)
        plot_widget.enableAutoRange(axis='y', enable=False)

        if ylim and len(ylim) == 2:
            plot_widget.setYRange(ylim[0], ylim[1], padding=0.05)

        x_ticks = [(i + 1, f'{i + 1}') for i in range(self.box_count)]
        plot_widget.getAxis('bottom').setTicks([x_ticks])

        x_min, x_max = 0.3, self.box_count + 0.7
        plot_widget.setXRange(x_min, x_max, padding=0)
        plot_widget.setLimits(xMin=x_min, xMax=x_max)  # hard limits

        self.plot_widgets[plot_name] = {
            'widget': plot_widget,
            'config': plot_config,
            'type': plot_type,
            'color': color,
            'x_min': x_min,
            'x_max': x_max,
        }
        self.plot_layout.addWidget(plot_widget)

    def _make_styled_plot_widget(self, plot_title, ylabel, palette):
        """Build a themed pg.PlotWidget with title + Y-label. Shared
        between line and static subplots."""
        plot_widget = _PageScrollPlotWidget()
        _theme_pg_plot(plot_widget)
        plot_widget.setTitle(
            f"<span style='color:{palette.text};font-weight:700'>{plot_title}</span>",
            size='12pt',
        )
        plot_widget.setLabel('left', ylabel,
                             **{'color': palette.text_muted, 'font-size': '10pt'})
        return plot_widget

    def updatePlot(self, plot_name, setup_id, value, sd=0.0, trial_count=None):
        """Update a specific plot with new data for a box

        Args:
            plot_name: Name of the plot
            setup_id: Box identifier
            value: The metric value
            sd: Standard deviation (for bar/scatter plots)
            trial_count: Current trial count (for line plots with x-axis="trials")
        """
        if plot_name not in self.plot_widgets:
            return

        self.plot_data[plot_name][setup_id] = value
        self.plot_sd[plot_name][setup_id] = sd
        self._dirty_plots.add(plot_name)

        # For line plots, also append to history and timestamp
        plot_info = self.plot_widgets[plot_name]
        if plot_info.get('type') == 'line':
            x_axis_type = plot_info.get('x_axis', 'trials')

            if x_axis_type == 'trials':
                if trial_count is not None:
                    # One point per trial, which is what the axis claims.
                    last_count = self.plot_last_trial_count[plot_name][setup_id]
                    if trial_count > last_count:
                        self.plot_history[plot_name][setup_id].append(value)
                        self.plot_last_trial_count[plot_name][setup_id] = trial_count
                else:
                    # No trial counter to key on. Append when the value CHANGES,
                    # which for a per-trial measure is once per trial anyway.
                    # Returning here instead, as this did, meant a config that
                    # named no trial_metric drew nothing at all, all session,
                    # silently. A line that is right about its shape and
                    # approximate about its x spacing beats an empty plot.
                    hist = self.plot_history[plot_name][setup_id]
                    if not hist or hist[-1] != value:
                        hist.append(value)
            else:
                # For time-based x-axis, append on every update
                self.plot_history[plot_name][setup_id].append(value)
                # Store elapsed time since plot creation
                start_time = plot_info.get('start_time', time.time())
                elapsed = time.time() - start_time
                self.plot_timestamps[plot_name][setup_id].append(elapsed)

    def refreshPlots(self, force=False):
        """Refresh plots that have new data (dirty-flag optimization).

        Dispatch:
            circular  -> always refreshed (cheap)
            line      -> persistent PlotDataItem per box, moving-avg smooth
            bar/scatter -> clear + redraw with per-box colours

        Args:
            force: If True, refresh all plots regardless of dirty state.
                   Use when clearing/resetting box data or updating labels.
        """
        for plot_name, plot_info in self.plot_widgets.items():
            plot_type = plot_info['type']

            # Circular widgets are cheap; always update.
            if plot_type == 'circular':
                self._refresh_circular_plot(plot_name, plot_info)
                continue

            # Skip non-dirty plots (no new data since last refresh).
            if not force and plot_name not in self._dirty_plots:
                continue

            if plot_type == 'line':
                self._refresh_line_plot(plot_name, plot_info)
            elif plot_type in ('bar', 'scatter'):
                self._refresh_static_plot(plot_name, plot_info)

        self._dirty_plots.clear()

    # ----- per-type refresh helpers -----

    def _refresh_circular_plot(self, plot_name, plot_info):
        """Update each box's circular-progress widget for this plot."""
        for setup_id, circ_widget in plot_info.get('circular_widgets', {}).items():
            if setup_id not in self.plot_data[plot_name]:
                continue
            circ_widget.setValue(self.plot_data[plot_name][setup_id])
            subj = self._get_subject_id(setup_id)
            circ_widget.subject_id = f"{subj}" if subj else ""
            circ_widget.update()

    def _refresh_line_plot(self, plot_name, plot_info):
        """Per-box history -> moving avg -> reuse or create a PlotDataItem.

        Per-box colours from the qualitative palette (matches bars + donuts
        so the same box gets the same colour everywhere).
        """
        plot_widget = plot_info['widget']
        window_size = plot_info.get('window', 10)
        x_axis_type = plot_info.get('x_axis', 'trials')
        line_items = plot_info.get('line_items', {})

        all_y_values = []
        max_x = 0

        for setup_id in range(1, self.box_count + 1):
            history = list(self.plot_history[plot_name][setup_id])
            if not history:
                # Hide existing line item if box has no data.
                if setup_id in line_items:
                    line_items[setup_id].setData([], [])
                continue

            if x_axis_type == 'time':
                timestamps = list(self.plot_timestamps[plot_name][setup_id])
                x_arr = np.array(timestamps) if timestamps else np.arange(1, len(history) + 1)
            else:
                x_arr = np.arange(1, len(history) + 1)

            y_arr = np.array(history)
            if len(y_arr) >= window_size:
                y_smooth = _rolling_mean(y_arr, window_size)
            else:
                y_smooth = y_arr

            # Reuse existing PlotDataItem or create a new one. Slightly
            # thicker stroke + smoothing for a polished look on dark bg.
            if setup_id in line_items:
                line_items[setup_id].setData(x_arr, y_smooth)
            else:
                subj = self._get_subject_id(setup_id)
                legend_name = f"Box {setup_id}" if not subj else f"{setup_id}-{subj}"
                pen = pg.mkPen(color=_box_color(setup_id), width=2.4, cosmetic=True)
                line_items[setup_id] = plot_widget.plot(
                    x_arr, y_smooth, pen=pen, name=legend_name, antialias=True)

            all_y_values.extend(y_smooth.tolist())
            if len(x_arr) > 0:
                max_x = max(max_x, float(np.max(x_arr)))

        plot_info['line_items'] = line_items
        if max_x > 0:
            plot_widget.setXRange(0, max_x * 1.05, padding=0.02)

        ylim = plot_info.get('config', {}).get('ylim', None)
        if ylim and len(ylim) == 2:
            plot_widget.setYRange(ylim[0], ylim[1], padding=0)
            plot_widget.setLimits(yMin=ylim[0], yMax=ylim[1])
        elif all_y_values:
            y_min = min(all_y_values)
            y_max = max(all_y_values)
            margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.5
            plot_widget.setYRange(y_min - margin, y_max + margin, padding=0)

    def _refresh_static_plot(self, plot_name, plot_info):
        """Bar or scatter, pull (x, y, sd) per box, clear, redraw, then
        set axes + tick labels. Shared between both static plot types so
        the data-prep + axes-handling logic is written once."""
        plot_widget = plot_info['widget']
        plot_type = plot_info['type']
        color = plot_info['color']

        x_arr, y_arr, sd_arr = self._collect_per_box_xy_sd(plot_name)
        self._apply_box_id_ticks(plot_widget)
        plot_widget.clear()

        if plot_type == 'bar':
            self._draw_bar_plot(plot_widget, x_arr, y_arr, sd_arr, color)
            self._draw_bar_value_labels(plot_widget, x_arr, y_arr, sd_arr)
        elif plot_type == 'scatter':
            self._draw_scatter_plot(plot_widget, x_arr, y_arr, sd_arr, color)

        # Fix x-range to the stored Box1..BoxN positions; no autoscaling.
        x_min, x_max = 0.3, self.box_count + 0.7
        plot_widget.setXRange(x_min, x_max, padding=0)
        plot_widget.setLimits(xMin=x_min, xMax=x_max)

        ylim = plot_info.get('config', {}).get('ylim', None)
        if ylim and len(ylim) == 2:
            plot_widget.setYRange(ylim[0], ylim[1], padding=0)
            plot_widget.setLimits(yMin=ylim[0], yMax=ylim[1])
        else:
            y_max = float(np.max(y_arr)) if y_arr.size else 0.0
            if sd_arr.size:
                y_max = max(y_max, float(np.max(y_arr + sd_arr)))
            upper = y_max * 1.1 if y_max > 0 else 1.0
            plot_widget.setYRange(0, upper, padding=0)
            plot_widget.setLimits(yMin=0, yMax=upper)

    # ----- static-plot drawing helpers -----

    def _collect_per_box_xy_sd(self, plot_name):
        """Return (x_arr, y_arr, sd_arr) covering box 1..box_count, with
        zero-fill for boxes that have no data yet."""
        x, y, sd = [], [], []
        for setup_id in range(1, self.box_count + 1):
            x.append(setup_id)
            if setup_id in self.plot_data[plot_name]:
                y.append(self.plot_data[plot_name][setup_id])
                sd.append(self.plot_sd[plot_name].get(setup_id, 0.0))
            else:
                y.append(0)
                sd.append(0)
        return np.array(x), np.array(y), np.array(sd)

    def _apply_box_id_ticks(self, plot_widget):
        """Bottom-axis tick labels: 'Box N' or 'N - <subject>' if subject set."""
        x_ticks = []
        for setup_id in range(1, self.box_count + 1):
            subj = self._get_subject_id(setup_id)
            label = f"{setup_id}" if not subj else f"{setup_id} - {subj}"
            x_ticks.append((setup_id, label))
        plot_widget.getAxis('bottom').setTicks([x_ticks])

    def _draw_bar_plot(self, plot_widget, x_arr, y_arr, sd_arr, color):
        """Per-box vertical bars with brush gradient (lighter top -> base
        colour at the bottom). Hand-coded template ``color`` wins over the
        qualitative palette when non-default."""
        p = _THEME.palette
        bar_width = 0.6
        template_color = color if color and color != '#4a90d9' else None
        for x, y, sd in zip(x_arr, y_arr, sd_arr):
            box_color_hex = template_color or _box_color(int(x))
            base = QtGui.QColor(box_color_hex)
            top = QtGui.QColor(base).lighter(135)
            grad = QtGui.QLinearGradient(
                QtCore.QPointF(x, 0), QtCore.QPointF(x, y))
            grad.setColorAt(0.0, top)
            grad.setColorAt(1.0, base)
            plot_widget.addItem(pg.BarGraphItem(
                x=np.array([x]), height=np.array([y]), width=bar_width,
                brush=QtGui.QBrush(grad),
                pen=pg.mkPen(color=QtGui.QColor(255, 255, 255, 60), width=1),
            ))
            if y > 0 and sd > 0:
                plot_widget.addItem(pg.ErrorBarItem(
                    x=np.array([x]), y=np.array([y]),
                    height=np.array([sd]),
                    pen=pg.mkPen(color=p.text_muted, width=1.2),
                ))

    def _draw_bar_value_labels(self, plot_widget, x_arr, y_arr, sd_arr):
        """Value labels above each bar (soft text, small tabular-numeric
        font so labels don't fight the bars)."""
        p = _THEME.palette
        f = QtGui.QFont(_THEME.font.family, 9, QtGui.QFont.Weight.DemiBold)
        for x, y, sd in zip(x_arr, y_arr, sd_arr):
            if y == 0:
                continue
            label_text = f'{y:.1f}±{sd:.1f}' if sd > 0.01 else f'{y:.1f}'
            item = pg.TextItem(text=label_text, anchor=(0.5, 1.2), color=p.text)
            item.setFont(f)
            item.setPos(x, y + sd)
            plot_widget.addItem(item)

    def _draw_scatter_plot(self, plot_widget, x_arr, y_arr, sd_arr, color):
        """Per-box scatter dots with soft white outline + optional error bars."""
        p = _THEME.palette
        for x, y, sd in zip(x_arr, y_arr, sd_arr):
            box_color_hex = (color if color and color != '#4a90d9'
                             else _box_color(int(x)))
            plot_widget.addItem(pg.ScatterPlotItem(
                x=np.array([x]), y=np.array([y]), size=12,
                pen=pg.mkPen(color=QtGui.QColor(255, 255, 255, 90), width=1.2),
                brush=QtGui.QColor(box_color_hex),
            ))
            if sd > 0:
                plot_widget.addItem(pg.ErrorBarItem(
                    x=np.array([x]), y=np.array([y]),
                    height=np.array([sd]),
                    pen=pg.mkPen(color=p.text_muted, width=1.2),
                ))

    def updateSubjectIds(self, subject_id_dict):
        """Update subject IDs for all boxes

        Args:
            subject_id_dict: Dictionary mapping box_id -> subject_id string
        """
        self.subject_ids = subject_id_dict.copy()
        logger.debug(f"Updated subject IDs: {self.subject_ids}")
        # Immediately refresh plots so ticks/circular labels update.
        self.refreshPlots(force=True)
        # Also propagate to statistics-level subject map so tables/metadata stay aligned.
        if hasattr(self, 'box_metadata'):
            for setup_id, subj in subject_id_dict.items():
                if setup_id in self.box_metadata:
                    self.box_metadata[setup_id]['subject_id'] = subj
    def clearBox(self, setup_id):
        """Clear all plot data for a specific box without affecting other boxes."""
        for plot_name in list(self.plot_data.keys()):
            self.plot_data[plot_name].pop(setup_id, None)
        for plot_name in list(self.plot_sd.keys()):
            self.plot_sd[plot_name].pop(setup_id, None)
        for plot_name in list(self.plot_history.keys()):
            self.plot_history[plot_name].pop(setup_id, None)
        for plot_name in list(self.plot_timestamps.keys()):
            self.plot_timestamps[plot_name].pop(setup_id, None)
        for plot_name in list(self.plot_last_trial_count.keys()):
            self.plot_last_trial_count[plot_name].pop(setup_id, None)
        # Remove persistent line items from PlotWidget (clears legend too)
        for plot_info in self.plot_widgets.values():
            if plot_info.get('type') == 'line':
                line_items = plot_info.get('line_items', {})
                if setup_id in line_items:
                    plot_info['widget'].removeItem(line_items[setup_id])
                    del line_items[setup_id]

    def _get_subject_id(self, setup_id):
        """Return subject ID for box, preferring explicit subject_ids mapping then box_metadata."""
        subj = self.subject_ids.get(setup_id, "")
        if subj:
            return subj.strip()
        if hasattr(self, "box_metadata") and setup_id in getattr(self, "box_metadata", {}):
            meta_subj = self.box_metadata[setup_id].get("subject_id", "")
            if meta_subj:
                return str(meta_subj).strip()
        return ""


class BoxStatisticsCalculator:
    """Calculate statistics for each box based on event counts.

    Supports pre-compiled formulas and regex patterns with hash-based cache
    invalidation, compilation happens once and is reused as long as the
    config (formulas/counters) doesn't change.
    """

    def __init__(self):
        self.event_counts = defaultdict(lambda: defaultdict(
            lambda: {'count': 0, 'sum': 0.0, 'last': None, 'values': []}
        ))
        self.stat_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=20)))  # Rolling window for SD
        self.latency_starts = defaultdict(dict)  # {box_id: {counter_key: mcu_timestamp}}
        # Ordered categorical sequences (e.g. per-trial C/F) captured by
        # ``type:"sequence"`` counters: {box_id: {counter_key: [token, ...]}}.
        # Kept out of the formula path (tokens are strings, not numbers) and
        # fanned into per-index table columns by the canvas.
        self.sequences = defaultdict(lambda: defaultdict(list))
        # First/last MCU timestamp seen per box → session duration in minutes.
        self.first_time = {}   # {box_id: fw_ms}
        self.last_time = {}    # {box_id: fw_ms}

        # Pre-compiled caches (invalidated by hash when config changes).
        self._compiled_formulas = {}   # {stat_name: compiled code object}
        self._formulas_snapshot = None  # last-compiled formulas (dict equality)
        self._compiled_regex = {}      # {counter_key: compiled re.Pattern}
        self._counters_snapshot = None  # last-compiled counters (dict equality)
        # Per-formula error dedup, key (box_id, stat_name, repr(exc)): log the
        # first occurrence, suppress repeats until config reload or run restart.
        self._logged_errors: set = set()

    def _ensure_compiled_formulas(self, formulas):
        """Compile formula strings to code objects if the formulas dict changed.

        Plain dict equality; this runs per box per tick, and the old
        json.dumps(sort_keys=True) + md5 per call was pure overhead (and
        an md5 in a djb2-only codebase)."""
        if formulas == self._formulas_snapshot:
            return  # Cache is still valid
        self._formulas_snapshot = dict(formulas)
        self._compiled_formulas = {}
        # Re-arm the per-formula error dedup so each broken formula in the
        # new config gets logged once.
        self._logged_errors.clear()
        for stat_name, formula in formulas.items():
            try:
                self._compiled_formulas[stat_name] = compile(formula, f'<stats:{stat_name}>', 'eval')
            except SyntaxError as e:
                logger.error(f"Syntax error compiling formula '{stat_name}': {e}")
                self._compiled_formulas[stat_name] = None

    def _ensure_compiled_counters(self, counters):
        """Compile regex patterns in counter configs if the counters dict changed."""
        if counters == self._counters_snapshot:
            return  # Cache is still valid
        import copy
        self._counters_snapshot = copy.deepcopy(counters)
        self._compiled_regex = {}
        for key, cfg in counters.items():
            regex_val = cfg.get("regex")
            if regex_val:
                try:
                    self._compiled_regex[key] = re.compile(regex_val)
                except re.error as e:
                    logger.error(f"Invalid regex for counter '{key}': {e}")
                    self._compiled_regex[key] = None

    def processData(self, setup_id, new_data, id2name=None, counters=None):
        """Process incoming data and update event/print counts.

        counters: optional dict from config describing patterns to count/sum/append/latency:
            {
              "key": {"type": "event"|"print"|"state"|"any"|"latency",
                      "match": "Exact", "regex": "pattern",
                      "agg": "count"|"sum"|"last"|"append",
                      "start": "Start_Print", "end": "End_Print",
                      "start_state": "state_name", "end_state": "state_name"}
            }
        """
        if counters is None:
            counters = {}
        else:
            self._ensure_compiled_counters(counters)

        for data_point in new_data:
            try:
                msg_type = data_point.type
                content = data_point.content
                mcu_time = getattr(data_point, 'time', None)
                name = None
                if msg_type == MsgType.EVENT:
                    if id2name and isinstance(content, int) and content in id2name:
                        name = id2name[content]
                    else:
                        name = str(content)
                elif msg_type == MsgType.PRINT:
                    name = str(content)

                # State transitions are resolved SEPARATELY and never folded
                # into ``name``. The framework emits a state row on every
                # transition without the task asking, so measuring between two
                # states needs no print added to the task at all - which is the
                # point: a latency should not require editing a validated task.
                # Kept out of ``name`` so the default per-name counting and the
                # auto-detector see exactly what they saw before.
                state_name = str(content) if msg_type == MsgType.STATE else None

                # Session duration bookkeeping, first/last MCU timestamp.
                if mcu_time is not None:
                    if setup_id not in self.first_time:
                        self.first_time[setup_id] = mcu_time
                    self.last_time[setup_id] = mcu_time

                # Default counting by name
                if name is not None:
                    self.event_counts[setup_id][name]['count'] += 1

                # Apply configured counters (pattern/regex/latency)
                for key, cfg in counters.items():
                    ctype = cfg.get("type", "any")

                    # --- Sequence counter (ordered categorical tokens) ---
                    # cfg["map"] = {token: match_prefix}; each PRINT whose text
                    # starts with a prefix appends that token in arrival order
                    # (e.g. "Correct L ..." → "C"). Powers per-trial C/F columns.
                    if ctype == "sequence":
                        if msg_type == MsgType.PRINT and name is not None:
                            for token, prefix in (cfg.get("map") or {}).items():
                                if prefix and name.startswith(prefix):
                                    self.sequences[setup_id][key].append(token)
                                    break
                        continue

                    # --- Latency counter ---
                    if ctype == "latency":
                        # ``start``/``end`` match a print or event name.
                        # ``start_state``/``end_state`` match a STATE name, so a
                        # latency can be measured from the state machine itself
                        # without the task printing anything. The two forms mix
                        # freely: "from entering the sample state to the
                        # Correct_response print" is a valid pair.
                        start_state = cfg.get("start_state")
                        end_state = cfg.get("end_state")
                        start_match = cfg.get("start", "")
                        end_match = cfg.get("end", "")
                        hit_start = ((start_state is not None
                                      and state_name == start_state)
                                     or (start_state is None and name is not None
                                         and name == start_match))
                        hit_end = ((end_state is not None
                                    and state_name == end_state)
                                   or (end_state is None and name is not None
                                       and name == end_match))
                        # A self-transition would otherwise start and stop the
                        # same measurement in one row; starting wins, so the
                        # next arrival at the end marker closes it.
                        if hit_start:
                            self.latency_starts[setup_id][key] = (
                                mcu_time if mcu_time is not None
                                else time.time() * 1000)
                        elif hit_end:
                            if key in self.latency_starts.get(setup_id, {}):
                                start_ts = self.latency_starts[setup_id].pop(key)
                                now_ms = (mcu_time if mcu_time is not None
                                          else time.time() * 1000)
                                latency_s = (now_ms - start_ts) / 1000.0
                                self.event_counts[setup_id][key]['values'].append(latency_s)
                                self.event_counts[setup_id][key]['count'] += 1
                        continue

                    # --- Standard counters (print/event/state/any) ---
                    if ctype == "event" and msg_type != MsgType.EVENT:
                        continue
                    if ctype == "print" and msg_type != MsgType.PRINT:
                        continue
                    if ctype == "state":
                        # Counting entries to a state needs no print either:
                        # "how many trials" is usually "how many times the task
                        # entered init_trial".
                        if state_name is None:
                            continue
                        name = state_name

                    match_val = cfg.get("match")
                    agg = cfg.get("agg", "count")

                    # Exact match
                    if match_val and name == match_val:
                        if agg == "sum":
                            continue
                        elif agg == "last":
                            self.event_counts[setup_id][key]['last'] = name
                            self.event_counts[setup_id][key]['count'] += 1
                        elif agg == "append":
                            self.event_counts[setup_id][key]['values'].append(1.0)
                            self.event_counts[setup_id][key]['count'] += 1
                        else:  # count
                            self.event_counts[setup_id][key]['count'] += 1
                        continue

                    # Regex match (uses pre-compiled pattern)
                    compiled_re = self._compiled_regex.get(key)
                    if compiled_re is not None:
                        m = compiled_re.search(name or "")
                        if m:
                            if agg == "sum":
                                try:
                                    val = float(m.group(1))
                                    self.event_counts[setup_id][key]['sum'] += val
                                except Exception:
                                    pass
                            elif agg == "last":
                                try:
                                    val = float(m.group(1))
                                    self.event_counts[setup_id][key]['last'] = val
                                    self.event_counts[setup_id][key]['count'] += 1
                                except Exception:
                                    pass
                            elif agg == "append":
                                try:
                                    val = float(m.group(1))
                                    self.event_counts[setup_id][key]['values'].append(val)
                                    self.event_counts[setup_id][key]['count'] += 1
                                except Exception:
                                    pass
                            else:  # count
                                self.event_counts[setup_id][key]['count'] += 1
            except Exception as e:
                logger.debug(f"Error processing data for box {setup_id}: {e}")

    def calculateStatistics(self, setup_id, formulas):
        """Calculate statistics based on formulas.

        Uses pre-compiled code objects (compiled once, reused until formulas change).
        Supports statistical functions: mean, stdev, cv, median, min, max, len, sum, abs, round.
        """
        self._ensure_compiled_formulas(formulas)
        results = {}

        # Flatten the nested dictionary structure for formulas. Only use
        # ``values`` when non-empty; otherwise fall back to last/count (the
        # default factory pre-populates an empty ``values`` list).
        counts = {}
        for key, val in self.event_counts[setup_id].items():
            if isinstance(val, dict):
                values_list = val.get('values') or []
                if values_list:
                    counts[key] = list(values_list)
                elif val.get('last') is not None:
                    counts[key] = val['last']
                else:
                    counts[key] = val.get('count', 0)
                counts[f"{key}_count"] = val.get('count', 0)
                counts[f"{key}_sum"] = val.get('sum', 0.0)
                if val.get('last') is not None:
                    counts[f"{key}_last"] = val['last']
                counts[f"{key}_values"] = list(values_list)
            else:
                counts[key] = val

        # Filter out counter keys that shadow FORMULA_GLOBALS function names
        # (e.g. a counter named 'sum', 'mean', 'len' would shadow the safe_ helper)
        _reserved = set(FORMULA_GLOBALS.keys())
        safe_counts = {k: v for k, v in counts.items() if k not in _reserved}
        eval_vars = _CountersDict(safe_counts)

        for stat_name in formulas.keys():
            try:
                code = self._compiled_formulas.get(stat_name)
                if code is None:
                    results[stat_name] = 0.0
                    continue

                result = eval(code, FORMULA_GLOBALS, eval_vars)

                if isinstance(result, list):
                    result = _safe_mean(result) if result else 0.0

                if isinstance(result, (int, float)):
                    if np.isnan(result) or np.isinf(result):
                        result = 0.0

                self.stat_history[setup_id][stat_name].append(result)
                results[stat_name] = result

            except ZeroDivisionError:
                results[stat_name] = 0.0
            except Exception as e:
                # Log each distinct (box, stat, error) once per session so a
                # broken formula doesn't spam the log every update interval.
                err_key = (setup_id, stat_name, repr(e))
                if err_key not in self._logged_errors:
                    self._logged_errors.add(err_key)
                    logger.error(
                        "Error calculating '%s' for box %s: %s "
                        "(further occurrences suppressed)",
                        stat_name, setup_id, e)
                results[stat_name] = 0.0

        return results

    def getStatSD(self, setup_id, stat_name):
        """Get standard deviation for a statistic"""
        history = list(self.stat_history[setup_id][stat_name])
        if len(history) > 1:
            return np.std(history)
        return 0.0

    def getSequence(self, setup_id, key):
        """Ordered token list captured by the ``key`` sequence counter."""
        return list(self.sequences.get(setup_id, {}).get(key, []))

    def getDuration(self, setup_id):
        """Session length in minutes from first→last MCU timestamp (0 if none)."""
        t0 = self.first_time.get(setup_id)
        t1 = self.last_time.get(setup_id)
        if t0 is None or t1 is None:
            return 0.0
        return max(0.0, (t1 - t0) / 60000.0)

    def clearBox(self, setup_id):
        """Clear statistics for a specific box"""
        if setup_id in self.event_counts:
            self.event_counts[setup_id].clear()
        if setup_id in self.stat_history:
            self.stat_history[setup_id].clear()
        if setup_id in self.latency_starts:
            self.latency_starts[setup_id].clear()
        if setup_id in self.sequences:
            self.sequences[setup_id].clear()
        self.first_time.pop(setup_id, None)
        self.last_time.pop(setup_id, None)
        # Re-arm the per-formula error dedup so a fresh run logs each error.
        self._logged_errors = {k for k in self._logged_errors
                               if k[0] != setup_id}

class StatisticsDataConsumer:
    """Wrapper to act as data consumer for a specific box"""

    def __init__(self, statistics_tab, setup_id):
        self.statistics_tab = statistics_tab
        self.setup_id = setup_id
        logger.debug(f"StatisticsDataConsumer created for box {setup_id}")

    def process_data(self, new_data):
        """Forward data to statistics tab with box_id"""
        logger.debug(f"Processing {len(new_data)} data items for box {self.setup_id}")
        self.statistics_tab.process_data_for_box(self.setup_id, new_data)


class _StatsGroup:
    """One config-signature group (small multiples).

    Boxes that share the same stats structure are rendered together in this
    block: its OWN table (only this group's columns) + its OWN plots (only
    this group's boxes as series). A rig running one task collapses to a
    SINGLE group, the unified view (header hidden when there is one group).
    """

    def __init__(self, signature, config):
        from source.gui.style_builders import groupbox_style
        self.signature = signature
        self.config = config
        self.box_ids = set()

        # Small floors on BOTH panels so the splitter can give almost the
        # whole height to whichever side the user drags toward (the other
        # side scrolls in its inner area).
        self.table = StatisticsTableWidget()
        self.table.setMinimumHeight(80)
        self.table.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                 QtWidgets.QSizePolicy.Policy.Expanding)
        self.plots = StatisticsPlotWidget()
        self.plots.setMinimumHeight(80)
        self.plots.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                 QtWidgets.QSizePolicy.Policy.Expanding)

        self.container = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(self.container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # Group header, hidden when there's only one group (unified view).
        self.header = QtWidgets.QLabel("")
        self.header.setStyleSheet(
            "QLabel { color: %s; font-size: 10pt; font-weight: 600; "
            "padding: 2px 4px; }" % _THEME.palette.text_dim)
        self.header.setVisible(False)
        outer.addWidget(self.header)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        # Collapsible so a user not interested in the plots can drag the
        # handle to the bottom and let the table take the entire height (and
        # vice versa). Floors above keep each side usable until collapsed.
        split.setChildrenCollapsible(True)
        split.setHandleWidth(4)
        tbox = QtWidgets.QGroupBox("Statistics Summary Table")
        tbox.setStyleSheet(groupbox_style("info"))
        tl = QtWidgets.QVBoxLayout(tbox)
        tl.setContentsMargins(8, 15, 8, 8)
        tl.addWidget(self.table)
        pbox = QtWidgets.QGroupBox("Real-Time Visualizations")
        pbox.setStyleSheet(groupbox_style("primary"))
        pl = QtWidgets.QVBoxLayout(pbox)
        pl.setContentsMargins(8, 15, 8, 8)
        pl.addWidget(self.plots)
        split.addWidget(tbox)
        split.addWidget(pbox)
        split.setSizes([300, 300])
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        outer.addWidget(split, 1)


class StatsCanvas(QtWidgets.QWidget):
    """Enhanced statistics tab with table + multi-subplot layout"""

    # Emitted with a setup_id when a row's Controls button is clicked; the
    # owning MainWindow opens that box's Controls dialog.
    box_controls_requested = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        # ``config`` is the ACTIVE STRUCTURE config (the union of every box's
        # columns + plots) that drives the table/plot LAYOUT. ``box_config``
        # holds each box's OWN config and is what each box is COMPUTED with, so
        # a box running Task A is never evaluated with Task B's formulas. When
        # ``box_config`` is empty the canvas falls back to the single global
        # ``config`` (manual load).
        self.config = None
        self.box_config = {}     # box_id -> that box's own stats config
        # Signature-keyed group blocks (small multiples). All-same boxes ->
        # one group -> the unified view.
        self.groups = {}         # signature -> _StatsGroup
        self.box_signature = {}  # box_id -> signature (which group it's in)
        self.config_path = None
        self.box_count = 0
        self.calculator = BoxStatisticsCalculator()
        self.framework_running = {}  # Track framework state per box_id
        self.box_metadata = {}  # Store subject_id, task, etc. per box_id
        self.box_data_consumers = {}  # Store StatisticsDataConsumer wrapper per box_id
        # Per-box error isolation: a box whose formula/calc raises is added
        # here and skipped on subsequent ticks (warned once) while the SHARED
        # update timer keeps running for the other boxes. Cleared for a box at
        # its onFrameworkStart (fresh run).
        self._stats_error_boxes = set()  # box_ids with a calc error this run
        self.id2name_map = {}  # Map box_id -> {ID: name} from sm_info

        # Universal statistics system
        self.auto_detectors = {}  # Per-box auto-detectors for multi-box setups
        self.config_mode = None  # "custom", "template", "auto", or None
        self.auto_config_generated = False  # Track if auto-config was used

        self._build_ui()

    def _build_ui(self):
        """Setup the statistics tab UI"""
        layout = QtWidgets.QVBoxLayout(self)
        # Zero left margin, the outer verticalLayout in operant.py /
        # maze.py already carries the 26 px sidebar-toggle gutter, so
        # adding 25 px here too would double-indent everything.
        layout.setContentsMargins(0, 8, 8, 8)
        layout.setSpacing(10)

        # Footer bar with export/detach buttons
        self.header_layout = QtWidgets.QHBoxLayout()
        self.header_layout.setSpacing(8)
        self.header_layout.setContentsMargins(0, 6, 0, 0)

        # Export button
        export_btn = QtWidgets.QPushButton("Export Data")
        export_btn.setFixedSize(110, 24)
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 3px;
                font-weight: bold;
                font-size: 9pt;
                padding: 2px 8px;
                min-height: 24px;
                max-height: 24px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        export_btn.clicked.connect(self.exportData)
        self.header_layout.addWidget(export_btn)

        self.header_layout.addSpacing(6)

        # Inline status banner, shows which config (or auto-mode) is driving
        # the tab, so a missing config.json isn't a silently empty tab.
        self.config_status_label = QtWidgets.QLabel("Stats: no task uploaded yet")
        self.config_status_label.setStyleSheet(
            "QLabel {"
            f" color: {_THEME.palette.text_muted};"
            " font-size: 10pt;"
            " padding: 2px 8px;"
            "}"
        )
        self.header_layout.addWidget(self.config_status_label)

        # Stacked group blocks (one per config signature) in a scroll area.
        # All-same boxes -> ONE block -> the unified view (header hidden).
        self.groups_scroll = QtWidgets.QScrollArea()
        self.groups_scroll.setWidgetResizable(True)
        self.groups_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._groups_host = QtWidgets.QWidget()
        self.groups_layout = QtWidgets.QVBoxLayout(self._groups_host)
        self.groups_layout.setContentsMargins(0, 0, 0, 0)
        self.groups_layout.setSpacing(12)

        # Placeholder shown until a config arrives.
        self.placeholder = QtWidgets.QLabel(
            "Statistics appear here once a task with a stats config is uploaded.")
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet(
            "QLabel { color: %s; font-size: 12pt; font-style: italic; "
            "padding: 40px; }" % _THEME.palette.text_dim)
        self.groups_layout.addWidget(self.placeholder)
        self.groups_layout.addStretch(1)

        self.groups_scroll.setWidget(self._groups_host)
        layout.addWidget(self.groups_scroll, stretch=1)

        # Footer bar anchored below content
        self.header_layout.addStretch()
        layout.addLayout(self.header_layout)

        # Update timer, parented to self so Qt owns lifetime + thread affinity.
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.timeout.connect(self.updateDisplay)

    def loadConfigFromTask(self, setup_id, task_path):
        """Auto-load stats.json / config.json from a task folder, stored PER BOX.

        ``box_id`` is the box that uploaded ``task_path``; its config is
        cached in ``self.box_config[box_id]`` so that box is computed with
        its own task's formulas (not whatever box uploaded last).

        ``task_path`` comes from the per-box NestedMenu, a string of the
        form ``"ReversalLearning/ReversalLearning"`` or, on Windows,
        ``"ReversalLearning\\ReversalLearning"`` (no ``.py`` suffix).
        Resolved against the canonical ``app_paths.tasks_dir``.
        """
        if not task_path:
            logger.debug("No task path provided")
            return

        logger.info(f"Loading statistics config for task: {task_path}")

        # Canonical absolute tasks_dir from config.paths, same source of
        # truth as NestedMenu.update_menu() and the run_task pipeline (a
        # relative path would break when the CWD isn't the project root).
        from source import paths as app_paths
        tasks_dir = Path(app_paths.tasks_dir)

        # Add .py extension if not present
        task_path_with_ext = task_path if task_path.endswith(".py") \
            else f"{task_path}.py"

        # Construct full path; resolve handles mixed / and \ on Windows.
        task_path_obj = (tasks_dir / task_path_with_ext).resolve()
        logger.debug(f"Resolved task path: {task_path_obj}")

        task_folder = task_path_obj.parent
        logger.debug(f"Task folder: {task_folder}")

        # ``stats.json`` is what tools/init_stats.py writes; ``config.json``
        # is the hand-authored name. Probe both so the tool's output is
        # actually loaded.
        config_file = task_folder / "stats.json"
        if not config_file.exists():
            config_file = task_folder / "config.json"
        logger.debug(f"Looking for config at: {config_file}")

        # ===== TIER 1: Try to load custom stats.json / config.json =====
        if config_file.exists():
            logger.info(f"Found custom config file: {config_file}")
            self.config_path = str(config_file)
            self.config_mode = "custom"
            self._set_status_banner(
                f"Stats: loaded {task_path_obj.parent.name}/{config_file.name}",
                ok=True)
            try:
                with open(config_file, 'r') as f:
                    cfg = to_canvas_config(json.load(f))
            except Exception as e:
                logger.error("Failed to read stats config %s: %s", config_file, e)
                return
            if setup_id is not None:
                # Per-box: cache THIS box's config + rebuild the union layout.
                self._set_box_config(setup_id, cfg)
            else:
                # Legacy global path (no box_id), apply as the single config.
                self.config = cfg
                if self.box_count > 0:
                    self._rebuild_groups()
                self.update_timer.start(self.config.get('update_interval_ms', 1000))
                self._stats_error_boxes.clear()
            return

        # ===== TIER 2 & 3: Use template or auto-detection =====
        logger.info(f"No custom config found at {config_file}, "
                    f"enabling universal mode for task {task_path}")
        self._set_status_banner(
            f"Stats: no stats/config JSON in {config_file.parent}, auto-detect mode",
            ok=False)
        self.config_mode = "auto"  # Will upgrade to "template" if pattern detected
        self.config_path = None
        self.auto_config_generated = False

        # Initialize auto-detector for active boxes
        for setup_id in self.framework_running:
            if setup_id not in self.auto_detectors:
                self.auto_detectors[setup_id] = StatisticsAutoDetector()
                logger.debug(f"Created auto-detector for box {setup_id}")

        logger.info("Universal statistics mode enabled - will auto-detect patterns")

    def _set_status_banner(self, text, ok=True):
        """Update the inline status label in the footer."""
        if not hasattr(self, "config_status_label"):
            return
        p = _THEME.palette
        colour = p.text_muted if ok else "#f87171"  # red-400 on failure
        self.config_status_label.setText(text)
        self.config_status_label.setStyleSheet(
            "QLabel {"
            f" color: {colour};"
            " font-size: 10pt;"
            " padding: 2px 8px;"
            "}"
        )

    def loadConfig(self):
        """Load statistics configuration from JSON"""
        if not self.config_path or not Path(self.config_path).exists():
            logger.warning("No valid config path")
            return

        try:
            with open(self.config_path, 'r') as f:
                self.config = to_canvas_config(json.load(f))

            logger.info(f"Loaded statistics config from: {self.config_path}")

            # Setup table and plots
            if self.box_count > 0:
                self._rebuild_groups()

            # Start update timer
            update_interval = self.config.get('update_interval_ms', 1000)
            self.update_timer.start(update_interval)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self,
                "JSON Error",
                f"Failed to parse JSON file:\n{str(e)}"
            )
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load configuration:\n{str(e)}"
            )

    @staticmethod
    def _config_columns(cfg):
        """Return a config's column list, whether at root or under ``table``."""
        if not cfg:
            return []
        if 'columns' in cfg:
            return cfg['columns'] or []
        if 'table' in cfg and 'columns' in cfg['table']:
            return cfg['table']['columns'] or []
        return []

    def _set_box_config(self, setup_id, cfg):
        """Cache one box's stats config and rebuild the signature groups.
        The box is then COMPUTED with ``cfg`` and rendered in the group that
        shares its structure; another box's upload can never blank this box's
        columns."""
        self.box_config[setup_id] = cfg
        self.config_mode = "custom"
        self._rebuild_groups()
        # New structure, give every box a fresh chance.
        self._stats_error_boxes.clear()

    @staticmethod
    def _config_signature(cfg):
        """Stable signature of a config's STRUCTURE, column keys + plot
        names/types + calculation keys. Boxes with the same signature render
        in one group; identical-structure different tasks share efficiently."""
        cols = tuple(c.get('key') for c in StatsCanvas._config_columns(cfg))
        plots = tuple((p.get('name'), p.get('type') or p.get('chart_type') or '')
                      for p in (cfg.get('plots') or []))
        calcs = tuple(sorted((cfg.get('calculations') or {}).keys()))
        return repr((cols, plots, calcs))

    def _group_for_box(self, setup_id):
        return self.groups.get(self.box_signature.get(setup_id))

    def _union_config(self):
        """Union of every box's columns + plots (dedup by key / name). Kept on
        ``self.config`` as the 'a config exists' marker + the global fallback
        used by boxes without a per-box config."""
        union_cols, seen_cols = [], set()
        union_plots, seen_plots = [], set()
        interval = None
        for cfg in self.box_config.values():
            for col in self._config_columns(cfg):
                key = col.get('key')
                if key is not None and key not in seen_cols:
                    seen_cols.add(key)
                    union_cols.append(col)
            for plot in (cfg.get('plots') or []):
                name = plot.get('name')
                if name is not None and name not in seen_plots:
                    seen_plots.add(name)
                    union_plots.append(plot)
            iv = cfg.get('update_interval_ms')
            if iv:
                interval = iv if interval is None else min(interval, iv)
        out = {'columns': union_cols, 'plots': union_plots}
        if interval:
            out['update_interval_ms'] = interval
        return out

    def _rebuild_groups(self):
        """(Re)build signature groups from each box's config. Existing group
        widgets + their live data are PRESERVED; only new groups are created
        and emptied groups removed. Cheap, runs on upload / box-count change,
        never per tick. One signature -> one group; all-same -> one group."""
        if not hasattr(self, "groups_layout"):
            return
        if self.box_config:
            self.config = self._union_config()
        # Assign each box a signature: its own config, else the global one.
        self.box_signature = {}
        desired = {}   # signature -> [config, set(box_ids)]
        place = set(self.box_config.keys())
        if not self.box_config and self.config:
            place |= set(range(1, self.box_count + 1))
        for setup_id in place:
            cfg = self.box_config.get(setup_id) or self.config
            if not cfg:
                continue
            sig = self._config_signature(cfg)
            self.box_signature[setup_id] = sig
            entry = desired.setdefault(sig, [cfg, set()])
            entry[1].add(setup_id)
        # Reconcile group objects (preserve existing widgets + data).
        for sig, (cfg, ids) in desired.items():
            g = self.groups.get(sig)
            if g is None:
                g = _StatsGroup(sig, cfg)
                # Re-emit each row's Controls click up to whoever owns the canvas.
                g.table.controls_clicked.connect(self.box_controls_requested)
                self.groups[sig] = g
            g.config = cfg
            g.box_ids = set(ids)
            g.table.setupTable(self._config_columns(cfg), self.box_count)
            g.plots.setupPlots(cfg, self.box_count)
        for sig in list(self.groups.keys()):
            if sig not in desired:
                self._destroy_group(sig)
        self._relayout_groups()

    def _destroy_group(self, sig):
        g = self.groups.pop(sig, None)
        if g is None:
            return
        try:
            self.groups_layout.removeWidget(g.container)
        except Exception:
            pass
        g.container.setParent(None)
        g.container.deleteLater()

    def _relayout_groups(self):
        """Stack the group blocks in box-id order; hide the header when there
        is a single group (the unified view)."""
        ordered = sorted(
            self.groups.values(),
            key=lambda g: (min(g.box_ids) if g.box_ids else 1 << 30))
        # Clear the layout WITHOUT deleting the widgets we keep.
        while self.groups_layout.count():
            item = self.groups_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        if not ordered:
            # Re-parent + add to the layout BEFORE making it visible, the
            # drain loop above set its parent to None, and showing it while
            # parentless would flash it as a top-level window for one frame.
            self.placeholder.setParent(self._groups_host)
            self.groups_layout.addWidget(self.placeholder)
            self.groups_layout.addStretch(1)
            self.placeholder.setVisible(True)
            return
        self.placeholder.setVisible(False)
        single = len(ordered) == 1
        for g in ordered:
            if single:
                g.header.setVisible(False)
            else:
                boxes = ", ".join(str(b) for b in sorted(g.box_ids))
                g.header.setText(f"Boxes {boxes}")
                g.header.setVisible(True)
            self._apply_group_rows(g)
            g.container.setParent(self._groups_host)
            # Single (unified) group fills the whole viewport so the user can
            # drag the table/plot splitter and give the table the full height.
            # With multiple groups each block keeps its preferred height and a
            # trailing stretch lets the outer scroll area scroll between them.
            self.groups_layout.addWidget(g.container, 1 if single else 0)
        if not single:
            self.groups_layout.addStretch(1)
        if not self.update_timer.isActive():
            interval = (self.config or {}).get('update_interval_ms', 1000)
            self.update_timer.start(interval)

    def _apply_group_rows(self, g):
        """Show only this group's boxes' rows in its table."""
        for r in range(g.table.rowCount()):
            g.table.setRowHidden(r, (r + 1) not in g.box_ids)

    def set_box_error(self, setup_id, on, msg=""):
        """Tint this box's stats row red on error / reset on clear. Routed
        centrally by box_alerts.apply_box_alerts; safe before the group exists."""
        g = self._group_for_box(int(setup_id))
        if g is not None and hasattr(g.table, "setRowError"):
            try:
                g.table.setRowError(int(setup_id), bool(on))
            except Exception as e:
                logger.debug("stats set_box_error box %s: %s", setup_id, e)

    def update_box_timer(self, setup_id, txt):
        """Write the per-box elapsed clock into the Timer column. Fed by the
        central process tick (one source shared with the box card / Live
        Status); safe before the box's group exists."""
        g = self._group_for_box(int(setup_id))
        if g is not None:
            try:
                g.table.updateCell(int(setup_id), "elapsed", txt)
            except Exception as e:
                logger.debug("stats update_box_timer box %s: %s", setup_id, e)

    def updateBoxCount(self, box_count):
        """Update when box count changes, re-row every group's table."""
        self.box_count = box_count
        logger.debug(f"Statistics tab updated for {box_count} boxes")
        self._rebuild_groups()

    def updateSubjectIds(self, subject_id_dict):
        """Update subject IDs for the circular-progress labels, routed to
        each box's group. ``{box_id: subject_id}``."""
        by_group = {}
        for setup_id, subj in subject_id_dict.items():
            g = self._group_for_box(int(setup_id))
            if g is not None:
                by_group.setdefault(g.signature, (g, {}))[1][setup_id] = subj
        for g, d in by_group.values():
            try:
                g.plots.updateSubjectIds(d)
            except Exception as e:
                logger.debug("stats updateSubjectIds: %s", e)
        logger.info(f"Subject IDs updated for {len(subject_id_dict)} boxes")

    def updateDisplay(self):
        """Update each group's table + plots, each box computed with ITS OWN
        config and rendered in its signature group."""
        if not self.groups or self.box_count == 0:
            return

        # Disable table updates during the batch on every group; ALWAYS
        # re-enable + refresh in the finally so the GUI never gets stuck.
        for g in self.groups.values():
            g.table.setUpdatesEnabled(False)
        try:
            # Calculate and update ONLY for boxes with running frameworks
            for setup_id in range(1, self.box_count + 1):
                # Only process if framework is running for this box
                if not self.framework_running.get(setup_id, False):
                    continue
                # skip a box that already errored this run; pausing just
                # THIS box keeps the others live.
                if setup_id in self._stats_error_boxes:
                    continue

                g = self._group_for_box(setup_id)
                if g is None:
                    continue
                # Per-box config: a box is computed with ITS OWN task config,
                # falling back to the group's config.
                cfg = self.box_config.get(setup_id) or g.config
                formulas = cfg.get('calculations') or {}
                if not formulas:
                    continue
                try:
                    self._update_box_display(setup_id, g, cfg, formulas)
                except Exception as e:
                    # Isolate the failure to THIS box; never stop the shared
                    # timer. Warn once per box so a bad formula doesn't spam
                    # the log every tick.
                    self._stats_error_boxes.add(setup_id)
                    logger.error(
                        "Box %s statistics error, updates paused for this "
                        "box only (other boxes keep running): %s", setup_id, e)
        finally:
            # Re-enable updates and refresh plot displays for every group.
            for g in self.groups.values():
                g.table.setUpdatesEnabled(True)
                g.plots.refreshPlots()

    def _update_box_display(self, setup_id, g, cfg, formulas):
        """Compute + render one box's stats row + plots in its group ``g``,
        using THIS box's own config. Raises on a bad formula/calc, the caller
        isolates the error to this box. ``updateRow`` fills only the keys
        this box's config defines."""
        logger.debug(f"Calculating statistics for box {setup_id}")
        results = self.calculator.calculateStatistics(setup_id, formulas)
        logger.debug(f"Calculated {len(results)} statistics for box {setup_id}")

        # Add metadata fields (subject_id, task, date, time, box_id)
        results['box_id'] = setup_id
        if setup_id in self.box_metadata:
            metadata = self.box_metadata[setup_id]
            results['subject_id'] = metadata.get('subject_id', '')
            results['task'] = metadata.get('task', '')
            results['box_number'] = setup_id
            # Date and time
            now = datetime.now()
            results['date'] = now.strftime("%Y-%m-%d")
            results['time'] = now.strftime("%H:%M:%S")
        else:
            results['subject_id'] = ''
            results['task'] = ''

        # Session duration in minutes (first→last MCU timestamp).
        results['duration_min'] = self.calculator.getDuration(setup_id)

        # Fan each sequence column's captured tokens into its per-index cells
        # (trials#1..trials#N). Cells past the captured length stay blank.
        for col in (self._config_columns(cfg) or []):
            if not col.get("sequence"):
                continue
            key = col.get("key")
            n = int(col.get("count", 0) or 0)
            seq = self.calculator.getSequence(setup_id, key)
            for i in range(1, n + 1):
                results[f"{key}#{i}"] = seq[i - 1] if i <= len(seq) else ""

        # Update this group's table row (only this box's own columns).
        if self._config_columns(cfg):
            g.table.updateRow(setup_id, results)

        # Update this group's plots, only THIS box's own plots.
        for plot_config in (cfg.get('plots') or []):
            plot_name = plot_config['name']
            metric_key = plot_config.get('metric', plot_config.get('data_key', plot_name))

            if metric_key in results:
                value = results[metric_key]
                sd = self.calculator.getStatSD(setup_id, metric_key)
                trial_count = None
                if plot_config.get('x-axis') == 'trials':
                    trial_count = self._trial_count_for_plot(
                        plot_config, results, cfg)
                g.plots.updatePlot(plot_name, setup_id, value, sd, trial_count=trial_count)

    #: Metrics tried, in order, when a trials plot names no ``trial_metric``.
    _TRIAL_METRIC_FALLBACKS = ("num_trials", "trials", "trial", "trial_count",
                               "n_trials")

    def _trial_count_for_plot(self, plot_config, results, cfg=None):
        """The x value for a ``"x-axis": "trials"`` line plot.

        A line plot only gains a point when this rises, so returning ``None``
        means the plot stays empty for the whole session. That is what happened
        to reversal learning: the config says ``"x-axis": "trials"`` and names
        no ``trial_metric``, only one config in the tree ever did, and nothing
        was logged, so the plot looked like it had simply stopped working.

        Order: the named metric, then the usual names for a trial counter, then
        ``None`` with a warning naming the plot and the config. The caller
        treats ``None`` as "append on any change of value", so a config that
        names nothing still draws a line rather than nothing at all.
        """
        name = plot_config.get("trial_metric", "")
        if name:
            if name in results:
                try:
                    return int(results[name])
                except (TypeError, ValueError):
                    return None
            self._warn_trial_metric(plot_config, cfg, missing=name)
            return None
        for candidate in self._TRIAL_METRIC_FALLBACKS:
            if candidate in results:
                try:
                    return int(results[candidate])
                except (TypeError, ValueError):
                    continue
        self._warn_trial_metric(plot_config, cfg, missing="")
        return None

    def _warn_trial_metric(self, plot_config, cfg, *, missing: str) -> None:
        """Say it once per plot, and say which file to edit."""
        warned = getattr(self, "_trial_metric_warned", None)
        if warned is None:
            warned = self._trial_metric_warned = set()
        key = str(plot_config.get("name", "?"))
        if key in warned:
            return
        warned.add(key)
        where = (cfg or {}).get("_config_path") or "the task's config.json"
        if missing:
            logger.warning(
                "Plot '%s' names trial_metric '%s', which is not a metric this "
                "task produces. Points are appended whenever the value changes "
                "instead. Fix the name in %s.", key, missing, where)
        else:
            logger.warning(
                "Plot '%s' uses x-axis 'trials' but names no trial_metric, and "
                "none of %s is a metric here. Points are appended whenever the "
                "value changes instead. Add \"trial_metric\" to %s to plot "
                "against the real trial count.",
                key, ", ".join(self._TRIAL_METRIC_FALLBACKS), where)

    def onFrameworkStart(self, setup_id, setup_widget=None):
        """Called when framework starts for a box - resets statistics and registers as data consumer"""
        logger.info(f"Framework started for Box {setup_id} - resetting statistics")

        # Mark framework as running
        self.framework_running[setup_id] = True
        logger.debug(f"Marked box {setup_id} as running")
        # (per-box error blacklist is cleared below, alongside the box reset)

        # Initialize or reset auto-detector for this box if in auto mode
        if self.config_mode == "auto" or self.config_mode is None:
            if setup_id in self.auto_detectors:
                self.auto_detectors[setup_id].reset()
                logger.debug(f"Reset auto-detector for box {setup_id}")
            else:
                self.auto_detectors[setup_id] = StatisticsAutoDetector()
                logger.info(f"Created auto-detector for box {setup_id}")

        # Register as data consumer for this box
        if setup_widget and hasattr(setup_widget, 'pycboard') and setup_widget.pycboard:
            logger.debug(f"Box widget and pycboard found for box {setup_id}")

            # Initialize data_consumers list if it doesn't exist or is None
            if not hasattr(setup_widget.pycboard, 'data_consumers') or setup_widget.pycboard.data_consumers is None:
                setup_widget.pycboard.data_consumers = []
                logger.debug(f"Initialized data_consumers list for box {setup_id}")

            # Remove old statistics consumer for this box if it exists (prevent duplicates)
            old_consumer = self.box_data_consumers.get(setup_id)
            if old_consumer and old_consumer in setup_widget.pycboard.data_consumers:
                setup_widget.pycboard.data_consumers.remove(old_consumer)
                logger.debug(f"Removed old consumer for box {setup_id}")

            # Create a new wrapper data consumer for this box
            consumer = StatisticsDataConsumer(self, setup_id)
            self.box_data_consumers[setup_id] = consumer
            logger.debug(f"Created StatisticsDataConsumer wrapper for box {setup_id}")

            # Register the new wrapper as data consumer
            setup_widget.pycboard.data_consumers.append(consumer)
            logger.info(f"Registered statistics tab as data consumer for box {setup_id}")

            # Store metadata and ID map
            try:
                subject_id = setup_widget.subject_id_edit.text().strip()
                task = setup_widget.task_combo.text()
            except (RuntimeError, AttributeError):
                subject_id = ""
                task = ""
            self.box_metadata[setup_id] = {
                'subject_id': subject_id,
                'task': task,
                'box_number': setup_id,
                'start_time': time.time()
            }
            logger.debug(f"Stored metadata for box {setup_id}: {self.box_metadata[setup_id]}")
            # Push subject ID into this box's group plot layer so labels render.
            _g = self._group_for_box(setup_id)
            if _g is not None:
                subj = self.box_metadata[setup_id].get('subject_id', '')
                _g.plots.subject_ids[setup_id] = subj
            try:
                if hasattr(setup_widget.pycboard, "sm_info") and hasattr(setup_widget.pycboard.sm_info, "ID2name"):
                    self.id2name_map[setup_id] = dict(setup_widget.pycboard.sm_info.ID2name)
            except Exception:
                pass
        else:
            logger.warning(f"No box_widget or pycboard available for box {setup_id}")

        # Make sure this box is grouped (its config was set at upload).
        self._rebuild_groups()

        # Clear statistics for this box to start fresh
        self.calculator.clearBox(setup_id)
        # a fresh run re-enables stats updates for this box even if its
        # previous run errored out.
        self._stats_error_boxes.discard(setup_id)

        # Clear this box's row + plot series in its group.
        g = self._group_for_box(setup_id)
        if g is not None:
            row = setup_id - 1
            if 0 <= row < g.table.rowCount():
                for col in range(g.table.columnCount()):
                    item = g.table.item(row, col)
                    if item:
                        item.setText("")
            g.plots.clearBox(setup_id)
            g.plots.refreshPlots(force=True)

        # Ensure update timer is running for this box
        self._ensureTimerState()

    def onFrameworkStop(self, setup_id):
        """Called when framework stops for a box"""
        logger.info(f"Framework stopped for Box {setup_id}")
        self.framework_running[setup_id] = False

        # Update timer state based on remaining active frameworks
        self._ensureTimerState()

    def _ensureTimerState(self):
        """Ensure update timer state matches current framework status.

        Timer should run if:
        - At least one framework is running AND
        - We have a valid config
        """
        if not hasattr(self, 'update_timer'):
            return

        has_active_framework = any(self.framework_running.values())
        has_config = self.config is not None
        timer_active = self.update_timer.isActive()

        if has_active_framework and has_config:
            # Should be running
            if not timer_active:
                update_interval = self.config.get('update_interval_ms', 1000)
                self.update_timer.start(update_interval)
                logger.debug(f"Update timer started ({update_interval}ms) - {sum(self.framework_running.values())} active box(es)")
        else:
            # Should be stopped
            if timer_active:
                self.update_timer.stop()
                if not has_active_framework:
                    logger.debug("Update timer stopped - no active frameworks")
                elif not has_config:
                    logger.debug("Update timer stopped - no config loaded")

    def process_data_for_box(self, setup_id, new_data):
        """Process incoming data from pycboard for a specific box

        This method is called by StatisticsDataConsumer wrapper when new data arrives.
        """
        logger.debug(f"Received {len(new_data)} data items for box {setup_id}")

        # ===== UNIVERSAL SYSTEM: Feed data to auto-detector =====
        if self.config_mode == "auto" and setup_id in self.auto_detectors:
            # Feed print statements to auto-detector
            for data_point in new_data:
                if data_point.type == MsgType.PRINT:
                    print_str = str(data_point.content)
                    self.auto_detectors[setup_id].analyze_print(print_str)

            # Check if we have enough data to generate config
            if not self.auto_config_generated:
                detector = self.auto_detectors[setup_id]
                if detector.has_sufficient_data(min_samples=5):
                    # Try to match template first (Tier 2)
                    task_type = detector.detect_task_type()

                    if task_type != "generic" and task_type != "unknown":
                        # Use template
                        logger.info(f"Detected task type: {task_type}, applying template")
                        self.config = apply_template(task_type, detector.print_counts)
                        self.config_mode = "template"
                    else:
                        # Use auto-generated config (Tier 3)
                        logger.info("No matching template, generating auto-config")
                        self.config = detector.generate_default_config()
                        self.config_mode = "auto"

                    self.auto_config_generated = True

                    # Apply the generated config
                    if self.box_count > 0:
                        self._rebuild_groups()

                    # Update timer state - will start since framework is running
                    self._ensureTimerState()

                    logger.info(f"Auto-config applied in {self.config_mode} mode")

        # Process the data using THIS box's OWN counters. ``self.config`` is
        # the cross-box UNION (columns + plots only, ``_union_config`` omits
        # counters), so counters must come from the per-box config; fall back
        # to the union (then {}) for a box with no per-box config.
        id_map = self.id2name_map.get(setup_id)
        box_cfg = self.box_config.get(setup_id) or self.config or {}
        counters = box_cfg.get("counters", {})
        self.calculator.processData(setup_id, new_data, id_map, counters)

        # Debug: log event counts
        if setup_id in self.calculator.event_counts:
            counts = self.calculator.event_counts[setup_id]
            logger.debug(f"Box {setup_id} event counts: {len(counts)} unique events")

    def exportData(self):
        """Export every group's statistics table to Excel (one sheet/group)."""
        try:
            if not self.groups:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Data",
                    "No statistics data to export. Please load a config and collect data first."
                )
                return

            import pandas as pd
            from datetime import datetime

            from source.datetime_formats import FILE_STEM_TS_FMT

            # Get default filename with timestamp
            timestamp = datetime.now().strftime(FILE_STEM_TS_FMT)
            default_filename = f"statistics_export_{timestamp}.xlsx"

            # Ask user for save location
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export Statistics to Excel",
                default_filename,
                "Excel Files (*.xlsx);;All Files (*)"
            )

            if not filename:
                return

            ordered = sorted(
                self.groups.values(),
                key=lambda g: (min(g.box_ids) if g.box_ids else 1 << 30))
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                for i, g in enumerate(ordered, 1):
                    tw = g.table
                    cols = tw.columnCount()
                    headers = []
                    for col in range(cols):
                        h = tw.horizontalHeaderItem(col)
                        headers.append(h.text() if h else f"Column {col}")
                    data = []
                    for row in range(tw.rowCount()):
                        if tw.isRowHidden(row):
                            continue   # only this group's boxes
                        row_data = []
                        for col in range(cols):
                            item = tw.item(row, col)
                            row_data.append(item.text() if item else "")
                        data.append(row_data)
                    sheet = f"Group{i}" if len(ordered) > 1 else "Statistics"
                    pd.DataFrame(data, columns=headers).to_excel(
                        writer, sheet_name=sheet, index=False)

            logger.info(f"Statistics exported to: {filename}")
            QtWidgets.QMessageBox.information(
                self,
                "Export Success",
                f"Statistics exported successfully to:\n{filename}"
            )

        except ImportError:
            QtWidgets.QMessageBox.critical(
                self,
                "Missing Dependencies",
                "Required libraries not found. Please install:\n\npip install pandas openpyxl"
            )
            logger.error("pandas or openpyxl not installed")
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export data:\n{str(e)}"
            )


    def getConfigPath(self):
        """Get current config path"""
        return self.config_path

    def setConfigPath(self, path):
        """Set config path and load"""
        if path and Path(path).exists():
            self.config_path = path
            self.loadConfig()
