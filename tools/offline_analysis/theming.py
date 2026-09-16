"""Theming bridge, the rig's design tokens and style builders, shared.

Imports ``source.gui.theme`` and ``source.gui.style_builders`` directly. It
used to ship private copies (``_design_tokens.py`` / ``_style_builders.py``)
so the folder could be lifted out of the repo and run standalone; the copies
drifted instead, the analyzer kept the old vivid ``success`` green after the
rig moved to a dim sage, so "saved OK" was a different colour in the two apps.
One palette now, by construction rather than by discipline.

The two SVG icons under ``icons/`` stay local: they are assets, not code.

One entry point per scope:

    setup_dark_theme(app), call once on QApplication after construction
    apply_window_chrome(w), call once on every top-level window after .show()
    style_button(btn, kind), convenience wrapper around button_style()

The global QApplication stylesheet is the heavy hitter, it styles every
widget kind the analyzer uses (QGroupBox, QPushButton, QTabWidget,
QComboBox, QLineEdit, QSpinBox, QTableWidget, QListWidget, QTextEdit,
QProgressBar, QScrollArea, QStatusBar) in one shot, so individual views
don't need to set per-widget stylesheets to get the rig-GUI dark look.
Per-widget builders (e.g. ``button_style("primary")``) still work and
override the global default where a specific look is wanted.
"""

from __future__ import annotations

from typing import Optional

from PySide6 import QtWidgets

from tools.offline_analysis.vendor.theme import THEME as TOKENS
from tools.offline_analysis.vendor import style_builders as _sb


# ---------------------------------------------------------------------------
# Global stylesheet, every widget kind the analyzer uses, in one string.
# ---------------------------------------------------------------------------

def _global_qss() -> str:
    """Composite app-level stylesheet built from the rig-GUI style builders.

    Applied to the QApplication so every widget the analyzer creates gets
    the rig-GUI dark look without per-widget setStyleSheet calls.
    """
    p = TOKENS.palette
    f = TOKENS.font
    base = (
        f"QWidget {{ background: {p.bg}; color: {p.text};"
        f" font: {f.body_pt}pt '{f.family}'; }}"
        f"QToolTip {{ background: {p.surface_elev}; color: {p.text};"
        f" border: 1px solid {p.surface_border_strong}; padding: 4px 6px;"
        f" border-radius: {TOKENS.radius.sm}px; }}"
        f"QSplitter::handle {{ background: {p.surface_border}; }}"
        f"QSplitter::handle:horizontal {{ width: 3px; }}"
        f"QSplitter::handle:vertical {{ height: 3px; }}"
        f"QMenuBar {{ background: {p.surface}; color: {p.text};"
        f" border-bottom: 1px solid {p.surface_border}; }}"
        f"QMenuBar::item:selected {{ background: {p.surface_elev_2}; }}"
        f"QMenu {{ background: {p.surface}; color: {p.text};"
        f" border: 1px solid {p.surface_border_strong}; }}"
        f"QMenu::item:selected {{ background: rgba(37,99,235,0.30); }}"
        f"QListWidget {{ background: {p.surface}; color: {p.text};"
        f" border: 1px solid {p.surface_border};"
        f" border-radius: {TOKENS.radius.sm}px; padding: 2px; }}"
        f"QListWidget::item {{ padding: 4px 6px;"
        f" border-radius: {TOKENS.radius.sm}px; }}"
        f"QListWidget::item:selected {{ background: rgba(37,99,235,0.30);"
        f" color: {p.text}; }}"
        f"QTextEdit, QPlainTextEdit {{ background: {p.surface};"
        f" color: {p.text}; border: 1px solid {p.surface_border};"
        f" border-radius: {TOKENS.radius.sm}px;"
        f" font: {f.body_pt}pt '{f.mono_family}'; padding: 4px; }}"
        f"QFormLayout {{ }}"
    )
    return "\n".join([
        _sb.main_window_style(),
        _sb.button_style("info"),
        _sb.input_style(),
        _sb.spinbox_style(),
        _sb.combobox_style(),
        _sb.checkbox_style(),
        _sb.groupbox_style("slate"),
        _sb.label_style("body"),
        _sb.tab_widget_style(),
        _sb.scroll_area_style(),
        _sb.status_bar_style(),
        _sb.table_style(),
        _sb.progress_bar_style(),
        base,
    ])


def setup_dark_theme(app: QtWidgets.QApplication) -> None:
    """Apply the rig-GUI palette + composite stylesheet to a QApplication."""
    _sb.apply_global_palette(app)
    app.setStyleSheet(_global_qss())


def apply_window_chrome(window: QtWidgets.QWidget) -> None:
    """Dark titlebar on Win32; no-op elsewhere.

    Call once after ``window.show()`` so the HWND exists.
    """
    _sb.force_dark_titlebar(window)


# ---------------------------------------------------------------------------
# Per-widget convenience wrappers (use sparingly, global QSS covers the
# default look; these are for buttons/labels that need a specific accent).
# ---------------------------------------------------------------------------

def style_button(btn: QtWidgets.QPushButton, variant: str = "info",
                 *, height: Optional[int] = None) -> None:
    """Apply one of the gradient button variants from style_builders."""
    btn.setStyleSheet(_sb.button_style(variant, height=height or 30))


def style_card(widget: QtWidgets.QWidget, *,
               elevated: bool = False, with_shadow: bool = False) -> None:
    widget.setStyleSheet(_sb.card_style(elevated=elevated, with_shadow=with_shadow))


def style_groupbox(box: QtWidgets.QGroupBox, accent: str = "slate") -> None:
    box.setStyleSheet(_sb.groupbox_style(accent))


def style_label(label: QtWidgets.QLabel, variant: str = "body") -> None:
    label.setStyleSheet(_sb.label_style(variant))


def style_chip(label: QtWidgets.QLabel, variant: str = "neutral") -> None:
    label.setStyleSheet(_sb.chip_style(variant))


__all__ = [
    "setup_dark_theme", "apply_window_chrome",
    "style_button", "style_card", "style_groupbox",
    "style_label", "style_chip",
    "TOKENS",
]
