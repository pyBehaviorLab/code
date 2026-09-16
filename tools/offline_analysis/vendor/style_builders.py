"""Stylesheet builders, one function per widget kind.

Functions read :data:`source.gui.theme.THEME` and return a Qt
stylesheet string.

Variants (where applicable):
    button_style(variant)   primary | info | success | warning | danger | secondary | ghost
    label_style(variant)    display | section | body | caption | muted
    timer_label_style(state) idle | running | error
    groupbox_style(accent)  primary | info | success | warning | danger | slate

Example:
    self.start_btn.setStyleSheet(button_style("success", height=32))
    self.note_input.setStyleSheet(input_style())
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from tools.offline_analysis.vendor.theme import THEME, Gradient


# Forward-slashed absolute paths for QSS ``image: url(...)``.

# White ✓ painted inside every checked QCheckBox indicator.
_TICK_WHITE_URL = (Path(__file__).parent / "icons" / "tick_white.svg") \
    .resolve().as_posix()

# Down-chevron painted in QComboBox and NestedMenu so both share the
# same visual vocabulary (SVG renders reliably on Windows Qt6).
_CHEVRON_URL = (Path(__file__).parent / "icons" / "chevron_down.svg") \
    .resolve().as_posix()


# ============================================================================
# Internals, gradient encoding
# ============================================================================

def _qlinear(g: Gradient) -> str:
    """Encode a 135° gradient as ``qlineargradient(...)`` for Qt QSS."""
    return (
        f"qlineargradient(x1:0, y1:0, x2:1, y2:1, "
        f"stop:0 {g.start}, stop:1 {g.end})"
    )


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """``#rrggbb`` → ``rgba(r,g,b,alpha)`` for glassy translucent fills."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# Per-variant accent hex (used to derive glass bg + border + hover).
_VARIANT_HEX = {
    "primary": THEME.gradient.primary.end,   # purple #a855f7
    "info":    THEME.gradient.info.end,      # blue   #2563eb
    "success": THEME.gradient.success.end,   # mint   #10b981
    "warning": THEME.gradient.warning.end,   # deep burnt amber/orange
    "danger":  THEME.gradient.danger.end,    # rose   #e11d48
}


# ============================================================================
# Buttons
# ============================================================================

_BUTTON_DISABLED_QSS = """
    QPushButton:disabled {{
        background: rgba(148,163,184,0.08);
        color: {muted};
        border: 1px solid rgba(148,163,184,0.15);
    }}
""".format(muted=THEME.palette.text_dim)


def _grad_for_variant(name: str) -> str:
    """135° qlineargradient for a button variant."""
    g = THEME.gradient
    return {
        "primary": _qlinear(g.primary),
        "info":    _qlinear(g.info),
        "success": _qlinear(g.success),
        "warning": _qlinear(g.warning),
        "danger":  _qlinear(g.danger),
    }[name]


def _grad_hover_for_variant(name: str) -> str:
    """Brighter hover variant, biases the gradient toward the lighter
    start stop to approximate a brightness boost Qt QSS can't express."""
    g = THEME.gradient
    pair = {
        "primary": g.primary, "info": g.info, "success": g.success,
        "warning": g.warning, "danger": g.danger,
    }[name]
    return (f"qlineargradient(x1:0, y1:0, x2:1, y2:1, "
            f"stop:0 {pair.start}, stop:0.7 {pair.start}, stop:1 {pair.end})")


def button_style(variant: str = "primary", *,
                 height: int = 30,
                 radius: Optional[int] = None,
                 padding_h: int = 14) -> str:
    """Build the QSS for a :class:`QPushButton`.

    Variants (vivid 135° gradient fills with a glassy white top-edge
    highlight):
        primary, violet → purple gradient.
        info, sky → royal blue.
        success, mint → emerald.
        warning, amber → orange.
        danger, rose → crimson.
        secondary, flat slate, muted text.
        ghost, transparent until hover.

    Hover brightens by biasing toward the lighter start stop; pressed
    nudges 1 px down for tactile feedback.
    """
    p = THEME.palette
    r = radius if radius is not None else THEME.radius.md
    font_qss = f"font: 600 {THEME.font.body_pt}pt '{THEME.font.family}';"

    if variant == "secondary":
        return (
            "QPushButton {"
            f" background: {p.surface_elev};"
            f" color: {p.text};"
            f" border: 1px solid {p.surface_border_strong};"
            f" border-radius: {r}px;"
            f" padding: 0 {padding_h}px;"
            f" min-height: {height}px;"
            f" {font_qss}"
            "}"
            "QPushButton:hover {"
            f" background: {p.surface_elev_2};"
            f" border-color: rgba(255,255,255,0.25);"
            "}"
            f"QPushButton:pressed {{ background: {p.surface_elev}; }}"
            + _BUTTON_DISABLED_QSS
        )

    if variant == "ghost":
        return (
            "QPushButton {"
            " background: transparent;"
            f" color: {p.text_muted};"
            " border: 1px solid transparent;"
            f" border-radius: {r}px;"
            f" padding: 0 {padding_h}px;"
            f" min-height: {height}px;"
            f" {font_qss}"
            "}"
            "QPushButton:hover {"
            " background: rgba(255,255,255,0.06);"
            f" color: {p.text};"
            "}"
            "QPushButton:pressed { background: rgba(255,255,255,0.10); }"
            + _BUTTON_DISABLED_QSS
        )

    grad       = _grad_for_variant(variant)
    grad_hover = _grad_hover_for_variant(variant)

    return (
        "QPushButton {"
        f" background: {grad};"
        f" color: #ffffff;"
        f" border: 1px solid rgba(255,255,255,0.18);"
        f" border-top: 1px solid rgba(255,255,255,0.32);"
        f" border-radius: {r}px;"
        f" padding: 0 {padding_h}px;"
        f" min-height: {height}px;"
        f" {font_qss}"
        "}"
        "QPushButton:hover {"
        f" background: {grad_hover};"
        f" border: 1px solid rgba(255,255,255,0.28);"
        f" border-top: 1px solid rgba(255,255,255,0.45);"
        "}"
        "QPushButton:pressed {"
        f" background: {grad};"
        f" padding-top: 1px;"
        "}"
        + _BUTTON_DISABLED_QSS
    )


def light_button_style(variant: str = "info", *,
                       height: int = 22,
                       padding_h: int = 0,
                       radius: Optional[int] = None) -> str:
    """Glassy, low-saturation variant of :func:`button_style`.

    Same colour family as ``button_style(variant)`` but rendered as a
    translucent tint + thin coloured border, not a vivid gradient.
    Used for the compact zone-adjust toolbar's 22 px icon buttons.
    ``variant="secondary"`` and ``variant="ghost"`` delegate to
    ``button_style`` unchanged.
    """
    if variant in ("secondary", "ghost"):
        return button_style(variant, height=height, padding_h=padding_h,
                            radius=radius)

    accent = _VARIANT_HEX.get(variant, _VARIANT_HEX["info"])
    bg       = _hex_to_rgba(accent, 0.16)
    bg_hover = _hex_to_rgba(accent, 0.28)
    border   = _hex_to_rgba(accent, 0.55)
    r = radius if radius is not None else THEME.radius.md
    font_qss = f"font: 600 {THEME.font.body_pt}pt '{THEME.font.family}';"
    p = THEME.palette
    return (
        "QPushButton {"
        f" background: {bg};"
        f" color: {p.text};"
        f" border: 1px solid {border};"
        f" border-radius: {r}px;"
        f" padding: 0 {padding_h}px;"
        f" min-height: {height}px;"
        f" {font_qss}"
        "}"
        "QPushButton:hover {"
        f" background: {bg_hover};"
        f" border-color: {accent};"
        "}"
        "QPushButton:pressed {"
        f" background: {bg};"
        f" padding-top: 1px;"
        "}"
        + _BUTTON_DISABLED_QSS
    )


# ============================================================================
# Inputs / selects / checkboxes
# ============================================================================

def input_style() -> str:
    """``QLineEdit`` style, slate background, focus ring in palette.focus.

    Text is pure white on an opaque dark surface for contrast on lab PCs.
    Padding aligns the input baseline with adjacent buttons at row height.
    """
    p = THEME.palette
    return f"""
        QLineEdit {{
            background: rgba(15,23,42,0.95);
            color: #ffffff;
            border: 1px solid {p.surface_border_strong};
            border-radius: {THEME.radius.md}px;
            padding: 2px 10px;
            min-height: 18px;
            font: 600 {THEME.font.body_pt}pt '{THEME.font.family}';
        }}
        QLineEdit:hover  {{ border-color: {p.text_muted}; }}
        QLineEdit:focus  {{
            border: {THEME.border.focus}px solid {p.focus};
            background: rgba(15,23,42,1.0);
            color: #ffffff;
        }}
        QLineEdit:disabled {{
            color: {p.text_muted};
            background: rgba(15,23,42,0.55);
            border-color: {p.surface_border};
        }}
    """


def spinbox_style() -> str:
    """``QSpinBox`` / ``QDoubleSpinBox`` style, matches input_style + glass arrows.

    Same slate-glass field and focus-ring as line edits so spin boxes
    line up next to QLineEdit fields. Up/down arrows use a translucent
    slot on the right edge instead of Windows-default chrome on dark.
    """
    p = THEME.palette
    return f"""
        QSpinBox, QDoubleSpinBox {{
            background: rgba(255,255,255,0.04);
            color: {p.text};
            border: 1px solid {p.surface_border_strong};
            border-radius: {THEME.radius.md}px;
            padding: 2px 22px 2px 8px;
            min-height: 18px;
            font: {THEME.font.body_pt}pt '{THEME.font.family}';
        }}
        QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {p.text_muted}; }}
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border: {THEME.border.focus}px solid {p.focus};
            background: rgba(255,255,255,0.06);
        }}
        QSpinBox:disabled, QDoubleSpinBox:disabled {{
            color: {p.text_muted};
            background: rgba(15,23,42,0.55);
            border-color: {p.surface_border};
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            background: rgba(255,255,255,0.05);
            border: none;
            width: 18px;
            margin: 1px;
            border-radius: {THEME.radius.sm}px;
        }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background: rgba(255,255,255,0.10);
        }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            image: none;
            width: 0; height: 0;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-bottom: 5px solid {p.text};
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            image: none;
            width: 0; height: 0;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid {p.text};
        }}
    """


def combobox_style() -> str:
    """``QComboBox`` style, pixel-matched to the per-box task selector
    (``source/gui/utility.py::NestedMenu``); same closed-state recipe.

    - Field: ``palette.surface_elev``.
    - Top edge: 1 px ``rgba(255,255,255,0.22)`` glass cap.
    - Padding: ``2px 30px 2px 12px``.
    - Chevron: SVG so it renders as a real V on every platform.
    - Popup: fully dark surface, vivid info-blue selection.
    """
    p = THEME.palette
    return f"""
        QComboBox {{
            background-color: {p.surface_elev};
            color: #ffffff;
            border: 1px solid {p.surface_border_strong};
            border-top: 1px solid rgba(255,255,255,0.22);
            border-radius: {THEME.radius.md}px;
            padding: 2px 30px 2px 12px;
            min-height: 18px;
            font: 600 {THEME.font.body_pt}pt '{THEME.font.family}';
        }}
        QComboBox:hover    {{ border-color: {p.text_muted}; }}
        QComboBox:focus    {{
            border: 1px solid {p.focus};
            border-top: 1px solid {p.focus};
        }}
        QComboBox:disabled {{
            color: {p.text_muted};
            background: rgba(15,23,42,0.55);
            border-color: {p.surface_border};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
            background: transparent;
            subcontrol-position: right center;
            subcontrol-origin: padding;
        }}
        /* Chevron from SVG, same icon NestedMenu uses. */
        QComboBox::down-arrow {{
            image: url({_CHEVRON_URL});
            width: 12px; height: 8px;
            margin-right: 10px;
        }}

        /* Dark popup panel, style both QAbstractItemView and the inner
           QListView frame to override the OS-light fallback on Windows. */
        QComboBox QAbstractItemView {{
            background-color: {p.surface};
            color: #ffffff;
            border: 1px solid {p.surface_border_strong};
            border-radius: {THEME.radius.md}px;
            selection-background-color: rgba(37,99,235,0.45);
            selection-color: #ffffff;
            outline: none;
            padding: 4px;
            font: 600 {THEME.font.body_pt}pt '{THEME.font.family}';
        }}
        QComboBox QAbstractItemView::item {{
            padding: 6px 10px;
            border-radius: {THEME.radius.sm}px;
            min-height: 22px;
            color: #ffffff;
            background: transparent;
        }}
        QComboBox QAbstractItemView::item:hover {{
            background: rgba(255,255,255,0.10);
        }}
        QComboBox QAbstractItemView::item:selected {{
            background: rgba(37,99,235,0.45);
            color: #ffffff;
        }}
        QComboBox QListView {{
            background-color: {p.surface};
            color: #ffffff;
            border: none;
        }}
    """


def checkbox_style() -> str:
    """``QCheckBox``, small chamfered indicator filled with info gradient
    on check, plus a white ✓ painted inside via SVG."""
    p = THEME.palette
    g = _grad_for_variant("info")
    return f"""
        QCheckBox {{
            color: {p.text};
            font: {THEME.font.caption_pt}pt '{THEME.font.family}';
            spacing: 6px;
        }}
        QCheckBox::indicator {{
            width: 16px; height: 16px;
            border: 1px solid {p.surface_border_strong};
            border-radius: {THEME.radius.sm}px;
            background: rgba(255,255,255,0.04);
        }}
        QCheckBox::indicator:checked {{
            background: {g};
            border-color: transparent;
            image: url({_TICK_WHITE_URL});
        }}
        QCheckBox::indicator:hover {{
            border-color: {p.focus};
        }}
        QCheckBox:disabled {{ color: {p.text_dim}; }}
    """


# ============================================================================
# Surfaces, cards, dialogs, group boxes, panels
# ============================================================================

def card_style(*, elevated: bool = False,
               radius: Optional[int] = None,
               with_shadow: bool = False) -> str:
    """Reusable card surface, used for box rows, sidebars, dialog
    inner panels.  Set ``with_shadow=True`` for top-level dialogs."""
    p = THEME.palette
    r = radius if radius is not None else THEME.radius.lg
    g = THEME.gradient.surface_elev if elevated else THEME.gradient.surface
    return f"""
        QWidget {{
            background: {_qlinear(g)};
            border: 1px solid {p.surface_border};
            border-radius: {r}px;
            color: {p.text};
        }}
    """


def dialog_style() -> str:
    """Top-level dialog (QDialog), dark surface + thin glassy white
    outline so the dialog edge is visible against a dark desktop."""
    p = THEME.palette
    return f"""
        QDialog {{
            background: {_qlinear(THEME.gradient.surface)};
            color: {p.text};
            border: 1px solid rgba(255,255,255,0.16);
        }}
        QDialog QLabel {{ color: {p.text}; }}
        QDialog QFrame[frameShape="4"] {{
            color: {p.surface_border_strong};
            background: {p.surface_border_strong};
        }}
    """


# (border_rgba, title_color) per accent. Borders are low-alpha rgba so
# they hint at the section's identity without screaming. Titles use the
# vivid accent so the section label still reads as e.g. "purple = primary".
_GROUPBOX_ACCENTS = {
    "primary": ("rgba(168,85,247,0.35)",  "#c084fc"),
    "info":    ("rgba(96,165,250,0.35)",  "#93c5fd"),
    "success": ("rgba(52,211,153,0.35)",  "#6ee7b7"),
    "warning": ("rgba(245,158,11,0.35)",  "#fdba74"),
    "danger":  ("rgba(239,68,68,0.35)",   "#fca5a5"),
    "slate":   (None,                     None),  # uses surface_border_strong
}


def groupbox_style(accent: str = "slate") -> str:
    """Outlined ``QGroupBox`` with a small label tucked into the top edge.
    Used for Maze's Box Setup / Camera Control / Experiment Control /
    Adjust Zone / Setup Control containers."""
    p = THEME.palette
    border_rgba, title_color = _GROUPBOX_ACCENTS.get(accent, (None, None))
    border = border_rgba or p.surface_border_strong
    title_color = title_color or p.text_muted
    return f"""
        QGroupBox {{
            background: transparent;
            border: 1px solid {border};
            border-radius: {THEME.radius.md}px;
            margin-top: 12px;
            padding: 12px 10px 10px;
            color: {p.text};
            font: 700 {THEME.font.caption_pt}pt '{THEME.font.family}';
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 14px;
            padding: 0 8px;
            color: {title_color};
            background: {p.bg};
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }}
    """


# ============================================================================
# Labels / typography
# ============================================================================

def label_style(variant: str = "body") -> str:
    """Variants: display | section | body | caption | muted."""
    f = THEME.font
    p = THEME.palette
    if variant == "display":
        return (f"QLabel {{ color: {p.text}; font: {f.display_weight} "
                f"{f.display_pt}pt '{f.family}'; }}")
    if variant == "section":
        return (f"QLabel {{ color: {p.text}; font: {f.section_weight} "
                f"{f.section_pt}pt '{f.family}'; }}")
    if variant == "caption":
        return (f"QLabel {{ color: {p.text_muted}; font: {f.caption_weight} "
                f"{f.caption_pt}pt '{f.family}'; }}")
    if variant == "muted":
        return (f"QLabel {{ color: {p.text_dim}; font: {f.body_weight} "
                f"{f.body_pt}pt '{f.family}'; }}")
    # body
    return (f"QLabel {{ color: {p.text}; font: {f.body_weight} "
            f"{f.body_pt}pt '{f.family}'; }}")


# ============================================================================
# Timer label, three states
# ============================================================================

def timer_label_style(state: str = "idle") -> str:
    """Timer states: idle | running | error.

    Running uses the success-mint colour so there is one "alive" green."""
    p = THEME.palette
    if state == "running":
        bg     = "rgba(16,185,129,0.18)"
        border = "rgba(16,185,129,0.45)"
        color  = "#34d399"
    elif state == "error":
        bg     = "rgba(225,29,72,0.18)"
        border = "rgba(225,29,72,0.45)"
        color  = "#fb7185"
    else:  # idle
        bg     = "rgba(255,255,255,0.05)"
        border = p.surface_border_strong
        color  = p.text_muted
    return f"""
        QLabel {{
            background: {bg};
            color: {color};
            border: 1px solid {border};
            border-radius: {THEME.radius.sm}px;
            padding: 1px 10px;
            font-weight: 800;
            font-size: {THEME.font.body_pt + 1}pt;
            font-family: 'Cascadia Mono', 'Consolas', monospace;
        }}
    """


# ============================================================================
# Tab widget
# ============================================================================

def tab_widget_style(accent: Optional[str] = None) -> str:
    """``QTabWidget``, top tab bar with vivid-gradient active tab.
    Selected tab gets the ``info`` (sky → royal blue) gradient + white
    text; inactive tabs are muted text on the dark canvas."""
    p = THEME.palette
    selected_grad = _qlinear(THEME.gradient.info)
    return f"""
        QTabWidget::pane {{
            background: transparent;
            border: 1px solid {p.surface_border_strong};
            border-top: 1px solid {p.surface_border_strong};
            border-radius: {THEME.radius.md}px;
            top: -1px;
        }}
        QTabBar {{
            background: transparent;
            qproperty-drawBase: 0;
        }}
        QTabBar::tab {{
            background: rgba(255,255,255,0.04);
            color: {p.text_muted};
            padding: 7px 22px;
            margin: 2px 4px 0 0;
            min-height: 24px;
            border: 1px solid rgba(255,255,255,0.10);
            border-top: 1px solid rgba(255,255,255,0.18);
            border-radius: {THEME.radius.md}px;
            font: 700 11pt '{THEME.font.family}';
            min-width: 110px;
        }}
        QTabBar::tab:selected {{
            background: {selected_grad};
            color: #ffffff;
            border: 1px solid rgba(255,255,255,0.18);
            border-top: 1px solid rgba(255,255,255,0.32);
        }}
        QTabBar::tab:hover:!selected {{
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.16);
            border-top: 1px solid rgba(255,255,255,0.28);
            color: {p.text};
        }}
    """


# ============================================================================
# Scroll area / status bar / main window
# ============================================================================

def scroll_area_style() -> str:
    """Thin slim scrollbars on dark, applies app-wide (QScrollArea +
    QAbstractScrollArea + standalone QScrollBar).

    Both axes are styled, with transparent track + arrow buttons so only
    the slim slate handle is visible.
    """
    return f"""
        QScrollArea {{
            background: transparent;
            border: none;
            border-radius: {THEME.radius.md}px;
        }}

        /* Vertical */
        QScrollBar:vertical {{
            background: transparent;
            width: 10px;
            margin: 4px 2px;
            border: none;
        }}
        QScrollBar::handle:vertical {{
            background: rgba(255,255,255,0.14);
            border-radius: 5px;
            min-height: 24px;
        }}
        QScrollBar::handle:vertical:hover    {{ background: rgba(255,255,255,0.24); }}
        QScrollBar::handle:vertical:pressed  {{ background: rgba(255,255,255,0.32); }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical        {{ height: 0; background: transparent; }}
        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical        {{ background: transparent; }}

        /* Horizontal */
        QScrollBar:horizontal {{
            background: transparent;
            height: 10px;
            margin: 2px 4px;
            border: none;
        }}
        QScrollBar::handle:horizontal {{
            background: rgba(255,255,255,0.14);
            border-radius: 5px;
            min-width: 24px;
        }}
        QScrollBar::handle:horizontal:hover    {{ background: rgba(255,255,255,0.24); }}
        QScrollBar::handle:horizontal:pressed  {{ background: rgba(255,255,255,0.32); }}
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal        {{ width: 0; background: transparent; }}
        QScrollBar::add-page:horizontal,
        QScrollBar::sub-page:horizontal        {{ background: transparent; }}

        /* Corner box between H + V scrollbars. */
        QAbstractScrollArea::corner {{ background: transparent; }}
    """


def status_bar_style() -> str:
    """``QStatusBar``, solid bg, muted text, focus-coloured accent on right."""
    p = THEME.palette
    return f"""
        QStatusBar {{
            background: #050810;
            color: {p.text_muted};
            border-top: 1px solid {p.surface_border};
            padding: 4px 12px;
            font: {THEME.font.caption_pt}pt '{THEME.font.family}';
        }}
        QStatusBar::item {{ border: none; }}
    """


def main_window_style() -> str:
    """``QMainWindow``, dark canvas + a thin glassy white outline so the
    window edge is visible against a dark desktop.

    The outline is painted on the central widget (Qt can't border the
    native frame), reached via the ``#mwCentral`` objectName that
    ``MainWindowBase`` sets on its centralwidget."""
    p = THEME.palette
    return f"""
        QMainWindow {{
            background: {p.bg};
            color: {p.text};
        }}
        QMainWindow > QWidget#mwCentral {{
            background: {p.bg};
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 0px;
        }}
        QWidget {{ font-family: '{THEME.font.family}'; color: {p.text}; }}
    """


# ============================================================================
# Chips / progress
# ============================================================================

def master_group_qss() -> str:
    """Bordered-card QGroupBox for the Setup Control master row, ONE
    definition for both MainWindows."""
    return (
        "QGroupBox {"
        f" border: 1px solid {THEME.palette.surface_border_strong};"
        f" border-radius: {THEME.radius.md}px;"
        " margin-top: 2px; padding: 6px;"
        f" background-color: {THEME.palette.surface};"
        "}"
    )


def chip_style(variant: str = "neutral") -> str:
    """Compact status pill, used in the Statistics tab "State" column,
    Live Status connection banner, hw_* highlight, etc.

    Variants: neutral | success | warning | danger | info | hw."""
    if variant == "hw":
        return ("QLabel { background: #fde68a; color: #b45309;"
                f" border-radius: {THEME.radius.sm}px;"
                f" padding: 2px 6px; font: 700 {THEME.font.caption_pt}pt"
                f" '{THEME.font.family}'; }}")
    fg_map = {
        "neutral": (THEME.palette.text_muted, "rgba(255,255,255,0.06)"),
        "success": ("#34d399", "rgba(16,185,129,0.18)"),
        "warning": ("#fdba74", "rgba(245,158,11,0.18)"),
        "danger":  ("#f87171", "rgba(220,38,38,0.18)"),
        "info":    ("#93c5fd", "rgba(37,99,235,0.18)"),
    }
    color, bg = fg_map.get(variant, fg_map["neutral"])
    return ("QLabel { "
            f"background: {bg}; color: {color};"
            f" border-radius: {THEME.radius.pill}px;"
            f" padding: 2px 8px;"
            f" font: 700 {THEME.font.caption_pt}pt '{THEME.font.family}'; }}")


def progress_bar_style() -> str:
    p = THEME.palette
    return f"""
        QProgressBar {{
            background: rgba(255,255,255,0.05);
            border: 1px solid {p.surface_border_strong};
            border-radius: {THEME.radius.sm}px;
            color: {p.text};
            text-align: center;
            min-height: 18px;
            font: 600 {THEME.font.caption_pt}pt '{THEME.font.family}';
        }}
        QProgressBar::chunk {{
            background: {_qlinear(THEME.gradient.info)};
            border-radius: {THEME.radius.sm}px;
        }}
    """


# ============================================================================
# Tables, used by Statistics tab + dialog tables
# ============================================================================

def table_style() -> str:
    """``QTableWidget`` styling, used by Statistics tab and by the
    Connect/Disconnect/Config dialogs."""
    p = THEME.palette
    return f"""
        QTableWidget {{
            background: {p.surface};
            alternate-background-color: {p.surface_2};
            gridline-color: {p.surface_border};
            color: {p.text};
            border: 1px solid {p.surface_border};
            border-radius: {THEME.radius.sm}px;
            font: {THEME.font.body_pt}pt '{THEME.font.family}';
        }}
        QHeaderView::section {{
            background: {p.surface_2};
            color: {p.text_muted};
            border: none;
            padding: 6px 8px;
            font: 700 {THEME.font.caption_pt}pt '{THEME.font.family}';
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        /* Vertical (row-number) header, without this rule Qt paints it
           the OS default light-grey, a bright stripe down the left. */
        QHeaderView::section:vertical {{
            background: {p.surface_2};
            color: {p.text_muted};
            border: none;
            padding: 4px 6px;
            text-transform: none;
            letter-spacing: 0;
        }}
        /* The corner button (top-left of a table with both headers
           visible) defaults to plain white on Windows. Paint it dark. */
        QTableCornerButton::section {{
            background: {p.surface_2};
            border: none;
        }}
        QTableWidget::item:selected {{
            background: rgba(59,130,246,0.20);
            color: {p.text};
        }}
    """


# ============================================================================
# Convenience: apply the global app palette
# ============================================================================

def apply_global_palette(app) -> None:
    """Apply a Qt :class:`QPalette` that matches the dark tokens.

    Called once on app startup. Sets the colours Qt uses for
    non-styleable bits (tooltips, message-box backgrounds, etc.).
    """
    from PySide6 import QtGui
    p = THEME.palette
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.ColorRole.Window,       QtGui.QColor(p.bg))
    pal.setColor(QtGui.QPalette.ColorRole.WindowText,   QtGui.QColor(p.text))
    pal.setColor(QtGui.QPalette.ColorRole.Base,         QtGui.QColor(p.surface))
    pal.setColor(QtGui.QPalette.ColorRole.AlternateBase,QtGui.QColor(p.surface_2))
    pal.setColor(QtGui.QPalette.ColorRole.ToolTipBase,  QtGui.QColor(p.surface_elev))
    pal.setColor(QtGui.QPalette.ColorRole.ToolTipText,  QtGui.QColor(p.text))
    pal.setColor(QtGui.QPalette.ColorRole.Text,         QtGui.QColor(p.text))
    pal.setColor(QtGui.QPalette.ColorRole.Button,       QtGui.QColor(p.surface_elev))
    pal.setColor(QtGui.QPalette.ColorRole.ButtonText,   QtGui.QColor(p.text))
    pal.setColor(QtGui.QPalette.ColorRole.Highlight,    QtGui.QColor(p.info))
    pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(p.text))
    pal.setColor(QtGui.QPalette.ColorRole.Link,         QtGui.QColor(p.link))
    pal.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor(p.text_dim))
    app.setPalette(pal)


def force_dark_titlebar(window) -> None:
    """Make a Qt window's native title bar use Windows dark mode.

    Qt can't theme the Win32 caption strip; one DwmSetWindowAttribute
    call per top-level window does it. No-op on non-Windows platforms.

    Call right *after* ``window.show()`` so the HWND exists. The DWM
    attribute number differs by Win 10 build: 20 is the documented
    DWMWA_USE_IMMERSIVE_DARK_MODE; we try 20 first and fall back to 19.
    """
    import sys
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        hwnd = int(window.winId())
        value = ctypes.c_int(1)
        dwmapi = ctypes.windll.dwmapi
        # 20 = DWMWA_USE_IMMERSIVE_DARK_MODE (Win 10 20H1+/Win 11)
        # 19 = same attribute pre-20H1
        res = dwmapi.DwmSetWindowAttribute(
            hwnd, 20, ctypes.byref(value), ctypes.sizeof(value)
        )
        if res != 0:
            dwmapi.DwmSetWindowAttribute(
                hwnd, 19, ctypes.byref(value), ctypes.sizeof(value)
            )
    except Exception:
        # Title bar will stay white but app keeps working.
        pass


def dialog_theme_qss(extras: str = "") -> str:
    """Composite dark QSS every multi-box dialog applies (dialog + table +
    inputs + combobox + checkbox + label). Pass extra rules, e.g.
    ``button_style("secondary")``: via ``extras``."""
    return (
        dialog_style()
        + table_style()
        + input_style()
        + combobox_style()
        + checkbox_style()
        + label_style()
        + (extras or "")
    )


def apply_dialog_theme(dialog, extras: str = "") -> None:
    """Set ``dialog_theme_qss(extras)`` as the dialog's stylesheet."""
    dialog.setStyleSheet(dialog_theme_qss(extras))


__all__ = [
    "button_style", "light_button_style",
    "input_style", "combobox_style", "checkbox_style",
    "card_style", "dialog_style", "groupbox_style",
    "label_style", "timer_label_style",
    "tab_widget_style", "scroll_area_style", "status_bar_style",
    "main_window_style", "chip_style", "master_group_qss", "progress_bar_style", "table_style",
    "apply_global_palette", "force_dark_titlebar",
    "dialog_theme_qss", "apply_dialog_theme",
]
