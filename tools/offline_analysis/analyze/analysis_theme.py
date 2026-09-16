"""The Analyze workbench's visual identity, in one place.

Two things make a technical panel read as crisp, and neither is the colour.

The first is that **one stylesheet dresses everything**. Set it on the window
and every child inherits; a widget only carries an ``objectName`` when it is
genuinely a different KIND of thing (a primary action, a toggle chip, a
section title). The workbench previously carried three inline button styles
and about forty `setStyleSheet` calls, which is why nothing quite matched
anything else and why a container's border kept leaking onto its labels.

The second is **neutrals that were chosen**. These are near-blacks biased
toward the periwinkle accent rather than pure grey, so the panel reads as one
material with the accent sitting in it, not as grey with blue stuck on.

Type is monospaced throughout: this is a tool for reading numbers in columns,
and a proportional face makes a column of measurements ragged.

Semantic colour is separate from the accent and never doubles as it: green
means a thing succeeded, amber that work is pending, red that something is
blocked. The accent means "this is a heading" and nothing else.
"""

from __future__ import annotations

# ── the palette ──────────────────────────────────────────────────────────────
#
# Surfaces, darkest to lightest. `LOG` is the deepest because a scrolling text
# well reads best as a hole in the panel.
LOG = "#0d0d14"
WELL = "#111118"        # lists, tables
SUNKEN = "#12121a"      # inputs
BG = "#1a1a1e"          # the window itself
PANEL = "#20202a"       # raised group
RAISED = "#252530"      # buttons

# Hairlines, quietest to loudest.
LINE = "#2a2a3a"
LINE2 = "#333344"
LINE3 = "#3a3a50"

# Ink.
INK = "#e8e8f2"         # primary text
INK2 = "#a8a8c4"        # labels, secondary
INK3 = "#d0d0ee"        # button text
INK_DIM = "#6e6e8a"     # disabled, and ONLY disabled
#: The name of a setting. It was INK_DIM, i.e. the same grey as a control you
#: cannot use, so every label in the panel read as switched off, and the
#: settings blended into the ground they sat on. 5.2:1 on SUNKEN.
INK_LABEL = "#9c9cbe"

# Accent, periwinkle. Headings and selection; never a status.
ACCENT = "#6677ff"
ACCENT2 = "#9090cc"     # group titles, one step back
FILL = "#3355cc"        # filled accent (progress, primary button)
FILL_HOVER = "#4466dd"
SEL = "#2a2a50"         # selected row

# Semantic. Each is a text/ground/border triple so a control can be built from
# it without inventing shades at the call site.
OK, OK_BG, OK_LINE = "#88ee88", "#1a3a1a", "#44aa44"
BAD, BAD_BG, BAD_LINE = "#ee8888", "#3a1a1a", "#aa4444"
#: The demo had no "pending" colour; the Plan column needs one. Amber, held
#: well down in saturation so it sits beside the indigo instead of fighting it.
WARN, WARN_BG, WARN_LINE = "#e8c07a", "#3a301a", "#aa8a44"

LOG_INK = "#b0ffb0"     # terminal green, for the run log only

MONO = "'Cascadia Mono', 'Consolas', 'DejaVu Sans Mono', monospace"


# ── the spin arrows ──────────────────────────────────────────────────────────
#
# Qt draws its own up/down indicator for a QSpinBox, and once a stylesheet
# touches the widget that indicator comes out as a solid white block. The
# transparent-border triangle every CSS answer suggests renders as a block
# too: Qt's ::up-arrow is an image slot, not a box it will draw borders on.
#
# So the arrows ARE images, written once to the user's cache directory. A
# nine-line PNG encoder beats a binary asset in the repo, and beats importing
# Qt here, this module is imported early, and a Qt import from a module that
# does not need one is how a fixture took down eleven unrelated tests.
def _png(rows, rgb) -> bytes:
    """A minimal RGBA PNG. `rows` is a list of equal-length mask strings."""
    import struct
    import zlib

    h, w = len(rows), len(rows[0])
    r, g, b = rgb
    raw = b""
    for row in rows:
        raw += bytes([0]) + b"".join(
            bytes((r, g, b, 255 if c != " " else 0)) for c in row)

    def chunk(tag, data):
        c = tag + data
        return (struct.pack(">I", len(data)) + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF))

    sig = bytes([137, 80, 78, 71, 13, 10, 26, 10])
    return (sig
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


_UP = ["    #    ",
       "   ###   ",
       "  #####  ",
       " ####### ",
       "#########"]
_DOWN = list(reversed(_UP))


def _arrow_paths():
    """(up, down) as forward-slashed paths a stylesheet can name.

    Returns ("", "") if they cannot be written, the arrows then fall back to
    Qt's own, which is what the panel had before, not a crash.
    """
    import os
    import tempfile

    try:
        d = os.path.join(tempfile.gettempdir(), "pybehavetrack_ui")
        os.makedirs(d, exist_ok=True)
        ink = (0x9c, 0x9c, 0xbe)
        out = []
        for name, rows in (("arrow_up.png", _UP), ("arrow_down.png", _DOWN)):
            p = os.path.join(d, name)
            data = _png(rows, ink)
            # Rewrite only when it differs, so a running second instance is
            # not handed a half-written file.
            if not os.path.exists(p) or open(p, "rb").read() != data:
                with open(p, "wb") as fh:
                    fh.write(data)
            out.append(p.replace(chr(92), "/"))
        return tuple(out)
    except Exception:
        return "", ""


ARROW_UP, ARROW_DOWN = _arrow_paths()
# A plain string, not an f-string: it is interpolated INTO STYLE as a value,
# so its braces are never re-processed and must be single ones.
_ARROW_CSS = ("""
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: url("__UP__"); width: 9px; height: 5px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: url("__DOWN__"); width: 9px; height: 5px;
}
""".replace("__UP__", ARROW_UP).replace("__DOWN__", ARROW_DOWN)
              if ARROW_UP else "")


#: Applied once, to the workbench window. Everything below inherits.
STYLE = f"""
QWidget {{
    background-color: {BG};
    color: {INK};
    font-family: {MONO};
    font-size: 12px;
}}

/* ── containers ───────────────────────────────────────────── */
QGroupBox {{
    background-color: {PANEL};
    border: 1px solid {LINE2};
    border-radius: 4px;
    margin-top: 9px;
    padding: 12px 10px 10px 10px;
    color: {ACCENT2};
    font-weight: bold;
    letter-spacing: 1px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 9px;
}}
QSplitter::handle {{ background-color: {LINE}; width: 3px; }}
QScrollArea {{ border: none; }}

/* The settings panel floats OVER the content, so it needs a ground of its
   own and an edge to separate it, a transparent overlay would show the
   options column straight through the controls. */
QStackedWidget#sidebarPanel {{
    background-color: {BG};
    border-left: 1px solid {LINE3};
}}

/* The settings sidebar's edge tabs. The height MUST be set here, not with
   setFixedHeight: a stylesheet that gives QPushButton padding makes Qt
   recompute the minimum height from the stylesheet, and that recomputed 34 px
   beat the fixed 130, so all three tabs collapsed and their rotated labels
   drew over one another. */
QPushButton#sidebarTab {{
    min-height: 132px;
    max-height: 132px;
    min-width: 24px;
    max-width: 24px;
    padding: 0;
    border: none;
}}

/* A foldable options block. The header IS the control, so it has to read as
   clickable without shouting louder than the values underneath it. */
QFrame#sectionBox {{
    background-color: {PANEL};
    border: 1px solid {LINE2};
    border-radius: 4px;
}}
QPushButton#sectionHead {{
    background-color: transparent;
    border: none;
    border-bottom: 1px solid {LINE2};
    border-radius: 0;
    color: {ACCENT2};
    font-weight: bold;
    letter-spacing: 1px;
    padding: 6px 10px;
    text-align: left;
}}
QPushButton#sectionHead:hover {{ color: {INK3}; background-color: {RAISED}; }}
QPushButton#sectionHead:checked {{ border-bottom: 1px solid {LINE2}; }}
QFrame#sectionBox[objectName="needs"] {{ border-color: {WARN_LINE}; }}

/* The "what is missing" group. Warning-coloured, because every line in it
   is something that will otherwise silently degrade the results, and each
   line is a button, so it has to read as one. */
QFrame#needs {{
    background-color: {WARN_BG};
    border: 1px solid {WARN_LINE};
    border-radius: 4px;
}}
QFrame#needs QPushButton#sectionHead {{
    color: {WARN};
    border-bottom: 1px solid {WARN_LINE};
}}
QPushButton#need {{
    background-color: transparent;
    border: 1px solid {WARN_LINE};
    border-radius: 3px;
    color: {WARN};
    padding: 5px 10px;
    text-align: left;
    font-weight: normal;
    letter-spacing: 0;
}}
QPushButton#need:hover {{
    background-color: {WARN_LINE};
    color: #1a1a1e;
}}

/* ── text ─────────────────────────────────────────────────── */
QLabel {{ color: {INK2}; background: transparent; }}
QLabel#section {{
    color: {ACCENT};
    font-weight: bold;
    letter-spacing: 2px;
    font-size: 11px;
    padding-top: 2px;
}}
QLabel#field  {{ color: {INK_LABEL}; }}
QLabel#value  {{ color: {INK}; }}
QLabel#plan   {{ color: {INK}; }}
QLabel#planBlocked {{ color: {BAD}; }}
QLabel#status {{ color: {INK_LABEL}; }}
QLabel#note   {{ color: {WARN}; }}

/* ── buttons ──────────────────────────────────────────────── */
QPushButton {{
    background-color: {RAISED};
    border: 1px solid {LINE3};
    border-radius: 3px;
    padding: 5px 11px;
    color: {INK3};
}}
QPushButton:hover  {{ background-color: #303045; border-color: {ACCENT}; }}
QPushButton:pressed {{ background-color: #1a1a2a; }}
QPushButton:disabled {{ color: {INK_DIM}; border-color: {LINE}; background: {SUNKEN}; }}
QPushButton::menu-indicator {{ width: 0px; }}

/* A quiet button: the ones that only change the selection. */
QPushButton#quiet {{ background: transparent; border-color: {LINE3}; color: {INK3}; }}
QPushButton#quiet:hover {{ background-color: {RAISED}; color: {INK}; border-color: {ACCENT}; }}

/* The one action that does the work. */
QPushButton#run {{
    background-color: {OK_BG};
    border: 1px solid {OK_LINE};
    color: {OK};
    font-weight: bold;
    font-size: 13px;
    padding: 8px 20px;
    letter-spacing: 1px;
}}
QPushButton#run:hover {{ background-color: #224422; }}
QPushButton#run:disabled {{
    background-color: {SUNKEN}; border-color: {LINE}; color: {INK_DIM};
}}
QPushButton#stop {{
    background-color: {BAD_BG}; border: 1px solid {BAD_LINE};
    color: {BAD}; font-weight: bold; padding: 8px 16px;
}}
QPushButton#stop:hover {{ background-color: #522424; }}

/* Setup actions, the two that unblock a recording. */
QPushButton#setup {{
    background-color: #24244a; border: 1px solid #4a4a8a; color: #c8c8ff;
}}
QPushButton#setup:hover {{ background-color: #2e2e5e; border-color: {ACCENT}; }}
QPushButton#setup[wanted="true"] {{ border: 1px solid {ACCENT}; color: #e0e0ff; }}

/* Workspace segmented control. */
QPushButton#seg {{
    background-color: {SUNKEN}; border: 1px solid {LINE2};
    color: {INK2}; font-weight: bold; padding: 5px 0; letter-spacing: 1px;
}}
QPushButton#seg:hover {{ color: {INK}; }}
QPushButton#seg:checked {{
    background-color: {SEL}; color: #d8d8ff; border-color: {ACCENT};
}}
QPushButton#seg:disabled {{ color: {INK_DIM}; background: {LOG}; }}

/* Measure toggles. */
QPushButton#chip {{
    background-color: {SUNKEN}; border: 1px solid {LINE2};
    border-radius: 11px; padding: 5px 12px; color: {INK2}; font-weight: bold;
}}
QPushButton#chip:hover {{ color: {INK}; border-color: {LINE3}; }}
QPushButton#chip:checked {{
    background-color: {SEL}; color: #d8d8ff; border-color: {ACCENT};
}}

/* ── inputs ───────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {SUNKEN};
    border: 1px solid {LINE3};
    border-radius: 3px;
    padding: 3px 6px;
    color: {INK};
    selection-background-color: {FILL};
}}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    border-color: {ACCENT2};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled {{ color: {INK_DIM}; border-color: {LINE}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 16px; }}

/* The spin buttons were never styled, so Qt drew its own pair OUTSIDE the
   rounded border, two loose squares beside every number in the panel. */
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    width: 15px;
    border: none;
    /* Opaque, not transparent: a transparent sub-control lets the style paint
       its own button primitive underneath, which came out as a white bar
       behind the arrow. */
    background-color: {SUNKEN};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-position: top right;
    margin: 1px 1px 0 0;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-position: bottom right;
    margin: 0 1px 1px 0;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {LINE2};
}}
{_ARROW_CSS}
QComboBox QAbstractItemView {{
    background-color: {WELL}; color: {INK};
    selection-background-color: {SEL}; border: 1px solid {LINE2};
}}
QCheckBox {{ color: {INK}; spacing: 6px; background: transparent; }}
QCheckBox:hover {{ color: #ffffff; }}
QCheckBox:disabled {{ color: {INK_DIM}; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border: 1px solid #4a4a68;
    border-radius: 2px; background: {SUNKEN};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background-color: {FILL}; border-color: {ACCENT}; }}

/* ── data ─────────────────────────────────────────────────── */
QTableWidget {{
    background-color: {WELL};
    alternate-background-color: #141420;
    gridline-color: {LINE};
    border: 1px solid {LINE};
    color: {INK};
    selection-background-color: {SEL};
    selection-color: {INK};
}}
QHeaderView::section {{
    background-color: {PANEL};
    color: {ACCENT2};
    border: none;
    border-bottom: 1px solid {LINE2};
    border-right: 1px solid {LINE};
    padding: 5px 6px;
    font-weight: bold;
}}
QTableWidget QTableCornerButton::section {{
    background-color: {PANEL}; border: none;
}}
QListWidget {{
    background-color: {WELL}; border: 1px solid {LINE};
    alternate-background-color: #141420;
}}
QListWidget::item:selected {{ background-color: {SEL}; }}

/* The run log reads as a terminal, because that is what it is. */
QTextEdit {{
    background-color: {LOG};
    border: 1px solid {LINE};
    color: {LOG_INK};
    font-family: {MONO};
    font-size: 11px;
}}

QTabWidget::pane {{ border: 1px solid {LINE}; top: -1px; }}
QTabBar::tab {{
    background: {SUNKEN}; color: {INK2};
    border: 1px solid {LINE}; border-bottom: none;
    padding: 6px 16px; margin-right: 2px;
    font-weight: bold; letter-spacing: 1px;
}}
QTabBar::tab:hover {{ color: {INK}; }}
QTabBar::tab:selected {{ background: {PANEL}; color: {ACCENT}; border-color: {LINE2}; }}

QMenu {{ background-color: {WELL}; border: 1px solid {LINE2}; color: {INK}; }}
QMenu::item:selected {{ background-color: {SEL}; }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 4px 8px; }}

QProgressBar {{
    background-color: {WELL}; border: 1px solid {LINE};
    border-radius: 3px; text-align: center; color: {INK2};
}}
QProgressBar::chunk {{ background-color: {FILL}; border-radius: 2px; }}

QScrollBar:vertical {{ background: {BG}; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {LINE3}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT2}; }}
QScrollBar:horizontal {{ background: {BG}; height: 11px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {LINE3}; border-radius: 5px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: {ACCENT2}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
"""


def section(text: str):
    """An uppercase, letter-spaced accent heading."""
    from PySide6.QtWidgets import QLabel

    lb = QLabel(text.upper())
    lb.setObjectName("section")
    return lb


def field(text: str):
    """A quiet name for the control beside it."""
    from PySide6.QtWidgets import QLabel

    lb = QLabel(text)
    lb.setObjectName("field")
    return lb
