"""Compatibility wrapper. All real styling lives in:
  source.gui.theme.THEME, every colour/gradient/radius/font value
  source.gui.style_builders, QSS composers per widget kind

The constants below derive their values from THEME. New code should
import from ``style_builders`` directly:

    from source.gui.style_builders import button_style, input_style
    btn.setStyleSheet(button_style("primary"))
"""

from __future__ import annotations


from source.gui.theme import THEME
from source.gui.style_builders import (
    apply_global_palette as _apply_global_palette,
    main_window_style, dialog_style, scroll_area_style, status_bar_style,
    button_style, input_style as _input_style,
    combobox_style as _combobox_style, table_style,
    timer_label_style, groupbox_style as _groupbox_style,
    checkbox_style as _checkbox_style,
    tab_widget_style, spinbox_style as _spinbox_style,
)
from source.log import get_logger

logger = get_logger()


# =============================================================================
# COLORS dict. Each entry that drives a button's fill is a Qt
# qlineargradient string, so ``BUTTON_STYLE.format(color=COLORS['primary'])``
# produces a vivid gradient (violet → purple, mint → emerald, etc.).
# Hover variants brighten toward the start stop so the button "lights up"
# without picking a new hue.
# =============================================================================
_p = THEME.palette
_g = THEME.gradient


def _qgrad(start: str, end: str) -> str:
    """Encode a 135° linear gradient as Qt's ``qlineargradient(...)``
    syntax. Drop straight into ``background: {color};``."""
    return (
        f"qlineargradient(x1:0, y1:0, x2:1, y2:1, "
        f"stop:0 {start}, stop:1 {end})"
    )


def _qgrad_hover(start: str, end: str) -> str:
    """Hover variant, same hue pair but biased toward the brighter
    start colour so more of the gradient sits at the lighter stop."""
    return (
        f"qlineargradient(x1:0, y1:0, x2:1, y2:1, "
        f"stop:0 {start}, stop:0.7 {start}, stop:1 {end})"
    )


# Named gradient pairs from the theme tokens.
_PR = (_g.primary.start, _g.primary.end)
_SU = (_g.success.start, _g.success.end)
_WA = (_g.warning.start, _g.warning.end)
_DA = (_g.danger.start,  _g.danger.end)
_IN = (_g.info.start,    _g.info.end)

COLORS = {
    'primary':              _qgrad(*_PR),       'primary_hover':       _qgrad_hover(*_PR),
    'success':              _qgrad(*_SU),       'success_hover':       _qgrad_hover(*_SU),
    'warning':              _qgrad(*_WA),       'warning_hover':       _qgrad_hover(*_WA),
    'danger':               _qgrad(*_DA),       'danger_hover':        _qgrad_hover(*_DA),
    'info':                 _qgrad(*_IN),       'info_hover':          _qgrad_hover(*_IN),
    'record':               _qgrad(*_DA),       'record_hover':        _qgrad_hover(*_DA),
    'record_light':         _qgrad(*_SU),       'record_light_hover':  _qgrad_hover(*_SU),
    'start_light':          _qgrad(*_SU),       'start_light_hover':   _qgrad_hover(*_SU),
    'config':               _qgrad(*_PR),       'config_hover':        _qgrad_hover(*_PR),
    'config_light':         _qgrad(*_PR),       'config_light_hover':  _qgrad_hover(*_PR),
    # Non-button anchors keep their solid values.
    'dark':                 _p.surface,
    'light':                _p.text,
    'border':               _p.surface_border_strong,
    'text':                 _p.text,
    'text_light':           _p.text_muted,
}


# =============================================================================
# BUTTON_STYLE, templated solid button. Used by callsites that do
#   BUTTON_STYLE.format(color=..., hover_color=...)
# Honours the colour they pass but applies the chamfered radius, padding,
# font, and disabled-state from the tokens.
#
# Built as ONE f-string. The `color`/`hover_color` placeholders must
# survive f-string eval for the later `.format()`, and the QSS rule-body
# braces must reach `.format()` as `{{`/`}}`. Both stages collapse a
# brace pair once, so we write FOUR braces `{{{{`/`}}}}` for literals.
# =============================================================================
BUTTON_STYLE = (
    f"QPushButton {{{{"
    f" background: {{color}};"
    f" color: #ffffff;"
    f" border: 1px solid rgba(255,255,255,0.18);"
    f" border-top: 1px solid rgba(255,255,255,0.32);"
    f" border-radius: {THEME.radius.md}px;"
    # 8 px horizontal; anything bigger clips labels like "Camera Config"
    # in the fixed-width box-row buttons.
    f" padding: 0 8px;"
    f" font: 600 {THEME.font.body_pt}pt '{THEME.font.family}';"
    f" min-height: 20px;"
    f" text-align: center;"
    f"}}}}"
    f"QPushButton:hover {{{{"
    f" background: {{hover_color}};"
    f" border: 1px solid rgba(255,255,255,0.28);"
    f" border-top: 1px solid rgba(255,255,255,0.45);"
    f"}}}}"
    f"QPushButton:pressed {{{{ background: {{color}}; padding-top: 1px; }}}}"
    f"QPushButton:disabled {{{{"
    f" background: rgba(148,163,184,0.10);"
    f" color: {_p.text_dim};"
    f" border: 1px solid rgba(148,163,184,0.18);"
    f"}}}}"
)


# Push button, info-variant of button_style.
PUSH_BUTTON_STYLE = button_style("info", height=24, padding_h=10)

LINE_EDIT_STYLE = _input_style()
COMBOBOX_STYLE = _combobox_style()
TIMER_LABEL_STYLE = timer_label_style("running")
STATUS_BAR_STYLE = status_bar_style()

# Outlined groupbox variants, delegate to the groupbox builder.
BOX_CONTROLS_GROUP_STYLE     = _groupbox_style("success")
BOX_CONTROLS_SUB_GROUP_STYLE = _groupbox_style("warning")

# Soft centre-weighted glow divider (control-row separators). Shared by both
# main-window modes; QSS ignores the internal whitespace.
HRULE_QSS = (
    "QFrame { border: none;"
    " background: qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5,"
    "  stop:0 rgba(255,255,255,0.0),"
    "  stop:0.2 rgba(255,255,255,0.18),"
    "  stop:0.5 rgba(255,255,255,0.30),"
    "  stop:0.8 rgba(255,255,255,0.18),"
    "  stop:1 rgba(255,255,255,0.0));"
    " margin: 8px 12px; padding: 0px; }"
)

# The global QSS includes every widget kind any window can hold so the
# styling is consistent regardless of which main_window applied it.
# combobox/checkbox/groupbox included here too, else a generic one in a
# dialog falls back to Qt's default (white on white).
GLOBAL_STYLE = (
    main_window_style()
    + dialog_style()
    + tab_widget_style()
    + status_bar_style()
    + scroll_area_style()
    + table_style()
    + _spinbox_style()
    + _combobox_style()
    + _checkbox_style()
    + _groupbox_style("slate")
)


# =============================================================================
# ThemeManager, dark-only stub. Any caller that asks for ``light`` gets dark.
# =============================================================================
class ThemeManager:
    """Dark-only theme manager. ``apply_theme`` always installs the dark
    palette via ``apply_global_palette``."""

    @staticmethod
    def apply_theme(app, theme_name="dark"):
        """Always applies dark, the requested ``theme_name`` is ignored."""
        try:
            _apply_global_palette(app)
            app.setStyleSheet(GLOBAL_STYLE)
        except Exception as e:
            logger.error(f"apply_theme failed: {e}")


__all__ = [
    "COLORS", "BUTTON_STYLE", "PUSH_BUTTON_STYLE",
    "LINE_EDIT_STYLE", "COMBOBOX_STYLE",
    "TIMER_LABEL_STYLE", "STATUS_BAR_STYLE",
    "BOX_CONTROLS_GROUP_STYLE", "BOX_CONTROLS_SUB_GROUP_STYLE",
    "HRULE_QSS", "GLOBAL_STYLE", "ThemeManager",
]
