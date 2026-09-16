"""Design tokens, single source of truth for every colour, gradient,
radius, font, and spacing value used by the Qt GUI.

Mirror of the ``:root`` CSS variables in ``docs/ui_mockup.html``.
Editing this file is the ONLY way to change the app's visual; every
widget pulls its style from ``style_builders``, which reads ``THEME``.

Dark-only by design; there is no light variant, no theme switch, no
runtime mutation. The dataclasses are frozen so anyone tempted to write
``THEME.palette.bg = "#fff"`` will get an attribute error.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Palette:
    """Solid colour anchors. Used directly for static surfaces and
    semantic chips; gradients in :class:`Gradients` reference these."""
    # bg is a *slightly lifted* near-black so cards and disabled controls
    # read against it.  Pure #000 made every disabled button vanish.
    bg:                       str = "#0a0f1e"
    surface:                  str = "#111a2e"
    surface_2:                str = "#1a2340"
    surface_elev:             str = "#162038"
    surface_elev_2:           str = "#1f2a4c"
    # rgba string used inside QSS – brighter than the mockup so subtle
    # 1 px borders are actually visible on the lifted bg.
    surface_border:           str = "rgba(255,255,255,0.10)"
    surface_border_strong:    str = "rgba(255,255,255,0.18)"
    text:                     str = "#f8fafc"
    text_muted:               str = "#cbd5e1"
    text_dim:                 str = "#64748b"
    accent:                   str = "#3b82f6"
    focus:                    str = "#a855f7"
    link:                     str = "#60a5fa"
    success:                  str = "#4a5c2a"
    warning:                  str = "#f59e0b"
    danger:                   str = "#e11d48"
    info:                     str = "#2563eb"


@dataclass(frozen=True)
class Gradient:
    """A two-stop linear gradient. ``angle`` is degrees clockwise from
    horizontal, Qt QSS uses x1/y1 to x2/y2; the helper in
    ``style_builders`` converts."""
    start: str
    end:   str
    angle: int = 135


@dataclass(frozen=True)
class Gradients:  
    # Vivid CTA gradients. Lightened a step for better legibility of
    # white labels + icons, the lift is weighted toward the darker end stop so
    # buttons read brighter without losing their hue identity or washing out.
    # Muted, desaturated CTA palette, calm tones that read clearly with white
    # labels on the dark slate UI without any colour "shouting". Same five
    # semantic hues (violet/blue/green/amber/red), but low saturation and a
    # harmonised lightness band so none of them dominate.
    primary: Gradient = field(default_factory=lambda: Gradient("#7b6ba8", "#5f5193"))
    # info (blue) stays vivid. success (green) is a DIM fig/sage green (not the
    # old vivid #22c55e), softer on the dark slate while still clearly green.
    # Drives every green button (Save, Start/Record via record_light/start_light).
    info:    Gradient = field(default_factory=lambda: Gradient("#3b82f6", "#2563eb"))
    success: Gradient = field(default_factory=lambda: Gradient("#6b8347", "#4a5f30"))
    warning: Gradient = field(default_factory=lambda: Gradient("#bf9450", "#9c7536"))
    danger:  Gradient = field(default_factory=lambda: Gradient("#b06a60", "#8a463c"))
    # Surfaces (cards / dialogs / panels)
    surface:      Gradient = field(default_factory=lambda: Gradient("#0c1424", "#1a2240"))
    surface_elev: Gradient = field(default_factory=lambda: Gradient("#101a36", "#1f2a55"))


@dataclass(frozen=True)
class Radius:
    """Chamfered, not pill, soft corners, not rounded."""
    sm:   int = 4   # inputs, checkboxes, chips
    md:   int = 6   # buttons, tabs
    lg:   int = 10  # cards, dialogs
    pill: int = 14  # reserved for hw_* chips (used sparingly)


@dataclass(frozen=True)
class Spacing:
    """4 / 8 / 12 / 16 / 24 / 32 px ladder."""
    xs:  int = 4
    sm:  int = 8
    md:  int = 12
    lg:  int = 16
    xl:  int = 24
    xxl: int = 32


@dataclass(frozen=True)
class FontSpec:
    """Type ramp. Sizes are in points (Qt convention). The HTML mockup
    uses pixels, px≈pt*1.33 on 96 dpi; the px-pt mapping here was
    eyeballed against the screenshot and pyOperant's existing UI."""
    family:           str = "Segoe UI"
    mono_family:      str = "Cascadia Mono"
    # 22 px in HTML → ~16 pt in Qt
    display_pt:       int = 16
    display_weight:   int = 700
    section_pt:       int = 12
    section_weight:   int = 700
    body_pt:          int = 10
    body_weight:      int = 500
    caption_pt:       int = 8
    caption_weight:   int = 400
    mono_pt:          int = 9


@dataclass(frozen=True)
class Borders:
    """Border widths. Surface borders use ``Palette.surface_border``
    rgba strings, see :func:`style_builders.card_style`."""
    thin:   int = 1
    medium: int = 2
    focus:  int = 2  # 2 px outline around focused inputs


@dataclass(frozen=True)
class Shadow:
    """Soft drop-shadow on cards/dialogs (T6 polish)."""
    card:   str = "0 6px 24px rgba(0,0,0,0.45)"
    dialog: str = "0 30px 60px rgba(0,0,0,0.6)"


@dataclass(frozen=True)
class Theme:
    """Root token tree. Import the module-level ``THEME`` singleton."""
    palette:  Palette   = field(default_factory=Palette)
    gradient: Gradients = field(default_factory=Gradients)
    radius:   Radius    = field(default_factory=Radius)
    spacing:  Spacing   = field(default_factory=Spacing)
    font:     FontSpec  = field(default_factory=FontSpec)
    border:   Borders   = field(default_factory=Borders)
    shadow:   Shadow    = field(default_factory=Shadow)


# Singleton. Import this everywhere; don't construct a second one.
THEME = Theme()


__all__ = [
    "Palette", "Gradient", "Gradients", "Radius", "Spacing",
    "FontSpec", "Borders", "Shadow", "Theme", "THEME",
]
