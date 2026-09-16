"""Per-box ERROR alert propagation.

When a box hits an error (board/MCU disconnect, PyboardError, serial drop,
camera drop, error-stop), its status goes sticky-red via
``run_task.set_status(..., "error")``. ``base.compute_ui_state`` derives an
``error`` flag per box from that. This module routes that flag to EVERY
per-box surface so the box (e.g. "Box 5") turns red everywhere at once:
  - the box control / setup widget
  - the Live Status panel
  - the Statistics tab (row + plot)
  - the Video tile

Single entry point: ``apply_box_alerts(main_window, box_states)``, called
from ``_apply_ui_state`` (operant + maze). Because ``refresh_ui_state`` runs
on every connect/disconnect/start/stop/error, alerts appear and AUTO-CLEAR
(the flag flips false when the box recovers) with no extra plumbing.

PERFORMANCE: each surface's ``set_error`` does a full Qt ``setStyleSheet``
(style unpolish/repolish + repaint), and ``refresh_ui_state`` fires once per
box during multi-box ops. The ``_alert_changed`` guard re-styles a surface
only when that box's (error, message) actually CHANGES. The cache lives on
each surface widget, so a rebuilt surface starts fresh and is styled on first
apply, the guard never hides a live alert.

Each surface owns its own ``set_error(on, msg)`` styling; this module only
maps box_id -> surface and is defensive so a missing surface is skipped.
"""

from source.log import get_logger

logger = get_logger()

ERROR_COLOR = "#ff6b6b"      # the one red; run_task.STATUS_COLOURS["error"] is this


# ---------------------------------------------------------------------------
# Shared vocabulary for the surfaces' own ``set_error``.
#
# Each surface still owns its stylesheet, they frame different widgets with
# different selectors and that is deliberate (see the header). What is shared
# is the wording and the identity-button styling, which were written out four
# times and drifted: two surfaces prefixed the warning glyph, two did not, and
# every one of them hard-coded the red rather than using ERROR_COLOR above.
# ---------------------------------------------------------------------------

def error_text(noun, number, on) -> str:
    """Header text for a surface that renames itself when the box errors."""
    return ("⚠ {} {}" if on else "{} {}").format(noun, number)


def error_tooltip(label, on, msg, *, idle="", bare=None) -> str:
    """Tooltip for an errored surface.

    ``idle`` is shown when there is no error, some surfaces use that slot for
    their normal hint rather than clearing it. ``bare`` is shown when the box
    is in error but carried no message; where it is None the surface falls
    back to ``idle``, which is what the two-way callers already did.
    """
    if on and msg:
        return f"{label} error: {msg}"
    if on and bare is not None:
        return bare
    return idle


def identity_button_style(colour) -> str:
    """The flat, transparent per-box identity button (operant box label and
    maze id button are the same control wearing different names)."""
    return ("QPushButton{background:transparent;border:none;"
            f"color:{colour};padding:0px;font-weight:bold;}}"
            "QPushButton:hover{color:#61dafb;text-decoration:underline;}")


def apply_box_alerts(main_window, box_states):
    """Render each box's derived ``error`` flag red across all its surfaces.

    Idempotent: the expensive per-surface ``set_error`` restyle only fires
    when a box's (error, message) changes."""
    if not box_states:
        return
    for setup_id, bs in box_states.items():
        on = bool(bs.get("error"))
        msg = str(bs.get("error_msg") or "") if on else ""
        _route(main_window, int(setup_id), on, msg)


def _alert_changed(owner, setup_id, on, msg) -> bool:
    """True only when ``(on, msg)`` differs from what was last applied to
    ``owner`` for ``box_id``: and records the new value.

    The cache lives on the OWNER widget (not the main window): a freshly
    rebuilt surface has no cache, so it's (correctly) styled on first apply
    rather than skipped. If the owner can't hold the attribute, we return
    True so correctness always wins over the optimisation."""
    if owner is None:
        return False
    cache = getattr(owner, "_alert_cache", None)
    if cache is None:
        cache = {}
        try:
            owner._alert_cache = cache
        except (AttributeError, TypeError):
            return True
    if cache.get(setup_id) == (on, msg):
        return False
    cache[setup_id] = (on, msg)
    return True


def _route(mw, setup_id, on, msg):
    # 1. Per-box widget: BoxControlWidget (operant) or SetupWidget (maze).
    #
    # ``_setup_widget_for`` is on MainWindowBase, so both modes have it. This
    # probed ``get_setup_widget``, which only operant defines, and surfaces 2
    # and 3 below are operant-only too, so in maze all three routes missed and
    # per-box error badges never rendered AT ALL, silently, even though
    # SetupWidget.set_error exists and works.
    try:
        getter = getattr(mw, "_setup_widget_for", None)
        w = getter(setup_id) if callable(getter) else None
        if (w is not None and hasattr(w, "set_error")
                and _alert_changed(w, setup_id, on, msg)):
            w.set_error(on, msg)
    except Exception as e:
        logger.debug("box-alert: control widget %s: %s", setup_id, e)

    # 2. Live Status panel (operant: _live_status_by_box keyed by box_number).
    try:
        for lw in (getattr(mw, "_live_status_by_box", None) or {}).values():
            if getattr(lw, "setup_number", None) == setup_id:
                if hasattr(lw, "set_error") and _alert_changed(lw, setup_id, on, msg):
                    lw.set_error(on, msg)
                break
    except Exception as e:
        logger.debug("box-alert: live status %s: %s", setup_id, e)

    # 3. Video tile (operant: _video_holders dict keyed by box_id).
    try:
        holders = getattr(mw, "_video_holders", None)
        vh = holders.get(setup_id) if isinstance(holders, dict) else None
        if (vh is not None and hasattr(vh, "set_error")
                and _alert_changed(vh, setup_id, on, msg)):
            vh.set_error(on, msg)
    except Exception as e:
        logger.debug("box-alert: video tile %s: %s", setup_id, e)

    # 4. Statistics tab (row + plot), only if the surface exposes it.
    try:
        stats = getattr(mw, "statisticsTab", None)
        if (stats is not None and hasattr(stats, "set_box_error")
                and _alert_changed(stats, setup_id, on, msg)):
            stats.set_box_error(setup_id, on, msg)
    except Exception as e:
        logger.debug("box-alert: stats %s: %s", setup_id, e)
