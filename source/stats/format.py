"""Stats-config format adapter, new flat schema ↔ legacy nested schema.

Two on-disk schemas are accepted; the canvas always sees the legacy
shape internally. The adapter is a thin one-way translator.

NEW (flat, one entry per metric), what users edit::

    {
      "task": "5-CSRTT",
      "metrics": [
        { "name": "accuracy",
          "track": "Correct_response,Incorrect_response",
          "widget": "ring", "color": "#2ecc71" },
        { "name": "omissions",
          "track": "Omission",
          "widget": "bar" }
      ],
      "table": ["box", "subject", "accuracy", "omissions"],
      "trend": ["accuracy"],
      "update_interval_ms": 1000
    }

LEGACY (4 parallel sections), what ``BoxStatisticsCalculator`` consumes::

    {
      "counters":     { "<tracked_name>": {"type": "print", "match": "<tracked_name>"} },
      "calculations": { "<metric>": "<formula>" },
      "columns":      [ {"key": ..., "label": ..., "format": ...} ],
      "plots":        [ {"type": ..., "metric": ..., "title": ..., "color": ...} ],
      "update_interval_ms": 1000
    }

If a config has no ``metrics`` key it's assumed to already be the
legacy shape and passed through unchanged. This keeps any
hand-written legacy file working, and once the bundled templates
are rewritten in the new shape, both formats coexist forever (no
schema_version branching, no migrator).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def is_new_format(cfg: Dict[str, Any]) -> bool:
    """True iff this looks like the new flat schema.

    The presence of a ``metrics`` *list* (not the legacy ``counters``
    dict) is the discriminator.
    """
    return isinstance(cfg, dict) and isinstance(cfg.get("metrics"), list)


def to_canvas_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a flat-schema config into the legacy 4-section shape.

    Pass-through when ``cfg`` is already legacy. Idempotent.
    """
    if not is_new_format(cfg):
        return cfg

    counters:     Dict[str, Dict[str, Any]] = {}
    calculations: Dict[str, str]            = {}
    columns:      List[Dict[str, str]]      = []
    plots:        List[Dict[str, Any]]      = []

    metrics: List[Dict[str, Any]] = cfg.get("metrics", []) or []

    # Build counters first, one per distinct tracked name across all
    # metrics. Three flavours, in priority order:
    #
    #   ``track``, simple print/event name match (the common case).
    #                     Comma-separated → one counter per name.
    #   ``track_regex``, regex pattern. ``agg`` controls aggregation
    #                     ("count" default, "sum", "last", "append").
    #                     Counter name = metric ``name``.
    #   ``track_event``, like track but matches MCU events not prints.
    seen_tracks: set[str] = set()
    for m in metrics:
        name = m.get("name", "")
        # Sequence metric: ordered categorical tokens (e.g. per-trial C/F).
        # ``sequence`` maps {token: print-prefix}; the canvas appends the
        # token each time a matching print arrives and fans the list into
        # per-trial columns. No count/regex counter, its own type.
        if m.get("sequence"):
            counters[name] = {"type": "sequence", "map": dict(m["sequence"])}
            seen_tracks.add(name)
            continue
        if m.get("track_regex"):
            counters[name] = {
                "type":  "print",
                "regex": m["track_regex"],
                "agg":   m.get("agg", "count"),
            }
            seen_tracks.add(name)
            continue
        if m.get("track_event"):
            for tname in _split_tracks(m.get("track_event")):
                if tname and tname not in seen_tracks:
                    counters[tname] = {"type": "event", "match": tname}
                    seen_tracks.add(tname)
            continue
        for tname in _split_tracks(m.get("track", "")):
            if tname and tname not in seen_tracks:
                counters[tname] = {"type": "print", "match": tname}
                seen_tracks.add(tname)

    # Custom counter overrides: power users can declare ``extra_counters``
    # in the legacy shape; they're merged on top of generated counters.
    for k, v in (cfg.get("extra_counters") or {}).items():
        counters[k] = v
        seen_tracks.add(k)

    # Per-metric: formula + column + plot
    for m in metrics:
        name = m.get("name")
        if not name:
            logger.warning("stats format: metric without name skipped: %r", m)
            continue

        # Sequence metric → one column that the table expands into ``count``
        # per-index cells. No formula, no plot.
        if m.get("sequence"):
            columns.append({
                "key": name,
                "label": m.get("label") or "Trial",
                "sequence": True,
                "count": int(m.get("count", 0) or 0),
                "format": "{}",
            })
            continue

        tracks = _split_tracks(m.get("track", ""))

        # Pick a default formula:
        #   ``formula`` explicit       → use it
        #   ``track_regex``            → the metric NAME is the counter, refer to it directly
        #   ``track`` simple           → _default_formula() on the track names
        if m.get("formula"):
            formula = m["formula"]
        elif m.get("track_regex") or m.get("track_event"):
            formula = name
        else:
            formula = _default_formula(tracks)
        calculations[name] = formula

        label = m.get("label") or name.replace("_", " ").title()
        fmt = m.get("format") or _default_format(m.get("widget"), len(tracks))
        columns.append({"key": name, "label": label, "format": fmt})

        widget = m.get("widget") or _default_widget(len(tracks))
        plot_type = _widget_to_plot_type(widget)
        if plot_type is not None:
            plot = {
                "type":   plot_type,
                # The canvas keys every plot builder on ``name``, omitting
                # it KeyErrors the box into _stats_error_boxes.
                "name":   name,
                "metric": name,
                "title":  m.get("title") or label,
            }
            if "color" in m:
                plot["color"] = m["color"]
            if "range" in m:
                plot["ylim"] = list(m["range"])
            plots.append(plot)

    # The ``table`` shorthand picks which columns to show + their order.
    # Default = box, subject, then every metric in declared order.
    if "table" in cfg and isinstance(cfg["table"], list):
        ordered = _filter_and_order_columns(columns, cfg["table"])
    else:
        # Prepend per-box identifying columns (always shown).
        identity = [
            {"key": "box_id",     "label": "Box",     "format": "{}"},
            {"key": "subject_id", "label": "Subject", "format": "{}"},
        ]
        ordered = identity + columns

    # The ``trend`` shorthand replaces auto-plot-by-widget for line
    # series when the user wants a single combined trend.
    if "trend" in cfg and isinstance(cfg["trend"], list) and cfg["trend"]:
        plots.append({
            "type":    "line",
            "name":    "trend",
            "metrics": list(cfg["trend"]),
            "title":   "Trend",
            "window":  cfg.get("trend_window", 50),
        })

    legacy: Dict[str, Any] = {
        "description":        cfg.get("description", ""),
        "auto_detect_keywords": list(seen_tracks),
        "update_interval_ms": cfg.get("update_interval_ms", 1000),
        "counters":     counters,
        "calculations": calculations,
        "columns":      ordered,
        "plots":        plots,
    }
    if "task" in cfg:
        legacy["task"] = cfg["task"]
    return legacy


# ── helpers ──────────────────────────────────────────────────────────


_TRACK_SPLIT = re.compile(r"[,\s]+")


def _split_tracks(track_value: Any) -> List[str]:
    """Accept ``"a,b,c"``, ``["a","b","c"]``, ``"a b c"``, ``""``, or None."""
    if not track_value:
        return []
    if isinstance(track_value, list):
        return [str(t).strip() for t in track_value if str(t).strip()]
    return [t for t in _TRACK_SPLIT.split(str(track_value)) if t]


def _default_formula(tracks: List[str]) -> str:
    """Auto-formula based on how many tracks the metric declared.

    1 track → raw count.
    2 tracks → ratio% (numerator / sum) * 100, the typical accuracy pattern.
    3+ tracks → sum.
    """
    if not tracks:
        return "0"
    if len(tracks) == 1:
        return tracks[0]
    if len(tracks) == 2:
        a, b = tracks
        return (
            f"({a} / ({a} + {b})) * 100 if ({a} + {b}) > 0 else 0"
        )
    joined = " + ".join(tracks)
    return joined


def _default_widget(n_tracks: int) -> str:
    return "ring" if n_tracks == 2 else "bar"


def _widget_to_plot_type(widget: str) -> str | None:
    """Map widget keyword to the legacy plot ``type``.

    ``table`` and ``text_label`` only contribute to the table, no plot
    entry. Unknown widget → fall back to bar with a warning.
    """
    if widget in ("table", "text_label"):
        return None
    if widget == "ring":
        return "circular"
    if widget in ("line_plot", "line"):
        return "line"
    if widget in ("bar_plot", "bar"):
        return "bar"
    if widget in ("scatter_plot", "scatter"):
        return "scatter"
    if widget == "histogram":
        return "bar"  # histogram approximation; legacy plot path has no histogram
    logger.warning("stats format: unknown widget %r, defaulting to bar", widget)
    return "bar"


def _default_format(widget: str | None, n_tracks: int) -> str:
    """Format string for the table cell.

    ``ring`` widgets get ``{:.1f}`` (percent-like); count-style widgets
    get ``{:.0f}``.
    """
    if widget == "ring" or n_tracks == 2:
        return "{:.1f}"
    return "{:.0f}"


def _filter_and_order_columns(
    columns: List[Dict[str, str]],
    table_keys: List[str],
) -> List[Dict[str, str]]:
    """Reduce ``columns`` to the entries named in ``table_keys`` (in order).

    ``box``/``subject`` shorthand auto-expand to the identity columns.
    """
    by_key = {c["key"]: c for c in columns}
    out: List[Dict[str, str]] = []
    for k in table_keys:
        if k == "box":
            out.append({"key": "box_id",     "label": "Box",     "format": "{}"})
        elif k == "subject":
            out.append({"key": "subject_id", "label": "Subject", "format": "{}"})
        elif k == "task":
            out.append({"key": "task",       "label": "Task",    "format": "{}"})
        elif k == "duration":
            out.append({"key": "duration_min", "label": "Duration (min)",
                        "format": "{:.2f}"})
        elif k in by_key:
            out.append(by_key[k])
        else:
            logger.warning(
                "stats format: table key %r not in metrics, skipped", k)
    return out


__all__ = ["is_new_format", "to_canvas_config"]
