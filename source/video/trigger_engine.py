"""Trigger visualisation shapes, annotation + live-plot data, no evaluator.

The trigger *evaluation* lives in the MCU push policy
(:class:`~source.video.framebus.mcu_pusher.TrackingPushPolicy`), the single
source of truth that fires the MCU events. This module holds only the
lightweight per-frame state the policy emits and the code that *shows* it:

* :class:`RuleState` / :class:`TriggerFrame`: one rule's outcome for one frame
  and a box's frame of them (the transport from the policy to the GUI);
* :class:`TriggerHistory`: a scrolling per-rule buffer for the Session-Plot
  Triggers lane;
* :func:`draw_triggers`: chips + geometry annotation onto a display frame.

Pure Python / OpenCV, no Qt, unit-tested without a camera.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

Geom = Optional[Dict[str, Any]]


# ── per-frame state (policy → GUI) ─────────────────────────────────────────

@dataclass
class RuleState:
    """One trigger's outcome for one frame, read by the MCU (fired), the
    overlay (active + geom) and the plot (value)."""
    id: str
    name: str
    active: bool
    fired: bool
    value: Optional[float] = None
    geom: Geom = None
    color: str = "#39c5cf"
    show_on_video: bool = True
    highlight_geometry: bool = True
    plot: bool = True
    threshold: Optional[float] = None


@dataclass
class TriggerFrame:
    setup_id: int
    cam_frame_id: int
    states: List[RuleState] = field(default_factory=list)


# ── live history (Session-Plot Triggers lane) ──────────────────────────────

class TriggerHistory:
    """A scrolling per-rule record of ``(t, active, value)`` for one box.
    Headless + Qt-free so the buffering + segment logic is unit-tested; the
    plot lane just reads it. ``t`` is seconds on the caller's clock."""

    def __init__(self, window_s: float = 10.0):
        self.window_s = float(window_s)
        self._series: Dict[str, Deque[Tuple[float, bool, Optional[float]]]] = {}
        self._meta: Dict[str, Tuple[str, str, bool, Optional[float]]] = {}

    def push(self, tf: "TriggerFrame", t_s: float) -> None:
        for s in tf.states:
            dq = self._series.setdefault(s.id, deque())
            dq.append((float(t_s), bool(s.active), s.value))
            self._meta[s.id] = (s.name or s.id, s.color, s.value is not None,
                                getattr(s, "threshold", None))
        self.prune(t_s)

    def prune(self, now_s: float) -> None:
        lo = now_s - self.window_s
        for dq in self._series.values():
            while dq and dq[0][0] < lo:
                dq.popleft()

    def rule_ids(self) -> List[str]:
        return list(self._series.keys())

    def meta(self, rule_id: str) -> Tuple[str, str, bool, Optional[float]]:
        return self._meta.get(rule_id, (rule_id, "#39c5cf", False, None))

    def series(self, rule_id: str) -> List[Tuple[float, bool, Optional[float]]]:
        return list(self._series.get(rule_id, ()))

    def active_segments(self, rule_id: str,
                        now_s: Optional[float] = None) -> List[Tuple[float, float]]:
        """``[(t_start, t_end), …]`` intervals where the rule was active, the
        raster rows. An interval still active at the last sample extends to
        ``now_s``."""
        pts = self.series(rule_id)
        segs: List[Tuple[float, float]] = []
        start: Optional[float] = None
        for (t, active, _v) in pts:
            if active and start is None:
                start = t
            elif not active and start is not None:
                segs.append((start, t))
                start = None
        if start is not None:
            end = now_s if now_s is not None else (pts[-1][0] if pts else start)
            segs.append((start, max(end, start)))
        return segs


# ── annotation, draw trigger state onto a display frame ───────────────────

def _hex_to_bgr(h: str) -> Tuple[int, int, int]:
    h = (h or "#39c5cf").lstrip("#")
    if len(h) != 6:
        return (207, 197, 57)  # default cyan in BGR
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def draw_triggers(frame, states, *, chip_corner: str = "top_right",
                  flash_on_fire: bool = True,
                  scale_x: float = 1.0, scale_y: float = 1.0,
                  off_x: float = 0.0, off_y: float = 0.0) -> None:
    """Draw trigger annotation onto ``frame`` in place (BGR, H×W×3): a stacked
    chip per rule (dim idle → colour active → white ring on fire) + geometry
    highlight (region circle / distance line / zone) when active. Geometry is in
    source-pixel coords; ``scale_*``/``off_*`` map it to the display frame.
    Pure OpenCV; a single bad state never aborts the rest."""
    import cv2
    if frame is None or getattr(frame, "ndim", 0) != 3 or not states:
        return
    h, w = frame.shape[:2]

    def _tx(x, y):
        return (int((x - off_x) * scale_x), int((y - off_y) * scale_y))

    for s in states:
        try:
            if not getattr(s, "highlight_geometry", True) or not s.active or not s.geom:
                continue
            col = _hex_to_bgr(s.color)
            g = s.geom
            if g.get("type") == "circle":
                cv2.circle(frame, _tx(g["x"], g["y"]),
                           max(1, int(g["r"] * scale_x)), col, 2)
            elif g.get("type") == "line" and g.get("a") and g.get("b"):
                a, b = g["a"], g["b"]
                cv2.line(frame, _tx(a[0], a[1]), _tx(b[0], b[1]), col, 2)
        except Exception:
            continue

    chips = [s for s in states if getattr(s, "show_on_video", True)]
    if not chips:
        return
    ch, pad, gap = 18, 8, 6
    top = "top" in chip_corner
    right = "right" in chip_corner
    y = pad if top else h - pad - (ch + gap) * len(chips)
    for s in chips:
        col = _hex_to_bgr(s.color) if s.active else (90, 90, 90)
        label = f"{s.name or s.id}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        bw = tw + 26
        x = (w - pad - bw) if right else pad
        cv2.rectangle(frame, (x, y), (x + bw, y + ch), col, -1)
        cv2.circle(frame, (x + 11, y + ch // 2), 5,
                   (255, 255, 255) if s.active else (150, 150, 150), -1)
        if s.fired and flash_on_fire:
            cv2.rectangle(frame, (x - 2, y - 2), (x + bw + 2, y + ch + 2),
                          (255, 255, 255), 2)
        cv2.putText(frame, label, (x + 20, y + ch - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
        y += ch + gap
