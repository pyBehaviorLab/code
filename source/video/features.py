"""Per-box kinematic / posture features from 2D pose keypoints.

The efficient online pattern (SimBA / B-SOiD / DeepLabCut kinematics): a small
rolling deque of recent keypoints per box, **smooth-then-difference** (never
differentiate raw coordinates), EMA for scalars, a running-median **body
length L** as the scale reference, and normalise distances by L (→
body-lengths) or by a px↔mm scale (→ real units) so thresholds port across
rigs. Angles use ``atan2(cross, dot)`` (never ``acos``, NaN near 0/π) and
angle differences use ``atan2(sinΔ, cosΔ)`` (streaming unwrap).

Headless (NumPy only), so it is unit-tested without a camera. The MCU push
policy owns one per box, calls :meth:`update` each pose frame, then reads the
feature properties in its condition evaluation.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Optional, Tuple

XY = Tuple[float, float]


def _angle_between(ux: float, uy: float, wx: float, wy: float) -> float:
    """Signed angle (radians, -pi..pi) from vector u to vector w, robustly."""
    cross = ux * wy - uy * wx
    dot = ux * wx + uy * wy
    return math.atan2(cross, dot)


def _wrap_delta(a: float, b: float) -> float:
    """Shortest signed difference a-b, wrapped to (-pi, pi] (streaming unwrap)."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


class FeatureExtractor:
    """Rolling per-box features. All distances are in pixels internally;
    :meth:`speed_in` / :meth:`distance_in` convert to real units when a scale
    (px per mm) is set, else to body-lengths, else raw px.

    ``axis`` names the (tail, head) keypoints that define the body/heading
    vector (default matches the app's rotation pair). ``conf`` gates keypoints:
    a point below it is treated as missing and held.
    """

    def __init__(self, window: int = 20, ema_alpha: float = 0.4,
                 axis: Tuple[str, str] = ("tailbase", "nose"),
                 px_per_mm: Optional[float] = None):
        self.window = int(window)
        self.ema_alpha = float(ema_alpha)
        self.axis = axis
        self.px_per_mm = px_per_mm
        # (t_s, {part: (x,y)}, centroid) history
        self._buf: Deque[Tuple[float, Dict[str, XY], Optional[XY]]] = deque(maxlen=window)
        self._L_samples: Deque[float] = deque(maxlen=200)  # body-length medians
        self._ema_speed: Optional[float] = None
        self._speed_px_s: float = 0.0

    # ── ingest ───────────────────────────────────────────────────────────

    def update(self, keypoints: Dict[str, XY], centroid: Optional[XY],
               speed_px_s: float, t_s: float) -> None:
        """Append one frame. ``keypoints`` = {part: (x,y)} of confident points
        (already confidence-gated by the caller); ``speed_px_s`` = the
        centroid speed the pipeline already computed; ``t_s`` = seconds."""
        self._buf.append((float(t_s), dict(keypoints or {}), centroid))
        # running-median body length from the configured axis
        tail, head = self.axis
        a, b = (keypoints or {}).get(tail), (keypoints or {}).get(head)
        if a is not None and b is not None:
            self._L_samples.append(math.hypot(a[0] - b[0], a[1] - b[1]))
        # EMA-smoothed speed (smooth-then-threshold)
        self._speed_px_s = float(speed_px_s)
        self._ema_speed = (speed_px_s if self._ema_speed is None
                           else self.ema_alpha * speed_px_s
                           + (1 - self.ema_alpha) * self._ema_speed)

    # ── scale ────────────────────────────────────────────────────────────

    def body_length_px(self) -> Optional[float]:
        """Running median of the body-axis length, or None if never seen."""
        if not self._L_samples:
            return None
        s = sorted(self._L_samples)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    def _unit_divisor(self, unit: str) -> float:
        """px per output unit: 'mm'/'cm' via scale, 'bodylen' via median L,
        'px' = 1. Falls back sensibly when the preferred basis is absent."""
        if unit in ("mm", "cm") and self.px_per_mm:
            return self.px_per_mm * (10.0 if unit == "cm" else 1.0)
        if unit == "bodylen":
            L = self.body_length_px()
            return L if L else 1.0
        return 1.0

    def speed_in(self, unit: str = "px") -> float:
        """EMA-smoothed speed in the requested unit per second."""
        v = self._ema_speed if self._ema_speed is not None else self._speed_px_s
        return float(v) / self._unit_divisor(unit)

    def distance_in(self, part_a: str, part_b: str, unit: str = "px") -> Optional[float]:
        """Distance between two current keypoints in ``unit``, or None if
        either is missing this frame."""
        if not self._buf:
            return None
        kps = self._buf[-1][1]
        a, b = kps.get(part_a), kps.get(part_b)
        if a is None or b is None:
            return None
        return math.hypot(a[0] - b[0], a[1] - b[1]) / self._unit_divisor(unit)

    # ── angles ───────────────────────────────────────────────────────────

    def _heading(self, kps: Dict[str, XY]) -> Optional[Tuple[float, float]]:
        tail, head = self.axis
        a, b = kps.get(tail), kps.get(head)
        if a is None or b is None:
            return None
        vx, vy = b[0] - a[0], b[1] - a[1]
        if math.hypot(vx, vy) < 1e-6:
            return None
        return (vx, vy)

    def heading_deg(self) -> Optional[float]:
        if not self._buf:
            return None
        h = self._heading(self._buf[-1][1])
        return math.degrees(math.atan2(h[1], h[0])) % 360.0 if h else None

    def turning_deg_s(self) -> Optional[float]:
        """Angular velocity of the heading (deg/s), unwrapped; None until two
        frames with a valid axis exist."""
        if len(self._buf) < 2:
            return None
        (t1, k1, _), (t0, k0, _) = self._buf[-1], self._buf[-2]
        h1, h0 = self._heading(k1), self._heading(k0)
        if h1 is None or h0 is None or t1 <= t0:
            return None
        a1 = math.atan2(h1[1], h1[0])
        a0 = math.atan2(h0[1], h0[0])
        return math.degrees(_wrap_delta(a1, a0)) / (t1 - t0)

    def head_body_angle_deg(self, neck: str) -> Optional[float]:
        """Signed angle (deg) of the head vector (neck→head) relative to the
        body vector (tail→neck). Needs tail, neck, head keypoints."""
        if not self._buf:
            return None
        kps = self._buf[-1][1]
        tail, head = self.axis
        A, B, C = kps.get(tail), kps.get(neck), kps.get(head)
        if A is None or B is None or C is None:
            return None
        return math.degrees(_angle_between(B[0] - A[0], B[1] - A[1],
                                           C[0] - B[0], C[1] - B[1]))

    def facing_deg(self, target: XY) -> Optional[float]:
        """Signed angle (deg) between the heading and the direction to
        ``target`` from the head keypoint. |value| small = facing it."""
        if not self._buf:
            return None
        kps = self._buf[-1][1]
        h = self._heading(kps)
        head = kps.get(self.axis[1])
        if h is None or head is None:
            return None
        dx, dy = target[0] - head[0], target[1] - head[1]
        if math.hypot(dx, dy) < 1e-6:
            return None
        return math.degrees(_angle_between(h[0], h[1], dx, dy))

    def elongation(self) -> Optional[float]:
        """Nose↔tail distance / running-median L. ~1 normal, <~0.6 compact
        (foreshortened, a rearing proxy), >~1.1 stretched."""
        if not self._buf:
            return None
        L = self.body_length_px()
        if not L:
            return None
        # current axis length (not the median)
        kps = self._buf[-1][1]
        a, b = kps.get(self.axis[0]), kps.get(self.axis[1])
        if a is None or b is None:
            return None
        return math.hypot(a[0] - b[0], a[1] - b[1]) / L
