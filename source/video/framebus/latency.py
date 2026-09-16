"""Per-stage latency accounting for the video → MCU pipeline.

Qt-free, thread-safe, dependency-free. Each named stage keeps a fixed-size
ring of recent samples in milliseconds; the budget reports p50/p95/mean per
stage and ``total_ms()`` = the sum of per-stage p95, the honest
capture→MCU delay that the latency HUD displays and the predict-ahead
horizon compensates for.

Observational only: recording a sample changes no pipeline behaviour, so
the budget can run live and be read (or ignored) without affecting timing.

Stages, capture → MCU (order matters for ``total_ms``):
  * ``capture_to_poll``  frame capture → tick-thread drain
  * ``poll_to_infer``    drain → pose/track inference finished
  * ``infer_to_push``    result → coord/event queued to the serial writer
  * ``push_to_wire``     queued → drained onto the serial port

A stage with no samples contributes 0 to ``total_ms`` and is reported with
``count == 0`` so a partially-wired pipeline is visible rather than silently
optimistic.
"""

from __future__ import annotations

import threading
from source import host_clock
from collections import deque
from typing import Dict, Iterable, List, Optional

# Canonical stage order, capture → MCU.
STAGES = ("capture_to_poll", "poll_to_infer", "infer_to_push", "push_to_wire")

# Reject absurd samples (clock glitch, paused thread, a stale frame whose
# start timestamp is from a previous run). Anything over this is noise.
_MAX_SAMPLE_MS = 10_000.0


def _percentile(sorted_vals: List[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in [0, 100]) of a sorted list.

    Matches numpy's default ('linear') method so unit expectations line up
    with the obvious hand-calculation. Empty → 0.0.
    """
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    rank = (q / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


class LatencyBudget:
    """Rolling per-stage latency samples with p50/p95/mean + a total.

    Thread-safe: appends and reads take a short lock. ``window`` is the
    per-stage ring depth (older samples fall off), so the stats track the
    recent steady state rather than the whole session.
    """

    def __init__(self, window: int = 240,
                 stages: Iterable[str] = STAGES) -> None:
        self._window = max(1, int(window))
        self._stages = tuple(stages)
        self._rings: Dict[str, deque] = {
            s: deque(maxlen=self._window) for s in self._stages
        }
        self._lock = threading.Lock()

    # ── Recording ------------------------------------------------------

    def record(self, stage: str, ms: float) -> None:
        """Record one duration sample (milliseconds) for ``stage``.

        Unknown stages, negatives, NaNs and absurd values are dropped so a
        wiring mistake can never poison the stats.
        """
        ring = self._rings.get(stage)
        if ring is None:
            return
        # NaN-safe: (ms != ms) is True only for NaN.
        if ms != ms or ms < 0.0 or ms > _MAX_SAMPLE_MS:
            return
        with self._lock:
            ring.append(float(ms))

    def record_from_ns(self, stage: str, start_ns: Optional[int],
                       end_ns: Optional[int] = None) -> None:
        """Record ``(end_ns - start_ns)`` for ``stage``, in ms.

        ``start_ns``/``end_ns`` are ``host_clock.host_ns()`` readings;
        ``end_ns`` defaults to now. A missing/zero ``start_ns`` is a no-op
        (the frame never got stamped for this stage).
        """
        if not start_ns:
            return
        if end_ns is None:
            end_ns = host_clock.host_ns()
        self.record(stage, (end_ns - start_ns) / 1e6)

    # ── Reading --------------------------------------------------------

    def stage_stats(self, stage: str) -> Dict[str, float]:
        """``{p50, p95, p99, mean, count}`` for one stage (ms). Empty → zeros.

        p99 is here because a closed loop is judged by its worst case, not its
        average: a stage whose mean is half the frame budget and whose p99 is
        over it misses deadlines, and the mean cannot say so.
        """
        empty = {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "count": 0}
        ring = self._rings.get(stage)
        if ring is None:
            return dict(empty)
        with self._lock:
            vals = list(ring)
        if not vals:
            return dict(empty)
        sv = sorted(vals)
        return {
            "p50": _percentile(sv, 50.0),
            "p95": _percentile(sv, 95.0),
            "p99": _percentile(sv, 99.0),
            "mean": sum(sv) / len(sv),
            "count": len(sv),
        }

    def deadline_misses(self, stage: str, budget_ms: float) -> Dict[str, float]:
        """How often ``stage`` ran over ``budget_ms``, and by how much.

        The count is what an operator needs: "it is usually fine" is not an
        answer when a trigger has to fire before the next frame arrives.
        """
        ring = self._rings.get(stage)
        if ring is None or budget_ms <= 0:
            return {"misses": 0, "count": 0, "fraction": 0.0, "worst_ms": 0.0}
        with self._lock:
            vals = list(ring)
        if not vals:
            return {"misses": 0, "count": 0, "fraction": 0.0, "worst_ms": 0.0}
        over = [v for v in vals if v > budget_ms]
        return {"misses": len(over), "count": len(vals),
                "fraction": len(over) / len(vals), "worst_ms": max(vals)}

    def total_ms(self) -> float:
        """Sum of per-stage p95, the end-to-end capture→MCU budget.

        Stages with no samples contribute 0, so this is the honest total
        over whatever stages are currently wired/observed.
        """
        return sum(self.stage_stats(s)["p95"] for s in self._stages)

    def predict_horizon_ms(self, fallback_ms: float,
                           cap_ms: float = 75.0) -> float:
        """Forecast horizon for latency compensation.

        The Kalman predict-ahead should extrapolate by the delay the frame
        actually incurs, capture→MCU, not a rate-derived guess. Returns
        the measured ``total_ms()`` once stages are populated, else
        ``fallback_ms`` (the FPS-based estimate) until then. Capped at
        ``cap_ms`` so a stale/huge reading can't fling the forecast (the
        constant-velocity KF overshoots direction changes past ~75 ms).
        """
        measured = self.total_ms()
        horizon = measured if measured > 0.0 else fallback_ms
        return max(0.0, min(cap_ms, horizon))

    def snapshot(self) -> Dict[str, object]:
        """Full read: per-stage stats + ``total_ms``. Cheap; safe to poll
        from the GUI thread at ~1 Hz."""
        stages = {s: self.stage_stats(s) for s in self._stages}
        total = sum(v["p95"] for v in stages.values())
        return {"stages": stages, "total_ms": total}


# ── HUD helpers (pure, so they unit-test without a GUI) ────────────────

_STAGE_LABEL = {
    "capture_to_poll": "capture→poll",
    "poll_to_infer": "poll→infer",
    "infer_to_push": "infer→push",
    "push_to_wire": "push→wire",
}


def format_hud_lines(snapshot: Dict[str, object]) -> List[str]:
    """Tooltip lines for a ``LatencyBudget.snapshot()``, p50/p95 per wired
    stage plus the capture→MCU total. Returns ``[]`` when nothing has been
    observed yet (so the HUD shows latency only once it's real).
    """
    stages = snapshot.get("stages", {}) or {}
    if not any(s.get("count") for s in stages.values()):
        return []
    lines = ["latency (pipeline, p50/p95 ms):"]
    for key in STAGES:
        s = stages.get(key, {})
        if s.get("count"):
            lines.append(
                f"  {_STAGE_LABEL.get(key, key):<12} "
                f"{s['p50']:.1f} / {s['p95']:.1f}")
    lines.append(f"  {'total (→MCU)':<12} {float(snapshot.get('total_ms', 0.0)):.1f}")
    return lines


def low_delivery(runtime_fps: float, target_fps: float,
                 ratio: float = 0.8) -> bool:
    """True when live delivery is meaningfully below the configured target
    (e.g. a UVC camera stuck in YUY2 at ~12 of 30 fps). Unknown/zero rates
    return False, no rate is not a slow rate.
    """
    if target_fps <= 0 or runtime_fps <= 0:
        return False
    return runtime_fps < ratio * target_fps
