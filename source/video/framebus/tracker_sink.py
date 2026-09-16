"""TrackerSink, blob (background subtraction) tracker, parallel per-box.

Per-box bounded queue, drop-oldest. Wraps TrackerManager; subscribes to
the FrameBus, hands frames to the tracker per box, computes centroid +
zone + speed, and publishes a result event for RecorderSink and MCUPusher.

Blob trackers run in parallel via a ThreadPoolExecutor (default 4 workers,
fewer on low-CPU hosts like Jetson) so 8-box throughput stays at camera
rate. A per-box in-flight gate keeps only one tracker call per box at a
time while different boxes run concurrently.
"""

from __future__ import annotations

import logging
import os
import queue as _queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

from source.video.framebus.types import BoxFrame
from source.video.tracking import TrackerManager
from source.video.tracking.blob import TRACK_LOST
from source.video.framebus.sink_base import (
    DropPolicy,
    Sink,
    estimate_capture_fps,
    forecast_point,
    notify_subscribers,
    track_speed,
    zone_occupancy_at,
)
from source.video.framebus.latency import LatencyBudget

logger = logging.getLogger(__name__)


# Result callback: (box_id, cam_frame_id, centroid (x, y), location, speed,
#                   zones_by_body_part, raw_position (x, y, w, h),
#                   *, capture_host_ns)
TrackerResultCallback = Callable[
    [int, int, Tuple[float, float], Optional[str], float, dict, tuple], None
]


class TrackerSink(Sink):
    """Blob tracker sink. One instance, multi-box, parallel across boxes.

    Per-box state (TrackerManager already owns trackers per box; we add
    only the prev-centroid for speed and the result subscriber list).
    """

    name = "tracker"

    def __init__(self, *,
                 tracker_manager: Optional[TrackerManager] = None,
                 queue_per_box_seconds: float = 1.0,
                 fps_hint: float = 30.0,
                 max_parallel: Optional[int] = None) -> None:
        """
        max_parallel: max concurrent blob detections. ``None`` sizes it from
            the machine, which is the only thing that scales: detection costs
            ~10-19 ms per box, so sixteen boxes need ~160-310 ms of CPU per
            frame-set and the pool width IS the frame rate. A fixed 4 capped a
            16-box rig at 12.9 fps while leaving an 8-core machine 60 % idle,
            the box count grew and this number did not. The inflight gate
            still keeps one box from running twice concurrently.
        """
        if max_parallel is None:
            max_parallel = max(2, (os.cpu_count() or 4) - 2)
        # ONE SECOND, whatever the rate. ``fps_hint`` defaults to 30 and
        # nothing has ever passed it, so this queue held one second at 30 fps
        # and half a second at 60 regardless of the camera. It is still the
        # starting depth; ``set_rate_hint`` corrects it once the pipeline
        # knows the real rate.
        maxsize = max(15, int(queue_per_box_seconds * fps_hint))
        # ONE dispatcher thread on purpose: this sink already runs detections
        # in parallel on its own ``_pool`` (``max_parallel`` wide), and the
        # dispatcher only pulls a frame and submits it. A second dispatcher
        # would contend for the same inflight gate without adding throughput.
        super().__init__(name="tracker", maxsize=maxsize,
                         seconds=float(queue_per_box_seconds),
                         policy=DropPolicy.DROP_OLDEST, workers=1)
        self._tm: TrackerManager = tracker_manager or TrackerManager()
        self._enabled: Dict[int, bool] = {}
        self._prev_centroid: Dict[int, Tuple[float, float, int]] = {}
        # Per-box previous capture monotonic_ns, used to estimate
        # instantaneous fps for the latency-compensation forecast.
        self._prev_capture_ns: Dict[int, int] = {}
        self._on_result: List[TrackerResultCallback] = []
        self._on_state: List[Callable] = []
        # Observational timing spine (set by the Pipeline); None = off.
        self._latency: Optional[LatencyBudget] = None
        # RLock overrides the base Sink._lock; guards the per-box state
        # collections and lets sub-helpers re-acquire safely.
        self._lock = threading.RLock()
        # Per-box inflight gate: at most one tracker call per box.
        self._inflight: Dict[int, Future] = {}
        self._inflight_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(max_parallel)),
            thread_name_prefix="tracker-pool",
        )

    @property
    def manager(self) -> TrackerManager:
        return self._tm

    def set_latency_budget(self, budget: Optional[LatencyBudget]) -> None:
        """Attach the pipeline's timing spine (observational)."""
        self._latency = budget

    # ── Public API ----------------------------------------------------

    def enable_for_box(self, setup_id: int) -> None:
        with self._lock:
            self._enabled[setup_id] = True
        self._tm.create_tracker(setup_id)

    def disable_for_box(self, setup_id: int) -> None:
        # Clear ALL per-box state, mirroring PoseSink.disable_for_box,
        # a stale _prev_capture_ns skews the first dt after re-enable, and
        # stale queued frames would be processed against the new session.
        with self._lock:
            self._enabled.pop(setup_id, None)
            self._prev_centroid.pop(setup_id, None)
            self._prev_capture_ns.pop(setup_id, None)
            dq = self._per_box.get(setup_id)
            if dq is not None:
                dq.clear()
        with self._inflight_lock:
            self._inflight.pop(setup_id, None)
        self._tm.stop_tracker(setup_id)

    def on_track_state(self, cb) -> Callable[[], None]:
        """Subscribe to ``(setup_id, cam_frame_id, state, capture_host_ns)``.

        Separate from ``on_result`` on purpose: a result carries a position
        and a state carries whether there IS one, so a lost frame has a state
        to report and no result to report it with.
        """
        with self._lock:
            self._on_state = [*self._on_state, cb]

        def _unsub_state():
            with self._lock:
                self._on_state = [c for c in self._on_state if c is not cb]
        return _unsub_state

    def _emit_track_state(self, frame) -> None:
        """Publish this box's track state for this frame."""
        try:
            tracker = self._tm.get_tracker(frame.setup_id)
            state = getattr(tracker, "track_state", TRACK_LOST) if tracker \
                else TRACK_LOST
        except Exception:
            state = TRACK_LOST
        with self._lock:
            subs = self._on_state
        if not subs:
            return
        notify_subscribers(subs, frame.setup_id, frame.cam_frame_id, state,
                           frame.capture_host_ns, log=logger,
                           label=f"TrackerSink state (box={frame.setup_id})")

    def on_result(self, cb: TrackerResultCallback) -> Callable[[], None]:
        with self._lock:
            self._on_result = [*self._on_result, cb]

        def _unsub():
            with self._lock:
                self._on_result = [c for c in self._on_result if c is not cb]
        return _unsub

    # ── Sink worker --------------------------------------------------
    #
    # _run is overridden so different boxes run in parallel on the pool.
    # ``_inflight`` keeps one tracker call per box at a time, a box's
    # per-box state (bg subtractor, prev position, Kalman) would corrupt
    # under concurrent updates; different boxes are independent.

    def _run(self, widx: int = 0) -> None:
        while self._running:
            try:
                bid = self._wake.get(timeout=0.5)
            except _queue.Empty:
                continue
            if bid is None:
                break

            # Per-box inflight gate: skip if this box is already running.
            # The on_done callback re-pushes the wake when the in-flight
            # call finishes, so the latest queued frame still gets
            # processed (no starvation).
            with self._inflight_lock:
                cur = self._inflight.get(bid)
                if cur is not None and not cur.done():
                    continue

            # Pull the box's latest frame (DROP_OLDEST keeps deque depth
            # ≤ maxsize; pop the oldest remaining).
            frame = None
            with self._lock:
                dq = self._per_box.get(bid)
                if dq:
                    frame = dq.popleft()
                    if not dq:
                        self._pending.discard(bid)
            if frame is None:
                continue

            # Submit to the parallel pool.
            try:
                future = self._pool.submit(self._run_one, frame)
            except RuntimeError:
                # Pool shut down, bail.
                return
            with self._inflight_lock:
                self._inflight[bid] = future
            future.add_done_callback(self._make_done_cb(bid))

    def _make_done_cb(self, bid: int):
        def _cb(_future):
            with self._inflight_lock:
                self._inflight.pop(bid, None)
            # If new frames arrived while this box was busy, re-wake.
            with self._lock:
                dq = self._per_box.get(bid)
                still = bool(dq)
                if still and bid not in self._pending:
                    self._pending.add(bid)
            if still:
                try:
                    self._wake.put_nowait(bid)
                except _queue.Full:
                    pass
        return _cb

    def _run_one(self, frame: BoxFrame) -> None:
        # Counted HERE, not in Sink._run: this sink overrides the worker loop
        # to dispatch onto its own pool, so the base-class counter never sees
        # its frames and the diagnostic reported "tracked 0.0" on a pipeline
        # that was tracking perfectly well.
        self.n_processed += 1
        try:
            self.process(frame)
            # Only when this box actually TRACKED. ``process`` returns
            # immediately for a box with no tracker enabled, and recording
            # that pass-through as an inference sample buries the real
            # number: pose and blob share this stage, so on a pose-only rig
            # the ring filled with the tracker's near-zero returns and a
            # 30 ms model was reported as 0.3 ms. Measured on the end-to-end
            # pose test, which is what found it.
            if (self._latency is not None
                    and self._enabled.get(frame.setup_id, False)):
                self._latency.record_from_ns(
                    "poll_to_infer", frame.poll_host_ns)
        except Exception as e:
            logger.exception("TrackerSink _run_one error (box=%s): %s",
                             frame.setup_id, e)

    def stop(self, timeout: float = 2.0) -> None:
        # Stop the base worker first (sentinel + join).
        super().stop(timeout=timeout)
        # Then shut down the parallel pool.
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def process(self, frame: BoxFrame) -> None:
        if not self._enabled.get(frame.setup_id, False):
            return

        # Pre-converted gray view: whole-frame BGR→GRAY computed once on
        # the parent CameraFrame, sliced here at O(1) for CCTV-shared boxes.
        gray = frame.image_gray
        try:
            success, position = self._tm.update_tracker(
                frame.setup_id, frame.image, frame.capture_host_ns,
                frame_gray=gray)
        except Exception as e:
            logger.error("TrackerSink: update_tracker error (box=%d): %s",
                         frame.setup_id, e)
            return

        # Auto-init the tracker if it has a background but isn't initialised.
        # self_norm needs no background, so it must also auto-init here, else
        # a self_norm box never initialises through the sink and stays blind.
        if not success and position is None:
            tracker = self._tm.get_tracker(frame.setup_id)
            if (tracker is not None and not tracker.is_initialized
                    and (tracker.background is not None
                         or getattr(tracker, "bg_mode", "") == "self_norm")):
                try:
                    if tracker.auto_initialize(frame.image):
                        success, position = self._tm.update_tracker(
                            frame.setup_id, frame.image, frame.capture_host_ns,
                            frame_gray=gray)
                except Exception as e:
                    logger.warning(
                        "TrackerSink: auto-init failed (box=%d): %s",
                        frame.setup_id, e)

        if not success or position is None:
            # Say the miss, do not just return. Returning silently leaves the
            # overlay on its last position until a TTL expires, the MCU on its
            # last coordinate, and the tracking log without a row, from which
            # the analysis reads a confident straight line across an
            # occlusion.
            self._emit_track_state(frame)
            return

        x, y, w, h = position[:4]
        # Prefer the moments centroid the tracker already computed (and KF-
        # smoothed when an enhancer is attached) over the bbox center. The
        # bbox center drifts from the true body center exactly when the animal
        # rears / grooms / extends its tail, which skews zone occupancy and the
        # MCU coordinate push. Fall back to the bbox center if unavailable.
        cx, cy = x + w / 2.0, y + h / 2.0
        try:
            tracker = self._tm.get_tracker(frame.setup_id)
            lc = getattr(tracker, "last_centroid", None) if tracker is not None else None
            if lc is not None:
                cx, cy = float(lc[0]), float(lc[1])
        except Exception:
            pass

        # ── Latency compensation ─────────────────────────────────────
        # With smooth_tracking on, forecast 1-2 frames ahead via the KF
        # posterior so the zone lookup (→ MCU push) reflects where the
        # mouse will be by the time the hardware reacts. ``zone_cx/cy``
        # drives zone occupancy; reported ``cx/cy`` stays at the current
        # smoothed position for display + speed.
        zone_cx, zone_cy = cx, cy
        try:
            enhancer = (self._tm.get_enhancer(frame.setup_id)
                        if hasattr(self._tm, "get_enhancer") else None)
            if enhancer is not None and getattr(enhancer, "_initialized", False):
                with self._lock:
                    prev_ts = self._prev_capture_ns.get(frame.setup_id)
                    self._prev_capture_ns[frame.setup_id] = frame.capture_host_ns
                fps = estimate_capture_fps(prev_ts, frame.capture_host_ns)
                fwd = forecast_point(enhancer, self._latency, fps)
                if fwd is not None:
                    zone_cx, zone_cy = fwd
        except Exception as e:
            logger.debug("TrackerSink: lookahead error (box=%d): %s",
                         frame.setup_id, e)

        # Zone lookup uses forecasted position (drives MCU push).
        # Centroid-only, innermost-wins (deepest), the SAME semantics as
        # PoseSink and the documented zone contract ("a query returns the
        # innermost containing zone"). First-zone-wins here made the
        # _video_data.txt location column depend on which tracker ran.
        location: Optional[str] = None
        zones_by_body_part: dict = {}
        try:
            zm = self._tm.get_zone_manager(frame.setup_id) \
                if hasattr(self._tm, "get_zone_manager") else None
            if zm is not None and hasattr(zm, "get_zones_at_point"):
                occ, loc = zone_occupancy_at(zm, zone_cx, zone_cy,
                                             deepest=True)
                if loc:
                    location = loc
                zones_by_body_part = {"centroid": occ}
        except Exception as e:
            logger.debug("TrackerSink: zone lookup error (box=%d): %s",
                         frame.setup_id, e)

        # Speed, shared helper with PoseSink.
        with self._lock:
            subs = self._on_result
        speed = track_speed(self._lock, self._prev_centroid,
                            frame.setup_id, cx, cy)

        # Notify. The state goes out alongside every result, not only on a
        # miss, so a consumer can tell a measured position from a coasted one
        # without having to infer it from gaps.
        self._emit_track_state(frame)
        notify_subscribers(subs, frame.setup_id, frame.cam_frame_id, (cx, cy),
                           location, speed, zones_by_body_part, position,
                           capture_host_ns=frame.capture_host_ns,
                           log=logger,
                           label=f"TrackerSink result (box={frame.setup_id})")
