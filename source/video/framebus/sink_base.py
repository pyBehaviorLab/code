"""Sink ABC, per-box bounded queue + worker thread + drop policy.

DeepStream-style fan-out: every consumer (sink) keeps a SEPARATE bounded
queue per box, so a backpressured queue for box X never affects what
boxes Y and Z see.  A single worker thread per sink drains across boxes
in round-robin order via a deduplicated wake queue.

Drop policies (apply WITHIN a single box's queue, never across boxes):

* ``LOSSLESS``       Per-box queue bounded; when full, ``submit`` returns
                     False and the caller emits a backpressure alarm.
                     Used by RecorderSink, losing a recorded frame is
                     a user-visible alarm, never silent.

* ``DROP_OLDEST``    When that box's queue is full, evict the oldest
                     queued frame for THAT box only to make room.  Used
                     by TrackerSink (process recent frames per box; old
                     ones are stale).

* ``DROP_NEWEST``    When that box's queue is full, drop the new
                     arrival; keep what's queued for that box.  Used by
                     PoseSink, once an inference is queued for a box,
                     replacing it with a slightly newer frame for the
                     same box buys nothing while inference is grinding.

Why per-box: a single shared queue lets dict iteration order in
``FrameBus.publish_frame`` decide which box's frame survives, starving the
rest (the last-iterated box wins ~all frames under a slow GUI paint).

The base class owns the worker lifecycle and per-box queueing.
Subclasses implement ``process(frame: BoxFrame)`` only.
"""

from __future__ import annotations

import collections
import enum
import logging
import os
import queue
import threading
from source import host_clock
from typing import Any, Dict, List, Optional, Set, Tuple

from source.video.framebus.types import BoxFrame
from source.video.recording.drop_log import drop_log as _drop_log
from source.video.tracking.speed import compute_speed

logger = logging.getLogger(__name__)


def default_sink_workers(cap: int = 4) -> int:
    """Worker count for a CPU-bound sink.

    Leaves two cores for the camera threads and the GUI, and caps low so a
    many-box rig cannot starve the machine with one sink. cv2 and the encoders
    release the GIL, so these threads scale nearly linearly on real work.
    """
    return max(1, min(cap, (os.cpu_count() or 4) - 2))


class DropPolicy(enum.Enum):
    LOSSLESS = "lossless"
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"


def notify_subscribers(subs, *args, log, label: str, **kwargs) -> None:
    """Invoke each callback with ``args``/``kwargs``, log-and-continue on
    error so one broken subscriber never starves the others. Shared by the
    sinks, FrameBus and the Pipeline fanouts."""
    for cb in subs:
        try:
            cb(*args, **kwargs)
        except Exception as e:
            log.error("%s callback error: %s", label, e)


def estimate_capture_fps(prev_ns: Optional[int], now_ns: int) -> float:
    """Instantaneous capture fps from two consecutive capture timestamps.
    Falls back to 30.0 on the first frame or outside the plausible
    2–200 fps window (dropped-frame gaps, clock hiccups)."""
    if prev_ns is not None and now_ns > prev_ns:
        dt_s = (now_ns - prev_ns) / 1e9
        if 0.005 < dt_s < 0.5:
            return 1.0 / dt_s
    return 30.0


def forecast_point(enhancer, latency, fps: float) -> Optional[Tuple[float, float]]:
    """Latency-compensated forecast of the tracked point.

    Horizon = the MEASURED capture→MCU delay from the timing spine when
    available, else the fps-based lookahead. Returns the forecast (x, y),
    or ``None`` when the KF has lost the target (occlusion / rear) so the
    caller keeps the raw position instead of forecasting into noise.
    Shared by PoseSink and TrackerSink.
    """
    fallback = lookahead_ms_from_fps(fps)
    horizon_ms = (latency.predict_horizon_ms(fallback)
                  if latency is not None else fallback)
    if enhancer.is_tracking_lost():
        return None
    return enhancer.predict_ahead(horizon_ms)


def zone_occupancy_at(zone_lookup: Any, x: float, y: float, *,
                      deepest: bool) -> Tuple[dict, Optional[str]]:
    """Zone occupancy for one point → ``({zone_name: True}, location)``.

    ``deepest=True`` (pose semantics): innermost-wins via
    ``deepest_zone_at_point`` when the lookup provides it, falling back to
    ``get_zones_at_point``. ``deepest=False`` (blob-tracker semantics):
    ``get_zones_at_point`` only, first zone wins. Empty dict + ``None``
    when the point is in no zone.
    """
    if deepest and hasattr(zone_lookup, "deepest_zone_at_point"):
        z = zone_lookup.deepest_zone_at_point(x, y)
        return ({z: True}, z) if z else ({}, None)
    if hasattr(zone_lookup, "get_zones_at_point"):
        occupied = list(zone_lookup.get_zones_at_point(x, y))
        return (dict.fromkeys(occupied, True),
                occupied[0] if occupied else None)
    return ({}, None)


def track_speed(lock, prev_map: Dict[int, Tuple[float, float, int]],
                setup_id: int, cx: float, cy: float) -> float:
    """Swap the per-box previous centroid under ``lock`` and return the
    instantaneous speed (px/s). Shared by PoseSink and TrackerSink."""
    now_ns = host_clock.host_ns()
    with lock:
        prev = prev_map.get(setup_id)
        prev_map[setup_id] = (cx, cy, now_ns)
    return compute_speed(prev, cx, cy, now_ns)


def lookahead_ms_from_fps(fps: float) -> float:
    """Pick a KF forecast horizon based on capture rate.

    Model: 1 frame ahead at <=30 fps, 2 frames ahead at >30 fps, capped
    at 75 ms (the constant-velocity KF overshoots direction changes
    beyond ~100 ms on rodent trajectories).

    Used by both :class:`~source.video.framebus.tracker_sink.TrackerSink` and
    :class:`~source.video.framebus.pose_sink.PoseSink` for latency-compensated
    zone-occupancy lookups.
    """
    if fps <= 0:
        return 33.0
    one_frame_ms = 1000.0 / fps
    horizon_ms = one_frame_ms if fps <= 30 else one_frame_ms * 2
    return min(horizon_ms, 75.0)


class Sink:
    """Worker-thread sink with PER-BOX bounded queue + configurable drop policy.

    Subclass and override ``process(frame: BoxFrame) -> None``. Lifetime
    is owned by the Pipeline: ``start()`` spawns the worker, ``stop()``
    joins it. Sinks may also expose extra ``on_*`` callbacks for results
    they produce (Pose, Tracker), wired by the Pipeline.
    """

    name: str = "sink"

    def __init__(self, *, name: Optional[str] = None,
                 maxsize: int = 1,
                 policy: DropPolicy = DropPolicy.DROP_OLDEST,
                 workers: int = 1,
                 seconds: Optional[float] = None) -> None:
        if name is not None:
            self.name = name
        self._policy = policy
        self._maxsize = max(1, int(maxsize))
        self._nworkers = max(1, int(workers))
        #: A queue bounded in SECONDS rather than frames, when the sink cares.
        #: Sinks are constructed before any camera exists, so a depth fixed in
        #: frames is a depth fixed for an assumed rate: the recorder's 90 is
        #: three seconds at 30 fps and one and a half at 60, and the tracker's
        #: was one second times an ``fps_hint`` nothing ever passed. Both were
        #: sized for a rate nobody checked. ``set_rate_hint`` recomputes this
        #: once the pipeline knows what the cameras actually deliver.
        self._seconds = float(seconds) if seconds else None
        self._floor_maxsize = self._maxsize

        # Per-box queues.  Lazily created on first submit per box so we
        # don't pre-allocate for boxes that never produce frames for
        # this sink.  ``deque`` (not queue.Queue) because we manage
        # bounded eviction explicitly, Queue's bounded semantics block
        # producers, which we never want here.
        self._per_box: Dict[int, "collections.deque[BoxFrame]"] = {}

        # Wake queues: signal a worker that some box has work.  We push
        # ``box_id`` only if it's not already pending, so a wake queue stays
        # bounded (≤ number of active boxes).  A worker processes ONE frame
        # per pop, then re-adds the box_id to the tail if more frames remain,
        # round-robin fairness across boxes without per-box threads.
        #
        # ONE QUEUE PER WORKER, and a box is pinned to a worker by
        # ``box_id % nworkers``.  Pinning is what makes multiple workers safe:
        # a box's frames are always handled by the same thread, so per-box
        # ordering is preserved and ``process`` never runs twice concurrently
        # for the same box, which matters because recorder and tracker both
        # keep per-box state that assumes sequential frames.
        self._wakes: List["queue.Queue[Optional[int]]"] = [
            queue.Queue() for _ in range(self._nworkers)]
        self._pending: Set[int] = set()

        # Lock guards _per_box and _pending.  The wake queues are their own
        # synchronisation (queue.Queue is thread-safe).
        self._lock = threading.Lock()

        self._running = False
        self._threads: List[threading.Thread] = []

        # Monotonic per-sink counters. Cheap ints, never reset, so any caller
        # can sample twice and divide by the interval to get a rate. Without
        # them the only way to find which stage is behind is to guess, and
        # guessing cost this pipeline several rounds of optimising the wrong
        # half while the operator kept reporting the same frame rate.
        self.n_submitted = 0     # handed to this sink
        self.n_processed = 0     # actually ran through process()
        self.n_dropped = 0       # evicted by the drop policy

    # ── Lifecycle -------------------------------------------------------

    def set_rate_hint(self, fps: float) -> None:
        """Re-bound the queue for a camera delivering ``fps``.

        Only for sinks that declared a ``seconds`` budget. Never shrinks below
        the depth the sink was constructed with, so a slow camera cannot make
        a queue too shallow to absorb a burst.
        """
        if not self._seconds or not fps or fps <= 0:
            return
        want = max(self._floor_maxsize, int(round(self._seconds * float(fps))))
        if want == self._maxsize:
            return
        logger.info("%s sink: queue %d -> %d frames (%.1f s at %.1f fps)",
                    self.name, self._maxsize, want, self._seconds, float(fps))
        self._maxsize = want

    def _worker_for(self, box_id: int) -> int:
        return int(box_id) % self._nworkers

    @property
    def _wake(self) -> "queue.Queue[Optional[int]]":
        """The single wake queue, valid only on a one-worker sink.

        PoseSink batches across every box on one queue, which is coherent
        precisely because it runs one worker. Asking for "the" queue on a
        multi-worker sink has no answer, so it raises rather than silently
        returning the first of several and losing the other workers' wakes.
        """
        if self._nworkers != 1:
            raise RuntimeError(
                f"Sink {self.name} has {self._nworkers} workers; use "
                "self._wakes[widx]; there is no single wake queue.")
        return self._wakes[0]

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._threads = []
        for i in range(self._nworkers):
            t = threading.Thread(
                target=self._run, args=(i,),
                name=f"Sink-{self.name}-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 2.0) -> None:
        self._running = False
        # Sentinel per worker so each unblocks immediately.
        for q in self._wakes:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=timeout)
                if t.is_alive():
                    logger.warning("Sink %s worker %s did not stop within %.1fs",
                                   self.name, t.name, timeout)
        self._threads = []
        # Frames still queued at shutdown are discarded, account them so
        # "every frame written or in _drops.tsv" stays true through stop.
        with self._lock:
            leftover = sum(len(d) for d in self._per_box.values())
            for d in self._per_box.values():
                d.clear()
        if leftover:
            _drop_log.record(self.name,
                             reason=f"shutdown_discard_x{leftover}")

    # ── Submission (called by FrameBus subscriber callback) -------------

    def submit(self, frame: BoxFrame) -> bool:
        """Enqueue ``frame`` into ``frame.box_id``'s per-box queue.

        Returns False ONLY for LOSSLESS / DROP_NEWEST when that BOX's
        queue is full.  DROP_OLDEST always returns True (it makes room
        by evicting an older frame from the SAME box).  Frames for
        other boxes are never affected.

        Caller is the FrameBus producer thread; do NOT block here.
        """
        bid = frame.setup_id
        self.n_submitted += 1
        accepted = False
        woke_this_box = False

        with self._lock:
            dq = self._per_box.get(bid)
            if dq is None:
                dq = collections.deque()
                self._per_box[bid] = dq

            if self._policy is DropPolicy.LOSSLESS:
                if len(dq) >= self._maxsize:
                    self._record_drop("lossless_overflow", frame)
                    accepted = False
                else:
                    dq.append(frame)
                    accepted = True

            elif self._policy is DropPolicy.DROP_NEWEST:
                if len(dq) >= self._maxsize:
                    self._record_drop("drop_newest", frame)
                    accepted = False
                else:
                    dq.append(frame)
                    accepted = True

            else:  # DROP_OLDEST
                while len(dq) >= self._maxsize:
                    old = dq.popleft()
                    self._record_drop("drop_oldest", old)
                dq.append(frame)
                accepted = True

            if accepted and bid not in self._pending and dq:
                self._pending.add(bid)
                woke_this_box = True

        if woke_this_box:
            try:
                self._wakes[self._worker_for(bid)].put_nowait(bid)
            except queue.Full:
                # The wake queues are unbounded; this branch is unreachable.
                pass
        return accepted

    # ── Worker loop -----------------------------------------------------

    def _run(self, widx: int = 0) -> None:
        """One worker.  Drains one frame per box per wake-queue pop, then
        re-queues the box_id at the tail if more frames remain.  Round-robin
        fairness emerges from the FIFO wake queue; parallelism across boxes
        comes from there being several of these, each owning its own slice of
        the boxes (see ``_worker_for``)."""
        wake = self._wakes[widx]
        while self._running:
            try:
                bid = wake.get(timeout=0.5)
            except queue.Empty:
                continue
            if bid is None:
                break

            frame: Optional[BoxFrame] = None
            still_has_work = False
            with self._lock:
                dq = self._per_box.get(bid)
                if dq:
                    frame = dq.popleft()
                if dq:
                    # Keep _pending[bid] = True; we re-push to wake tail below.
                    still_has_work = True
                else:
                    self._pending.discard(bid)

            if frame is not None:
                self.n_processed += 1
                try:
                    self.process(frame)
                except Exception as e:
                    logger.exception("Sink %s process error (box=%s): %s",
                                     self.name, bid, e)

            if still_has_work:
                try:
                    wake.put_nowait(bid)
                except queue.Full:
                    pass

    # ── Subclass surface ------------------------------------------------

    def process(self, frame: BoxFrame) -> None:
        raise NotImplementedError

    # ── Internal --------------------------------------------------------

    def _record_drop(self, reason: str, frame: BoxFrame) -> None:
        self.n_dropped += 1
        try:
            _drop_log.record(
                self.name,
                frame_idx=frame.cam_frame_id,
                capture_ts=frame.capture_host_ns / 1e9,
                reason=f"{reason}_box{frame.setup_id}",
            )
        except Exception:
            pass
