"""Pose-inference execution layer.

Owns the lifecycle of pose models (DLC, SLEAP) and the execution
strategy (thread vs subprocess) that feeds them frames.

  * The model cache key is (model_path, frame_w, frame_h, resize_factor,
    tracker_type). Reusing the same key returns the cached instance with
    no re-init.
  * Model instances are not thread-safe (DLCLive holds one TF/PyTorch
    session), so max_workers=1 in the default ThreadInferenceBackend is
    deliberate.
  * Frame size within a session is the user's responsibility; the cache
    key makes a violation visible (different shape -> different key ->
    rebuild) instead of silently mis-inferencing.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from source.video.recording.drop_log import drop_log as _drop_log
from source.video.tracking.pose import (PoseTracker, create_pose_tracker,
                                         filter_options)

logger = logging.getLogger(__name__)


ModelKey = Tuple[str, int, int, float, str, str]
"""(model_path, frame_w, frame_h, resize_factor, tracker_type, sleap_sig).

``sleap_sig`` is a compact string of the SLEAP-only options (centroid path /
model type / runtime / device / fp16), ``""`` for DLC/blob, so two SLEAP
configs that differ only in runtime or centroid model cache distinctly instead
of colliding."""


@dataclass
class ModelHandle:
    """Reference to a loaded pose model + its cache key."""

    key: ModelKey
    model: PoseTracker

    @property
    def is_initialized(self) -> bool:
        return self.model.is_initialized


def _new_tracker(tracker_type: str, model_path: str,
                 probe_frame: Optional[np.ndarray],
                 confidence_threshold: float, resize: float,
                 body_parts: Optional[list],
                 sleap_opts: Optional[Dict[str, Any]] = None,
                 dlc_opts: Optional[Dict[str, Any]] = None,
                 colour_mode: str = "auto") -> Optional[PoseTracker]:
    """Build kwargs, create a pose tracker for a key, and initialize it.

    Returns the initialized tracker, or None if ``initialize`` failed.
    Creation exceptions propagate so each backend logs its own context and
    runs its own cleanup. ``sleap_opts`` (centroid_path / model_type / runtime /
    device / fp16 / compile) is forwarded ONLY for SLEAP so the DLC/blob
    constructors never see unexpected kwargs.
    """
    kind = tracker_type.lower().strip()
    kwargs: Dict[str, Any] = {
        "resize_factor": resize,
        "colour_mode": colour_mode or "auto",
    }
    if body_parts:
        kwargs["body_parts"] = body_parts
    if sleap_opts and kind in ("sleap", "sleap-nn", "sleap_nn"):
        kwargs.update({k: v for k, v in sleap_opts.items() if v is not None})
    if dlc_opts and kind == "dlc":
        kwargs.update({k: v for k, v in dlc_opts.items() if v is not None})
    # Filtered against the constructor rather than trusted. Separating the two
    # option groups by hand is what kept this working, and it made an option
    # the caller spelled wrong a TypeError in the middle of a recording rather
    # than a line in the log. Same rule as the offline seam, so the two
    # pipelines cannot disagree about what a backend accepts.
    kwargs, ignored = filter_options(tracker_type, kwargs)
    if ignored:
        logger.info("%s ignores %s: not parameters of this backend",
                    kind, ", ".join(ignored))
    tracker = create_pose_tracker(tracker_type, model_path, **kwargs)
    if not tracker.initialize(probe_frame):
        return None
    return tracker


def _post_workers() -> int:
    """Threads for per-box pose post-processing.

    Capped deliberately: the work is numpy and cv2, which release the GIL only
    in parts, so past a handful of threads this contends with the camera and
    encoder threads for no gain. Two is the floor, because one would
    re-serialise the very chain the pool exists to break.
    """
    return max(2, min(4, (os.cpu_count() or 4) - 2))


def _resolve(marker: Optional[Future]) -> None:
    """Reopen a box's in-flight gate by completing its marker.

    Must run on every path out of a dispatch. A marker left pending shuts that
    box out of inference for the rest of the session, which looks like one
    camera quietly going blind rather than like an error.
    """
    if marker is None or marker.done():
        return
    try:
        marker.set_result(None)
    except Exception:                      # already resolved by a racing path
        pass


def _run_post(bid: int, pose: dict, on_done: Callable[[int, dict], None],
              ondone_err: str, marker: Optional[Future]) -> None:
    """One box's post-processing, off the inference thread."""
    try:
        on_done(bid, pose)
    except Exception as e:
        logger.error(ondone_err, bid, e)
    finally:
        _resolve(marker)


def _run_batched(handle: ModelHandle, items: list,
                 on_done: Callable[[int, dict], None],
                 predict_err: str, ondone_err: str,
                 post_pool: Optional[ThreadPoolExecutor] = None,
                 markers: Optional[Dict[int, Future]] = None) -> None:
    """Run ONE batched forward pass on ``handle`` then fan results out per box.

    Frames in ``items`` are assumed same-shape (the caller groups by shape);
    this stacks + dispatches. ``predict_err`` / ``ondone_err`` are %-format
    log templates used when the batched predict (``e``) or a per-box callback
    (``bid, e``) raises. Shared by both backends' executor tasks.

    With ``post_pool``, the per-box callbacks are handed to that pool and this
    thread returns as soon as the forward pass is done, so the next batch can
    start while the previous batch's boxes are still being post-processed.
    Only the forward pass has to be serialised here (the model object is not
    thread-safe); the callback after it touches only its own box's state.
    Leaving it on this thread made the cost of a frame scale with box count:
    measured at 2.9 ms per box, sixteen boxes spent 46 ms per frame set on the
    one thread that also owed the next forward pass.

    ``markers`` then carries one Future per box, resolved when THAT box's
    callback finishes. The in-flight gate keys on it, so a box still cannot be
    handed a second frame while its own post-processing is running, which is
    what keeps its smoother single-threaded.
    """
    markers = markers or {}
    box_ids = [bid for bid, _ in items]
    frames = [f for _, f in items]
    handed: set = set()
    try:
        try:
            poses = handle.model.predict_batch(frames)
        except Exception as e:
            logger.error(predict_err, e)
            poses = [{} for _ in items]
        if len(poses) != len(items):
            # Safety: pad / truncate so the per-box callback contract holds.
            logger.error(
                "Batched pose returned %d results for %d frames; aligning",
                len(poses), len(items))
            if len(poses) < len(items):
                poses = list(poses) + [{}] * (len(items) - len(poses))
            else:
                poses = poses[: len(items)]

        if post_pool is None:
            for bid, pose in zip(box_ids, poses):
                try:
                    on_done(bid, pose)
                except Exception as e:
                    logger.error(ondone_err, bid, e)
            return

        for bid, pose in zip(box_ids, poses):
            try:
                post_pool.submit(_run_post, bid, pose, on_done, ondone_err,
                                 markers.get(bid))
                handed.add(bid)
            except RuntimeError:           # pool shutting down
                _drop_log.record(
                    "tracker", frame_idx=None, capture_ts=time.time(),
                    reason=f"post_pool_shutdown_box{bid}",
                )
    finally:
        # Anything never handed off owns no resolver of its own.
        for bid in box_ids:
            if bid not in handed:
                _resolve(markers.get(bid))


#: How long a teardown waits for work already on the GPU. Long enough for a
#: large batch to finish, short enough that a wedged worker cannot hang the
#: GUI thread that is waiting to rebuild the model.
_DRAIN_TIMEOUT_S = 5.0


def _drain_inflight(inflight: Dict[Any, Future]) -> None:
    """Wait for submitted batches to finish, then forget them.

    A dropped reference does not stop a running thread. Without this the model
    is closed while the executor is still inside it, which is a use-after-free
    in a C extension rather than a Python error, it surfaces as a process
    crash somewhere unrelated.
    """
    futures = [f for f in list(inflight.values()) if f is not None]
    inflight.clear()
    if not futures:
        return
    deadline = time.monotonic() + _DRAIN_TIMEOUT_S
    for fut in futures:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning("inference teardown: drain timed out with %d batch(es) "
                           "still running; closing anyway", len(futures))
            return
        try:
            fut.exception(timeout=remaining)      # waits; never re-raises here
        except _FutureTimeout:
            logger.warning("inference teardown: batch did not finish within %.1fs",
                           _DRAIN_TIMEOUT_S)
            return
        except Exception:
            pass                                   # a failed batch is still done


def release_device_memory() -> None:
    """Return freed GPU blocks to the driver, if a framework is loaded.

    Dropping the last reference lets the allocator reuse the block, but PyTorch
    keeps it in its caching allocator, so the memory never comes back to the
    driver and a session that re-initialises a few times climbs until it fails.
    Only acts on frameworks that are ALREADY imported: importing torch here to
    free memory would allocate far more than it releases.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.debug("release_device_memory: %s", e)


class InferenceBackend(ABC):
    """Strategy interface for pose-model lifecycle + dispatch."""

    @abstractmethod
    def get_or_create(self, key: ModelKey,
                      probe_frame: Optional[np.ndarray] = None,
                      body_parts: Optional[list] = None,
                      confidence_threshold: float = 0.75,
                      sleap_opts: Optional[Dict[str, Any]] = None,
                      dlc_opts: Optional[Dict[str, Any]] = None,
                      colour_mode: str = "auto") -> Optional[ModelHandle]:
        """Return cached model for `key`, or create a new one and cache it.

        On a miss, closes the previously active model first (free GPU
        memory) so a session never holds more than one model at a time.

        Returns None if creation/init fails (caller logs and gives up).
        """

    @abstractmethod
    def submit_batch(self, model: ModelHandle,
                     items: list,                          # List[Tuple[int, np.ndarray]]
                     on_done: Callable[[int, dict], None]) -> int:
        """Submit a batch of (box_id, frame) for ONE inference call.

        Groups N boxes' frames into a single ``[N, H, W, C]`` tensor and
        runs one forward pass on the GPU, one call pays the kernel-launch
        overhead instead of N, a throughput win for multi-box pose.

        Returns the count of frames actually accepted into the batch.
        Frames whose previous inference for the same box is still in
        flight are dropped (per-box gate, same as ``submit``) and not
        counted in the return.

        ``on_done`` is invoked once per accepted box on the executor
        thread when the batch completes.
        """

    @abstractmethod
    def shutdown(self, wait: bool = False) -> None:
        """Stop the executor; close all models. Idempotent."""


class ThreadInferenceBackend(InferenceBackend):
    """In-process backend: ThreadPoolExecutor(max_workers=1).

    The single worker thread is mandatory: pose models are not
    thread-safe. The TF/PyTorch C-extension releases the GIL during
    inference, so the GUI thread isn't blocked.

    Per-box in-flight gating: submit() drops the frame if the previous
    one for that box is still in flight, so latency stays bounded and no
    backlog of stale frames builds up.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pose-infer")
        #: Per-box post-processing, off the inference thread. See _run_batched
        #: for why the callbacks must not run where the forward pass runs.
        self._post_pool = ThreadPoolExecutor(
            max_workers=_post_workers(), thread_name_prefix="pose-post")
        self._models: Dict[ModelKey, ModelHandle] = {}
        self._active_key: Optional[ModelKey] = None
        self._inflight: Dict[int, Future] = {}
        self._shutdown = False

    # ------------------------------------------------------------------
    # Model cache
    # ------------------------------------------------------------------

    def get_or_create(self, key: ModelKey,
                      probe_frame: Optional[np.ndarray] = None,
                      body_parts: Optional[list] = None,
                      confidence_threshold: float = 0.75,
                      sleap_opts: Optional[Dict[str, Any]] = None,
                      dlc_opts: Optional[Dict[str, Any]] = None,
                      colour_mode: str = "auto") -> Optional[ModelHandle]:
        if self._shutdown:
            return None

        cached = self._models.get(key)
        if cached is not None and cached.is_initialized:
            self._active_key = key
            return cached

        # Miss, evict the currently active model first (free GPU memory).
        if self._active_key is not None and self._active_key != key:
            self._evict(self._active_key)

        model_path, w, h, resize, tracker_type, _sig = key
        try:
            tracker = _new_tracker(tracker_type, model_path, probe_frame,
                                   confidence_threshold, resize, body_parts,
                                   sleap_opts, dlc_opts, colour_mode)
            if tracker is None:
                logger.error(
                    "InferenceBackend: %s initialize failed (path=%s, %dx%d, resize=%.2f)",
                    tracker_type, model_path, w, h, resize)
                return None
        except Exception as e:
            logger.error("InferenceBackend: create_pose_tracker raised: %s", e)
            return None

        handle = ModelHandle(key=key, model=tracker)
        self._models[key] = handle
        self._active_key = key
        logger.info(
            "InferenceBackend: loaded %s (path=%s, %dx%d, resize=%.2f, parts=%s)",
            tracker_type, model_path, w, h, resize, tracker.get_body_parts())
        return handle

    def _evict(self, key: ModelKey) -> None:
        """Drain, then close, then hand the memory back.

        Order is the whole point. Closing a model while a batch is still
        running on the executor frees the object under the thread using it, and
        the failure surfaces later as a crash inside the inference library with
        no connection to the setting the operator just changed.
        """
        handle = self._models.pop(key, None)
        if handle is None:
            return
        _drain_inflight(self._inflight)
        try:
            handle.model.close()
        except Exception as e:
            logger.debug("InferenceBackend: close error for %s: %s", key, e)
        release_device_memory()
        if self._active_key == key:
            self._active_key = None

    # ------------------------------------------------------------------
    # Submit / poll
    # ------------------------------------------------------------------

    def submit_batch(self, model: ModelHandle, items: list,
                     on_done: Callable[[int, dict], None]) -> int:
        """Run ONE batched inference for the accepted (box_id, frame) items.

        Per-box in-flight gate is preserved: if box X is still inferring
        from a previous batch, X's frame in this batch is dropped.  All
        accepted boxes share one Future (the batch task) so their
        ``_inflight`` entries clear together when the batch completes.
        Same-shape grouping is the tracker's responsibility (passed to
        ``predict_batch``); the backend just dispatches.
        """
        if self._shutdown or not items:
            return 0

        accepted: list = []
        for bid, f in items:
            prev = self._inflight.get(bid)
            if prev is not None and not prev.done():
                _drop_log.record(
                    "tracker", frame_idx=None, capture_ts=time.time(),
                    reason=f"inference_inflight_box{bid}",
                )
                continue
            accepted.append((bid, f))

        if not accepted:
            return 0

        # One marker per box, resolved when that box's post-processing ends.
        markers: Dict[int, Future] = {bid: Future() for bid, _ in accepted}
        try:
            future = self._executor.submit(
                _run_batched, model, accepted, on_done,
                "Batched pose predict raised: %s",
                "Box %d: pose on_done raised: %s",
                self._post_pool, markers,
            )
        except RuntimeError:
            for bid, _ in accepted:
                _resolve(markers[bid])
                _drop_log.record(
                    "tracker", frame_idx=None, capture_ts=time.time(),
                    reason=f"executor_shutdown_box{bid}",
                )
            return 0

        # A cancelled batch never reaches _run_batched's own cleanup, so its
        # markers would stay pending and gate those boxes out permanently.
        def _sweep(fut: Future, _m=markers) -> None:
            if fut.cancelled():
                for marker in _m.values():
                    _resolve(marker)

        future.add_done_callback(_sweep)

        for bid, _ in accepted:
            # The gate keys on post-processing completion, not on the forward
            # pass. The pass returning only means the GPU is free; this box's
            # callback may still be running and still owns its smoother.
            self._inflight[bid] = markers[bid]
        return len(accepted)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, wait: bool = False) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        try:
            self._executor.shutdown(wait=wait, cancel_futures=True)
        except Exception:
            pass
        # _evict drains the in-flight markers, which the post pool is what
        # resolves, so it has to still be running here. Closed after, never
        # before, or the drain waits out its full timeout on every teardown.
        for key in list(self._models):
            self._evict(key)
        try:
            self._post_pool.shutdown(wait=wait, cancel_futures=True)
        except Exception:
            pass
        self._inflight.clear()


class MultiInstanceInferenceBackend(InferenceBackend):
    """N independent model copies on the GPU, each in its own executor.

    ``ThreadInferenceBackend`` runs ``max_workers=1`` because one
    TF/PyTorch model object is not thread-safe. N separate model objects
    each have their own session and can run in parallel on the GPU's free
    SMs, so with enough boxes (or mixed ROI sizes that defeat batched
    single-instance) spreading them across N instances cuts per-box
    latency and uses idle GPU compute.

    Cost: ~N × model size in GPU memory. PoseSink defaults to 1 instance
    (same behaviour as ThreadInferenceBackend); the user opts in via the
    ``Pose model instances`` field in the tracking config dialog.
    """

    def __init__(self, n_instances: int = 2) -> None:
        n = max(1, int(n_instances))
        self._n = n
        # One executor per instance (each with max_workers=1 so the
        # instance's own model can never be hit by two threads).
        self._executors: list = [
            ThreadPoolExecutor(max_workers=1,
                               thread_name_prefix=f"pose-infer-{i}")
            for i in range(n)
        ]
        # Per-instance ModelHandle list (parallel to executors).
        self._handles: list = [None] * n
        self._active_key: Optional[ModelKey] = None
        # Per-box-per-instance inflight gate: (instance_idx, box_id) -> Future
        self._inflight: Dict[tuple, Future] = {}
        # Round-robin counter for fair instance assignment when items > N.
        self._rr = 0
        self._shutdown = False

    # Model cache --------------------------------------------------------

    def get_or_create(self, key: ModelKey,
                      probe_frame: Optional[np.ndarray] = None,
                      body_parts: Optional[list] = None,
                      confidence_threshold: float = 0.75,
                      sleap_opts: Optional[Dict[str, Any]] = None,
                      dlc_opts: Optional[Dict[str, Any]] = None,
                      colour_mode: str = "auto") -> Optional[ModelHandle]:
        if self._shutdown:
            return None
        # Already loaded for this key, first handle stands in as the
        # logical handle PoseSink stores; submit_batch uses the internal
        # pool instead.
        if (self._handles[0] is not None
                and self._handles[0].key == key
                and self._handles[0].is_initialized):
            self._active_key = key
            return self._handles[0]

        # Different key, close existing instances first.
        if self._active_key is not None and self._active_key != key:
            self._close_all()

        model_path, w, h, resize, tracker_type, _sig = key

        # Load N independent copies of the same model.  Each gets its
        # own session / weights / GPU allocation.
        try:
            for i in range(self._n):
                tracker = _new_tracker(tracker_type, model_path, probe_frame,
                                       confidence_threshold, resize, body_parts,
                                       sleap_opts, dlc_opts, colour_mode)
                if tracker is None:
                    logger.error(
                        "MultiInstance: instance %d initialise failed", i)
                    self._close_all()
                    return None
                self._handles[i] = ModelHandle(key=key, model=tracker)
                logger.info(
                    "MultiInstance: instance %d/%d loaded (%s, %dx%d)",
                    i + 1, self._n, tracker_type, w, h)
        except Exception as e:
            logger.error("MultiInstance: tracker create raised: %s", e)
            self._close_all()
            return None

        self._active_key = key
        return self._handles[0]

    def _close_all(self) -> None:
        """Drain every instance, close them all, then release once.

        N copies means N times the memory to hand back, so the release matters
        more here than in the single-model backend, and it is done once after
        the last close rather than per instance, because the allocator only
        needs telling when there is nothing left to reuse.
        """
        _drain_inflight(self._inflight)
        for i, h in enumerate(self._handles):
            if h is None:
                continue
            try:
                h.model.close()
            except Exception as e:
                logger.debug("MultiInstance: close instance %d: %s", i, e)
        self._handles = [None] * self._n
        self._active_key = None
        release_device_memory()

    # Submit -------------------------------------------------------------

    def submit_batch(self, model: ModelHandle, items: list,
                     on_done: Callable[[int, dict], None]) -> int:
        if self._shutdown or not items:
            return 0
        if self._handles[0] is None:
            return 0
        # Distribute items across instances.  Round-robin so a box
        # tends to land on the same instance across calls (cache /
        # session-warm effect) but rotates to balance load.
        per_instance: list = [[] for _ in range(self._n)]
        for idx, (bid, f) in enumerate(items):
            inst = (self._rr + idx) % self._n
            per_instance[inst].append((bid, f))
        self._rr = (self._rr + len(items)) % self._n

        accepted_total = 0
        for inst_idx, chunk in enumerate(per_instance):
            if not chunk:
                continue
            # Per-box inflight gate scoped per-instance.
            kept = []
            for bid, f in chunk:
                key = (inst_idx, bid)
                prev = self._inflight.get(key)
                if prev is not None and not prev.done():
                    _drop_log.record(
                        "tracker", frame_idx=None, capture_ts=time.time(),
                        reason=f"inference_inflight_inst{inst_idx}_box{bid}")
                    continue
                kept.append((bid, f))
            if not kept:
                continue
            try:
                future = self._executors[inst_idx].submit(
                    _run_batched, self._handles[inst_idx], kept, on_done,
                    "MultiInstance batch predict raised: %s",
                    "MultiInstance on_done (box %d): %s")
            except RuntimeError:
                continue
            for bid, _ in kept:
                self._inflight[(inst_idx, bid)] = future
            accepted_total += len(kept)
        return accepted_total

    def shutdown(self, wait: bool = False) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        for ex in self._executors:
            try:
                ex.shutdown(wait=wait, cancel_futures=True)
            except Exception:
                pass
        self._close_all()
        self._inflight.clear()
