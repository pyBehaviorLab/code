"""PoseSink, DLC/SLEAP inference with skip-but-write semantics.

When inference can't keep up, skipping is OK as long as the recorder
still writes the frame with an empty pose field (so the user knows
inference didn't run on it). PoseSink keeps at most one inference in
flight per box and publishes each result with the source
``cam_frame_id`` for RecorderSink's frame matching.

Drop policy is DROP_NEWEST with maxsize=1: while inference for a box is in
flight one frame may wait, and any frame arriving on top of that is dropped,
the ARRIVING frame, not the queued one. Both are already older than the
inference in flight, so swapping in the marginally fresher frame changes
nothing and costs a copy. The single-worker ThreadInferenceBackend then
dispatches.

GPU (Jetson / NVIDIA): the model runs in TF/PyTorch on GPU; no per-frame
torch tensor allocation here. The numpy frame is passed by reference;
the model copies it into device memory itself.
"""

from __future__ import annotations

import logging
import queue as _queue
import math
import os
import threading
import time
from source import host_clock
from typing import Any, Callable, Dict, List, Optional, Tuple

from source.video.framebus.types import BoxFrame
from source.video.tracking.crop_tracker import BoxCropState, crop_at
from source.video.tracking.letterbox import Letterbox, canonical_shape
from source.video.tracking.sleap_export import batch_bucket
from source.video.tracking.inference import (
    InferenceBackend,
    ModelHandle,
    ThreadInferenceBackend,
    MultiInstanceInferenceBackend,
    ModelKey,
)
from source.video.framebus.sink_base import (
    DropPolicy,
    Sink,
    estimate_capture_fps,
    forecast_point,
    notify_subscribers,
    track_speed,
    zone_occupancy_at,
)
from source.video.framebus.types import head_direction, DEFAULT_ROTATION_PAIR
from source.video.framebus.latency import LatencyBudget

logger = logging.getLogger(__name__)

# Set ``PYBL_POSE_DEBUG=1`` before launch to surface per-batch ingest,
# accept/skip counts, and per-result pose dict diagnostics.  Off by
# default, zero perf cost when not set.
_POSE_DEBUG = bool(os.environ.get("PYBL_POSE_DEBUG"))


# Result callback: (box_id, cam_frame_id, pose_array, location, speed,
#                   zones_by_body_part, raw_pose_dict, *, capture_host_ns)
PoseResultCallback = Callable[
    [int, int, list, Optional[str], float, dict, dict], None
]

# Failure callback: (box_id, reason_string, consecutive_empty_streak).
# Fired when the empty-streak crosses a warning threshold (3, 20, 100)
# so the GUI can surface a banner instead of silently-empty pose rows.
PoseFailedCallback = Callable[[int, str, int], None]


#: "the caller passed nothing", distinct from "the caller passed None": and
#: None is a real value here: it is the transform of a box in ``full`` mode.
_MISSING = object()


def _opts_sig(opts, keys) -> str:
    """Compact, stable signature of backend options for the cache key.

    Anything an SDK fixes at construction has to appear here. DeepLabCut-Live
    selects its runner from ``model_type`` and ``precision`` in ``__init__``,
    and a TensorRT engine bakes precision at export, so two configs differing
    only in precision are two different models, and sharing one cache entry
    would hand the operator the previous precision without a word.
    """
    if not opts:
        return ""
    return "|".join(f"{k}={opts.get(k)}" for k in keys
                    if opts.get(k) not in (None, ""))


def _sleap_sig(sleap_opts) -> str:
    """Signature of the SLEAP-only options (empty for DLC/blob)."""
    return _opts_sig(sleap_opts,
                     ("centroid_path", "sleap_model_type", "runtime", "device",
                      "fp16", "max_batch_size"))


def _dlc_sig(dlc_opts) -> str:
    """Signature of the DLC-only options (empty for SLEAP/blob)."""
    return _opts_sig(dlc_opts, ("dlc_model_type", "precision", "device",
                                "dynamic_crop", "dynamic_threshold",
                                "dynamic_margin", "max_batch_size"))


def pose_fingerprint(*, tracker_type: str, model_path: str,
                     resize_factor: float, confidence: float,
                     body_parts, sleap_opts=None, dlc_opts=None,
                     colour_mode: str = "auto", input_mode: str = "auto",
                     input_wh=None, crop_opts=None) -> tuple:
    """Everything about a pose model that only a re-load can change.

    ONE definition, used by both sides of the "is the loaded model still the
    one this box wants?" question: ``configure_model`` stores what it was
    called with, and the host builds the same tuple from the box config. The
    previous arrangement had the host DESCRIBE the model in its own second
    implementation, which silently omitted the DLC engine options and so
    answered "not loaded" on every DLC box for ever.

    Deliberately outside the fingerprint, because none of them can change what
    the network is or how it is built:

    * frame width and height, and the batch size derived from the box count -
      camera and rig facts, not user settings;
    * ``pose_n_instances``, the push gates, zones and triggers - runtime knobs;
    * everything that only changes how a result is drawn or written.
    """
    wh = tuple(int(v) for v in (input_wh or (0, 0)))
    crop = crop_opts or {}
    return (
        str(tracker_type or "").lower(),
        str(model_path or ""),
        round(float(resize_factor or 1.0), 6),
        round(float(confidence or 0.0), 6),
        tuple(body_parts or ()),
        _sleap_sig(sleap_opts),
        _dlc_sig(dlc_opts),
        str(colour_mode or "auto"),
        str(input_mode or "auto"),
        wh if len(wh) == 2 else (0, 0),
        (round(float(crop.get("conf_min", 0.20) or 0.20), 6),
         int(crop.get("good_min", 3) or 3),
         bool(crop.get("reacquire", True))),
    )


def _make_model_key(model_path: str, frame, resize_factor: float,
                    tracker_type: str, sleap_opts=None,
                    dlc_opts=None, colour_mode: str = "auto") -> ModelKey:
    h, w = frame.shape[:2]
    # Backend signatures and colour mode ride in the same slot: only one
    # backend is ever active, and colour mode changes what is fed to the
    # network, so it must rebuild rather than silently apply to a live model.
    sig = "|".join(s for s in (_sleap_sig(sleap_opts), _dlc_sig(dlc_opts),
                               (f"colour={colour_mode}"
                                if colour_mode and colour_mode != "auto" else ""))
                   if s)
    return (str(model_path), int(w), int(h), float(resize_factor),
            str(tracker_type), sig)


class PoseSink(Sink):
    """One sink that runs inference for any number of boxes.

    All boxes share one model handle (the project today loads a single
    DLC/SLEAP model at a time, the cache key in InferenceBackend
    enforces this). Per-box state (zone manager, prev centroid) lives
    here; the model is a singleton.
    """

    name = "pose"

    def __init__(self, *,
                 backend: Optional[InferenceBackend] = None,
                 confidence_threshold: float = 0.5,
                 batch_settle_s: float = 0.001,
                 max_batch_size: int = 0) -> None:
        """
        batch_settle_s: window the worker lingers after the first wake to
            let other boxes' submits arrive (so a producer iterating N
            boxes back-to-back yields one N-batch instead of N 1-batches).
            1 ms suits desktop CPUs; bump to 2-5 ms on slower hosts (Jetson
            Nano, embedded ARM) to compensate for slower context switching.
        max_batch_size: cap batches to this size to fit GPU memory.  0 =
            unlimited.  Set to 4 on Jetson Nano (2 GB), 8 on Xavier NX
            (8 GB), unlimited on a desktop with 12+ GB.  When the pending
            set exceeds the cap, the batch is split into multiple back-
            to-back inference calls (still amortizes per-call overhead
            across the boxes in each chunk).
        """
        # maxsize=1 + DROP_NEWEST = "in flight + at most one queued";
        # process the latest available rather than backlog.
        #
        # Deliberately ONE worker. Inference is GPU-bound, and the backend
        # batches across boxes itself, extra host threads would only contend
        # for the same device and break that batching.
        super().__init__(name="pose", maxsize=1, policy=DropPolicy.DROP_NEWEST,
                         workers=1)
        self._backend: InferenceBackend = backend or ThreadInferenceBackend()
        self.BATCH_SETTLE_S = float(batch_settle_s)
        self._max_batch_size = int(max_batch_size)
        #: The batch the LOADED engine was constructed for. A hard ceiling, not
        #: a preference: an exported engine's batch axis is fixed when it is
        #: built, so a larger submission fails outright. 0 = nothing built yet.
        self._built_for = 0
        self._warned_outgrew = False
        # Round-robin offset so chunked dispatches start at a different
        # position; otherwise when max_batch_size < N_boxes the first
        # ``cap`` boxes always win and trailing boxes starve.
        self._chunk_rotation = 0
        self._handle: Optional[ModelHandle] = None
        #: Settings the loaded model was built from. ``None`` until a model is
        #: configured; see :func:`pose_fingerprint`.
        self._fingerprint: Optional[Tuple] = None
        self._tracker_type: str = "dlc"
        self._resize: float = 1.0
        # One input shape for every box, so differently-cropped boxes share
        # one model instead of each evicting the last. ``None`` until a model
        # is configured; the per-box transforms are cached by source shape
        # because that shape only changes when the ROI does.
        self._canonical: Optional[Tuple[int, int]] = None       # (w, h)
        self._letterboxes: Dict[int, Letterbox] = {}
        self._box_shapes: Dict[int, Tuple[int, int]] = {}       # (w, h)
        # How the frame becomes the model's input: ``letterbox`` (scale + pad
        # to one shape) or ``crop_track`` (cut the training window at native
        # scale and follow the animal). ``full`` hands the frame over as it is.
        self._input_mode: str = "letterbox"
        self._crop_wh: Optional[Tuple[int, int]] = None
        self._crop_states: Dict[int, BoxCropState] = {}
        self._crop_opts: Dict[str, Any] = {}
        # The transform used for THIS frame, per box. Both a Letterbox and a
        # CropWindow answer ``pose_to_source``, so the inverse at the far end
        # does not care which mode produced it, which is the whole reason the
        # two live behind one slot.
        self._input_transform: Dict[int, Any] = {}
        self._body_parts: List[str] = []
        self._confidence: float = float(confidence_threshold)
        # Observational timing spine (set by the Pipeline). Records the
        # poll→inference-done latency; None = not measured.
        self._latency: Optional[LatencyBudget] = None
        # Per-box optional ZoneManager-like for location lookup; the
        # Pipeline injects this. Object must expose
        # ``get_zones_at_point(x, y) -> Iterable[str]``.
        self._zone_lookup: Dict[int, Any] = {}
        # Per-box optional TrackingEnhancer (set externally via
        # ``set_enhancer``). When attached, the enhancer's KF is
        # corrected with the pose centroid each frame and queried via
        # ``predict_ahead(horizon_ms)`` for the latency-compensated
        # zone lookup that drives MCU push.  Smoothed (current)
        # position still drives display + speed.
        self._enhancers: Dict[int, Any] = {}
        # Per-box per-keypoint filter. Absent = off, which is the state a box
        # is in until the tracking config asks for it.
        self._gap_fillers: Dict[int, Any] = {}
        #: Previous capture instant per box, for the filter's dt. Separate
        #: from ``_prev_capture_ns``, which the centroid forecast owns and
        #: consumes; sharing one would make each read the other's frame.
        self._prev_fill_ns: Dict[int, int] = {}
        # Per-box prev capture time for instantaneous fps estimate.
        self._prev_capture_ns: Dict[int, int] = {}
        # Per-box prev centroid for speed.  (cx, cy, monotonic_ns)
        self._prev_centroid: Dict[int, Tuple[float, float, int]] = {}
        # Per-box rotation config.  Off by default; when enabled the
        # configured keypoint pair (tail, head) defines the vector.
        # State holds the unit-circle EMA so successive frames smooth.
        self._rotation_enabled: Dict[int, bool] = {}
        self._rotation_pair: Dict[int, Tuple[str, str]] = {}
        self._rotation_state: Dict[int, dict] = {}
        # Per-box: which body part defines the "centroid" used for
        # location / MCU c.loc_center / zone_changed. ``"centroid"``
        # (default) keeps the first-confident-keypoint heuristic; any
        # other value names a keypoint, when it's below confidence,
        # cx,cy become None and location reports "na" instead of falling
        # back to another keypoint's zone.
        self._centroid_body_part: Dict[int, str] = {}
        # Result subscribers.
        self._on_result: List[PoseResultCallback] = []
        # Failure subscribers, fired on empty-streak threshold crossing
        # so the GUI can show a banner.
        self._on_failed: List[PoseFailedCallback] = []
        # Per-box enabled flag.
        self._enabled: Dict[int, bool] = {}
        # Per-box streak of consecutive all-zero-confidence results;
        # read in _on_pose_done to detect silent failure and warn.
        self._empty_result_streak: Dict[int, int] = {}
        #: When each box last warned about empty poses, so a flapping failure
        #: cannot re-warn every time its streak crosses 3 again.
        self._empty_warned_ns: Dict[int, int] = {}
        self._lock = threading.RLock()

    # ── Public API ----------------------------------------------------

    def current_n_instances(self) -> int:
        """How many parallel model copies the active backend holds.

        1 for ``ThreadInferenceBackend`` (the default), N for
        ``MultiInstanceInferenceBackend``.  Used by the GUI to decide
        whether ``set_n_instances`` needs to swap.
        """
        if isinstance(self._backend, MultiInstanceInferenceBackend):
            return int(getattr(self._backend, "_n", 1))
        return 1

    def set_n_instances(self, n: int) -> None:
        """Swap the inference backend so it holds ``n`` parallel model
        copies.  ``n=1`` uses :class:`ThreadInferenceBackend` (single
        shared session); ``n>1`` uses :class:`MultiInstanceInferenceBackend`
        with ``n`` independent copies running in parallel via
        ``ThreadPoolExecutor`` per copy.

        Caller is the GUI tracking config dialog, invoked just before
        Init DLC.  ``n`` should be ``ceil(N_pose_boxes / 4)``.

        Idempotent: if the active backend already has ``n`` copies the
        call is a no-op.  Otherwise the old backend is shut down and
        any cached model handle is dropped, the next
        ``configure_model`` call will reload onto the new backend.
        """
        n = max(1, int(n))
        if self.current_n_instances() == n:
            return
        with self._lock:
            old = self._backend
            try:
                old.shutdown(wait=False)
            except Exception as e:
                logger.debug("PoseSink: old backend shutdown raised: %s", e)
            if n == 1:
                self._backend = ThreadInferenceBackend()
            else:
                self._backend = MultiInstanceInferenceBackend(n_instances=n)
            self._handle = None
        logger.info("PoseSink: backend swapped → %d instance(s)", n)

    def configure_model(self, *, tracker_type: str, model_path: str,
                        probe_frame, resize_factor: float = 1.0,
                        body_parts: Optional[List[str]] = None,
                        confidence: float = 0.5,
                        sleap_opts: Optional[dict] = None,
                        dlc_opts: Optional[dict] = None,
                        colour_mode: str = "auto",
                        input_mode: str = "letterbox",
                        input_wh: Optional[Tuple[int, int]] = None,
                        crop_opts: Optional[dict] = None,
                        n_boxes: int = 0) -> Optional[ModelHandle]:
        """Load (or re-use cached) model. Cache key =
        (path,w,h,resize,type,backend_sig). ``sleap_opts`` (centroid_path /
        model_type / runtime / device / fp16) goes to the SLEAP backend only,
        ``dlc_opts`` (model_type / precision / device) to DLC only. Both are in
        the cache key because each SDK fixes them at construction, so a change
        has to produce a different model rather than be quietly ignored.
        ``colour_mode`` likewise changes what the network is fed."""
        if probe_frame is None:
            logger.error("PoseSink.configure_model: probe_frame required")
            return None
        # Pose models (DLC, SLEAP) need 3-channel colour input.  Refuse to
        # load against a 2D grayscale probe so the GUI gets a clear error
        # instead of silently running on a camera that's in the wrong mode.
        if getattr(probe_frame, "ndim", 0) == 2:
            logger.error(
                "PoseSink.configure_model: probe frame is 2D grayscale "
                "(shape=%s). Pose tracking (%s) needs a colour camera, "
                "switch the camera out of grayscale mode and retry.",
                getattr(probe_frame, "shape", None), tracker_type,
            )
            return None
        # Every box is fitted to one shape, so the key, and therefore the
        # model, is the same whatever box is being configured. The probe
        # travels through the same transform: DLC's TensorFlow engines build
        # their input placeholder from the frame given to ``init_inference``,
        # so a probe of the wrong size would compile a session that the real
        # frames cannot be fed to.
        # How many boxes may ride in one forward pass. One number, because an
        # exported engine's maximum batch is fixed at export: if the sink could
        # submit more than the engine takes, inference fails outright. The sink
        # caps itself at the same value and chunks the rest.
        # ``n_boxes`` is the caller's count of boxes that will run pose, and it
        # is the only one available HERE. The sink's own two sources,
        # ``_enabled`` and ``_box_shapes``, are both empty at Init: pose is
        # enabled afterwards, and shapes are recorded by the frame handler for
        # pose-enabled boxes only. So the engine was built for a batch of 1 on
        # every rig, and because that size is also the ceiling the sink may
        # never exceed, nothing could raise it later. Measured on 16 boxes:
        # 2186 forward passes of batch 1 in 8 s, GPU saturated, half the
        # camera's frames shed, against 236 passes of batch 16 once the
        # engine is built for the rig that exists.
        # Before the batch size is folded in. That number comes from how many
        # boxes the rig has, not from anything the operator set, so it must not
        # decide whether a re-load is needed.
        fingerprint = pose_fingerprint(
            tracker_type=tracker_type, model_path=model_path,
            resize_factor=resize_factor, confidence=confidence,
            body_parts=body_parts, sleap_opts=sleap_opts, dlc_opts=dlc_opts,
            colour_mode=colour_mode, input_mode=input_mode,
            input_wh=input_wh, crop_opts=crop_opts)
        eligible = max(int(n_boxes or 0), self._n_batchable_boxes())
        batch_cap = batch_bucket(eligible)
        if sleap_opts is not None:
            sleap_opts = dict(sleap_opts, max_batch_size=batch_cap)
        if dlc_opts is not None:
            # DLC needs it too, and for a different reason: ``auto`` picks the
            # PyTorch runner over the export on a multi-box rig, because only
            # the PyTorch runner can take a batch at all.
            #
            # The TRUE count, not the bucket. SLEAP is bucketed because the
            # number sizes an engine that gets BUILT, and rounding avoids a
            # multi-minute rebuild when a box is added. DLC builds nothing, so
            # rounding five up to eight would only make the sink and the
            # tracker resolve to different engines.
            dlc_opts = dict(dlc_opts, max_batch_size=eligible)
        # The TRUE box count, not the rounded bucket: the engine crossover is a
        # property of how many boxes there really are, and rounding five up to
        # eight would pick the batched engine for a rig the measurement says
        # should not have it.
        plan = self._input_plan(model_path, tracker_type, input_mode, input_wh,
                                probe_frame, dlc_opts, eligible)
        mode, crop_wh = plan.mode, (plan.size if plan.mode == "crop_track"
                                    else None)
        if mode == "crop_track":
            # The probe must be the shape the real frames will be, or a
            # TensorFlow session compiles its placeholder at the wrong size.
            probe_frame, _win = crop_at(
                probe_frame, probe_frame.shape[1] / 2.0,
                probe_frame.shape[0] / 2.0, crop_wh[0], crop_wh[1])
            canonical = crop_wh
        elif mode == "full":
            canonical = (probe_frame.shape[1], probe_frame.shape[0])
        else:
            canonical = (plan.size
                         or self._resolve_canonical(model_path, probe_frame))
            probe_lb = Letterbox.fit(probe_frame.shape[1], probe_frame.shape[0],
                                     canonical[0], canonical[1])
            probe_frame = probe_lb.apply(probe_frame)
        key = _make_model_key(model_path, probe_frame, resize_factor,
                              tracker_type, sleap_opts, dlc_opts, colour_mode)
        handle = self._backend.get_or_create(
            key, probe_frame=probe_frame, body_parts=body_parts,
            confidence_threshold=confidence, sleap_opts=sleap_opts,
            dlc_opts=dlc_opts, colour_mode=colour_mode,
        )
        if handle is None:
            return None
        # A tracker has ``_body_parts`` and ``get_body_parts()`` and no
        # ``body_parts`` attribute, so assigning one here only ever created a
        # new attribute nobody reads, an update that looked live and was not.
        # The names the sink labels output with are settled below instead.
        with self._lock:
            if canonical != self._canonical or mode != self._input_mode:
                # The transforms describe a mapping onto the OLD shape.
                self._letterboxes.clear()
                self._crop_states.clear()
                self._canonical = canonical
            self._input_mode = mode
            self._crop_wh = crop_wh
            self._crop_opts = dict(crop_opts or {})
            self._handle = handle
            self._max_batch_size = batch_cap
            # What this engine was BUILT for, which later box changes may not
            # exceed. Re-Init is the only thing that raises it.
            self._built_for = batch_cap
            self._warned_outgrew = False
            self._tracker_type = tracker_type
            self._resize = float(resize_factor)
            self._body_parts = self._reconcile_parts(
                body_parts, handle.model.get_body_parts())
            if tuple(self._body_parts) != tuple(body_parts or ()):
                # Describe the model by the names it actually reports, the
                # same names the host will hold once the dialog has read them
                # back. Fingerprinting the ASKED-for list left the two sides
                # describing one model differently the moment the model knew
                # its own keypoints: a SLEAP model reports its six names, the
                # dialog then adopts them, and every later readiness check
                # compared the model's list against the dialog's old one and
                # answered "not loaded" for ever. Tracking ran perfectly the
                # whole time, which is what made it a status bug and not an
                # inference one.
                fingerprint = pose_fingerprint(
                    tracker_type=tracker_type, model_path=model_path,
                    resize_factor=resize_factor, confidence=confidence,
                    body_parts=self._body_parts, sleap_opts=sleap_opts,
                    dlc_opts=dlc_opts, colour_mode=colour_mode,
                    input_mode=input_mode, input_wh=input_wh,
                    crop_opts=crop_opts)
            self._confidence = float(confidence)
            # What this model was built from, kept so the host can ask whether
            # a box still wants exactly this model without describing it again.
            self._fingerprint = fingerprint
            # Reset the one-shot "no handle" warning so a subsequent
            # eviction will warn the user again.
            self._warned_no_handle = False
        return handle

    def fingerprint(self) -> Optional[tuple]:
        """The settings the loaded model was built from, or ``None``.

        Compare with :func:`pose_fingerprint` built from a box config to decide
        whether that box needs an init. Both come from the same function, so
        they cannot describe the same model differently.
        """
        with self._lock:
            return self._fingerprint

    def _input_plan(self, model_path: str, tracker_type: str, asked: str,
                    input_wh, probe_frame, dlc_opts, n_boxes: int = 1):
        """How this frame becomes the model's input.

        The decision itself lives in ``input_policy`` so the offline retracker
        reaches the same answer from the same model; this only supplies what
        the sink knows, the frame size and, for DLC, which engine will run,
        since a fixed-shape export and a fully-convolutional PyTorch runner
        want different treatment of the same folder.
        """
        from source.video.tracking import input_policy

        engine = ""
        if str(tracker_type).lower() in ("dlc", "deeplabcut"):
            from source.video.tracking import dlc_engine
            # The same resolver the tracker will use, with the same requested
            # type, so the two cannot disagree about which engine is running.
            # The box count reaches the resolver too: ``auto`` prefers the
            # PyTorch runner on a multi-box rig, because only it can take a
            # batch. See dlc_engine.BATCH_CROSSOVER_BOXES.
            engine = dlc_engine.resolve(
                model_path, (dlc_opts or {}).get("dlc_model_type", "auto"),
                n_boxes=n_boxes)[0]

        plan = input_policy.plan(
            tracker_type, model_path,
            probe_frame.shape[1], probe_frame.shape[0],
            engine=engine, asked=asked or "auto",
            asked_size=tuple(input_wh) if input_wh else None)
        logger.info("Box pose input: %s", plan.describe())
        return plan

    def set_crop_opts(self, *, conf_min: Optional[float] = None,
                      good_min: Optional[int] = None,
                      reacquire: Optional[bool] = None) -> None:
        """Update the window's steering without rebuilding the model.

        These three change how a result is read, not what the network is fed,
        so they are live: an operator watching an animal can tighten the
        confidence floor and see the effect on the next frame.
        """
        with self._lock:
            if conf_min is not None:
                self._crop_opts["conf_min"] = float(conf_min)
            if good_min is not None:
                self._crop_opts["good_min"] = int(good_min)
            if reacquire is not None:
                self._crop_opts["reacquire"] = bool(reacquire)
            for st in self._crop_states.values():
                if conf_min is not None:
                    st.conf_min = float(conf_min)
                if good_min is not None:
                    st.good_min = int(good_min)
                if reacquire is not None:
                    st.reacquire = bool(reacquire)

    def _crop_hint(self, setup_id: int):
        """Where the motion filter thinks the animal will be, if anything knows.

        At speed the Kalman prediction beats the last centroid, which is
        exactly when a following window is most likely to be left behind. The
        filter already exists for the latency-compensated zone lookup, so this
        reuses it rather than adding a second motion model.
        """
        enhancer = self._enhancers.get(setup_id)
        if enhancer is None:
            return None
        try:
            point = enhancer.predict_ahead(0.0)
        except Exception as e:
            logger.debug("crop hint unavailable (box %d): %s", setup_id, e)
            return None
        if not point or len(point) < 2 or point[0] is None:
            return None
        return (float(point[0]), float(point[1]))

    def _crop_state(self, setup_id: int) -> BoxCropState:
        """This box's window, created on first use from the model's size."""
        st = self._crop_states.get(setup_id)
        w, h = self._crop_wh or (0, 0)
        if st is None or (st.w, st.h) != (w, h):
            o = self._crop_opts
            st = BoxCropState(
                w, h,
                conf_min=float(o.get("conf_min", 0.20)),
                good_min=int(o.get("good_min", 3)),
                reacquire=bool(o.get("reacquire", True)))
            self._crop_states[setup_id] = st
        return st

    def _n_batchable_boxes(self) -> int:
        """How many boxes could appear in one batch.

        Enabled boxes are the ones that will be inferred, but a box can be
        enabled after the model is built, so the boxes already delivering
        frames count too, every connected camera feeds this sink whether or
        not pose is on for it. Overestimating costs a larger engine; under-
        estimating costs a rebuild, so the union is the cheap direction.
        """
        with self._lock:
            return max(1, len(set(self._enabled) | set(self._box_shapes)))

    def _resolve_canonical(self, model_path: str, probe_frame) -> Tuple[int, int]:
        """The one input shape every box is fitted to, as ``(w, h)``.

        The model's own crop size wins when it states one: that is the size it
        was trained at, and a SLEAP engine exported for it has that size baked
        in. Failing that, the largest shape across the boxes seen so far,
        large enough that no box is upscaled past its own resolution, rounded
        up to the network's stride so a model that downsamples by that factor
        divides evenly.

        Fixed at Init: a box that first delivers frames afterwards is fitted
        into the existing shape rather than widening it, because widening it
        would mean rebuilding the model underneath boxes that are already
        running. Re-Init recomputes it.
        """
        ph, pw = probe_frame.shape[:2]
        stride = None
        try:
            from source.video.tracking.model_config import ModelInfo
            info = ModelInfo.read(model_path)
            stride = info.max_stride
            if info.input_w and info.input_h:
                return (int(info.input_w), int(info.input_h))
            if info.crop_size:
                side = int(info.crop_size)
                return (side, side)
        except Exception as e:
            logger.debug("canonical shape: model config unreadable (%s)", e)
        with self._lock:
            shapes = list(self._box_shapes.values())
        shapes.append((pw, ph))
        return canonical_shape(shapes, stride) or (pw, ph)

    def _letterbox_for(self, setup_id: int, w: int, h: int) -> Optional[Letterbox]:
        """Cached transform fitting box ``setup_id``'s frame to the canonical
        shape. ``None`` before a model is configured; there is no shape to
        fit to yet, and the caller passes the frame through untouched.
        """
        if self._canonical is None:
            return None
        lb = self._letterboxes.get(setup_id)
        # Rebuild when either end moved: the ROI was redrawn, or a re-Init
        # changed the canonical shape. A transform describing the wrong shape
        # would put keypoints somewhere the animal is not.
        if (lb is None or lb.src_w != w or lb.src_h != h
                or (lb.dst_w, lb.dst_h) != self._canonical):
            lb = Letterbox.fit(w, h, self._canonical[0], self._canonical[1])
            self._letterboxes[setup_id] = lb
            if not lb.is_identity:
                logger.info(
                    "Box %d: pose input %dx%d fitted to %dx%d (scale %.3f, "
                    "pad %d,%d) so all boxes share one model.",
                    setup_id, w, h, lb.dst_w, lb.dst_h, lb.scale,
                    lb.pad_x, lb.pad_y)
        return lb

    def current_signature(self) -> Optional[Tuple]:
        """Stable hash over the SETTINGS the current loaded model was
        built from, used by the host to detect "settings changed since
        last init" at Record click.

        Returns ``(model_path, tracker_type, resize, confidence,
        body_parts_tuple, sleap_sig)``, ``sleap_sig`` is ``""`` for DLC,
        or ``None`` when no model has been configured yet.

        Excluded from the signature on purpose:
          * frame width/height, camera-derived; not a user setting
          * pose_n_instances, runtime knob, no rebuild needed
          * online_tracking_enabled / push_*_to_mcu, gates, not model state

        Pair this with ``pose_subsystem.pose_signature_for(cfg)`` for
        the cfg-side comparator.
        """
        with self._lock:
            handle = self._handle
            if handle is None:
                return None
            model_path, _w, _h, resize, tracker_type, sig = handle.key
            confidence = float(self._confidence)
            body_parts = tuple(self._body_parts or ())
        return (str(model_path), str(tracker_type), float(resize),
                confidence, body_parts, str(sig))

    def enable_for_box(self, setup_id: int, *, zone_lookup: Any = None) -> None:
        with self._lock:
            self._enabled[setup_id] = True
            if zone_lookup is not None:
                self._zone_lookup[setup_id] = zone_lookup
        self._refresh_batch_cap()

    def disable_for_box(self, setup_id: int) -> None:
        with self._lock:
            self._enabled.pop(setup_id, None)
            self._zone_lookup.pop(setup_id, None)
            self._prev_centroid.pop(setup_id, None)
            self._enhancers.pop(setup_id, None)
            self._gap_fillers.pop(setup_id, None)
            self._prev_fill_ns.pop(setup_id, None)
            self._prev_capture_ns.pop(setup_id, None)
            self._rotation_enabled.pop(setup_id, None)
            self._rotation_pair.pop(setup_id, None)
            self._rotation_state.pop(setup_id, None)
        self._refresh_batch_cap()

    def _refresh_batch_cap(self) -> None:
        """Re-decide how many boxes may ride in one forward pass.

        Bounded by ``_built_for``, the size the loaded engine was actually
        constructed for. That is not a soft preference: an exported engine's
        batch axis is fixed when it is built, so submitting more fails outright
        rather than running slower, and rebuilding is a multi-minute TensorRT
        operation that cannot happen mid-session. A rig that grows past its
        engine keeps chunking at the built size and is told a re-Init would
        build a larger one.

        The reason this can raise the cap at all is that ``configure_model``
        now learns the box count from its caller, so the engine is built for
        the rig that exists rather than for the empty one the sink could see at
        Init. See the comment there.
        """
        from source.video.tracking.sleap_export import batch_bucket

        eligible = self._n_batchable_boxes()
        want = batch_bucket(eligible)
        with self._lock:
            built = self._built_for
            if built > 0 and want > built:
                if not self._warned_outgrew:
                    self._warned_outgrew = True
                    logger.warning(
                        "PoseSink: %d boxes now run pose but the engine was "
                        "built for a batch of %d, so it keeps chunking at %d. "
                        "Re-initialise tracking to build one for this rig.",
                        eligible, built, built)
                want = built
            if want == self._max_batch_size:
                return
            was, self._max_batch_size = self._max_batch_size, want
        logger.info("PoseSink: batch cap %d → %d (%d box(es) eligible)",
                    was, want, eligible)

    def set_centroid_body_part(self, setup_id: int, body_part: str) -> None:
        """Pick which keypoint defines the synthetic centroid for this box.

        ``"centroid"`` (default) keeps the first-confident-keypoint
        heuristic. Any other value names a keypoint whose coords become
        the centroid; when it's below confidence, cx,cy are None and
        ``location`` resolves to ``None`` (writes ``"na"``) instead of
        reporting another keypoint's zone.

        Mirror of the operator's "Zone-change body part" picker
        (``TrackingConfig.zone_change_body_part``). Wired by
        ``Pipeline.apply_tracking_config``.
        """
        bp = (body_part or "centroid").strip() or "centroid"
        with self._lock:
            if bp == "centroid":
                self._centroid_body_part.pop(setup_id, None)
            else:
                self._centroid_body_part[setup_id] = bp

    def set_rotation(self, setup_id: int, *,
                     enabled: bool,
                     kp_pair: Tuple[str, str] = DEFAULT_ROTATION_PAIR) -> None:
        """Enable/disable rotation estimation for one box and pick which
        two keypoints define the vector (tail→head). Off by default.

        Result rotation θ (radians, image plane) is attached to the raw
        pose dict under ``"_rotation_rad"`` so subscribers can read it
        without changing the PoseResultCallback signature. Value is
        ``None`` when below confidence or when the EMA has no history.
        """
        with self._lock:
            if enabled:
                self._rotation_enabled[setup_id] = True
                self._rotation_pair[setup_id] = tuple(kp_pair)
                self._rotation_state.setdefault(setup_id, {})
            else:
                self._rotation_enabled.pop(setup_id, None)
                self._rotation_pair.pop(setup_id, None)
                self._rotation_state.pop(setup_id, None)

    @staticmethod
    def _reconcile_parts(asked, reported):
        """Which keypoint names to label this model's output with.

        The MODEL is authoritative. The list arriving from the dialog is what
        was on screen when the operator pressed Init, and it can be stale, or,
        for a toolkit whose config the dialog cannot read, simply wrong.
        Letting it win silently relabels the network's channels: a six-node
        SLEAP model configured as ``["center"]`` reported one column named
        ``center`` that was really keypoint 0, the snout.

        The dialog's list is used only when the model reports nothing, which is
        the blob tracker and any backend that cannot introspect itself.
        """
        reported = list(reported or [])
        asked = list(asked or [])
        if reported and asked and len(asked) != len(reported):
            logger.warning(
                "keypoint names disagree: the model reports %d (%s) and the "
                "tracking dialog holds %d (%s). Using the model's, reopen "
                "the dialog to refresh what it shows.",
                len(reported), ", ".join(reported[:8]),
                len(asked), ", ".join(asked[:8]))
        return reported or asked

    def model_info(self):
        """``(body_parts, confidence_threshold)`` of the active model,
        the read the GUI overlay/annotation config needs, without
        touching sink privates."""
        return list(self._body_parts), float(self._confidence)

    def resolved_backend(self) -> str:
        """The loaded model's actual runtime/precision, or "" when none."""
        h = self._handle
        if h is None:
            return ""
        try:
            return str(h.model.resolved_backend)
        except Exception:
            return ""

    def set_confidence(self, value: float) -> None:
        """Change the keypoint confidence cutoff without rebuilding the model.

        Confidence is a post-inference filter applied here, not a property of
        the network; it is deliberately absent from the model cache key. So it
        can be changed on a running box, and it must be: routing it through a
        model rebuild would cost a drain, a close and a fresh load (minutes,
        for a TensorRT engine) to move a threshold the loaded model never sees.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            logger.debug("PoseSink.set_confidence: ignoring %r", value)
            return
        v = min(1.0, max(0.0, v))
        with self._lock:
            if self._confidence == v:
                return
            self._confidence = v
        logger.info("PoseSink: confidence -> %.2f (no model rebuild)", v)

    def set_enhancer(self, setup_id: int, enhancer: Any) -> None:
        """Attach a TrackingEnhancer for latency-compensated zone lookup.

        The same enhancer object that ``TrackerManager.set_enhancer``
        receives can be passed here so the KF state is shared (or two
        independent enhancers can be used for the same box if blob and
        pose run side-by-side).  Pass ``None`` to detach.
        """
        with self._lock:
            if enhancer is None:
                self._enhancers.pop(setup_id, None)
            else:
                self._enhancers[setup_id] = enhancer

    def set_latency_budget(self, budget: Optional[LatencyBudget]) -> None:
        """Attach the pipeline's timing spine (observational)."""
        self._latency = budget

    def is_enabled(self, setup_id: int) -> bool:
        return self._enabled.get(setup_id, False)

    def has_model(self) -> bool:
        return self._handle is not None

    def on_result(self, cb: PoseResultCallback) -> Callable[[], None]:
        with self._lock:
            self._on_result = [*self._on_result, cb]

        def _unsub():
            with self._lock:
                self._on_result = [c for c in self._on_result if c is not cb]
        return _unsub

    def on_failed(self, cb: PoseFailedCallback) -> Callable[[], None]:
        """Subscribe to silent-failure events (empty-streak threshold).

        Pipeline.controller wires this through to a Qt signal that the
        main window connects to its status bar, so a model returning
        all-zero confidence (grayscale frame, GPU OOM, post-eviction)
        lights up a red banner within ~0.1 s.
        """
        with self._lock:
            self._on_failed = [*self._on_failed, cb]

        def _unsub():
            with self._lock:
                self._on_failed = [c for c in self._on_failed if c is not cb]
        return _unsub

    # ── Sink worker --------------------------------------------------
    #
    # PoseSink overrides _run (not process) so it can drain all pending
    # boxes per wake cycle and submit them as one batched inference call
    # (DeepStream nvstreammux/nvinfer pattern): with 8 boxes batched the
    # GPU runs one forward pass instead of eight.

    def _run(self, widx: int = 0) -> None:
        wake = self._wakes[widx]
        while self._running:
            try:
                bid = wake.get(timeout=0.5)
            except _queue.Empty:
                continue
            if bid is None:
                break

            # Settle window: linger briefly to absorb wake tokens from a
            # producer iterating multiple boxes back-to-back. Trades
            # ≤BATCH_SETTLE_S latency for true multi-box batching.
            deadline = time.monotonic() + self.BATCH_SETTLE_S
            while True:
                # Stop as soon as the batch cannot get any fuller. Without
                # this the window is paid IN FULL on every cycle, not just
                # when a box is late: once the queued tokens are drained the
                # get() below blocks until the deadline, so the wait is a flat
                # tax on every box in the batch rather than the "≤" the
                # comment above promises. When every enabled box already has a
                # frame waiting there is nothing left to absorb.
                with self._lock:
                    want = sum(1 for on in self._enabled.values() if on)
                    if want and len(self._pending) >= want:
                        break
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    extra = self._wake.get(timeout=timeout)
                except _queue.Empty:
                    break
                if extra is None:
                    self._running = False
                    break

            # Drain ALL boxes that have a pending frame.
            batch: List[Tuple[int, BoxFrame]] = []
            with self._lock:
                for setup_id, dq in list(self._per_box.items()):
                    if dq:
                        # DROP_NEWEST means dq has at most 1 frame.
                        batch.append((setup_id, dq.popleft()))
                        self._pending.discard(setup_id)
            self._purge_wake_tokens({sid for sid, _ in batch})

            if not batch:
                continue
            try:
                self._dispatch_batch(batch)
            except Exception as e:
                logger.exception("PoseSink batch dispatch error: %s", e)

    def _purge_wake_tokens(self, drained: set) -> None:
        """Drop leftover wake tokens ONLY for boxes just drained.

        A token for a box whose frame arrived between the drain snapshot
        and this purge must survive, eating it left that frame waiting
        out the full 0.5 s ``get()`` timeout before the next drain (up to
        half a second of added latency per occurrence under multi-camera
        load; latency, not loss). The ``None`` shutdown sentinel is
        re-queued, never swallowed.
        """
        survivors = []
        try:
            while True:
                tok = self._wake.get_nowait()
                if tok is None or tok not in drained:
                    survivors.append(tok)
        except _queue.Empty:
            pass
        for tok in survivors:
            try:
                self._wake.put_nowait(tok)
            except _queue.Full:
                break

    def _dispatch_batch(self, batch: List[Tuple[int, BoxFrame]]) -> None:
        # Counted here for the same reason as TrackerSink._run_one: this sink
        # overrides the worker loop (it batches across boxes), so the
        # base-class counter never sees these frames.
        self.n_processed += len(batch)
        """Group by frame shape, fire one batched inference per shape.

        DLC requires identical input shape across the batch. Multi-box
        setups usually share one ROI size (one group); when ROIs differ
        each shape group is its own batched call."""
        with self._lock:
            handle = self._handle
            enabled = dict(self._enabled)

        # Filter to enabled boxes first. Every connected camera feeds the
        # pose sink unconditionally, so frames for non-enabled boxes are
        # normal and must not trip the "no model handle" warning.
        keep: List[Tuple[int, BoxFrame]] = []
        for setup_id, frame in batch:
            if not enabled.get(setup_id, False):
                continue
            keep.append((setup_id, frame))
        if _POSE_DEBUG:
            seen = [bid for bid, _ in batch]
            kept = [bid for bid, _ in keep]
            logger.info(
                "PoseSink dispatch: arrivals=%s enabled=%s kept=%s "
                "(handle=%s)",
                seen, sorted(enabled.keys()), kept,
                "set" if handle is not None else "None",
            )
        if not keep:
            return

        if handle is None:
            # Pose enabled for ≥1 box but no model loaded, one-shot
            # warning per lost-handle episode (reset when configure_model
            # succeeds). Camera-only / tracking-off sessions have no
            # enabled boxes, so they're filtered out above.
            with self._lock:
                if not getattr(self, "_warned_no_handle", False):
                    logger.warning(
                        "PoseSink: %d frames arrived for pose-enabled box(es) "
                        "but no model handle is loaded. Either "
                        "configure_pose_model() was never called, returned "
                        "None, or the model was evicted. Pose results will be "
                        "empty until reconfigure.",
                        len(keep),
                    )
                    self._warned_no_handle = True
            return

        # Group by frame shape using the RGB view so the backend never
        # cvtColors. CameraFrame.image_rgb does the whole-frame BGR→RGB
        # once; each BoxFrame in a shared-camera group slices the same
        # buffer (O(1) numpy view).
        #
        # Boxes are fitted to the canonical shape first, so differently-cropped
        # ROIs land in ONE group and one batched call, the grouping remains
        # because it is what guarantees that, not because shapes still differ.
        groups: Dict[tuple, List[Tuple[int, BoxFrame]]] = {}
        for setup_id, frame in keep:
            h, w = frame.image.shape[:2]
            self._box_shapes[setup_id] = (w, h)
            if self._input_mode == "crop_track" and self._crop_wh:
                # Every box cuts the SAME size window, so the batch is
                # homogeneous by construction, the crop does the letterbox's
                # shape-sharing job and keeps native pixel scale as well.
                shp = (self._crop_wh[1], self._crop_wh[0])
            elif self._input_mode == "full":
                self._input_transform[setup_id] = None
                shp = tuple(frame.image.shape[:2])
            else:
                lb = self._letterbox_for(setup_id, w, h)
                self._input_transform[setup_id] = lb
                shp = ((lb.dst_h, lb.dst_w) if lb is not None
                       else tuple(frame.image.shape[:2]))
            groups.setdefault(shp, []).append((setup_id, frame))

        for group in groups.values():
            # cam_frame_id + capture_host_ns are per-box.  The on_done
            # closure carries both so _on_pose_done can run the
            # enhancer's KF correction with the actual capture
            # timestamp (not the inference-completion time).
            id_map = {setup_id: fr.cam_frame_id for setup_id, fr in group}
            ts_map = {setup_id: fr.capture_host_ns for setup_id, fr in group}
            # Timing spine: poll instant per box, for poll→inference-done.
            poll_map = {setup_id: fr.poll_host_ns for setup_id, fr in group}
            # Skip boxes where image_rgb is None (grayscale source);
            # _to_model_input would have returned None too.
            # The transform THIS frame was cut with, carried per box beside the
            # frame id and the timestamps, for exactly the same reason they
            # are. Inference completes on the executor thread, so by the time a
            # result comes back the box has usually dispatched another frame;
            # a single ``_input_transform[setup_id]`` slot would by then hold
            # the NEWER crop window, and the keypoints would be un-mapped with
            # it. Under a letterbox that is harmless, the transform never
            # changes, but a following window moves with the animal, so every
            # keypoint landed a few pixels behind, which is precisely what a
            # slightly-offset overlay looks like.
            tf_map = {}
            # The grayscale frame the smoother's optical flow needs, carried
            # per box like the transform above: by the time a result comes
            # back the box has usually dispatched another frame, so a single
            # slot would hold the wrong image.
            gray_map = {}
            items_full = []
            for setup_id, fr in group:
                gray_map[setup_id] = fr.image_gray
                rgb = fr.image_rgb
                if rgb is None:
                    continue
                if self._input_mode == "crop_track" and self._crop_wh:
                    st = self._crop_state(setup_id)
                    fh, fw = rgb.shape[:2]
                    cx, cy = st.centre_for(fw, fh, self._crop_hint(setup_id))
                    rgb, window = crop_at(rgb, cx, cy, st.w, st.h)
                    self._input_transform[setup_id] = window
                    tf_map[setup_id] = window
                else:
                    lb = self._input_transform.get(setup_id)
                    if lb is not None and not lb.is_identity:
                        rgb = lb.apply(rgb)
                    tf_map[setup_id] = lb
                items_full.append((setup_id, rgb))
            if not items_full:
                continue
            # Rotate the start so chunked dispatches are fair across
            # boxes, a different ``cap`` of boxes leads each round.
            n = len(items_full)
            rot = self._chunk_rotation % n if n > 0 else 0
            items_full = items_full[rot:] + items_full[:rot]
            self._chunk_rotation = (self._chunk_rotation + 1) % max(1, n)

            def _on_done(bid: int, pose: dict,
                         _id_map=id_map, _ts_map=ts_map,
                         _poll_map=poll_map, _tf_map=tf_map,
                         _gray_map=gray_map) -> None:
                # Record poll→inference-done before result post-processing,
                # so the sample reflects inference time, not fan-out time.
                if self._latency is not None:
                    self._latency.record_from_ns(
                        "poll_to_infer", _poll_map.get(bid))
                self._on_pose_done(bid, _id_map.get(bid, 0), pose,
                                   capture_host_ns=_ts_map.get(bid, 0),
                                   transform=_tf_map.get(bid, _MISSING),
                                   frame_gray=_gray_map.get(bid))

            # Chunk by ``max_batch_size`` to fit GPU memory on limited
            # devices (Jetson Nano ≈4, Xavier NX ≈8, desktop unlimited).
            # 0 = unlimited. Trailing chunks may hit the backend inflight
            # gate; the rotation above keeps it fair over time.
            chunk = self._max_batch_size if self._max_batch_size > 0 else len(items_full)
            # A model that has already refused a batch says so, and is then
            # never handed another. Without this the sink kept submitting the
            # cap: a DLC export whose batch axis is fixed at 1 raised, logged
            # and fell back on EVERY frame, 290 times in one 8-second run,
            # costing a fifth of its throughput. The tracker's own fallback is
            # correct but it is a last resort, not a per-frame strategy.
            if not getattr(handle.model, "batched", True):
                chunk = 1
            chunk = max(1, chunk)
            for start in range(0, len(items_full), chunk):
                items = items_full[start:start + chunk]
                self._backend.submit_batch(handle, items, _on_done)

    # ── Inference completion (called on backend worker) ----------------

    def _on_pose_done(self, setup_id: int, cam_frame_id: int,
                      pose: dict,
                      capture_host_ns: int = 0,
                      transform: Any = _MISSING,
                      frame_gray=None) -> None:
        # Inference-done instant, start of the infer→push stage the MCU pusher
        # closes. Stamped here (not at notify) so it spans this post-processing
        # too, matching the "result → coord queued" stage definition.
        # Inference-done instant, stamped first so it spans this whole
        # post-processing, matching the "result → coord queued" stage the MCU
        # pusher closes.
        infer_done_ns = host_clock.host_ns()
        # Back to this box's own frame pixels before anything reads a
        # coordinate. Zones, triggers, the MCU push and the recorded frame log
        # all speak box pixels; the model spoke canonical ones. Identity
        # transforms return the dict untouched.
        # The transform THIS frame was prepared with, handed down from the
        # dispatch. Falling back to the current slot only when the caller did
        # not supply one, which is the direct-call path used by tests.
        if transform is _MISSING:
            transform = self._input_transform.get(setup_id)
        if transform is not None:
            pose = transform.pose_to_source(pose)
        if self._input_mode == "crop_track":
            # Steering happens on BOX coordinates, after the inverse, the
            # window has to be told where the animal is in the frame, not
            # where it was inside the previous cut.
            st = self._crop_states.get(setup_id)
            if st is not None:
                st.update(pose)
        cfg = self._snapshot_box_config(setup_id)
        # Smoothing, and how long it took. Both are per row in the session
        # file: an operator asking why a frame was late needs to see which
        # stage spent the time, and the filter runs on this thread, inside
        # the frame budget, so it can be the answer.
        filter_start_ns = host_clock.host_ns()
        pose = self._fill_pose_gaps(setup_id, pose, capture_host_ns, frame_gray)
        filter_ms = (host_clock.host_ns() - filter_start_ns) / 1e6
        parts, pose_array = self._build_pose_array(setup_id, pose)

        if _POSE_DEBUG:
            logger.info(
                "PoseSink result: box=%d cam_frame_id=%s parts=%d confs=%s",
                setup_id, cam_frame_id, len(pose_array),
                [round(pt[2], 2) for pt in pose_array],
            )

        self._note_empty_result(setup_id, pose_array, cfg["failed_subs"])
        cx, cy = self._pick_centroid(pose, pose_array, cfg["picked_bp"])
        zone_cx, zone_cy = self._forecast_centroid(
            setup_id, cfg["enhancer"], pose, cx, cy, capture_host_ns)
        location, zones_by_body_part = self._resolve_zones(
            setup_id, cfg["zone_lookup"], parts, pose_array,
            cx, cy, zone_cx, zone_cy)
        self._apply_rotation(setup_id, pose, cfg)

        # Speed, shared helper with TrackerSink.
        speed = 0.0
        if cx is not None and cy is not None:
            speed = track_speed(self._lock, self._prev_centroid,
                                setup_id, cx, cy)

        forecast_coords = self._forecast_coords_for_push(
            setup_id, cx, cy, zone_cx, zone_cy)

        # Two numbers the session file carries per row, both measured rather
        # than derived: how long after this frame was CAPTURED its pose
        # existed, and how much of that was the smoother. Ages that had to be
        # reconstructed by subtracting two timestamp columns are now one
        # value each, in milliseconds, and the operator does no arithmetic.
        pose_lag_ms = ((infer_done_ns - capture_host_ns) / 1e6
                    if capture_host_ns and infer_done_ns else None)
        notify_subscribers(cfg["subs"], setup_id, cam_frame_id, pose_array,
                           location, speed, zones_by_body_part, pose,
                           capture_host_ns=capture_host_ns,
                           forecast_coords=forecast_coords,
                           infer_done_ns=infer_done_ns,
                           pose_lag_ms=pose_lag_ms, filter_ms=filter_ms,
                           log=logger,
                           label=f"PoseSink result (box={setup_id})")

    # ── _on_pose_done steps ───────────────────────────────────────────

    def _snapshot_box_config(self, setup_id: int) -> dict:
        """Read the rarely-changing per-box config in ONE lock pass.

        It mutates only on apply_tracking_config / enable / disable, so taking
        it once keeps the per-frame path off the lock; the genuinely-mutating
        state (streak counter, prev-timestamp swap) re-locks briefly later.
        """
        with self._lock:
            rot_enabled = self._rotation_enabled.get(setup_id, False)
            return {
                "zone_lookup": self._zone_lookup.get(setup_id),
                "enhancer": self._enhancers.get(setup_id),
                "subs": self._on_result,
                "failed_subs": self._on_failed,
                "picked_bp": self._centroid_body_part.get(setup_id),
                "rot_enabled": rot_enabled,
                "rot_pair": self._rotation_pair.get(setup_id, DEFAULT_ROTATION_PAIR),
                "rot_state": (self._rotation_state.get(setup_id)
                              if rot_enabled else None),
            }

    def _build_pose_array(self, setup_id: int, pose: dict) -> tuple:
        """``(body_part_names, [[x, y, conf], ...])`` in body_parts order.

        One comprehension, to avoid a per-keypoint list realloc. A body part
        the model did not report becomes a zero row so the array stays aligned
        with the names, downstream code zips the two together.
        """
        try:
            parts = self._body_parts or list(pose.keys())
            pose_get = pose.get  # bind once
            return parts, [
                [round(float(pt[0]), 2),
                 round(float(pt[1]), 2),
                 round(float(pt[2]) if len(pt) > 2 else 1.0, 3)]
                if (pt := pose_get(name)) is not None and len(pt) >= 2
                else [0.0, 0.0, 0.0]
                for name in parts
            ]
        except Exception as e:
            logger.warning("PoseSink: pose_array build error (box=%d): %s",
                           setup_id, e)
            return [], []

    #: Quietest a box may be told its pose is empty. Long enough that a
    #: flapping model does not fill the log, short enough that an operator who
    #: looks after starting a session still sees it.
    _EMPTY_WARN_EVERY_NS = 30 * 1_000_000_000
    #: Consecutive empty results before the box is called stalled,
    #: ~10 s at 30 fps. Long enough that an empty arena, or an animal
    #: briefly out of view, never trips it.
    _EMPTY_STALL_FRAMES = 300

    def _note_empty_result(self, setup_id: int, pose_array: list,
                           failed_subs) -> None:
        """Watch for every keypoint at zero confidence, the signature of a
        pose pipeline that is broken but silent (model never initialised,
        camera switched to grayscale, GPU OOM).

        The threshold is a DURATION, not three frames. Three consecutive
        empties is 0.1 s at 30 fps, which is an empty arena, not a stall: with
        a following crop window on a scene with no animal, ~70% of results
        come back empty by design, because the window is searching. The old
        threshold fired one second after a model had loaded perfectly and
        blamed it for failing to initialise, a false alarm that teaches the
        operator to ignore the true ones.

        So: warn once a box has produced nothing for ``_EMPTY_STALL_FRAMES``
        in a row (~10 s at 30 fps), then at ten times that. A model that has
        stopped working stays empty; an arena with nothing in it recovers the
        moment the animal appears.
        """
        all_zero = bool(pose_array) and all(pt[2] <= 0.0 for pt in pose_array)
        with self._lock:
            streak = self._empty_result_streak.get(setup_id, 0)
            streak = streak + 1 if all_zero else 0
            self._empty_result_streak[setup_id] = streak
        if not (all_zero and streak in (self._EMPTY_STALL_FRAMES,
                                        self._EMPTY_STALL_FRAMES * 10)):
            return
        # A streak threshold alone throttles a PERSISTENT failure and not a
        # flapping one: a single frame with any confidence resets the count, so
        # a model that finds something once a second climbs back to 3 and warns
        # again, forever. Seen on a camera with no signal, hundreds of
        # identical lines in one run, which is how a log stops being read. The
        # floor keeps the fast first warning and drops the repeats.
        now = host_clock.host_ns()
        with self._lock:
            last = self._empty_warned_ns.get(setup_id, 0)
            if last and (now - last) < self._EMPTY_WARN_EVERY_NS:
                return
            self._empty_warned_ns[setup_id] = now
        # Name the backend that is actually running. The old text said "DLC"
        # and "DLCLiveTracker traceback" whatever the backend was, so a SLEAP
        # model was reported as a DLC failure, and every cause it listed was
        # one that had already been ruled out by the model loading.
        # ``_tracker_type`` always exists (it defaults to "dlc" and is set by
        # configure_model). Reading it directly rather than through a wider
        # try: the first version reached for an optional handle first and one
        # AttributeError threw away the answer that was already in hand.
        backend = str(getattr(self, "_tracker_type", "") or "")
        who = f"{backend} " if backend else ""
        seconds = streak / 30.0
        reason = (
            f"{who}pose has returned nothing for {streak} frames "
            f"(~{seconds:.0f}s): every body part at confidence 0. If the "
            f"arena is empty this is expected and will clear when the animal "
            f"is in view. If it does not clear, check that the camera is not "
            f"in grayscale, and the log for a tracker error."
        )
        logger.warning("PoseSink box %d: %s", setup_id, reason)
        notify_subscribers(failed_subs, setup_id, reason, streak,
                           log=logger,
                           label=f"PoseSink failed (box={setup_id})")

    def _pick_centroid(self, pose: dict, pose_array: list,
                       picked_bp: Optional[str]) -> tuple:
        """The animal's position, in priority order:

        1. the operator-picked body part (tracking dialog): and *only* that
           part, so below its confidence there is no centroid at all;
        2. a ``centroid`` key the backend supplied itself;
        3. the first body part above confidence.
        """
        try:
            if picked_bp:
                kp = pose.get(picked_bp)
                if (kp is not None and len(kp) >= 2
                        and (len(kp) < 3 or float(kp[2]) >= self._confidence)):
                    return float(kp[0]), float(kp[1])
                return None, None
            if "centroid" in pose:
                ct = pose["centroid"]
                return float(ct[0]), float(ct[1])
            for pt in pose_array:
                if pt[2] >= self._confidence:
                    return pt[0], pt[1]
        except Exception:
            pass
        return None, None

    def set_gap_fill(self, setup_id: int, max_gap_frames: int,
                     min_conf: float = 0.10) -> None:
        """Turn per-keypoint filtering and gap fill on for one box.

        ``max_gap_frames`` of 0 turns it off and drops the box's filters, so
        an operator who wants the raw network output gets exactly that.
        """
        from source.video.tracking.smoothing import PoseGapFiller

        with self._lock:
            if max_gap_frames and int(max_gap_frames) > 0:
                self._gap_fillers[setup_id] = PoseGapFiller(
                    max_gap=int(max_gap_frames), min_conf=float(min_conf))
                logger.info("Box %d: pose gap fill on, up to %d frames",
                            setup_id, int(max_gap_frames))
            else:
                self._gap_fillers.pop(setup_id, None)

    def _fill_pose_gaps(self, setup_id: int, pose: dict,
                        capture_host_ns: int, frame_gray=None) -> dict:
        """Smooth each keypoint and carry it across a brief disappearance.

        Runs BEFORE the pose array is built, so the overlay, the zone lookup,
        the triggers and the recorded row all see one answer rather than each
        deciding for itself what a missing part means.

        The parts that were predicted rather than measured are listed under
        ``_filled_parts``. The underscore is the established marker for a
        metadata key in this dict, which every consumer already skips, so the
        fact travels to the recorder without changing a single callback
        signature - and nothing downstream can mistake a coasted point for a
        measured one without ignoring a key that says so.
        """
        # ``getattr`` rather than a plain attribute: several call sites build
        # a sink without running __init__ to exercise this one method, and a
        # box with no filter is the normal state anyway.
        filler = (getattr(self, "_gap_fillers", None) or {}).get(setup_id)
        if filler is None or not pose:
            return pose
        with self._lock:
            seen = getattr(self, "_prev_fill_ns", None)
            if seen is None:
                seen = self._prev_fill_ns = {}
            prev = seen.get(setup_id)
            seen[setup_id] = capture_host_ns
        dt_s = 1.0 / 30.0
        if prev and capture_host_ns > prev:
            dt_s = (capture_host_ns - prev) / 1e9
        try:
            filled_pose, filled_names = filler.apply(pose, dt_s, frame_gray)
        except Exception as e:
            logger.debug("Box %d: gap fill failed: %s", setup_id, e)
            return pose
        if filled_names:
            filled_pose["_filled_parts"] = list(filled_names)
        return filled_pose

    def _forecast_centroid(self, setup_id: int, enhancer, pose: dict,
                           cx, cy, capture_host_ns: int) -> tuple:
        """Where the animal will be when a command lands, via the enhancer's
        Kalman filter.

        Only the zone/MCU path uses this. The reported ``cx``/``cy`` stays at
        the current position so the display overlay matches the captured
        frame. Returns the raw centroid unchanged when there is nothing to
        forecast from.
        """
        if not (enhancer is not None and cx is not None and cy is not None
                and capture_host_ns > 0):
            return cx, cy
        try:
            with self._lock:
                prev_ts = self._prev_capture_ns.get(setup_id)
                self._prev_capture_ns[setup_id] = capture_host_ns
            fps = estimate_capture_fps(prev_ts, capture_host_ns)
            # KF measurement-noise confidence, the centroid's own likelihood
            # when the backend supplied one, else neutral.
            conf = 0.9
            try:
                if "centroid" in pose and len(pose["centroid"]) > 2:
                    conf = float(pose["centroid"][2])
            except Exception:
                pass
            # No frame_gray → KF-only; optical flow is off on the pose path.
            enhancer.update(cx, cy, conf, None, capture_host_ns, detected=True)
            fwd = forecast_point(enhancer, self._latency, fps)
            if fwd is None:
                return cx, cy
            fx, fy = fwd
            # Trust only a finite forecast, and clamp to >=0: a near-edge
            # prediction must never land on a negative value, which would
            # collide with the MCU's COORD_LOST=-1.0 sentinel. A non-finite
            # forecast (runaway CA term / corrupted state) falls back to raw.
            if math.isfinite(fx) and math.isfinite(fy):
                return max(0.0, fx), max(0.0, fy)
        except Exception as e:
            logger.debug("PoseSink: lookahead error (box=%d): %s", setup_id, e)
        return cx, cy

    def _resolve_zones(self, setup_id: int, zone_lookup, parts: list,
                       pose_array: list, cx, cy, zone_cx, zone_cy) -> tuple:
        """``(location, zones_by_body_part)``, innermost zone wins.

        Every confident body part gets its deepest containing zone, so
        per-keypoint triggers ("nose enters reward_zone") work. The centroid
        is deliberately split by consumer: the FORECAST centroid feeds the
        synthetic ``centroid`` body part the MCU acts on, while the RAW
        centroid sets ``location``, which is what the overlay shows and
        _video_data.txt records, both must describe the captured frame.
        """
        location: Optional[str] = None
        zones_by_body_part: dict = {}
        if zone_lookup is None:
            return location, zones_by_body_part
        try:
            for name, pt in zip(parts, pose_array):
                if pt[2] < self._confidence:
                    continue
                occ, _ = zone_occupancy_at(zone_lookup, pt[0], pt[1],
                                           deepest=True)
                if occ:
                    zones_by_body_part[name] = occ

            if zone_cx is not None and zone_cy is not None:
                occ, _ = zone_occupancy_at(zone_lookup, zone_cx, zone_cy,
                                           deepest=True)
                if occ:
                    zones_by_body_part["centroid"] = occ
            if cx is not None and cy is not None:
                occ_raw, loc_raw = zone_occupancy_at(zone_lookup, cx, cy,
                                                     deepest=True)
                if occ_raw:
                    location = loc_raw
        except Exception as e:
            logger.debug("PoseSink: zone test error (box=%d): %s", setup_id, e)
        return location, zones_by_body_part

    def _apply_rotation(self, setup_id: int, pose: dict, cfg: dict) -> None:
        """Write head direction onto the pose dict when rotation is enabled."""
        if not (cfg["rot_enabled"] and cfg["rot_state"] is not None):
            return
        try:
            theta, valid = head_direction(pose, cfg["rot_pair"],
                                          cfg["rot_state"],
                                          conf_cutoff=self._confidence)
            pose["_rotation_rad"] = theta if valid else None
        except Exception as e:
            logger.debug("PoseSink: rotation error (box=%d): %s", setup_id, e)
            pose["_rotation_rad"] = None

    def _forecast_coords_for_push(self, setup_id: int, cx, cy,
                                  zone_cx, zone_cy) -> Optional[dict]:
        """The latency-compensated centroid, for the MCU push ONLY.

        Display and recording keep the current position, so the overlay
        matches the captured frame and _video_data.txt stays truthful; only
        closed-loop ``c.*`` coords act on where the animal WILL be. Covers the
        synthetic "centroid" and, when a specific keypoint defines the
        centroid, that keypoint too. ``None`` when there is no forecast.
        """
        if (zone_cx is None or zone_cy is None
                or (zone_cx, zone_cy) == (cx, cy)):
            return None
        coords = {"centroid": (float(zone_cx), float(zone_cy))}
        picked_bp = self._centroid_body_part.get(setup_id)
        if picked_bp and picked_bp != "centroid":
            coords[picked_bp] = (float(zone_cx), float(zone_cy))
        return coords

    # ── Cleanup ------------------------------------------------------

    def shutdown(self) -> None:
        try:
            self._backend.shutdown(wait=False)
        except Exception as e:
            logger.debug("PoseSink: backend shutdown error: %s", e)
