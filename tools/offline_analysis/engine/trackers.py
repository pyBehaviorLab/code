"""The one place the analyser may reach for the rig, and only if it is there.

Re-running a detector over a recording is the single thing offline analysis
cannot do by itself. Everything else here, resolving a clock, reading zones,
re-zoning poses, computing measures, needs nothing but the session file.

Two ways to get a detector, and both were wrong:

* **Vendor it.** Nearly five thousand lines of detector code, changing weekly,
  copied. The copy drifts, and the moment it does, an offline retrack stops
  reproducing what the rig recorded, two detectors giving two answers about
  the same animal. That is the failure this project keeps meeting: three
  readers of the session file, three column maps, a colour conversion applied
  twice.
* **Import it at module scope.** Then deleting the rig breaks the analyser
  outright, including the parts that never needed a detector.

So: imported lazily, inside a function, guarded, and named here rather than
scattered. When the rig is present a retrack uses *exactly* the detector the
rig uses live. When it is absent, :func:`available` is empty,
:func:`why_unavailable` says so in a sentence, and the tab blocks the one row
that wanted it while everything else runs.

This module is the only file under ``tools/offline_analysis`` allowed to name
``source.*``; the perimeter test grants it that by name and refuses it anywhere
else.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Input modes, as strings rather than an imported enum, so that a rig with a
#: different spelling cannot break the analyser at import time.
MODE_FULL = "full"
MODE_LETTERBOX = "letterbox"
MODE_CROP_TRACK = "crop_track"

_BLOB = "blob"
_POSE_BACKENDS = ("dlc", "sleap")


def _rig():
    """The rig's tracking package, or ``None``.

    The one guarded reach. Anything that raises here, no rig, a partial
    checkout, a detector whose own import fails, is answered the same way:
    there is no tracker, and the reason is kept for the operator.
    """
    try:
        import source.video.tracking as tracking  # noqa: F401
        return tracking
    except Exception as e:                        # ImportError and worse
        _remember(str(e))
        return None


_reason: str = ""
#: Import attempts already made, so a failing SDK is not re-imported on every
#: refresh; and why each backend is unusable, for the panel that offers them.
_IMPORTED: Dict[str, bool] = {}
_WHY_BACKEND: Dict[str, str] = {}


def _remember(why: str) -> None:
    global _reason
    _reason = why
    logger.debug("no rig trackers available: %s", why)


def why_unavailable() -> str:
    """A sentence for the blocked row, not a stack trace.

    Empty when a detector IS available: a reason offered alongside a working
    tracker reads as a warning about the tracker, and there is nothing wrong
    with it.
    """
    if available():
        return ""
    if not _reason:
        return ("the recording pipeline is not installed beside this analyser, "
                "so a detector cannot be run over the video here")
    return f"the recording pipeline's trackers could not be loaded: {_reason}"


def _exported_runtime() -> str:
    """A runtime that can execute an already-exported engine, or ``""``."""
    import importlib

    for module in ("tensorrt", "onnxruntime"):
        if module in _IMPORTED and not _IMPORTED[module]:
            continue
        try:
            importlib.import_module(module)
            _IMPORTED[module] = True
            return module
        except Exception:
            _IMPORTED[module] = False
    return ""


def why_backend_unavailable(backend: str) -> str:
    """Why THIS detector cannot run here, or ``""`` when it can.

    ``why_unavailable`` answers "is there any tracker at all", which is a
    different question and goes quiet as soon as one works. With blob
    installed and DLC broken, nothing said a word about DLC.
    """
    if backend in available():
        return ""
    why = _WHY_BACKEND.get(backend, "")
    module = {"dlc": "dlclive", "sleap": "sleap_nn"}.get(backend, backend)
    if backend == "sleap" and not why:
        return (f"neither {module} nor a runtime for an exported model "
                "(onnxruntime, tensorrt) is installed in this Python "
                "environment")
    if not why:
        return f"{module} is not installed in this Python environment"
    return f"{module} could not be imported: {why}"


def last_error() -> str:
    """The most recent reason a tracker could not be built or loaded."""
    return _reason


def available() -> List[str]:
    """Which detectors this machine can actually run, cheapest check first."""
    if _rig() is None:
        return []
    found: List[str] = []
    try:
        from source.video.tracking.blob import BlobTracker  # noqa: F401
        found.append(_BLOB)
    except Exception as e:
        _remember(str(e))
    for backend in _POSE_BACKENDS:
        if _pose_backend_importable(backend):
            found.append(backend)
    return found


def _pose_backend_importable(backend: str) -> bool:
    """Whether ``backend``'s own SDK is installed.

    A backend the rig defines but whose SDK is missing is not available: DLC
    without ``dlclive`` fails at the first frame, and finding that out then
    rather than now is how a six-hour retrack ends in nothing.
    """
    module = {"dlc": "dlclive", "sleap": "sleap_nn"}.get(backend)
    if not module:
        return False
    # An EXPORTED model needs a runtime, not the training SDK. The rig tries a
    # direct ONNX/TensorRT runner before it imports sleap_nn, so a folder
    # holding `model.onnx` runs here with onnxruntime alone, and demanding
    # sleap_nn reported "unavailable" on a machine that could run the model
    # perfectly well.
    if backend == "sleap" and _exported_runtime():
        return True
    if module in _IMPORTED:
        return _IMPORTED[module]
    import importlib

    try:
        # IMPORTED, not `find_spec`. A package can sit on disk and still fail
        # to import, `dlclive` is installed here and raises "No module named
        # torch", and `find_spec` says yes to it, which is precisely the
        # answer this function exists to avoid giving.
        importlib.import_module(module)
        _IMPORTED[module] = True
    except Exception as e:
        _IMPORTED[module] = False
        _WHY_BACKEND[backend] = str(e)
        logger.debug("%s is not usable: %s", module, e)
    return _IMPORTED[module]


def _have(module: str) -> bool:
    """Whether ``module`` imports here, remembered so a slow or failing SDK is
    not re-imported every time something asks."""
    if module in _IMPORTED:
        return _IMPORTED[module]
    import importlib

    try:
        importlib.import_module(module)
        _IMPORTED[module] = True
    except Exception as e:
        _IMPORTED[module] = False
        logger.debug("%s is not importable: %s", module, e)
    return _IMPORTED[module]


def _artifacts(model_path: str) -> Dict[str, List[str]]:
    """What is actually inside a model folder, grouped by what can run it.

    A folder is the only evidence available before a load is attempted, and it
    decides which runtime is even relevant: complaining that TensorRT is
    missing is noise for a folder that holds nothing but a checkpoint.
    """
    import os

    out: Dict[str, List[str]] = {"trt": [], "onnx": [], "ckpt": [],
                                 "torch": [], "tf": [], "config": []}
    if not model_path or not os.path.isdir(model_path):
        return out
    for name in sorted(os.listdir(model_path)):
        low = name.lower()
        if low.endswith(".trt") or low.endswith(".engine"):
            out["trt"].append(name)
        elif low.endswith(".onnx"):
            out["onnx"].append(name)
        elif low.endswith(".ckpt"):
            out["ckpt"].append(name)
        elif low.endswith(".pt") or low.endswith(".pth"):
            out["torch"].append(name)
        elif low.startswith("snapshot-") or low.endswith(".pb"):
            out["tf"].append(name)
        elif low in ("pose_cfg.yaml", "pose_cfg.yml", "pytorch_config.yaml",
                     "config.yaml", "training_config.yaml",
                     "training_config.json"):
            out["config"].append(name)
    return out


def _load_hint(backend: str, model_path: str) -> str:
    """What is missing for THIS model on THIS machine, as a sentence.

    A detector that will not load is nearly always a runtime that is not
    installed, and which runtime depends on what the folder holds. Working
    that out here means the operator is told "the folder holds model.onnx but
    onnxruntime is not installed" instead of "the model would not load", which
    is the difference between a one-line fix and an afternoon.
    """
    art = _artifacts(model_path)
    kind = str(backend or "").lower()
    bits: List[str] = []

    if kind.startswith("sleap"):
        if art["trt"] and not _have("tensorrt"):
            bits.append(f"the folder holds {', '.join(art['trt'])} but the "
                        "tensorrt package is not installed")
        if art["onnx"] and not _have("onnxruntime"):
            bits.append(f"the folder holds {', '.join(art['onnx'])} but "
                        "onnxruntime is not installed "
                        "(pip install onnxruntime-gpu)")
        if art["ckpt"] and not any(n.lower() == "best.ckpt"
                                   for n in art["ckpt"]):
            bits.append(f"the checkpoint here is named {art['ckpt'][0]}, and "
                        "sleap-nn's trained-model loader only opens "
                        "'best.ckpt', rename it, or export the model")
        if not any(art[k] for k in ("trt", "onnx", "ckpt")):
            bits.append("the folder holds no engine and no checkpoint, so "
                        "there is nothing here to load")
        if not _have("sleap_nn"):
            bits.append("sleap_nn is not installed (pip install sleap-nn)")

    elif kind.startswith("dlc"):
        if not _have("dlclive"):
            bits.append("dlclive is not installed "
                        "(pip install deeplabcut-live)")
        else:
            bits.append(f"dlclive can run {_dlc_backends()} here")
            if art["tf"] and not _have("tensorflow"):
                bits.append("this is a TensorFlow DLC model (it holds "
                            f"{art['tf'][0]}) and tensorflow is not installed "
                            "in this environment")
            if art["torch"] and not _have("torch"):
                bits.append("this is a PyTorch DLC model and torch is not "
                            "installed")

    if not bits:
        bits.append("the detector gave no reason; the full traceback is on "
                    "the 'source.video.tracking' logger")
    return "; ".join(bits) + "."


def _dlc_backends() -> str:
    """Which engines the installed dlclive can actually drive."""
    try:
        import dlclive
        names = [getattr(b, "value", str(b))
                 for b in (dlclive.get_available_backends() or ())]
        return ", ".join(names) or "no backend"
    except Exception:
        return "an unknown set of backends"


class BlobAsPose:
    """The rig's ``BlobTracker``, presented the way an offline retrack asks.

    The retrack loop was written against a detector that answers
    ``predict(frame) -> {name: [x, y, conf]}``. This rig's blob tracker answers
    ``update(frame) -> (ok, (x, y, w, h))``, needs a reference background, and
    has to be initialised before it will report anything. Nothing bridged the
    two, so the loop called ``detect``, which does not exist, one frame after
    the tracker was built.

    Two details are copied from the LIVE sink rather than reinvented, because
    an offline retrack of a video should produce what the rig would have
    produced from the same frames:

    * the **moments centroid** (``last_centroid``) is preferred over the box
      centre. The box centre drifts from the body exactly when the animal
      rears, grooms or extends its tail, which is when zone occupancy matters
      most; the live sink says so in as many words and falls back the same way.
    * the point is called **centroid**, which is the name the live pipeline
      gives it, so a session tracked here and one tracked on the rig describe
      the same thing.

    Confidence is 1.0 on a hit and the keypoint is absent on a miss. A blob
    detector has no confidence to report, inventing a number between would be
    inventing evidence, and the loop already treats an absent keypoint as "not
    seen this frame" rather than as a dropped frame.
    """

    #: What the live pipeline calls the point a differencing tracker produces.
    BODY_PART = "centroid"

    def __init__(self, tracker):
        self._t = tracker
        self._ready = False
        self._misses = 0
        self._hits = 0

    # ── what the retrack loop calls ──────────────────────────────────
    def predict(self, frame):
        """This frame's centroid, or ``{}`` when the animal was not found."""
        if not self._ready and not self._prime_from(frame):
            return {}
        try:
            ok, position = self._t.update(frame)
        except Exception as e:                       # never kill the run
            logger.debug("blob update failed: %s", e)
            return {}
        if not ok or not position:
            self._misses += 1
            return {}
        self._hits += 1
        x, y, w, h = (float(v) for v in position[:4])
        cx, cy = x + w / 2.0, y + h / 2.0
        centroid = getattr(self._t, "last_centroid", None)
        if centroid is not None and len(centroid) >= 2:
            cx, cy = float(centroid[0]), float(centroid[1])
        return {self.BODY_PART: [cx, cy, 1.0]}

    def get_body_parts(self):
        return [self.BODY_PART]

    def close(self):
        for name in ("stop", "reset"):
            fn = getattr(self._t, name, None)
            if fn is None:
                continue
            try:
                fn()
            except Exception:
                pass

    # ── the background this detector cannot work without ─────────────
    def prime(self, background) -> bool:
        """Give it the reference image and start it tracking.

        ``set_background`` then ``initialize``: the rig's own order, and
        ``initialize`` is what flips ``tracking_active``, without which
        ``update`` returns ``(False, None)`` for every frame of the video and
        the run produces nothing while looking like it worked.
        """
        if background is None:
            return False
        try:
            self._t.set_background(background)
            self._ready = bool(self._t.initialize(background))
        except Exception as e:
            logger.warning("could not prime the blob tracker: %s", e)
            self._ready = False
        return self._ready

    def _prime_from(self, frame) -> bool:
        """Last resort: prime from the frame in hand.

        Only reached when no background was built, a single frame contains
        the animal, so it subtracts itself away wherever it was standing. The
        run is still better than nothing and the caller has already been
        warned; ``self_norm`` mode needs no reference at all and works fine.
        """
        return self.prime(frame)

    @property
    def stats(self) -> dict:
        """Hits and misses, for the run's own report."""
        return {"found": self._hits, "missed": self._misses}


#: What the rig's own builder passes to a pose tracker's constructor, and
#: nothing else. Taken from ``source.video.tracking.inference._new_tracker``:
#: the SLEAP option group is forwarded only for SLEAP, deliberately, "so the
#: DLC/blob constructors never see unexpected kwargs": which is exactly how
#: the offline path failed, passing ``confidence_threshold`` to a constructor
#: that takes ``(model_path, resize_factor, colour_mode)``.
_POSE_KWARGS = ("resize_factor", "colour_mode", "body_parts")


def accepted_options(backend: str) -> set:
    """What this backend's constructor takes, asked through the seam.

    Exposed because the perimeter holds: everything under
    ``tools/offline_analysis`` except this file is forbidden to name
    ``source.*``, so a caller that wants to know what a backend accepts has to
    ask here rather than import the rig itself.
    """
    if _rig() is None:
        return set(_POSE_KWARGS)
    try:
        from source.video.tracking.pose import accepted_options as _accepted
        return _accepted(backend)
    except Exception as e:                                # pragma: no cover
        _remember(str(e))
        return set(_POSE_KWARGS)


def _pose_kwargs(backend: str, given: Dict[str, Any]) -> Dict[str, Any]:
    """Only the arguments this backend's constructor actually accepts.

    THE RIG'S OWN RULE, not a second copy of it. What a backend accepts is
    knowledge that belongs with the backend, and a copy here drifts from it: a
    list granting DLC only ``model_type`` drops ``device``, ``precision`` and
    ``dynamic_crop``: all real parameters, all settable in the tracking
    dialog, on the way to every offline retrack,
    and SLEAP lost ``peak_threshold`` and ``max_instances`` the same way.

    An option that IS dropped is said out loud. A setting that vanishes
    between the dialog and the model is only ever noticed as "the results look
    the same whatever I choose".
    """
    if _rig() is None:
        return {k: v for k, v in given.items()
                if k in _POSE_KWARGS and v is not None}
    try:
        from source.video.tracking.pose import filter_options
    except Exception as e:                                # pragma: no cover
        _remember(str(e))
        return {k: v for k, v in given.items()
                if k in _POSE_KWARGS and v is not None}
    kwargs, dropped = filter_options(backend, given)
    if dropped:
        logger.info("%s ignores %s: not parameters of this backend",
                    backend, ", ".join(dropped))
    return kwargs


def _between(centres, i):
    """A centre for index ``i`` from the nearest known one either side.

    Interpolated when the frame is bracketed, carried over when it sits at one
    end of the batch. ``None`` only when nothing in the batch tracked at all,
    which is a lost window rather than a missed frame.
    """
    before = next((j for j in range(i - 1, -1, -1)
                   if centres[j] is not None), None)
    after = next((j for j in range(i + 1, len(centres))
                  if centres[j] is not None), None)
    if before is None and after is None:
        return None
    if before is None:
        return centres[after]
    if after is None:
        return centres[before]
    span = after - before
    t = (i - before) / float(span)
    return (centres[before][0] + t * (centres[after][0] - centres[before][0]),
            centres[before][1] + t * (centres[after][1] - centres[before][1]))


class PoseAsPose:
    """A rig pose tracker, fed and initialised the way the live pipeline does.

    Three things stood between the offline retrack and a working detector, and
    none was visible until a tracker was actually built and given a frame:

    * the constructor was handed arguments it does not take
      (``confidence_threshold``, ``max_instances``, ``identities``, …). Those
      are the RUN's settings, not the model's, and the loop applies them
      itself.
    * ``initialize(frame)`` was never called. The live pipeline calls it with a
      probe frame and treats a false return as "no tracker"; without it the
      model is never loaded, and the first ``predict`` fails on a session that
      looked perfectly configured.
    * frames went to the model at whatever size the video happened to be. A
      UNet downsamples by its stride, so a height of 202 px tears the decoder's
      skip connections apart, "Axis 2 has mismatched dimensions of 24 and 25",
      and every frame fails. The live pipeline letterboxes to a canonical
      shape and maps the keypoints back; this does the same, with the same
      class, so a retrack lands in the same coordinates as the live run.

    Offline there is no probe frame at build time, the video is opened later,
    so the fit and the initialisation both happen on the first frame, once.
    """

    def __init__(self, tracker, model_path: str = "", backend: str = "",
                 input_mode: str = "auto", input_wh=None, engine: str = ""):
        self._t = tracker
        self._model_path = model_path
        self._backend = backend
        self._ready = False
        self._failed = ""
        self._lb = None
        #: How the frame becomes the model's input, decided by the rig's own
        #: policy on the first frame, the same call the live sink makes, so a
        #: retrack cannot present the model a different input from the
        #: recording it is reproducing.
        self._asked_mode = input_mode or "auto"
        self._asked_wh = tuple(input_wh) if input_wh else None
        self._engine = engine
        self._plan = None
        self._crop = None

    # ── what the retrack loop calls ──────────────────────────────────
    @property
    def failure(self) -> str:
        """Why this detector cannot run, or ``""``.

        The retrack loop asks BEFORE the first frame and stops the recording
        when there is an answer. Without it a model that would not load
        returned an empty pose for every frame of the video: 72,000 frames at
        5,000 fps, "0% with a detection", and the reason nowhere the operator
        could see it.
        """
        return self._failed

    def initialize(self, frame=None) -> bool:
        """Load the model, on the frame it will actually be given.

        Defined here rather than delegated through ``__getattr__``: the raw
        tracker would be handed the video's own frame, not the letterboxed one
        the model needs, so the probe and every later ``predict`` disagreed
        about the input shape. It also returns a truthful bool, which is what
        the caller checks to decide whether the run may proceed.
        """
        if frame is None:
            return self._ready
        plan = self._input_plan(frame)
        if plan is not None and plan.mode == "crop_track":
            # The probe has to be the shape the real frames will be, or a
            # session compiles its input placeholder at the wrong size.
            crop_at, state = self._crops()
            if crop_at is not None:
                cut, _win = crop_at(frame, frame.shape[1] / 2.0,
                                    frame.shape[0] / 2.0, state.w, state.h)
                return self._start(cut)
        letterbox = self._fit(frame)
        image = letterbox.apply(frame) if letterbox is not None else frame
        return self._start(image)

    def predict_batch(self, frames):
        """A whole batch through the model in one forward pass.

        Defined here rather than delegated through ``__getattr__`` for the
        same reason as ``initialize``: the raw tracker would be handed the
        video's own frames, not the letterboxed ones, and would either fail on
        the shape or silently return keypoints in the wrong coordinate space.

        This is where the offline retrack's speed is. Measured on the 6-point
        SLEAP export against an A6000: 2.92 ms/frame at batch 1 against 0.60
        at batch 16, kernel launch and transfer dominate a network this
        small, so the GPU spends most of a single-frame call idle.
        """
        if not frames:
            return []
        if self._failed:
            return [{} for _ in frames]
        self._input_plan(frames[0])
        if self._plan is not None and self._plan.mode == "crop_track":
            return self._crop_batch(frames)
        letterbox = self._fit(frames[0])
        images = ([letterbox.apply(f) for f in frames]
                  if letterbox is not None else list(frames))
        if not self._ready and not self._start(images[0]):
            return [{} for _ in frames]
        batch = getattr(self._t, "predict_batch", None)
        if batch is None:
            return [self._to_source(self._t.predict(im) or {}, letterbox)
                    for im in images]
        try:
            out = batch(images) or []
        except Exception as e:
            # A batch the engine will not take is not a dead run: the same
            # frames go through one at a time, and the caller is told why the
            # fast path was abandoned rather than left wondering about the
            # speed.
            logger.warning("batched predict failed (%s), falling back to "
                           "one frame at a time", e)
            self._no_batch = True
            return [self._to_source(self._t.predict(im) or {}, letterbox)
                    for im in images]
        poses = [self._to_source(p or {}, letterbox) for p in out]
        # Never return fewer poses than frames: a short list would silently
        # shift every later frame's pose onto the wrong picture.
        while len(poses) < len(frames):
            poses.append({})
        return poses[:len(frames)]

    def _crop_batch(self, frames):
        """A batch through a following window, then repaired from its
        neighbours.

        The window's position depends on where the animal was, and inside one
        batch that is not yet known, so every frame of the batch is cut at the
        centre the previous batch ended on. Over sixteen frames at 30 fps the
        animal can walk out of a window placed half a second ago, and those
        frames come back with too few confident keypoints.

        Which is where offline stops having to behave like online. The frames
        that DID track are, by then, known both before and after each failure,
        so the missed ones are re-cut at a centre interpolated from their
        neighbours on either side and run again. A live rig cannot do this: it
        has no later frame to interpolate from. That is the whole reason a
        retrack is worth running over a recording that already has pose.
        """
        crop_at, state = self._crops()
        if crop_at is None:
            return [{} for _ in frames]
        from source.video.tracking.crop_tracker import (confident_centroid,
                                                        count_confident)

        height, width = frames[0].shape[:2]
        centre = state.centre_for(width, height)
        cuts, windows = [], []
        for frame in frames:
            cut, window = crop_at(frame, centre[0], centre[1], state.w, state.h)
            cuts.append(cut)
            windows.append(window)
        if not self._ready and not self._start(cuts[0]):
            return [{} for _ in frames]
        poses = self._run(cuts)
        out = [w.pose_to_source(p or {}) for p, w in zip(poses, windows)]

        # Where each frame's animal actually was, in video coordinates.
        centres = [confident_centroid(p, state.conf_min, state.good_min)
                   for p in out]
        missed = [i for i, c in enumerate(centres) if c is None]
        for i in missed:
            guess = _between(centres, i)
            if guess is None:
                continue
            cut, window = crop_at(frames[i], guess[0], guess[1],
                                  state.w, state.h)
            retry = self._run([cut])
            fixed = window.pose_to_source((retry[0] if retry else {}) or {})
            # Kept only if it is actually better. A second window that finds
            # no more than the first is not evidence of anything, and taking
            # it anyway would trade a known-empty pose for an unknown one.
            if count_confident(fixed, state.conf_min) > count_confident(
                    out[i], state.conf_min):
                out[i] = fixed
                centres[i] = confident_centroid(fixed, state.conf_min,
                                                state.good_min)
        if missed:
            recovered = sum(1 for i in missed if centres[i] is not None)
            logger.debug("crop window: %d/%d frames re-cut from their "
                         "neighbours, %d recovered", len(missed), len(frames),
                         recovered)
        # The next batch starts where this one ended, so the window carries on
        # rather than restarting at the frame centre.
        for pose in out:
            state.update(pose)
        return out

    def _run(self, images):
        """One forward pass on already-transformed images."""
        batch = getattr(self._t, "predict_batch", None)
        if batch is None or getattr(self, "_no_batch", False):
            return [self._t.predict(im) or {} for im in images]
        try:
            got = list(batch(images) or [])
        except Exception as e:
            logger.warning("batched predict failed (%s), falling back to "
                           "one frame at a time", e)
            self._no_batch = True
            return [self._t.predict(im) or {} for im in images]
        while len(got) < len(images):
            got.append({})
        return got[:len(images)]

    @property
    def batched(self) -> bool:
        """Whether a batch is worth submitting to this tracker.

        The wrapped tracker's own answer wins where it has one. Merely HAVING
        a ``predict_batch`` says nothing: a DLC export whose batch axis is
        fixed at 1 has the method and refuses every batch, and taking the
        method's existence as consent made the retrack submit sixteen frames,
        get refused, and fall back, on every single call, warning each time.
        """
        if getattr(self, "_no_batch", False):
            return False
        own = getattr(self._t, "batched", None)
        if own is not None:
            return bool(own)
        return hasattr(self._t, "predict_batch")

    def _to_source(self, pose, letterbox):
        return letterbox.pose_to_source(pose) if letterbox is not None else pose

    def predict(self, frame):
        if self._failed:
            return {}
        self._input_plan(frame)
        if self._plan is not None and self._plan.mode == "crop_track":
            # One frame is a batch of one; the repair pass has no neighbours to
            # work with and simply does nothing.
            got = self._crop_batch([frame])
            return got[0] if got else {}
        letterbox = self._fit(frame)
        image = letterbox.apply(frame) if letterbox is not None else frame
        if not self._ready and not self._start(image):
            return {}
        try:
            pose = self._t.predict(image) or {}
        except Exception as e:
            logger.debug("pose predict failed: %s", e)
            return {}
        return letterbox.pose_to_source(pose) if letterbox is not None else pose

    def close(self):
        fn = getattr(self._t, "close", None) or getattr(self._t, "stop", None)
        if fn is not None:
            try:
                fn()
            except Exception:
                pass

    def __getattr__(self, name):
        """Anything else the loop asks for belongs to the real tracker."""
        return getattr(self._t, name)

    # ── the frame the model expects ──────────────────────────────────
    def _input_plan(self, frame):
        """The rig's own input decision for this video, made once.

        Asked of ``source.video.tracking.input_policy`` rather than re-derived,
        because a second copy of the rule is a second answer: the offline path
        used to letterbox onto whatever size the model declared, which for a
        crop-trained model shrinks the animal past anything the network saw and
        produced a retrack of confident nonsense. Same call, same answer, both
        pipelines.
        """
        if self._plan is not None or frame is None:
            return self._plan
        height, width = frame.shape[:2]
        try:
            from source.video.tracking.input_policy import plan as _plan
        except Exception as e:
            _remember(str(e))
            return None
        self._plan = _plan(self._backend or "sleap", self._model_path,
                           width, height, engine=self._engine,
                           asked=self._asked_mode, asked_size=self._asked_wh)
        logger.info("pose input for %dx%d video: %s", width, height,
                    self._plan.describe())
        return self._plan

    def _fit(self, frame):
        """The letterbox from this video's frames to the model's input.

        ``None`` when the plan is not a letterbox, ``full`` needs no
        transform, and ``crop_track`` uses a moving window instead, which is
        not one fixed mapping and so cannot be cached as one.

        Built once for the letterbox case: every frame of one video is the same
        size, and rebuilding it per frame would be arithmetic repeated 77,000
        times to reach the same answer.
        """
        if self._lb is not None or frame is None:
            return self._lb
        plan = self._input_plan(frame)
        if plan is None or plan.mode in ("crop_track", "full"):
            return None
        try:
            from source.video.tracking.letterbox import (Letterbox,
                                                         canonical_shape)
        except Exception as e:
            _remember(str(e))
            return None
        height, width = frame.shape[:2]
        if plan.size:
            dst_w, dst_h = plan.size
        else:
            shape = canonical_shape([(width, height)], self._stride())
            dst_w, dst_h = shape or (width, height)
        self._lb = Letterbox.fit(width, height, dst_w, dst_h)
        if not self._lb.is_identity:
            logger.info("pose input %dx%d fitted to %dx%d for the model "
                        "(scale %.3f, pad %d,%d)", width, height, dst_w, dst_h,
                        self._lb.scale, self._lb.pad_x, self._lb.pad_y)
        return self._lb

    def _stride(self):
        try:
            from source.video.tracking.model_config import ModelInfo
            return ModelInfo.read(self._model_path).max_stride
        except Exception as e:
            logger.debug("model config unreadable (%s)", e)
            return None

    def _crops(self):
        """``crop_at`` and this video's window state, or ``(None, None)``."""
        plan = self._plan
        if plan is None or plan.mode != "crop_track" or not plan.size:
            return None, None
        if self._crop is None:
            try:
                from source.video.tracking.crop_tracker import (BoxCropState,
                                                                crop_at)
            except Exception as e:
                _remember(str(e))
                return None, None
            self._crop = (crop_at,
                          BoxCropState(plan.size[0], plan.size[1]))
        return self._crop

    def _start(self, image) -> bool:
        """Load the model, on the frame it will actually be given.

        The reason a load failed is recorded in the operator's terms and named
        with the model it was asked for. "refused to initialise" said neither,
        and the detector's own explanation went to the ``source.video.tracking``
        logger, which this tool did not listen to, so the one sentence that
        would have identified the cause was written where nobody could read it.
        """
        what = self._backend or "pose"
        where = self._model_path or "the selected model"
        try:
            self._ready = bool(self._t.initialize(image))
        except Exception as e:
            self._failed = f"{what} could not load {where}: {e}"
            logger.warning("%s", self._failed)
            return False
        if not self._ready:
            self._failed = (
                f"{what} refused to load {where}. {_load_hint(what, where)}")
            logger.warning("%s", self._failed)
        return self._ready


def median_background(get_frame, samples: int = 30):
    """A reference background, by the rig's own method.

    The rig's ``capture_median``, not a second implementation of it. A
    differencing tracker is only as good as the image it subtracts, and using
    the same median the live tracker would have used is the difference between
    reproducing a session and approximating it.

    Lives here because this module is the ONE place allowed to reach into the
    rig; the offline retrack calls it rather than importing the rig itself.
    """
    if _rig() is None:
        return None
    try:
        from source.video.tracking.background import capture_median
    except Exception as e:
        _remember(str(e))
        return None
    # The sleep paces a live camera between grabs; offline there is nothing
    # to wait for.
    return capture_median(get_frame, samples=samples, sleep=lambda _s: None)


def make(backend: str, **kwargs) -> Optional[Any]:
    """Build a detector, or ``None`` with the reason recorded.

    The object returned is the rig's own tracker, so a retrack reproduces the
    live detector rather than approximating it.
    """
    if _rig() is None:
        return None
    try:
        if backend == _BLOB:
            from source.video.tracking.blob import BlobTracker
            return BlobTracker(**kwargs)
        from source.video.tracking.pose import create_pose_tracker
        # Filtered, then wrapped so it initialises on its first frame, the
        # two things the live pipeline does that the offline path did not.
        model_path = kwargs.pop("model_path", "")
        # The input contract travels with the run's settings, not with the
        # model constructor, it decides what the model is FED, not what it
        # IS, so it is pulled out before the kwargs are filtered.
        input_mode = kwargs.pop("pose_input_mode", "auto")
        input_wh = (int(kwargs.pop("pose_input_w", 0) or 0),
                    int(kwargs.pop("pose_input_h", 0) or 0))
        engine = ""
        if backend in ("dlc", "deeplabcut"):
            from source.video.tracking import dlc_engine
            engine = dlc_engine.resolve(
                model_path, kwargs.get("dlc_model_type", "auto"))[0]
        return PoseAsPose(
            create_pose_tracker(backend, model_path,
                                **_pose_kwargs(backend, kwargs)),
            model_path=model_path, backend=backend,
            input_mode=input_mode, input_wh=input_wh, engine=engine)
    except Exception as e:
        _remember(f"{backend}: {e}")
        logger.warning("could not build the %s tracker: %s", backend, e)
        return None


def model_info(model_path: str) -> Dict[str, Any]:
    """What a model declares about itself, or an empty dict.

    Read through the rig so the analyser cannot disagree with it about a
    model's body parts or input size, the numbers a retrack has to match.

    Everything the rig's reader states is carried, including the fields it
    works out for itself: ``multi_animal`` and ``needs_centroid`` are answers,
    not things to infer from the family name here and get subtly wrong.
    """
    if _rig() is None or not model_path:
        return {}
    try:
        from source.video.tracking.model_config import ModelInfo as _RigInfo
        info = _RigInfo.read(model_path)
        return {"family": info.family, "body_parts": list(info.body_parts),
                "identities": list(getattr(info, "identities", ()) or ()),
                "channels": info.channels, "max_stride": info.max_stride,
                "input_w": info.input_w, "input_h": info.input_h,
                "crop_size": info.crop_size,
                "multi_animal": bool(getattr(info, "multi_animal", False)),
                "needs_centroid": bool(getattr(info, "needs_centroid", False)),
                "backbone": getattr(info, "backbone", None),
                "config_path": getattr(info, "config_path", ""),
                "warnings": list(getattr(info, "warnings", ()) or ())}
    except Exception as e:
        _remember(str(e))
        return {}


class ModelInfo:
    """What a model declares, in the shape the ported tab expects.

    The tab and the tracker dialog were written against pyBehaveTrack's
    ``ModelInfo`` and read ``ok``, ``input_size``, ``needs_centroid``,
    ``summary()``, ``warnings`` and ``disagreements(...)`` off it. Rather than
    edit them, the point of the port is that they stay the same, the seam
    hands back an object of that shape, filled from the rig's own reader, and
    honest when the rig is not here: ``ok`` is False and every list is empty,
    so the panel shows "no model information" instead of confident blanks.
    """

    __slots__ = ("family", "body_parts", "identities", "channels",
                 "max_stride", "input_w", "input_h", "crop_size",
                 "multi_animal", "_needs_centroid", "backbone", "config_path",
                 "_warnings", "ok")

    def __init__(self, **kw):
        self.family = kw.get("family")
        self.body_parts = tuple(kw.get("body_parts") or ())
        self.identities = tuple(kw.get("identities") or ())
        self.channels = kw.get("channels")
        self.max_stride = kw.get("max_stride")
        self.input_w = kw.get("input_w")
        self.input_h = kw.get("input_h")
        self.crop_size = kw.get("crop_size")
        self.multi_animal = bool(kw.get("multi_animal"))
        self._needs_centroid = bool(kw.get("needs_centroid"))
        self.backbone = kw.get("backbone")
        self.config_path = kw.get("config_path", "")
        self._warnings = tuple(kw.get("warnings") or ())
        self.ok = bool(kw.get("ok", bool(kw)))

    @classmethod
    def read(cls, model_path: str) -> "ModelInfo":
        return cls(**model_info(model_path))

    @property
    def n_identities(self) -> int:
        return len(self.identities)

    @property
    def input_size(self) -> Optional[tuple]:
        """``(w, h)`` the model states, or ``None``. A square top-down crop is
        the other way it says the same thing."""
        if self.input_w and self.input_h:
            return (int(self.input_w), int(self.input_h))
        return (int(self.crop_size), int(self.crop_size)) if self.crop_size else None

    @property
    def needs_centroid(self) -> bool:
        """Whether this model needs a centroid stage before it can pose.

        The rig's reader decides; falling back to a family-name guess only
        when it said nothing, because a top-down model without its centroid
        model poses nothing at all and the operator should be told before the
        run, not after.
        """
        if self._needs_centroid:
            return True
        return str(self.family or "") in ("centered_instance", "topdown",
                                          "multi_class_topdown")

    @property
    def warnings(self) -> List[str]:
        """Whatever the rig's reader had to say, plus our own when it is mute."""
        out = list(self._warnings)
        if not self.ok:
            out.append(why_unavailable() or "no model information available here")
        return out

    def disagreements(self, *, n_animals: Optional[int] = None,
                      body_parts: Optional[List[str]] = None,
                      colour_mode: Optional[str] = None) -> List[str]:
        """Where the project contradicts the model, in the operator's words.

        A METHOD, with pyBehaveTrack's signature, because that is how the
        panel and the tracker dialog call it, ``info.disagreements(
        n_animals=…)``. It was a property returning a list here, so choosing
        any model at all raised ``TypeError: 'list' object is not callable``
        before the card could be drawn.

        Every one of these is otherwise silent: a multi-animal model loaded as
        single-animal poses one animal and drops the rest; a grayscale network
        fed three channels is the mirror of the BGR/RGB bug; a body-part list
        that is not the model's renames the keypoints without moving them.
        """
        out: List[str] = []
        if not self.ok:
            return out
        if n_animals is not None and n_animals > 1 and not self.multi_animal:
            out.append(
                f"this model is single-animal ({self.family}), but the run "
                f"asks for {n_animals} animals")
        if (n_animals is not None and self.identities
                and n_animals != self.n_identities):
            out.append(
                f"the model predicts {self.n_identities} identities "
                f"({', '.join(self.identities)}), the run asks for {n_animals}")
        if (body_parts and self.body_parts
                and tuple(body_parts) != tuple(self.body_parts)):
            out.append(
                f"the run names {len(body_parts)} body parts and the model "
                f"declares {len(self.body_parts)} "
                f"({', '.join(self.body_parts)})")
        if (colour_mode and colour_mode != "auto" and self.channels
                and colour_mode != self.channels):
            out.append(
                f"the model was trained on {self.channels} and the run asks "
                f"for {colour_mode}")
        if self.needs_centroid:
            out.append(
                "a top-down model needs a paired centroid model to locate the "
                "animal before it can be posed")
        # Ours, not pyBehaveTrack's: an input the backbone's stride does not
        # divide breaks the encoder shape arithmetic, which is exactly what a
        # hand-typed window gets wrong.
        size, stride = self.input_size, self.max_stride
        if size and stride and stride > 1 and (size[0] % stride
                                               or size[1] % stride):
            out.append(f"input {size[0]}x{size[1]} is not divisible by the "
                       f"backbone stride {stride}")
        return out

    def summary(self) -> str:
        if not self.ok:
            return "no model information"
        bits = [str(self.family or "unknown")]
        if self.body_parts:
            bits.append(f"{len(self.body_parts)} parts")
        if self.input_size:
            bits.append("{}x{}".format(*self.input_size))
        if self.channels:
            bits.append(str(self.channels))
        return " · ".join(bits)


def backends() -> Dict[str, bool]:
    """``{"dlc": bool, "sleap": bool, "blob": bool}`` for this machine.

    Answered when the operator CHOOSES a backend, not when the run starts: a
    six-hour job that dies on an import at minute one is the failure this
    prevents.
    """
    have = set(available())
    return {"dlc": "dlc" in have, "sleap": "sleap" in have,
            "blob": "blob" in have}


def crop_refusal(info: Optional[Dict[str, Any]] = None, n_instances: int = 1,
                 window: Optional[tuple] = None,
                 identities: Any = ()) -> str:
    """Why a following window cannot run for this model, or ``""``.

    Pure policy over what the model declares, so it holds whether or not the
    rig is installed, and it is asked at choosing time for the same reason as
    :func:`backends`.
    """
    # A dict OR a ModelInfo: the ported panel holds the object and passes it
    # straight in, the retrack path holds the dict `model_info` returns. This
    # asked `info.get(...)` and so raised AttributeError on the object, which
    # is what "no attribute get" meant when crop-track was chosen for SLEAP.
    if info is not None and not isinstance(info, dict):
        info = {"family": getattr(info, "family", ""),
                "multi_animal": bool(getattr(info, "multi_animal", False)),
                "needs_centroid": bool(getattr(info, "needs_centroid", False))}
    info = info or {}
    identities = [str(n) for n in (identities or []) if str(n)]
    n = int(n_instances or 1)
    if n > 1 and not identities:
        return ("crop-track follows ONE animal with one window; for "
                f"{n} animals give each an identity so each gets its own "
                "window, or use the full frame")
    if identities and len(identities) != n:
        return (f"{len(identities)} identities for {n} animals, one window "
                "per animal means one identity per animal")
    if not window or not window[0] or not window[1]:
        return ("crop-track needs the training window size, and the model's "
                "config does not state one, set it explicitly")
    family = str(info.get("family") or "")
    if family.startswith("bottomup"):
        return ("a bottom-up model finds every animal in the whole frame; a "
                "crop would hide the ones outside it")
    if family in ("topdown", "topdown_id", "multi_class_topdown",
                  "centered_instance", "centroid"):
        return ("a top-down model crops internally around its own centroid "
                "detections, cropping first would crop twice")
    if info.get("multi_animal") and not identities:
        return ("this model is multi-animal; one window would follow one of "
                "its animals and drop the rest")
    return ""


def confident_centroid(pose: dict, conf_min: float, good_min: int):
    """Steering point for a following window, from the rig's implementation.

    Falls back to the same arithmetic when the rig is absent, because this one
    is four lines and a retrack that already has poses should not be blocked
    for the want of them.
    """
    if _rig() is not None:
        try:
            from source.video.tracking.crop_tracker import (
                confident_centroid as _cc)
            return _cc(pose, conf_min, good_min)
        except Exception as e:
            _remember(str(e))
    xs, ys = [], []
    for value in (pose or {}).values():
        if value is None or len(value) < 3 or float(value[2]) < conf_min:
            continue
        xs.append(float(value[0]))
        ys.append(float(value[1]))
    if len(xs) < max(1, good_min):
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


__all__ = ["MODE_CROP_TRACK", "MODE_FULL", "MODE_LETTERBOX", "available",
           "confident_centroid", "make", "model_info", "why_unavailable"]
