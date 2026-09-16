"""Re-run tracking over a saved video, DeepLabCut, SLEAP, or the blob detector.

The offline path used to contain its own copy of DLC inference: it imported
``dlclive`` directly, read body-part names off an attribute that does not
exist (so every keypoint came back called ``bp_0``), timestamped poses
``frame_index / container_fps``, and could not reach SLEAP or the blob
detector at all even though the dialog offered them.

None of that is necessary. ``source.tracking`` already holds the tracker
classes the live pipeline uses, tested and with the body-part problem solved.
This module is orchestration only: open a video, hand frames to a tracker,
timestamp the results from a :class:`~source.analysis.clock.ClockModel`, and
stop when asked.

The tracker is injectable, so the orchestration is fully unit-tested with
no model, no GPU and no video, while the backends themselves are exercised on
the rig.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from tools.offline_analysis.engine.clock import ClockModel, SOURCE_INDEX

from tools.offline_analysis.engine.trackers import (MODE_CROP_TRACK,
                                                    MODE_FULL,
                                                    MODE_LETTERBOX)

BACKEND_DLC = "deeplabcut"
BACKEND_SLEAP = "sleap"
BACKEND_BLOB = "blob"

#: How many frames are sampled to build a blob background from the video
#: itself. There is no live camera offline, so the "point it at an empty
#: arena" step is not available, the median over a spread of frames is.
BG_SAMPLES = 60


# ── availability ─────────────────────────────────────────────────────────────

@dataclass
class BackendStatus:
    name: str
    available: bool
    reason: str = ""
    needs_model: bool = True

    @property
    def label(self) -> str:
        return self.name if self.available else f"{self.name}, {self.reason}"


def available_backends() -> Dict[str, BackendStatus]:
    """Which backends can actually run here, and why not when they cannot.

    Checked when the user chooses, not when the run starts. A six-hour job
    that dies on an import at minute one is the failure this prevents.
    """
    from tools.offline_analysis.engine import trackers as _tk

    have = _tk.backends()
    return {
        BACKEND_DLC: BackendStatus(
            BACKEND_DLC, bool(have.get("dlc")),
            "" if have.get("dlc") else "dlclive is not installed in this interpreter"),
        BACKEND_SLEAP: BackendStatus(
            BACKEND_SLEAP, bool(have.get("sleap")),
            "" if have.get("sleap") else "sleap / sleap-nn is not importable here"),
        # Blob needs OpenCV, which the whole app needs, and never needs a model.
        BACKEND_BLOB: BackendStatus(
            BACKEND_BLOB, _has_cv2(),
            "" if _has_cv2() else "OpenCV is not available", needs_model=False),
    }


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:                                   # pragma: no cover
        return False


# ── the spec ─────────────────────────────────────────────────────────────────

@dataclass
class TrackSpec:
    """Everything that decides what the poses will be.

    Its :meth:`fingerprint` keys the stored result, which is what lets two
    retracks of one recording coexist and be compared instead of one silently
    overwriting the other.
    """

    backend: str = BACKEND_BLOB
    model_path: str = ""
    confidence: float = 0.55
    resize: float = 1.0
    skip: int = 1
    n_instances: int = 1
    roi: Optional[Tuple[int, int, int, int]] = None      # x, y, w, h
    params: Dict[str, Any] = field(default_factory=dict)  # blob params, extras
    gpu_id: str = "0"

    def __post_init__(self):
        self.backend = _canonical_backend(self.backend)

    @property
    def needs_model(self) -> bool:
        return self.backend in (BACKEND_DLC, BACKEND_SLEAP)

    def validate(self) -> List[str]:
        """Reasons this cannot run, in the user's words. Empty = ready."""
        errs: List[str] = []
        st = available_backends().get(self.backend)
        if st is None:
            errs.append(f"unknown backend '{self.backend}'")
        elif not st.available:
            errs.append(st.reason)
        if self.needs_model and not self.model_path:
            errs.append(f"{self.backend} needs a model, none is set")
        elif self.needs_model and not os.path.exists(self.model_path):
            errs.append(f"model not found: {self.model_path}")
        if self.skip < 1:
            errs.append("skip must be at least 1")
        errs.extend(self._input_mode_problems())
        return errs

    def _input_mode_problems(self) -> List[str]:
        """Why the chosen input mode cannot run, in the user's words.

        Checked when the mode is CHOSEN rather than when the run starts, a
        six-hour job that discovers this at minute one is the failure the
        readiness rules exist to prevent.
        """
        mode = str(self.params.get("input_mode") or MODE_FULL)
        if mode == MODE_FULL:
            return []
        if mode not in (MODE_LETTERBOX, MODE_CROP_TRACK):
            return [f"unknown input mode '{mode}'"]
        if mode != MODE_CROP_TRACK:
            return []
        from tools.offline_analysis.engine import trackers as _tk

        window = self.crop_window
        why = _tk.crop_refusal(_tk.model_info(self.model_path or ""),
                               self.n_instances, window,
                               identities=self.identities)
        return [f"crop-track cannot run: {why}"] if why else []

    @property
    def identities(self) -> List[str]:
        """The names of the animals, when the run has any.

        In ``params`` rather than a field of its own so ``fingerprint()``
        hashes them: a two-animal retrack named male/female is not the same
        result as one named A/B, and the two must not share a cache entry.
        """
        return [str(n) for n in (self.params.get("identities") or []) if str(n)]

    @property
    def crop_window(self) -> Optional[Tuple[int, int]]:
        """The training window: what was typed, else what the model declares."""
        from tools.offline_analysis.engine import trackers as _tk

        w = int(self.params.get("input_w", 0) or 0)
        h = int(self.params.get("input_h", 0) or 0)
        if w > 0 and h > 0:
            return (w, h)
        info = _tk.model_info(self.model_path or "")
        iw, ih = info.get("input_w"), info.get("input_h")
        if iw and ih:
            return (int(iw), int(ih))
        # A square top-down crop is the model's other way of stating it.
        side = info.get("crop_size")
        return (int(side), int(side)) if side else None

    def fingerprint(self) -> str:
        """Stable short key over everything that changes the output."""
        payload = {
            "backend": self.backend,
            "model": os.path.basename(self.model_path.rstrip("/\\")),
            "model_sha": _model_fingerprint(self.model_path),
            "confidence": round(float(self.confidence), 4),
            "resize": round(float(self.resize), 4),
            "skip": int(self.skip),
            "n_instances": int(self.n_instances),
            "roi": list(self.roi) if self.roi else None,
            "params": {k: self.params[k] for k in sorted(self.params)},
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]

    def tracker_info(self, body_parts: Sequence[str]) -> dict:
        """The ``tracker`` header block: which backend, which model, what it
        was asked to do. Without it a reader cannot tell a blob centroid from
        a DLC keypoint set except by guessing from the names."""
        block = {
            "backend": self.backend,
            "model_id": os.path.basename(self.model_path.rstrip("/\\")),
            "model_path": self.model_path,
            "model_sha": _model_fingerprint(self.model_path),
            "runtime": "offline",
            "body_parts": list(body_parts),
            "n_animals": int(self.n_instances),
            # Ordered, because the columns are numbered rather than named:
            # `nose` is identities[0], `nose#2` is identities[1]. Without this
            # list a reader can count the animals but cannot name them.
            "identities": list(self.identities),
            "identity_method": "none",
            "coord_space": "px",
            # HOW the detector was tuned. Without it two blob retracks of one
            # clip are indistinguishable on disk, same backend, same model
            # (none), different numbers, so nothing downstream could say what
            # changed between them, including the parity check whose whole job
            # is to answer that.
            "params": {k: self.params[k] for k in sorted(self.params)
                       if k not in ("backgrounds", "identities")},
            "confidence": float(self.confidence),
            "resize": float(self.resize),
            "skip": int(self.skip),
            # HOW the frame became the model's input, which changes the poses
            # and until now was not recorded anywhere. Two sessions that differ
            # by this looked identical on disk.
            #
            # NOT under the key `runtime`: that one already exists here and
            # already means "offline", live versus offline, not torch versus
            # TensorRT. Re-using it would silently change the meaning of a
            # field in every recording already written.
            "input_mode": str(self.params.get("input_mode") or "full"),
            # The unambiguous half of the pair (see `TrackerSpec.tracker_info`):
            # `runtime` above says "offline" and means live-versus-offline,
            # while the live writer's `runtime` says "native" and means the
            # engine. Both meanings are already on disk, so neither is
            # changed and `source` is written beside them.
            "source": "offline",
            "fingerprint": self.fingerprint(),
        }
        for key in ("input_w", "input_h"):
            if self.params.get(key):
                block[key] = int(self.params[key])
        for key in ("engine", "precision", "device", "colour_mode",
                    "peak_threshold"):
            if self.params.get(key):
                block[key] = self.params[key]
        return block


def _canonical_backend(name: str) -> str:
    n = str(name or "").lower().strip()
    if n in ("dlc", "deeplabcut", "deep_lab_cut"):
        return BACKEND_DLC
    if n in ("sleap", "sleap-nn", "sleap_nn"):
        return BACKEND_SLEAP
    if n in ("blob", "background_subtraction", "bg", "simple"):
        return BACKEND_BLOB
    return n


def _model_fingerprint(path: str) -> str:
    """Identify the model well enough that two results can be told apart,
    without hashing gigabytes of weights: the sorted (name, size, mtime) of
    its files."""
    if not path or not os.path.isdir(path):
        return ""
    parts: List[str] = []
    try:
        for root, _dirs, files in os.walk(path):
            for fn in sorted(files):
                fp = os.path.join(root, fn)
                try:
                    st = os.stat(fp)
                except OSError:
                    continue
                parts.append(f"{os.path.relpath(fp, path)}:{st.st_size}")
            if len(parts) > 400:
                break
    except OSError:
        return ""
    return hashlib.sha1("|".join(sorted(parts)).encode("utf-8")).hexdigest()[:12]


# ── the result ───────────────────────────────────────────────────────────────

@dataclass
class PoseStream:
    """Poses for one video, with a record of what produced them."""

    frame_numbers: List[int] = field(default_factory=list)
    ts_ms: np.ndarray = field(default_factory=lambda: np.empty(0))
    poses: List[Optional[dict]] = field(default_factory=list)
    body_parts: List[str] = field(default_factory=list)
    tracker: Dict[str, Any] = field(default_factory=dict)
    #: True when the run was cancelled or died part-way. The frames that were
    #: tracked are kept, throwing them away would make a six-hour job an
    #: all-or-nothing bet.
    partial: bool = False
    n_frames_expected: int = 0
    #: (w, h) of the frames the tracker actually saw. These poses are in that
    #: space, and a re-track that does not say so reintroduces the gap
    #: `pose_resolution` exists to close.
    frame_size: Tuple[int, int] = (0, 0)
    elapsed_s: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.poses)

    @property
    def fps_achieved(self) -> float:
        return len(self.poses) / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def detected_fraction(self) -> float:
        """Frames where the detector actually LOCATED something.

        A pose dict of six ``None`` values has the shape of a result and the
        content of a failure, and counting it reported "100% with a detection"
        for a run whose every frame was empty, precisely the silent success
        this project keeps having to remove. A frame counts only if at least
        one keypoint has a position.
        """
        if not self.poses:
            return 0.0
        found = sum(1 for p in self.poses
                    if p and any(v is not None for v in p.values()))
        return found / len(self.poses)

    def xy(self, body_part: str) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.poses)
        x = np.full(n, np.nan)
        y = np.full(n, np.nan)
        for i, p in enumerate(self.poses):
            v = (p or {}).get(body_part)
            if v and len(v) >= 2 and v[0] is not None:
                x[i], y[i] = float(v[0]), float(v[1])
        return x, y


def prefetch(source: Iterable, depth: int = 64) -> Iterator:
    """Read ``source`` on a second thread, a bounded number of items ahead.

    The retrack loop decoded a frame, waited for the model, then decoded the
    next: the GPU waited for the decoder and the decoder waited for the GPU,
    and neither was ever busy at the same time. Measured on a 360x202 mp4,
    decoding costs 0.24 ms a frame against 0.60 ms of batched inference, so
    the whole of it can hide behind the model.

    BOUNDED, deliberately. An unbounded queue in front of a slow model reads
    an entire recording into memory, 72,000 frames of this video is 15 GB,
    and the first sign of it is the machine swapping.

    The producer's own exception is re-raised in the consumer rather than
    killing a daemon thread quietly, so a decode failure still stops the run
    with its reason.
    """
    queue_: "queue.Queue" = queue.Queue(maxsize=max(1, depth))
    done = object()

    def produce() -> None:
        try:
            for item in source:
                queue_.put(item)
        except BaseException as e:                # carried to the consumer
            queue_.put(e)
        finally:
            queue_.put(done)

    worker = threading.Thread(target=produce, name="retrack-decode",
                              daemon=True)
    worker.start()
    while True:
        item = queue_.get()
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


class Cancelled(RuntimeError):
    """Raised inside the worker when the user asked it to stop."""


class TrackerInitError(RuntimeError):
    """Raised when the detector will not load, before any frame is tracked.

    Its own class rather than a bare RuntimeError so the tab can tell "this
    model cannot run here", an environment problem the operator can fix,
    apart from "this recording failed", and say so without parsing a message.
    """


# ── tracker adapters ─────────────────────────────────────────────────────────

class _Adapter:
    """One shape for three backends: give it a frame, get a pose dict."""

    def __init__(self, tracker, spec: TrackSpec):
        self.t = tracker
        self.spec = spec
        self._parts: List[str] = []

    def start(self, frame) -> None:
        init = getattr(self.t, "initialize", None)
        started = None
        if init is not None:
            try:
                started = init(frame)
            except TypeError:                     # BlobTracker.initialize(frame, roi)
                started = init(frame, None)
        # A detector that will not load stops the recording HERE, with the
        # reason. Ignored, it yields an empty pose for every frame and finishes
        # hours later reporting "0% with a detection", which reads as a bad
        # model rather than one that never loaded at all. `started is None` is
        # a tracker that does not report,
        # several do, and is not treated as failure.
        reason = str(getattr(self.t, "failure", "") or "")
        if started is False or reason:
            raise TrackerInitError(
                reason or "the detector would not load, and gave no reason")
        gp = getattr(self.t, "get_body_parts", None)
        if gp is not None:
            try:
                self._parts = list(gp() or [])
            except Exception:
                self._parts = []

    @property
    def body_parts(self) -> List[str]:
        return self._parts

    def seed(self, centre) -> None:
        """Put the crop window(s) where the recording says the animal was.

        A mapping is N animals: identity to where that one was, which is what
        a multi-animal recording can say and a single-animal one cannot. A
        bare point is the one-window case, unchanged.
        """
        if isinstance(centre, dict):
            st = self._multi_state()
            if st is not None:
                st.seed(centre)
            return
        setter = getattr(self.t, "set_crop_seed", None)
        if setter is not None and centre is not None:
            setter(centre)

    def _multi_state(self):
        fn = getattr(self.t, "multi_crop_state", None)
        return fn(None, self.spec.identities) if fn is not None else None

    def crop_stats(self) -> Dict[str, int]:
        """What the window did, when there was one."""
        getter = getattr(self.t, "crop_stats", None)
        try:
            return dict(getter() or {}) if getter is not None else {}
        except Exception:                                 # pragma: no cover
            return {}

    def pose(self, frame) -> Optional[dict]:
        pcut = float(self.spec.confidence)
        if self.spec.n_instances > 1 and hasattr(self.t, "detect"):
            insts = self._instances(frame)
            out: Dict[str, list] = {}
            for i, inst in enumerate(insts[: self.spec.n_instances]):
                for bp, v in (getattr(inst, "body_parts", None) or {}).items():
                    key = bp if i == 0 else f"{bp}#{i + 1}"
                    out[key] = _kp(v, pcut)
            self._note_parts(out)
            return out or None
        predict = getattr(self.t, "predict", None)
        if predict is not None:
            raw = predict(frame) or {}
            out = {bp: _kp(v, pcut) for bp, v in raw.items()}
            self._note_parts(out)
            return out or None
        insts = self.t.detect(frame) or []
        if not insts:
            return None
        out = {bp: _kp(v, pcut)
               for bp, v in (getattr(insts[0], "body_parts", None) or {}).items()}
        self._note_parts(out)
        return out or None

    @property
    def can_batch(self) -> bool:
        """Whether these frames may be submitted together.

        Two modes are inherently one-at-a-time and must not be batched:

        * **crop-track**, where the window for frame N is placed from frame
          N-1's answer, a batch would put every window in the same place;
        * **multi-animal**, which goes through ``detect``/``detect_identities``
          rather than ``predict``, and whose per-frame identity assignment is
          the thing that keeps a column meaning one animal.
        """
        if str((self.spec.params or {}).get("input_mode") or "") == MODE_CROP_TRACK:
            return False
        if self.spec.n_instances > 1:
            return False
        return bool(getattr(self.t, "batched", False))

    def poses(self, frames) -> List[Optional[dict]]:
        """One pose per frame, from a single forward pass where possible.

        The same gating and naming as :meth:`pose`; it is the transport that
        changes, not the answer. A test asserts the two agree frame for frame,
        because a faster retrack that disagreed with the slow one would be
        worth nothing.
        """
        if not frames:
            return []
        if not self.can_batch:
            return [self.pose(f) for f in frames]
        pcut = float(self.spec.confidence)
        out: List[Optional[dict]] = []
        for raw in self.t.predict_batch(list(frames)):
            pose = {bp: _kp(v, pcut) for bp, v in (raw or {}).items()}
            self._note_parts(pose)
            out.append(pose or None)
        return out

    def _instances(self, frame):
        """This frame's animals, in a STABLE order.

        With per-identity windows the order is the identities' order, because
        the columns are numbered (`nose`, `nose#2`) and a run whose animals
        swap columns from frame to frame is worse than useless. Whoever is in
        window one is column one, every frame.

        Without identities it is whatever the detector returned, unchanged.
        """
        names = self.spec.identities
        fn = getattr(self.t, "detect_identities", None)
        if not names or fn is None:
            return self.t.detect(frame) or []
        # Collisions are NOT counted here: the windows already count them and
        # `crop_stats()` already reports them. A second tally would be a second
        # answer to the same question.
        found = {i.identity: i for i in (fn(frame, identities=names) or [])}
        return [found[n] for n in names if n in found]

    def _note_parts(self, pose: dict) -> None:
        if not self._parts:
            self._parts = list(pose.keys())

    def close(self) -> None:
        fn = getattr(self.t, "close", None) or getattr(self.t, "stop", None)
        if fn is not None:
            try:
                fn()
            except Exception:
                pass


def _kp(v, pcut: float) -> Optional[list]:
    """One keypoint, gated by confidence.

    Below the cut the keypoint is absent, ``None``, not a coordinate the
    caller might use anyway, and not a dropped frame either: the other
    keypoints of that frame are still perfectly good.
    """
    if v is None or len(v) < 2 or v[0] is None:
        return None
    conf = float(v[2]) if len(v) > 2 and v[2] is not None else 1.0
    if conf < pcut:
        return None
    return [float(v[0]), float(v[1]), conf]


def open_tracker(spec: TrackSpec, frame_size: Tuple[int, int] = (0, 0)):
    """Build the tracker the spec asks for, from the app's own classes."""
    from tools.offline_analysis.engine import trackers as _tk

    if spec.backend == BACKEND_BLOB:
        # `setup_id`, which is what the rig's BlobTracker takes. It was
        # `box_id`, so building it raised TypeError, `make` returned None, and
        # the run failed, with an EMPTY message, because `why_unavailable`
        # answers "is any tracker available" and blob was, right up until it
        # was asked to exist.
        t = _tk.make("blob", setup_id=0)
        if t is None:
            raise RuntimeError(
                _tk.last_error() or _tk.why_unavailable()
                or "the blob tracker could not be built")
        params = dict(spec.params or {})
        params.pop("backgrounds", None)
        if params:
            try:
                t.update_params(**{k: v for k, v in params.items()
                                   if k in _BLOB_PARAMS})
            except Exception:
                pass
        # Wrapped, not returned raw: the loop asks a detector for
        # `predict(frame)`, and this one answers `update(frame)` with a
        # bounding box. No `max_animals` is set: the rig's blob tracker has no
        # such attribute, so assigning one creates a field nothing reads, on a
        # detector that finds one animal.
        return _tk.BlobAsPose(t)

    # The GPU selector only works before the framework imports, and the old
    # code set it inside the worker after everything was already loaded.
    if spec.gpu_id:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(spec.gpu_id))
    # The same kwargs the LIVE pipeline builds (see
    # `tracking_controller._init_pose_model`): confidence, resize, and the
    # body-part names from the config when it carries them. Offline used to
    # drop the names, so a model whose parts are declared in the tracking
    # config rather than in its own folder came back unnamed.
    kw: Dict[str, Any] = {"confidence_threshold": float(spec.confidence)}
    if spec.resize and spec.resize != 1.0:
        kw["resize_factor"] = float(spec.resize)
    parts = spec.params.get("body_parts") or []
    if parts:
        kw["body_parts"] = list(parts)
    # How the frame becomes the model's input travels in `params`, which
    # `fingerprint()` hashes whole, so a retrack under a different input mode
    # gets its own stored result instead of being served the old one. A field
    # on the dataclass would NOT be hashed, which is the trap this avoids.
    mode = str(spec.params.get("input_mode") or MODE_FULL)
    if mode != MODE_FULL:
        kw["input_mode"] = mode
        w = int(spec.params.get("input_w", 0) or 0)
        h = int(spec.params.get("input_h", 0) or 0)
        if w > 0 and h > 0:
            kw["input_size"] = (w, h)
        elif mode == MODE_CROP_TRACK:
            # The window is the model's, not the frame's. Falling back to the
            # frame size here would silently turn crop-track into "no crop at
            # all", the mode would appear to run and change nothing.
            kw["input_size"] = spec.crop_window
        elif frame_size and frame_size[0] and frame_size[1]:
            kw["input_size"] = (int(frame_size[0]), int(frame_size[1]))
    if mode == MODE_CROP_TRACK:
        for key, arg in (("crop_conf_min", "crop_conf_min"),
                         ("crop_good_min", "crop_good_min"),
                         ("crop_reacquire", "crop_reacquire")):
            if key in spec.params:
                kw[arg] = spec.params[key]
        kw["max_instances"] = int(spec.n_instances)
        # Named animals are what makes multi-animal crop-track legal: one
        # window per identity. Without names the tracker refuses rather than
        # following one animal and dropping the rest.
        kw["identities"] = spec.identities
    # Which engine runs the model, and in what precision, on what device. All
    # three live in `params`, so `fingerprint()` hashes them and an fp16
    # retrack cannot be served the fp32 result.
    for key in ("runtime", "device", "precision", "fp16", "compile",
                "peak_threshold", "max_batch_size", "dynamic_crop",
                "dynamic_threshold", "dynamic_margin"):
        if spec.params.get(key) is not None:
            kw[key] = spec.params[key]
    # Passed under the backend's own name, never as a bare `model_type`: the
    # two backends spell the option the same way and read completely different
    # vocabularies from it (DLC base|pytorch|tensorrt|lite, SLEAP
    # auto|single|centroid|topdown). `_pose_kwargs` refuses the bare form.
    for key in ("dlc_model_type", "sleap_model_type"):
        if spec.params.get(key):
            kw[key] = spec.params[key]
    if spec.backend == BACKEND_SLEAP and spec.params.get("centroid_model_path"):
        kw["centroid_model_path"] = spec.params["centroid_model_path"]
    t = _tk.make(spec.backend, model_path=spec.model_path, **kw)
    if t is None:
        raise RuntimeError(
            _tk.why_backend_unavailable(spec.backend) or _tk.last_error()
            or f"the {spec.backend} tracker could not be built")
    if t is None:
        # The rig is not here, or its SDK is not installed. Said now,
        # with the reason, rather than at the first frame of a six-hour
        # run.
        raise RuntimeError(_tk.why_unavailable())
    return t


_BLOB_PARAMS = {
    "threshold", "min_area", "max_area", "detect_dark", "use_clahe",
    "clahe_clip_limit", "clahe_tile_size", "use_adaptive_threshold",
    "use_illumination_norm", "blur_kernel_size", "blur_mode", "bg_mode",
    "open_kernel_size", "close_kernel_size", "noise_min_area",
    "aspect_ratio_max", "solidity_min",
}


# ── frames ───────────────────────────────────────────────────────────────────

def iter_frames(video_path: str, *, skip: int = 1,
                roi: Optional[Tuple[int, int, int, int]] = None
                ) -> Iterator[Tuple[int, Any]]:
    """Yield ``(frame_index, frame)``.

    Skipped frames are not yielded at all, the caller records a gap at their
    true timestamp rather than compressing time, which is the difference
    between "we did not look" and "there was nothing there".
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % skip == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield idx, _crop(frame, roi)
            idx += 1
    finally:
        cap.release()


def _crop(frame, roi):
    if roi is None or frame is None:
        return frame
    x, y, w, h = (int(v) for v in roi)
    return frame[y:y + h, x:x + w]


def build_blob_background(video_path: str, tracker, *, samples: int = BG_SAMPLES,
                          roi=None, cancel: Optional[threading.Event] = None
                          ) -> bool:
    """A background for offline blob tracking, from the video itself.

    Sampled across the whole recording and combined with the median, so the
    animal, which is somewhere different in most frames, falls out. A mean
    would leave its ghost smeared along the path it walked, exactly where the
    detector then needs to see it.
    """
    import cv2

    from tools.offline_analysis.engine import trackers as _tk

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return False
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return False
        step = max(1, total // max(1, samples))
        wanted = iter(range(0, total, step))

        def next_sample():
            """One frame from across the recording, or None when they run out."""
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            try:
                i = next(wanted)
            except StopIteration:
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            return _crop(frame, roi) if ok else None

        # The RIG's median, through the seam: this module does not reach
        # into `source` itself, `engine/trackers.py` is the one place that
        # does, and the perimeter test enforces it.
        background = _tk.median_background(next_sample, samples=samples)
    finally:
        cap.release()
    if background is None:
        return False
    prime = getattr(tracker, "prime", None)
    return bool(prime(background)) if prime is not None else False


# ── the run ──────────────────────────────────────────────────────────────────

def track_video(video_path: str, spec: TrackSpec, clock: ClockModel, *,
                progress: Optional[Callable[[int, int, float], None]] = None,
                cancel: Optional[threading.Event] = None,
                frames: Optional[Iterable[Tuple[int, Any]]] = None,
                tracker: Optional[Any] = None,
                seed_fn: Optional[Callable[[int], Optional[Tuple[float, float]]]] = None,
                n_frames: int = 0) -> PoseStream:
    """Track a video and return the pose stream.

    ``frames`` and ``tracker`` are injectable, which is what makes the whole
    orchestration testable without a model, a GPU or a video file.

    ``seed_fn(frame_number)`` is the offline asymmetry: a re-track already
    knows where the animal was, because the recording says so. In crop-track
    mode the window is put there before each frame instead of having to find
    it, so re-acquisition becomes a fallback for frames the old tracker also
    lost rather than the normal recovery path.

    Timestamps come from ``clock``. There is deliberately no ``fps`` argument:
    the one thing this function must not do is invent time.
    """
    started = time.monotonic()
    stream = PoseStream(n_frames_expected=n_frames or clock.n_frames)
    if clock.source == SOURCE_INDEX:
        stream.warnings.append(
            "no clock: timestamps are frame indices, so nothing measured per "
            "second is meaningful")

    own_tracker = tracker is None
    if own_tracker:
        tracker = open_tracker(spec)
        if spec.backend == BACKEND_BLOB and video_path:
            try:
                if not build_blob_background(video_path, tracker, roi=spec.roi,
                                             cancel=cancel):
                    stream.warnings.append(
                        "could not build a background from this video, blob "
                        "detection needs one")
            except Cancelled:
                stream.partial = True
                return stream

    adapter = _Adapter(tracker, spec)
    src = frames if frames is not None else iter_frames(
        video_path, skip=spec.skip, roi=spec.roi)
    # Decode ahead of the model. Only for frames this function opened itself:
    # an injected iterator belongs to the caller, and a test that hands one in
    # should not have it consumed on another thread.
    if frames is None and int((spec.params or {}).get("prefetch", 64) or 0) > 0:
        src = prefetch(src, int((spec.params or {}).get("prefetch", 64)))
    started_adapter = False
    idxs: List[int] = []
    # How many frames go to the model at once. One at a time leaves the GPU
    # idle between kernel launches on a network this small; sixteen is 4.8x
    # cheaper per frame on the 6-point SLEAP export. Batching is refused
    # outright for the modes that must stay sequential, see `can_batch`.
    batch_size = max(1, int((spec.params or {}).get("batch_size", 16) or 1))
    pending: List[Any] = []
    pending_idx: List[int] = []

    def flush() -> None:
        """Send whatever has accumulated, and record it in frame order."""
        if not pending:
            return
        stream.poses.extend(adapter.poses(pending))
        idxs.extend(pending_idx)
        pending.clear()
        pending_idx.clear()
        if progress is not None:
            el = time.monotonic() - started
            progress(len(idxs), stream.n_frames_expected,
                     len(idxs) / el if el > 0 else 0.0)

    try:
        for frame_idx, frame in src:
            if cancel is not None and cancel.is_set():
                flush()                     # keep what was already inferred
                stream.partial = True
                break
            if not started_adapter:
                adapter.start(frame)
                started_adapter = True
                try:
                    stream.frame_size = (int(frame.shape[1]),
                                         int(frame.shape[0]))
                except Exception:               # not an ndarray in a test
                    pass
            if seed_fn is not None:
                adapter.seed(seed_fn(int(frame_idx)))
            pending.append(frame)
            pending_idx.append(int(frame_idx))
            if len(pending) >= (batch_size if adapter.can_batch else 1):
                flush()
        else:
            flush()                         # the tail, when the source ran dry
    except Cancelled:
        flush()
        stream.partial = True
    finally:
        if own_tracker:
            adapter.close()

    stream.frame_numbers = idxs
    stream.body_parts = list(adapter.body_parts)
    stream.elapsed_s = time.monotonic() - started
    stream.ts_ms = _timestamps_for(idxs, clock)
    stream.tracker = spec.tracker_info(stream.body_parts)
    # What the window did is part of what produced these poses: a run that
    # re-acquired forty times is worth looking at even when every frame ended
    # up with a pose.
    stats = adapter.crop_stats()
    if stats:
        stream.tracker["crop"] = stats
        if stats.get("lost"):
            stream.warnings.append(
                f"the crop window lost the animal on {stats['lost']} of "
                f"{stats['frames']} frames ({stats.get('reacquisitions', 0)} "
                f"re-acquisitions)")
    if progress is not None:
        progress(len(idxs), stream.n_frames_expected, stream.fps_achieved)
    return stream


def _timestamps_for(frame_numbers: Sequence[int], clock: ClockModel) -> np.ndarray:
    """Each tracked frame's time, from the clock, by frame number.

    Not ``index * 1000 / fps``: after a dropped frame or on a variable-rate
    recording those are different numbers, and the second one is wrong.
    """
    if not frame_numbers:
        return np.empty(0)
    from tools.offline_analysis.engine.clock import align_by_frame_number

    if clock.known_per_frame:
        return align_by_frame_number(frame_numbers, clock)
    step = 1000.0 / clock.fps if clock.fps > 0 else 1.0
    return np.asarray([fn * step for fn in frame_numbers], float)


# ── writing the result ───────────────────────────────────────────────────────

def write_pose_stream(out_path: str, stream: PoseStream, *,
                      source_header=None, space=None,
                      stage_by_frame: Optional[Dict[int, str]] = None,
                      mcu_ts_by_frame: Optional[Dict[int, str]] = None,
                      body_part: str = "",
                      source_locations: Optional[List[str]] = None) -> str:
    """Write a pose stream as a complete, valid session file.

    The carry-over rule in action: the source session's header travels with
    the poses, so ``units``/``px_per_cm``, ``start_time``, ``metadata``,
    ``metric_config`` and the zones all survive a retrack. Losing them is what
    turned a calibrated, staged, zoned session into an anonymous one.
    """
    from tools.offline_analysis import video_data_schema as vds

    if source_header is not None:
        header = source_header.with_tracker(stream.tracker, stream.body_parts)
        # The carried-over header describes the ORIGINAL poses. These are new
        # ones, produced from frames of a size we know exactly, so the pose
        # space is stated afresh rather than inherited, inheriting it is how
        # a re-track would keep a stale or missing value alive.
        if stream.frame_size and stream.frame_size[0]:
            header.info["pose_resolution"] = (f"{int(stream.frame_size[0])}"
                                              f"x{int(stream.frame_size[1])}")
        # A scale drawn while analysing beats the one the recorder wrote,
        # usually because the recorder wrote none. Carrying the header's 0
        # through produced a re-tracked file that had lost the calibration
        # the operator had just supplied: its speed column came out in px/s
        # and anything reading it on its own reported pixels, however clearly
        # the panel said 11.5 px/cm.
        if space is not None and getattr(space, "calibrated", False):
            units = dict(header.units or {})
            units["px_per_cm"] = float(space.px_per_cm)
            units["speed"] = "m/s"
            # `units` is a read-only view over the raw header line, so the
            # line itself is what gets rewritten.
            header.info["units"] = json.dumps(units, separators=(",", ":"))
    else:
        header = vds.header_for_new_session(
            body_parts=stream.body_parts, tracker=stream.tracker,
            px_per_cm=(space.px_per_cm if space else 0.0),
            zones=(space.zones if space else None),
            resolution=stream.frame_size,
            pose_resolution=stream.frame_size,
            frame_clock="offline_retrack")

    ppc = header.px_per_cm
    loc = _locations(stream, space, body_part, source_locations)
    speeds = _speeds(stream, space, body_part, ppc)

    rows = []
    for i, pose in enumerate(stream.poses):
        fn = stream.frame_numbers[i] if i < len(stream.frame_numbers) else i
        t = stream.ts_ms[i] if i < len(stream.ts_ms) else math.nan
        rows.append({
            "frame_number": fn,
            "frame_ts_ms": "none" if math.isnan(t) else int(round(t)),
            "pose_ts_ms": "none" if math.isnan(t) else int(round(t)),
            # Carried from the recording, not blanked. The rig stamps the
            # MCU's own clock against 75,651 of this session's 75,668 frames,
            # and that is the exact alignment between behaviour and task
            # state, hardcoding "NA" threw it away and left every retracked
            # file to re-derive the join from elapsed time instead.
            "mcu_ts_ms": (mcu_ts_by_frame or {}).get(fn, "NA"),
            "speed": speeds[i],
            "location": loc[i] or "none",
            "pose_array": vds.format_pose(pose),
            "stage": (stage_by_frame or {}).get(fn, ""),
        })
    return vds.write_session(out_path, header, rows)


def _locations(stream: PoseStream, space, body_part: str,
               source_locations=None) -> List[str]:
    """Which zone each frame was in.

    THE RECORDING'S OWN COLUMN WINS. It was decided at record time, in the
    space the poses are actually in, against these same zones -- and that is
    knowledge this function cannot reconstruct. Recomputing it here assigns
    normalized zones using the frame the header DECLARES, which on a rig that
    encodes the video smaller than it reads the camera is the video's size
    while the poses are in the camera's. Correcting one Y-maze recording that
    way turned 4,217 Home_arm frames into 119 and sent 80% of the session to
    "none", and because a written correction supersedes the original for
    everything downstream, every later measurement inherited it.

    Only a re-track earns a recomputation: those poses are new, and the
    caller states the frame they are in.
    """
    n = len(stream.poses)
    if source_locations is not None and len(source_locations) >= n:
        return [str(v or "") for v in source_locations[:n]]
    if space is None or not getattr(space, "zones", None):
        return [""] * n
    from tools.offline_analysis.engine.space import rezone

    bp = body_part or _default_part(stream.body_parts)
    x, y = stream.xy(bp) if bp else (np.full(len(stream), np.nan),) * 2
    res = rezone(space, body_xy=(x, y), dt=np.zeros(len(stream)))
    return res.location


def _speeds(stream: PoseStream, space, body_part: str, ppc: float) -> List[Any]:
    bp = body_part or _default_part(stream.body_parts)
    n = len(stream.poses)
    if not bp or n == 0:
        return ["none"] * n
    x, y = stream.xy(bp)
    t = np.asarray(stream.ts_ms, float)
    out: List[Any] = ["0"] * n
    scale = ppc if ppc > 0 else 1.0
    for i in range(1, n):
        dt = (t[i] - t[i - 1]) / 1000.0 if i < len(t) else 0.0
        if dt <= 0 or not (np.isfinite(x[i]) and np.isfinite(x[i - 1])):
            out[i] = "0"
            continue
        d = math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) / scale
        out[i] = round(d / dt, 4)
    return out


def _default_part(parts: Sequence[str]) -> str:
    """The point a single-animal measure should follow."""
    for want in ("center", "centre", "body", "middle", "back"):
        for p in parts:
            if p.lower() == want:
                return p
    return parts[0] if parts else ""
