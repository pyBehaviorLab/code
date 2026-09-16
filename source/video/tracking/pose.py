"""Unified pose-estimation interface.

Supports DLC-Live and SLEAP-NN backends with identical output format:
Dict[body_part_name, (x, y, confidence)] per frame.

Usage:
    tracker = create_pose_tracker("sleap", model_path="path/to/model")
    tracker.initialize(first_frame)
    pose = tracker.predict(frame)  # {"head": (x, y, 0.99), "center": (x, y, 0.95), ...}
"""

import logging
import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from source.video.tracking.model_config import GRAYSCALE, RGB, ModelInfo

logger = logging.getLogger(__name__)


def _to_ndarray(x):
    """Coerce a torch tensor / array-like to a detached numpy array. Kept
    dependency-light so the output parser is unit-testable with plain arrays."""
    if x is None:
        return None
    if hasattr(x, "detach"):        # torch.Tensor
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    return np.asarray(x)


def _dlclive_backends() -> str:
    """Which engines the installed DeepLabCut-Live can actually drive.

    A DLC model folder is TensorFlow or PyTorch, and dlclive only carries the
    runner whose framework is installed. Naming the set turns "init failed"
    into "this build of dlclive has only the pytorch runner, and this is a
    TensorFlow model".
    """
    try:
        import dlclive
        names = [getattr(b, "value", str(b))
                 for b in (dlclive.get_available_backends() or ())]
        return ", ".join(names) or "no backend"
    except Exception:                                     # pragma: no cover
        return "an unknown set of backends"


def _inference_device_after_import() -> Tuple[bool, str]:
    """``(cuda, description)`` for the pose network, queried AFTER the DL
    library has imported its framework, so torch/TF are already loaded and
    the probe is cheap (a dict lookup, not a fresh multi-second import)."""
    from source.video.gpu import inference_device
    # allow_import=True is safe here: dlclive/sleap already imported the
    # framework, so this is a dict lookup, not a fresh multi-second import.
    return inference_device(allow_import=True)


class PoseTracker:
    """Base class for pose estimation backends."""

    def __init__(self, model_path: str, resize_factor: float = 1.0,
                 colour_mode: str = "auto"):
        self.model_path = model_path
        self.resize_factor = resize_factor
        #: ``auto`` follows the model config; ``rgb``/``grayscale`` override it.
        self.colour_mode = (colour_mode or "auto").lower()
        self._initialized = False
        self._body_parts: List[str] = []
        self._model_info: Optional[ModelInfo] = None

    @property
    def model_info(self) -> ModelInfo:
        """What the model's own config file declares, read once and cached.

        Read from disk, not from a constructed backend object, so it is
        available before the model is built and on a machine with no inference
        stack installed.
        """
        if self._model_info is None:
            self._model_info = ModelInfo.read(self.model_path)
        return self._model_info

    def initialize(self, frame: Optional[np.ndarray] = None) -> bool:
        """Load model. Some backends need a frame for lazy init."""
        raise NotImplementedError

    def predict(self, frame: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
        """Run inference on a single frame.

        Returns:
            Dict mapping body_part_name -> [x, y, confidence] or None.
            ALL body parts are always included. None if not detected.
        """
        raise NotImplementedError

    def predict_batch(self, frames: List[np.ndarray]) -> List[Dict[str, Tuple[float, float, float]]]:
        """Run inference on a batch of frames.

        Default implementation falls back to serial ``predict`` calls so
        any subclass works out of the box. Subclasses should override
        this to use the model's true batched API (one forward pass on
        ``[N, H, W, C]``) for the multi-box throughput win.

        All frames MUST have the same H, W, C, the caller groups by
        shape before calling. Returns one pose dict per input frame, in
        order.
        """
        return [self.predict(f) for f in frames]

    def get_body_parts(self) -> List[str]:
        """Return ordered list of body part names from the model."""
        return list(self._body_parts)

    def _empty_result(self):
        """Return a pose dict with all body parts set to None."""
        return dict.fromkeys(self._body_parts)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def close(self):
        """Release resources."""
        self._initialized = False

    def _to_model_input(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Coerce the inference input to the channel count the model expects.

        The BGR→RGB conversion is done once per CameraFrame upstream
        (``BoxFrame.image_rgb``), not per box, so a 3-channel frame arrives
        ready. What this decides is what to do when the frame and the model
        disagree about channels.

        Refusing a grayscale frame outright would mean a model actually
        TRAINED on one channel could not be fed at all, and a grayscale camera
        could not run pose. The model's own config states its expectation
        (``ensure_grayscale`` / ``ensure_rgb``), so honour it:
        promote to 3 channels only for a model that wants 3, and pass one
        channel straight through to a model that wants one.
        """
        wants = self.expected_channels
        if frame.ndim == 2:
            if wants == GRAYSCALE:
                return frame                     # exactly what it was trained on
            # A model that wants colour cannot be given any: replicating the
            # single channel would feed it an image unlike its training data
            # and produce confident nonsense, which is worse than no pose.
            # An unstated expectation is treated the same way, refusing is the
            # safe reading of silence, but the message must not claim the file
            # said something it did not.
            if not getattr(self, "_warned_gray", False):
                why = ("the model expects colour" if wants == RGB else
                       "the model config does not state a channel expectation, "
                       "so colour is assumed")
                logger.error(
                    "%s: 2D grayscale frame (shape=%s) and %s, put the camera "
                    "in colour mode, or set the pose colour mode to grayscale "
                    "if the model was trained on one channel. "
                    "(logged once per session)",
                    type(self).__name__, frame.shape, why,
                )
                self._warned_gray = True
            return None
        if wants == GRAYSCALE and frame.ndim == 3 and frame.shape[2] == 3:
            # Colour camera, one-channel model: convert rather than refuse.
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame

    @property
    def resolved_backend(self) -> str:
        """What is ACTUALLY running, after any fallback, e.g.
        ``sleap_nn:tensorrt/fp16`` or ``dlc:base/FP32``.

        The requested runtime and the running one can differ: an engine that
        fails to build falls back to ONNX and then to native torch, which is
        the right behaviour but invisible if it is only ever logged. An
        operator who cannot see that they are on FP32 torch instead of
        TensorRT FP16 will not think to ask why the rig is slow.
        """
        base = getattr(self, "_backend", None) or type(self).__name__
        prec = getattr(self, "_precision", None)
        if prec is None and getattr(self, "_fp16", None) is not None:
            prec = "fp16" if self._fp16 else "fp32"
        return f"{base}/{prec}" if prec else str(base)

    @property
    def expected_channels(self) -> Optional[str]:
        """What the model wants fed to it, ``rgb`` | ``grayscale`` | None.

        ``colour_mode`` is the operator's override; ``auto`` (the default)
        defers to the model config, and None there means the file did not say,
        in which case nothing is coerced.
        """
        mode = (getattr(self, "colour_mode", "auto") or "auto").lower()
        if mode in (RGB, GRAYSCALE):
            return mode
        return self.model_info.channels


class DLCLiveTracker(PoseTracker):
    """DeepLabCut-Live backend.

    Requires: pip install deeplabcut-live
    Model: exported DLC model directory
    """

    def __init__(self, model_path: str, body_parts: Optional[List[str]] = None,
                 resize_factor: float = 1.0,
                 model_type: str = "auto", precision: str = "FP32",
                 device: str = "auto", colour_mode: str = "auto",
                 dynamic_crop: bool = False,
                 dynamic_threshold: float = 0.5,
                 dynamic_margin: int = 10,
                 max_batch_size: int = 1):
        super().__init__(model_path, resize_factor, colour_mode)
        self._body_parts = list(body_parts) if body_parts else ['head', 'center', 'tailbase']
        self._dlc_live = None
        # DeepLabCut-Live fixes these at construction (they select the runner),
        # so changing one needs a new instance, which is why they are part of
        # the model cache key rather than settable on a live tracker.
        self._model_type = (model_type or "auto").strip() or "auto"
        self._precision = (precision or "FP32").strip().upper() or "FP32"
        self._device = (device or "auto").strip() or "auto"
        #: Which of export / pytorch / tensorflow this run resolved to, and why
        #: it is not what was asked for. Settled in ``initialize``.
        self._engine = ""
        self._engine_note = ""
        #: How many boxes may ride in one pass. Reaches the engine choice
        #: because ``auto`` prefers the PyTorch runner on a multi-box rig,
        #: it is the only one that can take a batch at all.
        self._max_batch_size = max(1, int(max_batch_size))
        #: An exported ONNX/TensorRT graph, when the folder ships one and it
        #: runs here. Set by `_start_export`; None means DeepLabCut-Live.
        self._export = None
        #: Set after construction so ``resolved_backend`` reports what the
        #: model was actually built as, including a fallback to the baseline.
        self._backend = f"dlc:{self._model_type}"
        #: One notice per model that a multi-box batch runs serially here,
        #: per batch it would be log spam on the inference path.
        self._serial_batch_announced = False
        #: DeepLabCut-Live crops to the bounding box of the last detected
        #: keypoints and adds the offsets back on output, so this is its own
        #: version of a following window, and the reason we do not write one
        #: for DLC. Off by default: it changes what the network sees.
        self._dynamic_crop = bool(dynamic_crop)
        self._dynamic_threshold = float(dynamic_threshold)
        self._dynamic_margin = int(dynamic_margin)

    def initialize(self, frame: Optional[np.ndarray] = None) -> bool:
        if self._initialized:
            return True
        # Which of the folder's runnable artifacts to use. Previously the
        # export won unconditionally, which made the dialog's engine setting
        # dead: a folder holding an export could never run its own weights.
        from source.video.tracking import dlc_engine

        self._engine, engine_path, self._engine_note = dlc_engine.resolve(
            self.model_path, self._model_type, n_boxes=self._max_batch_size)
        if self._engine_note:
            logger.warning("DLC model %s: %s", self.model_path,
                           self._engine_note)
        if self._engine == "none":
            logger.error("DLC model %s cannot be run: %s", self.model_path,
                         self._engine_note or "no usable engine")
            return False
        # ``dlclive`` means nothing here recognised the folder, so the path goes
        # to DeepLabCut-Live untouched and its own detection decides. Every
        # other engine below is one we identified.
        if self._engine == "export":
            return self._start_export()
        if self._engine == "pytorch":
            # dlclive's PyTorch runner calls torch.load on this path, and DLC
            # pickles classes its allow-list refuses. Both are settled before
            # the constructor rather than surfacing as an opaque load failure.
            try:
                dlc_engine.allow_checkpoint_globals(engine_path)
            except Exception as e:
                logger.error(
                    "DLC snapshot %s could not be opened: %s: %s", engine_path,
                    type(e).__name__, e)
                return False
        try:
            from dlclive import DLCLive, Processor
            processor = Processor()
            # DLCLive's ``convert2rgb`` is not "make sure this is RGB", its
            # ``img_to_rgb`` REVERSES the channels of any 3-D array it is given
            # and only promotes a 2-D one. The pipeline already converts
            # BGR→RGB once per camera frame, so leaving the default True fed
            # the network BGR: measurably worse keypoints, and invisible to a
            # detection rate. Hand it the flag that matches what we hand it.
            dlc_kwargs = {"processor": processor,
                          "convert2rgb": self.expected_channels == GRAYSCALE}
            # ``resize`` is deliberately NOT handed over. DeepLabCut-Live
            # applies it inside ``process_frame``, which this tracker bypasses
            # (see ``_run_dlclive``), and scales the coordinates back inside
            # ``_post_process_pose``. Passing it would resize nothing and
            # unscale everything. We own the resize instead.
            # Engine + precision + placement. Passed only when they differ from
            # DLCLive's own defaults so an older dlclive without a given keyword
            # still constructs, the version floor for these is not uniform.
            #
            # Translated into dlclive's own vocabulary: ours distinguishes this
            # repo's ONNX export from DeepLabCut-Live's TensorFlow-TensorRT
            # runner, and dlclive has no name for the former.
            live_type = dlc_engine.dlclive_model_type(self._engine,
                                                      self._model_type)
            if live_type != "base":
                dlc_kwargs["model_type"] = live_type
            if self._precision and self._precision != "FP32":
                dlc_kwargs["precision"] = self._precision
            if self._device and self._device != "auto":
                dlc_kwargs["device"] = self._device
            if self._dynamic_crop:
                dlc_kwargs["dynamic"] = (True, self._dynamic_threshold,
                                         self._dynamic_margin)
            if self.model_info.family in ("topdown", "multi_class_topdown"):
                # DLCLive needs top_down_config (or top_down_dynamic) for these
                # and raises without one. Say so here rather than letting the
                # constructor fail with a message that does not name the cause.
                logger.error(
                    "DLC model %s is top-down, which DeepLabCut-Live cannot "
                    "run without a top_down_config. Use a single-animal model "
                    "or add that support.", self.model_path)
            try:
                # ``engine_path``, not ``model_path``: the PyTorch runner
                # torch.loads this, so it has to be the snapshot file. The
                # TensorFlow runner globs the directory, and for it the two are
                # the same path.
                self._dlc_live = DLCLive(engine_path, **dlc_kwargs)
            except TypeError as e:
                # An installed dlclive that predates one of these keywords.
                # Retry with the baseline set rather than refusing to track,
                # and say which options were dropped.
                dropped = [k for k in ("model_type", "precision", "device",
                                       "dynamic")
                           if k in dlc_kwargs]
                for k in dropped:
                    dlc_kwargs.pop(k, None)
                logger.warning(
                    "DLCLive rejected %s (%s), retrying without them. Update "
                    "deeplabcut-live to use them.", dropped, e)
                self._dlc_live = DLCLive(engine_path, **dlc_kwargs)
                # Report what is running, not what was asked for.
                self._backend = "dlc:base"
                self._precision = "FP32"
            self._inference_ready = False
            # ``init_inference`` IS the warm-up, DeepLabCut-Live's own
            # docstring says the first inference is very slow. Paying it here
            # keeps it out of the first frame of a running session.
            #
            # The pipeline always supplies a probe frame (configure_model
            # refuses without one), so the synthetic fallback is for direct API
            # use: warming on a blank frame of the right shape still builds the
            # session, which is the expensive part.
            probe = frame if frame is not None else self._synthetic_probe()
            rgb = self._to_model_input(probe)
            if rgb is None:
                if frame is not None:
                    return False           # a real frame the model cannot use
                logger.debug("DLC warm-up skipped: no usable probe frame")
            else:
                self._dlc_live.init_inference(rgb)
                self._inference_ready = True
            # Read actual body part names from model config (overrides user default)
            model_parts = self._read_model_body_parts()
            if model_parts:
                self._body_parts = model_parts
            self._initialized = True
            # What is RUNNING, which is the engine that was resolved and not
            # necessarily the one the dialog asked for.
            self._backend = f"dlc:{self._engine}"
            _cuda, _desc = _inference_device_after_import()
            logger.info(
                "DLC initialized on the %s engine: %s (%s), parts=%s | "
                "inference device=%s (%s)",
                self._engine, engine_path, self.model_info.summary(),
                self._body_parts, "CUDA" if _cuda else "CPU", _desc)
            # A multi-animal network run through the single-animal path returns
            # one animal and says nothing about it. Announce the mismatch so it
            # is a decision rather than a surprise in the data.
            for warn in self.model_info.warnings:
                logger.warning("DLC model %s: %s", self.model_path, warn)
            return True
        except ImportError as e:
            # NOT "DLCLive not installed" for every ImportError. DLCLive
            # dispatches to a per-engine runner and imports it lazily, so a
            # TensorFlow model on a machine without TensorFlow arrives here as
            # ModuleNotFoundError('tensorflow'), and the old message sent the
            # operator to reinstall a package that was already there.
            missing = getattr(e, "name", "") or ""
            if missing in ("dlclive", "") or "dlclive" in str(e):
                logger.error("DLCLive is not installed: %s. "
                             "pip install deeplabcut-live", e)
            else:
                logger.error(
                    "DLC model %s needs '%s', which is not installed in this "
                    "environment (%s). dlclive can run %s here; this model "
                    "requires the %s runtime.",
                    self.model_path, missing, e, _dlclive_backends(),
                    "TensorFlow" if missing == "tensorflow" else missing)
            return False
        except Exception as e:
            logger.error("DLC init failed for %s: %s: %s",
                         self.model_path, type(e).__name__, e)
            return False

    def _start_export(self) -> bool:
        """Drive this model's ONNX/TensorRT export, if it has one usable here.

        Returns False, quietly, when the folder holds no export, because
        that is the ordinary case for a trained-model directory and the
        DeepLabCut-Live path below is the right answer for it. A model that
        HAS an export and cannot run it says so, because then something the
        operator can fix is wrong.
        """
        from source.video.tracking import dlc_export

        if not dlc_export.usable(self.model_path):
            return False
        want_gpu = str(self._device or "auto").lower() != "cpu"
        try:
            self._export = dlc_export.DLCExportRunner(
                self.model_path, prefer_gpu=want_gpu,
                # So a graph with a dynamic batch axis always sees one shape.
                pad_to=self._max_batch_size)
        except Exception as e:
            logger.error(
                "DLC model %s ships an export that will not run here: %s",
                self.model_path, e)
            return False
        self._body_parts = self._export.body_parts or self._body_parts
        self._backend = self._export.backend
        self._initialized = True
        _cuda, _desc = _inference_device_after_import()
        logger.info(
            "DLC initialized from its export (%s): %s parts=%s | device=%s (%s)",
            self._backend, self.model_path, self._body_parts,
            "CUDA" if _cuda else "CPU", _desc)
        return True

    def _synthetic_probe(self) -> np.ndarray:
        """A blank frame to warm the session on when no real one is available.

        Content does not matter, the cost being paid here is building the
        session and running one forward pass, not what the pass sees. Size
        follows the model's own crop when it states one, so the warm-up
        exercises the shape the real frames will use.
        """
        side = self.model_info.crop_size or 256
        side = max(32, min(int(side), 2048))
        channels = 1 if self.expected_channels == GRAYSCALE else 3
        shape = (side, side) if channels == 1 else (side, side, 3)
        return np.zeros(shape, dtype=np.uint8)

    def _read_model_body_parts(self) -> List[str]:
        """Read body part names from the DLCLive model config."""
        if self._dlc_live is None:
            return []
        try:
            cfg = getattr(self._dlc_live, 'cfg', None)
            if cfg and 'all_joints_names' in cfg:
                return list(cfg['all_joints_names'])
        except Exception:
            pass
        # The live object had nothing: fall back to the file on disk, which is
        # the same source and is readable before DLCLive is even constructed.
        return list(self.model_info.body_parts)

    def predict(self, frame: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
        if not self._initialized:
            if not self.initialize(frame):
                return self._empty_result()
        if getattr(self, "_export", None) is not None:
            out = self.predict_batch([frame])
            return out[0] if out else self._empty_result()

        try:
            rgb = self._to_model_input(frame)
            if rgb is None:
                return self._empty_result()

            # Lazy-init TF session on first real frame.
            if not getattr(self, "_inference_ready", False):
                self._dlc_live.init_inference(rgb)
                self._inference_ready = True
                logger.info("DLC inference session initialised on first frame")

            pose = self._run_dlclive(rgb)

            result = {}
            for i, part in enumerate(self._body_parts):
                if i < len(pose):
                    x, y, conf = float(pose[i][0]), float(pose[i][1]), float(pose[i][2])
                    result[part] = [x, y, conf]
                else:
                    result[part] = None
            return result
        except Exception:
            # logger.exception so the full traceback reaches the log
            # sidebar; otherwise the empty-pose fallback looks like a
            # healthy-but-empty result.
            logger.exception(
                "DLCLiveTracker.predict raised, returning empty pose. "
                "Check the traceback above for the cause."
            )
            return self._empty_result()

    def predict_batch(self, frames):
        """A batch through the exported graph, or serial through DLCLive.

        The docstring below describes DeepLabCut-Live, which genuinely cannot
        batch. An EXPORTED graph is a different matter: it takes ``[N,C,H,W]``
        like any other, and this is where an offline retrack's speed is.

        Serial, because DeepLabCut-Live has no batched entry point.

        This override exists to record that, so the question is not reopened
        every time a multi-box rig looks slow. Both engines take one frame:

        * PyTorch, ``BaseRunner.get_pose(frame: np.ndarray)`` adds the batch
          dimension itself (``unsqueeze(0)``). The only stacking anywhere is
          top-down crops *within* one frame, never across frames.
        * TensorFlow (``base`` / ``tensorrt`` / ``lite``): ``init_inference``
          builds the input placeholder as ``[1, H, W, 3]``, so batch size 1 is
          compiled into the graph, and into the TensorRT engine or tflite
          interpreter built from it. Passing more frames is not a call-site
          change; it is a different engine.

        The throughput lever for DLC across boxes is therefore model copies
        (``Instances`` in the tracking dialog → ``MultiInstanceInferenceBackend``),
        which runs independent sessions in parallel. SLEAP, whose predictor does
        take a batch, overrides this with a real batched pass.
        """
        if not frames:
            return []
        export = getattr(self, "_export", None)
        if export is not None:
            if len(frames) > 1 and getattr(self, "_export_no_batch", False):
                # Already established that this graph's batch axis is fixed at
                # 1. `_export_batch` records that, but only AFTER raising and
                # logging, so re-entering it per frame turns a one-time fact
                # into a per-frame exception. Measured: 290 refusals in one
                # 8-second multi-box run, and a fifth off DLC's throughput.
                return [self._export_batch(export, [f])[0] for f in frames]
            return self._export_batch(export, frames)
        # DeepLabCut-Live has no batched entry point, but its PyTorch runner
        # holds an ordinary nn.Module that has always taken (N, C, H, W). The
        # batch is formed from the runner's OWN model and transforms, so this
        # is dlclive's arithmetic on more frames at once, not a second
        # implementation of it.
        if (len(frames) > 1 and self._dlc_live is not None
                and not getattr(self, "_no_torch_batch", False)):
            from source.video.tracking import dlc_engine
            if dlc_engine.batchable(self._dlc_live):
                try:
                    return self._torch_batch(frames)
                except Exception as e:
                    self._no_torch_batch = True
                    logger.warning(
                        "DLC batched forward failed (%s), falling back to one "
                        "frame at a time for the rest of this session.", e)
        if len(frames) > 1 and not self._serial_batch_announced:
            self._serial_batch_announced = True
            logger.info(
                "DLC is running %d boxes as %d forward passes. Its dynamic "
                "cropper and its TensorFlow runners cannot batch; a PyTorch "
                "runner without dynamic cropping can.", len(frames),
                len(frames))
        return [self.predict(f) for f in frames]

    def _scaled(self, rgb):
        """``(image, 1/scale)`` after our own resize, if one was asked for."""
        s = float(self.resize_factor or 1.0)
        if s == 1.0 or s <= 0:
            return rgb, 1.0
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))),
                           interpolation=cv2.INTER_AREA if s < 1
                           else cv2.INTER_LINEAR)
        return small, 1.0 / s

    def _run_dlclive(self, rgb):
        """One frame through the RUNNER, not through ``DLCLive.get_pose``.

        ``DLCLive.get_pose`` begins with

            if frame.ndim >= 2:
                self.convert2rgb = True

        and ``ndim >= 2`` is true of every frame there has ever been. So the
        flag is forced on at each call whatever was asked for at construction,
        and ``process_frame`` then runs ``img_to_rgb``, which REVERSES the
        channels of any 3-D array. The pipeline already hands pose RGB, so
        DeepLabCut-Live was quietly feeding this network BGR on every frame,
        with ``convert2rgb=False`` set, and a test asserting it was set.

        The runner is where the model actually lives, and it applies no colour
        conversion. Nothing else is lost by going straight to it: dynamic
        cropping lives in the runner too and still runs, static cropping is not
        used here, and the resize is ours (see ``_scaled``). Only the no-op
        Processor is skipped.
        """
        runner = getattr(self._dlc_live, "runner", None)
        image, back = self._scaled(rgb)
        if runner is None or not hasattr(runner, "get_pose"):
            # An older DeepLabCut-Live with no exposed runner. Its channel
            # handling is what it is; better a pose than none.
            pose = np.asarray(self._dlc_live.get_pose(image), float)
        else:
            pose = np.asarray(runner.get_pose(image), float)
        if back != 1.0 and pose.size:
            pose = pose.copy()
            pose[..., :2] *= back
        return pose

    def _torch_batch(self, frames):
        """One batched forward through DeepLabCut-Live's PyTorch runner."""
        from source.video.tracking import dlc_engine

        images = [self._to_model_input(f) for f in frames]
        if any(im is None for im in images):
            return [self._empty_result() for _ in frames]
        scaled = [self._scaled(im) for im in images]
        back = scaled[0][1]
        poses = dlc_engine.batched_get_pose(self._dlc_live,
                                            [im for im, _ in scaled])
        out = []
        for pose in poses:
            result = {}
            for i, part in enumerate(self._body_parts):
                if i < len(pose):
                    result[part] = [float(pose[i][0]) * back,
                                    float(pose[i][1]) * back,
                                    float(pose[i][2])]
                else:
                    result[part] = None
            out.append(result)
        return out

    def _export_batch(self, export, frames):
        """One forward pass through the exported graph for the whole batch."""
        from source.video.tracking import dlc_export
        from source.video.tracking.ort_runner import to_nchw

        try:
            images = [self._to_model_input(f) for f in frames]
            if any(im is None for im in images):
                return [self._empty_result() for _ in frames]
            batch = np.ascontiguousarray(
                np.concatenate([to_nchw(im) for im in images], axis=0))
            return dlc_export.to_pose_dicts(export(batch), self._body_parts)
        except Exception as e:
            # A graph built for one frame refuses a batch of sixteen rather
            # than running it slowly, and that must not end the recording.
            if len(frames) > 1:
                logger.warning(
                    "DLC export refused a batch of %d (%s), falling back to "
                    "one frame at a time.", len(frames), e)
                self._export_no_batch = True
                return [self._export_batch(export, [f])[0] for f in frames]
            logger.exception("DLC export predict raised, returning an empty "
                             "pose. The traceback above has the cause.")
            return [self._empty_result()]

    @property
    def batched(self) -> bool:
        """Whether a batch is worth submitting to this tracker.

        True for an export with a dynamic batch axis, and, since the batched
        torch path, for DeepLabCut-Live's PyTorch runner too, as long as it is
        not doing its own per-frame cropping. False for the TensorFlow runners,
        whose input placeholder is compiled at batch 1.
        """
        export = getattr(self, "_export", None)
        if export is not None:
            return (export.batched
                    and not getattr(self, "_export_no_batch", False))
        if self._dlc_live is not None and not getattr(
                self, "_no_torch_batch", False):
            from source.video.tracking import dlc_engine
            return dlc_engine.batchable(self._dlc_live)
        return False

    def close(self):
        export = getattr(self, "_export", None)
        if export is not None:
            try:
                export.close()
            except Exception:
                pass
            self._export = None
        if self._dlc_live:
            try:
                self._dlc_live.close()
            except Exception:
                pass
            self._dlc_live = None
        super().close()


def detect_sleap_model_type(model_path: str) -> str:
    """Return a sleap-nn model folder's type from its training config.

    One of ``single`` | ``centroid`` | ``centered_instance`` | ``bottomup`` |
    ``multi_class_topdown`` | ``multi_class_bottomup`` | ``unknown``.
    ``centroid`` / ``centered_instance`` together form a top-down pair.

    The reading lives in ``model_config`` so that the family, the body parts,
    the native scale, the crop size, the channel expectation and the identity
    names are all answered from one place, they come from one file, and a
    second reader is a second answer waiting to disagree.
    """
    return ModelInfo.read(model_path).family


class SLEAPTracker(PoseTracker):
    """SLEAP backend (sleap-nn v0.3.x PyTorch; falls back to legacy SLEAP TF).

    Requires: ``pip install sleap-nn`` (folder model) or ``pip install sleap``
    (legacy ``.zip``). Handles single-instance, top-down (centroid +
    centered-instance), and bottom-up; body parts auto-detected from the model
    skeleton. Single-animal contract: ``predict`` returns the primary instance
    (multi-instance list is a later phase; the parse already yields a list).
    """

    def __init__(self, model_path: str,
                 resize_factor: float = 1.0, max_instances: int = 1,
                 centroid_path: Optional[str] = None, model_type: str = "auto",
                 runtime: str = "auto", device: str = "auto",
                 fp16: bool = False, compile: bool = False,
                 peak_threshold: float = 0.2,
                 max_batch_size: int = 8,
                 body_parts: Optional[List[str]] = None,
                 colour_mode: str = "auto"):
        super().__init__(model_path, resize_factor, colour_mode)
        #: Largest batch the exported engine is built to accept. Fixed at
        #: export, so the sink must not submit more than this.
        self._max_batch_size = max(1, int(max_batch_size))
        self._predictor = None
        #: A direct engine driver (TensorRT or ONNX Runtime) when the export
        #: decodes peaks in-graph. It replaces sleap-nn in the hot loop, which
        #: is where most of the exported-runtime speed actually is: the same
        #: weights measure 1.09 ms driven directly against 4.79 ms through the
        #: library. ``None`` means the library path is in use.
        self._runner = None
        self._backend = "sleap_nn"
        self._max_instances = max_instances
        self._centroid_path = (centroid_path or "").strip() or None
        self._model_type = model_type or "auto"
        self._runtime = runtime or "auto"
        self._device = device or "auto"
        self._fp16 = bool(fp16)
        self._compile = bool(compile)
        self._peak_threshold = float(peak_threshold)
        if body_parts:
            self._body_parts = list(body_parts)

    # ── model construction ────────────────────────────────────────────

    def _resolved_type(self) -> str:
        mt = self._model_type
        if mt == "auto":
            mt = detect_sleap_model_type(self.model_path)
        return mt

    def _ordered_model_paths(self) -> List[str]:
        """Paths for ``Predictor.from_model_paths`` in the right order.

        Top-down needs BOTH ``[centroid, centered_instance]``. When the user
        supplied a ``centroid_path``, the main ``model_path`` is the
        centered-instance stage. Single-instance / bottom-up = one path.
        """
        mt = self._resolved_type()
        if self._centroid_path or mt in ("centroid", "centered_instance",
                                         "topdown", "multi_class_topdown"):
            if self._centroid_path:
                return [self._centroid_path, self.model_path]
            # A lone centroid can't pose; a lone centered-instance can't find
            # animals. Warn but still try (sleap-nn errors clearly if invalid).
            logger.warning(
                "SLEAP top-down model %s has no paired centroid model, "
                "set the Centroid model path in the tracking dialog.",
                self.model_path)
        return [self.model_path]

    def initialize(self, frame: Optional[np.ndarray] = None) -> bool:
        if self._initialized:
            return True
        try:
            built = self._build_predictor()
            if built is None:
                return False
            # A direct engine driver and a sleap-nn predictor are not the same
            # object with the same methods, so which one is held decides how
            # every later frame is run. Kept in two names rather than one, so
            # nothing has to guess by duck-typing.
            if hasattr(built, "backend") and callable(built):
                self._runner, self._predictor = built, None
            else:
                self._runner, self._predictor = None, built
            if self._runner is not None:
                # A bare engine carries no skeleton object, and probing it for
                # names would mean running inference to find out. The model's
                # own config already says, in order.
                self._body_parts = (list(self.model_info.body_parts)
                                    or self._body_parts)
            else:
                self._body_parts = (self._extract_body_parts()
                                    or self._body_parts
                                    or (self._infer_parts_from_frame(frame)
                                        if frame is not None else []))
            # A resize asked for here cannot be honoured: SLEAP takes its input
            # scale from the model config, and an exported engine fixes the
            # input size at export. Say so rather than let the operator believe
            # the frame is being downscaled.
            if self.resize_factor and float(self.resize_factor) != 1.0:
                logger.warning(
                    "SLEAP %s: resize %.2f ignored, input scale comes from the "
                    "model config (native %s) and, for an exported runtime, is "
                    "fixed at export. Re-export to change it.",
                    self.model_path, float(self.resize_factor),
                    self.model_info.native_scale)
            self._warmup(frame)
            self._initialized = True
            _cuda, _desc = _inference_device_after_import()
            logger.info(
                "SLEAP initialized (%s/%s, runtime=%s): %s parts=%s | device=%s (%s)",
                self._backend, self._resolved_type(), self._runtime,
                self.model_path, self._body_parts,
                "CUDA" if _cuda else "CPU", _desc)
            return True
        except ImportError as e:
            # Same trap as the DLC path: sleap-nn imports its engine drivers
            # lazily, so "onnxruntime is required for ONNXBackend" surfaces
            # here as an ImportError that has nothing to do with sleap-nn
            # being absent.
            missing = getattr(e, "name", "") or ""
            if missing in ("sleap_nn", "sleap", "") and "required for" not in str(e):
                logger.error("sleap-nn is not installed: %s. "
                             "pip install sleap-nn (or pip install sleap)", e)
            else:
                logger.error(
                    "SLEAP model %s needs '%s', which is not installed in "
                    "this environment: %s", self.model_path, missing or "a "
                    "runtime", e)
            return False
        except Exception as e:
            logger.error("SLEAP init failed for %s: %s: %s",
                         self.model_path, type(e).__name__, e)
            return False

    def _build_predictor(self):
        """Construct the predictor per runtime, with a TensorRT → ONNX → native
        fallback so a missing engine / export dep degrades gracefully instead of
        failing the box. ``auto`` prefers an exported engine when one builds,
        else native torch. Small + patchable so tests inject a fake predictor.
        """
        rt = self._runtime
        if rt in ("onnx", "tensorrt", "auto"):
            pred = self._build_exported(rt)
            if pred is not None:
                return pred
            if rt != "auto":
                logger.warning(
                    "SLEAP %s runtime unavailable, falling back to native torch",
                    rt)
        return self._build_native()

    #: Families whose graph emits a FIXED number of peaks, one per keypoint,
    #: each carrying its own confidence. For these the threshold can be
    #: exported permissively and applied on the host instead, which makes it a
    #: live control rather than a four-minute rebuild. Bottom-up decides how
    #: many candidates exist at all, so there it must stay baked.
    _FIXED_PEAK_FAMILIES = ("single", "centroid", "centered_instance",
                            "topdown", "multi_class_topdown")

    @property
    def export_peak_threshold(self) -> float:
        """The threshold to bake into the artifact.

        Permissive where the host can do the gating, the operator's value where
        it cannot. The same number is given to the native predictor, so the
        three runtimes cannot disagree about which peaks exist.
        """
        from source.video.tracking.sleap_export import PERMISSIVE_PEAK_THRESHOLD
        if self._resolved_type() in self._FIXED_PEAK_FAMILIES:
            return PERMISSIVE_PEAK_THRESHOLD
        return float(self._peak_threshold)

    @property
    def export_precision(self) -> str:
        """Precision to build an exported engine at, ``fp16`` | ``fp32``.

        sleap-nn's export takes ``fp32 | fp16 | tf32`` and defaults to fp16, so
        the operator's toggle has to be stated explicitly to mean anything.
        """
        return "fp16" if self._fp16 else "fp32"

    def _direct_runner(self, export_dir: str, runtime: str):
        """A bare engine driver for ``export_dir``, or None.

        Only possible when the graph decodes peaks itself, it returns
        ``peaks``/``peak_vals`` rather than heatmaps, which is what makes
        sleap-nn unnecessary at inference. Anything else falls through to the
        library path rather than failing the run.
        """
        want = ("tensorrt", "onnx") if runtime == "auto" else (runtime,)
        if "tensorrt" in want:
            try:
                from source.video.tracking import trt_runner
                if trt_runner.usable(export_dir):
                    engine = trt_runner.engine_path(export_dir)
                    r = trt_runner.TRTRunner(str(engine),
                                             device=self._resolve_device())
                    self._backend = f"sleap_nn:{r.backend}"
                    logger.info("SLEAP driving the TensorRT engine directly: %s",
                                engine)
                    return r
            except Exception as e:
                logger.warning("SLEAP direct TensorRT unavailable (%s), "
                               "falling back", e)
        if "onnx" in want:
            try:
                import os as _os

                from source.video.tracking.ort_runner import (
                    ORTRunner, graph_is_decoded)
                onnx = _os.path.join(export_dir, "model.onnx")
                if _os.path.isfile(onnx) and graph_is_decoded(onnx):
                    r = ORTRunner(onnx, prefer_gpu=self._wants_gpu(),
                                  pad_to=self._max_batch_size)
                    self._backend = f"sleap_nn:{r.backend}"
                    logger.info("SLEAP driving the ONNX graph directly: %s "
                                "(%d threads, %s)", onnx, r.threads, r.provider)
                    return r
            except Exception as e:
                logger.warning("SLEAP direct ONNX unavailable (%s), falling "
                               "back", e)
        return None

    def _wants_gpu(self) -> bool:
        return not str(self._resolve_device()).startswith("cpu")

    def _check_engine_batch(self, export_dir: str) -> None:
        """Warn when the engine cannot take the batch this rig will submit.

        Asked of the RUNNER that was actually built, not of the manifest beside
        it. ``export_metadata.json``'s ``max_batch_size`` describes the
        TensorRT engine build; the ONNX graph exported alongside it routinely
        has a dynamic batch axis and takes any size. ``models/sleap`` says 1
        and its graph runs a batch of 16 at 0.57 ms/frame against 3.06 ms at
        batch 1, so trusting the manifest told operators to spend an evening
        re-exporting a model that was already fine.
        """
        runner = self._runner
        limit = getattr(runner, "max_batch", None) if runner is not None else None
        if limit and limit < self._max_batch_size:
            logger.warning(
                "SLEAP engine at %s accepts a batch of %d but this rig may "
                "submit %d. Re-export it for the larger batch; until then the "
                "sink runs one frame at a time and pays for it.",
                export_dir, limit, self._max_batch_size)

    def _cached_engine_runner(self):
        """This machine's own TensorRT build of the model, if one is cached.

        A model folder can carry a ``model.trt`` built on ANOTHER machine, the
        training PC's, which TensorRT here refuses to deserialise. The ONNX
        graph beside it still loads, so the run used to settle for ONNX Runtime
        even when this machine had already built its own engine for exactly
        this (model, precision, batch, threshold) through ``sleap_export``.
        Only an engine already in the cache is used: building one takes
        minutes and stays the Export button's job.
        """
        try:
            from source.video.tracking.sleap_export import (
                export_cache_dir, is_exported)
            key = dict(device=self._resolve_device(),
                       centroid_path=self._centroid_path,
                       precision=self.export_precision,
                       max_batch_size=self._max_batch_size,
                       peak_threshold=self.export_peak_threshold)
            if not is_exported(self.model_path, "tensorrt", **key):
                return None
            export_dir = str(export_cache_dir(self.model_path, "tensorrt", **key))
        except Exception as e:
            logger.debug("SLEAP export cache lookup failed: %s", e)
            return None
        direct = self._direct_runner(export_dir, "tensorrt")
        if direct is not None:
            logger.info("SLEAP using this machine's cached TensorRT engine "
                        "instead of the model folder's: %s", export_dir)
            self._check_engine_batch(export_dir)
        return direct

    def _build_exported(self, rt: str):
        """Try exported runtimes in preference order (``auto`` = tensorrt→onnx;
        explicit = just that one). Exports once + caches per (model, device,
        runtime). Returns the predictor, or None if no exported runtime works."""
        # The model directory may already BE an export, a model.onnx/.trt
        # sitting beside the training config. Re-exporting that would need
        # sleap-nn and minutes to produce what is already on disk.
        direct = None
        if rt in ("auto", "tensorrt"):
            # TensorRT first, as before: the folder's own engine. Only when the
            # folder HAS an engine and it will not load here (built on another
            # GPU) is this machine's cached build tried, before the folder's
            # ONNX graph. A folder whose engine loads, or that ships none,
            # behaves exactly as it did.
            direct = self._direct_runner(self.model_path, "tensorrt")
            if (direct is None
                    and "model.trt" in self._export_artifacts(self.model_path)):
                direct = self._cached_engine_runner()
                if direct is not None:
                    return direct
        if direct is None and rt != "tensorrt":
            direct = self._direct_runner(self.model_path,
                                         "onnx" if rt == "auto" else rt)
        if direct is not None:
            self._check_engine_batch(self.model_path)
            return direct
        # An explicit runtime states a preference, not a prohibition. When the
        # asked-for engine cannot run here, no TensorRT on this machine, say,
        # the other artifact in the SAME directory produces the same
        # coordinates from the same weights, and is a far better answer than
        # falling through to a checkpoint load the folder cannot satisfy.
        if rt != "auto" and self._export_artifacts(self.model_path):
            direct = self._direct_runner(self.model_path, "auto")
            if direct is not None:
                logger.warning(
                    "SLEAP %s is unavailable here; using the %s export in the "
                    "same folder instead.", rt, self._backend.split(":")[-1])
                self._check_engine_batch(self.model_path)
                return direct

        order = ["tensorrt", "onnx"] if rt == "auto" else [rt]
        for runtime in order:
            try:
                from source.video.tracking.sleap_export import ensure_exported
                from sleap_nn.inference import Predictor
                # Precision comes from the operator's fp16 toggle, not from
                # ensure_exported's default: the toggle previously applied only
                # to the native torch path, so an exported engine was fp16
                # whatever the dialog said.
                export_dir = ensure_exported(
                    self.model_path, runtime,
                    device=self._resolve_device(),
                    centroid_path=self._centroid_path,
                    precision=self.export_precision,
                    max_batch_size=self._max_batch_size,
                    peak_threshold=self.export_peak_threshold)
                direct = self._direct_runner(export_dir, runtime)
                if direct is not None:
                    self._check_engine_batch(export_dir)
                    return direct
                pred = Predictor.from_export_dir(export_dir, runtime=runtime)
                self._backend = f"sleap_nn:{runtime}"
                logger.info("SLEAP using exported %s engine: %s", runtime, export_dir)
                return pred
            except Exception as e:
                logger.warning("SLEAP %s export/runtime failed: %s", runtime, e)
        return None

    @staticmethod
    def _export_artifacts(path: str) -> List[str]:
        """The exported engines sitting in ``path``, newest API first.

        An EXPORT directory holds ``model.onnx`` / ``model.trt`` beside
        ``training_config.yaml``; a TRAINING directory holds ``best.ckpt``.
        They are loaded by different sleap-nn entry points, and handing an
        export to the training loader is what produced
        ``No such file or directory: .../best.ckpt``.
        """
        import os as _os
        return [name for name in ("model.trt", "model.onnx")
                if _os.path.isfile(_os.path.join(path, name))]

    @staticmethod
    def _runtime_available(runtime: str) -> bool:
        """Whether the engine driver for ``runtime`` imports on this machine.

        Asked before an artifact is chosen, not after it fails: an engine file
        on disk says nothing about whether anything here can execute it.
        """
        import importlib
        module = "tensorrt" if runtime == "tensorrt" else "onnxruntime"
        try:
            importlib.import_module(module)
            return True
        except Exception as e:
            logger.debug("%s runtime unavailable: %s", runtime, e)
            return False

    @classmethod
    def _runnable_artifacts(cls, path: str):
        """``[(filename, runtime)]`` for the engines this machine can drive.

        Preference order (TensorRT then ONNX) is kept, but an artifact whose
        runtime is not installed is dropped rather than attempted and blamed.
        """
        pairs = [(name, "tensorrt" if name.endswith(".trt") else "onnx")
                 for name in cls._export_artifacts(path)]
        usable = [p for p in pairs if cls._runtime_available(p[1])]
        for name, runtime in pairs:
            if (name, runtime) not in usable:
                logger.info(
                    "SLEAP skipping %s in %s: its %s runtime is not installed "
                    "here.", name, path, runtime)
        return usable

    @staticmethod
    def _has_checkpoint(path: str) -> bool:
        """Whether the checkpoint loader has anything here to open."""
        import os as _os
        return _os.path.isfile(_os.path.join(path, "best.ckpt"))

    @staticmethod
    def _checkpoint_note(path: str) -> str:
        """What the folder offers the checkpoint loader, stated accurately.

        Asserting "no checkpoint" whenever an export is present sends the
        operator looking for a missing file that is sitting in the folder
        under a name sleap-nn does not open.
        """
        import os as _os
        try:
            ckpts = [n for n in _os.listdir(path) if n.lower().endswith(".ckpt")]
        except OSError:
            return ""
        if not ckpts:
            return ("The folder holds no checkpoint either, so the "
                    "trained-model loader cannot be used.")
        if any(n.lower() == "best.ckpt" for n in ckpts):
            return "A best.ckpt is present and will be tried next."
        return (f"The folder holds {', '.join(ckpts)}, but sleap-nn's "
                "trained-model loader only opens 'best.ckpt', rename it to "
                "load these weights natively.")

    def _build_native(self):
        """Native torch predictor (fp16/compile applied), or the legacy
        TensorFlow model when sleap-nn isn't installed."""
        exported = self._runnable_artifacts(self.model_path)
        if exported:
            # sleap-nn's own docs: pointing at a directory containing
            # model.onnx or model.trt is auto-detected as an exported model.
            # ``from_model_paths`` is the checkpoint loader and looks for
            # best.ckpt, which an export does not contain.
            #
            # Every artifact whose runtime is actually importable is tried, in
            # preference order. Taking model.trt purely because the file exists
            # meant a folder holding BOTH engines failed on a machine without
            # TensorRT, while the model.onnx beside it, the same weights, the
            # same coordinates, would have run.
            errors = []
            for name, runtime in exported:
                try:
                    from sleap_nn.inference import Predictor
                    pred = Predictor.from_export_dir(self.model_path,
                                                     runtime=runtime)
                    self._backend = f"sleap_nn:{runtime}"
                    logger.info("SLEAP loading the export directory %s via "
                                "sleap-nn (%s, from %s)",
                                self.model_path, runtime, name)
                    return pred
                except Exception as e:
                    errors.append(f"{runtime} ({name}): {e}")
                    logger.warning("SLEAP could not load %s as %s: %s",
                                   name, runtime, e)
            logger.error("SLEAP could not load the export at %s. Tried %s. %s",
                         self.model_path, "; ".join(errors),
                         self._checkpoint_note(self.model_path))
            return None
        on_disk = self._export_artifacts(self.model_path)
        if on_disk and not self._has_checkpoint(self.model_path):
            # An export this machine cannot drive, and no weights to fall back
            # on. Falling through to the checkpoint loader here produced
            # "No such file or directory: .../best.ckpt", which names a file
            # the operator never had and says nothing about the runtime that
            # is actually missing.
            missing = sorted({("tensorrt" if n.endswith(".trt")
                               else "onnxruntime") for n in on_disk})
            logger.error(
                "SLEAP cannot run the export at %s: it holds %s, and none of "
                "the runtimes that could execute them (%s) is installed in "
                "this Python environment. %s",
                self.model_path, " and ".join(on_disk), ", ".join(missing),
                self._checkpoint_note(self.model_path))
            return None
        try:
            from sleap_nn.inference import Predictor
            self._backend = "sleap_nn"
            pred = Predictor.from_model_paths(
                self._ordered_model_paths(),
                device=self._resolve_device(),
                # The sink hands this predictor a batch of boxes; sleap-nn's
                # own default is 4 and ours was 1, which quietly undid the
                # multi-box batching upstream.
                batch_size=self._max_batch_size,
                peak_threshold=self.export_peak_threshold)
            self._apply_fast_opts(pred)
            return pred
        except ImportError:
            from sleap import load_model
            self._backend = "sleap"
            return load_model(self.model_path)

    def _resolve_device(self) -> str:
        if self._device and self._device != "auto":
            return self._device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _apply_fast_opts(self, pred) -> None:
        """Best-effort fp16 / torch.compile on the torch backend (CUDA only).
        Guarded, the attribute path (``pred.layer.backend``) can shift between
        sleap-nn point releases, so every set is defensive and non-fatal."""
        dev = self._resolve_device()
        try:
            backend = getattr(getattr(pred, "layer", None), "backend", None)
            if backend is None:
                return
            if self._fp16 and dev == "cuda":
                backend.use_fp16 = True
            if self._compile:
                backend.use_compile = True
                # Fixed per-box ROI size → static shapes → cheaper compile.
                for attr in ("compile_dynamic", "dynamic"):
                    if hasattr(backend, attr):
                        setattr(backend, attr, False)
        except Exception as e:
            logger.debug("SLEAP fast-opts not applied: %s", e)

    def _warmup(self, frame: Optional[np.ndarray]) -> None:
        """Run a couple of dummy frames so compile/kernels/fp16 are hot before
        the first real latency-sensitive call."""
        if frame is None:
            return
        rgb = self._to_model_input(frame)
        if rgb is None:
            return
        try:
            for _ in range(2):
                if self._runner is not None:
                    self._run_direct([rgb])
                else:
                    self._run_predictor(rgb[None])
        except Exception as e:
            logger.debug("SLEAP warmup skipped: %s", e)

    # ── skeleton / body parts ─────────────────────────────────────────

    def _extract_body_parts(self) -> List[str]:
        """Node names from the loaded model's skeleton (v0.3: off a Labels /
        skeleton object; legacy: predictor.skeleton)."""
        p = self._predictor
        try:
            for obj in (getattr(p, "skeleton", None),
                        getattr(getattr(p, "labels", None), "skeleton", None)):
                if obj is None:
                    continue
                if getattr(obj, "node_names", None):
                    return list(obj.node_names)
                if getattr(obj, "nodes", None):
                    return [getattr(n, "name", str(n)) for n in obj.nodes]
        except Exception as e:
            logger.debug("SLEAP skeleton read: %s", e)
        return []

    def _infer_parts_from_frame(self, frame: np.ndarray) -> List[str]:
        try:
            rgb = self._to_model_input(frame)
            if rgb is None:
                return []
            insts = self._instances_from_prediction(self._run_predictor(rgb[None]))
            if insts:
                return list(insts[0].keys())
        except Exception:
            pass
        return []

    # ── inference ─────────────────────────────────────────────────────

    def _run_direct(self, frames: List[np.ndarray]) -> List[dict]:
        """One forward pass on the engine, decoded to the pose contract.

        uint8 NHWC in, NCHW out, the engine casts internally, so converting
        to float here would move four times the bytes for nothing. No colour
        conversion: the pipeline already hands pose RGB.
        """
        from source.video.tracking.ort_runner import batch_to_poses, to_nchw
        batch = np.ascontiguousarray(
            np.concatenate([to_nchw(f) for f in frames], axis=0))
        out = self._runner(batch)
        return batch_to_poses(out["peaks"], out["peak_vals"], self._body_parts)

    def _run_predictor(self, batch_nhwc: np.ndarray):
        """One forward pass on an ``(N,H,W,C)`` uint8 batch. Prefers the
        low-overhead streaming path (raw ``Outputs``); falls back to
        ``predict`` (``sio.Labels``). Returns whatever the predictor yields,
        parsing is centralised in ``_instances_from_prediction``."""
        p = self._predictor
        stream = getattr(p, "predict_streaming", None)
        if callable(stream):
            outs = list(stream(batch_nhwc))
            return outs[0] if outs else None
        return p.predict(batch_nhwc)

    def _batched_instances(self, preds) -> List[List[Dict[str, Tuple[float, float, float]]]]:
        """Normalise any sleap-nn / legacy output into ONE instance-list PER
        input frame, the single place output shapes are decoded.

        Returns ``[[{node:(x,y,conf)}, …instance…], …frame…]``. Raw ``Outputs``
        carry a batch dim (``pred_keypoints (B,I,N,2)``) so a batched forward
        pass yields B frames; ``sio.Labels`` iterate as labeled frames. Single
        callers take ``[0]``; ``predict_batch`` takes all, so a multi-box batch
        never collapses to frame 0.
        """
        if preds is None:
            return []
        names = self._body_parts

        # (1) Raw Outputs (streaming / make_labels=False), full batch.
        kp = getattr(preds, "pred_keypoints", None)
        if kp is not None:
            kp = _to_ndarray(kp)                          # (B,I,N,2) or (I,N,2)
            pv = _to_ndarray(getattr(preds, "pred_peak_values", None))
            if kp.ndim == 3:                              # single frame → add B
                kp = kp[None]
                pv = pv[None] if pv is not None else None
            # Confidence must be batch-aligned to kp (B,I,N). If a sleap-nn
            # point release hands back a differently-ranked pv, don't trust it
            # (drop to 1.0) rather than mis-index and attach conf to the wrong
            # keypoint.
            if pv is not None and (pv.ndim != 3 or pv.shape[0] != kp.shape[0]):
                pv = None
            frames = []
            for b in range(len(kp)):
                vb = pv[b] if pv is not None else None
                insts = []
                for i in range(len(kp[b])):
                    inst = {}
                    for n in range(len(kp[b][i])):
                        name = names[n] if n < len(names) else f"kp{n}"
                        x, y = float(kp[b][i][n][0]), float(kp[b][i][n][1])
                        try:
                            conf = float(vb[i][n]) if vb is not None else 1.0
                        except (IndexError, TypeError):
                            conf = 1.0
                        inst[name] = (x, y, conf)
                    insts.append(inst)
                frames.append(insts)
            return frames

        # (2) sio.Labels / list, labeled frames (one per input frame) or a flat
        # instance list (one frame). An object with ``.instances`` is one frame.
        if hasattr(preds, "instances"):
            return [[d for d in (self._instance_to_dict(i)
                                 for i in preds.instances) if d]]
        try:
            items = list(preds)
        except TypeError:
            return []
        if items and hasattr(items[0], "instances"):     # iterable of frames
            return [[d for d in (self._instance_to_dict(i)
                                 for i in lf.instances) if d] for lf in items]
        # Flat instance list → one frame.
        return [[d for d in (self._instance_to_dict(i) for i in items) if d]]

    def _instances_from_prediction(self, preds) -> List[Dict[str, Tuple[float, float, float]]]:
        """Instances for a SINGLE-frame prediction (the batch's frame 0)."""
        frames = self._batched_instances(preds)
        return frames[0] if frames else []

    def _instance_to_dict(self, inst) -> Dict[str, Tuple[float, float, float]]:
        """One PredictedInstance / point-container → ``{node:(x,y,conf)}``."""
        names = self._body_parts
        # sio point dict: {node: Point} + point_scores.
        pts = getattr(inst, "points", None)
        if isinstance(pts, dict):
            scores = getattr(inst, "point_scores", {}) or {}
            d = {}
            for node, point in pts.items():
                name = getattr(node, "name", str(node))
                x = getattr(point, "x", None)
                y = getattr(point, "y", None)
                if x is None and hasattr(point, "__getitem__"):
                    x, y = point[0], point[1]
                conf = scores.get(node, getattr(inst, "score", 1.0))
                d[name] = (float(x), float(y), float(conf))
            return d
        # numpy-backed instance: (N, 2 or 3).
        arr = inst.numpy() if hasattr(inst, "numpy") else (
            inst if isinstance(inst, np.ndarray) else None)
        if arr is not None:
            arr = np.asarray(arr)
            d = {}
            for i in range(len(arr)):
                name = names[i] if i < len(names) else f"kp{i}"
                x, y = float(arr[i][0]), float(arr[i][1])
                conf = float(arr[i][2]) if arr.shape[1] > 2 else 1.0
                d[name] = (x, y, conf)
            return d
        return {}

    def predict(self, frame: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
        if not self._initialized:
            if not self.initialize(frame):
                return self._empty_result()
        try:
            rgb = self._to_model_input(frame)
            if rgb is None:
                return self._empty_result()
            if self._runner is not None:
                got = self._run_direct([rgb])
                return got[0] if got else self._empty_result()
            insts = self._instances_from_prediction(self._run_predictor(rgb[None]))
            if not insts:
                return self._empty_result()
            # Single-animal contract: the primary (highest-confidence) instance.
            return insts[0]
        except Exception as e:
            logger.error("SLEAP predict error: %s", e)
            return self._empty_result()

    def predict_batch(self, frames):
        """True batched forward pass on same-shape frames (multi-box win)."""
        if not frames:
            return []
        if not self._initialized and not self.initialize(frames[0]):
            return [self._empty_result() for _ in frames]
        if len(frames) > 1 and getattr(self, "_no_batch", False):
            # An engine built for one frame refuses every batch. Asking again
            # on every frame would raise, log and fall back thirty times a
            # second for the length of the session.
            return [self.predict(f) for f in frames]
        try:
            if self._runner is not None:
                return self._run_direct(frames)
            batch = np.stack([np.ascontiguousarray(f) for f in frames])  # (N,H,W,C)
            preds = self._run_predictor(batch)
            per_frame = self._batched_instances(preds)   # one instance-list / frame
            # Single-animal contract: the primary instance of each frame.
            out = []
            for i in range(len(frames)):
                insts = per_frame[i] if i < len(per_frame) else []
                out.append(insts[0] if insts else self._empty_result())
            return out
        except Exception as e:
            if len(frames) > 1:
                # Remembered, not rediscovered: an engine whose batch axis was
                # fixed at export refuses every batch, and retrying one per
                # frame would log this thirty times a second for the session.
                self._no_batch = True
                logger.warning(
                    "SLEAP refused a batch of %d (%s), falling back to one "
                    "frame at a time for the rest of this session. Re-export "
                    "the engine for a larger batch to get the throughput back.",
                    len(frames), e)
            else:
                logger.error("SLEAP predict error: %s", e)
            return [self.predict(f) for f in frames]

    @property
    def batched(self) -> bool:
        """Whether submitting a batch is still worth it."""
        return not getattr(self, "_no_batch", False)

    def close(self):
        self._predictor = None
        super().close()


# ===================================================================
# Factory
# ===================================================================

#: Options both backends spell the same way and read differently. DLC reads
#: ``base|pytorch|tensorrt|lite``; SLEAP reads
#: ``auto|single|centroid|centered_instance|topdown|bottomup``. A value meant
#: for one is meaningless to the other, so callers pass ``dlc_model_type`` /
#: ``sleap_model_type`` and this resolves it.
AMBIGUOUS_OPTIONS = ("model_type",)


def tracker_class(tracker_type: str):
    """The class that backs a tracker type, or None."""
    t = (tracker_type or "").lower().strip()
    if t in ("dlc", "deeplabcut"):
        return DLCLiveTracker
    if t in ("sleap", "sleap-nn", "sleap_nn"):
        return SLEAPTracker
    return None


def accepted_options(tracker_type: str) -> set:
    """What this backend's constructor actually takes.

    Read from the signature, so it cannot drift from the constructor the way a
    hand-kept list does, and one did: the offline path granted DLC only
    ``model_type``, silently dropping ``device``, ``precision`` and
    ``dynamic_crop`` on the way to every retrack.
    """
    import inspect

    cls = tracker_class(tracker_type)
    if cls is None:
        return set()
    return {name for name in inspect.signature(cls.__init__).parameters
            if name not in ("self", "model_path")}


def filter_options(tracker_type: str, given: Dict[str, Any]):
    """``(kwargs this backend accepts, names dropped)``.

    THE rule for both pipelines. The live path relied on its caller to hand it
    a dict containing only that backend's options, which worked, but made an
    unknown key a ``TypeError`` in the middle of a recording rather than a
    line in the log.
    """
    accepted = accepted_options(tracker_type)
    prefix = "sleap" if (tracker_type or "").lower().startswith("sleap") else "dlc"
    out: Dict[str, Any] = {}
    dropped = []
    for key, value in (given or {}).items():
        if value is None or key in AMBIGUOUS_OPTIONS:
            continue
        if key in accepted:
            out[key] = value
        elif key.startswith(("dlc_", "sleap_")):
            continue                      # another backend's namespaced key
        else:
            dropped.append(key)
    for key in AMBIGUOUS_OPTIONS:
        named = f"{prefix}_{key}"
        if (given or {}).get(named) is not None and key in accepted:
            out[key] = given[named]
        elif (given or {}).get(key) is not None:
            dropped.append(f"{key} (pass it as {named})")
    return out, sorted(dropped)


def create_pose_tracker(tracker_type: str, model_path: str, **kwargs) -> PoseTracker:
    """Create a pose tracker by type.

    Args:
        tracker_type: "dlc" or "sleap"
        model_path: Path to model directory/file
        **kwargs: Additional args (body_parts, resize_factor, etc.)

    Returns:
        PoseTracker instance (not yet initialized -- call .initialize())
    """
    t = tracker_type.lower().strip()
    if t in ("dlc", "deeplabcut"):
        return DLCLiveTracker(model_path, **kwargs)
    elif t in ("sleap", "sleap-nn", "sleap_nn"):
        return SLEAPTracker(model_path, **kwargs)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}. Use 'dlc' or 'sleap'.")
