"""Run an exported DeepLabCut graph directly, without DeepLabCut-Live.

A DLC 3 project can be exported to ONNX or TensorRT, and the export decodes
its own peaks: the graph's single output is ``poses`` of shape
``(N, instances, keypoints, 3)`` carrying ``x, y, score``. Nothing about
running that needs DeepLabCut-Live, TensorFlow, or the ``deeplabcut`` package.

That matters here because the alternatives do not work:

* ``model_type="base"`` sends DeepLabCut-Live to its **TensorFlow** runner,
  and this environment has no TensorFlow, which is why a PyTorch model was
  refused with a message blaming TensorFlow, an error that named the wrong
  thing entirely;
* ``model_type="pytorch"`` wants the ``.pt`` file rather than the folder, and
  then fails inside ``torch.load(weights_only=True)`` because a DLC snapshot
  pickles ``deeplabcut…NetType``, a class this environment cannot import to
  allow-list.

The export sidesteps all of it, and is faster: the same reasoning as the SLEAP
direct runner, which measured 1.09 ms against 4.79 ms through the library.

This module only decides *whether* an export is usable and what it declares.
The session itself is built by :mod:`source.video.tracking.ort_runner`, which
already owns thread counts and the CUDA-provider check.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

#: File names a DLC export is written under. Checked in preference order,
#: TensorRT first, because where both exist the engine is the faster one.
ARTIFACTS = (("dlc_pose.trt", "tensorrt"), ("model.trt", "tensorrt"),
             ("dlc_pose.onnx", "onnx"), ("model.onnx", "onnx"))

#: The manifest a DLC export writes beside its graph. Distinguished from
#: SLEAP's manifest of the same name by the key it uses for the keypoints:
#: DLC says ``bodyparts``, SLEAP says ``node_names``.
MANIFEST = "export_metadata.json"


def artifacts(model_path: str) -> list[tuple[str, str]]:
    """``[(path, runtime)]`` for the DLC exports in ``model_path``."""
    if not model_path or not os.path.isdir(model_path):
        return []
    found = []
    for name, runtime in ARTIFACTS:
        path = os.path.join(model_path, name)
        if os.path.isfile(path):
            found.append((path, runtime))
    return found


def read_manifest(model_path: str) -> dict:
    """What a DLC export declares about itself, or ``{}``.

    Returns empty for a SLEAP export, so a folder is never mistaken for the
    wrong toolkit's: the two write the same filename with different keys.
    """
    path = os.path.join(model_path or "", MANIFEST)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError) as e:
        logger.debug("unreadable DLC manifest %s: %s", path, e)
        return {}
    return meta if isinstance(meta, dict) and meta.get("bodyparts") else {}


def input_shape(meta: dict) -> tuple[int, int] | None:
    """``(width, height)`` the export was built for, or None.

    The manifest writes ``input`` as ``[N, C, H, W]`` with ``layout: NCHW``;
    reading it as width-then-height letterboxes every frame to the transpose
    of the shape the graph accepts, which fails on the first call.
    """
    shape = meta.get("input")
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        return None
    layout = str(meta.get("layout") or "NCHW").upper()
    try:
        _n, a, b, c = (int(v) for v in shape)
    except (TypeError, ValueError):
        return None
    if layout == "NCHW":
        return c, b
    if layout == "NHWC":
        return b, a
    return None


def usable(model_path: str) -> bool:
    """Whether this folder holds a DLC export we can drive directly."""
    return bool(artifacts(model_path) and read_manifest(model_path))


class DLCExportRunner:
    """Callable: NCHW uint8 batch in, one pose dict per frame out.

    Deliberately the same shape as the tracker interface above it, so the
    retrack loop cannot tell this from DeepLabCut-Live, including
    ``predict_batch``, which is where the offline speed is.
    """

    def __init__(self, model_path: str, *, prefer_gpu: bool = True,
                 threads: int | None = None, pad_to: int = 0):
        #: Pad every batch to this size when the graph's batch axis is
        #: dynamic. A dynamic axis is only fast at a steady shape, see
        #: ORTRunner.__call__ for the measurement.
        self.pad_to = int(pad_to or 0)
        self.model_path = model_path
        self.meta = read_manifest(model_path)
        self.body_parts: list[str] = [str(b) for b in
                                      (self.meta.get("bodyparts") or ())]
        self.input_size = input_shape(self.meta)
        found = artifacts(model_path)
        if not found:
            raise FileNotFoundError(
                f"no DLC export in {model_path}, expected one of "
                + ", ".join(name for name, _ in ARTIFACTS))
        self._runner = None
        errors = []
        for path, runtime in found:
            try:
                self._runner = self._open(path, runtime, prefer_gpu, threads)
                self.runtime = runtime
                self.artifact = path
                break
            except Exception as e:
                errors.append(f"{os.path.basename(path)} ({runtime}): {e}")
                logger.info("DLC export %s unusable here: %s",
                            os.path.basename(path), e)
        if self._runner is None:
            raise RuntimeError(
                "none of this model's exports can run in this environment, "
                + "; ".join(errors))
        self.max_batch = self._graph_batch()
        logger.info("DLC driving the %s export directly: %s (%d parts, "
                    "input %s, max batch %s)", self.runtime,
                    os.path.basename(self.artifact), len(self.body_parts),
                    self.input_size, self.max_batch or "dynamic")
        if self.max_batch == 1:
            # Said once, at load, rather than discovered by a refused batch on
            # every call. A fixed batch axis is baked in at export time, so
            # this is a re-export away and worth naming: the SLEAP model beside
            # it runs 6x faster for exactly this reason.
            logger.info(
                "This export's batch axis is fixed at 1, so frames go through "
                "one at a time. Re-exporting it with a dynamic batch axis is "
                "what makes an offline retrack several times faster.")

    def _graph_batch(self) -> int | None:
        """The batch size the graph accepts, or None when it is dynamic."""
        sess = getattr(self._runner, "_sess", None)
        try:
            dim = sess.get_inputs()[0].shape[0] if sess is not None else None
        except Exception:
            return None
        # A symbolic axis comes back as a string ("batch"); a fixed one as an
        # int. Only the int constrains us.
        return int(dim) if isinstance(dim, int) else None

    @property
    def batched(self) -> bool:
        """Whether submitting more than one frame is worth trying."""
        return self.max_batch != 1

    def _open(self, path: str, runtime: str, prefer_gpu: bool, threads):
        if runtime == "tensorrt":
            from source.video.tracking import trt_runner
            return trt_runner.TRTRunner(path,
                                        device="cuda" if prefer_gpu else "cpu")
        from source.video.tracking.ort_runner import ORTRunner
        return ORTRunner(path, prefer_gpu=prefer_gpu, threads=threads,
                         pad_to=self.pad_to)

    @property
    def backend(self) -> str:
        return f"dlc:{getattr(self._runner, 'backend', self.runtime)}"

    @property
    def provider(self) -> str:
        return getattr(self._runner, "provider", "")

    def __call__(self, nchw_uint8: np.ndarray):
        """One forward pass, returning the raw ``poses`` array."""
        out = self._runner(np.ascontiguousarray(nchw_uint8))
        return _poses_of(out)

    def close(self) -> None:
        close = getattr(self._runner, "close", None)
        if close is not None:
            close()
        self._runner = None


def _poses_of(out):
    """The ``poses`` array out of whatever the runner returned.

    ORTRunner names its outputs for the SLEAP graph (``peaks``/``peak_vals``);
    a DLC graph has one output called ``poses``. Rather than teach the runner
    about a second graph, this accepts either shape of return.
    """
    if isinstance(out, dict):
        for key in ("poses", "pose", "output", "peaks"):
            if key in out:
                return np.asarray(out[key])
        return np.asarray(next(iter(out.values())))
    return np.asarray(out)


def to_pose_dicts(poses: np.ndarray, body_parts: list[str]
                  ) -> list[dict[str, list | None]]:
    """``(N, instances, keypoints, 3)`` → one ``{part: [x, y, score]}`` per frame.

    Single-animal: the first instance is taken and the rest ignored, which is
    the contract the rest of the pipeline is written to. A non-finite point is
    reported as absent rather than as a coordinate of NaN, the loop already
    treats a missing keypoint as "not seen this frame".
    """
    arr = np.asarray(poses, dtype=float)
    while arr.ndim > 3:                       # (N, I, K, 3) → (N, K, 3)
        arr = arr[:, 0]
    if arr.ndim == 2:                         # a single frame came back bare
        arr = arr[None]
    out: list[dict[str, list | None]] = []
    for frame in arr:
        pose: dict[str, list | None] = {}
        for i, name in enumerate(body_parts):
            if i >= len(frame):
                pose[name] = None
                continue
            x, y = float(frame[i][0]), float(frame[i][1])
            score = float(frame[i][2]) if len(frame[i]) > 2 else 1.0
            pose[name] = ([x, y, score] if np.isfinite(x) and np.isfinite(y)
                          else None)
        out.append(pose)
    return out


__all__ = ["ARTIFACTS", "DLCExportRunner", "artifacts", "input_shape",
           "read_manifest", "to_pose_dicts", "usable"]
