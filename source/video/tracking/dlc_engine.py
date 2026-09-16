"""Which engine runs a DeepLabCut model, and what DeepLabCut-Live needs to be handed.

A DLC folder can hold up to three runnable things at once, and they are not
interchangeable:

* an **export** (``dlc_pose.onnx`` / ``.trt``) driven by :mod:`dlc_export`,
  fastest, needs neither DeepLabCut-Live nor a framework, but its input size
  and batch are fixed at export time;
* a **PyTorch snapshot** (``*.pt``) driven by DeepLabCut-Live's PyTorch runner,
  accepts any frame size the stride divides, and measured more accurate here
  than the export letterboxed to its baked-in shape;
* a **TensorFlow snapshot** (``pose_cfg.yaml`` + ``*.pb``) driven by
  DeepLabCut-Live's TensorFlow runners (``base`` / ``tensorrt`` / ``lite``).

The choice was previously not a choice: the tracker tried the export first and
returned, so a folder holding one could never run its own weights however the
dialog was set. This module makes the preference explicit and, for ``auto``,
picks by what is actually on disk and importable rather than by a default that
names TensorFlow on a machine that has none.

Two things stand between a DLC 3 snapshot and ``torch.load``, and both are
handled here rather than left as "PyTorch does not initialise":

* DeepLabCut-Live's PyTorch runner takes the **snapshot file**, not the folder.
  Handing it a directory raises ``PermissionError`` on Windows and
  ``IsADirectoryError`` elsewhere, neither of which mentions the real problem.
* DeepLabCut pickles ``deeplabcut…NetType``, ``…MethodType`` and a
  ``pathlib.WindowsPath`` into the checkpoint's config, and dlclive loads with
  ``weights_only=True``, whose allow-list rejects all three. The classes cannot
  be allow-listed the ordinary way on a machine without ``deeplabcut``
  installed, and a ``WindowsPath`` cannot even be *constructed* off Windows,
  so importing the real class would not fix it on the Jetson. Each of the three
  stands in for a string, so a ``str`` subclass registered under the same
  qualified name reconstructs it faithfully and keeps every
  ``cfg["method"] == "td"`` comparison working.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import types
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Engines this module can resolve to, in the order ``auto`` prefers them.
#: Export first because it is ~3x faster; PyTorch second because it is the one
#: that reads the folder's own weights at any frame size.
ENGINES = ("export", "pytorch", "tensorflow")

#: The answer for a folder none of the checks recognise: hand it to
#: DeepLabCut-Live unchanged. See :func:`_passthrough`.
PASSTHROUGH = "dlclive"

#: DeepLabCut-Live's own ``model_type`` vocabulary for the TensorFlow runners.
#: ``tensorrt`` here is TensorFlow-TensorRT, which is NOT our ONNX/TRT export,
#: conflating the two is why a "tensorrt" selection used to land on a runner
#: that needed TensorFlow.
TF_MODEL_TYPES = ("base", "tensorrt", "lite")

_MISSING_GLOBAL = re.compile(r"Unsupported global: GLOBAL ([\w.]+) ")

#: Qualified names already stood in for, so a second model does not repeat the
#: probe-load. Process-wide because ``add_safe_globals`` is.
_allowed: set = set()


# ── what is on disk ──────────────────────────────────────────────────────

def snapshot_path(model_path: str) -> Optional[str]:
    """The PyTorch snapshot inside ``model_path``, or None.

    A folder may hold several (``snapshot-100.pt``, ``snapshot-best-720.pt``);
    the newest wins, because that is the one a training run just wrote and the
    one an operator means by "the model". A path given straight to a ``.pt`` is
    returned unchanged.
    """
    if not model_path:
        return None
    if os.path.isfile(model_path) and model_path.lower().endswith(".pt"):
        return model_path
    if not os.path.isdir(model_path):
        return None
    found = [os.path.join(model_path, n) for n in os.listdir(model_path)
             if n.lower().endswith(".pt")]
    if not found:
        return None
    return max(found, key=os.path.getmtime)


def has_tensorflow_snapshot(model_path: str) -> bool:
    """Whether ``model_path`` is a DLC TensorFlow export directory.

    The same test DeepLabCut-Live's own ``Engine.from_model_path`` uses: a
    ``pose_cfg.yaml`` beside a frozen ``.pb``.
    """
    if not model_path or not os.path.isdir(model_path):
        return False
    if not os.path.isfile(os.path.join(model_path, "pose_cfg.yaml")):
        return False
    return any(n.lower().endswith(".pb") for n in os.listdir(model_path))


def available(model_path: str) -> List[str]:
    """The engines this folder could be run with, in ``ENGINES`` order.

    "Could" means the artifact is present, not that its runtime imports,
    :func:`runnable` is the stricter question, and the two are kept apart so a
    missing package is reported as a missing package rather than a missing
    model.
    """
    from source.video.tracking import dlc_export

    out = []
    if dlc_export.usable(model_path):
        out.append("export")
    if snapshot_path(model_path):
        out.append("pytorch")
    if has_tensorflow_snapshot(model_path):
        out.append("tensorflow")
    return out


def _imports(module: str) -> bool:
    import importlib
    try:
        importlib.import_module(module)
        return True
    except Exception as e:
        logger.debug("%s is not importable here: %s", module, e)
        return False


def runnable(model_path: str) -> List[str]:
    """The engines that could actually execute here, artifact **and** runtime.

    An engine whose runtime is absent is dropped rather than attempted and
    blamed: a TensorFlow model on a machine without TensorFlow is a missing
    package, and saying so is the difference between an operator installing one
    and an operator re-exporting a model that was fine.
    """
    needs = {"export": "onnxruntime", "pytorch": "torch",
             "tensorflow": "tensorflow"}
    out = []
    for engine in available(model_path):
        if _imports(needs[engine]):
            out.append(engine)
        else:
            logger.info(
                "DLC model %s could run as %s, but %s is not installed in this "
                "environment.", model_path, engine, needs[engine])
    return out


#: Boxes at which ``auto`` switches from the export to the PyTorch runner.
#:
#: The export's batch axis is fixed at 1, so N boxes are N forward passes; the
#: PyTorch runner batches. That does NOT make it faster everywhere, because it
#: also carries about 11 ms of fixed per-call overhead the export does not.
#: Measured live, per box, with recording, zones and annotation running:
#:
#:     boxes    export    pytorch
#:         4      22.0       16.4      export, clearly
#:         8      13.2       13.8      a tie
#:        16       6.1        9.7      pytorch, by 60%
#:
#: Eight, from that table. An earlier value of four came from timing the two
#: engines in isolation on always-full batches, which flattered the batched
#: path: in a live session frames arrive jittered, batches are often partial,
#: and a partial batch pays the per-call overhead without earning the sharing.
#: The isolated number said four; the rig said eight, and the rig is right.
BATCH_CROSSOVER_BOXES = 8


def resolve(model_path: str, model_type: str = "auto",
            n_boxes: int = 1) -> Tuple[str, str, str]:
    """``(engine, path_for_dlclive, note)`` for a requested ``model_type``.

    ``model_type`` is one of ``auto`` | ``export`` | ``onnx`` | ``tensorrt`` |
    ``pytorch`` | ``base`` | ``lite`` | ``tensorflow``. The ONNX/TensorRT names
    both mean this repo's export runner, which picks the faster artifact
    itself; ``base``/``lite`` are DeepLabCut-Live's TensorFlow runners.

    ``note`` is empty when the request was honoured and carries the reason when
    it was not, a downgrade is a thing the operator has to be told, because a
    run that quietly fell back to a different engine is measuring something
    other than what was asked for.
    """
    asked = (model_type or "auto").strip().lower() or "auto"
    can = runnable(model_path)
    wanted = {"export": "export", "onnx": "export", "tensorrt_export": "export",
              "pytorch": "pytorch", "torch": "pytorch",
              "tensorflow": "tensorflow", "base": "tensorflow",
              "lite": "tensorflow", "tensorrt": "tensorflow"}.get(asked)

    if asked == "auto":
        order = list(ENGINES)
        if int(n_boxes or 1) >= BATCH_CROSSOVER_BOXES and "pytorch" in can:
            # A multi-box rig submits batches, and only the PyTorch runner can
            # take one. See BATCH_CROSSOVER_BOXES.
            order = ["pytorch"] + [e for e in order if e != "pytorch"]
        for engine in order:
            if engine in can:
                return engine, _path_for(engine, model_path), ""
        return _passthrough(model_path)

    if wanted is None:
        return "none", model_path, f"unknown DLC engine {model_type!r}"
    if wanted in can:
        return wanted, _path_for(wanted, model_path), ""
    if can:
        fallback = next(e for e in ENGINES if e in can)
        return fallback, _path_for(fallback, model_path), (
            f"{asked} is not available for this model (it offers "
            f"{', '.join(can)}), running as {fallback} instead")
    return _passthrough(model_path)


def _passthrough(model_path: str) -> Tuple[str, str, str]:
    """Hand the path to DeepLabCut-Live and let it decide.

    Reached when nothing here recognises the folder. That is NOT the same as
    the folder being unusable: the checks above look for the layouts this repo
    knows, and DeepLabCut-Live carries its own detection plus the model-zoo
    conventions. Refusing here would turn a folder it could have opened into a
    failure whose message named our checks rather than the real problem, so the
    permissive answer is the correct one, and dlclive's own error is a better
    error than any we could invent for a layout we do not recognise.
    """
    return PASSTHROUGH, model_path, ""


def _path_for(engine: str, model_path: str) -> str:
    """What DeepLabCut-Live has to be handed for ``engine``.

    The PyTorch runner calls ``torch.load`` on this path, so it must be the
    snapshot file. The TensorFlow runner globs the directory.
    """
    if engine == "pytorch":
        return snapshot_path(model_path) or model_path
    return model_path


def dlclive_model_type(engine: str, model_type: str) -> str:
    """The string DeepLabCut-Live's own factory expects for ``engine``.

    Our vocabulary is wider than dlclive's on purpose; it has no name for
    "this repo's ONNX export", so the translation happens once, here.
    """
    if engine == "pytorch":
        return "pytorch"
    asked = (model_type or "").strip().lower()
    return asked if asked in TF_MODEL_TYPES else "base"


# ── letting torch.load open a DeepLabCut snapshot ────────────────────────

def _stand_in(qualname: str):
    """A ``str`` subclass registered under ``qualname``, module tree included."""
    mod_name, _, cls_name = qualname.rpartition(".")
    parts = mod_name.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
        if i > 1:
            setattr(sys.modules[".".join(parts[:i - 1])], parts[i - 1],
                    sys.modules[name])
    existing = getattr(sys.modules[mod_name], cls_name, None)
    if existing is not None:
        return existing

    def __new__(cls, *args, **_kwargs):
        # Permissive on arity: the enum and the path reduce with different
        # argument counts, and refusing either would reintroduce the failure
        # this exists to remove.
        return str.__new__(cls, args[0] if args else "")

    cls = type(cls_name, (str,), {"__new__": __new__, "__module__": mod_name})
    setattr(sys.modules[mod_name], cls_name, cls)
    return cls


def batchable(dlc_live) -> bool:
    """Whether this DeepLabCut-Live instance's runner can take a whole batch.

    Only the PyTorch runner: it holds an ordinary ``nn.Module`` whose forward
    has always accepted ``(N, C, H, W)``. The TensorFlow runners compile their
    input placeholder as ``[1, H, W, 3]`` at ``init_inference``, so batch size
    one is baked into the graph and passing more is not a call-site change.

    Refused when the runner does its own per-frame cropping. A detector or a
    ``DynamicCropper`` gives each frame a different crop, so the frames stop
    sharing a shape and there is no batch to form, and the cropper carries
    per-frame offset state that a batched call would have to unpick.
    """
    runner = getattr(dlc_live, "runner", None)
    if runner is None or not hasattr(runner, "pose_transform"):
        return False
    if getattr(runner, "model", None) is None:
        return False                      # load_model has not run yet
    return (getattr(runner, "detector", None) is None
            and getattr(runner, "dynamic", None) is None)


def batched_get_pose(dlc_live, frames):
    """``[pose array]`` for a list of same-shape HWC frames, in one pass.

    DeepLabCut-Live has no batched entry point, its own docs list ``resize``,
    ``cropping``, ``dynamic`` and ``model_type`` as the speed levers and no
    batching among them, so a 16-box rig runs 16 forward passes for one camera
    frame. Measured on this rig's ResNet-50 at 360x202: 10.98 ms/frame one at a
    time against 1.58 ms/frame at a batch of 16, which is the difference
    between 208 and 633 inferences per second and therefore between missing and
    meeting a 16-box rig's 480/s.

    The model, the transform chain, the device and the precision are taken FROM
    the runner rather than rebuilt here. That is the whole point: the arithmetic
    is dlclive's own, so a batched call cannot drift from a single one as
    dlclive changes its preprocessing. What this adds is only the stacking.
    """
    import torch

    runner = dlc_live.runner
    with torch.inference_mode():
        # One contiguous uint8 block, one host->device copy, and the float
        # conversion and normalisation done ON the GPU.
        #
        # The order matters more than it looks. Transforming first and
        # uploading after would send float32, four times the bytes over PCIe
        # for the same picture, and would run the divide and the per-channel
        # normalise on the CPU, which is the resource a rig with sixteen
        # boxes, a video encoder, timestamp writing and an MCU push loop has
        # least of. Uploading uint8 first puts all of that on the GPU, which is
        # sitting at a third utilisation.
        stacked = np.ascontiguousarray(np.stack(frames))       # N,H,W,C uint8
        batch = torch.from_numpy(stacked).to(runner.device,
                                             non_blocking=True)
        batch = batch.permute(0, 3, 1, 2).contiguous()          # N,C,H,W
        batch = runner.pose_transform(batch)
        if getattr(runner, "precision", "FP32") == "FP16":
            batch = batch.half()
        outputs = runner.model(batch)
        poses = runner.model.get_predictions(outputs)["bodypart"]["poses"]
        if getattr(runner, "single_animal", True):
            # (N, individuals, parts, 3) -> (N, parts, 3), the same primary
            # instance a single-frame call returns.
            poses = poses[:, 0]
        return list(poses.cpu().numpy())


def allow_checkpoint_globals(snapshot: str, max_rounds: int = 24) -> List[str]:
    """Allow-list every global ``snapshot`` needs, so ``torch.load`` opens it.

    Driven by the error rather than by a fixed list: ``torch.load`` names one
    refused global per attempt, so each round stands that one in and retries.
    A hard-coded list would be a fourth place that has to learn about the next
    DeepLabCut release.

    Returns the qualified names newly stood in for. Raises whatever
    ``torch.load`` raised if the failure is not a refused global, a corrupt
    checkpoint must not look like a missing allow-list entry.
    """
    import torch

    added: List[str] = []
    for _ in range(max_rounds):
        try:
            torch.load(snapshot, map_location="cpu", weights_only=True)
            if added:
                logger.info(
                    "DLC snapshot %s: stood in for %s so torch.load could open "
                    "it with weights_only=True.", snapshot, ", ".join(added))
            return added
        except Exception as e:
            match = _MISSING_GLOBAL.search(str(e))
            if match is None:
                raise
            name = match.group(1)
            if name in _allowed:
                raise
            _allowed.add(name)
            added.append(name)
            torch.serialization.add_safe_globals([_stand_in(name)])
    logger.warning("DLC snapshot %s still refuses to load after %d allow-list "
                   "rounds", snapshot, max_rounds)
    return added


__all__ = ["ENGINES", "TF_MODEL_TYPES", "allow_checkpoint_globals",
           "available", "dlclive_model_type", "has_tensorflow_snapshot",
           "resolve", "runnable", "snapshot_path"]
