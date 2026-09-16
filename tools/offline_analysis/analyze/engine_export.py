"""Exported inference engines: where they live, when they may be used, and
when they must be built again.

A trained model is a set of weights. An **engine** is those weights compiled
for one runtime on one machine, ONNX for portability, TensorRT for speed. On
the sibling rig's measurements the difference is 4.79 ms per frame against
1.09, so this is the phase that buys the headline number.

Three facts drive every decision in this module.

**An engine is not portable.** A TensorRT plan is built for a specific GPU
architecture, TensorRT version and driver. Copying one to another machine, or
to a Jetson, is not slow, it is *wrong*, and it fails in ways that look like
bad tracking rather than like an error. So the cache lives per machine, the key
contains the device, and :func:`foreign_engine_reason` refuses one that does
not match rather than trusting a filename.

**The peak threshold is baked in at export.** ``sleap-nn export`` writes the
confidence cut into the ONNX graph. Changing the threshold afterwards does not
change what the engine emits, so the threshold is part of the cache key: a
different one is a different engine, not the same engine used differently.

**Re-training in place must invalidate the engine.** The signature hashes the
model's config *content*, not just its path, a folder that is retrained keeps
its name and would otherwise keep serving the old compiled weights.

Pure and import-light: nothing here imports torch, sleap-nn or tensorrt. The
one function that actually builds an engine shells out, and everything around
it, paths, keys, staleness, refusal, is decided here and unit-tested on a
machine with none of that installed.

See ``docs/pose_runtime_and_input_plan.html`` §5 (P5).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)

#: Runtimes a pose model can be executed by, in the order ``auto`` tries them.
#: Fastest first: a rig that can run TensorRT should not silently settle for
#: torch because nobody said which they wanted.
RUNTIME_AUTO = "auto"
RUNTIME_NATIVE = "native"
RUNTIME_ONNX = "onnx"
RUNTIME_TENSORRT = "tensorrt"
RUNTIMES = (RUNTIME_AUTO, RUNTIME_NATIVE, RUNTIME_ONNX, RUNTIME_TENSORRT)
#: The order ``auto`` resolves in. Native is last because it always works.
AUTO_ORDER = (RUNTIME_TENSORRT, RUNTIME_ONNX, RUNTIME_NATIVE)

#: Runtimes that are compiled ahead of time and therefore cached.
EXPORTED = (RUNTIME_ONNX, RUNTIME_TENSORRT)

PRECISION_FP32 = "fp32"
PRECISION_FP16 = "fp16"
PRECISIONS = (PRECISION_FP32, PRECISION_FP16)

#: The file an export writes, per runtime. Its presence is what "cached" means.
_ARTIFACT = {RUNTIME_ONNX: "model.onnx", RUNTIME_TENSORRT: "model.trt"}

#: Written beside the artifact so a foreign engine can be recognised as one.
STAMP = "engine.json"


# ── availability ─────────────────────────────────────────────────────────────

@dataclass
class RuntimeStatus:
    """Whether a runtime can run here, and why not when it cannot."""

    name: str
    available: bool
    reason: str = ""
    detail: str = ""

    @property
    def label(self) -> str:
        return self.name if self.available else f"{self.name}, {self.reason}"


def available_runtimes() -> Dict[str, RuntimeStatus]:
    """What this interpreter can actually execute, checked by importing.

    Asked when the operator CHOOSES, not when the run starts, the same rule
    the offline backends already follow. A six-hour job that dies on an import
    at minute one is the failure this prevents.
    """
    out: Dict[str, RuntimeStatus] = {
        RUNTIME_NATIVE: RuntimeStatus(RUNTIME_NATIVE, True,
                                      detail="the toolkit's own runtime"),
    }
    try:
        import onnxruntime as ort

        out[RUNTIME_ONNX] = RuntimeStatus(
            RUNTIME_ONNX, True,
            detail=", ".join(ort.get_available_providers()))
    except Exception as e:
        out[RUNTIME_ONNX] = RuntimeStatus(
            RUNTIME_ONNX, False, f"onnxruntime is not importable here ({e})")
    try:
        import tensorrt as trt

        version = str(getattr(trt, "__version__", "?"))
        major = int(version.split(".")[0]) if version[:1].isdigit() else 0
        if major >= 11:
            # TensorRT 11 removed NetworkDefinitionCreationFlag.EXPLICIT_BATCH,
            # which sleap-nn 0.3.3 still uses. Pinning is not a preference.
            out[RUNTIME_TENSORRT] = RuntimeStatus(
                RUNTIME_TENSORRT, False,
                f"TensorRT {version} removed EXPLICIT_BATCH, which the "
                f"exporter still uses, pin tensorrt>=10,<11")
        else:
            out[RUNTIME_TENSORRT] = RuntimeStatus(RUNTIME_TENSORRT, True,
                                                  detail=f"TensorRT {version}")
    except Exception as e:
        out[RUNTIME_TENSORRT] = RuntimeStatus(
            RUNTIME_TENSORRT, False, f"tensorrt is not importable here ({e})")
    return out


def resolve_runtime(requested: str, statuses: Optional[Dict[str, RuntimeStatus]] = None
                    ) -> Tuple[str, str]:
    """``(runtime_that_will_run, why_it_is_not_the_one_asked_for)``.

    ``auto`` walks :data:`AUTO_ORDER` and takes the first that is available.
    Anything else is honoured when it can be, and **falls back with a stated
    reason** when it cannot, the sibling repo's audit found precisely this
    fallback living in a log line while the dialog still showed the engine the
    operator had chosen, which is how a rig runs six months of sessions on a
    backend nobody meant to use.
    """
    st = statuses if statuses is not None else available_runtimes()
    want = (requested or RUNTIME_AUTO).lower()
    if want == RUNTIME_AUTO:
        for name in AUTO_ORDER:
            if st.get(name) and st[name].available:
                return name, ""
        return RUNTIME_NATIVE, ""
    if want not in RUNTIMES:
        return RUNTIME_NATIVE, f"unknown runtime '{requested}', using native"
    got = st.get(want)
    if got is not None and got.available:
        return want, ""
    reason = got.reason if got is not None else "unavailable"
    for name in AUTO_ORDER:
        if st.get(name) and st[name].available:
            return name, f"{want} unavailable ({reason}), using {name}"
    return RUNTIME_NATIVE, f"{want} unavailable ({reason}), using native"


# ── the cache ────────────────────────────────────────────────────────────────

def cache_root() -> str:
    """Where this machine keeps its compiled engines.

    Per machine, never inside the model folder: a model folder is often on a
    share, and an engine on a share is an engine on the wrong GPU.
    """
    from tools.offline_analysis.analyze.paths import CONFIG_DIR

    # Beside the camera capability cache, which is the rig's other
    # machine-specific store, same reasoning, same place.
    d = os.path.join(str(CONFIG_DIR), "pose_engines")
    os.makedirs(d, exist_ok=True)
    return d


def model_signature(model_path: str, centroid_path: str = "") -> str:
    """Short hash identifying this model AS IT IS NOW.

    Includes the config file's **content**, so re-training a model in place,
    which keeps the folder name, invalidates the engine built from the old
    weights. A path alone would keep serving it.
    """
    parts: List[str] = []
    for base in (model_path or "", centroid_path or ""):
        if not base:
            continue
        parts.append(os.path.normcase(os.path.abspath(base)))
        folder = base if os.path.isdir(base) else os.path.dirname(base)
        for name in ("training_config.yaml", "training_config.json",
                     "pose_cfg.yaml", "config.yaml"):
            cfg = os.path.join(folder, name)
            if os.path.isfile(cfg):
                try:
                    with open(cfg, "rb") as fh:
                        parts.append(hashlib.sha1(fh.read()).hexdigest())
                except OSError:                           # pragma: no cover
                    pass
                break
    blob = "|".join(parts) or "no-model"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def batch_bucket(n_streams: int) -> int:
    """Round a stream count up to a batch size an engine is built for.

    The maximum batch is fixed at export, so it must be at least as large as
    the biggest batch ever submitted, an engine built for four cannot run a
    batch of eight. Bucketing keeps a rig from rebuilding every time an arena
    is added or removed.
    """
    n = max(1, int(n_streams or 1))
    for size in (1, 2, 4, 8, 16):
        if n <= size:
            return size
    return 32


def export_dir(model_path: str, runtime: str, *, device: str = "cuda",
               precision: str = PRECISION_FP16, batch: int = 1,
               peak_threshold: float = 0.2, centroid_path: str = "") -> str:
    """Where the engine for exactly this combination lives.

    Every part of the key changes what the engine IS: a different device
    compiles different kernels, a different precision computes different
    numbers, a different batch is a different graph, and the peak threshold is
    baked into the graph at export.
    """
    name = "_".join([
        model_signature(model_path, centroid_path),
        str(runtime),
        _slug(device),
        str(precision or PRECISION_FP32).lower(),
        f"b{int(batch)}",
        f"p{float(peak_threshold):.3f}".replace(".", ""),
    ])
    return os.path.join(cache_root(), name)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in str(text or "")).strip("-")


def artifact_path(out_dir: str, runtime: str) -> str:
    """The file whose presence means "this engine is built"."""
    return os.path.join(out_dir, _ARTIFACT.get(runtime, "model.onnx"))


def is_exported(out_dir: str, runtime: str) -> bool:
    return os.path.isfile(artifact_path(out_dir, runtime))


# ── recognising an engine that does not belong here ──────────────────────────

def machine_stamp(device: str = "cuda") -> Dict[str, Any]:
    """What this machine is, as far as an engine is concerned."""
    stamp: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": ".".join(str(v) for v in sys.version_info[:2]),
        "device": str(device),
    }
    try:
        import tensorrt as trt

        stamp["tensorrt"] = str(getattr(trt, "__version__", ""))
    except Exception:
        pass
    try:
        import torch

        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            stamp["gpu"] = torch.cuda.get_device_name(0)
            stamp["driver"] = str(getattr(torch.version, "cuda", ""))
    except Exception:
        pass
    return stamp


def write_stamp(out_dir: str, runtime: str, device: str = "cuda",
                extra: Optional[Dict[str, Any]] = None) -> None:
    """Record what built this engine, beside it."""
    data = {"runtime": runtime, **machine_stamp(device), **(extra or {})}
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, STAMP), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
    except OSError as e:                                  # pragma: no cover
        logger.debug("could not stamp engine at %s: %s", out_dir, e)


def foreign_engine_reason(out_dir: str, runtime: str,
                          device: str = "cuda") -> str:
    """Why a cached engine may NOT be used here, or ``""`` when it may.

    A TensorRT plan built for another GPU, another TensorRT or another driver
    does not announce itself: it loads, or it crashes, or, worst; it runs and
    produces plausible nonsense. So the check is against a recorded stamp, and
    an engine with **no** stamp is refused too: it was built by something that
    did not know to say what it was.

    ONNX is portable by design, so only the TensorRT plan is machine-bound.
    """
    if runtime != RUNTIME_TENSORRT:
        return ""
    path = os.path.join(out_dir, STAMP)
    if not os.path.isfile(path):
        return "no build stamp, this engine cannot be shown to belong here"
    try:
        with open(path, encoding="utf-8") as fh:
            was = json.load(fh)
    except Exception:
        return "the build stamp is unreadable"
    now = machine_stamp(device)
    for field_name, label in (("gpu", "GPU"), ("tensorrt", "TensorRT version"),
                              ("driver", "CUDA driver"), ("machine", "machine")):
        before, after = was.get(field_name), now.get(field_name)
        if not before:
            continue
        if not after:
            # The plan says which GPU it was built for and this machine cannot
            # say which one it has. That is not agreement; it is an inability
            # to check, and the rule is that an engine which cannot be SHOWN to
            # belong here does not get used. Rebuilding costs minutes; a plan
            # from the wrong GPU costs a dataset.
            return (f"built for {label} {before}, and this machine cannot "
                    f"report its own, a TensorRT plan is not portable")
        if str(before) != str(after):
            return (f"built for {label} {before}, this machine has {after}, "
                    f"a TensorRT plan is not portable")
    return ""


# ── building one ─────────────────────────────────────────────────────────────

@dataclass
class ExportResult:
    ok: bool
    out_dir: str = ""
    runtime: str = ""
    reason: str = ""
    built: bool = False          # False when a valid cached engine was reused
    log: str = ""


def ensure_exported(model_path: str, runtime: str, *, device: str = "cuda",
                    precision: str = PRECISION_FP16, batch: int = 1,
                    peak_threshold: float = 0.2, centroid_path: str = "",
                    input_size: Optional[Tuple[int, int]] = None,
                    runner=None, timeout_s: float = 1800.0) -> ExportResult:
    """The engine for this combination, building it once if need be.

    ``runner`` is injectable; it takes the command as a list and returns
    ``(returncode, output)``: which is what makes the caching, the staleness
    rule and the refusal testable on a machine with no exporter at all. The
    default shells out to ``sleap-nn export``.
    """
    if runtime not in EXPORTED:
        return ExportResult(False, runtime=runtime,
                            reason=f"{runtime} is not an exported runtime")
    out_dir = export_dir(model_path, runtime, device=device,
                         precision=precision, batch=batch,
                         peak_threshold=peak_threshold,
                         centroid_path=centroid_path)

    if is_exported(out_dir, runtime):
        why = foreign_engine_reason(out_dir, runtime, device)
        if not why:
            return ExportResult(True, out_dir, runtime, built=False)
        # Refused, not reused: a plan from another machine fails in ways that
        # look like bad tracking rather than like an error.
        logger.warning("rebuilding the cached %s engine: %s", runtime, why)

    cmd = [sys.executable, "-m", "sleap_nn.export", model_path,
           "-o", out_dir, "-f", "both", "--device", device,
           "--precision", str(precision), "--max-batch-size", str(int(batch)),
           "--peak-threshold", str(float(peak_threshold))]
    if input_size and input_size[0] and input_size[1]:
        cmd += ["--input-width", str(int(input_size[0])),
                "--input-height", str(int(input_size[1]))]

    call = runner or _shell_out
    try:
        code, log = call(cmd, timeout_s)
    except Exception as e:                                # pragma: no cover
        return ExportResult(False, out_dir, runtime, reason=str(e))
    if code != 0 or not is_exported(out_dir, runtime):
        return ExportResult(False, out_dir, runtime,
                            reason=f"export failed (exit {code})", log=log)
    write_stamp(out_dir, runtime, device,
                extra={"peak_threshold": float(peak_threshold),
                       "precision": str(precision), "batch": int(batch)})
    return ExportResult(True, out_dir, runtime, built=True, log=log)


def _shell_out(cmd: List[str], timeout_s: float) -> Tuple[int, str]:
    """Run the exporter. Slow, needs the full stack, never runs in CI."""
    logger.info("exporting engine: %s", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# ── pre-warming, so the first Record is not the slow one ─────────────────────

def prewarm(model_path: str, runtime: str, *, device: str = "auto",
            precision: str = PRECISION_FP32, batch: int = 1,
            peak_threshold: float = 0.2, centroid_path: str = "",
            input_size: Optional[Tuple[int, int]] = None) -> ExportResult:
    """Build the engine for this configuration now, rather than at Record.

    An export is minutes. Paid at the start of a session it looks like the
    application has hung, and paid at the start of a RUN it eats the first
    minutes of an experiment. This is the "Export…" action: same cache, same
    key, just earlier.

    ``auto`` is resolved here, because an engine has to be built for something
    specific; there is no "auto" device to compile for.
    """
    chosen, note = resolve_runtime(runtime)
    if chosen not in EXPORTED:
        return ExportResult(False, runtime=chosen,
                            reason=note or f"{chosen} needs no export")
    dev = device
    if (dev or "auto").lower() == "auto":
        try:
            import torch

            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"
    res = ensure_exported(model_path, chosen, device=dev, precision=precision,
                          batch=batch_bucket(batch),
                          peak_threshold=peak_threshold,
                          centroid_path=centroid_path, input_size=input_size)
    if note and res.ok:
        res.reason = note          # it worked, but not with what was asked for
    return res


# ── the CPU thread budget ────────────────────────────────────────────────────

def onnx_threads(n_instances: int = 1, reserved: int = 2,
                 cores: Optional[int] = None) -> int:
    """How many threads one ONNX session may take.

    ONNX Runtime's default is **every core**, and the sibling's measurements
    show a 5.9× swing between one thread and all of them, with a p99 of
    45.27 ms at four threads against a 50 ms budget. A rig also needs cores for
    acquisition, encoding and closed-loop control, and N model copies each
    taking every core is oversubscription, not parallelism.

    So the pool owns the budget and divides it. Never ship the default.
    """
    total = int(cores or os.cpu_count() or 4)
    usable = max(1, total - max(0, int(reserved)))
    return max(1, usable // max(1, int(n_instances)))
