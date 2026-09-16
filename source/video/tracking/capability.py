"""What this machine can actually run pose on.

Every inference option that is offered but unusable ends the same way: a rig
that silently does something slower or different from what the dialog says. The
worst of them is quiet by design, ``onnxruntime-gpu`` built against the wrong
CUDA loads, reports success, and runs on the CPU; only
``session.get_providers()`` reveals it.

So the option surface is built from a probe, not from a static list, and the
probe answers three questions per capability: is it here, what is it, and if it
is not here, *why*, because a control greyed out saying "needs a CUDA GPU"
teaches an operator something and a control that has vanished does not.

Nothing here imports torch, tensorrt or onnxruntime at module scope: the probe
is called from the GUI thread when a dialog opens, and a multi-second import on
that thread is a freeze.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Capabilities:
    """One machine's inference facts. Cheap to hold, never persisted.

    These belong to the machine, not to the project: a project opened on the
    Jetson must not carry the workstation's answers.
    """

    cuda: bool = False
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    tensorrt: bool = False
    tensorrt_version: str = ""
    onnxruntime: bool = False
    ort_providers: Tuple[str, ...] = ()
    cpu_cores: int = 1
    torch: bool = False
    reasons: dict = field(default_factory=dict)

    # ── what the dialog asks ──────────────────────────────────────────

    def can(self, what: str) -> bool:
        """Is ``what`` usable here, ``tensorrt`` | ``onnx`` | ``cuda`` | ``fp16``."""
        return {
            "cuda": self.cuda,
            "fp16": self.cuda,          # half precision on CPU is slower, not faster
            "tensorrt": self.cuda and self.tensorrt,
            "onnx": self.onnxruntime,
        }.get(what, False)

    def why_not(self, what: str) -> str:
        """The sentence to put beside a disabled control."""
        if self.can(what):
            return ""
        return self.reasons.get(what, "not available on this machine")

    @property
    def ort_gpu(self) -> bool:
        """True when ONNX Runtime can actually reach the GPU here.

        Installed is not the same as usable: the CUDA provider is only real if
        it appears in ``get_available_providers()``.
        """
        return any("CUDA" in p or "Tensorrt" in p for p in self.ort_providers)

    def summary(self) -> str:
        bits = [f"{self.cpu_cores} cores"]
        if self.cuda:
            mem = f", {self.gpu_memory_gb:.1f} GB" if self.gpu_memory_gb else ""
            bits.append(f"CUDA: {self.gpu_name or 'yes'}{mem}")
        else:
            bits.append("no CUDA")
        if self.tensorrt:
            bits.append(f"TensorRT {self.tensorrt_version or 'yes'}")
        if self.onnxruntime:
            bits.append("ORT " + ("GPU" if self.ort_gpu else "CPU-only"))
        return " · ".join(bits)


def _probe_torch(reasons: dict):
    try:
        import torch
    except Exception:
        reasons["cuda"] = "PyTorch is not installed"
        reasons["fp16"] = reasons["cuda"]
        return False, False, "", 0.0
    if not torch.cuda.is_available():
        reasons["cuda"] = "PyTorch reports no CUDA device"
        reasons["fp16"] = reasons["cuda"]
        return True, False, "", 0.0
    name, mem = "", 0.0
    try:
        props = torch.cuda.get_device_properties(0)
        name = getattr(props, "name", "")
        mem = float(getattr(props, "total_memory", 0)) / (1024 ** 3)
    except Exception as e:
        logger.debug("GPU properties unreadable: %s", e)
    return True, True, name, mem


def _probe_tensorrt(cuda: bool, reasons: dict):
    if not cuda:
        reasons["tensorrt"] = "TensorRT needs a CUDA GPU"
        return False, ""
    try:
        import tensorrt
    except Exception:
        reasons["tensorrt"] = "the tensorrt package is not installed"
        return False, ""
    version = str(getattr(tensorrt, "__version__", ""))
    # sleap-nn 0.3.x uses NetworkDefinitionCreationFlag.EXPLICIT_BATCH, which
    # TensorRT 11 removed, so a too-new install fails at export with an error
    # that does not name the cause.
    major = version.split(".")[0] if version else ""
    if major.isdigit() and int(major) >= 11:
        reasons["tensorrt"] = (
            f"TensorRT {version} removed the export API sleap-nn uses "
            "(needs >=10,<11)")
        return False, version
    return True, version


def _probe_onnxruntime(reasons: dict):
    try:
        import onnxruntime as ort
    except Exception:
        reasons["onnx"] = "the onnxruntime package is not installed"
        return False, ()
    try:
        providers = tuple(ort.get_available_providers())
    except Exception as e:
        logger.debug("ORT provider list unreadable: %s", e)
        providers = ()
    return True, providers


def probe() -> Capabilities:
    """Ask this machine what it can do. Safe to call anywhere; never raises."""
    reasons: dict = {}
    torch_ok, cuda, gpu_name, gpu_mem = _probe_torch(reasons)
    trt, trt_version = _probe_tensorrt(cuda, reasons)
    ort_ok, providers = _probe_onnxruntime(reasons)
    caps = Capabilities(
        cuda=cuda, gpu_name=gpu_name, gpu_memory_gb=gpu_mem,
        tensorrt=trt, tensorrt_version=trt_version,
        onnxruntime=ort_ok, ort_providers=providers,
        cpu_cores=max(1, os.cpu_count() or 1),
        torch=torch_ok, reasons=reasons)
    return caps


_cached: Optional[Capabilities] = None


def capabilities(refresh: bool = False) -> Capabilities:
    """Cached probe. Hardware does not change mid-session; imports are slow."""
    global _cached
    if _cached is None or refresh:
        _cached = probe()
        logger.info("inference capabilities: %s", _cached.summary())
        for what, why in _cached.reasons.items():
            logger.debug("  %s unavailable: %s", what, why)
    return _cached


def inference_threads(caps: Optional[Capabilities] = None,
                      reserved: int = 4) -> int:
    """How many threads an ONNX Runtime session may use.

    ONNX Runtime's default is every core it can find, which on an idle bench
    looks like the fastest choice and on a live rig collides with frame
    acquisition, encoding and the control loop. The published measurement is
    stark: one thread is 55 ms/frame and all cores is 9.3 ms, but at four
    threads the *p99* is 45 ms against a 50 ms budget, so the number has to be
    chosen deliberately, and chosen against p99 rather than the mean.

    ``reserved`` is what the rest of the rig keeps. The result never drops
    below 2: a single-threaded session is slower than the budget it protects.
    """
    caps = caps or capabilities()
    return max(2, caps.cpu_cores - max(0, reserved))


def assert_provider(session, want_gpu: bool) -> str:
    """Return the provider actually in use, warning when it is not the one asked
    for.

    ``onnxruntime-gpu`` built against the wrong CUDA loads without complaint
    and runs on the CPU. Nothing surfaces that except this call, and a rig
    quietly at a fifth of its expected speed is a rig whose deadline misses
    have no visible cause.
    """
    try:
        providers = list(session.get_providers())
    except Exception as e:
        logger.debug("provider query failed: %s", e)
        return ""
    active = providers[0] if providers else ""
    if want_gpu and "CUDA" not in active and "Tensorrt" not in active:
        logger.warning(
            "ONNX Runtime fell back to %s despite a GPU being requested, "
            "available: %s. The onnxruntime-gpu build most likely does not "
            "match the installed CUDA.", active or "no provider", providers)
    return active


__all__ = ["Capabilities", "assert_provider", "capabilities",
           "inference_threads", "probe"]
