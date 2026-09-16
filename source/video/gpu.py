"""Unified GPU-capability probe.

ONE place to answer "is the GPU usable, and for what" across the pipeline:
encoding (FFmpeg NVENC / QSV / AMF / Jetson v4l2m2m), pose inference
(PyTorch / TensorFlow CUDA), and OpenCV CUDA, plus which platform we're on
(Windows / Linux x86_64 / Jetson ARM64).

Advisory only: every accelerated stage still tries-then-falls-back on its
own, exactly like the optional-Cython pattern (``source/cython/build.py``).
Nothing here forces a heavy import at start-up, the inference probe checks
already-imported frameworks unless explicitly asked to import.
"""

from __future__ import annotations

import functools
import logging
import platform
import sys

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def platform_kind() -> str:
    """One of ``jetson`` | ``windows`` | ``macos`` | ``linux-x86_64`` |
    ``linux-arm64``. Jetson is detected from the device-tree model that
    JetPack kernels populate (same probe the encoder uses)."""
    machine = platform.machine().lower()
    system = platform.system()
    if machine in ("aarch64", "arm64") and system == "Linux":
        try:
            with open("/proc/device-tree/model") as f:
                model = f.read().strip("\x00").strip().lower()
            if "jetson" in model or "tegra" in model or "nvidia" in model:
                return "jetson"
        except OSError:
            pass
        return "linux-arm64"
    if system == "Windows":
        return "windows"
    if system == "Darwin":
        return "macos"
    return "linux-x86_64"


def is_jetson() -> bool:
    return platform_kind() == "jetson"


@functools.lru_cache(maxsize=1)
def _inference_probe(allow_import: bool) -> tuple[bool, str]:
    """Return ``(cuda_available, description)`` for pose inference.

    When ``allow_import`` is False (the start-up default) we only inspect
    frameworks that are ALREADY imported, so probing never pays torch/TF's
    multi-second import cost. Pose init calls this after the DL library has
    already imported its framework, so it gets a real answer cheaply.
    """
    torch = sys.modules.get("torch")
    if torch is None and allow_import:
        try:
            import torch  # type: ignore
        except Exception:
            torch = None
    if torch is not None:
        try:
            if torch.cuda.is_available():
                n = torch.cuda.device_count()
                name = torch.cuda.get_device_name(0) if n else "?"
                return True, f"torch CUDA ({name}, {n} device[s])"
            return False, "torch present, no CUDA"
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("torch cuda probe failed: %s", e)

    tf = sys.modules.get("tensorflow")
    if tf is None and allow_import:
        try:
            import tensorflow as tf  # type: ignore
        except Exception:
            tf = None
    if tf is not None:
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                return True, f"tensorflow GPU x{len(gpus)}"
            return False, "tensorflow present, no GPU"
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("tf gpu probe failed: %s", e)

    return False, "no torch / tensorflow imported"


def inference_device(allow_import: bool = False) -> tuple[bool, str]:
    """``(cuda_available, human description)`` for pose inference. Cached."""
    return _inference_probe(bool(allow_import))


@functools.lru_cache(maxsize=1)
def cv2_cuda_available() -> bool:
    """True only when OpenCV was built WITH_CUDA (never the case for the
    stock ``opencv-python`` pip wheel, so almost always False)."""
    try:
        import cv2
        return int(cv2.cuda.getCudaEnabledDeviceCount()) > 0
    except Exception:
        return False


def encode_summary() -> dict:
    """Encoder capabilities via the existing FFmpeg probe (does its own
    detection + caching). Returns a small dict; never raises."""
    try:
        from source.video.recording.ffmpeg import EncoderCapabilities
        caps = EncoderCapabilities.get_instance()
        caps.detect_all()
        nvmpi = bool(caps.detect_nvmpi()) if caps.is_jetson() else False
        return {
            "ffmpeg": bool(caps.detect_ffmpeg()),
            "nvenc": bool(caps.detect_nvenc()),
            "hevc_nvenc": bool(caps.detect_hevc_nvenc()),
            "nvmpi": nvmpi,
            # Only asked when nvmpi is absent, as detect_all does. Probing it
            # anyway logged "fall back to CPU libx264" on a Jetson that was
            # recording on h264_nvmpi.
            "v4l2m2m": (bool(caps.detect_v4l2m2m())
                        if caps.is_jetson() and not nvmpi else False),
            "libx264": bool(caps.detect_libx264()),
            "hw_encode": bool(caps.gpu_force_required()),
        }
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("encode_summary failed: %s", e)
        return {"ffmpeg": False, "hw_encode": False}


def summary(allow_inference_import: bool = False) -> dict:
    """Full advisory snapshot of GPU capabilities for logging / diagnostics."""
    infer_ok, infer_desc = inference_device(allow_inference_import)
    return {
        "platform": platform_kind(),
        "encode": encode_summary(),
        "inference_cuda": infer_ok,
        "inference_desc": infer_desc,
        "cv2_cuda": cv2_cuda_available(),
    }


def log_summary(allow_inference_import: bool = False) -> None:
    """Emit one INFO line describing GPU acceleration state."""
    s = summary(allow_inference_import)
    enc = s["encode"]
    hw = "NVENC" if enc.get("nvenc") else (
        "nvmpi" if enc.get("nvmpi") else (
            "v4l2m2m" if enc.get("v4l2m2m") else (
                "QSV" if enc.get("qsv") else (
                    "AMF" if enc.get("amf") else "CPU"))))
    logger.info(
        "GPU capabilities: platform=%s | encode=%s (hw=%s) | pose=%s (%s) | cv2.cuda=%s",
        s["platform"], hw, enc.get("hw_encode"),
        "CUDA" if s["inference_cuda"] else "CPU", s["inference_desc"],
        s["cv2_cuda"],
    )
