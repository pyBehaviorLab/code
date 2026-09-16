"""Export a trained sleap-nn model to an ONNX / TensorRT engine, cached once
per (model, device, runtime).

Engines are **device-specific**, a TensorRT ``.trt`` built on a desktop GPU
must NOT be reused on a Jetson, so the cache key includes the device. The
export itself (``sleap-nn export``) is slow, so we run it once and reuse the
cached engine on every subsequent init; the GUI's "Export" button just
pre-warms this cache.

Host-safe: the actual export shells out to sleap-nn and only runs on a machine
that has it installed + a GPU. Everything else here (cache-path derivation,
skip-if-present) is pure and unit-tested without sleap-nn.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _cache_root() -> Path:
    """Per-PC exports dir: ``<user config>/pybehaviorlab/sleap_exports/``."""
    from source.video.cameras.calibration_store import _user_config_dir
    d = _user_config_dir() / "sleap_exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _model_sig(model_path: str, centroid_path: Optional[str] = None) -> str:
    """Short djb2 over the model folder identity (path + its config's content),
    so re-training a model in place invalidates the cached engine."""
    from source.config.hashing import djb2_hex_from_text, djb2_hex_from_file
    parts = [str(model_path)]
    for base in (model_path, centroid_path):
        if not base:
            continue
        cfg = None
        for fn in ("training_config.yaml", "training_config.json"):
            p = os.path.join(base, fn) if os.path.isdir(base) else None
            if p and os.path.isfile(p):
                cfg = p
                break
        try:
            parts.append(djb2_hex_from_file(cfg) if cfg else str(base))
        except Exception:
            parts.append(str(base))
    return djb2_hex_from_text("|".join(parts))[:12]


def batch_bucket(n_boxes: int) -> int:
    """Round a box count up to the batch size an engine is built for.

    The engine's maximum batch is fixed at export, so it has to be at least as
    large as the biggest batch that will be submitted. Keying it on the exact
    box count would mean a multi-minute TensorRT rebuild every time a box is
    added or taken out of a session, so it rounds to a power of two: a 5-box
    and a 7-box rig share the 8-engine, and the batch simply never fills it.
    """
    n = max(1, int(n_boxes))
    bucket = 1
    while bucket < n and bucket < _MAX_BATCH_BUCKET:
        bucket *= 2
    return bucket


#: Ceiling on the bucket. Past this the engine's memory cost stops being worth
#: the batching win, and the sink chunks the rest.
_MAX_BATCH_BUCKET = 32


#: Peak threshold exported into a graph whose peak count is fixed by the model
#: (single-instance and top-down emit one peak per keypoint, each carrying its
#: own confidence). Exporting permissively and gating on the host makes the
#: operator's threshold a live control instead of a four-minute rebuild, and
#: makes native torch, ONNX and TensorRT agree because all three then apply the
#: SAME host-side gate. Bottom-up models are the exception: there the threshold
#: decides how many candidate peaks are emitted at all, so it stays baked.
PERMISSIVE_PEAK_THRESHOLD = 0.01

#: ONNX opset pinned rather than defaulted, so a sleap-nn upgrade cannot move
#: it under a runtime that has not caught up.
ONNX_OPSET = 17


def export_cache_dir(model_path: str, runtime: str, device: str = "cuda",
                     centroid_path: Optional[str] = None,
                     precision: str = "fp16",
                     max_batch_size: int = 8,
                     peak_threshold: float = PERMISSIVE_PEAK_THRESHOLD) -> Path:
    """The cache folder an exported engine for this (model, runtime, device,
    precision, max batch) lives in.

    ``device`` normalised (``cuda:0`` → ``cuda``) so port/index don't fragment
    the cache; the engine is still GPU-arch specific by nature.

    ``precision`` is part of the key because it is baked into the engine at
    export. Leaving it out meant a request for fp32 was served the cached fp16
    engine, the operator changed the setting, the run did not change, and
    nothing said so.

    ``max_batch_size`` is in the key for the same reason and with a sharper
    consequence: an engine built for 4 boxes cannot run a batch of 8, so
    serving it to a grown rig would fail at inference rather than merely
    surprise. It is bucketed (see :func:`batch_bucket`) so adding one box does
    not force a rebuild.

    ``peak_threshold`` likewise: sleap-nn compiles peak decoding INTO the
    graph, so the threshold is a constant in the exported artifact and not an
    argument anything can pass later. The rule the key follows is the same one
    every time: anything baked into the artifact belongs in the name of the
    artifact.
    """
    dev = (device or "cuda").split(":")[0]
    sig = _model_sig(model_path, centroid_path)
    prec = (precision or "fp16").lower()
    thr = f"{float(peak_threshold):g}".replace(".", "")
    return (_cache_root() /
            f"{sig}-{dev}-{runtime}-{prec}-b{int(max_batch_size)}-p{thr}")


def _engine_marker(out_dir: Path, runtime: str) -> Path:
    return out_dir / ("model.trt" if runtime == "tensorrt" else "model.onnx")


def is_exported(model_path: str, runtime: str, device: str = "cuda",
                centroid_path: Optional[str] = None,
                precision: str = "fp16", max_batch_size: int = 8,
                peak_threshold: float = PERMISSIVE_PEAK_THRESHOLD) -> bool:
    """True when a cached engine for this (model, runtime, device, precision,
    max batch, peak threshold) exists."""
    out = export_cache_dir(model_path, runtime, device, centroid_path,
                           precision, max_batch_size, peak_threshold)
    return _engine_marker(out, runtime).exists()


def ensure_exported(model_path: str, runtime: str, device: str = "cuda",
                    precision: str = "fp16", max_batch_size: int = 8,
                    centroid_path: Optional[str] = None,
                    peak_threshold: float = PERMISSIVE_PEAK_THRESHOLD,
                    force: bool = False, timeout: float = 1800.0) -> str:
    """Return an export dir holding the requested engine, running the (slow)
    ``sleap-nn export`` once and caching by (model, runtime, device,
    precision, max batch).

    Raises on failure so the caller can fall back to a lower runtime. Never
    re-exports when the engine is already cached (unless ``force``).
    """
    out = export_cache_dir(model_path, runtime, device, centroid_path,
                           precision, max_batch_size, peak_threshold)
    marker = _engine_marker(out, runtime)
    if marker.exists() and not force:
        logger.info("SLEAP export cache hit: %s", out)
        return str(out)
    out.mkdir(parents=True, exist_ok=True)

    fmt = "tensorrt" if runtime == "tensorrt" else "onnx"
    # Top-down exports both stages: MODEL args = [centroid, centered_instance].
    model_args = [centroid_path, model_path] if centroid_path else [model_path]
    args = [*_cli(), "export", *model_args,
            "-o", str(out), "-f", fmt,
            "--precision", precision,
            "--max-batch-size", str(int(max_batch_size)),
            "--peak-threshold", str(float(peak_threshold)),
            "--opset-version", str(int(ONNX_OPSET)),
            "--device", (device or "cuda").split(":")[0]]
    logger.info("SLEAP export: %s", " ".join(args))
    done = subprocess.run(args, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = (done.stdout or b"").decode("utf-8", "replace").strip()
    if done.returncode != 0:
        # The exporter's own words, not just "returned non-zero exit status 1".
        # That message was all the caller had, so an export that failed for a
        # nameable reason, a missing dependency, a config it could not read,
        # was indistinguishable from one that failed for any other.
        raise RuntimeError(
            f"sleap-nn export failed (exit {done.returncode}): "
            + (_tail(output) or "it printed nothing"))
    if not marker.exists():
        raise RuntimeError(
            f"sleap-nn export produced no {marker.name} in {out}"
            + (f": {_tail(output)}" if output else ""))
    return str(out)


def _cli() -> list:
    """How to invoke sleap-nn's command line from this interpreter.

    NOT ``-m sleap_nn``: the package has no ``__main__``, so that spelling
    fails with "'sleap_nn' is a package and cannot be directly executed",
    every single time, for every model, which is why exporting never worked.
    The module is ``sleap_nn.cli``, and sleap-nn's own docs prefer running it
    that way (it sets ``__main__.__spec__``, which its Lightning path reads).
    The installed console script is the fallback for a layout where the module
    cannot be run directly.
    """
    import importlib.util
    import shutil

    if importlib.util.find_spec("sleap_nn.cli") is not None:
        return [sys.executable, "-m", "sleap_nn.cli"]
    script = shutil.which("sleap-nn", path=os.path.join(
        os.path.dirname(sys.executable), "Scripts")) or shutil.which("sleap-nn")
    if script:
        return [script]
    raise RuntimeError(
        "sleap-nn is not installed in this Python environment, so a model "
        "cannot be exported here (pip install sleap-nn)")


def _tail(text: str, lines: int = 12) -> str:
    """The last few lines of a subprocess's output, where its error is."""
    kept = [ln for ln in (text or "").splitlines() if ln.strip()][-lines:]
    return " | ".join(kept)


__all__ = ["export_cache_dir", "is_exported", "ensure_exported", "batch_bucket",
           "PERMISSIVE_PEAK_THRESHOLD", "ONNX_OPSET"]
