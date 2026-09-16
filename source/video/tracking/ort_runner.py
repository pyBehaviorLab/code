"""Run an exported pose graph on ONNX Runtime, with the thread count stated.

The exported graph decodes peaks itself, it returns ``peaks`` and
``peak_vals``, not heatmaps, so inference needs neither sleap-nn nor torch.
That makes ONNX Runtime the CPU deployment: same artifact, same coordinates, no
GPU.

The one number that decides whether it is usable is ``intra_op_num_threads``.
Published measurement on a 10-core host, 360x202 frames: 1 thread 55 ms/frame,
4 threads 25 ms, 8 threads 16 ms, and every core 9.3 ms. A 5.9x swing on a
setting whose default is "take everything I can find", which on an idle bench
looks like the right answer and on a live rig contends with frame acquisition,
encoding and the control loop. So it is always set explicitly, and chosen
against p99: at 4 threads the mean is 25 ms and the p99 is 45 ms against a
50 ms budget, which is a deadline miss waiting for a busy moment.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from source.video.tracking.capability import assert_provider, inference_threads

logger = logging.getLogger(__name__)

#: Whether the CUDA library directories have been put on the DLL search path.
#: Once per process: ``add_dll_directory`` accumulates, and re-adding the same
#: path on every session build would grow the loader's search list unboundedly.
_CUDA_DLLS_READY = False


def _cuda_dll_dirs() -> List[Path]:
    """Directories holding the CUDA libraries ONNX Runtime loads by name.

    Located WITHOUT importing torch, ``find_spec`` reads the package's
    location off disk, where importing it costs seconds and a large amount of
    memory for the sake of one directory path.
    """
    import importlib.util

    dirs: List[Path] = []
    for package, sub in (("torch", "lib"), ("nvidia", "")):
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):              # not installed
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        root = Path(list(spec.submodule_search_locations)[0])
        if sub:
            dirs.append(root / sub)
        else:
            # The nvidia-* wheels each ship their own bin/ (cudnn, cublas, …).
            dirs.extend(sorted(root.glob("*/bin")))
    return [d for d in dirs if d.is_dir()]


def enable_cuda_libraries() -> None:
    """Let ONNX Runtime find the CUDA libraries that are already installed.

    ``onnxruntime-gpu`` loads ``cudnn64_9.dll`` / ``cublas64_12.dll`` by name
    through the Windows loader, which searches PATH, not site-packages. The
    torch CUDA wheel ships exactly those files in ``torch/lib``, so on a
    machine with working GPU torch the libraries are present and merely
    unfindable. ORT then reports "Failed to create CUDAExecutionProvider",
    falls back to the CPU, and pose inference runs an order of magnitude
    slower for want of a directory on a search path.

    Idempotent and never fatal: a machine with no CUDA simply has nothing to
    add, and the CPU provider is a correct answer there.
    """
    global _CUDA_DLLS_READY
    if _CUDA_DLLS_READY:
        return
    _CUDA_DLLS_READY = True
    if os.name != "nt":
        return                        # POSIX resolves these through RPATH
    for directory in _cuda_dll_dirs():
        try:
            os.add_dll_directory(str(directory))
        except OSError as e:                          # pragma: no cover
            logger.debug("could not add %s to the DLL path: %s", directory, e)
            continue
        # Belt and braces: some loader paths consult PATH rather than the
        # added-directory list, and appending is harmless when they do not.
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get(
            "PATH", "")
        logger.debug("CUDA libraries: added %s to the DLL search path",
                     directory)


def graph_is_decoded(onnx_path: str) -> bool:
    """True when the graph returns keypoints rather than heatmaps.

    That is what makes this runner possible at all: without in-graph peak
    decoding the caller would have to reimplement sleap-nn's post-processing,
    which is the thing being avoided.
    """
    try:
        import onnxruntime as ort
    except Exception as e:
        # "Cannot tell" is not "no". A bare False here made a perfectly good
        # decoded graph look undecodable, and the caller then walked past a
        # usable model.onnx into an export attempt and finally into a native
        # checkpoint load, which cannot work on an export directory.
        logger.warning("cannot inspect %s: onnxruntime did not import (%s). "
                       "The graph may well decode peaks; this is a missing "
                       "dependency, not a bad model.", onnx_path, e)
        return False
    try:
        sess = ort.InferenceSession(str(onnx_path),
                                    providers=["CPUExecutionProvider"])
        names = {o.name for o in sess.get_outputs()}
        return {"peaks", "peak_vals"}.issubset(names)
    except Exception as e:
        logger.warning("ONNX graph probe failed for %s: %s", onnx_path, e)
        return False


class ORTRunner:
    """Callable: NCHW uint8 batch in, ``{"peaks", "peak_vals"}`` out.

    Deliberately the same signature as the TensorRT runner so the tracking loop
    above it does not know or care which one it holds.
    """

    def __init__(self, onnx_path: str, prefer_gpu: bool = False,
                 threads: Optional[int] = None, reserved_cores: int = 4,
                 pad_to: Optional[int] = None):
        # Before onnxruntime is imported: the CUDA provider resolves its
        # dependencies when it loads, and a search path widened afterwards is
        # widened too late.
        if prefer_gpu:
            enable_cuda_libraries()
        import onnxruntime as ort

        path = Path(onnx_path)
        if path.is_dir():
            path = path / "model.onnx"
        if not path.exists():
            raise FileNotFoundError(f"no ONNX graph at {path}")

        opts = ort.SessionOptions()
        # Errors only. The one warning this graph reliably produces is
        # "1 Memcpy nodes are added to the graph", which is expected and
        # harmless (see the provider comment below), and printing it on every
        # session start trains an operator to ignore the console. Nothing that
        # matters is lost: the failure this could hide - the CUDA provider not
        # loading - is caught explicitly by ``assert_provider`` a few lines
        # down, which names the provider that is ACTUALLY running.
        opts.log_severity_level = 3
        # Never the default. See the module docstring for why this one number
        # is the whole CPU story.
        self.threads = int(threads) if threads else inference_threads(
            reserved=reserved_cores)
        opts.intra_op_num_threads = self.threads
        # One session, one model: parallelising across nodes of the same graph
        # competes with the intra-op pool for the same cores.
        opts.inter_op_num_threads = 1

        # On the SLEAP exports a CUDA session prints "1 Memcpy nodes are added
        # to the graph main_graph". It is expected and it is not a fault: the
        # graph reads its own confidence-map width with Shape then Gather, ONNX
        # Runtime places that one scalar lookup on the CPU by its own rule, and
        # the 8-byte result is copied back for the three nodes that use it.
        # 62 of 63 nodes still run on the GPU, and measured cost is about 2% of
        # a batch, inside the run-to-run spread. It cannot be folded away
        # because the export accepts any height and width. See
        # docs/troubleshooting.md. A LARGE node count in that message is a
        # different matter and does mean real work fell back to the CPU.
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if prefer_gpu else ["CPUExecutionProvider"])
        self._sess = ort.InferenceSession(str(path), opts, providers=providers)
        # An onnxruntime-gpu built against the wrong CUDA loads happily and
        # runs on the CPU. This is the only place that shows.
        self.provider = assert_provider(self._sess, want_gpu=prefer_gpu)
        self._input = self._sess.get_inputs()[0].name
        self._input_dtype = self._sess.get_inputs()[0].type
        self._outputs = [o.name for o in self._sess.get_outputs()]
        # What the GRAPH accepts, which is the only honest answer to "can this
        # take a batch". A symbolic first dimension ('batch') means any size;
        # an integer means exactly that many. The manifest's max_batch_size
        # describes the TensorRT build sitting next to it and says nothing
        # about this graph, models/sleap declares 1 and runs 16 quite happily.
        first = self._sess.get_inputs()[0].shape[0]
        self.max_batch = first if isinstance(first, int) else None
        # Pad every batch out to this many frames, when the graph will take
        # them. See __call__, a dynamic batch axis is only fast at a STEADY
        # size.
        self.pad_to = int(pad_to) if (pad_to and self.max_batch is None) else 0
        if self.pad_to > 1:
            logger.info("ONNX Runtime: padding every batch to %d so the "
                        "graph sees one shape", self.pad_to)
        logger.info("ONNX Runtime: %s, %d intra-op threads, provider=%s",
                    path.name, self.threads, self.provider or "unknown")

    @property
    def backend(self) -> str:
        gpu = "CUDA" in (self.provider or "")
        return f"onnxruntime:{'gpu' if gpu else 'cpu'}/{self.threads}t"

    def __call__(self, nchw_uint8: np.ndarray) -> Dict[str, np.ndarray]:
        """One forward pass. ``nchw_uint8`` is ``[N, C, H, W]``.

        uint8 is handed straight to the graph, which casts internally,
        converting to float here would move four times the bytes for nothing.
        """
        arr = np.ascontiguousarray(nchw_uint8)
        real = arr.shape[0]
        if 0 < real < self.pad_to:
            # A dynamic batch axis is fast only at a STEADY size. ONNX Runtime
            # re-plans when the shape changes, and a live rig's batch size
            # varies with frame-arrival jitter, so the graph is re-planned
            # constantly. Measured on the DLC export with a dynamic axis:
            #
            #     steady 16            432 frames/s
            #     alternating 16,15,14 149
            #     alternating 16,8     117
            #     random 1..16          65
            #
            # A 16-box rig showed the end of that: 1.7 pose results per box
            # per second, against 24.7 at eight boxes where the batch happened
            # to be stable. Padding costs inference on frames nobody wanted;
            # re-planning costs six times more.
            #
            # The pad repeats the last frame rather than using zeros: a black
            # frame can produce a confident peak on some networks, and if the
            # slice below were ever wrong it is better to duplicate a real
            # answer than to invent one.
            pad = np.repeat(arr[-1:], self.pad_to - real, axis=0)
            arr = np.ascontiguousarray(np.concatenate([arr, pad], axis=0))
        # Keyed by the GRAPH's own output names, not by position. This
        # unpacked exactly two, the SLEAP export's `peaks`/`peak_vals`, and
        # a DeepLabCut export has one output called `poses`, so driving a
        # perfectly good DLC graph died on "not enough values to unpack
        # (expected 2, got 1)" with nothing to say which graph it meant.
        values = self._sess.run(None, {self._input: arr})
        out = {name: np.asarray(v) for name, v in zip(self._outputs, values)}
        if arr.shape[0] != real:
            # Give back only what was asked for. Every output of these graphs
            # is batch-major, so the slice is the same for all of them.
            out = {name: v[:real] if getattr(v, "ndim", 0) >= 1 and
                   v.shape[0] == arr.shape[0] else v
                   for name, v in out.items()}
        return out

    def close(self) -> None:
        self._sess = None


def to_nchw(frame: np.ndarray) -> np.ndarray:
    """``HWC`` (or ``HW``) uint8 → contiguous ``[1, C, H, W]``.

    ``np.ascontiguousarray`` rather than a reversed view: a negative-stride
    array is rejected outright by torch and copied silently by others, and the
    copy is the cheaper surprise to make explicit.

    No colour conversion happens here. The pipeline hands pose RGB already, and
    a second conversion is precisely the bug that cost DLC 48% of its accuracy.
    """
    if frame.ndim == 2:
        frame = frame[:, :, None]
    return np.ascontiguousarray(frame.transpose(2, 0, 1)[None])


def peaks_to_pose(peaks: np.ndarray, peak_vals: np.ndarray,
                  body_parts) -> Dict[str, Optional[list]]:
    """Graph output → the pipeline's ``{part: [x, y, conf]}`` contract.

    Confidence is carried through rather than gated: the host-side threshold is
    the operator's live control, and applying it twice in two places is how the
    two get to disagree.
    """
    pk = np.asarray(peaks).reshape(-1, 2)
    pv = np.asarray(peak_vals).reshape(-1)
    out: Dict[str, Optional[list]] = {}
    for i, name in enumerate(body_parts):
        if i >= len(pk):
            out[name] = None
            continue
        x, y = float(pk[i][0]), float(pk[i][1])
        conf = float(pv[i]) if i < len(pv) else 0.0
        out[name] = None if not (np.isfinite(x) and np.isfinite(y)) else [x, y, conf]
    return out


def batch_to_poses(peaks: np.ndarray, peak_vals: np.ndarray,
                   body_parts) -> list:
    """The same decode for a batch, one pose dict per frame, in order."""
    pk = np.asarray(peaks)
    pv = np.asarray(peak_vals)
    if pk.ndim < 3:
        return [peaks_to_pose(pk, pv, body_parts)]
    return [peaks_to_pose(pk[i], pv[i], body_parts) for i in range(pk.shape[0])]


__all__ = ["ORTRunner", "batch_to_poses", "enable_cuda_libraries",
           "graph_is_decoded", "peaks_to_pose", "to_nchw"]
