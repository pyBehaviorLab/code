"""Drive a TensorRT engine directly, engine in, keypoints out.

sleap-nn compiles peak decoding into the exported graph: it returns ``peaks``
and ``peak_vals`` rather than heatmaps. That is what makes this possible at
all, because it means inference needs no sleap-nn code, and the published gap
between doing it this way and going through the library on the same weights is
1.09 ms against 4.79 ms.

Device memory comes from torch tensors, so there is no pycuda dependency,
torch is already present for the native path.

Four things bite here, and each one is a silent failure rather than a crash:

* **Output shapes are unknown until the input shape is set.** With a dynamic
  profile ``get_tensor_shape`` returns -1s beforehand.
* **The output buffers must be kept alive.** ``set_tensor_address`` takes a raw
  pointer; if the tensor is garbage-collected, TensorRT writes into freed
  memory.
* **Execution is asynchronous.** Without ``torch.cuda.synchronize()`` you are
  timing queue submission, and the model looks impossibly fast.
* **The engine takes uint8** and casts internally. Pre-converting to float
  moves four times the bytes across PCIe for nothing.
"""
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def engine_path(export_dir: str) -> Optional[Path]:
    """The ``.trt`` in an export directory, or ``None``."""
    p = Path(export_dir)
    if p.is_file() and p.suffix == ".trt":
        return p
    candidate = p / "model.trt"
    return candidate if candidate.exists() else None


class TRTRunner:
    """Callable: NCHW uint8 in, ``{"peaks", "peak_vals"}`` out.

    Same signature as the ONNX Runtime runner, so whatever drives it does not
    know which one it holds.
    """

    def __init__(self, engine_file: str, device: str = "cuda"):
        import tensorrt as trt
        import torch

        self._torch = torch
        self._trt = trt
        self.device = device
        blob = Path(engine_file).read_bytes()
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            # Engines are built for one GPU architecture, TensorRT version and
            # driver. A silent None here is how a stale cache becomes a
            # mysterious fallback to a slower runtime.
            raise RuntimeError(
                f"TensorRT could not deserialise {engine_file}; it was most "
                "likely built for a different GPU, driver or TensorRT version. "
                "Re-export on this machine.")
        self.ctx = self.engine.create_execution_context()
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)
        #: Output buffers are held HERE for the lifetime of the runner, not as
        #: locals: TensorRT was handed their raw addresses.
        self._buffers: Dict[str, object] = {}
        logger.info("TensorRT engine loaded: in=%s out=%s",
                    self.inputs, self.outputs)

    @property
    def backend(self) -> str:
        return "tensorrt:direct"

    def _dtype(self, name):
        return self._trt.nptype(self.engine.get_tensor_dtype(name))

    def __call__(self, nchw_uint8: np.ndarray) -> Dict[str, np.ndarray]:
        torch = self._torch
        name = self.inputs[0]
        x = torch.from_numpy(np.ascontiguousarray(nchw_uint8)).to(self.device)
        want = torch.from_numpy(np.empty(0, dtype=self._dtype(name))).dtype
        if x.dtype != want:
            x = x.to(want)
        # Before the output shapes can be read, and once per call because the
        # profile is dynamic.
        self.ctx.set_input_shape(name, tuple(x.shape))
        self.ctx.set_tensor_address(name, x.data_ptr())

        out = {}
        for oname in self.outputs:
            shape = tuple(self.ctx.get_tensor_shape(oname))
            dtype = torch.from_numpy(np.empty(0, dtype=self._dtype(oname))).dtype
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            self._buffers[oname] = buf          # keep alive; TRT has the pointer
            self.ctx.set_tensor_address(oname, buf.data_ptr())
            out[oname] = buf

        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        # Not optional: without it the copies below race the kernel, and any
        # timing around this call measures submission rather than work.
        torch.cuda.synchronize()
        return {k: v.cpu().numpy() for k, v in out.items()}

    def close(self) -> None:
        self._buffers.clear()
        self.ctx = None
        self.engine = None


def usable(export_dir: str) -> bool:
    """Whether this export can be driven directly on this machine.

    Both halves have to hold: an engine file must exist, and TensorRT must be
    importable and of a version that can read it. Anything less falls back to
    the library path rather than failing the run.
    """
    if engine_path(export_dir) is None:
        return False
    from source.video.tracking.capability import capabilities
    return capabilities().can("tensorrt")


__all__ = ["TRTRunner", "engine_path", "usable"]
