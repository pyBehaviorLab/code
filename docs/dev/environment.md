# The `pybehlab_full` environment

One environment in which **all four inference runtimes work**. PyTorch,
TensorFlow, ONNX Runtime and TensorRT, so that every model format in
`models/` can be run from either pipeline without switching environments.

Built and verified on 2026-08-24. Windows 11, NVIDIA RTX A6000 (compute 8.6),
driver 595.95.

```
conda activate pybehlab_full
```

Everything below was measured on this machine, not taken from documentation.

---

## What is installed

| | version | device |
|---|---|---|
| Python | 3.11.15 | |
| PyTorch | 2.11.0+cu128 (CUDA 12.8) | **GPU** |
| TensorFlow | 2.18.1 | **CPU in this env**. GPU lives in `dlc-tf210`, see below |
| ONNX Runtime | onnxruntime-gpu 1.22.0 | **GPU** (CUDA + TensorRT EPs) |
| TensorRT | tensorrt-cu12 10.16.1.11 | **GPU** |
| DeepLabCut-Live | 1.1.0, backends `['tensorflow', 'pytorch']` | |
| sleap-nn / sleap-io | 0.3.3 / 0.9.2 | |
| numpy | 1.26.4 | |
| OpenCV | 4.11.0.86 | |
| PySide6 | 6.10.3 | |

`pip check` reports **no broken requirements**. The table above is the part
worth pinning by hand; the rest of the environment is whatever pip resolved.

A full freeze used to be committed here as a text file. It went stale the first
time anyone upgraded a package, and nothing checked it, so a reader could not
tell whether they were looking at the environment or at its ghost. Generate a
current one instead:

```bash
conda activate pybehlab
pip freeze > freeze.txt        # ~230 packages
```

---

## The three constraints that shaped this

### 1. TensorFlow on the GPU and the rest of this stack are mutually exclusive

TensorFlow **can** use the GPU on native Windows, and on this machine it does
- in the separate `dlc-tf210` environment:

```
dlc-tf210:  Python 3.10.18, TF 2.10.0, CUDA 11.2 / cuDNN 8
            tf.config.list_physical_devices('GPU') -> [PhysicalDevice(GPU:0)]
```

That works because conda puts `cudart64_110.dll` and `cudnn64_8.dll` in the
environment's `Library\bin`, which is on `PATH` only while the environment is
activated. (Running its `python.exe` by full path without activating reports
no GPU, the DLLs are never found. Worth knowing before concluding a TF
install is broken.)

What is *not* possible is having that in **this** environment:

* TF 2.10 is the last native-Windows GPU build, and it requires Python < 3.11;
  `sleap-nn` requires >= 3.11.
* TF 2.10 needs CUDA 11.2; PyTorch and ONNX Runtime here are CUDA 12.8, and
  two CUDA major versions cannot share a process.
* From TF 2.11, the Windows wheel has no CUDA in it at all, confirmed here,
  `tf.test.is_built_with_cuda()` is `False` for the 2.18.1 installed below.

**Decision: TensorFlow on the CPU *in this environment*, and `dlc-tf210` when
a TF model needs the GPU.** Dropping the whole stack to CUDA 11 for TF's sake
would cost GPU acceleration on SLEAP and the DLC ONNX path, which are the
runtimes carrying the throughput. TF is needed here only for the two legacy
DLC checkpoints, and it runs them.

### 2. sleap-nn and DLC's TensorFlow extra want different Pythons

```
sleap-nn 0.3.3           Requires-Python  >=3.11,<3.14
deeplabcut-live[tf]      Windows: python_version < "3.11", tensorflow<=2.10
```

Mutually exclusive, and this is why a naive "install everything" fails.

**Resolved by not needing the extra.** `deeplabcut-live` declares that pin for
its own TF *extra*; installing `tensorflow` separately alongside plain
`deeplabcut-live` gives its TensorFlow runner what it needs. Verified: dlclive
reports `backends=['tensorflow', 'pytorch']` on Python 3.11, and both legacy
checkpoints load and infer.

### 3. numpy is pinned from three directions

```
deeplabcut-live   numpy<2,>=1.20      <- the binding upper bound
tensorflow 2.18   numpy<2.2.0,>=1.26.0
ml-dtypes 0.6     numpy>=2.0.0        <- conflict, pulled in by keras 3.15
```

Installing TensorFlow 2.19 drags in `ml-dtypes 0.6`, which forces numpy 2 and
breaks DeepLabCut-Live. Pinning **`ml-dtypes<0.6`** (resolves to 0.5.4) keeps
numpy at 1.26.4 and satisfies all three. TensorFlow still imports and runs.

---

## Verified end to end

Every model in `models/`, loaded and given a frame through the offline seam:

```
[ok] sleap: sleap_6point_v4          ONNX export     6 parts   1.1 s to load
[ok] dlc:   dlc_6point_v2            ONNX export     6 parts   0.9 s to load
[ok] dlc:   Multimaze_mobilenet_v2   TF checkpoint   3 parts   2.2 s to load
[ok] dlc:   SuperAnimal-TopViewMouse TF checkpoint  27 parts   4.3 s to load
```

The two TensorFlow checkpoints could not run at all before this environment
existed. The project's own probes agree:

```
detectors available   dlc, sleap
capability probe      cuda=True  tensorrt=True  onnx=True  gpu=NVIDIA RTX A6000
```

Test suite: **2,210 passed**, 24 skipped, 6 failed, the same six pre-existing
failures this environment inherited, unrelated to inference
(`retake_bg_btn` missing in `tracking_panel`, camera-id refresh, trigger
semantics).

---

## Known issue: the `.trt` engines do not load here

```
[TRT] [E] IRuntime::deserializeCudaEngine: Error Code 6:
    The engine plan file is generated on an incompatible device,
    expecting compute 8.6 got compute 8.9, please rebuild.
```

A TensorRT engine is compiled for one GPU architecture. `model.trt` and
`dlc_pose.trt` were built on a **compute 8.9 (Ada)** card; this machine is
**compute 8.6 (Ampere, RTX A6000)**. The model metadata still points at
`D:\Autopose\..`, a path that does not exist here, so these models were
exported elsewhere and copied in.

This is not a fault and nothing is lost: the loader falls through to the ONNX
export in the same folder, which is the same weights and the same coordinates.
To use TensorRT, rebuild the engines **on this machine** from the ONNX files.

TensorRT itself is installed and working - `capability probe` reports
`tensorrt=True`, and ONNX Runtime lists `TensorrtExecutionProvider`.

---

## Rebuilding this environment

```bash
conda create -y -n pybehlab_full python=3.11
conda activate pybehlab_full

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "onnxruntime-gpu==1.22.0" "tensorrt-cu12==10.16.1.11"
pip install "tensorflow==2.18.1"
pip install "deeplabcut-live==1.1.0" "sleap-nn==0.3.3"
pip install "ml-dtypes<0.6" "numpy==1.26.4"     # AFTER tensorflow, see note 3
pip install "PySide6>=6.6,<6.11" pyserial pyqtgraph openpyxl orjson
pip install pytest pytest-qt pytest-timeout ruff cython
```

Order matters in two places. PyTorch goes first so the CUDA 12.8 wheel is not
later replaced by a CPU build pulled in as someone's dependency. The
`ml-dtypes` / `numpy` pin goes **last**, because installing TensorFlow raises
numpy to 2.x and only a later explicit pin brings it back.

### Two version pins that are not arbitrary

* **`onnxruntime-gpu==1.22.0`**, 1.29 requires CUDA 13 and silently falls
  back to the CPU provider against this CUDA 12.8 install. 1.22 is the
  CUDA-12 line.
* **`tensorrt-cu12==10.16.1.11`**, the version the existing `.trt` artifacts
  were built with, so a rebuilt engine stays interchangeable with them.

### cuDNN

ONNX Runtime loads `cudnn64_9.dll` by name through the Windows loader, which
searches `PATH` and not site-packages. The torch CUDA wheel already ships it
in `torch/lib`, so no separate cuDNN install is needed
`ort_runner.enable_cuda_libraries()` adds that directory to the DLL search
path before the session is built. Without it ORT reports
"Failed to create CUDAExecutionProvider" and runs on the CPU at roughly a
tenth of the speed.

---

## The other environment

`pybehlab` is unchanged and still works. It has no TensorFlow and no
TensorRT, so the two legacy DLC checkpoints cannot run there. Use
`pybehlab_full` for anything that needs them.

`dlc-tf210` is the TensorFlow-on-GPU environment (Python 3.10, TF 2.10,
CUDA 11.2). It is the right place to run or re-export a TF DLC checkpoint
with the GPU; it cannot run `sleap-nn`, which needs Python 3.11+. Activate it
properly - `conda activate dlc-tf210`, rather than calling its `python.exe`
by path, or its CUDA DLLs will not be on `PATH` and TF will report no GPU.
