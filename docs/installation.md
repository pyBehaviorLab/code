# Installation

There are three things to install and they are independent. The **application**
is a Python package. The **microcontroller** takes no Python packages at all:
MicroPython is flashed once, and everything else is uploaded to it from the
running application. **Pose estimation** is optional, and is the only part with
version constraints.

Start with the application. Add pose estimation only if an experiment needs it.

## Install the application

Python 3.11 is recommended. Pin NumPy below version 2 first, because the pose
packages require it and installing it later forces a downgrade.

::::{tab-set}

:::{tab-item} Windows
:sync: win

```powershell
conda create -y -n pybehaviorlab python=3.11
conda activate pybehaviorlab

pip install "numpy<2"
git clone https://github.com/pyBehaviorLab/code
cd code
pip install -e .
```
:::

:::{tab-item} Linux
:sync: linux

```bash
conda create -y -n pybehaviorlab python=3.11
conda activate pybehaviorlab

pip install "numpy<2"
git clone https://github.com/pyBehaviorLab/code
cd code
pip install -e .

sudo apt install ffmpeg build-essential      # video encoding + Cython
sudo usermod -a -G dialout $USER             # serial access; log out and back in
```
:::

:::{tab-item} NVIDIA Jetson
:sync: jetson

JetPack 6 ships Python 3.10, and that is the version to use.

```bash
python3.10 -m venv ~/pybehaviorlab-env
~/pybehaviorlab-env/bin/pip install --upgrade pip setuptools wheel

git clone https://github.com/pyBehaviorLab/code
cd code
sudo usermod -a -G dialout $USER
```

**Do not pass `--system-site-packages`.** The system Python carries NumPy 1.21
and an OpenCV 4.8 build, and both would shadow the versions the next steps
install. TensorRT is the one system package that is needed, and it is linked in
explicitly rather than inherited.

The Jetson needs pinned versions and wheels fetched by URL rather than by name,
for reasons that are not guessable from the error messages. Follow
[Install on NVIDIA Jetson](installation-jetson.md) from here; it is the
procedure that was actually run on the device.
:::

::::

### Check it worked

```bash
python pyOperant.py     # operant chambers
python pyMaze.py       # maze arenas
```

The window should open and the log panel should report a version. That is the
whole application: interface, camera acquisition, recording and the serial link.
None of it needs a GPU.

:::{note}
FFmpeg must be on `PATH` for recording. On Windows use
`winget install Gyan.FFmpeg`. Hardware encoding is used when the build supports
it and falls back to CPU otherwise. A C compiler is optional: the Cython
extensions build at first launch when one is present, and the application uses
pure Python when it is not.
:::

## Prepare the microcontroller

No Python packages are installed for this step.

1. **Flash MicroPython** for your Nucleo variant, over DFU or ST-Link.
2. **Connect the board** over USB. It appears as a serial port.
3. In the application, open the board dialog and press **Load Framework**, then
   **Load Hardware Definition**.

The framework stays on the board until its version changes. Tasks are uploaded
per session. Hardware definitions and task files from a standard pyControl rig
run unchanged. See [Boards](user-guide/boards.md) and
[The V2 Nucleo breakout](hardware/breakout.md).

## Add pose estimation

Only needed when a model has to run on the video. Sessions record, and the
controller writes its data, with none of this installed.

::::{tab-set}

:::{tab-item} NVIDIA GPU
:sync: win

Install PyTorch **first**, from the PyTorch index. On Windows the
`deeplabcut-live[pytorch]` extra pulls a CPU-only build, and inference then runs
on the CPU with no error to say why.

```bash
conda activate pybehaviorlab

# 1. PyTorch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. runtimes
pip install "onnxruntime-gpu==1.22.0" "tensorrt-cu12==10.16.1.11"

# 3. the toolkits
pip install "deeplabcut-live==1.1.0" "sleap-nn==0.3.3"

# 4. exactly one OpenCV (step 3 asks for two different ones)
pip uninstall -y opencv-python opencv-python-headless
pip install "opencv-python==4.11.0.86"

# 5. restore NumPy last
pip install "numpy==1.26.4"
```

Match the CUDA suffix to your driver: `cu128`, `cu126` or `cu118`. Check with
`nvidia-smi`.
:::

:::{tab-item} CPU only
:sync: cpu

Models exported to ONNX need no framework at all:

```bash
conda activate pybehaviorlab
pip install onnxruntime
```

To run SLEAP or DeepLabCut model files directly, add the CPU builds:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "deeplabcut-live==1.1.0" "sleap-nn==0.3.3"
pip uninstall -y opencv-python opencv-python-headless
pip install "opencv-python==4.11.0.86"
pip install "numpy==1.26.4"
```

Set the inference thread count explicitly. On a running rig those cores are also
acquiring frames and encoding video.
:::

:::{tab-item} NVIDIA Jetson
:sync: jetson

Install the JetPack builds. Generic pip CUDA wheels are built for x86-64 and
will not run on ARM64.

```bash
JP=https://pypi.jetson-ai-lab.io/jp6/cu126
P=~/pybehaviorlab-env/bin/pip

$P install "numpy==1.26.4"
$P install \
  "$JP/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl" \
  "$JP/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl"
$P install onnxruntime-gpu --index-url "$JP/+simple"
```

**Install torch by its wheel URL, not by name.** PyPI also publishes ARM builds,
and pip prefers their more specific `manylinux` tag over NVIDIA's
`linux_aarch64` wheel even when NVIDIA's index is given. Asking for
`torch==2.8.0` by name silently installed `2.8.0+cpu`; asking for `2.11.0` gave
a CUDA 13 build that fails with "The NVIDIA driver on your system is too old".
Neither says anything about the index it came from.

TensorRT's Python bindings come from JetPack (`python3-libnvinfer`) and are
symlinked into the environment, **not** installed with pip: the pip package is
TensorRT 11, which dropped the `EXPLICIT_BATCH` API that sleap-nn's export uses.
`sleap-nn` also needs `--ignore-requires-python`, and applying that to the whole
install rather than to sleap-nn alone lets pip pick pandas 3.0 and abort.

[Install on NVIDIA Jetson](installation-jetson.md) carries those steps, the
version pins and their reasons, and the on-device verification.

Build TensorRT engines on the device. An engine is specific to the GPU that built
it, and the device is part of the export cache key, so an engine built on a
workstation is never reused here. Keep the pose batch size small.
:::

::::

`pip check` will report that `deeplabcut-live` wants `opencv-python-headless`.
Ignore it. `opencv-python` provides the same `cv2` module, and installing both is
what breaks the environment.

**One environment runs both SLEAP and DeepLabCut.** Installed from empty on
Windows 11 with Python 3.11 and confirmed working: `torch` 2.11.0+cu128,
`numpy` 1.26.4, `sleap-nn` 0.3.3, `deeplabcut-live` 1.1.0 and `opencv-python`
4.11.0.86, with `torch.cuda.is_available()` true and both backends importing in
the same interpreter.

## Legacy TensorFlow models

Skip this unless you have a DeepLabCut 2.x model stored as a TensorFlow snapshot
(`pose_cfg.yaml` plus `.pb`) that is too slow on the CPU. ONNX exports and
DeepLabCut 3.x PyTorch snapshots do not need TensorFlow.

This is a **second environment**. It cannot run `sleap-nn`, because TensorFlow
2.10 caps Python at 3.10 while `sleap-nn` requires 3.11 or later.

```bash
conda create -y -n pybehaviorlab-tf python=3.10
conda activate pybehaviorlab-tf
pip install "tensorflow==2.10" "deeplabcut-live==1.1.0"
conda install -y cudatoolkit=11.2 cudnn=8
pip install "numpy<2" "protobuf<3.21"

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

An empty list means TensorFlow is running on the CPU.

:::{warning}
TensorFlow 2.10 was the last release with GPU support on native Windows. From
2.11 the Windows wheel contains no CUDA at all and GPU use needs WSL2. On Linux
this limit does not apply.

Activate this environment with `conda activate`, not by calling its `python.exe`
by full path. The CUDA libraries live in the environment's `Library\bin`, which
is only put on `PATH` by activation.
:::

## Verify

The repository carries its own check. It does not test whether packages import.
It asks each runtime to execute, reports the device it ran on, then loads every
model it can find and puts one frame through it.

```bash
python tools/verify_env.py
```

```text
── runtimes ─────────────────────────────────────────────────
  [ok  ] PyTorch          2.11.0+cu128  cuda=12.8  NVIDIA RTX A6000
  [ok  ] ONNX Runtime     1.22.0  providers=['TensorrtExecutionProvider',
                                             'CUDAExecutionProvider',
                                             'CPUExecutionProvider']

── pose SDKs ────────────────────────────────────────────────
  [ok  ] DeepLabCut-Live  1.1.0  backends=['tensorflow', 'pytorch']
  [ok  ] sleap-nn         0.3.3

── real inference ───────────────────────────────────────────
  [ok  ] dlc_6point_v2    ONNX export: 6 parts, 6 located, 0.5s to load
```

Read two lines in particular. **PyTorch** must name the GPU, not `CPU only`.
**ONNX Runtime** must list `CUDAExecutionProvider`. If it lists only
`CPUExecutionProvider`, the CUDA build does not match the installed CUDA and
inference will be roughly ten times slower with no error raised.

The test suite is the second check and needs no hardware:

```bash
python -m pytest
```

## Record what you installed

Pose results depend on the versions that produced them, so the environment is
part of the method. Record it when the experiment starts, not afterwards:

```bash
conda env export --no-builds > environment.yml
pip freeze > requirements-frozen.txt
```

Keep both files with the data. The application separately writes the task and
hardware-definition hashes into every session file, so the controller side
identifies itself. These two files do the same for the host side. See
[Run lineage](concepts/run-lineage.md).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Inference runs, but slowly, with no error | `onnxruntime-gpu` built against a different CUDA major version, silently on the CPU | Check `providers=` in `verify_env.py`; install the build matching your CUDA |
| `torch.cuda.is_available()` is `False` on Windows | A CPU-only wheel arrived as a dependency of `deeplabcut-live[pytorch]` | Reinstall from the PyTorch index **before** the toolkits |
| `import cv2` reports an unexpected version | Both `opencv-python` and `opencv-python-headless` are installed; the last one written wins | Uninstall both, install exactly one |
| `deeplabcut-live` fails after a TensorFlow install | NumPy was raised to 2.x by `ml-dtypes` | `pip install "ml-dtypes<0.6" "numpy==1.26.4"` |
| TensorFlow reports no GPU where it worked before | The interpreter was called by full path, so `Library\bin` was never put on `PATH` | `conda activate` the environment first |
| A `.trt` engine will not load | TensorRT engines are compiled for one GPU architecture | Rebuild on this machine from the ONNX file |
| The board does not appear in the port list | Missing USB serial driver, or a cable without data lines | Install the virtual-COM-port driver; confirm the port exists in the operating system first |
| Recording falls back to CPU encoding | FFmpeg on `PATH` was built without the hardware encoder | Install a full FFmpeg build. On a Jetson, build the one in [Install on NVIDIA Jetson](installation-jetson.md#10-build-ffmpeg-with-the-hardware-encoder) |

More symptoms: [Troubleshooting](troubleshooting.md).

---

## Appendix: why the versions are pinned

Not needed in order to install. This is the reasoning behind the pins above.
Each item was reproduced on a test machine rather than taken from release notes.

### Python must be 3.11 or 3.12

The usable range is the intersection of what the packages declare:

| Package | Declared `requires-python` |
|---|---|
| pyBehaviorLab | `>=3.9` |
| `sleap-nn` 0.3.3 | `>=3.11,<3.14` |
| `deeplabcut-live` 1.1.0 | `>=3.10,<3.13` |
| `tensorflow` 2.10 (last with GPU on native Windows) | `<=3.10` |

Python 3.10 excludes `sleap-nn`. Python 3.13 excludes `deeplabcut-live`.
TensorFlow 2.10 excludes `sleap-nn` outright, which is why GPU TensorFlow needs
its own environment.

### NumPy 2 against NumPy 1

`deeplabcut-live` requires `numpy<2`. Nothing else constrains NumPy, so a plain
install selects NumPy 2.x, and adding `deeplabcut-live` afterwards downgrades it.
Anything built against NumPy 2 is then left unsatisfied. Observed: a clean core
install selected NumPy 2.4.6 and `opencv-python` 5.0.0.93, and the later
downgrade left that OpenCV requiring a NumPy it no longer had.

### Two OpenCV distributions

`sleap-nn` requires `opencv-python`. `deeplabcut-live` requires
`opencv-python-headless`. Both install a module named `cv2` to the same path, so
pip writes one over the other and nothing reports it. Observed with both present:

```text
opencv-python           5.0.0.93
opencv-python-headless  4.11.0.86
import cv2 -> 4.11.0        # the headless build won, despite being older
```

The version that answers `import cv2` is whichever was unpacked last, not the
newest. The application draws through Qt rather than `cv2.imshow`, so either
distribution works. Only one may be installed.

### cuDNN needs no separate install

ONNX Runtime loads `cudnn64_9.dll` by name through the system loader, and the
PyTorch CUDA wheel already ships it in `torch/lib`. The application adds that
directory to the DLL search path before building a session. Without it, ONNX
Runtime reports that it could not create the CUDA provider and falls back to the
CPU.

### The two pinned runtimes

`onnxruntime-gpu` 1.22 is the CUDA-12 line; later releases require CUDA 13.
`tensorrt-cu12` is pinned to the version the existing engine files were built
with.
