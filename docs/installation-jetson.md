# Install on NVIDIA Jetson

This page installs the application and the full pose stack on a Jetson in a
single environment: SLEAP-NN, DeepLabCut-Live, CUDA PyTorch, ONNX Runtime and
TensorRT. It also builds FFmpeg with the Jetson's hardware H.264 encoder for
recording. The generic [Installation](installation.md) page is enough for a rig
without pose tracking.

Every step below was run and verified on the machine listed in
[Tested configuration](#tested-configuration). Several steps look unusual. Each
exists because the obvious version failed, and the reason is given with the
step. Skip one and the environment usually still installs, then runs inference
on the CPU with no error to say why.

## Before you start

### What this procedure assumes

| Item | Required | Check with |
|---|---|---|
| Board | Jetson AGX Orin (Orin NX / Nano are expected to work, not tested) | `cat /proc/device-tree/model` |
| JetPack | 6.2.x (L4T R36.x) | `dpkg -l nvidia-jetpack` |
| CUDA | 12.6 | `ls /usr/local \| grep cuda` |
| TensorRT | 10.3, from JetPack | `dpkg -l python3-libnvinfer` |
| Python | 3.10, the JetPack system Python | `python3.10 --version` |
| Free disk | 15 GB for the environment and pip cache | `df -h` |

:::{important}
**Use Python 3.10, even though sleap-nn declares 3.11 or newer.**
NVIDIA publishes CUDA builds of PyTorch, ONNX Runtime and TensorRT for JetPack
6 only for Python 3.10. There are no 3.11 builds on any Jetson index. A 3.11
environment installs cleanly and runs everything on the CPU. sleap-nn is pure
Python and runs correctly on 3.10: its keypoints match the reference machine to
0.000 px ([Verify](#verify)).
:::

### System packages

These are normally present on a JetPack install. Installing them again is
harmless.

```bash
sudo apt update
sudo apt install -y python3.10-venv python3.10-dev build-essential ffmpeg \
                    python3-libnvinfer python3-libnvinfer-dev
```

Give every user access to the boards' serial ports, once. Linux lets only
`root` and the `dialout` group open `/dev/ttyACM*`, and a `sudo chmod 666` on
the ports lasts only until a board is replugged or the Jetson reboots. A udev
rule applies at every plug-in instead, and also stops ModemManager from probing
the boards while the application is trying to connect:

```bash
sudo tee /etc/udev/rules/49-pyboard.rules > /dev/null <<'EOF'
# MicroPython pyboard (any USB mode)
ATTRS{idVendor}=="f055", ENV{ID_MM_DEVICE_IGNORE}="1"
SUBSYSTEM=="tty", ATTRS{idVendor}=="f055", GROUP="dialout", MODE="0666"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty --action=change
ls -l /dev/ttyACM*
```

Every port should read `crw-rw-rw-`. The rule matches the MicroPython vendor ID
only, because a board's product ID changes after "Disable Flash Drive".

Boards are bound to boxes by their USB serial number, which is the same on
Windows and Linux, so a project saved on either system finds its boards on the
other at whatever `COMn` or `/dev/ttyACMn` they have.

### Put the environment on the NVMe

The Jetson's internal eMMC is small, and the environment plus its pip cache
come to about 9 GB. Keep the environment, the pip cache and pip's temporary
files on the NVMe. The paths below use `/ssd_apps`; substitute your own mount
point.

```bash
export VENV=/ssd_apps/pybehlab
export PIP_CACHE_DIR=/ssd_apps/pip-cache
export TMPDIR=/ssd_apps/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
```

:::{note}
These three variables are used by every step below. If you open a new
terminal part-way through, set them again.
:::

## Install

### 1. Create the environment

```bash
python3.10 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip setuptools wheel
```

Do not use `--system-site-packages`. The system Python carries NumPy 1.21 and an
OpenCV 4.8 build that would shadow the versions installed below. TensorRT, the
one system package that is needed, is linked in explicitly in step 4.

### 2. Pin the versions

The pins stop later installs from replacing what the earlier steps put in
place.

```bash
cat > "$VENV/constraints.txt" <<'EOF'
numpy==1.26.4
torch==2.8.0
torchvision==0.23.0
opencv-python==4.11.0.86
opencv-python-headless==4.11.0.86
PySide6==6.8.0.2
PySide6-Essentials==6.8.0.2
PySide6-Addons==6.8.0.2
shiboken6==6.8.0.2
tables==3.10.1
EOF
```

| Pin | Why |
|---|---|
| `numpy==1.26.4` | `deeplabcut-live` requires NumPy below 2 |
| `torch` / `torchvision` | The JetPack 6 CUDA 12.6 build, installed in step 3 |
| `PySide6==6.8.0.2` | From 6.8.1 the ARM wheels need glibc 2.39; JetPack 6 is Ubuntu 22.04 with glibc 2.35 |
| `tables==3.10.1` | The last release with a Python 3.10 ARM wheel |
| `opencv-*==4.11.0.86` | See step 7 |

### 3. Install CUDA PyTorch and ONNX Runtime

```bash
JP=https://pypi.jetson-ai-lab.io/jp6/cu126
P="$VENV/bin/pip"
C="$VENV/constraints.txt"

$P install -c "$C" "numpy==1.26.4"
$P install -c "$C" \
  "$JP/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl" \
  "$JP/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl"
$P install -c "$C" onnxruntime-gpu --index-url "$JP/+simple"
```

:::{warning}
**Install torch by its wheel URL, not by name.** PyPI also publishes ARM builds
of torch, and pip prefers their more specific `manylinux` tag over NVIDIA's
`linux_aarch64` wheel, even with NVIDIA's index given. Installing
`torch==2.8.0` by name from the NVIDIA index silently produced `2.8.0+cpu`.
Installing `torch==2.11.0` produced a CUDA 13 build, which fails with
"The NVIDIA driver on your system is too old".
:::

If the URLs have moved, list the current files and copy the `cp310` links for
`torch-2.8.0` and `torchvision-0.23.0`:

```bash
curl -s "$JP/+simple/torch/" | grep -o 'torch-2.8.0-cp310[^"#]*'
```

Check that PyTorch reaches the GPU before going on:

```bash
"$VENV/bin/python" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.8.0 12.6 True Orin`. Stop and redo this step if it says `+cpu`,
`None` or `False`.

Also check that pip did not add CUDA libraries of its own. JetPack's CUDA is the
one to use:

```bash
"$VENV/bin/pip" list | grep -iE "^(nvidia|cuda)-" || echo "none, correct"
```

### 4. Link the JetPack TensorRT into the environment

TensorRT's Python bindings come from JetPack (`python3-libnvinfer`), not pip.

```bash
SP="$VENV/lib/python3.10/site-packages"
for d in tensorrt tensorrt-10.3.0.dist-info tensorrt_dispatch tensorrt_dispatch-10.3.0.dist-info \
         tensorrt_lean tensorrt_lean-10.3.0.dist-info; do
  ln -sfn "/usr/lib/python3.10/dist-packages/$d" "$SP/$d"
done
"$VENV/bin/python" -c "import tensorrt; print(tensorrt.__version__)"
```

Expected: `10.3.0`. Do not `pip install tensorrt`. The pip package is TensorRT
11, which removed the `EXPLICIT_BATCH` API that sleap-nn's export uses, and its
ARM build targets server GPUs, not the Jetson.

### 5. Install sleap-nn

```bash
"$VENV/bin/pip" install --no-deps --ignore-requires-python "sleap-nn==0.3.3"
```

`--ignore-requires-python` must apply to **sleap-nn only**, which is why its
dependencies are installed separately in step 6. Given to a whole install, the
flag lets pip choose pandas 3.0, which needs Python 3.11, and the install
aborts.

### 6. Install sleap-nn's dependencies, DeepLabCut-Live and the application

```bash
"$VENV/bin/pip" install --prefer-binary -c "$VENV/constraints.txt" \
  "sleap-io>=0.9.2,<0.10.0" lightning "skia-python>=87.0,<=138.0" jsonpickle attrs \
  omegaconf wandb seaborn rich "textual>=0.40.0" loguru psutil scikit-image hydra-core \
  jupyter jupyterlab pyzmq "rich-click>=1.9.5" "pykalman>=0.11.0" "onnx>=1.15.0" \
  "deeplabcut-live[pytorch]==1.1.0" \
  PySide6==6.8.0.2 opencv-python==4.11.0.86 pyserial shapely pandas matplotlib pyqtgraph \
  scipy pyyaml openpyxl orjson pillow markdown-it-py scikit-learn \
  cython pytest pytest-qt pytest-timeout
```

The first two lines are sleap-nn 0.3.3's own dependency list. This step
downloads about 200 packages and takes roughly 10 minutes.

### 7. Keep exactly one OpenCV

sleap-nn asks for `opencv-python` and DeepLabCut-Live for
`opencv-python-headless`. Both install a `cv2` package to the same folder, and
the one unpacked last wins, whatever its version. Remove both and install one:

```bash
"$VENV/bin/pip" uninstall -y opencv-python opencv-python-headless
"$VENV/bin/pip" install --no-deps "opencv-python==4.11.0.86"
```

`pip check` will then report that `deeplabcut-live` requires
`opencv-python-headless`. This is expected; ignore it.

### 8. Build the Cython extensions

```bash
source "$VENV/bin/activate"
cd ~/Documents/pyBehaviorLab          # your clone
python setup.py build_ext --inplace
```

This builds `zone_math`, `_image_ops` and `drawing_ops` for ARM. The application
would otherwise build them on its first launch.

### 9. Record the environment

```bash
pip freeze > "$VENV/requirements-frozen.txt"
```

Keep this file with the data from experiments run in this environment. See
[Record what you installed](installation.md#record-what-you-installed).

### 10. Build FFmpeg with the hardware encoder

The application records through the FFmpeg it finds on `PATH`, and uses a
hardware H.264 encoder when that FFmpeg has one that works. JetPack's FFmpeg
lists `h264_v4l2m2m`, but on JetPack 6 that encoder cannot find the Orin's
("Could not find a valid device"), so every box records with libx264 on the
CPU. This step builds FFmpeg with `h264_nvmpi` from
[jetson-ffmpeg](https://github.com/Keylost/jetson-ffmpeg), which drives the
encoder through NVIDIA's Multimedia API.

It installs to a folder of its own and leaves the system FFmpeg untouched. Only
the `pybehlab` environment uses it. It takes about 3 minutes to compile and
360 MB of disk.

:::{note}
The AGX Orin and Orin NX have a hardware video encoder. The Orin Nano does not:
skip this step there, and recording uses libx264.
:::

```bash
sudo apt install -y cmake git pkg-config nvidia-l4t-jetson-multimedia-api

PREFIX=/ssd_apps/ffmpeg-nvmpi
SRC=/ssd_apps/src
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/share/pkgconfig"
mkdir -p "$PREFIX" "$SRC" && cd "$SRC"

git clone --depth 1 -b stable https://code.videolan.org/videolan/x264.git
git clone https://github.com/Keylost/jetson-ffmpeg.git
git -C jetson-ffmpeg checkout 64df5aa
git clone --depth 1 -b n6.0.1 https://github.com/FFmpeg/FFmpeg.git ffmpeg-6.0

# libx264, so recording can still fall back to the CPU
(cd x264 && ./configure --prefix="$PREFIX" --enable-static --enable-pic --disable-cli \
   && make -j"$(nproc)" && make install)

# libnvmpi, the bridge to NVIDIA's Multimedia API
(cd jetson-ffmpeg && mkdir -p build && cd build \
   && cmake .. -DCMAKE_INSTALL_PREFIX="$PREFIX" \
               -DJETSON_MULTIMEDIA_LIB_DIR=/usr/lib/aarch64-linux-gnu/nvidia \
   && make -j"$(nproc)" && make install)

# FFmpeg with h264_nvmpi and libx264
(cd jetson-ffmpeg && ./ffpatch.sh "$SRC/ffmpeg-6.0")
cd ffmpeg-6.0
./configure --prefix="$PREFIX" \
  --enable-gpl --enable-nvmpi --enable-libx264 \
  --enable-static --disable-shared --disable-doc --disable-ffplay --disable-debug \
  --pkg-config-flags=--static \
  --extra-cflags="-I$PREFIX/include" \
  --extra-ldflags="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib -Wl,-rpath,/usr/lib/aarch64-linux-gnu/nvidia" \
  --extra-libs="-lpthread -lm -ldl"
make -j"$(nproc)" && make install
```

Link it into the environment, so the application finds it first on `PATH`
whenever the environment is active:

```bash
ln -s /ssd_apps/ffmpeg-nvmpi/bin/ffmpeg  "$VENV/bin/ffmpeg"
ln -s /ssd_apps/ffmpeg-nvmpi/bin/ffprobe "$VENV/bin/ffprobe"
source "$VENV/bin/activate"
which ffmpeg
ffmpeg -hide_banner -encoders | grep -E 'nvmpi|libx264'
```

Expected: `/ssd_apps/pybehlab/bin/ffmpeg`, then lines for `h264_nvmpi`,
`hevc_nvmpi`, `libx264` and `libx264rgb`. To go back to the system FFmpeg, remove
the two links.

Check that the encoder really encodes. It finishes in under a second; the
`NvMMLite` lines it prints come from NVIDIA's library and are normal:

```bash
timeout -s KILL 20 ffmpeg -hide_banner -v error -f lavfi -i testsrc2=size=320x240:rate=30 -t 3 \
  -c:v h264_nvmpi -b:v 5M -qmin 20 -qmax 51 -pix_fmt yuv420p -f null - && echo "encoder OK"
```

The application selects `h264_nvmpi` by itself when it finds this FFmpeg. It
also handles two limits of the Jetson encoder:

- **Frames under 160 px on a side** make the encoder hang, so a box that small
  records with libx264.
- **A very noisy frame** can make the encoder fail and stop encoding for the rest
  of the recording. The application sets a quantiser floor (`-qmin 20`), which
  prevents that. On real box video the quality matched libx264 at a similar file
  size ([Tested configuration](#tested-configuration)).

## Prepare the models

Skip this section if the rig only uses the exported ONNX files. Everything else
runs as shipped.

### TensorRT engines are built on the Jetson

The `.trt` files in `models/` were built on a desktop GPU. TensorRT refuses them
on a Jetson ("Platform specific tag mismatch detected"). The application then
uses an engine built on this machine, from its export cache in
`~/.config/pybehaviorlab/sleap_exports/`. Only when the cache has none does it
fall back to the model folder's ONNX file on the GPU, which tracks correctly
but about twice as slowly.

The application does not build that engine by itself when the model folder
already contains an ONNX export. Build it once per model, **for the box count
and precision you record with**. Both are fixed inside the engine, and an engine
built for a different box count or precision is not used.

| Recording setting | Engine it needs |
|---|---|
| Number of boxes | Rounded up to 1, 2, 4, 8, 16 or 32; a 3-box rig uses the 4-box engine |
| **fp16** in the tracking panel | Ticked: fp16 engine. Unticked: fp32 engine |

Build it from the tracking panel, or from the command line without changing the
model folder (see below). Each engine takes 1 to 3 minutes. On the tested Orin,
at 4 boxes, the fp16 engine infers in 1.71 ms per frame against 3.78 ms for the
ONNX file ([Tested configuration](#tested-configuration)).

**From the tracking panel.** Set up the boxes as they will be recorded, choose
Runtime `tensorrt`, set **fp16** as you will record, and press **Export**.
Runtime `auto` uses the engine too. The export only succeeds when two things are
true of the model folder.

**The checkpoint is named `best.ckpt`.** `sleap-nn export` looks for that name
only. Where a folder ships the checkpoint under another name, add a link:

```bash
cd models/sleap_6point_v4
ln -s dklab_v4.ckpt best.ckpt
```

**`training_config.yaml` points at no pretrained weights on another machine.**
A model fine-tuned from an earlier one records that model's path, and the
exporter tries to load it:

```yaml
  pretrained_backbone_weights: D:/Autopose/sleap_v3/models/dklab_v3/best.ckpt
  pretrained_head_weights: D:/Autopose/sleap_v3/models/dklab_v3/best.ckpt
```

On the Jetson set both to `null`. The trained weights are already inside the
checkpoint, so nothing is lost:

```yaml
  pretrained_backbone_weights: null
  pretrained_head_weights: null
```

:::{note}
Changing `training_config.yaml` changes the export cache key, so any engine
already cached for that model is rebuilt once.
:::

**From the command line, without changing the model folder.** Use this when
`models/` is shared with other machines, or when you would rather not edit it.
It exports from a temporary copy of the folder: the checkpoint is linked as
`best.ckpt` and the pretrained-weight paths are set to `null`. The engine goes
into the cache folder the application looks up for the real model folder. It is
for single-instance models; export top-down and bottom-up models from the
tracking panel.

Set the four variables on the `MODEL=` line. `MODEL` must be spelled exactly as
in the project's tracking settings (for example `models/sleap_6point_v8`),
because the path is part of the cache key.

```bash
source /ssd_apps/pybehlab/bin/activate
cd ~/Documents/pyBehaviorLab
MODEL=models/sleap_6point_v8 CKPT=dklab_v8.ckpt BOXES=4 PREC=fp16 python - <<'EOF'
import os, re, subprocess, tempfile
from pathlib import Path
from source.video.tracking.sleap_export import (
    export_cache_dir, batch_bucket, _cli, PERMISSIVE_PEAK_THRESHOLD, ONNX_OPSET)

model, prec = os.environ["MODEL"], os.environ["PREC"]
batch = batch_bucket(int(os.environ["BOXES"]))
src = Path(tempfile.mkdtemp())
cfg = Path(model, "training_config.yaml").read_text()
cfg = re.sub(r"(pretrained_(backbone|head)_weights:).*", r"\1 null", cfg)
(src / "training_config.yaml").write_text(cfg)
(src / "best.ckpt").symlink_to(Path(model, os.environ["CKPT"]).resolve())

out = export_cache_dir(model, "tensorrt", "cuda", None, prec, batch,
                       PERMISSIVE_PEAK_THRESHOLD)
out.mkdir(parents=True, exist_ok=True)
subprocess.run([*_cli(), "export", str(src), "-o", str(out), "-f", "tensorrt",
                "--precision", prec, "--max-batch-size", str(batch),
                "--peak-threshold", str(PERMISSIVE_PEAK_THRESHOLD),
                "--opset-version", str(ONNX_OPSET), "--device", "cuda"],
               check=True)
print("engine:", out / "model.trt")
EOF
```

Run it once per precision you record with. The last line prints the engine's
path, for example
`~/.config/pybehaviorlab/sleap_exports/f775931f-cuda-tensorrt-fp16-b4-p001/model.trt`.

### DeepLabCut PyTorch snapshots saved on Windows

A `.pt` snapshot trained on Windows can contain a pickled Windows path, and
opens on Linux with
`NotImplementedError: cannot instantiate 'WindowsPath' on your system`.
The exported `dlc_pose.onnx` of the same model is unaffected, so use the ONNX or
TensorRT runtime on the Jetson, or re-save the snapshot on Linux.

### TensorFlow models

DeepLabCut 2.x models stored as TensorFlow snapshots (`pose_cfg.yaml` plus
`.pb`) need TensorFlow, which this environment does not include. Export them to
ONNX on a workstation.

## Verify

Run these from the repository root with the environment active. Together they
take about 10 minutes.

### 1. Runtimes and SDKs

```bash
python tools/verify_env.py
```

Read these lines:

```text
  [ok  ] PyTorch                   2.8.0  cuda=12.6  Orin
  [ok  ] ONNX Runtime              1.24.0  providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
  [ok  ] TensorRT                  10.3.0
  [ok  ] DeepLabCut-Live           1.1.0  backends=['pytorch']
  [ok  ] sleap-nn                  0.3.3
  [ok  ] capability probe          cuda=True tensorrt=True onnx=True gpu=Orin
```

`TensorFlow` reports `FAIL`, which is expected. The script's real-inference
section uses Windows-style paths and reports every model as "not on this
machine" on Linux; step 3 covers inference instead.

### 2. SLEAP against the reference keypoints

Each SLEAP model folder with a `selfcheck/` directory ships reference crops and
the keypoints the reference machine produced for them. This is the decisive test
that sleap-nn is running correctly. The script expects a `best.ckpt` in the
folder given to `--model` (see [Prepare the models](#prepare-the-models)).

```bash
python models/sleap_6point_v4/selfcheck/deployment_selfcheck.py --model models/sleap_6point_v4
```

Expected:

```text
  PASS feeding RGB                mean deviation from reference 0.000 px
  PASS verdict                    model + environment reproduce the reference EXACTLY.
RESULT: 0 FAIL, 1 WARN, 11 PASS
```

The one `WARN` asks for a `--client` script and can be ignored. Anything above
0.5 px means a package or GPU difference; fix that before looking further.

### 3. Every model through every runtime

```bash
QT_QPA_PLATFORM=offscreen python tools/tracking_selftest.py --pipeline both
```

The last line must report **0 SILENT**. A `REFUSED` verdict is correct when it
names something this machine does not have, such as TensorFlow.

### 4. Test suite

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

On the tested Jetson: 2,165 passed, 26 failed, 7 skipped. The failures are
not caused by the environment:

- Camera dialog and capture-backend tests expect the Windows backends
  `dshow` and `msmf`; Linux offers `v4l2`.
- Some calibration-store tests pick up measurements left in the shared store
  by earlier runs.
- The two `test_dlc_engine` checkpoint tests hit the Windows path described in
  [DeepLabCut PyTorch snapshots](#deeplabcut-pytorch-snapshots-saved-on-windows).

A failure outside these files is worth investigating.

### 5. The application

```bash
python pyOperant.py
python pyMaze.py
```

Each prints `[CYTHON] All extensions up to date` and `GUI ready`, and its window
opens. The line
`GPU device discovery failed ... /sys/class/drm/card1/device/vendor` is printed
by ONNX Runtime on every Jetson and is harmless.

### 6. Hardware encoding and TensorRT in a session

Start `pyOperant.py` from the activated environment, load a project with pose
tracking, and record for a few seconds. The newest log in `data/log/` must
contain these lines:

```text
Encoder capabilities: FFmpeg=True H.264[... nvmpi=True ...]
Using Jetson HW H.264 (h264_nvmpi) for .../...-Box1-....mp4
SLEAP using this machine's cached TensorRT engine instead of the model folder's: ...
SLEAP initialized (sleap_nn:tensorrt:direct/single, runtime=auto): ...
```

- `nvmpi=False`, or `Using libx264` for every box: the application is not using
  the FFmpeg from [step 10](#10-build-ffmpeg-with-the-hardware-encoder).
- `onnxruntime` in the last line: the cache has no engine for this model, box
  count and precision. See [Prepare the models](#prepare-the-models).

## Daily use

```bash
source /ssd_apps/pybehlab/bin/activate
cd ~/Documents/pyBehaviorLab
python pyOperant.py        # or pyMaze.py
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `torch.__version__` ends in `+cpu` | pip took PyPI's CPU wheel instead of NVIDIA's | Step 3: install torch by wheel URL |
| `The NVIDIA driver on your system is too old (found version 12060)` | A CUDA 13 torch build, from PyPI or a newer index entry | Uninstall torch, torchvision and every `nvidia-*` / `cuda-*` package, then redo step 3 |
| `metadata-generation-failed` for pandas, "requires Python >=3.11" | `--ignore-requires-python` was given to more than sleap-nn | Steps 5 and 6 as written |
| `No matching distribution found for PySide6` | PySide6 newer than 6.8.0.2 needs glibc 2.39 | Keep the `PySide6==6.8.0.2` pin |
| `import tensorrt` fails | The JetPack TensorRT is not linked in | Step 4; check `dpkg -l python3-libnvinfer` |
| Keypoints a few pixels off in a script, but correct in the application | The script passes BGR frames from `cv2` straight to the tracker | Convert with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`; the application does this in the frame bus |
| `Checkpoint not found: .../best.ckpt` during export | The checkpoint has another name | Link it as `best.ckpt` |
| `No such file or directory: 'D:/.../best.ckpt'` during export | `training_config.yaml` names pretrained weights on another machine | Set `pretrained_backbone_weights` and `pretrained_head_weights` to `null` |
| TensorRT or `auto` selected, the log shows `onnxruntime` | No engine in the cache for this model, box count and fp16 setting | Export one; see [Prepare the models](#prepare-the-models) |
| Log says `Jetson detected but h264_v4l2m2m not in this FFmpeg build`, every box records with libx264 | The system FFmpeg is on `PATH`; its `h264_v4l2m2m` cannot reach the Orin's encoder | [Step 10](#10-build-ffmpeg-with-the-hardware-encoder), then start the application from the activated environment |
| One box records with libx264, the others with `h264_nvmpi` | That box's frame is under 160 px on a side, which the Jetson encoder refuses | Expected. Make the box's region larger to use the hardware encoder |
| A hand-run `ffmpeg ... -c:v h264_nvmpi` prints `DoWork failed` and never exits, at 100% CPU | A frame under 160 px, or a very noisy frame encoded without a quantiser floor | Stop it with `kill -9`. Keep frames at least 160 px and add `-qmin 20 -qmax 51`, as the application does |
| Serial ports `Permission denied`, or a board only connects after `sudo chmod 666 /dev/ttyACM*` | No udev rule, so the ports are `root`/`dialout` only again after every replug or reboot | Install the pyboard udev rule once (see [System packages](#system-packages)) |
| Install fills the root disk | pip cache and temp files on the eMMC | Set `PIP_CACHE_DIR` and `TMPDIR` on the NVMe before installing |

## Tested configuration

Verified on 13 September 2026. Hardware encoding and the 4-box TensorRT
figures were added on 15 September 2026.

| Component | Version |
|---|---|
| Board | Jetson AGX Orin Developer Kit, 12 cores, 32 GB |
| JetPack / L4T | 6.2.2 / R36.5.0 |
| OS / glibc | Ubuntu 22.04.5 / 2.35 |
| CUDA / cuDNN / TensorRT | 12.6 / 9.3.0 / 10.3.0 |
| Python | 3.10.12 |
| torch / torchvision | 2.8.0 / 0.23.0 (NVIDIA JetPack 6, cu126) |
| onnxruntime-gpu / onnx | 1.24.0 / 1.22.0 |
| sleap-nn / sleap-io | 0.3.3 / 0.9.2 |
| deeplabcut-live / timm | 1.1.0 / 1.0.29 |
| lightning / skia-python | 2.6.6 / 138.0 |
| PySide6 / opencv-python / numpy | 6.8.0.2 / 4.11.0.86 / 1.26.4 |
| FFmpeg for recording | n6.0.1 with jetson-ffmpeg `64df5aa` (`h264_nvmpi`) and x264 `stable` `b35605a` |

Measured on this configuration with `sleap_6point_v4`, batch 1:

| Runtime | Per frame (median) | Mean deviation from reference |
|---|---|---|
| sleap-nn checkpoint, CUDA | n/a | 0.000 px |
| ONNX Runtime, GPU | 9 to 15 ms | 0.226 px |
| TensorRT FP32, built on the Jetson | 8.77 ms | 0.226 px |
| TensorRT FP16, built on the Jetson | 4.48 ms | 0.300 px |

Measured with `sleap_6point_v8` through the application's tracker, 4 boxes in
one batch, on the model's reference frames:

| Runtime | Per batch of 4 | Per frame | Keypoints against ONNX Runtime |
|---|---|---|---|
| ONNX Runtime, GPU (the model folder's ONNX) | 15.1 ms | 3.78 ms | reference |
| TensorRT FP32, built on the Jetson | 14.5 ms | 3.63 ms | 0.000 px |
| TensorRT FP16, built on the Jetson | 6.8 ms | 1.71 ms | 0.000 px |

Recording, 60 s of real 240×240 box video encoded with the application's own
settings for each encoder:

| Encoder | SSIM against the source frames | File size |
|---|---|---|
| libx264 (`ultrafast`, `zerolatency`, CRF 23) | 0.968 | 1.19 MB |
| `h264_nvmpi` (`-b:v 5M -qmin 20 -qmax 51`) | 0.976 | 1.29 MB |

For scale: during a live 2-box session at 540×404 and 15 fps, each libx264
encoder averaged 0.10 of one core, on a 12-core Orin that was about 66% idle.
Hardware encoding removes that load; it does not change inference time.
