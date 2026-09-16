# pyBehaviorLab

**GUI v2.0 · pyControl firmware v2.1**

> **Authoritative current docs:** [`CLAUDE.md`](CLAUDE.md) (repo layout, build/test
> commands, conventions) and [`docs/pipeline_overview/`](docs/pipeline_overview/)
> (function-level map of the live pipeline). Some detailed module/path names
> further down predate the `source/video/` restructure, trust those two where
> they differ.

Single-process Qt6 application for behavioural-neuroscience experiments
on Micropython-based pyControl rigs. Two entry-point modes share one
unified pipeline:

| Mode      | Entry point    | Use case                                                   |
|-----------|----------------|------------------------------------------------------------|
| Maze      | `pyMaze.py`   | Multi-arena maze tasks (one camera per arena)              |
| Operant   | `pyOperant.py` | Multi-box operant chambers (often shared cameras + ROI)    |

Both modes load the **same** dialog widgets, tracking pipeline, recorder,
calibration UI, scale system, and lineage/history layer. The fork
between them is now just the entry-point file + the per-mode
`MainWindow` subclass; everything else is shared.

---

## Quick start

```bash
# 1. Install dependencies (see requirements.txt)
pip install -r requirements.txt
pip install cython          # required to build the C extensions

# 2. Build Cython extensions (auto-runs on first launch; this is for explicit pre-build)
python setup.py build_ext --inplace

# 3. Launch
python pyMaze.py     # maze mode
python pyOperant.py   # operant mode
```

## Documentation (Sphinx)

Build the documentation locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Both entry points call `source.cython.build.ensure_built()` at startup,
which checks staleness of every `.pyx` against its compiled `.pyd`/`.so`
and rebuilds only if needed. Pure-Python fallbacks exist for every Cython
function, if Cython isn't installed, the app still runs (slower).

---

## Build system

### Toolchain requirements
- **Python 3.11+** (tested 3.11 and 3.14)
- **Cython 3.x**: required to compile the extensions on first launch.
  Compiled `.pyd`/`.so` and generated `.c` files are git-ignored, so a
  fresh clone builds on-target via `ensure_built()`. If Cython is absent,
  the build is skipped and every consumer falls back to pure Python.
- **NumPy 1.24+** for headers used by `_image_ops`, `drawing_ops`
- **A C compiler** matching your Python's ABI (MSVC 14.x for CPython on
  Windows; gcc/clang on Linux/macOS)

### Files
- `setup.py`: extension declarations + cythonize directives:
  `boundscheck=False`, `wraparound=False`, `cdivision=True`, `language_level=3`.
- `source/cython/build.py`: runtime checker invoked at app startup. Public API:
  - `needs_build() -> bool`: true if any module is missing or older than its `.pyx`
  - `build(verbose=True) -> bool`: runs `setup.py build_ext --inplace` as subprocess
  - `ensure_built(verbose=True) -> bool`: `needs_build()` → `build()` → flush stale `sys.modules`

### Three Cython modules

| Module                        | Source                          | Purpose                                                              |
|-------------------------------|---------------------------------|----------------------------------------------------------------------|
| `source.cython.zone_math`     | `zone_math.pyx`                 | Polygon math (point-in-polygon, line-side, rotation, facing angle)   |
| `source.cython._image_ops`    | `_image_ops.pyx`                | `fused_illumination_normalize`, `batch_bgr_to_gray`                  |
| `source.cython.drawing_ops`   | `drawing_ops.pyx`               | Per-frame DLC keypoint extraction (`prepare_dlc_keypoints`)          |

### Force a full rebuild

```bash
python setup.py build_ext --inplace --force
```

### Availability gating

There is no package-level flag layer. Each consumer imports the compiled
symbols directly from the submodule and sets a local boolean, choosing the
Cython path or its pure-Python fallback at import time:

```python
try:
    from source.cython.zone_math import cross_line_side
    _HAS_ZONE_MATH = True
except ImportError:
    _HAS_ZONE_MATH = False
```

---

## Architecture overview

```
                ┌────────────┐
                │  Camera    │  (CameraThread per camera, runs off Qt main)
                │  backend   │
                └─────┬──────┘
                      │ frame batches via deque (lossless ring, drop-on-overflow logged)
                      ▼
            ┌───────────────────┐
            │ Pipeline          │  Tick thread drains cameras, fans frames out:
            │ (single owner)    │   ┌──► display sink(s) (worker thread)
            │                   │   ├──► pose sink (DLC/SLEAP, skip-but-write)
            └─────┬─────────┬───┘   └──► recorder sink (lossless target; backpressure alarms)
                  │         │
                  │         └────────────────► VideoSegmentProcessor (ROI split for shared cameras)
                  ▼
        ┌───────────────────┐
        │ TrackingPushPolicy │  → pycboard.queue_set_coordinates / queue_trigger_event
        └───────────────────┘    (sync write to serial; small per-frame messages)
```

Key invariants:
- **Lossless contract**: every captured frame goes through tracker AND recorder, with drop accounting for every overflow site (see `pipeline_metrics.drop_log`).
- **Display is decimated** (~20 Hz fixed throttle): the only consumer that may skip frames.
- **Serial port access**: the GUI's 10 ms `plot_update` timer reads MCU output (`Pycboard.process_data`); tracker pushes write through `pycboard.queue_set_coordinates` / `queue_trigger_event` directly. Small per-frame writes race with the reader but OS-level buffering makes this acceptable in practice.

---

## Module reference

### `source.pipeline.controller.Pipeline`

The single owner of the camera → bus → sinks → MCU pipeline.
Both modes construct exactly one `Pipeline` and interact only via its public API.

```python
from source.pipeline.controller import Pipeline

pipe = Pipeline(target_fps=30)
# Connect a camera to a box (optionally with ROI segmentation config):
pipe.connect_camera(camera_id=0, box_id=1, segment_config=None, camera_backend="opencv")
# Enable pose (after configure_pose_model):
pipe.configure_pose_model(model_path="...", tracker_type="dlc")
pipe.enable_pose(1, zone_lookup=...)
```

Public surface (high level):
- Box/sync: `register_box`, `unregister_box`, `update_fw_anchor`
- Cameras: `connect_camera`, `disconnect_camera`, `set_target_fps`, `set_target_resolution`, `set_segment_processor`
- Recording: `start_recording`, `stop_recording`, `is_recording`
- Pose: `configure_pose_model`, `enable_pose`, `disable_pose`
- Blob tracking: `enable_blob_tracking`, `disable_blob_tracking` (backgrounds via `tracker_manager.set_background`)
- Results: `add_display_callback`, `on_health`, pose/tracker result callbacks via `qt_bridge`
- Shutdown: `shutdown`

See `docs/PIPELINE.md` for a deeper dive (frame ordering, skip-but-write, backpressure).

### `source.video.recording.recorder.VideoRecorder`

Single canonical recorder used by both modes. Frames are always fed in
externally by the pipeline's `RecorderSink` via `add_frame`; an internal
thread drains and writes to disk via `VideoWriterFactory`.

```python
from source.video.recording.recorder import VideoRecorder

rec = VideoRecorder(camera_id=0, fps=20, resolution=(640, 480),
                    pycboard=board, codec="MJPEG", use_gpu="auto")
rec.start_recording(data_dir, subject_id, datetime_now=...)
# Per frame (pushed by RecorderSink):
rec.add_frame(frame_bgr, capture_time_ns=...)
rec.stop_recording()
```

Hooks:
- `recorder.drop_callback = fn(count, reason)`: fired on queue overflow
- `recorder.annotate_callback = fn(frame, capture_ts) -> annotated_frame`,
  set by `controller.set_annotate_saved` to burn overlays into the saved file

### `source.communication.video.VideoManager`

Camera lifecycle only. After the recorder unification, this class no
longer owns recorders or sessions.

| Method                                | What                                                                |
|---------------------------------------|---------------------------------------------------------------------|
| `start_camera(camera_id, box_id, ...)`| Create `CameraThread`, start acquisition                            |
| `stop_camera(box_id)`                 | Stop the camera thread, clean up FPS state                          |
| `register_recorder(box_id, recorder)` | Attach a `Video_recorder` for camera-thread direct routing          |
| `unregister_recorder(box_id)`         | Detach                                                              |
| `_handle_record_frame(batch, camera_id)` | Internal: route a camera batch to the box's recorder (segment-aware) |
| `getFPS(box_id)` / `getRuntimeFPS(box_id)` | Calibrated effective vs. live measured camera FPS              |
| `register_pose_callback(callback)`    | Wire the DLC/SLEAP per-frame inference dispatch                     |
| `cleanup()`                           | Stop everything, clear all maps                                     |

### `source.communication.tracking_push_policy.TrackingPushPolicy`

Translates per-frame tracker output into MCU coordinate pushes + edge-triggered
events. All pushes go through `pycboard.queue_set_coordinates` / `queue_trigger_event`,
which are thin sync wrappers over `set_coordinates` / `trigger_event`.

### `source.communication.pipeline_metrics`

Profiler + drop log singletons.

```python
from source.communication.pipeline_metrics import profiler, drop_log

# Profiler, off by default; enable via env var or programmatically
import os; os.environ['PYBL_PIPELINE_PROFILE'] = '1'   # or profiler.enable()
with profiler.section('blob_track'):
    tracker.update(frame)
profiler.maybe_dump()                                    # dumps every 5 s

# Drop log, always on; writes <session>/_drops.tsv
drop_log.set_session_path(session_dir / '_drops.tsv')    # done by SessionLogger
drop_log.record('recorder', frame_idx=42, capture_ts=1.5, reason='queue_full')
counts = drop_log.counts()                               # {stream: total}
```

Drop sites instrumented:
- `Video_recorder.add_frame` → `recorder/queue_full`
- `inference_backend.submit` → `tracker/inference_inflight_box{N}`
- `CameraThread._recording_buffer` → `camera_ring/ring_full_cam{id}_n{count}`

### `source.communication.config_schema` + `config_history`

Experiment configs are JSON files with task + hardware-definition file
hashes embedded for lineage.

```python
from source.communication.config_schema import Config, new_config, save, load, rehash_files

cfg = new_config('novelty', task_path='tasks/novelty.py', hwd_path='hardware/rig.py')
cfg.boxes['1'] = {'camera': {'id': 0}, 'tracking': {'mode': 'blob'}}
save(cfg, 'experiments/novelty.json')

cfg2 = load('experiments/novelty.json')
ok, drift = rehash_files(cfg2)   # detect file modifications on disk
```

Per-config sidecar JSONL log of every session that loaded it:

```python
from source.communication.config_history import append_session, list_recent

append_session(config_path, cfg=cfg, session_dir=...,
               subjects={1: 'M042'}, boxes_used=[1, 2],
               frames_total=27300, drops_total=0, duration_s=1820)
recent = list_recent(20)        # for the main-tab "Recent" dropdown
```

Status badges in the status bar (`MainWindowBase._build_status_panel`)
show two contextual labels: the loaded **project** and the **tracking** state
(ready / not-configured / actionable hint).

### `source.communication.pycboard.Pycboard`

Pyboard host-side wrapper. Both task hash AND hardware-definition hash
are now embedded into every session's `.tsv` header (see `data_logger.py`).

| Method                                       | What                                                   |
|----------------------------------------------|--------------------------------------------------------|
| `load_hardware_definition(hwd_path)`         | Transfer HD to MCU; cache `_loaded_hwd_path` + `_loaded_hwd_hash` |
| `setup_state_machine(sm_name, ...)`          | Upload task; populate `sm_info` (task_hash, HD hash, etc.) |
| `set_coordinates(name, value)`               | Direct serial write                                   |
| `queue_set_coordinates(name, value)`         | Sync wrapper used by tracker pipeline workers         |
| `trigger_event(name, source='u')`            | Direct serial write                                   |
| `queue_trigger_event(name, source='u')`      | Sync wrapper used by tracker pipeline workers         |
| `introspect_hardware()`                      | Query MCU's loaded HD (used by door dialog fallback)  |

### `source.communication.trackers.blob.BlobTracker`

Background-subtraction tracker. Default mode `running_avg` (Bonsai-style
masked update, drift-tolerant, doesn't absorb stationary subjects).
Other modes: `static`, `mog2`. The adaptive bg model auto-freezes
its learning rate when the centroid hasn't moved for `_stationary_threshold_frames`
consecutive frames.

```python
tracker = BlobTracker(box_id=1, callback=on_update)
tracker.set_background(bg_frame)
tracker.update_params(bg_mode='mog2', threshold=25, min_area=200, max_area=8000)
success, position = tracker.update(frame, timestamp=ts_ms)
metrics = tracker.get_quality_metrics()  # {success_rate, jitter_px, area_p10/median/p90, ...}
```

### `source.gui.tracker_calibration_dialog.TrackerCalibrationDialog`

Live blob calibration with multi-box support.

```python
TrackerCalibrationDialog(
    parent, box_id, get_frame_callback,
    initial_params={'detect_dark': True, 'threshold': 25, ...},
    background_path='backgrounds/box1.png',
    box_callbacks={1: cb1, 2: cb2, 3: cb3},   # multi-box dropdown shown when >1
    box_initial_params={1: ..., 2: ...},
    box_backgrounds={1: 'backgrounds/box1.png', ...},
)
# After dlg.exec() == Accepted:
single = dlg.get_calibration()       # current box
multi  = dlg.get_calibrations()      # {box_id: params} for everything tuned
```

Live quality readout (success-rate / jitter / area distribution) with
green/yellow/red traffic-light indicator.

### Scale calibration (`source.gui.zone_editor_tab.ZoneEditorWidget`)

Inline `value` + unit (cm/mm/m) widgets on the editor toolbar. Replaces
the old two-popup workflow.

```python
ze = ZoneEditorWidget(box_id=1, get_frame_callback=cb)
ze.set_zones(loaded_zones)
ze.has_valid_scale()    # True iff a scale zone exists with positive value
ze.get_scale_meta()     # {value, unit, pixel_length, px_per_unit, unit_per_px}
```

The tracking config dialog refuses to save zones for any box that has
zones drawn but no scale calibration. The analysis path warns when a
session's zones lack scale (distances reported in pixels, surfaced
loudly in the result table).

### `source.gui.door_control_dialog.DoorControlDialog`

Manual door control with a per-door state machine + per-motor enable.

| Column | What                                                                     |
|--------|--------------------------------------------------------------------------|
| `OPEN/UP`   | Move up. Disabled when door state is `up` OR motor not enabled.     |
| `CLOSE/DOWN`| Move down. Disabled when state is `down` OR motor not enabled.      |
| `HOME`      | (Door device only) close to limit switch; ends in `down` state.     |
| `STOP`      | `motor.stop()` only, never touches enable line. Resets state to unknown. |
| `EN`        | Toggle motor enable. Label flips between `ENABLE` / `DISABLE`. While off, UP/DOWN/HOME locked + grayed. |
| `UP=`       | (Legacy steppers) pick which motor phase = UP (`forward` or `backward`); depends on coil wiring. Door devices: N/A (firmware handles phase). |

State invariants:
- After UP / OPEN → state = `up` (button gray, dashed border)
- After DOWN / CLOSE → state = `down`
- After STOP → state = `unknown` (both UP and DOWN re-enabled)
- EN off → UP/DOWN/HOME all gray-disabled
- EN on → label changes to `DISABLE`, doors operable

`ENABLE ALL` toggles every per-row EN in lockstep. `OPEN ALL` / `CLOSE ALL`
skip already-there doors and skip de-energised motors.

---

## Run lineage + reproducibility

Every output file carries a chain of hashes so a session months later
can be matched back to the exact code that produced it.

### pyControl `.tsv` header (per session, per box)
```
I task_name        novelty
I task_file_hash   1234567890       (djb2 of tasks/novelty.py)
I hardware_def_name  rig_A.py
I hardware_def_hash  9876543210     (djb2 of hardware/rig_A.py)
I framework_version 1.7
I micropython_version 1.20
```

### `_video_data.txt` header (per session, per box)
Written by `SessionLogger`. Includes box-level config, camera info,
codec, and pose-tracker info. Coupled with FW-time anchor extrapolation.

### `<config>.usage.jsonl` (per experiment, append-only)
One line per session that loaded the experiment config:

```json
{"timestamp": "2026-05-02T14:30:00", "session_dir": "data/...",
 "subjects": {"1": "M042"}, "boxes_used": [1, 2],
 "config_djb2": "abc12345", "task_djb2": 1234567890,
 "hwd_djb2": 9876543210, "frames_total": 27300,
 "drops_total": 0, "duration_s": 1820, "outcome": "completed"}
```

### `~/.pybehaviorlab/recent.json`
Global cache of the 50 most-recently-used experiment configs for the
main-tab "Recent" dropdown. Pure cache, regenerable from per-config
JSONL files if lost.

### `<session>/_drops.tsv`
Append-only TSV of every dropped frame at every drop site:
```
wall_iso              monotonic_ms  stream       frame_idx  capture_ts  reason
2026-05-02T14:30:01   5.700         recorder     42         1.500000    queue_full
2026-05-02T14:30:01   5.797         tracker      na         na          inflight_box1
2026-05-02T14:30:01   26.243        camera_ring  100        2.000000    ring_full_cam0_n3
```

---

## Profiling

```bash
# Enable per-section ms accumulators for one run:
$env:PYBL_PIPELINE_PROFILE=1; python pyOperant.py     # PowerShell
PYBL_PIPELINE_PROFILE=1 python pyOperant.py           # bash / Linux
```

Or programmatically:
```python
from source.communication.pipeline_metrics import profiler
profiler.enable()
```

Output every 5 s in the application log:
```
[pipeline_profile] section            n        avg_ms     max_ms
  controller.tick                  150      0.832      3.110
  push_tracking                     45      1.927      4.604
  tracker.update                    45      6.118      9.220
```

---

## Theming

Two themes: `light` and `dark`. Default = **dark** (set in `ThemeManager.resolve_theme_name`).
Selectable via the status-bar toggle in both modes. Persisted to settings.

---

## Cross-platform notes

| Concern             | Windows                                              | Linux x86_64                       | Jetson (ARM64)                                      |
|---------------------|------------------------------------------------------|------------------------------------|-----------------------------------------------------|
| Cython build        | Pre-built `.pyd` for cp311/cp314; MSVC for source    | gcc at install time                | gcc at install time                                 |
| NVENC               | Available with NVIDIA GPU + drivers                  | Available with NVIDIA GPU          | Available natively (NVDEC/NVENC modules)            |
| DLC backend         | TF/PT GPU if CUDA                                    | Same                               | TensorRT preferred (out of scope here)              |
| pyserial            | COM port blocking reads OK                           | `/dev/ttyACM*` blocking reads OK   | Same as Linux                                       |

---

## Repository layout

```
code/
  pyMaze.py                            # Maze entry point
  pyOperant.py                          # Operant entry point
  setup.py                              # Cython extension build declarations
  README.md                             # this file
  requirements.txt
  hardware_definitions/                 # MCU hardware-definition .py files
  tasks/                                # MCU pyControl task .py files
  experiments/                          # Saved experiment configs (.json + .usage.jsonl)
  config/                               # App settings + paths
  source/
    pipeline/                           # ★ pipeline owner + sinks + FrameBus
      controller.py                     # ★ Pipeline (single owner)
      bus.py                            # FrameBus (cam fan-out + per-box ROI)
      sinks/                            # recorder/pose/tracker/display/push
    communication/                      # Camera backends, recorder, MCU, writers
      video_recorder.py                 # ★ canonical recorder
      video.py                          # camera lifecycle (VideoManager)
      pycboard.py                       # Pyboard host wrapper
      tracking_push_policy.py           # tracker → MCU translation
      pipeline_metrics.py               # profiler + drop_log singletons
      session_logger.py                 # _video_data.txt writer + FW sync
      config_schema.py                  # Config dataclass + hashing
      config_history.py                 # JSONL sidecar + recent cache
      ffmpeg_writer.py                  # encoder factory + capability detection
      inference_backend.py              # DLC/SLEAP execution layer
      fw_video_sync.py                  # FW-time anchor (lock-protected)
      trackers/                         # blob, pose, enhancer
      cameras/                          # backend abstractions (OpenCV, Spinnaker, Ximea)
    cython/                             # ★ compiled extensions
      build.py                          # ensure_built / needs_build
      *.pyx, *.c, *.pyd
    gui/                                # Qt widgets, dialogs, main windows
      main_window_base.py               # shared base (ROI, zones, lineage, …)
      main_window_maze.py               # maze entry MainWindow
      main_window_operant.py            # operant entry MainWindow
      widgets.py                        # SetupWidget + BoxControlWidget + tiles
      dialogs.py                        # all dialogs (unified across modes)
      tracking_tab.py                   # UnifiedTrackingDialog + TrackingSettingsPanel
      zone_editor_tab.py                # ZoneEditorWidget (matplotlib canvas)
      tracker_calibration_dialog.py     # blob calibration + multi-box
      door_control_dialog.py            # door + motor control with state machine
      metadata_manager.py               # subject metadata
      video_analysis_tab.py             # post-hoc analysis
      styles.py                         # ThemeManager (light / dark)
      config_manager.py                 # save/load experiment JSON
  docs/
    pipeline_schematic.html
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `[CYTHON] WARNING: Cython not installed` at startup | `pip install cython` missing | Install Cython, or accept pure-Python fallback |
| `ImportError: source.cython._image_ops` | Stale `.pyd` from a different Python ABI | `python setup.py build_ext --inplace --force` |
| Door dialog says "No Doors" but HD was uploaded | Standalone HD-load dialog wasn't caching `_hw_config` | Fixed, dialog now caches; door dialog also has live MCU `introspect_hardware()` fallback |
| Live video tile won't grow with window | Old chrome/Group-Box wrapper | Fixed, bare QWidget container, `Expanding` size policies, row/col stretches set in `updateVideoStreams` |
| Tracking config preview shows full image not ROI | Frame-callback double-cropping | Fixed, `get_last_frame(box_id)` already segments internally; wrapper trusts it for shared cameras |
| Save Zones rejects with "Scale Required" | Zones drawn without calibration | Click **Scale** with the inline value/unit set, then drag a calibration line |
| Distances in analysis output show as raw pixels | Scale missing on the saved session | Add Scale in tracking config and re-save |
