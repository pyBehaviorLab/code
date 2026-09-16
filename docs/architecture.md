# Architecture

pyBehaviorLab runs **behavioural-neuroscience experiments across
many enclosures concurrently** from a single desktop application. It belongs to
the same family as pyControl, [Bpod](https://sanworks.github.io/Bpod_Wiki/) and
[Autopilot](https://docs.auto-pi-lot.com): a **real-time finite-state machine
(FSM) executes on a microcontroller**, while a **host computer orchestrates,
records, and keeps a record of what produced each dataset**.

Its lineage is explicit. The on-device layer *is* pyControl, the `pyControl/`
package is the unmodified MicroPython framework, so the state/event/timer model
and the sub-millisecond transition guarantees are inherited rather than
reimplemented. A pyControl task is an **extended state machine** (finite states
and events, augmented with variables and arbitrary Python that can also determine
behaviour), and, as in Bpod, every trial is timestamped on a single firmware
clock. Where pyControl and Bpod target one rig and Autopilot scales by *adding
computers* (a Terminal commanding a swarm of Raspberry-Pi *Pilots*), pyBehaviorLab
scales by **boxes within one process**: one host drives N *setups* (each an MCU +
camera + pose tracker) consolidated on a single machine.

This document is the system-level map (Diátaxis *explanation* tier). Each
subsystem links to a deeper concept page; a [glossary](reference/glossary.md)
defines every term in **bold** on first use.

---

## 1. The governing principle: where real time lives

The single distinction that organises the whole codebase:

> **The microcontroller owns real time; the host owns everything else.**

Reward valve timing, state transitions, and the **authoritative event clock**
(the *framework clock*, see §6) execute deterministically on the MCU in
MicroPython, free of operating-system jitter. The host is explicitly **not**
real-time: it compiles and uploads the task, streams the resulting event log off
the wire, and expresses every host-side measurement (camera frames, pose
estimates) **in the MCU's own time** rather than wall-clock. This is the same
host/device contract pyControl and Bpod use, and it is what makes a recording
reproducible and temporally coherent across the behavioural, video, and (where
present) photometry streams.

The firmware's response **latency**, the delay from a digital input transition
to the corresponding output, was measured on an oscilloscope for the F767 and
H723 configurations under light and heavy load; it is a determinism budget the
host cannot match and therefore does not contend for. Pulse-duration error spans
approximately one polling interval, because a software timer expires on a
framework cycle rather than at an arbitrary point between cycles. Per-board
distributions and the measurement method are in
[Performance and validation](reference/performance.md).

---

## 2. The layer stack

Read top-to-bottom as "closest to the experimenter" → "closest to the animal".

```
┌──────────────────────────────────────────────────────────────────────────┐
│  EXPERIMENTER             pyOperant.py  /  pyMaze.py   (entry points)     │
│  ───────────              Qt (PySide6) desktop app, one process, dark UI   │
├──────────────────────────────────────────────────────────────────────────┤
│  HOST ORCHESTRATION       source/gui/                                      │
│  ─────────────────        MainWindowBase + mode subclass + per-box widgets │
│                           run lifecycle, dialogs, live status, statistics  │
├───────────────┬──────────────────────────┬───────────────────────────────┤
│  WHAT PRODUCED │  VISION & TRACKING       │  BEHAVIOURAL CONTROL           │
│  A DATASET     │  source/video/           │  source/communication/        │
│  source/config/│  cameras → frame bus →   │  pycboard ↔ serial ↔ MCU       │
│  schema, djb2  │  sinks (record / pose /  │  uploads task + hardware def,  │
│  hashing,      │  tracker / mcu-push) +   │  drains the timestamped event  │
│  snapshots,    │  frame-matched display   │  stream into a per-box TSV     │
│  change log    │                          │                               │
├───────────────┴──────────────────────────┴───────────────────────────────┤
│  TRANSPORT                USB serial (raw-REPL framing), one port per setup │
├──────────────────────────────────────────────────────────────────────────┤
│  FIRMWARE (on MCU)        pyControl/ MicroPython finite-state machine   │
│  ─────────────────        states, events, timers, hardware drivers;        │
│                           emits timestamped events on the framework clock  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Hardware compatibility

Existing pyControl rigs run unchanged. `source/pyControl/` IS the pyControl
framework, `devices/` ARE the pyControl device drivers, and
`hardware_definitions/` keeps the same pin-map format, so a breakout board
already wired in a lab, its hardware definition and its tasks work here without
modification. Supported boards include breakout 1.0, 1.2, F767 and F767 v2,
plus the Nucleo connector and port expander; the device drivers cover pokes,
lickometers, solenoids, LED and analog-LED drivers, steppers (incl. TMC),
doors, rotary encoders, load cells, Schmitt triggers, uRFID, the audio family
(audio board/player, PCM5102, TAS5825M, I2S), photometry, the ESP microphone,
MCP and UART handlers and the frame logger/trigger.

The video, tracking and position-feedback layers are additions alongside that
firmware, not replacements for it, which is why an existing task keeps its
behaviour and simply gains coordinates and zone events it can react to.

## 4. Runtime boundary: host (CPython) vs MCU (MicroPython)

The repository contains code for **two distinct runtimes that never share an
interpreter**. Confusing them is the most common source of error.

| Runtime | Lives in | Imports | Reaches the device via |
|---|---|---|---|
| **Host** (CPython 3.11+) | `source/` (except `source/pyControl/`), entry points, `tools/` | `PySide6`, `cv2`, `numpy` |, runs the app + pytest |
| **MCU** (MicroPython on STM32 pyboard) | `source/pyControl/`, `devices/`, `tasks/`, `hardware_definitions/` | `pyb`, `ujson`, `from pyControl.utility import *` | uploaded over serial by `pycboard` |

MCU modules **cannot be imported or executed on the host**; they are transmitted
to the board as source text and compiled there. The host-side test suite
excludes the on-device script directories from collection accordingly.

| Concern | Runs on | Module(s) | Reference-system analogue |
|---|---|---|---|
| Task logic (states/events/trials) | **MCU** | `tasks/<Task>.py` on `pyControl` | pyControl task · Bpod state matrix |
| Hardware abstraction (pokes, valves, audio) | **MCU** | hardware-definition `.py` + `pyControl.hardware` | pyControl hwdef · Bpod modules · Autopilot HAL |
| Event timing + framework clock | **MCU** | `pyControl.framework`, `.timer` | Bpod FSM (<100 µs transitions) |
| Serial transport + upload | Host | `communication.pycboard`, `.pyboard` | pyControl host API |
| Event stream → disk (TSV) | Host | `communication.data_logger` | pyControl data file |
| GUI, run lifecycle, multi-box | Host | `gui.base` + `gui.operant`/`gui.maze` | Autopilot Terminal |
| Camera + pose + position feedback | Host | `video.*` | Autopilot transform/hardware |
| Config, identity, what produced a dataset | Host | `config.*` | Autopilot Subject/Biography |

---

## 5. Process and thread topology

A **single OS process**. Qt's threading contract is honoured strictly: **every
widget is touched only from the GUI thread**; worker threads perform I/O and
inference and hand results back via queued signals.

```
QApplication  (Qt event loop, GUI thread)
│
├─ MainWindowBase + mode subclass + per-box widgets
│     • QTimer 1 Hz       idle refresh (status, hash staleness, clear-meta)
│     • QTimer ~100 Hz    process tick (drives the frame pipeline while live)
│     • debounced autosave (750 ms) → project_workflow.save_project
│
├─ Pycboard × N            one per setup; each owns a serial worker thread
│     • drains the MCU event stream (raw-REPL framed, bounded reads)
│     • data_logger, writes <session>/mcu/<box>.tsv (framework-timestamped)
│     • host↔framework clock anchor refreshed from every event (pycboard.fw_ms_at)
│
├─ Pipeline                 one pipeline; per-box subscriptions (Qt-free core)
│     • tick thread ~100 Hz drains CameraThreads, publishes FrameBus(box_id)
│     • per-box sink workers (I/O / inference only, never touch Qt):
│         RecorderSink   lossless  → mp4 + _video_data.txt
│         PoseSink       drop-newest, batched → DeepLabCut / SLEAP inference
│         TrackerSink    drop-oldest, parallel pool → centroid / zone logic
│         MCUPusher      event-driven → coordinates + intrinsic events to MCU
│     • frame-matched display channel → QtBridge → GUI (frame-locked overlay)
│
└─ CameraThread × N        one per camera; polls the grab API, 10 ms idle backoff
```

The pipeline lives in **one module family** (`video.framebus`); `operant` and
`maze` import it unchanged and only subscribe to its public events. The core
carries **no Qt dependency** - `framebus.qt_bridge.QtBridge` re-emits its
callbacks as queued `Signal`s so the GUI always sees them on the main thread.
See [concepts/frame-pipeline](concepts/frame-pipeline.md).

---

## 6. The four data flows

The system is most clearly understood as four flows that converge at the MCU.

### 5.1 Behavioural control loop

How a task runs end-to-end. Upload is the expensive step; *Record* merely starts
the framework.

```
   Task .py + hardware-definition .py
        │  pycboard.setup_state_machine()   (raw-REPL upload, source → MCU)
        ▼
   ┌───────────────┐   variables pushed at UPLOAD (one set_variables round-trip):
   │  MCU runs the │     1. hw_*        (rig store)
   │  pyControl    │     2. persistent  (last-session values; wins on collision)
   │  state machine│     everything else keeps the task-file default
   └──────┬────────┘
          │  emits timestamped events / prints on the framework clock
          ▼
   pycboard serial drain ──► data_logger ──► <session>/mcu/<box>.tsv
                              │
                              └──► live status, statistics, GUI consumers
          ▲
          │  at STOP: capture_persistent() → one get_variables() round-trip
          │           → persistent_variables.json   (carry-forward BY VARIABLE NAME)
```

The variable lifecycle, spec sidecar, runtime push/capture, persistent values,
is one module, `config.task_variables`. See
[user-guide/api-class](user-guide/api-class.md) and
[concepts/run-lineage](concepts/run-lineage.md).

### 5.2 Vision and position feedback

A frame's journey from sensor to disk, to the live display, and (optionally) back
into the task. Sinks subscribe to per-box frames; each owns a worker thread with
an explicit **back-pressure policy**:

| Sink | Policy | Rationale |
|---|---|---|
| **RecorderSink** | **lossless** (block/alarm when full) | every captured frame must be written; overflow raises a health alarm, never silently drops |
| **PoseSink** | **drop-newest**, ≤1 queued, batched | inference may lag capture; keep the newest frame, infer many setups in one forward pass |
| **MCUPusher** | event-driven (not a queued sink) | pushes only on coordinate/zone *change* to close the loop |

```
camera grab → CameraThread.buffer → Pipeline.tick → FrameBus.publish(box_id)
                                                                 │
            ┌──────────────────────┬───────────────────┬────────┴───────────┐
            ▼                      ▼                   ▼                    ▼
      RecorderSink            PoseSink            TrackerSink           (display)
      mp4 + _video_data.txt   pose inference      centroid / zone       see §6.4
       (BGR frame)            (RGB frame)              │
                                  │                    │
                                  └──────► MCUPusher ◄─┘
                                              │  coordinates + intrinsic events
                                              ▼  (position back to the task)
                                     MCU state machine (§6.1)
```

**Colour-space discipline.** The whole-frame BGR→RGB conversion is computed once
per camera frame and cached (`BoxFrame.image_rgb`). **Inference consumes RGB**
(`image_rgb`, as DLC/SLEAP expect); **recording and display consume BGR**
(`image`, as OpenCV and the Qt `Format_BGR888` path expect). The two are cached
views of the same capture, no per-box reconversion, and neither path is ever fed
the wrong colour order.

Pose estimation is **backend-agnostic**: DLC-Live is one of several
interchangeable estimators (DeepLabCut, SLEAP, exported ONNX/TensorRT engines).
See [concepts/tracking-pipeline](concepts/tracking-pipeline.md).

### 5.3 Timing and synchronisation

There is **one master clock: the MCU framework time** written into the TSV. The
host never anchors on wall-clock. Each camera frame is stamped with a host
**monotonic** capture instant (`capture_host_ns`, forced strictly increasing),
which is mapped into framework time per frame:

```
MCU framework clock ──(every event carries fw time)──► host↔fw anchor
        │                                                    │
        │                                   pycboard.fw_ms_at(capture_host_ns)
        ▼                                                    │
  data_logger TSV  ◄── authoritative per-box timeline        ▼
                                            RecorderSink stamps each frame row of
                                            _video_data.txt with:
                                              frame_fw_ms  (frame capture, mapped)
                                              pose_lag_ms  (capture → pose exists)
                                              filter_ms    (the smoothing's share)
```

`pycboard.get_timestamp()` (the MCU framework time) is the single clock every
session surface reads; `format_run_clock` is the one formatter. There is no
wall-clock anchor and no separate clock object. See
[concepts/time-sync](concepts/time-sync.md).

### 5.4 Frame-matched display

The live preview is decoupled from the lossless sinks. One path drives it for
every box:

- **Polling paint** (`base._paint_streaming_cameras_once`, on the process tick),
  shows the *latest* captured frame for **every** box, tracking or not, at the
  full capture FPS. This avoids a queued display sink that, under slow overlay
  drawing, silently dropped frames (a measured 20→7 fps regression, hence
  polling, not a sink). The overlay (pose keypoints) is drawn on top
  from the box's most recent inference result (`base._overlay[setup_id]`,
  TTL-bounded), so the *video* runs at camera rate while the *overlay* refreshes
  at the inference rate.

### 5.5 Identity and the record of what produced a dataset

Every artefact is named by a **djb2 hash**, and every project mutation is
appended to **one change log**. This is what lets a recording be reproduced or
audited months later.

```
HASHING  (config.hashing, the sole hash function; no SHA-256)
   • 4-byte-LE djb2 of a .py  → matches the MCU's task/hardware-def hash in the TSV
   • byte-wise djb2 of content → config self-hash, snapshot keys, pose manifests

SNAPSHOT STORE  (config.snapshot_store, content-addressable, one per project)
   upload ─► capture_source()      ─► _pending/box<N>_<kind>.<ext>
   record ─► commit_box_sources()  ─► <project>/source/<djb2>.<ext>
   pose   ─► capture_dlc()         ─► <project>/source/dlc/<djb2>.json  (manifest)

CHANGE LOG  (config.history, the single writer)
   one <project>/change_log.jsonl, two event families on one timeline:
     • config edits   {ts, action, source, path, old, new}
     • source capture {ts, kind, source, box_id, djb2, …}

RUN ROLLUP  (gui.project_workflow)
   <project>/runs/<task_family>/<date>.json, minimal 6-field run rows
```

On-disk layout of a recording session:

```
<data_dir>/<project>/<task_family>/<YYYY-MM-DD>/
    mcu/<box>.tsv                 framework-timestamped event stream (authoritative)
    video/<box>.mp4               recording
    video/<box>_video_data.txt    per-frame frame_fw_ms + pose_lag_ms
<project>/source/<djb2>.<ext>     committed task / hardware-def / action / api_class
<project>/persistent_variables.json   per-subject carry-forward values
<project>/change_log.jsonl        audit timeline
```

See [concepts/snapshot-tracking](concepts/snapshot-tracking.md) and
[concepts/run-lineage](concepts/run-lineage.md).

---

## 7. Configuration model. Project vs Run

The most important schema distinction in the codebase:

| | **Project** | **Run** |
|---|---|---|
| Is | Rig invariants + experimenter metadata | One recording session |
| Holds | hardware def, tracking config, ROIs/zones, background, action config, cohort metadata | task + subject_id + config snapshot + output paths |
| Stored | `experiment_config.json` (+ `source/`, `metadata/`) | `runs/<task>/<date>.json` + data folders |
| Lifetime | Persists across sessions | Created at *Record*, closed at *Stop* |

`task` and `subject_id` are **run-only**, stripped from the project config by
`_box_to_compact_dict` / `to_dict_for_hash` so the project hash is stable across
runs. All config conversion flows through one module
(`config.experiment.read_ui_into_config` / `apply_config_to_ui`), no translator
shims. On save, a **fresh** `Config` is rebuilt each time; non-widget state (meta
preferences, tracking config, COM port) is seeded from the active config so it is
not reset to defaults. See [concepts/project-vs-run](concepts/project-vs-run.md).

---

## 8. Module tour

By subsystem, with the one-line responsibility.

**`communication/`, the MCU serial layer (mirrors pyControl's host API)**
- `pycboard`, board handle: upload task/hardware-def, set/get variables, drain
  events, framework timestamps (`get_timestamp`, `fw_ms_at`)
- `pyboard`, raw-REPL transport (note: `PyboardError` subclasses
  `BaseException`, so `except Exception` will not catch it)
- `data_logger`, writes the per-box `.tsv` event stream
- `mcu_ports`, `message`, `controller`, `errors`, port discovery, framing, control

**`pyControl/`, the on-device MicroPython framework (runs on the MCU)**
- `framework`, `state_machine`, `timer`, `hardware`, `utility`, the FSM, the
  event clock, and hardware drivers. Exhaust these primitives before inventing
  host-side mechanisms.

**`config/`, schema, identity, dataset records (host)**
- `experiment`, the typed `Config` schema (FileRef, BoxConfig, Zone, Meta,
  tracking, setup_config), canonical JSON, and UI↔config conversion, the hub
- `hashing`, the sole djb2 implementation (file + content flavours)
- `snapshot_store`, content-addressable `source/` store + pose manifests
- `history`, the single `change_log.jsonl` writer + recent-configs cache
- `task_variables`, spec sidecar + runtime push/capture + persistent values
- `multi_instance`, file locks, atomic writes, PID-in-log, OCC auto-merge

**`gui/`, host orchestration (the "Terminal")**
- `base` - `MainWindowBase`: shared run lifecycle, save/load, status, display
- `operant` / `maze`, the two mode subclasses (see §9)
- `project_workflow`, project create/load/save + run open/close
- `pose_subsystem`, `config_manager`, `metadata_manager`, `main_window_mixins`
- `widgets/`, per-box widgets, `run_task` (per-box run lifecycle), tracking
  panel, zone editor, variable controls, live status
- `dialogs/`. MCU/config, camera connect, tracking, ROI, doors, subjects

**`video/`, vision and position feedback**
- `cameras/`, capture backends (OpenCV, Spinnaker, Ximea) behind one base
- `framebus/`, the pipeline: `controller`, `frame_bus`, `types`, `qt_bridge`,
  and the sinks (`recorder_sink`, `pose_sink`, `tracker_sink`, `mcu_pusher`,
  `sink_base`)
- `recording/` - `recorder`, `ffmpeg`/`encoder_pool`, `frame_log`,
  `mcu_row_mirror`, `drop_log`
- `tracking/` - `pose`, `inference`, `smoothing`, `speed`
- `zones/`. ROI geometry, triggering, coordinate spaces

**`stats/`, live statistics**, per-box configuration, signature-grouped canvas.

---

## 9. Operant vs Maze

Both entry points share `MainWindowBase`, all per-box widgets, the pipeline,
recorder, MCU subsystem, pose subsystem, and statistics. Mode code is narrow and
limited to box layout, the per-box status surface, and a few hooks:

- **operant** - `gui/operant.py` + `widgets/box_control.py`, a grid of boxes,
  per-box status bar; often shared cameras with per-box ROI segmentation.
- **maze** - `gui/maze.py` + `widgets/setup_widget.py`, a single setup widget,
  combined readout, door controls, tracking-coordinate experimental buttons;
  one camera per arena.

Everything tracking / recording / MCU / dataset records is shared; reachability
analysis confirms there is **no operant-only or maze-only subsystem**, only the
two `MainWindow` subclasses fork.

---

## 10. Design principles

These constraints, not incidental choices, shape the architecture:

- **One master clock.** All session time is MCU framework time via
  `pycboard.get_timestamp()`; host measurements are mapped into it
  (`fw_ms_at`). No wall-clock anchors.
- **One hash.** djb2 only, its file form matches the MCU upload hash, so
  analysis tooling joins across log surfaces by hash. No SHA-256.
- **One writer per file.** The change log, the recent cache, the persistent
  values, and the source store each have exactly one owning module.
- **Single-owner pipeline.** Camera→bus→sinks→MCU is owned by one
  `Pipeline`; both modes only touch its public API.
- **Lossless where it matters, lossy where it must.** Recording and tracking are
  lossless (with explicit drop accounting); only the live display may skip
  frames. Frame drops are logged, never silent.
- **Frame-accurate overlays.** Asynchronous results carry the `cam_frame_id`
  they were computed from; the display is locked to that frame.
- **Every dataset names its own origin.** Every uploaded artefact is content-addressed
  and committed; every config mutation is logged on one append-only timeline.
- **Per-setup, never global.** Each box runs its own MCU + tracking; gates apply
  per box; master buttons fan out via eligibility predicates and stay enabled for
  idle boxes while others run.
- **One feature, one module.** A new feature is one new module; existing files
  get short hooks, so a behaviour is debugged by opening one file.
- **No legacy scaffolding.** Schema bumps replace code inline, no migrators, no
  `schema_version` branches, no `.bak` sidecars.
- **Portable across Windows + Linux + Jetson.** The pipeline must run identically
  on all three; Jetson (ARM64) is the weakest link, so no x86-only assumptions.
  UI is PySide6, dark-only, with all visuals owned by `design_tokens` /
  `style_builders`.

---

## 11. References

- Saunders, J. L., Ott, L. A., & Wehr, M. (2019). *Autopilot: Automating
  behavioral experiments with lots of Raspberry Pis.* bioRxiv 807693.
  Autopilot: <https://docs.auto-pi-lot.com>
- Sanworks. *Bpod* finite-state-machine framework:
  <https://sanworks.github.io/Bpod_Wiki/>
- Mathis, A., et al. (2018). *DeepLabCut: markerless pose estimation of
  user-defined body parts with deep learning.* Nat Neurosci 21(9), 1281–1289.
- Pereira, T. D., et al. (2022). *SLEAP: a deep learning system for multi-animal
  pose tracking.* Nat Methods 19(4), 486–495.
- Kane, G. A., et al. (2020). *Real-time, low-latency closed-loop feedback using
  markerless posture tracking.* eLife 9, e61909.

Pose estimation is **delegated**, not reimplemented: DeepLabCut and SLEAP are
interchangeable backends, and DeepLabCut-Live! provides the online path. What
this platform contributes is their integration, one sync timestamp on every
frame and every event, one frame stream feeding recording and inference, and one stored
configuration binding the task, the hardware definition, the zones and the
trigger rules.

## See also

- [Project vs Run](concepts/project-vs-run.md)
- [Snapshot tracking](concepts/snapshot-tracking.md) · [Run lineage](concepts/run-lineage.md)
- [Frame pipeline](concepts/frame-pipeline.md) · [Tracking pipeline](concepts/tracking-pipeline.md)
- [Time sync](concepts/time-sync.md)
- [File formats](reference/file-formats.md) · [Glossary](reference/glossary.md)
