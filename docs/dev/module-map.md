# Module map, ownership rules, and flows

**Read this before any structural change.** It answers three questions:
where does a given responsibility live, what is shared between the two app modes
and what is legitimately split, and what actually happens, step by step,
between a click and a frame on disk.

It sits alongside two neighbours and does not repeat them:

| Document | Answers |
|---|---|
| [`architecture.md`](../architecture.md) | *Why* the system is shaped this way, the MCU-owns-real-time principle, the layer stack, comparison to pyControl / Bpod / Autopilot |
| [`contributing.md`](contributing.md) | The hard rules: no legacy, no migrators, no invented architectures |
| **this file** | *Where* everything is, *who owns what*, and *how a request flows* |

Everything below was verified against the tree, not remembered. Where a rule is
enforced by a test, the test is named. **A rule with no test is a rule that will
rot**, several in §7 already have.

---

## 1. Two runtimes. Never mix them.

| | Host | MCU |
|---|---|---|
| Runs on | The PC, CPython | The microcontroller, MicroPython |
| Code | `source/` except `source/pyControl/`, plus `tools/`, entry points | `source/pyControl/`, `devices/`, `tasks/`, `hardware_definitions/` |
| Imports | `PySide6`, `cv2`, `numpy` | `pyb`, `ujson`, `from pyControl.utility import *` |
| Gets there by | Just running | Upload over serial by `communication/pycboard.py` |

MCU code **cannot be imported on the host**. `source/tests/board_tests/` and
`framework_tests/` are MCU scripts, excluded from collection in `conftest.py`.

One thing that looks like MCU code and is not: **`api_classes/` is host code.**
Those modules subclass `communication/api.py::Api` and are loaded by name at
runtime from `run_task.py`. They run on the PC. A naive grep will call
`communication/api.py` dead; it is not.

---

## 2. Where the code is

68.5k production lines over 119 modules, plus ~23k of tests over 140 files.

| Package | Lines | Files | Largest member |
|---|---|---|---|
| `gui/` (top level) | 15.4k | 17 | `base.py` 6.1k |
| `gui/widgets/` | 11.7k | 17 | `tracking_panel.py` 2.4k |
| `gui/dialogs/` | 11.5k | 10 | `camera_connect.py` 3.9k |
| `video/cameras/` | 6.9k | 15 | `capture.py` 1.6k |
| `video/framebus/` | 5.2k | 10 | `controller.py` 1.1k |
| `config/` | 5.2k | 8 | `experiment.py` 2.5k |
| `video/tracking/` | 7.6k | 18 | `pose.py` |
| `stats/` | 3.0k | 3 | `canvas.py` 2.6k |
| `video/recording/` | 2.8k | 6 | `frame_log.py` 0.9k |
| `communication/` | 2.3k | 8 | `pycboard.py` 1.1k |

`base.py` and `camera_connect.py` are 15% of the tree between them.

**There are no unimported modules.** Before declaring anything dead, grep for its
name **inside string literals** too, this codebase dispatches by name, e.g.
`_safe_box_call(sid, 'clear_video')`, and loads `api_classes` and several
`tools/` views via `importlib`.

### Regenerating the map

```bash
"$HOME/.conda/envs/pyos5_new/Scripts/graphify.exe" update .
```

Deterministic (tree-sitter AST, no LLM, no API cost). Writes the gitignored
`graphify-out/`: `graph.html` (interactive), `GRAPH_REPORT.md`, `graph.json`
- currently 11,583 nodes and 21,616 edges. Useful queries:

```bash
graphify path "MainWindowBase" "RecorderSink"   # shortest path between two nodes
graphify explain "Pipeline"                     # a node and its neighbours
```

---

## 3. Ownership: one owner per fact

The single most common bug class here is **two places holding the same fact and
disagreeing**. This is the intended ownership; where reality differs, §7 says so.

| Fact | Owner | Rule |
|---|---|---|
| Camera → box binding | `Pipeline` | The GUI *requests* changes and *renders* the result; it must not keep its own copy. **Partially violated** (§7.1) |
| Frames from a camera | `FrameBus`, one per camera | Sinks subscribe. Nothing else pulls frames off a camera thread |
| Which frames get written | `RecorderSink` | Recorder is **lossless**; tracker (drop-oldest) and pose (drop-newest, queue of 1) may skip under load but account every drop to `_drops.tsv`. Display is the only consumer that skips unaccounted |
| What a camera can actually deliver | `video/cameras/probe.py` | `probe_all()` measures every backend, `merged_modes()` flattens to `(w, h, fps, backend)` rows. **Never read a mode list off the device without measuring it.** The dialog consumes the merged list, it is a plain list, not a `ProbeResult` |
| Which backend serves a mode | The chosen resolution row itself | Backend travels *with* the mode, set via `OpenCVCamera(cv_backend=…)` / `use_backend()`. There is deliberately **no separate backend control**, it asked the same question twice and hid itself exactly when an override was needed |
| Cached camera capabilities | `video/cameras/calibration_store.py` | Keyed on USB identity (`vid:pid:serial`) per machine; invalidated on bus-speed change. `probed_variants` also round-trips into the project |
| Hashes | `config/hashing.py` | **djb2 only, never sha256.** Pinned by `test_hashing_layering.py` |
| Project create / load / save | `gui/project_workflow.py` over `config/experiment.py` | Nothing else writes `experiment_config.json` |
| GUI dict ↔ `Config` | `config/experiment.py` | No separate translator module, ever |
| Zone geometry + policy | `video/zones/schema.py::Zone` | One class. Field `zone_type`, on-disk key `"type"`. Never `asdict` it, the key differs and the Shapely cache breaks JSON encoding. Build from `from_dict`, never by naming three fields |
| Session time | The MCU framework clock | Every clock reads `pycboard.get_timestamp()`. No wall-clock anchors. `format_run_clock` is the one formatter |
| How a frame becomes a model's input | `video/tracking/input_policy.py` | `letterbox` / `crop_track` / `full`, decided from `ModelInfo`. Both `PoseSink.configure_model` and the offline `PoseAsPose` call `plan()`; neither re-derives it. A model trained on native-scale crops must be **cut**, never scaled, see §"crop-trained models" below |
| Which engine runs a DLC model | `video/tracking/dlc_engine.py` | `resolve(path, model_type)` → `(engine, path_for_dlclive, note)`. The PyTorch runner needs the `.pt` file, not the folder; a `note` means the request was downgraded and must reach the operator |
| Cleaning a keypoint trace offline | `offline_analysis/engine/refine.py` (down one keypoint's timeline) over `pose_repair.py` (across keypoints, and forwards+backwards in time) | `refine` gates and fills per keypoint; `pose_repair` owns the body-shape model, swaps, mistracks, reconstruction, and the RTS smoother. See §8d |
| Secondary-window stacking | `gui/window_behavior.py` | §6 |

**Naming.** Time fields carry role + domain, never mechanism: `_fw_ms` (MCU
firmware time), `_host_ns`, `_cam_ns`. `monotonic_ns` / `capture_mono_ns` as
field names are banned (the pipeline's capture clock is `capture_host_ns`).

**Comments** say what and why, plus the use case, never fix history, dates or
plan codes. A comment describing behaviour the code no longer has is worse than
none: two docstrings in `base.py` described sha256 hashing that does not happen.

---

## 4. The maze ↔ operant contract

`QMainWindow → MainWindowUtilsMixin → MainWindowBase → MainWindow`.
~7.2k lines shared, ~3.2k mode-specific (operant 1.8k, maze 1.0k).

### Shared, do not fork

Every dialog in `gui/dialogs/` except `door.py` (maze-only, correct, doors are a
maze concept). Every widget in `gui/widgets/` except `box_control.py` (operant)
and `setup_widget.py` (maze), which are a legitimate pair: both subclass
`RunTask` and implement the same nine-method protocol. The whole video pipeline,
the whole MCU layer, `config/`, `stats/`.

### The camera dialog is not forked

Both modes call `base.show_camera_config_dialog`, which builds the one
`CameraConnectDialog`. Grepping its 3.9k lines for "maze" or "operant" returns a
single prose comment. It *looks* different for three reasons, one of which is a
defect:

1. Maze passes no `help_text` when building its camera group, so the header lacks
   operant's "?". Cosmetic.
2. The dialog auto-detects CCTV (2+ boxes on one camera) vs Individual and
   **hides the Camera-ID and Backend columns in CCTV mode**. Operant rigs share a
   camera; maze rigs do not. Same code, different data. **Correct, leave it.**
3. Maze hands the dialog a two-method fake instead of a real layout, so rows come
   out in dict-insertion order.

The bigger real gap is the *tracking* dialog: operant injects start/stop/trigger
callbacks via `_tracking_dialog_extra_kwargs`; the base default returns `{}`, so
maze's tracking dialog has no working in-dialog Start/Stop.

### The hook surface

Only two members are truly abstract: `_box_widgets` (property) and
`_load_mode_name`. Everything else in the "Hook surface" section of `base.py` is
a **default implementation** a mode may override.

`_safe_box_call(setup_id, name)` dispatches to the per-box widget through a
**closed set of nine names**: `append_status`, `camera_id_text`, `clear_video`,
`get_roi`, `notify_camera_disconnected`, `notify_camera_starting`,
`set_fps_text`, `update_frame`, `video_size`. Both widgets implement all nine;
adding a tenth means adding it to both.

### The rule that keeps being broken

> **Shared code must never probe the main window with `hasattr` for a method only
> one mode defines.**

The failure is always silent, the feature just does not happen in the other
mode, swallowed by a bare `except` or an `or []`. The worst case is fixed:
`gui/box_alerts.py` tried three surfaces to raise a per-box error badge
(`get_setup_widget`, `liveStatusWidgets`, `_video_holders`) and **maze had none
of the three**, so per-box error alerts never rendered there at all. Route 1 now
uses `MainWindowBase._setup_widget_for`, which both modes inherit, and operant's
duplicate accessor is gone. Routes 2 and 3 remain operant-only surfaces, that
is legitimate, since maze has no Live Status panel or video-holder grid.

If shared code needs something from the mode, put it on the base, a real default
or an explicit no-op. Not a probe. `test_mode_contract.py` ratchets the probe
count so it can fall but never rise.

### Drifted twins

Six methods exist in both modes where **each has a fix the other lacks**:

| Method | The drift |
|---|---|
| ~~`stop_recording`~~ | **Fixed.** It existed in both with opposite meanings, operant's is a full teardown including `stop_framework()`, maze's five lines never stopped the framework, and shared code reached it via `getattr`. Maze's is now `stop_box_recording`, and `closeEvent` names the shared `_stop_recording_for_box` instead of probing. Pinned by `test_mode_contract.py` |
| ~~`_on_record_stopped`~~ | **Not drifted, the finding was a false positive.** Operant closes the MCU data files in `BoxControlWidget`; maze re-enables its buttons through `refresh_ui_state → apply_global_state → _update_button_states`. The two main-window methods differ because each mode's per-box *widget* owns a different part of the teardown, which is legitimate. `test_record_stop_equivalence.py` now asserts the *outcome* on both sides rather than which file performs it |
| ~~`_box_camera_disconnected_cleanup`~~ | **Fixed.** Both already called `super()`, but `_pose_display_roi`, `_display_frame_counts` and `_fps_status_times` are **base-owned** dicts and each mode cleared a different subset, so whichever mode you ran leaked the other's. The base now clears all eight of its own per-box dicts; each override keeps only its genuinely mode-specific state (maze: `last_frame_update`; operant: the track-button gate and recording stop) |
| ~~`_load_clear_existing_state`~~ | **Fixed, narrower than reported.** Operant *did* reach `_clear_project_runtime_state`, via `_teardown_all_boxes`, but that early-returns on an empty rig, so a project loaded after one whose boxes were removed by hand inherited the previous pipeline registries. Now cleared unconditionally, as maze always did |
| ~~`_pose_after_disabled`~~ | **Fixed.** Operant cleared `pose_zone_state`, maze reset the Kalman enhancer, and both modes carry both. The union is on the base hook (reusing the existing shared `_detach_tracking_enhancer`); both mode overrides deleted |
| ~~`_refresh_test_tracking_button`~~ | **Fixed.** Only operant locked Tracking Config during a preview, so in maze the operator could reconfigure tracking underneath a running test. The lock is on the base now and both overrides call `super()`. The subtler half: maze's `_apply_ui_state` re-enabled the button on the next 1 Hz tick, so a one-shot lock would not have held, the enable predicate now includes `not _is_test_tracking_active()` |

---

## 5. Flows

### 5.1 Launch

```
pyOperant.py / pyMaze.py
  → app_bootstrap.py     Qt attrs, configure_logging(), ensure_built() (Cython,
                         with pure-Python fallback), encoder pre-warm, excepthook
  → MainWindow.__init__  (operant or maze)
       → MainWindowBase.__init__
            → Pipeline()                 builds VideoManager + 4 sinks, starts
                                         the sink worker threads
            → refresh_timer.start(1000)  1 Hz housekeeping
            → process_timer              10 ms; started only while a box runs or
                                         a camera streams
       → _build_ui()                     mode-specific layout
  → _auto_flow_after_load                cameras, pose init, MCU connect
```

### 5.2 Project load

`project_workflow.load_project` reads `experiment_config.json` →
`config/experiment.py` rebuilds the `Config` dataclass → the base fans it out to
boxes, zones, tracking config and the camera registry. Paths are stored
**relative to `source/paths.py::top_dir`**, so the tree is portable.
`_load_clear_existing_state` (mode hook) tears down the previous project;
`_load_post_restore` re-applies UI state.

### 5.3 Camera connect

```
CameraConnectDialog
  collect + dedupe selection      one entry per camera, N boxes each
  preflight resolution + FPS      each camera needs its own; probed if absent
  ROI prompt (shared cameras)     per-box regions, stored as percentages
  build + publish segment config  → main_window.video_segment_config
  connect each camera
      Pipeline.connect_camera(camera_id, setup_id, …)
        → disconnect-first if this box was on a DIFFERENT camera
        → VideoManager.start_camera      opens the device, spawns CameraThread
        → get-or-create FrameBus         (existing bus gets the NEW segmenter,
                                          so re-drawn ROIs take effect)
        → bus.register_box + subscribe the three sinks for this box
                                          (recorder / pose / tracker; MCUPusher
                                          is not a bus sink, it consumes
                                          tracker/pose RESULTS via Pipeline)
        → _ensure_tick_running()
      extra boxes on that camera: Pipeline.bind_box_to_camera (no reopen,
                                  disconnect-first guard applies here too)
  finalize: persist video settings, mark project dirty
```

### 5.4 A frame, end to end

```
CameraThread                    grabs, stamps capture time, publishes latest
   │
Pipeline tick (10 ms)           drains EVERY buffered frame, lossless
   │
FrameBus.publish_frame          per-box ROI crop (shared cameras), lens
   │                            correction, then fan-out
   ├─→ RecorderSink   lossless     → VideoRecorder → ffmpeg/NVENC → .mp4
   │                                 + frame_log row (frame_fw_ms, pose_lag_ms)
   ├─→ TrackerSink    drop-oldest  → centroid; a full queue
   │                                 evicts the oldest frame into _drops.tsv
   ├─→ PoseSink       drop-newest  → DLC/SLEAP inference; queue of 1, skips
   │                                 while busy (accounted in _drops.tsv) and
   │                                 writes an empty row so the gap is visible
   └─→ MCUPusher                  → queues coords / intrinsic events
                                    │
GUI process_timer (10 ms)           drains the queue and performs the serial
   │                                write on the GUI thread, as upstream
   ▼                                pyControl does
pycboard.serial.write → the board
```

**The display is not in this chain.** It polls
`CameraThread.get_latest_frame_versioned` from the GUI thread and paints the most
recent frame, skipping freely. An earlier queued display sink dropped frames
silently upstream and delivered 7 fps from a 20 fps camera; polling cannot.

**Latency.** `video/framebus/latency.py` measures `capture_to_poll`,
`poll_to_infer`, `infer_to_push`, `push_to_wire`; `total_ms()` (sum of per-stage
p95) is the horizon the pose forecast compensates for. Three 10 ms polling stages
sit in this path, camera backoff, pipeline tick, GUI tick, so roughly
10–15 ms mean / 20–30 ms p95 is pure scheduling overhead. A push-based framework
such as Bonsai pays none of that; the physics (exposure, readout, inference) is
identical in both.

### 5.5 Record start / stop

**Start**: install the per-box MCU row mirror **before** opening the MCU file,
open the recorder, open the run in project history.

**Stop**, in order: stop the framework → stop tracking → detach and stop the
recorder (which returns the encoder, so no zombie ffmpeg survives) → close the
MCU data-logger files → re-enable the buttons → close the run.
See the `_on_record_stopped` row in §4, neither mode currently does all of it.

### 5.6 Shutdown

`closeEvent` → flush autosave → stop timers → stop recordings →
`pipeline.shutdown()` → `close_independent_windows()` → `_close_event_extras`.

---

## 6. Windows and dialogs

**Modal dialogs** block the main window and sit above it. That is what modal
means; nothing to fix.

**Secondary windows the operator works alongside**, detached tabs, the Session
Plot window, must be able to go *behind* the main window. That is not a flag:

> A `QWidget` created with a **parent** and given the `Qt.Window` flag becomes an
> *owned* top-level. On Windows an owned window is permanently above its owner in
> the z-order, and no amount of `NonModal` or flag-twiddling changes it.

So these windows take **no Qt parent**. The cost is lifetime and shutdown, and
`gui/window_behavior.py` owns both: register the window, then
`close_independent_windows()` at shutdown. A parentless window nothing closes is
also a **heap-corruption risk**, the C++ object outlives the Python wrapper and
is freed at an arbitrary later moment. `conftest.py` carries a matching per-test
teardown, beside the one for leaked `Pipeline`s, which exists for the same
reason. Pinned by `test_window_behavior.py`.

---

## 7. Known violations, read before trusting a rule

1. **Partially fixed.** `base.py` still aliases
   `self.video_manager = self.pipeline.video_manager`, and read-only queries
   through it (`box_camera_map.get`, `get_last_frame`, `is_camera_streaming`,
   `get_fps`) remain, those are tolerable. Every **mutation** through the alias
   is gone: `Pipeline` grew `camera_id_for_box`, `force_color_for_box`,
   `set_capture_defaults` and `release_all_cameras`, and
   `test_pipeline_ownership.py` fails the build if new ones appear. Note
   `release_all_cameras` ≠ `shutdown`: the latter also stops the sink workers,
   after which the Pipeline cannot be reused, the remove-all-boxes sweep used
   to call `video_manager.cleanup()` and leave the buses pointing at dead
   threads. The ROI dialog no longer opens a second handle on a running camera
   either, it reads frames from `Pipeline.get_bus`, the same way, the lens
   wizard does, so drawing regions mid-session costs no device access at all.
   Still outstanding: recorder-geometry policy living in `base.py`.

2. ~~**Two `Zone` classes.**~~ **Fixed.** There is now one, in
   `video/zones/schema.py`, carrying all 17 fields; `config/experiment.py`
   imports it. Field name `zone_type`, on-disk key `"type"` (what 100% of real
   files use), mapped explicitly in `to_dict`/`from_dict` rather than via
   `asdict`. `to_dict` omits anything still at its default, so a plain rectangle
   is not padded. `transmit_mode`/`coord_var` are non-optional strings and
   `from_dict` maps null onto the working default, a `None` there matched no
   consumer branch and silently disabled the zone. Both zone-manager bridges
   (base and maze) build via `from_dict`, so authored policy survives.
   Pinned by `test_zone_unification.py`.

3. **Duplicate implementations, fewer than they look.** Counting by shape
   overcounts badly here; several "duplicates" answer different questions and
   merging them is a regression:

   * **ROI crop ×3.** *Not* mergeable. The recorder's own `roi` is the only
     cropper on the operant no-segment path, deleting it "because the bus
     already crops" would lose the crop entirely there. The real risk was that
     the one-cropper invariant was unenforced; `_recorder_geometry` now asserts
     it and `test_recorder_geometry_single_cropper` pins it.
   * **Colour convert ×5.** `types.convert_color` (RGB semantics, returns
     `None` for rgb-from-gray) and `cameras.base.normalize_channels` (*copies*
     a mono passthrough, because SDK drain loops hand it a recycled driver
     buffer) are genuinely different contracts. I merged the recorder's
     `_coerce_channels` into `normalize_channels` and reverted it: the copy is
     a pure per-frame allocation on the recording hot path, and a test caught
     it. Both now carry a comment saying why there are two.
   * **FPS estimators, fixed.** `measure_fps_at` was triplicated with
     *different* hardcoded windows (OpenCV settled 2.0 s and counted over
     3.0 s; the SDK backends 0.5 s and 2.0 s), so the three backends reported
     rates that were not comparable and a camera slower to settle than its
     window simply read low. One `measure_delivered_fps` in `cameras/base.py`
     now holds the settle-then-count loop and the arithmetic; each backend
     passes its own drain/count closures. The two window profiles are named
     constants and their difference is deliberate, a UVC camera renegotiates
     its mode and re-converges auto-exposure on every open, an industrial SDK
     camera does not.
   * **Point-in-polygon ×4, three of them are correct.** The Cython
     `zone_math` twin of `geometry.point_in_polygon` is the documented
     accelerator pattern, not a duplicate. What *was* duplicated: two display
     sites each calling `cv2.pointPolygonTest` and each re-deciding the
     boundary rule, now one `geometry.point_inside_px`. **Note the surviving
     divergence**: the overlay's test is boundary-*inclusive*, Shapely's
     `Zone.contains` is boundary-*exclusive*, so a centroid exactly on an edge
     can highlight while the tracker calls it outside. Sub-pixel, documented
     at both ends rather than silently unified.
   * **Two `CameraConfig` classes, fixed.** `cameras/base.py`'s is now
     `ResolvedCameraSettings`: frozen, small, the outcome of one `configure()`
     call on the device. `framebus/types.py` keeps `CameraConfig`, the
     persisted per-camera record the setup dialog edits and the project saves.
     Both were reachable from `gui/base.py` under the same name, so a call site
     told the reader nothing about which was in hand.

4. **Unfinished features that look dead.** Both were decided, and the entry is
   kept because half of one is still here:

   * `discover_plugins()` **no longer exists**. User camera backends in
     `experiments/config/camera_backends/*.py` were never loaded and the hook
     that would have loaded them is gone.
   * `HardwareVariablesDialog` **no longer exists** either, but
     `task_variables.py:379` still fills `result.hw_missing` for it and
     nothing reads the field. That is the sweep this entry warned against,
     done in one direction only: the consumer went, the producer stayed.

5. **`tools/offline_analysis/` is a deliberate fork, and still is.** This entry
   previously read "**De-vendored**", describing a removal that did not happen.
   `tools/offline_analysis/vendor/` is present and holds eight copies,
   ~2,800 lines, every one of them imported. Read the code before the entry.

   The design is not the muddle a mixed import list suggests. It is two rules:

   * **Vendor what the analyser needs to run at all.** Theme, style builders,
     the zone editor and the zone helpers are copied into `vendor/`, so the
     folder can be lifted out and still open, draw a zone and write it back in
     the same normalized form the rig uses. That last part is why the *editor*
     is copied rather than reimplemented: a zone drawn while analysing must
     obey the rules a zone drawn before the session obeys.
   * **Guard-import what is optional.** Re-tracking is the one thing the
     analyser genuinely cannot do alone, so `engine/trackers.py` reaches for
     `source.video.tracking` inside `_rig()`, which returns `None` and keeps
     the reason when the rig is absent. No module-level `source.*` import
     exists anywhere on the active path, so the folder still starts without
     the rig; it simply reports that there is no tracker.

   The cost of copies is drift, and this repository has paid it once: the
   analyzer shipped a stale vivid `success` green after the rig moved to a dim
   sage, so "saved OK" was a different colour in the two apps for months.

   **Both guards against that were deleted and have been restored.**
   `tests/test_vendor_matches_rig.py` compares every vendored file's
   definitions against the module it came from, imports stripped (rewriting
   them *is* the vendoring) with an `EXPECTED_DIFFERENCES` entry for the one
   deliberate divergence, `zone_geometry`'s Cython import. When it came back it
   immediately found `vendor/zone_schema.py`, copied and imported by nothing,
   now deleted. `tests/test_no_live_rig_imports.py` holds the perimeter: the
   seam may name the tracking package and nothing else, never at module scope,
   and the rig must not import the analyser.

   The pair had been gone long enough to matter and nothing had gone wrong yet:
   when they were restored, every difference between a copy and its origin was
   an import rewrite. The alarms were off, not the building on fire.

   **The separation that matters most is about runtime, not imports**: the
   analyzer must never touch a camera, a serial port, a frame bus or the
   firmware, because it is meant to be safe to run *while a session is
   recording*.

---

## 8. Traps that have each cost a debugging session

| Trap | Consequence |
|---|---|
| `PyboardError` subclasses `BaseException` | `except Exception` does **not** catch it. Use `BaseException` or `PyboardError` |
| `str(d.get(k, ""))` when `d[k]` is `None` | Yields the string `"None"`. Use `or ""` *after* `.get()`. Comparing two of them makes `"None" == "None"` true, this is why clearing a camera-ID cell wiped the capture card |
| `pycboard.print` writes Qt widgets | Worker threads must swap it for a signal, or "parallel" work silently serialises |
| An `autoDelete` QRunnable solely owning a queued-signal QObject | PySide6 frees it mid-emit → crash. Keep a reference |
| The app launches on Python **3.9**; pytest runs **3.14** | Newer syntax passes tests and breaks startup. `test_runtime_python_syntax.py` parses the whole host tree under 3.9 |
| A missed dict lookup in the camera dialog | Never a harmless `None`, it selects the slowest or most broken fallback |
| `w.close(); w.deleteLater(); QApplication.processEvents()` in a test | Looks like cleanup, leaks anyway - `processEvents` does not deliver `DeferredDelete`. Use `source/tests/qt_dispose.py::dispose`. The C++ widget otherwise outlives its Python wrapper, and freeing the wrapper later is heap corruption (0xc0000374) landing in an *unrelated* test, as a hang or an exit-255 with every test reported passing |
| Calling `close()` on a widget you did not write | `closeEvent` is application code. `UnifiedTrackingDialog` opens a modal "save zones?" prompt in its own; driving that with no event loop corrupts the heap. To dispose of an unknown widget, `deleteLater()` it, never close it |
| An orientation trigger authored against a single-keypoint model | Never fires, and nothing says so. See below |
| Reading `in_zone` as a level condition | It is an edge like every other rule. See below |
| Letterboxing a **crop-trained** pose model | Confident keypoints in the wrong places, and a detection rate that looks healthy. See §8c |
| `model_type` in a `dlc_opts` / `sleap_opts` dict | Dropped as ambiguous between the two backends. It must be `dlc_model_type` / `sleap_model_type`; `filter_options` resolves the prefix. The bare key still changed the model **cache key**, so every setting rebuilt the model and none of them reached it |
| DLC `model_type="base"` | That is DeepLabCut-Live's **TensorFlow** runner. On a rig without TensorFlow a PyTorch model fails with `ModuleNotFoundError: tensorflow`, an error naming the wrong package entirely. The default is `auto` |
| Handing DLC's PyTorch runner a model **folder** | It `torch.load`s the path, so it wants the `.pt`. A folder raises `PermissionError` on Windows, `IsADirectoryError` elsewhere |
| Deciding the pose batch cap in `configure_model` | It reads `_enabled` + `_box_shapes`, and at Init **both are empty**, pose is enabled afterwards, and `_box_shapes` is filled by the frame handler for pose-enabled boxes only. The cap came out 1 and nothing recomputed it, so every multi-box rig ran serial inference. See §8f |
| `max_batch_size` in a SLEAP `export_metadata.json` | Describes the **TensorRT** engine build, not the ONNX graph beside it. `models/sleap` says 1 and its ONNX takes a batch of 16 quite happily, its batch axis is dynamic. Check the graph (`sess.get_inputs()[0].shape`), not the manifest |
| Setting `DLCLive(convert2rgb=False)` | **It does not stick.** `DLCLive.get_pose` opens with `if frame.ndim >= 2: self.convert2rgb = True`, and `ndim >= 2` is true of every frame, so it is forced back on at each call and `process_frame` reverses the channels. The tracker therefore calls `dlc_live.runner.get_pose` instead. See §8g |
| Timing two inference engines in isolation | Flatters whichever one batches: an isolated loop always hands it a full batch, and a live session hands it a partial one. The DLC crossover measured at 4 boxes in isolation and 8 on the rig |

---

## 8c. Crop-trained pose models

sleap-nn writes the training image size as `preprocessing.max_width` /
`max_height` whether the labels were whole frames or crops of them. Nothing in
the file distinguishes the two, and the difference decides whether the frame
should be **scaled** onto that size or **cut** at it.

`models/sleap` is the second kind: 224x176 at `scale: 1.0`, from a skeleton
named `dklab_crop`. Fitting a 513x315 arena onto it scales by 0.437, so the
animal reaches the network at 44% of the size it was trained on. Measured
against the model's own `selfcheck/golden.json`:

| input path | max error | min confidence |
|---|---|---|
| native 224x176 crop (as trained) | 0.29–0.46 px | 0.73 |
| `crop_track` window at native scale | 1.7–2.1 px | 0.80 |
| letterbox 513x315 → 224x176 | 9–11 px, or dropped | 0.26 |

Over a 32-frame retrack of a moving animal the letterbox found 24% of keypoints
at a median 11.6 px and a p95 of 272 px, keypoints on the far side of the
arena. The window found 98% at a median 1.1 px.

`input_policy.crop_trained()` separates them by one field: an **explicitly**
native `scale: 1.0` beside a declared input size. DeepLabCut states no scale and
its declared input is an export's baked-in graph shape, so silence is
deliberately not read as native, doing so made every DLC export look
crop-trained.

The size relationship is deliberately *not* part of that test. It was, and a
16-box rig showed why: each ROI is then 160×120, smaller than the 224×176
window, so it failed the "much larger" test and fell through to a letterbox that
**upscaled** it 1.4×, wrong in the opposite direction and just as invisible.
When the frame is smaller than the window the whole frame is used at native
scale, padded only as far as the backbone stride requires (160×120 → 160×128),
not out to the model's window, which would run inference over a large black
border for nothing.

## 8g. DeepLabCut: the colour bug, and the two engines

**`convert2rgb` never took effect.** `DLCLive.get_pose` and `init_inference`
both begin `if frame.ndim >= 2: self.convert2rgb = True`, and `ndim >= 2` is
true of every frame there has ever been, so whatever was asked for at
construction is forced on at each call. `process_frame` then runs `img_to_rgb`,
which *reverses* the channels of any 3-D array. The pipeline already hands pose
RGB, so DeepLabCut-Live was feeding this network BGR on every frame, with
`convert2rgb=False` set and a test asserting it was set.

`DLCLiveTracker._run_dlclive` therefore calls `dlc_live.runner.get_pose`, where
the model lives and no colour conversion happens. Nothing is lost: dynamic
cropping lives in the runner and still runs, static cropping is not used, the
resize is ours, and only a no-op Processor is skipped. Measured: the two paths
disagreed by **484 px** before and **0.017 px** after.

**Two engines, and which is faster depends on the rig.** The export's batch
axis is fixed at 1, so N boxes are N forward passes. The PyTorch runner batches
- not through DeepLabCut-Live, which has no batched entry point, but through
the `nn.Module` underneath it, driven with the runner's *own* model and
transforms so the arithmetic cannot drift. It also carries ~11 ms of fixed
per-call overhead the export does not. Measured live, per box:

| boxes | export | pytorch |
|---|---|---|
| 4 | **22.0** | 16.4 |
| 8 | 13.2 | **13.8** |
| 16 | 6.1 | **9.7** |

Hence `BATCH_CROSSOVER_BOXES = 8`. An earlier value of 4 came from timing the
engines in isolation on always-full batches; a live session delivers partial
ones, which pay the per-call overhead without earning the sharing.

The batch is uploaded as **uint8** and converted and normalised on the GPU,
transforming first would send float32, four times the bytes, and spend the CPU
a 16-box rig needs for encoding, timestamps and the MCU push loop.

**A dynamic batch axis is only fast at a steady size.** ONNX Runtime re-plans
whenever the input shape changes, and a live rig's batch varies with
frame-arrival jitter. Re-exporting DLC with a dynamic axis made 16 boxes *worse*
than the fixed-batch graph, 1.7 pose results per box per second against 6.1,
until every batch was padded out to one constant shape (`ORTRunner(pad_to=…)`,
extra rows sliced off the result):

| batch pattern | unpadded | padded |
|---|---|---|
| steady 16 | 436/s | 453/s |
| alternating 16, 15, 14 | 149/s | 413/s |
| random 1–16 | 73/s | 188/s |

With both, that export reaches 15.2 per box at sixteen and 25.1 at eight, nine
times the shipped one. A smaller 224×176 export is faster still and was
rejected: confidence fell from 0.78–0.84 to 0.46–0.67, the window being too
small for a network trained on 448×448 crops.

## 8f. The pose batch cap, and why it was always 1

`PoseSink` chunks its dispatch by `_max_batch_size`, which exists to fit GPU
memory on small devices (Jetson Nano ≈4, Xavier NX ≈8, desktop unlimited). It
was set only in `configure_model`, from `_n_batchable_boxes()` = `_enabled` ∪
`_box_shapes`, and at Init both are empty, because the GUI enables pose *after*
loading the model and `_box_shapes` is populated by the frame handler for
pose-enabled boxes. So it was decided at the one moment the sink knows least
about the rig, resolved to 1, and was never asked again.

Measured on a 16-box rig, 8 seconds:

| | forward passes | batch | poses/s | per box | GPU busy |
|---|---|---|---|---|---|
| before | 2186 | all 1 | 273 | 17.1 | **96%** |
| after | 236 | all 16 | **472** | **29.5** | **29%** |

The camera delivers 30 fps, so before the fix roughly half of every box's frames
were being shed while the GPU sat saturated doing the same work sixteen times
over. `_refresh_batch_cap()` now runs on enable/disable, the moments the count
actually changes.

An engine that genuinely cannot take the larger batch (a TensorRT engine with a
baked batch axis) is not a hazard: `predict_batch` catches the refusal, falls
back to one frame at a time, and **remembers**, the sticky `_no_batch` flag
exists because retrying meant raising, logging and falling back thirty times a
second for the length of the session.

## 8d. The two cropping mechanisms are not variants of each other

| | ours (`crop_track`) | DeepLabCut-Live's (`dlc_dynamic`) |
|---|---|---|
| window | fixed, the model's training size | the keypoints' bounding box + margin |
| steered by | mean of the confident keypoints | whatever it last found |
| recovery | coarse grid scan, one window per frame | analyses the whole frame again |
| settings | `pose_crop_conf_min`, `pose_crop_good_min`, `pose_crop_reacquire` | `dlc_dynamic_threshold`, `dlc_dynamic_margin` |
| available to | SLEAP single-instance (written for it, the one family with no SDK equivalent) | **dlclive's PyTorch runner only** |

`dlc_dynamic` reaches nothing else: `dlclive.factory.build_runner` filters the
`dynamic` keyword out for the TensorFlow runners, and an exported ONNX/TensorRT
graph never goes through dlclive at all. The dialog therefore offers each mode
only where it does something, and the two settings groups are never shown
together, a follow-confidence typed under `dlc-dynamic` was read by nothing.

## 8e. Repairing a trace offline

`refine.py` works down one keypoint's timeline: confidence gate, jump limit, gap
fill. It cannot see the two errors that matter most, and `pose_repair.py` does:

* **a confident keypoint in the wrong place.** The confidence gate passes it
  (it is confident) and the jump limit often passes it (a snout mislabelled
  onto a nearby ear has not moved far). The body shape sees it. Detection is
  iterative trimming, one part per frame per round, and only when its residual
  stands clear of the *rest of its own frame*. Both halves are required: a
  mistrack drags its whole frame's fit, so testing each part against a fixed
  threshold condemns the innocent ones, and gating on the frame's median
  residual refuses to condemn anyone (the outlier inflates the median that
  licenses accusing it).
* **left/right swaps**, which keep the pose plausible and barely move the
  whole-frame residual. Tested against the canonical shape, per frame, a swap
  that *persisted* would be a naming convention, not an error.

Two limits worth knowing. `canonical_shape` learns from partly-observed frames
(EM Procrustes) because gating makes complete frames rare, the old version
needed 20 frames with every part visible and switched itself off without them.
And per-frame fitting has a breakdown point: once the corrupted keypoints
outnumber the good ones the fit follows them and every residual looks fine, so
`temporally_consistent` rejects frames whose fitted body pose leaps away from
their neighbours in time.

The smoother is a fixed-interval RTS (forward Kalman + backward pass) on a
constant-velocity model, replacing a centred Savitzky-Golay window. Both are
zero-lag; the smoother also weights each sample by its own confidence
and carries a velocity across gaps. One trap: the process noise **must** have
the measurement contribution subtracted before use. Differencing twice amplifies
independent errors sixfold, so the naive estimate on ~1 px of jitter came out
~30× the animal's real acceleration, which tells the filter the animal can go
anywhere and it then smooths nothing.

Measured on a synthetic six-part animal with injected faults (median / p95 /
max positional error, and the fraction of keypoints present):

| | present | median | p95 | max |
|---|---|---|---|---|
| raw | 88% | 1.30 px | 2.95 px | 58.0 px |
| old: savgol only | 100% | 0.78 px | 2.28 px | 50.1 px |
| + mistrack detection | 100% | 0.78 px | 2.15 px | 16.6 px |
| + swap correction | 100% | 0.76 px | 1.92 px | 12.9 px |
| + RTS smoother (default) | 100% | **0.58 px** | **1.42 px** | **11.2 px** |

---

## 8a. Event triggers: what each condition really does

`TrackingPushPolicy.evaluate` (`source/video/framebus/mcu_pusher.py`) is the
**single** evaluator, overlay, plot and MCU event all come from one pass, so
what you see annotated is what the board received.

**Every condition is edge-fired.** `fire = active and not previous_active`.
There is no level mode. That has two consequences worth knowing:

- `in_zone` and `enter_zone` are the *same rule under two names*, and so are
  `not_in_zone` and `exit_zone`. Both names stay - `in_zone` is the default and
  sits in saved configs, and the project takes no migrators, but the panel's
  hover help now says they are equivalent instead of implying a difference.
- A rule fires once per crossing. To require the animal to *stay*, use **Hold**
  (`duration_ms`), which gates how long the raw condition must hold before it
  counts as active.

**Orientation and posture need two keypoints.** `rotation_gt`, `rotation_lt`,
`facing_line`, `elongation_gt`, `rearing`, `head_angle_gt` and `head_angle_lt`
all derive heading from `FeatureTracker.axis = (tail, head)`. A tracker that
delivers only one point has no heading, so these rules can never fire, they sit
in the config looking configured and produce nothing. The tracking panel disables
them when the active tracker has no axis (`AXIS_ONLY_CONDITIONS` +
`_tracker_has_axis`), following the mode radios live.

Pinned by `source/tests/gui_tests/test_trigger_condition_semantics.py`, which
drives the real Pipeline → MCUPusher → board chain rather than asserting on the
UI: if the evaluator changes, the panel's gating and help are wrong and that
test says so.

---

## 8b. Why the suite leaks widgets, and what stops it

Qt widgets are C++ objects. A test that builds a dialog and returns does not
destroy it: `deleteLater` only *queues* a delete that never runs without an
event loop, and any widget something still holds, a module registry, a
signal connected to a long-lived object, the `Pipeline` it owns, stays alive
for the whole session.

Left unchecked this is quadratic, and it presents as a **hang with no failing
test**. Measured across the GUI suite before the fix:

| after test | top-level widgets | all widgets | live Pipelines |
|---|---|---|---|
| 125 | 0 | 0 | 0 |
| 250 | 1169 | 16 179 | 2 |
| 600 | 1375 | 19 875 | 2 |
| 1225 | 2549 | 35 188 | 48 |

Every leaked widget keeps its `Pipeline` alive, so the WeakSet behind
`shutdown_leaked_pipelines` grows too, and that fixture then re-shuts-down
every pipeline every earlier test leaked, after every later test. The run
degraded to minutes per test and never reached the end.

`destroy_widgets_created_by_this_test` (defined **last** in `conftest.py`, so
it tears down **first** and lets the pipelines fall out of the WeakSet on
their own) destroys any top-level widget that appeared during the test **and
fails the test**. A leak's cost is paid by some unrelated test much later, so
it has to be loud where it happens. The widgets are destroyed *before* the
assertion, so a failure costs one test rather than the run.

Only classes defined under `source.`/`tools.` are blamed. Qt makes its own
parentless `QFrame`/`QWidget` top-levels (combo popups, tooltip frames) that
no test can control; those are destroyed silently. Opt out with
`@pytest.mark.leaks_widgets("why")`.

Identity is the shiboken C++ pointer, not the Python wrapper. PySide hands
out a fresh wrapper per call, so comparing wrappers marks every widget as new.
It never calls `close()`; see the trap above. `test_widget_leak_net.py` is the
regression test.

### Disposing a widget in a test

Use `source/tests/qt_dispose.py`:

```python
from source.tests.qt_dispose import dispose, WidgetBin

dispose(widget)                       # destroy; does NOT close
dispose(roi_dialog, close=True)       # only when you want its closeEvent
```

The obvious incantation **does not work** and was used repo-wide for a long
time:

```python
w.close(); w.deleteLater(); QApplication.processEvents()   # still leaks
```

`deleteLater` posts a `DeferredDelete`, which Qt only delivers when the event
loop unwinds to the level the delete was requested from. A test has no running
loop, so `processEvents` never delivers it. `sendPostedEvents(None,
DeferredDelete)` is the call that does, and it is what `dispose` adds.

For a module-level helper that builds widgets and has nowhere to register
cleanup, `WidgetBin` plus one autouse fixture covers the whole file, see its
docstring.

---

## 9. Verifying

```bash
python -m pytest                # host suite + the analyzer's own tests
python -m ruff check source     # line length 100; E,F,W,I,UP,B,SIM,RUF
mypy source                     # strict
python setup.py build_ext --inplace --force
```

Cython is optional, every compiled module has a pure-Python fallback chosen at
import time.

**Unit tests are necessary and not enough for UI or flow work.** Every defect
found in the camera dialog was a *sequence* bug, connect, change the id,
disconnect, which single-call tests cannot see. Launch the app and walk the
actual flow before reporting that a feature works end to end.
