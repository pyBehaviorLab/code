# Projects

A project is a folder. Every saved-state-related file for one rig + one experimental
program lives there.

```{figure} /_static/media/gui/experiment-info-sidebar.png
:alt: The experiment information sidebar with the project's metadata fields
:width: 100%

The experiment sidebar. What is typed here is written into the project and travels with every session recorded under it.
```

## The two buttons

There is no "New Project" button and no "Open Project" button. The whole of
project handling is **Save** and **Load**, in the Experiment group.

```{figure} /_static/media/gui/group-experiment.png
:alt: The Experiment group with two buttons, Save and Load

**Save** and **Load** are the entire interface. Which of the three things
below happens depends on what you pick in the file dialog, not on which button
you press.
```

## Create a project

The application starts as a **draft**. Boxes, cameras and tracking can all be
set up before a project exists: the snapshot store runs in a temporary folder
and nothing is written to `experiments/projects/` yet.

Press **Save**. Because no project is active, you get a save dialog; type a
name in the filename field. The name you type becomes the folder, and the
draft workspace moves into it.

That first save writes:

- `experiment_config.json`, the rig itself
- `template.json`, the same config minus the identifying metadata
- `runs/`, ready for the first Record
- `data/<project>/`, so the recording target exists before you press Record

Everything else appears when first needed: `source/` when a task or hardware
definition is captured, `background_images/` on the first reference frame,
`metadata/` only if you attach a cohort sheet, `_pending/` while a capture is
in flight.

Saving is allowed with boxes that have no camera, no port and no board. The
one refusal is **zero boxes**.

Later saves are silent: no dialog, no prompt, both files rewritten in place.

## Create a project from a template

Every save also writes `template.json` beside the config, so **every project is
already a template**. It is the same file with the identifying fields removed:
`experiment_name`, `created_at`, `last_modified` and `config_djb2` at the top,
and `experimenter`, `project`, `session_label`, `data_dir`, `metadata_file`
and `created_at` from the metadata. What survives is the structure: boxes,
cameras, tracking, zones, stats.

To start a new project from one:

1. Press **Load**.
2. Navigate into the project you want to copy and pick its **`template.json`**,
   not its `experiment_config.json`.
3. A folder dialog opens. The name you give becomes the new project's name.
4. The new project is created and opened, and a dialog tells you the full path
   it landed at.

The same dispatch drives both, which is why there is no separate button: the
file you pick decides.

It refuses, with a dialog rather than silently, when the template was written
by an app with a different schema major version, when its `mode` does not match
the running app (an operant template in pyMaze), or when a project of that name
already exists at that location.

## Open an existing project

Press **Load** and pick an `experiment_config.json`. The picker dispatches on
the filename:

- `experiment_config.json` loads it as the active project
- `template.json` starts a new project from it, as above
- anything else warns and does nothing

On load the GUI:
1. Reads the JSON and stamps a `ProjectFileGuard` for later OCC saves.
2. **Refuses the file** if its `schema_version` has a different major version, or if its
   `mode` doesn't match the running app, the file is left untouched.
3. **Auto-recovers wiped tracking config** from `source/configs/<hex>.json`
   snapshots if needed (see [snapshot-tracking](../concepts/snapshot-tracking.md)).
4. Builds one box widget per `setup_config.boxes` entry.
5. Bridges every cfg field into the matching widget via
   `apply_config_to_ui(cfg, host)`.
6. Installs per-box `TrackingConfig`s into the pipeline.
7. Restores `_mcu_serial` per widget (auto-upgrades legacy `com_port` if needed).
8. Re-applies zone configs into the renderer + push policy.
9. Re-loads the cohort sheet named by `cfg.meta.metadata_file`, if present.

## Save

- **Explicit Save** (toolbar) writes a fresh snapshot to `source/configs/<hex>.json`,
  refreshes `template.json`, and appends to `change_log.jsonl`.
- **Autosave** runs debounced at 750 ms on any `_project_changed` trigger
  (box wiring change, dialog Apply, camera config change, bg capture, etc.).
  Autosave does NOT touch the snapshot store, only `experiment_config.json`.

Both paths go through `read_ui_into_config(host) → save_experiment(cfg, …)`.

## Where everything is saved

Two trees, kept apart on purpose. The project folder holds the setup and stays small enough to
copy, back up or hand to a colleague. The `data/` tree holds the sessions and grows without bound.

```
code/                                  the repo root, paths.py::top_dir
  tasks/<Family>/                      task .py, its variables sidecar, its stats config.json
  hardware_definitions/                the wiring, one .py per board layout
  devices/                             per-peripheral MicroPython drivers
  api_classes/                         host-side adaptive logic
  models/                              DLC and SLEAP model folders
  experiments/
    config/settings.json               app-wide defaults
    config/stats_templates/            stats configs to start from
    projects/<project>/                one folder per rig + program
  data/                                every session ever recorded
```

Inside one project folder:

```
experiments/projects/DKLab_aCG_reversal_learning_4box/
  experiment_config.json               the rig: boxes, cameras, boards, tracking, zones
  template.json                        the same file minus the identifying metadata
  change_log.jsonl                     config edits and source captures, one timeline
  metadata/MetaData.xlsx               the cohort sheet, copied in on first attach
  background_images/box<N>.png         reference frames for the blob tracker
  runs/2026-09-08.json                 one row per Record press, per box, that day
  ReversalLearning/                    one folder per task family
    persistent_variables.json          remembered variable values for this project
  source/                              pinned copies of everything a run used
    index.jsonl                        one line per commit
    <djb2>.py                          task and api_class snapshots
    hd/<djb2>.py                       hardware definitions
    devices/<djb2>.py                  drivers
    configs/<djb2>.json                the config as it stood at that save
  _pending/                            transient staging for a capture in flight
```

`source/` is the reason a result stays reproducible. Every file that reached a board is copied in
under the djb2 hash of its own content, and the run row and the session file headers carry those
same hashes, so the exact task and wiring behind a six-month-old session can be recovered even
after the working copies have moved on. Nothing there is meant to be edited; `_pending/` is swept
of orphans at project load. See [snapshot tracking](../concepts/snapshot-tracking.md).

### Where the recordings go

Sessions land in the global `data/` tree, not in the project:

```
data/<project>/<task>/<YYYY-MM-DD>/
  mcu/    39-Box1-2026-09-08-132130.tsv                 the board's own record
  video/  39-Box1-2026-09-08-132130.mp4                 the recording
          39-Box1-2026-09-08-132130_video_data.txt      one row per camera frame
          _drops.tsv                                    any frame the camera lost
```

Both files of a run share the stem `<subject>-Box<N>-<YYYY-MM-DD-HHMMSS>`, so a session is one
`ls` away from being complete or visibly not. A run with no subject set is a dry run: it writes to
`data/temp/` under fixed names that the next dry run overwrites, and it appends no row to
`runs/`.

`_drops.tsv` is created lazily, so a session with no dropped frames writes no file at all.

### What the data looks like

The MCU `.tsv` is what the task itself wrote, four tab-separated fields per row, times in seconds
from the board's own clock:

```
0.000   info      task_name           ReversalLearning
0.000   info      task_file_hash      ead04e1f
0.000   state                         init_trial
6.300   event     user                poke_3
6.301   state                         choice_state
6.480   print     task                Correct_response
```

The `_video_data.txt` is one row per camera frame, nine tab-separated columns, and it carries the
board's time for that frame, which is what joins it to the `.tsv` above:

```
#columns  frame  elapsed  capture_host_ns  frame_fw_ms  pose_lag_ms  filter_ms  pose  zone  state
1         00:00.033   35742119033   16    41.7   0.42   [[121,88,0.99],…]   Left_poke   init_trial
2         00:00.066   35742152366   49    41.7   0.42   [[122,88,0.99],…]   Left_poke   -
3         00:00.100   35742185699   82    na     na     na                  na          choice_state
```

`capture_host_ns` is the raw host instant the frame was stamped with, before
`elapsed` rounded it to a millisecond. It is also the only key that matches
across boxes on a shared camera: `frame` is numbered from each box's own first
frame, and the boxes start recording up to a second or two apart.

Join the two on `frame_fw_ms`, never on the frame index: the video is written at a constant
nominal rate while the camera acquires at its own slightly different one, so an index does not
convert linearly to time. Every column, every header line and every other file the app writes is
in the [file-format reference](../reference/file-formats.md).

## What survives save / load

| Field | Where |
|---|---|
| Rig wiring (boxes, cameras, MCUs, ROI, geometry) | `cfg.setup_config.boxes[]` |
| Tracking mode + model + body parts + push gates | `cfg.tracking` |
| Per-box zones | `cfg.setup_config.boxes[].zones` |
| Stats config path | `cfg.stats_config` |
| Operator-facing metadata (experimenter, project label) | `cfg.meta` |
| Dialog overrides (coord_mapping, triggers, annotate_parts, marker_size) | `cfg.ui.dialog_overrides` |
| Last-active box / panel expansion | `cfg.ui` |

What's NOT in the project (= per-run):
- subject_id
- task name (the upload selection)
- task_hash, hd_hash (those live in the run row + the data file headers)

See [project-vs-run](../concepts/project-vs-run.md).

## Multi-instance safety

`source/config/multi_instance.py`, one module by intent, stops two GUIs editing the same
project file from clobbering each other:

- `write_atomic(path, payload)`, tmp file + `os.replace`, so a save is never half-written
- `file_lock(path, timeout)`, cross-process exclusive lock via a `.lock` sidecar
- `ProjectFileGuard.stamp(path)` at load, `guard.save_json(obj, on_conflict=…)` at save:
  if the on-disk file is unchanged, write; if it changed but the edits are disjoint,
  three-way merge silently; if any field genuinely conflicts, prompt the user to
  overwrite / reload / cancel

Each GUI process also writes to its own `_pid<N>.log`, so concurrent instances never
interleave into one log file.

## Project autosave + cohort metadata

If a cohort Excel sheet is attached (`cfg.meta.metadata_file`), it's copied INTO
`<project>/metadata/` on first attach. From then on, references are relative to the
project. Cohort changes trigger `_project_changed(reason="metadata_loaded")` which
debounces an autosave.

The metadata file requires two columns: **`Subject`** and **`SetupID`**. Header names are
trimmed of surrounding whitespace but matched **case-sensitively** - `subject` or `setupid`
will be rejected. `SetupID` is coerced to int, `Subject` to str.

The cohort row populates per-box `subject_id` via the metadata-assign dialog (sidebar
button + per-box subject combo, sharing the same modal).
