# Project vs Run

A clean split between **what's permanent** (project) and **what's per-session** (run).

## Project = rig + setup

Lives in `experiment_config.json`. Captures rig invariants:

- Per-box wiring: `mcu_serial`, `camera_id`, `init_hw_def`, geometry, ROI, zones
- Rig-level tracking: mode, model, body parts, push gates, confidence/resize/instances
- Stats template path
- Metadata sidebar values

**Edited often** during setup, **rarely** once the rig is calibrated. Round-trips through
`Config.from_dict` ↔ `Config.to_compact` (see [reference/file-formats](../reference/file-formats.md)).

## Run = one Record press, per box

Lives in `runs/YYYY-MM-DD.json` (or `runs/<task_family>/YYYY-MM-DD.json` when the family is
known). The row is deliberately **six fields**, anything already recorded in the MCU TSV
header, or derivable from the pinned config snapshot, is excluded on purpose:

- `id` - `YYYY-MM-DD-HHMMSS_box<N>`
- `subject_id` (only set on the row, not on the project)
- `box_number`
- `config_djb2`, points at `source/configs/<djb2>.json`
- `mcu_tsv`, `video_mp4`, filled in post-hoc by `close_run`

Task name, `task_hash`, `hd_hash` and the session timings are **not** stored here; they live
in the MCU TSV header. A dry run (empty `subject_id`) writes no row at all.

The 6-field row format is enforced, see `feedback_runs_json_minimal` in the codebase.
Anything derivable from the MCU TSV header or the project snapshot is **forbidden** from
the run row.

## What's stripped at save time

`_box_to_compact_dict` and `to_dict_for_hash` explicitly drop:

- `subject_id`, run-only
- `task`, run-only

This is why opening the same project, recording two subjects, then closing doesn't
leave the last subject_id hanging in `experiment_config.json`.

## Why this matters

A clean split lets you:

- Reuse a project across many subjects + sessions without copy-paste.
- Hash the project for lineage without subject_id polluting the hash.
- Reconstruct exactly what ran on day X by joining the run row → project snapshot
  → task snapshot.
