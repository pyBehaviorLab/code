# Recording

One Record button per box. Pressing it starts: MCU framework → camera capture →
video writer → tracking writer → MCU push subsystems. Stop reverses cleanly.

```{figure} /_static/media/gui/operant-live-status.png
:alt: The Live Status tab, one panel per box with its elapsed time and state
:width: 100%

**Live Status** while boxes are running: one panel per box with its elapsed time and the state the task is in.
```

## Pre-flight checks

Per-box Record is enabled only when:

- MCU connected
- Task uploaded
- Camera connected (if `tracking_enabled` or `save_video`)
- `subject_id` set on the box (or operator confirmed empty = dry run)

`source/gui/widgets/run_task.py` orchestrates. `subject_id == ""` → dry run, no data
written to disk (framework still runs, used for hardware testing).

## What gets written per Record

Each Record creates one row in `runs/YYYY-MM-DD.json` and these files in
`<project>/data/<task>/<YYYY-MM-DD>/`:

```
mcu/<subject>-Box<N>-<YYYY-MM-DD-HHMMSS>.tsv     ← pyControl event/state/print log
mcu/<subject>-Box<N>-<YYYY-MM-DD-HHMMSS>.log     ← per-session text log
video/<subject>-Box<N>-<YYYY-MM-DD-HHMMSS>.mp4   ← recorded video
video/<subject>-Box<N>-<YYYY-MM-DD-HHMMSS>_video_data.txt   ← per-frame log
```

`build_session_stem(subject_id, box_id, start_dt)` is the single source of truth for the
filename pattern, all four files use the same stem.

## The `_video_data.txt` row format (v3)

One row per camera frame, **9** tab-separated columns:

```
#columns  frame elapsed capture_host_ns frame_fw_ms pose_lag_ms filter_ms pose zone state
190  00:06.331  35742119033  6255  30.6  0.42  [[173.2,140.1,0.85]]  R1  choice_state
```

Every column is defined once, in
[File formats](../reference/file-formats.md#_video_datatxt-v2), including the
self-describing `#tracker` header block and the footer totals. Two points matter
while recording:

- `frame_fw_ms` is the controller framework time at frame capture, and is what
  joins this file to the MCU TSV. **Do not convert a frame index to time**, the
  encoder writes at a constant nominal rate while the camera acquires at its own,
  so an index is right on average and wrong locally.
- `pose_lag_ms` is how old that row's pose is, measured from the frame's own
  capture: the tick, the queue, the fan-out and the model together, not the model
  alone. `filter_ms` is the smoothing's share of it. A row with no pose carries
  `na` in all three, so a gap is visible rather than interpolated.

See [time-sync](../concepts/time-sync.md) for why `frame_fw_ms` is an estimate
and how to cross-reference authoritative event timing via the MCU TSV.

## The run row

```json
{
  "start_ms": 1748627951000,
  "end_ms":   1748627991000,
  "subject_id": "A31",
  "task": "ReversalLearning/ReversalLearning",
  "task_hash": "af425e28",
  "hd_hash":   "00000000"
}
```

6 fields. The "minimal runs json" rule (`feedback_runs_json_minimal` in memory):
anything derivable from the MCU TSV header or the project snapshot is **forbidden** in
this row.

## Multi-box record

Master Record button (toolbar) fans out per-box record. `ParallelStartCoordinator`
serialises the MCU framework starts (one at a time per box, because the microcontroller's
USB-CDC serial doesn't multiplex). Camera + recorder start parallel.

## Stop

Per-box Stop or master Stop. Each box:

1. Sends `b'\x03'` to MCU to stop the framework.
2. Drains remaining MCU data into the TSV.
3. Closes the recorder (flushes mp4 footer + video txt footer).
4. Captures any subject-persistent variables and writes to
   `<project>/<task>/persistent_variables.json`.
5. Closes the per-session `.log`.

## Annotated video

Per-box "Save annotated video" checkbox (in the tracking dialog), when on,
RecorderSink writes the overlay (zones + pose dots + ROI) baked into the mp4
in addition to the raw stream. Off → raw stream only.
