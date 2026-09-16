# File formats

Every file the project writes, row-by-row.

## `experiment_config.json`

Schema v3.0. Top-level keys:

```
schema_version      "3.0"
mode                "operant" | "maze"
experiment_name     str
created_at          "YYYY-MM-DD-HHMMSS"
last_modified       "YYYY-MM-DD-HHMMSS"
config_djb2         8-hex self-hash (this field excluded from its own input)
meta                Meta dataclass
setup_config        SetupConfig {boxes: List[BoxConfig]}
cameras             CamerasConfig {video_defaults, registry}
tracking            TrackingConfig (rig-level)
stats_config        StatsConfig
ui                  UIState (NOT included in config_djb2)
```

### `setup_config.boxes[]`

```
box_number        int
mcu_serial        str       USB serial of the microcontroller (e.g. "315535563234")
com_port          str       legacy device path; back-compat fallback only
init_hw_def       FileRef   {name, path, djb2, size_bytes}
camera_id         str       "" = no camera
geometry          {x, y, w, h}  pixel rect on the camera frame
roi_normalized    [x, y, w, h]  in 0-1
save_video        bool
bg_captured_at    str       "" or "YYYY-MM-DD-HHMMSS"
tracking_enabled  bool      per-box on/off
save_tracking     bool      save annotated video
zones             List[Zone]
action_config     FileRef
api_class         FileRef   optional, per-task adaptive logic
```

### `tracking` (rig-level)

```
enabled                bool
mode                   "none" | "dlc" | "sleap"
dlc                    DLCConfig  (D8: {} when mode != dlc)
sleap                  SleapConfig (D8: {} when mode != sleap)
push_zones_to_mcu      bool   default true, fire `zone_changed` on the MCU
push_coords_to_mcu     bool   default true, push c.* every frame
push_frame_event       bool   default FALSE, opt-in per-frame `frame_event`
zone_change_body_part  str    "centroid" | <keypoint_name>
```

`DLCConfig`:
```
model_djb2     str    filesystem path (legacy name)
body_parts     List[str]
confidence     float
resize         float
instances      int
marker_size    int
zone_body_part str
```

### `ui.dialog_overrides`

```
coord_mapping     {coord_name: body_part_name}
triggers          [{condition, body_part, zones, event_name, threshold}]
annotate_parts    [body_part_name..]
zone_body_part    str  (legacy mirror of tracking.zone_change_body_part)
marker_size       int
annotate_saved    {box_id_str: bool}
task_name         str
tracker_type      str
scale             {box_id_str: {value, unit}}
```

## `_video_data.txt` (v2)

```
# header (one per session, lazily written on first frame)
#version           3
#session_id        <subject>_box<N>_<YYYY-MM-DD-HHMMSS>
#session_start     YYYY-MM-DD HH:MM:SS.fff
#box_id            int
#video_file        <stem>.mp4
#camera            {"resolution":"WxH","fps":F,"roi":[x,y,w,h]}
#video_codec       {"encoder":"…","container":"mp4"}
#pycontrol         {"task":"…","subject":"…","task_hash":"…","hd_hash":"…"}
#zone_config_begin
{ "scale":…, "arena":…, "zones":{…} }
#zone_config_end
#tracker           {"backend":"sleap","model_type":"…","tracker":"…","n_animals":N,
                    "identities":[…],"bodyparts":[…],"skeleton":[…]}
#sync              {"source":"mcu_event_stream"}
#units             {"speed":"m/s"}    # or "px/s" when no calibration
#columns           frame elapsed capture_host_ns frame_fw_ms pose_lag_ms
                   filter_ms pose zone state
# ============================================================================
```

The `#tracker` block is what makes the file self-describing: a reader knows which
backend produced the `pose` column, and which body parts its entries are, without
needing the model folder. Only the keys that apply are written.

Rows (one per camera frame, **8** tab-separated columns):

```
frame         1-based, recording-local
elapsed       mm:ss.fff (hh:mm:ss.fff if ≥1h) since session_start, at capture
frame_fw_ms   the board's own time at this frame's capture, written out raw, or "na".
              The column every later alignment to the task is built on.
pose_lag_ms   ms from this frame being CAPTURED to its pose existing: the whole
              journey (tick, queue, fan-out, model), not the model alone, or "na"
filter_ms     how much of pose_lag_ms the Kalman and optical-flow smoothing took,
              over every keypoint, or "na"
pose          JSON [[x,y,c],…]; a per-identity map {"male":[[x,y,c],…],…} when n_animals > 1; or "na"
zone          the zone the CENTROID is in, or "na"
state         pipe-joined state names that landed this interval, or "-"
```

Read `zone` literally: with small zones such as pokes the centroid stays outside
while the snout is well inside.

`pose_lag_ms` is the cost of getting position back to the task, not a model benchmark. Measured on
this machine the model is 3.7 ms of a 30.6 ms lag; the rest is the tick, the
queue and the fan-out. The column was called `infer_ms`, which invited exactly
the wrong comparison.

Five columns were deliberately cut, and the reason each one went is worth knowing
before you go looking for it:

| gone | where it lives now |
|---|---|
| `speed` | follows from the pose and the scale calibration; the analyser recomputes it |
| `events` | the task's own events, which the MCU TSV records and is the canonical account of |
| `pose_fw_ms` | the pose's age, now stated outright as `pose_lag_ms` instead of left as a subtraction |
| `track` | a blob-tracker state that says nothing on a pose rig |
| `part_zones`, `filled_parts` | per-part membership and which parts were predicted; both belong with the analysis |

Rows for frames the tracker did not reach carry no pose, so **absence is visible,
never interpolated**.

Footer on close: `#session_end YYYY-MM-DD HH:MM:SS.fff  total_frames=N  dropped_frames=N  total_pose_inferences=N`

:::{admonition} Use frame_fw_ms, not the frame index
:class: warning

The video is written at a constant nominal rate while the camera acquires at its
own slightly different rate, so the encoder resamples and a frame index does not
convert linearly to time, locally by up to several hundred frames in measured
recordings, even when correct on average. `frame_fw_ms` is the controller stamp
for that frame and is what joins this file to the MCU TSV. See
[Performance and validation](performance.md#agreement-between-the-pose-record-and-the-controller-record).
:::

## `_drops.tsv`

Always on, written next to the video. One row per dropped frame, six
tab-separated columns; the file is created lazily on the first drop, so **a
session with no drops writes no file**.

```
wall_iso       ISO wall-clock time of the drop
monotonic_ms   ms since the drop recorder started
stream         recorder | tracker | camera_ring | mcu_push | <other site>
frame_idx      monotonic counter from the dropping subsystem, or "na"
capture_ts     camera capture timestamp in seconds, or "na"
reason         short tag, e.g. queue_full, inflight_box3
```

This is the accounting behind the losslessness claim. The recorder never
discards **silently** and never evicts a frame already queued, but it can still
drop one if its 90-frame buffer is exhausted; that appears here as
`lossless_overflow` and raises a health alarm. Treat a `recorder` row as a fault
to investigate. A `tracker` or `pose` row is normal: those consumers skip frames
by design to stay current. The per-session total also appears in the
`_video_data.txt` footer as `dropped_frames=N`, so a count is available without
opening this file.

## MCU TSV (`.tsv` written by pyControl data_logger)

```
time    type      subtype  content
0.000   info      task_name           ReversalLearning
0.000   info      task_file_hash      af425e28
0.000   info      hardware_def_hash   00000000
0.000   info      subject_id          A31
0.000   variable  run_start           {"…":…}
0.000   state                          init_trial
6.300   event     user                poke_3
6.301   state                          choice_state
…
```

Every row is exactly four tab-separated fields. `time` is `fw.current_time`. MCU framework
time since the run started, written as **seconds with three decimals** (`f"{ms/1000:.3f}"`),
not as milliseconds. Tabs and newlines inside `content` are escaped, so a row never breaks
the column count.

Row types: `info`, `state`, `event`, `print`, `variable`, `threshold`, `warning`, `error`.
The `subtype` column carries the source - `input`/`timer`/`user`/`api`/`publish`/`sync` for
events, `task`/`api`/`user`/`trigger` for prints, and
`run_start`/`run_end`/`user_set`/`api_set`/`get`/`print` for variables.

The header also carries a `devices` line holding a JSON dict of `{driver_file.py: djb2hex}`;
together with `task_file_hash` and `hardware_def_hash` those hex values name the snapshot
files under `<project>/source/`, `source/hd/` and `source/devices/`.

## `runs/YYYY-MM-DD.json`

One file per day. When the task family is known the file sits in a per-family subfolder
(`runs/<task_family>/YYYY-MM-DD.json`); otherwise it lands directly in `runs/`, there is
no `_loose` placeholder directory. Each file is `{"date": "<iso>", "runs": [..]}`.

One row per Record press per box that day. Dry runs (empty `subject_id`) write no row at
all. The 6-field row (anything already in
the MCU TSV header or derivable from the pinned config snapshot is deliberately
excluded - `config_djb2` points at `source/configs/<djb2>.json`):

```json
{
  "id":          "2026-06-15-175911_box1",
  "subject_id":  "A31",
  "box_number":  1,
  "config_djb2": "af425e28",
  "mcu_tsv":     "",
  "video_mp4":   null
}
```

`mcu_tsv` / `video_mp4` are filled in post-hoc by `close_run` once the artefacts
are finalised; timing comes from the MCU TSV header, not stored here.

## `change_log.jsonl`

Append-only, one JSON object per line. Single writer: `source/config/history.py`
(`append_change` for config edits, `append_event` for source captures). Every line
carries a `ts` (`%Y-%m-%d %H:%M:%S`); the rest of the schema varies by discriminator
(`action` for config edits, `kind` for captures):

```text
{"ts":"..","actor":"user","action":"save","source":"..","path":"..","old":"<djb2>","new":"<djb2>"}
{"ts":"..","kind":"upload","source":"task","box_id":1,"label":"..","djb2":"..","tier":"new","new":true}
{"ts":"..","kind":"commit","source":"task","box_id":1,"djb2":"..","name":".."}
{"ts":"..","kind":"dlc_capture","djb2":"..","new":true}
```

## `<project>/<task_family>/persistent_variables.json`

Per-task, per-project persisted variable values. **Keyed by variable name only, not by
subject.**

```json
{
  "flags":  {"stage": true, "n_trials": false},
  "values": {"stage": 3, "ITI_duration": 1000}
}
```

`flags` is the project-specific persistent/reset classification; it overrides the per-task
sidecar below on a per-name basis. `values` holds the last-run value of each persistent
variable, captured in one `get_variables()` round-trip at Stop and pushed back at the next
Upload. Only flat scalars are stored, non-scalar values are dropped.

Writes go through a cross-process `file_lock` on a `.json.lock` sidecar, then an atomic
tmp + replace.

:::{note}
Older files may still carry the pre-3.0 shape `{"persistent": [..], "values": {"<subject>":
{..}}}`. The current reader ignores both the `persistent` key and the subject nesting, and
rewrites the file in the `{flags, values}` shape on the next capture.
:::

## Task variable spec sidecar

`tasks/<Task>/<Task>.variables.json`, sits next to the task `.py`. **Per-task**
(shared across subjects and projects): which variables persist across sessions. There is
no `reset` flag, a variable absent from the list, or listed with `persistent: false`,
resets to its task-file default on the next Upload. Loaded/saved by
`source/config/task_variables.py`.

```json
{
  "task_hash": 1234567890,
  "task_path": "Reversal/Reversal",
  "saved_at": "2026-05-13 18:42:11",
  "variables": [
    {"name": "n_trials", "persistent": false},
    {"name": "weight",   "persistent": true}
  ]
}
```

`task_hash` is the signed int from the 4-byte-LE djb2 of the task file, and `saved_at` is
`%Y-%m-%d %H:%M:%S`. A row is exactly `{name, persistent}`, older files may carry an extra
`user_default` key, which nothing reads and which is dropped on the next write.

The spec is per-task; the remembered *values* live in `persistent_variables.json` above,
and a project can override the persistent/reset classification per name via that file's
`flags`.
