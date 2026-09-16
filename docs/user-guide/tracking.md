# Tracking

Two backends: **DeepLabCut** and **SLEAP**. Both are configured through the
unified tracking dialog (toolbar Tracking button).

```{figure} /_static/media/gui/tracking-mode.png
:alt: The tracking settings tab showing the Blob, Simple, DLC and SLEAP modes
:width: 100%

**Tracking Config → Tracking Settings**. Blob needs no model and no GPU; DLC and SLEAP need both.
```

## Backend picker

Two chips at the top of the panel:

- **DLC**. DeepLabCut model loaded via `deeplabcut-live`. Pick the exported model
  folder. Body parts auto-loaded from the model.
- **SLEAP**. See the SLEAP notes below. Same panel flow as DLC.

## SLEAP

SLEAP is handled by `source/video/tracking/pose.py::SLEAPTracker`, built on **sleap-nn**
(the PyTorch rewrite, `pip install sleap-nn`) and falling back to **legacy SLEAP**
(TensorFlow, `pip install sleap`) when sleap-nn isn't installed.

### Model families (auto-detected)

The tracking dialog reads the model folder's `training_config.yaml`/`.json` and shows the
detected type as a badge next to the Model field:

| Type | Models | Use |
|---|---|---|
| **single-instance** | 1 folder | one animal per box/ROI, the rig default, fastest |
| **top-down** | 2 folders (**centroid** + centered-instance) | a few non-overlapping animals |
| **bottom-up** | 1 folder | crowded / occluded scenes (uses Part-Affinity Fields) |
| **multi-class** (top-down-id / bottom-up-id) | 1–2 | fixed identities predicted by the network |

For a **top-down** model the badge reveals a **Centroid model** picker, set it to the
paired centroid folder (both stages are required to run live). Single-instance and
bottom-up need only the one Model path. Body parts are auto-detected from the skeleton.

Browse opens a **folder** picker first (sleap-nn models are folders); press Cancel to get a
file picker for a legacy SLEAP `.zip`/`.pt`/`.pb`.

### Runtime & optimisation

The **Runtime** dropdown selects how inference runs, with an automatic
**TensorRT → ONNX → native** fallback if an engine or dependency is missing:

| Runtime | What it is | When |
|---|---|---|
| **native** | PyTorch (sleap-nn) | dev / desktop; combine with **fp16** (CUDA, ~1.5×) |
| **onnx** | exported ONNX engine | portable CPU/GPU |
| **tensorrt** | exported TensorRT engine | **Jetson / production, ~5×** faster |
| **auto** | prefer an exported engine, else native | default |

- **fp16** (CUDA only) and `torch.compile` are the native-path speed knobs; the tracker
  warms up a couple of frames on load so the first Record isn't cold.
- The **Export…** button pre-builds the ONNX/TensorRT engine for the chosen runtime and
  **caches it per-PC** (`<user config>/pybehaviorlab/sleap_exports/<hash>-<device>-<runtime>/`).
  TensorRT engines are **GPU-specific**, build them on the Jetson itself; the device is
  part of the cache key so a desktop engine is never reused on the Jetson.
- Multi-box: same-shape frames are batched into one forward pass automatically.

If neither package imports, init logs `pip install sleap-nn (or pip install sleap)` and the
box records video + the per-frame TXT (empty pose column) for offline scoring later.

### Multi-animal & identity (SLEAP only)

DLC-Live is single-instance. Multi-animal is a SLEAP capability, set by
**Animals** (`n_animals`) + **Identity**:

- **none**, single animal (default).
- **tracker**, online frame-to-frame association (OKS/IoU + Hungarian); IDs can swap on
  occlusion. A stateful tracker is kept per box.
- **id_model**, a multi-class model predicts a fixed identity per animal, so an
  identity survives an occlusion. Identities must be labelled in training.

See the data-format note below for how multiple animals appear in `_video_data.txt`.

## Per-box checkboxes

Boxes group at the top: tick which boxes use the configured tracker. Unchecked boxes
get the rig-level fields (model_path, body_parts) too, but `online_tracking_enabled`
stays False so they don't run inference at record time. This lets you configure once
and toggle per-box without re-picking the model.

## Confidence / Resize / Instances + Initialize

DLC params row:

- **Zone part**, single body-part picker (see below)
- **Confidence**, minimum keypoint confidence to accept (default 0.5)
- **Resize**, frame downscale factor for inference (0.5 = half-res, 2× faster)
- **Instances**, parallel inference workers per box (default 1)
- **Initialize**, load + warm the model

Init button colour code:
- Disabled (green) - `PoseSink.current_signature()` matches the configured one, model
  already loaded
- Enabled (warning yellow), settings changed since last init, re-init needed
- Enabled (red), last init failed

On Record: if `online_tracking_enabled` AND mode is pose AND signature mismatches,
operator gets a modal: **Initialize** / **Disable tracking for this run** / **Cancel**.

## The single body-part picker

`zone_body_part_combo` always shows `["centroid", *body_parts]`. Its selection drives
**both**:

1. **PoseSink centroid**, which keypoint defines the synthetic centroid used for the
   `zone` column in `_video_data.txt`, the `location` returned in pose results, and
   `zones_by_body_part["centroid"]`.
2. **MCUPusher zone-change diff target**, which keypoint's zone occupancy fires the
   intrinsic `zone_changed` event.

When `"centroid"` is selected (default), the legacy first-confident-keypoint heuristic
runs. When a specific keypoint is selected, only that keypoint's zone is reported,
if its confidence drops below threshold for a frame, `location = "na"` rather than
silently falling back to another keypoint.

## Push-to-MCU group

Two checkboxes (bottom right of dialog, next to Coord Mapping):

- **Zone-change events**, gate on the intrinsic `zone_changed` event being pushed
  to MCU. The event is auto-injected by `state_machine.setup_state_machine` (like
  entry/exit), the task doesn't have to declare it. Dispatched **silently** via
  the `b'Z'` wire byte (no MCU TSV row), matching entry/exit semantics. Off ⇒ no
  zone events at all.
- **c.* coordinates**, gate on per-zone coord mappings being pushed via
  `set_coordinates`. Each push writes `ut.c.<coord_name> = <zone_name_or_value>`
  on the MCU; silent (no TSV entry). Off ⇒ no coord pushes.

## Coord mapping (zone-derived)

Bottom-left of dialog. Maps `coord_name` → `body_part`. Each `coord_name` resolves to:

- `"speed"` → scalar speed (px/s)
- `"x"`, `"y"`, `"<name>_x"`, `"<name>_y"` → keypoint pixel coordinate
- anything else → name of the zone the body_part is currently in, or `""`

E.g. `{"loc_center": "centroid"}` writes `ut.c.loc_center = "R1"` whenever the
centroid is in zone R1.

## Triggers

A trigger turns a tracked quantity into a **named task event**, so the state
machine handles it exactly as it handles a nose-poke. Configured in the **Event
Triggers** table, pick a condition, a body part, the zones, an event name, and
optionally a threshold and a minimum hold time. Fired via
`pycboard.queue_trigger_event(event_name)` and logged in the MCU TSV.

Each rule fires when its condition **becomes** true, not continuously while it
stays true.

### Conditions

| Condition | Fires on | Needs |
|---|---|---|
| `in_zone`, `enter_zone` | Body part inside any listed zone | zones |
| `not_in_zone`, `exit_zone` | Body part outside every listed zone | zones |
| `exit_edge` | The true→false transition of zone occupancy | zones |
| `speed_gt`, `speed_lt` | Speed above / below threshold | threshold |
| `freezing` | Speed below threshold (default 1.0) |, |
| `rotation_gt`, `rotation_lt` | Turn rate (deg/s) above / below threshold | threshold |
| `head_angle_gt`, `head_angle_lt` | Head-to-body angle above / below threshold | a `neck` part |
| `facing_line` | Heading within threshold of a target direction (default 30°) | a `target` point |
| `elongation_gt` | Body elongation above threshold | threshold |
| `rearing` | Foreshortened **and** slow, elongation below `threshold` (default 0.6) with speed under `speed_max` |, |
| `distance_gt`, `distance_lt` | Distance between two keypoints | `part_a`, `part_b` |

Thresholds carry a `unit` (`px`, `mm`, `cm` or `bodylen`); metric units require
the scale line drawn in the zone editor.

**Minimum hold time.** `duration_ms` requires the raw condition to hold for that
long before the rule goes active, this is what separates a genuine zone entry
from a transient excursion across a boundary. Leave it at 0 for instantaneous
firing.

:::{admonition} Two conditions that need more than the table gives them
:class: warning

`facing_line` needs a `target` point, which **no editor currently authors**, a
rule without one is disabled and logs a warning once, rather than silently
measuring the angle to the frame origin. Author `target` in the trigger JSON if
you need it.

Conditions requiring an orientation axis - `head_angle_*`, `facing_line`,
`elongation_gt`, `rearing`, need at least two keypoints, so a model tracking a
single point cannot evaluate them.
:::

See [Set up cameras, tracking and zones](setup-cameras-tracking-zones.md) for the
flow, and [Writing tasks](../tasks/writing-tasks.md) for handling the event on
the board.

## How tracking appears in `_video_data.txt`

Format **version 2**, self-describing. One row per camera frame; the header
describes how to read the pose column so a script never has to guess which
tracker produced it. The full column list lives in
[File formats](../reference/file-formats.md#_video_datatxt-v2), this section
covers only what the tracker choice changes.

- **`#tracker` header** records `{backend, model_type, tracker, n_animals, identities,
  bodyparts}`, e.g. `{"backend":"sleap","model_type":"topdown","n_animals":2,
  "identities":["male","female"],…}`. DLC → `{"backend":"dlc","bodyparts":[…]}`.
- **`pose` column**, single-animal is a flat array `[[x,y,c],…]` (unchanged / byte-compatible).
  When `n_animals > 1` it becomes a per-track map `{"male":[[x,y,c],…],"female":[…]}` keyed
  by identity. `frame_fw_ms` still pairs each row 1:1 with the MCU TSV, so frames are never
  split across rows.
- Timing (`frame_fw_ms`, `pose_lag_ms`, `filter_ms`), `zone` and `state` columns are
  unchanged; the focal (first) track drives the `zone` column.

The model tree is hashed (djb2) into the run snapshot, so a recording records exactly which
model produced its pose.

## What persists in the project file

| Field | Lives in |
|---|---|
| Rig-level mode + model_path + body_parts + confidence + resize + instances | `cfg.tracking.dlc` / `.sleap` |
| Push gates + body-part picker | `cfg.tracking.push_zones_to_mcu`, `push_coords_to_mcu`, `zone_change_body_part` |
| Per-box zones | `cfg.setup_config.boxes[].zones` |
| Per-box "tracking enabled" + "save annotated" | `cfg.setup_config.boxes[].tracking_enabled` / `.save_tracking` |
| Coord mapping + triggers + annotate_parts + marker_size | `cfg.ui.dialog_overrides` |

See [concepts/tracking-pipeline](../concepts/tracking-pipeline.md) for the end-to-end
data flow.
