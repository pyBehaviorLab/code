# API overview

Most people never touch a Python API here. An experiment is configured in the
interface and stored in the project file; the only code you write is a **task**
and a **hardware definition**, and both are uploaded to the microcontroller.

This page lists the surfaces in the order you are likely to meet them. The
module-level reference at the bottom is for people modifying the application
itself.

## 1 · What a task file can call

This is the API for writing an experiment. Full reference:
[Task API](../tasks/task-api.md).

| Surface | What it is |
|---|---|
| `states`, `events` | The two lists at the top of a task file that declare its state machine |
| `goto_state(name)` | Move to another state |
| `set_timer(event, ms)`, `timed_goto_state(state, ms)` | Schedule an event or a transition |
| `v.<name>` | Session variables. Readable and settable live from the interface |
| `hw.<name>` | The devices named in the hardware definition |
| `print(..)` | Writes a line into the session file; the statistics tab is computed from these |
| `stop_framework()` | End the session from within the task |

Time is in **milliseconds** throughout. `second`, `minute` and `hour` are
provided as multipliers.

## 2 · What tracking adds to a task

Present only when a camera and a tracker are configured. Nothing here needs new
task syntax. A zone entry arrives as a logged event, like any other.

| Surface | What it is |
|---|---|
| `c.<name>` | Coordinate namespace, written by the host every frame. A mapping such as `c.loc_center` holds the zone a body part occupies; `c.snout_x` / `c.snout_y` hold pixel coordinates |
| `zone_changed` | Intrinsic event raised when the configured body part changes zone. The task does not have to declare it |
| Trigger events | Any name you type into the trigger table. It reaches the task exactly as a nose-poke event does, so it **must appear verbatim in the task's `events` list** |
| `frame_event` | Optional per-frame event, off by default |

Which of these are sent is set by the **Push to MCU** options in the tracking
dialog. See [Tracking](../user-guide/tracking.md).

## 3 · What a hardware definition declares

A hardware definition is the wiring, written once per apparatus. It names a board
and attaches devices to its ports; the names it defines become `hw.<name>` in
every task uploaded to that board.

```python
from devices import *

board       = Breakout_F767_v2()
five_poke   = Five_poke(ports=[board.port_1, board.port_7])
reward_pump = Stepper_motor(direction_pin='PG9', step_pin='PG13')
```

Port map, motor channels and the full device list:
[The V2 Nucleo breakout](../hardware/breakout.md) and
[Peripheral modules](../hardware/peripherals.md).

:::{admonition} The one contract that fails silently
:class: warning

An event named in a hardware definition, or produced by a tracking trigger, must
appear **verbatim** in the task's `events` list. If it does not, the input is
never delivered, with no error and no warning.
:::

## 4 · Host-side task logic (`api_classes/`)

Optional. Some logic is awkward to run on the microcontroller: adaptive
staircases, curve fitting, anything that needs the filesystem. An *API class*
runs that part on the host and exchanges variables with the running task. See
[API class](../user-guide/api-class.md).

## 5 · The project file

`experiment_config.json` is written by the interface, not by hand, but its schema
is documented because analysis scripts read it: setups, cameras and regions,
zones, tracker settings and the trigger table.

- [Configuration schema](config-schema.md)
- [File formats](../reference/file-formats.md), the session outputs: the
  controller `.tsv`, the video, the per-frame tracking table and the drop log

---

## Internal module reference

Below this line is the application's own structure. It is here for people
changing the application; **nothing in it is needed to run an experiment or to
write a task.**

| Module | Responsibility |
|---|---|
| [`Pipeline`](pipeline.md) | Owns camera → frame bus → sinks → microcontroller |
| [`Pycboard`](pycboard.md) | Serial transport, uploads, the framework clock |
| [`VideoRecorder`](recorder.md) | Encoding and the per-frame log |
| [Trackers](trackers.md) | Pose backends behind one interface |
| [GUI](gui.md) | Windows, dialogs and per-box widgets |

Architecture and ownership rules live in
[Architecture](../architecture.md) and
[Module map](../dev/module-map.md).
