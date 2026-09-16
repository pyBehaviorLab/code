# Configuration files

The task file says what the box does. Everything around it, which variables survive to the next
session, what the live stats panel counts, how the rig is wired, is configuration, and each piece
lives in its own small file. This page says what each file decides, where it has to sit, and who
writes it.

Only two of them are normally written by hand: the variable sidecar and the stats config. The rest
are either produced by the GUI or belong to the hardware.

| file | decides | where it goes | written by |
|---|---|---|---|
| `experiment_config.json` | the rig: boxes, cameras, boards, tracking, zones | `experiments/projects/<project>/` | the GUI |
| `<Task>.variables.json` | which task variables carry over to the next session | next to the task, in `tasks/<Family>/` | the variables grid, or by hand |
| `config.json` | what the live stats panel counts and plots | next to the task, in `tasks/<Family>/` | by hand |
| `<hardware>.py` | the wiring: which pin is which port, and its event names | `hardware_definitions/` | by hand |
| `<name>.py` | host-side adaptive logic during a run | `api_classes/` | by hand |
| `<cohort>.xlsx` | which subject is in which box | `<project>/metadata/` | your spreadsheet editor |
| `settings.json` | app-wide defaults: fonts, folders, encoder | `experiments/config/` | by hand, rarely |

## The project config

`experiment_config.json` is the rig description: one entry per box, each carrying its board serial,
its camera and ROI, its hardware definition, its zones and its tracking settings. It is written by
the GUI every time you change something, so **edit it through the interface, not in a text editor**.
An autosave lands on any change, and toolbar Save additionally pins a content-addressed copy under
`<project>/source/configs/<djb2>.json` so a past run can be traced back to the exact configuration
that produced it.

It sits at the top of the project folder, alongside `template.json`, which is the same file minus
the identifying metadata. Opening a `template.json` asks for a folder and stamps out a new project
from it, which is how a second rig gets the same layout without repeating the setup.

Two constraints matter if you ever do open the file:

- `mode` must match the launcher. `pyOperant.py` loads only `"operant"`, `pyMaze.py` only `"maze"`,
  and a mismatch is refused rather than fixed up.
- `config_djb2` is a self-hash and is recomputed on save. Leave it alone.

Field by field, the schema is in [Configuration schema](../api/config-schema.md), and the shape on
disk is in the [file-format reference](../reference/file-formats.md#experiment_configjson). Where
the file sits relative to everything else is in [Projects](../user-guide/projects.md).

## Which variables carry over

By default every task variable resets to the value written in the task file at the next upload. A
variable that should instead continue from where the last session left it has to be named in a
sidecar next to the task:

```
tasks/ReversalLearning/
  ReversalLearning.py
  ReversalLearning.variables.json
```

```json
{
  "task_path": "ReversalLearning/ReversalLearning",
  "variables": [
    {"name": "stage",    "persistent": true},
    {"name": "n_trials", "persistent": false}
  ]
}
```

A name absent from the list resets, exactly as if it were listed with `persistent: false`. The
sidecar is per task and shared by every project and every subject that runs it; it says only which
names persist. The remembered *values* are per project and live in
`<project>/<task_family>/persistent_variables.json`, captured in one round trip when the run stops
and pushed back at the next upload. Only flat scalars are kept, so a dict or a list will not
survive.

That same file can also override the classification for one project, through its `flags` block, so
a training project can persist `stage` while a test project resets it, without touching the task.

:::{tip}
You do not have to write this file. The variables grid in the GUI has a persist checkbox per
variable and saves the sidecar for you. Writing it by hand is worth it when you are setting up a
task family for the first time and want the decision recorded with the task.
:::

## The live stats config

`tasks/<Family>/config.json` drives the statistics canvas: what it counts while the session runs,
how it combines those counts, and which widget shows the result. There is no naming convention to
follow beyond the file name, and the choice of file is remembered in the project so it reopens with
the right one.

The flat schema is the one to hand-author:

```json
{
  "task": "Reversal learning",
  "update_interval_ms": 1000,
  "metrics": [
    {"name": "accuracy",  "track": "Correct_response,Incorrect_response",
     "widget": "ring", "color": "#2ecc71"},
    {"name": "omissions", "track": "Omission", "widget": "bar"}
  ],
  "table": ["box", "subject", "accuracy", "omissions"],
  "trend": ["accuracy"],
  "trend_window": 50
}
```

Each metric needs a `name` and one source: `track` for a printed string, `track_event` for an
event, `track_regex` to pull a number out of a printed line, or `sequence`. Optional keys are
`agg` (`count`, `sum`, `last`, `append`), `formula`, `label`, `format`, `widget`, `color`, `range`
and `title`. Widgets available are `ring`, `bar`, `line`, `scatter`, `histogram`, `table` and
`text_label`. With no formula, one tracked name counts, two are read as a ratio
`(a / (a + b)) * 100`, and three or more sum.

The older four-section schema (`counters`, `calculations`, `columns`, `plots`) still loads and is
what the calculator uses internally, so existing task folders keep working. It adds counter types
`print`, `event`, `any`, `latency` (with `start` and `end` state or event names) and `sequence`,
and its formulas may call `mean`, `stdev`, `cv`, `median`, `min`, `max`, `len`, `sum`, `abs` and
`round`.

The string in `track` or `match` must be exactly what the task prints. This is the one contract on
this page that fails silently: a counter watching `Correct_response` while the task prints
`correct_response` simply stays at zero. Ready-made starting points sit in
`experiments/config/stats_templates/`, and the panel itself is described in
[Statistics canvas](../user-guide/stats.md).

## Hardware definitions

The hardware definition is a `.py` file in `hardware_definitions/` that names the physical wiring:
which breakout port a poke is plugged into, which pin drives a pump, and what each input is called.
It is MicroPython and it runs on the board, not on your PC.

The event names it declares are the same strings your task lists in `events`. They must match
verbatim, and when they do not, the input is dropped at the queue with no error, which at the rig
looks like broken hardware. See
[Contracts that fail silently](writing-tasks.md#contracts-that-fail-silently) and, for the ports
and pin maps themselves, the [Hardware](../hardware/breakout.md) section.

## API classes

An api class is host-side Python in `api_classes/`, attached to a box in the project config. It
runs on the PC while the task runs on the board and can read the event stream and set task
variables mid-session, which is how adaptive staircases and any logic too heavy for the
microcontroller are implemented. The lifecycle and the method contract are in
[API class](../user-guide/api-class.md).

## Cohort metadata

An Excel sheet in `<project>/metadata/` maps subjects to boxes. Two columns are required, spelled
exactly `Subject` and `SetupID`, matched case-sensitively; `SetupID` is the integer box number. Any
other columns you keep, genotype, sex, date of birth, ride along into the run metadata. The file is
copied into the project the first time you attach it, so the project stays portable.

## App-level settings

`experiments/config/settings.json` holds preferences that are not part of any project: the folder
names the app searches for tasks, devices and hardware definitions, the plot history lengths, the
GUI font sizes, and the video encoder defaults.

```json
{
  "folders": {"tasks": "tasks", "devices": "devices",
              "hardware_definitions": "hardware_definitions",
              "api_classes": "api_classes", "data": "data"},
  "plotting": {"update_interval": 10, "event_history_len": 200},
  "GUI": {"ui_font_size": 11, "log_font_size": 9, "theme": "dark"},
  "video": {"prefer_hevc": false, "quality_crf": 18, "allow_cpu_fallback": true}
}
```

It is global. Anything that differs between two rigs or two experiments belongs in the project
config instead. Every key is listed in [Settings](../reference/settings.md), and the environment
variables that override some of them in [Environment variables](../reference/env-vars.md).

## Where each file goes

```
code/
  tasks/<Family>/<Task>.py                  the task
                 <Task>.variables.json      which variables persist
                 config.json                the live stats panel
  hardware_definitions/<hardware>.py        the wiring
  api_classes/<name>.py                     host-side adaptive logic
  devices/<peripheral>.py                   per-peripheral drivers
  experiments/
    config/settings.json                    app-wide defaults
    config/stats_templates/                 stats configs to start from
    projects/<project>/
      experiment_config.json                the rig
      template.json                         the rig, minus identity
      metadata/<cohort>.xlsx                subjects and boxes
      runs/<YYYY-MM-DD>.json                one row per run
      source/                               pinned copies of everything a run used
  data/<project>/<task>/<YYYY-MM-DD>/       the sessions themselves
```

Session data does not land inside the project. It goes to the `data/` tree, which keeps a project
folder small enough to copy or version. [Projects](../user-guide/projects.md) walks the whole
layout.

:::{warning}
`controls_dialogs/*.json`, `experiments/config/action_buttons.json` and
`experiments/config/control_templates/*.json` look authorable and are not. Nothing loads them:
declarative custom control panels were never implemented, and the action-button editor was removed.
Use the trigger-event buttons and the variables grid instead.
:::

## Where to read next

- [Writing tasks](writing-tasks.md), the state machine these files configure.
- [Task API reference](task-api.md), every function a task may call.
- [Projects](../user-guide/projects.md), what a project folder holds and where runs are saved.
- [File formats](../reference/file-formats.md), every file the app writes, row by row.
