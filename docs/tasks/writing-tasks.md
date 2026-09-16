# Writing tasks

Behavioural tasks are written as **state machines** that run on the box's microcontroller. A task is
a plain `.py` file with three module-level lists and one function per state. The host uploads it to
the board over serial; from then on the board runs the task by itself and streams what happens back
to the GUI.

The framework underneath is [pyControl](https://pycontrol.readthedocs.io/en/latest/), so everything
in its
[Programming tasks](https://pycontrol.readthedocs.io/en/latest/user-guide/programming-tasks/) guide
applies here unchanged. This page covers that skeleton briefly and then the part pyControl does not
have: tasks that read what the camera sees.

:::{tip}
A task never imports host code (`PySide6`, `cv2`, …) and never runs on your PC. It runs on the board
under MicroPython, and imports only `pyControl.utility` and its hardware definition. See the
[Task API reference](task-api.md) for everything a task may call.
:::

## Three kinds of task

Pick the row you need before you start writing. They share the skeleton and differ only in where
the information a state reacts to comes from.

| | reacts to | needs a camera | start here |
|---|---|---|---|
| **Operant, board alone** | pokes, levers, timers | no | [Path A](#path-a-an-operant-task-the-board-runs-alone) |
| **Operant, pose driven** | the above, plus where the animal is | yes, plus a tracking model | [Path B](#path-b-an-operant-task-driven-by-pose) |
| **Maze** | which zone the animal is in | yes, plus zones drawn in the editor | [Path C](#path-c-a-maze-task) |

Paths B and C use the same machinery. The host tracks the animal, decides which zone each body part
is in, and pushes that to the board. Only the configuration differs, so anything below written for
one applies to the other.

## The skeleton

Every task file starts the same way:

```python
from pyControl.utility import *      # goto_state, timed_goto_state, v, c, print, units …
import hardware_definition as hw     # the wiring: hw.five_poke, hw.pump2, hw.speaker …

states        = ['reward_state', 'iti']         # every state your task can be in
events        = ['poke_1', 'session_timer']     # every event you want to react to
initial_state = 'reward_state'                  # the state the run begins in
```

Three module-level names define the machine:

- **`states`**, the list of state names. The task is always in exactly one of them.
- **`events`**, the names your states react to. An event is either a **hardware** event (a beam
  break fires `'poke_1'`), a **timer** event you set yourself, or an event the **host** fires from
  what the camera saw. Only names listed here are delivered, and this is the single most common
  cause of a task that silently ignores an input. See
  [Contracts that fail silently](#contracts-that-fail-silently).
- **`initial_state`**, where the run starts.

### Variables live in `v`

Task variables live in the `v` namespace. Assign them at module level to set their starting value;
they persist for the whole session and are visible to the GUI, which can read and set them while
the task runs. Units come from the toolkit (`ms`, `second`, `minute`, `hour`, all in milliseconds):

```python
v.reward_rate  = 500          # stepper pulse rate
v.reward_steps = 200          # steps per reward
v.session_dur  = 1 * hour     # 3_600_000 ms
```

### One function per state

Each state is a function `def state(event):`. The framework calls it with the string `'entry'` the
moment the state begins and `'exit'` the moment it ends, so bracket your setup and teardown with
those, and with an event name whenever a listed event fires while that state is active:

```python
def reward_state(event):
    if event == 'entry':
        hw.five_poke.poke_1.LED.on()            # cue light on when we arrive
    elif event == 'poke_1':                     # animal poked aperture 1
        hw.pump2.forward(v.reward_rate, v.reward_steps)
        print('reward_delivered')               # timestamped row in the data log
        goto_state('iti')                       # leave for the ITI
    elif event == 'exit':
        hw.five_poke.poke_1.LED.off()           # cue off when we leave
```

Two ways to change state:

- **`goto_state('name')`**, switch immediately.
- **`timed_goto_state('name', interval_ms)`**, switch after a delay. It is cancelled automatically
  if a `goto_state()` happens first, so it doubles as a per-state timeout.

```python
def iti(event):
    if event == 'entry':
        timed_goto_state('reward_state', 2 * second)   # 2 s inter-trial interval
```

### `run_start`, `run_end`, `all_states`

Three optional functions the framework calls for you:

- **`run_start()`**, once at the very start of the run. Enable hardware drivers, arm the session
  timer, seed variables.
- **`run_end()`**, once at the end. Turn everything off; `hw.off()` cuts all outputs.
- **`all_states(event)`**, called for **every** event, whatever state you are in. The place to
  handle a session-end timer or a global abort.

```python
def all_states(event):
    if event == 'session_timer':
        stop_framework()          # end the run cleanly
```

:::{note}
`print(..)` in a task does not go to a console. It writes a **timestamped row to the session data
log** (the `.tsv`). That is how trials, choices and outcomes get recorded. Use it liberally; it is
the primary data channel alongside hardware events.
:::

(path-a-an-operant-task-the-board-runs-alone)=
## Path A: an operant task the board runs alone

Nothing here needs a camera. The board reads its own inputs and decides everything, so the task
keeps running even with the video pipeline switched off.

A minimal one-poke reward loop. Poke aperture 1 to earn a pump reward, sit through a 2 s ITI,
repeat, until the session timer ends the run:

```python
from pyControl.utility import *
import hardware_definition as hw

states        = ['reward_state', 'iti']
events        = ['poke_1', 'session_timer']
initial_state = 'reward_state'

v.reward_rate  = 500
v.reward_steps = 200
v.session_dur  = 1 * hour

def run_start():
    hw.enable_pumps.off()                     # off() enables the (inverted) pump driver
    set_timer('session_timer', v.session_dur) # fire 'session_timer' once, after session_dur

def reward_state(event):
    if event == 'entry':
        hw.five_poke.poke_1.LED.on()
    elif event == 'poke_1':
        hw.pump2.forward(v.reward_rate, v.reward_steps)
        print('reward_delivered')
        goto_state('iti')
    elif event == 'exit':
        hw.five_poke.poke_1.LED.off()

def iti(event):
    if event == 'entry':
        timed_goto_state('reward_state', 2 * second)

def all_states(event):
    if event == 'session_timer':
        stop_framework()

def run_end():
    hw.off()                                  # turn off all outputs
```

:::{tip}
Many rigs wire the pump enable **inverted**, so `off()` powers the driver on and `on()` powers it
off. Whether a line is inverted is decided once in the hardware definition
(`Digital_output(pin, inverted=True)`); the task just calls `on()` and `off()`.
:::

For richer patterns, a running accuracy that reverses the reward rule, per-poke sounds, polling a
sound-level sensor, read the real
[`ReversalLearning.py`](https://github.com/pyBehaviorLab/code/blob/main/tasks/ReversalLearning/ReversalLearning.py)
alongside the [Task recipes](recipes.md).

### Audio

The audio device named in the hardware definition (here `speaker`) is called directly. The first
`play_*` or `load_*` call powers up the amp, so you do not manage its lifecycle:

```python
hw.speaker.play_sine(10000, 100, ramp_ms=5, level=0.5)   # 10 kHz tone, 100 ms
hw.speaker.play_noise(80, level=0.25)                    # 80 ms white noise
hw.speaker.play_click(ms=1)                              # a click
hw.speaker.play_sweep(5000, 15000, ms=80)                # FM sweep

hw.speaker.load_sweep(10000, 15000, ms=80, name='cue')   # pre-render once …
hw.speaker.play('cue')                                   # … replay with no compute delay

hw.speaker.start_noise(level=0.2)                        # continuous masker
hw.speaker.play_cue_in_noise(8000, 100)                  # tone mixed into the masker
hw.speaker.play_wav('/sd/cue.bin')                       # WAV streamed from the SD card
```

Pre-loading in `run_start()` is the common pattern: the render cost is paid once, and each trial's
`play('name')` is free. From `ReversalLearning.py`:

```python
def run_start():
    hw.speaker.load_sweep(10000, 15000, ms=80, name='sweep')  # poke 2
    hw.speaker.load_click(ms=1, name='click')                 # poke 3
    hw.speaker.load_sine(10000, 80, name='sine')              # poke 4
    hw.speaker.set_volume(50)                                 # ~-30 dB; 80+ may clip
```

:::{warning}
At the default 48 kHz sample rate the usable band tops out just below Nyquist (~21.6 kHz). Keep
stimulus frequencies under that. Two amp variants share this exact API, `TAS5825MAudio` (I²C
class-D, hardware volume) and `PCM5102Audio` (strap-pin DAC).
:::

(path-b-an-operant-task-driven-by-pose)=
## Path B: an operant task driven by pose

Everything in Path A still applies. What changes is that the host now pushes what the camera sees
into the running task, so a state can react to where the animal is as well as to what it touched.

Two channels carry it, and they behave differently:

- **Coordinates**, the `c` namespace. Pushed **only when the value changes**, so a stationary animal
  costs no serial traffic. Read them like variables.
- **Events**, plain names in your `events` list. Fired on the **false to true transition** of a
  condition, so a zone entry gives you one event, not one per frame.

Declare each coordinate at module level with a starting value. The name you choose here is the name
you map to a body part in the tracking dialog.

:::{important}
**The name decides what arrives in it.** There is no separate setting: the resolver reads the
variable name and sends the matching kind of value.

| you name it | the board receives |
|---|---|
| `x`, `y`, or anything ending `_x` / `_y` | the body part's **pixel coordinate**, `-1.0` when it was not detected this frame |
| `speed` | the body part's **speed** |
| **anything else** (`loc_center`, `where`, `arm`) | the **name of the zone** the body part is in, `''` when it is in none |

So `c.head_x` is a number and `c.loc_center` is a zone-name string, from the same mechanism. Naming
a zone variable `zone_x` gets you a pixel float and no zone name; the host logs a warning when it
sees that, but the task will already be comparing a float against strings. Zone names are used in
*Path B* and *Path C* below.
:::

```python
events = ['poke_1', 'at_port', 'session_timer']   # 'at_port' is fired by the host
initial_state = 'wait'

c.head_x = 0.0        # ends in _x, so: pixel x of the mapped body part
c.head_y = 0.0
c.speed  = 0.0        # m/s if a scale is calibrated, otherwise px/s

def wait(event):
    if event == 'at_port':                  # host saw the head enter the port zone
        goto_state('reward_state')
```

:::{warning}
A coordinate for a body part the tracker did **not** detect this frame is pushed as `-1.0`, not left
at its last value and not `NaN`. Pixel coordinates are never negative, so this is unambiguous, and a
task that treats a lost part as a real position will act on the last place it saw the animal:

```python
if c.head_x < 0:
    return          # not detected this frame, do not decide anything on it
```
:::

The condition behind an event is chosen in the tracking dialog, not written in the task. The task
sees only the resulting event name, which keeps the geometry in the GUI where it can be drawn and
checked against the video. These are the conditions available, each attached to one body part and
one event name:

| condition | fires when the body part |
|---|---|
| `in_zone`, `enter_zone` | is inside the named zone |
| `not_in_zone`, `exit_zone` | is outside it |
| `exit_edge` | leaves it, on the transition only |
| `speed_gt`, `speed_lt` | moves faster or slower than a threshold |
| `freezing` | moves slower than a threshold, default 1 unit/s |
| `rotation_gt`, `rotation_lt` | turns faster or slower than a threshold, deg/s |
| `head_angle_gt`, `head_angle_lt` | holds a head-to-body angle past a threshold |
| `facing_line` | points within a threshold angle of a target point, default 30° |
| `elongation_gt` | is stretched past a threshold |
| `rearing` | is foreshortened and slow, the rearing proxy |
| `distance_gt`, `distance_lt` | is further from or nearer to a second part than a threshold |

Speed and distance thresholds carry a unit: `px`, `mm`, `cm` or `bodylen`, the last one being the
animal's own body length so a threshold survives a change of camera height. Every condition also
takes an optional `duration_ms`: the condition has to hold for that long before the event fires,
which is what stops a threshold flickering at its boundary from firing a burst of events.

:::{warning}
`facing_line` needs a `target` point and no editor writes one today. Without it the rule is disabled
and logs a warning once rather than measuring the angle to the frame origin, which is what it used
to do and which looked like it was working.
:::

:::{note}
Pose reaches the board about 30 ms after the frame was captured on the reference rig, and that is
the delivered figure including queueing, not a model benchmark. If a state must act within one frame
of an event, use a hardware input rather than pose. See
[Performance and validation](../reference/performance.md).
:::

(path-c-a-maze-task)=
## Path C: a maze task

A maze task is a pose-driven task whose main input is **which zone the animal is in**. The host
tracks the animal, decides zone occupancy per frame, and pushes it in silently, with no data-log
clutter, because the video file already records occupancy for every frame.

- **`c.loc_center`**, the name of the zone the animal is in (`''` or `0` for none), pushed on change.
- **`c.speed`**, the centroid's speed, m/s when a scale zone is set, otherwise px/s.
- **`'zone_changed'`**, fired on any zone entry or exit.
- **`'frame_event'`**, an opt-in per-frame event, for tasks that must re-check position every frame
  rather than only on a change.

Declare the coordinate slot, list the zone names your task expects (they must match the names you
drew in the zone editor), and re-check position on a small **POLL** set:

```python
events = ['zone_changed', 'frame_event', 'end_task_timer']   # host-fired
POLL   = ('entry', 'zone_changed', 'frame_event')            # re-check position on these

c.loc_center = 0                       # zone name pushed by the tracker ('' = none)
right_zones  = ['RightArm']            # names must match the zones you drew

def sample_arm(event):
    if event not in POLL:              # ignore events that cannot have moved the animal
        return
    if c.loc_center in right_zones:    # animal reached the start arm
        print('Sample arm reached: {}'.format(c.loc_center))
        hw.home_door.close()           # confine it to the maze
        timed_goto_state('return_to_center', 0)
```

Two idioms keep a zone state correct when frames are missed, both from
`SpontMaze_RightStart_v2.py`:

- **The `if event not in POLL: return` guard**, so a state acts only when the animal's position may
  actually have changed, not on every unrelated timer or hardware event.
- **A one-shot latch** (`v.choice_made`, reset on `'entry'`), so a centroid flickering on a zone
  boundary cannot score the same trial twice.

```python
def choice_phase(event):
    if event == 'entry':
        v.choice_made = False
    if event not in POLL or v.choice_made:
        return
    chosen = None
    if   c.loc_center in left_zones:  chosen = 'L'
    elif c.loc_center in right_zones: chosen = 'R'
    if chosen:
        v.choice_made = True               # score this trial once
        ..
```

Each zone also carries a **transmit policy** set in the editor: push the zone name to a coordinate
variable (`loc_center` by default), or fire a named event on entry (`event_enter`) or on entry and
exit (`event_enter_exit`). The latter arrives as an event you handle like any hardware input. See the [maze recipe](recipes.md#example-t-maze-spontaneous-alternation) for a full T-maze
walk-through.

## Contracts that fail silently

Three files have to agree, and when they do not, nothing raises. The input simply never arrives,
which reads at the rig like broken hardware.

1. **An event name in the hardware definition must appear verbatim in the task's `events` list.**
   The hardware definition names an input `poke_1`; if the task lists `poke1`, the framework drops
   every one of them at the queue. Same string, same case.
2. **A zone name drawn in the editor must match the string the task compares against.** `RightArm`
   and `right_arm` are different zones, and a comparison against the wrong one is simply never true.
3. **A coordinate name declared with `c.` must match the name mapped in the tracking dialog.** An
   unmapped coordinate keeps its module-level starting value for the whole session, so the task runs
   and the animal never appears to move.

The quickest check is the session `.tsv`: if an input is wired and the animal is triggering it, the
event name appears there. If the file has no such rows, the name never reached the task.

## Upload it and find the data

1. Drop your `.py` in `tasks/` (a single file) or `tasks/<YourTask>/` (a folder, if it has helper
   modules). Assign it to a box in the GUI, connect the board and click **Upload Task**. The app
   transfers the framework, the hardware definition and your task over serial.
2. Press **Record**. The board runs the state machine; every `print(..)`, every state change and
   every event lands in the per-session `.tsv`, with the task and hardware-definition hashes stamped
   in its header.

A run writes two files per box, both named `<subject>-Box<N>-<YYYY-MM-DD-HHMMSS>`:

```
data/<project>/<task>/<YYYY-MM-DD>/
  mcu/    39-Box1-2026-09-08-132130.tsv               the board's own record
  video/  39-Box1-2026-09-08-132130.mp4               the recording
          39-Box1-2026-09-08-132130_video_data.txt    one row per frame
          _drops.tsv                                  any frame the camera lost
```

The `.tsv` is what your task wrote: one row per state change, per event and per `print(..)`, with
the task and hardware-definition hashes in its header so the exact code that produced it can be
recovered later. The `_video_data.txt` is one row per camera frame, carrying the pose, the zone and
the board time of that frame, which is what joins the two files together.

## Where to read next

- [Configuration files](configs.md), what to configure and where each file goes.
- [Task API reference](task-api.md), every function a task may call.
- [Task recipes](recipes.md), worked tasks for both rigs.
- [pyControl: Programming tasks](https://pycontrol.readthedocs.io/en/latest/user-guide/programming-tasks/),
  the upstream framework guide, including the parts this page does not repeat: the full function
  reference, task file structure and writing performant task code.
