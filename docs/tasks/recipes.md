# Task recipes & use cases

The setups people actually run, operant nose-poke rigs and mazes, shown as **real task code** and
explained in plain language. Each recipe is a hardware-definition line (what to wire) plus the few
task lines that use it. New to task files? Start with [Writing tasks](writing-tasks.md); every call
below is listed in the [Task API reference](task-api.md).

## Pick your rig

Find your setup, grab the matching **hardware definition** and **example task**, and jump to the
recipe that explains it.

| I want to… | Hardware definition | Example task |
|---|---|---|
| Run a **5-poke** operant task | `pyBehLab_v2.py` (`Five_poke`) | `tasks/three_poke_test.py` · [how →](#nose-poke-walls-5-poke-and-9-poke) |
| Run a **9-poke** operant task | HD with `Nine_poke` on an I²C port | [how →](#nose-poke-walls-5-poke-and-9-poke) |
| Run **reversal learning** (pump reward) | `pyBehLab_v2.py` (pumps + `Five_poke`) | `tasks/ReversalLearning/ReversalLearning.py` · [how →](#reversal-learning-with-pump-rewards) |
| Drive **motorised doors** (limit switches) | `pyBehaviorLab_TM_Board_v2.py` (`Door`) | *(used by the T-maze task)* · [how →](#doors-as-motors-with-limit-switches) |
| Record **fiber photometry** | `pyBehLab_photometry.py` (`PyPhotometry_board`) | [how →](#fiber-photometry-pyphotometry) |
| Run a **T-maze** with zones | `pyBehaviorLab_TM_Board_v2.py` | `tasks/TM_spontAlternation/SpontMaze_RightStart_v2.py` · [how →](#example-t-maze-spontaneous-alternation) |
| Test **audio** output (I2S vs DAC) | `pyBehLab_auditory.py` | `tasks/AudioCompare/AudioCompare.py` |

:::{tip}
A **task** is a small state machine that runs on the microcontroller. It declares `states`,
`events` and an `initial_state`, then one function per state. The framework sends `'entry'` when a
state begins and `'exit'` when it ends, and a named event (like `'poke_1'`) whenever hardware fires.
You react to those in the state functions, see [Writing tasks](writing-tasks.md) for the skeleton.
:::

## Operant chambers - `pyOperant.py`

### Nose-poke walls: 5-poke and 9-poke

**In plain terms.** A "poke" is one aperture with an infrared beam (detects the nose), an LED (the
cue light) and a solenoid (a valve, e.g. for reward). When the animal breaks the beam the firmware
fires a `poke_N` event; from the task you turn the cue LED on/off and open the valve.

You wire the whole wall as **one device** on a breakout port, then reach each poke as
`hw.<device>.poke_N`:

::::{tab-set}

:::{tab-item} 5-poke
Hardware definition, a 5-aperture wall spanning two ports:
```python
five_poke = Five_poke(ports=[board.port_1, board.port_7])
```
In a task, light poke 3, wait for a poke there, flash it off:
```python
events = ['poke_3']

def cue(event):
    if event == 'entry':
        hw.five_poke.poke_3.LED.on()      # cue light on
    elif event == 'poke_3':               # animal poked aperture 3
        hw.five_poke.poke_3.SOL.on()      # open the reward valve
        print('poke_3 correct')
        goto_state('iti')
    elif event == 'exit':
        hw.five_poke.poke_3.LED.off()     # cue off when we leave
```
:::

:::{tab-item} 9-poke
The 9-poke wall uses an I²C port expander, so it needs a port with I²C; the task-side API is
otherwise identical (`poke_1 … poke_9`, each with `.LED`, and valves `SOL_1 … SOL_9`):
```python
nine_poke = Nine_poke(port=board.port_1)
```
```python
hw.nine_poke.poke_5.LED.on()   # cue aperture 5
hw.nine_poke.SOL_5.on()        # open its valve
```
:::

::::

### Reversal learning with pump rewards

**In plain terms.** The animal learns that one side (say left) pays out. Once it is reliably
choosing that side, the rule **reverses**, the other side now pays, and it has to relearn. The
task keeps a running accuracy and flips `good_side` once the animal has clearly learned.

Reward here comes from a **syringe/peristaltic pump** (a stepper motor), not a solenoid. Pumps are
wired as `Stepper_motor` with a shared enable line:

```python
pump2        = Stepper_motor(direction_pin='PG9', step_pin='PG13')
enable_pumps = Digital_output(pin='PD9')
```

In the task, enable the driver at the start (the enable line is inverted, so `off()` = enabled) and
deliver reward by stepping the pump forward. Track accuracy with an exponential moving average and
reverse when it clears a threshold:

```python
v.good_side  = choice(['left', 'right'])            # random starting rule
v.reward_ul  = 20                                   # microlitres per reward
v.correct_ma = exp_mov_ave(tau=8, init_value=0.5)   # running accuracy

def run_start():
    hw.enable_pumps.off()                           # power up the pump driver

def deliver_reward(event):
    if event == 'entry':
        hw.pump2.forward(v.step_rate, v.n_steps)    # dispense the reward
        print('reward_delivered {}ul'.format(v.reward_ul))
        v.correct_ma.update(1)                      # this trial was correct
        if v.correct_ma.value > 0.8:                # learned it → reverse the rule
            v.good_side = 'right' if v.good_side == 'left' else 'left'
            print('Reversal: good_side now {}'.format(v.good_side))
        goto_state('iti')
```

The shipped `tasks/ReversalLearning/ReversalLearning.py` goes further, probabilistic reward, a
post-threshold delay before the reversal, and per-poke sounds, but this is its core loop.

### Doors as motors, with limit switches

**In plain terms.** A motorised door is a stepper motor that raises/lowers a gate. **Limit
switches** are little sensors that tell the firmware when the door has reached the **top** (closed)
or **bottom** (open), so it stops there instead of grinding the motor. You can wire **both** limits,
or **just the bottom** one (then "close" stops by a step count instead of a top switch).

The `Door` device wraps all of that, ramped motion, limit-switch awareness, and simple
`open()` / `close()` / `home()` calls:

::::{tab-set}

:::{tab-item} Top + bottom limits
```python
home_door = Door(direction_pin='PF2', step_pin='PF9',
                 limit_close_pin='PD6',   # top switch    → fully closed
                 limit_open_pin='PD5',    # bottom switch → fully open
                 max_steps=4000, step_rate=2000, min_speed=1000,
                 doorup_dir='backward')   # motor direction that RAISES the door
HomeDoor_enable = Digital_output(pin='PD1', inverted=True)   # on() = enabled
```
:::

:::{tab-item} Bottom limit only
No top switch - `close()` stops after `max_steps` instead of at a limit:
```python
home_door = Door(direction_pin='PF2', step_pin='PF9',
                 limit_open_pin='PD5',    # only the bottom (open) switch
                 max_steps=4000, step_rate=2000, doorup_dir='backward')
```
:::

::::

In a task you just command the door, the motion, ramp and stopping-at-the-limit are handled:

```python
def run_start():
    hw.HomeDoor_enable.on()          # hold the driver enabled

def open_gate(event):
    if event == 'entry':
        hw.home_door.open()          # lower to the bottom limit
        if hw.home_door.is_open():
            timed_goto_state('trial', 0)
```

:::{note}
`doorup_dir` is the motor direction that *raises* the door toward its closed/top limit. If a door
travels the wrong way on your rig, flip that one value (`'forward'` ↔ `'backward'`), no rewiring.
:::

### Fiber photometry (pyPhotometry)

**In plain terms.** Photometry records a fluorescent signal from the brain. It plugs into the
breakout's dedicated photometry connector: **two photodetector channels** stream in as analog
inputs, and **two excitation LEDs** are driven from the microcontroller's DAC. Because it runs on
the **same firmware clock** as the behaviour and video, the samples are already time-aligned, no
external sync box.

```python
photometry = PyPhotometry_board(board.photometry1, sampling_rate=130)
```

:::{warning}
The F767 has only two DAC channels, and the audio board's control and the photometry LEDs both want
one of them, so a box runs **either** the audio board **or** photometry LED control, not both at
once.
:::

(maze-arenas-pymaze-py)=
## Maze arenas - `pyMaze.py`

Maze tasks react to **where the animal is**. The camera tracks the animal, decides which **zone**
the centroid is in, and pushes that zone name to the microcontroller. Your task reads it as
`c.loc_center` and reacts. The zone-handling idioms are covered in
[Writing tasks, Path C](writing-tasks.md#path-c-a-maze-task); here is a complete task.

### Example: T-maze spontaneous alternation

**The behaviour, in plain terms.** A T-maze has a home/stem and two arms (left, right). A healthy
animal tends to **alternate**, if it went right last time, it prefers left this time. This task
scores each choice: alternating = *correct*, repeating the same arm = *incorrect*. Motorised doors
open and close to guide the animal through the trial.

The actual task (`tasks/TM_spontAlternation/SpontMaze_RightStart_v2.py`), header first:

```python
from pyControl.utility import *
import hardware_definition as hw

states = ['init_state', 'habituation', 'open_sample_arm',
          'sample_arm', 'return_to_center', 'choice_phase', 'last_state']
events = ['zone_changed', 'frame_event', 'end_task_timer']   # host-fired, intrinsic
initial_state = 'init_state'

c.loc_center = 0                       # zone the tracker last reported ('' = none)
left_zones   = ['LeftArm']            # names must match the zones you drew
right_zones  = ['RightArm']
center_zones = ['Center_MidZone']
POLL = ('entry', 'zone_changed', 'frame_event')   # re-check position on these
```

What each state does, in words:

1. **`init_state`**, close all three doors (a known start) and start the session timer.
2. **`habituation`**, let the animal settle for `habituation_dur`, then continue.
3. **`open_sample_arm`**, open the home door and the *start* arm; the animal is forced down one arm.
4. **`sample_arm`**, *wait* until `c.loc_center` is the start arm, then close the home door to
   confine the animal to the maze.
5. **`return_to_center`**, *wait* until the animal is back in the centre, count the trial, and open
   the *opposite* arm so both are now available.
6. **`choice_phase`**, *wait* until the animal commits to an arm. Opposite the last visit →
   **alternation → correct**; same arm → incorrect. Close the arm not chosen and loop to the centre.

The heart of it is a "wait for a zone" state. Note the `if event not in POLL: return` guard (act
only when position may have changed) and the one-shot latch (`v.choice_made`) so a flickering
centroid can't score the same trial twice:

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
        v.choice_made = True                    # score this trial once
        if chosen != v.init_arm:                # alternated → correct
            v.n_correct += 1
        else:
            v.n_incorrect += 1
        (hw.right_door if chosen == 'L' else hw.left_door).close()
        v.init_arm = chosen                     # this arm is the new reference
        timed_goto_state('return_to_center', 0)
```

### How to create zones

Zones are drawn once per box, in the GUI, and stored with the project:

1. Open **Tracking Config** for the box (Camera Control on the Main Control tab).
2. In the **zone editor**, pick a shape (rectangle, circle, ellipse, polygon, or line) and draw it
   over the camera image, one zone per region you care about (each maze arm, the centre).
3. **Name each zone** exactly as your task expects, for the T-maze: `LeftArm`, `RightArm`,
   `Center_MidZone`. The names are the contract between the editor and the task.
4. Wire the zones to the firmware on the **Tracking Settings** tab (drawing a zone doesn't send
   anything on its own):
   - to read the current zone as `c.loc_center`, add a **Coord Mapping** row and tick
     **Push to MCU → c.\* coordinates**;
   - to fire an event on entry/exit, add an **Event Trigger** (`enter_zone` / `exit_zone`) and tick
     **Push to MCU → Zone-change events**.
5. Optionally add a **scale** (a ruler of known length) so speed is reported in real units
   (m/s) instead of px/s.

See [Set up cameras, tracking and zones](../user-guide/setup-cameras-tracking-zones.md) for the full
click-by-click walkthrough.

Zones are stored in resolution-independent coordinates, so they don't move if the camera resolution
changes, and they can **nest**, a query returns the innermost zone first.

### How zones reach the task

Every frame the tracker pushes the current zone name into `c.loc_center` (and speed into `c.speed`)
and fires `zone_changed` on any entry/exit, all **silently** (no clutter in the data log; the video
file already records occupancy). Your task:

- declares the coordinate slot: `c.loc_center = 0`;
- lists the zone names it expects, matching the editor (`left_zones = ['LeftArm']`, …);
- re-checks position on the `POLL` events and compares `c.loc_center` against those lists.

```python
def sample_arm(event):
    if event not in POLL:                  # only act when position may have changed
        return
    if c.loc_center in right_zones:        # animal reached the start arm
        hw.home_door.close()               # confine it to the maze
        timed_goto_state('return_to_center', 0)
```

:::{tip}
Coordinates (`c.*`) are pushed **before** events each tick, so an event handler always reads a fresh
position. To poll `c.*` on every single frame (not just on zone changes), enable the opt-in
`frame_event` and add it to your `POLL` tuple, the T-maze task above already does.
:::
