# Task API reference

Everything a task (or a hardware definition) may call. The task toolkit comes in with
`from pyControl.utility import *`; the hardware primitives and device drivers are what a
hardware-definition file instantiates and a task then drives as `hw.<name>`.

- Task utilities - `source/pyControl/utility.py`
- Hardware primitives - `source/pyControl/hardware.py`
- Device drivers - `devices/`

For a narrative introduction see [Writing tasks](writing-tasks.md); for wiring examples see
[Task recipes](recipes.md).

## State control

```{list-table}
:header-rows: 1
:widths: 34 66

* - Call
  - Effect
* - `goto_state(next_state)`
  - Transition to `next_state` immediately.
* - `timed_goto_state(next_state, interval)`
  - Transition to `next_state` after `interval` ms; cancelled if a `goto_state()` happens first.
* - `current_state()`
  - Return the name of the state the task is currently in.
* - `stop_framework()`
  - End the run cleanly (triggers `run_end()` and hardware teardown).
* - `get_current_time()`
  - Current framework time in ms (the clock stamped on every event and shared with the video).
```

## Timers

Timers fire a named event after an interval. The event must be listed in `events`.

```{list-table}
:header-rows: 1
:widths: 34 66

* - Call
  - Effect
* - `set_timer(event, interval, output_event=True)`
  - Fire `event` once after `interval` ms. `output_event=False` schedules it silently (no data row).
* - `disarm_timer(event)`
  - Cancel all pending timers for `event`.
* - `reset_timer(event, interval, output_event=True)`
  - Disarm any pending `event` timer and set a fresh one for `interval` ms.
* - `pause_timer(event)`
  - Pause all timers for `event` (time remaining is frozen).
* - `unpause_timer(event)`
  - Resume paused timers for `event`.
* - `timer_remaining(event)`
  - Milliseconds until `event`'s timer elapses; `0` if none is set.
```

## Logging & events

```{list-table}
:header-rows: 1
:widths: 34 66

* - Call
  - Effect
* - `print(print_string)`
  - Write a timestamped row to the session data log (the `.tsv`). The task's primary data channel.
* - `print_variables(variables="all", when="p")`
  - Dump the given variables (or all) to the data log as a JSON string.
* - `warning(message)`
  - Write a warning row to the log.
* - `publish_event(event)`
  - Put a named event into the event queue yourself, as if hardware had fired it.
```

## Randomisation & maths

```{list-table}
:header-rows: 1
:widths: 40 60

* - Call
  - Returns
* - `random()`
  - Random float `x` with `0 <= x < 1`.
* - `withprob(p)`
  - Boolean that is `True` with probability `p`.
* - `shuffled(L)`
  - A shuffled copy of list `L`.
* - `randint(a, b)`
  - Random integer `N` with `a <= N <= b`.
* - `choice(L)`
  - A randomly selected item from list `L`.
* - `exp_rand(m)`
  - Exponentially distributed random number with mean `m`.
* - `gauss_rand(m, s)`
  - Gaussian random number, mean `m`, standard deviation `s`.
* - `mean(x)`
  - Arithmetic mean of sequence `x`.
* - `Sample_without_replacement(items)`
  - Sampler object; `.next()` draws items without replacement, reshuffling when exhausted.
* - `Exp_mov_ave(tau, init_value=0)`
  - Exponential moving-average object; `.update(sample)` folds in a sample, `.value` reads it.
```

:::{note}
`exp_mov_ave` / `sample_without_replacement` are lower-case aliases for the classes above, so both
`Exp_mov_ave(tau=8)` and `exp_mov_ave(tau=8)` work. `ReversalLearning.py` uses
`exp_mov_ave(tau=v.tau, init_value=0.5)` to track a running accuracy that drives reversals.
:::

## Namespaces & units

```{list-table}
:header-rows: 1
:widths: 26 74

* - Name
  - Meaning
* - `v`
  - Task-variable namespace. `v.foo = 1` declares/holds a session variable; the GUI can read and set it live.
* - `c`
  - Coordinate namespace. Tracking data (`c.loc_center`, `c.speed`) is pushed in from the host each frame (maze).
* - `ms`
  - Time unit, `1` (all framework time is in milliseconds).
* - `second`
  - `1000 * ms`.
* - `minute`
  - `60 * second`.
* - `hour`
  - `60 * minute`.
```

## Hardware primitives

The low-level building blocks in `pyControl.hardware`. A hardware definition wires these directly
or through the higher-level device drivers below.

```{list-table}
:header-rows: 1
:widths: 46 54

* - Constructor
  - Purpose
* - `Digital_input(pin, rising_event=None, falling_event=None, debounce=5, pull=None)`
  - Fire a framework event on a pin's rising/falling edge (beams, buttons, limit switches).
* - `Digital_output(pin, inverted=False)`
  - A driven output (`on()`/`off()`/`toggle()`/`pulse()`); `inverted=True` flips the active level.
* - `Analog_input(pin, name, sampling_rate, threshold=None, rising_event=None, falling_event=None, data_type="H", triggers=None)`
  - Sample a voltage at `sampling_rate` and stream it; optional threshold fires events.
* - `Rsync(pin, event_name="rsync", mean_IPI=5000, pulse_dur=50)`
  - Emit random-interval sync pulses for aligning to external acquisition hardware.
* - `Port(DIO_A, DIO_B, POW_A, POW_B, DIO_C=None, POW_C=None, DAC=None, I2C=None, UART=None)`
  - One RJ port bundling its DIO, POW and bus lines; a device driver claims the lines it needs.
```

## Device drivers

Peripheral drivers in `devices/`. A hardware definition instantiates the ones your rig has; the
task drives them as `hw.<name>`.

```{list-table}
:header-rows: 1
:widths: 50 50

* - Constructor
  - Purpose
* - `Poke(port, rising_event=None, falling_event=None, debounce=5)`
  - One nose-poke aperture: `.input` beam, `.LED` cue, `.SOL` valve.
* - `Five_poke(ports, rising_event_1="poke_1", …, debounce=5)`
  - A 5-aperture wall spanning two ports; reach each as `.poke_1 … .poke_5`.
* - `Nine_poke(port, rising_event_1="poke_1", …, debounce=5, solenoid_driver=True)`
  - A 9-aperture wall via an I²C port expander; `.poke_1 … .poke_9`, valves `SOL_1 … SOL_9`.
* - `Lickometer(port, rising_event_A="lick_1", falling_event_A="lick_1_off", rising_event_B="lick_2", falling_event_B="lick_2_off", debounce=5)`
  - Two lick sensors (`.lick_1`/`.lick_2`) plus two valves (`.SOL_1`/`.SOL_2`).
* - `Stepper_motor(port=None, direction_pin=None, step_pin=None)`
  - Low-level stepper: `.forward(rate, steps)` / `.backward(rate, steps)` / `.stop()`. Reward pumps.
* - `Door(motor_port=None, motor=None, direction_pin=None, step_pin=None, limit_close_pin=None, limit_open_pin=None, max_steps=4000, step_rate=1500, min_speed=None, doorup_dir='backward', limit_active_high=True)`
  - Motorised maze door with ramped motion and limit-switch awareness: `.open()`/`.close()`/`.home()`.
* - `TAS5825MAudio(i2c_id=1, i2c_scl="PB8", i2c_sda="PB9", pdn_pin="PB6", level=50, i2s_id=2, i2s_ws="PB12", i2s_sck="PB13", i2s_sd="PB15", sample_rate=48000, …)`
  - I²C class-D I2S amp with hardware volume; full `play_*`/`load_*` stimulus API.
* - `PCM5102Audio(xsmt_pin=None, i2s_id=2, i2s_ws="B12", i2s_sck="B13", i2s_sd="B15", sample_rate=48000, …)`
  - Strap-pin I2S DAC; same `play_*`/`load_*` API as the TAS5825M.
* - `Audio_board(port)`
  - Legacy DAC audio board on a port with DAC + I²C; `.set_volume(V)` plus tone playback.
* - `ESP_mic(uart, tx, rx, req_pin=None, alert_pin=None, alert_event=None, alert_off_event=None, enable_pin=None, baudrate=115200, timeout=15)`
  - ESP32-S3 two-mic dB(A) meter: `.read_levels()` → `(dB_L, dB_R)`; `.set_alert_thresholds(hi, lo)` arms `alert_event`.
* - `PyPhotometry_board(photometry_port, sampling_rate=1000)`
  - Fiber photometry: two photodetector analog inputs + two DAC-driven excitation LEDs, on the shared clock.
```

:::{tip}
The audio drivers share one playback API on the I2S base class - `play_sine`, `play_noise`,
`play_click`, `play_sweep`, `play_chord`, `play_cue_in_noise`, `start_noise`/`stop_noise`,
`play_wav`, and the `load_*` pre-render variants replayed with `play('name')`. See
[Audio from a task](writing-tasks.md#audio).
:::
