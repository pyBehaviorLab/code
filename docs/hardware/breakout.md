# The V2 Nucleo breakout

The hardware definition *is* the wiring, every RJ port, motor and bus is one line of Python. A
station is an STM32 Nucleo seated on the **V2 breakout board**, which fans the Nucleo's GPIO out to
standardised RJ ports plus dedicated motor, photometry and microphone connectors.

**One board serves a complete apparatus.** It extends the pyControl design,
retaining its peripheral connector standard and its task and hardware-definition
format, and adding the capacity a full apparatus needs. Where the pyControl board
provides six behaviour ports and four BNC connectors, this board provides **ten
general-purpose RJ ports and eight stepper-motor channels**, the motor channels
driving both the reward pumps and the maze doors, so adding doors or pumps never
consumes the ports needed for pokes, lickometers or lights.

That extra capacity is what makes the larger apparatus possible at all: a six-arm
radial maze needs one motor channel and one end-stop input per door, and would
exhaust a six-port board before a single sensor was connected.

```python
from devices import *
import pyControl.hardware as _h

board = Breakout_F767_v2()                       # the breakout board
five_poke   = Five_poke(ports=[board.port_1, board.port_7])
reward_port = Poke(port=board.port_2, rising_event='poke_6', falling_event='poke_6_out')
speaker     = Audio_board(port=board.port_8)     # audio plugs into port 8
```

```{figure} /_static/media/intro/breakout-board.jpg
:alt: pyBehaviorLab breakout board
:width: 70%

The V2 breakout board, Nucleo carrier with RJ ports, motor drivers and dedicated connectors.
```

## RJ ports

Each RJ port is one `Port(..)` carrying **DIO** lines (digital I/O, pokes, beams, digital outputs),
**POW** lines (driven outputs, solenoids, LEDs, house lights), and, on some ports, a **bus** (UART,
DAC, I²C). This is the exact map from `Breakout_F767_v2`:

| Port | DIO lines | POW lines | Bus | Typical device |
|---|---|---|---|---|
| `port_1` | PD5, PD6 | PD11, PD13, PD12 | UART2 | Five-poke / serial device |
| `port_2` | PG2, PG3 | PB10, PA6, PB6 |, | Reward poke |
| `port_4` | PB8, PB9, PA5 | PB2, PC4 | DAC2 · I²C1 | Analog / I²C peripheral |
| `port_5` | PC6, PC7, PB1 | PA2, PE8 |, | Poke / beam / sensor |
| `port_6` | PF5, PA3, PF4 | PE7, PD14 |, | Poke / beam / sensor |
| `port_7` | PF10, PE12, PD15 | PE10, PE9 |, | Five-poke (with port 1) |
| `port_8` | PF14, PF15, PA4 | PE14, PE15 | DAC1 · I²C4 | I2S / DAC audio board |
| `port_9` | PF12, PE0, PC2 | PE13, PE11 |, | General |
| `port_10` | PG11, PG6 | PF13, PB13, PF3 |, | General |

`Port(DIO_A, DIO_B, POW_A, POW_B, DIO_C=None, POW_C=None, DAC=None, I2C=None, UART=None)`, a driver
takes a whole port and claims the lines it needs.

:::{note}
There is no `port_3`: the numbering follows the physical RJ jacks on the board, which skip that
index. Ports `4`–`10` are contiguous.
:::

## Dedicated interfaces

Some functions are awkward to carry over a general-purpose connector, so they get
their own:

| Interface | Lines | What it is for |
|---|---|---|
| **I2S** | I2S | Auditory stimuli as arbitrary **calibrated waveforms**, rather than the square waves a toggled timer output can generate |
| **MIC** |, | ESP32-based acoustic recording module |
| **FP** | DAC1, DAC2 | pyPhotometry, fibre-photometry acquisition is timed by the controller itself |
| **UART** | UART3 | Serial peripherals |
| **HDMI** | HDMI 1–7 | Synchronisation with electrophysiology |
| **GPIO / JST** |, | Additional sync and general-purpose lines |
| **12 V input** |, | Main board supply; also feeds pumps, fans and lighting |

**Why HDMI for sync.** It carries several lines in one shielded, latching cable,
so a single connection to an electrophysiology system replaces a set of
individual coaxial leads, and cannot be partly detached mid-session. See
[Peripheral modules](peripherals.md#synchronising-external-recording-equipment)
for the pulse pattern that makes two records align without an assumed common
start.

## Pokes, lights and the board tester

Pokes, lickometers, LEDs and house lights are pyControl device drivers that each take a whole port.
A `Five_poke` spans two ports (`port_1` + `port_7`); a single `Poke` or `Lickometer` takes one. Each
poke exposes its IR-beam events and an `LED` output you drive from a task
(`hw.five_poke.poke_1.LED.on()`).

```{figure} /_static/media/intro/home-arm-sensors.png
:alt: Home-arm poke and beam sensors
:width: 70%

Poke and IR-beam sensors wired into an arm, each maps to one RJ port on the breakout.
```

Before running animals, the **board tester** verifies every port end to end, it walks the DIO/POW
lines so you can confirm pokes, LEDs and solenoids fire on the right pins.

```{figure} /_static/media/intro/board-tester.jpg
:alt: Breakout board tester
:width: 70%

The board tester, exercises each RJ port to confirm wiring before a session.
```

## Motors and doors

Eight stepper-motor channels (`board.motor1 … board.motor8`), each a `Motor(DIR, STEP, EN)` triple,
for reward pumps and maze doors. Two driver levels sit on top of them.

### Low-level stepper

`Stepper_motor` is a bare direction+step pair, call `forward` / `backward` with a step rate and a
step count, or `stop`:

```python
# reward pump, low-level stepper
pump2 = Stepper_motor(direction_pin='PG9', step_pin='PG13')
```

From a task: `hw.pump2.forward(step_rate, n_steps)` / `hw.pump2.backward(step_rate, n_steps)` /
`hw.pump2.stop()`. A `Stepper_motor(port=..)` form takes `DIO_A`/`DIO_B` as direction/step instead
of explicit pins.

### Doors with limit switches

`Door` drives a maze door with a derived acceleration ramp and software end-stop awareness (plain
pin reads, no interrupts). It comes in two wirings:

- **Top + bottom limits**, both `limit_close_pin` (top/closed) and `limit_open_pin` (bottom/open)
  are wired; each move stops on its switch.
- **Bottom-only**, only `limit_open_pin` is wired; `close()` stops by `max_steps` instead of a top
  switch, while `open()`/`home()` still stop on the bottom limit.

```python
# maze door, ramped motion, top + bottom limit switches
home_door = Door(direction_pin='PG9', step_pin='PG13',
                 limit_close_pin='PD6', limit_open_pin='PD5',
                 max_steps=4000, step_rate=1800, min_speed=800, doorup_dir='backward')

# bottom-only, no top switch; close() stops by max_steps
home_door = Door(direction_pin='PG9', step_pin='PG13',
                 limit_open_pin='PD5',
                 max_steps=4000, step_rate=1800, doorup_dir='backward')
```

`doorup_dir` (`'forward'` / `'backward'`) is the motor direction that *raises* the door toward its
closed/top limit, flip it per door if a rig travels the wrong way. `step_rate` is the cruise speed
(steps/s) and `min_speed` the start/end speed; the acceleration curve is derived per move from those
two and the planned travel, there are no ramp knobs to tune.

From a task: `hw.home_door.open()` / `.close()` / `.home()` / `.stop()`, each accepting per-call
`n_steps`, `step_rate` and `min_speed` overrides. Query with `hw.home_door.is_open()`,
`.is_closed()`, `.is_moving()` and `.position`. A maze board typically defines one `Door` per arm
plus an inverted `Digital_output` enable per driver:

```python
home_door = Door(direction_pin='PF2', step_pin='PF9',
                 limit_close_pin='PD6', limit_open_pin='PD5',
                 max_steps=4000, step_rate=2000, doorup_dir='backward', min_speed=1000)
HomeDoor_enable = Digital_output(pin='PD1', inverted=True)   # off() enables the driver
```

## Preparing the Nucleo

1. **Free the GPIO**, remove the on-board solder-bridge resistor(s) so the pin is free for the
   breakout. Check the ST user manual for your exact Nucleo variant.

```{figure} /_static/media/intro/nucleo-resistor.png
:alt: Nucleo resistor to remove
:width: 80%

The resistor(s) to remove on the Nucleo. *(final photo pending, refer to the ST user manual for your
variant in the meantime.)*
```

2. **Flash MicroPython** for your exact Nucleo variant (DFU / ST-Link), connect the board and click
   **Upload Task**, the app transfers the framework, hardware definition and task over serial.

:::{tip}
Disable the board's USB mass-storage (VCP-only) so the host never writes to the flash filesystem
while the app uploads over the raw REPL, this prevents the upload-corruption class of failure.
:::

## Choosing a Nucleo

Two modules have been tested on this breakout. They run the same firmware and
differ in clock and memory:

| Nucleo | Core | Clock | Flash | RAM | Notes |
|---|---|---|---|---|---|
| **F767ZI** | Cortex-M7 | 216 MHz | 2 MB | 512 KB | The default, and the board every experiment here was run on. |
| **H723ZG** | Cortex-M7 | 550 MHz | 1 MB | ~564 KB | Higher clock; used for timing measurements, not for experiments. |

Other STM32 Nucleo-144 modules share the connector and may work, but have not
been tested here and are not documented as supported.

A Nucleo module is used because its clock speed and its number of GPIO lines are
what allow ten ports, eight motor channels and the dedicated interfaces to come
off **one** board; the modules are also inexpensive, widely available and already
supported by the firmware.

**Why clock speed matters here** is not that task logic is computationally
demanding. The framework runs a single loop in which it samples inputs, services
timers, drives outputs and streams data to the host, so the interval between
successive iterations bounds the delay from an input transition to the
corresponding output, and spare capacity within that interval is what absorbs
concurrent analogue sampling, step-pulse generation and serial transmission.

Response latency and pulse-duration accuracy were measured for the F767 and the
H723 (the latter at low and high polling frequencies) under both light and heavy
load; see [Performance and validation](../reference/performance.md).
