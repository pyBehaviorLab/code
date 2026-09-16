# Peripheral modules

An apparatus is defined by **which modules are connected and declared in its
hardware definition**. The breakout board carries ten general-purpose RJ ports,
eight stepper channels and a set of dedicated interfaces. Every module below
plugs into one of these and is named in one line of the hardware definition.

Every driver listed here is a pyControl device driver. A module already wired
into a pyControl rig works here with the same declaration. See
[Hardware compatibility](../introduction.md).

## Behavioural modules

| Module | Connects to | Driver |
|---|---|---|
| **Nose-poke**. IR beam + LED, carries a spare port | One RJ port | `poke.py` |
| **Five-poke**, the five-aperture operant response wall | Two RJ ports | `five_poke.py` |
| **Nine-poke** | Two RJ ports | `nine_poke.py` |
| **Lickometer** | One RJ port | `lickometer.py` |
| **House light**, independently switchable IR, green and white arrays, selected by DIP switch | One RJ port | `LED_driver.py` / `analog_LED.py` |
| **Solenoid driver** | One RJ port | `solenoid_driver.py` |
| **Rotary encoder** | One RJ port | `rotary_encoder.py` |
| **Load cell** | One RJ port | `load_cell.py` |
| **Schmitt trigger** | One RJ port | `schmitt_trigger.py` |
| **uRFID** reader | UART port | `uRFID.py` |

### House light and illumination choice

The house-light board carries **infrared, green and white** LED arrays, each
switchable independently. Choose by experiment, not appearance:

- **Infrared** is outside the range mice see, so video can be acquired in an
  otherwise dark chamber.
- **Green** is used for visible illumination and cue presentation on a reversed
  light cycle: mouse M-cones peak near 510 nm, so a green cue is salient at low
  irradiance, while the photoreceptive system driving circadian entrainment is
  most sensitive near 478 nm.
- **Dim red** illumination, long assumed invisible to rodents, has been shown to
  evoke retinal responses and to alter circadian rhythms, and is **not** used.

## Motors, doors and pumps

Eight stepper channels (`M1`–`M8`) serve **both** reward pumps and maze doors.
Adding doors or pumps therefore never consumes the ports needed for pokes,
lickometers or lights.

This capacity is what makes a six-arm radial maze possible. One channel and one
end-stop input per door would exhaust a six-port board before any sensor was
connected.

| Module | Driver | Notes |
|---|---|---|
| **Peristaltic reward pump** | `stepper_motor.py` | Volume is set by commanded step count, no valves, no pressure regulation |
| **Maze door** | `door.py` | Ramped motion, end-stop aware. See [Interfacing a maze arm](#interfacing-a-maze-arm) |
| **TMC / Trinamic stepper** | `tmc_stepper.py` | Chopper mode, an apparatus needs only mains power and USB |

### Interfacing a maze arm

One arm consumes one stepper channel and one port:

| Arm component | Connects to |
|---|---|
| Door stepper motor | One motor channel (`M1`–`M8`) |
| Upper + lower limit switches | One peripheral port, through the port adapter |
| Reward pump (rewarded arms) | One motor channel |

The board takes a 12 V supply and connects to the host over USB; a separate 12 V
motor supply is optional. The drivers run in chopper mode, so a complete maze
needs only mains power and one USB cable.

The **arm count is bounded by the eight stepper channels**: one channel per door,
plus one per pump on rewarded arms.

Each door is declared as one `Door` object. The controller **learns the step
count to each limit on first use** instead of trusting a configured value, and
derives the acceleration ramp per move from the cruise and start speeds:

```python
home_door = Door(direction_pin='PF2', step_pin='PF9',
                 limit_close_pin='PD6', limit_open_pin='PD5',
                 max_steps=4000, step_rate=2000, min_speed=1000,
                 doorup_dir='backward')
HomeDoor_enable = Digital_output(pin='PD1', inverted=True)   # off() enables the driver
```

`doorup_dir` is the motor direction that *raises* the door. Flip it per door if
an arm travels the wrong way. The full `Door` API and the bottom-limit-only
wiring variant are in [The V2 Nucleo breakout](breakout.md).

:::{admonition} Direction and polarity cannot be checked from source
:class: warning

Motor direction, switch polarity, port assignment and reward calibration are
properties of the physical build. A hardware definition can look correct and
still drive a door the wrong way. Verify each one on the apparatus, at low speed,
before running animals.
:::

**Why peristaltic pumps.** The delivered volume follows the commanded step count
directly, so no valves or pressure regulation are needed. The liquid touches only
the tubing, so cleaning between subjects means flushing or replacing a length of
2 mm OD tube.

Mount pumps outside the animal-accessible volume. **Calibrate before each
experiment**: dispense a known number of deliveries at the task volume, weigh the
total, and check again at the end.

## Audio

| Module | Interface | Driver |
|---|---|---|
| **I2S audio board**, drives two 4–8 Ω speakers | Dedicated I2S connector | `i2s.py` |
| **Audio board** | RJ port with DAC | `audio_board.py` |
| **Audio player** | RJ port | `audio_player.py` |
| **PCM5102** DAC | I2S | `pcm5102_audio.py` |
| **TAS5825M** amplifier | I2S / I²C | `tas5825m_audio.py` |

An I2S codec is provided rather than a toggled pin. This allows auditory stimuli
to be **arbitrary calibrated waveforms**, rather than the square waves a timer
output can generate. See [Audio](audio.md).

## Recording and physiology

| Module | Interface | Driver |
|---|---|---|
| **pyPhotometry board**, analog inputs, LED drivers, digital I/O | Dedicated FP connector (DAC1, DAC2) | `pyphotometry_board.py`, `photometry_acquisition.py` |
| **ESP32 microphone module** | Dedicated MIC connector | `esp_mic.py` |
| **Frame logger / frame trigger** | GPIO | `frame_logger.py`, `frame_trigger.py` |

Photometry connects to the controller directly so acquisition **shares the
controller's own time** rather than being aligned afterwards. See
[Photometry](photometry.md) and [Microphone](microphone.md).

### Synchronising external recording equipment

Controller-generated TTL signals are available on **GPIO**, **JST** headers and a
dedicated **HDMI** connector (lines HDMI 1–7).

HDMI carries several lines in one shielded, latching cable. A single connection
to an electrophysiology system replaces a set of individual coaxial leads, and
**cannot be partly detached during a session**.

Emit sync pulses at **randomly varying intervals** and log each one as an event
on the controller. The two records then align on a unique pulse pattern rather
than on an assumed common start:

```python
# in the task, a sync pulse whose interval is drawn per pulse
hw.sync_out.pulse(10)
print('sync_pulse')
```

## Board-level modules

| Module | Driver |
|---|---|
| Breakout 1.0 / 1.2 / F767 / F767 v2 | `breakout_*.py` |
| Nucleo connector | `nucleo_connector.py` |
| Port expander | `port_expander.py` |
| MCP I/O expander | `MCP.py` |
| UART handler | `uart_handler.py` |
| Grid maze | `grid_maze.py` |

## Flashing an ESP32 module

The microphone modules run MicroPython. Download the firmware `.bin`: the
[ESP32 generic build](https://micropython.org/download/ESP32_GENERIC/) for
pySync max, or the
[Seeed XIAO ESP32S3 build](https://micropython.org/download/SEEED_XIAO_ESP32S3/)
for the mini. Then:

```bash
esptool.py --port <PORT> erase_flash
esptool.py --chip esp32 --port <PORT> write_flash 0x1000 ESP32_BOARD_NAME-DATE-VERSION.bin
```

:::{admonition} An event name that never fires
:class: warning

A peripheral reaches a task only through **name matching**. An event named in a
hardware definition must appear **verbatim** in the task's `events` list.
Otherwise the input is never delivered, with no error and no warning.

Check the names before a session, not after: an event a task lists
and a hardware definition does not is the one cross-file mistake
that produces no message at all.
:::
