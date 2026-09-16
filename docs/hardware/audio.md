# Audio

Two audio paths plug into the breakout. The **I2S audio board** delivers clean, low-latency
synthesised and recorded stimuli through a dedicated amplifier; the **DAC audio board** is the
pyControl-standard tone generator driven straight off the STM32 DAC. A box uses one or the other.

## I2S audio board

A dedicated I2S audio board delivers tones, noise, FM sweeps, chords, pulsed/stepped stimuli, a
continuous masking noise with cue-in-noise, and streamed WAV playback. The I2S clocks run
continuously and each stimulus is pre-rendered and played with one gapless blocking write, so the
sound stays clean. Two amplifier variants share the same `I2S_Audio` API and differ only in how the
amp is controlled.

::::{tab-set}

:::{tab-item} TAS5825M
An I²C class-D amplifier with hardware volume (`set_volume(0-100)`). Configure the I²C control bus,
the power-down pin and the I2S data lines:

```python
# TAS5825M class-D amp: I²C1 control (SCL=PB8, SDA=PB9, PDN=PB6),
# I2S2 data (WS=PB12, SCK=PB13, SD=PB15)
i2s_speaker = TAS5825MAudio(
    i2c_id=1, i2c_scl='PB8', i2c_sda='PB9', pdn_pin='PB6',
    i2s_id=2, i2s_ws='PB12', i2s_sck='PB13', i2s_sd='PB15',
    level=50, sample_rate=48000)
```

`level` (0–100) sets the hardware digital volume; `set_volume(0-100)` and `mute()` change it at
runtime. Output is capped at 90 % of full scale, so `level=100` is never hard-clipped.
:::

:::{tab-item} PCM5102
A strap-pin DAC, no I²C, no registers, no hardware volume. Wire only the I2S data lines, plus an
optional soft-mute (XSMT) pin if it reaches an MCU GPIO:

```python
# PCM5102 DAC: I2S2 data (WS=B12, SCK=B13, SD=B15), optional XSMT mute pin
i2s_speaker = PCM5102Audio(
    i2s_id=2, i2s_ws='B12', i2s_sck='B13', i2s_sd='B15',
    xsmt_pin=None, sample_rate=48000)
```

Loudness is set per stimulus by the `level` argument (the DAC has no volume register), so
`set_volume` is a no-op here. Leave `xsmt_pin=None` if XSMT is strapped high on the board.
:::

::::

### Playing from a task

The device named in the hardware definition (here `i2s_speaker`) is called directly. `play_*`
helpers render and play in one call; `load_*` pre-renders a named stimulus once so `play(name)`
replays it with no build latency:

```python
hw.i2s_speaker.play_sine(10000, 100, ramp_ms=5, level=0.5, sync=True)  # 10 kHz tone, 100 ms
hw.i2s_speaker.play_noise(80, level=0.25)                              # 80 ms white noise
hw.i2s_speaker.play_sweep(5000, 15000, ms=80)                          # FM sweep
hw.i2s_speaker.play_chord([4000, 6000, 9000], 120)                     # sum-of-sines chord

hw.i2s_speaker.load_sweep(10000, 15000, ms=80, name='cue')            # pre-render once …
hw.i2s_speaker.play('cue')                                            # … replay with no latency

hw.i2s_speaker.start_noise(level=0.2)                                 # continuous masker
hw.i2s_speaker.play_cue_in_noise(8000, 100)                          # tone mixed into the masker
hw.i2s_speaker.stop_noise()

hw.i2s_speaker.play_wav('/sd/cue.bin')                               # WAV streamed from SD
```

Also available: `play_square`, `play_click`, `play_pulsed_sine`, `play_stepped_sine`, and the
matching `load_*` builders. `sync=True` pulses the sync TTL for the stimulus so its true onset is
recorded.

:::{note}
Every frequency is clamped below Nyquist, at 48 kHz the usable band tops out at ~21.6 kHz
(0.45 × sample rate). Keep synthesised cues short: RAM is 8 bytes per frame, so a 300 ms tone
(~115 KB) will `MemoryError` on the small MCU heap. Load long audio from SD instead.
:::

### WAV playback from SD

`play_wav` streams a **headerless raw blob** from an SD card via the non-blocking IRQ feed, so
arbitrarily long audio plays without a RAM ceiling and without freezing the framework. Pre-convert a
`.wav` into the engine's playback format (16-bit sample left-justified in a 32-bit stereo slot at
the board's sample rate) with the host-side tool:

```bash
python tools/wav_to_i2s_blob.py input.wav /sd/cue.bin --rate 48000
```

Then `hw.i2s_speaker.play_wav('/sd/cue.bin')` (add `loop=True` for continuous playback, `stop_wav()`
to end it). The SD card must be mounted, pin/slot wiring is the hardware definition's job.

```{figure} /_static/media/intro/i2s-audio-with-esp32.jpg
:alt: I2S audio board with ESP32
:width: 60%

The I2S audio board with the on-board ESP32 mic populated.
```

```{figure} /_static/media/intro/i2s-audio-without-esp32.jpg
:alt: I2S audio board without ESP32
:width: 60%

The same board without the ESP32, audio only.
```

## DAC audio board

`Audio_board` is the pyControl-standard tone generator: it plays directly off an STM32 DAC channel
with an I²C digital pot for volume. It plugs into a port that provides both a DAC and an I²C bus
(`port_8` on the breakout):

```python
speaker = Audio_board(port=board.port_8)     # needs a port with DAC + I2C
```

From a task the API is the plain pyControl audio interface:

```python
hw.speaker.sine(5000)               # 5 kHz sine tone (runs until changed / off)
hw.speaker.square(5000)             # square wave
hw.speaker.noise()                  # white noise
hw.speaker.click()                  # single click
hw.speaker.clicks(20)               # clicks at 20 Hz
hw.speaker.pulsed_sine(5000, 10)    # 5 kHz sine pulsed at 10 Hz
hw.speaker.stepped_sine(2000, 8000, 5, 4)  # 5 steps 2→8 kHz at 4 Hz
hw.speaker.set_volume(90)           # digital-pot volume, 0–127
hw.speaker.off()                    # silence
```

:::{warning}
The `Audio_board` claims a DAC channel (DAC1/PA4 on `port_8`), which conflicts with the pyPhotometry
LED drive, a box runs **either** the DAC audio board **or** photometry LED control, not both. See
[Fiber photometry](photometry.md).
:::
