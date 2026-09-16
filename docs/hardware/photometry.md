# Fiber photometry

A pyPhotometry acquisition board plugs into the breakout's dedicated photometry connector. It is a
pyControl analog device on the same MCU, two photodetector channels stream in as analog inputs
while two excitation LEDs are driven from the STM32 DAC, all on the shared firmware clock, so
photometry aligns with video and behaviour offline for free.

```python
photometry = PyPhotometry_board(board.photometry1, sampling_rate=130)
```

The connector (`board.photometry1`) carries six lines: two analog signals (`SIGNAL1`/`SIGNAL2` from
the photodetectors), two DAC LED-control outputs (`LED1CON`/`LED2CON`), and two digital sync outputs
(`DIGITAL1`/`DIGITAL2`).

```{figure} /_static/media/intro/pyphotometry.jpg
:alt: pyPhotometry board on the breakout
:width: 60%
:figclass: pbl-side

The pyPhotometry acquisition board on its dedicated breakout connector.
```

## From a task

The two signal channels are `Analog_input`s that log continuously at `sampling_rate`; drive the LEDs
from the DAC and read back the streamed signal:

```python
def run_start():
    hw.photometry.set_led1(180)     # LED1 drive level (0–255 8-bit / 0–4095 12-bit DAC)
    hw.photometry.set_led2(120)     # LED2 drive level

def run_end():
    hw.photometry.leds_off()        # both LEDs to zero
```

The board exposes `set_led1(value)`, `set_led2(value)` and `leds_off()`; the analog channels
(`photometry.signal1`, `photometry.signal2`, named `photometry_signal1`/`photometry_signal2`) stream
their samples into the data log at the configured `sampling_rate`.

:::{warning}
The F767 has only two DAC channels (PA4 = DAC1, PA5 = DAC2). The photometry LED1 drive uses
DAC1/PA4, the same channel the audio board's control path claims on `port_8`. So a box runs
**either** the audio board **or** photometry LED control, not both. See [Audio](audio.md).
:::
