# Microphone (ESP32 mic)

An ESP32-S3 two-microphone module measures the chamber's sound-pressure level and can capture the
actual acoustic waveform, for tasks that react to what they hear. It connects to the breakout's dedicated
ESP-UART port (UART7), and the driver is `ESP_mic`:

```python
sound_sensor = ESP_mic(uart=7, tx='PF7', rx='PF6',
                       req_pin='PA13',            # OUTPUT: MCU → ESP (record/read trigger)
                       alert_pin='PA14',          # INPUT : ESP → MCU (raised when loud)
                       alert_event='loud_sound')  # framework event on the ALERT rising edge
```

The ESP does the acoustic work continuously (A-weighted dB level, threshold detection, waveform
capture); the MCU driver just fetches results over UART. On the STM32 the UART number fixes the
tx/rx pins (UART7 = PF7/PF6) - `tx`/`rx` here only document the wiring.

:::{note}
Only two GPIOs reach the ESP: **PA13** (the request / record trigger the MCU drives) and **PA14**
(the alert line the ESP drives). Everything else travels over the UART. Wire the two handshake pins
the right way round by direction or the handshake does nothing.
:::

## Sound levels, dB(A)

`read_levels()` returns the latest `(dB_left, dB_right)` pair (or `None` on a failed read). If
`req_pin` is wired it pulses that GPIO so the ESP replies instantly; otherwise it falls back to a
UART poll:

```python
def sample(event):
    if event == 'measure':
        levels = hw.sound_sensor.read_levels()
        if levels:
            print('SPL L={} R={}'.format(*levels))
```

- `read_peak()`, latest peak-hold `(dB_L, dB_R)`.
- `set_calibration(offset_db)`, adjust the ESP's dB calibration offset at runtime.
- `set_alert_thresholds(hi_db, lo_db)`, arm the ALERT line with hysteresis (`lo_db < hi_db`); when
  the level crosses `hi_db` the ESP raises PA14 and the `alert_event` framework event fires, so a
  loud sound can drive a task transition with no polling.

## Waveform recording (reacting to sound)

For verifying that a commanded stimulus was actually delivered, the ESP records the raw mic waveform
over a window gated by the hardware **rec_trig** TTL: the task raises the trigger just before playing
a stimulus and lowers it after, so the ESP captures the real acoustic output at its native sample
rate. Then `get_recording()` streams the waveform back:

```python
def play_and_check(event):
    if event == 'entry':
        hw.sound_sensor.arm_record()        # follow the rec_trig TTL window
        hw.rec_trig.on()                    # HD-owned Digital_output, start of window
        hw.i2s_speaker.play_sine(8000, 100)
        hw.rec_trig.off()                   # end of window
        left, right, rate = hw.sound_sensor.get_recording()
        print('recorded {} samples @ {} Hz'.format(len(left), rate))
```

`arm_record()` with no argument follows the TTL window; pass `arm_record(ms)` to ask the ESP for a
fixed-duration capture over UART instead (a fallback when no TTL is wired). `get_recording()` returns
`(left, right, rate)`, two lists of int16 samples and the sample rate; reads are bounded so a
partial transfer can never hang the task.

:::{note}
`rec_trig` is a plain `Digital_output` owned by the hardware definition, not by `ESP_mic`, declare
it alongside the mic in the HD (see `hardware_definitions/pyBehLab_auditory.py`). The ESP firmware
itself lives in `ESP32S3/mic_firmware.py` and is flashed to the module separately from the pyControl
task upload.
:::
