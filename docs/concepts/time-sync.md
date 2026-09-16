# Time sync (host ↔ MCU framework clock)

## Two clocks

| Clock | Source | Used for |
|---|---|---|
| **Host monotonic** | `time.monotonic_ns()` | Camera capture timestamps; the host side of the framework-clock anchor |
| **MCU framework** | `fw.current_time` on the microcontroller (1 ms tick) | Source of truth for event/state timing in the MCU TSV |

There is no physical sync line between them. The host estimates MCU framework
time at any instant by **linear extrapolation from the most recent message
anchor**, the same technique pyControl uses, but the anchor lives in `pycboard`,
not a separate clock object.

## The anchor lives in pycboard

Every received MCU message updates two fields on the board handle:

- `self.timestamp`, the framework ms carried by that message
- `self._last_message_mono` - `time.monotonic()` when the host received it

One formula maps any host monotonic instant to framework ms
(`source/communication/pycboard.py`):

```python
def _fw_ms_for_mono(self, mono_s):
    return self.timestamp + round(1000 * (mono_s - self._last_message_mono))

def get_timestamp(self):       # framework ms "now" (synthesised WARN/ERROR rows)
    return self._fw_ms_for_mono(time.monotonic())

def fw_ms_at(self, host_ns):   # framework ms at a PAST capture instant
    if not self._fw_anchored:  # gate: no estimate before the first anchor
        return None
    return self._fw_ms_for_mono(host_ns / 1e9)
```

`get_timestamp()` and `fw_ms_at()` share the one extrapolator, so synthesised
MCU rows and video frame rows are expressed on the **same** clock, no
inter-source drift. The `_fw_anchored` gate returns `None` for frames captured
before the first message anchor exists, so a stale estimate is never written.

## What the video data row looks like

`RecorderSink` calls `pyc.fw_ms_at(frame.capture_host_ns)` per frame to write
`frame_fw_ms`, and states the pose's age against that same capture as
`pose_lag_ms` (`source/video/recording/frame_log.py`):

```
columns: frame elapsed capture_host_ns frame_fw_ms pose_lag_ms filter_ms pose zone state
190     00:06.331  35742119033  6255      30.6      0.42     […]  na   choice_state
            ▲           ▲         ▲         ▲         ▲                     ▲
       elapsed      raw host   framework ms  capture   the         MCU-authoritative
       (mm:ss.fff)  instant    at frame    → pose    smoothing's state name (folded
                   capture     exists    share of it into the interval
                   (extrapolated)                    it arrived in)
```

- **`frame_fw_ms`** advances every frame (host extrapolation of MCU time at
  capture) and is the column every later alignment is built on.
- **`pose_lag_ms`** is the age of this row's pose, capture to pose existing:
  the tick, the queue, the fan-out and the model together, not the model alone.
  **`filter_ms`** is how much of it the Kalman and optical-flow smoothing took.
- State transitions are MCU-authoritative; they are folded into the frame row for
  the interval in which they arrived (host arrival order). Events are not in this
  file at all: the MCU TSV records them against the board's own clock and is the
  account to use, cross-referenced by `frame_fw_ms`.

## Inter-message gap consequence

The MCU only sends a message when something happens (event, state, variable
change). During silent periods (long ITI, mid-trial wait), no anchors arrive and
the extrapolation drifts at ~1 ms per second of silence from the last anchor.

Where the task reacts to position this is acceptable, the periods that matter are active, so
events arrive frequently and the anchor stays fresh. Cross-reference the MCU TSV
when you need sub-millisecond accuracy for a specific event.

See `source/communication/pycboard.py` (`_fw_ms_for_mono`, `get_timestamp`,
`fw_ms_at`, `_last_message_mono`).
