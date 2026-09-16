# `source/video/recording/`, video + per-frame log

## `recorder.py` - `VideoRecorder`

Frames are fed in by the pipeline's `RecorderSink` via `add_frame()`; a background
thread encodes and writes them so the GUI never blocks. Per-frame metadata is owned
by `FrameLog` / the tracking writer, not the recorder.

### Encoder selection (`VideoWriterFactory.create_writer`)

Platform-aware, best-first: `h264_nvenc` → Intel `h264_qsv` → AMD `h264_amf` →
Jetson `h264_v4l2m2m` → CPU `libx264` → OpenCV MJPEG/XVID (`prefer_hevc=True`
swaps in `hevc_nvenc` / `libx265`).

**NVENC session-limit admission control.** Consumer GeForce drivers cap concurrent
NVENC sessions (~12); a 16-box rig can exceed that. When a hardware encoder is
present but its writer won't open:

- `allow_cpu_fallback=True` (default, config `video.allow_cpu_fallback`): fall
  through to CPU `libx264` so the box **still records**, with a loud warning. This
  is the "rock solid, don't lose the box" behaviour.
- `allow_cpu_fallback=False`: refuse and return `None` (strict hardware-only).

Live hardware sessions are counted by `ffmpeg.hw_encoder_sessions()` (incremented
on a successful HW writer open, decremented on `release()`).

### When a frame fails to write

The encode loop guards each frame: a transient `writer.write()` / `resize` error is
logged and skipped (it never kills the encode thread with `recording` stuck True).
After `_WRITE_ERROR_FATAL` (150 ≈ 5 s @ 30 fps) consecutive failures it gives up and
fires the `drop_callback` with reason `"encoder_write_failed"` so the box can alarm.

`VideoRecorder(.., use_gpu="auto"|True|False, allow_cpu_fallback=True)`;
`add_frame(frame, capture_time_ns)` queues a frame; `stop_recording()` + `close()`
finalise the mp4 (reap the ffmpeg child off the GUI thread via
`base._async_stop_recorder`).

## `frame_log.py` - `FrameLog`

Writes `_video_data.txt` v2. One row per camera frame, 9 tab-separated columns.

### Construction

```python
log = open_with_headers(
    filepath,
    subject_id=.., setup_id=.., start_dt=..,
    task_name=.., task_hash=.., hd_name=.., hd_hash=..,
    tracking_mode=.., resolution=(w, h), fps=..,
    zones=.., roi=.., video_filename=..,
    px_per_m=<float|None>,
)
```

Framework time is not passed in. RecorderSink maps each frame's
`capture_host_ns` through `pycboard.fw_ms_at` when it writes the row.

### Per-frame API

```python
log.write_frame(frame_number, timestamp_ms, *,
                speed=None, location=None, pose_array=None,
                capture_host_ns=None)
log.note_event(name)
log.note_state(name)
log.on_dropped_frame(count=1, note="dropped")
log.close()
```

### Internal flow

- `write_frame` opens a "pending" row for the next frame; if there's a previous
  pending it's written first (one-frame lookahead so state/events arriving slightly
  after a frame's capture still fold into the same row).
- `note_event` / `note_state` append into the current pending's `events` / `state`
  list.
- Per-session footer is appended on `close()`.

### Speed conversion

`_inv_px_per_m` cached at construction. `_format_speed` uses multiplication so the
per-frame hot path doesn't do a division.

## Framework-time mapping

There is no separate clock object. The host↔framework anchor lives on the board
handle (`pycboard`): every received MCU message updates `self.timestamp` +
`self._last_message_mono`, and `pycboard.fw_ms_at(host_ns)` extrapolates MCU
framework ms for any host capture instant. RecorderSink calls it per frame to
write `frame_fw_ms`; the pose's age relative to that capture is written as
`pose_lag_ms`, so the row states the lag rather than leaving it as a subtraction.

See [time-sync](../concepts/time-sync.md).

## `drop_log.py` - `drop_log`

Process-wide ring buffer (1024 entries) for dropped frames. Periodically flushed to
`<video_dir>/_drops.tsv`. RecorderSink calls `drop_log.record(..)` whenever a
LOSSLESS queue overflows.

## Build session stem

```python
from source.video.recording import build_session_stem
stem = build_session_stem(subject_id, box_id, start_dt)
# → e.g. "A31-Box1-2026-05-30-175911"
```

Used by FrameLog, Recorder, MCU TSV, and the per-session .log so all four files share
the same name root.
