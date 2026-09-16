# `source/video/tracking/`, pose backends

Estimators are **backend-agnostic**: PoseSink drives any of them through one
interface. Files: `pose.py`, `inference.py`, `smoothing.py`,
`speed.py`.

## `pose.py`, pose estimators

A common base, `PoseTracker`, with two methods. **The frame handed in is already
RGB** (`BoxFrame.image_rgb`, converted once per camera frame); `_to_model_input`
only validates the channel count.

```python
PoseTracker.predict(frame) -> Dict[str, [x, y, conf] | None]      # one frame
PoseTracker.predict_batch(frames) -> List[Dict[..]]              # one forward pass
```

Subclasses:

- **`DLCLiveTracker`**. DeepLabCut-Live (`dlclive.DLCLive(model_path, resize=…)`).
  `resize_factor` (default 1.0) is DLC's own internal resize; coordinates come back
  in full-frame space. The TF/PyTorch session lazy-initialises on the first real
  frame.
- **`SLEAPTracker`**. SLEAP backend (tries sleap-nn / PyTorch, falls back to
  TensorFlow). Same `predict` contract.

Body-part names are read from the model config and override the user default. On
`initialize()` each backend logs the resolved **inference device** ("… inference
device=CUDA (…)" vs "CPU"), via `source/video/gpu.py::inference_device`, so a
silent CUDA/driver mismatch that demotes inference to CPU is visible in the log.

## `inference.py`, inference backends

- **`ThreadInferenceBackend`** (default), one `ThreadPoolExecutor(max_workers=1)`;
  pose models are not thread-safe, but the C-extension releases the GIL during the
  forward pass so the GUI thread is not blocked. A per-box in-flight gate drops a
  new frame while that box's previous inference is still running (bounded latency).
- **`MultiInstanceInferenceBackend`**. N independent model copies for parallel
  inference; selected via `PoseSink.set_n_instances(n)`.

`PoseSink` groups same-shape frames and submits them as **one batched forward
pass** (the DeepStream `nvstreammux`/`nvinfer` pattern), so M boxes cost one GPU
call, not M.

## Result shape

`PoseSink` converts the estimator's dict to `pose_array` (`[[x, y, conf], …]` in
body-part order) and fires subscribers with
`(box_id, cam_frame_id, pose_array, location, speed, zones_by_body_part,
raw_pose_dict, *, capture_host_ns)`. Framework time is **not** attached
here; the recorder maps `capture_host_ns` through `pycboard.fw_ms_at` when it
writes the row.

## `smoothing.py` - `TrackingEnhancer` (Kalman + optical flow, optional)

Per-box latency compensator. Smooths the centroid and predicts 1–2 camera frames
ahead; the forecast drives zone occupancy for the MCU push (closing the
inference-latency gap), while the smoothed current position drives display + speed.
Off by default; per-box toggle.

## `speed.py` - `compute_speed`

Shared by PoseSink + TrackerSink. Takes `prev_centroid = (cx, cy, mono_ns)` +
`(cx, cy, now_ns)` → speed in px/s (or m/s when a scale calibration exists), so the
figure on the display and the `c.speed` pushed to the task are comparable across
backends. It is no longer written to `_video_data.txt`: speed follows from the pose
and the scale calibration, so the analyser recomputes it rather than storing it.

## Configuring the backend per box

`PoseSink.configure_model(tracker_type, model_path, probe_frame, resize_factor,
body_parts, confidence)` loads or re-uses a cached model handle (cache key =
`(path, w, h, resize, type)`). `PoseSink.current_signature()` returns
`(model_path, tracker_type, resize, confidence, body_parts)`; the tracking dialog
compares it against the configured signature to gate the Init button.
