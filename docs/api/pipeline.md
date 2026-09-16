# `source/video/framebus/`, pipeline + sinks

## `controller.py` - `Pipeline`

Owns the tick thread, the per-camera FrameBuses, the sinks, and the per-box
`TrackingConfig` registry.

### Lifecycle

```text
pipe = Pipeline(..)
pipe.add_camera(cam_id, camera_thread)
pipe.subscribe_box(box_id, cam_id)             # wire box -> bus
pipe.update_tracking_config(box_id, **fields)  # partial-merge field update
pipe.apply_tracking_config(box_id)             # full apply: cfg -> all subsystems
pipe.start_recording(box_id, paths..)       # RecorderSink begins
pipe.stop_recording(box_id)
```

### Internal state

- `_tracking_configs: Dict[int, TrackingConfig]`, per-box live config
- `_pycboards: Dict[int, Pycboard]`, register via `register_box(bid, pyc)`
- `_buses: Dict[Any, FrameBus]`, per camera id
- `_box_unsubs: Dict[int, List[Callable]]`, cleanup hooks per box

### Tick

`_tick_loop` runs every `_tick_interval_s` (10 ms = 100 Hz). Drains each connected
CameraThread, publishes each frame to its bus.

### Camera liveness watchdog

Each tick also runs `_observe_camera_liveness()`: it reads each CameraThread's
frame-flow state (`streaming` | `stalled` | `reconnecting`)
and, **on a state change only**, fans a per-box health banner to every box on that
camera via `_notify_health` → `QtBridge.health` → `base._on_box_health`:

- `stalled` → "Camera stopped delivering frames"
- `reconnecting` → "Camera lost, reconnecting…"
- recovery → "Camera recovered"

The state itself is owned by `CameraThread` (see `docs/api/…` / `capture.py`): a
brief loss marks the camera STALLED as soon as frames stop (`_stall_after_s`), and
for OpenCV it retries `reconnect()` on a backoff forever, the thread is never
killed on frame loss. `VideoManager.is_camera_streaming` gates on this so a box
never "records" from a stalled camera.

## `types.py` - `BoxFrame`, `TrackingConfig`

`BoxFrame`:
```python
@dataclass
class BoxFrame:
    image: np.ndarray               # BGR (per-box crop for shared cameras)
    setup_id: int
    cam_frame_id: int               # links capture → derived BoxFrames → pose
    capture_host_ns: int            # host monotonic ns at grab (ordering + latency)
    crop_origin / crop_size: Optional  # None for a dedicated camera
    poll_host_ns: Optional[int]     # host ns when the tick drained + published it
    image_rgb / image_gray          # cached colour views (RGB for inference)
```

MCU framework time is **not stored** on the frame, each sink derives it on
demand by mapping `capture_host_ns` through `pycboard.fw_ms_at`.

`TrackingConfig`:
- box_id, tracker_type, dlc_model_path, keypoint_names, confidence_threshold,
  pose_resize_factor, pose_n_instances, rotation_enabled, rotation_keypoints
- zones (List[Zone])
- annotation_enabled, online_tracking_enabled
- **push_zones_to_mcu, push_coords_to_mcu, zone_change_body_part**
- user_applied (set True when the user clicks Apply in the dialog)

`to_json` / `from_json` for round-trip.

## `frame_bus.py` - `FrameBus`

One per camera. `on_box_frame(box_id, callback)` adds a per-box subscription.
`publish_frame(camera_frame)` fans the raw frame to camera subscribers, then
derives one `BoxFrame` per registered box (cropped for shared cameras) and
fans it to that box's subscribers.

## Sinks

### `recorder_sink.py` - `RecorderSink`

LOSSLESS per-box queue. On every BoxFrame:
- Append to mp4 (via FFmpeg / NVENC / cv2.VideoWriter, picked at session start)
- Append row to `_video_data.txt`

### `pose_sink.py` - `PoseSink`

DROP_NEWEST per-box queue depth=1. Backend = DLC or SLEAP. Per box:
- `_centroid_body_part: Dict[int, str]`, which keypoint defines the centroid
- `_rotation_*`, optional rotation estimation per box
- `_confidence`, global threshold

On every inference result, fires `on_result` callbacks with
`(box_id, cam_frame_id, pose_array, location, speed, zones_by_body_part, raw_pose_dict, *, capture_host_ns)`.

### `tracker_sink.py` - `TrackerSink`

DROP_OLDEST. Zone occupancy. Fires the same shape of result as
PoseSink so MCUPusher can consume both uniformly.

### `mcu_pusher.py` - `MCUPusher` + `TrackingPushPolicy`

Not a sink, consumes pose/tracker result callbacks. Per box:
- Intrinsic `zone_changed` event diff (single body part, silent dispatch via
  `pycboard.queue_trigger_intrinsic_event`)
- Per-zone enter/exit triggers (TSV-logged via `queue_trigger_event`)
- `c.*` coord mapping (silent set via `queue_set_coordinates`)

### Display, not a sink

The live preview is driven from the GUI thread, not a queued sink (an early
`DisplaySink` dropped frames under slow overlay drawing). A single path drives
every box:

- **Polling paint** - `base._paint_streaming_cameras_once` shows the latest
  captured frame for **every** box at camera rate. The overlay (pose keypoints /
  keypoints) is drawn on top from the box's most recent result
  (`base._overlay[setup_id]`, TTL-bounded), so video runs at capture FPS while
  the overlay refreshes at the inference rate.
