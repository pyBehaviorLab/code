# Frame pipeline

The video pipeline is a single producer (CameraThread) feeding a FrameBus that fans out
to per-box sinks. Each sink runs on its own worker thread with a per-box queue.

## What each consumer does when it falls behind

Every consumer has its **own queue, per box**. A queue is bounded, and each
consumer handles a full queue differently. Those three policies are the reason
the platform supports two different ways of working.

Because the queues are per box, a consumer falling behind on one apparatus never
costs another apparatus a frame.

| Consumer | Policy | Queue per box | When that queue is full |
|---|---|---|---|
| **Recorder** | `LOSSLESS` | 90 frames | The arriving frame is dropped and a **health alarm** is raised. Queued frames are never evicted to make room. |
| **Pose (online inference)** | `DROP_NEWEST` | 1 frame | The **arriving** frame is dropped. The queued frame is kept and inferred. |
| **Display** | none |, | Paints whatever frame is most recent. Skips are not counted. |

Every drop by the three queued consumers is written to `_drops.tsv` with its
site and reason. Only the display skips without being counted.

### What "lossless" does and does not mean

The recorder is lossless in two specific senses:

- it never evicts a frame that is already queued, so a frame accepted for
  writing is written;
- it never drops silently. A drop is logged and raises an alarm.

It does **not** mean the recorder cannot lose a frame. If disk or encoder
throughput stays below the capture rate long enough to exhaust the 90-frame
buffer, the arriving frame is dropped and recorded as `lossless_overflow`. A
recorder drop is a fault to investigate, not normal operation.

### Why pose keeps the older frame

With a queue of one and `DROP_NEWEST`, a frame arriving while inference is in
flight is discarded, and the frame already waiting is the one inferred.

Swapping in the newer frame would not help. Both frames are already older than
the inference that is running, so replacing one stale frame with a marginally
less stale one changes nothing and costs a copy. Under load the pose stream
therefore thins out, but the frames it does process are processed promptly.

### The two ways of working

- **Online.** Use it when the task must act on posture during the session.
  Inference skips frames under load, so write position-driven rules on states that
  persist across several frames, zone occupancy rather than instantaneous
  position.
- **Offline.** Use it when the analysis needs pose on every frame. Re-run
  inference over the complete recording afterwards. Every recorded frame carries
  a controller sync timestamp, so the recovered pose aligns with the trial log
  exactly as an online result would. Inference no longer has to keep pace with the
  camera, so many setups can be recorded at once.

One acquisition supports both. See [Time sync](time-sync.md) and
[Performance and validation](../reference/performance.md).

## Producer → bus

```
CameraThread (one per physical camera)
    │  polls camera.grab() in a tight loop
    │  10 ms sleep when batch is empty
    │  pushes (CameraFrame, capture_host_ns) into a buffer
    ▼
Pipeline._tick_loop @ 100 Hz (single thread, all cameras)
    │  drains every CameraThread's buffer
    │  for each frame, splits into per-box BoxFrame via VideoSegmentProcessor (if shared cam)
    │  publishes BoxFrame on FrameBus(cam_id)
    ▼
FrameBus (per camera): publish(BoxFrame) fans to all subscribed callbacks
    │
    │  per-box subscriptions:
    └──► RecorderSink  ─►  mp4 + _video_data.txt (LOSSLESS queue per box)
         PoseSink      ─►  DLC/SLEAP inference (DROP_NEWEST queue, depth=1)
         MCUPusher     ─►  coords + intrinsic events to MCU (event-driven, no frame queue)
```

The GUI video tiles are **not** a sink, they are painted by polling the latest
frame each `process_timer` tick (`base._paint_streaming_cameras_once`). The old
queued DisplaySink dropped frames upstream when overlay drawing got slow, so it
was replaced by polling the most recent frame.

**Every** box paints this way, at camera rate, tracking boxes included. The
overlay (pose keypoints) is drawn from the last result the box
produced (`base._overlay[setup_id]`, TTL-bounded), so the video stays at the
full capture FPS while the overlay updates at the inference rate. There is no
separate per-frame display path for tracking boxes.

## Sink ownership of the FrameBus

`source/video/framebus/sink_base.py` defines the base. Each sink:

- Owns one worker thread.
- Owns one per-box queue (deque or bounded).
- Pops from a wake queue, picks the box with work, processes, re-adds if more frames
  pending (round-robin fairness).
- Drop policy is per-sink: LOSSLESS for recorder, DROP_NEWEST for pose, DROP_OLDEST
  for tracker.

## Why per-sink threads, not per-box threads

Per-sink keeps the number of OS threads bounded (3 frame-consuming sinks: recorder,
pose, tracker; MCUPusher is event-driven and owns no frame queue) regardless of how many boxes
are in the rig. Per-box queues inside each sink give isolation: a slow recorder on box 1
doesn't block recording on box 2 because they're separate queues processed by the same
worker round-robin.

## Tick rate + camera backoff

- Pipeline tick: 100 Hz (`_tick_interval_s = 0.010` in `controller.py:162`).
  Camera frames at 30 fps = 1 frame every ~33 ms = drained on ~every 3rd tick.
- Camera thread empty-batch sleep: 10 ms (`capture.py:610`).
  USB cameras deliver every ~16-33 ms; 10 ms backoff keeps latency invisible while
  cutting idle wakeups 10× vs the previous 1 ms loop.

## Pose subsystem

Per-box `current_signature` = `(model_path, tracker_type, resize, confidence, body_parts)`.
(`n_instances` is a runtime knob, changing it does NOT rebuild the model, so it is not
part of the signature.) Init button in the tracking dialog stays disabled when the
signature matches the live PoseSink; re-enables when any of those fields changes
(`_mark_dlc_dirty`).

The dialog's body-part picker (single combo, `zone_body_part_combo`) drives both:

- `PoseSink.set_centroid_body_part(box_id, name)`, which keypoint defines the
  centroid used for `location` + `zones_by_body_part["centroid"]`.
- `MCUPusher.policy.zone_change_body_part`, which keypoint's zone occupancy fires
  the intrinsic `zone_changed` event.

See [tracking pipeline](tracking-pipeline.md).
