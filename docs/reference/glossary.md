# Glossary

| Term | Meaning |
|---|---|
| **Box** | One physical operant/maze setup. Owns one MCU + (usually) one camera. |
| **Setup** | Maze synonym for "box". Same widget contract; the maze-mode shell calls it a "setup widget". |
| **MCU** | The microcontroller (STM32 + MicroPython + pyControl). One per box. |
| **pyControl** | The MicroPython framework on the MCU (`source/pyControl/`). Extended state machine + timer + IO. |
| **Extended state machine** | Finite states + events augmented with variables and arbitrary code, pyControl's task model. |
| **Pycboard** | Host-side serial connection to one MCU (`source/communication/pycboard.py`). |
| **Pipeline** | The video processing controller (`source/video/framebus/controller.py`). Owns sinks + tick thread. |
| **Sink** | One processing stage downstream of FrameBus, each with a back-pressure policy: RecorderSink (lossless), PoseSink (drop-newest), TrackerSink (drop-oldest). MCUPusher consumes results but is not a queued sink; the live display is painted on the GUI thread, not a sink. |
| **Lossless / lossy** | A sink is lossless if it processes every captured frame (recorder, tracker) with explicit drop accounting; only the live display may skip frames. |
| **Run lineage** | The recorded origin and code version of every artefact that lets a session be reproduced, here via djb2 content-addressed snapshots and the append-only change log. |
| **Latency / jitter** | Delay between a cause and its effect, and the variability of that delay. The MCU bounds both for real-time control; the host overlay is made frame-accurate to avoid perceived lag. |
| **FrameBus** | Per-camera fan-out. Sinks subscribe per-box. |
| **TC** | `framebus.TrackingConfig`, per-box live tracking state in the pipeline. |
| **cfg.tracking** | Rig-level tracking config in the project YAML. Lifted from + applied to per-box TCs. |
| **mcu_serial** | USB serial number of the microcontroller. Stable across replug. Replaces `com_port` as the binding. |
| **com_port** | Device path (`/dev/ttyACMn` / `COMn`). Now legacy / fallback only. |
| **DLC** | DeepLabCut. Pose-estimation backend. |
| **SLEAP** | SLEAP. Alternative pose-estimation backend. |
| **Zone** | Polygon / rectangle drawn on the camera frame; tracker reports inside/outside per body part. |
| **zone_changed** | Framework-intrinsic MCU event fired when the configured body part's zone occupancy changes. Silent dispatch (no TSV row). |
| **Centroid** | Synthetic centre PoseSink computes from a configured body part (or first-confident fallback). |
| **fw_ms_at / get_timestamp** | `pycboard` methods that map a host monotonic instant to MCU framework ms via the last-message anchor (`_fw_ms_for_mono`). One extrapolator, gated by `_fw_anchored`. |
| **frame_fw_ms** | MCU framework time (ms) at frame capture, written raw into `_video_data.txt`. The column every alignment to the task is built on. |
| **pose_lag_ms / filter_ms** | Ms from a frame being captured to its pose existing (tick, queue, fan-out and model, not the model alone), and how much of that the Kalman and optical-flow smoothing took. `pose_lag_ms` was called `infer_ms`. |
| **Frame-matched display** | Live overlay painted on the exact frame it was computed from (tagged by `cam_frame_id`), so keypoints don't trail a moving subject under inference latency. |
| **task_hash, hd_hash, model_djb2, config_djb2** | djb2 8-hex content hashes for cross-file lineage. |
| **Snapshot store** | `<project>/source/` content-addressable copies of every saved artifact. |
| **Run** | One Record press per box. One 6-field row in `runs/YYYY-MM-DD.json`. |
| **change_log** | Append-only event log at `<project>/change_log.jsonl`. |
| **Persistent variable** | Task variable whose final value is captured at Stop and restored at next Upload (per subject). |
| **API class** | Per-task host-side adaptive hook (`api_classes/<TaskName>.py`). |
| **VID/PID** | USB vendor ID / product ID. Microcontrollers: 0xF055 / 0x9800 (or 0x9801). |
| **Stats canvas** | Live MCU-event-driven dashboard (`source/stats/canvas.py`). |
| **djb2** | Hash function used everywhere for content addressing. Fast, 8-hex output. |
| **`_video_data.txt`** | Per-session per-box frame log next to the mp4. v2 format = 8 cols, one row per frame, ending `pose  zone  state`. |
