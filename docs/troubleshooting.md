# Troubleshooting

## "I opened my project and the DLC model is gone"

Symptom: tracking dialog shows empty model_path, no body parts. Config file shows
`cfg.tracking.dlc: {}`.

Cause: pre-fix autosave bug (now fixed in `read_ui_into_config`). Some non-tracking
event (camera change, box wiring, bg capture) triggered an autosave with an empty
live pipe; that autosave wiped `cfg.tracking` to defaults.

Fix: nothing for you to do. On next project load, `_recover_tracking_from_snapshots`
walks `<project>/source/configs/*.json` and restores the most-recent populated
snapshot. Check the per-session `.log` for:
```
tracking recovery: cfg.tracking was empty; restored mode=…, dlc.model=…, body_parts=… from snapshot <hex>.json. Next autosave will persist the recovered config.
```

If recovery itself fails (every snapshot is also wiped, project never had a good
save), pick the model in the dialog. The fix prevents re-occurrence; next save sticks.

## "Box 1 connected to box 2's MCU"

Symptom: After replug or reboot, the box widget connects but the wrong physical
microcontroller responds (poke pin events show up on the wrong box).

Cause: Linux's `/dev/ttyACMn` numbering shuffled. The old behaviour was to bind on
the device path (whatever `ttyACM` got assigned), not the physical board.

Fix: as of the May 2026 stable-serial work, projects bind on USB serial number
(`BoxConfig.mcu_serial`). If your project pre-dates the fix, it has `mcu_serial=""`
and `com_port="/dev/ttyACMn"`. The first connect captures the live USB serial and
auto-upgrades the project. From the next session forward, port shuffling is invisible.

See [user-guide/boards](user-guide/boards.md).

## "MCU not in the dropdown"

Cause: The MCU's MicroPython firmware doesn't expose a USB serial-number descriptor.
`pyserial` returns `serial_number = None` for it, so it's excluded from the picker.

Fix: re-flash MicroPython with the standard build (includes the USB descriptor by
default). The MCU's `pyb.unique_id()` is NOT the same thing, that's an internal
chip id, not a USB descriptor.

## "zone_changed doesn't appear in MCU TSV"

Expected. As of the May 2026 silent-dispatch change, `zone_changed` is dispatched
to the MCU state machine without being logged, same semantics as `entry` and
`exit`. If you want it visible in your data, add `print(..)` to your state
handler when `event == "zone_changed"`.

## "My task crashed: KeyError on event_ID"

Probably: the MCU's pyControl framework is stale. The host triggered an event
whose ID isn't in `sm.events`. Most common: `zone_changed` event missing.

Fix: re-upload `source/pyControl/` to the MCU. The current `state_machine.py`
auto-injects `zone_changed`; older versions don't.

## "Camera says connected but no frames"

Check:
1. Camera Connect dialog reports streaming (status flips to green).
2. Per-box preview widget shows the camera feed within ~3s of connect.
3. `_drops.tsv` next to the video dir for excessive drop counts.

If preview is black but logs say connected: per-box ROI may be cropped out of frame
range. Check `cfg.setup_config.boxes[].geometry` vs camera resolution.

## "Project save says 'conflict, deferred'"

`source/config/multi_instance.py` is doing OCC and detected another process touched
the project file. The save is deferred; an auto-merge will fire on the next change,
or you can dismiss + retry.

Hard conflict (auto-merge can't reconcile) → modal asks you to pick the winning
version.

## "DLC won't initialise"

Init button stuck red:
1. Open the per-session `.log`. DLCLiveTracker prints its load error there.
2. Check the model path exists on disk (`cfg.tracking.dlc.model_djb2`).
3. Check GPU availability if you expect GPU inference (`PYBL_DLC_GPU=1` to force).
4. Check the per-box "Pose result" empty-streak warning, fires after 3 / 20 /
   100 consecutive empty results.

## "A video tile shows red 'Camera not connected'"

Expected. In the Video Stream tab (and maze setup tiles) a tile turns its
placeholder text red when a camera id is configured for that box but the camera
isn't connected yet. Connect the camera (Camera Connect, or auto-connect on load)
and the tile clears to normal once it streams.

## "Camera won't disconnect"

As of the June 2026 pass every disconnect routes through one path
`MainWindowBase.disconnect_all_cameras()` → `pipeline.disconnect_camera(setup_id)`
(sink unsubscribe + FrameBus unregister + device release). The Camera Connect
dialog's "Disconnect Cameras" uses it too; it no longer pokes `video_manager`
directly (the old shortcut left sinks subscribed and corrupted the next reconnect).

For a **shared camera** (several boxes segmented from one device) the physical
device is released only when the **last** box on it disconnects, disconnecting one
of six boxes on camera 0 is expected to leave the device open for the other five.
The thread join is now bounded (3 s) so a flaky device can't freeze the GUI.

## "Parallel task upload hangs"

Fixed. The serial port has no global read timeout, so a board that goes silent
mid-upload (raw-REPL desync, common when several boards enter raw REPL at once)
used to block a worker forever, and the progress dialog waited for it with no exit.
Reads in `pyboard.read_until` and `pycboard.transfer_file` are now deadline-bounded,
so a silent board surfaces as the normal upload-failure path instead of a hang.

If a box still fails to upload: disable the USB mass-storage / flash drive on that
microcontroller (a known source of raw-REPL desync) and retry. Tune concurrency with
`PYBEHAVIORLAB_MCU_PARALLELISM` (default 4).

## "PoseSink: frames arrived but no model handle is loaded"

This warning now fires **only** when at least one box has pose tracking *enabled*
but no model is loaded, a camera-only or tracking-off session no longer triggers
it. The pose model loads at the first Record click (or via the Camera Connect
dialog), not on auto-connect-on-load. If you see it during a real tracking run, the
model failed to load, see "DLC won't initialise" above.

## "1 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider"

Harmless, and only the SLEAP exports print it. Nothing is misconfigured and the
GPU is being used.

The SLEAP graph reads the width of its own confidence maps with a `Shape` node
followed by a `Gather`, and uses that one number to turn a flat peak index into
x and y. ONNX Runtime has a rule that shape lookups run faster on the CPU, so it
places that single node there. The number then has to be copied back to the GPU
for the three nodes that use it, and the copy is what the message counts. On the
v8 export that is 62 of 63 nodes on the GPU and one 8-byte value crossing per
forward pass.

Measured here at batch 8 and 256x256, against a copy of the graph with the
lookup replaced by a fixed number: 9.9 ms per batch as exported against 9.7 ms
without the copy. About 2%, and inside the run to run spread. The warning also
mentions CUDA graph capture, which pyBehaviorLab does not use.

The lookup cannot simply be baked in: the export takes any height and width, so
the confidence map size is only known at run time. The DeepLabCut export does
not print the message at all, all 438 of its nodes are placed on the GPU.

What would be worth acting on is the same message with a **large** node count,
or `Failed to create CUDAExecutionProvider`. Both mean real work fell back to
the CPU. To see which nodes, set `session_options.log_severity_level = 1` and
read the `Force fallback to CPU execution for node:` lines.

## "Auto-connect opened the camera at the wrong FPS / it took several seconds"

FPS: auto-connect uses the project's saved capture rate
(`cameras.video_defaults.target_fps`), re-asserted onto every camera on load, not
the camera's maximum. Change it in Camera Connect and re-save if it's wrong.

Slow first open: a **cold** camera open + driver settle (`camera_settle_ms`,
default 5 s) takes ~6 s on Windows, that's the driver, not a hang. The streaming
wait is sized to outlast it, so the tile comes up cleanly; the manual dialog feels
instant only because its preview already warmed the device.

## Where to look first

- Per-session log: `<project>/data/<task>/<date>/mcu/<stem>.log`
- Per-launch log: `data/log/<APP>_<YYYY-MM-DD>_pid<N>.log`
- Change log: `<project>/change_log.jsonl`
- Snapshot store: `<project>/source/configs/`, `source/<hex>.py`, `source/dlc/<hex>.json`
