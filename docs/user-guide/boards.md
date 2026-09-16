# MCU boards

Each box has one physical microcontroller. The host opens a USB-CDC serial connection and
drives the pyControl framework.

## Stable identity (post May-2026 fix)

`/dev/ttyACMn` (Linux) and `COMn` (Windows) shuffle on every replug. The Box ↔ MCU
binding lives on the **USB serial number**, not the device path.

- `BoxConfig.mcu_serial` stores the serial (e.g. `"315535563234"`).
- `BoxConfig.com_port` is legacy, kept for back-compat as a fallback only.

See [reference/file-formats](../reference/file-formats.md) for the saved schema.

## Connecting

Toolbar → **Connect MCU** → dialog lists every plugged-in microcontroller by serial number.
```{figure} /_static/media/gui/group-setup-control.png
:alt: The Setup Control strip: Clear Meta, Connect boards, Session Plot, Config boards, Multi-Start, Multi-Stop, Upload Task, Disconnect and Analysis

**Setup Control**, along the bottom of Main Control, wrapped onto two lines
to fit the page. These act on every eligible box at once; the per-box
equivalents live on each box row.
```

```{figure} /_static/media/gui/board-connect.png
:alt: The Connect dialog with one row per box, a port dropdown and tick each, a Connect button and a refresh button
:figclass: pbl-side

One row per box, and a rescan button for a board plugged in after the dialog was opened.
```


Each row tooltip shows the current `/dev/ttyACMn` (or `COMn`) the resolver will open.

| Display | Means |
|---|---|
| `315535563234` | Microcontroller with that USB serial is online, ready |
| `315535563234  (not connected)` | Project remembers this MCU but it's unplugged right now |
| `--- Select MCU ---` | No MCU picked yet for this box |

Click the **Refresh** button in the dialog footer after plugging in or removing microcontrollers.

## How the resolver works

`source/communication/mcu_ports.py`:

```text
list_mcu_serials() -> [(serial, device)..]   # filtered by VID/PID 0xF055/9800
resolve(mcu_serial) -> device path | None       # serial -> live /dev/ttyACMn
serial_for_device(dev) -> serial | None         # reverse, for legacy upgrade
```

Cross-platform via pyserial. No OS-specific code in the GUI.

## Auto-upgrade for legacy projects

Old projects from before the serial fix have `mcu_serial=""` and `com_port="/dev/ttyACM0"`.
On load, `_apply_per_box`:

1. If `box.mcu_serial` is set → use it.
2. Else if `box.com_port` is set AND a live microcontroller is at that path → capture its
   serial, set `bw._mcu_serial`, set `box.mcu_serial`, and the next autosave
   persists the upgrade.
3. Else → empty; operator picks from the dialog.

After one successful connect, the project is permanently port-shuffle-immune.

## Edge cases

- **MCU not plugged in**: dialog row shows `(not connected)`; connect attempt fails
  cleanly.
- **Firmware without USB serial**: `pyserial`'s `serial_number` is `None`; MCU
  excluded from the dropdown. Fix: re-flash MicroPython with the standard build.
- **Two microcontrollers with the same serial**: USB spec says serials are unique, if
  duplicated, dropdown shows both rows; operator picks.


```{figure} /_static/media/gui/board-upload-task.png
:alt: The Upload Task dialog with a row per box, a Parallel option and the upload button
:figclass: pbl-side

Upload sends the task and the hardware definition together. That pair is what the session header hashes, so a session can name the exact code it ran.
```

## Upload + run

After connect:

1. **Upload Task**, picks `.py` from `tasks/`; transfers to MCU + runs
   `sm.setup_state_machine(task_file)`. Hashes the file, snapshots to
   `<project>/source/<hex>.py`, writes change_log line.
2. **Upload HD**, same for hardware definition.
3. **Record**, starts framework (`fw.run()`), starts camera + recorder + writer.

State machine + framework live on the MCU. The host is a controller: it sends events
+ coords, receives state/event/print/variable messages.


```{figure} /_static/media/gui/board-disconnect.png
:alt: The Disconnect dialog listing the connected boxes
:figclass: pbl-side

Disconnect releases the serial port. A board still claimed by a window that closed badly is the usual reason the next launch cannot see it.
```

## Disconnect

- Closing the project disconnects every box.
- Per-box Disconnect button stops the framework cleanly first, then closes serial.
