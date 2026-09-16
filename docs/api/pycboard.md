# `source/communication/`. MCU communication

## `pycboard.py` - `Pycboard`

Host-side wrapper around `pyboard.Pyboard` (which wraps `serial.Serial`). One
instance per box.

### Construction

```python
pyc = Pycboard(serial_port=device_path..)
# serial_port can be a device path (/dev/ttyACM0), resolution from mcu_serial
# happens upstream in widget.connect_mcu
```

### Key methods

```python
pyc.setup_state_machine(task_path)     # transfer + initialise task
pyc.start_framework()                  # fw.run() on MCU
pyc.stop_framework()                   # send b'\x03'
pyc.process_data()                     # drain MCU → data_consumers
pyc.trigger_event(name, source="u")    # TSV-logged event
pyc.queue_trigger_event(name)          # thread-safe enqueue
pyc.trigger_intrinsic_event(name)      # SILENT dispatch (zone_changed, etc.)
pyc.queue_trigger_intrinsic_event(name)
pyc.set_coordinates(c_name, c_value)   # writes ut.c.<name> on MCU
pyc.queue_set_coordinates(c_name, value)
pyc.set_variable(v_name, v_value)
pyc.get_variable(v_name)
pyc.get_timestamp()                    # extrapolated MCU fw_ms (monotonic anchor)
```

### Threading

All serial I/O happens on the GUI thread via `process_data()` which is called from
the main process tick. Worker threads (pipeline / pose / tracker / sinks) push
requests via `queue_*` methods. `_drain_pending_writes` runs at the top of
`process_data` and converts queued requests into the same `trigger_event` /
`set_coordinates` calls Original pyControl uses.

### State

```python
pyc.framework_running  # True between start_framework + stop
pyc.sm_info            # State_machine_info: states, events, variables, hashes
pyc.timestamp          # last received MCU fw_ms
pyc._last_message_mono # host monotonic ns at that last message
pyc.data_logger        # writes the .tsv
pyc.data_consumers     # list of objects with .process_data(new_data)
```

## `pyboard.py` - `Pyboard`

Original-pyboard layer; raw REPL + serial I/O. Don't touch directly; go through
Pycboard.

## `port_resolver.py`. USB-serial-number wrapper

```text
list_mcu_serials() -> [(serial, device)..]   # filter by VID/PID (0xF055 / 0x9800-1)
resolve(mcu_serial) -> device_path | None       # serial -> live /dev/ttyACMn
serial_for_device(dev) -> serial | None         # reverse, for legacy upgrade
```

Cross-platform via `pyserial.tools.list_ports`. The GUI talks to MCUs by serial
number; this is the only module that knows about device paths.

## `message.py`. MCU message types

`MsgType` enum: `EVENT, STATE, PRINT, HARDW, VARBL, WARNG, ERROR, STOPF, ANLOG, THRSH`.
Each MCU message decoded by `pycboard.process_data` into a `Datatuple(time, type,
subtype, content)`.

## `data_logger.py`

Writes the `.tsv` per session. One row per MCU message. Format:
`time \t type \t subtype \t content`.

## `api.py`, per-task adaptive class hook

Loader + run loop for `api_classes/<TaskName>.py`. Instantiated at Upload, `update(new_data)`
called on every `process_data` drain.
