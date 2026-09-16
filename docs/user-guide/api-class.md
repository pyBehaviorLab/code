# API class (per-task adaptive hooks)

A task can ship an optional **API class** that runs on the host alongside the MCU's
state machine. Useful for adaptive logic that's awkward inside MicroPython
(matrix math, scipy, calling external services), or to change the task while it runs.

## Where it lives

- Per-task: `api_classes/<TaskName>.py` (e.g. `api_classes/ReversalLearning.py`)
- Per box: assigned via the box widget - `BoxConfig.api_class: FileRef`

Each box can use a different api_class (or none).

## Lifecycle

`source/communication/api.py` hosts the loader + run loop:

1. On Upload (per box), the api_class file is imported (relative to `api_classes/`).
2. The class is instantiated with the live `Pycboard`.
3. On every MCU data batch (`pycboard.process_data` tick), the api_class's
   `update(new_data)` is called.

## Contract

```python
class MyTaskApi:
    def __init__(self, board):
        self.board = board
        # snapshot starting state, set up classifiers, etc.

    def update(self, new_data):
        # called on every drain. new_data is a list of Datatuples.
        # use self.board.trigger_event(..) / set_variable(..) to act on it.
```

See `api_classes/Example_user_class.py` for the minimal working template.

## Threading

`update()` runs on the GUI thread (inside the pycboard `process_data` drain). Don't
block, push heavy work to a thread + queue back via `pycboard.queue_trigger_event` /
`queue_set_coordinates` (thread-safe).

## Persistence

The API class file is snapshotted (djb2 hashed) like the task script. Path lives in
`BoxConfig.api_class.path`. Optional, empty FileRef means "no api class for this box".

## Restored as of May 2026

The api_class support was disabled for a period; restored on 2026-05-11 with the
RunTask integration (see `project_api_class_restored` in the codebase memory).
Complements the stats canvas but runs at MCU-message granularity rather than 1 Hz.
