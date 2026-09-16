# Testing

`source/tests/` holds unit + integration tests. Pytest is the runner.

## Run everything

There are **two** suites, and the first command does not collect the second:

```bash
python -m pytest source/tests -q            # the rig
python -m pytest tools/offline_analysis/tests -q   # the analyser
```

Current state: **1870 passed, 4 skipped** (rig) and **38 passed** (analyser).

Qt runs offscreen: `conftest.py` sets `QT_QPA_PLATFORM=offscreen`, so no
display is needed and nothing has to be ignored. One consequence is worth
knowing: **text metrics are not the ones on screen**. The offscreen platform
substitutes a font that measures about 1.7x wider, so a test asserting that a
label fits its widget passes or fails for the wrong reason. Those checks live
in `tools/check_cell_widths.py`, which refuses to run offscreen; the pytest
counterpart skips itself.

The MCU tree (`tasks/`, `devices/`, `hardware_definitions/`,
`source/pyControl/`) is excluded from collection because it imports `pyb` and
cannot run on the host. Note that this means **nothing parses it either**: a
task with a syntax error is discovered at upload time, in front of an animal.

## Layout

```
source/tests/
├── conftest.py
├── gui_tests/             ← widget-level + dialog flows (158 files)
└── integration_tests/     ← cross-module integration (6 files)

tools/offline_analysis/tests/   ← the analyser's own suite
```

Markers are declared in `pyproject.toml` but only three are used: `unit` (70),
`integration` (3), `communication` (34). `config` and `slow` match nothing, so
`-m "config"` runs zero tests and reports success.

## What's covered

| Surface | Test file(s) |
|---|---|
| Config schema round-trip | `test_dialog_config_persists_rig_fields.py`, `test_push_fields_roundtrip.py` |
| Per-subject variable specs | `test_subject_scoped_variables.py` |
| FrameLog v2 row format | `test_frame_log_v2.py` |
| Pose centroid body-part picker | `test_pose_centroid_body_part.py` |
| MCU push policy + zone_changed | `test_dlc_push.py` |
| Tracker field persistence | `test_dlc_path_roundtrip.py` |
| Autosave preserves cfg.tracking (SK fix) | `test_autosave_preserves_tracking.py` |
| Snapshot-based tracking recovery | `test_tracking_recovery_from_snapshot.py` |
| MCU stable-serial binding | `test_mcu_serial_binding.py` |
| MCU / video integration | `test_mcu_video_integration.py` |
| Per-subject variable apply pipeline | `test_persistent_variables_v0.py` |
| Auto-enable tracking gate | `test_auto_enable_tracking.py` |
| Other GUI flows | `test_*.py` in `gui_tests/` |

## Writing a test

Tests run headless under Qt's `offscreen` platform. Most tests don't need a
QApplication; the ones that do create one at module top:

```python
from PySide6 import QtWidgets
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
```

Use `MagicMock` from `unittest.mock` for the host shim (`MainWindowBase` is too
heavy to instantiate; create a host with just the fields your code under test
reads). Pin to a `spec_set` if you're using `_pose_already_loaded_for` or other
new helpers, see the existing tests.

## Guards

A guard is a test that protects a property rather than a behaviour: nothing
breaks the moment it is deleted, which is exactly why it needs naming here.

| Guard | Holds |
|---|---|
| `test_no_em_dashes.py` | No em dash in any text this project writes. Covers `.py`, `.pyx`, `.pxd`, `.md`, `.html`, `.json`, and more; generated Cython `.c` is out, `.pyx` is in and was the last hiding place |
| `test_vendor_matches_rig.py` | Every file in `tools/offline_analysis/vendor/` still matches the module it was copied from, comparing definitions with imports stripped |
| `test_no_live_rig_imports.py` | The analyser's perimeter: no `source.*` on its active path except the guarded tracker seam, never at module scope, and the rig never imports the analyser |
| `test_app_boots.py` | The analyser opens without the rig codebase |
| `test_pipeline_ownership.py` | No new mutations through the `video_manager` alias |
| `test_runtime_python_syntax.py` | The host tree parses under the interpreter the app actually launches on, which is older than the one pytest runs |
| `test_*_contract.py` | The call shapes between sinks, recorder and MCU push |

Two of these were deleted at some point and restored later, and in between
nothing failed: the properties still held, the alarms were simply off. If you
find a guard inconvenient, that is the moment it is doing its job. Change the
rule deliberately and say so in the test, or leave it alone.

## Regression tests are sacred

Every bug fix in the last audit pass shipped with a regression test. The
"SK project lost its config" bug has two regression tests:

- `test_autosave_preserves_tracking.py`, proves the fix prevents the wipe
- `test_tracking_recovery_from_snapshot.py`, proves the heal restores from snapshot

If you change either of those files' behaviour, walk the SK project
end-to-end before merging.
