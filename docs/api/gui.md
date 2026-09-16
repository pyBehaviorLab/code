# `source/gui/`. Qt frontend

## Top-level windows

- `base.py` - `MainWindowBase` (shared between operant + maze)
- `operant.py` - `MainWindow` for operant grids
- `maze.py` - `MainWindow` for maze setup widgets

Both override the same hooks: `_load_create_widgets`, `_load_clear_existing_state`,
`_load_post_restore`, `apply_mode_extras`, `read_mode_extras`.

## Per-box widgets

- `widgets/box_control.py` - `BoxControlWidget` (operant)
- `widgets/setup_widget.py` - `SetupWidget` (maze)

Both implement the same contract for the bridge:
- `box_number` (int)
- `pycboard` (or `mcu_*` methods if no pycboard yet)
- `connect_mcu(mcu_id)`, picks serial → resolves device → opens
- `disconnect_mcu()`
- `is_connected` property
- `_mcu_serial`, `com_text` getter
- `subject_id_edit` (or equivalent), `task_combo`
- `tracking_enabled`, `save_tracking`, `save_video_enabled`

Plus `RunTask` mixin owns the framework start/stop + record + variable apply pipeline.

## Tracking dialog

`widgets/tracking_panel.py` - `UnifiedTrackingDialog` + `TrackingSettingsPanel`.

Init-button state machine: `_mark_dlc_clean(status)`, `_mark_dlc_dirty(reason)`,
`_mark_dlc_failed(message)` + `set_dlc_init_result(success, message)` entry point.

`_loading` flag short-circuits the mode-change → load-from-model cascade during
`_apply_tracking_settings` (saved selection wins over auto-detected body parts).

## MCU dialog

`dialogs/mcu.py` - `UniversalConnectDialog`. Lists microcontrollers by USB serial number via
`port_resolver.list_mcu_serials()`. Tooltip per row shows current device path.
Refresh button in the footer re-scans live USB.

## Pose subsystem mixin

`pose_subsystem.py`. Helpers shared by RunTask:
- `pose_signature_for(cfg) → Optional[Tuple]`, canonical signature for a config
- `init_dlc_for_launch(cfg)`, pre-warm pose on first Record
- Lazy init logic at Record click time (warn-and-prompt modal)

## Plotting

`plotting.py` - `PlotWindow` + `TaskPlot`. One singleton timer @ 20 ms (50 Hz)
walks per-box tabs and updates plots that aren't paused. Per-box pause checked
inside the callback (skipping paused boxes is free).

## Stats canvas

`stats/canvas.py` - `StatsCanvas` (per-task tab). 1 Hz update timer.

## Configuration manager

`config_manager.py` - `get_setting(category, key)`, `set_setting`. Reads/writes
`config/settings.json` (app-wide, not per-project).

## Project workflow

`project_workflow.py`, load/save/save-as orchestration. Handles the `ProjectFileGuard`
OCC + auto-merge save path, schema-version and mode gating on load, `runs/` bootstrap,
`data/<project>/` pre-creation, cohort-metadata auto-reload, and the snapshot store sweep.
