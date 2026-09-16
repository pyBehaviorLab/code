# `source/config/experiment.py`, schema + load/save bridge

Single module owns the project schema, JSON round-trip, and the host-side bridge that
applies a loaded `Config` to widgets / reads widget state back into a `Config`.

## Public dataclasses

| Class | Purpose |
|---|---|
| `Config` | Top-level project root |
| `Meta` | Experimenter, project label, metadata sidebar |
| `SetupConfig` | Wrapper around `List[BoxConfig]` |
| `BoxConfig` | One box (mcu_serial, camera_id, geometry, ROI, zones, …) |
| `BoxGeometry` | Pixel rect `{x, y, w, h}` |
| `Zone` | Polygon / scale / arena entry |
| `CamerasConfig` | Rig-level camera defaults + registry |
| `CameraEntry` | One registered camera (unique_id + backend info) |
| `TrackingConfig` | Rig-level tracking (mode + dlc/sleap + push gates) |
| `DLCConfig`, `SleapConfig` | Backend-specific subtrees |
| `StatsConfig` | Stats template path |
| `UIState` | active_box, expanded panels, dialog overrides |
| `FileRef` | `{name, path, djb2, size_bytes}` for any tracked file |

## Top-level functions

```python
apply_config_to_ui(cfg, host)    # load: cfg → host widgets + pipeline
read_ui_into_config(host)        # save: host widgets + pipeline → fresh cfg
save_experiment(cfg, *, project_dir_path, cached_payload=None, guard=None)
save_template(cfg, project_dir, *, cached_payload=None)
serialize_for_save(cfg)          # canonical dict (sort + redact ephemerals)
```

### Save-side gotcha: seed-from-base

`read_ui_into_config` deep-copies `cfg.tracking` from `host._active_config` before
calling `_read_tracking`. This prevents a non-tracking autosave (camera config change,
box wiring change, bg capture, etc.) from wiping `cfg.tracking` when the live pipe
happens to be empty at that moment.

See `test_autosave_preserves_tracking.py` for the regression coverage.

### Load-side gotcha: auto-recovery

`apply_config_to_ui` calls `_recover_tracking_from_snapshots` BEFORE `_apply_tracking`.
If `cfg.tracking` is wiped but `<project>/source/configs/<hex>.json` has a populated
snapshot, the recovery restores it transparently.

See `test_tracking_recovery_from_snapshot.py`.

## Per-box bridge

`_apply_per_box(cfg, host)`:

- For each `box`: set widget `_mcu_serial`, com_port, camera_id, ROI, geometry, …
- Auto-upgrade legacy `com_port` → `mcu_serial` via `port_resolver.serial_for_device`

`_read_per_box(host, cfg)`:

- For each widget: read `_mcu_serial`, com_port, camera_id, …
- Falls back to `base.mcu_serial` when widget hasn't been connected yet (avoids
  resetting saved serial to empty just because the connect dialog isn't open).

## Tracking bridge

`_apply_tracking(cfg, host)`:

1. `pipe._tracking_configs.clear()`
2. For each box, build raw dict from `cfg.tracking` + per-box fields
3. `pipe.install_tracking_configs(payload)` → per-box TCs
4. `pose.set_centroid_body_part(bid, cfg.tracking.zone_change_body_part)`
5. `push.configure_tracking(bid, derived_coord_mapping, …)`

`_read_tracking(host, cfg)`:

- Walk `pipe.all_tracking_configs()`
- For the first TC with `user_applied=True`, lift rig-level fields onto `cfg.tracking`
- Mirror per-box state (`online_tracking_enabled` etc.) onto `cfg.setup_config.boxes[bid]`

## Hashing

`config_djb2` is a self-hash: `djb2_hex_from_text(canonical_json(payload))`.
Three top-level keys are dropped from the payload before hashing, not one:
`config_djb2` itself, `last_modified`, and `ui`. Hashing with only
`config_djb2` removed gives a different value and the config reads as
tampered. `canonical_json` and the exclusion list live in
`source/config/experiment.py`; `djb2_hex_from_text` is in
`source/config/hashing.py`.
