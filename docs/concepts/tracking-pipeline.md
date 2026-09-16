# Tracking pipeline

End-to-end: dialog edit → live TC → pipeline subsystems → MCU push + per-frame log.

## Layers

```
┌─ Dialog (UnifiedTrackingDialog) ───────────────────────────────────┐
│   TrackingSettingsPanel: mode, model path, body parts, confidence, │
│   resize, instances, push gates, single body-part picker, init btn │
└────────────────────────┬────────────────────────────────────────────┘
                         │ Apply
                         ▼
┌─ MainWindowBase._apply_dialog_config ──────────────────────────────┐
│   for each box: pipeline.update_tracking_config(bid, **fields)     │
│   then host._dialog_overrides for coord_mapping, triggers, etc.    │
│   then _project_changed → debounced autosave                       │
└────────────────────────┬────────────────────────────────────────────┘
                         ▼
┌─ framebus.TrackingConfig per box (source/video/framebus/types.py) ─┐
│   tracker_type, dlc_model_path, keypoint_names, confidence,        │
│   pose_resize_factor, pose_n_instances, zones,                      │
│   push_zones_to_mcu, push_coords_to_mcu, zone_change_body_part,    │
│   user_applied, online_tracking_enabled, …                         │
└──────────┬────────────────────────┬─────────────────────────────────┘
           │ apply_tracking_config  │
           ▼                        ▼
   PoseSink              MCUPusher (TrackingPushPolicy)
   .set_centroid_body_part(bid, name)   .zone_change_body_part = name
   .set_rotation(bid, …)                .coord_mapping = {…}
   .set_n_instances(n)                   .push_zone_changed = bool
   .enable_for_box(bid)                  .configure_zones(zones)
   .configure_pose_model(…)
```

## Save side (autosave)

```
read_ui_into_config(host)
    seed cfg.tracking from host._active_config.tracking  ← prevents wipe (SK fix)
    _read_tracking(host, cfg)
        for each live TC with user_applied=True:
            lift rig-level fields → cfg.tracking
            mirror per-box state → cfg.setup_config.boxes[bid]
```

## On load

```
apply_config_to_ui(cfg, host)
    _recover_tracking_from_snapshots(cfg, host)   ← one-time heal
    _apply_per_box(cfg, host)
        for each box: widget._mcu_serial = box.mcu_serial (or upgrade from com_port)
    _apply_tracking(cfg, host)
        pipe._tracking_configs.clear()
        for each box: install TC from cfg.tracking + per-box fields
        pose.set_centroid_body_part(bid, cfg.tracking.zone_change_body_part)
        push.configure_tracking(bid, derived_coord_mapping, triggers, push gates,
                                zone_change_body_part)
```

## Per-frame push to MCU

`MCUPusher.TrackingPushPolicy.push(pycboard, zones_by_body_part, speed, body_part_coords)`:

1. **Intrinsic zone_changed event**, diff `zones_by_body_part[zone_change_body_part]`
   vs previous tick's set. On change, fire via
   `pycboard.queue_trigger_intrinsic_event("zone_changed")` (silent, does NOT appear
   in MCU TSV, like entry/exit).
2. **Per-zone triggers** (event_enter / event_enter_exit), edge-detect per condition;
   fire via `pycboard.queue_trigger_event(event_name)` (TSV-logged).
3. **c.* coord mapping**, resolve each `coord_name` → zone name or X/Y; if cache
   diff, push via `pycboard.queue_set_coordinates(name, value)` (silent, sets
   `ut.c.<name>` on MCU, no TSV entry).

## The single body-part picker

One combo in the DLC params row of the dialog (`zone_body_part_combo`). Always shows
`["centroid", *body_parts]`. Drives **both**:

- PoseSink centroid (per-frame `cx, cy` + `zones_by_body_part["centroid"]` + `location`)
- MCUPusher `zone_changed` event, diff target

"centroid" (default) keeps the legacy first-confident-keypoint heuristic. Any specific
keypoint name pins to that keypoint only, if it's below confidence on a given frame,
`cx,cy = None` and `location = "na"` rather than silently picking a different keypoint.

See `source/video/framebus/pose_sink.py:set_centroid_body_part` + `_emit_result`
centroid block.
