# Snapshot store + auto-recovery

Every Save (and every task / HD upload) writes a content-addressable copy of the
artifact into `<project>/source/`. Two purposes: **lineage** (link a run back to
the exact config it used) and **recovery** (heal a damaged config from a past snapshot).

## What gets snapshotted

| Artifact | Where | Trigger |
|---|---|---|
| `experiment_config.json` | `source/configs/<config_djb2>.json` | every Save (explicit or autosave with state changes) |
| Task script `.py` | `source/<task_djb2>.py` | every successful Upload (tier-new) |
| DLC model manifest | `source/dlc/<model_djb2>.json` | first time a model is touched by the snapshot store |
| Hardware-def `.py` | `source/<hd_djb2>.py` | every successful HD upload |

The hash is djb2 (8 hex). All cross-file references use the same hash so the MCU TSV's
`task_file_hash` joins the video txt's `task_hash` joins `source/<hex>.py`.

## change_log.jsonl, append-only event log

Every upload, commit, save, capture writes one line. **`source/config/history.py`
is the single writer** - `SnapshotStore` routes its source-capture events through
`history.append_event`, and config edits go through `history.append_change`. Both
families share one file, one `ts` format (`%Y-%m-%d %H:%M:%S`), and are told apart
by their discriminator key (`kind` for captures, `action` for config edits). Used
by analysis tooling to reconstruct "what happened on day X". Example:

```text
{"ts":"2026-05-29 11:39:54","kind":"upload","source":"task","box_id":1,"label":"ReversalLearning.py","djb2":"21a41cdc","tier":"new","new":true}
{"ts":"2026-05-29 16:47:35","actor":"user","action":"save","source":"Save Config","path":"../experiment_config.json","old":null,"new":"b9c6ddef"}
{"ts":"2026-05-30 12:35:37","kind":"dlc_capture","djb2":"8998445c","new":true}
```

## Auto-recovery of wiped tracking config

Specific defense against the pre-fix autosave bug that wiped `cfg.tracking` to defaults.
On project load (`apply_config_to_ui`), `_recover_tracking_from_snapshots` runs:

1. If `cfg.tracking.enabled` is `True` or `mode` is non-empty → no-op (config healthy).
2. Else walk `<project>/source/configs/*.json` newest-first.
3. First snapshot whose `tracking.enabled` is `True` → adopt its `tracking` block.
4. Log a WARNING with the source snapshot name + restored model/body_parts.
5. Next autosave persists the recovered config (the seed-from-base fix in
   `read_ui_into_config` prevents the wipe from recurring).

The recovery itself is `_recover_tracking_from_snapshots` in
`source/config/experiment.py`. Step 5, the part that stops the wipe coming
back, is held by `source/tests/gui_tests/test_autosave_preserves_tracking.py`.

## When no snapshot can help

If every snapshot is also wiped (project never had a good save), recovery is a no-op
and the operator picks the model fresh in the dialog. Once they Apply, the fixed
autosave path persists correctly.
