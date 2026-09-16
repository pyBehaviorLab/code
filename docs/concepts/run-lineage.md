# Run lineage, joining runs to the config that produced them

Each run-relevant artifact gets a content hash (djb2, 8 hex). The hash appears in
**every file the artifact touches**, so you can reconstruct exactly what code +
config produced a given data file. This hash-based linkage is the run's
**lineage**.

## Hashes used

| Artifact | Hash field | Appears in |
|---|---|---|
| Task `.py` | `task_hash` | MCU TSV header, video txt header, `source/<hex>.py`, change_log |
| Hardware def `.py` | `hd_hash` | MCU TSV header, video txt header, `source/<hex>.py`, change_log |
| DLC model | `model_djb2` | `cfg.tracking.dlc.model_djb2`, `source/dlc/<hex>.json` manifest |
| experiment_config | `config_djb2` | `source/configs/<hex>.json`, change_log |

## Cross-reference example

`2026-05-30/mcu/A31-Box1-2026-05-30-175911.tsv` header:
```
0.000  info  task_file_hash    af425e28
0.000  info  hardware_def_hash 00000000
0.000  info  subject_id        A31
0.000  info  start_time        2026-05-30 17:59:11
```

`2026-05-30/video/A31-Box1-2026-05-30-175911_video_data.txt` header:
```
#pycontrol  {"task":"ReversalLearning/ReversalLearning","subject":"A31","task_hash":"af425e28","hd_hash":"00000000"}
```

Same hashes. Look up `<project>/source/af425e28.py` to get the exact task code that
produced both files.

The `runs/2026-05-30.json` row does not repeat those hashes, it points at the pinned config
snapshot instead, and names the two artefacts:
```text
{"id": "2026-05-30-175911_box1", "subject_id": "A31", "box_number": 1,
 "config_djb2": "6ee98184",
 "mcu_tsv": "data/…/mcu/A31-Box1-2026-05-30-175911.tsv",
 "video_mp4": "data/…/video/A31-Box1-2026-05-30-175911.mp4"}
```

So the join runs the other way: the run row gives you the TSV, and the TSV header gives you
`task_file_hash` / `hardware_def_hash` / `devices`.

## Why djb2 (not SHA256)

DJB2 is fast, fits in 8 hex chars, collision-resistant enough at the scale of
"a few thousand task variants over a lab's lifetime." SHA256 was considered and
explicitly removed, see `feedback_djb2_only` in the codebase memory.

`source/config/hashing.py` is the single source of truth for the hash function.
