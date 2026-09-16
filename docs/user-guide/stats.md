# Statistics canvas

A configurable live dashboard that watches the MCU event/state/print/variable stream
and updates plots + counters in real time.

```{figure} /_static/media/gui/operant-statistics.png
:alt: The Statistics tab with a per-task panel of live trial counters
:width: 100%

The **Statistics** tab: live counters per task, updated as the session runs.
```

## Configuring

Each task has an adjacent JSON config file (e.g.
`tasks/ReversalLearning/config.json`). The path is stored in
`cfg.stats_config.template_file` so projects round-trip the choice.

The Statistics window opens via the toolbar; it loads the config at first frame +
re-reads on every Upload.

## Authoring

The config defines:
- Data sources (which MCU events/variables to track)
- Layout (rows × cols of cells)
- Per-cell widget type (counter, plot, table, gauge, …)
- Per-widget bindings (which source → which display)

See `source/stats/canvas.py` for the live widget set + binding grammar.

## Per-task tabs

Multi-task projects get one tab per task. The active tab follows the most recently
uploaded box's task; switching tabs reloads the matching config.

## Update cadence

`update_timer` @ 1 Hz per stats canvas. Gated by `framework_running` on the bound
boxes, idle boxes contribute zero work.

## Detachable

Toolbar button pops the stats window out of the main window into its own top-level,
useful for multi-monitor setups.

## See also

- [reference/file-formats](../reference/file-formats.md) for the stats config JSON shape
- `source/stats/format.py` for the formatter functions referenced from config
