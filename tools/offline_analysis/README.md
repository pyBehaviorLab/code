# Offline Analyzer

Standalone Qt application for post-hoc analysis of saved pyBehaviorLab
sessions. Runs in its own process, does not share state with the rig
GUI, and never opens a serial port or camera. Read-only access to
session files; exports go to a folder you choose.

## Launching

**From a terminal at the project root:**
```
python -m tools.offline_analysis.app
```

**From inside the rig GUI:**
Master toolbar → **Analysis**, in both pyOperant and pyMaze.

Both launch this same `python -m tools.offline_analysis.app` as a
detached subprocess. The rig GUI does not block while the analyzer is
open; you can close either side independently.

## One view

A session picker sits above the window; below it is the **Analyze**
view, which is pyBehaveTrack's Analyze tab ported whole and 2D only.
It reads `*_video_data.txt` recordings (and bare videos with no data
file yet), and it is the whole application; there is no tab bar.

The MCU Analysis tab that used to sit beside it is gone: pyControl logs
are read on the rig, and a tab bar holding one tab is a control that
cannot be operated.

**3D is absent, not hidden.** pyBehaveTrack's second workspace,
`*_pose3d.txt` reconstructions, solid zone volumes, height and rearing,
the mvEKS smoother, DLC-3D export, is removed from the tab and from
the engine. This rig records one video per session and reconstructs
nothing, so a control with no data behind it is worse than no control.
`tests/test_app_boots.py::test_no_3d_anywhere_on_the_tab` reads widgets
rather than visibility, so a *hidden* 3D control fails it too.

## Layout

```
tools/offline_analysis/
├── app.py                  # python -m entry point
├── main_window.py          # QMainWindow: picker + the Analyze view
├── session_selector_panel.py / session_picker_dialog.py / session_catalog.py
├── sidebar_log.py          # slide-out Log sidebar
├── theming.py              # window chrome
│
├── analyze/                # the ported Analyze tab
│   ├── analyze_tab.py      #   the tab itself
│   ├── native_zone_editor.py / zone_workbench.py / zone_renderer.py
│   ├── pose_engine_panel.py / offline_tracker.py / verify_view.py
│   └── analysis_theme.py / flow_layout.py / rotated_button.py / …
│
├── engine/                 # the analysis, with no Qt in it
│   ├── session_bundle.py   #   per-recording derived state + Bundle.plan()
│   ├── pipeline.py         #   stage hashes; run_all
│   ├── clock.py / space.py / measures.py / offline_analysis.py
│   ├── retrack2d.py        #   offline re-inference
│   ├── report_html.py / validate.py
│   └── trackers.py         #   THE ONLY file allowed to name source.*
│
├── vendor/                 # copies of rig modules, so the rig is optional
│   ├── theme.py / style_builders.py + icons/
│
├── video_data_schema.py    # the recording format, pure stdlib
├── video_data_parser.py / analysis_input.py / legacy_v0.py
├── zone_stats_full.py      # per-zone metrics used by the CLI
├── analyze_cli.py          # headless entry point
│
├── core.py / template_engine.py / analysis_hook.py
└── scripts/                # pluggable pyControl analyses loaded by name
```

`engine/` carries no Qt and `analyze/` carries no analysis: the tab
asks the engine what a recording needs and what running would do, and
the engine never asks the tab anything.

## Perimeter, what the analyzer must NOT import

By design this folder is **independent of the rig**: delete
`tools/offline_analysis` and the rig is untouched; copy it elsewhere and
it still runs. The one seam is `engine/trackers.py`, which reaches for
the live detectors lazily so an offline re-track uses exactly the
detector the rig used, and reports them as unavailable when the rig is
not there.

```
grep -rn "from source\.\|import source\." tools/offline_analysis/
```

Anything outside `engine/trackers.py` and docstring mentions is a
regression. `tests/test_no_live_rig_imports.py` enforces it, and
`tests/test_vendor_matches_rig.py` fails when a vendored copy drifts
from the module it came from while both live in this repository.

Not allowed:
- `source.communication.*`: pycboard / framework / live MCU
- `source.video.*`: live camera pipeline (except through `trackers.py`)
- `source.gui.*`: rig main window or widgets
- `pyControl.*`: MicroPython framework

Allowed:
- Stdlib + `PySide6`, `pandas`, `numpy`, `matplotlib`, `pyqtgraph`,
  `shapely` (optional), `openpyxl`, `cv2` (optional)
- Sibling files inside `tools/offline_analysis/`

## Tests

```
python -m pytest tools/offline_analysis/tests -q
```

## Troubleshooting

- **"No module named tools.offline_analysis"**: you're not running from
  the project root. Either `cd` to it, or use the `.bat` launcher.
- **Qt offscreen / no display**: for CI / headless testing, set
  `QT_QPA_PLATFORM=offscreen` before launching.
- **Matplotlib font warnings on first plot**: harmless, cached
  on second run.
- **"no tracker available"** on a re-track, the detector packages
  (`dlclive`, `sleap-nn`) are not installed in this interpreter;
  `engine/trackers.py::why_unavailable()` says which.
