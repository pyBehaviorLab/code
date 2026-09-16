# Offline analysis

Analysis runs in its own application, not in the rig GUI. It never opens a serial
port or a camera, reads session files read-only, and writes only where you point
it, so a mistake in analysis cannot disturb a recording session, and a crash in
one cannot take down the other.

## Launch

From the rig GUI, press **Analysis** in the master toolbar; it starts the analyser
as a separate process and the rig GUI keeps running. Either window can be closed
independently.

From a terminal at the project root:

```bash
python -m tools.offline_analysis.app
```

There is also a command-line entry point for batch work without a window:

```bash
python -m tools.offline_analysis.analyze_cli --help
```

## The window

One view. A source bar along the top, the recordings you have chosen on the left,
and what to do with them on the right.

```{figure} /_static/media/analysis/analyzer-main.png
:alt: The offline analyser, source bar, recordings table and the plan panel
:width: 100%

The analyser: choose recordings on the left, choose what to do on the right, run
it, and read the results underneath.
```

**Choosing recordings.** **Files** picks individual recordings, **Folder** takes
everything beneath a directory, **Project** reads a project's own session index,
and **Videos** accepts bare video files that have no data file beside them yet.
Recordings can also be dragged onto the table. Each row can be ticked in or out
with **Use**, so a run covers exactly the sessions you meant.

**Choosing what to do.** Three plans, and the panel states in one line what each
will do before you run it:

- **Analyse** measures the poses each recording already has, and writes nothing
  except the results.
- **Correct** repairs an existing track.
- **Re-track** runs a pose model over the video again, for a session recorded
  without online tracking, or one you want re-tracked with a different model.

**Zones** are drawn on the video itself with *Draw / edit zones on video*, so a
session recorded without zones can still be scored by zone afterwards.

**Running.** **Dry run** reports what would happen without writing anything.
**Run plan** executes it, and **Save results to…** chooses where the output goes.
The panel lists the measure columns you will get before the run starts, so the
output shape is never a surprise.

**Reading the output.** **Results** is the table, **Plots** the figures, and
**Verify** shows the tracking over the video so a number can be checked against
the frame it came from.

```{figure} /_static/media/analysis/analyzer-results.png
:alt: Placeholder for the analyser results and plots view
:width: 100%

Results and plots after a run.
```

The right-hand rail holds **Zones**, **Measure**, **Correct**, **Objects** and
**Options**; the **Log** tab on the left edge slides out the run log.

## What it reads

- `*_video_data.txt`, one row per frame, and the mp4 beside it
- the MCU `.tsv` for the same session
- the project's session index, when a whole project is opened
- the task source captured with the run, so the code that produced a session is
  available with it

A recording with no data file is still loadable: that is what **Re-track** is for.

## Working with the data yourself

```python
from tools.offline_analysis.video_data_parser import parse_video_data

sess = parse_video_data("…_video_data.txt")
sess.frames_df       # one row per frame
sess.events          # events reconstructed from the folded columns
sess.zones           # zone definitions from the header
sess.info            # header: units, task hash, hardware-definition hash, …
```

Rows are one per frame, with event and state names folded into the frame interval
they landed in. The column-by-column breakdown is in
[recording](recording.md#the-_video_datatxt-row-format-v3).

:::{note}
The session header records whose clock the frame times came from and the rate that
was requested alongside the rate that was delivered, so a session analysed months
later can still be checked against the conditions it was recorded under.
:::
