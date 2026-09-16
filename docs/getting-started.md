# Getting started

This page goes from an installed application to a first recording on one box.
It takes about fifteen minutes with the hardware already wired.

At the end there is a project folder holding the task, the configuration it ran
under, and the data from one session.

If the application is not installed yet, start at
[Installation](installation.md).

## What is needed first

A plain operant chamber needs a microcontroller wired to the box, connected over USB,
and a pyControl task file. Nothing else: no camera, no GPU.

A camera is needed only to record video or to track the animal. Live tracking
also needs a CUDA GPU and a trained DeepLabCut or SLEAP model. Those steps are
marked below and can be skipped.

## Launch

```bash
python pyOperant.py     # operant chambers
python pyMaze.py        # maze arenas
```

The two launchers open different main windows over the same engine. A project
saved by one opens in the other when its `mode` matches.
```{figure} /_static/media/gui/operant-main-control.png
:alt: The pyOperant main window with its four tabs, three control groups across the top, two box rows and the setup-control strip at the bottom

**Main Control** with two boxes added. Three groups across the top: **Box Setup**, **Experiment**, **Camera Control**. One row per box under them, and **Setup Control** along the bottom. The large empty area is the box list, which fills as sessions run.
```



## A first project

1. Press **Save** and type a name. There is no "New Project" button: the
   application starts as a draft, and the first Save is what names it and
   writes the folder. See [Projects](user-guide/projects.md) for the template
   route.
2. Press **Add Box** once per physical setup. Each box gets its own row, and
   from here on every setting belongs to a box rather than to the application.

```{figure} /_static/media/gui/group-box-setup.png
:alt: The Box Setup group, with an Add Box button and a Remove Box button

The **Box Setup** group, top left of Main Control. **Remove Box** takes away
the last row; a box that is running has to be stopped first.
```

```{figure} /_static/media/gui/box-rows.png
:alt: One box row wrapped over three lines: ID field, task selector, Upload, Start, Stop, elapsed time, Controls, connection state, Cam and COM

One box row, wrapped into three lines to fit the page. Everything on it belongs to that box alone: its ID, its task, its own Upload, Start and Stop, its elapsed time, its connection state, and its own camera and port.
```

3. Press **Connect boards** and pick the USB serial number of the microcontroller wired
   to that box. Serial numbers are used rather than COM ports because a port
   changes when the cable moves, and a serial number does not. See
   [MCU boards](user-guide/boards.md).

```{figure} /_static/media/gui/board-connect.png
:alt: The Connect dialog: a Display MCUs as selector, then one row per box with a port dropdown and a tick, a Connect button and a refresh button
:figclass: pbl-side

One row per box. **Display MCUs as** switches the list between the USB serial number and the COM port the board currently answers on. The serial is the one that survives a replug, which is why it is what gets stored. The circular button rescans, for a board plugged in after the dialog was opened.
```



4. Press **Upload Task** and choose the task file. It is compiled and copied to
   the board, and a copy is kept in the project so the session can be traced
   back to the exact code that ran. See [Writing tasks](tasks/writing-tasks.md).

```{figure} /_static/media/gui/board-upload-task.png
:alt: The Upload Task dialog: one row per box with its port and a tick, a Parallel checkbox and a Select and Upload Task button
:figclass: pbl-side

Tick the boxes to upload to, then pick the file once. **Parallel (faster)** uploads to all of them at the same time; untick it if a board is answering slowly and you want a failure to name one box rather than the set.
```



5. Press **Save**. Autosave has been running since step 1; this writes the
   snapshot the session will be recorded against.
6. Press **Record** on the box row. The task starts and the data file opens.

A plain operant session is complete at this point. The two sections below add
video and tracking.

## Adding a camera

Recording video needs a camera and a region of the frame that belongs to the
box. One camera can cover several boxes, with a region drawn for each.




```{figure} /_static/media/gui/group-camera-control.png
:alt: The Camera Control group with three buttons: Camera Config, Tracking Config and Test Tracking

The **Camera Control** group, top right of Main Control. **Tracking Config** and **Test Tracking** stay greyed until a camera is connected, because there is nothing for them to act on until then.
```

1. Press **Camera Config** and pick a camera for the box. Cameras are
   identified by a hash of the USB device, not by index, so a camera keeps its
   identity when another one is plugged in.
2. Choose a resolution. The rate each resolution reaches is measured and shown
   beside it, so a resolution that cannot hold the required rate is visible
   before the session rather than in the drop log after it.
3. Draw the region for each box.

```{figure} /_static/media/camera/regions-panel.png
:alt: The Regions, lens and connect tab of the camera dialog, with a chip per box showing its current region, a Draw regions button, lens calibration below, and record selection on the right

The **Regions** tab. The green chips are the current answer for each box, here
**whole frame** for both; **Draw regions** is how one camera comes to carry
several boxes. The tab header says the same thing in words, so an unfinished
step is visible without opening the panel.
```

Video is written alongside the task data, one file per box. See
[Recording](user-guide/recording.md).

## Adding tracking

Live tracking needs a CUDA GPU and a trained model. The animal's position is
found in every frame and sent to the task while the session runs, which is what
allows a trial to depend on where the animal is.

1. Press **Tracking Config** and pick DeepLabCut or SLEAP, then the model
   folder. The body parts are read from the model.

```{figure} /_static/media/gui/tracking-mode.png
:alt: The Tracking Mode group: a row of Blob, Simple, DLC and SLEAP, with the blob threshold, area limits and blur settings below

The **Tracking Mode** row decides what finds the animal, and with it what the
rig needs: **Blob** subtracts a background image and runs on any machine, with
the threshold and area limits shown here; **DLC** and **SLEAP** need a trained
model and a CUDA GPU.
```



2. Draw the zones the task refers to. A zone name is what the task receives, so
   it has to match the string the task compares against.

```{figure} /_static/media/gui/zone-editor-frames.png
:alt: The Zone Editor: the box's camera view beside a Zones list and the Add zone shape tools

The **Zone Editor**, working on the box's own view rather than a generic
frame, so a zone is placed against what the camera actually sees. Shapes come
from **Add zone** on the right, and the name given to each one is the string
the task compares against.
```



3. Press **Initialize**. The model loads and the box reports ready.

See [Tracking](user-guide/tracking.md), and
[Writing tasks](tasks/writing-tasks.md) for how position and zone names reach
the task.

## Where the data lands

```
<project>/
├── experiment_config.json    # the rig and per-box configuration
├── change_log.jsonl          # append-only log of uploads, saves and captures
├── background_images/        # per-box reference images
├── source/                   # snapshots: tasks, configs, model manifests
├── runs/YYYY-MM-DD.json      # one row per Record press
└── data/<task>/YYYY-MM-DD/
    ├── mcu/                  # per-box .tsv from the board, plus the session log
    └── video/                # per-box .mp4 and per-frame _video_data.txt
```

Everything under `source/` is content addressed, so a session can be traced
back to the task file and configuration it actually ran under, even after both
have been edited. See [Projects](user-guide/projects.md) for the whole layout
and [File formats](reference/file-formats.md) for the shape of each row.

## Next

- [GUI tour](user-guide/gui-tour.md), what every control does.
- [Writing tasks](tasks/writing-tasks.md), the state machine and its API.
- [Troubleshooting](troubleshooting.md), when a board or camera does not appear.
