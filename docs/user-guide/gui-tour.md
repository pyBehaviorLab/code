# GUI tour

pyBehaviorLab is configured and operated through a common graphical user interface with two
launch modes that share the same project format, dialogs, board-management tools and
video/tracking pipeline. Starting `pyOperant.py` opens **operant mode**, compact rows in a
multi-tab window, one row per box. Starting `pyMaze.py` opens **maze mode**, one tab per maze
with a large live-camera view. The recommended workflow is the same in both modes:

1. Create or load an experiment and enter the cohort metadata
2. Connect and configure the pyControl boards
3. Configure the cameras
4. Configure tracking, zones and position triggers
5. Run and monitor the session

Steps 3 and 4 are **optional**. A classic operant chamber senses everything it needs through the
board itself, infrared beam breaks in the poke wall, lickometers, levers, so a rig with no camera
simply skips them: leave Camera Config and Tracking Config untouched and the pipeline never starts
a camera, a tracker or a video writer. The session then produces the MCU `.tsv` alone, and Live
Status, Session Plot and Statistics all work normally because they read the MCU stream rather than
the camera. Video tracking earns its place when the variable of interest is *where the animal is*
(maze arms, zone occupancy, speed) rather than *what it touched*. Adding a camera later is not a
migration: assign it, draw a region, and the same project starts recording video alongside the
existing MCU log.

Mode-specific differences are noted where relevant; each step below links to the page that
covers it in depth.

**pyOperant.** Four tabs across the top switch between **Main Control**, **Live Status**, a
colour-coded event stream from the running boxes, **Statistics**, live trial counters and
plots, and **Video Stream**, a grid with one camera tile per box. Each is shown in its own
section below.

In the top action bar, **BOX SETUP** adds or removes boxes, **EXPERIMENT** saves and loads the
project, and **CAMERA CONTROL** opens the camera configuration, tracking configuration and
tracking-test tools used in steps 3 and 4. Three buttons on the left edge fold out the
**Experimental Info** panel, the message **Logger** and the built-in **Documentation** panel.

Each operant box occupies one row: box label, subject ID, task selector, Upload / Start / Stop
controls, session timer, task-variable **Controls**, board connection state and camera / COM
indicators. The **SETUP CONTROL** row underneath provides master actions for the ticked boxes:
board connection and configuration, task upload, multi-start and multi-stop, disconnection,
session plots and analysis.

**pyMaze** shares the experiment, camera and setup-control workflow, but each arena is a
separate tab whose header shows the setup and assigned subject. Within a tab, a state banner
displays the current task state and state timer; the live camera view occupies the main panel
and shows the configured tracking zones; and a status column carries camera number, COM port,
frame rate, connection state and the **LIVE STATUS** log, where uploads, state changes and
events appear with timestamps.

**EXPERIMENT CONTROL** provides Pause, Next and Prev for protocol stages, a manual **Doors**
panel and task-variable **Controls**. The **ADJUST ZONE** toolbar lets a zone be nudged,
rotated or scaled live, except while that arena is **recording**: zone geometry is frozen
during a recording and the toolbar greys out, so the session's zone events keep one meaning
from start to stop. The subject field, task selector and Upload / Record / Stop controls keep
the same function as in operant mode. **Detach** opens the current arena in a separate window,
useful when several mazes are spread across monitors.

## Operant tabs [pyOperant.py]{.pbl-pill}

**Main Control**, add or remove boxes, assign a subject and task per box, and start / record /
reset / stop each box independently. The bottom bar fans master actions across every eligible
box while leaving idle boxes usable.

```{figure} /_static/media/gui/operant-main-control.png
:alt: Operant GUI, Main Control tab

Per-box subject/task/timer rows and the master setup bar.
```

**Live Status**, a live, colour-coded event stream for every running box.

```{figure} /_static/media/gui/operant-live-status.png
:alt: Operant GUI, Live Status tab

One firmware event panel per box.
```

**Statistics**, a summary table plus live visualisations (rolling accuracy and per-box gauges),
configured per task and exportable.

```{figure} /_static/media/gui/operant-statistics.png
:alt: Operant GUI, Statistics tab

Summary table, rolling accuracy plot and per-box gauges.
```

**Video Stream**, every box's camera in one grid; one physical camera can serve several boxes
via per-box ROI cropping.

```{figure} /_static/media/gui/operant-video-stream.png
:alt: Operant GUI, Video Stream tab

Live per-box camera tiles.
```

## Maze mode [pyMaze.py]{.pbl-pill}

Same pipeline and dialogs, with one camera (or a shared CCTV camera) per arena and
maze-specific controls: zone-driven triggers, motorised door control, and tracking coordinates
pushed live to the firmware.

```{figure} /_static/media/gui/maze-main-window.png
:alt: The pyMaze main window with two arenas
:width: 100%

pyMaze: one panel per arena, each with its own task selector, session timer,
camera view and MCU log.
```

```{figure} /_static/media/gui/run-session.png
:alt: Placeholder for a recording session in progress
:width: 100%

A session in progress: state banner, live view with zone overlays, and the event
log filling as the task runs.
```

## Step 1. Create the experiment and enter the cohort

Full detail: {doc}`projects`.

```{figure} /_static/media/gui/experiment-info-sidebar.png
:alt: The Experimental Info sidebar open over the main window
:width: 100%

The Experimental Info sidebar: experimenter, project, session and data folder.
```

```{figure} /_static/media/gui/edit-cohort.png
:alt: The Edit Cohort dialog
:width: 80%

Edit Cohort. Subject and SetupID are required and must stay populated.
```

1. Create one setup per physical maze or operant box with **Add Maze / Add Box**. Each setup
   appears as its own maze tab or operant-box row.
2. Open the **Experimental info** sidebar and enter the experimenter's name; the project name is
   set when the project is first saved. Both are written into the header of every data file so
   sessions remain attributable later.
3. Select the **data directory** (only if the default is not wanted), the root folder under
   which all project data is written. Sessions are stored below it by task and date, so
   different tasks and recording days never overwrite one another.
4. Keep **Track HD / task snapshots** enabled (recommended). Every task and hardware-definition
   file uploaded to a board is then stored with the project together with a content hash, and
   the same hash is written into the session data header, a direct record of the exact
   versions used for each session, even if the source files are edited later.
5. Enter the animals with **Load Metadata** (imports a cohort table from an Excel file) or
   **Edit Metadata** (opens the same table for editing in the GUI). The first two columns
   **Subject** and **SetupID**, are required: they assign each animal to a physical setup, so
   selecting a subject in the GUI associates the correct setup automatically. Additional
   columns (cage, ear number, group, run, sex, …) are free cohort descriptors and are carried
   into the session records.
6. Press **Save** in the EXPERIMENT group to write the project file; subsequent changes are
   autosaved. **Load** restores the saved configuration, setups, cameras, regions, zones and
   tracking settings, so the rig can be reopened on later days.

## Step 2. Connect and configure the pyControl boards

Full detail: {doc}`boards`.

Each setup is controlled by its own pyControl-compatible microcontroller board running
MicroPython; once uploaded, the behavioural task executes on the board in real time,
independently of the PC. Four dialogs in the SETUP CONTROL row manage the boards. Each lists
the available setups with a tick box, so an operation is applied only to the selected boards.

1. **Connect boards** assigns each setup its serial port. *Display MCUs as* can show native
   port names (e.g. `COM27`), hashed IDs or hardware serial numbers, helpful for telling
   identical boards apart. Tick the required setups (**ALL** selects every setup) and press
   Connect; the LIVE STATUS log reports the MicroPython and pyControl framework versions
   detected on each board.

```{figure} /_static/media/gui/board-connect.png
:alt: The Connect dialog, one COM port per box
:width: 60%
:figclass: pbl-side

Connect: one port per box, with a tick box so an operation touches only the
setups you select.
```

2. For a new or freshly flashed board, or to update the framework or hardware configuration,
   open **Config boards** (B). **Load Framework** installs the pyControl framework;
   **Load Hardware Definition** uploads the pin map describing that setup's peripherals (pokes,
   doors, LEDs, audio devices, …). The hardware definition stays on the board and normally only
   needs re-uploading when the rig wiring changes. With **Parallel (faster)** enabled, all
   ticked boards are programmed concurrently.

```{figure} /_static/media/gui/board-config.png
:alt: The Config boards dialog
:width: 70%
:figclass: pbl-side

Config boards: framework and hardware definition, flash-drive and DFU
maintenance, and parallel programming.
```

3. **Enable / Disable Flash Drive** and **DFU Mode** are maintenance functions. Keep the
   board's USB flash drive *disabled* during experiments, a mounted flash drive can interfere
   with serial communication, and use DFU mode only to update the MicroPython firmware itself.
4. **Upload Task**: *Select & Upload Task* opens a file picker and sends the chosen task to
   every ticked board, convenient when several setups run the same protocol. To give one setup
   a different task, use that setup's own task selector and Upload control instead.

```{figure} /_static/media/gui/board-upload-task.png
:alt: The Upload Task dialog
:width: 60%
:figclass: pbl-side

Upload Task: send one task to every ticked board.
```

5. **Disconnect** cleanly releases the selected boards, for example at the end.
   Individual boards can be disconnected while other setups stay available.

```{figure} /_static/media/gui/board-disconnect.png
:alt: The Disconnect dialog
:width: 60%
:figclass: pbl-side

Disconnect: release selected boards while the others keep running.
```

```{admonition} Per-box by design
All board operations are strictly per box: unticked boxes are never touched, so one setup can
be maintained, re-flashed or restarted while the other boxes keep running their sessions.
```

## Step 3. Configure the cameras

Full detail: {doc}`cameras` and {doc}`setup-cameras-tracking-zones`.

*Skip this step entirely on a camera-less rig, see the note above.*

Camera configuration is stored with the project and normally only needs doing at the start of
an experiment or when the camera arrangement changes. Open **Camera Config** from CAMERA
CONTROL, or from the Video Stream tab. The dialog has two tabs, worked through left to right;
afterwards the saved cameras can reconnect automatically whenever the project loads.

1. On **Setups & cameras**, choose the arrangement. *One camera per box* gives each setup its
   own device.

```{figure} /_static/media/camera/camera-dialog-multicam.png
:alt: The camera table in one-camera-per-box mode, two boxes with different cameras
:width: 100%

One camera per box: each row is an independent camera with its own settings.
```

   *Shared camera (CCTV)* divides one camera's image into a region per setup, so several boxes
   are acquired from the same stream.

```{figure} /_static/media/camera/camera-dialog-cctv.png
:alt: The camera table in shared-camera CCTV mode
:width: 100%

Shared camera: one device serves every box, each box being a region of its frame.
```

   Maze mode uses the same dialog, one arena per row.

```{figure} /_static/media/camera/camera-dialog-maze.png
:alt: The camera dialog in maze mode
:width: 100%

pyMaze: the same dialog, one row per arena.
```

2. Set each row left to right: **Backend → Format → Resolution → FPS**. Each choice decides
   what the next may contain. The backend is the operating system's path to the camera
   (DirectShow or Media Foundation on Windows; V4L2 or GStreamer on Linux), and the format
   decides which sizes and rates exist at all.
3. Press **Detect**. It measures the backend and format that row is set to, and no others: it
   reads the camera's mode descriptors, then opens each size and counts the frames that
   arrive. If the camera will not serve the chosen format, Detect says so and offers the
   alternative rather than switching silently.
4. Read the resolution cell as *accepted → delivered*. `640×480 (30)` means the two agree;
   `1920×1080 (60→30)` means the driver accepts 60 here and the camera was measured
   delivering 30. A gap is a link or lighting limit, not a wrong choice of rate.
5. On **Regions, lens & connect**, press **Draw regions** and draw one region per setup on the
   live image. A region is required in both arrangements: it defines the image used for
   display, recording and tracking. *Record selection* determines which setups write video. If
   needed, run **Lens calibration** with a printed checkerboard; the correction is saved with
   the camera configuration. Enable **Auto-connect cameras when this project loads** if the
   same cameras are normally used.

```{figure} /_static/media/camera/camera-dialog-regions.png
:alt: The regions, lens calibration and connect tab
:width: 100%

Regions, lens calibration and connect.
```

6. Press **Connect Cameras**. The live image appears in each setup, with the measured frame
   rate and camera status shown alongside the MCU state.

```{admonition} One stream, one truth
Recording and tracking use the same captured image stream and the same setup region, so the
stored video corresponds exactly to the image presented to the tracking pipeline. The complete
camera configuration is saved with the project.
```

:::{admonition} Why the backend is yours to choose
:class: tip

The same camera does not perform the same through two backends, and neither is better in
general. Measured on one rig at 640×480: one camera delivered 19.9 fps through DirectShow and
29.9 through Media Foundation, while the camera beside it managed 28.3 through DirectShow and
23.9 through Media Foundation. No per-platform default is right for both, so the backend is a
control in the table and Detect measures the one you pick.

A backend that cannot open the camera is reported when you press Detect, naming the camera and
offering the fallback, rather than being recorded as a camera with no modes.
:::

## Step 4. Configure tracking, zones and position triggers

Full detail: {doc}`tracking`.

*Skip this step entirely on a camera-less rig, a task without tracking reacts to board events and
its own timers, and declares no `c.*` coordinate slots.*

Tracking converts the live video into behavioural variables usable both for offline analysis
and for driving the task from position: the animal's position or pose is resolved relative to named
zones, optionally converted to metric coordinates, and selected information is transmitted to
the microcontroller as events or task variables. Open **Tracking Config** from CAMERA CONTROL.

1. In the **Zone Editor**, choose a shape. Rectangle, Ellipse, Polygon, Manual Poly or
   Line, draw it over the live image, and name it in the list on the right. Zones can be
   moved, rotated, resized, duplicated, aligned with *Attach edges*, and copied to other setups
   with *Copy zones to all boxes*.
2. Draw a **metric scale**: enter the real length and unit, press *Draw Scale* and drag the
   line over a feature of known size in the image. The scale converts image coordinates from
   pixels to physical units, so position, distance and speed are stored in metric units.

```{figure} /_static/media/gui/tracking-zone-editor.png
:alt: The Zone Editor tab of the Tracking Configuration dialog
:width: 100%

Zone Editor: draw and name zones over the live image, and set the metric scale.
```

3. On **Tracking Settings**, select the backend. **DLC** and **SLEAP** both use trained
   pose-estimation models and track individual body parts: select the model, choose the body
   parts, set the confidence threshold and pick the **zone part**, the body part used to
   determine zone occupancy, then press **Initialize** to load the tracker.
4. Optionally enable **Optical Flow + Kalman** smoothing to stabilise noisy detections, and
   **save annotated video** to record a copy with the tracking overlay burnt in.

```{figure} /_static/media/gui/tracking-settings.png
:alt: The Tracking Settings tab of the Tracking Configuration dialog
:width: 100%

Tracking Settings: backend, model, body parts and the zone part.
```

5. On **Event & Trigger**, define what the task can see. **Coord Mapping** rows map a
   selected body part to `c.*` variables the task can read (for example `c.loc_center` and
   `c.speed` derived from the Center body part). The **Push to MCU** options select whether
   zone-change events, `c.*` coordinates and/or an optional per-frame pose event are
   transmitted to the microcontroller.
6. Add behavioural rules in the **Event Triggers** table. Each rule combines a condition
   (`in_zone`, `not_in_zone`, `speed`, `distance`, `angle`, …), the relevant body part and
   zone(s), the MCU event name, an optional threshold and a minimum hold time. A rule fires
   when its condition *becomes* true, not continuously while it stays true. *VID* and *PLOT*
   choose whether the rule appears in the video overlay and live plots.

```{figure} /_static/media/gui/tracking-events.png
:alt: The Event and Trigger tab of the Tracking Configuration dialog
:width: 100%

Event & Trigger: coordinate mapping, what is pushed to the MCU, and the
trigger table.
```

7. The dialog is **transactional**. **Apply & Close** commits everything, tracker settings,
   zones, coord mapping and triggers, to the boxes and the project in one step. **Cancel**,
   **Discard** or closing with the window ✕ rolls *everything* back to the state at open,
   including zones drawn during the session. To keep zone work without applying tracker
   settings, press **Save Zones to Project** in the Zone Editor, a zones-only commit that
   survives a later Discard. The **Export** buttons write standalone JSON snippets for reuse
   on another rig; they do **not** save anything to the project.

```{admonition} Before the first real session
:class: warning

Event names generated by tracking triggers must match the event names expected by the uploaded
task, otherwise the board cannot respond to them. Conditions that require an orientation axis
need at least two keypoints, so a model tracking a single point cannot evaluate them. Use
**Test Tracking** from CAMERA CONTROL to verify the complete path from image acquisition and
tracking to zone detection and task events. Test Tracking only runs on boxes with an
**initialised model**; drawn zones alone do not make a box trackable.
```

## Step 5. Run a session

Full detail: {doc}`recording`.

1. In the setup tab or box row, select the task and press **Upload**. The LIVE STATUS log
   reports the upload, then every state change and event of the running task with timestamps.
2. Press **Record** to start the session: video recording, tracking and MCU data logging start
   together on a shared clock, so video frames and behavioural events align later without extra
   synchronisation hardware. The session timer runs while recording; with several setups,
   sessions start and stop independently per box.
3. Control the running protocol with **EXPERIMENT CONTROL**: *Pause* halts and resumes the
   task, *Next* / *Prev* switch the protocol stage, *Doors* opens the manual door panel (maze
   mode) and *Controls* gives direct read/write access to the running board's task variables.
4. If the camera or apparatus has shifted slightly, correct a zone live with the **ADJUST
   ZONE** toolbar: select the zone number, then nudge, rotate or scale it. The change takes
   effect immediately, without opening the tracking dialog, but **not while that box is
   recording**: zone geometry is frozen for the duration of a recording (the buttons grey
   out) so the recorded zone events keep one meaning.
5. Follow the behaviour live with **Session Plot**, the event raster of all running boxes in
   one scrollable view, and with **Statistics**, the live per-task trial counters (both in the
   setup-control row).
6. Press **Stop** to end the session. All files are finalised into the project's data
   directory, one folder per task and date: the MCU event log (`.tsv`, carrying the content
   hashes of the exact task and hardware definition in its header), the session video (`.mp4`),
   the per-frame tracking and synchronisation table (`_video_data.txt`) and the frame-drop
   accounting (`_drops.tsv`). Disconnect the boards when the day is done.

```{admonition} Master buttons and losslessness
Master buttons act only on eligible boxes: a box that is already running is locked out of
global start actions while idle boxes stay available.

Every captured frame is either written to the video or accounted for in
`_drops.tsv`. Tracking skips frames under load rather than falling behind, and
each skip is logged there too. Only the on-screen display drops frames without
counting them.
```

## Video walkthroughs

Four clips accompany the paper. Two of them are the operating walkthroughs for
this page; drop the files at the paths below and they render here.

| Clip | Shows | Asset path |
|---|---|---|
| Supplementary Video 1 | Rewarded T-maze during electrophysiological recording | not on this page |
| Supplementary Video 2 | CAD animation: T-, plus- and six-arm radial maze | not on this page |
| **Supplementary Video 3** | **Operating the pyOperant GUI** | `_static/media/intro/workflow-operant.mp4` |
| **Supplementary Video 4** | **Operating the pyMaze GUI** | `_static/media/intro/maze-overview.mp4` |

:::{note}
**Placeholder.** The two clips below are not yet in the repository. Each player
shows a still of the window it walks through and will not play; dropping an
`.mp4` at the path named in the table above is all that is needed, no edit to
this page.
:::

<video class="pbl-video" controls preload="metadata" poster="/_static/media/gui/maze-main-window.png">
  <source src="/_static/media/intro/maze-overview.mp4" type="video/mp4">
  Your browser does not support embedded video.
</video>
<p class="pbl-vidcap">Supplementary Video 4, operating the pyMaze GUI.</p>

<video class="pbl-video" controls preload="metadata" poster="/_static/media/gui/operant-main-control.png">
  <source src="/_static/media/intro/workflow-operant.mp4" type="video/mp4">
  Your browser does not support embedded video.
</video>
<p class="pbl-vidcap">Supplementary Video 3, operating the pyOperant GUI:
project → camera and tracking → run.</p>
