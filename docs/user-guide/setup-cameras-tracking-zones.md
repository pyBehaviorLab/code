# Set up cameras, tracking and zones

Connect a camera, choose a tracking backend, draw the zones your task reacts to, and wire those zones
back to the microcontroller, end to end, using the real buttons in the app. This is the hands-on
companion to the reference pages [Cameras](cameras.md), [Tracking](tracking.md) and
[Task recipes](../tasks/recipes.md).

:::{note}
**Before you start.** Load or save a project first, backgrounds and zones are stored under it.
Everything below lives in the **Camera Control** group on the Main Control tab, which has three
buttons: **Camera Config**, **Tracking Config**, and **Test Tracking** (Tracking Config is disabled
until a camera is connected).
:::

## 1. Connect a camera

Open **Camera Config** to launch the *Connect Camera* dialog, three tabs, worked left to right.

```{figure} /_static/media/camera/camera-dialog-multicam.png
:alt: Connect Camera dialog, Setups & cameras, Camera options, Regions lens & connect

The three tabs of the Connect Camera dialog.
```

1. **Setups & cameras**, choose the arrangement: **One camera per box**, or **Shared camera
   (CCTV)** where one overhead camera is split into per-box regions. Assign each box (or the
   shared arrangement) its camera from the ID list; **Refresh camera IDs** rescans after a replug.
   Per row: **Detect** measures the camera's supported resolutions and true frame rate once
   (cached per machine), and the **Record** toggle selects video saving.
2. **Camera options**, pick the resolution / frame rate, keep the recommended **MJPEG** pixel
   format for USB cameras (uncompressed formats hit the USB bandwidth ceiling and silently cap
   FPS), and set per-camera **Mirror / Flip** and the **Output quality** of the recording.
3. **Regions, lens & connect**, click **Draw regions…** and drag one rectangle per box on the
   live frame; a region is required in both arrangements. Optionally run **Lens calibration**
   once per camera (checkerboard undistortion, stored against the camera's USB identity) and tick
   **Auto-connect cameras when this project loads**.
4. Click **Connect Cameras**. Each camera appears as a live tile on the Video Stream tab.

Regions are stored as **percentages**, so they survive a resolution change and reload with the
project. See [Cameras → ROI + geometry](cameras.md) for the underlying representation.

```{figure} /_static/media/gui/tracking-mode.png
:alt: The tracking settings tab with the Blob, Simple, DLC and SLEAP mode row
:width: 100%

Pick the mode first: it decides whether a model folder and a GPU are needed at all.
```

## 2. Configure tracking

Open **Tracking Config**. The dialog has three tabs, **Zone Editor**, **Tracking Settings** and
**Event & Trigger**. Start on **Tracking Settings**:

1. In the **Boxes** row, tick the boxes you want to track.
2. Leave **Smooth tracking (Optical Flow + Kalman)** on unless you have a reason not to.
3. Choose the **Tracking Mode**:

- Click the **DLC** or **SLEAP** chip, then **Browse** to the model folder, the body-part
  list loads automatically.
- Set the **Zone part** (the body part that drives zone logic), **Confidence** (0.5), **Resize**
  (0.8) and **Instances** (1).
- Click **Initialize**, the button is amber while the network loads and turns green when it is
  ready.

Finish with **Apply & Close**, the dialog is transactional: Apply commits tracker settings,
zones, coord mapping and triggers to the boxes and the project in one step, while **Cancel /
Discard / ✕** rolls *everything* back to the state at open (including zones drawn this
session). **Export Config** writes a reusable JSON snippet only, it does not save to the
project.

## 3. Draw zones

Switch to the **Zone Editor** tab. Pick the **Box** from the selector at the top.

1. Click a shape button, **Rectangle**, **Circle/Ellipse**, **Polygon** (prompts for 3–12 sides),
   **Manual Poly** (click vertices, right-click or Enter to finish), or **Line**, and give the zone
   a **name** when prompted.
2. **Name each zone exactly as your task expects**, e.g. `LeftArm`, `RightArm`, `Center` for a
   T-maze. The names are the contract between the editor and the task.
3. Refine: **Ctrl+drag** rotates about the centre (`[` / `]` nudge ±5°), the mouse wheel zooms,
   middle-drag pans, and **Attach edges** + the **magnet** weld neighbouring arms together.
   **Copy / Paste / Dup** and **Undo / Redo** are on the panel.
4. Click **Save Zones to Project**, a zones-only commit for every box, kept even if you later
   close the dialog with Discard (only tracker-settings changes stay pending for Apply &
   Close). **Export Box Zones** writes this box's zones to a snippet file for reuse on another
   rig; it does **not** save to the project. Zones are stored in normalized `[0, 1]`
   coordinates, so they don't move if the resolution changes, and they **nest**, a query
   returns the innermost zone first. While a box is **recording**, its zones are locked
   (editor and adjust buttons refuse) so the session's zone events keep one meaning.

## 4. Calibrate real-world scale

Still in the Zone Editor, set a **length** and **unit** (cm / mm / m), click **Draw Scale**, and drag
the ruler along a known real-world distance (hold **Shift** to constrain to horizontal/vertical).
Speed is then reported in **m/s** instead of px/s.

## 5. Wire zones to the task

Drawing a zone doesn't send anything on its own, you connect it to the firmware on the
**Tracking Settings** tab. There are two ways a zone reaches your task.

### Push the zone name into a coordinate

To make the current zone available as `c.loc_center`:

1. Under **Coord Mapping**, add a row mapping `c.loc_center` (or another name) to the body part /
   centroid.
2. In **Push to MCU**, tick **c.\* coordinates**.

Your task then reads it directly:

```python
if c.loc_center in right_zones:      # right_zones = ['RightArm']
    print('Sample arm reached')
    timed_goto_state('return', 0)
```

### Fire an event on entry or exit

To turn a zone crossing into a framework event:

1. Under **Event Triggers**, click **+ Add Trigger**.
2. Set **Condition** to `enter_zone` or `exit_zone`, choose the **Body Part**, select the **Zones**,
   and type an **Event** name. (Other conditions include `in_zone`, `not_in_zone`, `speed_gt`,
   `speed_lt`, `cross_line` and `rotation_gt`.)
3. In **Push to MCU**, tick **Zone-change events**.

Declare that event name in your task's `events` list and handle it like any hardware event.

:::{tip}
To re-check position on **every frame** rather than only on zone changes, tick **Per-frame pose
event** under Push to MCU and add `frame_event` to your task's `POLL` tuple. See
[Task recipes](#maze-arenas-pymaze-py) for a complete T-maze that uses this.
:::

## 6. Verify

Use **Test Tracking** (operant) or the live overlay and confirm the centroid dot sits in the right
zone and `c.loc_center` updates as the animal moves. Once it does, you're ready to
[record](recording.md).
