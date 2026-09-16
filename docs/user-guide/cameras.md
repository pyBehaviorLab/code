# Cameras

A camera is reached in two steps, and both are yours to choose:

- the **family** that owns the device: **OpenCV** for any UVC camera (USB webcams,
  analogue feeds through a capture card, IP cameras), or a vendor SDK,
  **Spinnaker** (FLIR) and **Ximea**, for a scientific camera;
- for a UVC camera only, the **backend** the operating system opens it through:
  **DirectShow** or **Media Foundation** on Windows, **V4L2** or **GStreamer** on
  Linux and Jetson. A scientific camera has no such choice, its SDK is the path.

One `CameraThread` per physical camera. The pipeline pulls frames and dispatches
per-box (see [frame-pipeline](../concepts/frame-pipeline.md)).

Camera choice is an **experimental parameter**, not an implementation detail. Where
the timing of individual frames matters, use a camera delivering a known, stable
frame interval. Where many setups must be recorded at once, analogue cameras
combined through a multiplexer cost far less per chamber, and their longer, more
variable acquisition delay is acceptable when pose is computed afterwards. Both
are inputs to the same frame bus, see
[Running many chambers at once](../hardware/operant-chamber.md#running-many-chambers-at-once).

## Four settings, in the order they depend on each other

```{figure} /_static/media/camera/camera-dialog-multicam.png
:alt: The per-box camera table, one row per box, with backend, format, resolution and rate
:width: 100%

One row per box. Each column is decided by the one before it.
```

**Backend → Format → Resolution → FPS.** The order is not cosmetic: each choice
determines what the next one may contain.

- **Backend** decides everything below it. The same camera does not perform the
  same through two backends, and neither is better in general. Measured on one
  rig at 640×480: one camera delivered 19.9 fps through DirectShow and 29.9
  through Media Foundation, while the camera beside it managed 28.3 through
  DirectShow and 23.9 through Media Foundation.
- **Format** decides which sizes and rates exist at all. A camera that offers
  640×480 at 120 fps compressed may offer the same size at 30 fps uncompressed,
  and nothing else. If you need a particular frame rate, this is usually the
  control that gets it for you.
- **Resolution** then lists the sizes that backend and format support.
- **FPS** lists the frame intervals the driver will accept at that size.

## What the camera can actually do

Capabilities are **asked for and then measured**, never assumed.

**Detect** runs for the backend and format the row is set to, and for no others.
It reads the camera's own mode descriptors, then opens each size and counts the
frames that arrive. Both halves are necessary: a descriptor says what the device
advertises, and only counting says what it delivers.

If the camera will not serve the format you picked, Detect says so, names the
camera and what it returned instead, and offers the alternative. It never swaps
the format silently, because a rate measured in a format you did not choose
describes a camera you are not going to run.

### Reading the resolution cell

A size is labelled with the rate the driver accepts and, when the two differ, the
rate the camera was actually seen delivering:

| label | meaning |
|---|---|
| `640×480 (30)` | accepted and delivered agree |
| `1920×1080 (60→30)` | the driver accepts 60 here; the camera delivered 30 |

A gap between the two is a link or lighting limit, not a wrong choice of rate,
and it is worth knowing before a session rather than after.

### Why a mode sometimes offers a single rate

Cameras describe their frame intervals in one of two ways, and the difference
decides what you can pick:

- **A discrete interval.** The device reports one interval for that size and
  format and has no other. Its rate list holds exactly one entry, and asking for
  anything else makes the driver round to it. This is correct, not a broken
  control; the cell explains itself on hover and points at the modes where the
  rate you want does exist.
- **A range.** The device reports a fastest and a slowest interval. Which rates
  exist between them is **not** in the descriptor and can only be found by
  measurement, so a rate offered from inside a range is a candidate until the
  camera has been seen to hold it. A rate the camera turns out to round away is
  remembered and stops being offered for that camera, in that format, at that
  size.

:::{note}
A driver accepts any frame rate you set and reports it back unchanged, whether or
not it can hold it. `cap.get(CAP_PROP_FPS)` is therefore never evidence, and every
rate in this application comes from counting delivered frames.
:::

### What Detect stores, and what a second Detect replaces

Measurements are kept **per machine**, because the achievable rate depends on this
PC's USB controller and the port, and they are filed **per backend and per format**,
because both change the answer.

Pressing Detect again replaces what it measured and leaves the rest alone:

- the rates for that backend, in that format: **replaced**, including by a lower
  reading;
- rates learned to be unavailable: **cleared**;
- the other backend, and the other format: **untouched**, because this Detect
  made no statement about them.

## Connect

**Camera Config** opens the dialog above. It is reachable from the Camera Control
group on the Main Control tab, and from the Video Stream tab, where the same two
controls sit beside the detach button so a detached video window carries them.

```{figure} /_static/media/camera/video-stream-tab.png
:alt: The Video Stream tab with Camera Config and Test Tracking beside the detach button
:width: 100%

Camera Config and Test Tracking are duplicated into the Video Stream tab, and
follow the state of the originals.
```

Pick which camera goes to which box. Cameras can be shared between boxes when
several boxes are cropped from one frame (CCTV mode); regions, lens correction and
the connect step live on the second tab.

```{figure} /_static/media/camera/camera-dialog-regions.png
:alt: The regions, lens calibration and connect tab of the camera dialog
:width: 100%

Regions, lens calibration and connect.
```

For a step-by-step walkthrough see
[Set up cameras, tracking and zones](setup-cameras-tracking-zones.md).

## Camera config persistence

A camera is identified by what the operating system says it is, not by the index
it happens to open on. Plugging in a second camera can take index 0 and push the
first to 1, so an index recorded in a project silently opens different hardware.

- **UVC cameras** are identified by their USB device path, shown as `cam` followed
  by a short hash (`cam5b4028a9`). Because the path includes the port, moving a
  camera to a different socket gives it a new identity and its row needs
  re-picking. When the OS cannot describe the device, the identity falls back to a
  fingerprint of the camera's own capabilities, written `fp…`.
- **Scientific cameras** are identified by their serial or device id.

The chosen backend, format, resolution, rate, orientation and regions are saved
with the project, so a rig reopened on another day restores the same
configuration. A project written before the backend was a choice is given this
platform's default and records it the next time the project is saved.

Measurements are *not* saved with the project. They live in a per-machine store,
because the rate a camera can hold depends on the PC it is plugged into; carrying
them to another rig would state numbers nobody measured there.

## ROI + geometry

Two representations per box:

- `geometry`: pixel rect `{x, y, w, h}`, absolute on the camera frame
- `roi_normalized`: `[x, y, w, h]` in 0–1, survives resolution changes

The video segment processor uses these to split a shared camera frame into per-box
sub-frames. Single-cam-per-box rigs leave both empty / passthrough.

## Calibration scale (pixels-per-meter)

Per-box `scale` calibration drawn in the zone editor (a "scale zone" with known
real-world length). Persisted under `cfg.ui.dialog_overrides.scale[<box_id>]`.

Used by `FrameLog.resolve_px_per_m` to convert speed from px/s → m/s. When set, the
video txt header shows `#units {"speed":"m/s"}`; otherwise `px/s`.

## Recording the camera

When Record fires, RecorderSink:

1. Opens an mp4 writer (NVENC if available, else FFmpeg libx264, else OpenCV
   VideoWriter) at `<project>/data/<task>/<date>/video/`.
2. Opens a `_video_data.txt` v3 (one row per frame, 9 tab-separated columns) next to
   the mp4, see [File formats](../reference/file-formats.md#_video_datatxt-v2).
3. On every BoxFrame: appends one row.
4. On Stop: closes both files and writes a footer with total_frames /
   dropped_frames / pose_count.

Frame drops are tracked separately in `<video_dir>/_drops.tsv` (per-box ring buffer),
one row per drop, and totalled in the `_video_data.txt` footer as `dropped_frames=N`.

## Display

The live preview is **not** a sink. Two paths drive it, one active per box:

- **Polling paint** (`base._paint_streaming_cameras_once`, on the process tick),
  for boxes that are not tracking, the per-box widget shows the latest captured
  frame. A queued display sink was tried first but silently dropped frames when
  overlay drawing got slow, so polling the most recent frame replaced it.
- **Frame-matched overlay**, for boxes that are tracking, the pose result
  carries the source frame and its `cam_frame_id`; the GUI paints that frame with
  that overlay so keypoints stay locked to their own frame. The polling loop
  stands down for these boxes.

In both cases the overlay layer (zones, pose dots, ROI rect) is composited on top
by the per-box renderer.

### Aspect ratio

Tiles **letterbox** by default: the image scales with the window and keeps its
aspect ratio, with padding on the short axis rather than distortion. This holds in
both modes and at any window size, so enlarging the GUI does not stretch the
picture, and, more importantly, does not move an overlay off the animal.

Scaling uses the pixel-centre convention (`cv2.resize` semantics): source pixel
*i* maps to destination centre `i·s + (s−1)/2`. Overlay geometry is transformed
through the same mapping as the image, so keypoints and zone outlines stay locked
to the pixels they were computed from at every zoom level.

## Lens calibration

Where the field of view requires it, correct lens distortion from a printed
checkerboard; the correction is stored with the camera configuration and applied
to display, recording and tracking alike. A ready-to-print board is included at
`docs/calibration_board_9x6_25mm_A4.png`, print at 100 % scale on A4 and verify
the squares measure 25 mm before use.
