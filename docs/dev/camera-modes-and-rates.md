# Camera modes and rates

How a camera's resolution, pixel format and frame rate are really decided, what
each capture backend can and cannot tell us, and why the rig kept offering
rates its cameras do not have.

This document exists because a whole evening was spent chasing symptoms. Every
fix was correct and none of them addressed the cause, because the cause is
architectural: **OpenCV cannot enumerate a camera's modes, so the rig was
guessing them by trial, and the trial was unreliable.** Everything below is
measured on the DKLab rig on 10 September 2026 unless stated otherwise.

## 1. The ground truth is in the camera

A USB Video Class camera describes itself in its descriptors. For each pixel
format it lists frame sizes, and for each frame size it lists the exact frame
intervals it supports. This is not a range to be interpolated: it is a list,
and a camera that lists 30, 20 and 15 does not have 25.

The host cannot invent an interval that is not in that list. Ask for one and
the driver picks the nearest it does have, reports success, and reports the
requested value back when asked. That last part is what makes this hard to
see from the application.

Machine vision cameras (Spinnaker, XIMEA) work the opposite way. Their rate is
continuous and derived from exposure, region of interest and link bandwidth, so
the SDK can be asked "what rate is achievable right now" and will answer with a
number that changes when exposure changes. The two families need different
handling and the rig must not treat one as the other.

## 2. What each backend exposes

| Backend | Platform | Sizes and formats | Rates | From OpenCV? |
|---|---|---|---|---|
| DirectShow | Windows | Exact, `IAMStreamConfig::GetStreamCaps` | **Range only**, a min and a max interval per format and size | No |
| Media Foundation | Windows | Exact, `IMFMediaType` native types | Exact, `MF_MT_FRAME_RATE` per type | No |
| V4L2 | Linux | Exact, `VIDIOC_ENUM_FRAMESIZES` | **Exact list**, `VIDIOC_ENUM_FRAMEINTERVALS` | No |
| GStreamer | Linux, Jetson | Exact, caps negotiation | Exact, caps carry a framerate list | Not through `VideoCapture` |
| Spinnaker, XIMEA | any | Exact | Continuous, a live function of exposure and bandwidth | Not applicable, own SDK |

The rate column is the one that matters and the platforms differ. V4L2 gives
the camera's actual interval LIST. DirectShow gives only the bounds, so a
camera whose real list is 30, 20 and 15 is reported as "5 to 30" and the
missing 25 is invisible. Measured on this rig: FFmpeg lists 1920x1080 MJPEG as
`fps=15 max fps=30` for `cam3cbb52d3`, and asking that camera for 25 fps
produces 30.03.

So on Windows the enumeration settles sizes and formats and BOUNDS the rates,
and the exact list within those bounds still has to be confirmed by opening
the camera. On Linux it settles all three.

Every backend knows the answer. OpenCV's `VideoCapture` surfaces none of it.
Its entire interface for this is `set(CAP_PROP_*)` followed by `get`, and `get`
returns what was requested rather than what was negotiated on most UVC drivers.

So an OpenCV-only application has exactly one way to learn a camera's
capabilities: set a mode, grab frames, and count. That is what this rig does,
and it is why the numbers were wrong.

## 3. Three different quantities, routinely confused

Naming these separately is most of the fix.

**Offered.** What the camera lists in its descriptors: the sizes and the exact
rates it can be set to. A property of the hardware and the link. Does not
change with lighting.

**Requested.** What the operator picked. A number written into the project.

**Delivered.** What the camera actually produces once running. A function of
the requested mode AND the current conditions: exposure, illumination, USB
bandwidth shared with other cameras, host load.

A fourth follows from them:

**Recorded.** The rate written into the video file's header. If this does not
match delivered, the file plays back at the wrong speed and every time derived
from its nominal rate is wrong.

The rig had one number where it needed four. The picker showed a ladder built
from a measurement and called it capability; the tile showed the request and
labelled it "camera can deliver"; the recorder wrote the request into the file
regardless of what arrived.

## 4. What the rig measured, and why it was wrong

Three cameras, all UVC, on one Windows host.

### 4.1 Rates that do not exist were offered

`cam3cbb52d3` at 640x480, asking for each rung of the ladder:

```
asked 30 -> delivered 30.03   honoured
asked 25 -> delivered 30.03   the camera has no 25, it rounds up
asked 20 -> delivered 19.99   honoured
asked 15 -> delivered 15.08   honoured
asked 10 -> delivered 14.98   no 10, rounds to 15
asked  5 -> delivered 14.99   no 5, rounds to 15
```

The picker offered 10, 15, 20, 25, 30, built as an arithmetic sequence up to
the measured ceiling. The camera has 30, 20 and 15. Picking 25 always produced
30, and nothing said so.

### 4.2 The stored ceilings were the uncompressed format's

`camc1aefd1e` was stored as 1.0 fps at 3840x2160 and 1.0 at 1920x1080 through
DirectShow, which is absurd for a camera that records 4K. Asking FFmpeg to
enumerate the same device takes about a second and explains it exactly:

```
vcodec=mjpeg          3840x2160  fps 5..30
vcodec=mjpeg          1920x1080  fps 5..30
pixel_format=yuyv422  3840x2160  fps 1..1
pixel_format=yuyv422  1920x1080  fps 3..3
```

The stored 1.0 and 3.0 are the **yuyv422** rows. The probe believed it was
measuring MJPEG and was measuring uncompressed: the FOURCC had not stuck. A
UVC driver renegotiates the pixel format whenever the resolution or the rate
changes, and lands back on uncompressed, whose bandwidth ceiling over USB is
what produced 1 fps at 4K.

The same enumeration also lists **2592x1944 and 2048x1536**, two modes the
rig's fixed `RESOLUTION_LADDER` never tries, so they could never be offered.

### 4.3 Measuring on a shared handle contaminates the next mode

`measure_fps_for_modes` walks the ladder largest first on one open handle to
save time. It guards against the handle failing to switch, and that guard
fires. What it cannot see is a handle that reports the new size while frames
already in flight belong to the previous, much slower mode. With a 0.5 s drain
and a 1.2 s measurement window, a mode measured right after 4K at 1 fps is
measured against an almost empty pipeline.

### 4.4 The delivered rate moves with the room

The same camera, same mode, same door, minutes apart:

```
camc1aefd1e 1920x1080 DirectShow:  30.13 fps   then   24.88 fps
camc1aefd1e  640x480  DirectShow:  auto-exposure on 24.67, off 30.10
```

Auto-exposure lengthens integration in dimmer light and the camera drops
frames to fit. So a measured ceiling is not a capability at all: it is a
capability multiplied by the lighting at the moment of measurement. Calibrate
in a dim room and the rig will refuse to offer a rate the camera has.

This is the single most important consequence in this document, because it
means **no amount of careful measurement can produce a correct list of
offered rates.** Only enumeration can.

## 5. The design

### 5.1 Enumerate first, measure second

Ask the platform what the camera offers, and measure only to find out what it
delivers under present conditions. The two answers serve different questions
and must be stored separately.

| | Source | Cost | Changes with |
|---|---|---|---|
| Offered sizes and formats | FFmpeg `-list_options` (Windows), `v4l2-ctl` (Linux) | about 1 s | replugging, different port |
| Offered rates | exact on Linux; bounded on Windows, confirmed on open | about 1 s | as above |
| Delivered rate | count frames after opening | 1 to 2 s per mode | light, load, bandwidth |

Measured on this rig, three cameras, whole enumeration: 1.23 s, 0.65 s and
0.78 s. The trial probe it replaces takes about 40 s per camera and answers a
different question badly.

FFmpeg is already a dependency of this project for recording, and it is on the
path on every target. On Windows it enumerates DirectShow devices by name and
by a stable `@device_pnp_` moniker that carries the vendor and product id, so
it maps onto the identity the rig already uses. On Linux the same information
comes from V4L2, which FFmpeg also reads.

Media Foundation has no FFmpeg lister. Its native types are usually the same
set as DirectShow's for a UVC device, so the enumerated list is used for both
doors and the delivered rate is measured per door.

### 5.2 What each control is for

| Control | Answers | Fed by |
|---|---|---|
| Resolution | which of the camera's frame sizes to use | enumerated sizes |
| Format | which pixel format, because it changes the achievable rate at the same size | enumerated formats |
| FPS | which of the camera's actual intervals at that size and format | enumerated intervals, not an arithmetic ladder |
| Backend | which OS door, because the doors differ in what they negotiate and how reliably | platform |
| Detect | refresh the enumeration and measure delivery | both |

The FPS list must be the intervals the camera lists. That removes the guessing
entirely, and with it the need to learn which rates fail.

### 5.3 Where each number goes

```
enumerate  ->  offered modes            -> Resolution / Format / FPS lists
operator   ->  requested mode + rate    -> project config, camera open
camera     ->  delivered rate           -> video file header, live tile
```

The recorder stamps the file with the requested rate when the camera is
holding it, because a measurement carries about a frame of noise and 30 is
more accurate than 29.99. It stamps the delivered rate when the camera is not
holding it, and says so. The live tile shows all three, because when they
disagree the operator needs to see which one is wrong.

### 5.4 Scientific cameras stay on their own path

Spinnaker and XIMEA report an achievable rate that already accounts for
exposure, region and link speed. They are asked, not probed, and their answer
is refreshed whenever exposure changes. Nothing in this document applies to
them except the separation of requested from delivered.

## 6. What the video pipeline needs from a camera

The rate is not the only property that matters, and the file header is not the
only thing that breaks when it is wrong.

### 6.1 The pipeline is paced by delivery, not by the request

`OpenCVCamera.get_available_images` calls `cap.read()`, which blocks until the
next frame, and `CameraThread.run` spins with no sleep. So the whole pipeline
runs at whatever the camera delivers. Nothing anywhere throttles to the
requested rate.

That is worth stating plainly: **the requested rate is a request to the driver
and nothing else in the rig enforces it.** A camera asked for 25 and
delivering 30 runs every sink 20 percent faster than the operator believes.

### 6.2 What each consumer depends on

| Consumer | Needs | Policy | What a wrong rate costs |
|---|---|---|---|
| CameraThread ring | monotonic capture stamp, gap-free frame id | clamps the stamp | nothing, it repairs |
| RecorderSink | every frame, and a TRUE rate for the header | lossless, fixed 90-frame queue | wrong header plays back at the wrong speed; the queue is 3 s at 30 fps and 1.5 s at 60 |
| TrackerSink | every frame | drop-oldest, `maxsize = max(15, 1 s x fps_hint)` | `fps_hint` defaults to 30 and nothing passes it, so at 60 fps the queue holds half a second |
| PoseSink | throughput, and a frame id to match results | drop-newest, maxsize 1 | a higher rate means more skips and lower coverage |
| MCUPusher | low latency and ordering | event driven | nothing directly |
| Display | nothing, it may skip | skips | nothing |

Two of those queue depths are derived from an assumed 30 fps. They are sized
for a rate nobody checks against the rate the camera is running at.

### 6.3 The capture stamp is not a capture time

`capture_host_ns` is taken when `cap.read()` returns, and the field the
backends report is literally `timestamp_source: "host_backstamped"`. It
carries the whole USB and driver delay. Measured on this rig from a lamp
lighting to the frame being in hand:

```
direct USB camera        72 ms
CCTV chain, 4 boxes     121 to 125 ms
```

Nothing downstream can recover that. Anything aligning video to controller
time inherits it as a fixed offset, and it is not the same offset on every
camera. Spinnaker and XIMEA report `timestamp_source: "hw"` and carry a real
device timestamp plus a frame id in chunk data, so the two families are not
comparable and the session log does not currently record which one produced a
given file.

## 7. The roots, and what each can and cannot do

| | Timestamp | Rate control | Enumeration | Main caveat |
|---|---|---|---|---|
| UVC / DirectShow | host, at `read()` | set and hope; driver rounds to its nearest interval and reports the request back | sizes and formats exact, rates as a RANGE | will not release the device promptly; opens reliably in about 0.6 s |
| UVC / Media Foundation | host, at `read()` | same | exact per native type, not reachable from FFmpeg | intermittent. Refuses a device DirectShow has just closed, measured failing 1.5 s and 3.0 s later. Sometimes the only door that works at all |
| UVC / V4L2 | host, at `read()`; a monotonic buffer stamp exists and is not used | same | **exact interval list** | the only root that answers the rate question properly |
| UVC / GStreamer, CAP_ANY | host | caps negotiation | exact, caps carry a framerate list | needed on Jetson CSI nodes where V4L2 reports not-opened |
| Spinnaker | **hardware**, chunk timestamp and FrameID | continuous, derived from exposure, region and link | exact, and live | a different model entirely; asking is correct, probing is not |
| XIMEA | **hardware** | continuous | exact, and live | as Spinnaker |

Caveats that apply to every UVC root, whichever door:

- **The pixel format does not stick.** A driver renegotiates it whenever the
  size or the rate changes and lands back on uncompressed, whose bandwidth
  ceiling over USB is roughly a tenth of MJPEG's. This is why the format must
  be sealed LAST, after the size and the rate, and re-sealed after any rate
  change.
- **Auto-exposure caps the rate.** In dim light the camera lengthens
  integration and drops frames to fit. Measured: 24.67 fps with auto-exposure
  on and 30.10 with it off, same camera, same mode, same minute.
- **Bandwidth is shared per USB controller and nothing models it.** Four
  cameras on one hub divide one budget. The calibration store records
  `bus_speed` so a camera moved from USB3 to USB2 is re-probed, but there is
  no model of two cameras competing, which is exactly the CCTV and 2-camera
  rigs.
- **`CAP_PROP_BUFFERSIZE=1` reduces but does not remove batching.** Frames can
  still arrive in bursts, which is what the ring buffer and the drop log exist
  to absorb and count.

## 8. Where the gaps are

Each of these is a requirement the pipeline has that no root currently
satisfies on its own.

**G1. No exact rate list on Windows.** DirectShow bounds the rates. Closed by
enumerating for the bounds and confirming the list when the camera opens.

**G2. No hardware timestamp on UVC.** The capture stamp carries the transport
delay. Cannot be closed, only named: the session log should record
`timestamp_source` so an analysis knows whether frame times are hardware or
host, and the measured transport delay should be recorded with the session
rather than assumed to be zero.

**G3. Nothing paces to the requested rate.** Either the rig decimates to hold
the request, or it stops presenting the request as a rate. Decimating conflicts
with the lossless recorder contract and discards frames the tracker wants, so
the honest choice is the second: the delivered rate is the rate, the recorder
already stamps it, and the picker should only offer rates the camera has.

**G4. Sink queues are sized from an assumed 30 fps.** `RecorderSink` uses a
fixed 90 and `TrackerSink` uses an `fps_hint` nobody passes. Both should be
sized from the rate the camera is actually delivering.

**G5. No model of cameras sharing a controller.** Out of scope to solve, but
the enumeration makes it visible: a camera offering 30 fps at 1080p that
delivers 18 with three siblings running is reporting a bandwidth problem, not
a capability.

## 9. Implementation plan

Ordered so each step is useful alone and testable without the next, and so the
steps that change what the operator sees come after the steps that make the
data true.

### Stage 1, know what the camera offers

1. **An enumerator.** `source/video/cameras/enumerate_modes.py`. Returns
   `(size, pixel_format, rates)` per camera: FFmpeg `-list_options` on Windows,
   `v4l2-ctl --list-formats-ext` on Linux where it gives the exact list, FFmpeg
   V4L2 otherwise. Empty means the question could not be asked, never "no
   modes". **Done.**
2. **Store offered beside delivered.** A new `offered` field in the per-machine
   calibration entry, alongside the existing `variants`. Neither overwrites the
   other: one is a property of the camera, the other of the room.

### Stage 2, make the pickers honest

3. **Resolution, Format and FPS come from the enumeration.** Sizes from the
   offered sizes, formats from the offered formats, rates from the offered
   rates for the selected size and format. This deletes the arithmetic ladder,
   which is what offered 25 fps to a camera without it and hid the 30 it has.
4. **Detect becomes enumerate then verify.** Enumerate (about 1 s), then
   measure delivery at the SELECTED mode only. Seconds instead of the current
   minute, and it answers both questions instead of confusing them.

### Stage 3, make the pipeline agree with the camera

5. **Size the sink queues from the delivered rate.** `RecorderSink`'s fixed 90
   and `TrackerSink`'s unset `fps_hint` both assume 30. At 60 fps they hold
   half what they were designed to.
6. **Record `timestamp_source` in the session log.** An analysis reading frame
   times needs to know whether they came from hardware or from the host, and
   the transport delay measured for that camera belongs beside them.

### Stage 4, say what is true

7. **Three numbers wherever a rate is shown**, requested, offered, delivered,
   and a warning when delivered leaves requested. Partly done: the recorder
   stamps the delivered rate and the live tile separates the three.

### Not doing, and why

**Decimating to hold the requested rate.** It would make the request real, and
it conflicts with the lossless recorder contract, discards frames the tracker
needs, and adds a second place where a rate is decided. The delivered rate is
the honest one. The right fix is to stop offering rates the camera does not
have, which is stage 2.

**Modelling shared USB bandwidth.** Out of scope. The enumeration makes it
diagnosable instead: a camera that offers 30 and delivers 18 alongside three
siblings is reporting contention, not capability.

## 10. What this replaces

The arithmetic FPS ladder derived from a measured ceiling goes. It is the
mechanism that offered 25 fps to a camera without it and, on a camera measured
in dim light, hid the 30 it does have.

Two mechanisms STAY, and the reason is the DirectShow range in section 2.

- The delivered-rate measurement at open. It answers a different question and
  it is the only thing that catches a camera offered 30, set to 30, producing
  24 because the room is dark.
- The record of rates a camera was seen to round away. On Windows the
  enumeration bounds the rates without listing them, so the exact list inside
  those bounds is still learned by use. On Linux, where the list is exact,
  nothing will ever be recorded and the mechanism costs nothing.

An earlier draft of this document proposed deleting both. That was written
before the DirectShow output was read carefully enough to notice it reports a
range, and it was wrong.

---

## 9. The FOURCC ordering bug, and what it invalidated

Found 2026-09-11, on a rig where every stored rate had looked wrong for days.

`_apply_fourcc` set the pixel format and then re-asserted width/height, on the
stated belief that "the resolution must follow the FOURCC or the driver clamps
it". DirectShow does the opposite: a resolution set *after* the FOURCC
renegotiates the format, and lands back on uncompressed. Measured on one
camera at 1920x1080, same device, same minute:

```
fourcc, then size   ->  YUY2  10.7 fps
size, then fourcc   ->  MJPG  56.7 fps
```

Nothing checked the negotiated format afterwards, so every camera on the rig
streamed YUY2 while the application logged MJPEG, and **every rate the probe
ever stored was an uncompressed rate**. That is the whole of "10 fps at 1080p"
on a camera that does nearly 60. It is also why each fix layered on top of it
changed nothing visible: the pickers, the ceilings and the tile were all
faithfully reporting numbers that had been measured in the wrong format.

The ordering is fixed in all three places that set it
(`_apply_fourcc`, `probe_supported_resolutions`, `measure_fps_for_modes`), and
the format is now a parameter rather than a hardcoded `MJPG`, threaded
`probe_all -> probe_camera -> probe_supported_resolutions / measure_fps_at /
measure_fps_for_modes`. `OpenCVCamera.supports_format` confirms the
negotiation before anything is measured, and `measure_fps_at` returns 0.0
rather than a rate taken in a format nobody asked for.

**Anything measured before this fix is worthless.** Re-run Detect per camera.

### 9.1 `min == max` is a discrete interval, not a ceiling

DirectShow reports one capability per (format, size) with a minimum and a
maximum frame interval, and the two cases mean different things:

```
cam5b4028a9  mjpeg 640x480   min=10       max=60.0002   a RANGE
camc1aefd1e  mjpeg 640x480   min=5        max=30        a RANGE
cam519bc077  mjpeg 640x480   min=120.101  max=120.101   a SINGLE
```

cam519bc077 reports `min == max` on every line it has: it is a
discrete-interval device (UVC `bFrameIntervalType` >= 1), and its 640x480
MJPEG really is 120 fps and nothing else. Asking it for 30 there delivered 95.
Padding such a value out into a ladder invents rates the camera does not have,
so a collapsed range is taken verbatim.

### 9.2 A reported range does NOT give the discrete list

This is the one still open. For a camera reporting `min=10 max=60`, the
standard rates inside are *inferred*, not read. Measured on cam5b4028a9 at
640x480 MJPEG by setting each rate and counting frames:

```
asked 10 -> 10   asked 20 -> 20   asked 30 -> 30   asked 60 -> 60
asked 15 -> 20   asked 24 -> 30   asked 25 -> 30   asked 50 -> 60
```

Only four of the eight offered rates exist. `cap.get(CAP_PROP_FPS)` returned
the request every single time, so the driver's own report is worthless and
only frame counting tells the truth. Until Detect measures each candidate
rate, the FPS list for a range-reporting camera is partly a guess, narrowed
after the fact by `calibration_store.note_rate_rejected`.

### 9.3 One format has several FOURCC spellings

`YUYV` and `YUY2` are the same format. Comparing the raw codes reported a
mismatch on every YUY2 open and warned that the camera had ignored a request
it had honoured. `_same_format` normalises through `_FORMAT_FOR_FOURCC`. A
warning that fires when nothing is wrong is the one that teaches people to
ignore the real ones.

### 9.4 The lesson

Every wrong number in this file's earlier sections came from trusting a stored
or parsed copy instead of asking the device. The fault was found in one step
by running `ffmpeg -list_options` on the camera and counting delivered frames,
after days of reasoning about our own tables. Go to the hardware first.
