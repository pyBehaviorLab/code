# Performance and validation

This page records what was measured on the assembled system, and how.

The end-to-end numbers apply to **one camera and one host**. They are not a
property of the software. Measure your own configuration before relying on it to
act on a given event.

## Two different latencies

The system has two latencies. They are measured separately and are not pooled,
because they describe different parts of the system.

```{mermaid}
flowchart LR
  A["Animal<br/>moves"] --> B["Camera<br/>exposure + transfer"]
  B --> C["Frame handling<br/>on host"]
  C --> D["Pose<br/>inference"]
  D --> E["Event to<br/>controller"]
  E --> F["Task<br/>processing"]
  F --> G["Actuation"]
  H["Digital<br/>input"] --> F
  classDef fw fill:#fdf0e3,stroke:#c07a2c,color:#3a2a15;
  classDef path fill:#efeafd,stroke:#5b4bd6,color:#2b2340;
  class F,G fw;
  class B,C,D,E path;
```

**Framework response latency** covers the orange span only: a digital input
transition to the corresponding output transition.

**Camera-to-actuator latency** covers the whole chain. It is not the sum of the
framework latency and a published inference time. Each stage's cost depends on
the others, so the figure is measured end to end.

## Framework response latency and timing accuracy

Measured with a **PicoScope 2205A** on the breakout board, recording the
controller output at `PA5` and the input at `PB8` with a common ground.

| | Method |
|---|---|
| **Response latency** | `input_follower.py`, the output follows externally driven input transitions; latency computed separately for rising and falling edges |
| **Timing accuracy** | `triggered_pulses.py`, each input transition triggers an output pulse of predefined duration; error is measured minus target |
| **Low load** | The timing task alone |
| **High load** | `ANALOG_IN_1` (PF4) and `ANALOG_IN_2` (PF10) acquired concurrently, plus framework events supplied from a second board |

Two independent boards were tested per configuration.

**Results.** Response latency was shortest on the **H723 with high-frequency
polling**, intermediate on the **H723 with low-frequency polling**, and longest
on the **F767**. The ordering held under both loads. The two boards of each
configuration gave closely matching distributions, so the differences belong to
the configuration, not to an individual board.

Pulse-duration error spanned about **one polling interval**, biased toward
durations shorter than the target. A software timer expires on a framework cycle
rather than between cycles, so **the polling interval sets the timing
resolution**.

:::{admonition} Load widens the distribution, it does not merely shift it
:class: important

Concurrent activity defers input handling. Budget for the tail, not the mean, if
your task acts on brief events.
:::

Per-board distributions are in Figure 9 of the paper.

## Camera-to-actuator latency

The acquisition chain adds delay that device specifications do not state,
particularly when frames pass through a capture device. The measurement
therefore uses a reference the software cannot influence: three LEDs inside the
camera's field of view, driven by the breakout board.

| LED | Driven | Gives |
|---|---|---|
| Session LED | On for the session | Session boundary in the video |
| 100 Hz LED | Pulsed at 100 Hz | A fine time reference **visible in the frames themselves** |
| Frame-trigger LED | Toggled on each frame event returned by the host | The host's frame handling, recorded in the video |

This writes the controller's timing into the video itself. Each acquisition path
is then measured against **the sync timestamp that stamps the behaviour**, with
no external synchronisation device.

The normal online pathway detects the configured condition, transmits the event
and generates the digital output. Latency is computed between the reference
transition and the controller response, from independently recorded hardware
signals.

Both acquisition routes were measured: a USB camera connected directly to the
host, and an analogue camera through the CCTV system and an HDMI-to-USB capture
device. See
[The operant chamber](../hardware/operant-chamber.md#running-many-chambers-at-once)
for how the two routes compare.

:::{note}
This quantity includes camera acquisition and transfer, frame handling,
inference, communication of the derived event and task processing. **It is not
inference time alone**, and it should not be compared against published
inference benchmarks.
:::

### Which instant is which

The walkthrough below follows one trial: the lamp lit, the frame carrying it
arriving, and the answer reaching the controller, with the three timestamps
that bracket them.

```{raw} html
<iframe src="../_static/media/latency/latency_measurement_flow.html"
        title="How latency is measured"
        style="width:100%;height:640px;border:1px solid var(--pst-color-border,#e2e6ee);border-radius:6px"
        loading="lazy"></iframe>
```

[Open it in its own tab](../_static/media/latency/latency_measurement_flow.html)
if the frame is too small to read.

Two things it makes explicit, because both are easy to get wrong when comparing
against a published figure. The answer lamp is lit **after** its arrival is
timestamped, so it sits outside its own measurement. And the interval starts at
the **stimulus**, not at the frame reaching the host, so it contains the camera
and the return path that an image-to-pose figure excludes by construction.

## Does automation change the behaviour?

Automation is only useful if it does not alter what is being measured. This was
tested directly.

The same mice were tested in the same maze under **manual and fully automated**
operation, order counterbalanced, in both a rewarded and a spontaneous
alternation paradigm. In the automated condition the state machine controlled the
doors and tracked position advanced the task phases; in the manual condition the
experimenter performed the same operations, preserving the behavioural sequence.

**Result:** performance did not differ between modes in either task, and trial
duration in spontaneous alternation was likewise unchanged (p > 0.05, paired
t-tests). Removing the experimenter from the room did not alter the measured
behaviour.

## Tracking against manual scoring and commercial software

The same object- and social-interaction recordings were scored three ways:
manually by an observer blind to the automated output, by pyBehaviorLab, and by
ANY-maze. Differences therefore reflect the **analysis method**, not different
sessions.

Taking manual scoring as the reference:

| Measure | pyBehaviorLab | ANY-maze |
|---|---|---|
| Time in head interaction with object | **r = 0.997** | 0.974 |
| Time with head in social zone | **r = 0.973** | 0.795 |
| Number of head interactions with object | **r = 0.982** | 0.762 |
| Number of head entries into social zone | **r = 0.763** | 0.209 |

Note where the two methods diverge. On **cumulative durations** both agree
closely with manual scoring. On **event counts** the gap is much wider.

Measuring how long an animal spent in a region is easier than deciding how many
separate interactions that period contained. Resolving the head as a distinct
body point is what improves the second measure.

## Agreement between the pose record and the controller record

In the sixteen-chamber reversal-learning experiment, every poke has **both** a
controller timestamp and a tracked snout position, so the two records can be
checked against each other directly.

Doing so exposed one requirement. The video is written at a constant nominal
rate while the camera acquires at its own slightly different rate, so the encoder
resamples the stream and **a frame index does not convert linearly to time**. In
these recordings the implied time drifted locally by up to several hundred
frames, while remaining correct on average.

Timing each frame against the controller stamp in the acquisition log removes
this. No additional synchronisation hardware is required. With that correction:

- the pose recovered the chosen port for a median of **99.8 % of trials** per
  session;
- the matched frame fell a median of **6 ms after** the controller timestamp;
- **88.5 %** of matches lay within one frame;
- agreement was **equivalent when the same recordings were re-tracked with
  DeepLabCut**, so it reflects the alignment rather than the pose backend.

:::{admonition} Do not convert frame index to time
:class: danger

Use the per-frame controller timestamp in `_video_data.txt`. A frame index looks
correct on average and is wrong locally. See
[Time sync](../concepts/time-sync.md) and
[File formats](file-formats.md#_video_datatxt-v2).
:::

Where agreement was poorer it was attributable to **the image**, not the tracking
or the timing. See
[camera placement](../hardware/operant-chamber.md#camera-placement-decides-what-tracking-can-resolve).

## Known limits

Each of these bounds what an experiment can ask.

- **Analogue CCTV acquisition** constrains resolution and frame rate and adds a
  longer, more variable acquisition delay than a triggered machine-vision camera.
  Use it for monitoring, recording and zone-based control, not for events lasting
  a few tens of milliseconds. The same software path accepts machine-vision
  cameras.
- **The online tracker is deliberately lossy.** It skips frames under load to
  stay current, so position-driven rules are best written on states that persist
  across several frames, such as zone occupancy rather than instantaneous
  position.
  Analyses needing per-frame pose should use an offline pass over the complete
  recording.
- **Pose-based control inherits the properties of its model.** Tracking quality
  depends on the training set, the illumination and the camera placement, and an
  occluded body point cannot be localised however good the model. The clearest
  case is a snout inside a port.
- **Camera geometry sets a floor.** Adjacent response locations may not be
  separable in an image where the apparatus occupies a small part of the frame.
- **The maze is a mechanical system** and needs its own alignment and end-stop
  checks before use. The number of arms is bounded by the controller's eight
  stepper channels.
