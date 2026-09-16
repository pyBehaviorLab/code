# The operant chamber

The chamber develops the published open five-choice design, with three changes:

- the pyControl breakout is replaced by the pyBehaviorLab board;
- video acquisition is integrated into the same workflow;
- the response wall and house light are rebuilt.

The trapezoidal wall geometry, externally mounted pumps and centrally positioned
overhead camera are retained.

## What sits outside the chamber

The board in its printed enclosure, the three peristaltic pumps and a 12 V
ventilation fan mount **outside** the inner volume. All are supplied and
switched from the board. This placement:

- leaves the inner volume unobstructed beneath the overhead camera, which
  matters for a tethered animal;
- holds pump noise and vibration away from the chamber;
- lets every surface the animal contacts be cleaned without disturbing
  electronics.

Each chamber lives in a sound-attenuating cubicle carrying all of that.

## The response wall

The five-aperture wall is a printed circuit board with **reduced centre-to-centre
aperture spacing** relative to the published design. The **number of active
apertures is set by the task, not by the hardware.** The reversal-learning
paradigm in the paper uses three of the five.

Each aperture carries an IR beam for entry detection and an LED for cue
presentation, driven from the task:

```python
hw.five_poke.poke_2.LED.on()      # cue the centre aperture
```

The `Five_poke` driver spans **two RJ ports** (typically `port_1` + `port_7`).

## Lighting

The house light is a board carrying independently switchable **infrared, green
and white** LED arrays selected by DIP switch. Infrared allows video in an
otherwise dark chamber; green is used for cues on a reversed light cycle. See
[Peripheral modules](peripherals.md#house-light-and-illumination-choice) for why
red is not used.

## Reward delivery

Three peristaltic pumps deliver liquid reward through 2 mm OD tubing, driven from
motor channels `M1`–`M3`. Calibrate by dispensing a known number of deliveries at
the task volume and weighing the total, before the experiment and again at the
end.

## Camera placement decides what tracking can resolve

An overhead camera gives the top-down view used for recording and pose.
**Mounting height and viewing angle set a floor on what any tracker can
resolve.** What matters is how far apart two response locations fall in the
image, relative to the scatter of the tracked point.

This was measured. In the sixteen-chamber experiment, per-chamber agreement
between the pose record and the controller record scaled with that ratio. It did
**not** scale with the proportion of untracked frames. Chambers whose camera
viewed the response wall obliquely, so that adjacent ports overlapped in the
image, gave the weakest agreement.

:::{admonition} Position the response wall as close to normal to the optical axis as the enclosure allows
:class: important

Ports that overlap in the image cannot be separated by a better model or a
higher frame rate. This is a geometry problem, fixed at mounting time.
:::

Occlusion has the same character. A snout inside a port cannot be localised, no
matter how good the model.

## Wiring for a three-port task

The layout used for probabilistic reversal learning:

| Connection | Goes to |
|---|---|
| Host | Board over USB |
| Camera | Host over USB, **or** the CCTV system |
| Task-relevant nose-poke ports | Corresponding RJ peripheral ports |
| Three peristaltic pumps | Motor channels `M1`–`M3` |
| 12 V ventilation fan | Board-switched 12 V output |
| House light | One RJ port |

## Running many chambers at once

Each chamber has **its own controller**; several controllers share one host over
a USB hub. Sixteen chambers were run concurrently from a single host in the
paper.

Video reaches those chambers by either of two routes, which are alternative
inputs to the same frame bus. The acquisition route is therefore a
**configuration choice, not a different implementation**:

```{mermaid}
flowchart LR
  U1["USB camera<br/>per chamber"] --> FB
  A1["Analogue cameras"] --> MUX["CCTV multiplexer"]
  MUX --> CAP["HDMI-to-USB<br/>capture device"]
  CAP --> FB
  FB["Frame bus<br/>per-box ROI split"] --> R["Recorder"]
  FB --> T["Tracker"]
  FB --> D["Display"]
  classDef hl fill:#efeafd,stroke:#5b4bd6,color:#2b2340;
  class FB hl;
```

Choose per experiment:

| | Individual USB cameras | Analogue + CCTV + one capture device |
|---|---|---|
| Frame interval | Known and stable | Longer and more variable |
| Cost per chamber | Higher | Much lower |
| Scales to 16 boxes | Needs the USB bandwidth | Yes, one capture device |
| Suits | Timing of individual frames matters; reacting to brief events | Monitoring, recording, zone-based control; pose computed afterwards |

With the shared route, one camera's image is divided into **one region of
interest per chamber** in software, and each region stays bound to its own
controller, subject, task, video file and tracking output. See
[Cameras](../user-guide/cameras.md) for how regions are drawn and stored.

:::{admonition} The limit of the low-cost route
:class: warning

Analogue CCTV acquisition constrains resolution and frame rate. It also adds a
longer and more variable acquisition delay than a triggered machine-vision
camera.

Use it for monitoring, recording and zone-based control. Avoid it for
experiments that must act on events lasting a few tens of milliseconds. The
**same software path** accepts machine-vision cameras for those.
:::

## Cleaning

Clean every surface the animal contacts between animals. Because the pumps sit
outside and the liquid path is tubing only, pump cleaning is flushing or
replacing a length of tube.
