"""How a box's frame becomes the model's input, decided from the model, once.

Three ways to present a frame to a pose network, each right for a different
kind of model, and choosing the wrong one is silent:

``full``
    Hand the frame over at its own size. Correct for a fully-convolutional
    network with no baked-in input shape, DeepLabCut's PyTorch runner pads to
    the backbone stride itself and its config asks for no resize at inference.
    Nothing is scaled, so nothing has to be scaled back.

``letterbox``
    Scale by one factor and pad to a fixed shape. Correct for a model trained
    on whole frames, and the only option for an exported graph whose input size
    was fixed at export.

``crop_track``
    Cut a window of exactly the training size at native resolution and follow
    the animal with it. Correct for a model trained on **crops**, where scaling
    the frame to the model's input is precisely the wrong move: it changes the
    animal's pixel size, which is the one thing the network was trained to
    expect.

The distinction that matters is the last one, because the two look identical in
a config file. sleap-nn writes the training image size as
``preprocessing.max_width`` / ``max_height`` whether the labels were whole
frames or crops. What separates them is arithmetic, not a flag: a model trained
on whole frames declares an input about the size of a frame, and a model
trained on crops declares one much smaller. Combined with ``scale: 1.0``
(no downscaling at training) that is a sound reading, and it is self-calibrating,
the same rule leaves a frame-trained model on the letterbox it needs.

This was worth writing down because the default cost real data. A 224x176
crop-trained SLEAP model was letterboxed from a 513x315 arena at scale 0.437,
making the animal 2.3x smaller than anything the network had seen. Measured
against the model's own shipped self-check, the same weights gave 0.3 px error
on their native crop, 1.7-2.1 px through ``crop_track``, and 9-11 px through
the letterbox, with four of six keypoints landing in the padding and being
dropped outright.

Both the live sink and the offline retracker ask here, so the two pipelines
cannot disagree about what a model wants.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from source.video.tracking.model_config import ModelInfo

logger = logging.getLogger(__name__)

#: How much larger than its declared input a frame has to be before the window
#: has to FOLLOW the animal rather than merely sit on it. Below this the window
#: covers essentially the whole frame and cannot be anywhere else; above it,
#: where the window goes is the whole question.
#:
#: It is not a test for whether to cut rather than scale. It was, and a
#: multi-box rig showed why that was wrong: at sixteen boxes each ROI is
#: 160x120, smaller than this model's 224x176 window, so the frame failed the
#: margin and fell through to the letterbox, which then UPSCALED it by 1.4x,
#: making the animal larger than anything the network was trained on. Wrong in
#: the opposite direction from the bug this module was written for, and just as
#: invisible.
CROP_MARGIN = 1.25

#: The scale that means "the network was trained on pixels at their original
#: size". An ABSENT scale is deliberately not counted: DeepLabCut states none,
#: and its declared input size comes from an export's baked-in graph shape
#: rather than from a training crop, reading silence as 1.0 made every DLC
#: export look crop-trained and cut a window out of a model that wanted the
#: whole frame.
NATIVE_SCALE = 1.0

#: Model families that do their own cropping. A window of ours would crop a
#: crop for a top-down pose stage, and a bottom-up network is looking for every
#: animal in the frame, so centring on one is the wrong input by construction.
SELF_CROPPING = ("centroid", "centered_instance", "topdown",
                 "multi_class_topdown", "bottomup", "multi_class_bottomup",
                 "multi-animal")

#: DLC engines whose input shape is fixed when the graph is built.
FIXED_SHAPE_ENGINES = ("export", "tensorflow")

#: DLC engines positively known to accept a frame at whatever size it arrives,
#: fully convolutional, padding to the backbone stride internally. Stated as a
#: whitelist rather than "not fixed-shape", because an engine nobody here
#: recognised is an engine whose input rules are unknown, and the letterbox is
#: the answer that cannot be wrong for one of those.
ANY_SIZE_ENGINES = ("pytorch",)


@dataclass(frozen=True)
class InputPlan:
    """What to do with a frame before it reaches the network."""

    mode: str                                  # full | letterbox | crop_track
    size: Optional[Tuple[int, int]] = None     # (w, h) the mode works to
    reason: str = ""                           # why, in one line, for humans
    from_model: bool = False                   # size read from the model config

    def describe(self) -> str:
        if self.size:
            return f"{self.mode} {self.size[0]}x{self.size[1]}, {self.reason}"
        return f"{self.mode}, {self.reason}"


def crop_trained(info: ModelInfo, frame_w: int, frame_h: int) -> bool:
    """Whether this model must be fed pixels at their original size.

    The question is only ever about scale, and one field answers it: a model
    that states ``scale: 1.0`` was trained on pixels as the camera produced
    them, so the frame must be **cut or padded** to reach the input size, never
    resized to it. Cutting preserves the animal's pixel size; scaling, in
    either direction, destroys the one property the network was trained on.

    Deliberately independent of how the frame compares to the window. An
    earlier version required the frame to be substantially LARGER, which is
    right for the case it was written for and silently wrong for the case a
    16-box rig produces: an ROI smaller than the window fell through to a
    letterbox that upscaled it. Padding a small frame is exactly as
    scale-preserving as cropping a large one, and ``crop_at`` already does
    both.

    False whenever the config does not say enough to be sure. DeepLabCut states
    no scale and its declared input is an export's baked-in graph shape, so
    silence is not read as native, doing so made every DLC export look
    crop-trained.
    """
    if not (info.input_w and info.input_h):
        return False
    return info.native_scale == NATIVE_SCALE


def plan(tracker_type: str, model_path: str, frame_w: int, frame_h: int,
         *, engine: str = "", asked: str = "auto",
         asked_size: Optional[Tuple[int, int]] = None) -> InputPlan:
    """The input plan for one model against one box's frame size.

    ``asked`` is the operator's choice: ``auto`` lets the model decide, any
    other value is honoured unless the model cannot work that way, in which
    case the refusal is carried in ``reason`` rather than applied silently.
    ``asked_size`` overrides the window size for ``crop_track``: an operator
    who wants a wider window than the training crop can have one.
    """
    info = _read(model_path)
    asked = (asked or "auto").strip().lower() or "auto"
    is_sleap = str(tracker_type or "").lower().startswith("sleap")

    if asked == "dlc_dynamic":
        # DeepLabCut-Live cuts and restores its own window inside the runner,
        # so the frame reaches it the ordinary way and the mode name is carried
        # through only so the sink can hand the flag to the constructor.
        #
        # It exists in exactly one place: dlclive's PyTorch runner. The
        # TensorFlow runners never receive the keyword (their factory filters
        # it out), and an exported ONNX/TensorRT graph is not driven through
        # dlclive at all, so asking for it anywhere else is a setting that
        # would do nothing, silently, which is the failure this module exists
        # to stop.
        if is_sleap or engine != "pytorch":
            where = "SLEAP" if is_sleap else f"the {engine or 'chosen'} engine"
            return InputPlan(
                "letterbox", _letterbox_size(info, frame_w, frame_h),
                f"DeepLabCut-Live's dynamic cropping is not available to "
                f"{where}, it exists only in its PyTorch runner",
                bool(info.input_w and info.input_h))
        return InputPlan("dlc_dynamic", None,
                         "DeepLabCut-Live does its own dynamic cropping, "
                         "around the keypoints it last found", False)

    if asked not in ("auto", "full", "letterbox", "crop_track"):
        logger.warning("unknown pose input mode %r, deciding from the model",
                       asked)
        asked = "auto"

    if asked == "crop_track" or (asked == "auto"
                                 and crop_trained(info, frame_w, frame_h)):
        refusal = _crop_refusal(info, engine, is_sleap)
        if refusal:
            return InputPlan("letterbox", _letterbox_size(info, frame_w,
                                                          frame_h),
                             refusal, bool(info.input_w and info.input_h))
        size = _crop_size(info, asked_size)
        if size is None:
            return InputPlan(
                "letterbox", _letterbox_size(info, frame_w, frame_h),
                "crop-track needs a window size and neither the project nor "
                "the model declares one", False)
        from_model = not (asked_size and asked_size[0] and asked_size[1])
        if frame_w <= size[0] and frame_h <= size[1]:
            # The window is bigger than the picture, so there is nothing to cut
            # and nowhere to follow. Use the whole frame at native scale.
            #
            # Padding it out to the model's window instead would run inference
            # over a large black border for nothing, on a 16-box rig that is
            # 224x176 of work for a 160x120 picture, twice the pixels. The only
            # padding kept is what the backbone stride actually requires: a
            # UNet that halves four times cannot take a height of 120, and
            # feeding it one tears the decoder's skip connections apart.
            size = _stride_pad(frame_w, frame_h, info.max_stride)
            return InputPlan(
                "crop_track", size,
                "the frame is smaller than the model's window, so the whole "
                f"frame is used at native scale (padded to {_wh(size)} for the "
                "backbone stride)", False)
        follows = (frame_w > size[0] * CROP_MARGIN
                   or frame_h > size[1] * CROP_MARGIN)
        # Only a size the OPERATOR typed is checked against the stride. One the
        # model declared is by definition a shape the model accepts, DLC's
        # export is built at a height its own backbone stride does not divide,
        # because the runner pads internally, and rejecting that would refuse
        # the model its own input size.
        if not from_model:
            stride_note = _stride_problem(info, size)
            if stride_note:
                return InputPlan("letterbox",
                                 _letterbox_size(info, frame_w, frame_h),
                                 stride_note,
                                 bool(info.input_w and info.input_h))
        if asked != "auto":
            why = "asked for"
        elif follows:
            why = "the model works at native scale, so the window follows"
        else:
            # The window is as big as the frame or bigger. Still a cut, not a
            # scale, it just has nowhere to move, and `crop_at` pads the
            # remainder rather than stretching the animal to fill it.
            why = ("the model works at native scale, so the frame is padded "
                   "to the window rather than resized to it")
        return InputPlan("crop_track", size, why, from_model)

    if asked == "full":
        # An exported graph accepts exactly the shape it was built for. Handing
        # it the raw frame does not run slower, it throws on every frame, and
        # the batch path catches that and returns an empty pose, so the run
        # looks healthy while recording nothing. Refused here, with the reason,
        # rather than discovered as a session of zeroes.
        if engine in FIXED_SHAPE_ENGINES:
            size = _letterbox_size(info, frame_w, frame_h)
            return InputPlan(
                "letterbox", size,
                f"full-frame refused: the {engine} graph only accepts "
                f"{_wh(size)}, which is fixed at export",
                bool(info.input_w and info.input_h))
        return InputPlan("full", (frame_w, frame_h), "asked for", False)

    if asked == "letterbox":
        return InputPlan("letterbox", _letterbox_size(info, frame_w, frame_h),
                         "asked for", bool(info.input_w and info.input_h))

    # auto, and not crop-trained. A network with no baked-in shape is happiest
    # with the frame exactly as it is, nothing scaled means nothing to scale
    # back, and it measured the most accurate of the three DLC paths.
    if not is_sleap and engine in ANY_SIZE_ENGINES:
        return InputPlan("full", (frame_w, frame_h),
                         f"the {engine} engine takes the frame at its own size",
                         False)
    size = _letterbox_size(info, frame_w, frame_h)
    if engine in FIXED_SHAPE_ENGINES:
        why = f"the {engine} graph's input size is fixed at {_wh(size)}"
    elif size and info.native_scale == NATIVE_SCALE:
        # A crop-trained model whose window is already about the size of this
        # box's frame. The letterbox is near-identity here, so it is the right
        # answer, but say WHY, so it does not read as "this is a frame model".
        why = ("this box's frame is already about the model's own "
               f"{_wh(size)} window, so it is fitted rather than cut")
    else:
        why = "the model was trained on whole frames"
    return InputPlan("letterbox", size, why, bool(info.input_w and info.input_h))


def _wh(size: Optional[Tuple[int, int]]) -> str:
    return f"{size[0]}x{size[1]}" if size else "its own size"


def _stride_pad(w: int, h: int, stride: Optional[int]) -> Tuple[int, int]:
    """``(w, h)`` rounded up to what the backbone can divide.

    Not to the model's declared window: the point is to use the frame itself,
    so only the arithmetic the network genuinely requires is added.
    """
    if not stride or stride <= 1:
        return (int(w), int(h))
    step = int(stride)
    return (int(-(-int(w) // step) * step), int(-(-int(h) // step) * step))


def _read(model_path: str) -> ModelInfo:
    try:
        return ModelInfo.read(model_path)
    except Exception as e:                                # never break loading
        logger.debug("input policy: model config unreadable (%s)", e)
        return ModelInfo()


def _crop_refusal(info: ModelInfo, engine: str, is_sleap: bool) -> str:
    """Why this model cannot use a following window, or ``""``."""
    if info.family in SELF_CROPPING:
        return (f"crop-track refused for a {info.family} model; it does its "
                f"own cropping, so a window of ours would crop a crop")
    if not is_sleap and engine in FIXED_SHAPE_ENGINES and not (
            info.input_w and info.input_h):
        return (f"crop-track refused: the {engine} engine has a fixed input "
                f"shape and the manifest does not state it")
    return ""


def _crop_size(info: ModelInfo,
               asked: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """The window to cut: the operator's, else the model's own training size.

    ``min_crop_size`` is deliberately never consulted; it is a floor for
    dynamic cropping, not an input size, and is typically not stride-divisible.
    """
    if asked and asked[0] and asked[1]:
        return (int(asked[0]), int(asked[1]))
    if info.input_w and info.input_h:
        return (int(info.input_w), int(info.input_h))
    if info.crop_size:
        return (int(info.crop_size), int(info.crop_size))
    return None


def _stride_problem(info: ModelInfo, size: Tuple[int, int]) -> str:
    stride = info.max_stride
    if not stride or stride <= 1:
        return ""
    if size[0] % stride or size[1] % stride:
        return (f"window {size[0]}x{size[1]} is not divisible by the model's "
                f"stride {stride}, which breaks the encoder shape arithmetic")
    return ""


def _letterbox_size(info: ModelInfo, frame_w: int,
                    frame_h: int) -> Optional[Tuple[int, int]]:
    """The shape to letterbox onto, or None to let the sink pick one.

    None rather than the frame size, because the sink fits EVERY box to one
    shared shape and only it knows the other boxes.
    """
    if info.input_w and info.input_h:
        return (int(info.input_w), int(info.input_h))
    if info.crop_size:
        return (int(info.crop_size), int(info.crop_size))
    return None


def default_for(tracker_type: str, model_path: str, frame_w: int,
                frame_h: int, *, engine: str = "") -> InputPlan:
    """The plan a project that has never chosen gets. Thin alias for ``auto``."""
    return plan(tracker_type, model_path, frame_w, frame_h, engine=engine,
                asked="auto")


__all__ = ["CROP_MARGIN", "InputPlan", "crop_trained", "default_for", "plan"]
