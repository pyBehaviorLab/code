"""What a pose model declares about itself, read from the file beside its weights.

Both toolkits ship a config next to the checkpoint that already answers most of
what the tracking dialog asks the operator to type, body parts, the family of
model, the input scale it was trained at, how big a crop it wants, how many
channels it expects, and for SLEAP identity models the animal names themselves.

    ModelInfo.read(path)   ->  ModelInfo   (never raises; ``ok`` says whether
                                            anything was actually read)

The rule this module exists to enforce: **the model config is authoritative for
what the model IS; the project is authoritative for what the operator WANTS.**
Anything typed into the dialog that the file already states is a second source
of truth, and the two can disagree, a multi-animal DeepLabCut model loaded as
single-animal, or a grayscale-trained network fed RGB, are both silent today.

Pure and import-light on purpose: no torch, no sleap-nn, no dlclive, no Qt. It
reads YAML/JSON and nothing else, so it is unit-testable on a machine with no
inference stack installed, which is the machine most of this is written on.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

from source.log import get_logger

logger = get_logger()

# SLEAP head names, in the order we probe them. ``single_instance`` is reported
# as ``single`` for continuity with the string the rest of the code already uses.
_SLEAP_HEADS = ("multi_class_topdown", "multi_class_bottomup",
                "single_instance", "centered_instance", "centroid", "bottomup")

# SLEAP families that need a centroid model in front of them: the pose stage
# only sees crops somebody else located.
_NEEDS_CENTROID = ("centered_instance", "multi_class_topdown")

RGB = "rgb"
GRAYSCALE = "grayscale"


@dataclass(frozen=True)
class ModelInfo:
    """Everything a model file states about itself.

    Every field is optional because the two toolkits describe themselves
    differently and older exports omit things. ``None`` means "the file did not
    say", which is a different claim from a default, a caller that needs to
    fall back should do so knowingly rather than inherit a made-up number.
    """

    backend: str = ""                       # "dlc" | "sleap" | ""
    family: str = "unknown"                 # see _SLEAP_HEADS / "multi-animal"
    body_parts: Tuple[str, ...] = ()
    # Connected pairs of body-part NAMES, in the order the model declares them.
    # Names rather than indices, because the two toolkits index differently and
    # a name survives a part list being reordered or filtered.
    skeleton: Tuple[Tuple[str, str], ...] = ()
    identities: Tuple[str, ...] = ()        # SLEAP multi-class only
    native_scale: Optional[float] = None     # the scale the model was trained at
    crop_size: Optional[int] = None          # centered-instance / top-down crop
    # The input the model was trained to take, in pixels. sleap-nn states this
    # as ``preprocessing.max_width`` / ``max_height``, NOT as ``crop_size``,
    # which is the top-down crop, and emphatically not as ``min_crop_size``,
    # which is a floor for dynamic cropping and is typically neither the real
    # size nor divisible by the backbone stride.
    input_w: Optional[int] = None
    input_h: Optional[int] = None
    channels: Optional[str] = None           # RGB | GRAYSCALE
    max_stride: Optional[int] = None
    backbone: Optional[str] = None
    multi_animal: bool = False
    needs_centroid: bool = False
    config_path: str = ""
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """True when a config was found and parsed."""
        return bool(self.config_path)

    @property
    def n_identities(self) -> int:
        """Animal count as the model defines it, the length of its class list.

        SLEAP has no separate count field; the identities ARE the count.
        """
        return len(self.identities)

    def summary(self) -> str:
        """One line for a log or a status label."""
        if not self.ok:
            return "no model config found"
        bits = [f"{self.backend}/{self.family}", f"{len(self.body_parts)} parts"]
        if self.skeleton:
            bits.append(f"{len(self.skeleton)} edges")
        if self.identities:
            bits.append(f"{self.n_identities} identities")
        if self.native_scale is not None:
            bits.append(f"scale {self.native_scale:g}")
        if self.crop_size:
            bits.append(f"crop {self.crop_size}")
        if self.channels:
            bits.append(self.channels)
        return " · ".join(bits)

    # ── reading ──────────────────────────────────────────────────────────

    @classmethod
    def read(cls, model_path: str) -> ModelInfo:
        """Read whichever config sits at ``model_path``.

        Accepts a model directory or the config file itself. Backend is decided
        by which file is present, not by asking the caller, so a mis-set backend
        in the project cannot make us parse the wrong schema.
        """
        if not model_path:
            return cls()
        try:
            sleap_cfg = _first_existing(model_path, ("training_config.yaml",
                                                     "training_config.json"))
            if sleap_cfg:
                return _read_sleap(sleap_cfg)
            dlc_cfg = _first_existing(model_path, ("pose_cfg.yaml",
                                                   "pose_cfg.yml"))
            if dlc_cfg:
                return _read_dlc(dlc_cfg)
            # DLC 3 (PyTorch) writes a different file, and a model exported
            # from it may ship only the export manifest. Neither was read, so
            # every PyTorch DLC model came back family="unknown" with no body
            # parts, and the run silently fell back to the hardcoded
            # ['head', 'center', 'tailbase'], renaming six keypoints to three.
            torch_cfg = _first_existing(model_path, ("pytorch_config.yaml",
                                                     "config.yaml"))
            if torch_cfg and _is_dlc_torch_config(torch_cfg):
                return _read_dlc_torch(torch_cfg)
            export_meta = _first_existing(model_path, ("export_metadata.json",))
            if export_meta:
                info = _read_dlc_export(export_meta)
                if info is not None:
                    return info
        except Exception as e:                     # never break model loading
            logger.debug("model config read %s: %s", model_path, e)
            return cls(warnings=(f"config unreadable: {e}",))
        return cls()


def _first_existing(model_path: str, names: Tuple[str, ...]) -> str:
    """The config path for a model dir, or the file itself if one was given."""
    if os.path.isfile(model_path):
        base = os.path.basename(model_path).lower()
        return model_path if base in [n.lower() for n in names] else ""
    for name in names:
        p = os.path.join(model_path, name)
        if os.path.isfile(p):
            return p
    return ""


def _load(path: str) -> Optional[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith(".json"):
        cfg = json.loads(text)
    else:
        import yaml  # PyYAML is optional
        cfg = yaml.safe_load(text)
    return cfg if isinstance(cfg, dict) else None


# ── SLEAP ────────────────────────────────────────────────────────────────

def _read_sleap(path: str) -> ModelInfo:
    cfg = _load(path)
    if cfg is None:
        return ModelInfo(backend="sleap", config_path=path,
                         warnings=("config is not a mapping",))
    warnings: list = []

    model_cfg = cfg.get("model_config") or {}
    heads = model_cfg.get("head_configs") or cfg.get("head_configs") or {}
    heads = heads if isinstance(heads, dict) else {}

    family = "unknown"
    head_block: Dict[str, Any] = {}
    for key in _SLEAP_HEADS:
        # Presence of the key is the signal, not its contents: a minimal config
        # can carry an empty head block, and testing truthiness would read that
        # as "no head" and report the model as unknown.
        if key in heads and heads[key] is not None:
            family = "single" if key == "single_instance" else key
            block = heads[key]
            head_block = block if isinstance(block, dict) else {}
            break

    # Identities live under the multi-class head, and their COUNT is just the
    # length of the list, SLEAP has no separate n_classes field.
    identities: Tuple[str, ...] = ()
    for holder in ("class_vectors", "class_maps"):
        block = head_block.get(holder) if isinstance(head_block, dict) else None
        if isinstance(block, dict):
            names = block.get("classes")
            if isinstance(names, (list, tuple)) and names:
                identities = tuple(str(n) for n in names)
                break
    if family.startswith("multi_class") and not identities:
        warnings.append("identity model declares no class names "
                        "(they were inferred from the labels at training time)")

    data_cfg = cfg.get("data_config") or {}
    pre = data_cfg.get("preprocessing") or {}
    pre = pre if isinstance(pre, dict) else {}

    channels = None
    if pre.get("ensure_grayscale"):
        channels = GRAYSCALE
    elif pre.get("ensure_rgb"):
        channels = RGB

    backbone = None
    max_stride = None
    bb = model_cfg.get("backbone_config") or {}
    if isinstance(bb, dict):
        for name, block in bb.items():
            if block:
                backbone = str(name)
                if isinstance(block, dict):
                    max_stride = _as_int(block.get("max_stride"))
                break

    parts = _sleap_parts(cfg)
    skeleton = _keep_known_edges(_sleap_edges(cfg, parts), parts, warnings)

    return ModelInfo(
        backend="sleap",
        family=family,
        body_parts=parts,
        skeleton=skeleton,
        identities=identities,
        native_scale=_as_float(pre.get("scale")),
        crop_size=_as_int(pre.get("crop_size")),
        input_w=_as_int(pre.get("max_width")),
        input_h=_as_int(pre.get("max_height")),
        channels=channels,
        max_stride=max_stride,
        backbone=backbone,
        multi_animal=family in ("bottomup", "centered_instance",
                                "multi_class_bottomup", "multi_class_topdown"),
        needs_centroid=family in _NEEDS_CENTROID,
        config_path=path,
        warnings=tuple(warnings),
    )


def _sleap_parts(cfg: Dict[str, Any]) -> Tuple[str, ...]:
    """Node names, wherever this config generation put them."""
    for holder in (cfg.get("data_config") or {}, cfg):
        if not isinstance(holder, dict):
            continue
        for key in ("skeletons", "skeleton"):
            sk = holder.get(key)
            if isinstance(sk, list) and sk and isinstance(sk[0], dict):
                nodes = sk[0].get("nodes")
                if isinstance(nodes, list):
                    out = []
                    for n in nodes:
                        if isinstance(n, dict):
                            out.append(str(n.get("name", "")))
                        else:
                            out.append(str(n))
                    return tuple(p for p in out if p)
    return ()


def _sleap_edges(cfg: Dict[str, Any],
                 parts: Tuple[str, ...]) -> Tuple[Tuple[str, str], ...]:
    """The skeleton's edges, as name pairs.

    ``training_config.yaml`` is the source sleap-nn's own docs name for this:
    an exported runtime "reads the full training skeleton from
    training_config.yaml". The export metadata beside the weights is NOT a
    substitute, ``edge_inds`` there is empty for a single-instance model,
    because grouping edges only exist for bottom-up inference, so a reader that
    trusts it concludes a skeleton-less model.
    """
    for holder in (cfg.get("data_config") or {}, cfg):
        if not isinstance(holder, dict):
            continue
        for key in ("skeletons", "skeleton"):
            sk = holder.get(key)
            if not (isinstance(sk, list) and sk and isinstance(sk[0], dict)):
                continue
            edges = sk[0].get("edges")
            if isinstance(edges, list) and edges:
                out = []
                for e in edges:
                    pair = _edge_pair(e, parts)
                    if pair:
                        out.append(pair)
                if out:
                    return tuple(out)
    return ()


def _edge_pair(edge: Any, parts: Tuple[str, ...]) -> Optional[Tuple[str, str]]:
    """One edge, however this config generation spells it.

    Seen in the wild: ``{source: {name: A}, destination: {name: B}}`` (current
    sleap-nn), ``{source: A, destination: B}`` (names inline), and ``[i, j]``
    index pairs.
    """
    if isinstance(edge, dict):
        src, dst = edge.get("source"), edge.get("destination")
        src = src.get("name") if isinstance(src, dict) else src
        dst = dst.get("name") if isinstance(dst, dict) else dst
        if src is None or dst is None:
            return None
        return (str(src), str(dst))
    if isinstance(edge, (list, tuple)) and len(edge) == 2:
        a, b = edge
        if isinstance(a, int) and isinstance(b, int):
            if 0 <= a < len(parts) and 0 <= b < len(parts):
                return (parts[a], parts[b])
            return None
        return (str(a), str(b))
    return None


def _keep_known_edges(edges: Tuple[Tuple[str, str], ...],
                      parts: Tuple[str, ...],
                      warnings: list) -> Tuple[Tuple[str, str], ...]:
    """Drop edges naming a part the model does not have.

    An edge to a part that is not in the list cannot be drawn and cannot be
    selected, so keeping it only produces a silent no-op later. Dropping it is
    said out loud instead.
    """
    if not edges:
        return ()
    known = set(parts)
    kept, dropped = [], []
    for a, b in edges:
        if a in known and b in known:
            kept.append((a, b))
        else:
            dropped.append(f"{a} -> {b}")
    if dropped:
        warnings.append("skeleton edges name unknown parts and were dropped: "
                        + ", ".join(dropped))
    return tuple(kept)


# ── DeepLabCut ───────────────────────────────────────────────────────────

def _read_dlc(path: str) -> ModelInfo:
    cfg = _load(path)
    if cfg is None:
        return ModelInfo(backend="dlc", config_path=path,
                         warnings=("config is not a mapping",))
    warnings: list = []

    parts = cfg.get("all_joints_names")
    body_parts = tuple(str(p) for p in parts) if isinstance(parts, (list, tuple)) else ()

    # A DeepLabCut config announces multi-animal two ways, and either is enough:
    # the dataset type it was trained with, or part-affinity-field prediction.
    dataset_type = str(cfg.get("dataset_type") or "")
    has_paf = bool(cfg.get("partaffinityfield_predict"))
    multi = ("multi-animal" in dataset_type) or has_paf
    if multi:
        warnings.append(
            "multi-animal model: DLCLive needs single_animal=False and a "
            "top_down_config to return more than one animal")

    identities = ()
    individuals = cfg.get("individuals")
    if isinstance(individuals, (list, tuple)) and individuals:
        identities = tuple(str(i) for i in individuals)

    skeleton = _keep_known_edges(_dlc_skeleton(path), body_parts, warnings)

    return ModelInfo(
        backend="dlc",
        family="multi-animal" if multi else "single",
        body_parts=body_parts,
        skeleton=skeleton,
        identities=identities,
        # global_scale is a TRAINING field, DLCLive does not read it, so it is
        # reported for information and never used as an inference scale.
        native_scale=None,
        crop_size=_as_int(cfg.get("crop_size")),
        channels=RGB,                      # DLC models are RGB
        max_stride=None,
        backbone=str(cfg.get("net_type")) if cfg.get("net_type") else None,
        multi_animal=multi,
        needs_centroid=False,
        config_path=path,
        warnings=tuple(warnings),
    )


def _is_dlc_torch_config(path: str) -> bool:
    """Whether ``path`` is a DLC 3 pytorch config rather than some other yaml.

    A DLC PROJECT root also holds a ``config.yaml``, and it is a different
    schema entirely. Matching on the two keys only the pytorch config has
    keeps a project directory from being read as a model.
    """
    try:
        cfg = _load(path)
    except Exception:
        return False
    return bool(cfg) and "model" in cfg and (
        "net_type" in cfg or "config_version" in cfg or "metadata" in cfg)


def _read_dlc_torch(path: str) -> ModelInfo:
    """A DeepLabCut 3 (PyTorch) model, from its ``pytorch_config.yaml``.

    Body parts live under ``metadata.bodyparts`` here, not
    ``all_joints_names``; the backbone's ``output_stride`` is the stride the
    input has to divide by, and ``method`` says bottom-up vs top-down. An
    ``export_metadata.json`` beside it, when present, is preferred for the
    input size because that shape is baked into the exported engine.
    """
    cfg = _load(path) or {}
    warnings: list = []
    meta = cfg.get("metadata") or {}
    parts = meta.get("bodyparts")
    body_parts = (tuple(str(p) for p in parts)
                  if isinstance(parts, (list, tuple)) else ())
    if not body_parts:
        warnings.append(
            "this PyTorch DLC config declares no bodyparts under 'metadata'")

    individuals = meta.get("individuals")
    identities = (tuple(str(i) for i in individuals)
                  if isinstance(individuals, (list, tuple)) and individuals
                  else ())
    multi = len(identities) > 1 or bool(meta.get("with_identity"))

    method = str(cfg.get("method") or "").lower()
    family = "multi-animal" if multi else "single"
    if method in ("td", "topdown"):
        family = "topdown"
        warnings.append(
            "top-down model: DeepLabCut-Live needs a top_down_config to run "
            "it, and will refuse without one")

    backbone = (cfg.get("model") or {}).get("backbone") or {}
    colormode = str(((cfg.get("data") or {}).get("colormode") or "")).lower()

    info = ModelInfo(
        backend="dlc",
        family=family,
        body_parts=body_parts,
        skeleton=_keep_known_edges(_dlc_skeleton(path), body_parts, warnings),
        identities=identities,
        native_scale=None,
        crop_size=None,
        channels=GRAYSCALE if colormode in ("grey", "gray", "grayscale")
        else RGB,
        max_stride=_as_int(backbone.get("output_stride")),
        backbone=str(cfg.get("net_type")) if cfg.get("net_type") else None,
        multi_animal=multi,
        needs_centroid=family == "topdown",
        config_path=path,
        warnings=tuple(warnings),
    )
    return _with_export_shape(info, os.path.dirname(os.path.abspath(path)))


def _read_dlc_export(path: str) -> Optional[ModelInfo]:
    """A DLC ONNX/TensorRT export described only by its manifest.

    The manifest names its keypoints under ``bodyparts``; SLEAP's manifest
    uses ``node_names``, so the key is also what tells the two apart when a
    folder carries nothing else.
    """
    try:
        meta = _load(path) or {}
    except Exception:
        return None
    parts = meta.get("bodyparts")
    if not isinstance(parts, (list, tuple)) or not parts:
        return None
    info = ModelInfo(
        backend="dlc",
        family="single",
        body_parts=tuple(str(p) for p in parts),
        channels=RGB,
        backbone=None,
        multi_animal=False,
        needs_centroid=False,
        config_path=path,
        warnings=("read from the export manifest: this folder carries no "
                  "training config, so only what was exported is known",),
    )
    return _with_export_shape(info, os.path.dirname(os.path.abspath(path)))


def _with_export_shape(info: ModelInfo, folder: str) -> ModelInfo:
    """Fill in the input size an export bakes in, when a manifest states one.

    DLC's manifest writes ``input`` as ``[N, C, H, W]`` (``layout: NCHW``);
    reading it as width-then-height would letterbox every frame to the
    transpose of the shape the engine accepts.
    """
    meta_path = os.path.join(folder, "export_metadata.json")
    if not os.path.isfile(meta_path):
        return info
    try:
        meta = _load(meta_path) or {}
    except Exception:
        return info
    shape = meta.get("input")
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        return info
    layout = str(meta.get("layout") or "NCHW").upper()
    _n, a, b, c = [_as_int(v) for v in shape]
    if layout == "NCHW":
        height, width = b, c
    elif layout == "NHWC":
        height, width = a, b
    else:
        return info
    if not (width and height):
        return info
    return replace(info, input_w=width, input_h=height)


def _dlc_skeleton(pose_cfg_path: str) -> Tuple[Tuple[str, str], ...]:
    """DeepLabCut's drawing skeleton, from the PROJECT config.

    ``pose_cfg.yaml`` sits under ``dlc-models/iteration-N/<task>/train/`` and
    does not carry one; the project ``config.yaml`` at the root does, as a list
    of ``[partA, partB]`` pairs. So this walks up looking for it, and an
    exported model folder shipped without the project simply has no skeleton.

    ``partaffinityfield_graph`` is deliberately NOT used. It is the training
    graph for part-affinity fields and is routinely the complete graph, 351
    edges for a 27-part model, so drawing it would connect every part to every
    other part and assert anatomy that does not exist.
    """
    directory = os.path.dirname(os.path.abspath(pose_cfg_path))
    for _ in range(5):                       # train/ -> task/ -> iteration/ -> project
        candidate = os.path.join(directory, "config.yaml")
        if os.path.isfile(candidate):
            try:
                cfg = _load(candidate) or {}
            except Exception as e:
                logger.debug("DLC project config unreadable %s: %s",
                             candidate, e)
                return ()
            raw = cfg.get("skeleton")
            if isinstance(raw, (list, tuple)):
                return tuple((str(edge[0]), str(edge[1])) for edge in raw
                             if isinstance(edge, (list, tuple))
                             and len(edge) == 2)
            return ()
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    return ()


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, (list, tuple)):     # crop given as [h, w]
            v = v[0]
        return int(v)
    except (TypeError, ValueError, IndexError):
        return None
