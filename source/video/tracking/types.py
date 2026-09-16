"""Canonical tracking data structures, the one format every tracker emits.

PoseSnapshot     Per-frame body-part positions, immutable, tracker-agnostic.

Pure-types layer (no I/O, no Qt) shared by runtime, analysis, and gui.
Body-parts dict shape by backend:

Blob backend -> {"center": (cx, cy, 1.0)}
DLC backend  -> {"nose": (x,y,conf), "head": (...), "tail_base": (...)}
SLEAP        -> {"head": (...), "spine1": (...), ...} (model-dependent)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class InstanceSnapshot:
    """Immutable per-frame pose of ONE animal.

    Fields:
        body_parts       {name: (x_px, y_px, confidence)} for this instance.
        track_id         stable identity across frames (multi-animal), or None
                         for single-animal / untracked.
        score            instance-level confidence (SLEAP), 1.0 otherwise.
        confidence_threshold   parts below this are treated as missing.
    """

    body_parts: Dict[str, Tuple[float, float, float]]
    track_id: Optional[str] = None
    score: float = 1.0
    confidence_threshold: float = 0.5

    @property
    def centroid(self) -> Optional[Tuple[float, float]]:
        sx, sy, n = 0.0, 0.0, 0
        for x, y, conf in self.body_parts.values():
            if conf >= self.confidence_threshold:
                sx += x
                sy += y
                n += 1
        if n == 0:
            return None
        return (sx / n, sy / n)

    @property
    def primary_position(self) -> Optional[Tuple[float, float]]:
        if len(self.body_parts) == 1:
            (x, y, conf), = self.body_parts.values()
            return (x, y) if conf >= self.confidence_threshold else None
        return self.centroid

    def get_part(self, name: str) -> Optional[Tuple[float, float]]:
        if name == "centroid":
            return self.centroid
        if name == "any":
            for x, y, conf in self.body_parts.values():
                if conf >= self.confidence_threshold:
                    return (x, y)
            return None
        bp = self.body_parts.get(name)
        if bp is None:
            return None
        x, y, conf = bp
        return (x, y) if conf >= self.confidence_threshold else None

    @property
    def confident_parts(self) -> Dict[str, Tuple[float, float]]:
        return {
            name: (x, y)
            for name, (x, y, conf) in self.body_parts.items()
            if conf >= self.confidence_threshold
        }

    @property
    def part_names(self) -> List[str]:
        return list(self.body_parts.keys())

    def to_dict(self) -> dict:
        d = {"body_parts": {name: [x, y, conf]
                            for name, (x, y, conf) in self.body_parts.items()}}
        if self.track_id is not None:
            d["track_id"] = self.track_id
        if self.score != 1.0:
            d["score"] = self.score
        return d

    @classmethod
    def from_dict(cls, d: dict, confidence_threshold: float = 0.5) -> "InstanceSnapshot":
        bp = {name: (vals[0], vals[1], vals[2])
              for name, vals in (d.get("body_parts", {}) or {}).items()}
        return cls(body_parts=bp, track_id=d.get("track_id"),
                   score=float(d.get("score", 1.0)),
                   confidence_threshold=confidence_threshold)


@dataclass(frozen=True)
class PoseSnapshot:
    """Immutable per-frame pose from any tracker, a list of animal instances.

    Single-animal is the 1-instance case (the default everywhere): the
    ``body_parts`` / ``centroid`` / ``get_part`` accessors delegate to the
    ``primary`` instance so every existing single-animal caller is untouched.
    Multi-animal (SLEAP top-down / bottom-up) fills more ``instances``, each
    carrying its ``track_id``.

    Fields:
        capture_fw_ms    pyControl framework time (ms) at frame capture.
        instances        per-animal poses ([] none · [x] single · [a,b] multi).
        confidence_threshold   parts below this are treated as missing.
        inference_fw_ms  pyControl FW time (ms) when inference finished; None
                         for a non-inference source (e.g. blob fallback).
    """

    capture_fw_ms: float
    instances: Tuple[InstanceSnapshot, ...] = ()
    confidence_threshold: float = 0.5
    inference_fw_ms: Optional[float] = None

    @classmethod
    def single(cls, capture_fw_ms: float,
               body_parts: Dict[str, Tuple[float, float, float]],
               confidence_threshold: float = 0.5,
               inference_fw_ms: Optional[float] = None,
               track_id: Optional[str] = None,
               score: float = 1.0) -> "PoseSnapshot":
        """Build a single-animal snapshot (the common case)."""
        inst = InstanceSnapshot(body_parts=body_parts, track_id=track_id,
                                score=score,
                                confidence_threshold=confidence_threshold)
        return cls(capture_fw_ms=capture_fw_ms, instances=(inst,),
                   confidence_threshold=confidence_threshold,
                   inference_fw_ms=inference_fw_ms)

    @property
    def primary(self) -> Optional[InstanceSnapshot]:
        """The focal instance, instance 0 (single-animal = the only one)."""
        return self.instances[0] if self.instances else None

    @property
    def n_instances(self) -> int:
        return len(self.instances)

    # ── single-animal back-compat accessors (delegate to primary) ─────────

    @property
    def body_parts(self) -> Dict[str, Tuple[float, float, float]]:
        p = self.primary
        return p.body_parts if p else {}

    @property
    def centroid(self) -> Optional[Tuple[float, float]]:
        p = self.primary
        return p.centroid if p else None

    @property
    def primary_position(self) -> Optional[Tuple[float, float]]:
        p = self.primary
        return p.primary_position if p else None

    def get_part(self, name: str) -> Optional[Tuple[float, float]]:
        p = self.primary
        return p.get_part(name) if p else None

    @property
    def confident_parts(self) -> Dict[str, Tuple[float, float]]:
        p = self.primary
        return p.confident_parts if p else {}

    @property
    def part_names(self) -> List[str]:
        p = self.primary
        return p.part_names if p else []

    @property
    def inference_latency_ms(self) -> Optional[float]:
        if self.inference_fw_ms is None:
            return None
        return self.inference_fw_ms - self.capture_fw_ms

    def to_dict(self) -> dict:
        return {
            "capture_fw_ms": self.capture_fw_ms,
            "instances": [i.to_dict() for i in self.instances],
            "confidence_threshold": self.confidence_threshold,
            "inference_fw_ms": self.inference_fw_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PoseSnapshot":
        conf = d.get("confidence_threshold", 0.5)
        if "instances" in d:
            instances = tuple(InstanceSnapshot.from_dict(i, conf)
                              for i in (d.get("instances") or []))
        else:
            # Back-compat: old single-instance ``{body_parts: …}`` shape.
            bp = d.get("body_parts") or {}
            instances = ((InstanceSnapshot.from_dict({"body_parts": bp}, conf),)
                         if bp else ())
        return cls(
            capture_fw_ms=d.get("capture_fw_ms", 0.0),
            instances=instances,
            confidence_threshold=conf,
            inference_fw_ms=d.get("inference_fw_ms"),
        )


__all__ = ["InstanceSnapshot", "PoseSnapshot"]
