"""Where a run's results end up, under names you can read.

Everything an analysis produces already exists somewhere, the retracked
poses, the figures, the workbook. The problem was *where*: inside a hidden
``<video>.pbanalysis`` folder, under a name like ``06360b98_video_data.txt``.
That layout is right for a cache; it is what makes a re-run cheap and lets
two retracks of one recording coexist without either overwriting the other,
and wrong for the thing you actually take away.

So this does not replace the cache. It copies out of it, once, at the end of a
run, into a folder the operator chose, with every file named after the
recording it came from:

    <out>/282-Box1-2026-05-27-110433_retracked_video_data.txt
    <out>/282-Box1-2026-05-27-110433_track.png

Nothing is deleted and nothing is moved; publishing twice overwrites the
published copy and leaves the cache alone.
"""

from __future__ import annotations

import logging
import os
import shutil

logger = logging.getLogger(__name__)

#: What a published file is called, given the recording's stem. Keyed by the
#: kind of artifact so the names are decided in one place rather than at each
#: call site, which is how ``_retracked`` and ``_retracked_video_data`` came
#: to mean the same file in different parts of the code.
NAMES = {
    "poses": "{stem}_retracked_video_data.txt",
    "corrected": "{stem}_corrected_video_data.txt",
    "video": "{stem}_annotated.mp4",
}


#: The folder a run writes into when the operator has not chosen one. Beside
#: the recording, so results sit with the data they came from.
DEFAULT_DIRNAME = "retracked"


def default_dir(bundle) -> str:
    """Where this recording's results go when nothing else was asked for.

    ONE folder per source folder, ``<where the recording lives>/retracked``,
    not one per session. A date folder holding twelve boxes gets twelve
    files in a single ``retracked`` directory beside them, which is what makes
    a day's results something you can open; a folder per recording is twelve
    folders to click through.
    """
    base = bundle.txt_path or bundle.video_path
    if not base:
        return ""
    return os.path.join(os.path.dirname(os.path.abspath(base)),
                        DEFAULT_DIRNAME)


def publish(bundle, out_dir: str = "", *, plots: list[str] | None = None,
            video: str = "") -> dict[str, str]:
    """Copy this recording's artifacts into ``out_dir``.

    Returns ``{kind: written path}`` for whatever was there to publish. A
    missing artifact is not an error, a run that only measured has no poses
    to publish, and saying so by omission is the honest answer.
    """
    out_dir = out_dir or default_dir(bundle)
    if not out_dir:
        return {}
    stem = bundle.stem or "session"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        logger.error("cannot write results to %s: %s", out_dir, e)
        return {}

    written: dict[str, str] = {}
    for kind, src in _sources(bundle, video).items():
        if not src or not os.path.exists(src):
            continue
        dest = os.path.join(out_dir, NAMES[kind].format(stem=stem))
        if os.path.abspath(src) == os.path.abspath(dest):
            written[kind] = dest              # already where it belongs
            continue
        try:
            shutil.copy2(src, dest)
            written[kind] = dest
        except OSError as e:
            logger.error("could not publish the %s for %s: %s", kind, stem, e)

    for i, plot in enumerate(plots or []):
        if not plot or not os.path.exists(plot):
            continue
        # The figure keeps its own descriptive tail (`track`, `heatmap`) so a
        # folder of forty recordings sorts by animal and then by panel.
        tail = os.path.basename(plot)
        base, ext = os.path.splitext(tail)
        suffix = base.split("_", 1)[1] if "_" in base else f"figure{i + 1}"
        dest = os.path.join(out_dir, f"{stem}_{suffix}{ext}")
        try:
            shutil.copy2(plot, dest)
            written[f"plot:{suffix}"] = dest
        except OSError as e:
            logger.error("could not publish a figure for %s: %s", stem, e)

    if written:
        logger.info("%s: published %d file(s) to %s",
                    stem, len(written), out_dir)
    return written


def _sources(bundle, video: str) -> dict[str, str]:
    """The cache paths this recording could publish, by kind."""
    from tools.offline_analysis.engine.session_bundle import STAGE_CORRECT, STAGE_TRACK

    def artifact(stage):
        state = bundle.stages.get(stage)
        return getattr(state, "artifact", "") if state else ""

    return {"poses": artifact(STAGE_TRACK),
            "corrected": artifact(STAGE_CORRECT),
            "video": video}


__all__ = ["DEFAULT_DIRNAME", "NAMES", "default_dir", "publish"]
