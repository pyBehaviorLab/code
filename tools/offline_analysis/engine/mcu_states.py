"""The task state each video frame was recorded under.

The rig writes two logs per session and, until now, they never met:

* the **video log** (``<stem>_video_data.txt``): one row per frame, carrying
  the acquisition time and the pose;
* the **MCU log** (``<stem>.tsv``): the pyControl state machine's own record,
  one row per state entry and event.

Every analysis worth reporting is per state: time in a zone *during
``choice_state``* is a number, time in a zone across the whole hour is not. The
analyser had a ``stage`` column ready for this and filled it from the video
log, whose ``stage`` cell is empty on this rig, so every frame of every
session came back stateless, every metric was computed over the whole
recording, and nothing said a word. This is the join that was missing.

**Aligning the two clocks.** By elapsed time, not by wall clock. Both logs
start counting at zero when the session starts, and on the recording this was
built against they agree to 0.28 s over a full hour (MCU 0 → 3600.002 s; video
0 → 3600.281 s). Their *stamped* start times do not agree at all, the video
log records 11:04:33.661 and the MCU log 09:04:33.664, the same instant in two
timezones, so trusting the strings would put the entire session two hours out
and silently label every frame with the wrong state.

When the recording carries a real ``mcu_ts_ms`` per frame, that is used
instead: it is the rig's own answer and needs no alignment at all.
"""

from __future__ import annotations

import bisect
import logging
import os
import re

logger = logging.getLogger(__name__)

#: The ``type`` column value marking a state entry in a pyControl session log.
STATE_ROW = "state"

#: A state timeline: ``(milliseconds since session start, state name)``, in
#: time order. A state is in force from its own entry until the next one.
Timeline = list[tuple[float, str]]


# ── finding the log ──────────────────────────────────────────────────────

def find_log(txt_path: str = "", video_path: str = "", stem: str = "") -> str:
    """The MCU session log for this recording, or ``""``.

    The rig puts the MCU log in the date folder and the video log one level
    down in ``video/``, so looking only beside the video log finds nothing.
    Two cheap steps first, beside the recording, then up to two folders above
    it and their ``temp``/``mcu`` siblings, and only if both come back empty,
    a sweep of the project the recording belongs to, for older sessions whose
    logs were filed somewhere else entirely.

    The sweep stops at the project and never goes above it. A search that
    climbed further would eventually match a *different* project's session by
    name, and a wrong state timeline is worse than none: nothing downstream
    can tell that it is wrong.
    """
    stem = stem or _stem_of(txt_path or video_path)
    if not stem:
        return ""
    seen = set()
    roots = []
    for base in (txt_path, video_path):
        if not base:
            continue
        folder = os.path.dirname(os.path.abspath(base))
        for _ in range(3):                      # here, up one, up two
            if folder in seen:
                break
            seen.add(folder)
            for candidate in _candidates(folder, stem):
                if os.path.isfile(candidate):
                    return candidate
            parent = os.path.dirname(folder)
            if parent == folder:
                break
            folder = parent
        roots.append(_project_root(base))
    # Last resort: sweep the PROJECT for a log by this name. Climbing parents
    # finds the rig's own layout, date folder, `video/` beneath it, and
    # misses anything filed differently, which older sessions are. A log moved
    # into a per-animal or per-cohort folder is still unambiguously this
    # recording's, because the stem carries box, date and time.
    #
    # Bounded by the project, never by a folder count: "three levels up" from
    # one recording is a different project's data, and this would then answer
    # with another session's states rather than none. Wrong states are worse
    # than no states, because nothing downstream can tell they are wrong.
    for root in roots:
        found = _search_root(root, stem) if root else ""
        if found:
            logger.info("found the MCU log for %s away from the recording, "
                        "at %s", stem, os.path.dirname(found))
            return found
    return ""


#: A folder named like a recording date, ``270526`` or ``2026-05-27``. The rig
#: files one of these per session day, so the folder ABOVE it is the project,
#: and that is as far as a search for one session's log may reach.
_DATE_DIR = re.compile(r"^(\d{6}|\d{4}-\d{2}-\d{2})$")

#: How many directories the sweep will open before giving up. A project holds
#: a few thousand; this is the difference between a slow search and a hung
#: analysis on a network share.
SEARCH_LIMIT = 4000


def _project_root(path: str) -> str:
    """The project folder ``path`` sits in, the parent of its date folder.

    ``…/data/aCG_RL/270526/video/x_video_data.txt`` → ``…/data/aCG_RL``.
    ``""`` when there is no date folder above it, which means the layout is
    not the rig's and there is nothing safe to sweep.
    """
    folder = os.path.dirname(os.path.abspath(path))
    for _ in range(6):
        parent = os.path.dirname(folder)
        if parent == folder:
            return ""
        if _DATE_DIR.match(os.path.basename(folder)):
            return parent
        folder = parent
    return ""


#: ``{project root: {stem: path}}``. Built once per root and reused.
#:
#: A cohort run opens thirty recordings and a corpus sweep opens a thousand,
#: and every one of them that needs this would otherwise walk the same project
#: tree from scratch. Measured: without the cache, indexing one project of
#: 1,001 recordings did not finish in ten minutes; with it, the walk happens
#: once and every later lookup is a dict hit.
_INDEX: dict[str, dict[str, str]] = {}


def forget_indexed_logs() -> None:
    """Drop the cached project indexes.

    For the case the cache is wrong about: a log written, moved or renamed
    while the analyser is open. Nothing calls this on a timer, an index that
    silently refreshed would make "why did it find it this time" unanswerable.
    """
    _INDEX.clear()


def _index_root(root: str) -> dict[str, str]:
    """``{stem: path}`` for every ``.tsv`` under ``root``."""
    known = _INDEX.get(root)
    if known is not None:
        return known
    found: dict[str, str] = {}
    looked = 0
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        looked += 1
        if looked > SEARCH_LIMIT:
            logger.warning(
                "stopped indexing %s after %d folders, MCU logs deeper than "
                "that will not be found by name", root, SEARCH_LIMIT)
            break
        for name in files:
            if name.endswith(".tsv"):
                # First wins: the shallowest copy of a duplicated log is the
                # one filed where the rig put it.
                found.setdefault(name[:-4], os.path.join(base, name))
    logger.info("indexed %d MCU logs under %s", len(found), root)
    _INDEX[root] = found
    return found


def _search_root(root: str, stem: str) -> str:
    """``<stem>.tsv`` anywhere under ``root``, or ``""``."""
    return _index_root(root).get(stem, "")


#: Sibling folders the MCU log turns up in besides the date folder itself.
#: A session that crashed and was recovered leaves its log in ``temp``: 62 of
#: them across this corpus, every one a complete log, and every one invisible
#: to a search that only climbs parents.
SIBLING_DIRS = ("temp", "mcu")


def _candidates(folder: str, stem: str):
    """Where ``<stem>.tsv`` could be, given this folder."""
    yield os.path.join(folder, f"{stem}.tsv")
    for sub in SIBLING_DIRS:
        yield os.path.join(folder, sub, f"{stem}.tsv")


def _stem_of(path: str) -> str:
    """``…/282-Box1-…_video_data.txt`` → ``282-Box1-…``."""
    if not path:
        return ""
    name = os.path.splitext(os.path.basename(path))[0]
    for suffix in ("_video_data", "_retracked_video_data", "_retracked",
                   "_corrected_video_data", "_annotated"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


# ── reading it ───────────────────────────────────────────────────────────

def read_timeline(log_path: str) -> Timeline:
    """Every state entry in ``log_path``, as ``(ms, name)`` in time order.

    Parsed directly rather than through ``tools.data_import``: that reader
    wants pandas and builds the whole session, events, prints, variables,
    to answer a question that is four columns wide, and the analyser has to
    stay light enough to open thirty recordings at once.
    """
    out: Timeline = []
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line or line[0] == "#":
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4 or parts[1] != STATE_ROW:
                    continue
                name = parts[3].strip()
                if not name:
                    continue
                try:
                    seconds = float(parts[0])
                except ValueError:              # the header row
                    continue
                out.append((seconds * 1000.0, name))
    except OSError as e:
        logger.warning("could not read the MCU log %s: %s", log_path, e)
        return []
    out.sort(key=lambda row: row[0])
    return out


#: What the recorder writes in the state column when nothing changed on this
#: frame. Not caught by ``is_na``, which knows "na"/"none" and not this.
NO_CHANGE = ("-", "--", "")


def state_of_cell(cell) -> str:
    """The state a recording's own ``stage`` cell names, or ``""``.

    Two things this has to survive, both found by reading the rig's recordings
    rather than its writer:

    * ``"-"`` means "no change on this frame", not a state called ``-``;
    * a frame whose interval spanned a transition is written pipe-joined,
      ``poke4_reward|inter_trial_interval``. Taken literally that becomes a
      state of its own, so a session grew states that the task never had. The
      LAST name is the one in force when the frame ended, which is what the
      next frame will carry anyway.
    """
    from tools.offline_analysis import video_data_schema as vds

    text = str(cell or "").strip()
    if not text or text in NO_CHANGE or vds.is_na(text):
        return ""
    return text.rsplit("|", 1)[-1].strip()


def state_at(timeline: Timeline, ms: float) -> str:
    """The state in force at ``ms``, or ``""`` before the first entry.

    A step function, which is what a state machine is: a state holds from its
    own entry until the next. Bisected rather than scanned, seventy thousand
    frames against fifteen hundred transitions is a hundred million
    comparisons done the naive way, and it runs on every recording.
    """
    if not timeline or ms != ms:                # NaN never matches a state
        return ""
    i = bisect.bisect_right(timeline, (float(ms), "￿"))
    return timeline[i - 1][1] if i else ""


# ── the thing the pipeline asks for ──────────────────────────────────────

def own_by_frame(txt_path: str, header=None) -> tuple[dict[int, str], int, int]:
    """``({frame: state}, frames stamped, frames seen)`` from the recording's
    own ``stage`` column.

    Carried forward, because this column is written two ways. One recorder
    stamps every frame; two others stamp only the frames where the state
    CHANGED and write ``-`` in between, 824 stamped rows in a 75,875-frame
    session. Read literally that is a recording that was in a state for 1% of
    itself. A state machine holds its state until the next transition, so the
    carry is not an assumption about the data; it is what the data means.
    """
    from tools.offline_analysis import video_data_schema as vds

    out: dict[int, str] = {}
    held = ""
    stamped = frames = 0
    for row in vds.iter_rows(txt_path, header):
        frame = vds.parse_int(row.get("frame_number"), -1)
        if frame < 0:
            continue
        frames += 1
        name = state_of_cell(row.get("stage"))
        if name:
            held = name
            stamped += 1
        if held:
            out[frame] = held
    return out, stamped, frames


def merge_by_frame(txt_path: str, *, video_path: str = "", stem: str = "",
                   log_path: str = "", header=None) -> dict[int, str]:
    """The state each frame was in, from the recording AND the MCU log.

    Both sources are consulted whenever both exist, rather than one standing in
    for the other. The recording's own column used to win outright, and the
    ``.tsv`` beside it was never opened, so a session whose column covered
    part of the recording reported states for that part and nothing for the
    rest, with a complete log sitting next to it the whole time.

    Where they overlap they are compared and the agreement reported. They are
    two independent records of the same state machine: if they disagree, one of
    the clocks is wrong, and that is worth knowing before any per-state number
    is believed.
    """
    own, stamped, frames = own_by_frame(txt_path, header)
    joined = by_frame(txt_path, video_path=video_path, stem=stem,
                      log_path=log_path, header=header)
    label = stem or os.path.basename(txt_path)

    if own and joined:
        shared = own.keys() & joined.keys()
        agree = sum(1 for f in shared if own[f] == joined[f])
        pct = 100.0 * agree / len(shared) if shared else 0.0
        say = logger.info if pct >= 95.0 else logger.warning
        say("%s: the recording's own stage column and the MCU log agree on "
            "%.1f%% of the %d frames both describe", label, pct, len(shared))
        if pct < 95.0:
            say("A disagreement this size means the two clocks are not "
                "aligned. The recording's own column is used where it has an "
                "answer, so the mismatch shows up only on the frames the log "
                "had to fill.")

    if not own and not joined:
        return {}
    # The recording's own column wins where it has an answer: it is stamped
    # against the frame itself and needs no alignment at all. The log fills
    # everything else, which on this corpus is most of it.
    out = dict(joined)
    out.update(own)
    from_log = len(out) - len(own)
    logger.info(
        "%s: %d of %d frames carry a state, %d from the recording "
        "(%d stamped, the rest carried forward), %d joined from the MCU log",
        label, len(out), frames, len(own), stamped, from_log)
    return out


def by_frame(txt_path: str, *, video_path: str = "", stem: str = "",
             log_path: str = "", header=None) -> dict[int, str]:
    """``{frame_number: state}`` for a recording, or ``{}``.

    Empty is a legitimate answer, a rig with no state machine, or a log that
    is not beside the recording, and the caller treats it as "no phases",
    which is what such a session has.
    """
    log_path = log_path or find_log(txt_path, video_path, stem)
    if not log_path:
        logger.info("no MCU log found for %s, frames will carry no state",
                    stem or txt_path)
        return {}
    timeline = read_timeline(log_path)
    if not timeline:
        logger.warning("the MCU log %s holds no state entries", log_path)
        return {}

    from tools.offline_analysis import video_data_schema as vds

    out: dict[int, str] = {}
    on_mcu_clock = 0
    frames = 0
    for row in vds.iter_rows(txt_path, header):
        frame = vds.parse_int(row.get("frame_number"), -1)
        if frame < 0:
            continue
        frames += 1
        # The rig's own per-frame MCU time when it recorded one; otherwise the
        # frame's acquisition time, which starts from the same zero.
        ms = vds.parse_elapsed(row.get("mcu_ts_ms"))
        if ms == ms:
            on_mcu_clock += 1
        else:
            ms = vds.parse_elapsed(row.get("frame_ts_ms"))
        state = state_at(timeline, ms)
        if state:
            out[frame] = state

    names = sorted({s for s in out.values()})
    # Counted, not flagged: a single row carrying an mcu_ts_ms would otherwise
    # have this claim the whole session was aligned by the rig's own clock
    # when 75,667 of its frames were aligned by elapsed time.
    how = (f"the recorded mcu_ts_ms on {on_mcu_clock} of {frames} frames"
           if on_mcu_clock else "elapsed time")
    logger.info(
        "MCU states joined for %s: %d transitions over %.0f s → %d of %d "
        "frames carry one of %d states (%s), aligned by %s",
        stem or os.path.basename(txt_path), len(timeline),
        timeline[-1][0] / 1000.0, len(out), frames, len(names),
        ", ".join(names[:6]) + ("…" if len(names) > 6 else ""), how)
    if frames and on_mcu_clock < frames // 2:
        # Measured on the recording this was built against: where both clocks
        # are present they run ~0.43 s apart, which is longer than a
        # `choice_state` lasts. On the recordings that HAVE mcu_ts_ms none of
        # that matters; where it is missing, a frame either side of a
        # transition can land in the neighbouring state.
        logger.warning(
            "%s carries no per-frame MCU time, so states are aligned by "
            "elapsed time. The two clocks differ by a few hundred ms where "
            "both exist, so frames close to a transition may be attributed to "
            "the neighbouring state.", stem or os.path.basename(txt_path))
    _warn_unsampled(timeline, names, log_path)
    return out


def ts_by_frame(txt_path: str, header=None) -> dict[int, str]:
    """``{frame_number: mcu_ts_ms}`` exactly as the recording stated it.

    The rig stamps the MCU's clock against each frame it can, and that is the
    only exact alignment between the behaviour and the task state. A retrack
    that blanks the column forces everything downstream back onto elapsed
    time, which is a few hundred milliseconds out, longer than some states
    last. Kept as the recorded TEXT rather than a parsed float so a retracked
    file is byte-comparable with its source on this column.
    """
    from tools.offline_analysis import video_data_schema as vds

    out: dict[int, str] = {}
    for row in vds.iter_rows(txt_path, header):
        frame = vds.parse_int(row.get("frame_number"), -1)
        cell = row.get("mcu_ts_ms")
        if frame >= 0 and cell and not vds.is_na(cell):
            out[frame] = str(cell)
    return out


def _warn_unsampled(timeline: Timeline, seen: list[str], log_path: str) -> None:
    """Name the states the camera never caught.

    A state shorter than the frame interval can be entered hundreds of times
    and still contain no frame, ``poke4_reward`` lasts 2 ms and the camera
    samples every 50, so it is absent from the results for a good reason. Said
    out loud, because "my reward state has no data" otherwise looks like this
    join dropped it.
    """
    missed = sorted({name for _ms, name in timeline} - set(seen))
    if missed:
        logger.warning(
            "these states were entered but never coincided with a frame: %s. "
            "They are shorter than the interval between frames, so no video "
            "measurement of them is possible (%s).",
            ", ".join(missed), os.path.basename(log_path))


__all__ = ["STATE_ROW", "Timeline", "by_frame", "find_log", "merge_by_frame",
           "own_by_frame", "read_timeline", "state_at", "state_of_cell",
           "ts_by_frame"]
