"""Words that were rejected, each on its own occasion, stay rejected.

The objection is not to the words as words. It is that each of them stands in
for a specific technical claim without making it: "closed-loop" names a shape
of experiment rather than saying that position reaches the task while it runs,
and "timebase" names nothing at all. The instruction when one of these is the
natural word is to rewrite the sentence around the mechanism, not to reach for
a synonym.

This exists because the list was already given and the words came back anyway:
the landing page opened with "makes closed-loop designs possible" within an
hour of the list being restated. A rule that is only enforced by remembering
is a rule that lasts until the next page.

``ALLOWED`` is for the cases where the word is not ours to change. A cited
paper's title is the obvious one; rewriting it would misquote the reference.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: word -> what to write instead. The replacement is a direction, not a
#: substitution: say what actually happens.
BANNED = {
    r"timebase":
        "name the actual time: the controller's own timestamps, or aligned to "
        "the controller record",
    r"closed[- ]loop":
        "say what reaches what: position is sent to the task while it runs",
    r"\bprovenance\b":
        "say what is recorded: the task file and configuration the session ran under",
    r"\bfootage\b": "video, or the recording",
    r"\bheadroom\b": "state the margin: how many frames or milliseconds are spare",
    r"\bsimultaneous(ly)?\b":
        "say what shares a clock, or what happens in the same frame",
}

#: Where a banned word is quoted rather than used.
ALLOWED = {
    # A cited paper's own title. Changing it would misquote the reference.
    ("docs/architecture.md", r"closed[- ]loop"),
}

#: PROSE ONLY, and that is the point. The note bans these words as an
#: organising frame, not as technical vocabulary, and three places are
#: deliberately out of scope:
#:
#: * ``experiments/projects/*/source/`` holds djb2 content-addressed
#:   snapshots of the task and configuration a session actually ran under.
#:   Editing one changes its hash and breaks the lineage it exists to prove.
#: * ``tools/latency_*`` and ``tasks/Latency/`` measure the quantity Kane et
#:   al. (2020) Figure 5 reports, under the name that paper gives it. That is
#:   a citation, not a frame.
#: * Code comments generally: a comment naming the path it guards is doing
#:   the job the note asks for, which is to be specific.
#: Copies of the tree kept beside it, from a snapshot or a backup. Their
#: prose is not this project's to police: every file in them is a duplicate,
#: so each hit is reported twice and a citation that is already allowed at its
#: real path is flagged again under the copy's.
SNAPSHOT_DIRS = {"data - Copy", "from jetson"}

SKIP_DIRS = {"_build", "_export", "__pycache__", ".git", "data", "models",
             "vendor", ".pytest_cache", "node_modules", "build",
             "experiments", "paper"} | SNAPSHOT_DIRS
EXTS = {".md", ".html", ".rst"}


def _files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTS:
            continue
        if any(p in SKIP_DIRS for p in path.parts):
            continue
        if path.name == pathlib.Path(__file__).name:
            continue          # this file names every word it forbids
        yield path


@pytest.mark.parametrize("pattern,instead", sorted(BANNED.items()))
def test_the_word_was_rejected_and_stays_rejected(pattern, instead):
    found = []
    for path in _files():
        rel = path.relative_to(ROOT).as_posix()
        if (rel, pattern) in ALLOWED:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if re.search(pattern, line, re.I):
                found.append(f"  {rel}:{n}  {line.strip()[:88]}")
    assert not found, (
        f"{len(found)} use(s) of a rejected word:\n" + "\n".join(found[:25])
        + f"\n\nWrite instead: {instead}.\n"
        "Rewrite the sentence around the mechanism rather than swapping in a "
        "synonym. If the word is quoted rather than used, add it to ALLOWED "
        "with the reason.")


def test_the_check_reads_something():
    """A sweep with an empty file list passes for the wrong reason."""
    seen = sum(1 for _ in _files())
    assert seen > 50, f"only {seen} prose files found, is the scope wrong?"
