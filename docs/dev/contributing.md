# Contributing

The codebase follows a few hard conventions distilled from many regressions. They
override personal preference.

## Hard rules

1. **No legacy / no schema_version branches.** Schema bumps replace old code inline.
   No migrators. No `.bak` sidecars. No `if schema_version < X:` branches. Old
   code goes away; the new path handles legacy YAML via *additive* default values
   (e.g. `mcu_serial: str = ""`), not via branches.

2. **DJB2 only.** No SHA256 anywhere. The hash function lives in
   `source/config/hashing.py`. Other modules import from there.

3. **Pipeline lives in one module.** Camera + tracker + DLC + recording is
   `source/video/framebus/`. Don't fan it out across modules without explicit
   discussion.

4. **Consolidate new features.** A new feature = one new module file. Existing
   files get 1-3 line hooks only.

5. **Only what was asked.** Change exactly the property the user named.
   Don't bundle "while I'm here" cleanups into a bug fix.

6. **Drive UI before claiming done.** Unit tests for save/load aren't enough.
   Launch the app, walk the flow, confirm the bug is fixed end-to-end.

7. **PyboardError trap.** `PyboardError` extends `BaseException`, not `Exception`.
   `except Exception` won't catch it. Use `except BaseException` or
   `except PyboardError` explicitly when wrapping pycboard calls.

8. **No Qt from worker threads.** ParallelStartCoordinator workers do serial I/O
   only. UI updates (`_after_stop`, timer ops, `refresh_ui_state`) belong in
   `finalize_record_*` on the GUI thread.

## Layering

- `source/communication/`. MCU + serial. No Qt.
- `source/video/`, pipeline + sinks + tracking + recording. No Qt.
- `source/config/`, schema + load/save bridge. No Qt.
- `source/pyControl/`. MCU framework code (uploaded to MCU). MicroPython only.
- `source/gui/`. Qt frontend. Imports everything else.
- `source/stats/`, live dashboard. Qt allowed.
- `source/tests/`, pytest.

## Editing the docs

The pages under `docs/` are MyST markdown built by Sphinx:

```bash
python -m sphinx -b html docs docs/_build/html     # the check; xref warnings matter
python tools/docs_edit.py                          # click-to-edit in a browser
```

`tools/docs_edit.py` serves a local page listing every `.md`, turns any section
into a text box on click, and writes the real file on Save; Export writes a
standalone `.html` into `docs/_export/`. It refuses a save if the file changed on
disk since the browser loaded it, so a parallel edit in an editor is never
silently overwritten.

Build warnings are the link check. A `myst.xref_missing` warning means a
cross-page anchor does not exist, and the MyST anchor slug is **not** the id in
the built HTML for a heading carrying punctuation. Link to a punctuation-free
sub-heading rather than guessing.

## How the pages are written

The reader is a behavioural neuroscientist with an animal waiting. They are
not reading for pleasure and they are not reading the whole page. Everything
below follows from that.

**The register is plain and declarative.** State what is true. No second
person, no imperatives stacked on each other, no rhetorical turns. This is the
register of the reference the reader will come back to at eleven at night, and
it is the one that stays consistent across fifty pages written months apart.

> A zone is a region drawn on the video. When the tracked body part enters
> one, the host sends its name to the task. The task compares it like any
> other variable.

Three sentences, three facts, nobody addressed. Compare the version this
replaced, which said the same thing in the shape of a sales line: *"Draw a
zone on the video and you get the zone name in your task, so you can react to
where the animal went, not just what it pressed."* Second person, a promise,
and a flourish at the end.

Procedures are the one exception: a numbered sequence of steps is written as
instructions, because that is what a step is.

**Write from their side of the screen.** The landing page used to open:

> A single-process behavioural-rig platform for operant chambers and maze
> arenas, sharing one codebase. Each box owns one microcontroller, the host
> pushes coordinates + zone events back to the MCU for the task to act on.

Every fact there is true and every one is about *us*: our process model, our
repository, our data flow. "Single-process" and "sharing one codebase" answer
a question no researcher has. It now opens with what they can do, and the
architecture arrives later, where someone is looking for it. Compare what
pyControl leads with, which is one sentence and a benefit: *"pyControl makes
it easy to program complex behavioural tasks."*

**Say what to do, then why.** Rationale is worth writing down; this project
has paid for most of it. But it goes after the instruction or into a note, not
in front of the reader who is trying to press the button. The old pages
routinely spent a paragraph justifying a setting before saying where it was.

**One idea per paragraph, two to four sentences.** Reference documentation
that people actually finish runs short. If a paragraph passes six sentences it
is usually two paragraphs, or a paragraph and a table.

**A technical term is introduced once, in a sentence that does not need it.**
Letterbox, crop-track, lineage, coord_var, djb2: each of these appears in the
docs before anything grounds it. Give the plain meaning first and the name
second, then use the name freely. Where the term has a reference page, link it
on first use and keep going, the way Autopilot does, rather than stopping to
define it inline.

**Anchor a new concept to one the reader already has.** They arrive knowing
pyControl state machines. Zones, coordinate pushes and the follow window are
all easier against that than from first principles.

**Name the thing the reader sees.** Buttons, dialogs and columns get the label
that is printed on them, not the attribute behind them. `dlc_resize_spin` is
"the Resize box on the tracking panel".

**Say what it costs.** A page that needs a GPU, an internet connection, a
particular camera or a calibration says so at the top. This is what the rig
badges are for.

**No em dash.** Enforced by `source/tests/test_no_em_dashes.py`; use a comma, a
semicolon, a colon or a hyphen.

## Commit etiquette

- One concern per commit.
- Reference the test that covers the change in the body.
- Don't push to `main` directly; PRs only.

## When in doubt

Check `MEMORY.md` (the auto-memory the agent maintains), it holds the codified
"why we don't do X" lessons from past regressions. If a rule there contradicts
what you're about to do, ask in the issue thread before merging.
