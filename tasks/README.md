# Tasks

Task state machines live here, one Python file per task, uploaded to the
board when a session starts. This folder ships empty: a task encodes one
lab's experiment and there is no useful default.

A task declares `states`, `events` and a handler per state. Every event
name it lists must appear verbatim in the hardware definition, or the
input is never wired and nothing says so.

See `docs/` for the task API and a worked example.
