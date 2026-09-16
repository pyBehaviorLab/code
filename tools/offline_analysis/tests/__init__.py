"""Tests for the offline analyzer.

Run with:
    python -m pytest tools/offline_analysis/tests -v

They are collected by the top-level suite too (``pyproject.toml`` lists this
folder in ``testpaths``); they were not for a long time, which is how three of
the analyzer's imports rotted into permanent ``ImportError``s unnoticed.

The analyzer shares the rig's data and presentation layers by importing them;
it must never touch the live-rig subsystems (camera, serial, frame bus,
firmware), since it is meant to be safe to run while a session is recording.
``test_no_live_rig_imports.py`` guards that. The older, broader rule, no
``source.*`` imports at all, so the folder could be copied out and run
standalone, was dropped deliberately; see the package docstring.
"""
