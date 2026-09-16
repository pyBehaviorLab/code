"""The headless analysis engine: what a recording knows, and what is missing.

Ported from pyBehaveTrack, whose Analyze tab stopped asking the operator to
pick a "mode". A recording is not in a mode, it either has a clock, poses, a
space and measures, or it is missing some of them, and the work to be done is
whatever is missing.

Nothing here imports Qt. The GUI renders what these modules compute; it does
not compute anything itself, which is what makes the plan the tab shows and the
work the run performs the same object rather than two descriptions that can
disagree.
"""
