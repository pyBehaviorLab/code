"""pyBehaviorLab source package."""

# ``VERSION`` is the pyControl FIRMWARE version the host EXPECTS on the board:
# pycboard compares the board's ``fw.VERSION`` against this and warns to reload
# the framework on mismatch, so it MUST stay in lockstep with
# source/pyControl/framework.py (VERSION = "2.1"). It is NOT the GUI version.
VERSION = "2.1"          # firmware (board) version
FW_VERSION = VERSION
# GUI / desktop-application version, separate from the firmware. Mirrors the
# package version in pyproject.toml.
GUI_VERSION = "2.0"

__all__ = ["VERSION", "GUI_VERSION", "FW_VERSION"]
