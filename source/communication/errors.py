"""MCU-side error + hint message templates.

Pure module: no I/O, no Qt, no logger import, so anyone can import from here
without dragging in heavier layers.

Format strings use ``.format(...)`` at the call site::

    raise PyboardError(ERR_TASK_FILE_NOT_FOUND.format(path=sm_path))
"""

# Framework
ERR_FRAMEWORK_NOT_LOADED = "Framework not loaded on device."
ERR_FRAMEWORK_IMPORT_ERROR = "Error importing framework."

# Task
ERR_TASK_NOT_SELECTED = "No task selected."
ERR_TASK_FILE_NOT_FOUND = "Task file not found: {path}"
ERR_TASK_UPLOAD_FAILED = "Unable to upload task to device."
ERR_TASK_SETUP_FAILED = "Unable to setup state machine."

# File transfer
ERR_FILE_TRANSFER_NO_SPACE = "Insufficient space on device filesystem to transfer file."
ERR_FILE_TRANSFER_FAILED = "Unable to transfer file."

# Hardware definition
ERR_HW_DEF_NOT_FOUND = "Hardware definition file not found."
ERR_HW_DEF_IMPORT_ERROR = "Error importing hardware definition."

# Data integrity
ERR_BAD_CHECKSUM = "Bad data checksum."
ERR_UNEXPECTED_MCU_INPUT = "Unexpected input received from board: {data}"

# Hints
HINT_UPLOAD_FRAMEWORK = "Upload the framework first using Config Boards."
HINT_TROUBLESHOOTING_CHECKS = (
    "Check the USB cable and port, that no other program has the board "
    "open, and that the framework is uploaded."
)


__all__ = [
    "ERR_FRAMEWORK_NOT_LOADED", "ERR_FRAMEWORK_IMPORT_ERROR",
    "ERR_TASK_NOT_SELECTED", "ERR_TASK_FILE_NOT_FOUND",
    "ERR_TASK_UPLOAD_FAILED", "ERR_TASK_SETUP_FAILED",
    "ERR_FILE_TRANSFER_NO_SPACE", "ERR_FILE_TRANSFER_FAILED",
    "ERR_HW_DEF_NOT_FOUND", "ERR_HW_DEF_IMPORT_ERROR",
    "ERR_BAD_CHECKSUM", "ERR_UNEXPECTED_MCU_INPUT",
    "HINT_UPLOAD_FRAMEWORK", "HINT_TROUBLESHOOTING_CHECKS",
]
