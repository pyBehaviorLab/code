"""Dialog package, dialogs grouped by domain.

One file per domain, multiple dialog classes per file. Re-exported here
so existing callers (``from source.gui.dialogs import X``) keep working.
"""

# MCU / board
from .mcu import (  # noqa: F401
    UniversalConnectDialog,
    UniversalDisconnectDialog,
    UniversalStartDialog,
    UniversalStopDialog,
    UniversalConfigDialog,
    UniversalUploadDialog,
)
from .door import DoorControlDialog  # noqa: F401

# Camera (big single-purpose dialogs kept in their own files)
from .camera_connect import CameraConnectDialog  # noqa: F401

# Video / ROI
from .video_roi import ROISegmentationDialog  # noqa: F401

# Tracking, the main runtime dialog is UnifiedTrackingDialog in
# source.gui.widgets.tracking_panel. TrackerCalibrationDialog is the only
# companion: LIVE, opened from the blob settings panel's "Calibrate" button.
from .tracking import TrackerCalibrationDialog  # noqa: F401

# Subjects / metadata
from .subjects import (  # noqa: F401
    AssignSubjectsDialog,
    MetadataEditorDialog,
)

# Task controls
from .controls import (  # noqa: F401
    ControlsDialog,
    ConfigSelectionDialog,
)
