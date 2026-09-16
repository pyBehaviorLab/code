"""Widget package, every non-dialog UI element, one class per file.

Tab/window framework, per-box widgets, frame display, zone editor+overlay,
and the live tab-shaped panels (statistics, tracking). Offline-analysis
tabs live outside the rig in tools/offline_analysis/.
"""

# Detachable tab framework
from .tab_window import DetachableTabWindow  # noqa: F401
from .tab_widget import DetachableTabWidget  # noqa: F401

# Reusable common widgets
from .common import (  # noqa: F401
    CollapsibleSidebar,
    RotatedButton,
)
from .error_log_panel import ErrorLogPanel  # noqa: F401
from .markdown_view import MarkdownView  # noqa: F401
from .run_task import RunTask  # noqa: F401
from .frame_display import FrameDisplay, ROIDrawCanvas  # noqa: F401

# Per-box widgets
from .video_stream import VideoStreamHolder  # noqa: F401
from .live_status import LiveStatusWidget  # noqa: F401
from .setup_widget import SetupWidget  # noqa: F401
from .box_control import BoxControlWidget  # noqa: F401

# Tab-shaped panels
from .tracking_panel import TrackingSettingsPanel, UnifiedTrackingDialog  # noqa: F401

# Zone editor + overlay
from .zone_editor import ZoneEditorWidget  # noqa: F401
from .zone_overlay import *  # noqa: F401, F403
