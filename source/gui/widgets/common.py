"""
Reusable, purely-presentational GUI widgets + constants (no backend coupling).

Exports:
    make_button / make_lineedit  One-line styled per-box widget builders
    SUBJECT_PRESETS       Calibration-parameter dicts keyed by subject profile
    PRESET_NAMES          Ordered list for populating preset dropdowns
    CollapsibleSidebar / BusyDialog / RotatedButton  Reusable widgets
"""

from __future__ import annotations

from typing import List, Optional

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets


# ============================================================================
# Per-box widget builders, one-line constructors that replace ~6 lines of
# QPushButton/QLineEdit boilerplate per widget across box_control.py,
# setup_widget.py, video_stream.py. The whole point of these is debuggability:
# every per-box button now reads as a single line saying *what* it is, not 6
# lines of fixed-size / icon / stylesheet plumbing.
# ============================================================================

# Standard widget height used by both per-box widgets (24-25 px row).
DEFAULT_ROW_H = 25


def make_button(
    label: str,
    *,
    color: str = "info",
    icon: str | Path | None = None,
    width: int | None = None,
    height: int = DEFAULT_ROW_H,
    tooltip: str | None = None,
    enabled: bool = True,
    flat: bool = False,
) -> QtWidgets.QPushButton:
    """Build a styled QPushButton with the standard per-box look.

    ``color`` is a key from ``styles.COLORS`` (e.g. ``"info"``, ``"danger"``,
    ``"success"``, ``"warning"``, ``"primary"``, ``"secondary"``,
    ``"start_light"``). ``"ghost"`` / ``"plain"`` skip the gradient and use
    ``PUSH_BUTTON_STYLE`` (flat dark button).
    """
    from source.gui.styles import BUTTON_STYLE, PUSH_BUTTON_STYLE, COLORS

    btn = QtWidgets.QPushButton(label)
    if icon is not None:
        ipath = Path(icon)
        if ipath.exists():
            btn.setIcon(QtGui.QIcon(str(ipath)))
            btn.setIconSize(QtCore.QSize(14, 14))
    if width is not None:
        btn.setFixedSize(int(width), int(height))
    else:
        btn.setFixedHeight(int(height))
    if color in ("ghost", "plain"):
        btn.setStyleSheet(PUSH_BUTTON_STYLE)
    else:
        hover_key = color + "_hover"
        btn.setStyleSheet(BUTTON_STYLE.format(
            color=COLORS.get(color, COLORS["info"]),
            hover_color=COLORS.get(hover_key, COLORS["info_hover"]),
        ))
    btn.setStyleSheet(btn.styleSheet()
                      + f"QPushButton {{ min-height: {int(height)}px;"
                      f" max-height: {int(height)}px; }}")
    if tooltip:
        btn.setToolTip(tooltip)
    btn.setEnabled(enabled)
    if flat:
        btn.setFlat(True)
    return btn


def make_lineedit(
    *,
    placeholder: str = "",
    width: int | None = None,
    height: int = DEFAULT_ROW_H,
    readonly: bool = False,
    value: str = "",
    expanding: bool = False,
    align: QtCore.Qt.AlignmentFlag | None = None,
) -> QtWidgets.QLineEdit:
    """Build a styled QLineEdit with the standard per-box look."""
    from source.gui.styles import LINE_EDIT_STYLE

    le = QtWidgets.QLineEdit()
    if placeholder:
        le.setPlaceholderText(placeholder)
    if value:
        le.setText(value)
    if expanding:
        le.setFixedHeight(int(height))
        le.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                         QtWidgets.QSizePolicy.Policy.Fixed)
        if width is not None:
            le.setMinimumWidth(int(width))
    elif width is not None:
        le.setFixedSize(int(width), int(height))
    else:
        le.setFixedHeight(int(height))
    if readonly:
        le.setReadOnly(True)
    if align is not None:
        le.setAlignment(align)
    le.setStyleSheet(LINE_EDIT_STYLE)
    return le


# ============================================================================
# Subject presets, drop-in calibration parameter packs
# ============================================================================

# Area bounds are the subject's contour in pixels². A mouse is 1500–8000 px²
# and a rat 6000–30000 at typical rig resolutions; max_area is there to reject
# a whole-arena blob, so it sits well above the animal rather than below it.
#
# use_adaptive_threshold (Otsu) is off everywhere on purpose: Otsu assumes a
# bimodal histogram, and a background-difference image is ~99% near-zero, so
# it picks a threshold inside the sensor noise floor and turns an empty arena
# into one frame-sized blob.
# Each preset names a SITUATION and sets the detector for it, ``bg_mode``
# included, the detection method is not a separate expert choice, it IS the
# choice, and the measurements say so plainly. On a 17.8-minute operant
# session, differencing against a captured background tracked the animal in
# 96 % / 81 % of frames across two windows; the background-free detector
# managed 47 % / 20 % at 2-4x the CPU. So every subject preset is
# ``static``, and the two modes that lose are kept only for the cases they
# genuinely win: adaptive for drifting light, background-free for a textured
# floor with no stable empty view.
SUBJECT_PRESETS = {
    "Dark Mouse / Light BG": {
        # CLAHE OFF by default. It costs 1.9x the detection time (19.3 ms per
        # box against 10.2 ms measured), and on a 16-box rig detection time IS
        # the frame rate, that difference alone is 13 fps against 26. It also
        # does not earn it on accuracy where contrast is already good: 96.0 %
        # without against 97.0 % with in the easy window, and 81.2 % against
        # 79.5 % in the hard one, i.e. within noise and pointing both ways.
        # The low-contrast preset below turns it on, which is the case for.
        "bg_mode": "static",
        "detect_dark": True,
        "threshold": 25,
        "clahe_clip_limit": 3.0,
        "min_area": 150,
        "max_area": 20000,
        "blur_kernel_size": 5,
        "use_clahe": False,
        "use_adaptive_threshold": False,
    },
    "Dark Mouse / Dark BG": {
        # Low contrast / cluttered arena. CLAHE more than halves the median
        # error here (17.1 -> 10.2 px measured); this is the case it is for.
        "bg_mode": "static",
        "detect_dark": True,
        "threshold": 15,
        "clahe_clip_limit": 5.0,
        "min_area": 150,
        "max_area": 20000,
        "blur_kernel_size": 5,
        "use_clahe": True,
        "use_adaptive_threshold": False,
    },
    "Light Mouse / Dark BG": {
        "bg_mode": "static",
        "detect_dark": False,
        "threshold": 20,
        "clahe_clip_limit": 3.0,
        "min_area": 150,
        "max_area": 20000,
        "blur_kernel_size": 5,
        "use_clahe": True,
        "use_adaptive_threshold": False,
    },
    "Dark Rat / Light BG": {
        "bg_mode": "static",
        "detect_dark": True,
        "threshold": 25,
        "clahe_clip_limit": 3.0,
        "min_area": 400,
        "max_area": 60000,
        "blur_kernel_size": 7,
        "use_clahe": True,
        "use_adaptive_threshold": False,
    },
    "Changing Light / Long Session": {
        # Adaptive background for light cycles and projector stimuli. Comes
        # with a real cost: it learns a STATIONARY animal into its own
        # background, measured 96.7 % while the animal moved, 15.1 % in a
        # window where it rested. Right for drifting light, wrong as a
        # default, and wrong for a species that freezes.
        "bg_mode": "running_avg",
        "detect_dark": True,
        "threshold": 20,
        "clahe_clip_limit": 3.0,
        "min_area": 150,
        "max_area": 20000,
        "blur_kernel_size": 5,
        "use_clahe": False,
        "use_adaptive_threshold": False,
    },
    "Textured Bedding (no background)": {
        # mousefinder-style: no reference image at all. The case it wins is a
        # textured floor (gravel, sawdust, mesh) where no clean empty view
        # exists to subtract. On a furnished arena it locks onto the
        # furniture instead, see the Simple-mode notes in blob.py.
        "bg_mode": "self_norm",
        "detect_dark": True,
        "threshold": 25,
        "clahe_clip_limit": 3.0,
        "min_area": 150,
        "max_area": 20000,
        "blur_kernel_size": 5,
        "use_clahe": False,
        "use_adaptive_threshold": False,
    },
    "Custom": None,
}

PRESET_NAMES: List[str] = list(SUBJECT_PRESETS.keys())


# ============================================================================
# CollapsibleSidebar, overlay sidebar that slides from the left
# ============================================================================

class CollapsibleSidebar(QtWidgets.QFrame):
    """Modern overlay sidebar that slides from the left.

    Hosts arbitrary content via :meth:`addField`. Collapses on outside
    click (event-filter) and animates the width transition.
    """

    def __init__(self, parent=None, header_text="SIDEBAR", expanded_width=300):
        super().__init__(parent)
        self.is_collapsed = True
        self.collapsed_width = 0
        self.expanded_width = expanded_width
        self.parent_widget = parent
        self.on_collapse_callback = None
        self.header_text = header_text
        self._setup_ui()

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if not self.is_collapsed and self._outside(event):
                self.collapse()
                # Not consumed.  Swallowing the press meant the first click
                # after opening a sidebar did nothing except close it, so
                # every control needed clicking twice.
                return False
        # Keep our height matched to the parent when it resizes while
        # we're expanded.  Without this the parent can grow but the
        # sidebar stays short, exposing the tab content below it.
        if (event.type() == QtCore.QEvent.Type.Resize
                and obj is self.parent_widget
                and not self.is_collapsed):
            self.setFixedHeight(self.parent_widget.height())
        return super().eventFilter(obj, event)

    def _outside(self, event) -> bool:
        """Whether this press landed anywhere but on the sidebar.

        Compared in SCREEN coordinates.  The press is delivered to whichever
        widget was clicked, so its own ``pos()`` is in that widget's frame and
        says nothing about where the sidebar is.

        The width is the one the sidebar is heading for, not the one it holds
        mid-slide.  The open animation runs 300 ms up from zero, so measuring
        the live width makes every point fall outside an empty rectangle and
        shuts the sidebar an instant after it was asked to open.
        """
        try:
            point = event.globalPosition().toPoint()
        except AttributeError:                      # pragma: no cover
            point = event.globalPos()
        top_left = self.mapToGlobal(self.rect().topLeft())
        width = self.expanded_width if not self.is_collapsed else self.width()
        return not QtCore.QRect(
            top_left, QtCore.QSize(width, self.height())).contains(point)

    def mousePressEvent(self, event):
        if not self.is_collapsed and self.width() < self.expanded_width:
            # Still sliding in.  Nothing occupies the point under the cursor
            # yet, so childAt() calls every position background, and clicking
            # the background is what closes the sidebar.
            super().mousePressEvent(event)
            return
        widget_at_pos = self.childAt(event.position().toPoint())
        interactive_types = (
            QtWidgets.QPushButton, QtWidgets.QLineEdit, QtWidgets.QTextEdit,
            QtWidgets.QTextBrowser, QtWidgets.QComboBox, QtWidgets.QSpinBox,
            QtWidgets.QCheckBox, QtWidgets.QRadioButton, QtWidgets.QScrollBar,
        )
        if widget_at_pos is None or not isinstance(widget_at_pos, interactive_types):
            parent = widget_at_pos.parent() if widget_at_pos else None
            if parent and isinstance(parent, interactive_types):
                super().mousePressEvent(event)
            else:
                self.collapse()
        else:
            super().mousePressEvent(event)

    def _setup_ui(self):
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setFixedWidth(self.collapsed_width)
        # Paint a solid background, without this the QFrame is
        # transparent and the tab widget underneath bleeds through the
        # expanded sidebar (looks like Main Control / Experimental Info
        # overlap).  Both the frame and the inner scroll area need it.
        self.setAutoFillBackground(True)

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Header
        header = QtWidgets.QWidget()
        header.setFixedHeight(50)
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(15, 0, 15, 0)

        from source.gui.theme import THEME as _T
        from source.gui.styles import BUTTON_STYLE as _BS, COLORS as _C
        header_label = QtWidgets.QLabel(self.header_text)
        header_label.setStyleSheet(
            f"font-weight: 700; font-size: 11px; color: {_T.palette.text};"
            " text-transform: uppercase; letter-spacing: 0.5px;")
        header_layout.addWidget(header_label)
        header_layout.addStretch()

        close_btn = QtWidgets.QPushButton("◀")
        close_btn.setFixedSize(30, 30)
        close_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.collapse)
        # Vivid primary gradient (pink→purple), apex action of the
        # sidebar header.
        close_btn.setStyleSheet(_BS.format(
            color=_C['primary'], hover_color=_C['primary_hover']))
        header_layout.addWidget(close_btn)

        # Content with scroll
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll_area.setAutoFillBackground(True)

        self.content_widget = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(15, 15, 15, 15)
        self.content_layout.setSpacing(12)
        self.scroll_area.setWidget(self.content_widget)

        self.main_layout.addWidget(header)
        self.main_layout.addWidget(self.scroll_area)

        # Animation
        self.animation = QtCore.QPropertyAnimation(self, b"minimumWidth")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

        self.animation_max = QtCore.QPropertyAnimation(self, b"maximumWidth")
        self.animation_max.setDuration(300)
        self.animation_max.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

        # Shadow
        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(2)
        shadow.setColor(QtGui.QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)

    def addField(self, label_text, widget, help_text=None):
        """Add a label + widget row to the sidebar's scroll area."""
        label_container = QtWidgets.QWidget()
        label_layout = QtWidgets.QHBoxLayout(label_container)
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.setSpacing(5)

        from source.gui.theme import THEME as _Tl
        label = QtWidgets.QLabel(label_text)
        label.setStyleSheet(
            f"font-weight: 700; font-size: 10px; color: {_Tl.palette.focus};")
        label_layout.addWidget(label)

        if help_text:
            help_btn = QtWidgets.QPushButton("?")
            help_btn.setFixedSize(18, 18)
            help_btn.setToolTip(help_text)
            help_btn.setStyleSheet(
                "QPushButton {"
                f" background-color: {_Tl.palette.surface_elev};"
                f" color: {_Tl.palette.text}; border: none;"
                f" border-radius: 9px;"
                "}"
                f"QPushButton:hover {{ background-color: {_Tl.palette.surface_elev_2}; }}"
            )
            label_layout.addWidget(help_btn)

        label_layout.addStretch()
        self.content_layout.addWidget(label_container)
        self.content_layout.addWidget(widget)

    def toggle(self):
        if self.is_collapsed:
            self.expand()
        else:
            self.collapse()

    def expand(self):
        self.is_collapsed = False
        # Match the parent's current height *now*, MainWindowBase's
        # _sync_sidebar_heights only fires on resizeEvent / moveEvent
        # and can miss interim parent resizes (e.g. the central widget
        # growing after init).  Without this snap the sidebar stays at
        # whatever fixed-height it last had and the Main Control tab
        # peeks out below its bottom edge.
        if self.parent_widget is not None:
            self.setFixedHeight(self.parent_widget.height())
        self.show()
        self.raise_()

        self.animation.setStartValue(self.width())
        self.animation.setEndValue(self.expanded_width)
        self.animation.start()

        self.animation_max.setStartValue(self.width())
        self.animation_max.setEndValue(self.expanded_width)
        self.animation_max.start()

        # The parent only ever sees presses that land on its OWN background.
        # Every click on something clickable, a tab, a button, a table, is
        # delivered to that child and never reaches the parent, so a sidebar
        # watching only its own background stays open whatever the operator
        # clicks. Watching the application catches the press wherever it lands.
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        if self.parent_widget:
            self.parent_widget.installEventFilter(self)

    def collapse(self):
        self.is_collapsed = True

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        if self.parent_widget:
            self.parent_widget.removeEventFilter(self)

        self.animation.setStartValue(self.width())
        self.animation.setEndValue(self.collapsed_width)
        self.animation.finished.connect(self._on_collapse_finished)
        self.animation.start()

        self.animation_max.setStartValue(self.width())
        self.animation_max.setEndValue(self.collapsed_width)
        self.animation_max.start()

    def _on_collapse_finished(self):
        self.hide()
        try:
            self.animation.finished.disconnect(self._on_collapse_finished)
        except TypeError:
            pass
        if self.on_collapse_callback:
            self.on_collapse_callback()


# ============================================================================
# BusyDialog, minimal "what's happening" progress popup for long ops
# ============================================================================
#
# Two long-running operations need user feedback:
#
#   1. Camera FPS calibration (~14 s wait, 2 s drain + 10 s measure + 2 s
#      slack). Phase labels advance on a QTimer; no work runs in the GUI
#      thread so the bar can tick on a wall-clock schedule.
#
#   2. DLC / SLEAP model init (variable, model-dependent). Work runs in
#      the GUI thread; advance the bar by calling ``set_step()`` between
#      stages and call ``QApplication.processEvents()`` so the dialog
#      repaints.
#
# Cancel is disabled on purpose: the underlying camera thread / model
# loader can't be safely interrupted mid-operation.

class BusyDialog(QtWidgets.QDialog):
    """Compact modal progress dialog for long operations.

    Two flavours:
        BusyDialog.timed(parent, title, total_seconds, phases)
            Ticks a wall-clock progress bar over ``total_seconds``.
            ``phases`` is a list of (cumulative_seconds, label): the
            label shown at any given tick is the latest phase whose
            cumulative time has elapsed.

        BusyDialog.stepped(parent, title, n_steps, detail="")
            Manual progress; caller calls ``set_step(i, label)`` between
            work items. Use this for in-thread work (DLC init etc.) and
            call ``QtWidgets.QApplication.processEvents()`` after each
            step so the dialog repaints.

    Both close automatically when ``finish(success=True/False, msg="")``
    is called, or when the timed variant's clock runs out.
    """

    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedWidth(360)
        self.setWindowFlags(
            self.windowFlags()
            & ~QtCore.Qt.WindowType.WindowCloseButtonHint
            & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        self._header = QtWidgets.QLabel(title)
        self._header.setStyleSheet("font-weight: 600; font-size: 11pt;")
        layout.addWidget(self._header)

        self._detail = QtWidgets.QLabel("")
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet("color: #cccccc; font-size: 9pt;")
        layout.addWidget(self._detail)

        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(0, 0)  # indeterminate by default
        self._bar.setTextVisible(False)
        layout.addWidget(self._bar)

        # Timed-mode state (populated by .timed())
        self._tick_timer: Optional[QtCore.QTimer] = None
        self._phases: List = []           # list of (cum_sec, label)
        self._elapsed_ms: int = 0
        self._total_ms: int = 0
        self._on_timeout = None           # optional callable on auto-close

    # ------------------------------------------------------------------
    # Timed factory, used for the camera FPS calibration wait.
    # ------------------------------------------------------------------

    @classmethod
    def timed(cls, parent, title: str, total_seconds: float,
              phases: Optional[List] = None,
              on_timeout=None) -> "BusyDialog":
        """Create + show a timed progress dialog.

        ``phases`` is a list of (cumulative_seconds, label) entries.
        ``on_timeout`` runs once at the end (typically the same callback
        that would have fired from a manual QTimer.singleShot).
        """
        dlg = cls(parent, title)
        dlg._phases = sorted(phases or [], key=lambda p: p[0])
        dlg._total_ms = int(total_seconds * 1000)
        dlg._on_timeout = on_timeout
        dlg._bar.setRange(0, dlg._total_ms)
        dlg._bar.setTextVisible(True)
        dlg._bar.setFormat("%v / %m ms")
        if dlg._phases:
            dlg._detail.setText(dlg._phases[0][1])

        dlg._tick_timer = QtCore.QTimer(dlg)
        dlg._tick_timer.setInterval(100)
        dlg._tick_timer.timeout.connect(dlg._tick)
        dlg._tick_timer.start()
        dlg.show()
        QtWidgets.QApplication.processEvents()
        return dlg

    def _tick(self) -> None:
        self._elapsed_ms += 100
        self._bar.setValue(min(self._elapsed_ms, self._total_ms))
        # Walk phases and pick the latest one whose cumulative time has elapsed
        elapsed_s = self._elapsed_ms / 1000.0
        current_label = self._phases[0][1] if self._phases else ""
        for cum, label in self._phases:
            if elapsed_s >= cum:
                current_label = label
        self._detail.setText(current_label)
        if self._elapsed_ms >= self._total_ms:
            self._tick_timer.stop()
            cb = self._on_timeout
            self._on_timeout = None  # one-shot
            self.accept()
            if callable(cb):
                try:
                    cb()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Stepped factory, used for DLC / SLEAP init.
    # ------------------------------------------------------------------

    @classmethod
    def stepped(cls, parent, title: str, n_steps: int,
                detail: str = "") -> "BusyDialog":
        """Create + show a stepped progress dialog. Caller drives via
        ``set_step()`` and finishes with ``finish()``."""
        dlg = cls(parent, title)
        dlg._bar.setRange(0, max(1, n_steps))
        dlg._bar.setValue(0)
        dlg._bar.setTextVisible(True)
        dlg._bar.setFormat("%v / %m")
        dlg._detail.setText(detail)
        dlg.show()
        QtWidgets.QApplication.processEvents()
        return dlg

    def set_step(self, step: int, detail: str = "") -> None:
        """Update the bar + detail label and force a repaint. Use after
        each in-thread step so the user sees progress."""
        try:
            self._bar.setValue(step)
            if detail:
                self._detail.setText(detail)
            QtWidgets.QApplication.processEvents()
        except RuntimeError:
            pass  # dialog destroyed

    def finish(self, success: bool = True, msg: str = "") -> None:
        """Close the dialog. Intended for stepped mode; safe for timed too."""
        try:
            if self._tick_timer is not None:
                self._tick_timer.stop()
        except Exception:
            pass
        try:
            if msg:
                self._detail.setText(msg)
                self._bar.setValue(self._bar.maximum())
                QtWidgets.QApplication.processEvents()
            self.accept() if success else self.reject()
        except RuntimeError:
            pass


class RotatedButton(QtWidgets.QPushButton):
    """Button with vertical (270-degree rotated) text for sidebar toggles."""

    def __init__(self, text, color="#bd93f9", hover_color="#ff79c6",
                 pressed_color="#8be9fd", parent=None):
        super().__init__(text, parent)
        self._color = color
        self._hover_color = hover_color
        self._pressed_color = pressed_color

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(self.rect()), 5, 5)
        if self.isDown():
            painter.fillPath(path, QtGui.QColor(self._pressed_color))
        elif self.underMouse():
            painter.fillPath(path, QtGui.QColor(self._hover_color))
        else:
            painter.fillPath(path, QtGui.QColor(self._color))
        painter.translate(int(self.width() / 2), int(self.height() / 2))
        painter.rotate(-90)
        painter.setPen(QtGui.QColor("white"))
        painter.setFont(self.font())
        text_rect = QtCore.QRect(
            int(-self.height() / 2), int(-self.width() / 2),
            int(self.height()), int(self.width())
        )
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, self.text())
        painter.end()
