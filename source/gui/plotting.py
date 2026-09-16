"""Real-time event and analog data plotting using pyqtgraph.

One plot per data kind, all fed from the box's MCU message stream:
``StatesPlot`` (state entries), ``EventsPlot`` (discrete events) and
``AnalogPlot`` (sampled signals), composed by ``TaskPlot``. The x axis
is MCU framework time, not host time, so a plot lines up with the
session's own TSV rather than with when the GUI happened to redraw.
"""

import logging

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets, QtCore
from source.communication.message import MsgType
from source.datetime_formats import format_run_clock
from source.gui.window_behavior import register_independent_window

logger = logging.getLogger(__name__)


def _fw_clock_for(setup_widget):
    """Zero-arg callable → MCU framework time in SECONDS for this box's
    plot sweep + on-canvas clock. Holds at 0 when the board is absent."""
    def _clock():
        pyc = getattr(setup_widget, "pycboard", None)
        return (pyc.get_timestamp() / 1000.0) if pyc is not None else 0.0
    return _clock

# Default history sizes for the live plots.
DEFAULT_STATE_HISTORY_LEN = 100
DEFAULT_EVENT_HISTORY_LEN = 100
DEFAULT_ANALOG_HISTORY_DUR = 10

# Minimum height for one box block in the scrolling Session Plot. These are
# floors so each box stays readable; blocks have no maximum, so they expand to
# fill when the window has room, and the window scrolls once the stacked blocks
# exceed the viewport. The taller floor applies when the task has an analog
# lane (e.g. pyPhotometry), three stacked lanes instead of two.
MIN_BOX_PLOT_H = 280          # states + events, floor for an unknown task
MIN_BOX_PLOT_H_ANALOG = 380   # kept: callers/tests refer to the 3-lane floor

#: A y-axis tick row needs about this much height to stay readable at the
#: application font. The lane STRETCHES were already proportional to the
#: number of states and events, so the lanes divided the block correctly,
#: but the block itself was a constant: a task with 20 states and 15 events
#: split 280 px between 35 tick rows, eight pixels each, and the labels
#: collided. Height has to follow the content that has to fit in it.
PLOT_ROW_H = 16
#: However few rows a lane has, it still needs axis, title and padding.
MIN_LANE_H = 72
#: Header label above each block.
BLOCK_HEADER_H = 34
#: The analog lane is a continuous trace, not tick rows, so it is a fixed add.
ANALOG_LANE_H = 100
#: What one more lane costs the floor. The trigger lane is hidden at build
#: time and revealed later, on the first frame that carries rule states, so
#: the floor has to be recomputed then. Without that the block keeps its
#: two-lane floor while holding three lanes, and the task plot is squeezed
#: into the 3/5 the trigger lane leaves it.
TRIGGER_LANE_H = 100


class _ScrollFriendlyPlotWidget(pg.PlotWidget):
    """``pg.PlotWidget`` whose plain mouse wheel scrolls the surrounding page.

    In the scrolling Session Plot the wheel should move the page up/down, not
    rescale the X axis. So a plain wheel event is ignored and bubbles to the
    enclosing ``QScrollArea``; holding **Ctrl** restores pyqtgraph's wheel
    behaviour (X-axis zoom, since these plots enable the mouse on X only) for
    inspecting a lane.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hide pyqtgraph's auto-range "A" button (bottom-left of the viewbox).
        # It toggles autorange, which fights the fixed −10→0 s sweep window and
        # makes the trace jump, so it reads as broken rather than useful.
        self.getPlotItem().hideButtons()

    def wheelEvent(self, ev):
        if ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(ev)   # Ctrl+wheel -> zoom the X axis
            ev.accept()              # consume it so the page doesn't also scroll
        else:
            ev.ignore()              # plain wheel -> let the page scroll


# ----------------------------------------------------------------------------------------
# TaskPlot
# ----------------------------------------------------------------------------------------


class TaskPlot(QtWidgets.QWidget):
    """Widget for plotting the states, events and analog inputs output by a state machine."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create widgets
        self.states_plot = StatesPlot(self, data_len=DEFAULT_STATE_HISTORY_LEN)
        self.events_plot = EventsPlot(self, data_len=DEFAULT_EVENT_HISTORY_LEN)
        self.analog_plot = AnalogPlot(self, data_dur=DEFAULT_ANALOG_HISTORY_DUR)
        self.run_clock = RunClock(self.states_plot.axis)

        # Setup plots
        self.pause_button = QtWidgets.QPushButton()
        self.pause_button.setEnabled(False)
        self.pause_button.setCheckable(True)
        self.events_plot.axis.setXLink(self.states_plot.axis)
        self.analog_plot.axis.setXLink(self.states_plot.axis)
        self.analog_plot.axis.setVisible(False)

        # create layout

        self.layout = QtWidgets.QGridLayout()
        self.layout.addWidget(self.states_plot.axis, 0, 0, 1, 3)
        self.layout.addWidget(self.events_plot.axis, 1, 0, 1, 3)
        self.layout.addWidget(self.analog_plot.axis, 2, 0, 1, 3)
        self.layout.addWidget(self.pause_button, 3, 0, 1, 3, QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.pause_button.clicked.connect(self.update_pause_btn_text)
        self.update_pause_btn_text()

    def set_state_machine(self, sm_info):
        # Initialise plots with state machine information.

        font = QtGui.QFont()  # inherits application font size
        metrics = QtGui.QFontMetrics(font)
        max_tick_label_width = max([metrics.horizontalAdvance(n) for n in list(sm_info.states) + list(sm_info.events)])
        padding = 10
        self.axiswidth = max_tick_label_width + padding

        self.states_plot.set_state_machine(sm_info)
        self.events_plot.set_state_machine(sm_info)
        self.analog_plot.set_state_machine(sm_info)

        # size the states and events plots relative to the number of states and events
        num_states = max(1, len(sm_info.states))
        num_events = max(1, len(sm_info.events))
        total = num_states + num_events
        states_fraction = num_states / total
        events_fraction = num_events / total

        if self.analog_plot.inputs:
            # state and event plots are 2/3 of the total height
            # analog plot is 1/3 of the total height
            self.analog_plot.axis.setVisible(True)
            self.events_plot.axis.getAxis("bottom").setLabel("")
            self.layout.setRowStretch(0, int(states_fraction * 66))  # States plot
            self.layout.setRowStretch(1, int(events_fraction * 66))  # Events plot
            self.layout.setRowStretch(2, 33)  # Analog plot
        else:
            # state and event plots - EQUAL SIZE (50/50 split)
            self.analog_plot.axis.setVisible(False)
            self.events_plot.axis.getAxis("bottom").setLabel("Time (seconds)")
            self.layout.setRowStretch(0, 50)  # States plot - equal size
            self.layout.setRowStretch(1, 50)  # Events plot - equal size
            self.layout.setRowStretch(2, 0)  # No analog plot

    def run_start(self, recording=False, clock=None, initial_state_id=None):
        """``clock`` is a zero-arg callable returning the MCU framework time
        in SECONDS (``pycboard.get_timestamp()/1000``). The plotted data is
        stored at fw seconds (``na.time/1000``), so driving the sweep + the
        on-canvas clock from the same fw clock keeps the axis "now" and the
        data on one timeline, no host wall clock, no drift vs the TSV."""
        self.pause_button.setChecked(False)
        self.pause_button.setEnabled(True)
        self.update_pause_btn_text()
        self._clock = clock
        self._last_run_time = 0.0
        self.states_plot.run_start(initial_state_id=initial_state_id)
        self.events_plot.run_start()
        self.analog_plot.run_start()
        if recording:
            self.run_clock.recording()

    def run_stop(self):
        self.pause_button.setEnabled(False)
        self.run_clock.run_stop()

    def process_data(self, new_data):
        """Store new data from board."""
        self.states_plot.process_data(new_data)
        self.events_plot.process_data(new_data)
        self.analog_plot.process_data(new_data)

    def _fw_run_time(self):
        """Latest fw run time in seconds; holds last value if the clock is
        missing or raises (e.g. board went None mid-teardown)."""
        clock = getattr(self, "_clock", None)
        if clock is not None:
            try:
                self._last_run_time = float(clock())
            except Exception:
                pass
        return getattr(self, "_last_run_time", 0.0)

    def update(self):
        """Update plots."""
        if not self.pause_button.isChecked():
            run_time = self._fw_run_time()
            self.states_plot.update(run_time)
            self.events_plot.update(run_time)
            self.analog_plot.update(run_time)
            self.run_clock.update(run_time)

    def update_pause_btn_text(self):
        if self.pause_button.isChecked():
            self.pause_button.setText("Resume plotting")
        else:
            self.pause_button.setText("Pause plotting")


# StatesPlot --------------------------------------------------------


class StatesPlot:
    def __init__(self, parent=None, data_len=100):
        self.task_plot = parent
        self.data_len = data_len
        # A lane with no state machine yet. ``set_state_machine`` fills these
        # in, and until it has, every method below must be a no-op rather
        # than an AttributeError: ``update_boxes`` calls ``run_start`` for
        # EVERY running box but only sets the state machine on boxes whose
        # board already reports one, so opening the Session Plot while a box
        # was still starting raised inside the button's slot. An exception
        # crossing back into Qt's C++ takes the process with it, which is why
        # the crash left three registered boxes and no traceback.
        # ``EventsPlot`` has guarded itself this way all along.
        self.state_IDs = []
        self.plots = {}
        self.data = np.zeros([data_len * 2, 2], int)
        self.axis = _ScrollFriendlyPlotWidget(title="States")
        self.axis.showAxis("right")
        self.axis.hideAxis("left")
        self.axis.setRange(xRange=[-10.2, 0], padding=0)
        self.axis.setMouseEnabled(x=True, y=False)
        self.axis.showGrid(x=True, alpha=0.75)
        self.axis.setLimits(xMax=0)

    def set_state_machine(self, sm_info):
        self.state_IDs = list(sm_info.states.values())
        self.axis.clear()
        self.axis.getAxis("right").setTicks([[(i, n) for (n, i) in sm_info.states.items()]])
        self.axis.setYRange(min(self.state_IDs), max(self.state_IDs), padding=0.1)
        self.n_colours = len(sm_info.states) + len(sm_info.events)
        self.plots = {
            ID: self.axis.plot(pen=pg.mkPen(pg.intColor(ID, self.n_colours), width=3)) for ID in self.state_IDs
        }
        self.axis.getAxis("right").setWidth(self.task_plot.axiswidth)
        self.axis.getAxis("right").setStyle(hideOverlappingLabels=False)

    def run_start(self, initial_state_id=None):
        self.data = np.zeros([self.data_len * 2, 2], int)
        if not self.state_IDs:
            return          # no state machine yet, same rule as EventsPlot
        for plot in self.plots.values():
            plot.setData(x=[], y=[])
        # If initial state provided, add it to show something on plot immediately
        # If not provided, use the first (lowest) state ID as default
        if initial_state_id is None and self.state_IDs:
            initial_state_id = min(self.state_IDs)
        if initial_state_id is not None and initial_state_id in self.state_IDs:
            # Initialize with first state entry at time 0
            self.data[-2, 0] = 0  # Entry time
            self.data[-2, 1] = initial_state_id
            self.data[-1, 0] = 0  # Exit time (will be updated)
            self.data[-1, 1] = initial_state_id

    def process_data(self, new_data):
        """Store new data from board"""
        new_states = [nd for nd in new_data if nd.type == MsgType.STATE]
        if new_states:
            n_new = len(new_states)
            self.data = np.roll(self.data, -2 * n_new, axis=0)
            for i, nd in enumerate(new_states):  # Update data array.
                j = 2 * (-n_new + i)  # Index of state entry in self.data
                self.data[j - 1 :, 0] = nd.time
                self.data[j:, 1] = nd.content

    def update(self, run_time):
        """Update plots."""
        if not self.state_IDs:
            return
        self.data[-1, 0] = run_time * 1000  # Update exit time of current state to current time.
        for ID in self.state_IDs:
            state_data = self.data[self.data[:, 1] == ID, :]
            timestamps, IDs = (state_data[:, 0] / 1000 - run_time, state_data[:, 1])
            if timestamps.size > 0:
                self.plots[ID].setData(x=timestamps, y=IDs, connect="pairs")


# EventsPlot--------------------------------------------------------


class EventsPlot:
    def __init__(self, parent=None, data_len=100):
        self.task_plot = parent
        self.axis = _ScrollFriendlyPlotWidget(title="Events")
        self.axis.showAxis("right")
        self.axis.hideAxis("left")
        self.axis.setRange(xRange=[-10.2, 0], padding=0)
        self.axis.setMouseEnabled(x=True, y=False)
        self.axis.showGrid(x=True, alpha=0.75)
        self.axis.setLimits(xMax=0)
        self.event_IDs = []
        self.data_len = data_len

    def set_state_machine(self, sm_info):
        self.event_IDs = list(sm_info.events.values())
        self.axis.clear()
        self.axis.getAxis("right").setTicks([[(i, n) for (n, i) in sm_info.events.items()]])
        self.axis.getAxis("right").setStyle(hideOverlappingLabels=False)
        self.axis.getAxis("right").setWidth(self.task_plot.axiswidth)
        if self.event_IDs:  # Task has events.
            self.axis.setYRange(min(self.event_IDs), max(self.event_IDs), padding=0.1)
            self.n_colours = len(sm_info.states) + len(sm_info.events)
            self.plot = self.axis.plot(pen=None, symbol="o", symbolSize=6, symbolPen=None)

    def run_start(self):
        if not self.event_IDs:
            return  # State machine can have no events.
        self.plot.clear()
        self.data = np.zeros([self.data_len, 2])

    def process_data(self, new_data):
        """Store new data from board."""
        if not self.event_IDs:
            return  # State machine can have no events.
        new_events = [nd for nd in new_data if nd.type == MsgType.EVENT]
        if new_events:
            n_new = len(new_events)
            self.data = np.roll(self.data, -n_new, axis=0)
            for i, nd in enumerate(new_events):
                self.data[-n_new + i, 0] = nd.time / 1000
                self.data[-n_new + i, 1] = nd.content

    def update(self, run_time):
        """Update plots"""
        if not self.event_IDs:
            return
        self.plot.setData(
            x=self.data[:, 0] - run_time,
            y=self.data[:, 1],
            symbolBrush=[pg.intColor(int(ID), self.n_colours) for ID in self.data[:, 1]],
        )


# ------------------------------------------------------------------------------------------


class AnalogPlot:
    def __init__(self, parent=None, data_dur=10):
        self.task_plot = parent
        self.data_dur = data_dur
        self.axis = _ScrollFriendlyPlotWidget(title="Analog")
        self.axis.showAxis("right")
        self.axis.hideAxis("left")
        self.axis.setRange(xRange=[-10.2, 0], padding=0)
        self.axis.setMouseEnabled(x=True, y=False)
        self.axis.showGrid(x=True, alpha=0.75)
        self.axis.setLimits(xMax=0)
        self.inputs = {}

    def set_state_machine(self, sm_info):
        self.inputs = {ID: ai for ID, ai in sm_info.analog_inputs.items() if ai.get("plot", True)}
        if not self.inputs:
            return  # State machine may not have analog inputs.
        self.axis.clear()
        self.legend = self.axis.addLegend(offset=(10, 10))
        self.plots = {
            ID: self.axis.plot(name=ai["name"], pen=pg.mkPen(pg.intColor(i, len(self.inputs))))
            for i, (ID, ai) in enumerate(sorted(self.inputs.items()))
        }
        self.axis.getAxis("bottom").setLabel("Time (seconds)")
        self.axis.getAxis("right").setWidth(self.task_plot.axiswidth)

    def run_start(self):
        if not self.inputs:
            return  # State machine may not have analog inputs.
        for plot in self.plots.values():
            plot.clear()
        self.data = {ID: np.zeros([ai["fs"] * self.data_dur, 2]) for ID, ai in self.inputs.items()}
        self.updated_inputs = []

    def process_data(self, new_data):
        """Store new data from board."""
        if not self.inputs:
            return  # State machine may not have analog inputs.
        new_analog = [nd for nd in new_data if nd.type == MsgType.ANLOG]
        for na in new_analog:
            ID, data = na.content
            if ID in self.plots:
                new_len = len(data)
                t = na.time / 1000 + np.arange(new_len) / self.inputs[ID]["fs"]
                self.data[ID] = np.roll(self.data[ID], -new_len, axis=0)
                self.data[ID][-new_len:, :] = np.vstack([t, data]).T

    def update(self, run_time):
        """Update plots."""
        if not self.inputs:
            return  # State machine may not have analog inputs.
        for ID in self.inputs:
            self.plots[ID].setData(x=self.data[ID][:, 0] - run_time, y=self.data[ID][:, 1])


# -----------------------------------------------------


class RunClock:
    # Class for displaying the run time.

    def __init__(self, axis):
        self.clock_text = pg.TextItem(text="")
        self.clock_text.setFont(QtGui.QFont("arial", 11, QtGui.QFont.Weight.Bold))
        axis.getViewBox().addItem(self.clock_text, ignoreBounds=True)
        self.clock_text.setParentItem(axis.getViewBox())
        self.clock_text.setPos(10, -5)
        self.recording_text = pg.TextItem(text="", color=(255, 0, 0))
        self.recording_text.setFont(QtGui.QFont("arial", 12, QtGui.QFont.Weight.Bold))
        axis.getViewBox().addItem(self.recording_text, ignoreBounds=True)
        self.recording_text.setParentItem(axis.getViewBox())
        self.recording_text.setPos(80, -5)

    def update(self, run_time):
        # Same HH:MM:SS formatter as the box-card / LiveStatus clocks so all
        # session clocks read identically. ``run_time`` is fw seconds.
        self.clock_text.setText(format_run_clock(run_time * 1000))

    def recording(self):
        self.recording_text.setText("Recording")

    def run_stop(self):
        self.clock_text.setText("")
        self.recording_text.setText("")


# ----------------------------------------------------------------------------------------
# UniversalPlotWindow
# ----------------------------------------------------------------------------------------


class UniversalPlotWindow(QtWidgets.QMainWindow):
    """Session Plot window: every running box stacked in one vertical scroll.

    Each box is a block (header + its ``TaskPlot``) with a minimum height and
    no maximum, so blocks expand to fill a tall window and the area scrolls
    once the stack outgrows the viewport. Boxes are independent; there is no
    shared X-axis between them (a ``TaskPlot`` X-links only its own
    States/Events/Analog lanes)."""

    def __init__(self, parent=None):
        # Deliberately NOT parented, even though callers pass the main window.
        # A parented widget with the Window flag is an *owned* top-level, which
        # Windows pins permanently above its owner: the operator could not put
        # the plots behind the rig UI. Forcing NonModal (as this did before)
        # fixes input blocking but not stacking, because both come from the
        # ownership rather than the flags. See source/gui/window_behavior.py.
        # MainWindowBase.universal_plot_window holds the reference.
        super().__init__(None)
        register_independent_window(self)
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.setWindowTitle("Live Plots")
        self.setGeometry(100, 100, 760, 500)
        self.setMinimumSize(640, 420)

        # Token-driven dark surface so the window has visual hooks instead
        # of the OS-default near-black.
        from source.gui.theme import THEME as _T
        self._tokens = _T
        self.setStyleSheet(
            "QMainWindow {"
            f" background-color: {_T.palette.bg};"
            f" color: {_T.palette.text};"
            "}"
        )

        # Stacked central widget, switches between the empty-state message
        # (when no boxes are running) and the scrolling box stack (when there's
        # data).
        self._central = QtWidgets.QStackedWidget(self)
        self.setCentralWidget(self._central)

        # Empty-state widget, informative placeholder.
        self._empty_state = self._build_empty_state()
        self._central.addWidget(self._empty_state)

        # Scrolling per-box stack. Every running box is one block stacked
        # vertically in a single scroll, so all boxes
        # are watchable at once. Each block carries a minimum height and no
        # maximum: blocks expand to fill when the window is tall, and the area
        # scrolls vertically once the stack outgrows the viewport. Width is one
        # column that follows the window (horizontal scrollbar only if narrow).
        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._blocks_host = QtWidgets.QWidget()
        self._blocks_layout = QtWidgets.QVBoxLayout(self._blocks_host)
        self._blocks_layout.setContentsMargins(8, 8, 8, 8)
        self._blocks_layout.setSpacing(12)
        # No trailing stretch: blocks carry an Expanding policy so they grow to
        # fill spare height rather than clustering at the top.
        self._scroll.setWidget(self._blocks_host)
        self._central.addWidget(self._scroll)

        # Default to the empty state until update_boxes() finds something.
        self._central.setCurrentWidget(self._empty_state)

        self.box_plots = {}     # box_num -> TaskPlot
        self.box_blocks = {}    # box_num -> per-box block container widget
        self.box_headers = {}   # box_num -> header QLabel (for subject relabel)
        self.box_triggers = {}  # box_num -> TriggerPlot (hidden until it fires)
        self.running_boxes = []

        # Persistent one-line interaction hint along the bottom: plain wheel
        # moves through the stacked boxes, Ctrl+wheel zooms a lane's time axis.
        p = self._tokens.palette
        hint = QtWidgets.QLabel(
            "Scroll to move between boxes    ·    "
            "Ctrl + scroll to zoom a plot's time axis")
        hint.setStyleSheet(
            f"QLabel {{ color: {p.text_muted}; font-size: 9pt; padding: 2px 8px; }}")
        self.statusBar().addWidget(hint)
        self.statusBar().setStyleSheet(
            f"QStatusBar {{ background: {p.surface}; "
            f"border-top: 1px solid {p.surface_border}; }}"
            "QStatusBar::item { border: none; }")
        self.statusBar().setSizeGripEnabled(False)

        # Update timer, 10 Hz (100 ms). Display-only; decoupled from
        # process_data, so no effect on recording / TSV / fw timestamps.
        # Parented to self so Qt owns lifetime + thread affinity.
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.timeout.connect(self.update_all_plots)
        self.update_timer.start(100)  # 10 Hz

    def _build_empty_state(self) -> QtWidgets.QWidget:
        """Empty placeholder shown when no boxes are running: a graph glyph
        plus a hint to start a recording."""
        p = self._tokens.palette
        wrap = QtWidgets.QWidget()
        wrap.setStyleSheet(f"QWidget {{ background: {p.bg}; }}")
        outer = QtWidgets.QVBoxLayout(wrap)
        outer.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        outer.setSpacing(8)

        glyph = QtWidgets.QLabel("📈")
        glyph.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        glyph.setStyleSheet(
            f"font-size: 56pt; color: {p.text_dim}; background: transparent;")
        outer.addWidget(glyph)

        title = QtWidgets.QLabel("No live plots yet")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"font: 700 16pt '{self._tokens.font.family}';"
            f" color: {p.text}; background: transparent;")
        outer.addWidget(title)

        hint = QtWidgets.QLabel(
            "Start a box (Record button) and its plots will appear here."
        )
        hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"color: {p.text_muted}; font-size: 10pt;"
            " background: transparent; padding: 0 24px;")
        outer.addWidget(hint)

        return wrap

    def _refresh_central(self):
        """Swap empty-state ↔ scrolling box stack based on box count."""
        if not hasattr(self, "_central"):
            return
        if not self.box_plots:
            self._central.setCurrentWidget(self._empty_state)
        else:
            self._central.setCurrentWidget(self._scroll)

    @staticmethod
    def _block_floor(task_plot, trig_plot=None) -> int:
        """Minimum height for a block, from the content it has to show.

        The states lane draws one y-axis row per state and the events lane one
        per event, so a task with twenty states needs a taller lane than one
        with three. The row STRETCHES were already proportional to those
        counts, so the lanes divided the block correctly; the block itself was
        a constant, which is why a rich task collided its own tick labels.

        A floor, never a cap: blocks stay Expanding, so they grow into spare
        window height, and the stack scrolls once it outgrows the viewport.
        Capping would put the overlap back for exactly the tasks this is for.
        """
        n_states = len(getattr(getattr(task_plot, "states_plot", None),
                               "state_IDs", None) or ())
        n_events = len(getattr(getattr(task_plot, "events_plot", None),
                               "event_IDs", None) or ())

        def lane(rows):
            return max(MIN_LANE_H, rows * PLOT_ROW_H)

        floor = BLOCK_HEADER_H + lane(n_states) + lane(n_events)
        if getattr(getattr(task_plot, "analog_plot", None), "inputs", None):
            floor += ANALOG_LANE_H
        # Never below the two-lane floor: a box whose board has not reported
        # a state machine yet knows no counts, and a block that collapsed to
        # its lane minimums would look broken rather than empty.
        floor = max(MIN_BOX_PLOT_H, floor)
        # ``isHidden``, not ``isVisible``: a child of a window that has not
        # been shown yet is never "visible", so isVisible() would answer for
        # the window rather than for this lane and the floor would miss it.
        # Added after the clamp so revealing the lane always grows the block.
        if trig_plot is not None and not trig_plot.isHidden():
            floor += TRIGGER_LANE_H
        return floor

    def _refresh_block_floor(self, box_num) -> None:
        """Re-apply a block's floor after its visible lanes changed."""
        block = self.box_blocks.get(box_num)
        task_plot = self.box_plots.get(box_num)
        if block is None or task_plot is None:
            return
        floor = self._block_floor(task_plot, self.box_triggers.get(box_num))
        if block.minimumHeight() != floor:
            block.setMinimumHeight(floor)
            # The stack lives in a resizable scroll area, which sizes its
            # widget from the layout's minimum; that has to be recomputed
            # or the new floor is not honoured until some other event
            # happens to relayout.
            if getattr(self, "_blocks_host", None) is not None:
                self._blocks_host.adjustSize()

    def _make_box_block(self, box_num, subject_text, task_plot, trig_plot=None):
        """Build one per-box block (header label + framed ``TaskPlot``).

        Returns ``(block, header_label)``. The block carries a minimum height
        (taller when the task has an analog lane) and an Expanding policy so it
        fills spare window height; the header is kept so the subject label can
        be refreshed without rebuilding the plot."""
        p = self._tokens.palette
        block = QtWidgets.QWidget()
        block.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                            QtWidgets.QSizePolicy.Policy.Expanding)
        # The floor is set at the END of this method, once the lanes are in
        # their starting state: a child widget created with a parent is not
        # hidden until it is told to be, so asking here would count the
        # trigger lane that the next few lines are about to hide.

        v = QtWidgets.QVBoxLayout(block)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        header = QtWidgets.QLabel(f"Box {box_num}{subject_text}")
        header.setStyleSheet(
            "QLabel {"
            f" color: {p.text}; background: {p.surface_elev};"
            f" border: 1px solid {p.surface_border};"
            " border-top-left-radius: 8px; border-top-right-radius: 8px;"
            " font-weight: 700; font-size: 11pt; padding: 6px 12px; }"
        )
        v.addWidget(header)

        plot_frame = QtWidgets.QFrame()
        plot_frame.setStyleSheet(
            "QFrame {"
            f" background: {p.surface}; border: 1px solid {p.surface_border};"
            " border-top: none;"
            " border-bottom-left-radius: 8px; border-bottom-right-radius: 8px; }"
        )
        fl = QtWidgets.QVBoxLayout(plot_frame)
        fl.setContentsMargins(6, 6, 6, 6)
        fl.addWidget(task_plot, 3)
        # Composable-trigger lane, inline under the task plot. Stays hidden
        # until the box actually fires a trigger frame with rules, so boxes
        # without triggers keep a compact block.
        if trig_plot is not None:
            trig_plot.setVisible(False)
            fl.addWidget(trig_plot, 2)
        v.addWidget(plot_frame, 1)

        block.setMinimumHeight(self._block_floor(task_plot, trig_plot))
        return block, header

    def update_boxes(self, running_box_widgets):
        """Update the plot window with currently running boxes.

        This is called each time the Plot button is clicked.
        We add blocks for NEW boxes and remove blocks for stopped boxes,
        keeping the stack ordered by box number.
        """
        # Restart update timer if it was stopped (e.g., after window was closed)
        if not self.update_timer.isActive():
            self.update_timer.start(100)  # 10 Hz

        # Get current and new box numbers (use set to avoid duplicates)
        new_box_numbers = {box.setup_number for box in running_box_widgets}
        old_box_numbers = set(self.box_plots.keys())

        logger.info(f"Plot update: old_boxes={old_box_numbers}, new_boxes={new_box_numbers}")

        # Remove blocks for boxes that stopped
        for box_num in (old_box_numbers - new_box_numbers):
            task_plot = self.box_plots[box_num]

            # Unregister from data consumer
            for setup_widget in self.running_boxes:
                if setup_widget.setup_number == box_num:
                    if hasattr(setup_widget.pycboard, 'data_consumers') and task_plot in setup_widget.pycboard.data_consumers:
                        setup_widget.pycboard.data_consumers.remove(task_plot)
                        logger.info(f"Unregistered plot for stopped box {box_num}")
                    break

            task_plot.run_stop()
            self._remove_box_block(box_num)
            del self.box_plots[box_num]

        # Add blocks for NEW boxes only (running boxes without a block yet)
        for setup_widget in running_box_widgets:
            box_num = setup_widget.setup_number

            if box_num in self.box_plots:
                # Block already exists, refresh its header subject label.
                self._set_box_header(box_num, self._subject_text(setup_widget))
                continue

            logger.info(f"Creating new plot block for box {box_num}")

            # Create new TaskPlot
            task_plot = TaskPlot(parent=self)

            # Set state machine info
            if hasattr(setup_widget, 'pycboard') and setup_widget.pycboard:
                if hasattr(setup_widget.pycboard, 'sm_info') and setup_widget.pycboard.sm_info:
                    task_plot.set_state_machine(setup_widget.pycboard.sm_info)

                # CRITICAL: Register as data consumer
                if not hasattr(setup_widget.pycboard, 'data_consumers'):
                    setup_widget.pycboard.data_consumers = []

                setup_widget.pycboard.data_consumers.append(task_plot)
                logger.info(f"Registered plot for box {box_num}, total consumers: {len(setup_widget.pycboard.data_consumers)}")

            # Drive the sweep + clock off this box's MCU framework time.
            recording = getattr(setup_widget, 'recording_data', False)
            task_plot.run_start(recording=recording,
                                clock=_fw_clock_for(setup_widget))

            # Inline trigger lane, on the same FW clock as the task plot so
            # the raster/signal time axes align with the session timeline.
            trig_plot = TriggerPlot(parent=self, window_s=10.0,
                                    clock=_fw_clock_for(setup_widget))
            self.box_triggers[box_num] = trig_plot

            # Build the block and insert it in box-number order.
            self.box_plots[box_num] = task_plot
            self._add_box_block(box_num, self._subject_text(setup_widget),
                                task_plot, trig_plot)

        self.running_boxes = running_box_widgets
        # Swap to the empty-state placeholder if no boxes left, or
        # back to the box stack as soon as one shows up.
        self._refresh_central()

    @staticmethod
    def _subject_text(setup_widget):
        """`` - <subject>`` suffix for a box header, or '' when none set."""
        try:
            if hasattr(setup_widget, "subject_id_edit"):
                subj = setup_widget.subject_id_edit.text().strip()
                if subj:
                    return f" - {subj}"
        except Exception:
            pass
        return ""

    def _add_box_block(self, box_num, subject_text, task_plot, trig_plot=None):
        """Insert a box's block into the stack, ordered by box number."""
        block, header = self._make_box_block(box_num, subject_text, task_plot,
                                             trig_plot)
        index = sum(1 for n in self.box_blocks if n < box_num)
        self._blocks_layout.insertWidget(index, block)
        self.box_blocks[box_num] = block
        self.box_headers[box_num] = header

    def _remove_box_block(self, box_num):
        """Remove and delete a box's block (its ``TaskPlot`` child too)."""
        block = self.box_blocks.pop(box_num, None)
        self.box_headers.pop(box_num, None)
        trig = self.box_triggers.pop(box_num, None)
        if trig is not None:
            trig.stop()
        if block is not None:
            self._blocks_layout.removeWidget(block)
            block.setParent(None)
            block.deleteLater()

    def _set_box_header(self, box_num, subject_text):
        """Refresh a box block's header text (subject may have been typed in)."""
        header = self.box_headers.get(box_num)
        if header is not None:
            header.setText(f"Box {box_num}{subject_text}")

    def update_all_plots(self):
        """Update all plots - called by the 10 Hz timer.

        Only updates plots for boxes that are still running.
        """
        by_num = {box.setup_number: box for box in self.running_boxes}
        for box_num, task_plot in list(self.box_plots.items()):
            setup_widget = by_num.get(box_num)
            # Only update if box is still running AND not paused
            if setup_widget and setup_widget.framework_running and not task_plot.pause_button.isChecked():
                task_plot.update()

    def push_trigger_frame(self, setup_id, trigger_frame):
        """Feed a box's TriggerFrame into its inline trigger lane. Reveals the
        lane the first time a frame carries rule states, so boxes without
        triggers stay compact. Called from the GUI ``trigger_ready`` relay."""
        trig = self.box_triggers.get(int(setup_id))
        if trig is None:
            return
        try:
            trig.push(trigger_frame)
            if getattr(trigger_frame, "states", None) and trig.isHidden():
                trig.setVisible(True)
                # A third lane just appeared in a block sized for two.
                self._refresh_block_floor(int(setup_id))
        except Exception as e:
            logger.debug("push_trigger_frame (box=%s): %s", setup_id, e)

    def closeEvent(self, event):
        """Handle window close - stop timer and unregister all plots."""
        # Stop update timer
        if hasattr(self, 'update_timer') and self.update_timer:
            self.update_timer.stop()

        # Unregister all plots from data consumers
        for box_num, task_plot in list(self.box_plots.items()):
            for setup_widget in self.running_boxes:
                if setup_widget.setup_number == box_num:
                    if hasattr(setup_widget, 'pycboard') and setup_widget.pycboard:
                        if hasattr(setup_widget.pycboard, 'data_consumers') and setup_widget.pycboard.data_consumers:
                            if task_plot in setup_widget.pycboard.data_consumers:
                                setup_widget.pycboard.data_consumers.remove(task_plot)
                                logger.info(f"Unregistered plot for box {box_num} on window close")
                    break
            task_plot.run_stop()

        # CRITICAL: Delete the actual block widgets, not just the dictionary!
        for box_num in list(self.box_blocks.keys()):
            self._remove_box_block(box_num)
        self.box_plots.clear()
        self.running_boxes.clear()
        self._refresh_central()

        logger.info("Plot window closed, all blocks cleared")
        event.accept()



# ── Trigger live plot (T3) ────────────────────────────────────────────────

class TriggerPlot(QtWidgets.QWidget):
    """Real-time view of one box's composable triggers.

    Two stacked, toggleable pyqtgraph lanes fed from a
    :class:`~source.video.trigger_engine.TriggerHistory`:

    * **raster**: one row per rule, a coloured bar for each active interval
      over a scrolling ``window_s`` window ("what fired, when");
    * **signal**: for scalar rules (speed / distance / region) the value
      trace with its threshold line ("why it fired").

    Fed by ``push(TriggerFrame)`` from the GUI's ``trigger_ready`` slot; a timer
    coalesces redraws to ~30 Hz so a fast camera never floods the UI. Detachable
    simply by parenting it into its own window. ``clock`` is a zero-arg callable
    → seconds (default wall/monotonic); pass the box FW clock to align with the
    session timeline.
    """

    def __init__(self, parent=None, *, window_s: float = 10.0, clock=None):
        super().__init__(parent)
        import time as _time
        from source.video.trigger_engine import TriggerHistory
        self._history = TriggerHistory(window_s)
        self._clock = clock or _time.monotonic

        root = QtWidgets.QVBoxLayout(self)
        # controls
        ctl = QtWidgets.QHBoxLayout()
        ctl.addWidget(QtWidgets.QLabel("Window"))
        self._win_spin = QtWidgets.QSpinBox()
        self._win_spin.setRange(2, 120); self._win_spin.setValue(int(window_s))
        self._win_spin.setSuffix(" s")
        self._win_spin.valueChanged.connect(self._on_window_changed)
        ctl.addWidget(self._win_spin)
        self._raster_chk = QtWidgets.QCheckBox("Raster"); self._raster_chk.setChecked(True)
        self._signal_chk = QtWidgets.QCheckBox("Signal"); self._signal_chk.setChecked(True)
        self._raster_chk.toggled.connect(self._apply_visibility)
        self._signal_chk.toggled.connect(self._apply_visibility)
        ctl.addWidget(self._raster_chk); ctl.addWidget(self._signal_chk)
        ctl.addStretch(1)
        root.addLayout(ctl)

        self._raster = _ScrollFriendlyPlotWidget()
        self._signal = _ScrollFriendlyPlotWidget()
        for pw in (self._raster, self._signal):
            pw.setBackground("#0e1117")
            pw.getPlotItem().showGrid(x=True, y=False, alpha=0.15)
            pw.setMouseEnabled(x=True, y=False)
        self._raster.getPlotItem().setLabel("left", "rules")
        self._signal.getPlotItem().setLabel("left", "value")
        self._signal.getPlotItem().setLabel("bottom", "time", units="s")
        root.addWidget(self._raster, 2)
        root.addWidget(self._signal, 2)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._redraw)
        self._timer.start(33)   # ~30 Hz coalesced repaint

    # ── feed ────────────────────────────────────────────────────────────
    def push(self, trigger_frame) -> None:
        """Append a TriggerFrame (called from the trigger_ready slot)."""
        try:
            self._history.push(trigger_frame, float(self._clock()))
        except Exception as e:
            logger.debug("TriggerPlot.push: %s", e)

    def set_window_seconds(self, s: float) -> None:
        self._history.window_s = float(s)

    # ── options ─────────────────────────────────────────────────────────
    def _on_window_changed(self, v):
        self._history.window_s = float(v)

    def _apply_visibility(self, *_):
        self._raster.setVisible(self._raster_chk.isChecked())
        self._signal.setVisible(self._signal_chk.isChecked())

    # ── redraw ──────────────────────────────────────────────────────────
    @staticmethod
    def _pen(color):
        return pg.mkPen(color=color, width=8)

    def _redraw(self):
        now = float(self._clock())
        self._history.prune(now)
        lo = now - self._history.window_s
        ids = self._history.rule_ids()

        if not ids:
            # Nothing to draw, skip the full clear+replot every tick
            # until a rule appears (one final clear on the transition to
            # empty so stale segments don't linger).
            if not getattr(self, "_drew_empty", False):
                self._raster.getPlotItem().clear()
                self._signal.getPlotItem().clear()
                self._drew_empty = True
            return
        self._drew_empty = False

        if self._raster_chk.isChecked():
            self._raster.getPlotItem().clear()
            ticks = []
            for i, rid in enumerate(ids):
                name, color, _has_val, _thr = self._history.meta(rid)
                ticks.append((i, name))
                for (t0, t1) in self._history.active_segments(rid, now):
                    self._raster.getPlotItem().plot([t0, t1], [i, i], pen=self._pen(color))
            ax = self._raster.getPlotItem().getAxis("left")
            ax.setTicks([ticks] if ticks else [])
            self._raster.setXRange(lo, now, padding=0)
            self._raster.setYRange(-0.5, max(0.5, len(ids) - 0.5), padding=0.1)

        if self._signal_chk.isChecked():
            self._signal.getPlotItem().clear()
            for rid in ids:
                name, color, has_val, thr = self._history.meta(rid)
                if not has_val:
                    continue
                pts = [(t, v) for (t, _a, v) in self._history.series(rid) if v is not None]
                if pts:
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    self._signal.getPlotItem().plot(xs, ys, pen=pg.mkPen(color=color, width=2))
                if thr is not None:
                    line = pg.InfiniteLine(pos=thr, angle=0,
                                           pen=pg.mkPen(color=color, style=QtCore.Qt.PenStyle.DashLine))
                    self._signal.getPlotItem().addItem(line)
            self._signal.setXRange(lo, now, padding=0)

    def stop(self):
        self._timer.stop()
