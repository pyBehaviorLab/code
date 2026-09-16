"""User API base class.

Lets a task ship a desktop-side Python class that runs alongside the MCU
state machine. The task opts in by declaring a variable

    v.api_class = "MyApiName"

inside the task .py file. ``RunTask.initialise_API`` then imports
``api_classes.MyApiName`` and instantiates the class ``MyApiName`` from
that module. The instance is inserted into ``pycboard.data_consumers``
so it receives every state / event / print / variable / analog message
in real time, and can call back into the board via ``set_variable``,
``trigger_event``, ``print_message``.

The subclass overrides any of ``run_start``, ``run_stop``,
``process_data_user``, ``plot_update``. ``interface`` and
``process_data`` are framework hooks, don't override them.
"""

import json
from collections import namedtuple

from source.communication.message import MsgType


class Api:
    # ----------------------------------------------------------
    # Functions the user can override
    # ----------------------------------------------------------

    def __init__(self):
        """Called once when the task is uploaded (before ``interface``)."""

    def run_start(self):
        """Called once when the framework starts."""

    def run_stop(self):
        """Called once when the framework stops."""

    def process_data_user(self, data):
        """Called whenever the MCU sends new data.

        ``data`` is a dict with keys ``states``, ``events``, ``prints``,
        ``vars``, ``analog``. Each is a list of namedtuples:
            State(name, time)        Event(name, time)
            Print(data, time)        Var(name, value, time)
            Analog(name, data, time)
        """

    def plot_update(self):
        """Called every plot tick (~10 ms)."""

    # ----------------------------------------------------------
    # Methods the user can call from their overrides
    # ----------------------------------------------------------

    def set_variable(self, v_name, v_value):
        """Set a task variable on the running MCU."""
        if v_name in self.board.sm_info.variables:
            self.board.set_variable(v_name, v_value, source="a")
        else:
            self.print_to_log(
                f"Variable {v_name} not defined in task file "
                f"{self.board.sm_info.name} so cannot be set by API"
            )

    def trigger_event(self, event):
        """Fire an event on the running MCU."""
        if event in self.board.sm_info.events:
            self.board.trigger_event(event, "a")
        else:
            self.print_to_log(
                f"Event {event} not defined in task file "
                f"{self.board.sm_info.name} so cannot be set by API"
            )

    def print_message(self, msg):
        """Append a message to the data log."""
        self.board.data_logger.print_message(msg, "a")

    # Note: get_variable is not exposed because pycboard.get_variable does
    # not return a value when the framework is running, it just asks the
    # board to print it out. Use process_data_user to track variables via
    # the data stream instead.

    # ----------------------------------------------------------
    # Framework hooks, do NOT override or call from user code
    # ----------------------------------------------------------

    def interface(self, board, print_to_log):
        """Called by ``RunTask.initialise_API`` after the state machine
        is uploaded, before any data starts flowing. Wires the api to
        the board and caches ID-to-name maps."""
        self.board = board
        self.print_to_log = print_to_log
        self.ID2name = self.board.sm_info.ID2name
        self.ID2analog = {}
        for ID, info in (self.board.sm_info.analog_inputs or {}).items():
            self.ID2analog[ID] = info["name"]

        self.event_tup  = namedtuple("Event",  "name time")
        self.state_tup  = namedtuple("State",  "name time")
        self.print_tup  = namedtuple("Print",  "data time")
        self.var_tup    = namedtuple("Var",    "name value time")
        self.analog_tup = namedtuple("Analog", "name data time")

    def process_data(self, new_data):
        """Called by pycboard every time new data arrives. Parses the
        raw datatuples into the user-friendly ``data`` dict and dispatches
        to ``process_data_user``."""
        data = {"states": [], "events": [], "prints": [],
                "vars": [], "analog": []}

        for nd in new_data:
            if nd.type == MsgType.PRINT:
                data["prints"].append(self.print_tup(nd.content, nd.time))
            elif nd.type == MsgType.VARBL:
                var_change_dict = json.loads(nd.content)
                name = next(iter(var_change_dict.keys()))
                value = next(iter(var_change_dict.values()))
                data["vars"].append(self.var_tup(name, value, nd.time))
            elif nd.type == MsgType.STATE:
                name = self.ID2name.get(nd.content, str(nd.content))
                data["states"].append(self.state_tup(name, nd.time))
            elif nd.type == MsgType.EVENT:
                name = self.ID2name.get(nd.content, str(nd.content))
                data["events"].append(self.event_tup(name, nd.time))
            elif nd.type == MsgType.ANLOG:
                aname = self.ID2analog.get(nd.content[0], str(nd.content[0]))
                data["analog"].append(self.analog_tup(aname, nd.content[1], nd.time))

        self.process_data_user(data)
