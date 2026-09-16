"""Example user API class, ported from pyControl_v0.

To use this with a task, add inside the task .py file:

    v.api_class = "Example_user_class"

The class must have the SAME NAME as the file (without .py) and inherit
``Api``. See ``source/gui/api.py`` for the methods you can override.
"""

import random

from source.communication.api import Api


class Example_user_class(Api):
    def __init__(self):
        self.off_count = 0

    def run_start(self):
        self.print_to_log("\nMessage from api_classes/Example_user_class.py "
                          "at the start of the run")

    def run_stop(self):
        self.print_to_log("\nMessage from api_classes/Example_user_class.py "
                          "at the end of the run")

    def process_data_user(self, data):
        # Count transitions into the LED_off state and randomise blink rate.
        LED_off_happened = any(state.name == "LED_off"
                               for state in data["states"])
        if LED_off_happened:
            self.off_count += 1
            new_duration = random.triangular(0.1, 1, 5.5)
            self.set_variable("LED_duration", round(new_duration, 3))
            if self.off_count % 4 == 0:
                self.trigger_event("event_a")

        # Look for prints from the task of the form ``vals_from_task=x,y,z``.
        msgs_from_task = [printed.data.split("=")[1]
                          for printed in data["prints"]
                          if "vals_from_task=" in printed.data]
        for msg in msgs_from_task:
            x, y, z = msg.split(",")
            total = int(x) + int(y) + int(z)
            self.print_message(f"{x} and {y} and {z} total to {total}")
