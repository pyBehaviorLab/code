# devices/__init__.py
# Import hardware primitives from pyControl
import os
from pyControl.hardware import Digital_input, Digital_output, Analog_input, Rsync, off

# Dynamically import all device classes from device driver files
_driver_files = [f.split(".")[0] for f in os.listdir("devices") if "init" not in f]

for _driver_file in _driver_files:
    exec("from devices.{} import *".format(_driver_file))
