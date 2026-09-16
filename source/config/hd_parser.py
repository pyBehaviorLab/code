"""
Parse hardware_definition.py files to extract device information.

Reads files like pyMultimaze_F767.py and extracts:
- Digital inputs (limit switches) with event names
- Stepper motors (Stepper_motor, TMC_motor, Door) with port/motor assignments
- Motor enable pins (Digital_output)

Supports motor styles:
  Stepper_motor(port=board.port_N), direction+step on port DIO_A/DIO_B
  TMC_motor(motor=board.motorN), dedicated DIR/STEP/EN header (has built-in enable)
  Door(motor=board.motorN, ...), composite device with ISR limit protection
  Door(motor_port=board.port_N, ...)
"""

import re
import os
from dataclasses import dataclass, field


@dataclass
class DeviceInfo:
    name: str
    device_type: str  # 'Digital_input', 'Digital_output', 'Stepper_motor', 'TMC_motor', 'Door'
    pin: str = ''
    port: str = ''
    motor: str = ''  # motor header name for TMC_motor/Door (e.g. 'motor1')
    events: dict = field(default_factory=dict)  # {'rising': ..., 'falling': ..., 'open': ..., 'close': ...}
    params: dict = field(default_factory=dict)  # Door params: max_steps, step_rate, etc.


@dataclass
class HardwareConfig:
    digital_inputs: list = field(default_factory=list)
    stepper_motors: list = field(default_factory=list)  # Stepper_motor, TMC_motor, Door
    enables: list = field(default_factory=list)          # enable pins for motors


def parse_hardware_definition(filepath):
    """Parse a hardware_definition.py file and return HardwareConfig."""
    config = HardwareConfig()

    if not os.path.isfile(filepath):
        return config

    with open(filepath, 'r') as f:
        content = f.read()

    # Extract Digital_input declarations
    # Supports both pin=board.xxx and pin=board.port_N.DIO_X forms,
    # and positional first arg (board.xxx, ...).
    # Events are extracted order-independently from the arg body.
    di_base = re.compile(
        r'(\w+)\s*=\s*Digital_input\s*\(([^)]+)\)'
    )
    _pin_kw = re.compile(r'pin\s*=\s*board\.([\w.]+)')
    _pin_pos = re.compile(r'^\s*board\.([\w.]+)')
    _rising = re.compile(r'rising_event\s*=\s*[\'"](\w+)[\'"]')
    _falling = re.compile(r'falling_event\s*=\s*[\'"](\w+)[\'"]')

    for m in di_base.finditer(content):
        name = m.group(1)
        args = m.group(2)
        # Extract pin (keyword or positional)
        pin_m = _pin_kw.search(args) or _pin_pos.search(args)
        if not pin_m:
            continue
        dev = DeviceInfo(
            name=name,
            device_type='Digital_input',
            pin=pin_m.group(1),
        )
        rise_m = _rising.search(args)
        fall_m = _falling.search(args)
        if rise_m:
            dev.events['rising'] = rise_m.group(1)
        if fall_m:
            dev.events['falling'] = fall_m.group(1)
        config.digital_inputs.append(dev)

    # Extract Stepper_motor declarations
    # Pattern: name = Stepper_motor(port=board.port_N)
    sm_pattern = re.compile(
        r'(\w+)\s*=\s*Stepper_motor\s*\(\s*port\s*=\s*board\.(\w+)\s*\)'
    )
    for m in sm_pattern.finditer(content):
        dev = DeviceInfo(
            name=m.group(1),
            device_type='Stepper_motor',
            port=m.group(2),
        )
        config.stepper_motors.append(dev)

    # Extract TMC_motor declarations
    # Pattern: name = TMC_motor(motor=board.motorN)
    tmc_pattern = re.compile(
        r'(\w+)\s*=\s*TMC_motor\s*\(\s*motor\s*=\s*board\.(\w+)\s*\)'
    )
    for m in tmc_pattern.finditer(content):
        dev = DeviceInfo(
            name=m.group(1),
            device_type='TMC_motor',
            motor=m.group(2),
        )
        config.stepper_motors.append(dev)

    # Extract Door composite device declarations
    # Door(...) can span multiple lines and contain nested parens like mcp.Pin('A0').
    # Match balanced parentheses: allow one level of nesting inside Door(...).
    _door_block = re.compile(
        r'(\w+)\s*=\s*Door\s*\(((?:[^()]*|\([^()]*\))*)\)',
        re.DOTALL
    )
    _motor_kw = re.compile(r'(?<!\w)motor\s*=\s*board\.(\w+)')
    _motor_port_kw = re.compile(r'motor_port\s*=\s*board\.(\w+)')
    # Limit pins: board.port_N.DIO_X or mcp.Pin('A0')
    _open_pin = re.compile(r'limit_open_pin\s*=\s*(?:board\.([\w.]+)|(\w+)\.Pin\([\'"](\w+)[\'"]\))')
    _close_pin = re.compile(r'limit_close_pin\s*=\s*(?:board\.([\w.]+)|(\w+)\.Pin\([\'"](\w+)[\'"]\))')
    _open_event = re.compile(r'open_event\s*=\s*[\'"](\w+)[\'"]')
    _close_event = re.compile(r'close_event\s*=\s*[\'"](\w+)[\'"]')
    _max_steps = re.compile(r'max_steps\s*=\s*(\d+)')
    _step_rate = re.compile(r'step_rate\s*=\s*(\d+)')

    for m in _door_block.finditer(content):
        name = m.group(1)
        args = m.group(2)

        dev = DeviceInfo(
            name=name,
            device_type='Door',
        )

        # Motor source: TMC header or port
        mm = _motor_kw.search(args)
        mp = _motor_port_kw.search(args)
        if mm:
            dev.motor = mm.group(1)
        elif mp:
            dev.port = mp.group(1)

        # Limit pins, board.port_N.DIO_X or mcp.Pin('A0')
        op = _open_pin.search(args)
        cp = _close_pin.search(args)
        if op:
            if op.group(1):  # board.xxx match
                dev.params['limit_open_pin'] = op.group(1)
            else:  # mcp.Pin('X') match
                dev.params['limit_open_pin'] = f"{op.group(2)}.Pin('{op.group(3)}')"
        if cp:
            if cp.group(1):
                dev.params['limit_close_pin'] = cp.group(1)
            else:
                dev.params['limit_close_pin'] = f"{cp.group(2)}.Pin('{cp.group(3)}')"

        # Events
        oe = _open_event.search(args)
        ce = _close_event.search(args)
        if oe:
            dev.events['open'] = oe.group(1)
        if ce:
            dev.events['close'] = ce.group(1)

        # Stepper config
        ms = _max_steps.search(args)
        sr = _step_rate.search(args)
        if ms:
            dev.params['max_steps'] = int(ms.group(1))
        if sr:
            dev.params['step_rate'] = int(sr.group(1))

        config.stepper_motors.append(dev)

    # Extract Digital_output declarations. The pin may be a board alias
    # (pin=board.xxx) or a raw string (pin='PD9'), and extra kwargs such as
    # inverted=True may follow, motor enables use exactly that form.
    do_pattern = re.compile(
        r"(\w+)\s*=\s*Digital_output\s*\(\s*pin\s*=\s*"
        r"(?:board\.([\w.]+)|['\"]([\w.]+)['\"])"
        r"[^)]*\)"
    )
    for m in do_pattern.finditer(content):
        if 'enable' in m.group(1).lower():
            config.enables.append(DeviceInfo(
                name=m.group(1),
                device_type='Digital_output',
                pin=m.group(2) or m.group(3),
            ))

    return config
