"""Door Control Dialog, manual door control when the MCU is connected
but the framework is NOT running.

Door device layout:   # | Door Name | OPEN | CLOSE | HOME | STOP
Stepper layout:       # | Door Name | UP   | DOWN  | STOP

Detects Door devices from cached _hw_config (set during upload) or a
user-selected hardware_definition.py. Auto-imports the hw module on the
MCU when the dialog opens.
"""

import os
from pathlib import Path
from PySide6 import QtCore, QtWidgets
from source.communication.pyboard import PyboardError
from source.log import get_logger

logger = get_logger()


def safe_mcu_exec(pycboard, code, *, parent_widget=None, silent=False):
    """Run a MicroPython snippet on the MCU. Never raises.

    Returns ``(ok: bool, result_or_error: str)``. Logs every failure;
    pops a QMessageBox.warning when ``parent_widget`` is set and
    ``silent`` is False. Used by door buttons so a single bad command
    doesn't crash the dialog.
    """
    if pycboard is None:
        msg = "MCU not connected"
        logger.warning("safe_mcu_exec: %s (code=%r)", msg, code)
        return False, msg
    try:
        result = pycboard.exec(code)
        return True, result
    # PyboardError subclasses BaseException, so a bare Exception clause
    # would let board errors escape into the Qt event loop.
    except (Exception, PyboardError) as e:
        msg = f"{type(e).__name__}: {e}"
        logger.error("safe_mcu_exec failed (code=%r): %s", code, msg)
        if parent_widget is not None and not silent:
            try:
                QtWidgets.QMessageBox.warning(
                    parent_widget, "MCU Error",
                    f"Could not run command on MCU:\n{msg}\n\nCommand: {code}")
            except Exception:
                pass
        return False, msg


class DoorControlDialog(QtWidgets.QDialog):
    """Dialog for direct door control.

    Requires:
    - MCU connected
    - Framework NOT running
    - Hardware definition uploaded to MCU

    Detects Door device instances from the parsed hw def, auto-imports hw
    on the MCU, shows OPEN / CLOSE / HOME / STOP per door.
    """

    def __init__(self, main_window, setup_id, parent=None):
        super().__init__(parent or main_window)
        self.main_window = main_window
        self.setup_id = setup_id
        self._step_rate = 1500
        self._n_steps = 4000
        self._doors = []
        self._has_door_devices = False

        self.setWindowTitle(f"Door Control, Setup {setup_id}")
        self.setMinimumWidth(580)
        # Shared dark-dialog QSS + info groupbox accent.
        from source.gui.style_builders import apply_dialog_theme
        from source.gui.style_builders import groupbox_style
        apply_dialog_theme(self, extras=groupbox_style("info"))

        self._build_ui()
        self._detect_doors()

    def keyPressEvent(self, event):
        # Enter closes the door panel; it must NOT trigger a motor action
        # (OPEN/CLOSE/HOME/STOP/ENABLE). Other keys behave normally.
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Status ──
        self._status_label = QtWidgets.QLabel("Detecting doors...")
        self._status_label.setStyleSheet(
            "color: #fbbf24; font-size: 9pt; padding: 4px; font-weight: bold;")
        layout.addWidget(self._status_label)

        # ── Stepper config (hidden for Door-only setups) ──
        self._config_group = QtWidgets.QGroupBox("Stepper Config")
        config_layout = QtWidgets.QHBoxLayout(self._config_group)
        config_layout.setSpacing(10)

        from source.gui.style_builders import spinbox_style as _spin_st
        _spin_qss = _spin_st()

        config_layout.addWidget(QtWidgets.QLabel("Step Rate:"))
        self._rate_spin = QtWidgets.QSpinBox()
        self._rate_spin.setRange(100, 20000)
        self._rate_spin.setValue(self._step_rate)
        self._rate_spin.setStyleSheet(_spin_qss)
        self._rate_spin.valueChanged.connect(self._on_rate_changed)
        config_layout.addWidget(self._rate_spin)

        config_layout.addWidget(QtWidgets.QLabel("Steps:"))
        self._steps_spin = QtWidgets.QSpinBox()
        self._steps_spin.setRange(100, 50000)
        self._steps_spin.setValue(self._n_steps)
        self._steps_spin.setStyleSheet(_spin_qss)
        self._steps_spin.valueChanged.connect(self._on_steps_changed)
        config_layout.addWidget(self._steps_spin)

        config_layout.addStretch()
        layout.addWidget(self._config_group)

        # ── Door grid ──
        self._door_group = QtWidgets.QGroupBox("Doors")
        self._door_layout = QtWidgets.QGridLayout(self._door_group)
        self._door_layout.setSpacing(6)

        # Placeholder
        self._no_doors_label = QtWidgets.QLabel(
            "No doors detected.\nUse 'Load HW Def' or upload a task with HW Def selected.")
        self._no_doors_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        from source.gui.theme import THEME as _T_d
        self._no_doors_label.setStyleSheet(
            f"color: {_T_d.palette.text_dim}; padding: 20px;")
        self._door_layout.addWidget(self._no_doors_label, 0, 0, 1, 6)

        layout.addWidget(self._door_group)

        # ── Bottom: Load HW + Close ──
        bottom = QtWidgets.QHBoxLayout()

        detect_btn = QtWidgets.QPushButton("Load HW Def")
        detect_btn.setToolTip("Select a hardware_definition.py to detect doors")
        detect_btn.setStyleSheet(self._btn_style('#4b5563', '#374151'))
        detect_btn.clicked.connect(self._detect_from_file)
        bottom.addWidget(detect_btn)

        bottom.addStretch()

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setStyleSheet(self._btn_style('#4b5563', '#374151'))
        close_btn.clicked.connect(self.close)
        bottom.addWidget(close_btn)

        layout.addLayout(bottom)

    # ── Detection ────────────────────────────────────────────────

    def _detect_doors(self):
        """Auto-detect doors from cached hw config or hw_def_path."""
        sw = self.main_window._setup_widget_for(self.setup_id)
        if not sw:
            self._status_label.setText("Setup not found.")
            return

        if not sw.is_connected:
            self._status_label.setText("MCU not connected. Connect first.")
            return

        # Cached parsed config (set during task upload).
        hw_config = getattr(sw, '_hw_config', None)
        if hw_config:
            self._build_doors_from_config(hw_config)
            self._import_hw_silent()
            return

        # Cached file path.
        hw_path = getattr(sw, '_hw_def_path', '')
        if hw_path and os.path.isfile(hw_path):
            self._load_hw_file(hw_path)
            self._import_hw_silent()
            return

        # Last fallback: resolve the UPLOADED HD file path host-side and parse
        # it, NO board query. Doors come from the hardware definition the user
        # uploaded, not from introspecting the live MCU.
        hd_path = self._resolve_hd_path_for_box(sw)
        if hd_path:
            self._load_hw_file(hd_path)
            self._import_hw_silent()
            return

        self._status_label.setText(
            "No hardware definition found. Upload a Hardware Definition "
            "(or click 'Load HW Def').")

    def _resolve_hd_path_for_box(self, sw) -> str:
        """Return the uploaded HD file path for this box, host-side only.

        Order: the active project config's per-box ``init_hw_def`` (stored
        relative to ``top_dir``), then the board's recorded ``_loaded_hwd_path``
        from this session. Never queries the MCU. Returns "" when unresolved.
        """
        mw = self.main_window
        cfg = getattr(mw, "_active_config", None)
        if cfg is not None:
            for box in getattr(cfg.setup_config, "boxes", None) or []:
                if getattr(box, "setup_number", None) != self.setup_id:
                    continue
                ref = getattr(box, "init_hw_def", None)
                p = getattr(ref, "path", "") if ref else ""
                if p:
                    if not os.path.isabs(p):
                        try:
                            from source.paths import top_dir
                            p = os.path.join(str(top_dir), p)
                        except Exception:
                            pass
                    if os.path.isfile(p):
                        return p
                break
        pyc = (mw.mcu.get(self.setup_id) if hasattr(mw, "mcu") else None) \
            or getattr(sw, "pycboard", None)
        p = getattr(pyc, "_loaded_hwd_path", "") if pyc is not None else ""
        return p if p and os.path.isfile(p) else ""

    def _detect_from_file(self):
        """Let user pick a hardware_definition.py file."""
        hw_dir = str(Path(__file__).resolve().parents[2] / "hardware_definitions")
        if not os.path.isdir(hw_dir):
            hw_dir = str(Path(__file__).resolve().parents[2])

        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Hardware Definition", hw_dir,
            "Python Files (*.py)")
        if filepath:
            self._load_hw_file(filepath)
            # Cache on setup widget
            sw = self.main_window._setup_widget_for(self.setup_id)
            if sw:
                sw._hw_def_path = filepath
            self._import_hw_silent()

    def _load_hw_file(self, filepath):
        """Parse hardware file and build door list."""
        try:
            from source.config.hd_parser import parse_hardware_definition
            config = parse_hardware_definition(filepath)
            sw = self.main_window._setup_widget_for(self.setup_id)
            if sw:
                sw._hw_config = config
            self._build_doors_from_config(config)
        except Exception as e:
            self._status_label.setText(f"Failed to parse: {e}")
            logger.error(f"Door control HW parse error: {e}")

    def _import_hw_silent(self):
        """Auto-import hw module on MCU (no user action needed)."""
        self._exec_commands(["import hardware_definition as hw"])

    # ── Build door list from parsed config ───────────────────────

    def _build_doors_from_config(self, config):
        """Build door list from HardwareConfig.

        Door devices  → open()/close()/home()/stop() API
        TMC_motor     → forward/backward + enable_motor()
        Stepper_motor → forward/backward + separate enable pin
        """
        self._doors = []
        seen = set()

        enable_map = {d.name: d for d in config.enables}
        input_map = {d.name: d for d in config.digital_inputs}

        for motor in config.stepper_motors:
            name = motor.name
            if name in seen:
                continue
            seen.add(name)

            display = name[:-5] if name.endswith('_door') else name
            is_door = (motor.device_type == 'Door')
            is_tmc = (motor.device_type == 'TMC_motor')

            if is_door:
                # Door devices have no built-in enable (unlike TMC_motor); the
                # driver EN is a separate Digital_output in the HD. Match it by
                # name (home_door → HomeDoor_enable) so the EN button energises
                # the right driver. None → EN acts as a pure permission gate.
                base = display.lower()
                en = next((n for n in enable_map
                           if n.lower().startswith(base)), None)
                door = {
                    'display': display,
                    'motor': name,
                    'door_device': True,
                    'enable': en,
                    'events': motor.events,
                    'max_steps': motor.params.get('max_steps', 4000),
                    'step_rate': motor.params.get('step_rate', 1500),
                    'limit_open': motor.params.get('limit_open_pin', ''),
                    'limit_close': motor.params.get('limit_close_pin', ''),
                }
            else:
                en_name = f"{name}_enable"
                door = {
                    'display': display,
                    'motor': name,
                    'door_device': False,
                    'tmc': is_tmc,
                    'enable': en_name if en_name in enable_map else None,
                    'limit_bot': f"{name}_lmt_bot" if f"{name}_lmt_bot" in input_map else None,
                    'limit_top': f"{name}_lmt_top" if f"{name}_lmt_top" in input_map else None,
                }

            # Per-door runtime state used by _refresh_door_buttons:
            #   _state  -- 'up' | 'down' | 'unknown' (initial / after STOP)
            #   _en     -- True when motor coils are energised (EN button)
            #   up_dir  -- 'forward' or 'backward' (which phase = UP)
            door['_state'] = 'unknown'
            door['_en'] = False
            door['up_dir'] = 'forward'
            self._doors.append(door)

        self._has_door_devices = any(d.get('door_device') for d in self._doors)
        all_door = all(d.get('door_device') for d in self._doors)

        # Hide stepper config when all Door devices (params baked into device)
        self._config_group.setVisible(not all_door)

        if self._doors:
            self._populate_doors()
            n_dd = sum(1 for d in self._doors if d.get('door_device'))
            n_leg = len(self._doors) - n_dd
            parts = []
            if n_dd:
                parts.append(f"{n_dd} Door devices")
            if n_leg:
                parts.append(f"{n_leg} stepper motors")
            self._status_label.setText(f"Found {' + '.join(parts)}. Ready.")
            self._status_label.setStyleSheet(
                "color: #34d399; font-size: 9pt; padding: 4px; font-weight: bold;")
        else:
            self._status_label.setText(
                "No doors found in hardware definition.")

    # ── Populate grid ────────────────────────────────────────────

    def _populate_doors(self):
        """Rebuild the door control grid from ``self._doors``."""
        while self._door_layout.count():
            item = self._door_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._no_doors_label.hide()

        has_dd = self._has_door_devices
        cols = self._door_columns(has_dd)
        self._add_door_headers(cols, has_dd)
        for i, door in enumerate(self._doors):
            self._add_door_row(i + 1, i, door, cols, has_dd)
        self._add_all_row(cols, has_dd)

        # Initial gating: a disabled motor blocks UP / DOWN / HOME.
        for door in self._doors:
            self._refresh_door_buttons(door)

    # ── _populate_doors steps ─────────────────────────────────────────

    @staticmethod
    def _door_columns(has_dd: bool) -> dict:
        """Column index per field.

        A Door-device rig gets a HOME column between CLOSE and STOP, which
        pushes STOP, EN and UP= one place right. Deriving the whole map in one
        place keeps that shift from being applied to some columns and not
        others, which shows up as two widgets stacked in one cell.
        """
        cols = {"num": 0, "name": 1, "open": 2, "close": 3}
        cols["home"] = 4 if has_dd else None
        cols["stop"] = 5 if has_dd else 4
        cols["en"] = cols["stop"] + 1
        cols["dir"] = cols["en"] + 1
        return cols

    def _add_door_headers(self, cols: dict, has_dd: bool) -> None:
        """Header row. ``EN`` is the per-motor enable, independent of STOP;
        ``UP=`` picks which motor phase means up, which depends on how the
        coils were wired and is meaningless for a Door device."""
        headers = ["#", "Door"]
        headers += (["OPEN", "CLOSE", "HOME", "STOP"] if has_dd
                    else ["UP", "DOWN", "STOP"])
        headers += ["EN", "UP="]
        for col, text in enumerate(headers):
            lbl = QtWidgets.QLabel(f"<b>{text}</b>")
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #9ca3af; font-size: 9pt;")
            self._door_layout.addWidget(lbl, 0, col)

    def _add_door_row(self, row: int, index: int, door: dict,
                      cols: dict, has_dd: bool) -> None:
        """One door: number, name, the motion buttons, enable, and phase.

        Every widget is stashed back onto ``door`` so ``_refresh_door_buttons``
        can gate them later without re-walking the layout.
        """
        is_dd = door.get('door_device')

        num = QtWidgets.QLabel(str(index + 1))
        num.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        num.setStyleSheet("color: #6b7280; font-size: 9pt;")
        self._door_layout.addWidget(num, row, cols["num"])

        name_lbl = QtWidgets.QLabel(door['display'])
        name_lbl.setStyleSheet(
            "font-weight: bold; font-size: 10pt; padding: 2px 8px;")
        name_lbl.setToolTip(self._door_tooltip(door))
        self._door_layout.addWidget(name_lbl, row, cols["name"])

        open_btn = self._make_action_btn(
            "OPEN" if is_dd else "UP", '#059669', '#047857',
            lambda checked=False, d=door: self._door_open(d))
        self._door_layout.addWidget(open_btn, row, cols["open"])
        door['_up_btn'] = open_btn

        close_btn = self._make_action_btn(
            "CLOSE" if is_dd else "DOWN", '#dc2626', '#b91c1c',
            lambda checked=False, d=door: self._door_close(d))
        self._door_layout.addWidget(close_btn, row, cols["close"])
        door['_down_btn'] = close_btn

        if has_dd:
            self._add_home_button(row, door, cols, is_dd)

        # STOP is a pure motor.stop(); it does NOT touch the enable line.
        self._door_layout.addWidget(
            self._make_action_btn("STOP", '#d97706', '#b45309',
                                  lambda checked=False, d=door: self._door_stop(d)),
            row, cols["stop"])

        self._add_enable_button(row, door, cols)
        self._add_direction_combo(row, door, cols, is_dd)

    def _add_home_button(self, row: int, door: dict, cols: dict,
                         is_dd: bool) -> None:
        """HOME closes slowly onto the limit switch to calibrate. Only a Door
        device has one; an older stepper is offered the button (so the column
        stays aligned) but disabled, with a tooltip saying why."""
        home_btn = self._make_action_btn(
            "HOME", '#2563eb', '#1d4ed8',
            lambda checked=False, d=door: self._door_home(d))
        home_btn.setEnabled(bool(is_dd))
        home_btn.setToolTip(
            f"Slowly close {door['display']} until limit switch (calibrate)"
            if is_dd else "Home not available for older stepper motors")
        self._door_layout.addWidget(home_btn, row, cols["home"])
        door['_home_btn'] = home_btn

    def _add_enable_button(self, row: int, door: dict, cols: dict) -> None:
        """Per-motor enable, independent of STOP.

        Always clickable. With a hardware enable line it energises the coils
        (Door device / TMC_motor: enable_motor(); stepper: hw.<enable>.on()).
        Without one it still acts as a permission gate, so UP / DOWN / HOME
        stay blocked until the operator affirms a move, a motor should not
        run off one stray click.
        """
        en_btn = QtWidgets.QPushButton("ENABLE")
        en_btn.setCheckable(True)
        en_btn.setChecked(False)
        en_btn.setFixedHeight(30)
        en_btn.setStyleSheet(self._toggle_btn_style())
        en_btn.setEnabled(True)
        has_en = bool(door.get('tmc') or door.get('enable'))
        en_btn.setToolTip(
            "Energise / de-energise the motor coils. Independent of STOP.\n"
            "While disabled, UP / DOWN / HOME are blocked." if has_en else
            "No hardware enable line for this motor.\n"
            "Click to permit movement; UP / DOWN are blocked otherwise.")
        en_btn.toggled.connect(
            lambda checked=False, d=door: self._on_en_toggled(d, checked))
        door['_en_btn'] = en_btn
        self._door_layout.addWidget(en_btn, row, cols["en"])

    def _add_direction_combo(self, row: int, door: dict, cols: dict,
                             is_dd: bool) -> None:
        """Which motor phase means UP. Only meaningful for a bare stepper,
        a Door device's firmware abstracts phase behind open()/close()."""
        combo = QtWidgets.QComboBox()
        combo.addItems(["forward", "backward"])
        combo.setCurrentText(door.get('up_dir', 'forward'))
        combo.setFixedHeight(30)
        combo.setMaximumWidth(90)
        combo.setEnabled(not is_dd)
        combo.setToolTip(
            "Phase handled by the Door device firmware (open/close)." if is_dd
            else "Pick which motor phase means UP for this door.\n"
                 "Depends on how the stepper coils are wired.")
        combo.currentTextChanged.connect(
            lambda s, d=door: d.__setitem__('up_dir', s))
        door['_up_combo'] = combo
        self._door_layout.addWidget(combo, row, cols["dir"])

    def _add_all_row(self, cols: dict, has_dd: bool) -> None:
        """The fan-out row: one action per column, driving every door."""
        sep_row = len(self._doors) + 1
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #374151;")
        sep.setFixedHeight(1)
        self._door_layout.addWidget(sep, sep_row, 0, 1, 8 if has_dd else 7)

        row = sep_row + 1
        all_lbl = QtWidgets.QLabel("ALL")
        all_lbl.setStyleSheet("font-weight: bold; font-size: 11pt; "
                              "color: #60a5fa; padding: 2px 8px;")
        all_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._door_layout.addWidget(all_lbl, row, 0, 1, 2)

        for label, colour, hover, slot, col in (
                ("OPEN ALL" if has_dd else "UP ALL",
                 '#059669', '#047857', self._all_open, cols["open"]),
                ("CLOSE ALL" if has_dd else "DOWN ALL",
                 '#dc2626', '#b91c1c', self._all_close, cols["close"]),
                ("STOP ALL", '#d97706', '#b45309', self._all_stop,
                 cols["stop"])):
            btn = QtWidgets.QPushButton(label)
            btn.setStyleSheet(self._btn_style(colour, hover))
            btn.setFixedHeight(34)
            btn.clicked.connect(slot)
            self._door_layout.addWidget(btn, row, col)

        if has_dd:
            all_home = QtWidgets.QPushButton("HOME ALL")
            all_home.setStyleSheet(self._btn_style('#2563eb', '#1d4ed8'))
            all_home.setFixedHeight(34)
            all_home.clicked.connect(self._all_home)
            self._door_layout.addWidget(all_home, row, cols["home"])

        # Kept on the dialog: the caption flips between ENABLE ALL and
        # DISABLE ALL so the operator always knows what the next click does.
        self._all_en_btn = QtWidgets.QPushButton("ENABLE ALL")
        self._all_en_btn.setCheckable(True)
        self._all_en_btn.setChecked(False)
        self._all_en_btn.setFixedHeight(34)
        self._all_en_btn.setStyleSheet(self._toggle_btn_style())
        self._all_en_btn.toggled.connect(self._all_set_enable)
        self._door_layout.addWidget(self._all_en_btn, row, cols["en"])

    @staticmethod
    def _door_tooltip(door):
        parts = []
        if door.get('door_device'):
            parts.append(f"hw.{door['motor']}  (Door device)")
            parts.append("open() / close() / home() / stop()")
            if door.get('enable'):
                parts.append(f"Enable: hw.{door['enable']}")
            if door.get('limit_open'):
                parts.append(f"Open limit: {door['limit_open']}")
            if door.get('limit_close'):
                parts.append(f"Close limit: {door['limit_close']}")
            ev = door.get('events', {})
            if ev:
                parts.append(f"Events: {ev}")
            parts.append(f"max_steps={door.get('max_steps')}  "
                         f"step_rate={door.get('step_rate')}")
        else:
            mtype = "TMC_motor" if door.get('tmc') else "Stepper_motor"
            parts.append(f"hw.{door['motor']}  ({mtype})")
            if door.get('tmc'):
                parts.append("enable_motor() / disable_motor() (built-in)")
            elif door.get('enable'):
                parts.append(f"Enable: hw.{door['enable']}")
            if door.get('limit_top'):
                parts.append(f"Top limit: hw.{door['limit_top']}")
            if door.get('limit_bot'):
                parts.append(f"Bot limit: hw.{door['limit_bot']}")
        return '\n'.join(parts)

    # ── Door actions ─────────────────────────────────────────────

    @staticmethod
    def _opposite(direction: str) -> str:
        return "backward" if direction == "forward" else "forward"

    @staticmethod
    def _motor_blocked(door) -> bool:
        """True when the EN gate hasn't been clicked. UP / DOWN / HOME are
        blocked until the user affirms a move via EN, regardless of whether
        the motor has a hardware enable line, so doors can't move on dialog
        open."""
        return not door.get('_en', False)

    def _door_open(self, door):
        """UP / OPEN. Door devices use the firmware open(); steppers / TMCs
        use the user-picked UP direction. Door must be enabled and not
        already at the top."""
        if self._motor_blocked(door):
            return
        if door.get('_state') == 'up':
            return
        if door.get('door_device'):
            self._exec_commands([f"hw.{door['motor']}.open()"])
        else:
            direction = door.get('up_dir', 'forward')
            self._exec_commands([
                f"hw.{door['motor']}.{direction}({self._step_rate}, {self._n_steps})"
            ])
        door['_state'] = 'up'
        self._refresh_door_buttons(door)

    def _door_close(self, door):
        """DOWN / CLOSE. Opposite of UP. Door must be enabled and not
        already at the bottom."""
        if self._motor_blocked(door):
            return
        if door.get('_state') == 'down':
            return
        if door.get('door_device'):
            self._exec_commands([f"hw.{door['motor']}.close()"])
        else:
            direction = self._opposite(door.get('up_dir', 'forward'))
            self._exec_commands([
                f"hw.{door['motor']}.{direction}({self._step_rate}, {self._n_steps})"
            ])
        door['_state'] = 'down'
        self._refresh_door_buttons(door)

    def _door_home(self, door):
        """HOME closes the door against the limit switch, ending at the
        bottom. Only meaningful for Door devices."""
        if not door.get('door_device'):
            return
        if self._motor_blocked(door):
            return
        self._exec_commands([f"hw.{door['motor']}.home()"])
        door['_state'] = 'down'
        self._refresh_door_buttons(door)

    def _door_stop(self, door):
        """Pure stop, only motor.stop(); the enable line is untouched.
        Position is now unknown, so re-enable both UP and DOWN."""
        self._exec_commands([f"hw.{door['motor']}.stop()"])
        door['_state'] = 'unknown'
        self._refresh_door_buttons(door)

    def _on_en_toggled(self, door, checked: bool):
        """User toggled the per-row EN button. Sends the right
        enable/disable command, updates internal state, flips the
        button label between ENABLE / DISABLE, and refreshes the
        UP/DOWN/HOME enabled state (movement requires EN on)."""
        cmd = self._enable_cmd_for(door, checked)
        if cmd:
            self._exec_commands([cmd])
        door['_en'] = bool(checked)
        btn = door.get('_en_btn')
        if btn is not None:
            try:
                btn.setText("DISABLE" if checked else "ENABLE")
            except Exception:
                pass
        self._refresh_door_buttons(door)

    def _refresh_door_buttons(self, door):
        """Single source of truth for per-row enabled state.

        Rules:
          - UP/OPEN  disabled when state == 'up'   OR EN not clicked
          - DOWN     disabled when state == 'down' OR EN not clicked
          - HOME     disabled when state == 'down' OR EN not clicked
          - STOP     always enabled (only way to interrupt a move)
          - EN       always enabled (permission gate, even with no
                     hardware enable line)
        """
        state = door.get('_state', 'unknown')
        # ``_en`` is the unified gate, UP / DOWN / HOME require EN clicked
        # first regardless of a hardware enable line.
        movable = door.get('_en', False)

        up_btn = door.get('_up_btn')
        if up_btn is not None:
            up_btn.setEnabled(movable and state != 'up')
        down_btn = door.get('_down_btn')
        if down_btn is not None:
            down_btn.setEnabled(movable and state != 'down')
        home_btn = door.get('_home_btn')
        if home_btn is not None and door.get('door_device'):
            home_btn.setEnabled(movable and state != 'down')

    @staticmethod
    def _enable_cmd_for(door, enabled: bool) -> str:
        """Build the right enable/disable command for this door's
        underlying device, or '' when it has no enable line.

        Only TMC_motor has built-in enable_motor()/disable_motor(). A Door
        device drives a separate Digital_output enable pin (door['enable'],
        matched at discovery); with none, EN is a pure permission gate."""
        if door.get('tmc'):
            method = "enable_motor" if enabled else "disable_motor"
            return f"hw.{door['motor']}.{method}()"
        if door.get('enable'):
            method = "on" if enabled else "off"
            return f"hw.{door['enable']}.{method}()"
        return ""

    def _all_open(self):
        """Send OPEN/UP to every door that's not already at the top
        AND whose motor is enabled. State + buttons updated per-row."""
        cmds = []
        moved = []
        for d in self._doors:
            if d.get('_state') == 'up':
                continue
            if self._motor_blocked(d):
                continue
            if d.get('door_device'):
                cmds.append(f"hw.{d['motor']}.open()")
            else:
                direction = d.get('up_dir', 'forward')
                cmds.append(
                    f"hw.{d['motor']}.{direction}({self._step_rate}, {self._n_steps})")
            moved.append(d)
        if cmds:
            self._exec_commands(cmds)
        for d in moved:
            d['_state'] = 'up'
            self._refresh_door_buttons(d)

    def _all_close(self):
        cmds = []
        moved = []
        for d in self._doors:
            if d.get('_state') == 'down':
                continue
            if self._motor_blocked(d):
                continue
            if d.get('door_device'):
                cmds.append(f"hw.{d['motor']}.close()")
            else:
                direction = self._opposite(d.get('up_dir', 'forward'))
                cmds.append(
                    f"hw.{d['motor']}.{direction}({self._step_rate}, {self._n_steps})")
            moved.append(d)
        if cmds:
            self._exec_commands(cmds)
        for d in moved:
            d['_state'] = 'down'
            self._refresh_door_buttons(d)

    def _all_set_enable(self, enabled: bool):
        """Toggle every per-row EN button to match. Each toggle's signal
        fires _on_en_toggled which sends the command, updates state and
        flips the per-row label between ENABLE / DISABLE."""
        for d in self._doors:
            btn = d.get('_en_btn')
            if btn is None or not btn.isEnabled():
                continue
            if btn.isChecked() != enabled:
                btn.setChecked(enabled)
        # Flip the bulk button's own label to match the new state.
        if hasattr(self, '_all_en_btn') and self._all_en_btn is not None:
            try:
                self._all_en_btn.setText("DISABLE ALL" if enabled else "ENABLE ALL")
            except Exception:
                pass

    def _all_home(self):
        """Send HOME to every door device whose motor is energised and
        isn't already at the bottom; respects the same per-row gates as the
        individual HOME buttons."""
        cmds = []
        moved = []
        for d in self._doors:
            if not d.get('door_device'):
                continue
            if d.get('_state') == 'down':
                continue
            if self._motor_blocked(d):
                continue
            cmds.append(f"hw.{d['motor']}.home()")
            moved.append(d)
        if cmds:
            self._exec_commands(cmds)
        for d in moved:
            d['_state'] = 'down'
            self._refresh_door_buttons(d)

    def _all_stop(self):
        cmds = [f"hw.{d['motor']}.stop()" for d in self._doors]
        self._exec_commands(cmds)
        # Position is unknown after a bulk stop; mark so both UP and DOWN
        # re-enable on every row.
        for d in self._doors:
            d['_state'] = 'unknown'
            self._refresh_door_buttons(d)

    # ── Command execution ────────────────────────────────────────

    def _exec_commands(self, commands):
        """Send commands to MCU via EXEC_RAW."""
        if not commands:
            return

        sw = self.main_window._setup_widget_for(self.setup_id)
        if not sw or not sw.is_connected:
            QtWidgets.QMessageBox.warning(
                self, "Not Connected",
                "MCU is not connected for this setup.")
            return

        if getattr(sw, 'is_running', False) or getattr(sw, 'framework_running', False):
            QtWidgets.QMessageBox.warning(
                self, "Framework Running",
                "Cannot control doors while the framework is running.\n"
                "Stop the framework first.")
            return

        # Route every door command through safe_mcu_exec so a single bad
        # command logs + pops one QMessageBox instead of crashing the dialog.
        pyc = self.main_window.mcu.get(self.setup_id) \
            if hasattr(self.main_window, "mcu") else None
        if pyc is None:
            sw_lookup = self.main_window._setup_widget_for(self.setup_id) \
                if hasattr(self.main_window, "_setup_widget_for") else None
            pyc = getattr(sw_lookup, "pycboard", None) if sw_lookup else None
        if pyc is None:
            QtWidgets.QMessageBox.warning(
                self, "Not Connected",
                "MCU not connected for this setup.")
            return
        for code in commands:
            ok, _ = safe_mcu_exec(pyc, code, parent_widget=self)
            if not ok:
                # safe_mcu_exec already logged + showed a dialog. Stop on
                # first failure, more commands after a board error won't help.
                break
            logger.debug(f"Door control exec: {code}")

    # ── Config callbacks ─────────────────────────────────────────

    def _on_rate_changed(self, value):
        self._step_rate = value

    def _on_steps_changed(self, value):
        self._n_steps = value

    # ── Styling ──────────────────────────────────────────────────

    def _make_action_btn(self, label, color, hover, on_click):
        """OPEN/CLOSE/HOME/STOP-style 30px gradient button (shared shape)."""
        btn = QtWidgets.QPushButton(label)
        btn.setStyleSheet(self._btn_style(color, hover))
        btn.setFixedHeight(30)
        btn.clicked.connect(on_click)
        return btn

    @staticmethod
    def _btn_style(color, hover):
        """Action button (OPEN / CLOSE / HOME / STOP / Detect).

        135° gradient + glass top-edge highlight, using ``color`` as the
        gradient end stop and ``hover`` as the start. Disabled state stays
        neutral slate so unavailable actions are obvious at a glance.
        """
        return f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                            stop:0 {hover}, stop:1 {color});
                color: #ffffff;
                border: 1px solid rgba(255,255,255,0.18);
                border-top: 1px solid rgba(255,255,255,0.32);
                border-radius: 6px;
                padding: 4px 12px; font-weight: 700; font-size: 9pt;
                min-width: 64px; min-height: 26px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                            stop:0 {hover}, stop:0.7 {hover}, stop:1 {color});
                border: 1px solid rgba(255,255,255,0.28);
                border-top: 1px solid rgba(255,255,255,0.45);
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                            stop:0 {hover}, stop:1 {color});
                padding-top: 5px;
            }}
            QPushButton:disabled {{
                background: rgba(148,163,184,0.10);
                color: #64748b;
                border: 1px solid rgba(148,163,184,0.18);
            }}
        """

    @staticmethod
    def _toggle_btn_style():
        """Two-state style for the EN button: gray when disabled (off),
        green when checked (motor coils energised)."""
        return """
            QPushButton {
                background-color: #4b5563; color: #e5e7eb;
                border: 1px solid #374151; border-radius: 4px;
                padding: 4px 12px; font-weight: bold; font-size: 9pt;
                min-width: 50px;
            }
            QPushButton:hover { background-color: #6b7280; }
            QPushButton:checked {
                background-color: #16a34a; color: white;
                border: 1px solid #166534;
            }
            QPushButton:checked:hover { background-color: #15803d; }
            QPushButton:disabled {
                background-color: #1f2937; color: #4b5563;
                border: 1px dashed #374151;
            }
        """
