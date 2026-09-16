import pyb
import ujson
from ucollections import namedtuple
from . import timer
from . import state_machine as sm
from . import hardware as hw
from . import utility as ut

VERSION = "2.1"


class pyControlError(BaseException):  # Exception for pyControl errors.
    pass


Datatuple = namedtuple("Datatuple", ["time", "type", "subtype", "content"])

# Constants used to indicate data types, corresponding data tuple indicated in comment.

EVENT_TYP = b"E"  # Event            : (time, EVENT_TYP, [i]nput/[t]imer/[s]ync/[p]ublish/[u]ser/[a]pi, event_ID)
STATE_TYP = b"S"  # State transition : (time, STATE_TYP, "", state_ID)
PRINT_TYP = b"P"  # User print       : (time, PRINT_TYP, "", print_string)
HARDW_TYP = b"H"  # Harware callback : (time, HARDW_TYP, "", hardware_ID)
VARBL_TYP = b"V"  # Variable change  : (time, VARBL_TYP, [g]et/user_[s]et/[a]pi_set/[p]rint/s[t]art/[e]nd, json_str)
WARNG_TYP = b"!"  # Warning          : (time, WARNG_TYP, "", print_string)
STOPF_TYP = b"X"  # Stop framework   : (time, STOPF_TYP, "", "")
THRSH_TYP = b"T"  # Threshold        : (time, THRSH_TYP, [s]et)
NAV_TYP   = b"N"  # Maze nav command : 1-byte payload (0=pause, 1=resume, 2=next_stage, 3=prev_stage)

# Event_queue -----------------------------------------------------------------


class Event_queue:
    # First-in first-out event queue.
    def __init__(self):
        self.reset()

    def reset(self):
        # Empty queue.
        self.Q = []
        self.available = False

    def put(self, event_tuple):
        # Put event in queue.
        self.Q.append(event_tuple)
        self.available = True

    def get(self):
        # Get event tuple from queue
        self.available = len(self.Q) > 1
        return self.Q.pop(0)


# Framework variables and objects ---------------------------------------------

event_queue = Event_queue()  # Instantiate event que object.

data_output_queue = Event_queue()  # Queue used for outputing events to serial line.

data_output = True  # Whether to output data to the serial line.

current_time = None  # Time since run started (milliseconds).

running = False  # Set to True when framework is running, set to False to stop run.

usb_serial = pyb.USB_VCP()  # USB serial port object.

clock = pyb.Timer(1)  # Timer which generates clock tick.

check_timers = False  # Flag to say timers need to be checked, set True by clock tick.

start_time = 0  # Time at which framework run is started.

# Framework functions ---------------------------------------------------------


def _clock_tick(t):
    # Set flag to check timers, called by hardware timer once each millisecond.
    global check_timers, current_time
    current_time = pyb.elapsed_millis(start_time)
    check_timers = True


def output_data(event):
    # Output data to computer.
    if not data_output:
        return
    timestamp = event.time.to_bytes(4, "little")
    subtype_byte = event.subtype.encode() if event.subtype else b"_"
    content_bytes = str(event.content).encode() if event.content else b""
    message = timestamp + event.type + subtype_byte + content_bytes
    message_len = len(message).to_bytes(2, "little")
    checksum = (sum(message) & 0xFFFF).to_bytes(2, "little")
    usb_serial.send(b"\x07" + checksum + message_len + message)


# Host-command failures that would otherwise be invisible. Reported ONCE per
# run each: the coordinate and intrinsic-event commands arrive at pose rate,
# so a systematic fault (bad encoding, stale event IDs, a task handler that
# raises every time) would put a warning on the data queue tens of times a
# second, competing with the science data it shares that queue with. One row
# is what tells the operator the run is compromised; the rest only cost time.
_reported_faults = set()


def _warn_once(key, message):
    if key in _reported_faults:
        return
    _reported_faults.add(key)
    data_output_queue.put(Datatuple(current_time, WARNG_TYP, "", message))


def receive_data():
    # Read and process data from computer.
    global running
    new_byte = usb_serial.read(1)
    if new_byte == b"\x03":  # Serial command to stop run.
        running = False
    elif new_byte in (VARBL_TYP, EVENT_TYP):
        data_len = int.from_bytes(usb_serial.read(2), "little")
        data_and_checksum = usb_serial.recv(data_len + 2, timeout=1)
        checksum = int.from_bytes(data_and_checksum[-2:], "little")
        if checksum != (sum(data_and_checksum[:-2]) & 0xFFFF):
            # Dropped set/trigger (fire-and-forget), flag it so the host knows.
            data_output_queue.put(
                Datatuple(current_time, WARNG_TYP, "", "Host command dropped (bad checksum)")
            )
            return
        data_str = data_and_checksum[:-2].decode()
        if new_byte == VARBL_TYP:  # Get/set variables command.
            if data_str[0] in ("s", "a"):  # Set variable.
                v_name, v_value = eval(data_str[1:])
                if sm.set_variable(v_name, v_value):
                    data_output_queue.put(
                        Datatuple(current_time, VARBL_TYP, data_str[0], ujson.dumps({v_name: v_value}))
                    )
            elif data_str[0] == "g":  # Get variable.
                v_name = data_str[1:]
                v_value = sm.get_variable(v_name)
                data_output_queue.put(Datatuple(current_time, VARBL_TYP, "g", ujson.dumps({v_name: v_value})))
        elif new_byte == EVENT_TYP:  # Trigger event command.
            subtype = data_str[0]
            event_ID = int(data_str[1:])
            event_queue.put(Datatuple(current_time, EVENT_TYP, subtype, event_ID))
    elif new_byte == NAV_TYP:  # GUI nav: pause / resume / next / prev.
        # Wire format: b'N' + 1 byte nav code (0=pause, 1=resume, 2=next, 3=prev).
        # Dispatched directly to state_machine, no separate stage module.
        nav_code = usb_serial.read(1)
        if not nav_code:
            return
        code = nav_code[0]
        nav_handlers = (sm.pause, sm.resume, sm.next_state, sm.prev_state)
        if 0 <= code < len(nav_handlers):
            nav_handlers[code]()
    elif new_byte == b"c":  # Set coordinate command, GUI pushes tracking data
                            # OR control flags (e.g. c.pause_state, c.pass_inter_state)
                            # into the task's `c` namespace at runtime.
        # Wire format (matches pycboard.set_coordinates):
        #   b'c' + len(2) + data + checksum(2)
        # where  data = repr((c_name, c_value)).encode() + b'c'
        # The trailing b'c' byte is a sanity-check tag. eval() restores the
        # native value, so a zone name comes back as the str "RightArm".
        data_len = int.from_bytes(usb_serial.read(2), "little")
        data = usb_serial.read(data_len)
        checksum = int.from_bytes(usb_serial.read(2), "little")
        if checksum != (sum(data) & 0xFFFF):
            _warn_once("c_sum", "Coordinate push dropped (bad checksum)")
            return
        if data[-1:] == b"c":  # Sanity-tag matches.
            try:
                c_name, c_value = eval(data[:-1])
                setattr(ut.c, c_name, c_value)
            except Exception:
                # Keep running, an aborted framework loses the rest of the
                # session, but say so, or the task silently reads a stale
                # coordinate for the whole run.
                _warn_once("c_bad", "Coordinate push failed (malformed payload)")
    elif new_byte == b"Z":  # Intrinsic event dispatch, like entry/exit:
                            # process the event in the state machine but do
                            # NOT log to the data stream. Used by the host's
                            # tracking pipeline for the framework-intrinsic
                            # ``zone_changed`` event so the MCU TSV isn't
                            # flooded with one row per frame's zone flicker.
                            # If the user task wants visibility, they write
                            # ``print(...)`` in their state handler.
        # Wire format: b'Z' + len(2) + event_ID_string + checksum(2)
        data_len = int.from_bytes(usb_serial.read(2), "little")
        data = usb_serial.read(data_len)
        checksum = int.from_bytes(usb_serial.read(2), "little")
        if checksum != (sum(data) & 0xFFFF):
            _warn_once("z_sum", "Intrinsic event dropped (bad checksum)")
            return
        try:
            event_ID = int(data.decode())
            sm.process_event(event_ID)
        except Exception:
            # Still swallowed: letting a stale event ID, or a raise inside the
            # task's own handler, abort the framework would cost the rest of
            # the session, which is worse. But an unreported raise means the
            # task never responded to zone_changed all run and the data file
            # looks entirely normal, so the run must be marked.
            _warn_once("z_bad", "Intrinsic event failed (bad ID or task handler raised)")


def run():
    # Run framework for specified number of seconds.
    # Pre run
    global current_time, start_time, running
    timer.reset()
    event_queue.reset()
    data_output_queue.reset()
    _reported_faults.clear()  # Each run reports its own faults afresh.
    if not hw.initialised:
        hw.initialise()
    usb_serial.setinterrupt(-1)  # Disable 'ctrl+c' on serial raising KeyboardInterrupt.
    current_time = 0
    ut.print_variables(when="t")
    start_time = pyb.millis()
    clock.init(freq=1000)
    clock.callback(_clock_tick)
    sm.start()
    hw.run_start()
    running = True
    # try/finally so teardown runs even if a task handler raises (else hardware
    # like the I2S amp stays powered after the run "stops").
    try:
        while running:
            # Priority 1: Process hardware interrupts.
            if hw.interrupt_queue.available:
                hw.IO_dict[hw.interrupt_queue.get()]._process_interrupt()
            # Priority 2: Process event from queue.
            elif event_queue.available:
                event = event_queue.get()
                data_output_queue.put(event)
                sm.process_event(event.content)
            # Priority 3: Check for elapsed timers.
            elif check_timers:
                timer.check()
            # Priority 4: Process timer event.
            elif timer.elapsed:
                event = timer.get()
                if event.type == EVENT_TYP:
                    if event.subtype:
                        data_output_queue.put(event)
                    sm.process_event(event.content)
                elif event.type == HARDW_TYP:
                    hw.IO_dict[event.content]._timer_callback()
                elif event.type == STATE_TYP:
                    sm.goto_state(event.content)
            # Priority 5: Check for serial input from computer.
            elif usb_serial.any():
                receive_data()
            # Priority 6: Stream analog data.
            elif hw.stream_data_queue.available:
                hw.IO_dict[hw.stream_data_queue.get()].send_buffer()
            # Priority 7: Output framework data.
            elif data_output_queue.available:
                output_data(data_output_queue.get())
    finally:
        # Post run, always runs (clean stop or task exception); tears down hw.
        ut.print_variables(when="e")
        data_output_queue.put(Datatuple(current_time, STOPF_TYP, "", ""))
        usb_serial.setinterrupt(3)  # Enable 'ctrl+c' on serial raising KeyboardInterrupt.
        clock.deinit()
        hw.run_stop()
        sm.stop()
        while data_output_queue.available:
            output_data(data_output_queue.get())
