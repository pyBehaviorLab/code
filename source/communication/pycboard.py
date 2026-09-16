"""Host-side wrapper around one pyControl board.

Sits between the raw serial transport (``pyboard.py``) and everything that
wants to talk to an MCU: uploads the hardware definition and the task, sets and
reads task variables, starts and stops the framework, and pushes coordinates
and trigger events at the tracker's rate.

Two things about it bite callers who assume ordinary Python:

* **``PyboardError`` subclasses ``BaseException``**, so ``except Exception``
  does NOT catch a board failure. Catch ``PyboardError`` or ``BaseException``
  explicitly, a bare ``except Exception`` around board I/O silently lets a
  disconnected board take down the caller.
* **The board is a second machine.** The files uploaded here run under
  MicroPython on the STM32 and cannot be imported on the host; the framework
  version this module checks ``fw.VERSION`` against is mirrored in
  ``source/__init__.py::VERSION`` and the two must move together.

Mirrors upstream pyControl's ``pycboard`` so shared infrastructure stays
recognisable; the additions are this rig's (host clock, coordinate push,
per-session TSV hand-off to ``data_logger``).
"""

import io
import logging
import os
import re
import time
from source import host_clock
import json
import queue as _queue
import inspect
import tokenize
from serial import SerialException

logger = logging.getLogger(__name__)
from array import array
from .pyboard import Pyboard, PyboardError
from .data_logger import Data_logger
from source.communication.message import MsgType, Datatuple
from source import VERSION
from source.communication.errors import (
    ERR_FILE_TRANSFER_NO_SPACE, ERR_FILE_TRANSFER_FAILED,
    ERR_FRAMEWORK_IMPORT_ERROR, ERR_HW_DEF_NOT_FOUND,
    ERR_HW_DEF_IMPORT_ERROR, ERR_TASK_FILE_NOT_FOUND,
    ERR_TASK_SETUP_FAILED, ERR_BAD_CHECKSUM,
    ERR_UNEXPECTED_MCU_INPUT, HINT_TROUBLESHOOTING_CHECKS,
)
from source.config.settings import user_folder
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------------------
#  Helper functions.
# ----------------------------------------------------------------------------------------


# djb2 hashing algorithm used to check integrity of transfered files.
def _djb2_file(file_path):
    with open(file_path, "rb") as f:
        h = 5381
        while True:
            c = f.read(4)
            if not c:
                break
            h = ((h << 5) + h + int.from_bytes(c, "little")) & 0xFFFFFFFF
    return h


# Used on pyboard for file transfer (shipped as source at connect, executed on
# the board, must stay MicroPython-compatible). If the host aborts mid-file it
# self-exits after 10 s without data, deletes the partial file (a truncated .py
# on flash SyntaxErrors every later import), and re-enables ctrl-C so the host
# can always reclaim the REPL.
def _receive_file(file_path, file_size):
    import time

    usb = pyb.USB_VCP()
    usb.setinterrupt(-1)
    buf_size = 512
    buf = bytearray(buf_size)
    buf_mv = memoryview(buf)
    bytes_remaining = file_size
    complete = False
    try:
        with open(file_path, "wb") as f:
            last_rx_ms = time.ticks_ms()
            while bytes_remaining > 0:
                bytes_read = usb.recv(buf, timeout=5)
                usb.write(b"OK")
                if bytes_read:
                    bytes_remaining -= bytes_read
                    f.write(buf_mv[:bytes_read])
                    last_rx_ms = time.ticks_ms()
                elif time.ticks_diff(time.ticks_ms(), last_rx_ms) > 10000:
                    break  # Host stopped sending: abandon, clean up in finally.
        complete = bytes_remaining <= 0
    except:
        fs_stat = os.statvfs("/flash")
        fs_free_space = fs_stat[0] * fs_stat[3]
        if fs_free_space < bytes_remaining:
            usb.write(b"NS")  # Out of space.
        else:
            usb.write(b"ER")
    finally:
        usb.setinterrupt(3)  # 3 = ctrl-C character (default interrupt char).
        if not complete:
            try:
                os.remove(file_path)
            except OSError:
                pass


@dataclass
class State_machine_info:
    name: str
    task_hash: int
    states: dict
    events: dict
    ID2name: dict
    analog_inputs: dict
    variables: dict
    coordinates: dict           # maze tracking-data namespace; empty for tasks that don't use it
    framework_version: str
    micropython_version: float
    hardware_def_name: str = ""
    hardware_def_hash: int = 0
    devices: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------
#  Pycboard class.
# ----------------------------------------------------------------------------------------

# setup_state_machine self-heal: how many times to retry the WHOLE upload
# (reset -> transfer -> load -> sm_info read-back) on a raw-REPL desync, and
# the base settle gap between attempts (escalated per attempt so a just-
# rebooted board has time to quiesce before the next handshake).
_TASK_SETUP_ATTEMPTS = 3
_TASK_SETUP_SETTLE_S = 0.5

# Connect-time link priming: one clean soft-reboot + heavy round-trip BEFORE
# the first real reset(), so the F767 native-USB "first transaction" glitch is
# absorbed at connect (where nobody waits) instead of stalling the first Upload
# with a '<stdin>' SyntaxError + retry.
_PRIME_ATTEMPTS = 3      # clean round-trips to try before giving up
_PRIME_SETTLE_S = 0.3    # gap between prime attempts
_DTR_SETTLE_S = 0.1      # let the CDC connect_state settle after asserting DTR

# File-transfer stall tolerance. Writing a chunk can force a 256 KB internal-
# flash sector erase on the F76x, during which MicroPython disables IRQs and
# USB goes silent for seconds, budgets must sit well above that worst case
# (~4 s) or a healthy board gets aborted mid-file, leaving a truncated .py.
_TRANSFER_ACK_S = 15.0    # per-chunk ack wait
_TRANSFER_EOF_S = 10.0    # end-of-file raw-REPL completion wait
_TRANSFER_RESYNC_S = 12.0  # > _receive_file's 10 s idle self-exit


class Pycboard(Pyboard):
    """Pycontrol board inherits from Pyboard and adds functionality for file transfer
    and pyControl operations.
    """

    device_class2file = {}  # Dict mapping device classes to file where they are defined {class_name: device_file}

    def __init__(self, serial_port, baudrate=115200, verbose=True, print_func=print, data_consumers=None):
        self.serial_port = serial_port
        self.print = print_func  # Function used for print statements.
        self.data_logger = Data_logger(board=self, print_func=print_func)
        self.data_consumers = data_consumers
        self.status = {"serial": None, "framework": None, "usb_mode": None}
        self.device_files_on_pyboard = {}  # Dict {file_name:file_hash} of files in devices folder on pyboard.
        # {target_path: host_djb2} recorded by transfer_file each upload, the
        # single hash source reused by sm_info (task_hash / _loaded_hwd_hash).
        self._upload_hashes = {}
        # HD lineage, set by load_hardware_definition(); read by
        # setup_state_machine() to populate State_machine_info, then
        # written to the .tsv header by data_logger.
        self._loaded_hwd_path = ""
        self._loaded_hwd_hash = 0
        self._pending_writes: "_queue.Queue[tuple]" = _queue.Queue()
        if not Pycboard.device_class2file:  # Scan devices folder to find files where device classes are defined.
            self.make_device_class2file_map()
        try:
            super().__init__(self.serial_port, baudrate=baudrate)
            self.status["serial"] = True
            self._settle_link()   # assert DTR + let the CDC connect_state settle
            self._prime_link()    # warm the link so the first Upload doesn't glitch
            self.reset()
            self.unique_ID = eval(self.eval("pyb.unique_id()").decode())
            v_tuple = eval(
                self.eval("sys.implementation.version if hasattr(sys, 'implementation') else (0,0,0)").decode()
            )
            self.micropython_version = float("{}.{}{}".format(*v_tuple))
        except SerialException as e:
            self.status["serial"] = False
            raise (e)
        if verbose:  # Print status.
            if self.status["serial"]:
                self.print("\nMicropython version: {}".format(self.micropython_version))
            else:
                self.print("Error: Unable to open serial connection.")
                return
            if self.status["framework"]:
                self.print(f"Framework version: {self.framework_version}")
                if self.framework_version != VERSION:
                    self.print(
                        "\nThe pyControl framework version on the board does not match the GUI version. "
                        "It is recommended to reload the pyControl framework to the pyboard to ensure compatibility."
                    )
            else:
                if self.status["framework"] is None:
                    self.print("pyControl Framework: Not loaded")
                else:
                    self.print("pyControl Framework: Import error")
                return

    def _settle_link(self):
        """Assert DTR and let the USB-CDC connect_state settle before the first
        command. MicroPython's stm32 CDC only treats the host as connected once
        DTR is asserted; data crossing that boundary can be dropped, so a brief
        settle avoids losing the very first bytes."""
        try:
            self.serial.dtr = True
        except Exception:
            pass
        time.sleep(_DTR_SETTLE_S)

    def _prime_link(self):
        """Warm the link with one clean soft-reboot + heavy round-trip before
        the real reset().

        On the F767's native USB-CDC the FIRST soft-reboot-then-heavy round-trip
        after connect garbles once (CDC warm-up), which otherwise surfaces as a
        '<stdin>' SyntaxError on the first Upload and triggers the slow upload
        retry. We exercise that exact shape here, a long command in AND a long
        print out, until one round-trip comes back intact, so the operator's
        first real Upload is already warm. Best-effort: if it never converges
        the board still works (reset() + the upload self-heal cover it); it just
        may pay the glitch once on the first Upload."""
        probe = "X" * 256
        for _attempt in range(_PRIME_ATTEMPTS):
            try:
                self.enter_raw_repl()  # soft reboot
                if self.exec("_p='%s'\nprint(_p)" % probe).strip().decode() == probe:
                    return True        # one clean heavy round-trip → link warm
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                pass
            try:
                self.serial.reset_input_buffer()
            except Exception:
                pass
            time.sleep(_PRIME_SETTLE_S)
        logger.warning("link prime did not converge for %s "
                       "(first upload may glitch once)", self.serial_port)
        return False

    def reset(self):
        """Enter raw repl (soft reboots pyboard), import modules."""
        self.enter_raw_repl()  # Soft resets pyboard.
        self.exec(inspect.getsource(_djb2_file))  # define djb2 hashing function.
        self.exec(inspect.getsource(_receive_file))  # define receive file function.
        self.exec("import os; import gc; import sys; import pyb")
        self.framework_running = False
        error_message = None
        self.status["usb_mode"] = self.eval("pyb.usb_mode()").decode()
        self.data_logger.reset()
        try:
            self.exec("from pyControl import *; import devices")
            self.status["framework"] = True  # Framework imported OK.
        except PyboardError as e:
            error_message = e.args[2].decode()
            if ("ImportError: no module named 'pyControl'" in error_message) or (
                "ImportError: no module named 'devices'" in error_message
            ):
                self.status["framework"] = None  # Framework not installed.
            else:
                self.status["framework"] = False  # Framework import error.
        # Device-file hash map is best-effort and must not affect framework
        # status; an empty map just makes the next task upload re-transfer the
        # device files.
        self.device_files_on_pyboard = {}
        if self.status["framework"]:
            try:
                self.device_files_on_pyboard = eval(self.eval(
                    "{f:_djb2_file('devices/'+f) for f in os.listdir('devices') if f.endswith('.py')}").decode())
            except Exception:
                self.device_files_on_pyboard = {}
        try:
            self.framework_version = self.eval("fw.VERSION").decode()
        except PyboardError:
            self.framework_version = "<1.8"
        return error_message

    def hard_reset(self, reconnect=True):
        self.print("\nResetting pyboard.")
        try:
            self.exec_raw_no_follow("pyb.hard_reset()")
        except PyboardError:
            pass
        self.close()  # Close serial connection.
        if reconnect:
            time.sleep(5.0)  # Wait 5 seconds before trying to reopen serial connection.
            try:
                super().__init__(self.serial_port, baudrate=115200)  # Reopen serial conection.
                self.reset()
            except SerialException:
                self.print("Unable to reopen serial connection.")
        else:
            self.print("\nSerial connection closed.")

    def gc_collect(self, timeout=10):
        """Run a garbage collection on pyboard to free up memory."""
        self.exec("gc.collect()", timeout=timeout)

    def DFU_mode(self):
        """Put the pyboard into device firmware update mode."""
        self.exec("import pyb")
        try:
            self.exec_raw_no_follow("pyb.bootloader()")
        except PyboardError:
            pass  # Error occurs on older versions of micropython but DFU is entered OK.
        self.print("\nEntered DFU mode, closing serial connection.\n")
        self.close()

    def disable_mass_storage(self):
        """Modify the boot.py file to make the pyboards mass storage invisible to the
        host computer."""
        self.print("\nDisabling USB flash drive")
        self.write_file("boot.py", "import machine\nimport pyb\npyb.usb_mode('VCP')")
        self.hard_reset(reconnect=False)

    def enable_mass_storage(self):
        """Modify the boot.py file to make the pyboards mass storage visible to the
        host computer."""
        self.print("\nEnabling USB flash drive")
        self.write_file("boot.py", "import machine\nimport pyb\npyb.usb_mode('VCP+MSC')")
        self.hard_reset(reconnect=False)

    # ------------------------------------------------------------------------------------
    # Pyboard filesystem operations.
    # ------------------------------------------------------------------------------------

    def write_file(self, target_path, data):
        """Write data to file at specified path on pyboard, any data already
        in the file will be deleted."""
        try:
            self.exec("with open('{}','w') as f: f.write({})".format(target_path, repr(data)))
        except PyboardError as e:
            raise PyboardError(e)

    def get_file_hash(self, target_path):
        """Get the djb2 hash of a file on the pyboard."""
        try:
            file_hash = int(self.eval("_djb2_file('{}')".format(target_path)).decode())
        except PyboardError:  # File does not exist.
            return -1
        return file_hash

    def transfer_file(self, file_path, target_path=None):
        """Copy file at file_path to location target_path on pyboard.

        Returns the file's host djb2 hash (also recorded in
        ``self._upload_hashes[target_path]``) so callers reuse the one hash
        for sm_info instead of re-computing it.
        """
        if not target_path:
            target_path = os.path.split(file_path)[-1]
        file_size = os.path.getsize(file_path)
        file_hash = _djb2_file(file_path)
        self._upload_hashes[target_path] = file_hash
        error_message = (
            "\n\nError: " + ERR_FILE_TRANSFER_FAILED + "\n"
            + HINT_TROUBLESHOOTING_CHECKS
        )
        # Send until the board's hash matches the host's. A failed attempt
        # (ack stall, lost sync) is resynced and retried rather than aborting
        # the whole upload on the first hiccup.
        for _attempt in range(10):
            if file_hash == self.get_file_hash(target_path):
                return file_hash
            try:
                self.exec_raw_no_follow("_receive_file('{}',{})".format(target_path, file_size))
                with open(file_path, "rb") as f:
                    while True:
                        chunk = f.read(512)
                        if not chunk:
                            break
                        self.serial.write(chunk)
                        response_bytes = self._read_transfer_ack()
                        if response_bytes == b"NS":
                            self.print("\n\n" + ERR_FILE_TRANSFER_NO_SPACE)
                            raise PyboardError(ERR_FILE_TRANSFER_NO_SPACE)
                        if response_bytes != b"OK":
                            raise PyboardError("chunk ack timeout")
                self.follow(_TRANSFER_EOF_S)
            except PyboardError as e:
                if e.args and e.args[0] == ERR_FILE_TRANSFER_NO_SPACE:
                    raise  # Retrying cannot create free space.
                self._resync_after_failed_transfer()
        # Unable to transfer file.
        self.print(error_message)
        raise PyboardError(ERR_FILE_TRANSFER_FAILED)

    def _read_transfer_ack(self):
        """Read one 2-byte _receive_file ack, polling up to _TRANSFER_ACK_S.
        Bounded (a dead board surfaces instead of hanging the worker), but
        long enough to span an internal-flash sector erase during which the
        board's USB is silent."""
        response_bytes = b""
        deadline = time.monotonic() + _TRANSFER_ACK_S
        while len(response_bytes) < 2 and time.monotonic() < deadline:
            avail = self.serial.in_waiting
            if avail > 0:
                response_bytes += self.serial.read(min(2 - len(response_bytes), avail))
            else:
                time.sleep(0.005)
        return response_bytes

    def _resync_after_failed_transfer(self):
        """Regain a clean raw-REPL prompt after an aborted _receive_file.

        The board-side receiver self-exits ~10 s after the last byte and
        deletes its partial file. Anything sent before that would be consumed
        as file content (ctrl-C included, the receiver disables it), so wait
        passively for the function's raw-REPL completion, then drain."""
        try:
            self.follow(_TRANSFER_RESYNC_S)
        except PyboardError:
            pass
        time.sleep(0.1)
        try:
            self.serial.reset_input_buffer()
        except Exception:
            pass

    def transfer_folder(
        self, folder_path, target_folder=None, file_type="all", files="all", remove_files=True, show_progress=False
    ):
        """Copy a folder into the root directory of the pyboard.  Folders that
        contain subfolders will not be copied successfully.  To copy only files of
        a specific type, change the file_type argument to the file suffix (e.g. 'py').
        To copy only specified files pass a list of file names as files argument."""
        if not target_folder:
            target_folder = os.path.split(folder_path)[-1]
        if files == "all":
            files = os.listdir(folder_path)
            if file_type != "all":
                files = [f for f in files if f.split(".")[-1] == file_type]
        try:
            self.exec("os.mkdir({})".format(repr(target_folder)))
        except PyboardError:
            # Folder already exists.
            if remove_files:  # Remove any files not in sending folder.
                target_files = self.get_folder_contents(target_folder)
                remove_files = list(set(target_files) - set(files))
                for f in remove_files:
                    target_path = target_folder + "/" + f
                    self.remove_file(target_path)
        for f in files:
            file_path = os.path.join(folder_path, f)
            target_path = target_folder + "/" + f
            self.transfer_file(file_path, target_path)
            if show_progress:
                self.print(".", end="")

    def remove_file(self, file_path):
        """Remove a file from the pyboard."""
        try:
            self.exec("os.remove({})".format(repr(file_path)))
        except PyboardError:
            pass  # File does not exist.

    def get_folder_contents(self, folder_path):
        """Return a list of the files in a folder on the pyboard."""
        return eval(self.eval("os.listdir({})".format(repr(folder_path))).decode())

    # ------------------------------------------------------------------------------------
    # pyControl operations.
    # ------------------------------------------------------------------------------------

    def load_framework(self):
        """Copy the pyControl framework folder to the board, reset the devices folder
        on pyboard by removing all devices files, and rebuild the device_class2file dict.

        Raises PyboardError on framework-import failure so the GUI surfaces
        a real error (red status) instead of falsely reporting success.
        """
        self.print("\nTransferring pyControl framework to pyboard.", end="")
        self.transfer_folder(os.path.join("source", "pyControl"), file_type="py", show_progress=True)
        self.transfer_folder(user_folder("devices"), files=["__init__.py"], remove_files=True, show_progress=True)
        self.remove_file("hardware_definition.py")
        self.make_device_class2file_map()
        error_message = self.reset()
        if not self.status["framework"]:
            self.print("\n" + ERR_FRAMEWORK_IMPORT_ERROR)
            if error_message:
                self.print(error_message)
            # Surface to callers (universal_config dialog, RunTask.mcu_load_framework)
            # so the per-box status line flips to red instead of green.
            raise PyboardError(ERR_FRAMEWORK_IMPORT_ERROR,
                               (error_message or "").encode())
        self.print(" OK")

    def load_hardware_definition(self, hwd_path):
        """Transfer a hardware definition file to pyboard.

        Raises PyboardError on import failure (e.g. ``Pin(PD5) doesn't exist``)
        and FileNotFoundError when the path doesn't exist, so the GUI shows a
        red error instead of green OK.
        """
        if not os.path.exists(hwd_path):
            self.print(ERR_HW_DEF_NOT_FOUND)
            raise FileNotFoundError(ERR_HW_DEF_NOT_FOUND + f" ({hwd_path})")

        self.transfer_device_files(hwd_path)
        # Exact device sync (HD context only, the HD knows the real used set):
        # remove any device .py on the MCU that this HD does not use, so
        # devices/ holds exactly the HD's drivers after switching HDs. HD-based
        # only, never task-based, which would wipe the HD's drivers.
        used = set(self._get_used_device_files(hwd_path))
        for f in list(self.device_files_on_pyboard):
            if f == "__init__.py" or not f.endswith(".py"):
                continue
            if f not in used:
                self.remove_file("devices/" + f)
                self.device_files_on_pyboard.pop(f, None)
        self.print("\nTransferring hardware definition to pyboard.", end="")
        self.transfer_file(hwd_path, target_path="hardware_definition.py")
        self.reset()
        try:
            self.exec("import hardware_definition")
        except PyboardError as e:
            error_message = e.args[2].decode() if len(e.args) > 2 else str(e)
            self.print("\n\n" + ERR_HW_DEF_IMPORT_ERROR + "\n")
            self.print(error_message)
            raise PyboardError(ERR_HW_DEF_IMPORT_ERROR,
                               e.args[2] if len(e.args) > 2 else b"")

        self.print(" OK")
        # Snapshot HD lineage for State_machine_info → the .tsv header.
        # Reuse the hash transfer_file just recorded (one hash source).
        self._loaded_hwd_path = hwd_path
        self._loaded_hwd_hash = self._upload_hashes.get(
            "hardware_definition.py", _djb2_file(hwd_path))

    def transfer_device_files(self, ref_file_path):
        """Transfer device driver files defining classes used in ref_file to
        the pyboard devices folder. Driver files already on the pyboard
        are only transferred if they have changed on the computer.
        """
        used_device_files = self._get_used_device_files(ref_file_path)
        files_to_transfer = []
        for device_file in used_device_files:
            if device_file not in self.device_files_on_pyboard:
                files_to_transfer.append(device_file)
            else:
                file_hash = _djb2_file(os.path.join(user_folder("devices"), device_file))
                if file_hash != self.device_files_on_pyboard[device_file]:
                    files_to_transfer.append(device_file)
        if files_to_transfer:
            self.print(f"\nTransfering device driver files {files_to_transfer} to pyboard", end="")
            self.transfer_folder(
                user_folder("devices"), files=files_to_transfer, remove_files=False, show_progress=True
            )
            self.reset()
            self.print(" OK")

    @staticmethod
    def _identifier_names_in_source(source):
        """Return the set of identifier (NAME) tokens used in ``source``.

        Comments and string literals are skipped via ``tokenize`` so a
        class name that appears only inside a ``# Door stepper motor``
        comment or a docstring no longer triggers a false-positive
        device-file transfer. Falls back to an empty set on parse error;
        callers degrade to the older substring search in that case so
        a malformed file never silently transfers nothing.
        """
        names = set()
        try:
            for tok in tokenize.generate_tokens(io.StringIO(source).readline):
                if tok.type == tokenize.NAME:
                    names.add(tok.string)
        except (tokenize.TokenizeError, IndentationError, SyntaxError):
            return None  # signal caller to fall back
        return names

    def _get_used_device_files(self, ref_file_path):
        """Return a list of device driver file names containing device classes
        used in ``ref_file``.
        """
        ref_file_name = os.path.split(ref_file_path)[-1]
        with open(ref_file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
        names = Pycboard._identifier_names_in_source(file_content)
        if names is None:
            # Tokenise failed, fall back to the older substring match so
            # an unparseable HW def still gets *something* through.
            device_files = [
                device_file
                for device_class, device_file in Pycboard.device_class2file.items()
                if device_class in file_content and ref_file_name != device_file
            ]
        else:
            device_files = [
                device_file
                for device_class, device_file in Pycboard.device_class2file.items()
                if device_class in names and ref_file_name != device_file
            ]
        # Recurse into transferred device files so nested deps (e.g.
        # five_poke.py legitimately uses the Poke class) come along.
        for device_file in device_files.copy():
            device_files += self._get_used_device_files(os.path.join(user_folder("devices"), device_file))
        device_files = list(set(device_files))  # Remove duplicates.
        return device_files

    def make_device_class2file_map(self):
        """Make dict mapping device class names to file in devices folder containing
        the class definition.

        Built into a LOCAL dict and published with ONE reference assignment.
        ``device_class2file`` is a CLASS attribute shared by every box's
        Pycboard and read concurrently during a PARALLEL Universal Config
        (framework load on N boxes at once); a single rebind is atomic from a
        reader's view (it sees either the complete old map or the complete new
        map), so no box's ``_get_used_device_files`` observes a half-filled map
        and transfers no drivers."""
        mapping = {}  # {device_classname: device_filename}
        all_device_files = [f for f in os.listdir(user_folder("devices")) if f.endswith(".py")]
        for device_file in all_device_files:
            with open(os.path.join(user_folder("devices"), device_file), "r") as f:
                file_content = f.read()
            pattern = r"[\n\r]class\s*(?P<dcname>\w+)\s*"
            device_classes = list(set(re.findall(pattern, file_content)))
            for device_class in device_classes:
                mapping[device_class] = device_file
        Pycboard.device_class2file = mapping

    @staticmethod
    def _error_text(err):
        """Flatten a PyboardError's args (str + bytes) into one string."""
        parts = []
        for a in getattr(err, "args", ()):
            if isinstance(a, (bytes, bytearray)):
                parts.append(bytes(a).decode(errors="replace"))
            elif isinstance(a, str):
                parts.append(a)
        return " ".join(parts) or str(err)

    @classmethod
    def _is_genuine_task_error(cls, err):
        """True for a real Python error in the user's task/HD code (won't be
        fixed by a re-sync → surface now); False for a raw-REPL desync
        ('<stdin>' SyntaxError, could-not-enter-raw-repl, exec/transfer/
        timeout, SerialException), which is transient → retry."""
        if isinstance(err, SerialException):
            return False
        text = cls._error_text(err)
        if "<stdin>" in text:
            return False
        return ("task_file.py" in text) or ("hardware_definition.py" in text)

    def _raise_task_setup_failed(self, err):
        """Surface a task-setup failure with its real traceback detail."""
        detail = (err.args[2] if isinstance(err, PyboardError)
                  and len(err.args) > 2 else str(err).encode())
        detail = bytes(detail) if isinstance(detail, (bytes, bytearray)) else str(detail).encode()
        self.print("\n\n" + ERR_TASK_SETUP_FAILED + "\n\n" + detail.decode(errors="replace"))
        raise PyboardError(ERR_TASK_SETUP_FAILED, detail)

    def setup_state_machine(self, sm_name, sm_dir=None, uploaded=False):
        """Transfer state machine descriptor file sm_name.py from folder sm_dir
        to board and setup state machine on pyboard."""
        if sm_dir is None:
            sm_dir = user_folder("tasks")
        sm_path = os.path.join(sm_dir, sm_name + ".py")
        if not uploaded and not os.path.exists(sm_path):
            self.print(ERR_TASK_FILE_NOT_FOUND.format(path=sm_path))
            raise PyboardError(ERR_TASK_FILE_NOT_FOUND.format(path=sm_path))

        def _attempt():
            # One full, self-contained upload attempt. EVERY raw-REPL step is
            # here, soft-reboot, transfers, the two task-load execs, and the
            # sm_info read-back, so a desync at ANY step is retried as a unit
            # by the loop below.
            self.serial.reset_input_buffer()
            self.reset()
            if uploaded:
                self.print("\nResetting task. ", end="")
            else:
                self.transfer_device_files(sm_path)
                self.print("\nTransferring state machine {} to pyboard. ".format(sm_name), end="")
                self.transfer_file(sm_path, "task_file.py")
            self.gc_collect()
            # Two separate execs (one short command each), less desync-prone
            # than one combined multi-line exec.
            self.exec("import task_file")
            self.exec("sm.setup_state_machine(task_file)")
            # states/events/variables/coordinates in ONE repr/eval round-trip
            # (repr preserves int keys + tuple values JSON would mangle).
            info = eval(self.eval(
                "{'states': sm.states, 'events': sm.events,"
                " 'variables': {k: v for k, v in sm.variables.__dict__.items() if not hasattr(v, '__init__')},"
                " 'coordinates': ({k: v for k, v in ut.c.__dict__.items() if not hasattr(v, '__init__')}"
                "                 if 'ut' in dir() and hasattr(ut, 'c') else {})}").decode())
            info["analog_inputs"] = self.get_analog_inputs()
            self.print("OK")
            return info

        # Retry the whole upload on a transient raw-REPL desync; surface a
        # genuine task/HD code error immediately (a re-sync can't fix it).
        info = None
        last_err = None
        for attempt in range(_TASK_SETUP_ATTEMPTS):
            try:
                info = _attempt()
                last_err = None
                break
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                if self._is_genuine_task_error(e):
                    self._raise_task_setup_failed(e)
                last_err = e
                self.print(f" upload attempt {attempt + 1} failed, re-syncing board. ")
                time.sleep(_TASK_SETUP_SETTLE_S * (attempt + 1))
        if last_err is not None:
            self._raise_task_setup_failed(last_err)
        states = info["states"]
        events = info["events"]
        devices = {f: h for f, h in self.device_files_on_pyboard.items()
                   if f.endswith(".py") and f != "__init__.py"}
        # Reuse the HD hash transfer_file recorded this session; else ask the
        # MCU for the hash of the hardware_definition.py it holds.
        hd_hash = self._loaded_hwd_hash
        if not hd_hash:
            mcu_hd_hash = self.get_file_hash("hardware_definition.py")
            hd_hash = mcu_hd_hash if mcu_hd_hash > 0 else 0
        self.sm_info = State_machine_info(
            name=sm_name,
            # Reuse the hash transfer_file recorded (one hash source); fall back
            # to a fresh hash on the reset-task path that skips the transfer.
            task_hash=self._upload_hashes.get("task_file.py", _djb2_file(sm_path)),
            states=states,  # {name:ID}
            events=events,  # {name:ID}
            ID2name={ID: name for name, ID in {**states, **events}.items()},  # {ID:name}
            analog_inputs=info["analog_inputs"],  # {ID: {'name':, 'fs':, 'dtype': 'plot':}}
            variables=info["variables"],
            coordinates=info["coordinates"],
            framework_version=self.framework_version,
            micropython_version=self.micropython_version,
            hardware_def_name=os.path.basename(self._loaded_hwd_path) if self._loaded_hwd_path else "",
            hardware_def_hash=hd_hash,
            devices=devices,
        )
        self.data_logger.reset()
        self.timestamp = 0
        self._last_message_mono = host_clock.host_s()
        # False until the first MCU message of THIS run lands. ``fw_ms_at``
        # returns None while False so video rows stamp ``na`` instead of a
        # stale carry-over from the previous run (or a pre-first-message 0).
        self._fw_anchored = False
        # Propagate sm_info to any data_consumer that wants it (TaskInfo,
        # TaskPlot, etc.). Single attach point, widgets don't need to
        # re-implement this for maze and operant separately.
        for consumer in (self.data_consumers or []):
            attach = getattr(consumer, "set_state_machine", None)
            if callable(attach):
                try:
                    attach(self.sm_info)
                except Exception:
                    pass

    def get_analog_inputs(self):
        """Return analog_inputs as a dictionary: {ID: {'name':, 'fs':, 'dtype': 'plot':}}"""
        return eval(self.exec("hw.get_analog_inputs()").decode().strip())

    def start_framework(self, data_output=True, timeout=2):
        """Start the pyControl framework running on the pyboard.

        ``timeout`` bounds the gc.collect + data_output round-trips so a board
        that stalls mid-output can't freeze the GUI for the default 10 s.
        """
        # Flush stale bytes so they aren't misread as the first exec response.
        self.serial.reset_input_buffer()
        self.gc_collect(timeout=timeout)
        self.exec("fw.data_output = " + repr(data_output), timeout=timeout)
        self.serial.reset_input_buffer()
        self.last_message_time = time.time()
        self._last_message_mono = host_clock.host_s()
        self.timestamp = 0
        self._fw_anchored = False   # re-anchored on this run's first message
        self.exec_raw_no_follow("fw.run()")
        self.framework_running = True

    def stop_framework(self):
        """Stop framework running on pyboard by sending stop command."""
        self.serial.write(b"\x03")  # Stop signal
        self.framework_running = False

    # Frozen sets for fast type membership checks in process_data hot loop.
    _EVENT_STATE = frozenset((MsgType.EVENT, MsgType.STATE))
    _PRINT_WARNG_THRSH = frozenset((MsgType.PRINT, MsgType.WARNG, MsgType.THRSH))

    def _resync_to_message_start(self):
        """After a misframed / short / bad-checksum read, discard input up to
        the next message-start byte (0x07) so one corrupt frame costs one row
        instead of cascading into every following message (a misframed
        message_len would otherwise over/under-consume the rest of the stream).
        Consumes the 0x07; returns True if found (stream now positioned just
        after it, ready to re-parse), False if the input drained first."""
        while self.serial.in_waiting > 0:
            if self.serial.read(1) == b"\x07":
                return True
        return False

    def process_data(self):
        """Read data from serial line, generate list new_data of data tuples,
        pass new_data to data_logger and print_func if specified, return new_data."""
        # Flush cross-thread write requests (camera pipeline events) on this
        # GUI-thread tick so every serial.write stays on one thread.
        self._drain_pending_writes()
        new_data = []
        error_message = None
        unexpected_input = []
        while self.serial.in_waiting > 0:
            new_byte = self.serial.read(1)
            if new_byte == b"\x07":  # Start of pyControl message.
                # Output any unexpected characters recived prior to message start.
                if unexpected_input:
                    new_data.append(
                        Datatuple(
                            time=self.get_timestamp(),
                            type=MsgType.WARNG,
                            content=ERR_UNEXPECTED_MCU_INPUT.format(data="".join(unexpected_input)),
                        )
                    )
                    unexpected_input = []
                # Parse the framed message after this \x07. On a short/bad/
                # malformed frame, resync to the next \x07 so one bad frame
                # costs one row instead of misframing the rest of the stream.
                at_start = True
                while at_start:
                    at_start = False
                    bad_frame = False
                    # Read header (checksum + length) in one call to reduce syscalls.
                    header = self.serial.read(4)
                    if len(header) < 4:  # Short read, board silent / desynced.
                        bad_frame = True
                    else:
                        checksum = int.from_bytes(header[:2], "little")
                        message_len = int.from_bytes(header[2:4], "little")
                        message = self.serial.read(message_len)
                        if len(message) < message_len:  # Short body.
                            bad_frame = True
                        else:
                            try:
                                msg_type = MsgType.from_byte(message[4:5])
                                msg_subtype = msg_type.get_subtype(message[5:6].decode())
                                content_bytes = message[6:]
                                # Compute checksum
                                if msg_type == MsgType.ANLOG:  # Extract analog data to compute checksum.
                                    ID = int.from_bytes(content_bytes[:2], "little")
                                    data = array(self.sm_info.analog_inputs[ID]["dtype"], content_bytes[2:])
                                    content = (ID, data)
                                    msg_sum = sum(message[:8]) + sum(data)
                                else:
                                    msg_sum = sum(message)
                                if checksum == (msg_sum & 0xFFFF):  # Checksum OK.
                                    # First message of the run → host↔fw clock map is valid.
                                    self._fw_anchored = True
                                    msg_timestamp = int.from_bytes(message[:4], "little")
                                    if msg_timestamp > self.timestamp:
                                        # Anchor both wall + monotonic clocks; get_timestamp
                                        # extrapolates from the monotonic one (immune to
                                        # system-clock corrections).
                                        self.last_message_time = time.time()
                                        _mono_now = host_clock.host_s()
                                        self._last_message_mono = _mono_now
                                        self.timestamp = msg_timestamp
                                        # Atomic pair for cross-thread readers
                                        # (RecorderSink worker): two separate
                                        # loads of timestamp+mono can interleave
                                        # with these stores and skew frame_fw_ms
                                        # by one inter-message interval.
                                        self._fw_anchor = (msg_timestamp, _mono_now)
                                    if msg_type in Pycboard._EVENT_STATE:
                                        content = int(content_bytes.decode())  # Event/state ID.
                                    elif msg_type in Pycboard._PRINT_WARNG_THRSH:
                                        content = content_bytes.decode()  # Print or error string.
                                    elif msg_type == MsgType.VARBL:
                                        content = content_bytes.decode()  # JSON string
                                        self.sm_info.variables.update(json.loads(content))
                                    new_data.append(
                                        Datatuple(time=msg_timestamp, type=msg_type, subtype=msg_subtype, content=content)
                                    )
                                else:  # Bad checksum.
                                    bad_frame = True
                            except Exception:  # Malformed payload (short msg_len, bad ID, bad JSON…).
                                bad_frame = True
                    if bad_frame:
                        new_data.append(
                            Datatuple(time=self.get_timestamp(), type=MsgType.WARNG, content=ERR_BAD_CHECKSUM)
                        )
                        at_start = self._resync_to_message_start()
            elif new_byte == b"\x04":  # End of framework run.
                self.framework_running = False
                data_err = self.read_until(2, b"\x04>", timeout=10)
                if len(data_err) > 2:  # Error during framework run.
                    error_message = data_err[:-3].decode()
                    new_data.append(Datatuple(time=self.get_timestamp(), type=MsgType.ERROR, content=error_message))
                break
            else:
                unexpected_input.append(new_byte.decode())
        if new_data:
            self.data_logger.process_data(new_data)
            if self.data_consumers:
                for data_consumer in self.data_consumers:
                    if data_consumer is None:
                        continue
                    data_consumer.process_data(new_data)
        if error_message:
            raise PyboardError(error_message)

    def trigger_event(self, event_name, source="u"):
        """Trigger specified task event on the pyboard."""
        if self.framework_running:
            event_ID = str(self.sm_info.events[event_name])
            self.send_serial_data(event_ID, "E", source)

    def queue_trigger_event(self, event_name, source="u"):
        """Enqueue an MCU event from a worker thread.
        Used by the camera pipeline's MCUPusher worker. The actual
        ``serial.write`` happens on the GUI thread inside
        ``process_data`` (which drains ``_pending_writes`` at the
        start of each tick), same thread Original pyControl uses
        for every serial op. Latency = at most one ``process_data``
        tick (10 ms by default).
        """
        self._pending_writes.put(("event", host_clock.host_ns(),
                                  event_name, source))

    def trigger_intrinsic_event(self, event_name):
        """Dispatch a FRAMEWORK-INTRINSIC event without TSV logging.
        """
        if not self.framework_running:
            return
        event_ID = str(self.sm_info.events[event_name])
        data = event_ID.encode()
        data_len = len(data).to_bytes(2, "little")
        checksum = (sum(data) & 0xFFFF).to_bytes(2, "little")
        self.serial.write(b"Z" + data_len + data + checksum)

    def queue_trigger_intrinsic_event(self, event_name):
        """Enqueue a silent intrinsic event from a worker thread.

        Same threading pattern as ``queue_trigger_event``, the actual
        ``serial.write`` happens on the GUI thread via
        ``_drain_pending_writes``.
        """
        self._pending_writes.put(("intrinsic_event", host_clock.host_ns(),
                                  event_name))

    # Maze nav protocol, wire format mirrors framework.py NAV_TYP parser:
    #   b"N" + bytes([code])  where code is 0/1/2/3
    NAV_CODES = {"pause": 0, "resume": 1, "next_stage": 2, "prev_stage": 3}

    def trigger_nav(self, name):
        """Send a maze nav command (pause/resume/next_stage/prev_stage) to the
        pyboard. Bypasses the events dict, uses the dedicated NAV_TYP byte.
        No-op if name is unknown or framework not running."""
        if not self.framework_running:
            return
        code = self.NAV_CODES.get(name)
        if code is None:
            return
        self.serial.write(b"N" + bytes([code]))

    def _fw_ms_for_mono(self, mono_s):
        """Map a host ``host_clock.host_s()`` instant (seconds) to MCU
        framework ms via the last-message anchor. One formula shared by
        ``get_timestamp`` (now) and ``fw_ms_at`` (a past capture instant).

        Reads the anchor as ONE tuple load; this runs on sink worker
        threads while the GUI-thread reader updates the anchor."""
        anchor = getattr(self, "_fw_anchor", None)
        if anchor is None:
            anchor = (self.timestamp, self._last_message_mono)
        fw_ms, anchor_mono = anchor
        return fw_ms + round(1000 * (mono_s - anchor_mono))

    def get_timestamp(self):
        """Get the current pyControl timestamp in ms since start of
        framework run.

        Uses ``host_clock.host_s()`` for the host-side elapsed delta,
        immune to NTP slew / DST jumps (a ``time.time()`` delta can go
        backwards on a clock correction and emit a stale/negative ms).
        """
        return self._fw_ms_for_mono(host_clock.host_s())

    def fw_ms_at(self, host_ns):
        """MCU framework ms at a host capture instant
        (``host_clock.host_ns()``, the same clock the cameras stamp with,
        which is what makes this mapping meaningful)."""
        if not getattr(self, "_fw_anchored", False):
            return None
        return self._fw_ms_for_mono(host_ns / 1e9)

    def send_serial_data(self, data, command, cmd_type=""):
        """Send data to the pyboard while framework is running.

        The checksum is masked to 16 bits because that is what the board
        compares against (``sum(...) & 0xFFFF`` in ``framework.receive_data``).
        Unmasked, ``to_bytes(2)`` raises OverflowError once the payload's byte
        sum passes 65535, about 650 characters of text or a list of a few
        hundred numbers, so a large ``set_variable`` mid-run aborted the call
        instead of sending it. Below that threshold the mask changes nothing,
        so every payload that worked before is byte-identical on the wire.
        """
        encoded_data = cmd_type.encode() + data.encode()
        data_len = len(encoded_data).to_bytes(2, "little")
        checksum = (sum(encoded_data) & 0xFFFF).to_bytes(2, "little")
        self.serial.write(command.encode() + data_len + encoded_data + checksum)

    # ------------------------------------------------------------------------------------
    # Getting and setting variables.
    # ------------------------------------------------------------------------------------

    def set_variable(self, v_name, v_value, source="s"):
        """Set the value of a state machine variable. If framework is not running
        returns True if variable set OK, False if set failed.  Returns None framework
        running, but variable event is later output by board."""
        if v_name not in self.sm_info.variables:
            raise PyboardError("Invalid variable name: {}".format(v_name))
        if self.framework_running:  # Set variable with serial command.
            self.send_serial_data(repr((v_name, v_value)), "V", source)
            return None
        else:  # Set variable using REPL.
            try:
                set_OK = eval(self.eval(f"sm.set_variable({repr(v_name)}, {repr(v_value)})").decode())
            except Exception as e:
                # Garbled REPL reply → treat as "set failed", don't propagate.
                logger.warning("set_variable(%s) REPL parse failed: %r", v_name, e)
                return False
            if set_OK:
                self.sm_info.variables[v_name] = v_value
            return set_OK

    def set_coordinates(self, c_name, c_value):
        """Push a coordinate value into the task's c namespace at runtime.

        Used for streaming tracking data (animal x/y/speed) AND for GUI control
        flags consumed by maze tasks (c.pause_state, c.pass_inter_state, etc.).
        Wire format matches the b'c' command handler in pyControl framework.
        Only effective while framework is running.
        """
        if not self.framework_running:
            return
        # ONE repr of the (name, value) tuple. The MCU evals it back to the
        # native value, so a string zone name arrives as the str "RightArm"
        # (not "'RightArm'") and ``c.loc_center in right_zones`` matches.
        # A second repr on the value would double-quote strings and turn
        # numbers into strings, breaking every task-side comparison.
        data = repr((c_name, c_value)).encode() + b"c"
        data_len = len(data).to_bytes(2, "little")
        checksum = (sum(data) & 0xFFFF).to_bytes(2, "little")
        self.serial.write(b"c" + data_len + data + checksum)

    def queue_set_coordinates(self, c_name, c_value):
        """Enqueue a coordinate push from a worker thread.

        Same pattern as ``queue_trigger_event``, the camera pipeline's
        MCUPusher worker calls this; the actual ``serial.write`` happens
        on the GUI thread inside ``process_data``."""
        self._pending_writes.put(("coord", host_clock.host_ns(),
                                  c_name, c_value))

    def set_write_latency_budget(self, budget) -> None:
        """Attach the pipeline's LatencyBudget so each drained write records
        how long it waited between being queued on the pusher thread and
        actually reaching the port. Observational; ``None`` detaches."""
        self._write_latency_budget = budget

    def _drain_pending_writes(self):
        """Run any cross-thread write requests on the GUI thread.

        Pycboard is deliberately single-threaded, mirroring upstream
        pyControl: worker threads enqueue, this drain is the only writer.
        """
        if self._pending_writes.empty():
            return
        while True:
            try:
                item = self._pending_writes.get_nowait()
            except _queue.Empty:
                return
            if not self.framework_running:
                continue  # drop silently, same as the underlying methods
            kind, queued_ns = item[0], item[1]
            try:
                if kind == "event":
                    _, _, event_name, source = item
                    # The event name must exist in the running task's events
                    # list, or ``trigger_event`` raises KeyError deep inside
                    # ``sm_info.events[...]``, which the generic handler below
                    # mislabels as "port may be down". This is the silent
                    # cross-file contract from the authoring guide: name it
                    # plainly and actionably instead (rate-limited per name).
                    # Guarded so a board object without ``sm_info`` (test
                    # doubles) still routes through ``trigger_event``.
                    events = getattr(getattr(self, "sm_info", None), "events", None)
                    if events is not None and event_name not in events:
                        seen = getattr(self, "_unknown_event_warned", None)
                        if seen is None:
                            seen = self._unknown_event_warned = set()
                        if event_name not in seen:
                            seen.add(event_name)
                            logger.warning(
                                "MCU trigger event %r is not in the running "
                                "task's events list, event dropped. Add %r to "
                                "the task's `events` list (see "
                                "AI_AUTHORING_GUIDE cross-file contract).",
                                event_name, event_name)
                        continue  # this event is not deliverable; drop it
                    self.trigger_event(event_name, source=source)
                elif kind == "intrinsic_event":
                    _, _, event_name = item
                    # Same contract as the named-event branch: the framework
                    # auto-injects zone_changed / frame_event, so a missing one
                    # means a STALE framework, not a dead port. Name it, the
                    # generic handler would otherwise raise the false "port may
                    # be down" banner for exactly this case.
                    events = getattr(getattr(self, "sm_info", None), "events", None)
                    if events is not None and event_name not in events:
                        seen = getattr(self, "_unknown_event_warned", None)
                        if seen is None:
                            seen = self._unknown_event_warned = set()
                        if event_name not in seen:
                            seen.add(event_name)
                            logger.warning(
                                "MCU intrinsic event %r is not in the running "
                                "task's events, the pyControl framework on the "
                                "board is stale. Re-upload framework.py + "
                                "state_machine.py so %r is auto-injected.",
                                event_name, event_name)
                        continue  # stale framework on the board; drop it
                    self.trigger_intrinsic_event(event_name)
                elif kind == "coord":
                    _, _, c_name, c_value = item
                    self.set_coordinates(c_name, c_value)
                # Timing spine: the write just happened, so this is the real
                # "queued -> on the wire" delay. Recorded only on success,
                # a dropped or failed write never reached the MCU, and
                # timing it would report a latency that did not occur.
                budget = getattr(self, "_write_latency_budget", None)
                if budget is not None:
                    budget.record_from_ns("push_to_wire", queued_ns)
            except BaseException as e:
                # Catch PyboardError too; rate-limit the warning to once / 5 s.
                now = time.monotonic()
                if now - getattr(self, "_drain_warn_at", 0.0) > 5.0:
                    self._drain_warn_at = now
                    logger.warning(
                        "pycboard drain write failed (port may be down): %r", e)
                    # Opt-in surfacing: when the owner set ``health_hook`` the
                    # failure raises a per-box banner (coords/events silently
                    # stopped reaching the board) instead of a log line only.
                    hook = getattr(self, "health_hook", None)
                    if hook is not None:
                        try:
                            hook("mcu_write_failed")
                        except Exception:
                            pass

    def get_variable(self, v_name):
        """Get the value of a state machine variable. If framework not running returns
        variable value if got OK, None if get fails.  Returns None if framework
        running, but variable event is later output by board."""
        if v_name not in self.sm_info.variables:
            raise PyboardError("Invalid variable name: {}".format(v_name))
        if self.framework_running:  # Get variable with serial command.
            self.send_serial_data(v_name, "V", "g")
        else:  # Get variable using REPL.
            var_str = self.eval(f"sm.get_variable({repr(v_name)})").decode()
            try:
                return eval(var_str)
            except Exception:  # Variable is a string.
                return var_str

    def get_variables(self):
        """Return variables as a dictionary {v_name: v_value}.
        """
        try:
            return eval(self.eval(
                "{k: v for k, v in sm.variables.__dict__.items() if not hasattr(v, '__init__')}"
            ).decode())
        except Exception as e:
            logger.warning("get_variables REPL parse failed (no capture this stop): %r", e)
            return {}

    def set_variables(self, var_dict, source="s"):
        """Set MANY state-machine variables in ONE REPL round-trip.
        """
        if not var_dict:
            return {}
        if self.framework_running:
            return {k: self.set_variable(k, v, source)
                    for k, v in var_dict.items()}
        items = {k: v for k, v in var_dict.items()
                 if k in self.sm_info.variables}
        if not items:
            return {}
        # One round-trip: the MCU sets each and returns {name: ok}. The dict
        # comprehension preserves the literal's key order, so no host-side
        # zip alignment is needed.
        try:
            results = eval(self.eval(
                "{_k: sm.set_variable(_k, _v) for _k, _v in "
                + repr(items) + ".items()}").decode())
        except Exception as e:
            # Batch reply garbled (transient desync) → fall back to per-variable
            # sets so an Upload still applies what it can.
            logger.warning("set_variables batch parse failed (%r); retrying per-variable", e)
            results = {k: self.set_variable(k, v, source) for k, v in items.items()}
            if not all(results.values()):
                logger.error("set_variables: %d of %d variables not applied",
                             sum(1 for ok in results.values() if not ok), len(items))
            return results
        for k, ok in results.items():
            if ok:
                self.sm_info.variables[k] = items[k]
        return results
