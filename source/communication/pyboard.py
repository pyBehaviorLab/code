#!/usr/bin/env python

"""
pyboard interface
This module provides the Pyboard class, used to communicate with and
control the pyboard over a serial USB connection.
Example usage:
    import pyboard
    pyb = pyboard.Pyboard('/dev/ttyACM0')
    pyb.enter_raw_repl()
    pyb.exec('pyb.LED(1).on()')
    pyb.exit_raw_repl()
"""

import time
import serial


class PyboardError(BaseException):
    pass


class Pyboard:
    def __init__(self, serial_device, baudrate=115200):
        # 2 s read timeout so a silent/desynced board surfaces as a short read
        # instead of blocking the GUI thread forever; healthy USB-CDC frames
        # arrive in <<1 ms so it never trips mid-frame.
        self.serial = serial.Serial(serial_device, baudrate=baudrate, interCharTimeout=1, timeout=2)

    def close(self):
        self.serial.close()

    def read_until(self, min_num_bytes, ending, timeout=10, data_consumer=None):
        # Poll in_waiting up to `timeout` rather than a blocking read, so a
        # silent board can't hang the handshake.
        deadline = None if timeout is None else time.monotonic() + timeout
        data = b""
        while len(data) < min_num_bytes:
            avail = self.serial.in_waiting
            if avail > 0:
                data += self.serial.read(min(min_num_bytes - len(data), avail))
            elif deadline is not None and time.monotonic() > deadline:
                break
            else:
                time.sleep(0.01)
        if data_consumer and data:
            data_consumer(data)
        timeout_count = 0
        while True:
            if data.endswith(ending):
                break
            elif self.serial.in_waiting > 0:
                new_data = self.serial.read(1)
                data = data + new_data
                if data_consumer:
                    data_consumer(new_data)
                timeout_count = 0
            else:
                timeout_count += 1
                if timeout is not None and timeout_count >= 50 * timeout:
                    break
                time.sleep(0.02)
        return data

    def enter_raw_repl(self):
        self.serial.write(b"\r\x03\x03")  # ctrl-C twice: interrupt any running program
        # flush input (without relying on serial.flushInput())
        n = self.serial.in_waiting
        while n > 0:
            self.serial.read(n)
            n = self.serial.in_waiting
        self.serial.write(b"\r\x01")  # ctrl-A: enter raw REPL
        data = self.read_until(1, b"to exit\r\n>")
        if not data.endswith(b"raw REPL; CTRL-B to exit\r\n>"):
            print(data)
            raise PyboardError("could not enter raw repl")
        self.serial.write(b"\x04")  # ctrl-D: soft reset
        data = self.read_until(1, b"to exit\r\n>")
        if not data.endswith(b"raw REPL; CTRL-B to exit\r\n>"):
            print(data)
            raise PyboardError("could not enter raw repl")

    def exit_raw_repl(self):
        self.serial.write(b"\r\x02")  # ctrl-B: enter friendly REPL

    def follow(self, timeout, data_consumer=None):
        # wait for normal output
        data = self.read_until(1, b"\x04", timeout=timeout, data_consumer=data_consumer)
        if not data.endswith(b"\x04"):
            raise PyboardError("timeout waiting for first EOF reception")
        data = data[:-1]

        # wait for error output
        data_err = self.read_until(2, b"\x04>", timeout=timeout)
        if not data_err.endswith(b"\x04>"):
            raise PyboardError("timeout waiting for second EOF reception")
        data_err = data_err[:-2]

        # return normal and error output
        return data, data_err

    def exec_raw_no_follow(self, command, timeout=10):
        if isinstance(command, bytes):
            command_bytes = command
        else:
            command_bytes = bytes(command, encoding="utf8")

        # 256-byte chunks with an inter-chunk sleep to pace the MCU buffer.
        for i in range(0, len(command_bytes), 256):
            self.serial.write(command_bytes[i : min(i + 256, len(command_bytes))])
            if i + 256 < len(command_bytes):
                time.sleep(0.01)
        self.serial.write(b"\x04")

        # Wait for the raw-REPL "OK" receipt ack on the caller's timeout budget
        # (not the 2 s port timeout), so a slow-but-alive board under contention
        # isn't declared failed while its command actually landed.
        data = b""
        deadline = time.monotonic() + timeout
        while len(data) < 2:
            avail = self.serial.in_waiting
            if avail > 0:
                data += self.serial.read(min(2 - len(data), avail))
            elif time.monotonic() > deadline:
                break
            else:
                time.sleep(0.005)
        if data != b"OK":
            raise PyboardError("could not exec command")

    def exec_raw(self, command, timeout=10, data_consumer=None):
        self.exec_raw_no_follow(command, timeout=timeout)
        return self.follow(timeout, data_consumer)

    def eval(self, expression):
        ret = self.exec("print({})".format(expression))
        ret = ret.strip()
        return ret

    def exec(self, command, timeout=10):
        ret, ret_err = self.exec_raw(command, timeout=timeout)
        if ret_err:
            raise PyboardError("exception", ret, ret_err)
        return ret
