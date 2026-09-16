"""MCU message types and parsed-message namedtuple.

Defines the binary message types used in PC <-> MCU communication and the
Datatuple namedtuple for representing parsed messages internally.
"""

from enum import Enum
from collections import namedtuple

Datatuple = namedtuple("Datatuple", ["time", "type", "subtype", "content"], defaults=[None] * 4)


class MsgType(Enum):
    EVENT = b"E"  # External event
    STATE = b"S"  # State transition
    PRINT = b"P"  # User print
    HARDW = b"H"  # Hardware callback
    VARBL = b"V"  # Variable change
    WARNG = b"!"  # Warning
    ERROR = b"!!"  # Error
    STOPF = b"X"  # Stop framework
    ANLOG = b"A"  # Analog
    THRSH = b"T"  # Threshold

    @classmethod
    def from_byte(cls, byte_value):
        """Get member given value byte. Uses pre-built dict for O(1) lookup."""
        return cls._byte_lookup.get(byte_value, byte_value)

    def get_subtype(self, subtype_char):
        """Get subtype name from character. Uses pre-built lookup tables."""
        if subtype_char == "_":
            return None
        subtypes = _SUBTYPE_LOOKUP.get(self)
        if subtypes is None:
            raise KeyError(f"No subtypes defined for {self}")
        return subtypes[subtype_char]


# Bytes value -> MsgType member, built once at import time for O(1) lookup.
MsgType._byte_lookup = {member.value: member for member in MsgType}

# Subtype lookup dicts, built once at import time.
_SUBTYPE_LOOKUP = {
    MsgType.VARBL: {
        "g": "get",
        "s": "user_set",
        "a": "api_set",
        "p": "print",
        "t": "run_start",
        "e": "run_end",
    },
    MsgType.EVENT: {
        "i": "input",
        "t": "timer",
        "p": "publish",
        "u": "user",
        "a": "api",
        "s": "sync",
    },
    MsgType.PRINT: {
        "t": "task",
        "a": "api",
        "u": "user",
        "s": "trigger",
    },
    MsgType.THRSH: {
        "s": "run_start",
        "t": "task",
    },
}


__all__ = ["MsgType", "Datatuple"]
