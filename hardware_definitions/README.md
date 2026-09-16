# Hardware definitions

One file per rig, naming the board and mapping each device onto a port.
This folder ships empty because a definition describes physical wiring,
which is specific to the bench it was written for.

A definition instantiates a breakout board and then one object per
device, each on a named port. The board revision is part of the wiring:
the same port can carry different pins between revisions, and declaring
the wrong one drives a pin with nothing on it, which looks like a broken
device rather than a wrong file.

See `docs/` for the device reference.
