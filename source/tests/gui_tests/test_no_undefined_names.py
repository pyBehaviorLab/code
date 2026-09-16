"""A name used without its import is a crash that waits for a click.

``NameError: name 'dialog_dir' is not defined`` reached a user from the offline
analyser's "Save results to" button: five call sites in that file imported the
helper locally and the sixth did not, so the button raised the first time it
was pressed. Nothing caught it; it is not a syntax error, no test clicked that
button, and the module imports perfectly well.

A static scan catches the whole class in one pass. The allowlist below is every
hit that is deliberate; anything else is a real one, and this test is what says
so before the operator does.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

#: Hits that are correct, with the reason. A tuple of (path fragment, name).
#:
#: MicroPython builtins exist only on the board, ``source/pyControl`` and the
#: device drivers are uploaded there and cannot be imported on the host at all,
#: which is the whole point of the two-runtime split.
ALLOWED = {
    ("pyControl", "micropython"),
    ("pyControl", "const"),
    ("pycboard.py", "pyb"),
    # Forward references inside string annotations. Never evaluated.
    ("gui/base.py", "SnapshotStore"),
    ("gui/base.py", "RotatedButton"),
    # Closures over a name the enclosing function later ``del``s in its
    # ``finally``. They run inside the ``try``, long before that.
    ("spinnaker.py", "cam"),
    # Deliberately unreachable: the block sits after an unconditional
    # ``return []`` and documents a check this analyser will not perform,
    # because importing the live rig is exactly what would end its
    # independence. Left as a named gap rather than a quiet skip.
    ("validate.py", "MetricsComputer"),
    ("validate.py", "MetricsConfig"),
}


def _allowed(path: str, name: str) -> bool:
    norm = path.replace("\\", "/")
    return any(frag.replace("\\", "/") in norm and name == allowed_name
               for frag, allowed_name in ALLOWED)


def _has_ruff() -> bool:
    """Whether ``python -m ruff`` runs, which is how the project invokes it.

    Checked by running it, not by looking for an executable on PATH: ruff is a
    dependency of this environment, not a system tool, and ``shutil.which``
    misses it entirely.
    """
    try:
        return subprocess.run([sys.executable, "-m", "ruff", "--version"],
                              capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _has_ruff(),
                    reason="ruff is not installed in this environment")
def test_no_name_is_used_without_being_defined():
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "source", "tools",
         "--select", "F821", "--output-format", "concise", "--no-cache"],
        cwd=ROOT, capture_output=True, text=True)

    unexpected = []
    for line in proc.stdout.splitlines():
        if "F821" not in line:
            continue
        # e.g.  source\gui\base.py:3983:32: F821 Undefined name `SnapshotStore`
        location, _, message = line.partition(" F821 ")
        name = message.split("`")[1] if "`" in message else message.strip()
        path = location.split(":")[0]
        if not _allowed(path, name):
            unexpected.append(f"{location.strip()} -> {name}")

    assert not unexpected, (
        "a name is used without being defined or imported; this is the "
        "`dialog_dir` crash class, and it only shows when that line runs:\n  "
        + "\n  ".join(unexpected))
