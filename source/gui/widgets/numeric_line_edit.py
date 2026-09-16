"""QLineEdit-based replacement for QSpinBox / QDoubleSpinBox.

Drop-in: exposes the same ``value()`` / ``setValue()`` / ``valueChanged``
/ ``setRange()`` / ``setSingleStep()`` / ``setDecimals()`` API so call
sites swap the constructor and nothing else.

Behaviour:

* Validator clamps typed input to ``[minimum, maximum]``.
* Integer mode (``decimals=0``) uses ``QIntValidator``.
* Float mode (``decimals>0``) uses ``QDoubleValidator`` in standard
  notation (no exponent, neuroscientists type "0.5", not "5e-1").
* ``valueChanged`` fires on ``editingFinished`` (focus-out or Enter),
  not on every keystroke, matches QSpinBox semantics where partial
  edits don't trip downstream side effects.
* ``auto_value`` gives one out-of-range value a word instead of a
  number ("Auto"), the way QSpinBox's ``specialValueText`` does. It is
  how a setting says "derive this from the data" without spending a
  magic number the operator would otherwise have to know.

Used by the tracking-settings dialog (replaces the spin-box pickers
the operators preferred to type into directly).
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class NumericLineEdit(QtWidgets.QLineEdit):

    valueChanged = QtCore.Signal(float)

    def __init__(self,
                 *,
                 minimum: float = 0,
                 maximum: float = 100,
                 decimals: int = 0,
                 default: float = 0,
                 single_step: float = 1,
                 auto_value=None,
                 auto_text: str = "Auto",
                 parent=None):
        super().__init__(parent)
        self._decimals = int(max(0, decimals))
        self._min = float(minimum)
        self._max = float(maximum)
        self._step = float(single_step)
        self._auto_value = auto_value
        self._auto_text = auto_text
        self._install_validator()
        self.setValue(default)
        self.editingFinished.connect(self._on_edit_done)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        # GLOBAL_STYLE doesn't ship a QLineEdit rule, so QLineEdit
        # widgets render with Qt's default light palette and look
        # washed out on the dark theme. Apply the canonical input
        # style here so every NumericLineEdit gets the slate
        # background + white text + focus ring without each call site
        # having to remember.
        try:
            from source.gui.style_builders import input_style
            self.setStyleSheet(input_style())
        except Exception:
            pass

    # ---- validator ----

    def _install_validator(self) -> None:
        if self._auto_value is not None:
            # A numeric validator would reject the word outright, keystroke by
            # keystroke. Accept either the word or a number and let ``value``
            # do the clamping it already does.
            number = r"-?\d*\.?\d*" if self._decimals > 0 else r"-?\d*"
            word = "".join(f"[{c.lower()}{c.upper()}]" for c in self._auto_text)
            v = QtGui.QRegularExpressionValidator(
                QtCore.QRegularExpression(f"^(?:{word}|{number})$"), self)
        elif self._decimals > 0:
            v = QtGui.QDoubleValidator(self._min, self._max, self._decimals, self)
            v.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
        else:
            v = QtGui.QIntValidator(int(self._min), int(self._max), self)
        self.setValidator(v)

    # ---- value access ----

    def value(self):
        text = self.text().strip()
        if self._auto_value is not None and text.lower() == self._auto_text.lower():
            return self._auto_value
        try:
            v = float(text) if text else self._min
        except (TypeError, ValueError):
            v = self._min
        v = max(self._min, min(self._max, v))
        return v if self._decimals > 0 else int(v)

    def setValue(self, v) -> None:
        if self._auto_value is not None:
            try:
                is_auto = float(v) == float(self._auto_value)
            except (TypeError, ValueError):
                is_auto = False
            if is_auto:
                self.setText(self._auto_text)
                return
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = self._min
        x = max(self._min, min(self._max, x))
        if self._decimals > 0:
            self.setText(f"{x:.{self._decimals}f}")
        else:
            self.setText(str(int(x)))

    # ---- QSpinBox-compatible setters ----

    def setRange(self, minimum, maximum) -> None:
        self._min = float(minimum)
        self._max = float(maximum)
        self._install_validator()
        self.setValue(self.value())

    def setMinimum(self, minimum) -> None:
        self.setRange(minimum, self._max)

    def setMaximum(self, maximum) -> None:
        self.setRange(self._min, maximum)

    def minimum(self):
        return self._min if self._decimals > 0 else int(self._min)

    def maximum(self):
        return self._max if self._decimals > 0 else int(self._max)

    def setDecimals(self, decimals: int) -> None:
        cur = self.value()
        self._decimals = int(max(0, decimals))
        self._install_validator()
        self.setValue(cur)

    def setSingleStep(self, step) -> None:
        self._step = float(step)

    def setSuffix(self, _suffix: str) -> None:
        # SpinBox suffix has no analog in QLineEdit; ignore so callers
        # don't have to special-case.
        return

    # ---- internal ----

    def _on_edit_done(self) -> None:
        # Normalise what is shown: a half-typed "aut" or "007" becomes the
        # canonical rendering of whatever it parsed to.
        self.setValue(self.value())
        self.valueChanged.emit(float(self.value()))


__all__ = ["NumericLineEdit"]
