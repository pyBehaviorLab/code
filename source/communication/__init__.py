"""Host side of the pyControl MCU link.

Layering, lowest first: ``pyboard.py`` (raw-REPL transport) →
``pycboard.py`` (pyControl host wrapper: upload, variables, coordinates) →
``controller.py`` / ``data_logger.py`` (session orchestration and the per-run
TSV). ``api.py`` is the base class user task controllers in ``api_classes/``
subclass; it is loaded by name at runtime, not imported statically.

``PyboardError`` subclasses ``BaseException``, so ``except Exception`` will not
catch it, see ``errors.py``.

Deliberately empty of imports, so importing one layer never drags in the rest.
Import the module you need directly::

    from source.communication.pycboard import Pycboard

This file exists so setuptools' (non-namespace) package discovery ships the
package at all, without it, ``source.communication`` is absent from any built
wheel even though it imports fine from a source checkout.
"""
