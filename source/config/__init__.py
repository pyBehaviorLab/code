"""Project configuration, lineage and on-disk state.

``experiment.py`` owns the ``Config`` dataclass and the GUI-dict conversion,
``hashing.py`` is the single djb2 implementation, ``snapshot_store.py``
content-addresses captured sources, and ``multi_instance.py`` guards
concurrent GUI instances.

Deliberately empty of imports: several modules here are imported by both the
GUI and the MCU layer, and re-exporting them from the package would create
import cycles. Import the module you need directly::

    from source.config.experiment import Config
    from source.config.hashing import djb2_int_from_file

This file exists so setuptools' (non-namespace) package discovery ships the
package at all, without it, ``source.config`` is absent from any built wheel
even though it imports fine from a source checkout.
"""
