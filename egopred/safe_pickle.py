"""Unpickler restricted to the plain containers and numpy arrays found in OpenScene metadata.

The metadata ships as pickles downloaded from the internet; a regular unpickler would
execute whatever callable they reference. This one refuses anything outside an allow-list.
"""

import io
import pickle

import numpy as np

_NUMPY2 = int(np.__version__.split(".")[0]) >= 2

_ALLOWED = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy._core.numeric", "_frombuffer"),
    ("collections", "OrderedDict"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"blocked global in pickle: {module}.{name}")
        if _NUMPY2 and module.startswith("numpy.core"):
            module = "numpy._core" + module[len("numpy.core"):]
        return super().find_class(module, name)


def loads(data: bytes):
    return _RestrictedUnpickler(io.BytesIO(data)).load()
