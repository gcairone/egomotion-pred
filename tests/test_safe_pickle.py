import os
import pickle
from collections import OrderedDict

import numpy as np
import pytest

from egopred import safe_pickle


def test_loads_plain_containers_and_arrays():
    obj = [{"a": np.arange(4.0).reshape(2, 2), "b": np.float64(1.5), "c": np.int64(3),
            "d": OrderedDict(x=[1, 2]), "e": None, "f": "s", "g": np.array(["x", "yy"])}]
    out = safe_pickle.loads(pickle.dumps(obj, protocol=4))
    np.testing.assert_array_equal(out[0]["a"], obj[0]["a"])
    assert out[0]["b"] == 1.5 and out[0]["d"]["x"] == [1, 2] and out[0]["g"][1] == "yy"


class Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_blocks_arbitrary_callables():
    with pytest.raises(pickle.UnpicklingError, match="blocked"):
        safe_pickle.loads(pickle.dumps(Evil()))
