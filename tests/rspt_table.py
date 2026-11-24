"""Build, with a real ``cffi.FFI``, the ``impmod_field`` table RSPt would pass.

The struct is read from ``src/run_impurityModel.h`` so the layout under test is the layout shipped.
"""

import re
from pathlib import Path

import numpy as np

HEADER = Path(__file__).resolve().parents[1] / "src" / "run_impurityModel.h"

_KIND = {"c": 1, "f": 2, "i": 3, "S": 5}


def make_ffi():
    import cffi

    text = HEADER.read_text()
    typedef = re.search(r"typedef struct \{.*?\} impmod_field;", text, re.S).group(0)
    ffi = cffi.FFI()
    ffi.cdef(typedef)
    return ffi


def build_table(ffi, arrays, logical=()):
    """``{name: array-or-scalar-or-str}`` -> ``(fields, n_fields, keepalive)``.

    Arrays are copied into Fortran order allocations owned by ``keepalive``; ``None`` is a field RSPt
    did not allocate (NULL data). Names in ``logical`` are stored as 4-byte logicals.
    ``keepalive["buffers"][name]`` is the raw allocation, so a test can read RSPt's memory back.
    """
    fields = ffi.new("impmod_field[]", max(len(arrays), 1))
    buffers = {}
    for i, (name, value) in enumerate(arrays.items()):
        record = fields[i]
        record.name = name.encode("ascii")
        if value is None:
            record.data = ffi.NULL
            continue
        if isinstance(value, str):
            raw = ffi.new("char[]", value.encode("ascii"))
            record.data, record.kind, record.rank, record.itemsize = raw, 5, 0, len(value)
            buffers[name] = raw
            continue
        array = np.array(value, order="F")
        if name in logical:
            array = array.astype(np.int32)
            kind = 4
        else:
            kind = _KIND[array.dtype.kind]
        raw = ffi.new("char[]", max(array.nbytes, 1))
        ffi.buffer(raw)[: array.nbytes] = array.tobytes(order="F")
        buffers[name] = raw
        record.data, record.kind, record.rank, record.itemsize = raw, kind, array.ndim, array.dtype.itemsize
        for axis, extent in enumerate(array.shape):
            record.shape[axis] = extent
    return fields, len(arrays), {"fields": fields, "buffers": buffers}


def raw(ffi, keepalive, name, dtype=complex):
    """RSPt's allocation behind ``name`` read back as flat bytes, bypassing every view."""
    buffer = keepalive["buffers"][name]
    return np.frombuffer(bytes(ffi.buffer(buffer)), dtype=dtype)
