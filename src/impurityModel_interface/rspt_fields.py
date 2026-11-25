"""RSPt's cluster object, seen from Python as a mapping from Fortran names to numpy views.

RSPt cannot hand over its ``clustertype`` itself (allocatable components, compiler-private
layout), so ``green_impmod_interface.F90`` builds a flat table of ``impmod_field`` records --
name, pointer, kind, rank, item size, shape; see ``run_impurityModel.h`` -- and passes that.
:class:`RsptFields` turns the table into ``fields["ham"]``, ``fields["observables%ms"]``,
``fields["symm(3)%rep"]``, ``fields["ctx%u4"]`` ...  The values are views of RSPt's memory, not
copies, and are valid only during the call that delivered the table.

What the solver makes of the fields -- ``h_dft``, the rotation, the operators -- is
:meth:`cluster.ClusterCall.from_fields`, which takes any ``Mapping`` so a test can feed it a plain dict.
"""

from collections.abc import Mapping

import numpy as np

#: Layout version of ``impmod_field``; RSPt passes its own value and any other is refused.
#: 3 replaced the positional argument list by the field table.
ABI_VERSION = 3

#: ``sizeof(impmod_field)``; checked against the compiled struct and RSPt's ``c_sizeof``.
FIELD_SIZE = 128

KIND_COMPLEX, KIND_REAL, KIND_INTEGER, KIND_LOGICAL, KIND_CHARACTER = 1, 2, 3, 4, 5

#: Fields RSPt reads back (or that the solve writes through on non-root ranks); every other view
#: is read-only so a stray in-place operation cannot corrupt RSPt's cluster.
WRITABLE = frozenset({"sig", "sig_real", "sig_static", "sig_dc", "ctx%u4", "ctx%hyb"})

_DTYPES = {
    (KIND_COMPLEX, 16): np.dtype(np.complex128),
    (KIND_COMPLEX, 8): np.dtype(np.complex64),
    (KIND_REAL, 8): np.dtype(np.float64),
    (KIND_REAL, 4): np.dtype(np.float32),
    (KIND_INTEGER, 8): np.dtype(np.int64),
    (KIND_INTEGER, 4): np.dtype(np.int32),
    (KIND_INTEGER, 2): np.dtype(np.int16),
    (KIND_INTEGER, 1): np.dtype(np.int8),
    # Fortran logicals are integers whose truth is "non-zero" (gfortran: 1, ifx: -1).
    (KIND_LOGICAL, 8): np.dtype(np.int64),
    (KIND_LOGICAL, 4): np.dtype(np.int32),
    (KIND_LOGICAL, 2): np.dtype(np.int16),
    (KIND_LOGICAL, 1): np.dtype(np.int8),
}


class RsptFields(Mapping):
    """``name -> numpy view`` over the ``impmod_field`` array RSPt passed.

    ``fields`` is a cffi ``impmod_field *`` and ``ffi`` the handle that made it. Scalars come
    back as Python ``complex``/``float``/``int``/``bool``/``str`` (strings stripped of Fortran's
    blank padding); arrays as Fortran-ordered views. A field RSPt did not allocate is simply
    absent from the mapping.
    """

    def __init__(self, ffi, fields, n_fields):
        self._ffi = ffi
        self._records = {}
        for i in range(n_fields):
            record = fields[i]
            if record.data == ffi.NULL:
                continue
            name = ffi.string(record.name).decode("ascii")
            self._records[name] = record

    def __len__(self):
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def __contains__(self, name):
        return name in self._records

    def __getitem__(self, name):
        try:
            record = self._records[name]
        except KeyError:
            raise KeyError(
                f"RSPt did not pass the field {name!r}. Either the cluster does not have it "
                "(it is not allocated) or RSPt is older than this interface: rebuild RSPt "
                "(green_impmod_interface.F90) and this library together."
            ) from None
        return self._value(name, record)

    def _value(self, name, record):
        kind, rank, itemsize = int(record.kind), int(record.rank), int(record.itemsize)
        shape = tuple(int(record.shape[i]) for i in range(rank))
        count = int(np.prod(shape)) if rank else 1
        if kind == KIND_CHARACTER:
            raw = bytes(self._ffi.buffer(record.data, count * itemsize))
            if rank == 0:
                return raw.decode("ascii").rstrip("\0 ")
            return np.ndarray(shape=shape, order="F", dtype=f"S{itemsize}", buffer=raw)
        try:
            dtype = _DTYPES[(kind, itemsize)]
        except KeyError:
            raise ValueError(f"RSPt field {name!r} has kind {kind} with item size {itemsize}; not supported.") from None
        view = np.ndarray(buffer=self._ffi.buffer(record.data, count * itemsize), shape=shape, order="F", dtype=dtype)
        if rank == 0:
            value = view[()].item()
            return bool(value) if kind == KIND_LOGICAL else value
        if name not in WRITABLE:
            view.flags.writeable = False
        return view != 0 if kind == KIND_LOGICAL else view
