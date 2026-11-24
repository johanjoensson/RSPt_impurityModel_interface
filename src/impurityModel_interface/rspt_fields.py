"""RSPt's cluster object, seen from Python as a mapping from Fortran names to numpy views.

RSPt cannot hand over its ``clustertype`` itself (allocatable components, compiler-private
layout), so ``green_impmod_interface.F90`` builds a flat table of ``impmod_field`` records --
name, pointer, kind, rank, item size, shape; see ``run_impurityModel.h`` -- and passes that.
:class:`RsptFields` turns the table into ``fields["ham"]``, ``fields["observables%ms"]``,
``fields["symm(3)%rep"]``, ``fields["ctx%u4"]`` ...  The values are views of RSPt's memory, not
copies, and are valid only during the call that delivered the table.

Everything that used to be computed on the Fortran side -- the chemical potential shift of
``h_dft``, the harmonic-to-correlated rotation, the dipole from its observables, the stacked
symmetry representations -- is :func:`solve_arguments`, which takes any ``Mapping`` so a test
can feed it a plain dict.
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


def _n_orb(fields):
    return int(fields["ham"].shape[0])


def _square(fields, name, n_orb):
    """``fields[name]`` cut to the first ``n_orb`` rows and columns, as the Fortran side did."""
    value = np.asarray(fields[name])
    if value.shape[0] < n_orb or value.shape[1] < n_orb:
        raise ValueError(f"RSPt field {name!r} has shape {value.shape}, smaller than the {n_orb}x{n_orb} cluster.")
    return value[:n_orb, :n_orb]


def _operator(fields, name, n_orb):
    """A one-body operator in RSPt's correlated basis, or ``None`` if absent (not allocated, or all zero)."""
    if name not in fields:
        return None
    value = np.array(_square(fields, name, n_orb), order="F")
    return value if np.any(value) else None


def _dipole(fields, n_orb):
    """Cartesian dipole ``(n_orb, n_orb, 3)`` from RSPt's observables, or ``None``.

    The observables line lists Dxyz and/or Dpm. Dxyz is used as is; Dpm (Dp, Dm, Dz) is converted
    back: ``Dx = (Dp + Dm)/sqrt(2)``, ``Dy = (Dp - Dm)/(i sqrt(2))``.
    """
    names = []
    k = 1
    while f"observables%obs({k})%name" in fields:
        names.append(fields[f"observables%obs({k})%name"].strip())
        k += 1

    def block(first):
        keys = [f"observables%obs({first + i})%t" for i in range(3)]
        if first + 2 > len(names) or not all(key in fields for key in keys):
            return None
        return [np.array(_square(fields, key, n_orb), order="F") for key in keys]

    for k, name in enumerate(names, start=1):
        if name == "(Dipole) Dx" and (parts := block(k)) is not None:
            return np.stack(parts, axis=2)
    for k, name in enumerate(names, start=1):
        if name == "(Dipole) Dp" and (parts := block(k)) is not None:
            dp, dm, dz = parts
            half = np.sqrt(0.5)
            return np.stack([half * (dp + dm), -1j * half * (dp - dm), dz], axis=2)
    return None


def _symmetries(fields, n_orb):
    """``(symm_rep, symm_timerev)`` stacked over the cluster's point-group operations, or ``(None, None)``.

    As before, a cluster where any operation has no representation counts as having none.
    """
    n_symm = int(fields.get("symm%n", 0))
    keys = [f"symm({i})%rep" for i in range(1, n_symm + 1)]
    if n_symm == 0 or not all(key in fields for key in keys):
        return None, None
    rep = np.stack([_square(fields, key, n_orb) for key in keys], axis=2)
    timerev = np.array([int(bool(fields.get(f"symm({i})%timerev", False))) for i in range(1, n_symm + 1)], dtype=np.intc)
    return np.asfortranarray(rep), timerev


def solve_arguments(fields):
    """The keyword arguments of :func:`solve.solve`, derived from RSPt's cluster fields.

    ``h_dft`` is a copy (``ham - mu``): RSPt's ``ham`` is never written. Every other array is
    the view RSPt owns, so ``sig``, ``sig_real``, ``sig_static`` and ``sig_dc`` are written in
    place and ``u4`` and ``hyb`` can be broadcast through. The output shapes are checked rather
    than reshaped: a mismatch would write past RSPt's allocation.
    """
    ham = fields["ham"]
    n_orb = _n_orb(fields)
    if ham.shape != (n_orb, n_orb):
        raise ValueError(f"RSPt's ham has shape {ham.shape}; expected a square matrix.")
    iw = fields["ctx%wmsl"]
    w = fields["ctx%emesh"]
    n_iw, n_w = len(iw), len(w)

    expected = {
        "sig": (n_orb, n_orb, n_iw),
        "sig_real": (n_orb, n_orb, n_w),
        "sig_static": (n_orb, n_orb),
        "sig_dc": (n_orb, n_orb),
        "ctx%u4": (n_orb,) * 4,
        "ctx%hyb": (n_orb, n_orb, n_w),
    }
    for name, shape in expected.items():
        if fields[name].shape != shape:
            raise ValueError(f"RSPt field {name!r} has shape {fields[name].shape}, expected {shape}.")

    h_dft = np.array(ham, order="F")
    h_dft[np.diag_indices(n_orb)] -= fields["ctx%green_mu"]

    harm2corr = fields["harm2corr"]
    operator_arrays = {
        name: _operator(fields, f"observables%{name}", n_orb) for name in ("ms", "msp", "ml", "mlp", "msaxis")
    }
    dxyz = _dipole(fields, n_orb)
    if dxyz is not None:
        operator_arrays["dxyz"] = dxyz
    symm_rep, symm_timerev = _symmetries(fields, n_orb)
    if symm_rep is not None:
        operator_arrays["symm_rep"] = symm_rep
        operator_arrays["symm_timerev"] = symm_timerev

    return dict(
        label=fields["label"],
        solver_line=fields["solver_paramstr"],
        dc_line=fields["dc_paramstr"],
        dc_flag=int(fields["ctx%dc_flag"]),
        h_dft=h_dft,
        u4=fields["ctx%u4"],
        hyb=fields["ctx%hyb"],
        iw=iw,
        w=w,
        sig=fields["sig"],
        sig_real=fields["sig_real"],
        sig_static=fields["sig_static"],
        sig_dc=fields["sig_dc"],
        # RSPt's harm2corr is (n_orb_full, n_orb); the solver wants corr -> harmonic.
        corr_to_spherical_in=np.asfortranarray(harm2corr.conj().T),
        n_orb=n_orb,
        n_rot_cols=int(fields["corr2cf"].shape[1]),
        n_orb_full=int(harm2corr.shape[0]),
        eim=fields["ctx%eim"],
        tau=fields["ctx%ttt"],
        verbosity=int(fields["ctx%verbosity"]),
        operator_arrays=operator_arrays,
    )
