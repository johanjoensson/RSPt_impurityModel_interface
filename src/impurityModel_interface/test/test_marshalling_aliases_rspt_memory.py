"""The numpy views handed to ``_solve`` are RSPt's own memory, not copies of it.

This is the one property of the boundary that no amount of Python-level testing will notice
if it breaks. RSPt passes pointers and reads its results back out of the same allocations, so
``sig``, ``sig_real``, ``sig_static`` and ``sig_dc`` get their values from in-place assignment
inside ``_solve`` -- there is no copy-out step. A stray ``np.asarray(..., dtype=complex)``
would produce a perfectly working function that silently returns nothing to RSPt.

The inputs alias too, and for a different reason: ``_solve`` calls ``comm.Bcast`` on
``h_dft``, ``u4`` and ``hyb``, which on every non-root rank writes *through* the view into
RSPt's buffer. Copy those and a multi-rank run diverges while a single-rank one stays green.

These tests drive ``_run_impmod_ed`` with a real ``cffi.FFI`` over real allocations, so they
pin the aliasing without needing an embedded interpreter.
"""

import numpy as np
import pytest

cffi = pytest.importorskip("cffi")
pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: F401,E402

from impurityModel_interface import lib  # noqa: E402

N_ORB, N_IW, N_W = 4, 6, 8
SIZE_REAL, SIZE_COMPLEX = 8, 16

# name -> element count, in the shape RSPt allocates it with.
COMPLEX_ARRAYS = {
    "u4": N_ORB**4,
    "hyb": N_ORB * N_ORB * N_W,
    "h_dft": N_ORB * N_ORB,
    "sig": N_ORB * N_ORB * N_IW,
    "sig_real": N_ORB * N_ORB * N_W,
    "sig_static": N_ORB * N_ORB,
    "sig_dc": N_ORB * N_ORB,
    "corr_to_spherical": N_ORB * N_ORB,
    "corr_to_cf": N_ORB * N_ORB,
}
REAL_ARRAYS = {"iw": N_IW, "w": N_W}

OUTPUTS = ("sig", "sig_real", "sig_static", "sig_dc")
INPUTS = ("h_dft", "u4", "hyb")


@pytest.fixture
def rspt_memory(monkeypatch):
    """Allocations standing in for RSPt's, reachable both as pointers and as raw bytes."""
    ffi = cffi.FFI()
    monkeypatch.setattr(lib, "ffi", ffi)

    mem = {"_ffi": ffi}
    for name, count in COMPLEX_ARRAYS.items():
        mem[name] = ffi.new("char[]", count * SIZE_COMPLEX)
    for name, count in REAL_ARRAYS.items():
        mem[name] = ffi.new("char[]", count * SIZE_REAL)
    # Fortran character fields: fixed width, space padded, no NUL terminator.
    mem["label"] = ffi.new("char[18]", b"Ni-3d" + b" " * 13)
    mem["solver"] = ffi.new("char[100]", b"8 10 2" + b" " * 94)
    mem["dc"] = ffi.new("char[100]", b"gap 0.5" + b" " * 93)
    return mem


def _raw(mem, name):
    """Read an allocation back as complex, bypassing the numpy views entirely."""
    count = COMPLEX_ARRAYS[name]
    return np.frombuffer(mem["_ffi"].buffer(mem[name], count * SIZE_COMPLEX), dtype=complex)


def _invoke(mem, monkeypatch, recorder):
    monkeypatch.setattr(lib, "_solve", recorder)
    return lib._run_impmod_ed(
        mem["label"],
        mem["solver"],
        mem["dc"],
        0,
        mem["u4"],
        mem["hyb"],
        mem["h_dft"],
        mem["sig"],
        mem["sig_real"],
        mem["sig_static"],
        mem["sig_dc"],
        mem["iw"],
        mem["w"],
        mem["corr_to_spherical"],
        mem["corr_to_cf"],
        N_ORB,
        N_ORB,
        N_ORB,
        N_IW,
        N_W,
        0.1,
        0.01,
        0,
        SIZE_REAL,
        SIZE_COMPLEX,
    )


def test_writes_into_the_outputs_land_in_rspt_memory(rspt_memory, monkeypatch):
    """The only way results reach RSPt. Copy any of these and the solve returns nothing."""

    def recorder(**kw):
        for i, name in enumerate(OUTPUTS):
            kw[name][...] = complex(i + 1, -(i + 1))
        return 0

    assert _invoke(rspt_memory, monkeypatch, recorder) == 0
    for i, name in enumerate(OUTPUTS):
        assert np.all(_raw(rspt_memory, name) == complex(i + 1, -(i + 1))), name


def test_writes_into_the_inputs_land_in_rspt_memory(rspt_memory, monkeypatch):
    """comm.Bcast writes through these on every non-root rank."""

    def recorder(**kw):
        for i, name in enumerate(INPUTS):
            kw[name][...] = complex(10 + i, 0)
        return 0

    assert _invoke(rspt_memory, monkeypatch, recorder) == 0
    for i, name in enumerate(INPUTS):
        assert np.all(_raw(rspt_memory, name) == complex(10 + i, 0)), name


def test_the_views_are_writable_fortran_ordered_and_own_nothing(rspt_memory, monkeypatch):
    seen = {}

    def recorder(**kw):
        seen.update(kw)
        return 0

    _invoke(rspt_memory, monkeypatch, recorder)
    for name in COMPLEX_ARRAYS:
        arr = seen[name if name not in ("corr_to_spherical", "corr_to_cf") else f"{name}_in"]
        assert arr.dtype == complex, name
        assert arr.flags.writeable, name
        assert not arr.flags.owndata, f"{name} is a copy, not a view of RSPt's memory"
        if arr.ndim > 1:
            assert arr.flags.f_contiguous, name


def test_the_shapes_match_what_rspt_allocated(rspt_memory, monkeypatch):
    seen = {}

    def recorder(**kw):
        seen.update(kw)
        return 0

    _invoke(rspt_memory, monkeypatch, recorder)
    assert seen["h_dft"].shape == (N_ORB, N_ORB)
    assert seen["u4"].shape == (N_ORB,) * 4
    assert seen["hyb"].shape == (N_ORB, N_ORB, N_W)
    assert seen["sig"].shape == (N_ORB, N_ORB, N_IW)
    assert seen["sig_real"].shape == (N_ORB, N_ORB, N_W)
    assert seen["sig_static"].shape == (N_ORB, N_ORB)
    assert seen["sig_dc"].shape == (N_ORB, N_ORB)
    assert seen["iw"].shape == (N_IW,)
    assert seen["w"].shape == (N_W,)


def test_the_fixed_width_fortran_strings_arrive_as_str(rspt_memory, monkeypatch):
    """Space padded and not NUL terminated; _solve must get str, not bytes."""
    seen = {}

    def recorder(**kw):
        seen.update(kw)
        return 0

    _invoke(rspt_memory, monkeypatch, recorder)
    assert seen["label"] == "Ni-3d" + " " * 13
    assert seen["label"].strip() == "Ni-3d"
    assert seen["solver_line"].strip() == "8 10 2"
    assert seen["dc_line"].strip() == "gap 0.5"


def test_the_return_value_is_forwarded(rspt_memory, monkeypatch):
    assert _invoke(rspt_memory, monkeypatch, lambda **kw: -3) == -3
