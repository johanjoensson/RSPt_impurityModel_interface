"""The numpy views handed to ``solve`` are RSPt's own memory, not copies of it.

This is the one property of the boundary that no amount of Python-level testing will notice
if it breaks. RSPt passes pointers and reads its results back out of the same allocations, so
``sig``, ``sig_real``, ``sig_static`` and ``sig_dc`` get their values from in-place assignment
inside ``solve`` -- there is no copy-out step. A stray ``np.asarray(..., dtype=complex)``
would produce a perfectly working function that silently returns nothing to RSPt.

``u4`` and ``hyb`` alias too, for a different reason: ``solve`` calls ``comm.Bcast`` on them,
which on every non-root rank writes *through* the view into RSPt's buffer. Copy those and a
multi-rank run diverges while a single-rank one stays green. ``ham`` must NOT alias: ``h_dft``
is ``ham - mu`` and RSPt keeps its Hamiltonian.

These tests drive ``_run_impmod_ed`` with a real ``cffi.FFI`` over real allocations, so they
pin the aliasing without needing an embedded interpreter.
"""

import numpy as np
import pytest

pytest.importorskip("cffi")
pytest.importorskip("mpi4py")
from rspt_table import build_table, make_ffi, raw  # noqa: E402

from impurityModel_interface import entry  # noqa: E402

N_ORB, N_IW, N_W = 4, 6, 8
OUTPUTS = ("sig", "sig_real", "sig_static", "sig_dc")
INPUTS = {"ctx%u4": "u4", "ctx%hyb": "hyb"}


def cluster_arrays(n_orb=N_ORB, n_iw=N_IW, n_w=N_W, **extra):
    """The fields a minimal RSPt cluster carries, as plain arrays."""
    rng = np.random.default_rng(1)
    arrays = {
        "label": "Ni-3d" + " " * 13,
        "solver_paramstr": "8 10" + " " * 116,
        "dc_paramstr": "gap 0.5" + " " * 93,
        "ham": rng.normal(size=(n_orb, n_orb)) + 1j * rng.normal(size=(n_orb, n_orb)),
        "harm2corr": np.eye(n_orb, dtype=complex),
        "corr2cf": np.eye(n_orb, dtype=complex),
        "sig": np.zeros((n_orb, n_orb, n_iw), complex),
        "sig_real": np.zeros((n_orb, n_orb, n_w), complex),
        "sig_static": np.zeros((n_orb, n_orb), complex),
        "sig_dc": np.zeros((n_orb, n_orb), complex),
        "ctx%u4": np.zeros((n_orb,) * 4, complex),
        "ctx%hyb": np.zeros((n_orb, n_orb, n_w), complex),
        "ctx%wmsl": np.linspace(0.1, 2.0, n_iw),
        "ctx%emesh": np.linspace(-1.0, 1.0, n_w),
        "ctx%green_mu": np.array(0.25),
        "ctx%eim": np.array(0.1),
        "ctx%ttt": np.array(0.01),
        "ctx%verbosity": np.array(0, dtype=np.int32),
        "ctx%dc_flag": np.array(0, dtype=np.int32),
    }
    arrays.update(extra)
    return arrays


@pytest.fixture
def rspt_memory(monkeypatch):
    ffi = make_ffi()
    monkeypatch.setattr(entry, "ffi", ffi)
    return ffi


def _invoke(ffi, monkeypatch, recorder, arrays=None, abi_version=entry.ABI_VERSION):
    monkeypatch.setattr(entry, "solve", recorder)
    arrays = cluster_arrays() if arrays is None else arrays
    fields, n, keep = build_table(ffi, arrays, logical=())
    status = entry._run_impmod_ed(fields, n, abi_version)
    return status, keep


def test_writes_into_the_outputs_land_in_rspt_memory(rspt_memory, monkeypatch):
    """The only way results reach RSPt. Copy any of these and the solve returns nothing."""

    def recorder(**kw):
        for i, name in enumerate(OUTPUTS):
            kw[name][...] = complex(i + 1, -(i + 1))
        return 0

    status, keep = _invoke(rspt_memory, monkeypatch, recorder)
    assert status == 0
    for i, name in enumerate(OUTPUTS):
        assert np.all(raw(rspt_memory, keep, name) == complex(i + 1, -(i + 1))), name


def test_writes_into_the_inputs_land_in_rspt_memory(rspt_memory, monkeypatch):
    """comm.Bcast writes through these on every non-root rank."""

    def recorder(**kw):
        for i, name in enumerate(INPUTS.values()):
            kw[name][...] = complex(10 + i, 0)
        return 0

    status, keep = _invoke(rspt_memory, monkeypatch, recorder)
    assert status == 0
    for i, name in enumerate(INPUTS):
        assert np.all(raw(rspt_memory, keep, name) == complex(10 + i, 0)), name


def test_h_dft_is_a_copy_and_rspts_hamiltonian_is_untouched(rspt_memory, monkeypatch):
    seen = {}

    def recorder(**kw):
        seen.update(kw)
        kw["h_dft"][...] = 99  # what Bcast and the fit may do
        return 0

    arrays = cluster_arrays()
    _, keep = _invoke(rspt_memory, monkeypatch, recorder, arrays)
    assert seen["h_dft"].flags.owndata
    np.testing.assert_array_equal(raw(rspt_memory, keep, "ham").reshape(N_ORB, N_ORB, order="F"), arrays["ham"])


def test_the_views_are_fortran_ordered_and_own_nothing(rspt_memory, monkeypatch):
    seen = {}

    def recorder(**kw):
        seen.update(kw)
        return 0

    _invoke(rspt_memory, monkeypatch, recorder)
    for name in (*OUTPUTS, "u4", "hyb"):
        arr = seen[name]
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
    assert seen["iw"].shape == (N_IW,)
    assert seen["w"].shape == (N_W,)
    assert (seen["n_orb"], seen["n_rot_cols"], seen["n_orb_full"]) == (N_ORB, N_ORB, N_ORB)


def test_the_fixed_width_fortran_strings_arrive_as_stripped_str(rspt_memory, monkeypatch):
    seen = {}

    def recorder(**kw):
        seen.update(kw)
        return 0

    _invoke(rspt_memory, monkeypatch, recorder)
    assert seen["label"] == "Ni-3d"
    assert seen["solver_line"] == "8 10"
    assert seen["dc_line"] == "gap 0.5"


def test_a_wrong_abi_version_is_refused_before_anything_is_read(rspt_memory, monkeypatch):
    def recorder(**kw):
        raise AssertionError("solve must not be reached")

    with pytest.raises(RuntimeError, match="ABI version"):
        _invoke(rspt_memory, monkeypatch, recorder, abi_version=entry.ABI_VERSION + 1)


def test_the_compiled_record_is_the_size_rspt_builds(rspt_memory):
    assert rspt_memory.sizeof("impmod_field") == entry.FIELD_SIZE == 128
