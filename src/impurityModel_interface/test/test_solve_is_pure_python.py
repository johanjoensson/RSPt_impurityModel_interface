"""``_solve`` runs on numpy arrays, with no cffi handle anywhere in sight.

Before the split, everything below the argument marshalling lived in one function whose first
act was ``ffi.string(rspt_label, 18)``. Reaching any of it from a test meant either an
embedded interpreter or a fake cffi handle, so none of it was reached. ``_run_impmod_ed`` now
owns the pointers and ``_solve`` owns the physics, which makes the physics callable.

What this pins is the seam itself: ``_solve`` accepts ``str`` and ``numpy`` and gets far
enough in to enforce its own preconditions.
"""

import sys

import numpy as np
import pytest

pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: F401

from impurityModel_interface import lib

N_ORB = 4
N_IW = 6
N_W = 8


@pytest.fixture
def restore_stdout():
    """``_solve`` redirects sys.stdout and only restores it on the paths it completes.

    In production ``run_impmod_ed`` is the net for the rest (see
    ``test_entry_point_returns_failure``). A test calling ``_solve`` directly is its own net.
    """
    saved = sys.stdout
    yield
    if sys.stdout is not saved:
        sys.stdout.close()
        sys.stdout = saved


def _arrays():
    """The shapes RSPt sends, in the Fortran order it sends them in."""
    return dict(
        h_dft=np.zeros((N_ORB, N_ORB), order="F", dtype=complex),
        u4=np.zeros((N_ORB,) * 4, order="F", dtype=complex),
        hyb=np.zeros((N_ORB, N_ORB, N_W), order="F", dtype=complex),
        iw=np.linspace(0.1, 1.0, N_IW),
        w=np.linspace(-5.0, 5.0, N_W),
        sig=np.zeros((N_ORB, N_ORB, N_IW), order="F", dtype=complex),
        sig_real=np.zeros((N_ORB, N_ORB, N_W), order="F", dtype=complex),
        sig_static=np.zeros((N_ORB, N_ORB), order="F", dtype=complex),
        sig_dc=np.zeros((N_ORB, N_ORB), order="F", dtype=complex),
        corr_to_spherical_in=np.eye(N_ORB, dtype=complex, order="F"),
        corr_to_cf_in=np.eye(N_ORB, dtype=complex, order="F"),
    )


def _call(tmp_path, monkeypatch, solver_line):
    monkeypatch.chdir(tmp_path)
    return lib._solve(
        label="Ni-3d",
        solver_line=solver_line,
        dc_line="",
        dc_flag=0,
        n_orb=N_ORB,
        n_rot_cols=N_ORB,
        n_orb_full=N_ORB,
        eim=0.1,
        tau=0.01,
        verbosity=0,
        **_arrays(),
    )


def test_an_impossible_nominal_occupation_is_rejected(tmp_path, monkeypatch, restore_stdout):
    """Eight electrons will not fit in four spin-orbitals, and _solve says so itself."""
    with pytest.raises(RuntimeError, match="out of bounds"):
        _call(tmp_path, monkeypatch, "8 10 2")


def test_a_negative_nominal_occupation_is_rejected(tmp_path, monkeypatch, restore_stdout):
    with pytest.raises(RuntimeError, match="out of bounds"):
        _call(tmp_path, monkeypatch, "-1 10 2")


def test_it_got_there_without_a_cffi_handle(tmp_path, monkeypatch, restore_stdout):
    """The point of the split: the stub would have raised long before the bounds check.

    ``_FFIStub.string`` raises RuntimeError("ffi.string is only available when embedded in
    RSPt"), so if any of this path still went through cffi, that is the message we would see
    instead of the occupation complaint.
    """
    with pytest.raises(RuntimeError) as excinfo:
        _call(tmp_path, monkeypatch, "8 10 2")
    assert "only available when embedded" not in str(excinfo.value)


def test_it_opens_the_per_cluster_output_file(tmp_path, monkeypatch, restore_stdout):
    """The redirect is part of _solve, so a direct call produces the same .out file."""
    with pytest.raises(RuntimeError):
        _call(tmp_path, monkeypatch, "8 10 2")
    assert (tmp_path / "impurityModel-Ni-3d.out").exists()
