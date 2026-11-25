"""RSPt's observable matrices (``ms``, ``msp``, ``ml``, ``mlp``, ``msaxis``, ``Dxyz``) through ``solve``.

They are built by :meth:`ClusterCall.from_fields` (see ``test_cluster_call``) and handed to the model rotated
into the fitting basis.  No ED runs: ``from_blocks`` is a spy and ``calc_selfenergy`` is replaced.
"""

import numpy as np
import pytest

pytest.importorskip("mpi4py")
from fake_cluster import cluster  # noqa: E402
from mpi4py import MPI  # noqa: E402
from test_symm_prec_solve_wiring import (  # noqa: E402,F401
    _hermitian,
    _install_spies,
    _problem,
    _solve,
    _unitary,
    restore_stdout,
)

from impurityModel.api import LocalOperators  # noqa: E402
from impurityModel_interface import problem as problem_module  # noqa: E402
from impurityModel_interface.cluster import ClusterCall  # noqa: E402


def test_field_names_map_to_operator_names():
    n = 4
    rng = np.random.default_rng(0)
    ms, msp = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    f = cluster(extra={"observables%ms": ms, "observables%msp": msp, "observables%ml": None})
    del f["observables%ml"]
    ops = ClusterCall.from_fields(f).operators
    np.testing.assert_allclose(ops.sz, ms)
    np.testing.assert_allclose(ops.s_plus, msp)
    assert ops.lz is None and ops.source == "rspt" and ops.frame == "global"


def test_the_model_gets_the_operators_in_the_fitting_basis(shared_tmp_path, monkeypatch):
    """The model's operators are ``R^dagger O R`` with ``R`` the fit rotation -- not ``R O R^dagger``.

    The rotation is read back from the model's own ``rot_to_spherical`` (``corr_to_spherical`` is the
    identity in the helper), and is made non-trivial so that the two directions differ.
    """
    captured = []
    real = problem_module.ImpurityModel.from_blocks.__func__

    def from_blocks(cls, H_imp, V, H_bath, **kwargs):
        captured.append((kwargs.get("operators"), kwargs["rot_to_spherical"]))
        return real(cls, H_imp, V, H_bath, **kwargs)

    _install_spies(monkeypatch, sigma_static_cf=_hermitian(1))
    monkeypatch.setattr(problem_module.ImpurityModel, "from_blocks", classmethod(from_blocks))
    problem = _problem(0.0, unitary=_unitary(3))
    ms = np.asfortranarray(problem[0].copy())
    _solve(
        shared_tmp_path,
        monkeypatch,
        "3 3",
        problem,
        operators=LocalOperators(source="rspt", frame="global", sz=np.array(ms, dtype=complex)),
    )

    ops, rot_to_spherical = captured[0]
    R = rot_to_spherical.conj().T
    assert np.abs(R - np.eye(len(R))).max() > 0.1, "the test needs a non-trivial fit rotation"
    np.testing.assert_allclose(ops.sz, R.conj().T @ ms @ R, atol=1e-10)
    assert np.abs(ops.sz - R @ ms @ R.conj().T).max() > 0.1


def test_a_malformed_operator_fails_on_synchronise_not_silently():
    f = cluster(extra={"observables%ms": np.zeros(9)})
    call = ClusterCall.from_fields(f)
    assert call.operators is None and call.operator_error
    with pytest.raises(ValueError, match="observables%ms"):
        call.synchronise(MPI.COMM_WORLD)
