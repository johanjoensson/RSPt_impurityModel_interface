"""RSPt's observable matrices (``ms``, ``msp``, ``ml``, ``mlp``, ``msaxis``, ``Dxyz``) through ``solve``.

They are marshalled by :func:`solve.build_operators` and handed to the model rotated into the fitting basis.
No ED runs: ``from_blocks`` is a spy and ``calc_selfenergy`` is replaced.
"""

import numpy as np
import pytest

pytest.importorskip("mpi4py")
from test_symm_prec_solve_wiring import _hermitian, _install_spies, _problem, _solve, restore_stdout  # noqa: E402,F401

from impurityModel_interface import solve  # noqa: E402


def test_build_operators_maps_names_and_dipole_layout():
    n = 4
    rng = np.random.default_rng(0)
    ms, msp = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    dxyz = rng.normal(size=(n, n, 3))  # Fortran layout (n_orb, n_orb, 3)
    ops = solve.build_operators({"ms": ms, "msp": msp, "ml": None, "dxyz": dxyz}, n)
    np.testing.assert_allclose(ops.sz, ms)
    np.testing.assert_allclose(ops.s_plus, msp)
    assert ops.lz is None and ops.source == "rspt" and ops.frame == "global"
    np.testing.assert_allclose(ops.dipole[1], dxyz[:, :, 1])
    assert solve.build_operators(None, n) is None
    assert solve.build_operators({"ms": None}, n) is None
    with pytest.raises(ValueError):
        solve.build_operators({"bogus": ms}, n)


def test_the_model_gets_the_operators_in_the_fitting_basis(shared_tmp_path, monkeypatch):
    captured = []
    real = solve.ImpurityModel.from_blocks.__func__

    def from_blocks(cls, H_imp, V, H_bath, **kwargs):
        captured.append(kwargs.get("operators"))
        return real(cls, H_imp, V, H_bath, **kwargs)

    _install_spies(monkeypatch, sigma_static_cf=_hermitian(1))
    monkeypatch.setattr(solve.ImpurityModel, "from_blocks", classmethod(from_blocks))
    ms = np.asfortranarray(_hermitian(5))
    import impurityModel_interface.solve as s

    original = s.solve

    problem = _problem(0.0)
    arrays_holder = {}

    def spy_prepare(**kw):
        arrays_holder["operators"] = kw["operators"]
        return real_prepare(**kw)

    real_prepare = s.prepare_problem
    monkeypatch.setattr(s, "prepare_problem", spy_prepare)
    # route operator_arrays through the test helper's solve() call
    monkeypatch.setattr(s, "solve", lambda **kw: original(operator_arrays={"ms": ms}, **kw))
    _solve(shared_tmp_path, monkeypatch, "3 3", problem)
    assert captured and captured[0] is not None
    op_in = arrays_holder["operators"]
    np.testing.assert_allclose(op_in.sz, ms)
    # the rotation into the fit basis is a unitary change of basis: the spectrum is preserved
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(captured[0].sz)), np.sort(np.linalg.eigvalsh(ms)), atol=1e-10
    )
