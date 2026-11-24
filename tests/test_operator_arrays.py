"""RSPt's observable matrices (``ms``, ``msp``, ``ml``, ``mlp``, ``msaxis``, ``Dxyz``) through ``solve``.

They are marshalled by :func:`solve.build_operators` and handed to the model rotated into the fitting basis.
No ED runs: ``from_blocks`` is a spy and ``calc_selfenergy`` is replaced.
"""

import numpy as np
import pytest

pytest.importorskip("mpi4py")
from test_symm_prec_solve_wiring import (  # noqa: E402,F401
    _hermitian,
    _install_spies,
    _problem,
    _solve,
    _unitary,
    restore_stdout,
)  
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
    """The model's operators are ``R^dagger O R`` with ``R`` the fit rotation -- not ``R O R^dagger``.

    The rotation is read back from the model's own ``rot_to_spherical`` (``corr_to_spherical`` is the
    identity in the helper), and is made non-trivial so that the two directions differ.
    """
    captured = []
    real = solve.ImpurityModel.from_blocks.__func__

    def from_blocks(cls, H_imp, V, H_bath, **kwargs):
        captured.append((kwargs.get("operators"), kwargs["rot_to_spherical"]))
        return real(cls, H_imp, V, H_bath, **kwargs)

    _install_spies(monkeypatch, sigma_static_cf=_hermitian(1))
    monkeypatch.setattr(solve.ImpurityModel, "from_blocks", classmethod(from_blocks))
    problem = _problem(0.0, unitary=_unitary(3))
    ms = np.asfortranarray(problem[0].copy())
    original = solve.solve
    monkeypatch.setattr(solve, "solve", lambda **kw: original(operator_arrays={"ms": ms}, **kw))
    _solve(shared_tmp_path, monkeypatch, "3 3", problem)

    ops, rot_to_spherical = captured[0]
    R = rot_to_spherical.conj().T
    assert np.abs(R - np.eye(len(R))).max() > 0.1, "the test needs a non-trivial fit rotation"
    np.testing.assert_allclose(ops.sz, R.conj().T @ ms @ R, atol=1e-10)
    assert np.abs(ops.sz - R @ ms @ R.conj().T).max() > 0.1


def test_operator_arrays_must_have_the_stated_shape():
    with pytest.raises(ValueError, match="shape"):
        solve.build_operators({"ms": np.zeros(9)}, 3)
    with pytest.raises(ValueError, match="shape"):
        solve.build_operators({"dxyz": np.zeros((3, 3, 2))}, 3)
