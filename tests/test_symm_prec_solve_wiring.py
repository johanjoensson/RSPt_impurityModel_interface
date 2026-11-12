"""``symm_prec`` through ``solve``: the solver line value reaches the fit, and the solver and the
double-counting search are told whether the fit neglected anything.

``solve`` is driven on the synthetic 2 + 3 orbital problem of ``test_symm_prec_truncation`` with
``calc_selfenergy`` and ``fixed_occupation_dc`` replaced by spies, so no ED runs. The bath fit and
the archive are real.
"""

import sys
from pathlib import Path

import h5py as h5
import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

pytest.importorskip("mpi4py")
from mpi4py import MPI
from rspt2spectra.utils import rotate_matrix

from impurityModel_interface import double_counting, solve

N = 5
N_IW = 4


@pytest.fixture(autouse=True)
def restore_stdout():
    saved = sys.stdout
    yield
    if sys.stdout is not saved:
        try:
            sys.stdout.close()
        except Exception:
            pass
    sys.stdout = saved


@pytest.fixture
def tmp_path(tmp_path_factory):
    """One working directory for all ranks: only rank 0 writes the archive the tests read back."""
    path = MPI.COMM_WORLD.bcast(str(tmp_path_factory.mktemp("wiring")), root=0)
    return Path(path)


def _problem(coupling, unitary=None, weak=True, connect_last=False):
    w = np.linspace(-3, 3, 300)
    eim = 0.05
    eps = np.array([0.5, 0.5, -0.3, -0.2, -0.25])
    hop = np.array([0.4, 0.4, 0.3, 0.3, 0.3])
    hyb = np.zeros((len(w), N, N), dtype=complex)
    for i in range(N):
        hyb[:, i, i] = hop[i] ** 2 / (w + 1j * eim - eps[i])
    hyb[:, 0, 2] = hyb[:, 2, 0] = coupling * 0.5 / (w + 1j * eim)
    H = np.diag([0.6, 0.6, -0.1, -0.1, -0.1]).astype(complex)
    if weak:
        H[0, 1] = H[1, 0] = 0.02
        H[2, 3] = H[3, 2] = 0.01
    H[1, 3] = H[3, 1] = coupling
    if connect_last:
        H[3, 4] = H[4, 3] = coupling
    if unitary is not None:
        hyb = np.einsum("ji,wjk,kl->wil", unitary.conj(), hyb, unitary)
        H = unitary.conj().T @ H @ unitary
    return H, hyb, w, eim


def _unitary(seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
    return q


def _hermitian(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    return 0.3 * (a + a.conj().T)


def _solve(tmp_path, monkeypatch, solver_line, problem, dc_flag=0, dc_line="occ 3", sig_dc=None):
    monkeypatch.chdir(tmp_path)
    H, hyb, w, eim = problem
    n_w = len(w)
    sig_dc = np.zeros((N, N), order="F", dtype=complex) if sig_dc is None else sig_dc
    arrays = dict(
        h_dft=np.asfortranarray(H),
        u4=np.zeros((N,) * 4, order="F", dtype=complex),
        hyb=np.asfortranarray(np.moveaxis(hyb, 0, -1)),
        iw=np.linspace(0.1, 1.0, N_IW),
        w=w,
        sig=np.zeros((N, N, N_IW), order="F", dtype=complex),
        sig_real=np.zeros((N, N, n_w), order="F", dtype=complex),
        sig_static=np.zeros((N, N), order="F", dtype=complex),
        sig_dc=sig_dc,
        corr_to_spherical_in=np.eye(N, dtype=complex, order="F"),
        corr_to_cf_in=np.eye(N, dtype=complex, order="F"),
    )
    status = solve.solve(
        label="t",
        solver_line=solver_line,
        dc_line=dc_line,
        dc_flag=dc_flag,
        n_orb=N,
        n_rot_cols=N,
        n_orb_full=N,
        eim=eim,
        tau=0.01,
        verbosity=0,
        **arrays,
    )
    assert status == 0
    return arrays


def _n_components(matrix, tol=1e-12):
    return connected_components(np.abs(matrix) > tol, directed=False)[0]


def _install_spies(monkeypatch, dc_in_corr=None, sigma_static_cf=None):
    """Replace the two solver entry points; returns the list the calls are recorded in.

    ``("model", n_blocks_H_imp, n_blocks_dc)`` is recorded for every model built: the number of
    connected components of the impurity block and of the double counting handed to the solver.
    """
    calls = []
    real_from_blocks = solve.ImpurityModel.from_blocks.__func__

    def from_blocks(cls, H_imp, V, H_bath, *, dc=None, **kwargs):
        calls.append(("model", _n_components(H_imp), None if dc is None else _n_components(dc)))
        return real_from_blocks(cls, H_imp, V, H_bath, dc=dc, **kwargs)

    monkeypatch.setattr(solve.ImpurityModel, "from_blocks", classmethod(from_blocks))
    real_fit_h0 = solve.fit_h0

    def fit_h0(*args, **kwargs):
        options = args[5]
        calls.append(("fit_h0", options.symm_prec))
        return real_fit_h0(*args, **kwargs)

    def corr_to_cf_of(model):
        return model.rot_to_spherical.conj().T  # corr_to_spherical_in is the identity

    def fixed_occupation_dc(model, basis, solver, **kwargs):
        calls.append(("dc", basis.interaction_merges_blocks))
        return rotate_matrix(dc_in_corr.pop(0), corr_to_cf_of(model)), 3

    def calc_selfenergy(model, meshes, basis, solver, **kwargs):
        calls.append(("sigma", basis.interaction_merges_blocks))
        n = sigma_static_cf.shape[0]
        return {
            "sigma_static": sigma_static_cf,
            "sigma": np.broadcast_to(sigma_static_cf, (N_IW, n, n)).copy(),
            "sigma_real": np.broadcast_to(sigma_static_cf, (len(meshes.w), n, n)).copy(),
            "thermal_rho": np.zeros((n, n)),
            "rhos": np.zeros((1, n, n)),
            "gs_matsubara": None,
            "gs_realaxis": None,
        }

    monkeypatch.setattr(solve, "fit_h0", fit_h0)
    if dc_in_corr is not None:
        monkeypatch.setattr(double_counting, "fixed_occupation_dc", fixed_occupation_dc)
    if sigma_static_cf is not None:
        monkeypatch.setattr(solve, "calc_selfenergy", calc_selfenergy)
    return calls


@pytest.mark.parametrize("key", ["symm_prec", "symmetry_prec"])
def test_value_reaches_the_fit_and_the_solver_is_told_blocks_are_authoritative(tmp_path, monkeypatch, key):
    sigma = _hermitian(1)
    calls = _install_spies(monkeypatch, sigma_static_cf=sigma)
    _solve(tmp_path, monkeypatch, f"3 3 {key} 0.05", _problem(3e-3))
    assert ("fit_h0", 0.05) in calls
    assert ("sigma", False) in calls
    # the model handed to the solver is block diagonal: the weak 2 <-> 3 couplings are gone
    models = [c for c in calls if c[0] == "model"]
    assert models and all(n_h >= 2 for _, n_h, _ in models)


def test_default_keeps_the_interaction_merging(tmp_path, monkeypatch):
    calls = _install_spies(monkeypatch, sigma_static_cf=_hermitian(1))
    _solve(tmp_path, monkeypatch, "3 3", _problem(3e-3))
    assert ("fit_h0", 1e-6) in calls
    assert ("sigma", True) in calls


def test_a_raised_symm_prec_that_neglects_nothing_keeps_merging_on(tmp_path, monkeypatch):
    calls = _install_spies(monkeypatch, sigma_static_cf=_hermitian(1))
    _solve(tmp_path, monkeypatch, "3 3 symm_prec 0.05", _problem(0.3, weak=False))
    assert ("sigma", True) in calls


def test_the_double_counting_search_gets_the_same_flag(tmp_path, monkeypatch):
    calls = _install_spies(monkeypatch, dc_in_corr=[_hermitian(2)])
    _solve(tmp_path, monkeypatch, "3 3 symm_prec 0.05", _problem(3e-3), dc_flag=1)
    assert ("dc", False) in calls
    # ... and the double counting guess it starts from carries no coupling between the blocks
    models = [c for c in calls if c[0] == "model"]
    assert models and all(n_h >= 2 and n_dc >= 2 for _, n_h, n_dc in models)


def test_the_self_energy_comes_back_in_the_correlated_basis(tmp_path, monkeypatch):
    sigma = _hermitian(3)
    _install_spies(monkeypatch, sigma_static_cf=sigma)
    out = _solve(tmp_path, monkeypatch, "3 3 symm_prec 0.05", _problem(3e-3, _unitary(7)))
    with h5.File(tmp_path / "impurityModel_data.h5", "r") as f:
        corr_to_cf = np.asarray(f["t 1/corr_to_cf"])
    # the fit rotated the problem, so the composed rotation is not the identity ...
    assert np.abs(corr_to_cf - np.eye(N)).max() > 1e-2
    # ... and the solver's answer is carried back through it
    np.testing.assert_allclose(out["sig_static"], corr_to_cf @ sigma @ corr_to_cf.conj().T, atol=1e-12)
    np.testing.assert_allclose(out["sig"][:, :, 0], corr_to_cf @ sigma @ corr_to_cf.conj().T, atol=1e-12)


def test_archive_records_the_requested_value_and_the_measured_flag(tmp_path, monkeypatch):
    _install_spies(monkeypatch, sigma_static_cf=_hermitian(1))
    _solve(tmp_path, monkeypatch, "3 3 symm_prec 0.05", _problem(3e-3))
    with h5.File(tmp_path / "impurityModel_data.h5", "r") as f:
        attrs = f["t 1"].attrs
        assert attrs["symm_prec"] == pytest.approx(0.05)
        assert bool(attrs["interaction_merges_blocks"]) is False


def test_damped_double_counting_is_added_in_one_basis(tmp_path, monkeypatch):
    """Each iteration's solver basis is its own fitting basis; the damping must not add matrices of
    different bases. With one block (strong coupling) the damped answer in RSPt's basis is exactly
    ``D1 + alpha (D2 - D1)``."""
    d1, d2 = _hermitian(11), _hermitian(12)
    _install_spies(monkeypatch, dc_in_corr=[d1, d2])
    first = _solve(tmp_path, monkeypatch, "3 3", _problem(0.3, _unitary(21), connect_last=True), dc_flag=1)
    np.testing.assert_allclose(first["sig_dc"], d1, atol=1e-10)
    second = _solve(tmp_path, monkeypatch, "3 3", _problem(0.3, _unitary(22), connect_last=True), dc_flag=1)
    with h5.File(tmp_path / "impurityModel_data.h5", "r") as f:
        c1, c2 = (np.asarray(f[f"t {it}/corr_to_cf"]) for it in (1, 2))
    assert np.abs(c1 - c2).max() > 1e-2, "the two iterations must fit in different bases"
    np.testing.assert_allclose(second["sig_dc"], d1 + 0.5 * (d2 - d1), atol=1e-10)


@pytest.mark.parametrize(("line", "expected"), [("3 3", 2.0), ("3 3 u_densify 4", 4.0), ("3 3 u_densify inf", np.inf)])
def test_the_fit_gets_the_interaction_and_the_densification_budget(tmp_path, monkeypatch, line, expected):
    seen = {}

    def fit_h0(*args, **kwargs):
        seen.update(kwargs)
        seen["options"] = args[5]
        raise RuntimeError("stop after the fit call")

    monkeypatch.setattr(solve, "fit_h0", fit_h0)
    with pytest.raises(RuntimeError, match="stop after the fit call"):
        _solve(tmp_path, monkeypatch, line, _problem(3e-3))
    assert seen["options"].u_densify == expected
    assert seen["u4"].shape == (N,) * 4
    # corr_to_cf and corr_to_spherical are the identity here
    assert np.allclose(seen["rot_to_spherical"], np.eye(N))
