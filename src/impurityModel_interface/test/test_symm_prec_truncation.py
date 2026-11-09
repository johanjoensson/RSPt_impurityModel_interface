"""End to end: couplings below ``symm_prec`` are neglected by the bath fit, and above it they are kept.

A synthetic 5-orbital problem (2 + 3 orbitals) with a weak coupling between the two groups, both in
the hybridization function and in H_dft. ``get_ed_h0`` must hand back an impurity block that is exactly
block diagonal in the fitting basis when ``symm_prec`` exceeds the coupling, and the same block must
carry the coupling when it does not.
"""

import numpy as np
import pytest

pytest.importorskip("mpi4py")
from mpi4py import MPI

from impurityModel_interface.lib import get_ed_h0

N = 5
COUPLING = 3e-3


def _problem():
    w = np.linspace(-3, 3, 300)
    eim = 0.05
    eps = np.array([0.5, 0.5, -0.3, -0.2, -0.25])
    hop = np.diag([0.4, 0.4, 0.3, 0.3, 0.3])
    hyb = np.zeros((len(w), N, N), dtype=complex)
    for i in range(N):
        hyb[:, i, i] = hop[i, i] ** 2 / (w + 1j * eim - eps[i])
    hyb[:, 0, 2] = hyb[:, 2, 0] = COUPLING * 0.5 / (w + 1j * eim)
    H = np.diag([0.6, 0.6, -0.1, -0.1, -0.1])
    H[0, 1] = H[1, 0] = 0.02
    H[2, 3] = H[3, 2] = 0.01
    H[1, 3] = H[3, 1] = COUPLING
    return H, hyb, w, eim


def _run(tmp_path, monkeypatch, symm_prec):
    monkeypatch.chdir(tmp_path)
    H, hyb, w, eim = _problem()
    return get_ed_h0(
        H,
        hyb,
        3,
        w,
        eim,
        0.0,
        symm_prec=symm_prec,
        verbose=False,
        label="t",
        hdf5_filename="fit.h5",
        comm=MPI.COMM_WORLD,
    )


def _off_block(mat, blocks):
    mask = np.ones(mat.shape, bool)
    for b in blocks:
        mask[np.ix_(b, b)] = False
    return np.abs(mat[mask]).max()


def test_coupling_below_symm_prec_is_neglected(tmp_path, monkeypatch):
    H_imp, *_rest, Q, bs = _run(tmp_path, monkeypatch, 0.05)
    assert len(bs.blocks) > 1
    assert _off_block(Q.conj().T @ H_imp @ Q, bs.blocks) < 1e-12


def test_coupling_above_symm_prec_is_kept(tmp_path, monkeypatch):
    H_imp, *_rest, Q, bs = _run(tmp_path, monkeypatch, 1e-6)
    # the weak e_g/t_2g coupling joins the two groups into one block, so it survives in the
    # returned impurity block (which also carries the static hybridization shift)
    assert any(len(b) >= 4 for b in bs.blocks)
    Hq = Q.conj().T @ H_imp @ Q
    between_groups = np.abs(Hq[np.ix_([0, 1], [2, 3, 4])]).max()
    assert between_groups > 0.1 * COUPLING


def test_neglecting_removes_exactly_that_coupling(tmp_path, monkeypatch):
    kept, *_rest, Qk, _ = _run(tmp_path, monkeypatch, 1e-6)
    cut, *_rest, Qc, _ = _run(tmp_path / "cut", monkeypatch, 0.05) if (tmp_path / "cut").mkdir() is None else None
    Hk = Qk.conj().T @ kept @ Qk
    Hc = Qc.conj().T @ cut @ Qc
    assert np.abs(Hc[np.ix_([0, 1], [2, 3, 4])]).max() < 1e-12
    assert np.abs(Hk[np.ix_([0, 1], [2, 3, 4])]).max() > 1e-4
