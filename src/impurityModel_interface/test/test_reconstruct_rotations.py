"""
Tests for reconstruct_rotations, covering every combination of shapes RSPt can
send (see green_locust.F90 on the ImpModED branch):

  n_orb       = size(cluster%ham, 1)      -- always 2 spins
  n_rot_cols  = n_orb * nspmat / 2        -- corr2cf columns
  n_orb_full  = spherical dim, nspmat spins

with nspmat = 1 (non-spin-polarized, no SOC) or 2 (spin-polarized or fulrel),
and the correlated set either a full l-shell or a subset (t2g, eg, ...).
"""

import numpy as np
import pytest

# impurityModel accesses MPI.COMM_WORLD at import time. Embedded in RSPt, MPI
# is initialized by Fortran; here we have to initialize it ourselves *before*
# importing lib (which sets mpi4py.rc.initialize = False).
pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: F401

from impurityModel_interface.lib import reconstruct_rotations


def _one_spin_inputs(n_orb, n_rot_cols, n_orb_full, rng):
    """Build inputs the way RSPt lays them out for nspmat = 1: the spin-down
    corr rows carry the rotation, the spin-up rows are zero."""
    n_half = n_orb // 2
    corr_to_spherical = np.zeros((n_orb, n_orb_full), dtype=complex)
    corr_to_spherical[:n_half, :] = rng.random((n_half, n_orb_full)) + 1j * rng.random((n_half, n_orb_full))
    corr_to_cf = np.zeros((n_orb, n_rot_cols), dtype=complex)
    corr_to_cf[:n_half, :] = rng.random((n_half, n_rot_cols)) + 1j * rng.random((n_half, n_rot_cols))
    return corr_to_spherical, corr_to_cf


def test_nonmagnetic_full_d_shell():
    # d shell: n_orb = 10, one spin sent: n_rot_cols = n_orb_full = 5
    rng = np.random.default_rng(0)
    sph_in, cf_in = _one_spin_inputs(10, 5, 5, rng)
    sph, cf = reconstruct_rotations(sph_in, cf_in, 10, 5, 5)

    assert sph.shape == (10, 10)
    assert cf.shape == (10, 10)
    # Block diagonal with identical spin blocks
    np.testing.assert_allclose(sph[:5, :5], sph_in[:5, :])
    np.testing.assert_allclose(sph[5:, 5:], sph_in[:5, :])
    assert np.all(sph[:5, 5:] == 0)
    assert np.all(sph[5:, :5] == 0)
    np.testing.assert_allclose(cf[:5, :5], cf_in[:5, :])
    np.testing.assert_allclose(cf[5:, 5:], cf_in[:5, :])
    assert np.all(cf[:5, 5:] == 0)
    assert np.all(cf[5:, :5] == 0)
    # Matches the historical roll-based construction, which was correct for
    # full shells (n_orb == 2 * n_orb_full)
    legacy_sph = np.empty((10, 10), dtype=complex)
    legacy_sph[:, :5] = sph_in
    legacy_sph[:, 5:] = np.roll(sph_in, 5, axis=0)
    np.testing.assert_allclose(sph, legacy_sph)


def test_nonmagnetic_full_shell_unitary_stays_unitary():
    rng = np.random.default_rng(1)
    n = 5
    block, _ = np.linalg.qr(rng.random((n, n)) + 1j * rng.random((n, n)))
    cf_in = np.zeros((2 * n, n), dtype=complex)
    cf_in[:n, :] = block
    _, cf = reconstruct_rotations(np.zeros((2 * n, n), dtype=complex), cf_in, 2 * n, n, n)
    np.testing.assert_allclose(np.conj(cf.T) @ cf, np.eye(2 * n), atol=1e-12)


def test_nonmagnetic_t2g_subset():
    # t2g only: n_orb = 6 correlated spin-orbitals out of a d shell,
    # one spin sent: n_rot_cols = 3, n_orb_full = 5.
    rng = np.random.default_rng(2)
    sph_in, cf_in = _one_spin_inputs(6, 3, 5, rng)
    sph, cf = reconstruct_rotations(sph_in, cf_in, 6, 3, 5)

    assert sph.shape == (6, 10)
    assert cf.shape == (6, 6)
    # The spin-up block sits at rows n_orb//2 = 3, not n_orb_full = 5 (the
    # np.roll(..., n_orb_full) of the old implementation garbled this case).
    np.testing.assert_allclose(sph[:3, :5], sph_in[:3, :])
    np.testing.assert_allclose(sph[3:, 5:], sph_in[:3, :])
    assert np.all(sph[:3, 5:] == 0)
    assert np.all(sph[3:, :5] == 0)
    np.testing.assert_allclose(cf[:3, :3], cf_in[:3, :])
    np.testing.assert_allclose(cf[3:, 3:], cf_in[:3, :])
    assert np.all(cf[:3, 3:] == 0)
    assert np.all(cf[3:, :3] == 0)


def test_two_spin_full_shell_passthrough():
    # Spin-polarized or fulrel full d shell: everything square, pass through.
    rng = np.random.default_rng(3)
    sph_in = rng.random((10, 10)) + 1j * rng.random((10, 10))
    cf_in = rng.random((10, 10)) + 1j * rng.random((10, 10))
    sph, cf = reconstruct_rotations(sph_in, cf_in, 10, 10, 10)
    np.testing.assert_allclose(sph, sph_in)
    np.testing.assert_allclose(cf, cf_in)


def test_two_spin_t2g_subset_passthrough():
    # Spin-polarized t2g: matrices already carry both spins, corr_to_spherical
    # rectangular (6, 10). The old implementation crashed on this case.
    rng = np.random.default_rng(4)
    sph_in = rng.random((6, 10)) + 1j * rng.random((6, 10))
    cf_in = rng.random((6, 6)) + 1j * rng.random((6, 6))
    sph, cf = reconstruct_rotations(sph_in, cf_in, 6, 6, 10)
    np.testing.assert_allclose(sph, sph_in)
    np.testing.assert_allclose(cf, cf_in)
    assert sph.shape == (6, 10)
    assert cf.shape == (6, 6)


def test_inconsistent_shapes_raise():
    with pytest.raises(RuntimeError, match="Inconsistent rotation shapes"):
        reconstruct_rotations(
            np.zeros((6, 5), dtype=complex),
            np.zeros((6, 2), dtype=complex),
            6,
            2,
            5,
        )
