"""``freeze_bath_energies``: reuse the previous fit's bath energies, re-solve only the hoppings.

The interface side is ``_previous_bath_energies`` -- it digs the most recent stored bath
energies out of the archive, ignoring the hybridization fingerprint (unlike the reuse path in
``fit_hyb_star``) because the point is to hold them fixed across the fingerprint change a new
DMFT iteration brings. This pins that lookup; the fit itself is exercised in rspt2spectra's
``test_freeze_bath_energies_solves_only_hoppings``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mpi4py")
h5 = pytest.importorskip("h5py")
from mpi4py import MPI  # noqa: F401,E402

from impurityModel_interface.lib import _previous_bath_energies, h5_write_dataset  # noqa: E402

LABEL = "Ni 1"
BLOCKS = SimpleNamespace(inequivalent_blocks=[0, 2])


def _write_fit(f, it, eb0, eb2):
    g = f.require_group(f"{LABEL} {it}/Bath fit/ebs_star")
    h5_write_dataset(g, "0", np.asarray(eb0, dtype=float))
    h5_write_dataset(g, "2", np.asarray(eb2, dtype=float))


def test_returns_none_without_an_archive(tmp_path):
    assert _previous_bath_energies(tmp_path / "missing.h5", LABEL, BLOCKS) is None


def test_returns_none_when_no_fit_is_stored(tmp_path):
    path = tmp_path / "a.h5"
    with h5.File(path, "w") as f:
        f.attrs["last iteration"] = 3
        f.create_group(f"{LABEL} 3")
    assert _previous_bath_energies(path, LABEL, BLOCKS) is None


def test_reads_the_most_recent_stored_energies_and_dedupes(tmp_path):
    path = tmp_path / "a.h5"
    with h5.File(path, "w") as f:
        f.attrs["last iteration"] = 4
        _write_fit(f, 2, [-0.5, -0.2], [-0.6, -0.6, -0.3])  # older
        _write_fit(f, 4, [-0.51, -0.19, -0.19], [-0.61, -0.29])  # newest, with a flatten duplicate

    eb = _previous_bath_energies(path, LABEL, BLOCKS)
    assert eb is not None
    np.testing.assert_allclose(eb[0], [-0.51, -0.19])  # duplicate collapsed, sorted
    np.testing.assert_allclose(eb[1], [-0.61, -0.29])


def test_skips_iterations_without_a_fit(tmp_path):
    path = tmp_path / "a.h5"
    with h5.File(path, "w") as f:
        f.attrs["last iteration"] = 5
        _write_fit(f, 2, [-0.4], [-0.7])
        f.create_group(f"{LABEL} 5")  # a later iteration that failed before fitting

    eb = _previous_bath_energies(path, LABEL, BLOCKS)
    np.testing.assert_allclose(eb[0], [-0.4])
    np.testing.assert_allclose(eb[1], [-0.7])
