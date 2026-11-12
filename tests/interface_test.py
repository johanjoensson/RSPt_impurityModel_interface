"""
Regression test for the RSPt <-> impurityModel interface.

Recomputes the self-energy from a saved `impurityModel_data.h5` archive
(written by a previous wrapper run) and compares against the self-energy
stored in the same archive. The archive is self-contained: it holds the full
one-particle solver hamiltonian ("H solver"), the Coulomb interaction ("U"),
the frequency meshes, the corr -> CF rotation and all solver options.

Point IMPMOD_INTERFACE_TEST_DATA at an archive to run the test; it is skipped
otherwise. Run under MPI with e.g. `mpirun -n 2 pytest interface_test.py`.

Archives written before the archive was versioned are refused by the loader; regenerate the archive
with a current wrapper run.
"""

import os

import numpy as np
import pytest

h5 = pytest.importorskip("h5py")
pytest.importorskip("mpi4py")
api = pytest.importorskip("impurityModel.api")

from impurityModel.api import calc_selfenergy, load_selfenergy_archive  # noqa: E402
from mpi4py import MPI  # noqa: E402
from rspt2spectra.utils import rotate_Greens_function, rotate_matrix  # noqa: E402

TEST_DATA = os.environ.get(
    "IMPMOD_INTERFACE_TEST_DATA",
    os.path.join(os.path.dirname(__file__), "impurityModel_data.h5"),
)

requires_data = pytest.mark.skipif(
    not os.path.exists(TEST_DATA),
    reason=(
        "No test archive found. Set IMPMOD_INTERFACE_TEST_DATA to an impurityModel_data.h5 produced by the wrapper."
    ),
)


def read_references(filename, label, iteration):
    """The rotation to the correlated basis and the stored self-energies of one iteration."""
    with h5.File(filename, "r") as f:
        g = f[f"{label} {iteration}"]
        return {
            "corr_to_solver": g["corr_to_solver"][...],
            "sigma_static_ref": g["Sigma Static"][...] if "Sigma Static" in g else None,
            "sigma_ref": g["Sigma Matsubara"][...] if "Sigma Matsubara" in g else None,
            "sigma_real_ref": g["Sigma real"][...] if "Sigma real" in g else None,
        }


@requires_data
def test_selfenergy_against_reference():
    model, meshes, basis, solver, label = load_selfenergy_archive(TEST_DATA)
    with h5.File(TEST_DATA, "r") as f:
        iteration = int(f.attrs["last iteration"])
    data = read_references(TEST_DATA, label, iteration)
    if data["sigma_ref"] is None:
        pytest.skip("Archive holds no reference self-energy (double counting run?)")

    results = calc_selfenergy(model, meshes, basis, solver, comm=MPI.COMM_WORLD, verbosity=0, cluster_label=label)

    if MPI.COMM_WORLD.rank != 0:
        return
    # calc_selfenergy returns the self-energy in its input (fitting) basis; the
    # stored references are in the corr basis.
    u = np.conj(data["corr_to_solver"].T)
    sigma_static = rotate_matrix(results["sigma_static"], u)
    sigma = rotate_Greens_function(results["sigma"], u)
    sigma_real = rotate_Greens_function(results["sigma_real"], u)

    assert np.allclose(sigma_static, data["sigma_static_ref"], atol=1e-4)
    assert np.allclose(sigma, data["sigma_ref"], atol=1e-4)
    assert np.allclose(sigma_real, data["sigma_real_ref"], atol=1e-4)
