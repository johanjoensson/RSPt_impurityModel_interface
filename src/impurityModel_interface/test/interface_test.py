"""
Regression test for the RSPt <-> impurityModel interface.

Recomputes the self-energy from a saved `impurityModel_data.h5` archive
(written by a previous wrapper run) and compares against the self-energy
stored in the same archive. The archive is self-contained: it holds the full
one-particle solver hamiltonian ("H solver"), the Coulomb interaction ("U"),
the frequency meshes, the corr -> CF rotation and all solver options.

Point IMPMOD_INTERFACE_TEST_DATA at an archive to run the test; it is skipped
otherwise. Run under MPI with e.g. `mpirun -n 2 pytest interface_test.py`.

Note: archives written before the switch to RSPt's u4 index convention stored
"U" with the first two indices swapped (the old `np.moveaxis(u4, 1, 0)`
workaround) and are not compatible with this test; regenerate the archive
with a current wrapper run.
"""

import os

import numpy as np
import pytest

h5 = pytest.importorskip("h5py")
pytest.importorskip("mpi4py")
api = pytest.importorskip("impurityModel.api")

from impurityModel.api import calc_selfenergy, matrixToIOp  # noqa: E402
from mpi4py import MPI  # noqa: E402
from rspt2spectra.utils import rotate_Greens_function, rotate_matrix  # noqa: E402

TEST_DATA = os.environ.get(
    "IMPMOD_INTERFACE_TEST_DATA",
    os.path.join(os.path.dirname(__file__), "impurityModel_data.h5"),
)

requires_data = pytest.mark.skipif(
    not os.path.exists(TEST_DATA),
    reason=(
        "No test archive found. Set IMPMOD_INTERFACE_TEST_DATA to an "
        "impurityModel_data.h5 produced by the wrapper."
    ),
)


def read_cluster(filename):
    """Load the newest iteration of the first cluster in the archive."""
    with h5.File(filename, "r") as f:
        it = f.attrs["last iteration"]
        cluster_names = [name for name in f if name.endswith(f" {it}")]
        assert cluster_names, f"No cluster group for iteration {it} in {filename}"
        g = f[cluster_names[0]]
        label = cluster_names[0].rsplit(" ", 1)[0]

        data = {
            "label": label,
            "H_solver": g["H solver"][...],
            "u4": g["U"][...],
            "w": g["Real frequency mesh"][...],
            "iw": g["Matsubara frequency mesh"][...],
            "rot_to_spherical": g["Rot to spherical"][...],
            "corr_to_cf": g["corr_to_cf"][...],
            "impurity_orbitals": [int(i) for i in g["Impurity orbitals"][...]],
            "tau": g.attrs["tau"],
            "delta": g.attrs["delta"],
            "nominal_occ": int(g.attrs["nominal occupation"]),
            "options": {key: g.attrs[key] for key in g.attrs},
            "sigma_static_ref": g["Sigma Static"][...] if "Sigma Static" in g else None,
            "sigma_ref": g["Sigma Matsubara"][...] if "Sigma Matsubara" in g else None,
            "sigma_real_ref": g["Sigma real"][...] if "Sigma real" in g else None,
        }
    return data


@requires_data
def test_selfenergy_against_reference():
    data = read_cluster(TEST_DATA)
    if data["sigma_ref"] is None:
        pytest.skip("Archive holds no reference self-energy (double counting run?)")
    options = data["options"]
    dN = options["dN"]
    dN = None if isinstance(dN, str) else int(dN)
    mv = options["mv"]
    mixed_valence = None if isinstance(mv, str) else {0: int(mv)}

    h_op = matrixToIOp(data["H_solver"])
    results = calc_selfenergy(
        h0=h_op,
        u4=data["u4"],
        iw=1j * data["iw"],
        w=data["w"],
        delta=data["delta"],
        nominal_occ={0: data["nominal_occ"]},
        mixed_valence=mixed_valence,
        impurity_orbitals={0: data["impurity_orbitals"]},
        tau=data["tau"],
        verbosity=0,
        rot_to_spherical=data["rot_to_spherical"],
        cluster_label=data["label"],
        comm=MPI.COMM_WORLD,
        reort=options["reort"],
        dense_cutoff=int(options["dense_cutoff"]),
        spin_flip_dj=bool(options["spin_flip_dj"]),
        chain_restrict=bool(options["chain_restrict"]),
        occ_cutoff=float(options["occ_cutoff"]),
        truncation_threshold=int(options["truncation_threshold"]),
        slaterWeightMin=float(options["slater_min"]),
        dN=dN,
        sparse_green=bool(options["sparse_green"]),
    )

    if MPI.COMM_WORLD.rank != 0:
        return
    # calc_selfenergy returns the self-energy in its input (CF) basis; the
    # stored references are in the corr basis.
    u = np.conj(data["corr_to_cf"].T)
    sigma_static = rotate_matrix(results["sigma_static"], u)
    sigma = rotate_Greens_function(results["sigma"], u)
    sigma_real = rotate_Greens_function(results["sigma_real"], u)

    assert np.allclose(sigma_static, data["sigma_static_ref"], atol=1e-4)
    assert np.allclose(sigma, data["sigma_ref"], atol=1e-4)
    assert np.allclose(sigma_real, data["sigma_real_ref"], atol=1e-4)
