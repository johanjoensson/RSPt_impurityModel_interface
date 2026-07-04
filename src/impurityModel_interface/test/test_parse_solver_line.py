"""
Unit tests for the pure python helpers in impurityModel_interface.lib.

These tests do not need MPI, RSPt, or a built CFFI embedding library.
"""

import numpy as np
import pytest

# impurityModel accesses MPI.COMM_WORLD at import time. Embedded in RSPt, MPI
# is initialized by Fortran; here we have to initialize it ourselves *before*
# importing lib (which sets mpi4py.rc.initialize = False).
pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: F401
from rspt2spectra.block_structure import build_block_structure

from impurityModel_interface.lib import (
    get_weight_function,
    h5_write_dataset,
    parse_solver_line,
)


def test_minimal_line():
    n0, n_baths, options = parse_solver_line("8 10")
    assert n0 == 8
    assert n_baths == 10
    # Star is the default geometry; it disables chain restrictions and sets dN
    assert options["bath_geometry"] == "star"
    assert options["chain_restrict"] is False
    assert options["collapse_chains"] is True
    assert options["dN"] == 4
    assert options["weight_function"] == "unit"


@pytest.mark.parametrize("comment_char", ["!", "#"])
def test_comments_are_stripped(comment_char):
    n0, n_baths, options = parse_solver_line(
        f"8 10 {comment_char} chain gamma 0.5 trailing comment"
    )
    assert n0 == 8
    assert n_baths == 10
    assert options["bath_geometry"] == "star"
    assert options["gamma"] == 0.01


def test_full_option_line():
    line = (
        "8 10 haver full dense_cutoff 500 gamma 0.1 gaussian weight 3 "
        "weight_w0 -1.5 spin_flip_dj occ_cutoff 1e-4 truncation_threshold 1e7 "
        "slater_min 1e-8 dn 2 mv 1 sparse_green no_chain_restrict fit_unocc"
    )
    n0, n_baths, options = parse_solver_line(line)
    assert n0 == 8
    assert n_baths == 10
    assert options["bath_geometry"] == "haver"
    assert options["reort"] == "full"
    assert options["dense_cutoff"] == 500
    assert options["gamma"] == pytest.approx(0.1)
    assert options["weight_function"] == "gaussian"
    assert options["weight"] == pytest.approx(3)
    assert options["weight_w0"] == pytest.approx(-1.5)
    assert options["spin_flip_dj"] is True
    assert options["occ_cutoff"] == pytest.approx(1e-4)
    assert options["truncation_threshold"] == int(1e7)
    assert options["slater_min"] == pytest.approx(1e-8)
    assert options["dN"] == 2
    assert options["mv"] == 1
    assert options["sparse_green"] is True
    assert options["chain_restrict"] is False
    assert options["fit_unocc"] is True


def test_pro_maps_to_partial():
    _, _, options = parse_solver_line("8 10 chain pro")
    assert options["reort"] == "partial"


@pytest.mark.parametrize("reort", ["partial", "selective", "full", "periodic"])
def test_reort_modes(reort):
    _, _, options = parse_solver_line(f"8 10 chain {reort}")
    assert options["reort"] == reort


def test_chain_keeps_chain_restrict():
    _, _, options = parse_solver_line("8 10 chain")
    assert options["bath_geometry"] == "chain"
    assert options["chain_restrict"] is True
    assert options["dN"] is None


def test_unknown_argument_raises():
    with pytest.raises(RuntimeError, match="Unknown solver parameter"):
        parse_solver_line("8 10 bogus_option")


def test_too_few_arguments_raises():
    with pytest.raises(AssertionError):
        parse_solver_line("8")


def test_non_integer_arguments_raise():
    with pytest.raises(RuntimeError):
        parse_solver_line("eight 10")


def test_weight_function_unit():
    f = get_weight_function("unit", 0.0, 2.0)
    w = np.linspace(-5, 5, 11)
    assert np.allclose(f(w), 1.0)


def test_weight_function_gaussian():
    f = get_weight_function("gaussian", 1.0, 2.0)
    w = np.linspace(-5, 5, 101)
    assert f(np.array([1.0]))[0] == pytest.approx(1.0)
    assert np.all(f(w) <= 1.0)


def test_weight_function_unknown_raises():
    with pytest.raises(KeyError):
        get_weight_function("does_not_exist", 0.0, 2.0)


def test_combined_block_structure_merges_h_coupling():
    # The hybridization function is diagonal, but the local hamiltonian
    # couples the two orbitals; they must end up in the same block.
    n_w = 3
    phase_hyb = np.zeros((n_w, 2, 2), dtype=complex)
    phase_hyb[:, 0, 0] = 1.0
    phase_hyb[:, 1, 1] = 2.0
    H_local = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=complex)
    bs = build_block_structure(phase_hyb, mat=H_local)
    assert bs.blocks == [[0, 1]]
    assert bs.inequivalent_blocks == [0]


def test_combined_block_structure_keeps_disconnected_blocks():
    n_w = 3
    phase_hyb = np.zeros((n_w, 2, 2), dtype=complex)
    phase_hyb[:, 0, 0] = 1.0
    phase_hyb[:, 1, 1] = 2.0
    H_local = np.diag([0.0, 1.0]).astype(complex)
    bs = build_block_structure(phase_hyb, mat=H_local)
    assert bs.blocks == [[0], [1]]
    # Different hybridization and local energies: not equivalent
    assert bs.inequivalent_blocks == [0, 1]


def test_combined_block_structure_identical_blocks():
    n_w = 3
    phase_hyb = np.zeros((n_w, 2, 2), dtype=complex)
    phase_hyb[:, 0, 0] = 1.0
    phase_hyb[:, 1, 1] = 1.0
    H_local = np.diag([0.5, 0.5]).astype(complex)
    bs = build_block_structure(phase_hyb, mat=H_local)
    assert bs.blocks == [[0], [1]]
    # Same hybridization and local energies: one inequivalent block
    assert bs.inequivalent_blocks == [0]


def test_h5_write_dataset_overwrites(tmp_path):
    h5 = pytest.importorskip("h5py")
    with h5.File(tmp_path / "test.h5", "w") as f:
        g = f.create_group("g")
        h5_write_dataset(g, "data", np.arange(3))
        # Overwriting must not raise, and must store the new data
        h5_write_dataset(g, "data", np.arange(5))
        assert np.array_equal(g["data"][...], np.arange(5))
