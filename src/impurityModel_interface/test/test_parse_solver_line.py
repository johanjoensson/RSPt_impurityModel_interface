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
    format_settings_header,
    get_weight_function,
    h5_write_dataset,
    parse_solver_line,
    solver_line_attrs,
)


def test_minimal_line():
    n0, n_baths, fit_options, basis, _solver = parse_solver_line("8 10 2")
    assert n0 == 8
    assert n_baths == 10
    assert basis.excitation_budget == 2
    # peeled_linked_chain is the default geometry; it keeps chain restrictions and the
    # star-only defaults (collapse_chains, dN) do not kick in.
    assert fit_options["bath_geometry"] == "peeled"
    assert basis.chain_restrict is True
    assert fit_options["collapse_chains"] is False
    assert basis.dN is None
    assert fit_options["weight_function"] == "unit"
    # The nominal occupation is carried on the basis options.
    assert basis.nominal_occ == {0: 8}


def test_freeze_bath_energies_is_off_by_default_and_opt_in():
    _, _, fit_options, _, _ = parse_solver_line("8 10 4 peeled")
    assert fit_options["freeze_bath_energies"] is False
    _, _, fit_options, _, _ = parse_solver_line("8 10 4 peeled freeze_bath_energies")
    assert fit_options["freeze_bath_energies"] is True


@pytest.mark.parametrize("comment_char", ["!", "#"])
def test_comments_are_stripped(comment_char):
    n0, n_baths, fit_options, basis, _solver = parse_solver_line(
        f"8 10 4 {comment_char} chain gamma 0.5 trailing comment"
    )
    assert n0 == 8
    assert n_baths == 10
    assert basis.excitation_budget == 4
    # Everything after the comment char is stripped, so the geometry stays at its default.
    assert fit_options["bath_geometry"] == "peeled"
    assert fit_options["gamma"] == 0.01


def test_full_option_line():
    line = (
        "8 10 4 linked_chain full dense_cutoff 500 gamma 0.1 gaussian weight 3 "
        "weight_w0 -1.5 spin_flip_dj occ_cutoff 1e-4 truncation_threshold 1e7 "
        "slater_min 1e-8 dn 2 mv 1 no_chain_restrict fit_unocc"
    )
    n0, n_baths, fit_options, basis, solver = parse_solver_line(line)
    assert n0 == 8
    assert n_baths == 10
    assert basis.excitation_budget == 4
    assert fit_options["bath_geometry"] == "linked_chain"
    assert solver.reort == "full"
    assert solver.dense_cutoff == 500
    assert fit_options["gamma"] == pytest.approx(0.1)
    assert fit_options["weight_function"] == "gaussian"
    assert fit_options["weight"] == pytest.approx(3)
    assert fit_options["weight_w0"] == pytest.approx(-1.5)
    assert basis.occ_cutoff == pytest.approx(1e-4)
    assert basis.truncation_threshold == int(1e7)
    assert basis.slater_weight_min == pytest.approx(1e-8)
    assert basis.dN == 2
    assert basis.mixed_valence == {0: 1}
    # No dense_green token, so the sparse Green's function path is the default
    assert solver.sparse_green is True
    assert basis.chain_restrict is False
    assert fit_options["fit_unocc"] is True


def test_pro_maps_to_partial():
    *_, solver = parse_solver_line("8 10 4 chain pro")
    assert solver.reort == "partial"


@pytest.mark.parametrize("reort", ["partial", "selective", "full", "periodic"])
def test_reort_modes(reort):
    *_, solver = parse_solver_line(f"8 10 4 chain {reort}")
    assert solver.reort == reort


def test_chain_keeps_chain_restrict():
    _, _, fit_options, basis, _solver = parse_solver_line("8 10 4 chain")
    assert fit_options["bath_geometry"] == "chain"
    assert basis.chain_restrict is True
    assert basis.dN is None


def test_solver_line_attrs_reproduces_the_flat_record():
    # The parsed groups round-trip to the flat attribute record the HDF5 archive stores.
    _, _, fit_options, basis, solver = parse_solver_line(
        "8 10 4 chain full dense_cutoff 500 gamma 0.1 gaussian weight 3 weight_w0 -1.5 "
        "spin_flip_dj occ_cutoff 1e-4 truncation_threshold 1e7 slater_min 1e-8 dn 2 mv 1 "
        "dense_green no_chain_restrict fit_unocc"
    )
    attrs = solver_line_attrs(fit_options, basis, solver)
    assert attrs["reort"] == "full"
    assert attrs["dense_cutoff"] == 500
    assert attrs["sparse_green"] is False
    assert "spin_flip_dj" not in attrs
    assert attrs["chain_restrict"] is False
    assert attrs["occ_cutoff"] == pytest.approx(1e-4)
    assert attrs["dN"] == 2
    assert attrs["truncation_threshold"] == int(1e7)
    assert attrs["slater_min"] == pytest.approx(1e-8)
    assert attrs["mv"] == 1
    assert attrs["bath_geometry"] == "chain"
    assert attrs["gamma"] == pytest.approx(0.1)
    assert attrs["weight_function"] == "gaussian"
    assert attrs["fit_unocc"] is True
    assert attrs["freeze_bath_energies"] is False
    assert attrs["excitation_budget"] == 4
    assert attrs["gf_method"] == "lanczos"
    assert attrs["e_pt2_tol"] is None
    # Every option the archive stores is present (20 keys; spin_flip_dj was removed). n_baths is
    # only added when the caller passes it.
    assert len(attrs) == 20
    assert "n_baths" not in attrs
    assert solver_line_attrs(fit_options, basis, solver, n_baths=10)["n_baths"] == 10


def test_excitation_budget_is_archived():
    """The budget is the third mandatory solver-line token; without it in the archive an offline
    rerun silently falls back to impurityModel's default and solves a different problem."""
    _, n_baths, fit_options, basis, solver = parse_solver_line("3 3 8 peeled")
    attrs = solver_line_attrs(fit_options, basis, solver, n_baths=n_baths)
    assert attrs["excitation_budget"] == 8
    assert attrs["n_baths"] == 3


def test_negative_excitation_budget_disables_it(capsys):
    """A negative budget disables the budget (with a notice) instead of building an empty window;
    the archive then stores it as None, which impurityModel's reader reads back as disabled."""
    _, n_baths, fit_options, basis, solver = parse_solver_line("3 3 -1 peeled")
    assert "is negative: the excitation budget is disabled" in capsys.readouterr().out
    assert basis.excitation_budget is None
    assert solver_line_attrs(fit_options, basis, solver, n_baths=n_baths)["excitation_budget"] is None


def test_settings_header_lists_every_archived_setting():
    n0, n_baths, fit_options, basis, solver = parse_solver_line(
        "3 3 8 chain full gamma 0.1 gaussian dn 2 mv 1 dense_green"
    )
    header = format_settings_header(n0, fit_options, basis, solver, n_baths=n_baths, tau=0.025, delta=0.01)
    rows = dict(line.split(" |> ") for line in header.strip().splitlines())
    rows = {label.strip(): value for label, value in rows.items()}
    attrs = solver_line_attrs(fit_options, basis, solver, n_baths=n_baths)
    # One row per archived setting, plus the nominal occupation, tau and the broadening.
    assert len(rows) == len(attrs) + 3
    assert rows["Nominal imp. occupation"] == "3"
    assert rows["Excitation budget"] == "8"
    assert rows["Bath states per imp. orb."] == "3"
    assert rows["Fit regularization gamma"] == "0.1"
    assert rows["Sparse Green's function"] == "False"
    assert rows["Temperature tau"] == "0.025"
    assert rows["Real-axis broadening"] == "0.01"


@pytest.mark.parametrize(
    "written, expected",
    [("auto", None), ("unlimited", float("inf")), ("inf", float("inf")), ("2e6", 2_000_000), ("5000", 5000)],
)
def test_truncation_threshold_takes_the_shared_vocabulary(written, expected):
    """auto | unlimited (inf) | a positive integer, as on the CLI and in TOML. `inf` used to raise
    OverflowError here and `auto` a ValueError."""
    _, _, _, basis, _ = parse_solver_line(f"3 3 8 chain full truncation_threshold {written}")
    assert basis.truncation_threshold == expected


@pytest.mark.parametrize("written", ["0", "-10", "1.5", "lots"])
def test_truncation_threshold_rejects_nonsense(written):
    with pytest.raises(ValueError):
        parse_solver_line(f"3 3 8 chain full truncation_threshold {written}")


@pytest.mark.parametrize("written, shown", [("auto", "auto"), ("unlimited", "unlimited"), ("2000", "2000")])
def test_settings_header_names_the_cap_in_the_solver_line_words(written, shown):
    n0, n_baths, fit_options, basis, solver = parse_solver_line(f"3 3 8 chain full truncation_threshold {written}")
    header = format_settings_header(n0, fit_options, basis, solver, n_baths=n_baths)
    rows = {label.strip(): value for label, value in (line.split(" |> ") for line in header.strip().splitlines())}
    assert rows["Truncation threshold"] == shown


def test_unknown_argument_raises():
    with pytest.raises(RuntimeError, match="Unknown solver parameter"):
        parse_solver_line("8 10 4 bogus_option")


def test_single_argument_raises():
    with pytest.raises(AssertionError):
        parse_solver_line("8")


def test_two_arguments_raises():
    with pytest.raises(AssertionError):
        parse_solver_line("8 10")


def test_non_integer_arguments_raise():
    with pytest.raises(RuntimeError):
        parse_solver_line("eight 10 4")


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


def test_spin_flip_dj_is_accepted_and_ignored():
    """An existing green.inp may still carry the token; it must parse to the same options as the
    line without it, since the option it named never had an effect and no longer exists."""
    with_token = parse_solver_line("8 10 4 chain spin_flip_dj occ_cutoff 1e-4")
    without = parse_solver_line("8 10 4 chain occ_cutoff 1e-4")
    assert with_token == without


def test_e_pt2_is_unset_by_default_so_impurity_model_uses_its_own():
    _, _, _, basis, _ = parse_solver_line("8 10 4 peeled")
    assert basis.e_pt2_tol is None


@pytest.mark.parametrize("spelling", ["e_pt2", "E_PT2", "e_pt2_tol"])
def test_e_pt2_sets_the_ground_state_tolerance(spelling):
    _, _, _, basis, _ = parse_solver_line(f"8 10 4 peeled {spelling} 1e-6 slater_min 0")
    assert basis.e_pt2_tol == 1e-6
    assert basis.slater_weight_min == 0.0, "the next token must still parse"


@pytest.mark.parametrize("tail", ["e_pt2", "e_pt2 0", "e_pt2 -1e-8"])
def test_e_pt2_needs_a_positive_value(tail):
    with pytest.raises(AssertionError, match="e_pt2"):
        parse_solver_line(f"8 10 4 peeled {tail}")


def test_e_pt2_is_archived_and_printed():
    n0, n_baths, fit_options, basis, solver = parse_solver_line("8 10 4 peeled e_pt2 1e-6")
    assert solver_line_attrs(fit_options, basis, solver)["e_pt2_tol"] == 1e-6
    header = format_settings_header(n0, fit_options, basis, solver, n_baths=n_baths)
    assert "GS residual PT2 tolerance" in header and "1e-06" in header
