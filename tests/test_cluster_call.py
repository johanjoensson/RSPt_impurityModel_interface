"""What used to be computed in Fortran: ``h_dft``, the rotation, the dipole, the stacked symmetries."""

import numpy as np
import pytest
from fake_cluster import N, cluster

from impurityModel_interface.cluster import ClusterCall


def test_h_dft_is_ham_minus_mu_on_the_diagonal_and_ham_is_untouched():
    f = cluster()
    ham = f["ham"].copy()
    call = ClusterCall.from_fields(f)
    np.testing.assert_allclose(call.h_dft, ham - 0.25 * np.eye(N))
    np.testing.assert_array_equal(f["ham"], ham)
    assert call.h_dft.flags.f_contiguous


def test_the_rotation_is_corr_to_harmonic_and_the_layout_comes_from_the_shapes():
    f = cluster(n=4, n_full=6)  # a subset of a larger shell
    call = ClusterCall.from_fields(f)
    np.testing.assert_allclose(call.corr_to_spherical, f["harm2corr"].conj().T)
    assert call.corr_to_spherical.flags.f_contiguous
    assert call.shells.nspmat == 2 and call.shells.preoffset == (0, 6) and call.shells.offset == (0, 4)
    assert (call.label, call.solver_line, call.dc_line, call.task) == ("Ni-3d", "8 10", "gap 0.5", "dc")
    assert (call.eim, call.tau, call.verbosity) == (0.1, 0.01, 1)


def test_the_task_is_the_dc_flag_unless_rspt_names_it():
    assert ClusterCall.from_fields(cluster(extra={"ctx%dc_flag": 0})).task == "selfenergy"
    assert ClusterCall.from_fields(cluster(extra={"ctx%dc_flag": 0, "ctx%task": "spectra "})).task == "spectra"


def test_the_outputs_and_inputs_are_rspts_arrays_not_copies():
    f = cluster()
    call = ClusterCall.from_fields(f)
    assert call.out.sig is f["sig"] and call.out.sig_dc is f["sig_dc"]
    assert call.u4 is f["ctx%u4"] and call.hyb is f["ctx%hyb"]


def test_operators_are_cut_to_the_cluster_and_zero_counts_as_absent():
    big = np.arange(36.0).reshape(6, 6) + 1j
    f = cluster(extra={"observables%ms": big, "observables%ml": np.zeros((N, N), complex)})
    ops = ClusterCall.from_fields(f).operators
    np.testing.assert_array_equal(ops.sz, big[:N, :N])
    assert ops.lz is None and ops.s_plus is None and ops.dipole is None
    assert ops.source == "rspt" and ops.frame == "global"


def test_no_operators_at_all_is_none():
    assert ClusterCall.from_fields(cluster()).operators is None


def _obs(names_and_matrices):
    extra = {}
    for k, (name, t) in enumerate(names_and_matrices, start=1):
        extra[f"observables%obs({k})%name"] = name
        extra[f"observables%obs({k})%t"] = t
    return extra


def test_a_cartesian_dipole_is_stacked_as_is():
    rng = np.random.default_rng(4)
    d = [rng.normal(size=(N, N)) + 0j for _ in range(3)]
    f = cluster(extra=_obs([("(Dipole) Dx", d[0]), ("(Dipole) Dy", d[1]), ("(Dipole) Dz", d[2])]))
    dipole = ClusterCall.from_fields(f).operators.dipole
    assert dipole.shape == (3, N, N)
    for i in range(3):
        np.testing.assert_array_equal(dipole[i], d[i])


def test_a_spherical_dipole_is_converted_to_cartesian():
    rng = np.random.default_rng(5)
    dp, dm, dz = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)) for _ in range(3))
    f = cluster(extra=_obs([("S+", dz), ("(Dipole) Dp", dp), ("(Dipole) Dm", dm), ("(Dipole) Dz", dz)]))
    dipole = ClusterCall.from_fields(f).operators.dipole
    np.testing.assert_allclose(dipole[0], (dp + dm) / np.sqrt(2))
    np.testing.assert_allclose(dipole[1], (dp - dm) / (1j * np.sqrt(2)))
    np.testing.assert_allclose(dipole[2], dz)


def test_symmetries_are_stacked_with_their_time_reversal_flags():
    reps = [np.eye(N) * (i + 1) + 0j for i in range(3)]
    extra = {"symm%n": 3}
    for i, rep in enumerate(reps, start=1):
        extra[f"symm({i})%rep"] = rep
        extra[f"symm({i})%timerev"] = i == 2
    ops = ClusterCall.from_fields(cluster(extra=extra)).operators
    assert [flag for _, flag in ops.symmetry_ops] == [False, True, False]
    np.testing.assert_array_equal(ops.symmetry_ops[1][0], reps[1])


def test_one_operation_without_a_representation_means_no_symmetry_data():
    extra = {"symm%n": 2, "symm(1)%rep": np.eye(N, dtype=complex), "symm(1)%timerev": False}
    assert ClusterCall.from_fields(cluster(extra=extra)).operators is None


def test_an_output_of_the_wrong_shape_is_refused_not_reshaped():
    f = cluster()
    f["sig_static"] = np.zeros((N + 1, N + 1), complex)
    with pytest.raises(ValueError, match="sig_static"):
        ClusterCall.from_fields(f)


def test_a_missing_required_field_names_itself():
    f = cluster()
    del f["harm2corr"]
    with pytest.raises(KeyError, match="harm2corr"):
        ClusterCall.from_fields(f)


def test_inconsistent_rotation_shapes_are_refused():
    f = cluster(extra={"corr2cf": np.zeros((N, N - 1), complex)})
    with pytest.raises(RuntimeError, match="Inconsistent rotation shapes"):
        ClusterCall.from_fields(f)
