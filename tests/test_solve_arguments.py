"""What used to be computed in Fortran: ``h_dft``, the rotation, the dipole, the stacked symmetries."""

import numpy as np
import pytest

from impurityModel_interface.rspt_fields import solve_arguments

N = 4


def cluster(n=N, extra=None, n_full=None):
    n_full = n_full or n
    rng = np.random.default_rng(3)
    c = lambda *shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)  # noqa: E731
    fields = {
        "label": "Ni-3d",
        "solver_paramstr": "8 10",
        "dc_paramstr": "gap 0.5",
        "ham": c(n, n),
        "harm2corr": c(n_full, n),
        "corr2cf": c(n, n - 1),
        "sig": np.zeros((n, n, 3), complex),
        "sig_real": np.zeros((n, n, 5), complex),
        "sig_static": np.zeros((n, n), complex),
        "sig_dc": np.zeros((n, n), complex),
        "ctx%u4": np.zeros((n,) * 4, complex),
        "ctx%hyb": np.zeros((n, n, 5), complex),
        "ctx%wmsl": np.arange(3.0),
        "ctx%emesh": np.arange(5.0),
        "ctx%green_mu": 0.25,
        "ctx%eim": 0.1,
        "ctx%ttt": 0.01,
        "ctx%verbosity": 1,
        "ctx%dc_flag": 1,
    }
    fields.update(extra or {})
    return fields


def test_h_dft_is_ham_minus_mu_on_the_diagonal_and_ham_is_untouched():
    f = cluster()
    ham = f["ham"].copy()
    args = solve_arguments(f)
    np.testing.assert_allclose(args["h_dft"], ham - 0.25 * np.eye(N))
    np.testing.assert_array_equal(f["ham"], ham)
    assert args["h_dft"].flags.f_contiguous


def test_the_rotation_is_corr_to_harmonic_and_the_counts_come_from_the_shapes():
    f = cluster(n=3, n_full=5)  # a subset of a larger shell
    args = solve_arguments(f)
    np.testing.assert_allclose(args["corr_to_spherical_in"], f["harm2corr"].conj().T)
    assert args["corr_to_spherical_in"].flags.f_contiguous
    assert (args["n_orb"], args["n_rot_cols"], args["n_orb_full"]) == (3, 2, 5)
    assert (args["label"], args["solver_line"], args["dc_line"], args["dc_flag"]) == ("Ni-3d", "8 10", "gap 0.5", 1)
    assert (args["eim"], args["tau"], args["verbosity"]) == (0.1, 0.01, 1)


def test_operators_are_cut_to_the_cluster_and_zero_counts_as_absent():
    big = np.arange(36.0).reshape(6, 6) + 1j
    f = cluster(extra={"observables%ms": big, "observables%ml": np.zeros((N, N), complex)})
    ops = solve_arguments(f)["operator_arrays"]
    np.testing.assert_array_equal(ops["ms"], big[:N, :N])
    assert ops["ml"] is None and ops["msp"] is None and "dxyz" not in ops


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
    dxyz = solve_arguments(f)["operator_arrays"]["dxyz"]
    assert dxyz.shape == (N, N, 3)
    for i in range(3):
        np.testing.assert_array_equal(dxyz[:, :, i], d[i])


def test_a_spherical_dipole_is_converted_to_cartesian():
    rng = np.random.default_rng(5)
    dp, dm, dz = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)) for _ in range(3))
    f = cluster(extra=_obs([("S+", dz), ("(Dipole) Dp", dp), ("(Dipole) Dm", dm), ("(Dipole) Dz", dz)]))
    dxyz = solve_arguments(f)["operator_arrays"]["dxyz"]
    np.testing.assert_allclose(dxyz[:, :, 0], (dp + dm) / np.sqrt(2))
    np.testing.assert_allclose(dxyz[:, :, 1], (dp - dm) / (1j * np.sqrt(2)))
    np.testing.assert_allclose(dxyz[:, :, 2], dz)


def test_symmetries_are_stacked_with_their_time_reversal_flags():
    reps = [np.eye(N) * (i + 1) + 0j for i in range(3)]
    extra = {"symm%n": 3}
    for i, rep in enumerate(reps, start=1):
        extra[f"symm({i})%rep"] = rep
        extra[f"symm({i})%timerev"] = i == 2
    ops = solve_arguments(cluster(extra=extra))["operator_arrays"]
    assert ops["symm_rep"].shape == (N, N, 3) and ops["symm_rep"].flags.f_contiguous
    np.testing.assert_array_equal(ops["symm_rep"][:, :, 1], reps[1])
    np.testing.assert_array_equal(ops["symm_timerev"], [0, 1, 0])
    assert ops["symm_timerev"].dtype == np.intc


def test_one_operation_without_a_representation_means_no_symmetry_data():
    extra = {"symm%n": 2, "symm(1)%rep": np.eye(N, dtype=complex), "symm(1)%timerev": False}
    ops = solve_arguments(cluster(extra=extra))["operator_arrays"]
    assert "symm_rep" not in ops


def test_an_output_of_the_wrong_shape_is_refused_not_reshaped():
    f = cluster()
    f["sig_static"] = np.zeros((N + 1, N + 1), complex)
    with pytest.raises(ValueError, match="sig_static"):
        solve_arguments(f)


def test_a_missing_required_field_names_itself():
    f = cluster()
    del f["harm2corr"]
    with pytest.raises(KeyError, match="harm2corr"):
        solve_arguments(f)
