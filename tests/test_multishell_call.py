"""A cluster of two shells (here s + p, the smallest) through ``solve``: layout, ``N0`` per shell, archive.

The p shell has no hybridization, as a core shell has.  No ED runs: ``calc_selfenergy`` is a spy.
"""

import h5py as h5
import numpy as np
import pytest

pytest.importorskip("mpi4py")
from fake_cluster import cluster  # noqa: E402
from test_symm_prec_solve_wiring import N_IW, _install_spies, restore_stdout  # noqa: E402,F401

from impurityModel_interface import problem as problem_module  # noqa: E402
from impurityModel_interface import solve  # noqa: E402
from impurityModel_interface.cluster import ClusterCall, Shells  # noqa: E402

N = 8  # s: 2 spin-orbitals, p: 6
N_W = 200


def _fields(task=0, dc_line="amf"):
    w = np.linspace(-3, 3, N_W)
    hyb = np.zeros((N, N, N_W), complex)
    for i in range(2):
        hyb[i, i, :] = 0.16 / (w + 0.05j - 0.4)
    f = cluster(
        n=N,
        extra={
            "ham": np.diag([0.1, 0.1, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]).astype(complex),
            "harm2corr": np.eye(N, dtype=complex),
            "corr2cf": np.eye(N, dtype=complex),
            "sig": np.zeros((N, N, N_IW), complex),
            "sig_real": np.zeros((N, N, N_W), complex),
            "ctx%hyb": hyb,
            "ctx%wmsl": np.linspace(0.1, 1.0, N_IW),
            "ctx%emesh": w,
            "ctx%eim": 0.05,
            "ctx%green_mu": 0.0,
            "ctx%dc_flag": task,
            "ctx%verbosity": 0,
            "solver_paramstr": "2,2 2",
            "dc_paramstr": dc_line,
            "preoffset": np.array([0, 2, 8]),
            "offset": np.array([0, 8]),
        },
    )
    return f


def _run(shared_tmp_path, monkeypatch, fields):
    monkeypatch.chdir(shared_tmp_path)
    return solve.solve(ClusterCall.from_fields(fields))


def test_the_shells_are_read_from_the_layout():
    call = ClusterCall.from_fields(_fields())
    assert call.shells.shell_ls == (0, 1) and call.shells.preoffset == (0, 2, 8)


def test_a_layout_that_is_not_a_partition_is_refused():
    with pytest.raises(RuntimeError, match="preoffset"):
        Shells.from_layout(8, 8, 8, preoffset=[0, 3, 8], offset=[0, 8])  # a width-3 block is not a spin-doubled shell
    with pytest.raises(RuntimeError, match="partition"):
        Shells.from_layout(8, 8, 8, preoffset=[0, 2, 7], offset=[0, 8])


def test_the_model_gets_the_shells_and_the_archive_records_them(shared_tmp_path, monkeypatch, restore_stdout):
    seen = {}
    real = problem_module.ImpurityModel.from_blocks.__func__

    def from_blocks(cls, H_imp, V, H_bath, **kwargs):
        seen.update(kwargs)
        return real(cls, H_imp, V, H_bath, **kwargs)

    _install_spies(monkeypatch, sigma_static_cf=np.zeros((N, N)))
    monkeypatch.setattr(problem_module.ImpurityModel, "from_blocks", classmethod(from_blocks))
    assert _run(shared_tmp_path, monkeypatch, _fields()) == 0
    assert seen["shell_ls"] == (0, 1)
    with h5.File(shared_tmp_path / "impurityModel_data.h5", "r") as f:
        group = f[next(k for k in f if k != "last iteration")]
        assert list(group["Spherical shell l"][()]) == [0, 1]
        assert list(group["nominal occupation per shell"][()]) == [2, 2]
        assert group.attrs["nominal occupation"] == 4
    out = (shared_tmp_path / "impurityModel-Ni-3d.out").read_text()
    assert "shell 0 (l=0): 2 of 2 spin-orbitals, complete" in out
    assert "shell 1 (l=1): 6 of 6 spin-orbitals, complete, no hybridization" in out
    assert "2,2" in out


def test_n0_must_list_one_value_per_shell(shared_tmp_path, monkeypatch, restore_stdout):
    f = _fields()
    f["solver_paramstr"] = "4 2"
    with pytest.raises(RuntimeError, match="one occupation per shell"):
        _run(shared_tmp_path, monkeypatch, f)


def test_n0_cannot_exceed_the_orbitals_of_its_shell(shared_tmp_path, monkeypatch, restore_stdout):
    f = _fields()
    f["solver_paramstr"] = "3,1 2"
    with pytest.raises(RuntimeError, match="shell 0"):
        _run(shared_tmp_path, monkeypatch, f)


@pytest.mark.parametrize("scheme", sorted(solve.SINGLE_SHELL_SCHEMES))
def test_a_single_shell_double_counting_is_refused_for_several_shells(
    scheme, shared_tmp_path, monkeypatch, restore_stdout
):
    with pytest.raises(ValueError, match="one shell"):
        _run(shared_tmp_path, monkeypatch, _fields(task=1, dc_line=scheme))


def test_sigma_inf_is_not_refused_up_front(shared_tmp_path, monkeypatch, restore_stdout):
    """It is the multi-shell scheme; whatever it does next, it is not the request check that stops it."""
    try:
        _run(shared_tmp_path, monkeypatch, _fields(task=1, dc_line="sigma_inf"))
    except ValueError as error:
        assert "one shell" not in str(error)


def test_a_real_solve_keeps_the_core_shell_full_and_uncoupled(shared_tmp_path, monkeypatch, restore_stdout):
    """s (hybridizing) + p (core, N0 = 6): the real solver consumes ``shell_ls``."""
    f = _fields()
    f["solver_paramstr"] = "1,6 2"
    f["ham"] = np.diag([0.1, 0.1] + [-2.0] * 6).astype(complex)
    u4 = np.zeros((N,) * 4, complex)
    for i in range(N):
        for j in range(N):
            if i != j:
                u4[i, j, j, i] = 0.2
    f["ctx%u4"] = u4
    assert _run(shared_tmp_path, monkeypatch, f) == 0
    sigma = f["sig"]
    assert not np.abs(sigma[:2, 2:]).any() and not np.abs(sigma[2:, :2]).any()
    with h5.File(shared_tmp_path / "impurityModel_data.h5", "r") as ar:
        group = ar[next(k for k in ar if k != "last iteration")]
        rho = group["thermal_rho"][()]
    assert np.trace(rho[2:8, 2:8]).real == pytest.approx(6.0, abs=1e-8)


def test_the_spectra_task_names_the_valence_and_core_shell(shared_tmp_path, monkeypatch, restore_stdout):
    import impurityModel.api as api

    seen = {}

    def calc_spectroscopy(model, requests, basis, solver=None, **kwargs):
        seen.update(valence=model.valence_shell, core=model.core_shell, requests=requests)
        return {"E": 0.0}

    saved = []
    monkeypatch.setattr(api, "calc_spectroscopy", calc_spectroscopy)
    monkeypatch.setattr(api, "find_environment_file", lambda *a, **k: "impurityModel.toml")
    monkeypatch.setattr(api, "load_spectroscopy_requests", lambda path, comm=None: ["xas"])
    monkeypatch.setattr(api, "save_spectra", lambda result, name: saved.append(name))
    f = _fields()
    f["ctx%task"] = "spectra "
    assert _run(shared_tmp_path, monkeypatch, f) == 0
    assert seen["valence"] == 0 and seen["core"] == 1 and seen["requests"] == ["xas"]
    from mpi4py import MPI

    assert saved == (["spectra-Ni-3d.h5"] if MPI.COMM_WORLD.rank == 0 else [])


def test_two_shells_without_spin_polarisation_get_their_up_columns():
    """2p + 3d at nspmat=1: each shell's columns are [dn m | up m], the up rows copy the dn rows."""
    rng = np.random.default_rng(7)
    n_full, n_orb = 8, 16  # p: 3, d: 5 orbitals per spin; rows [dn 8 | up 8]
    pre = [0, 3, 8]
    shells = Shells.from_layout(n_orb, n_full, n_full, nspmat=1, preoffset=pre, offset=[0, n_orb])
    assert shells.shell_ls == (1, 2)
    corr_in = rng.normal(size=(n_orb, n_full)) + 0j
    out = shells.corr_to_spherical(corr_in, n_orb)
    assert out.shape == (n_orb, 2 * n_full)
    half = n_orb // 2
    # p block: columns 0..5, d block: columns 6..15
    np.testing.assert_array_equal(out[:half, 0:3], corr_in[:half, 0:3])
    np.testing.assert_array_equal(out[half:, 3:6], corr_in[:half, 0:3])
    np.testing.assert_array_equal(out[:half, 6:11], corr_in[:half, 3:8])
    np.testing.assert_array_equal(out[half:, 11:16], corr_in[:half, 3:8])
    assert not out[:half, 3:6].any() and not out[half:, 0:3].any()
