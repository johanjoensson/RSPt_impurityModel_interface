"""The bath fit is warm-started from the previous iteration's fit by default.

``fit_h0`` reads the most recent stored bath energies and hands them to rspt2spectra's
``fit_bath`` as a warm start; rspt2spectra keeps the warm fit unless a global search beats it by
more than ``warm_tol`` (the basin-stability itself is tested in rspt2spectra's
``test_warm_start.py``). This pins the wiring: what is seeded when, and the audit trail it leaves
in the archive.
"""

import numpy as np
import pytest

pytest.importorskip("mpi4py")
h5 = pytest.importorskip("h5py")
from mpi4py import MPI  # noqa: E402
from rspt2spectra import FitOptions  # noqa: E402

from impurityModel_interface.h0_fit import fit_h0  # noqa: E402

LABEL = "Ni 1"
W = np.linspace(-5, 5, 351)
EIM = 0.1
Z = W + 1j * (EIM * (1 + 0.5 * np.abs(W) ** 2))


def _hyb(s):
    eb = np.array([-2.0, -1.2, -0.4])
    v = np.array([0.7 * s, 0.7, 0.7 / s]).reshape(-1, 1)
    return np.einsum("bi,wb,bj->wij", v, 1 / (Z[:, None] - eb[None, :]), v)


def _fit(path, s, **options):
    hyb = _hyb(s)
    options = {"bath_geometry": "star", "fit_unocc": True, **options}
    return fit_h0(
        np.zeros((1, 1)),
        hyb,
        2,
        W,
        EIM,
        FitOptions(**options),
        u4=np.zeros((1, 1, 1, 1), dtype=complex),
        rot_to_spherical=np.eye(1, dtype=complex),
        label=LABEL,
        filename=path,
        verbose=False,
        comm=MPI.COMM_WORLD,
    )


def _fit_group(path, it):
    with h5.File(path, "r") as f:
        g = f[f"{LABEL} {it}/Bath fit"]
        return {key: g.attrs[key] for key in ("warm start", "cost warm", "cost global")}, int(f.attrs["last iteration"])


def test_first_iteration_is_cold_and_the_next_is_warm(tmp_path):
    path = tmp_path / "a.h5"
    _fit(path, 1.15)
    audit, last = _fit_group(path, 1)
    assert last == 1
    assert list(audit["warm start"]) == ["cold"]
    assert np.isnan(audit["cost warm"][0]) and np.isfinite(audit["cost global"][0])

    _fit(path, 1.154)  # a new iteration: new hybridization fingerprint
    audit, last = _fit_group(path, 2)
    assert last == 2
    assert list(audit["warm start"]) == ["kept"]
    assert np.isfinite(audit["cost warm"][0])


def test_cold_fit_ignores_the_previous_fit(tmp_path):
    path = tmp_path / "a.h5"
    _fit(path, 1.15)
    _fit(path, 1.154, warm_start=False)
    audit, _ = _fit_group(path, 2)
    assert list(audit["warm start"]) == ["cold"]


def test_freezing_still_holds_the_previous_energies(tmp_path):
    path = tmp_path / "a.h5"
    first = _fit(path, 1.15)
    second = _fit(path, 1.3, freeze_bath_energies=True)
    audit, _ = _fit_group(path, 2)
    assert list(audit["warm start"]) == ["frozen"]
    np.testing.assert_allclose(np.unique(np.diag(second.H_bath).round(10)), np.unique(np.diag(first.H_bath).round(10)))


def test_the_same_hybridization_reuses_the_stored_fit(tmp_path, capsys):
    # The DC pass and the self-energy pass of one iteration share one fit; no warm start runs.
    path = tmp_path / "a.h5"
    first = _fit(path, 1.15)
    capsys.readouterr()
    again = _fit(path, 1.15)
    assert "REUSED" in capsys.readouterr().out
    np.testing.assert_array_equal(again.H_bath, first.H_bath)
    with h5.File(path, "r") as f:
        assert int(f.attrs["last iteration"]) == 1
