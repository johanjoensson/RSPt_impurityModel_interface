"""The double-counting damping anchor must be *our* previous answer.

RSPt does not carry impurityModel's double counting between CSC iterations: it zeroes
``sig_dc`` at the top of every ``double_counting`` call (``green_double_counting.F90:73``) and
refills it with its own FLL/AMF potential, and our answer is never written back into
``solver_wisdom``. Damping toward the incoming ``sig_dc`` therefore returned
``alocal + alpha*mu`` -- a double counting that does not satisfy the criterion the search just
converged -- on every iteration.

``_previous_damped_dc`` closes that by reading the last answer back out of the HDF5 archive.
"""

import numpy as np
import pytest

pytest.importorskip("mpi4py")
pytest.importorskip("h5py")
import h5py as h5  # noqa: E402
from mpi4py import MPI  # noqa: F401,E402

from impurityModel_interface.lib import (  # noqa: E402
    _double_counting_sector,
    _previous_damped_dc,
    _split_total_over_groups,
    h5_write_dataset,
)

LABEL = "Ni 1"


def _archive(path, iterations):
    """Write ``{iteration: dc_matrix_or_None}`` into a fresh archive at ``path``."""
    with h5.File(path, "w") as f:
        f.attrs["last iteration"] = max(iterations) + 1
        for it, dc in iterations.items():
            group = f.create_group(f"{LABEL} {it}")
            if dc is not None:
                h5_write_dataset(group, "DC damped", dc)


def test_returns_none_without_an_archive(tmp_path):
    """First call of a fresh run: nothing to damp against, so the caller returns the
    converged value undamped rather than mixing it with RSPt's unrelated guess."""
    assert _previous_damped_dc(tmp_path / "missing.h5", LABEL, None) is None


def test_returns_none_on_the_first_iteration(tmp_path):
    path = tmp_path / "archive.h5"
    with h5.File(path, "w") as f:
        f.attrs["last iteration"] = 1
    assert _previous_damped_dc(path, LABEL, None) is None


def test_returns_the_previous_iterations_answer(tmp_path):
    path = tmp_path / "archive.h5"
    dc_prev = np.diag([24.1, 24.1]).astype(complex)
    _archive(path, {1: np.diag([23.0, 23.0]).astype(complex), 2: dc_prev})

    np.testing.assert_allclose(_previous_damped_dc(path, LABEL, None), dc_prev)


def test_skips_an_iteration_that_recorded_no_dc(tmp_path):
    """An iteration where the search was unreachable (or the DC pass never ran) writes no
    ``DC damped``; the chain must reach past it instead of falling back to no damping."""
    path = tmp_path / "archive.h5"
    dc_older = np.diag([22.5, 22.5]).astype(complex)
    _archive(path, {1: dc_older, 2: None, 3: None})

    np.testing.assert_allclose(_previous_damped_dc(path, LABEL, None), dc_older)


def test_ignores_other_clusters(tmp_path):
    path = tmp_path / "archive.h5"
    with h5.File(path, "w") as f:
        f.attrs["last iteration"] = 3
        h5_write_dataset(f.create_group("Fe 2 1"), "DC damped", np.diag([9.0, 9.0]).astype(complex))
        h5_write_dataset(f.create_group(f"{LABEL} 2"), "DC damped", np.diag([24.1, 24.1]).astype(complex))

    np.testing.assert_allclose(_previous_damped_dc(path, LABEL, None), np.diag([24.1, 24.1]))


def test_damping_formula_reproduces_the_criterion_on_the_first_iteration(tmp_path):
    """The property the whole change exists for: with no previous answer, what RSPt receives
    is exactly what the criterion converged to."""
    dc_converged = np.diag([25.3, 25.3]).astype(complex)
    previous = _previous_damped_dc(tmp_path / "none.h5", LABEL, None)

    damped = dc_converged if previous is None else previous + 0.5 * (dc_converged - previous)

    np.testing.assert_allclose(damped, dc_converged)


def test_no_sector_recorded_yields_none(tmp_path):
    """A static DC scheme, an unreachable target, or a plain self-energy run records no sector;
    the following solve then just uses RSPt's nominal, as before."""
    path = tmp_path / "archive.h5"
    with h5.File(path, "w") as f:
        f.attrs["last iteration"] = 2
        f.create_group(f"{LABEL} 2")
    assert _double_counting_sector(path, LABEL, None) is None


def test_the_sector_is_read_back_for_the_current_iteration(tmp_path):
    """The DC pass and the self-energy pass share one iteration group, so the sector written by
    the first is what the second reads -- not the previous iteration's."""
    path = tmp_path / "archive.h5"
    with h5.File(path, "w") as f:
        f.attrs["last iteration"] = 3
        f.create_group(f"{LABEL} 2").attrs["DC ground state sector"] = 7
        f.create_group(f"{LABEL} 3").attrs["DC ground state sector"] = 9
    assert _double_counting_sector(path, LABEL, None) == 9


@pytest.mark.parametrize(
    "total, nominal, expected_total",
    [
        (9, {0: 8}, 9),
        (7, {0: 8}, 7),
        (9, {0: 4, 1: 4}, 9),
        (6, {0: 4, 1: 4}, 6),
        (0, {0: 2, 1: 2}, 0),
    ],
)
def test_the_sector_split_preserves_the_total(total, nominal, expected_total):
    """Whatever the group layout, the seed must carry the sector the DC search found."""
    occ = _split_total_over_groups(total, nominal)
    assert sum(occ.values()) == expected_total
    assert set(occ) == set(nominal)
    assert all(v >= 0 for v in occ.values()), occ


def test_the_sector_split_keeps_the_incoming_shape():
    """Only the difference moves: a split impurity stays near the filling RSPt asked for."""
    occ = _split_total_over_groups(9, {0: 4, 1: 4})
    assert sum(occ.values()) == 9
    # One group takes the extra electron; neither is re-derived from scratch.
    assert sorted(occ.values()) == [4, 5]
