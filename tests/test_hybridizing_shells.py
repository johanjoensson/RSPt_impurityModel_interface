"""Which shells hybridize is measured as the bath fit measures it: on the fit window."""

import numpy as np
from rspt2spectra import FitOptions

from impurityModel_interface.problem import hybridizing_shells, shells_with_bath
from impurityModel_interface.solver_line import parse_solver_line

W = np.linspace(-3, 3, 601)


def _hyb(core_in_window, core_above_ef):
    """Shell 0 (2 orbitals) hybridizes with weight 1 per orbital in [-1, 0]; shell 1 (2 orbitals)
    carries ``core_in_window`` there and ``core_above_ef`` in [1, 2], above the Fermi level, where
    the fit of the occupied bath does not look."""
    hyb = np.zeros((len(W), 4, 4), complex)
    inside = (W >= -1) & (W <= 0)
    above = (W >= 1) & (W <= 2)
    for i in (0, 1):
        hyb[inside, i, i] = -1j
    for i in (2, 3):
        hyb[inside, i, i] = -1j * core_in_window
        hyb[above, i, i] = -1j * core_above_ef
    return hyb


SHELL = np.array([0, 0, 1, 1])


def test_a_core_shell_at_two_percent_in_the_window_does_not_hybridize():
    # Over the whole mesh the core shell would carry 2% + 50% of the valence weight.
    hyb = _hyb(0.02, 0.5)
    assert hybridizing_shells(hyb, W, SHELL, 2, FitOptions()) == {0}


def test_the_tolerance_is_the_fit_option():
    hyb = _hyb(0.02, 0.0)
    assert hybridizing_shells(hyb, W, SHELL, 2, FitOptions(hyb_tol=1e-3)) == {0, 1}
    line = parse_solver_line("8,6 5 hyb_tol 1d-3")
    assert line.fit.hyb_tol == 1e-3
    assert hybridizing_shells(hyb, W, SHELL, 2, line.fit) == {0, 1}


def test_shells_with_bath_reads_the_hopping():
    v = np.zeros((3, 4))
    v[0, 1] = 0.3
    assert shells_with_bath(v, SHELL, 2) == {0}
    assert shells_with_bath(np.zeros((0, 4)), SHELL, 2) == set()


def test_rounding_noise_in_the_rotation_does_not_count_as_a_bath():
    v = np.zeros((3, 4))
    v[:, 0] = 0.3
    v[:, 2] = 1e-15
    assert shells_with_bath(v, SHELL, 2) == {0}
