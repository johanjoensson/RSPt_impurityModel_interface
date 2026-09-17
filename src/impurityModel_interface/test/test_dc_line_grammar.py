"""RSPt's 100-character double-counting line, and what each criterion on it means.

The line itself is fixed by RSPt -- it rewrites it for DC in {-4,-5,-14,-15} -- so the grammar can
only ever be *extended*, never restructured. That makes back-compatibility of the existing spellings
a standing property rather than a one-off check, which is why every one of them is pinned here
alongside the new ``gap``.
"""

import pytest

pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: F401,E402

from impurityModel_interface.lib import _parse_dc_line  # noqa: E402


@pytest.mark.parametrize(
    "line, expected",
    [
        # gap: Karolak et al.'s insulator prescription (arXiv:1004.4569). The bare word means
        # "centre the gap on the Fermi level", which is the prescription as stated.
        ("gap", ("gap", 0.0)),
        ("gap 0.5", ("gap", 0.5)),
        ("gap -0.3", ("gap", -0.3)),
        ("GAP 1.0", ("gap", 1.0)),
        # Everything that already worked, still working.
        ("occ", ("occupation", None)),
        ("occ 8.0", ("occupation", 8.0)),
        ("occupation 8.0", ("occupation", 8.0)),
        ("1.2", ("peak", 1.2)),
        ("-1.5", ("peak", -1.5)),
        ("fll", ("fll", None)),
        ("amf", ("amf", None)),
        ("sigma_inf", ("sigma_inf", None)),
        ("nominal", ("nominal", None)),
    ],
)
def test_the_criterion_and_its_target(line, expected):
    mode, target, _alpha, _gs_manifold, _de2 = _parse_dc_line(line)
    assert (mode, target) == expected


@pytest.mark.parametrize("line", ["gap", "gap 0.5", "occ 8.0", "1.2", "nominal"])
def test_alpha_is_orthogonal_to_the_criterion(line):
    """``alpha`` is stripped before the mode-specific parsing, so it composes with every
    criterion including the new one -- and its absence still means the default 0.5."""
    assert _parse_dc_line(line)[2] == 0.5
    mode, target, alpha, _gs_manifold, _de2 = _parse_dc_line(f"{line} alpha 1.0")
    assert alpha == 1.0
    assert (mode, target) == _parse_dc_line(line)[:2]


@pytest.mark.parametrize("comment", ["!", "#"])
def test_comments_are_stripped(comment):
    assert _parse_dc_line(f"gap 0.5 {comment} centre it") == ("gap", 0.5, 0.5, False, None)


def test_a_bare_number_is_still_a_peak_position_not_a_gap_offset():
    """The un-prefixed number has meant "peak position" since before ``gap`` existed, and RSPt
    input files in the wild carry it. Adding a keyword must not have moved it."""
    assert _parse_dc_line("2.5")[:2] == ("peak", 2.5)


def test_a_gap_line_with_too_many_arguments_is_rejected():
    with pytest.raises(AssertionError, match="at most 1 argument"):
        _parse_dc_line("gap 0.5 0.7")


@pytest.mark.parametrize("line", ["", "   ", "alpha 1.0", "! only a comment"])
def test_a_line_naming_no_criterion_is_rejected_cleanly(line):
    """The degenerate cases the extraction out of ``run_impmod_ed`` had to preserve.

    ``alpha`` is stripped from the token list before the criterion is read, so ``alpha 1.0`` alone
    leaves that list *empty* -- and an empty list must reach the same assertion an unrecognised
    keyword does, not an ``IndexError`` from indexing past the end of it.
    """
    with pytest.raises(AssertionError):
        _parse_dc_line(line)


def test_alpha_without_a_value_says_so():
    with pytest.raises(AssertionError, match="needs a value"):
        _parse_dc_line("gap alpha")


def test_an_unknown_keyword_names_the_criteria_it_could_have_been():
    """A typo must not be silently read as a peak position -- ``float()`` raises, and the
    assertion above it lists what was available, ``gap`` included."""
    with pytest.raises((AssertionError, ValueError)):
        _parse_dc_line("gapp 0.5")


# ---- ground_state_manifold ---------------------------------------------------------------


@pytest.mark.parametrize("line", ["gap", "gap 0.5", "gap -0.3", "1.2", "-1.5"])
def test_ground_state_manifold_is_orthogonal_to_the_criterion(line):
    """Like ``alpha``, it is stripped before the mode-specific parsing, so it composes with every
    criterion that accepts it and leaves the criterion and its target untouched."""
    assert _parse_dc_line(line)[3] is False
    mode, target, alpha, gs_manifold, _de2 = _parse_dc_line(f"{line} ground_state_manifold")
    assert gs_manifold is True
    assert (mode, target, alpha) == _parse_dc_line(line)[:3]


def test_ground_state_manifold_composes_with_alpha():
    """Both are stripped by independent passes, so neither consumes the other's tokens and the
    order they appear in must not matter."""
    assert _parse_dc_line("gap 0.5 alpha 0.15 ground_state_manifold") == ("gap", 0.5, 0.15, True, None)
    assert _parse_dc_line("gap 0.5 ground_state_manifold alpha 0.15") == ("gap", 0.5, 0.15, True, None)


@pytest.mark.parametrize("line", ["GROUND_STATE_MANIFOLD", "Ground_State_Manifold"])
def test_ground_state_manifold_is_case_insensitive(line):
    """Every other keyword on this line is matched case-insensitively; RSPt input in the wild is
    inconsistently cased."""
    assert _parse_dc_line(f"gap {line}")[3] is True


@pytest.mark.parametrize("scheme", ["fll", "amf", "sigma_inf", "nominal"])
def test_a_static_scheme_rejects_ground_state_manifold(scheme):
    """The static schemes run no ground-state solve, so there is no manifold to narrow. Rejected
    rather than ignored: a flag that reports as set and changes nothing is the failure mode a
    misnamed knob already produces once in this stack."""
    with pytest.raises(AssertionError, match="runs no ground-state solve"):
        _parse_dc_line(f"{scheme} ground_state_manifold")


@pytest.mark.parametrize("line", ["occ", "occ 8.0", "occupation 8.0"])
def test_the_occupation_criterion_rejects_ground_state_manifold(line):
    """Excluded on different grounds from the static schemes, and the message has to say which.

    ``fixed_occupation_dc`` runs plenty of solves -- but its observable *is* the thermal impurity
    occupation, so narrowing the manifold would change the criterion rather than the cost of
    evaluating it. It accordingly does not accept the argument at all (it runs through
    ``_OccupationContext``/``solve_ground_state``, not ``_SectorContext.sector_solve``), which is
    why this has to fail here rather than reach a ``TypeError`` at the call site.
    """
    with pytest.raises(AssertionError, match="would change the criterion"):
        _parse_dc_line(f"{line} ground_state_manifold")


def test_ground_state_manifold_is_off_by_default_on_every_criterion():
    """It is opt-in by design: it also switches the reported impurity occupation from the thermal
    average to the ground state's, and those differ on SrMnO3 (``occupation_spread`` up to 0.065).
    A default flip would move a reported ``mu`` resolution with no line in any input file to
    explain it."""
    for line in ("gap", "gap 0.5", "occ 8.0", "1.2", "fll", "amf", "sigma_inf", "nominal"):
        assert _parse_dc_line(line)[3] is False, line


# --------------------------------------------------------------------------- #
# de2_min: the PT2 admission floor of the charge-sector solves
# --------------------------------------------------------------------------- #


def test_de2_min_defaults_to_none_so_the_solver_keeps_its_own_default():
    """Absent, the line must say nothing about the threshold -- adding this token must not move
    any existing answer."""
    assert _parse_dc_line("gap 0.5")[4] is None
    assert _parse_dc_line("2.5")[4] is None
    assert _parse_dc_line("occ 8")[4] is None


@pytest.mark.parametrize("line", ["gap 0.5", "2.5", "occ 8"])
def test_de2_min_parses_on_every_criterion(line):
    mode, target, alpha, gs, de2 = _parse_dc_line(f"{line} de2_min 1e-6")
    assert de2 == 1e-6
    # and stripping it leaves the rest of the grammar untouched
    assert (mode, target, alpha, gs) == _parse_dc_line(line)[:4]


def test_de2_min_composes_with_alpha_and_ground_state_manifold_in_any_order():
    expected = ("gap", 0.5, 0.15, True, 1e-6)
    assert _parse_dc_line("gap 0.5 alpha 0.15 ground_state_manifold de2_min 1e-6") == expected
    assert _parse_dc_line("gap 0.5 de2_min 1e-6 ground_state_manifold alpha 0.15") == expected
    assert _parse_dc_line("gap 0.5 ground_state_manifold de2_min 1e-6 alpha 0.15") == expected


def test_de2_min_needs_a_value():
    with pytest.raises(AssertionError, match="de2_min"):
        _parse_dc_line("gap 0.5 de2_min")


def test_de2_min_must_be_positive():
    """Zero would admit every candidate the expansion can reach, which is not a loosening but an
    unbounded one; a negative value is meaningless."""
    for bad in ("0", "-1e-6"):
        with pytest.raises(AssertionError, match="positive"):
            _parse_dc_line(f"gap 0.5 de2_min {bad}")


def test_de2_min_is_stripped_before_the_criterion_argument_count():
    """The token must not be mistaken for a criterion argument -- `gap` takes at most one, and
    `gap 0.5 de2_min 1e-6` must not read as `gap` with two."""
    assert _parse_dc_line("gap de2_min 1e-6")[:2] == ("gap", 0.0)
    assert _parse_dc_line("gap 0.5 de2_min 1e-6")[:2] == ("gap", 0.5)


@pytest.mark.parametrize("scheme", ["fll", "amf", "sigma_inf", "nominal"])
def test_de2_min_is_rejected_on_the_static_schemes(scheme):
    """A static scheme runs no charge-sector solve, so the token could only be ignored. Reject it
    loudly, the same discipline the line already applies to `ground_state_manifold`."""
    with pytest.raises(AssertionError, match="de2_min"):
        _parse_dc_line(f"{scheme} de2_min 1e-6")


def test_de2_min_is_accepted_on_occ_even_though_it_only_acts_on_gap_and_peak():
    """`occ` solves, so the token is meaningful grammar there; it is `fixed_occupation_dc` that
    routes through the production ground state rather than per-sector CIPSI, which is a solver
    decision rather than a grammar one."""
    assert _parse_dc_line("occ 8 de2_min 1e-6")[4] == 1e-6
