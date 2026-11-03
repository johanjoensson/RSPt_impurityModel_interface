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
    mode, target, _alpha, _de2, _e_pt2 = _parse_dc_line(line)
    assert (mode, target) == expected


@pytest.mark.parametrize("line", ["gap", "gap 0.5", "occ 8.0", "1.2", "nominal"])
def test_alpha_is_orthogonal_to_the_criterion(line):
    """``alpha`` is stripped before the mode-specific parsing, so it composes with every
    criterion including the new one -- and its absence still means the default 0.5."""
    assert _parse_dc_line(line)[2] == 0.5
    mode, target, alpha, _de2, _e_pt2 = _parse_dc_line(f"{line} alpha 1.0")
    assert alpha == 1.0
    assert (mode, target) == _parse_dc_line(line)[:2]


@pytest.mark.parametrize("comment", ["!", "#"])
def test_comments_are_stripped(comment):
    assert _parse_dc_line(f"gap 0.5 {comment} centre it") == ("gap", 0.5, 0.5, None, None)


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


# --------------------------------------------------------------------------- #
# de2_min: the PT2 admission floor of the charge-sector solves
# --------------------------------------------------------------------------- #


def test_de2_min_defaults_to_none_so_the_solver_keeps_its_own_default():
    """Absent, the line must say nothing about the threshold -- adding this token must not move
    any existing answer."""
    assert _parse_dc_line("gap 0.5")[3] is None
    assert _parse_dc_line("2.5")[3] is None
    assert _parse_dc_line("occ 8")[3] is None


@pytest.mark.parametrize("line", ["gap 0.5", "2.5", "occ 8"])
def test_de2_min_parses_on_every_criterion(line):
    mode, target, alpha, de2, _e_pt2 = _parse_dc_line(f"{line} de2_min 1e-6")
    assert de2 == 1e-6
    # and stripping it leaves the rest of the grammar untouched
    assert (mode, target, alpha) == _parse_dc_line(line)[:3]


def test_de2_min_composes_with_alpha_and_ground_state_manifold_in_any_order():
    expected = ("gap", 0.5, 0.15, 1e-6, None)
    assert _parse_dc_line("gap 0.5 alpha 0.15 de2_min 1e-6") == expected
    assert _parse_dc_line("gap 0.5 de2_min 1e-6 alpha 0.15") == expected
    assert _parse_dc_line("gap 0.5 de2_min 1e-6 alpha 0.15") == expected


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
    loudly, the same discipline the line applies to every modifier that would do nothing."""
    with pytest.raises(AssertionError, match="de2_min"):
        _parse_dc_line(f"{scheme} de2_min 1e-6")


def test_de2_min_is_accepted_on_occ_even_though_it_only_acts_on_gap_and_peak():
    """`occ` solves, so the token is meaningful grammar there; it is `fixed_occupation_dc` that
    routes through the production ground state rather than per-sector CIPSI, which is a solver
    decision rather than a grammar one."""
    assert _parse_dc_line("occ 8 de2_min 1e-6")[3] == 1e-6


# --------------------------------------------------------------------------- #
# e_pt2: the residual PT2 energy the gap/peak charge-sector solves converge to
# --------------------------------------------------------------------------- #


def test_e_pt2_defaults_to_none_so_the_solves_inherit_the_solver_line():
    for line in ("gap 0.5", "2.5", "occ 8", "fll"):
        assert _parse_dc_line(line)[4] is None


@pytest.mark.parametrize("line", ["gap 0.5", "2.5", "gap"])
@pytest.mark.parametrize("spelling", ["e_pt2", "E_PT2", "e_pt2_tol"])
def test_e_pt2_parses_on_gap_and_peak_and_leaves_the_rest_alone(line, spelling):
    parsed = _parse_dc_line(f"{line} {spelling} 1e-5")
    assert parsed[4] == 1e-5
    assert parsed[:4] == _parse_dc_line(line)[:4]


def test_e_pt2_composes_with_every_other_modifier_in_any_order():
    expected = ("gap", 0.5, 0.15, 1e-6, 1e-5)
    assert _parse_dc_line("gap 0.5 alpha 0.15 de2_min 1e-6 e_pt2 1e-5") == expected
    assert _parse_dc_line("gap 0.5 e_pt2 1e-5 de2_min 1e-6 alpha 0.15") == expected
    assert _parse_dc_line("gap e_pt2 1e-5 0.5 alpha 0.15 de2_min 1e-6") == expected


def test_e_pt2_needs_a_positive_value():
    with pytest.raises(AssertionError, match="needs a value"):
        _parse_dc_line("gap 0.5 e_pt2")
    for bad in ("0", "-1e-8"):
        with pytest.raises(AssertionError, match="positive"):
            _parse_dc_line(f"gap 0.5 e_pt2 {bad}")


@pytest.mark.parametrize("line", ["occ 8", "occ", "fll", "amf", "sigma_inf", "nominal"])
def test_e_pt2_is_rejected_where_it_would_do_nothing(line):
    """`occ` converges on the production path, to the solver line's e_pt2; the static schemes
    solve nothing. Either way the token could only be ignored, so it is refused."""
    with pytest.raises(AssertionError, match="e_pt2"):
        _parse_dc_line(f"{line} e_pt2 1e-5")


def test_e_pt2_fits_rspts_100_character_line():
    line = "gap 0.5 alpha 0.15 de2_min 1e-6 e_pt2 1e-5"
    assert len(line) <= 100
    assert _parse_dc_line(line.ljust(100))[4] == 1e-5
