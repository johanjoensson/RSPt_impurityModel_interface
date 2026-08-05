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
    mode, target, _alpha = _parse_dc_line(line)
    assert (mode, target) == expected


@pytest.mark.parametrize("line", ["gap", "gap 0.5", "occ 8.0", "1.2", "nominal"])
def test_alpha_is_orthogonal_to_the_criterion(line):
    """``alpha`` is stripped before the mode-specific parsing, so it composes with every
    criterion including the new one -- and its absence still means the default 0.5."""
    assert _parse_dc_line(line)[2] == 0.5
    mode, target, alpha = _parse_dc_line(f"{line} alpha 1.0")
    assert alpha == 1.0
    assert (mode, target) == _parse_dc_line(line)[:2]


@pytest.mark.parametrize("comment", ["!", "#"])
def test_comments_are_stripped(comment):
    assert _parse_dc_line(f"gap 0.5 {comment} centre it") == ("gap", 0.5, 0.5)


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
