"""The solver line grammar: ``N0 nBaths`` and the hand-over of the rest to the two option sets.

What each option means and how it is spelled is tested where it is defined
(``rspt2spectra/tests/test_fit_options.py``, impurityModel's ``test_options_registry.py``).
This pins what the wrapper adds: the two leading integers, the comment syntax, an error for any
word nobody owns, the RSPt-route defaults, and the rules that tie the fit to the solver.
"""

import pytest

pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: F401

from impurityModel_interface.solver_line import (
    format_settings_header,
    parse_solver_line,
    with_temperature,
)


def test_minimal_line():
    line = parse_solver_line("8 10")
    assert (line.nominal_occ, line.n_baths) == (8, 10)
    assert line.basis.nominal_occ == {0: 8}
    # peeled_linked_chain is the default geometry: chains are restricted, no star-only defaults
    assert line.fit.bath_geometry == "peeled_linked_chain"
    assert line.basis.chain_restrict is True
    assert line.basis.dN is None


def test_rspt_route_defaults():
    line = parse_solver_line("8 10")
    assert line.basis.occ_cutoff == 1e-6
    assert line.basis.mixed_valence is True
    assert line.solver.dense_cutoff == 1000


def test_the_route_defaults_yield_to_the_line():
    line = parse_solver_line("8 10 dense_cutoff 50")
    assert line.solver.dense_cutoff == 50


@pytest.mark.parametrize("comment_char", ["!", "#"])
def test_comments_are_stripped(comment_char):
    line = parse_solver_line(f"8 10 star {comment_char} this is not an option")
    assert line.fit.bath_geometry == "star"


def test_star_geometry_has_no_chains_and_a_wider_occupation_window():
    line = parse_solver_line("8 10 star")
    assert line.basis.chain_restrict is False
    assert line.basis.dN == 4


def test_the_window_of_a_star_can_be_set_explicitly():
    assert parse_solver_line("8 10 star dN 2").basis.dN == 2


def test_options_of_the_fit_and_of_the_solver_mix_in_any_order():
    line = parse_solver_line("8 10 gamma 0.02 star symm_prec 0.05")
    assert line.fit.gamma == 0.02
    assert line.fit.symm_prec == 0.05
    assert line.fit.bath_geometry == "star"


@pytest.mark.parametrize("text", ["", "8", "   ! only a comment"])
def test_fewer_than_two_words_is_an_error(text):
    with pytest.raises(ValueError, match="at least 2 arguments"):
        parse_solver_line(text)


@pytest.mark.parametrize("text", ["eight 10", "8 ten", "8 1.5"])
def test_the_leading_words_must_be_integers(text):
    with pytest.raises(ValueError, match="must be integers"):
        parse_solver_line(text)


def test_the_old_three_integer_grammar_is_an_unknown_option():
    with pytest.raises(ValueError, match="Unknown solver parameter"):
        parse_solver_line("8 10 3")


def test_an_unknown_word_is_an_error():
    with pytest.raises(ValueError, match=r"Unknown solver parameter.*not_an_option"):
        parse_solver_line("8 10 not_an_option")


def test_an_error_of_the_fit_options_passes_through():
    with pytest.raises(ValueError, match="symm_prec"):
        parse_solver_line("8 10 symm_prec")


def test_a_contradiction_between_fit_options_passes_through():
    with pytest.raises(ValueError, match="contradict"):
        parse_solver_line("8 10 cold_fit freeze_bath_energies")


@pytest.mark.parametrize(
    ("line", "merges"),
    [
        ("8 10", True),
        ("8 10 symm_prec 1e-9", True),  # below the default: only negligible couplings are dropped
        ("8 10 symm_prec 0.05", False),  # asks for truncation: the blocks are authoritative
        ("8 10 symmetry_prec 0.05", False),
    ],
)
def test_neglecting_couplings_stops_the_interaction_from_merging_blocks(line, merges):
    assert parse_solver_line(line).basis.interaction_merges_blocks is merges


def test_with_temperature_sets_it_on_the_basis_options_only():
    line = parse_solver_line("8 10 symm_prec 0.05")
    warm = with_temperature(line, 0.01)
    assert warm.basis.tau == 0.01
    assert warm.fit == line.fit
    assert warm.solver == line.solver
    assert warm.basis.interaction_merges_blocks is False


def test_settings_header_lists_the_line_and_the_run():
    header = format_settings_header(parse_solver_line("8 10 symm_prec 0.05"), tau=0.01, delta=0.1)
    rows = {row.split("|>")[0].strip(): row.split("|>")[1].strip() for row in header.splitlines()}
    assert rows["Nominal imp. occupation"] == "8"
    assert rows["Bath states per imp. orb."] == "10"
    assert rows["Bath geometry"] == "peeled_linked_chain"
    assert rows["Neglected coupling (relative)"] == "0.05"
    assert rows["Temperature tau"] == "0.01"
    assert rows["Real-axis broadening"] == "0.1"


def test_settings_header_aligns_the_columns():
    lines = format_settings_header(parse_solver_line("8 10")).splitlines()
    assert len({row.index("|>") for row in lines}) == 1
