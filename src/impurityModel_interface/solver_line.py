"""The solver line RSPt passes: ``N0 nBaths [options]``.

The wrapper owns only the first two words.  Everything after them is an option of the fit
(:func:`rspt2spectra.parse_fit_tokens`) or of the solver
(:func:`impurityModel.api.parse_option_tokens`); a word neither knows is an error.
"""

from dataclasses import dataclass, replace

from impurityModel.api import BasisOptions, SolverOptions, describe_options, parse_option_tokens
from rspt2spectra import FitOptions, parse_fit_tokens
from rspt2spectra.fit_options import DEFAULT_SYMM_PREC

#: Defaults the RSPt route sets where the solver's own differ.
BASIS_DEFAULTS = {"occ_cutoff": 1e-6, "mixed_valence": True}
SOLVER_DEFAULTS = {"dense_cutoff": 1000}


@dataclass(frozen=True)
class SolverLine:
    """A parsed solver line."""

    #: Nominal impurity occupation.
    nominal_occ: int
    #: Bath states per impurity orbital.
    n_baths: int
    fit: FitOptions
    basis: BasisOptions
    solver: SolverOptions


def parse_solver_line(solver_line):
    """Parse the solver line (``N0 nBaths [options]``, ``!`` or ``#`` start a comment).

    Raises
    ------
    ValueError
        Fewer than two words, a first two words that are not integers, an option without a value or
        out of range, or a word that is no option.
    """
    words = solver_line.split("!")[0].split("#")[0].split()
    if len(words) < 2:
        raise ValueError(f"The impurityModel ED solver requires at least 2 arguments; N0 nBaths. Got {solver_line!r}")
    try:
        nominal_occ, n_baths = int(words[0]), int(words[1])
    except ValueError:
        raise ValueError(f"N0 and nBaths must be integers, got {words[0]!r} {words[1]!r}") from None
    fit, rest = parse_fit_tokens(words[2:])
    basis_kwargs, solver_kwargs, unknown = parse_option_tokens(rest)
    if unknown:
        raise ValueError(f"Unknown solver parameter(s) {unknown}.\n--->Solver line {solver_line.strip()!r}")
    basis_kwargs = {**BASIS_DEFAULTS, **basis_kwargs}
    solver_kwargs = {**SOLVER_DEFAULTS, **solver_kwargs}
    if fit.bath_geometry == "star":
        # A star bath has no chains to restrict, and its occupation window is wider.
        basis_kwargs["chain_restrict"] = False
        basis_kwargs.setdefault("dN", 4)
    # Neglecting couplings makes the one-body blocks authoritative: the solver must then not merge
    # them back because the interaction couples them.
    basis_kwargs["interaction_merges_blocks"] = not fit.symm_prec > DEFAULT_SYMM_PREC
    basis = BasisOptions(nominal_occ={0: nominal_occ}, **basis_kwargs)
    try:
        solver = SolverOptions(**solver_kwargs)
    except ValueError as error:
        raise ValueError(f"Invalid Green's function setting on the solver line: {error}") from error
    return SolverLine(nominal_occ, n_baths, fit, basis, solver)


def format_settings_header(line, tau=None, delta=None):
    """The output-file header listing every setting of this run, from the options' own descriptions."""
    rows = [("Nominal imp. occupation", line.nominal_occ), ("Bath states per imp. orb.", line.n_baths)]
    rows += line.fit.describe()
    rows += describe_options(line.basis, line.solver)
    if tau is not None:
        rows.append(("Temperature tau", tau))
    if delta is not None:
        rows.append(("Real-axis broadening", delta))
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{width}} |> {value}" for label, value in rows) + "\n"


def with_temperature(line, tau):
    """Return ``line`` with the external temperature set on its basis options."""
    return replace(line, basis=replace(line.basis, tau=tau))
