"""The problem both kinds of call share: RSPt's data rotated into the fitting basis, the fitted h0, the solver model.

:func:`prepare_problem` is the one place the bath is fitted; :class:`Problem` carries the result to
:func:`solve.run_double_counting` / :func:`solve.run_selfenergy` and to the archive.
"""

from dataclasses import dataclass, replace

import numpy as np
from impurityModel.ed.observables import shell_index_of_orbitals
from impurityModel.api import (
    CapLedger,
    ImpurityModel,
    LocalOperators,
    discretized_impurity_occupation,
    report_continuum_reference,
    save_Greens_function,
    site_frames,
)
from rspt2spectra.block_structure import project_to_blocks
from rspt2spectra.h0 import bath_hybridization, fit_window
from rspt2spectra.hyb_fit import hybridization_weight
from rspt2spectra.weight_functions import weight_functions
from rspt2spectra.utils import rotate_4index_U, rotate_Greens_function, rotate_matrix

from . import archive
from .h0_fit import fit_h0
from .solver_line import format_settings_header, with_temperature


@dataclass
class Problem:
    """Everything the two kinds of call share, in the fitting basis."""

    model: ImpurityModel
    basis: object
    solver: object
    #: Block partition of the fit; the double counting carries no coupling between blocks.
    blocks: list
    #: Rotation from the fitting basis (the solver's) to RSPt's correlated basis: columns are the
    #: fitting orbitals expressed in RSPt's orbitals.
    corr_to_solver: np.ndarray
    #: RSPt's incoming double counting, in the fitting basis.
    sig_dc: np.ndarray
    cap_ledger: CapLedger
    #: The parsed solver line, with the basis the solve actually uses.
    line: object
    fitted: object
    #: The fitted one-body Hamiltonian of impurity and bath, in the fitting basis.
    H_solver: np.ndarray
    #: RSPt's inputs in the fitting basis (``hyb`` has the frequency first).
    h_dft: np.ndarray
    u4: np.ndarray
    hyb: np.ndarray
    v: np.ndarray
    #: ``(n_orb, n_orb_full)``-style map from the fitting basis to the spherical harmonics.
    rot_to_spherical: np.ndarray
    #: Indices of the shells that hybridize with the bath (empty when the shells are not known).
    hybridizing: frozenset = frozenset()
    #: Number of shells RSPt sent.
    n_shells: int = 1

    def to_rspt(self, matrix):
        """A matrix of the fitting basis in RSPt's correlated basis."""
        return rotate_matrix(matrix, np.conj(self.corr_to_solver.T))

    def to_rspt_green(self, green):
        """A Green's function (frequency first) of the fitting basis in RSPt's correlated basis."""
        return rotate_Greens_function(green, np.conj(self.corr_to_solver.T))


def spectra_shells(hybridizing, n_shells):
    """``(valence_shell, core_shell)`` for ``calc_spectroscopy``: the valence shell is the one that hybridizes.

    With two shells the other is the core shell.  Raises where the valence shell is ambiguous.
    """
    if n_shells == 1:
        return 0, None
    if len(hybridizing) != 1:
        raise RuntimeError(
            f"Spectra need exactly one shell with a hybridization (the valence shell); "
            f"found {sorted(hybridizing)} among {n_shells} shells."
        )
    (valence,) = hybridizing
    others = [a for a in range(n_shells) if a != valence]
    return valence, (others[0] if len(others) == 1 else None)


def split_total_over_groups(total, nominal_occ, movable=None):
    """Distribute an impurity charge ``total`` over the groups of ``nominal_occ``.

    Keeps the incoming split as the shape and moves only the difference, so a multi-group
    impurity (an ``eg``/``t2g`` split) stays near the filling RSPt asked for instead of being
    re-derived from scratch. The groups redistribute freely at fixed total inside the solver
    anyway (``basis_generation.generate_initial_basis``), so this only has to be a sensible seed.

    ``movable`` limits the groups that may take or give charge (a core shell without hybridization
    cannot); ``None`` means all of them.
    """
    occ = dict(nominal_occ)
    delta = int(total) - sum(occ.values())
    keys = sorted(occ if movable is None else (key for key in occ if key in movable))
    step = 1 if delta > 0 else -1
    while delta != 0 and keys:
        moved = False
        for key in keys:
            if delta == 0:
                break
            if occ[key] + step >= 0:
                occ[key] += step
                delta -= step
                moved = True
        if not moved:
            break
    return occ


def _report_operator_check(operators, rot_to_spherical, shell_ls, comm):
    """Print the site frame RSPt's operators define and how well they fit the spherical rebuild.

    RSPt's ``ml``/``mlp`` are along the global (crystal) axes; the spherical rebuild is in the frame
    ``rot_to_spherical`` defines. A rotated site therefore shows as a rotation with a roundoff residual, not as
    an error; a large residual means the basis is not what ``rot_to_spherical`` says. The spin operators are
    compared directly (RSPt keeps spin in the global frame).
    """
    if comm.rank != 0 or operators is None:
        return
    # Only a rotation that spans whole spin-resolved shells has a spherical rebuild to compare with.
    if rot_to_spherical.shape[0] != rot_to_spherical.shape[1] or rot_to_spherical.shape[1] != sum(
        2 * (2 * l + 1) for l in shell_ls
    ):
        print("RSPt operators: no complete spin-resolved shells, so no spherical rebuild to compare with.", flush=True)
        return
    spherical = LocalOperators.from_spherical(rot_to_spherical, tuple(shell_ls))
    parts = []
    for name in ("sz", "s_plus"):
        got, ref = getattr(operators, name), getattr(spherical, name)
        if got is not None and ref is not None and got.shape == ref.shape:
            parts.append(f"{name} {np.max(np.abs(got - ref)):.2e}")
    if parts:
        print("RSPt spin operators vs spherical rebuild (max abs deviation): " + ", ".join(parts), flush=True)
    for a, frame in enumerate(site_frames(operators, rot_to_spherical, tuple(shell_ls))):
        if frame is None:
            continue
        warn = "" if frame.residual < 1e-6 else "  WARNING: L is not a rotation of the spherical L in this basis"
        print(f"RSPt site frame (shell {a}, l={shell_ls[a]}): {frame.describe()}{warn}", flush=True)


def shell_orbital_counts(corr_to_spherical, shell_ls):
    """Correlated orbitals per shell (RSPt's basis): the shell holding each orbital's weight, counted."""
    if not shell_ls or corr_to_spherical.shape[1] != sum(2 * (2 * l + 1) for l in shell_ls):
        return None
    index = shell_index_of_orbitals(corr_to_spherical, shell_ls)
    return [int(np.sum(index == a)) for a in range(len(shell_ls))], index


def hybridizing_shells(hyb, w, orbital_shell, n_shells, options):
    """Shells whose hybridization weight is above ``options.hyb_tol`` of the strongest one.

    ``hyb`` has the frequency first and is in RSPt's basis.  The weight is measured the way the
    bath fit allocates its states (:func:`rspt2spectra.hyb_fit.hybridization_weight`): on the fit
    window (cut at the Fermi level unless ``fit_unocc``), with the fit's weight function.  Measured
    over the whole mesh a core shell can look strongly hybridizing from its weight above the Fermi
    level (Ni 2p in NiO: 18% of the 3d weight instead of 1.4%).
    """
    lo, hi = fit_window(w, hyb, not options.fit_unocc)
    mask = (lo <= w) & (w < hi)
    weight_fun = weight_functions[options.weight_function](options.weight_w0, options.weight)
    weights = np.zeros(n_shells)
    for a in range(n_shells):
        idx = np.flatnonzero(orbital_shell == a)
        weights[a] = max(hybridization_weight(hyb[mask][:, idx][:, :, idx], w[mask], weight_fun), 0.0)
    return {a for a in range(n_shells) if weights[a] > options.hyb_tol * weights.max()}


def shells_with_bath(v, orbital_shell, n_shells):
    """Shells that at least one bath state couples to; ``v`` is (n_bath, n_orb) in RSPt's basis."""
    scale = np.max(np.abs(v), initial=0.0)
    if scale == 0:
        return set()
    return {a for a in range(n_shells) if np.max(np.abs(v[:, orbital_shell == a]), initial=0.0) > 1e-8 * scale}


def _check_shells(call, line):
    """Check ``N0`` against the shells RSPt sent.  Returns ``(counts, orbital_shell)`` or ``(None, None)``."""
    shell_ls = call.shells.shell_ls
    per_shell = line.nominal_per_shell
    if len(shell_ls) > 1 and len(per_shell) != len(shell_ls):
        raise RuntimeError(
            f"The cluster has {len(shell_ls)} correlated shells (l = {list(shell_ls)}); "
            f"N0 must list one occupation per shell, e.g. {','.join('N' for _ in shell_ls)}, got {len(per_shell)}."
        )
    if len(per_shell) > 1 and len(per_shell) != max(len(shell_ls), 1):
        raise RuntimeError(f"N0 lists {len(per_shell)} occupations but the cluster has {len(shell_ls)} shell(s).")
    found = shell_orbital_counts(call.corr_to_spherical, shell_ls)
    if found is None:
        return None, None
    counts, index = found
    for a, (n0, n_a) in enumerate(zip(per_shell, counts)):
        if not 0 <= n0 <= n_a:
            raise RuntimeError(
                f"Nominal occupation {n0} of shell {a} (l={shell_ls[a]}) is out of bounds [0, {n_a}]."
            )
    return counts, index


def prepare_problem(call, line, comm):
    """Fit h0 and build the solver model for ``call`` (already synchronised and parsed into ``line``).

    Prints the settings header, fits the bath (or reuses this iteration's fit), rotates everything
    into the fitting basis -- the basis the solver then works in; the fit chooses it from the data
    as RSPt sends it -- and returns the problem there.
    """
    rank = comm.rank
    h_dft, u4, sig_dc, w, eim, tau = call.h_dft, call.u4, call.out.sig_dc, call.w, call.eim, call.tau
    corr_to_spherical, operators, verbosity = call.corr_to_spherical, call.operators, call.verbosity
    is_dc = call.task == "dc"
    n_orb = h_dft.shape[0]
    filename = archive.FILENAME
    label = call.label.strip()
    # For python, it makes more sense to put the frequency index first, instead of last
    hyb = np.moveaxis(call.hyb, -1, 0)
    line = with_temperature(line, tau)
    print(format_settings_header(line, tau=tau, delta=eim), flush=True)
    basis, solver = line.basis, line.solver
    nominal_occ = basis.nominal_occ
    if any(n0 > n_orb for n0 in nominal_occ.values()) or any(n0 < 0 for n0 in nominal_occ.values()):
        raise RuntimeError(f"Nominal impurity occupation {nominal_occ} out of bounds [0, {n_orb}]")
    shell_ls = call.shells.shell_ls
    counts, orbital_shell = _check_shells(call, line)
    movable = None
    hybridizing = frozenset({0})
    if orbital_shell is not None:
        hybridizing = hybridizing_shells(hyb, w, orbital_shell, len(shell_ls), line.fit)
        # Charge is restricted to the hybridizing shells only where there is a choice to make; a single
        # shell (or none hybridizing at all) keeps the unrestricted rule.
        movable = hybridizing if len(shell_ls) > 1 and hybridizing else None
        if rank == 0:
            for a, l in enumerate(shell_ls):
                kind = "complete" if counts[a] == 2 * (2 * l + 1) else "partial"
                print(
                    f"shell {a} (l={l}): {counts[a]} of {2 * (2 * l + 1)} spin-orbitals, {kind}"
                    + ("" if a in hybridizing else ", no hybridization")
                    + (", charge is not moved onto it" if movable is not None and a not in movable else "")
                )

    cap_ledger = CapLedger()
    if not is_dc:
        # The cap the DC search of this iteration solved at bounds this solve's auto cap.
        cap_ledger = CapLedger(ceiling=archive.double_counting_cap(filename, label, comm))
        # Seed the ground-state search with the charge sector the double-counting search settled
        # on for this same iteration.  The DC is only meaningful if this solve lands on the state
        # it was measured against, and the sector walk is a discrete choice that a small change in
        # the incoming hybridization can flip.  The walk still runs and can still move.
        dc_sector = archive.double_counting_sector(filename, label, comm)
        if dc_sector is not None and dc_sector != sum(nominal_occ.values()):
            if rank == 0:
                print(
                    f"Seeding the ground-state search at the double-counting search's sector "
                    f"{dc_sector} (nominal {sum(nominal_occ.values())})."
                )
            nominal_occ = split_total_over_groups(dc_sector, nominal_occ, movable=movable)
            basis = replace(basis, nominal_occ=nominal_occ)

    if abs(w[1] - w[0]) > eim / 2 and rank == 0:
        print(
            "WARNING: Your real frequency mesh is rather coarse. Recommended dE <= eim/5, "
            "in order to guarantee resolution of fine structures in the self energy."
        )
    if rank == 0 and corr_to_spherical.shape[0] != corr_to_spherical.shape[1]:
        print(
            "The correlated orbitals span only part of the shell "
            f"({corr_to_spherical.shape[0]} of {corr_to_spherical.shape[1]} "
            "spin-orbitals); the solver will skip the L/S/J observables."
        )

    fitted = fit_h0(
        h_dft,
        hyb,
        line.n_baths,
        w,
        eim,
        line.fit,
        u4=u4,
        rot_to_spherical=corr_to_spherical,
        label=label,
        filename=filename,
        verbose=(verbosity >= 1 or is_dc),
        extra_verbose=(verbosity >= 2),
        comm=comm,
    )
    if orbital_shell is not None and line.n_baths > 0 and rank == 0:
        # The shell rule above and the fit's block rule compare against different references (the
        # strongest shell, the strongest block); say so when they disagree.
        with_bath = shells_with_bath(fitted.v, orbital_shell, len(shell_ls))
        if with_bath != set(hybridizing):
            print(
                f"WARNING: shells {sorted(with_bath)} got bath states but shells {sorted(hybridizing)} count as "
                f"hybridizing (hyb_tol {line.fit.hyb_tol:g}); the charge restriction follows the latter."
            )
    # The fit neglected the couplings between blocks (symm_prec) in its own basis, in which the
    # impurity block, the hopping and the double counting are exactly block diagonal.  That is the
    # basis the solver must work in: rotated back to RSPt's basis the neglected couplings are
    # spread over all orbitals again and the solver would merge the blocks.  So the interaction, the
    # Green's functions and the double counting follow the fit, and ``corr_to_solver`` is the
    # rotation back to RSPt's correlated basis.
    rotation = fitted.rotation
    if rank == 0:
        deviation = np.max(np.abs(rotation - np.eye(n_orb)))
        print(f"Max abs deviation of the fitting basis from RSPt's correlated basis: {deviation:.3e}")
    blocks = fitted.block_structure.blocks
    n_total = fitted.H_solver.shape[0]
    full_rotation = np.eye(n_total, dtype=complex)
    full_rotation[np.ix_(fitted.impurity_indices, fitted.impurity_indices)] = rotation
    H_solver = rotate_matrix(fitted.H_solver, full_rotation)
    H_imp = project_to_blocks(rotate_matrix(fitted.H_imp, rotation), blocks)
    v = fitted.v @ rotation
    hyb = rotate_Greens_function(hyb, rotation)
    h_dft = rotate_matrix(h_dft, rotation)
    u4 = rotate_4index_U(u4, rotation)
    sig_dc_fit = project_to_blocks(rotate_matrix(sig_dc, rotation), blocks)
    # Only if the fit neglected something are its one-body blocks authoritative and the interaction
    # must not merge them back; otherwise the safety net stays on.
    basis = replace(basis, interaction_merges_blocks=not fitted.neglected_couplings)
    if rank == 0:
        symm_prec = line.fit.symm_prec
        if fitted.neglected_couplings:
            print(
                f"symm_prec {symm_prec:g}: couplings between {len(blocks)} blocks are neglected "
                "in the fit and in the solver (the interaction does not merge them).",
                flush=True,
            )
        elif line.fit.neglects_couplings:
            print(
                f"symm_prec {symm_prec:g} neglects no coupling of this problem; "
                "the solver may still merge blocks the interaction couples.",
                flush=True,
            )
    # One ImpurityModel from the fitted blocks: it assembles the operator and derives the
    # impurity/bath orbital layout from the block sizes.  model.h0 is the raw h_dft + bath (RSPt
    # passes h_dft WITHOUT the double counting subtracted) and model.dc is RSPt's current dc:
    # calc_selfenergy subtracts it and the fixed_*_dc searches use it as their guess.
    rot_to_spherical = np.conj(rotation.T) @ corr_to_spherical
    if operators is not None and verbosity >= 1:
        _report_operator_check(operators, corr_to_spherical, call.shells.shell_ls, comm)
    valence_shell, core_shell = (
        spectra_shells(hybridizing, max(len(shell_ls), 1)) if call.task == "spectra" else (None, None)
    )
    model = ImpurityModel.from_blocks(
        H_imp,
        v,
        fitted.H_bath,
        u4=u4,
        dc=sig_dc_fit,
        rot_to_spherical=rot_to_spherical,
        bath_valence_conduction=(fitted.valence_bath_indices, fitted.conduction_bath_indices),
        operators=None if operators is None else operators.rotated(rotation),
        shell_ls=tuple(shell_ls) if shell_ls else None,
        valence_shell=valence_shell,
        core_shell=core_shell,
    )
    return Problem(
        model=model,
        basis=basis,
        solver=solver,
        blocks=blocks,
        corr_to_solver=rotation,
        sig_dc=sig_dc_fit,
        cap_ledger=cap_ledger,
        line=replace(line, basis=basis),
        fitted=fitted,
        H_solver=H_solver,
        h_dft=h_dft,
        u4=u4,
        hyb=hyb,
        v=v,
        rot_to_spherical=rot_to_spherical,
        hybridizing=frozenset(hybridizing) if orbital_shell is not None else frozenset({0}),
        n_shells=max(len(shell_ls), 1),
    )


def _report_fit(problem, *, w, eim, tau, label, comm):
    """Verbosity >= 1: how well the bath reproduces the hybridization, and the DFT reference occupation."""
    hyb = problem.hyb
    hyb_fit = bath_hybridization(w, eim, problem.fitted.H_bath, problem.v)
    # Unoccupied states are only fitted with fit_unocc, so also report the deviation on the occupied side alone.
    fit_dev = np.max(np.abs(hyb_fit - hyb))
    fit_dev_occ = np.max(np.abs(hyb_fit[w <= 0] - hyb[w <= 0]))
    print(f"Max abs deviation of the fitted hybridization function: {fit_dev_occ:.6f} (w <= 0), {fit_dev:.6f} (all w)")
    save_Greens_function(problem.to_rspt_green(hyb_fit), w, "hyb-fit", label.strip())
    # Is the occupation fixed_occupation_dc pins to the true DFT impurity occupation, or an artefact of
    # how finely the bath was discretized?  Diagnostic only; see dc_reference.report_continuum_reference.
    n0_disc = discretized_impurity_occupation(problem.model, tau)
    report_continuum_reference(problem.h_dft, hyb, hyb_fit, w, eim, tau, n0_disc, rank=comm.rank)
