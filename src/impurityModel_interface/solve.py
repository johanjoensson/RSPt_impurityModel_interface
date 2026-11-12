"""One RSPt call: the double-counting search or the self-energy solve, on marshalled arrays.

RSPt calls the solver twice per self-consistency iteration, once for the double counting
(``dc_flag == 1``) and once for the self-energy.  Both start from the same problem -- RSPt's
quantities rotated into the fitting basis, the fitted h0, the solver model -- built by
:func:`prepare_problem`; :func:`run_double_counting` and :func:`run_selfenergy` finish the call.

The arrays are RSPt's own memory and are mutated in place (see :mod:`impurityModel_interface.entry`).
"""

import re
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, replace
from os import devnull

import numpy as np
from impurityModel.api import (
    CapLedger,
    ImpurityModel,
    Meshes,
    calc_selfenergy,
    discretized_impurity_occupation,
    report_continuum_reference,
    save_Greens_function,
)
from mpi4py import MPI
from rspt2spectra.block_structure import project_to_blocks
from rspt2spectra.h0 import bath_hybridization
from rspt2spectra.utils import rotate_4index_U, rotate_Greens_function, rotate_matrix

from . import archive
from .dc_line import parse_dc_line
from .double_counting import run_double_counting as search_double_counting
from .h0_fit import fit_h0
from .mpi import on_root
from .solver_line import format_settings_header, parse_solver_line, with_temperature


def reconstruct_rotations(corr_to_spherical_in, corr_to_cf_in, n_orb, n_rot_cols, n_orb_full):
    """
    Return (corr_to_spherical, corr_to_cf) with both spin blocks present.

    RSPt dimensions the rotation matrices with nspmat spins (1 for
    non-spin-polarized calculations without SOC, 2 otherwise), while the
    hamiltonian, hybridization function and selfenergies always carry both
    spins (n_orb rows). When only one spin block is sent
    (n_rot_cols == n_orb/2), duplicate it onto the spin-up block; the
    correlated basis rows are ordered [spin-down block, spin-up block].

    Parameters:
    corr_to_spherical_in -- (n_orb, n_orb_full) rotation from the correlated
                            basis to spherical harmonics, n_orb_full counts
                            nspmat spins.
    corr_to_cf_in        -- (n_orb, n_rot_cols) rotation from the correlated
                            basis to the CF basis, n_rot_cols counts nspmat
                            spins.
    n_orb                -- Number of correlated spin-orbitals (2 spins).
    n_rot_cols           -- Number of CF columns sent by RSPt.
    n_orb_full           -- Spherical-harmonics dimension sent by RSPt.
    """
    if n_rot_cols == n_orb:
        # The matrices already carry both spins (spin-polarized or SOC).
        # corr_to_spherical may be rectangular (n_orb < n_orb_full) when the
        # correlated set is a subset of the shell (e.g. t2g only).
        return np.array(corr_to_spherical_in), np.array(corr_to_cf_in)
    if 2 * n_rot_cols == n_orb:
        n_half = n_orb // 2
        corr_to_spherical = np.zeros((n_orb, 2 * n_orb_full), dtype=complex)
        corr_to_spherical[:n_half, :n_orb_full] = corr_to_spherical_in[:n_half, :]
        corr_to_spherical[n_half:, n_orb_full:] = corr_to_spherical_in[:n_half, :]
        corr_to_cf = np.zeros((n_orb, n_orb), dtype=complex)
        corr_to_cf[:n_half, :n_rot_cols] = corr_to_cf_in[:n_half, :]
        corr_to_cf[n_half:, n_rot_cols:] = corr_to_cf_in[:n_half, :]
        return corr_to_spherical, corr_to_cf
    raise RuntimeError(
        f"Inconsistent rotation shapes from RSPt: n_orb={n_orb}, "
        f"n_rot_cols={n_rot_cols}, n_orb_full={n_orb_full}; "
        "expected n_rot_cols == n_orb or 2*n_rot_cols == n_orb."
    )


def split_total_over_groups(total, nominal_occ):
    """Distribute an impurity charge ``total`` over the groups of ``nominal_occ``.

    Keeps the incoming split as the shape and moves only the difference, so a multi-group
    impurity (an ``eg``/``t2g`` split) stays near the filling RSPt asked for instead of being
    re-derived from scratch. The groups redistribute freely at fixed total inside the solver
    anyway (``basis_generation.generate_initial_basis``), so this only has to be a sensible seed.
    """
    occ = dict(nominal_occ)
    delta = int(total) - sum(occ.values())
    keys = sorted(occ)
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


@contextmanager
def redirected_stdout(label, dc_flag, comm, verbosity):
    """Send ``print`` to this call's output file; restore ``sys.stdout`` on every path out.

    Rank 0 writes ``impurityModel-<label>[-dc].out``; the other ranks write their own file when
    ``verbosity > 0`` and otherwise discard their output.
    """
    suffix = "-dc" if dc_flag == 1 else ""
    if comm.rank == 0:
        name = f"impurityModel-{label.strip()}{suffix}.out"
    elif verbosity > 0:
        name = f"impurityModel-{label.strip()}{suffix}-{comm.rank}.out"
    else:
        name = devnull
    saved = sys.stdout
    sys.stdout = open(name, "w")  # noqa: SIM115 - closed below
    try:
        yield
    finally:
        try:
            sys.stdout.close()
        finally:
            sys.stdout = saved


@dataclass
class Problem:
    """Everything the two kinds of call share, in the fitting basis."""

    model: ImpurityModel
    basis: object
    solver: object
    #: Block partition of the fit; the double counting carries no coupling between blocks.
    blocks: list
    #: Rotation from RSPt's correlated basis to the fitting basis.
    corr_to_cf: np.ndarray
    #: RSPt's incoming double counting, in the fitting basis.
    sig_dc: np.ndarray
    cap_ledger: CapLedger


def prepare_problem(
    *, label, line, dc_flag, h_dft, u4, hyb, w, corr_to_spherical, corr_to_cf, sig_dc, eim, tau, comm, verbosity
):
    """Fit h0 and build the solver model.  ``h_dft``, ``u4``, ``hyb`` are already in the CF basis.

    Prints the settings header, fits the bath (or reuses this iteration's fit), rotates everything
    into the fitting basis -- the basis the solver then works in -- and writes the problem into the
    archive.
    """
    rank = comm.rank
    n_orb = h_dft.shape[0]
    filename = archive.FILENAME
    label = label.strip()
    line = with_temperature(line, tau)
    print(format_settings_header(line, tau=tau, delta=eim), flush=True)
    basis, solver = line.basis, line.solver
    nominal_occ = basis.nominal_occ
    if any(n0 > n_orb for n0 in nominal_occ.values()) or any(n0 < 0 for n0 in nominal_occ.values()):
        raise RuntimeError(f"Nominal impurity occupation {nominal_occ} out of bounds [0, {n_orb}]")

    cap_ledger = CapLedger()
    if dc_flag != 1:
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
            nominal_occ = split_total_over_groups(dc_sector, nominal_occ)
            basis = replace(basis, nominal_occ=nominal_occ)

    if abs(w[1] - w[0]) > eim / 2 and rank == 0:
        print(
            "WARNING: Your real frequency mesh is rather coarse. Recommended dE <= eim/5, "
            "in order to guarantee resolution of fine structures in the self energy."
        )
    # Log how different RSPt's correlated basis and the CF basis are, so basis mixups are visible.
    if rank == 0:
        print(f"Max abs deviation of corr_to_cf from identity: {np.max(np.abs(corr_to_cf - np.eye(n_orb))):.3e}")
        if corr_to_spherical.shape[0] != corr_to_spherical.shape[1]:
            print(
                "The correlated orbitals span only part of the shell "
                f"({corr_to_spherical.shape[0]} of {corr_to_spherical.shape[1]} "
                "spin-orbitals); the solver will skip the L/S/J observables."
            )
    # RSPt supplies the double counting in the corr basis, the solver needs it in the CF basis.
    sig_dc_cf = rotate_matrix(sig_dc, corr_to_cf)

    fitted = fit_h0(
        h_dft,
        hyb,
        line.n_baths,
        w,
        eim,
        line.fit,
        u4=u4,
        rot_to_spherical=np.conj(corr_to_cf.T) @ corr_to_spherical,
        label=label,
        filename=filename,
        verbose=(verbosity >= 1 or dc_flag == 1),
        extra_verbose=(verbosity >= 2),
        comm=comm,
    )
    # The fit neglected the couplings between blocks (symm_prec) in its own basis, in which the
    # impurity block, the hopping and the double counting are exactly block diagonal.  That is the
    # basis the solver must work in: rotated back to the CF basis the neglected couplings are
    # spread over all orbitals again and the solver would merge the blocks.  From here on the "CF"
    # basis is the fitting basis, so the interaction, the Green's functions and the rotation back
    # to RSPt's correlated basis (corr_to_cf) all follow it.
    rotation = fitted.rotation
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
    sig_dc_cf = project_to_blocks(rotate_matrix(sig_dc_cf, rotation), blocks)
    corr_to_cf = corr_to_cf @ rotation
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
    rot_to_spherical = np.conj(corr_to_cf.T) @ corr_to_spherical
    model = ImpurityModel.from_blocks(
        H_imp,
        v,
        fitted.H_bath,
        u4=u4,
        dc=sig_dc_cf,
        rot_to_spherical=rot_to_spherical,
        bath_valence_conduction=(fitted.valence_bath_indices, fitted.conduction_bath_indices),
    )
    return (
        Problem(model, basis, solver, blocks, corr_to_cf, sig_dc_cf, cap_ledger),
        dict(
            line=replace(line, basis=basis),
            fitted=fitted,
            H_solver=H_solver,
            h_dft=h_dft,
            u4=u4,
            hyb=hyb,
            v=v,
            rot_to_spherical=rot_to_spherical,
        ),
    )


def _report_fit(problem, context, *, w, eim, tau, label, comm):
    """Verbosity >= 1: how well the bath reproduces the hybridization, and the DFT reference occupation."""
    fitted = context["fitted"]
    hyb = context["hyb"]
    hyb_fit = bath_hybridization(w, eim, fitted.H_bath, context["v"])
    # Unoccupied states are only fitted with fit_unocc, so also report the deviation on the occupied side alone.
    fit_dev = np.max(np.abs(hyb_fit - hyb))
    fit_dev_occ = np.max(np.abs(hyb_fit[w <= 0] - hyb[w <= 0]))
    print(f"Max abs deviation of the fitted hybridization function: {fit_dev_occ:.6f} (w <= 0), {fit_dev:.6f} (all w)")
    save_Greens_function(rotate_Greens_function(hyb_fit, np.conj(problem.corr_to_cf.T)), w, "hyb-fit", label.strip())
    # Is the occupation fixed_occupation_dc pins to the true DFT impurity occupation, or an artefact of
    # how finely the bath was discretized?  Diagnostic only; see dc_reference.report_continuum_reference.
    n0_disc = discretized_impurity_occupation(problem.model, tau)
    report_continuum_reference(context["h_dft"], hyb, hyb_fit, w, eim, tau, n0_disc, rank=comm.rank)


def solve(
    label,
    solver_line,
    dc_line,
    dc_flag,
    h_dft,
    u4,
    hyb,
    iw,
    w,
    sig,
    sig_real,
    sig_static,
    sig_dc,
    corr_to_spherical_in,
    corr_to_cf_in,
    n_orb,
    n_rot_cols,
    n_orb_full,
    eim,
    tau,
    verbosity,
):
    """Run the double-counting or self-energy solve on already-marshalled arrays.

    Pure Python: ``label``/``solver_line``/``dc_line`` are ``str`` and every array is a numpy
    array.  The arrays are the caller's and are mutated in place.  Returns 0, or -1 if the
    self-energy solve failed (after adding infinity to ``sig`` and aborting the communicator).
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    corr_to_spherical, corr_to_cf = reconstruct_rotations(
        corr_to_spherical_in, corr_to_cf_in, n_orb, n_rot_cols, n_orb_full
    )
    comm.Bcast(corr_to_spherical)
    comm.Bcast(corr_to_cf)
    comm.Bcast(h_dft)
    comm.Bcast(u4)
    comm.Bcast(sig_dc)
    comm.Bcast(hyb)

    # For python, it makes more sense to put the frequency index first, instead of last
    sig_python = np.moveaxis(sig, -1, 0)
    sig_real_python = np.moveaxis(sig_real, -1, 0)
    hyb = np.moveaxis(hyb, -1, 0)

    hyb = rotate_Greens_function(hyb, corr_to_cf)
    h_dft = rotate_matrix(h_dft, corr_to_cf)
    u4 = rotate_4index_U(u4, corr_to_cf)

    with redirected_stdout(label, dc_flag, comm, verbosity):
        verbosity = comm.bcast(verbosity)
        line = parse_solver_line(solver_line)
        if rank == 0 and len(solver_line.strip()) >= 100 and not re.search(r"[!#]", solver_line):
            print(
                "WARNING: the solver line fills all 100 characters RSPt passes; the last option may have been cut off.",
                flush=True,
            )
        request = parse_dc_line(dc_line) if dc_flag == 1 else None
        problem, context = prepare_problem(
            label=label,
            line=line,
            dc_flag=dc_flag,
            h_dft=h_dft,
            u4=u4,
            hyb=hyb,
            w=w,
            corr_to_spherical=corr_to_spherical,
            corr_to_cf=corr_to_cf,
            sig_dc=sig_dc,
            eim=eim,
            tau=tau,
            comm=comm,
            verbosity=verbosity,
        )
        fitted = context["fitted"]
        on_root(
            comm,
            lambda: archive.write_problem(
                archive.FILENAME,
                label,
                line=context["line"],
                solver_line=solver_line,
                tau=tau,
                delta=eim,
                w=w,
                iw=iw,
                h_solver=context["H_solver"],
                h_dft=context["h_dft"],
                h_bath=fitted.H_bath,
                v=context["v"],
                u4=context["u4"],
                dc=problem.sig_dc,
                corr_to_cf=problem.corr_to_cf,
                rot_to_spherical=context["rot_to_spherical"],
                layout=(fitted.impurity_indices, fitted.valence_bath_indices, fitted.conduction_bath_indices),
            ),
        )
        if verbosity >= 1:
            on_root(comm, lambda: _report_fit(problem, context, w=w, eim=eim, tau=tau, label=label, comm=comm))

        if dc_flag == 1:
            return run_double_counting(
                request, problem, context, label=label, sig_dc=sig_dc, w=w, tau=tau, comm=comm, verbosity=verbosity
            )
        status = run_selfenergy(
            problem,
            iw=iw,
            w=w,
            eim=eim,
            label=label,
            sig=sig,
            sig_static=sig_static,
            sig_python=sig_python,
            sig_real=sig_real,
            sig_real_python=sig_real_python,
            comm=comm,
            verbosity=verbosity,
        )
    comm.barrier()
    return status


def run_double_counting(request, problem, context, *, label, sig_dc, w, tau, comm, verbosity):
    """The ``dc_flag == 1`` call: write the damped double counting into ``sig_dc`` (RSPt's basis).

    A search that fails leaves ``sig_dc`` holding RSPt's incoming value and still returns 0; see
    :func:`impurityModel_interface.double_counting.run_double_counting`.
    """
    damped = search_double_counting(
        request,
        problem.model,
        problem.basis,
        problem.solver,
        guess=problem.sig_dc,
        blocks=problem.blocks,
        corr_to_cf=problem.corr_to_cf,
        w=w,
        tau=tau,
        nominal_total=sum(problem.basis.nominal_occ.values()),
        comm=comm,
        verbosity=verbosity,
        cap_ledger=problem.cap_ledger,
        filename=archive.FILENAME,
        label=label,
    )
    if damped is not None:
        # The double counting is calculated in the fitting basis, RSPt expects it in the corr basis.
        sig_dc[:, :] = rotate_matrix(damped, np.conj(problem.corr_to_cf.T))
    comm.barrier()
    return 0


def run_selfenergy(
    problem, *, iw, w, eim, label, sig, sig_static, sig_python, sig_real, sig_real_python, comm, verbosity
):
    """The ``dc_flag != 1`` call: fill ``sig``, ``sig_real``, ``sig_static`` in RSPt's basis."""
    try:
        meshes = Meshes(iw=1j * iw, w=w, delta=eim)
        results = calc_selfenergy(
            problem.model,
            meshes,
            problem.basis,
            problem.solver,
            comm=comm,
            verbosity=verbosity,
            cluster_label=label.strip(),
            cap_ledger=problem.cap_ledger,
        )
        # calc_selfenergy returns everything in the fitting basis; RSPt expects the corr basis.
        u = np.conj(problem.corr_to_cf.T)
        if comm.rank == 0:
            sig_static[:, :] = rotate_matrix(results["sigma_static"], u)
            sig_python[:, :, :] = rotate_Greens_function(results["sigma"], u)
            sig_real_python[:, :, :] = rotate_Greens_function(results["sigma_real"], u)
        comm.Bcast(sig_static, root=0)
        comm.Bcast(sig_real, root=0)
        comm.Bcast(sig, root=0)
        if comm.rank == 0:
            archive.write_selfenergy(
                archive.FILENAME, label, results, sig_static, sig_real_python, sig_python, rotate_Greens_function, u
            )
    except Exception as error:
        print("!" * 100)
        print(f"Exception {error!r} caught on rank {comm.rank}!")
        print(traceback.format_exc())
        print(
            "Adding positive infinity to the imaginary part of the selfenergy at the last matsubara frequency.",
            flush=True,
        )
        print("!" * 100)
        sig[:, :, -1] += 1j * np.inf
        comm.Abort(-1)
        return -1
    print("Self energy calculated! impurityModel shutting down.", flush=True)
    return 0
