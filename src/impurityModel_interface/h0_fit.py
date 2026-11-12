"""The impurity Hamiltonian h0 for one call: rspt2spectra fits it, this module reuses what it can.

The fit itself (basis, partition, bath, geometry) is rspt2spectra's.  What is the wrapper's is the
reuse across the two calls RSPt makes per iteration -- the DC search and the self-energy solve must
run on the *same* bath fit -- and the seed for the next iteration's warm start.
"""

from dataclasses import dataclass

import numpy as np
from rspt2spectra.h0 import assemble_h0, bath_fit_fingerprint, fit_bath, fit_window, prepare_hyb_fit
from rspt2spectra.plot import plot_hyb_fit

from . import archive
from .mpi import on_root


@dataclass
class FittedH0:
    """The fitted blocks, in RSPt's correlated-to-CF input basis (rotate with ``rotation``)."""

    H_imp: np.ndarray
    impurity_indices: list
    valence_bath_indices: list
    conduction_bath_indices: list
    v: np.ndarray
    H_bath: np.ndarray
    #: The full one-particle solver matrix (impurity + bath).
    H_solver: np.ndarray
    #: ``Q``: columns are the fitting orbitals in the input basis.  ``Q^dagger H_imp Q`` and
    #: ``v Q`` are exactly block diagonal; in the input basis the neglected couplings are spread
    #: over all orbitals.
    rotation: np.ndarray
    block_structure: object
    #: Whether ``symm_prec`` neglected a coupling the default threshold would have kept.
    neglected_couplings: bool


def fit_h0(
    h_dft,
    hyb,
    n_baths,
    w,
    eim,
    options,
    *,
    u4,
    rot_to_spherical,
    label,
    filename,
    verbose,
    extra_verbose=False,
    comm=None,
):
    """Fit the bath and assemble h0.

    Parameters
    ----------
    h_dft, hyb : np.ndarray
        Local Hamiltonian and real-frequency hybridization function (``hyb`` has the frequency
        first), in the CF basis.
    n_baths : int
        Bath states per impurity orbital.
    options : rspt2spectra.FitOptions
    u4, rot_to_spherical : np.ndarray
        The Coulomb tensor and the map to spherical harmonics, which steer the choice of fitting
        basis (``options.u_densify``).
    label, filename : str
        Cluster label and the archive the stored fit is read from and written to.
    """
    rank = 0 if comm is None else comm.rank
    label = label.strip()
    # Every rank fits in rank 0's window and rank 0's basis and partition.
    x_lim = on_root(comm, lambda: fit_window(w, hyb, not options.fit_unocc))
    fit_basis = prepare_hyb_fit(
        hyb,
        h_dft,
        tol=options.symm_prec,
        verbose=verbose,
        w=w,
        x_lim=x_lim,
        u4=u4,
        to_spherical=rot_to_spherical,
        max_u_densify=options.u_densify,
        comm=comm,
    )
    block_structure = fit_basis.block_structure
    # Everything the stored fit depends on: the hybridization and basis, and nBaths, mesh, gamma, weight.
    fit_key = bath_fit_fingerprint(fit_basis, w, eim, n_baths, options)

    # RSPt calls run_impmod_ed twice per CSC iteration (the DC search, then the self-energy); each
    # call runs this fit.  If the two did not use the same bath fit, the DC would have been
    # determined on a different model than the one it is applied to.
    def look_up():
        stored, _found, iteration = archive.load_stored_fit(filename, label, fit_key, block_structure)
        where = "iteration ?" if iteration is None else f"iteration {iteration}"
        if stored is not None:
            print("Reading stored bath energies and hopping parameters")
            print(
                f"Bath fit REUSED for {label!r} {where} (hybridization fingerprint matches): the DC search and "
                "the selfenergy solve share the identical bath model.",
                flush=True,
            )
        else:
            print(f"Bath fit computed for {label!r} {where} (no fit stored yet this iteration).", flush=True)
        return stored

    stored_fit = on_root(comm, look_up)

    # Seed a new fit with the latest stored bath energies: frozen (hoppings re-solved only) or
    # warm-started (kept unless a global search beats it by warm_tol).  Gated on the fit being new,
    # not on a flag only rank 0 knows, so every rank takes the same branch into the collective fit.
    previous_energies = None
    if (options.freeze_bath_energies or options.warm_start) and stored_fit is None:
        previous_energies = on_root(comm, lambda: archive.previous_bath_energies(filename, label, block_structure))
        if rank == 0:
            mode = "freeze_bath_energies is set" if options.freeze_bath_energies else "Warm start"
            if previous_energies is None:
                print(f"{mode}: no usable previous bath fit is stored; doing a cold, full fit.", flush=True)
            elif options.freeze_bath_energies:
                print("Bath energies FROZEN at the previous fit; re-solving hoppings only.", flush=True)
            else:
                print(f"Bath fit warm-started from the previous fit (margin {options.warm_tol:g}).", flush=True)

    bath_fit = fit_bath(
        fit_basis,
        w,
        eim,
        n_baths,
        options,
        stored_fit=stored_fit,
        previous_energies=previous_energies,
        verbose=verbose,
        comm=comm,
    )
    if stored_fit is None:
        on_root(comm, lambda: archive.store_bath_fit(filename, label, fit_key, block_structure, bath_fit))

    # rspt2spectra turns the flattened star fit into the requested geometry and assembles h0,
    # including the star reference that classifies bath states as valence/conduction.
    (H, _H_star, impurity, valence, conduction, v, H_bath, H_imp) = assemble_h0(
        bath_fit.ebs_star,
        bath_fit.vs_star,
        bath_fit.shifts,
        h_dft,
        fit_basis.H_local_Q,
        fit_basis.Q,
        block_structure,
        bath_geometry=options.bath_geometry,
        w=w,
        eim=eim,
        label=label,
        verbose=verbose,
        extra_verbose=extra_verbose,
        comm=comm,
    )
    if extra_verbose and rank == 0:
        figures = plot_hyb_fit(
            w,
            eim,
            fit_basis.phase_hyb,
            bath_fit.ebs_star,
            bath_fit.vs_star,
            bath_fit.shifts,
            fit_basis.H_local_Q,
            block_structure,
            options.bath_geometry,
        )
        for index, figure in enumerate(figures):
            figure.savefig(f"bath_state_resolved_hybridization_fit_block_{index}.png")
    return FittedH0(
        H_imp=H_imp,
        impurity_indices=impurity,
        valence_bath_indices=valence,
        conduction_bath_indices=conduction,
        v=v,
        H_bath=H_bath,
        H_solver=H,
        rotation=fit_basis.Q,
        block_structure=block_structure,
        neglected_couplings=fit_basis.neglected_couplings,
    )
