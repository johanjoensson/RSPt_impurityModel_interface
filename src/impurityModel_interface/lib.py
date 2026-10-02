import mpi4py

# All imports below run after the mpi4py.rc settings and the OMP_NUM_THREADS
# block, so E402 is expected and suppressed throughout this preamble. The
# ordering is deliberate: mpi4py.rc must be set before importing MPI, and
# OMP_NUM_THREADS must be set before importing the thread-spawning numeric
# libraries (numpy, h5py).
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
import sys  # noqa: E402
from os import devnull, environ  # noqa: E402

from mpi4py import MPI  # noqa: E402

if "OMP_NUM_THREADS" not in environ:
    print(
        "OMP_NUM_THREADS will be set to 1 from now on!.",
        file=sys.stderr,
    )
    environ["OMP_NUM_THREADS"] = "1"

import hashlib  # noqa: E402
import traceback  # noqa: E402
from dataclasses import replace  # noqa: E402
from importlib.metadata import PackageNotFoundError  # noqa: E402
from importlib.metadata import version as package_version  # noqa: E402

import h5py as h5  # noqa: E402
import numpy as np  # noqa: E402

try:
    from run_impurityModel import ffi
except ImportError:
    # Not running embedded in RSPt (e.g. unit tests). Provide a stub so that
    # the module can still be imported and the pure python helpers used.
    class _FFIStub:
        @staticmethod
        def def_extern(*args, **kwargs):
            # The real ffi.def_extern takes an optional name and an optional error=; accept
            # and ignore both, so the decorator below keeps working outside the embedding.
            return lambda func: func

        @staticmethod
        def string(*args, **kwargs):
            raise RuntimeError("ffi.string is only available when embedded in RSPt")

        @staticmethod
        def buffer(*args, **kwargs):
            raise RuntimeError("ffi.buffer is only available when embedded in RSPt")

    ffi = _FFIStub()
from rspt2spectra.h0 import (
    assemble_h0,
    flatten_star_levels,
    prepare_hyb_fit,
)  # noqa: E402
from rspt2spectra.hyb_fit import fit_hyb  # noqa: E402
from rspt2spectra.plot import plot_hyb_fit  # noqa: E402
from rspt2spectra.utils import (  # noqa: E402
    rotate_4index_U,
    rotate_Greens_function,
    rotate_matrix,
)
from rspt2spectra.weight_functions import weight_functions  # noqa: E402

try:
    # The stable external surface of the solver; everything under
    # impurityModel.ed.* is internal.
    from impurityModel.api import (
        BasisOptions,
        DoubleCountingUnreachable,
        ImpurityModel,
        Meshes,
        SolverOptions,
        amf_dc,
        apply_environment,
        calc_selfenergy,
        dc_levels,
        dc_spread,
        discretized_impurity_occupation,
        emit_dc_record,
        fixed_gap_dc,
        fixed_occupation_dc,
        fixed_peak_dc,
        find_environment_file,
        fll_dc,
        load_environment,
        nominal_dc,
        parse_truncation_threshold,
        report_continuum_reference,
        save_Greens_function,
        sigma_inf_dc,
    )
except ImportError as import_error:
    raise ImportError(
        f"Failed to import impurityModel under Python "
        f"{sys.version_info.major}.{sys.version_info.minor}: {import_error}\n"
        "impurityModel ships a compiled extension (ManyBodyUtils) built for a "
        "specific Python minor version. If the import above mentions "
        "ManyBodyUtils, rebuild/reinstall impurityModel for this interpreter, "
        "e.g. `pip install --force-reinstall --no-build-isolation "
        "<path-to-impurityModel>`."
    ) from import_error


def parse_solver_line(solver_line):
    """
    Parse the RSPt solver line for the impurityModel ED solver.

    Format: N0 Nbath [options]
      N0     -- Nominal impurity occupation.
      Nbath  -- Number of bath states to fit per impurity orbital.
    Options (whitespace separated, case insensitive):
      periodic | partial (pro) | selective | full -- Reorthogonalization mode.
      star | chain | linked_chain | peeled -- Bath geometry.
      fit_unocc | fit_occ                  -- Also fit unoccupied bath states / only occupied (default).
      freeze_bath_energies                 -- Reuse the previous iteration's bath energies and
                                              only re-solve the hoppings (least squares); no
                                              bath-energy optimization. Off by default; the
                                              first iteration (nothing stored) does a full fit.
      gamma X                              -- Regularization parameter for the bath fit.
      dense_cutoff N                       -- Use dense eigensolver below this matrix size.
      <weight function name>               -- Weight function for the fit; one of
                                              unit, exponential, gaussian, sqrtgauss,
                                              lingauss, quadgauss, step.
      weight X                             -- Weight function decay/steepness factor.
      weight_w0 X                          -- Center of the weight function (default 0).
      spin_flip_dj                         -- Accepted and ignored: it never had an effect on the
                                              CIPSI solver and was removed from impurityModel.
      no_chain_restrict                    -- Disable chain occupation restrictions.
      occ_cutoff X                         -- Occupation cutoff.
      truncation_threshold N               -- Determinant cap per basis: auto (default; sized from memory,
                                              separately for the ground state and the Green's-function
                                              units, may be held lower at run time), unlimited (alias
                                              inf; no cap, memory guard still active), or a positive
                                              integer such as 2e6 (final: never lowered, only warned about).
      slater_min X                         -- Minimal Slater determinant weight.
      e_pt2 X                              -- Residual PT2 energy the ground-state CIPSI
                                              expansion is converged to (default: impurityModel's
                                              GS_E_PT2_TOL, 1e-8). The double-counting search
                                              inherits it unless its own line sets 'e_pt2'.
      dn N                                 -- Allowed impurity occupation window (+-dN).
      mv N                                 -- Mixed valence scalar, forwarded per group to
                                              impurityModel's Basis (see impurityModel docs).
      dense_green                          -- Use the dense block-Lanczos Green's function path.
      gf_method lanczos|bicgstab           -- Green's function kernel (default lanczos): one block-Lanczos
                                              recurrence per work unit serving the whole frequency mesh, or
                                              one linear solve per frequency point on its own rebuilt basis.
                                              Retired kernels are refused with the reason.
      gf_admission auto|all|outer          -- Basis growth of each per-frequency solve (needs gf_method
                                              bicgstab). all admits every determinant the solver produces.
                                              outer solves on a frozen basis, scores the residual outside it,
                                              admits only what clears gf_admit_tol and records a measured
                                              error bound in the output. auto (default) leaves it to the
                                              GF_BICGSTAB_ADMISSION environment knob, else all.
      gf_admit_tol X                       -- Admission threshold of 'gf_admission outer', relative to the
                                              seed norm (default 1e-4). Smaller admits more and is more
                                              accurate; the reported error bound says what was left out.
    """
    # Remove comments from the solver line
    solver_line = solver_line.split("!")[0]
    solver_line = solver_line.split("#")[0]
    solver_array = solver_line.strip().split()
    assert (
        len(solver_array) >= 3
    ), "The impurityModel ED solver requires at least 3 arguments; N0 nBaths excitation_budget"
    try:
        nominal_occ = int(solver_array[0])
        nBaths = int(solver_array[1])
        excitation_budget = int(solver_array[2])
    except Exception as e:
        raise RuntimeError(
            f"{e}\n--->N0 {solver_array[0]}\n--->Nbaths {solver_array[1]}\n"
            f"--->excitation_budget {solver_array[2]}\n--->Other params {solver_array[3:]}"
        ) from e
    if excitation_budget < 0:
        # A negative budget would build an empty admissible window; it means "no budget".
        print(f"excitation_budget {excitation_budget} is negative: the excitation budget is disabled.", flush=True)
        excitation_budget = None
    options = {
        "dense_cutoff": 1000,
        "reort": "none",
        "fit_unocc": False,
        "freeze_bath_energies": False,
        "gamma": 0.01,
        "weight_function": "unit",
        "weight": 2,
        "weight_w0": 0.0,
        "bath_geometry": "peeled",
        "occ_cutoff": 1e-6,
        "dN": None,
        "mv": None,
        "chain_restrict": True,
        "truncation_threshold": None,
        "slater_min": np.sqrt(np.finfo(float).eps),
        "e_pt2_tol": None,
        "collapse_chains": False,
        "sparse_green": True,
        "gf_method": "lanczos",
        "gf_admission": None,
        "gf_admit_tol": None,
    }
    if len(solver_array) > 3:
        skip_next = False
        for i in range(3, len(solver_array)):
            if skip_next:
                skip_next = False
                continue
            arg = solver_array[i]
            if arg.lower() in {
                "none",
                "pro",
                "partial",
                "selective",
                "full",
                "periodic",
            }:
                if arg.lower() == "pro":
                    options["reort"] = "partial"
                else:
                    options["reort"] = arg.lower()
            elif arg.lower() in {
                "star",
                "chain",
                "linked_chain",
                "peeled",
                "peeled_linked_chain",
            }:
                if arg.lower() == "peeled":
                    options["bath_geometry"] = "peeled_linked_chain"
                else:
                    options["bath_geometry"] = arg.lower()
            elif arg.lower() == "fit_unocc":
                options["fit_unocc"] = True
            elif arg.lower() == "fit_occ":
                options["fit_unocc"] = False
            elif arg.lower() == "freeze_bath_energies":
                options["freeze_bath_energies"] = True
            elif arg.lower() == "weight_w0":
                options["weight_w0"] = float(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "gamma":
                options["gamma"] = float(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "dense_cutoff":
                options["dense_cutoff"] = int(solver_array[i + 1])
                skip_next = True
            elif arg.lower() in weight_functions.keys():
                options["weight_function"] = arg.lower()
            elif arg.lower() == "weight":
                options["weight"] = float(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "spin_flip_dj":
                # Kept parseable so an existing green.inp still runs: the option was a no-op
                # (only the pre-CIPSI Basis.expand read it) and impurityModel no longer has it.
                print("spin_flip_dj on the solver line is ignored: it had no effect and has been removed.")
            elif arg.lower() == "no_chain_restrict":
                options["chain_restrict"] = False
            elif arg.lower() == "occ_cutoff":
                options["occ_cutoff"] = float(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "truncation_threshold":
                # The one vocabulary shared with the CLI and TOML input: auto | unlimited/inf | N.
                options["truncation_threshold"] = parse_truncation_threshold(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "slater_min":
                options["slater_min"] = float(solver_array[i + 1])
                skip_next = True
            elif arg.lower() in {"e_pt2", "e_pt2_tol"}:
                assert i + 1 < len(solver_array), f"'{arg}' on the solver line needs a value"
                options["e_pt2_tol"] = float(solver_array[i + 1])
                assert options["e_pt2_tol"] > 0, f"'{arg}' must be positive"
                skip_next = True
            elif arg.lower() == "dn":
                options["dN"] = int(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "mv":
                options["mv"] = int(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "dense_green":
                options["sparse_green"] = False
            elif arg.lower() in {"gf_method", "gf_admission", "gf_admit_tol"}:
                assert i + 1 < len(solver_array), f"'{arg}' on the solver line needs a value"
                value = solver_array[i + 1].lower()
                if arg.lower() == "gf_method":
                    options["gf_method"] = value
                elif arg.lower() == "gf_admission":
                    # "auto" is the TOML spelling of "not specified"; both mean the environment decides.
                    options["gf_admission"] = None if value == "auto" else value
                else:
                    options["gf_admit_tol"] = float(value)
                    assert options["gf_admit_tol"] > 0, f"'{arg}' must be positive"
                skip_next = True
            else:
                raise RuntimeError(f"Unknown solver parameter {arg}.\n--->Other solver params {solver_array[2:]}")
    if options["bath_geometry"] == "star":
        options["chain_restrict"] = False
        options["collapse_chains"] = True
        if options["dN"] is None:
            options["dN"] = 4

    # Split the parsed tokens into: the bath-fit parameters (consumed by rspt2spectra and stored
    # as provenance) and the solver option groups impurityModel consumes. tau is external (a
    # separate run_impmod_ed argument) and is injected onto BasisOptions by the caller.
    fit_options = {
        key: options[key]
        for key in (
            "gamma",
            "weight_function",
            "weight",
            "weight_w0",
            "fit_unocc",
            "freeze_bath_energies",
            "bath_geometry",
            "collapse_chains",
        )
    }
    basis = BasisOptions(
        nominal_occ={0: nominal_occ},
        mixed_valence=None if options["mv"] is None else {0: options["mv"]},
        dN=options["dN"],
        truncation_threshold=options["truncation_threshold"],
        chain_restrict=options["chain_restrict"],
        occ_cutoff=options["occ_cutoff"],
        slater_weight_min=options["slater_min"],
        excitation_budget=excitation_budget,
        e_pt2_tol=options["e_pt2_tol"],
    )
    try:
        solver = SolverOptions(
            reort=options["reort"],
            dense_cutoff=options["dense_cutoff"],
            sparse_green=options["sparse_green"],
            gf_method=options["gf_method"],
            gf_admission=options["gf_admission"],
            gf_admit_tol=options["gf_admit_tol"],
        )
    except ValueError as error:
        # The solver refuses a combination it would otherwise ignore (outer admission on the Lanczos
        # kernel) or a kernel it does not have: say it where the user wrote it, before any solve.
        raise RuntimeError(f"Invalid Green's function setting on the solver line: {error}") from error
    return nominal_occ, nBaths, fit_options, basis, solver


def solver_line_attrs(fit_options, basis, solver, n_baths=None):
    """Flat ``{name: value}`` record of a parsed solver line, for the HDF5 archive attrs.

    Reproduces the historical ``options``-dict attribute dump from the parsed option groups so
    archived runs read back unchanged (see ``impurityModel.ed.model._read_archive_group``), and
    records every other setting the solve depends on. ``excitation_budget`` in particular must be
    stored: the archive reader cannot recover it otherwise and falls back to impurityModel's
    default, so an offline rerun would silently solve a differently restricted problem.
    ``n_baths`` (bath states per impurity orbital) is included when given.
    """
    attrs = {
        "reort": solver.reort,
        "dense_cutoff": solver.dense_cutoff,
        "sparse_green": solver.sparse_green,
        "gf_method": solver.gf_method,
        # None = not specified (the environment decides, else "all" / the default threshold); stored
        # as given, so an archive read back resolves them the way the run did.
        "gf_admission": solver.gf_admission,
        "gf_admit_tol": solver.gf_admit_tol,
        "chain_restrict": basis.chain_restrict,
        "occ_cutoff": basis.occ_cutoff,
        "dN": basis.dN,
        "truncation_threshold": basis.truncation_threshold,
        "slater_min": basis.slater_weight_min,
        "mv": None if basis.mixed_valence is None else basis.mixed_valence[0],
        "excitation_budget": basis.excitation_budget,
        # None = impurityModel's default (GS_E_PT2_TOL); stored as requested, so an archive read
        # back resolves it the same way the run did.
        "e_pt2_tol": basis.e_pt2_tol,
        **fit_options,
    }
    if n_baths is not None:
        attrs["n_baths"] = n_baths
    return attrs


# Output-header label for every key of the solver_line_attrs record, in print order.
_SETTINGS_LABELS = (
    ("n_baths", "Bath states per imp. orb."),
    ("excitation_budget", "Excitation budget"),
    ("bath_geometry", "Bath geometry"),
    ("collapse_chains", "Collapse chains"),
    ("fit_unocc", "Fit unoccupied states"),
    ("freeze_bath_energies", "Freeze bath energies"),
    ("gamma", "Fit regularization gamma"),
    ("weight_function", "Fitting weight function"),
    ("weight", "Fitting weight factor"),
    ("weight_w0", "Fitting weight center w0"),
    ("reort", "Reorthogonalizaion mode"),
    ("dense_cutoff", "Dense matrix size cutoff"),
    ("sparse_green", "Sparse Green's function"),
    ("gf_method", "Green's function method"),
    ("gf_admission", "GF basis admission"),
    ("gf_admit_tol", "GF admission threshold"),
    ("occ_cutoff", "Occupation cutoff"),
    ("dN", "dN"),
    ("mv", "Mixed valence"),
    ("chain_restrict", "Chain occ. restrictions"),
    ("slater_min", "Minimal Slater weight"),
    ("e_pt2_tol", "GS residual PT2 tolerance"),
    ("truncation_threshold", "Determinant cap (truncation_threshold)"),
)


def format_settings_header(nominal_occ, fit_options, basis, solver, n_baths=None, tau=None, delta=None):
    """The output-file header listing every solver setting of this run.

    Built from :func:`solver_line_attrs`, the same record the HDF5 archive stores, so the header
    and the archive cannot disagree; a key without a label in ``_SETTINGS_LABELS`` is still
    printed under its raw name rather than dropped.
    """
    attrs = solver_line_attrs(fit_options, basis, solver, n_baths=n_baths)
    if "truncation_threshold" in attrs:
        # Printed in the words the solver line takes, not as Python's None/inf.
        cap = attrs["truncation_threshold"]
        attrs["truncation_threshold"] = (
            "auto" if cap is None else ("unlimited" if not cap < float("inf") else f"{int(cap):,} (final)")
        )
    # Unspecified policy/threshold are printed in the words the solver line takes, not as Python's None.
    if attrs.get("gf_admission", "") is None:
        attrs["gf_admission"] = "auto"
    if attrs.get("gf_admit_tol", "") is None:
        attrs["gf_admit_tol"] = "default" if attrs.get("gf_admission") == "outer" else "n/a"
    rows = [("Nominal imp. occupation", nominal_occ)]
    labelled = set()
    for key, label in _SETTINGS_LABELS:
        if key in attrs:
            rows.append((label, attrs[key]))
            labelled.add(key)
    rows += [(key, value) for key, value in attrs.items() if key not in labelled]
    if tau is not None:
        rows.append(("Temperature tau", tau))
    if delta is not None:
        rows.append(("Real-axis broadening", delta))
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{width}} |> {value}" for label, value in rows) + "\n"


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


def get_weight_function(weight_function_name, w0, e):
    """
    Get the weight function matching the weight function name
    """
    return weight_functions[weight_function_name](w0, e)


def h5_write_dataset(group, name, data):
    """
    Create the dataset name in group, overwriting any existing dataset.
    Guards against 'name already exists' errors when re-entering a
    partially written iteration group (e.g. after a crash-restart, or when
    the DC and selfenergy passes share one iteration group).
    """
    if name in group:
        del group[name]
    group.create_dataset(name, data=data)


def _previous_damped_dc(hdf5_filename, label, comm):
    """The double counting this interface returned on the most recent earlier iteration.

    This -- not the ``sig_dc`` RSPt hands in -- is the anchor the damping has to pull toward.
    RSPt rebuilds ``sig_dc`` from its own FLL/AMF potential on every ``double_counting`` call
    and never stores our answer, so the incoming value carries no memory of what the criterion
    converged to last time.

    Searches the archive backwards from the current iteration for the newest group holding a
    ``"DC damped"`` dataset, so an iteration that skipped or failed the DC search does not
    break the chain. Returns ``None`` when there is no earlier answer (the first iteration),
    which the caller reads as "return the converged value undamped".

    Read on rank 0 and broadcast: every rank writes ``sig_dc`` from the result, so a rank-local
    value would return a different double counting per rank.
    """
    previous = None
    if comm is None or comm.rank == 0:
        try:
            with h5.File(hdf5_filename, "r") as f:
                current = int(f.attrs.get("last iteration", 1))
                for it in range(current - 1, 0, -1):
                    group_name = f"{label.strip()} {it}"
                    if group_name in f and "DC damped" in f[group_name]:
                        previous = np.asarray(f[group_name]["DC damped"])
                        break
        except (OSError, KeyError):
            # No archive yet (first call of a fresh run), or an unreadable one: no anchor.
            previous = None
    if comm is not None:
        previous = comm.bcast(previous, root=0)
    return previous


def _previous_bath_energies(hdf5_filename, label, block_structure):
    """The bath energies from the most recent stored fit, one array per inequivalent block.

    For ``freeze_bath_energies``: unlike the fingerprint-keyed reuse in ``fit_hyb_star``, this
    ignores the hybridization fingerprint and the iteration boundary -- it just wants the last
    energies the fitter produced, to hold them fixed while only the hoppings are re-solved
    against this iteration's hybridization. Returns ``None`` when nothing is stored yet (the
    first iteration), which the caller reads as "fall back to a full fit this once".

    Read on rank 0; the caller broadcasts. Energies are de-duplicated (the archive stores the
    post-``flatten_star_levels`` list, with one entry per coupled orbital component).
    """
    try:
        with h5.File(hdf5_filename, "r") as ar:
            current = int(ar.attrs.get("last iteration", 1))
            for it in range(current, 0, -1):
                fit_g = ar.get(f"{label.strip()} {it}/Bath fit")
                if fit_g is None or "ebs_star" not in fit_g:
                    continue
                return [
                    np.unique(np.asarray(fit_g[f"ebs_star/{block_index}"], dtype=float))
                    for block_index in block_structure.inequivalent_blocks
                ]
    except (OSError, KeyError):
        pass
    return None


def _record_dc_audit_attr(hdf5_filename, label, comm, attr, value):
    """Stamp one audit attribute onto this iteration's cluster group (rank 0 only).

    Used by the DC branch's failure handlers -- ``"DC search unreachable"`` (the target has
    no solution) and ``"DC search failed"`` (a solver blew up) -- so a charge-self-consistent
    run can afterwards be audited for which iterations actually determined a double counting
    and which fell back to the previous value. Creates the group if the self-energy pass has
    not written it yet. No-op on non-root ranks; the caller does not broadcast because nothing
    downstream reads the attribute back.
    """
    if comm is not None and comm.rank != 0:
        return
    with h5.File(hdf5_filename, "a") as f:
        it = f.attrs.get("last iteration", 1)
        group_name = f"{label.strip()} {it}"
        if group_name not in f:
            f.create_group(group_name)
        f[group_name].attrs[attr] = value


def _double_counting_sector(hdf5_filename, label, comm):
    """The charge sector the double-counting search settled on, for the current iteration.

    Written by the ``rspt_dc_flag=1`` call and read back by the ``rspt_dc_flag=0`` one, which
    RSPt makes in the same process a moment later. ``None`` when this iteration ran no DC search
    (a static scheme, an unreachable target, or a plain self-energy run).

    Read on rank 0 and broadcast: it decides the nominal occupation every rank builds its basis
    from, and a rank-local answer would have different ranks generating different determinants.
    """
    sector = None
    if comm is None or comm.rank == 0:
        try:
            with h5.File(hdf5_filename, "r") as f:
                it = int(f.attrs.get("last iteration", 1))
                group_name = f"{label.strip()} {it}"
                if group_name in f:
                    sector = f[group_name].attrs.get("DC ground state sector", None)
        except (OSError, KeyError):
            sector = None
    if comm is not None:
        sector = comm.bcast(sector, root=0)
    return None if sector is None else int(sector)


def _split_total_over_groups(total, nominal_occ):
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


def _parse_dc_line(dc_line):
    """Parse RSPt's 100-character double-counting line into its criterion and modifiers.

    Pulled out of :func:`run_impmod_ed` so the grammar can be tested without a cffi handle and a
    live RSPt call. Nothing about it changed in the move; the 100-character line itself is fixed
    by RSPt (it rewrites that line for DC in {-4,-5,-14,-15}), so the grammar can only be
    extended, never restructured.

    Returns
    -------
    (str, float or None, float, bool, float or None, float or None)
        The criterion name, its numeric target (``None`` where the criterion takes none), the
        damping factor, whether ``gap``/``peak`` narrow their sector solves to the ground
        multiplet, the PT2 admission floor for those sector solves, and the residual PT2 energy
        they converge to (``None`` for either = the solver's own default).
    """
    dc_line = dc_line.split("!")[0]
    dc_line = dc_line.split("#")[0]
    dc_array = dc_line.strip().split()

    # Optional damping, 'alpha X' anywhere on the line (B4): RSPt applies the returned DC
    # verbatim, with no mixing of its own against the previous iteration's value -- an
    # undamped full Newton step on a problem whose target (the charge density) moves
    # underneath it each CSC iteration, combined with a search that can return a point right
    # at a charge-sector boundary (dc_search._refine_bracket), is a d8/d9 limit-cycle
    # generator. Damp on our side, against OUR previous answer read back from the archive
    # (_previous_damped_dc): DC_new = DC_prev + alpha * (DC_converged - DC_prev), and
    # DC_new = DC_converged when there is no previous answer. Parsed and stripped before
    # the mode-specific parsing below, so it can follow either a peak position or 'occ [N]'.
    dc_alpha = 0.5
    for _i, _token in enumerate(dc_array):
        if _token.lower() == "alpha":
            assert _i + 1 < len(dc_array), "'alpha' on the double-counting line needs a value"
            dc_alpha = float(dc_array[_i + 1])
            del dc_array[_i : _i + 2]
            break

    # Optional 'ground_state_manifold' anywhere on the line, stripped the same way and for the
    # same reason: it modifies *how* a criterion solves, not which criterion runs, so it has to
    # compose with the spellings below rather than appear in each of their argument counts. Bare
    # flag, no value.
    #
    # What it does: 'gap' and 'peak' ask each charge sector for its whole thermal manifold
    # (`energy_cut(tau)`), then widen `num_wanted` until that window is exhausted. Their residual
    # only ever reads `min(es)`, so on a workload whose N +- 1 spectrum is dense inside the window
    # that widening is bought and thrown away -- measured on SrMnO3 cubic at a 512,000-determinant
    # cap, four stacked solves per sector ending at 160 states. Setting this asks for the
    # degenerate ground multiplet alone (`max_energy=0.0`).
    #
    # NOT free, and deliberately not the default: it also switches the criterion's reported
    # impurity occupation from the thermal average to the ground state's, which are only the same
    # where `occupation_spread` is negligible -- true on every fixture checked so far EXCEPT
    # SrMnO3 (up to 0.065). That spread feeds `delta_sum` and so the reported `mu` resolution, not
    # the root itself. Check it from a run with this flag OFF before turning it on.
    dc_ground_state_manifold = False
    for _i, _token in enumerate(dc_array):
        if _token.lower() == "ground_state_manifold":
            dc_ground_state_manifold = True
            del dc_array[_i]
            break

    # Optional 'de2_min X' anywhere on the line, stripped like 'alpha' and for the same reason:
    # it modifies how a criterion solves, not which criterion runs.
    #
    # What it does: the Epstein-Nesbet PT2 admission floor of the charge-sector CIPSI expansions.
    # Unset, those use `groundstate.GS_DE2_MIN` (1e-8), which buys *parity* -- the double counting
    # determined on the same variational space as the self-energy run that consumes it -- and
    # explicitly not accuracy. Measured on `nio_5peeled`, 1e-6 -> 1e-8 moves the gap centre 0.3 meV,
    # i.e. ~2.7 meV in `mu` after the 1/|chi| amplification, at 4x the cost; a cap change from
    # 2,000 to 8,000 moves it ~54 meV. Truncation drift dominates, the PT2 floor does not.
    #
    # Weigh that against the tolerance the search itself admits -- the SrMnO3 gap-DC record
    # reports "dc determined to +- 5.51e-02 by the search tolerance alone", so at 1e-8 the sector
    # energies are held ~20x finer than the answer is resolved. And the parity argument lapses
    # wherever the sectors are cap- or memory-bound rather than PT2-converged, which is exactly
    # the workload where the cost hurts.
    #
    # Loosening this is a BOUNDED approximation: the skipped weight is reported as
    # `subthreshold_de2_mass`. Lowering the determinant cap is not -- it truncates the basis and
    # loses spectral weight where the bath lives, which is how a self-energy turns non-causal.
    dc_de2_min = None
    for _i, _token in enumerate(dc_array):
        if _token.lower() == "de2_min":
            assert _i + 1 < len(dc_array), "'de2_min' on the double-counting line needs a value"
            dc_de2_min = float(dc_array[_i + 1])
            assert dc_de2_min > 0, "'de2_min' must be positive"
            del dc_array[_i : _i + 2]
            break

    # Optional 'e_pt2 X' (or 'e_pt2_tol X') anywhere on the line, stripped like 'de2_min'.
    #
    # What it does: the residual Epstein-Nesbet PT2 energy the charge-sector CIPSI expansions are
    # converged to -- the *summed* PT2 contribution of every determinant left out, which is what
    # the energy error follows (impurityModel's doc/plans/cipsi_pt2_convergence.md). This, not
    # 'de2_min', is the accuracy control: 'de2_min' bounds each refused determinant, not their
    # sum, and at 1e-8 alone left the SrMnO3 ground state 5.0e-5 above its converged energy.
    #
    # Unset, the sector solves inherit the solver line's own 'e_pt2' (else 1e-8), so the double
    # counting is measured on a space converged as far as the self-energy run's. Loosen it (e.g.
    # 1e-5) when the search is the cost -- on SrMnO3 converging the N-1 sector to 1e-8 is expected
    # to hit the memory guard. A solve that stops short of it warns with its residual.
    dc_e_pt2_tol = None
    for _i, _token in enumerate(dc_array):
        if _token.lower() in {"e_pt2", "e_pt2_tol"}:
            assert _i + 1 < len(dc_array), f"'{_token}' on the double-counting line needs a value"
            dc_e_pt2_tol = float(dc_array[_i + 1])
            assert dc_e_pt2_tol > 0, f"'{_token}' must be positive"
            del dc_array[_i : _i + 2]
            break

    # Double counting criteria:
    #   <peak_position>        -- place a spectral peak at the given energy
    #                             (E[N+1]-E[N] if positive, E[N]-E[N-1] if
    #                             negative)
    #   gap [offset]           -- centre mu_dc in the impurity gap, i.e. put the midpoint
    #                             (E[N+1]-E[N-1])/2 of the removal and addition excitations
    #                             at `offset` (default 0, the Fermi level). RECOMMENDED FOR
    #                             INSULATORS: Karolak et al. (arXiv:1004.4569) show the
    #                             occupation condition below "essentially breaks down" for a
    #                             charge-transfer insulator -- inside a gap the occupation is
    #                             flat, so a whole interval of mu satisfies it and none of
    #                             them is picked out. For NiO they get 25.3 eV this way
    #                             against 20.4 (SC)AMF, and show 21 eV is qualitatively wrong.
    #   occ <occupation>       -- fix the thermal impurity occupation (Karolak's Eq. 2;
    #                             the right criterion for metals)
    #   fll | amf | sigma_inf  -- static schemes (dc_static.py), evaluated at the DFT
    #                             reference occupation/density matrix (no ED solve)
    #   nominal                -- FLL at the NOMINAL (integer) occupation, not the DFT
    #                             reference (M4): needs no reference filling, so it cannot
    #                             saturate and cannot inherit the fit-resolution sensitivity
    #                             B1 measures for the other schemes. The natural dc_guess for
    #                             CSC iteration 1, or a reference to check a converged
    #                             fixed_occupation_dc answer against.
    # Any may be followed by 'alpha <value>' or 'de2_min <value>' (both parsed and removed
    # above; 'de2_min' only reaches the criteria that run charge-sector solves, so on a static
    # scheme it parses and is simply unused). 'gap' and 'peak' may also carry the bare flag
    # 'ground_state_manifold' and 'e_pt2 <value>' (likewise already parsed); every other
    # spelling rejects them rather than silently ignoring them -- see the assertions below for
    # why 'occ' is excluded on different grounds from the static schemes.
    _STATIC_SCHEMES = {"fll", "amf", "sigma_inf", "sigmainf", "nominal"}
    if len(dc_array) > 0 and dc_array[0].lower() == "gap":
        assert len(dc_array) <= 2, (
            "impurityModel gap double counting takes at most 1 argument, the offset of the "
            f"gap centre from the Fermi level. Got: {dc_array}"
        )
        dc_mode = "gap"
        dc_target = float(dc_array[1]) if len(dc_array) == 2 else 0.0
    elif len(dc_array) > 0 and dc_array[0].lower() in {"occ", "occupation"}:
        assert len(dc_array) <= 2, (
            "impurityModel occupation double counting takes at most 1 "
            f"argument, the target impurity occupation. Got: {dc_array}"
        )
        dc_mode = "occupation"
        dc_target = None
        if len(dc_array) == 2:
            dc_target = float(dc_array[1])
    elif len(dc_array) > 0 and dc_array[0].lower() in _STATIC_SCHEMES:
        assert len(dc_array) == 1, (
            "impurityModel static double counting (fll, amf, sigma_inf, nominal) takes no "
            f"further arguments (besides 'alpha <value>', already parsed). Got: {dc_array}"
        )
        dc_mode = dc_array[0].lower()
        dc_target = None
    else:
        assert len(dc_array) == 1, (
            "impurityModel double counting takes 1 argument, peak_position, "
            "'gap [offset]', 'occ [target impurity occupation]', or one of "
            "fll/amf/sigma_inf/nominal. "
            f"Got: {dc_array}"
        )
        dc_mode = "peak"
        dc_target = float(dc_array[0])

    # Only 'gap' and 'peak' take it, and the exclusions are two different facts:
    #
    #   * the static schemes (fll/amf/sigma_inf/nominal) run no ground-state solve at all, so
    #     there is no manifold to narrow;
    #   * 'occ' runs plenty of them, but its observable *is* the thermal impurity occupation
    #     (`_evaluate_occupation_and_energy_at_mu` -> `thermal_average_scale_indep`), so
    #     narrowing the manifold would change the criterion rather than the cost of evaluating
    #     it. `fixed_occupation_dc` accordingly does not accept the argument -- it runs through
    #     `_OccupationContext`/`solve_ground_state`, not `_SectorContext.sector_solve`.
    #
    # Rejected rather than ignored either way: a flag that reports as set and does nothing is
    # the exact failure mode this file already documents for a misnamed knob.
    assert not (dc_de2_min is not None and dc_mode in _STATIC_SCHEMES), (
        "'de2_min' sets the PT2 admission floor of the charge-sector CIPSI solves, and the static "
        f"double-counting schemes run no solve at all -- '{dc_mode}' would ignore it silently. "
        "Remove it, or pick a criterion that solves ('gap', 'peak' or 'occ')."
    )
    assert not (dc_ground_state_manifold and dc_mode not in {"gap", "peak"}), (
        "'ground_state_manifold' applies to the 'gap' and 'peak' double-counting criteria, whose "
        "residual reads only each sector's lowest energy. "
        + (
            "'occ' pins the thermal impurity occupation itself, so narrowing the manifold would "
            "change the criterion, not just its cost."
            if dc_mode == "occupation"
            else f"The static scheme '{dc_mode}' runs no ground-state solve."
        )
        + " Remove it from the double-counting line."
    )
    # 'occ' is excluded on the parity ground: it solves on the production ground-state path, so
    # it converges to the solver line's 'e_pt2' and only to that -- the same tolerance the
    # self-energy run's ground state uses. Rejected, not ignored, like the flag above.
    assert not (dc_e_pt2_tol is not None and dc_mode not in {"gap", "peak"}), (
        "'e_pt2' on the double-counting line sets the convergence of the 'gap' and 'peak' "
        "criteria's charge-sector solves. "
        + (
            "'occ' solves on the production ground-state path and converges to the solver line's "
            "'e_pt2', so set it there."
            if dc_mode == "occupation"
            else f"The static scheme '{dc_mode}' runs no solve at all."
        )
        + " Remove it from the double-counting line."
    )

    return dc_mode, dc_target, dc_alpha, dc_ground_state_manifold, dc_de2_min, dc_e_pt2_tol


def _run_impmod_ed(
    rspt_label,
    rspt_solver_line,
    rspt_dc_line,
    rspt_dc_flag,
    rspt_u4,
    rspt_hyb,
    rspt_h_dft,
    rspt_sig,
    rspt_sig_real,
    rspt_sig_static,
    rspt_sig_dc,
    rspt_iw,
    rspt_w,
    rspt_corr_to_spherical,
    rspt_corr_to_cf,
    n_orb,
    n_rot_cols,
    n_orb_full,
    n_iw,
    n_w,
    eim,
    tau,
    verbosity,
    size_real,
    size_complex,
):
    """Wrap RSPt's raw pointers as numpy views and hand them to :func:`_solve`.

    The only function in this module that touches a cffi handle. Everything the physics
    needs -- decoded strings and aliased numpy arrays -- is produced here, so :func:`_solve`
    is plain Python and can be called directly from a test.

    The views alias RSPt's memory; they are never copies. The four outputs (``sig``,
    ``sig_real``, ``sig_static``, ``sig_dc``) are written back by in-place assignment, and
    the inputs matter just as much -- the ``comm.Bcast`` calls in :func:`_solve` write
    *through* ``h_dft``, ``u4`` and ``hyb`` into RSPt's own memory on non-root ranks. An
    ``np.asarray(..., dtype=complex)`` anywhere below would silently break both, and no
    single-rank test would notice.

    ``n_iw``, ``n_w``, ``size_real`` and ``size_complex`` exist only to compute the byte
    counts below, and are deliberately not forwarded.
    """
    label = ffi.string(rspt_label, 18).decode("ascii")
    solver_line = ffi.string(rspt_solver_line, 100).decode("ascii")
    dc_line = ffi.string(rspt_dc_line, 100).decode("ascii")

    h_dft = np.ndarray(
        buffer=ffi.buffer(rspt_h_dft, n_orb * n_orb * size_complex),
        shape=(n_orb, n_orb),
        order="F",
        dtype=complex,
    )
    u4 = np.ndarray(
        buffer=ffi.buffer(rspt_u4, n_orb * n_orb * n_orb * n_orb * size_complex),
        shape=(n_orb, n_orb, n_orb, n_orb),
        order="F",
        dtype=complex,
    )
    hyb = np.ndarray(
        buffer=ffi.buffer(rspt_hyb, n_w * n_orb * n_orb * size_complex),
        shape=(n_orb, n_orb, n_w),
        order="F",
        dtype=complex,
    )
    iw = np.ndarray(buffer=ffi.buffer(rspt_iw, n_iw * size_real), shape=(n_iw,), dtype=float)
    w = np.ndarray(buffer=ffi.buffer(rspt_w, n_w * size_real), shape=(n_w,), dtype=float)
    sig = np.ndarray(
        buffer=ffi.buffer(rspt_sig, n_iw * n_orb * n_orb * size_complex),
        shape=(n_orb, n_orb, n_iw),
        order="F",
        dtype=complex,
    )
    sig_real = np.ndarray(
        buffer=ffi.buffer(rspt_sig_real, n_w * n_orb * n_orb * size_complex),
        shape=(n_orb, n_orb, n_w),
        order="F",
        dtype=complex,
    )
    sig_static = np.ndarray(
        buffer=ffi.buffer(rspt_sig_static, n_orb * n_orb * size_complex),
        shape=(n_orb, n_orb),
        order="F",
        dtype=complex,
    )
    sig_dc = np.ndarray(
        buffer=ffi.buffer(rspt_sig_dc, n_orb * n_orb * size_complex),
        shape=(n_orb, n_orb),
        order="F",
        dtype=complex,
    )
    rspt_corr_to_spherical_arr = np.ndarray(
        buffer=ffi.buffer(rspt_corr_to_spherical, n_orb * n_orb_full * size_complex),
        shape=(n_orb, n_orb_full),
        order="F",
        dtype=complex,
    )
    rspt_corr_to_cf_arr = np.ndarray(
        buffer=ffi.buffer(rspt_corr_to_cf, n_orb * n_rot_cols * size_complex),
        shape=(n_orb, n_rot_cols),
        order="F",
        dtype=complex,
    )

    return _solve(
        label=label,
        solver_line=solver_line,
        dc_line=dc_line,
        dc_flag=rspt_dc_flag,
        h_dft=h_dft,
        u4=u4,
        hyb=hyb,
        iw=iw,
        w=w,
        sig=sig,
        sig_real=sig_real,
        sig_static=sig_static,
        sig_dc=sig_dc,
        corr_to_spherical_in=rspt_corr_to_spherical_arr,
        corr_to_cf_in=rspt_corr_to_cf_arr,
        n_orb=n_orb,
        n_rot_cols=n_rot_cols,
        n_orb_full=n_orb_full,
        eim=eim,
        tau=tau,
        verbosity=verbosity,
    )


def _solve(
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
    """Run the double-counting or selfenergy solve on already-marshalled arrays.

    Pure Python: ``label``/``solver_line``/``dc_line`` are ``str`` and every array is a numpy
    array. Split out of :func:`_run_impmod_ed` so it can be exercised without a cffi handle.

    The arrays are the caller's, and are mutated in place -- see :func:`_run_impmod_ed`.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank if comm is not None else 0

    corr_to_spherical, corr_to_cf = reconstruct_rotations(
        corr_to_spherical_in,
        corr_to_cf_in,
        n_orb,
        n_rot_cols,
        n_orb_full,
    )
    comm.Bcast(corr_to_spherical)
    comm.Bcast(corr_to_cf)
    comm.Bcast(h_dft)
    comm.Bcast(u4)

    # For python, it makes more sense to put the frequency index first, instead of last
    sig_python = np.moveaxis(sig, -1, 0)
    sig_real_python = np.moveaxis(sig_real, -1, 0)
    comm.Bcast(hyb)
    hyb = np.moveaxis(hyb, -1, 0)

    hyb = rotate_Greens_function(hyb, corr_to_cf)
    h_dft = rotate_matrix(h_dft, corr_to_cf)
    u4 = rotate_4index_U(u4, corr_to_cf)

    stdout_save = sys.stdout
    # sys.stdout is redirected to a file that must stay open for the remainder
    # of the call (restored from stdout_save at the end), so a context manager
    # is intentionally not used here.
    if rank == 0:
        sys.stdout = open(  # noqa: SIM115
            f"impurityModel-{label.strip()}{'-dc' if dc_flag == 1 else ''}.out",
            "w",
        )
    elif verbosity > 0:
        sys.stdout = open(  # noqa: SIM115
            f"impurityModel-{label.strip()}{'-dc' if dc_flag == 1 else ''}-{rank}.out",
            "w",
        )
    else:
        sys.stdout = open(devnull, "w")
    verbosity = comm.bcast(verbosity)

    hdf5_filename = "impurityModel_data.h5"
    _nominal_occ, bath_states_per_orbital, fit_options, basis, solver = parse_solver_line(solver_line)
    # tau is not part of the solver line; inject the external temperature onto the basis options.
    basis = replace(basis, tau=tau)
    print(
        format_settings_header(
            _nominal_occ, fit_options, basis, solver, n_baths=bath_states_per_orbital, tau=tau, delta=eim
        ),
        flush=True,
    )
    nominal_occ = basis.nominal_occ
    if any(n0 > n_orb for n0 in nominal_occ.values()) or any(n0 < 0 for n0 in nominal_occ.values()):
        raise RuntimeError(f"Nominal impurity occupation {nominal_occ} out of bounds [0, {n_orb}]")

    if dc_flag != 1:
        # Seed the ground-state search with the charge sector the double-counting search settled
        # on for this same iteration. The DC is only meaningful if this solve lands on the state
        # the DC was measured against, and the sector walk is the one step that can legitimately
        # land elsewhere -- it is a discrete choice, so a small change in the incoming
        # hybridization can flip it. RSPt calls run_impmod_ed twice per CSC iteration in one
        # process, so the answer from the first call is available here; the walk still runs and
        # can still move, but it starts where the DC search finished rather than at RSPt's
        # nominal.
        dc_sector = _double_counting_sector(hdf5_filename, label, comm)
        if dc_sector is not None and dc_sector != sum(nominal_occ.values()):
            if rank == 0:
                print(
                    f"Seeding the ground-state search at the double-counting search's sector "
                    f"{dc_sector} (nominal {sum(nominal_occ.values())})."
                )
            nominal_occ = _split_total_over_groups(dc_sector, nominal_occ)
            basis = replace(basis, nominal_occ=nominal_occ)

    if abs(w[1] - w[0]) > eim / 2 and rank == 0:
        print(
            "WARNING: Your real frequency mesh is rather coarse. Recommended dE <= eim/5, "
            "in order to guarantee resolution of fine structures in the self energy."
        )
    # The solver works in the CF basis; RSPt sends and expects quantities in
    # the corr basis. Log how different the two bases are, so basis mixups are
    # visible in the output file.
    corr_to_cf_dev = np.max(np.abs(corr_to_cf - np.eye(n_orb)))
    if rank == 0:
        print(f"Max abs deviation of corr_to_cf from identity: {corr_to_cf_dev:.3e}")
        if corr_to_spherical.shape[0] != corr_to_spherical.shape[1]:
            print(
                "The correlated orbitals span only part of the shell "
                f"({corr_to_spherical.shape[0]} of {corr_to_spherical.shape[1]} "
                "spin-orbitals); the solver will skip the L/S/J observables."
            )
    # RSPt supplies the double counting in the corr basis, the solver needs it
    # in the CF basis.
    sig_dc_cf = rotate_matrix(sig_dc, corr_to_cf)

    (
        H_imp,
        impurity_indices,
        valence_bath_indices,
        conduction_bath_indices,
        v,
        H_bath,
        H_solver,
    ) = get_ed_h0(
        h_dft,
        hyb,
        bath_states_per_orbital,
        w,
        eim,
        tau,
        gamma=fit_options["gamma"],
        weight_function=fit_options["weight_function"],
        weight_w0=fit_options["weight_w0"],
        exp_weight=fit_options["weight"],
        imag_only=False,
        valence_bath_only=not fit_options["fit_unocc"],
        bath_geometry=fit_options["bath_geometry"],
        freeze_bath_energies=fit_options["freeze_bath_energies"],
        label=label.strip(),
        hdf5_filename=hdf5_filename,
        verbose=(verbosity >= 1 or dc_flag == 1),
        extra_verbose=(verbosity >= 2),
        comm=comm,
    )
    # Build one ImpurityModel from the fitted blocks: it assembles the operator and derives the
    # impurity/bath orbital layout from the block sizes (no orbital indices passed). The
    # (valence, conduction) split feeds the double-counting path (calc_selfenergy re-derives its
    # own from h0). model.h0 is the raw h_dft + bath (RSPt passes h_dft WITHOUT the double
    # counting subtracted) and model.dc is RSPt's current dc: calc_selfenergy subtracts it
    # (h0 - dc + U) and the fixed_*_dc searches use it as dc_guess -- their DFT reference
    # occupation is the Fermi filling of the raw h0, independent of model.dc.
    rot_to_spherical = np.conj(corr_to_cf.T) @ corr_to_spherical
    model = ImpurityModel.from_blocks(
        H_imp,
        v,
        H_bath,
        u4=u4,
        dc=sig_dc_cf,
        rot_to_spherical=rot_to_spherical,
        bath_valence_conduction=(valence_bath_indices, conduction_bath_indices),
    )
    if comm.rank == 0:
        # h5py cannot store None attributes
        opt = {
            key: value if value is not None else "None"
            for key, value in solver_line_attrs(fit_options, basis, solver, n_baths=bath_states_per_orbital).items()
        }
        with h5.File(hdf5_filename, "a") as f:
            if "last iteration" not in f.attrs:
                f.attrs["last iteration"] = 1
            it = f.attrs["last iteration"]

            if f"{label.strip()} {it}" not in f:
                f.create_group(f"{label.strip()} {it}")
            cluster_g = f[f"{label.strip()} {it}"]
            cluster_g.attrs.update(opt)
            cluster_g.attrs["tau"] = tau
            cluster_g.attrs["delta"] = eim
            cluster_g.attrs["nominal occupation"] = nominal_occ[0]
            # Provenance, for reproducing results
            cluster_g.attrs["solver line"] = solver_line.strip()
            for package in ("impurityModel", "rspt2spectra", "impurityModel_interface"):
                try:
                    cluster_g.attrs[f"{package} version"] = package_version(package)
                except PackageNotFoundError:
                    pass
            h5_write_dataset(cluster_g, "Real frequency mesh", w)
            h5_write_dataset(cluster_g, "Matsubara frequency mesh", iw)
            h5_write_dataset(cluster_g, "Rot to spherical", rot_to_spherical)
            h5_write_dataset(cluster_g, "Impurity orbitals", impurity_indices)
            h5_write_dataset(cluster_g, "Valence orbitals", valence_bath_indices)
            h5_write_dataset(cluster_g, "Conduction orbitals", conduction_bath_indices)

            h5_write_dataset(cluster_g, "H DFT", h_dft)
            h5_write_dataset(cluster_g, "H bath", H_bath)
            h5_write_dataset(cluster_g, "V", v)
            h5_write_dataset(cluster_g, "DC", sig_dc_cf)
            h5_write_dataset(cluster_g, "U", u4)
            # The full one-particle solver hamiltonian (impurity + bath, CF
            # basis) and the corr -> CF rotation; together they make the
            # archive self-contained for reproducing the solver run.
            h5_write_dataset(cluster_g, "H solver", H_solver)
            h5_write_dataset(cluster_g, "corr_to_cf", corr_to_cf)
    if verbosity >= 1 and rank == 0:
        hyb_fit = np.conj(v).T @ np.linalg.solve(
            (w + 1j * eim)[:, None, None] * np.identity(H_bath.shape[0], dtype=complex)[None, :, :] - H_bath,
            v,
        )
        # Report the fit quality. Unoccupied states are only fitted with
        # fit_unocc, so also report the deviation on the occupied side alone.
        fit_dev = np.max(np.abs(hyb_fit - hyb))
        fit_dev_occ = np.max(np.abs(hyb_fit[w <= 0] - hyb[w <= 0]))
        print(
            f"Max abs deviation of the fitted hybridization function: {fit_dev_occ:.6f} (w <= 0), {fit_dev:.6f} (all w)"
        )
        save_Greens_function(
            rotate_Greens_function(hyb_fit, np.conj(corr_to_cf.T)),
            w,
            "hyb-fit",
            label.strip(),
        )
        # B1: is the occupation fixed_occupation_dc pins to the true DFT impurity occupation, or
        # an artefact of how finely the bath was discretized? Diagnostic only -- it does not
        # change the criterion's target; see dc_reference.report_continuum_reference's docstring.
        n0_disc = discretized_impurity_occupation(model, tau)
        report_continuum_reference(h_dft, hyb, hyb_fit, w, eim, tau, n0_disc, rank=rank)

    if dc_flag == 1:
        dc_mode, dc_target, dc_alpha, dc_gs_manifold, dc_de2_min, dc_e_pt2_tol = _parse_dc_line(dc_line)

        # The charge sector the ED criteria settle on, handed to this iteration's self-energy
        # call so it starts from the state the DC was measured against (see
        # _double_counting_sector). None for the static schemes, which run no ground-state solve.
        dc_sector = None
        try:
            if dc_mode == "occupation":
                # Scale the shift search with the real-frequency mesh, so the
                # steps are sensible in any energy unit (RSPt supplies Ry).
                bandwidth = w[-1] - w[0]
                dc_cf, dc_sector = fixed_occupation_dc(
                    model,
                    basis,
                    solver,
                    occupation=dc_target,
                    comm=comm,
                    verbosity=verbosity,
                    initial_step=bandwidth / 100,
                    max_shift=bandwidth,
                    return_sector=True,
                )
            elif dc_mode == "gap":
                dc_cf, dc_sector = fixed_gap_dc(
                    model,
                    basis,
                    solver,
                    offset=dc_target,
                    comm=comm,
                    verbosity=verbosity,
                    ground_state_manifold=dc_gs_manifold,
                    de2_min=dc_de2_min,
                    e_pt2_tol=dc_e_pt2_tol,
                    return_sector=True,
                )
            elif dc_mode == "peak":
                dc_cf, dc_sector = fixed_peak_dc(
                    model,
                    basis,
                    solver,
                    peak_position=dc_target,
                    comm=comm,
                    verbosity=verbosity,
                    ground_state_manifold=dc_gs_manifold,
                    de2_min=dc_de2_min,
                    e_pt2_tol=dc_e_pt2_tol,
                    return_sector=True,
                )
            elif dc_mode == "fll":
                dc_cf = fll_dc(model, tau=tau)
            elif dc_mode == "amf":
                dc_cf = amf_dc(model, tau=tau)
            elif dc_mode in ("sigma_inf", "sigmainf"):
                dc_cf = sigma_inf_dc(model, tau=tau)
            else:
                assert dc_mode == "nominal"
                dc_cf = nominal_dc(model, sum(nominal_occ.values()))
            # Damp against OUR OWN previous answer, read back from the archive -- not against
            # sig_dc_cf. sig_dc_cf is not a continuation of anything we returned: RSPt zeroes
            # sig_dc at the top of every double_counting call (green_double_counting.F90:73)
            # and refills it with its own FLL/AMF `alocal` (:340-371, trace-renormalised against
            # solver_wisdom at :406-425), and impurityModel's answer is never written back into
            # solver_wisdom. Damping toward it therefore returned `alocal + alpha*mu` -- a double
            # counting that does NOT satisfy the criterion the search just converged, on every
            # CSC iteration. With no previous answer on record (the first iteration) there is
            # nothing to damp against, so the converged value is returned undamped.
            previous_dc_cf = _previous_damped_dc(hdf5_filename, label, comm)
            if previous_dc_cf is None:
                damped_dc_cf = dc_cf
            else:
                damped_dc_cf = previous_dc_cf + dc_alpha * (dc_cf - previous_dc_cf)
            if comm.rank == 0:
                # Persist the full damped DC matrix (never the bare mu) and fingerprint it
                # against the dc_guess it was computed from, so the archive can be audited for
                # whether the next iteration's incoming DC is actually a continuation of this
                # one's answer or something else changed it underneath.
                with h5.File(hdf5_filename, "a") as f:
                    it = f.attrs.get("last iteration", 1)
                    group_name = f"{label.strip()} {it}"
                    if group_name not in f:
                        f.create_group(group_name)
                    cluster_g = f[group_name]
                    cluster_g.attrs["DC damping alpha"] = dc_alpha if previous_dc_cf is not None else 1.0
                    cluster_g.attrs["DC guess fingerprint"] = hashlib.sha256(
                        np.ascontiguousarray(sig_dc_cf).tobytes()
                    ).hexdigest()
                    # The value actually converged by the criterion, before damping. "DC damped"
                    # is what RSPt receives; this is what the criterion says the answer is, and
                    # the two coincide on the first iteration by construction.
                    h5_write_dataset(cluster_g, "DC converged", dc_cf)
                    h5_write_dataset(cluster_g, "DC damped", damped_dc_cf)
                    if dc_sector is not None:
                        cluster_g.attrs["DC ground state sector"] = int(dc_sector)
            # A second record, in the same format and through the same formatter, for the one
            # number the criterion could not report: it printed the dc it *converged*, and what
            # RSPt receives is that value damped toward the previous iteration's answer. A reader
            # taking `dc_level` from the criterion's block as "what was applied" would be wrong on
            # every iteration after the first. A bare adjacent line was the first attempt and is
            # worse than useless for the stated goal -- it falls outside the delimiters, so
            # anything parsing between header and footer never sees it.
            applied_alpha = dc_alpha if previous_dc_cf is not None else 1.0
            applied = {
                "criterion": f"{dc_mode} (applied)",
                "status": "damped" if previous_dc_cf is not None else "undamped",
                "alpha": applied_alpha,
            }
            applied["dc_trace"], applied["dc_level"] = dc_levels(damped_dc_cf)
            applied["dc_spread"] = dc_spread(damped_dc_cf)
            emit_dc_record(applied, rank=comm.rank)
            # The double counting is calculated in the CF basis, RSPt expects
            # it in the corr basis.
            sig_dc[:, :] = rotate_matrix(damped_dc_cf, np.conj(corr_to_cf.T))
            er = 0
        except DoubleCountingUnreachable as e:
            # A modelling verdict, not a solver failure (dc_search.DoubleCountingUnreachable's
            # docstring): the target has no solution with this bath/truncation. sig_dc was never
            # written above, so it already holds RSPt's incoming DC unchanged -- distinguish that
            # from success (er=0 with a silent, indistinguishable "nothing happened") with an
            # unmistakable banner and a record in the archive, so a CSC run can be audited
            # afterward for which iterations actually determined a DC.
            print("!" * 100)
            print(f"Exception {e!r} caught on rank {rank}!")
            print("DOUBLE COUNTING SEARCH COULD NOT REACH ITS TARGET. Returning initial DC unchanged.")
            print("!" * 100, flush=True)
            _record_dc_audit_attr(hdf5_filename, label, comm, "DC search unreachable", str(e))
            er = 0
        except Exception as e:
            # A solver failure inside the DC search (a Lanczos/SVD non-convergence, an
            # out-of-memory sector solve, ...) is not a reason to destroy a multi-hour CSC
            # run: like the DoubleCountingUnreachable branch above, leave sig_dc holding
            # RSPt's incoming DC unchanged (it was never written on this path), record the
            # failure in the archive so the run can be audited for which iterations actually
            # determined a DC, and let the self-consistency loop carry on with the previous
            # value. The DC search raises rank-synchronously by construction (its verdicts are
            # broadcast and genuinely rank-local failure conditions Abort rather than raise --
            # see dc_search / dc_criteria), so every rank reaches this handler together and a
            # plain continue does not desync the collective barrier at the end of the call.
            print("!" * 100)
            print(f"Exception {e!r} caught on rank {rank}!")
            print(traceback.format_exc())
            print(
                "DOUBLE COUNTING SEARCH FAILED. Returning initial DC unchanged.",
                flush=True,
            )
            print("!" * 100)
            _record_dc_audit_attr(hdf5_filename, label, comm, "DC search failed", f"{e!r}\n{traceback.format_exc()}")
            er = 0
    else:
        try:
            # The ImpurityModel and the basis/solver option groups were built above (from the
            # fitted blocks and the parsed solver line); only the frequency meshes are per-call.
            # No file round-trip: the solver is called directly in memory.
            meshes = Meshes(iw=1j * iw, w=w, delta=eim)
            results = calc_selfenergy(
                model,
                meshes,
                basis,
                solver,
                comm=comm,
                verbosity=verbosity,
                cluster_label=label.strip(),
            )
            if comm.rank == 0:
                # calc_selfenergy returns everything in its input (CF) basis,
                # RSPt expects the selfenergy in the corr basis.
                u = np.conj(corr_to_cf.T)
                sig_static[:, :] = rotate_matrix(results["sigma_static"], u)
                sig_python[:, :, :] = rotate_Greens_function(results["sigma"], u)
                sig_real_python[:, :, :] = rotate_Greens_function(results["sigma_real"], u)

            comm.Bcast(sig_static, root=0)
            comm.Bcast(sig_real, root=0)
            comm.Bcast(sig, root=0)

            if comm.rank == 0:
                with h5.File(hdf5_filename, "a") as f:
                    it = f.attrs["last iteration"]
                    cluster_g = f[f"{label.strip()} {it}"]
                    h5_write_dataset(cluster_g, "thermal_rho", results["thermal_rho"])
                    h5_write_dataset(cluster_g, "rhos", results["rhos"])
                    h5_write_dataset(cluster_g, "Sigma Static", sig_static)
                    h5_write_dataset(cluster_g, "Sigma real", sig_real_python)
                    h5_write_dataset(cluster_g, "Sigma Matsubara", sig_python)
                    if results["gs_matsubara"] is not None:
                        h5_write_dataset(
                            cluster_g,
                            "Gimp Matsubara",
                            rotate_Greens_function(results["gs_matsubara"], u),
                        )
                    if results["gs_realaxis"] is not None:
                        h5_write_dataset(
                            cluster_g,
                            "Gimp real",
                            rotate_Greens_function(results["gs_realaxis"], u),
                        )

            er = 0

        except Exception as e:
            print("!" * 100)
            print(f"Exception {e!r} caught on rank {rank}!")
            print(traceback.format_exc())
            print(
                "Adding positive infinity to the imaginary part of the selfenergy at the last matsubara frequency.",
                flush=True,
            )
            print("!" * 100)
            sig[:, :, -1] += 1j * np.inf
            er = -1
            comm.Abort(er)
        else:
            print("Self energy calculated! impurityModel shutting down.", flush=True)

    sys.stdout.close()
    sys.stdout = stdout_save
    comm.barrier()
    return er


# ``error=-1`` is load bearing. cffi's default for an uncaught exception in a callback is to
# report it as unraisable and return **0**, and RSPt reads 0 as success
# (``green_impmod_interface.F90``: ``if (er .ne. 0) call stopgreen(...)``), so a failed solve
# would let the DMFT loop carry on with a stale ``acluster%sig``. The handler below is what
# normally fires; this is the backstop for anything that escapes it, including a failure
# inside the handler itself.
def bind_rspt_callback():
    """Attach :func:`run_impmod_ed` to the extern "Python" slot, if import time could not.

    Normally the decorator below does this: RSPt's first call starts the interpreter, cffi
    registers the ``run_impurityModel`` module, and only then does ``embedding_init_code``
    import this package -- so ``from run_impurityModel import ffi`` at the top of this file
    succeeds and the decorator is the real one.

    Anything that imports this package *earlier* breaks that. A ``sitecustomize.py``, a
    ``usercustomize.py`` or a ``.pth`` file on the path runs during ``Py_InitializeEx``,
    before cffi has registered its module, so the import at the top falls back to
    ``_FFIStub`` and the decorator becomes a no-op. cffi then answers every call with
    ``"no code was attached to it yet ... Returning 0"`` -- RSPt reads 0 as a successful
    solve, which is the exact failure this module exists to prevent, arriving by a third
    route that neither ``error=-1`` nor the ``cffi_start_python()`` check in
    ``library_builder.py`` can see.

    ``embedding_init_code`` calls this after the import, when the module is guaranteed to
    exist. It is a no-op on the normal path.
    """
    # Both of these are the point of the function, not an oversight: the module-level import
    # is what failed, so the retry has to be deferred to here, and rebinding the module global
    # is what makes ffi.string/ffi.buffer in _run_impmod_ed resolve to the real handle.
    global ffi  # noqa: PLW0603
    from run_impurityModel import ffi as embedded_ffi  # noqa: PLC0415

    if ffi is embedded_ffi:
        return
    ffi = embedded_ffi
    embedded_ffi.def_extern("run_impmod_ed_py", error=-1)(run_impmod_ed)


@ffi.def_extern("run_impmod_ed_py", error=-1)
def run_impmod_ed(*args):
    """Entry point RSPt calls, wrapping the solve in whatever ``[environment]`` asks for.

    RSPt configures the solver through two fixed-width strings and a label, and this callback
    has no argument to pass a file path through -- nor are RSPt's own sources ours to change.
    So the input file is found by convention: ``impurityModel.toml`` in the working directory,
    or wherever ``IMPURITYMODEL_INPUT`` points. Only its ``[environment]`` table is read; the
    rest of such a file describes a model, and on this path RSPt supplies the model itself, so
    reading more would create two sources of truth for the same physics.

    Two rules this path needs and the command line does not:

    * The knobs are **restored on the way out**. This callback runs once per cluster label per
      self-consistency iteration, plus again for the double-counting pass, so a knob left set
      would silently carry into the next one.
    * A variable **already set in the environment wins**, and is reported as skipped -- the
      same rule this module already applies to ``OMP_NUM_THREADS`` above. A value someone
      exported in their submit script should not be quietly overridden by a file.
    """
    stdout_save = sys.stdout
    try:
        comm = MPI.COMM_WORLD
        path = find_environment_file()
        knobs = load_environment(path, comm=comm) if path else {}
        with apply_environment(knobs, override=False) as skipped:
            if knobs and comm.rank == 0:
                print(f"Applying {len(knobs)} tuning knob(s) from {path}.", file=sys.stderr)
                for name in skipped:
                    print(f"  {name}: kept the value already set in the environment.", file=sys.stderr)
            return _run_impmod_ed(*args)
    except BaseException:
        # Diagnostics go to sys.__stderr__, never print(): _solve redirects sys.stdout to the
        # per-cluster .out file, so an exception raised while that redirect is live would send
        # the traceback into the file nobody is watching -- or into a closed handle.
        # BaseException, not Exception: a MemoryError or a signal mid-solve must still reach
        # RSPt as a failure rather than as a silent success.
        stream = sys.__stderr__ if sys.__stderr__ is not None else stdout_save
        try:
            rank = MPI.COMM_WORLD.rank
        except Exception:
            rank = "?"
        print("!" * 100, file=stream)
        print(f"impurityModel: unhandled exception on rank {rank}; returning -1 to RSPt.", file=stream)
        traceback.print_exc(file=stream)
        print("!" * 100, file=stream, flush=True)
        return -1
    finally:
        # _solve restores sys.stdout itself on every path it runs to completion. This is the
        # net for the paths it does not: without it a failure leaves sys.stdout pointing at a
        # half-written .out file for the rest of the process, and the diagnostic for the very
        # failure being reported is the thing that gets truncated.
        if sys.stdout is not stdout_save:
            try:
                sys.stdout.close()
            except Exception:
                pass
            sys.stdout = stdout_save


def get_ed_h0(
    H_dft,
    hyb,
    bath_states_per_orbital,
    w,
    eim,
    tau,
    gamma=0.001,
    exp_weight=2,
    weight_function="unit",
    weight_w0=0,
    imag_only=False,
    valence_bath_only=True,
    bath_geometry="peeled_linked_chain",
    freeze_bath_energies=False,
    label=None,
    hdf5_filename="impurityModel_data.h5",
    verbose=True,
    extra_verbose=False,
    comm=None,
):
    """
    Calculate the non-interacting hamiltonian, h0, for use in exact diagonalization.
    Bath states are fitted to the real frequency hybridization function.
    In block form h0 can be written
    [ h_dft  V^+ ]
    [  V     Eb ],
    where h_dft is the dft hamiltonian projected onto the correlated orbitals, V is
    the hopping amplitudes between the impurity and the bath, and Eb is a diagonal
    matrix with energies of the bath states along the diagonal.
    Parameters:
    hyb           -- The real frequency hybridiaztion function, in the CF basis. Used to fit the bath states.
    hdft          -- The DFT hamiltonian, projected onto the impurity orbitals, in the CF basis.
    sig_dc        -- The double counting correction, in the CF basis (or scalar 0).
    bath_states   -- Number of bath states to fit per impurity orbital.
    w             -- Real frequency mesh.
    eim           -- All real frequency quantities are evaluated i*eim above the real frequency axis.
    gamma         -- Regularization parameter.
    imag_only     -- Currently ignored; rspt2spectra's fit_hyb does not support
                     fitting only the imaginary part. Kept for API stability.
    valence_bath_only -- Only fit bath stated in the valence band, default: True.
    label          -- Label for the cluster, used for saving a copy of the Hamiltonian
                      that can be plugged into the Matsubara ED solver in RSPt, default: None,

    Returns:
    A tuple ``(H_imp, impurity_indices, valence_bath_indices, conduction_bath_indices,
    v_solver, H_bath, H)``: the effective impurity block, the orbital-index classification,
    the impurity-bath hopping ``v_solver``, the bath block ``H_bath``, and the full assembled
    solver matrix ``H``. Hand ``(H_imp, v_solver, H_bath)`` to
    ``ImpurityModel.from_blocks`` to build the model.
    """

    rank = 0 if comm is None else comm.rank
    # rspt2spectra block-diagonalizes the hybridization function, rotates the
    # local hamiltonian into the same (fitting) basis and builds the block
    # partition from the union of both connectivities.
    Q, phase_hyb, H_local_Q, block_structure = prepare_hyb_fit(hyb, H_dft, tol=1e-6, verbose=verbose, w=w)

    # Fingerprint of the hybridization function, used to decide whether a
    # stored bath fit can be reused (identical hybridization) or the fit has
    # to be redone (new DMFT iteration).
    hyb_fingerprint = hashlib.sha256(np.ascontiguousarray(hyb).tobytes()).hexdigest()
    ebs_star, vs_star, shifts, block_structure = fit_hyb_star(
        phase_hyb,
        w,
        eim,
        bath_states_per_orbital,
        block_structure,
        gamma,
        valence_bath_only,
        weight_function,
        weight_w0,
        exp_weight,
        label,
        hdf5_filename,
        verbose,
        comm,
        hyb_fingerprint=hyb_fingerprint,
        freeze_bath_energies=freeze_bath_energies,
    )

    # rspt2spectra turns the (flattened) star fit into the requested bath
    # geometry and assembles the full one-particle Hamiltonian, including the
    # star-geometry reference used to classify the bath states as
    # valence/conduction and the star-vs-chain G0 consistency check.
    (
        H,
        _H_star,
        impurity_indices,
        valence_bath_indices,
        conduction_bath_indices,
        v_solver,
        H_bath,
        H_imp,
    ) = assemble_h0(
        ebs_star,
        vs_star,
        shifts,
        H_dft,
        H_local_Q,
        Q,
        block_structure,
        bath_geometry=bath_geometry,
        w=w,
        eim=eim,
        label=label,
        verbose=verbose,
        extra_verbose=extra_verbose,
        comm=comm,
    )

    if extra_verbose and rank == 0:
        figs = plot_hyb_fit(
            w,
            eim,
            phase_hyb,
            ebs_star,
            vs_star,
            shifts,
            H_local_Q,
            block_structure,
            bath_geometry,
            # peel_weight=peel_weight,
        )
        for block_idx, fig in enumerate(figs):
            fig.savefig(f"bath_state_resolved_hybridization_fit_block_{block_idx}.png")

    # Hand back the impurity / hybridization / bath blocks; the caller builds the ImpurityModel
    # (which assembles the operator and derives the orbital layout) via ImpurityModel.from_blocks.
    return (
        H_imp,
        impurity_indices,
        valence_bath_indices,
        conduction_bath_indices,
        v_solver,
        H_bath,
        H,
    )


def fit_hyb_star(
    phase_hyb,
    w,
    eim,
    bath_states_per_orbital,
    block_structure,
    gamma,
    valence_bath_only,
    weight_function,
    weight_w0,
    exp_weight,
    label,
    hdf5_filename,
    verbose,
    comm,
    hyb_fingerprint="",
    freeze_bath_energies=False,
):
    vs_star = None
    ebs_star = None
    shifts = None
    read_hopping = False
    # M2: RSPt calls run_impmod_ed twice per CSC iteration (once to determine the DC,
    # then 0 to solve); each call re-runs this fit. If the DC search and the selfenergy solve did
    # not end up using the SAME bath fit, the DC was determined on a different model than the one
    # it is applied to -- precisely the parity failure this branch's caching exists to prevent.
    # stored_fingerprint (found but not necessarily matching) distinguishes a genuine mismatch
    # (two calls this iteration disagree on hyb) from the ordinary first-computation case (no
    # stored fit yet), which the printed message below reports either way, not gated on verbose.
    stored_fingerprint = None
    it = None
    if comm is None or comm.rank == 0:
        # Reuse the stored bath fit if, and only if, it was produced from this
        # exact hybridization function and block structure. This lets the
        # selfenergy pass reuse the fit from the double counting pass within
        # one DMFT step, while a new DMFT iteration (new hybridization), or a
        # changed block partition, triggers a refit.
        try:
            with h5.File(
                hdf5_filename,
                "r",
            ) as ar:
                it = ar.attrs["last iteration"]
                fit_g = ar[f"{label} {it}/Bath fit"]
                stored_fingerprint = fit_g.attrs.get("hyb fingerprint", None)
                if stored_fingerprint == hyb_fingerprint and fit_g.attrs.get("block structure", "") == repr(
                    block_structure
                ):
                    print("Reading stored bath energies and hopping parameters")
                    vs_star = []
                    ebs_star = []
                    shifts = []
                    for block_index in block_structure.inequivalent_blocks:
                        vs_star.append(np.array(fit_g[f"vs_star/{block_index}"]))
                        ebs_star.append(np.array(fit_g[f"ebs_star/{block_index}"]))
                        shifts.append(np.array(fit_g[f"shifts/{block_index}"]))
                    read_hopping = True
        except (FileNotFoundError, KeyError):
            vs_star = None
            ebs_star = None
            shifts = None
        # it may still be None here (no archive file yet, or "last iteration" not written yet --
        # the very first call this process ever makes): report that plainly rather than crashing
        # on an undefined iteration number.
        iteration_label = "iteration ?" if it is None else f"iteration {it}"
        if read_hopping:
            print(
                f"Bath fit REUSED for {label!r} {iteration_label} (hybridization fingerprint "
                "matches): the DC search and the selfenergy solve share the identical bath "
                "model.",
                flush=True,
            )
        else:
            print(
                f"Bath fit computed fresh for {label!r} {iteration_label} (no fit stored yet this iteration).",
                flush=True,
            )
    if comm is not None:
        ebs_star = comm.bcast(ebs_star, root=0)
        vs_star = comm.bcast(vs_star, root=0)
        shifts = comm.bcast(shifts, root=0)

    # freeze_bath_energies: hold the bath energies fixed at the most recent stored fit and
    # re-solve only the hoppings against this iteration's hybridization. Skipped when the exact
    # fit was just reused (``ebs_star`` already set, rank-consistent after the bcast above) --
    # that path already returns a consistent model -- and on the first iteration, where nothing
    # is stored yet, we do a full fit once. Gated on ``ebs_star is None`` rather than
    # ``read_hopping`` so every rank takes the same branch into the collective below.
    frozen_ebs = None
    if freeze_bath_energies and ebs_star is None:
        if comm is None or comm.rank == 0:
            frozen_ebs = _previous_bath_energies(hdf5_filename, label, block_structure)
        if comm is not None:
            frozen_ebs = comm.bcast(frozen_ebs, root=0)
        if comm is None or comm.rank == 0:
            if frozen_ebs is None:
                print(
                    "freeze_bath_energies is set but no previous bath fit is stored; doing a full fit this iteration.",
                    flush=True,
                )
            else:
                print("Bath energies FROZEN at the previous fit; re-solving hoppings only.", flush=True)

    if ebs_star is None:
        trace_phase_hyb = np.sum(np.diagonal(phase_hyb, axis1=1, axis2=2), axis=1)
        w_min = w[np.argmax(np.abs(trace_phase_hyb) > 1e-6)]
        w_max = w[-(np.argmax(np.abs(trace_phase_hyb)[::-1] > 1e-6) + 1)]
        ebs_star, vs_star, shifts = fit_hyb(
            w,
            eim,
            phase_hyb,
            bath_states_per_orbital,
            block_structure,
            gamma=gamma,
            x_lim=(w_min, min(0, w_max) if valence_bath_only else w_max),
            verbose=verbose,
            comm=comm,
            weight_fun=get_weight_function(weight_function, weight_w0, exp_weight),
            ebs_guess=frozen_ebs,
            optimize_bath_energies=frozen_ebs is None,
        )
    for i in range(len(ebs_star)):
        if len(ebs_star[i]) == 0:
            continue
        sorted_indices = np.argsort(ebs_star[i], kind="stable")
        ebs_star[i], vs_star[i] = flatten_star_levels(
            ebs_star[i][sorted_indices], vs_star[i][sorted_indices], verbose=verbose
        )
    assert len(vs_star) == len(block_structure.inequivalent_blocks), "Number of inequivalent blocks is inconsitent"

    if verbose:
        print("Star bath energies and hopping parameters:")
        for bi, (eb, vb) in enumerate(zip(ebs_star, vs_star)):
            print(
                f"Energy   :  Hopping  (impurity orbitals "
                f"{block_structure.blocks[block_structure.inequivalent_blocks[bi]]})"
            )
            for eb_i, vb_i in zip(eb, vb):
                print(
                    f"{eb_i: 9.6f}:  ",
                    "  ".join(f"{val: 9.6f}" for vb_row in vb_i for val in vb_row),
                )
            print("")
        print("=" * 80)
    if (comm is None or comm.rank == 0) and not read_hopping:
        with h5.File(
            hdf5_filename,
            "a",
        ) as ar:
            if "last iteration" not in ar.attrs:
                ar.attrs["last iteration"] = 1
            it = ar.attrs["last iteration"]
            while f"{label.strip()} {it}" in ar:
                it += 1
            ar.attrs["last iteration"] = it

            fit_g = ar.require_group(f"{label} {it}/Bath fit")
            fit_g.attrs["hyb fingerprint"] = hyb_fingerprint
            fit_g.attrs["block structure"] = repr(block_structure)
            vs_g = fit_g.require_group("vs_star")
            ebs_g = fit_g.require_group("ebs_star")
            shifts_g = fit_g.require_group("shifts")
            for i, block_index in enumerate(block_structure.inequivalent_blocks):
                h5_write_dataset(vs_g, f"{block_index}", vs_star[i])
                h5_write_dataset(ebs_g, f"{block_index}", ebs_star[i])
                h5_write_dataset(shifts_g, f"{block_index}", shifts[i])
    return ebs_star, vs_star, shifts, block_structure
