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
        def def_extern():
            return lambda func: func

        @staticmethod
        def string(*args, **kwargs):
            raise RuntimeError("ffi.string is only available when embedded in RSPt")

        @staticmethod
        def buffer(*args, **kwargs):
            raise RuntimeError("ffi.buffer is only available when embedded in RSPt")

    ffi = _FFIStub()
from rspt2spectra.h0 import assemble_h0, flatten_star_levels, prepare_hyb_fit  # noqa: E402
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
        calc_selfenergy,
        discretized_impurity_occupation,
        fixed_occupation_dc,
        fixed_peak_dc,
        report_continuum_reference,
        save_Greens_function,
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
      gamma X                              -- Regularization parameter for the bath fit.
      dense_cutoff N                       -- Use dense eigensolver below this matrix size.
      <weight function name>               -- Weight function for the fit; one of
                                              unit, exponential, gaussian, sqrtgauss,
                                              lingauss, quadgauss, step.
      weight X                             -- Weight function decay/steepness factor.
      weight_w0 X                          -- Center of the weight function (default 0).
      spin_flip_dj                         -- Generate spin flipped determinants.
      no_chain_restrict                    -- Disable chain occupation restrictions.
      occ_cutoff X                         -- Occupation cutoff.
      truncation_threshold N               -- Basis truncation threshold (default: None => automatically determined).
      slater_min X                         -- Minimal Slater determinant weight.
      dn N                                 -- Allowed impurity occupation window (+-dN).
      mv N                                 -- Mixed valence scalar, forwarded per group to
                                              impurityModel's Basis (see impurityModel docs).
      dense_green                          -- Use the dense block-Lanczos Green's function path.
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
    options = {
        "dense_cutoff": 1000,
        "reort": "none",
        "fit_unocc": False,
        "gamma": 0.01,
        "weight_function": "unit",
        "weight": 2,
        "weight_w0": 0.0,
        "spin_flip_dj": False,
        "bath_geometry": "peeled",
        "occ_cutoff": 1e-6,
        "excitation_budget": 4,
        "dN": None,
        "mv": None,
        "chain_restrict": True,
        "truncation_threshold": None,
        "slater_min": np.sqrt(np.finfo(float).eps),
        "collapse_chains": False,
        "sparse_green": True,
    }
    if len(solver_array) > 3:
        skip_next = False
        for i in range(3, len(solver_array)):
            if skip_next:
                skip_next = False
                continue
            arg = solver_array[i]
            if arg.lower() in {"none", "pro", "partial", "selective", "full", "periodic"}:
                if arg.lower() == "pro":
                    options["reort"] = "partial"
                else:
                    options["reort"] = arg.lower()
            elif arg.lower() in {"star", "chain", "linked_chain", "peeled", "peeled_linked_chain"}:
                if arg.lower() == "peeled":
                    options["bath_geometry"] = "peeled_linked_chain"
                else:
                    options["bath_geometry"] = arg.lower()
            elif arg.lower() == "fit_unocc":
                options["fit_unocc"] = True
            elif arg.lower() == "fit_occ":
                options["fit_unocc"] = False
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
                options["spin_flip_dj"] = True
            elif arg.lower() == "no_chain_restrict":
                options["chain_restrict"] = False
            elif arg.lower() == "occ_cutoff":
                options["occ_cutoff"] = float(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "truncation_threshold":
                options["truncation_threshold"] = int(float(solver_array[i + 1]))
                skip_next = True
            elif arg.lower() == "slater_min":
                options["slater_min"] = float(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "dn":
                options["dN"] = int(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "mv":
                options["mv"] = int(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "dense_green":
                options["sparse_green"] = False
            else:
                raise RuntimeError(f"Unknown solver parameter {arg}.\n--->Other solver params {solver_array[2:]}")
    if options["bath_geometry"] == "star":
        options["chain_restrict"] = False
        options["collapse_chains"] = True
        if options["dN"] is None:
            options["dN"] = 4

    print(
        f"Nominal imp. occupation   |> {nominal_occ}\n"
        f"Bath states per imp. orb. |> {nBaths}\n"
        f"Bath geometry             |> {options['bath_geometry']}\n"
        f"Fit unoccupied states     |> {options['fit_unocc']}\n"
        f"Generate spin fliped Djs  |> {options['spin_flip_dj']}\n"
        f"Reorthogonalizaion mode   |> {options['reort']}\n"
        f"Dense matrix size cutoff  |> {options['dense_cutoff']}\n"
        f"Fitting weight function   |> {options['weight_function']}\n"
        f"Fitting weight factor     |> {options['weight']}\n"
        f"Fitting weight center w0  |> {options['weight_w0']}\n"
        f"Occupation cutoff         |> {options['occ_cutoff']}\n"
        f"dN                        |> {options['dN']}\n"
        f"Mixed valence             |> {options['mv']}\n"
        f"Chain occ. restrictions   |> {options['chain_restrict']}\n"
        f"Minimal Slater weight     |> {options['slater_min']}\n"
        f"Truncation threshold      |> {options['truncation_threshold']}\n",
        flush=True,
    )
    # Split the parsed tokens into: the bath-fit parameters (consumed by rspt2spectra and stored
    # as provenance) and the solver option groups impurityModel consumes. tau is external (a
    # separate run_impmod_ed argument) and is injected onto BasisOptions by the caller.
    fit_options = {
        key: options[key]
        for key in ("gamma", "weight_function", "weight", "weight_w0", "fit_unocc", "bath_geometry", "collapse_chains")
    }
    basis = BasisOptions(
        nominal_occ={0: nominal_occ},
        mixed_valence=None if options["mv"] is None else {0: options["mv"]},
        dN=options["dN"],
        truncation_threshold=options["truncation_threshold"],
        chain_restrict=options["chain_restrict"],
        spin_flip_dj=options["spin_flip_dj"],
        occ_cutoff=options["occ_cutoff"],
        slater_weight_min=options["slater_min"],
        excitation_budget=excitation_budget,
    )
    solver = SolverOptions(
        reort=options["reort"],
        dense_cutoff=options["dense_cutoff"],
        sparse_green=options["sparse_green"],
    )
    return nominal_occ, nBaths, fit_options, basis, solver


def solver_line_attrs(fit_options, basis, solver):
    """Flat ``{name: value}`` record of a parsed solver line, for the HDF5 archive attrs.

    Reproduces the historical ``options``-dict attribute dump from the parsed option groups so
    archived runs read back unchanged (see ``impurityModel.ed.model._read_archive_group``).
    """
    return {
        "reort": solver.reort,
        "dense_cutoff": solver.dense_cutoff,
        "sparse_green": solver.sparse_green,
        "spin_flip_dj": basis.spin_flip_dj,
        "chain_restrict": basis.chain_restrict,
        "occ_cutoff": basis.occ_cutoff,
        "dN": basis.dN,
        "truncation_threshold": basis.truncation_threshold,
        "slater_min": basis.slater_weight_min,
        "mv": None if basis.mixed_valence is None else basis.mixed_valence[0],
        **fit_options,
    }


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


@ffi.def_extern()
def run_impmod_ed(
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
    comm = MPI.COMM_WORLD
    rank = comm.rank if comm is not None else 0

    label = ffi.string(rspt_label, 18).decode("ascii")
    solver_line = ffi.string(rspt_solver_line, 100).decode("ascii")

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

    corr_to_spherical, corr_to_cf = reconstruct_rotations(
        rspt_corr_to_spherical_arr,
        rspt_corr_to_cf_arr,
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
            f"impurityModel-{label.strip()}{'-dc' if rspt_dc_flag == 1 else ''}.out",
            "w",
        )
    elif verbosity > 0:
        sys.stdout = open(  # noqa: SIM115
            f"impurityModel-{label.strip()}{'-dc' if rspt_dc_flag == 1 else ''}-{rank}.out",
            "w",
        )
    else:
        sys.stdout = open(devnull, "w")  # noqa: SIM115

    _nominal_occ, bath_states_per_orbital, fit_options, basis, solver = parse_solver_line(solver_line)
    # tau is not part of the solver line; inject the external temperature onto the basis options.
    basis = replace(basis, tau=tau)
    nominal_occ = basis.nominal_occ
    if any(n0 > n_orb for n0 in nominal_occ.values()) or any(n0 < 0 for n0 in nominal_occ.values()):
        raise RuntimeError(f"Nominal impurity occupation {nominal_occ} out of bounds [0, {n_orb}]")

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

    hdf5_filename = "impurityModel_data.h5"
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
        label=label.strip(),
        hdf5_filename=hdf5_filename,
        verbose=(verbosity >= 1 or rspt_dc_flag == 1),
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
            for key, value in solver_line_attrs(fit_options, basis, solver).items()
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

    if rspt_dc_flag == 1:
        dc_line = ffi.string(rspt_dc_line, 100).decode("ascii")
        dc_line = dc_line.split("!")[0]
        dc_line = dc_line.split("#")[0]
        dc_array = dc_line.strip().split()
        # Two double counting criteria:
        #   <peak_position>        -- place a spectral peak at the given energy
        #                             (E[N+1]-E[N] if positive, E[N]-E[N-1] if
        #                             negative)
        #   occ <occupation>       -- fix the thermal impurity occupation
        if len(dc_array) > 0 and dc_array[0].lower() in {"occ", "occupation"}:
            assert len(dc_array) <= 2, (
                "impurityModel occupation double counting takes at most 1 "
                f"argument, the target impurity occupation. Got: {dc_array}"
            )
            dc_mode = "occupation"
            dc_target = None
            if len(dc_array) == 2:
                dc_target = float(dc_array[1])
        else:
            assert len(dc_array) == 1, (
                "impurityModel double counting takes 1 argument, peak_position, "
                f"or 'occ [target impurity occupation]'. Got: {dc_array}"
            )
            dc_mode = "peak"
            dc_target = float(dc_array[0])

        try:
            if dc_mode == "occupation":
                # Scale the shift search with the real-frequency mesh, so the
                # steps are sensible in any energy unit (RSPt supplies Ry).
                bandwidth = w[-1] - w[0]
                dc_cf = fixed_occupation_dc(
                    model,
                    basis,
                    solver,
                    occupation=dc_target,
                    comm=comm,
                    verbosity=verbosity,
                    initial_step=bandwidth / 100,
                    max_shift=bandwidth,
                )
            else:
                dc_cf = fixed_peak_dc(
                    model,
                    basis,
                    solver,
                    peak_position=dc_target,
                    comm=comm,
                    verbosity=verbosity,
                )
            # The double counting is calculated in the CF basis, RSPt expects
            # it in the corr basis.
            sig_dc[:, :] = rotate_matrix(dc_cf, np.conj(corr_to_cf.T))
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
            if comm.rank == 0:
                with h5.File(hdf5_filename, "a") as f:
                    it = f.attrs.get("last iteration", 1)
                    group_name = f"{label.strip()} {it}"
                    if group_name not in f:
                        f.create_group(group_name)
                    f[group_name].attrs["DC search unreachable"] = str(e)
            er = 0
        except Exception as e:
            print("!" * 100)
            print(f"Exception {e!r} caught on rank {rank}!")
            print(traceback.format_exc())
            print(
                "Adding positive infinity to the imaginary part of the DC selfenergy.",
                flush=True,
            )
            print("!" * 100)
            sig_dc[:, :] = np.inf + 1j * np.inf
            er = -1
            comm.Abort(er)
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
    Q, phase_hyb, H_local_Q, block_structure = prepare_hyb_fit(hyb, H_dft, tol=1e-6, verbose=verbose)

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
):
    vs_star = None
    ebs_star = None
    shifts = None
    read_hopping = False
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
                if fit_g.attrs.get("hyb fingerprint", "") == hyb_fingerprint and fit_g.attrs.get(
                    "block structure", ""
                ) == repr(block_structure):
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
    if comm is not None:
        ebs_star = comm.bcast(ebs_star, root=0)
        vs_star = comm.bcast(vs_star, root=0)
        shifts = comm.bcast(shifts, root=0)
    if ebs_star is not None and verbose:
        print("Read bath energies and hopping parameters", flush=True)
    elif ebs_star is None:
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
            ebs_guess=ebs_star,
            vs_guess=vs_star,
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
