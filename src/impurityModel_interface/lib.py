from os import devnull, environ
import sys

if "OMP_NUM_THREADS" not in environ or int(environ["OMP_NUM_THREADS"]) != 1:
    print(
        "Warning, OMP parallelization can cause the eigensystem solvers to hang indefinitely.",
        file=sys.stderr,
    )
    print(
        "Therefore OMP_NUM_THREADS will be forcefully set to 1 from now on!.",
        file=sys.stderr,
    )
    environ["OMP_NUM_THREADS"] = "1"

import traceback
import hashlib
from importlib.metadata import version as package_version, PackageNotFoundError
import numpy as np
import h5py as h5

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
import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
from rspt2spectra.hyb_fit import fit_hyb
from rspt2spectra.weight_functions import weight_functions

try:
    from impurityModel.ed.block_structure import (
        BlockStructure,
        print_block_structure,
        get_n_blocks_block_indices_mask,
        get_identical_blocks,
        get_transposed_blocks,
        get_particle_hole_blocks,
        get_particle_hole_and_transpose_blocks,
        get_inequivalent_blocks,
    )
    from impurityModel.ed.greens_function import (
        save_Greens_function,
        block_diagonalize_hyb,
    )
    from impurityModel.ed import finite
    from impurityModel.ed.greens_function import (
        rotate_Greens_function,
        rotate_matrix,
        rotate_4index_U,
    )
    from impurityModel.ed.edchain import (
        build_H_bath_v,
        build_imp_bath_blocks,
        build_full_bath,
    )
    from impurityModel.ed.selfenergy import (
        calc_selfenergy,
        fixed_peak_dc,
        fixed_occupation_dc,
    )
    from impurityModel.ed.utils import matrix_print
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
      star | chain | haver                 -- Bath geometry.
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
      truncation_threshold N               -- Basis truncation threshold.
      slater_min X                         -- Minimal Slater determinant weight.
      dn N                                 -- Allowed impurity occupation window (+-dN).
      mv N                                 -- Mixed valence scalar, forwarded per group to
                                              impurityModel's Basis (see impurityModel docs).
      sparse_green                         -- Use the sparse block-Lanczos Green's function path.
    """
    # Remove comments from the solver line
    solver_line = solver_line.split("!")[0]
    solver_line = solver_line.split("#")[0]
    solver_array = solver_line.strip().split()
    assert (
        len(solver_array) >= 2
    ), "The impurityModel ED solver requires at least 2 arguments; N0 nBaths"
    try:
        nominal_occ = int(solver_array[0])
        nBaths = int(solver_array[1])
    except Exception as e:
        raise RuntimeError(
            f"{e}\n"
            f"--->N0 {solver_array[0]}\n"
            f"--->Nbaths {solver_array[1]}\n"
            f"--->Other params {solver_array[2:]}"
        )
    options = {
        "dense_cutoff": 1000,
        "reort": "partial",
        "fit_unocc": False,
        "gamma": 0.01,
        "weight_function": "unit",
        "weight": 2,
        "weight_w0": 0.0,
        "spin_flip_dj": False,
        "bath_geometry": "star",
        "occ_cutoff": 1e-6,
        "dN": None,
        "mv": None,
        "chain_restrict": True,
        "truncation_threshold": int(1e8),
        "slater_min": np.sqrt(np.finfo(float).eps),
        "collapse_chains": False,
        "sparse_green": False,
    }
    if len(solver_array) > 2:
        skip_next = False
        for i in range(2, len(solver_array)):
            if skip_next:
                skip_next = False
                continue
            arg = solver_array[i]
            if arg.lower() in {"pro", "partial", "selective", "full", "periodic"}:
                if arg.lower() == "pro":
                    options["reort"] = "partial"
                else:
                    options["reort"] = arg.lower()
            elif arg.lower() in {"star", "chain", "haver"}:
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
            elif arg.lower() == "sparse_green":
                options["sparse_green"] = True
            else:
                raise RuntimeError(
                    f"Unknown solver parameter {arg}.\n"
                    f"--->Other solver params {solver_array[2:]}"
                )
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
    return nominal_occ, nBaths, options


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
    iw = np.ndarray(
        buffer=ffi.buffer(rspt_iw, n_iw * size_real), shape=(n_iw,), dtype=float
    )
    w = np.ndarray(
        buffer=ffi.buffer(rspt_w, n_w * size_real), shape=(n_w,), dtype=float
    )
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

    if n_rot_cols == n_orb_full and n_orb == n_orb_full:
        corr_to_spherical = rspt_corr_to_spherical_arr
        corr_to_cf = rspt_corr_to_cf_arr
    else:
        corr_to_spherical = np.empty((n_orb, 2 * n_orb_full), dtype=complex)
        corr_to_cf = np.empty((n_orb, n_orb), dtype=complex)
        corr_to_spherical[:, :n_orb_full] = rspt_corr_to_spherical_arr
        corr_to_spherical[:, n_orb_full:] = np.roll(
            rspt_corr_to_spherical_arr, n_orb_full, axis=0
        )
        corr_to_cf[:, :n_rot_cols] = rspt_corr_to_cf_arr
        corr_to_cf[:, n_rot_cols:] = np.roll(rspt_corr_to_cf_arr, n_rot_cols, axis=0)
    comm.Bcast(corr_to_spherical)
    comm.Bcast(corr_to_cf)
    comm.Bcast(h_dft)
    comm.Bcast(u4)
    # impurityModel uses a weird convention for the U-matrix
    u4 = np.moveaxis(u4, 1, 0)

    # For python, it makes more sense to put the frequency index first, instead of last
    sig_python = np.moveaxis(sig, -1, 0)
    sig_real_python = np.moveaxis(sig_real, -1, 0)
    comm.Bcast(hyb)
    hyb = np.moveaxis(hyb, -1, 0)

    hyb = rotate_Greens_function(hyb, corr_to_cf)
    h_dft = rotate_matrix(h_dft, corr_to_cf)
    u4 = rotate_4index_U(u4, corr_to_cf)

    stdout_save = sys.stdout
    if rank == 0:
        sys.stdout = open(
            f"impurityModel-{label.strip()}{'-dc' if rspt_dc_flag == 1 else ''}.out",
            "w",
        )
    elif verbosity > 0:
        sys.stdout = open(
            f"impurityModel-{label.strip()}{'-dc' if rspt_dc_flag == 1 else ''}-{rank}.out",
            "w",
        )
    else:
        sys.stdout = open(devnull, "w")

    nominal_occ, bath_states_per_orbital, options = parse_solver_line(solver_line)
    nominal_occ = {0: nominal_occ}
    mixed_valence = None
    if options["mv"] is not None:
        mixed_valence = {0: options["mv"]}
    if any(n0 > n_orb for n0 in nominal_occ.values()) or any(
        n0 < 0 for n0 in nominal_occ.values()
    ):
        raise RuntimeError(
            f"Nominal impurity occupation {nominal_occ} out of bounds [0, {n_orb}]"
        )

    if abs(w[1] - w[0]) > eim / 2 and rank == 0:
        print(
            "WARNING: Your real frequency mesh is rather coarse. Recommended dE <= eim/5, in order to guarantee resolution of fine structures in the self energy."
        )
    # The solver works in the CF basis; RSPt sends and expects quantities in
    # the corr basis. Log how different the two bases are, so basis mixups are
    # visible in the output file.
    corr_to_cf_dev = np.max(np.abs(corr_to_cf - np.eye(n_orb)))
    if rank == 0:
        print(f"Max abs deviation of corr_to_cf from identity: {corr_to_cf_dev:.3e}")
    # RSPt supplies the double counting in the corr basis, the solver needs it
    # in the CF basis.
    sig_dc_cf = rotate_matrix(sig_dc, corr_to_cf)

    hdf5_filename = "impurityModel_data.h5"
    (
        h_op,
        impurity_indices,
        valence_bath_indices,
        conduction_bath_indices,
        v,
        H_bath,
        H_solver,
    ) = get_ed_h0(
        h_dft,
        0 if rspt_dc_flag == 1 else sig_dc_cf,
        hyb,
        bath_states_per_orbital,
        w,
        eim,
        tau,
        gamma=options["gamma"],
        weight_function=options["weight_function"],
        weight_w0=options["weight_w0"],
        exp_weight=options["weight"],
        imag_only=False,
        valence_bath_only=not options["fit_unocc"],
        bath_geometry=options["bath_geometry"],
        label=label.strip(),
        hdf5_filename=hdf5_filename,
        verbose=(verbosity >= 1 or rspt_dc_flag == 1),
        extra_verbose=(verbosity >= 2),
        comm=comm,
    )
    if comm.rank == 0:
        opt = options.copy()
        # h5py cannot store None attributes
        opt["dN"] = options["dN"] if options["dN"] is not None else "None"
        opt["mv"] = options["mv"] if options["mv"] is not None else "None"
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
            h5_write_dataset(
                cluster_g,
                "Rot to spherical",
                np.conj(corr_to_cf.T) @ corr_to_spherical,
            )
            h5_write_dataset(cluster_g, "Impurity orbitals", impurity_indices)
            h5_write_dataset(cluster_g, "Valence orbitals", valence_bath_indices)
            h5_write_dataset(cluster_g, "Conduction orbitals", conduction_bath_indices)

            h5_write_dataset(cluster_g, "H DFT", h_dft)
            h5_write_dataset(cluster_g, "H bath", H_bath)
            h5_write_dataset(cluster_g, "V", v)
            h5_write_dataset(cluster_g, "U", u4)
            # The full one-particle solver hamiltonian (impurity + bath, CF
            # basis) and the corr -> CF rotation; together they make the
            # archive self-contained for reproducing the solver run.
            h5_write_dataset(cluster_g, "H solver", H_solver)
            h5_write_dataset(cluster_g, "corr_to_cf", corr_to_cf)
    if verbosity >= 1 and rank == 0:
        hyb_fit = np.conj(v).T @ np.linalg.solve(
            (w + 1j * eim)[:, None, None]
            * np.identity(H_bath.shape[0], dtype=complex)[None, :, :]
            - H_bath,
            v,
        )
        # Report the fit quality. Unoccupied states are only fitted with
        # fit_unocc, so also report the deviation on the occupied side alone.
        fit_dev = np.max(np.abs(hyb_fit - hyb))
        fit_dev_occ = np.max(np.abs(hyb_fit[w <= 0] - hyb[w <= 0]))
        print(
            f"Max abs deviation of the fitted hybridization function: "
            f"{fit_dev_occ:.6f} (w <= 0), {fit_dev:.6f} (all w)"
        )
        save_Greens_function(
            rotate_Greens_function(hyb_fit, np.conj(corr_to_cf.T)),
            w,
            "hyb-fit",
            label.strip(),
        )

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
            assert len(dc_array) == 2, (
                "impurityModel occupation double counting takes exactly 1 "
                f"argument, the target impurity occupation. Got: {dc_array}"
            )
            dc_mode = "occupation"
            dc_target = float(dc_array[1])
        else:
            assert len(dc_array) == 1, (
                "impurityModel double counting takes 1 argument, peak_position, "
                f"or 'occ <target impurity occupation>'. Got: {dc_array}"
            )
            dc_mode = "peak"
            dc_target = float(dc_array[0])

        dc_kwargs = dict(
            N0=nominal_occ,
            mixed_valence=mixed_valence,
            impurity_orbitals={0: impurity_indices},
            bath_states=(
                {0: valence_bath_indices},
                {0: conduction_bath_indices},
            ),
            u4=u4,
            dc_guess=sig_dc_cf,
            spin_flip_dj=options["spin_flip_dj"],
            tau=tau,
            rank=rank,
            verbose=verbosity > 0,
            dense_cutoff=options["dense_cutoff"],
            slaterWeightMin=options["slater_min"],
            truncation_threshold=options["truncation_threshold"],
        )
        try:
            if dc_mode == "occupation":
                # Scale the shift search with the real-frequency mesh, so the
                # steps are sensible in any energy unit (RSPt supplies Ry).
                bandwidth = w[-1] - w[0]
                dc_cf = fixed_occupation_dc(
                    h_op,
                    occupation=dc_target,
                    initial_step=bandwidth / 100,
                    max_shift=bandwidth,
                    **dc_kwargs,
                )
            else:
                dc_cf = fixed_peak_dc(h_op, peak_position=dc_target, **dc_kwargs)
            # The double counting is calculated in the CF basis, RSPt expects
            # it in the corr basis.
            sig_dc[:, :] = rotate_matrix(dc_cf, np.conj(corr_to_cf.T))
            er = 0
        except Exception as e:
            print("!" * 100)
            print(f"Exception {repr(e)} caught on rank {rank}!")
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

            results = calc_selfenergy(
                h0=h_op,
                u4=u4,
                iw=1j * iw,
                w=w,
                delta=eim,
                nominal_occ=nominal_occ,
                mixed_valence=mixed_valence,
                impurity_orbitals={0: impurity_indices},
                tau=tau,
                verbosity=verbosity,
                rot_to_spherical=np.conj(corr_to_cf.T) @ corr_to_spherical,
                cluster_label=label.strip(),
                comm=comm,
                reort=options["reort"],
                dense_cutoff=options["dense_cutoff"],
                spin_flip_dj=options["spin_flip_dj"],
                chain_restrict=options["chain_restrict"],
                occ_cutoff=options["occ_cutoff"],
                truncation_threshold=options["truncation_threshold"],
                slaterWeightMin=options["slater_min"],
                dN=options["dN"],
                sparse_green=options["sparse_green"],
            )
            if comm.rank == 0:
                # calc_selfenergy returns everything in its input (CF) basis,
                # RSPt expects the selfenergy in the corr basis.
                u = np.conj(corr_to_cf.T)
                sig_static[:, :] = rotate_matrix(results["sigma_static"], u)
                sig_python[:, :, :] = rotate_Greens_function(results["sigma"], u)
                sig_real_python[:, :, :] = rotate_Greens_function(
                    results["sigma_real"], u
                )

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
            print(f"Exception {repr(e)} caught on rank {rank}!")
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


def build_combined_block_structure(phase_hyb, H_local, tol=1e-6):
    """
    Build a block structure from the union of the connectivity of the
    hybridization function and the local hamiltonian.

    The hybridization function and the local hamiltonian do not necessarily
    share a block structure. The bath geometries (in particular the linked
    double chain) are anchored on the local hamiltonian block of each
    hybridization block, so every orbital pair coupled by either the
    hybridization or the local hamiltonian must end up in the same block.
    Block equivalence (identical/transposed/particle-hole) is likewise tested
    against both, since equivalent blocks share one bath fit and one chain
    construction.
    """
    n_blocks, block_idxs = get_n_blocks_block_indices_mask(phase_hyb, H_local, tol=tol)
    blocks = [[] for _ in range(n_blocks)]
    for orb_i, block_i in enumerate(block_idxs):
        blocks[block_i].append(orb_i)
    identical_blocks = get_identical_blocks(blocks, phase_hyb, H_local, tol=tol)
    transposed_blocks = get_transposed_blocks(blocks, phase_hyb, H_local, tol=tol)
    particle_hole_blocks = get_particle_hole_blocks(blocks, phase_hyb, H_local, tol=tol)
    particle_hole_and_transposed_blocks = get_particle_hole_and_transpose_blocks(
        blocks, phase_hyb, H_local, tol=tol
    )
    inequivalent_blocks = get_inequivalent_blocks(
        identical_blocks,
        transposed_blocks,
        particle_hole_blocks,
        particle_hole_and_transposed_blocks,
    )
    return BlockStructure(
        blocks,
        identical_blocks,
        transposed_blocks,
        particle_hole_blocks,
        particle_hole_and_transposed_blocks,
        inequivalent_blocks,
    )


def get_ed_h0(
    H_dft,
    sig_dc,
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
    bath_geometry="star",
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
    label          -- Label for the cluster, used for saving a copy of the Hamiltonian that can be plugged into the Matsubara
    ED solver in RSPt, default: None,

    Returns:
    h0   -- The non-interacting impurity hamiltonian in operator form.
    eb   -- The bath states used for fitting the hybridization function.
    """

    # We do the fitting by first transforming the hyridization function into a basis
    # where each block is (hopefully) close to diagonal
    # np.conj(Q.T) @ hyb @ Q is the transformation performed
    phase_hyb, Q = block_diagonalize_hyb(hyb)

    # The local hamiltonian in the fitting basis. It anchors the bath geometry
    # construction, so the block structure must respect its connectivity as
    # well as that of the hybridization function.
    H_local_Q = rotate_matrix(H_dft + sig_dc, Q)
    block_structure = build_combined_block_structure(phase_hyb, H_local_Q, tol=1e-6)
    # Guaranteed by the union connectivity above; guard against regressions.
    _off_block = np.abs(H_local_Q.copy())
    for _orbs in block_structure.blocks:
        _off_block[np.ix_(_orbs, _orbs)] = 0
    if np.max(_off_block) > 1e-6:
        raise RuntimeError(
            "The local hamiltonian is not block diagonal on the combined "
            f"block partition. Max off-block element: {np.max(_off_block):.3e}"
        )
    if verbose:
        print_block_structure(block_structure)

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

    H_shift = np.zeros_like(H_dft)
    for inequiv_block_i, shift in zip(block_structure.inequivalent_blocks, shifts):
        for block_i in block_structure.identical_blocks[inequiv_block_i]:
            orbs = block_structure.blocks[block_i]
            H_shift[np.ix_(orbs, orbs)] = shift
    if verbose:
        matrix_print(H_shift, r"Shift of $\Delta(\omega=0)$")
    # The double counting was removed from the DFT hamiltonian before calling the solver.
    # In order to properly set up the linked double chain geometry for the bath states we need to add it back in.
    # Otherwise we will end up with 2 separate chains that only link to the impurity, not to each other.
    H_baths, vs = build_H_bath_v(
        H_local_Q - H_shift,
        ebs_star,
        vs_star,
        bath_geometry,
        block_structure,
        verbose,
        extra_verbose,
    )
    H_bath, v = build_full_bath(H_baths, vs, block_structure)
    if comm is not None:
        comm.Bcast(H_bath)
        comm.Bcast(v)

    n_orb = H_dft.shape[0]
    H = np.zeros((n_orb + H_bath.shape[0], n_orb + H_bath.shape[0]), dtype=complex)
    H[:n_orb, :n_orb] = H_dft - rotate_matrix(H_shift, np.conj(Q.T))
    H[n_orb:, n_orb:] = H_bath
    H[n_orb:, :n_orb] = v @ np.conj(Q.T)
    H[:n_orb, n_orb:] = np.conj(H[n_orb:, :n_orb].T)

    if verbose:
        print(f"Total number of spin orbitals: {H.shape[0]}")
        print(f"----> Impurity orbitals: {n_orb}")
        print(f"----> Bath orbitals: {H_bath.shape[0]}")

    # The star geometry hamiltonian is needed for classifying bath states as
    # valence/conduction (the star diagonal holds the bath energies) and for
    # the isospectrality check of the chain construction.
    if bath_geometry == "star":
        # build_H_bath_v with "star" would rebuild exactly H.
        H_tmp = H
    else:
        H_baths_star, vs_star = build_H_bath_v(
            H_local_Q - H_shift,
            ebs_star,
            vs_star,
            "star",
            block_structure,
            verbose,
            extra_verbose,
        )
        H_bath_star, v_star = build_full_bath(H_baths_star, vs_star, block_structure)
        H_tmp = np.zeros(
            (n_orb + H_bath_star.shape[0], n_orb + H_bath_star.shape[0]), dtype=complex
        )
        H_tmp[:n_orb, :n_orb] = H_dft - rotate_matrix(H_shift, np.conj(Q.T))
        H_tmp[n_orb:, n_orb:] = H_bath_star
        H_tmp[n_orb:, :n_orb] = v_star @ np.conj(Q.T)
        H_tmp[:n_orb, n_orb:] = np.conj(H_tmp[n_orb:, :n_orb].T)
        if H.shape != H_tmp.shape:
            # The chain constructions drop bath states that decouple from the
            # impurity. The uncoupled states are pruned right after the fit,
            # so a size mismatch here means the valence/conduction
            # classification below would be invalid.
            raise RuntimeError(
                f"The {bath_geometry} bath construction changed the number of "
                f"bath states ({H_tmp.shape[0] - n_orb} -> {H.shape[0] - n_orb}). "
                "Cannot classify bath states as valence/conduction."
            )
        # The star and chain geometries must describe the same impurity
        # physics; compare the impurity-projected Green's functions.
        z_check = w[::10] + 1j * eim
        G0 = np.linalg.inv(
            z_check[:, None, None] * np.identity(H.shape[0])[None] - H[None]
        )[:, :n_orb, :n_orb]
        G0_star = np.linalg.inv(
            z_check[:, None, None] * np.identity(H_tmp.shape[0])[None] - H_tmp[None]
        )[:, :n_orb, :n_orb]
        if not np.allclose(G0, G0_star, atol=1e-8):
            warning = (
                "WARNING: The bath geometry transformation changed the impurity "
                "Green's function!\n"
                f"Max abs deviation: {np.max(np.abs(G0 - G0_star)):.3e}"
            )
            print(warning, flush=True)
            # stdout is redirected to a file; make sure the warning is also
            # visible on the terminal.
            print(warning, file=sys.stderr, flush=True)

    if extra_verbose:
        print("DFT hamiltonian, with baths, in solver basis")
        matrix_print(H)
        print("=" * 80)

        print()
        print("DFT hamiltonian, with star geometry baths, in solver basis")
        matrix_print(H_tmp)
        print("=" * 80, flush=True)
        with open(f"Ham-{label}.inp", "w") as f:
            for i in range(H_tmp.shape[0]):
                for j in range(H_tmp.shape[1]):
                    f.write(
                        f" 0 0 0 {i+1} {j+1} {np.real(H_tmp[i, j])} {np.imag(H_tmp[i, j])}\n"
                    )
    impurity_indices, valence_bath_indices, conduction_bath_indices = (
        build_imp_bath_blocks(H_tmp, n_orb)
    )

    h_op = finite.matrixToIOp(H)
    return (
        h_op,
        impurity_indices,
        valence_bath_indices,
        conduction_bath_indices,
        v @ np.conj(Q.T),
        H_bath,
        H,
    )


def flatten_star_levels(ebs, vs, coupling_tol=1e-6, verbose=False):
    r"""
    Split each fitted bath level into its coupled orbital components.

    A fitted bath level at energy e carries an (n_orb x n_orb) hopping matrix
    v; the level expands into n_orb degenerate bath orbitals with hopping rows
    v[b, :]. If v is rank deficient (common when the block structure merges
    orbitals whose hybridization is block diagonal, e.g. orbitals only coupled
    through the local hamiltonian), some unitary combinations of the level's
    bath orbitals decouple from the impurity. The chain constructions
    (Lanczos) silently drop such decoupled orbitals, which would leave the
    star and chain geometries with different numbers of bath states and break
    the positional valence/conduction classification.

    Rotate the degenerate orbitals of each level with the SVD
    $v = U S W^\dagger$ (the level energy block $e\,\mathbb{1}$ is invariant,
    the hopping becomes $U^\dagger v = S W^\dagger$) and keep only rows with
    singular value above coupling_tol. The result is packed in the flat
    (N, 1, n_orb) form accepted by all bath geometry builders.

    Returns:
    ebs_flat -- (N,) bath energies, one per kept bath orbital.
    vs_flat  -- (N, 1, n_orb) hopping rows.
    """
    n_imp = vs.shape[2]
    flat_e = []
    flat_v = []
    n_dropped = 0
    for e, v in zip(ebs, vs):
        if v.shape[0] == 1:
            # Already flat; keep the row as is (an SVD would only change the
            # gauge), just drop it if it is uncoupled.
            if np.linalg.norm(v[0]) > coupling_tol:
                flat_e.append(e)
                flat_v.append(v[0])
            else:
                n_dropped += 1
            continue
        _, s, wh = np.linalg.svd(v)
        for s_k, row in zip(s, wh[: len(s)]):
            if s_k > coupling_tol:
                flat_e.append(e)
                flat_v.append(s_k * row)
            else:
                n_dropped += 1
    if n_dropped > 0 and verbose:
        print(
            f"Dropped {n_dropped} bath orbitals that do not couple to the "
            "impurity (rank deficient bath level hopping)."
        )
    return (
        np.array(flat_e, dtype=float),
        np.array(flat_v, dtype=complex).reshape((len(flat_e), 1, n_imp)),
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
                if fit_g.attrs.get(
                    "hyb fingerprint", ""
                ) == hyb_fingerprint and fit_g.attrs.get("block structure", "") == repr(
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
    assert len(vs_star) == len(
        block_structure.inequivalent_blocks
    ), "Number of inequivalent blocks is inconsitent"

    if verbose:
        print("Star bath energies and hopping parameters:")
        for bi, (eb, vb) in enumerate(zip(ebs_star, vs_star)):
            print(
                f"Energy   :  Hopping  (impurity orbitals {block_structure.blocks[block_structure.inequivalent_blocks[bi]]})"
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
