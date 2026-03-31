from os import devnull, remove, environ
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
import pickle
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import h5py as h5
from run_impurityModel import ffi
import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
from rspt2spectra.hyb_fit import fit_hyb
from rspt2spectra.weight_functions import weight_functions

from impurityModel.ed.block_structure import (
    BlockStructure,
    build_block_structure,
    print_block_structure,
)
from impurityModel.ed.greens_function import (
    save_Greens_function,
    build_full_greens_function,
    block_diagonalize_hyb,
)
from impurityModel.ed import finite
from impurityModel.ed.lanczos import Reort
from impurityModel.ed.greens_function import (
    rotate_Greens_function,
    rotate_matrix,
    rotate_4index_U,
)
from impurityModel.ed.manybody_basis import CIPSI_Basis
from impurityModel.ed.selfenergy import fixed_peak_dc
from impurityModel.ed.edchain import build_H_bath_v, build_imp_bath_blocks
from impurityModel.ed.selfenergy import calc_selfenergy
from impurityModel.ed.utils import matrix_print


def parse_solver_line(solver_line):
    """
    N0 dN dVal dCon Nbath [[pro, full] [dense_cutoff 50] [no_block], [fit_unocc] [weight 2]]
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
        "reort": Reort.NONE,
        "blocked": True,
        "fit_unocc": False,
        "gamma": 0.01,
        "weight_function": "none",
        "weight": 2,
        "spin_flip_dj": False,
        "bath_geometry": "star",
        "occ_cutoff": 1e-6,
        "occ_restrict": False,
        "dN": 4,
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
            if arg.lower() in {"pro", "full", "periodic"}:
                if arg.lower() == "pro":
                    options["reort"] = Reort.PARTIAL
                elif arg.lower() == "full":
                    options["reort"] = Reort.FULL
                elif arg.lower() == "periodic":
                    options["reort"] = Reort.PERIODIC
            elif arg.lower() in {"star", "chain", "haver"}:
                options["bath_geometry"] = arg.lower()
            elif arg.lower() == "fit_unocc":
                options["fit_unocc"] = True
            elif arg.lower() == "fit_occ":
                options["fit_unocc"] = False
            elif arg.lower() == "gamma":
                options["gamma"] = float(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "dense_cutoff":
                options["dense_cutoff"] = int(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "no_block":
                options["blocked"] = False
            elif arg.lower() in weight_functions.keys():
                options["weight_function"] = arg.lower()
            elif arg.lower() == "weight":
                options["weight"] = float(solver_array[i + 1])
                skip_next = True
            elif arg.lower() == "spin_flip_dj":
                options["spin_flip_dj"] = True
            elif arg.lower() == "occ_restrict":
                options["occ_restrict"] = True
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
                    f"--->Other solver params {solver_array[5:]}"
                )
    if options["bath_geometry"] == "star":
        options["chain_restrict"] = False
        options["collapse_chains"] = True
        options["occ_restrict"] = True
        if options["dN"] is None:
            options["dN"] = 2
    if not options["occ_restrict"]:
        options["dN"] = None

    print(
        f"Nominal imp. occupation   |> {nominal_occ}\n"
        f"Bath states per imp. orb. |> {nBaths}\n"
        f"Bath geometry             |> {options['bath_geometry']}\n"
        f"Fit unoccupied states     |> {options['fit_unocc']}\n"
        f"Generate spin fliped Djs  |> {options['spin_flip_dj']}\n"
        f"Use block structure       |> {options['blocked']}\n"
        f"Reorthogonalizaion mode   |> {options['reort']}\n"
        f"Dense matrix size cutoff  |> {options['dense_cutoff']}\n"
        f"Fitting weight function   |> {options['weight_function']}\n"
        f"Fitting weight factor     |> {options['weight']}\n"
        f"Occupation cutoff         |> {options['occ_cutoff']}\n"
        f"Occupation restrictions   |> {options['occ_restrict']}\n"
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
    # impurityModel uses a weird convention for the U-matrix
    u4 = np.moveaxis(u4, 1, 0)

    # For python, it makes more sense to put the frequency index first, instead of last
    sig_python = np.moveaxis(sig, -1, 0)
    sig_real_python = np.moveaxis(sig_real, -1, 0)
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

    (nominal_occ, bath_states_per_orbital, options) = parse_solver_line(solver_line)
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

    hdf5_filename = "impurityModel_data.h5"
    (
        h_op,
        impurity_indices,
        valence_bath_indices,
        conduction_bath_indices,
        block_structure,
        v,
        H_bath,
    ) = get_ed_h0(
        h_dft,
        0 if rspt_dc_flag == 1 else sig_dc,
        hyb,
        bath_states_per_orbital,
        w,
        eim,
        tau,
        gamma=options["gamma"],
        weight_function=options["weight_function"],
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
    if verbosity >= 1 and rank == 0:
        hyb = np.conj(v).T @ np.linalg.solve(
            (w + 1j * eim)[:, None, None]
            * np.identity(H_bath.shape[0], dtype=complex)[None, :, :]
            - H_bath,
            v,
            # v[None, :, :],
        )
        save_Greens_function(
            # hyb,
            rotate_Greens_function(hyb, np.conj(corr_to_cf.T)),
            w,
            "hyb-fit",
            label.strip(),
        )
    if not options.get("blocked", True):
        impurity_indices = [sorted(orb for block in impurity_indices for orb in block)]
        valence_bath_indices = [
            sorted(orb for block in valence_bath_indices for orb in block)
        ]
        conduction_bath_indices = [
            sorted(orb for block in conduction_bath_indices for orb in block)
        ]
        original_block_structure = block_structure
        block_structure = BlockStructure(
            impurity_indices,
            [[0]],
            [[]],
            [[]],
            [[]],
            [0],
        )

    if rspt_dc_flag == 1:
        dc_line = ffi.string(rspt_dc_line, 100).decode("ascii")
        dc_line = dc_line.split("!")[0]
        dc_line = dc_line.split("#")[0]
        dc_array = dc_line.strip().split()
        assert (
            len(dc_array) == 1
        ), f"impurityModel double counting correction only accepts 1 argument, peak_position. Got options: {dc_array} "
        peak_position = float(dc_array[0])

        try:
            sig_dc[:, :] = fixed_peak_dc(
                h_op,
                N0=nominal_occ,
                mixed_valence=mixed_valence,
                impurity_orbitals={0: impurity_indices},
                bath_states=(
                    {0: valence_bath_indices},
                    {0: conduction_bath_indices},
                ),
                u4=u4,
                peak_position=peak_position,
                dc_guess=sig_dc,
                spin_flip_dj=options["spin_flip_dj"],
                tau=tau,
                rank=rank,
                verbose=verbosity > 0,
                dense_cutoff=options["dense_cutoff"],
                slaterWeightMin=options["slater_min"],
                truncation_threshold=options["truncation_threshold"],
            )
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
                bath_states=(
                    {0: valence_bath_indices},
                    {0: conduction_bath_indices},
                ),
                tau=tau,
                verbosity=verbosity,
                block_structure=block_structure,
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
                sig_static[:, :] = results["sigma_static"]
                sig_python[:, :, :] = build_full_greens_function(
                    results["sigma"], block_structure
                )
                sig_real_python[:, :, :] = build_full_greens_function(
                    results["sigma_real"], block_structure
                )

                # Rotate self energy from CF basis to RSPt's corr basis
                u = np.conj(corr_to_cf.T)
                sig_python[:, :, :] = rotate_Greens_function(sig_python, u)
                sig_real_python[:, :, :] = rotate_Greens_function(sig_real_python, u)
                sig_static[:, :] = rotate_matrix(sig_static, u)

            comm.Bcast(sig_static, root=0)
            comm.Bcast(sig_real, root=0)
            comm.Bcast(sig, root=0)

            if comm.rank == 0:
                opt = options.copy()
                opt.pop("reort", None)
                if opt["dN"] is None:
                    opt.pop("dN", None)
                if opt["mv"] is None:
                    opt.pop("mv", None)
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
                    cluster_g.create_dataset("Real frequency mesh", data=w)
                    cluster_g.create_dataset("Matsubara frequency mesh", data=iw)
                    cluster_g.create_dataset(
                        "Rot to spherical",
                        # data=corr_to_spherical,
                        data=np.conj(corr_to_cf.T) @ corr_to_spherical,
                    )
                    bs_g = cluster_g.create_group("block structure")
                    bs_g.attrs["Num blocks"] = len(block_structure.blocks)
                    b_g = bs_g.create_group("Blocks")
                    for i, block in enumerate(block_structure.blocks):
                        b_g.create_dataset(f"{i}", data=block)
                    b_g = bs_g.create_group("Identical blocks")
                    for i, block in enumerate(block_structure.identical_blocks):
                        if len(block) == 0:
                            continue
                        b_g.create_dataset(f"{i}", data=block)
                    b_g = bs_g.create_group("Transposed blocks")
                    for i, block in enumerate(block_structure.transposed_blocks):
                        if len(block) == 0:
                            continue
                        b_g.create_dataset(f"{i}", data=block)
                    b_g = bs_g.create_group("Particle hole blocks")
                    for i, block in enumerate(block_structure.particle_hole_blocks):
                        if len(block) == 0:
                            continue
                        b_g.create_dataset(f"{i}", data=block)
                    b_g = bs_g.create_group("Particle hole transposed blocks")
                    for i, block in enumerate(
                        block_structure.particle_hole_transposed_blocks
                    ):
                        if len(block) == 0:
                            continue
                        b_g.create_dataset(f"{i}", data=block)
                    bs_g.create_dataset(
                        "Inequivalent blocks", data=block_structure.inequivalent_blocks
                    )
                    ibb_g = cluster_g.create_group("Impurity bath blocks")
                    ibb_g.attrs["Num blocks"] = len(block_structure.blocks)
                    imp_orb_g = ibb_g.create_group("Impurity orbitals")
                    val_orb_g = ibb_g.create_group("Valence baths")
                    con_orb_g = ibb_g.create_group("Conduction baths")
                    for i, (
                        impurity_block,
                        valence_block,
                        conduction_block,
                    ) in enumerate(
                        zip(
                            impurity_indices,
                            valence_bath_indices,
                            conduction_bath_indices,
                        )
                    ):
                        imp_orb_g.create_dataset(f"{i}", data=impurity_block)
                        val_orb_g.create_dataset(f"{i}", data=valence_block)
                        con_orb_g.create_dataset(f"{i}", data=conduction_block)

                    cluster_g.create_dataset("H DFT", data=h_dft)
                    cluster_g.create_dataset("H bath", data=H_bath)
                    cluster_g.create_dataset("V", data=v)
                    cluster_g.create_dataset("U", data=u4)
                    cluster_g.create_dataset(
                        "thermal_rho", data=results["thermal_rho"]
                    ),
                    cluster_g.create_dataset("rhos", data=results["rhos"]),
                    cluster_g.create_dataset("Sigma Static", data=sig_static)
                    cluster_g.create_dataset("Sigma real", data=sig_real_python)
                    cluster_g.create_dataset("Sigma Matsubara", data=sig_python)
                    cluster_g.create_dataset(
                        "Gimp Matsubara",
                        data=rotate_Greens_function(
                            build_full_greens_function(
                                results["gs_matsubara"], block_structure
                            ),
                            u,
                        ),
                    )
                    cluster_g.create_dataset(
                        "Gimp real",
                        data=rotate_Greens_function(
                            build_full_greens_function(
                                results["gs_realaxis"], block_structure
                            ),
                            u,
                        ),
                    )
                    for i, inequiv_block in enumerate(
                        block_structure.inequivalent_blocks
                    ):
                        orbs = block_structure.blocks[inequiv_block]
                        block_g = cluster_g.create_group(f"block {inequiv_block}")
                        block_g.create_dataset(
                            "orbitals", data=np.array(orbs, dtype=int)
                        )
                        block_g.create_dataset(
                            "Gimp Matsubara", data=results["gs_matsubara"][i]
                        )
                        block_g.create_dataset(
                            "Gimp real", data=results["gs_realaxis"][i]
                        )
                        block_g.create_dataset(
                            "Sigma Matsubara", data=results["sigma"][i]
                        )
                        block_g.create_dataset(
                            "Sigma real", data=results["sigma_real"][i]
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
    weight_function="none",
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
    hyb           -- The real frequency hybridiaztion function. Used to fit the bath states.
    hdft          -- The DFT hamiltonian, projected onto the impurity orbitals.
    bath_states   -- Number of bath states to fit per impurity orbital.
    rot_spherical -- Transformation matrix to transform to spherical harmonics basis.
    w             -- Real frequency mesh.
    eim           -- All real frequency quantities are evaluated i*eim above the real frequency axis.
    gamma         -- Regularization parameter.
    imag_only     -- Only fit the imaginary part of the hybridization function, default: False.
    valence_bath_only -- Only fit bath stated in the valence band, default: True.
    label          -- Label for the cluster, used for saving a copy of the Hamiltonian that can be plugged into the Matsubara
    ED solver in RSPr, default: None,

    Returns:
    h0   -- The non-interacting impurity hamiltonian in operator form.
    eb   -- The bath states used for fitting the hybridization function.
    """

    # We do the fitting by first transforming the hyridization function into a basis
    # where each block is (hopefully) close to diagonal
    # np.conj(Q.T) @ hyb @ Q is the transformation performed
    phase_hyb, Q = block_diagonalize_hyb(hyb)

    block_structure = build_block_structure(phase_hyb, tol=1e-6)

    ebs_star, vs_star, block_structure = fit_hyb_star(
        phase_hyb,
        w,
        eim,
        bath_states_per_orbital,
        block_structure,
        gamma,
        imag_only,
        valence_bath_only,
        weight_function,
        weight_w0,
        exp_weight,
        label,
        hdf5_filename,
        verbose,
        comm,
    )
    w_min = w[0]
    w_max = w[-1]
    if valence_bath_only:
        w_max = 0
    filtered_ebs_star, filtered_vs_star = ([], [])
    shifts = []
    for ebs, vs in zip(ebs_star, vs_star):
        shift = 0
        filtered_ebs = np.empty((0,), dtype=float)
        filtered_vs = np.empty((0, vs.shape[1], vs.shape[2]), dtype=vs.dtype)
        for i in range(ebs.shape[0]):
            eb = ebs[i]
            v = vs[i]
            if w_min <= eb <= w_max:
                filtered_ebs = np.append(filtered_ebs, eb)
                filtered_vs = np.append(filtered_vs, [v], axis=0)
                continue
            shift += np.conj(v.T) @ v / eb
        filtered_ebs_star.append(filtered_ebs)
        filtered_vs_star.append(filtered_vs)
        shifts.append(shift)
    ebs_star = filtered_ebs_star
    vs_star = filtered_vs_star

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
        rotate_matrix(H_dft + sig_dc, Q) - H_shift,
        ebs_star,
        vs_star,
        bath_geometry,
        block_structure,
        verbose,
        extra_verbose,
    )
    H_bath, v = build_full_bath(H_baths, vs, block_structure)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, H_bath, op=MPI.SUM)
        H_bath /= comm.size
        comm.Allreduce(MPI.IN_PLACE, v, op=MPI.SUM)
        v /= comm.size

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

    H_baths_star, vs_star = build_H_bath_v(
        rotate_matrix(H_dft + sig_dc, Q) - H_shift,
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
    assert np.allclose(
        np.linalg.eigvalsh(H), np.linalg.eigvalsh(H_tmp)
    ), "Eigenvalues have changed!"
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
    impurity_indices, valence_bath_indices, conduction_bath_indices, block_structure = (
        build_imp_bath_blocks(H, n_orb)
    )

    h_op = finite.matrixToIOp(H)
    return (
        h_op,
        impurity_indices,
        valence_bath_indices,
        conduction_bath_indices,
        block_structure,
        v @ np.conj(Q.T),
        H_bath,
    )


def fit_hyb_star(
    phase_hyb,
    w,
    eim,
    bath_states_per_orbital,
    block_structure,
    gamma,
    imag_only,
    valence_bath_only,
    weight_function,
    weight_w0,
    exp_weight,
    label,
    hdf5_filename,
    verbose,
    comm,
):
    vs_star = None
    ebs_star = None
    read_hopping = False
    if comm is None or comm.rank == 0:
        # Check to see if we have already done a fit
        vs_star = []
        ebs_star = []
        try:
            with h5.File(
                hdf5_filename,
                "r",
            ) as ar:
                it = None
                if "last iteration" in ar.attrs:
                    it = ar.attrs["last iteration"]
                if it is None or "tau" in ar[f"{label} {it}"].attrs:
                    vs_star = None
                    ebs_star = None
                else:
                    print(f"Reading hopping parameters")
                    for block_index in block_structure.inequivalent_blocks:
                        vs_star.append(
                            np.array(ar[f"{label} {it}/Bath fit/vs_star/{block_index}"])
                        )
                        ebs_star.append(
                            np.array(
                                ar[f"{label} {it}/Bath fit/ebs_star/{block_index}"]
                            )
                        )
                    read_hopping = True
        except (FileNotFoundError, KeyError):
            vs_star = None
            ebs_star = None
    if comm is not None:
        ebs_star = comm.bcast(ebs_star, root=0)
        vs_star = comm.bcast(vs_star, root=0)
    if ebs_star is not None and verbose:
        print("Read bath energies and hopping parameters", flush=True)
    elif ebs_star is None:
        ebs_star, vs_star = fit_hyb(
            w,
            eim,
            phase_hyb,
            bath_states_per_orbital,
            block_structure,
            gamma=gamma,
            x_lim=(w[0], 0 if valence_bath_only else w[-1]),
            verbose=verbose,
            comm=comm,
            weight_fun=get_weight_function(weight_function, weight_w0, exp_weight),
            ebs_guess=ebs_star,
            vs_guess=vs_star,
        )
    for ebss, vss in zip(ebs_star, vs_star):
        if len(ebss) == 0:
            continue
        sorted_indices = np.argsort(ebss, kind="stable")
        ebss[:] = ebss[sorted_indices]
        vss[:] = vss[sorted_indices]
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

            if f"{label} {it}" not in ar:
                ar.create_group(f"{label} {it}")
            if f"{label} {it}/Bath fit" not in ar:
                ar.create_group(f"{label} {it}/Bath fit")
            if f"{label} {it}/Bath fit/vs_star" not in ar:
                ar.create_group(f"{label} {it}/Bath fit/vs_star")
            if f"{label} {it}/Bath fit/ebs_star" not in ar:
                ar.create_group(f"{label} {it}/Bath fit/ebs_star")
            for i, block_index in enumerate(block_structure.inequivalent_blocks):
                if f"{block_index}" in ar[f"{label} {it}/Bath fit/vs_star"]:
                    del ar[f"{label} {it}/Bath fit/vs_star/{block_index}"]
                if f"{block_index}" in ar[f"{label} {it}/Bath fit/ebs_star"]:
                    del ar[f"{label} {it}/Bath fit/ebs_star/{block_index}"]
                ar[f"{label} {it}/Bath fit/vs_star"].create_dataset(
                    f"{block_index}", data=vs_star[i]
                )
                ar[f"{label} {it}/Bath fit/ebs_star"].create_dataset(
                    f"{block_index}", data=ebs_star[i]
                )
    return ebs_star, vs_star, block_structure


def build_full_bath(H_bath_inequiv, v_inequiv, block_structure: BlockStructure):
    (
        blocks,
        identical_blocks,
        transposed_blocks,
        particle_hole_blocks,
        particle_hole_and_transposed_blocks,
        inequivalent_blocks,
    ) = block_structure
    n_orb = sum(len(b) for b in blocks)
    H_baths = [None] * len(blocks)
    vs = [None] * len(blocks)
    for i, block_i in enumerate(inequivalent_blocks):
        H_bath = H_bath_inequiv[i]
        v = v_inequiv[i]
        for b in identical_blocks[block_i]:
            v_tmp = np.zeros((v.shape[0], n_orb), dtype=complex)
            H_baths[b] = H_bath.copy()
            v_tmp[:, blocks[b]] = v
            vs[b] = v_tmp
        for b in transposed_blocks[block_i]:
            v_tmp = np.zeros((v.shape[0], n_orb), dtype=complex)
            H_baths[b] = H_bath.copy().T
            v_tmp[:, blocks[b]] = v
            vs[b] = v_tmp
        for b in particle_hole_blocks[block_i]:
            v_tmp = np.zeros((v.shape[0], n_orb), dtype=complex)
            H_baths[b] = H_bath.copy() @ (-np.identity(H_bath.shape[0]))
            v_tmp[:, blocks[b]] = v
            vs[b] = v_tmp
        for b in particle_hole_and_transposed_blocks[block_i]:
            v_tmp = np.zeros((v.shape[0], n_orb), dtype=complex)
            H_baths[b] = H_bath.copy().T @ (-np.identity(H_bath.shape[0]))
            v_tmp[:, blocks[b]] = v
            vs[b] = v_tmp
    return sp.linalg.block_diag(*H_baths), np.vstack(vs)
