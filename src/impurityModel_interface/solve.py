"""One RSPt call: the double-counting search, the self-energy solve or the spectra, from a :class:`ClusterCall`.

RSPt calls the solver twice per self-consistency iteration, once for the double counting
(task ``"dc"``) and once for the self-energy (``"selfenergy"``).  Both start from the same
problem -- RSPt's quantities rotated into the fitting basis, the fitted h0, the solver model --
built by :func:`problem.prepare_problem`; the entries of :data:`TASKS` finish the call.

The arrays are RSPt's own memory and are mutated in place (see :mod:`impurityModel_interface.entry`).
"""

import re
import sys
import traceback
from contextlib import contextmanager
from os import devnull

import numpy as np
from impurityModel.api import Meshes, calc_selfenergy
from mpi4py import MPI
from rspt2spectra.utils import rotate_Greens_function

from . import archive
from .cluster import TASKS as TASK_NAMES
from .dc_line import parse_dc_line
from .double_counting import run_double_counting as search_double_counting
from .mpi import on_root
from .problem import Problem, _report_fit, prepare_problem, split_total_over_groups  # noqa: F401
from .solver_line import parse_solver_line


def _check_fll_occupations(target, shell_ls):
    """``fll N`` on one shell, ``fll N1,N2,...`` with one occupation per shell on several."""
    count = len(target) if isinstance(target, tuple) else 1
    n_shells = max(1, len(shell_ls))  # no layout sent: one block, the single-shell path
    if count != n_shells:
        raise ValueError(
            f"'fll' was given {count} occupation(s), but this cluster has {n_shells} shell(s) "
            f"(l = {list(shell_ls)}): give one per shell, comma-separated in the order of the shells "
            "(as N0 on the solver line), or none to use each shell's DFT occupation."
        )


@contextmanager
def redirected_stdout(label, task, comm, verbosity):
    """Send ``print`` to this call's output file; restore ``sys.stdout`` on every path out.

    Rank 0 writes ``impurityModel-<label>[-dc].out``; the other ranks write their own file when
    ``verbosity > 0`` and otherwise discard their output.
    """
    suffix = {"dc": "-dc", "selfenergy": ""}.get(task, f"-{task}")
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


def solve(call, comm=None):
    """Run ``call.task`` on a :class:`ClusterCall`.

    Pure Python: nothing here knows about cffi.  The arrays of the call are RSPt's and are mutated in
    place.  Returns 0, or -1 if the self-energy solve failed (after poisoning ``sig`` and aborting the
    communicator).  An unknown task raises on every rank.
    """
    comm = MPI.COMM_WORLD if comm is None else comm
    if call.task not in TASKS:
        raise ValueError(f"unknown task {call.task!r}; expected one of {sorted(TASKS)}")
    call.synchronise(comm)

    with redirected_stdout(call.label, call.task, comm, call.verbosity):
        solver_line = call.solver_line
        line = parse_solver_line(solver_line)
        if comm.rank == 0 and len(solver_line.strip()) >= 120 and not re.search(r"[!#]", solver_line):
            print(
                "WARNING: the solver line fills all 120 characters of RSPt's solver string; the last option may have been cut off.",
                flush=True,
            )
        request = parse_dc_line(call.dc_line) if call.task == "dc" else None
        if request is not None and request.mode == "fll" and request.target is not None:
            _check_fll_occupations(request.target, call.shells.shell_ls)
        problem = prepare_problem(call, line, comm)
        on_root(
            comm,
            lambda: archive.write_problem(
                archive.FILENAME,
                call.label,
                line=problem.line,
                solver_line=solver_line,
                tau=call.tau,
                delta=call.eim,
                w=call.w,
                iw=call.iw,
                h_solver=problem.H_solver,
                h_dft=problem.h_dft,
                h_bath=problem.fitted.H_bath,
                v=problem.v,
                u4=problem.u4,
                dc=problem.sig_dc,
                corr_to_solver=problem.corr_to_solver,
                rot_to_spherical=problem.rot_to_spherical,
                layout=(
                    problem.fitted.impurity_indices,
                    problem.fitted.valence_bath_indices,
                    problem.fitted.conduction_bath_indices,
                ),
                operators=problem.model.operators,
                shell_ls=call.shells.shell_ls,
            ),
        )
        if call.verbosity >= 1:
            on_root(
                comm,
                lambda: _report_fit(problem, w=call.w, eim=call.eim, tau=call.tau, label=call.label, comm=comm),
            )
        status = TASKS[call.task](problem, call, request, comm)
    comm.barrier()
    return status


def run_double_counting(problem, call, request, comm):
    """The ``"dc"`` task: write the damped double counting into ``sig_dc`` (RSPt's basis).

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
        corr_to_solver=problem.corr_to_solver,
        w=call.w,
        tau=call.tau,
        nominal_occ=problem.basis.nominal_occ,
        comm=comm,
        verbosity=call.verbosity,
        cap_ledger=problem.cap_ledger,
        filename=archive.FILENAME,
        label=call.label,
    )
    if damped is not None:
        # The double counting is calculated in the fitting basis, RSPt expects it in the corr basis.
        call.out.write_double_counting(problem.to_rspt(damped))
    comm.barrier()
    return 0


def run_selfenergy(problem, call, request, comm):
    """The ``"selfenergy"`` task: fill ``sig``, ``sig_real``, ``sig_static`` in RSPt's basis."""
    out = call.out
    try:
        meshes = Meshes(iw=1j * call.iw, w=call.w, delta=call.eim)
        results = calc_selfenergy(
            problem.model,
            meshes,
            problem.basis,
            problem.solver,
            comm=comm,
            verbosity=call.verbosity,
            cluster_label=call.label.strip(),
            cap_ledger=problem.cap_ledger,
        )
        # calc_selfenergy returns everything in the fitting basis; RSPt expects the corr basis.
        out.write_selfenergy(
            problem.to_rspt(results["sigma_static"]) if comm.rank == 0 else None,
            problem.to_rspt_green(results["sigma"]) if comm.rank == 0 else None,
            problem.to_rspt_green(results["sigma_real"]) if comm.rank == 0 else None,
            comm,
        )
        if comm.rank == 0:
            u = np.conj(problem.corr_to_solver.T)
            archive.write_selfenergy(
                archive.FILENAME,
                call.label,
                results,
                out.sig_static,
                out.sig_real_python,
                out.sig_python,
                rotate_Greens_function,
                u,
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
        out.poison()
        comm.Abort(-1)
        return -1
    print("Self energy calculated! impurityModel shutting down.", flush=True)
    return 0


def run_spectra(problem, call, request, comm):
    """Spectra of the fitted model; the requests come from ``impurityModel.toml``.  Nothing goes back to RSPt."""
    from impurityModel.api import calc_spectroscopy, find_environment_file, load_spectroscopy_requests, save_spectra

    try:
        requests = load_spectroscopy_requests(find_environment_file(), comm=comm)
        result = calc_spectroscopy(
            problem.model,
            requests,
            problem.basis,
            problem.solver,
            comm=comm,
            verbosity=call.verbosity,
            cluster_label=call.label,
        )
        if comm.rank == 0:
            save_spectra(result, f"spectra-{call.label}.h5")
    except Exception:
        print(f"Exception caught on rank {comm.rank}!\n{traceback.format_exc()}", flush=True)
        comm.Abort(-1)
        return -1
    print("Spectra calculated! impurityModel shutting down.", flush=True)
    return 0


#: What each task does, ``(problem, call, dc_request, comm) -> status``.
TASKS = {"dc": run_double_counting, "selfenergy": run_selfenergy, "spectra": run_spectra}
assert set(TASKS) <= set(TASK_NAMES)
