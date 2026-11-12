"""Run the double-counting criterion RSPt asked for and hand the damped answer back."""

import hashlib
import traceback

import numpy as np
from impurityModel.api import (
    DoubleCountingUnreachable,
    amf_dc,
    dc_levels,
    dc_spread,
    emit_dc_record,
    fixed_gap_dc,
    fixed_occupation_dc,
    fixed_peak_dc,
    fll_dc,
    nominal_dc,
    sigma_inf_dc,
)
from rspt2spectra.block_structure import project_to_blocks

from . import archive
from .mpi import on_root


def _banner(rank, headline, error, with_traceback):
    print("!" * 100)
    print(f"Exception {error!r} caught on rank {rank}!")
    if with_traceback:
        print(traceback.format_exc())
    print(headline, flush=True)
    print("!" * 100)


def _criterion(request, model, basis, solver, *, w, tau, nominal_total, comm, verbosity, cap_ledger):
    """Evaluate the requested criterion.  Returns ``(dc, sector)``; ``sector`` is ``None`` for static schemes."""
    solves = {
        "comm": comm,
        "verbosity": verbosity,
        "return_sector": True,
        "cap_ledger": cap_ledger,
        "allow_memory_bound": request.allow_memory_bound,
    }
    if request.mode == "occupation":
        # Scale the shift search with the real-frequency mesh, so the steps suit any energy unit.
        bandwidth = w[-1] - w[0]
        return fixed_occupation_dc(
            model,
            basis,
            solver,
            occupation=request.target,
            initial_step=bandwidth / 100,
            max_shift=bandwidth,
            **solves,
        )
    if request.mode == "gap":
        return fixed_gap_dc(
            model,
            basis,
            solver,
            offset=request.target,
            de2_min=request.de2_min,
            e_pt2_tol=request.e_pt2_tol,
            **solves,
        )
    if request.mode == "peak":
        return fixed_peak_dc(
            model,
            basis,
            solver,
            peak_position=request.target,
            de2_min=request.de2_min,
            e_pt2_tol=request.e_pt2_tol,
            **solves,
        )
    if request.mode == "fll":
        return fll_dc(model, tau=tau), None
    if request.mode == "amf":
        return amf_dc(model, tau=tau), None
    if request.mode in ("sigma_inf", "sigmainf"):
        return sigma_inf_dc(model, tau=tau), None
    assert request.mode == "nominal"
    return nominal_dc(model, nominal_total), None


def run_double_counting(
    request,
    model,
    basis,
    solver,
    *,
    guess,
    blocks,
    corr_to_cf,
    w,
    tau,
    nominal_total,
    comm,
    verbosity,
    cap_ledger,
    filename,
    label,
):
    """Determine the double counting; return it in the solver basis, damped, or ``None`` on failure.

    A search that cannot reach its target (:class:`DoubleCountingUnreachable`, a modelling verdict)
    or that fails in the solver returns ``None`` after a banner and an audit record in the archive:
    the caller leaves RSPt's incoming DC unchanged and the self-consistency loop carries on with the
    previous value rather than dying hours into a CSC run.  The search raises rank-synchronously, so
    every rank reaches the handlers together.

    ``guess`` is RSPt's incoming DC in the solver basis; the converged answer is damped against the
    previous iteration's answer from the archive, not against it (``doc/double_counting.md``).
    """
    rank = comm.rank
    try:
        converged, sector = _criterion(
            request,
            model,
            basis,
            solver,
            w=w,
            tau=tau,
            nominal_total=nominal_total,
            comm=comm,
            verbosity=verbosity,
            cap_ledger=cap_ledger,
        )
        previous = archive.previous_damped_dc(filename, label, comm, corr_to_cf=corr_to_cf)
        damped = converged if previous is None else previous + request.alpha * (converged - previous)
        # The solver neglected the couplings between blocks, so the double counting carries none.
        damped = project_to_blocks(damped, blocks)
        alpha = request.alpha if previous is not None else 1.0
        # A failed write on rank 0 is raised on every rank, so none of them applies a DC the others do not.
        on_root(
            comm,
            lambda: archive.write_double_counting(
                filename,
                label,
                alpha=alpha,
                guess_fingerprint=hashlib.sha256(np.ascontiguousarray(guess).tobytes()).hexdigest(),
                converged=converged,
                damped=damped,
                sector=sector,
                cap=cap_ledger.resolved,
            ),
        )
        # The criterion printed the DC it *converged*; RSPt receives it damped.  A second record, in
        # the same format and between the same delimiters, states what was applied.
        applied = {
            "criterion": f"{request.mode} (applied)",
            "status": "damped" if previous is not None else "undamped",
            "alpha": alpha,
        }
        applied["dc_trace"], applied["dc_level"] = dc_levels(damped)
        applied["dc_spread"] = dc_spread(damped)
        emit_dc_record(applied, rank=rank)
        return damped
    except DoubleCountingUnreachable as error:
        _banner(
            rank, "DOUBLE COUNTING SEARCH COULD NOT REACH ITS TARGET. Returning initial DC unchanged.", error, False
        )
        archive.record_dc_audit_attr(filename, label, comm, "DC search unreachable", str(error))
    except Exception as error:
        _banner(rank, "DOUBLE COUNTING SEARCH FAILED. Returning initial DC unchanged.", error, True)
        archive.record_dc_audit_attr(filename, label, comm, "DC search failed", f"{error!r}\n{traceback.format_exc()}")
    return None


__all__ = ["run_double_counting"]
