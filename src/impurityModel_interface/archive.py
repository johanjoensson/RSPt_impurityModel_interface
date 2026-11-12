"""Everything the wrapper keeps in ``impurityModel_data.h5`` -- the only module that imports h5py.

The file holds one group ``"<label> <iteration>"`` per DMFT iteration.  The solver problem in it
(options, meshes, blocks, ``H solver``) is written by ``impurityModel.api.write_selfenergy_group``;
what is the wrapper's own is the iteration counter, the bath fit (stored so the DC pass and the
self-energy pass of one iteration share it), the damped double counting, the sector and cap the DC
search settled on, and the results handed back to RSPt.

Reads happen on rank 0 and the caller broadcasts (:func:`impurityModel_interface.mpi.on_root`), so
every rank sees the same answer.  Writes happen on rank 0 only.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

import h5py as h5
import numpy as np
from impurityModel.api import write_selfenergy_group
from rspt2spectra.utils import rotate_matrix

from .mpi import on_root

FILENAME = "impurityModel_data.h5"


def write_dataset(group, name, data):
    """Create dataset ``name`` in ``group``, replacing an existing one.

    Re-entering a partly written iteration group (after a crash-restart, or because the DC and
    self-energy passes share one group) must not fail on "name already exists".
    """
    if name in group:
        del group[name]
    group.create_dataset(name, data=data)


def _group_name(label, iteration):
    return f"{label.strip()} {iteration}"


def _last_iteration(f):
    return int(f.attrs.get("last iteration", 1))


def _iteration_group(f, label):
    """This iteration's group, created if the self-energy pass has not written it yet."""
    name = _group_name(label, _last_iteration(f))
    return f[name] if name in f else f.create_group(name)


def _read_on_root(comm, read):
    """``read()`` on rank 0 (``None`` when the file or group is not there yet), broadcast."""

    def safe():
        try:
            return read()
        except (OSError, KeyError):
            return None

    return on_root(comm, safe)


# --- bath fit ------------------------------------------------------------------------------


def load_stored_fit(filename, label, fingerprint, block_structure):
    """The bath fit stored for this iteration, if it was made from exactly this input.

    Returns ``(fit, stored_fingerprint, iteration)``.  ``fit`` is ``(ebs_star, vs_star, shifts)``
    when the fingerprint and block structure match, else ``None``; ``stored_fingerprint`` is the
    one found (``None`` if nothing is stored).  This is what lets the self-energy pass reuse the fit
    of the double-counting pass within one DMFT step, while a new iteration (new hybridization) or
    another block partition refits.  Rank 0 only.
    """
    try:
        with h5.File(filename, "r") as ar:
            iteration = ar.attrs["last iteration"]
            fit_g = ar[f"{label} {iteration}/Bath fit"]
            stored = fit_g.attrs.get("hyb fingerprint", None)
            if stored != fingerprint or fit_g.attrs.get("block structure", "") != repr(block_structure):
                return None, stored, iteration
            fit = tuple(
                [np.array(fit_g[f"{key}/{index}"]) for index in block_structure.inequivalent_blocks]
                for key in ("ebs_star", "vs_star", "shifts")
            )
            return fit, stored, iteration
    except (FileNotFoundError, KeyError):
        return None, None, None


def previous_bath_energies(filename, label, block_structure):
    """The bath energies of the most recent stored fit, one array per inequivalent block.

    For the warm start and ``freeze_bath_energies``: unlike :func:`load_stored_fit` this ignores the
    fingerprint and the iteration boundary -- it wants the last energies the fitter produced.
    ``None`` when nothing is stored yet, and also when the latest fit was made on another block
    partition (its per-block energies would seed the wrong blocks).  Rank 0 only.
    """
    try:
        with h5.File(filename, "r") as ar:
            for iteration in range(_last_iteration(ar), 0, -1):
                fit_g = ar.get(f"{label.strip()} {iteration}/Bath fit")
                if fit_g is None or "ebs_star" not in fit_g:
                    continue
                stored_structure = fit_g.attrs.get("block structure", None)
                if stored_structure is not None and stored_structure != repr(block_structure):
                    print(
                        f"The previous bath fit ({label.strip()} {iteration}) was made on another block "
                        "partition; its bath energies are not reused.",
                        flush=True,
                    )
                    return None
                return [
                    np.unique(np.asarray(fit_g[f"ebs_star/{index}"], dtype=float))
                    for index in block_structure.inequivalent_blocks
                ]
    except (OSError, KeyError):
        pass
    return None


def store_bath_fit(filename, label, fingerprint, block_structure, fit):
    """Store a freshly made fit under a new iteration.  Rank 0 only.

    A fit that was not reused starts the next iteration: the counter advances past every group the
    cluster already has.
    """
    with h5.File(filename, "a") as ar:
        if "last iteration" not in ar.attrs:
            ar.attrs["last iteration"] = 1
        iteration = ar.attrs["last iteration"]
        while _group_name(label, iteration) in ar:
            iteration += 1
        ar.attrs["last iteration"] = iteration
        fit_g = ar.require_group(f"{label} {iteration}/Bath fit")
        fit_g.attrs["hyb fingerprint"] = fingerprint
        fit_g.attrs["block structure"] = repr(block_structure)
        if fit.warm_start_info is not None:
            # Audit trail, one entry per inequivalent block: whether the previous bath was kept or
            # a better global fit replaced it ("cold" when there was nothing to start from), and the
            # two costs the decision compared (NaN when not computed).
            fit_g.attrs["warm start"] = [info["status"] for info in fit.warm_start_info]
            for key in ("cost warm", "cost global"):
                fit_g.attrs[key] = np.array([info.get(key, np.nan) for info in fit.warm_start_info], dtype=float)
        for key, values in (("vs_star", fit.vs_star), ("ebs_star", fit.ebs_star), ("shifts", fit.shifts)):
            group = fit_g.require_group(key)
            for index, value in zip(block_structure.inequivalent_blocks, values):
                write_dataset(group, f"{index}", value)


# --- the solver problem --------------------------------------------------------------------


def write_problem(
    filename,
    label,
    *,
    line,
    solver_line,
    tau,
    delta,
    w,
    iw,
    h_solver,
    h_dft,
    h_bath,
    v,
    u4,
    dc,
    corr_to_cf,
    rot_to_spherical,
    layout,
):
    """Write the solver problem of this iteration into its group.  Rank 0 only.

    ``line`` is the parsed :class:`~impurityModel_interface.solver_line.SolverLine` (with the
    temperature on its basis options), ``layout`` the ``(impurity, valence, conduction)`` orbital
    indices.  Next to the solver's own record this stamps the fit options, the bath size, the
    ``corr_to_cf`` rotation (so the archive is self-contained) and the versions that made it.
    """
    impurity, valence, conduction = layout
    with h5.File(filename, "a") as f:
        if "last iteration" not in f.attrs:
            f.attrs["last iteration"] = 1
        group = _iteration_group(f, label)
        write_selfenergy_group(
            group,
            basis=line.basis,
            solver=line.solver,
            delta=delta,
            w=w,
            iw=iw,
            h_solver=h_solver,
            u4=u4,
            dc=dc,
            rot_to_spherical=rot_to_spherical,
            impurity_orbitals=impurity,
            valence_orbitals=valence,
            conduction_orbitals=conduction,
            h_dft=h_dft,
            h_bath=h_bath,
            v=v,
        )
        group.attrs["n_baths"] = line.n_baths
        for key, value in line.fit.to_attrs().items():
            group.attrs[key] = value
        group.attrs["solver line"] = solver_line.strip()
        for package in ("impurityModel", "rspt2spectra", "impurityModel_interface"):
            try:
                group.attrs[f"{package} version"] = package_version(package)
            except PackageNotFoundError:
                pass
        write_dataset(group, "corr_to_cf", corr_to_cf)


def write_selfenergy(filename, label, results, sig_static, sig_real, sig, rotate_green, u):
    """Store the self-energy pass's results (already in RSPt's basis where RSPt receives them).

    ``rotate_green(G, u)`` rotates the solver's Green's functions back to the correlated basis.
    Rank 0 only.
    """
    with h5.File(filename, "a") as f:
        group = f[_group_name(label, f.attrs["last iteration"])]
        write_dataset(group, "thermal_rho", results["thermal_rho"])
        write_dataset(group, "rhos", results["rhos"])
        write_dataset(group, "Sigma Static", sig_static)
        write_dataset(group, "Sigma real", sig_real)
        write_dataset(group, "Sigma Matsubara", sig)
        if results["gs_matsubara"] is not None:
            write_dataset(group, "Gimp Matsubara", rotate_green(results["gs_matsubara"], u))
        if results["gs_realaxis"] is not None:
            write_dataset(group, "Gimp real", rotate_green(results["gs_realaxis"], u))


# --- double counting -----------------------------------------------------------------------


def previous_damped_dc(filename, label, comm, corr_to_cf=None):
    """The double counting this interface returned on the most recent earlier iteration.

    Stored in the solver basis of the iteration that produced it, which is the fitting basis and
    changes from one iteration to the next.  With ``corr_to_cf`` (the current iteration's
    solver-basis-to-correlated-basis rotation) the stored matrix is carried through the correlated
    basis into the current solver basis, so the damping adds like to like; a group without a stored
    ``corr_to_cf`` is returned as it is.

    This -- not the ``sig_dc`` RSPt hands in -- is the anchor the damping pulls toward
    (``doc/double_counting.md``).  Searches backwards from the current iteration for the newest group
    holding ``"DC damped"``, so an iteration that skipped or failed the search does not break the
    chain.  ``None`` when there is no earlier answer.
    """

    def read():
        with h5.File(filename, "r") as f:
            for iteration in range(_last_iteration(f) - 1, 0, -1):
                name = _group_name(label, iteration)
                if name in f and "DC damped" in f[name]:
                    previous = np.asarray(f[name]["DC damped"])
                    if corr_to_cf is not None and "corr_to_cf" in f[name]:
                        previous_c = np.asarray(f[name]["corr_to_cf"])
                        if previous_c.shape == np.shape(corr_to_cf) and previous.shape == previous_c.shape:
                            return rotate_matrix(rotate_matrix(previous, np.conj(previous_c.T)), np.asarray(corr_to_cf))
                        print(
                            f"The stored double counting of {name} has another basis shape; "
                            "damping against it unrotated.",
                            flush=True,
                        )
                    return previous
        return None

    return _read_on_root(comm, read)


def write_double_counting(filename, label, *, alpha, guess_fingerprint, converged, damped, sector, cap):
    """Record the double counting of this iteration.  Rank 0 only.

    The full damped matrix is stored (never the bare mu) and fingerprinted against the guess it was
    computed from, so the archive can be audited for whether the next iteration's incoming DC is a
    continuation of this answer.  ``converged`` is what the criterion says, before damping; the two
    coincide on the first iteration.
    """
    with h5.File(filename, "a") as f:
        group = _iteration_group(f, label)
        group.attrs["DC damping alpha"] = alpha
        group.attrs["DC guess fingerprint"] = guess_fingerprint
        write_dataset(group, "DC converged", converged)
        write_dataset(group, "DC damped", damped)
        if sector is not None:
            group.attrs["DC ground state sector"] = int(sector)
        if cap is not None:
            group.attrs["DC ground state cap"] = int(cap)


def record_dc_audit_attr(filename, label, comm, attr, value):
    """Stamp one audit attribute on this iteration's group (``"DC search unreachable"`` / ``"failed"``).

    Lets a charge-self-consistent run be audited for which iterations determined a double counting
    and which fell back to the previous value.  Writes on rank 0; nothing reads it back, so there is
    no broadcast.  A failed stamp is reported and swallowed: this runs in a failure handler, and an
    exception on rank 0 alone would leave the other ranks waiting at the next barrier.
    """
    if comm is not None and comm.rank != 0:
        return
    try:
        with h5.File(filename, "a") as f:
            _iteration_group(f, label).attrs[attr] = value
    except Exception as error:
        print(f"Could not record {attr!r} in {filename}: {error!r}", flush=True)


def _iteration_attr(filename, label, comm, attr):
    def read():
        with h5.File(filename, "r") as f:
            name = _group_name(label, _last_iteration(f))
            return f[name].attrs.get(attr, None) if name in f else None

    value = _read_on_root(comm, read)
    return None if value is None else int(value)


def double_counting_sector(filename, label, comm):
    """The charge sector the DC search settled on this iteration (``None``: it ran no search).

    Written by the ``rspt_dc_flag=1`` call and read by the ``rspt_dc_flag=0`` one RSPt makes a moment
    later.  Broadcast from rank 0: it decides the nominal occupation every rank builds its basis from.
    """
    return _iteration_attr(filename, label, comm, "DC ground state sector")


def double_counting_cap(filename, label, comm):
    """The auto determinant cap the DC search resolved this iteration (``None``: none recorded).

    An auto cap is sized from the memory free *now*, so the self-energy solve, made as a separate
    call, would otherwise run at a different one.  It uses this as a ceiling on its own auto cap (it
    can only lower it; a cap set on the solver line is never touched).
    """
    return _iteration_attr(filename, label, comm, "DC ground state cap")
