"""The door RSPt knocks on: the cffi callback, the raw pointers and the process environment.

Nothing here knows any physics.  :func:`run_impmod_ed` is what RSPt calls; it wraps the solve in
whatever ``[environment]`` asks for and turns every failure into the ``-1`` RSPt reads as one.
:func:`_run_impmod_ed` turns RSPt's pointers into numpy views and calls :func:`solve.solve`.
"""

import mpi4py

# mpi4py.rc must be set before importing MPI, and OMP_NUM_THREADS before the thread-spawning
# numeric libraries (numpy, h5py), hence the E402 below.
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
import sys  # noqa: E402
import traceback  # noqa: E402
from os import environ  # noqa: E402

from mpi4py import MPI  # noqa: E402

if "OMP_NUM_THREADS" not in environ:
    print("OMP_NUM_THREADS will be set to 1 from now on!.", file=sys.stderr)
    environ["OMP_NUM_THREADS"] = "1"

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

try:
    import impurityModel.api as _solver_api
    import rspt2spectra as _fit_api
except ImportError as import_error:
    raise ImportError(
        f"Failed to import impurityModel or rspt2spectra under Python "
        f"{sys.version_info.major}.{sys.version_info.minor}: {import_error}\n"
        "impurityModel ships a compiled extension (ManyBodyUtils) built for a "
        "specific Python minor version. If the import above mentions "
        "ManyBodyUtils, rebuild/reinstall impurityModel for this interpreter, "
        "e.g. `pip install --force-reinstall --no-build-isolation "
        "<path-to-impurityModel>`."
    ) from import_error

#: The versions of the two upstream interfaces this wrapper is written against.
EXPECTED_SOLVER_API = 1
EXPECTED_FIT_API = 1
for _name, _module, _expected in (
    ("impurityModel", _solver_api, EXPECTED_SOLVER_API),
    ("rspt2spectra", _fit_api, EXPECTED_FIT_API),
):
    if getattr(_module, "API_VERSION", None) != _expected:
        raise ImportError(
            f"{_name} has API version {getattr(_module, 'API_VERSION', None)!r}, this interface needs "
            f"{_expected}. Rebuild/reinstall {_name} (and the others) for this interpreter."
        )

from impurityModel.api import apply_environment, find_environment_file, load_environment  # noqa: E402

from .solve import solve  # noqa: E402


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
    """Wrap RSPt's raw pointers as numpy views and hand them to :func:`solve.solve`.

    The only function that touches a cffi handle. Everything the physics needs -- decoded
    strings and aliased numpy arrays -- is produced here, so ``solve`` is plain Python and can be
    called directly from a test.

    The views alias RSPt's memory; they are never copies. The four outputs (``sig``,
    ``sig_real``, ``sig_static``, ``sig_dc``) are written back by in-place assignment, and
    the inputs matter just as much -- the ``comm.Bcast`` calls in ``solve`` write
    *through* ``h_dft``, ``u4`` and ``hyb`` into RSPt's own memory on non-root ranks. An
    ``np.asarray(..., dtype=complex)`` anywhere below would silently break both, and no
    single-rank test would notice.

    ``n_iw``, ``n_w``, ``size_real`` and ``size_complex`` exist only to compute the byte
    counts below, and are deliberately not forwarded.
    """

    def view(pointer, shape, dtype=complex, itemsize=size_complex):
        count = int(np.prod(shape))
        return np.ndarray(buffer=ffi.buffer(pointer, count * itemsize), shape=shape, order="F", dtype=dtype)

    return solve(
        label=ffi.string(rspt_label, 18).decode("ascii"),
        solver_line=ffi.string(rspt_solver_line, 100).decode("ascii"),
        dc_line=ffi.string(rspt_dc_line, 100).decode("ascii"),
        dc_flag=rspt_dc_flag,
        h_dft=view(rspt_h_dft, (n_orb, n_orb)),
        u4=view(rspt_u4, (n_orb,) * 4),
        hyb=view(rspt_hyb, (n_orb, n_orb, n_w)),
        iw=view(rspt_iw, (n_iw,), float, size_real),
        w=view(rspt_w, (n_w,), float, size_real),
        sig=view(rspt_sig, (n_orb, n_orb, n_iw)),
        sig_real=view(rspt_sig_real, (n_orb, n_orb, n_w)),
        sig_static=view(rspt_sig_static, (n_orb, n_orb)),
        sig_dc=view(rspt_sig_dc, (n_orb, n_orb)),
        corr_to_spherical_in=view(rspt_corr_to_spherical, (n_orb, n_orb_full)),
        corr_to_cf_in=view(rspt_corr_to_cf, (n_orb, n_rot_cols)),
        n_orb=n_orb,
        n_rot_cols=n_rot_cols,
        n_orb_full=n_orb_full,
        eim=eim,
        tau=tau,
        verbosity=verbosity,
    )


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
        # Diagnostics go to sys.__stderr__, never print(): solve redirects sys.stdout to the
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
        # solve restores sys.stdout itself on every path (redirected_stdout); this is the net
        # for anything that escapes before that context is entered or after it is left.
        if sys.stdout is not stdout_save:
            try:
                sys.stdout.close()
            except Exception:
                pass
            sys.stdout = stdout_save
