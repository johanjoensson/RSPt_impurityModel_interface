"""RSPt must never read a failed solve as a successful one.

RSPt's only channel for "this went wrong" is the integer this callback returns:
``green_impmod_interface.F90`` does ``if (er .ne. 0) call stopgreen(...)``, and ``stopgreen``
aborts the job. Anything that returns 0 after a failure lets the DMFT loop continue with the
previous iteration's ``acluster%sig`` still in place -- a wrong answer that looks like a
converging run.

That was the behaviour until the ``error=-1`` on the ``def_extern`` and the handler in
``run_impmod_ed``: cffi's default for an uncaught exception in a callback is to report it as
unraisable and return **0**, and most of the solve ran outside any ``try``. These tests pin
the return value, which is the part RSPt actually reads.
"""

import sys

import pytest

pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: F401

from impurityModel_interface import lib


def _call(monkeypatch, impl):
    """Invoke the entry point with the marshalling half replaced."""
    monkeypatch.setattr(lib, "_run_impmod_ed", impl)
    return lib.run_impmod_ed()


def test_success_is_forwarded_unchanged(monkeypatch):
    """The happy path still returns what the solve returned, guard or no guard."""
    assert _call(monkeypatch, lambda *args: 0) == 0


def test_a_deliberate_nonzero_is_forwarded_unchanged(monkeypatch):
    assert _call(monkeypatch, lambda *args: -7) == -7


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("malformed solver line"),
        ValueError("bath fit did not converge"),
        OSError("No space left on device"),
        MemoryError(),
    ],
)
def test_an_exception_becomes_a_nonzero_return(monkeypatch, exc):
    """Every one of these used to return 0, and RSPt carried on with a stale selfenergy."""

    def boom(*args):
        raise exc

    assert _call(monkeypatch, boom) != 0


def test_even_a_baseexception_becomes_a_nonzero_return(monkeypatch):
    """A signal or a MemoryError mid-solve is still a failed solve, not a successful one."""

    def boom(*args):
        raise KeyboardInterrupt

    assert _call(monkeypatch, boom) != 0


def test_the_traceback_does_not_go_to_the_redirected_stdout(monkeypatch, capfd):
    """_solve points sys.stdout at the per-cluster .out file; the diagnostic must dodge it.

    capfd, not capsys: the handler writes to ``sys.__stderr__`` on purpose, so that it still
    lands somewhere visible when ``sys.stderr`` has been swapped out too. Only file-descriptor
    level capture sees that.
    """

    def boom(*args):
        raise RuntimeError("kaboom")

    assert _call(monkeypatch, boom) != 0
    captured = capfd.readouterr()
    assert "kaboom" in captured.err
    assert "kaboom" not in captured.out


def test_stdout_is_restored_when_the_solve_leaves_it_redirected(monkeypatch, tmp_path):
    """Without the restore, the rest of the process writes into a half-written .out file."""
    before = sys.stdout
    out_file = tmp_path / "impurityModel-Ni-3d.out"

    def redirect_then_fail(*args):
        sys.stdout = out_file.open("w")
        print("partial output from a solve that is about to fail")
        raise RuntimeError("failed after redirecting")

    assert _call(monkeypatch, redirect_then_fail) != 0
    assert sys.stdout is before
    # Closed, so the partial output was flushed rather than lost with the process.
    assert out_file.read_text().strip() == "partial output from a solve that is about to fail"


def test_a_failure_before_the_redirect_leaves_stdout_alone(monkeypatch):
    before = sys.stdout

    def boom(*args):
        raise RuntimeError("failed before redirecting")

    assert _call(monkeypatch, boom) != 0
    assert sys.stdout is before


class _RecordingFFI:
    """Stands in for the ffi handle cffi publishes once it has registered its module."""

    def __init__(self):
        self.registrations = []

    def def_extern(self, name=None, error=None):
        def register(func):
            self.registrations.append((name, error, func))
            return func

        return register


def test_bind_rspt_callback_registers_with_the_error_backstop(monkeypatch):
    """The third silent-zero route: an early import leaves the callback unregistered.

    A sitecustomize.py, usercustomize.py or .pth file imports this package during
    Py_InitializeEx, before cffi has registered ``run_impurityModel``. lib.py then takes the
    _FFIStub fallback and the decorator is a no-op, so cffi answers every call with "no code
    was attached to it yet ... Returning 0" -- and RSPt reads 0 as a successful solve.
    """
    fake = _RecordingFFI()
    monkeypatch.setitem(sys.modules, "run_impurityModel", type("M", (), {"ffi": fake}))
    monkeypatch.setattr(lib, "ffi", object())  # whatever the stub left behind

    lib.bind_rspt_callback()

    assert lib.ffi is fake, "ffi.string/ffi.buffer would still resolve to the stub"
    assert len(fake.registrations) == 1
    name, error, func = fake.registrations[0]
    assert name == "run_impmod_ed_py"
    assert error == -1, "without this the re-registered callback returns 0 on an exception"
    assert func is lib.run_impmod_ed


def test_bind_rspt_callback_is_a_no_op_on_the_normal_path(monkeypatch):
    """RSPt's first call imports the package after cffi is ready; nothing to repair."""
    fake = _RecordingFFI()
    monkeypatch.setitem(sys.modules, "run_impurityModel", type("M", (), {"ffi": fake}))
    monkeypatch.setattr(lib, "ffi", fake)

    lib.bind_rspt_callback()

    assert fake.registrations == []
