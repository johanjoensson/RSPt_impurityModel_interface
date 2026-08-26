"""Tuning knobs taken from an input file on the RSPt-driven path.

RSPt configures the solver through two fixed-width strings and a label, so the callback has
no argument to pass a file path through and RSPt's own sources are not ours to change. The
file is therefore found by convention. What these tests pin is the part that is easy to get
wrong: the knobs must be scoped to one invocation, because the callback runs once per cluster
label per self-consistency iteration plus once more for the double-counting pass.
"""

import os

import pytest

from impurityModel.api import find_environment_file
from impurityModel_interface import lib

KNOB = "GF_BICGSTAB_ATOL"

INPUT_FILE = f"""
[format]
version = [1, 0]

[environment]
{KNOB} = 1e-9
"""


@pytest.fixture
def in_a_run_directory(tmp_path, monkeypatch):
    """A working directory holding the conventional input file."""
    monkeypatch.delenv("IMPURITYMODEL_INPUT", raising=False)
    monkeypatch.delenv(KNOB, raising=False)
    (tmp_path / "impurityModel.toml").write_text(INPUT_FILE)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _call_with_recorder(monkeypatch, recorder):
    """Invoke the wrapper with the real solve replaced by a recorder."""
    monkeypatch.setattr(lib, "_run_impmod_ed", recorder)
    return lib.run_impmod_ed()


def test_the_knob_is_set_during_the_call_and_gone_after(in_a_run_directory, monkeypatch):
    """Without the restore, self-consistency iteration two inherits iteration one's knobs."""
    seen = {}

    def recorder(*args):
        seen["during"] = os.environ.get(KNOB)
        return 0

    _call_with_recorder(monkeypatch, recorder)
    # Compared by value: the knob is stored as the string os.environ takes, and TOML's 1e-9
    # round-trips through float as "1e-09".
    assert float(seen["during"]) == pytest.approx(1e-9)
    assert KNOB not in os.environ


def test_a_value_already_in_the_environment_wins(in_a_run_directory, monkeypatch):
    """The rule this module already applies to OMP_NUM_THREADS: a shell value beats a file."""
    monkeypatch.setenv(KNOB, "1e-7")
    seen = {}

    def recorder(*args):
        seen["during"] = os.environ.get(KNOB)
        return 0

    _call_with_recorder(monkeypatch, recorder)
    assert seen["during"] == "1e-7"
    assert os.environ[KNOB] == "1e-7"


def test_no_input_file_means_no_knobs_and_no_complaint(tmp_path, monkeypatch):
    """The overwhelmingly common case: RSPt runs, there is no such file, nothing happens."""
    monkeypatch.delenv("IMPURITYMODEL_INPUT", raising=False)
    monkeypatch.delenv(KNOB, raising=False)
    monkeypatch.chdir(tmp_path)
    assert find_environment_file() is None

    seen = {}

    def recorder(*args):
        seen["during"] = os.environ.get(KNOB)
        return 7

    assert _call_with_recorder(monkeypatch, recorder) == 7
    assert seen["during"] is None


def test_the_result_of_the_solve_is_passed_straight_through(in_a_run_directory, monkeypatch):
    """The wrapper must be transparent: RSPt reads this return value."""
    assert _call_with_recorder(monkeypatch, lambda *args: 42) == 42


def test_the_arguments_reach_the_solve_unchanged(in_a_run_directory, monkeypatch):
    """26 positional arguments from cffi; the wrapper must not reorder or drop any."""
    captured = {}

    def recorder(*args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(lib, "_run_impmod_ed", recorder)
    lib.run_impmod_ed("label", "solver line", 3, 4.5)
    assert captured["args"] == ("label", "solver line", 3, 4.5)


def test_only_the_environment_table_is_read(in_a_run_directory, monkeypatch):
    """A file describing a model too must not have that model used: RSPt supplies its own."""
    (in_a_run_directory / "impurityModel.toml").write_text(
        INPUT_FILE + '\n[hamiltonian.file]\npath = "nowhere.h0"\n[selfenergy]\n'
    )
    seen = {}
    _call_with_recorder(monkeypatch, lambda *args: seen.setdefault("during", os.environ.get(KNOB)))
    assert float(seen["during"]) == pytest.approx(1e-9)
