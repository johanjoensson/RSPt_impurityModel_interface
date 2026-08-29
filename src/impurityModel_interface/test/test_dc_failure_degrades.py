"""A DC-search failure must not abort a charge-self-consistent run.

The DC branch of ``_run_impmod_ed`` used to answer *any* exception from the criterion
(``fixed_gap_dc`` and friends) by writing ``inf`` into ``sig_dc`` and calling
``comm.Abort`` -- one bad sector solve killed the whole multi-hour RSPt job. It now
degrades the way the ``DoubleCountingUnreachable`` branch already did: leave RSPt's incoming
DC untouched, stamp an audit attribute on the archive, and let the loop continue. This pins
the shared archive-recording helper both branches use.
"""

import numpy as np
import pytest

pytest.importorskip("mpi4py")
h5 = pytest.importorskip("h5py")
from mpi4py import MPI  # noqa: F401,E402

from impurityModel_interface.lib import _record_dc_audit_attr  # noqa: E402

LABEL = "Ni 1"


def test_records_onto_an_existing_cluster_group(tmp_path):
    path = tmp_path / "archive.h5"
    with h5.File(path, "w") as f:
        f.attrs["last iteration"] = 3
        f.create_group(f"{LABEL} 3").attrs["something else"] = "kept"

    _record_dc_audit_attr(path, LABEL, None, "DC search failed", "LinAlgError('SVD did not converge')")

    with h5.File(path, "r") as f:
        g = f[f"{LABEL} 3"]
        assert g.attrs["DC search failed"] == "LinAlgError('SVD did not converge')"
        assert g.attrs["something else"] == "kept"


def test_creates_the_group_when_the_selfenergy_pass_has_not_written_it(tmp_path):
    path = tmp_path / "archive.h5"
    with h5.File(path, "w") as f:
        f.attrs["last iteration"] = 7

    _record_dc_audit_attr(path, LABEL, None, "DC search unreachable", "no root in range")

    with h5.File(path, "r") as f:
        assert f[f"{LABEL} 7"].attrs["DC search unreachable"] == "no root in range"


def test_is_a_noop_on_non_root_ranks(tmp_path):
    path = tmp_path / "archive.h5"
    with h5.File(path, "w") as f:
        f.attrs["last iteration"] = 1

    class _FakeComm:
        rank = 1

    _record_dc_audit_attr(path, LABEL, _FakeComm(), "DC search failed", "boom")

    with h5.File(path, "r") as f:
        assert f"{LABEL} 1" not in f or "DC search failed" not in f[f"{LABEL} 1"].attrs


def test_default_iteration_is_one_without_the_attribute(tmp_path):
    path = tmp_path / "archive.h5"
    with h5.File(path, "w"):
        pass

    _record_dc_audit_attr(path, LABEL, None, "DC search failed", "boom")

    with h5.File(path, "r") as f:
        assert f[f"{LABEL} 1"].attrs["DC search failed"] == "boom"
