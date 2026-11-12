"""Initialize MPI before the interface sets ``mpi4py.rc.initialize = False``.

Embedded in RSPt, Fortran has initialized MPI by the time this package is imported; under pytest
nothing has, and the solver touches ``MPI.COMM_WORLD`` while it is imported.
"""

from pathlib import Path

import pytest
from mpi4py import MPI


@pytest.fixture
def shared_tmp_path(tmp_path_factory):
    """One working directory for all ranks, for tests where only rank 0 writes.

    The plain ``tmp_path`` is per rank, which is what tests that write on every rank need.
    """
    path = MPI.COMM_WORLD.bcast(str(tmp_path_factory.mktemp("shared")), root=0)
    return Path(path)
