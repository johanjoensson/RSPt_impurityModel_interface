"""Initialize MPI before the interface sets ``mpi4py.rc.initialize = False``.

Embedded in RSPt, Fortran has initialized MPI by the time this package is imported; under pytest
nothing has, and the solver touches ``MPI.COMM_WORLD`` while it is imported.
"""

from mpi4py import MPI  # noqa: F401
