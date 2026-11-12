"""Glue between RSPt and the impurityModel ED solver.  See the README for who owns what."""

from . import entry  # noqa: F401 - registers the cffi callback and sets the MPI/OpenMP environment first
