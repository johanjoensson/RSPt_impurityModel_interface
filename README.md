# RSPt_impurityModel_interface
Interface for running impurityModel from RSPt

## Responsibilities
This package is glue only: it marshals RSPt's Fortran arrays through CFFI
(views, no copies), rotates between the correlated and CF bases, parses the
solver line, keeps the HDF5 archive (including bath-fit reuse between the DC
and Σ passes of one DMFT step) and drives MPI. The physics lives upstream:
[rspt2spectra](https://github.com/johanjoensson/rspt2spectra) owns everything
from the hybridization function to the non-interacting Hamiltonian h0 (block
partitioning, bath fitting, bath geometries), and
[impurityModel](https://github.com/johanjoensson/impurityModel) owns the
many-body ED solve (self-energy, double counting) behind its stable
`impurityModel.api` façade. The two upstream packages are independent of each
other; this wrapper is their only meeting point.

## Requirements
The idea is that all python requirements will be installed as part of the CMake
configuration. It is usually a good idea to set up some form of virtual
environment (or conda, or miniconda, or ...) for managing the python
environment.

Note that `impurityModel` ships a compiled extension (`ManyBodyUtils`) built
for a specific Python minor version. If you switch Python versions (e.g.
3.13 -> 3.14), reinstall/rebuild `impurityModel` for the new interpreter,
otherwise importing this interface fails with an `ImportError` mentioning
`ManyBodyUtils`.

## Build instructions
To build the shared library `libimpurityModel_interface.so` first generate all
the files, and update your python environment with the command
`cmake -B build -S .`, then build the library `cmake --build build`. The shared
library, `libimpurityModel_interface.so` is in the `build` directory. Recompile
RSPt, add `-DEXTERNAL_ED` flag to `FCPPFLAGS` and `CPPFLAGS` and link with this
library (`-limpurityModel_interface`).

To build the threaded version of `impurityModel`, configure with
`-DPARALLEL_IMPURITYMODEL=ON`, i.e.
`cmake -B build -S . -DPARALLEL_IMPURITYMODEL=ON`. This sets
`IMPURITYMODEL_PARALLEL=1` when `impurityModel` is built during
`cmake --build build`.
