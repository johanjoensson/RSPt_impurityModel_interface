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

## Supported orbital representations
RSPt can define the correlated orbitals in spherical harmonics, crystal-field
(cubic) combinations — including subsets of a shell such as t2g or eg only —
user supplied `Irr` projections, and the relativistic JJ basis, with or
without spin polarization and spin-orbit coupling. The wrapper handles all of
these: RSPt sends the rotation matrices with one spin block when the
calculation is neither spin polarized nor relativistic (`nspmat = 1`), and the
wrapper duplicates that block onto both spins (`reconstruct_rotations` in
`lib.py`); otherwise the matrices are used as received. The impurityModel ED
solver requires the cluster to contain exactly one correlated orbital set and
nothing else. When the correlated set spans only part of a shell, the
rotation to spherical harmonics is rectangular and impurityModel skips the
L/S/J observables (everything else, including the self-energy and double
counting, is unaffected).

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

`impurityModel` is built in its optimized `release` mode by default. Choose the
mode with `-DIMPURITYMODEL_BUILD=release|debug|safe` (passed on as the
`IMPURITYMODEL_BUILD` environment variable); `-DCMAKE_BUILD_TYPE=Debug` makes
`debug` the default. Use `safe` for builds shared across heterogeneous cluster
nodes, since `release` compiles with `-march=native`. The value is cached, so
to change it in an existing build directory pass `-DIMPURITYMODEL_BUILD=...`
explicitly.

By default `impurityModel` and `rspt2spectra` are installed from GitHub, which
replaces any local editable installs in the environment. To develop against
local checkouts instead, configure with `-DEDITABLE_DEPS=ON`, e.g.
```
cmake -B build -S . -DEDITABLE_DEPS=ON \
      -DIMPURITYMODEL_SOURCE_DIR=$HOME/src/impurityModel \
      -DRSPT2SPECTRA_SOURCE_DIR=$HOME/src/rspt2spectra
```
Both are then installed with `pip install --editable`, and the interface itself
with `--no-deps` so the GitHub copies are not pulled back in. The source
directories default to `dependencies/impurityModel` and
`dependencies/rspt2spectra`; a directory that does not exist is cloned from
`IMPURITYMODEL_GIT_URL` / `RSPT2SPECTRA_GIT_URL` (the forks by default) at
configure time.
