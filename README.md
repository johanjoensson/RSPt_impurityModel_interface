# RSPt_impurityModel_interface
Interface for running impurityModel from RSPt

## Requirements
The idea is that all python requirements will be installed as part of the CMake
configuration. It is usually a good idea to set up some form of virtual
environment (or conda, or miniconda, or ...) for managing the python
environment.

## Build instructions
To build the shared library `libimpurityModel_interface.so` first generate all
the files, and update your python environment with the command
`cmake -B build -S .`, then build the library `cmake --build build`. The shared
library, `libimpurityModel_interface.so` is in the `build` directory. Recompile
RSPt, add `-DEXTERNAL_ED` flag to `FCPPFLAGS` and `CPPFLAGS` and link with this
library (`-limpurityModel_interface`).
