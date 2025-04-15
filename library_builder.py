import cffi

ffibuilder = cffi.FFI()

with open("src/run_impurityModel.h") as f:
    data = "".join([line for line in f if not line.startswith("#")])
    ffibuilder.embedding_api(data)

ffibuilder.set_source(
    "run_impurityModel",
    r"""
    #include "run_impurityModel.h"
""",
)

ffibuilder.embedding_init_code(
    r"""
    import impurityModel_interface
"""
)

ffibuilder.emit_c_code("src/run_impurityModel.c")
# ffibuilder.compile(target="libimpomod_interface.*", verbose=True)
