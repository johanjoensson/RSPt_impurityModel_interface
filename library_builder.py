import cffi

ffibuilder = cffi.FFI()

with open("run_impurityModel.h") as f:
    data = "".join([line for line in f if not line.startswith("#")])
    ffibuilder.embedding_api(data)

# The public symbol RSPt links against. It is NOT the cffi callback: cffi's own fallback for
# a failed bootstrap prints one line to stderr, zeroes the result buffer and returns 0, and
# RSPt reads 0 as a successful solve (green_impmod_interface.F90: `if (er .ne. 0) call
# stopgreen(...)`), so the DMFT loop would carry on with a stale acluster%sig. cffi_start_python()
# is emitted by the embedding support above this block and returns -1 if starting the
# interpreter or running the init code failed; checking it is the only way to see that case.
#
# The sibling failure -- an exception inside the callback, which also returned 0 by default --
# is handled on the Python side by @ffi.def_extern(..., error=-1) in lib.py.
ffibuilder.set_source(
    "run_impurityModel",
    r"""
    #include <stdio.h>
    #include "run_impurityModel.h"

CFFI_DLLEXPORT int run_impmod_ed(char label[], char solver_param[], char dc_param[],
                                 int dc_flag, double _Complex *U_mat,
                                 double _Complex *hyb, double _Complex *h_dft,
                                 double _Complex *sig, double _Complex *sig_real,
                                 double _Complex *sig_static, double _Complex *sig_dc,
                                 double *iw, double *w,
                                 double _Complex *corr_to_spherical,
                                 double _Complex *corr_to_cf, size_t n_orb,
                                 size_t n_rot_cols, size_t n_orb_full, size_t n_iw,
                                 size_t n_w, double eim, double tau, int verbosity,
                                 size_t size_real, size_t size_complex)
{
    if (cffi_start_python() != 0) {
        fprintf(stderr,
                "impurityModel: the embedded Python interpreter failed to start. "
                "Returning -99 to RSPt.\n");
        fflush(stderr);
        return -99;
    }
    return run_impmod_ed_py(label, solver_param, dc_param, dc_flag, U_mat, hyb, h_dft,
                            sig, sig_real, sig_static, sig_dc, iw, w, corr_to_spherical,
                            corr_to_cf, n_orb, n_rot_cols, n_orb_full, n_iw, n_w, eim,
                            tau, verbosity, size_real, size_complex);
}
""",
)

# bind_rspt_callback() is a no-op unless something imported the package before cffi
# registered this module -- a sitecustomize.py or a .pth file, which run during
# Py_InitializeEx. In that case lib.py took its ImportError fallback and the @ffi.def_extern
# decorator was the stub's no-op, so cffi would answer every call with "no code was attached
# to it yet ... Returning 0" and RSPt would read 0 as a successful solve.
ffibuilder.embedding_init_code(r"""
    import impurityModel_interface
    impurityModel_interface.lib.bind_rspt_callback()
""")

ffibuilder.emit_c_code("run_impurityModel.c")
