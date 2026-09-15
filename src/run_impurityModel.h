#ifndef IMPMOD_INTERFACE_H
#define IMPMOD_INTERFACE_H
#include <complex.h>
#include <stdbool.h>
#include <stdlib.h>
/*
 * This header is fed to cffi's embedding_api() by library_builder.py, which strips every
 * line starting with '#' and treats what is left as the set of extern "Python" callbacks.
 * So it declares ONLY the callback, and deliberately not the public symbol.
 *
 * The symbol RSPt links against is `run_impmod_ed`, defined in the C body that
 * library_builder.py passes to set_source(). It checks that the embedded interpreter
 * actually came up and then forwards here. Adding `run_impmod_ed` to this file would make
 * cffi generate a second callback for it instead.
 */
extern int run_impmod_ed_py(char label[], char solver_param[], char dc_param[],
                         int dc_flag, double _Complex *U_mat,
                         double _Complex *hyb, double _Complex *h_dft,
                         double _Complex *sig, double _Complex *sig_real,
                         double _Complex *sig_static, double _Complex *sig_dc,
                         double *iw, double *w,
                         double _Complex *corr_to_spherical,
                         double _Complex *corr_to_cf, size_t n_orb,
                         size_t n_rot_cols, size_t n_orb_full, size_t n_iw,
                         size_t n_w, double eim, double tau, int verbosity,
                         size_t size_real, size_t size_complex);
#endif // IMPMOD_INTERFACE_H
