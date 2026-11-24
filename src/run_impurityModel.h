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
 * The symbol RSPt links against is `run_impmod_cluster`, defined in the C body that
 * library_builder.py passes to set_source(). It checks that the embedded interpreter
 * actually came up and then forwards here. Adding `run_impmod_cluster` to this file would make
 * cffi generate a second callback for it instead.
 */
/* One named field of RSPt's cluster object (or of the call context). RSPt builds an array of
 * these and passes it, so that a new quantity never changes the argument list again.
 *   name      NUL-terminated Fortran name: "ham", "observables%ms", "symm(3)%rep", "ctx%u4"
 *   data      pointer into RSPt's own memory (column-major); NULL if not allocated
 *   kind      1 complex, 2 real, 3 integer, 4 logical, 5 character
 *   rank      0 for a scalar
 *   itemsize  bytes per element (for character: the length of the string)
 *   shape     extents, the first `rank` entries are meaningful
 * The layout is 128 bytes; the Fortran side checks that with c_sizeof. */
typedef struct {
    char name[48];
    void *data;
    int kind;
    int rank;
    long long itemsize;
    long long shape[7];
} impmod_field;

extern int run_impmod_cluster_py(const impmod_field *fields, size_t n_fields, int abi_version);
#endif // IMPMOD_INTERFACE_H
