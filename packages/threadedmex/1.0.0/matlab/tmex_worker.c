/*
 * tmex_worker.c - the OpenMP-parallel core of threadedmex.
 *
 * Kept separate from the MEX gateway (threadedmex.c) so it can be compiled
 * directly with `gcc -fopenmp` for full flag control, then mex-linked with
 * -lgomp (the channel's MinGW links -static, baking libgomp into the .mexw64).
 * It uses only plain C + OpenMP, so it needs no MATLAB headers.
 *
 * The point of the parallel region is its SIDE EFFECT: libgomp spins up a
 * persistent worker-thread pool on the first parallel region and keeps those
 * threads alive (idle-waiting) for the rest of the process. Because libgomp is
 * statically linked into the .mexw64, those parked threads live in code inside
 * the MEX module and pin it - so the file stays locked even after `clear`.
 */

#include <stddef.h>
#include <omp.h>

/* y = x + 1, element-wise, in parallel. Returns the number of threads used. */
int add_one_parallel(const double *in, double *out, ptrdiff_t n)
{
    int used = 1;
    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        used = omp_get_num_threads();

        ptrdiff_t i;
        #pragma omp for
        for (i = 0; i < n; i++) {
            out[i] = in[i] + 1.0;
        }
    }
    return used;
}
