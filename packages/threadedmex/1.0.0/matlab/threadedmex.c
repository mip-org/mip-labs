/*
 * threadedmex.c - MEX gateway for the OpenMP test fixture.
 *
 *   [y, nthreads] = threadedmex(x)
 *
 * y = x + 1 (element-wise, real double), computed in parallel by
 * add_one_parallel (tmex_worker.c). nthreads (optional) reports how many
 * OpenMP threads the parallel region used.
 *
 * The gateway itself has no OpenMP pragmas, so it is compiled by `mex`; the
 * parallel work lives in tmex_worker.c (compiled with -fopenmp). See compile.m.
 */

#include <stddef.h>
#include "mex.h"

int add_one_parallel(const double *in, double *out, ptrdiff_t n);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize n;
    double *in, *out;
    int used;

    if (nrhs != 1) {
        mexErrMsgIdAndTxt("threadedmex:nrhs", "threadedmex requires exactly one input.");
    }
    if (nlhs > 2) {
        mexErrMsgIdAndTxt("threadedmex:nlhs", "threadedmex returns at most two outputs.");
    }
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("threadedmex:type", "Input must be a real double array.");
    }

    plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[0]),
                                   mxGetDimensions(prhs[0]),
                                   mxDOUBLE_CLASS, mxREAL);

    n   = mxGetNumberOfElements(prhs[0]);
    in  = mxGetPr(prhs[0]);
    out = mxGetPr(plhs[0]);

    used = add_one_parallel(in, out, (ptrdiff_t)n);

    if (nlhs >= 2) {
        plhs[1] = mxCreateDoubleScalar((double)used);
    }
}
