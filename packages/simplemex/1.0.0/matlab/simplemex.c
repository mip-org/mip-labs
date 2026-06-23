/*
 * simplemex.c - a deliberately trivial, single-threaded MEX.
 *
 *   y = simplemex(x)   returns x + 1, element-wise, for a real double array,
 *                      preserving the input's size.
 *
 * It exists to exercise mip's package lifecycle (load / clear / uninstall /
 * update) and Windows MEX file-locking behavior with a binary that uses NO
 * threads and NO external runtime, so a `clear simplemex` fully unloads it.
 */

#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize n, i;
    double *in, *out;

    if (nrhs != 1) {
        mexErrMsgIdAndTxt("simplemex:nrhs", "simplemex requires exactly one input.");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("simplemex:nlhs", "simplemex returns at most one output.");
    }
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("simplemex:type", "Input must be a real double array.");
    }

    plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[0]),
                                   mxGetDimensions(prhs[0]),
                                   mxDOUBLE_CLASS, mxREAL);

    n   = mxGetNumberOfElements(prhs[0]);
    in  = mxGetPr(prhs[0]);
    out = mxGetPr(plhs[0]);
    for (i = 0; i < n; i++) {
        out[i] = in[i] + 1.0;
    }
}
