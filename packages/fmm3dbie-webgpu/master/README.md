# fmm3dbie-webgpu

A limitation of running [fmm3dbie](https://github.com/fastalgorithms/fmm3dbie) in numbl is that its numerical core is Fortran with OpenMP, which does not translate readily to WebAssembly. This package takes a different route for a subset of it. fmm3dbie's own MATLAB code is shipped unmodified, and the Helmholtz impedance and Neumann solvers are reimplemented with the same signatures on WebGPU, so that a script written for fmm3dbie runs after

```matlab
mip install --channel mip-org/labs fmm3dbie-webgpu
mip load fmm3dbie-webgpu
```

It runs in numbl in a browser with WebGPU (recent Chrome or Edge; on Linux, Chrome may need `chrome://flags/#enable-unsafe-webgpu`). In numbl on Node there is no WebGPU; the package loads, but the solvers report that WebGPU is not available.

## What is supported

- `helm3d.impedance.solver`, `helm3d.impedance.eval`, `helm3d.neumann.solver` and `helm3d.neumann.eval`, with fmm3dbie's signatures and its representation u = S_k[σ] + iα D_k S_{i|k|}[σ].
- fmm3dbie's `surfer` class, `+koorn`, `+polytens`, `helm3d.kern`, `helm3d.green` and the other pure-MATLAB code, unmodified.
- `quad_sphere(a, na, c0, norder)`, a sphere of quad patches in pure MATLAB, standing in for `geometries.sphere`, which needs the MEX.
- `mie_sound_hard(zk, a, x)`, the exact field scattered by a sound-hard sphere, for checking results.

Everything else that calls the fmm3dbie MEX (the Laplace, Stokes and Maxwell solvers, `helm3d.dirichlet`, `helm3d.transmission`, the `geometries` constructors) fails with a message saying so.

## Differences from fmm3dbie

The GPU version needs quad patches (`iptype` 11) of a single order, a real wavenumber and a real constant impedance, and it evaluates off the surface only. All GPU arithmetic is single precision, so the attainable accuracy is about 1e-6, and GMRES tolerances below 1e-6 are raised to 1e-6 with a warning. The far field is a direct O(N²) sum rather than an FMM, which limits problems to roughly 10⁵ points; the near-field weights also have to fit in GPU memory. The returned `Q` holds the GPU state rather than quadrature weights: passing it back as `opts.quadrature_correction` (solver) or `opts.precomp_quadrature` (eval) skips the near-quadrature setup, and `fmm3dbie_gpu('free', Q.gpu_handle)` releases it.

On a laptop's integrated GPU, the plane wave example on a sphere of 2646 points (54 patches of order 6) takes about 8 s to solve, nearly all of it near-quadrature setup, and agrees with the exact solution to 9e-6.

## Layout

- `matlab/+helm3d/+impedance`, `matlab/+helm3d/+neumann`: the GPU versions of `solver.m` and `eval.m`, replacing upstream's.
- `matlab/webgpu/fmm3dbie_gpu.numbl.js`: the gateway, called by the solvers; its header documents the lower-level interface (dense point sums, layer potentials, solve, eval, handles).
- `matlab/webgpu/_fmm3dbie_gpu_backend.numbl.js`: the GPU code, bundled from `backend/*.ts` by `backend/build.mjs`. It runs in numbl's GPU helper worker (the `gpu` global of `.numbl.js` files). The near quadrature, far-field sum and GMRES in `backend/poc/` come from an earlier WebGPU proof of concept for fmm3dbie's impedance solver.
- `matlab/webgpu/fmm3dbie_routs.numbl.js`: a stand-in for the fmm3dbie MEX that reports what is not supported.

Upstream's Fortran (`src/`, the FMM3D submodule) and its example, test and documentation trees are not shipped.

## Test

`test_fmm3dbie_webgpu.m` checks the surfer class on a quad sphere and the messages for unsupported routines. When WebGPU is available, it also solves an exterior Neumann problem whose solution is the field of a point source inside the sphere.
