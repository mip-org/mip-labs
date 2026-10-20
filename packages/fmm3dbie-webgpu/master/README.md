# fmm3dbie-webgpu

A limitation of running [fmm3dbie](https://github.com/fastalgorithms/fmm3dbie) in numbl is that its numerical core is Fortran with OpenMP, which does not translate readily to WebAssembly. This package takes a different route for a subset of it. fmm3dbie's own MATLAB code is shipped unmodified, and the Helmholtz impedance and Neumann solvers are reimplemented with the same signatures on WebGPU, so that a script written for fmm3dbie that stays within this subset runs after

```matlab
mip install --channel mip-org/labs fmm3dbie-webgpu
mip load fmm3dbie-webgpu
```

It runs in numbl in a browser with WebGPU (recent Chrome or Edge; on Linux, Chrome may need `chrome://flags/#enable-unsafe-webgpu`). In numbl on Node there is no WebGPU; the package loads, but the solvers report that WebGPU is not available. `example.m` solves plane wave scattering from a sound-hard sphere and compares the result with the exact solution.

## Coverage compared with fmm3dbie

The GPU covers the Helmholtz impedance and Neumann problems. fmm3dbie's pure-MATLAB code runs unmodified; anything that needs the fmm3dbie MEX fails with a message naming what is supported.

### Solvers and evaluators

| | fmm3dbie | fmm3dbie-webgpu |
|---|---|---|
| `helm3d.impedance.solver` / `eval` | yes | yes, same signatures |
| `helm3d.neumann.solver` / `eval` | yes | yes, same signatures |
| `helm3d.dirichlet`, `helm3d.transmission` | yes | no |
| `lap3d` (Dirichlet, Neumann) | yes | no |
| `stok3d` (velocity, traction) | yes | no |
| `em3d` (PEC with NRCCIE, dielectric with Müller) | yes | no |
| `belpde` (Laplace- and Helmholtz-Beltrami) | yes | no |
| `kernel3d`, `surfermat`, `surfermatapply`, `surferkerneval` | yes | no (they need the MEX for quadrature corrections) |
| `get_quadrature_correction` | yes | no (the solvers return a GPU handle instead) |

### Geometry and utilities

| | fmm3dbie | fmm3dbie-webgpu |
|---|---|---|
| `surfer` class (`area`, `volume`, `vals2coefs`, transforms, `merge`, ...) | yes | yes, unmodified |
| `surfer.oversample` (uses `ismembertol`, in numbl since 2026-10-20) | yes | yes |
| `+koorn`, `+polytens`, `helm3d.kern`, `helm3d.green` | yes | yes, pure MATLAB |
| `geometries.*` (sphere, ellipsoid, stellarator, torus, disk) | yes | no (they call the MEX); `quad_sphere` stands in for the sphere |
| Gmsh / `.go3` loaders | yes | pure MATLAB, not tested |
| `multiscale_mesher` | yes | no |
| `helm3d.fmm`, `lap3d.fmm`, `stok3d.fmm`, `em3d.fmm` | yes | yes, through the `fmm3d` package (a dependency; its numbl build is single-threaded WebAssembly) |

### Differences within the supported solvers

| | fmm3dbie | fmm3dbie-webgpu |
|---|---|---|
| Patch types | triangles (iptype 1), Gauss-Legendre quads (11), Chebyshev quads (12), mixed orders | Gauss-Legendre quads (11) of a single order |
| Wavenumber | real or complex | real |
| Impedance `zlams` | complex, varying per point | real constant |
| Precision | double; tolerances to about 1e-12 | single on the GPU; attainable accuracy about 1e-6, and GMRES tolerances below 1e-6 are raised to 1e-6 with a warning |
| Far field | FMM, O(N) | direct sum, O(N²); practical up to roughly 10⁵ points, with the near-field weights also having to fit in GPU memory |
| Evaluation targets | on or off the surface | off the surface only |
| Reusing the quadrature | `Q` holds the weights | `Q` holds the GPU state (`Q.gpu_handle`); pass it as `opts.quadrature_correction` (solver) or `opts.precomp_quadrature` (eval), and release it with `fmm3dbie_gpu('free', Q.gpu_handle)` |
| Platform | MATLAB on Linux, macOS, Windows; CPU threads (OpenMP) | numbl in a browser with WebGPU |

Also included: `quad_sphere(a, na, c0, norder)`, a sphere of quad patches in pure MATLAB, and `mie_sound_hard(zk, a, x)`, the exact field scattered by a sound-hard sphere, for checking results.

## Speed

We compared the two on `example.m` (sound-hard sphere, ka = 6, quad patches of order 6, eps = 1e-6, evaluation at 13,360 points) on a laptop with an i7-1355U (2 performance and 8 efficiency cores, 12 threads) and its integrated Intel Iris Xe GPU, timing the second run of each case.

| Points | fmm3dbie, MATLAB, 12 threads | fmm3dbie-webgpu, numbl in Chrome | Error vs exact (fmm3dbie / webgpu) |
|---|---|---|---|
| 2,646 | solve 7.2 to 9.2 s, eval 0.6 s | solve 3.1 s, eval 0.85 s | 9.1e-6 / 9.1e-6 |
| 10,584 | solve 37 s, eval 0.7 s | solve 12.4 s, eval 0.5 s | 1.2e-6 / 1.0e-6 |

Both take 9 GMRES iterations. The near-field quadrature costs about the same on both (about 2.6 s at 2,646 points); the difference is in GMRES, where fmm3dbie's FMM-based matrix-vector products take about 5 to 6 s in all and the dense products on the GPU about 0.4 s. At larger sizes the FMM's O(N) scaling will win. Note that on this machine MATLAB's `maxNumCompThreads` defaults to 2 and the fmm3dbie MEX follows it; with that default, fmm3dbie took 13.5 s and 59 s. A discrete GPU would widen the gap, and a CPU with more cores would narrow it.

## Layout

- `matlab/+helm3d/+impedance`, `matlab/+helm3d/+neumann`: the GPU versions of `solver.m` and `eval.m`, replacing upstream's.
- `matlab/webgpu/fmm3dbie_gpu.numbl.js`: the gateway, called by the solvers; its header documents the lower-level interface (dense point sums, layer potentials, solve, eval, handles).
- `matlab/webgpu/_fmm3dbie_gpu_backend.numbl.js`: the GPU code, bundled from `backend/*.ts` by `backend/build.mjs`. It runs in numbl's GPU helper worker (the `gpu` global of `.numbl.js` files). The near quadrature, far-field sum and GMRES in `backend/poc/` come from an earlier WebGPU proof of concept for fmm3dbie's impedance solver.
- `matlab/webgpu/fmm3dbie_routs.numbl.js`: a stand-in for the fmm3dbie MEX that reports what is not supported.

Upstream's Fortran (`src/`, the FMM3D submodule) and its example, test and documentation trees are not shipped.

## Test

`test_fmm3dbie_webgpu.m` checks the surfer class on a quad sphere and the messages for unsupported routines. When WebGPU is available, it also solves an exterior Neumann problem whose solution is the field of a point source inside the sphere.
