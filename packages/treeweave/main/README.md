# treeweave

A piecewise-polynomial function approximator: it builds an adaptive tree whose
leaves are polynomial fits over a box domain, refining until a requested
tolerance is met, then evaluates the approximant fast (single point or batched).
This package ships the **MATLAB MEX binding** from
[DiamonDinoia/treeweave](https://github.com/DiamonDinoia/treeweave) (author:
Marco Barbone; BSD-3-Clause).

Branch-tracked at `main` (upstream has no release tags yet).

## Install

```matlab
mip install --channel mip-org/labs treeweave
mip load treeweave
```

```matlab
f   = @(x) exp(0.5*x(1)) + sin(3*x(1));
obj = treeweave(f, 0, 1, 1e-8);     % fit on [0,1] to tol 1e-8
Y   = obj(linspace(0, 1, 1000)');   % evaluate (N x out_dim)
delete(obj);                        % free C-side memory
```

See [`example.m`](example.m) for a 2-D → 3-D vector fit, and the upstream
[`bindings/matlab/README.md`](https://github.com/DiamonDinoia/treeweave/blob/main/bindings/matlab/README.md)
for the full API (`'sorted'`, `'transposed'`, `memory_usage`, `print_stats`, …).

## What is shipped

After `mip load`, the `bindings/matlab` directory is on the path:

- `treeweave.m` — the user-facing handle class.
- `tw_*.m` — the mwrap stubs (checked in upstream, copied here at build time).
- `treeweave_mex.<ext>` — the compiled MEX gateway.

`mip load treeweave --with examples` additionally adds the upstream MATLAB
examples (`bindings/matlab/examples`).

## How it is built

The binding is compiled entirely by treeweave's own CMake build (there is no
Makefile): the mwrap gateway (`treeweave_mex_gen.cpp` + `tw_*.m`) is checked in
upstream under `bindings/matlab/generated/` — mwrap's output is platform-
independent, so it is generated once and committed, and no platform runs mwrap
(or needs bison/flex). CMake fetches only the header-only deps (polyfit, POET,
xsimd), compiles the checked-in gateway, builds the C ABI static archive, and
links the MEX. `compile.m` drives that build and stages the outputs into
`bindings/matlab`.

### Architecture matrix

| Architecture | Compiler | Notes |
| --- | --- | --- |
| `linux_x86_64` | gcc-toolset ≥ 11 | C++20 needs gcc ≥ 11; the channel's default gcc-toolset-10 is too old, so a newer toolset is installed at build time. |
| `macos_arm64` | Apple clang | `apple-m1` baseline (Apple Silicon). |
| `windows_x86_64` | MSVC (VS 2022) | C++20; SSE2 baseline, dispatch ladder up to AVX-512 via `/arch:` flags. |

**Static linking.** The MEX statically links the treeweave C ABI and (on
Linux/GNU) `-static-libstdc++ -static-libgcc`, so each binary is self-contained:
the only dynamic dependencies are OS-provided libraries and MATLAB's own
`libmex`/`libmx`. The C ABI does **runtime ISA dispatch** (`TREEWEAVE_C_MULTIARCH`),
so a single portable baseline binary picks the best AVX/AVX-512 (or NEON) kernel
on the end-user CPU — on MSVC the ladder is SSE2/AVX/AVX2/AVX512 (`/arch:` flags).

### Not shipped

- **`numbl_wasm`.** Deferred. treeweave's API passes a MATLAB `function_handle`
  into the C `fit` as a callback that is invoked many times. numbl's WASM
  builtin model is a stateless one-way dispatch with no WASM→runtime callback
  path, so the trampoline cannot be reproduced without a new bridge. The other
  language bindings (Python/Julia/JS/Fortran) are also not part of this MATLAB
  package.

## Tests

[`test_treeweave.m`](test_treeweave.m) exercises a 1-D scalar fit and a 2-D → 3-D
vector fit (accuracy, output shapes, the `transposed` layout, out-of-domain
NaN), which loads the shipped MEX and satisfies the channel's MEX-coverage gate.
