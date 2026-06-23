# simplemex

A deliberately trivial, **single-threaded** C MEX:

```matlab
y = simplemex(x)   % returns x + 1, element-wise, for a real double array
```

It exists as a test fixture for `mip` itself — specifically to exercise the
package lifecycle (`load` / `clear` / `uninstall` / `update`) and **Windows MEX
file-locking** behavior with a binary that uses no threads and no external
runtime. Because nothing pins the DLL, a `clear simplemex` fully unloads it and
the `.mexw64` becomes deletable in the same MATLAB session — the clean baseline
to contrast against thread-pool MEX (OpenMP/TBB) that stay locked until the
process exits.

## Install

```matlab
mip install --channel mip-org/labs simplemex
mip load simplemex
y = simplemex([1 2 3]);   % -> [2 3 4]
```

## What is shipped

After `mip load`, the `matlab/` directory is on the path:

- `simplemex.<ext>` — the compiled MEX.

`matlab/simplemex.c` (the source) is overlaid alongside it but is irrelevant to
MATLAB (there is no `simplemex.m`).

## Architecture matrix

| Architecture | Compiler | Notes |
| --- | --- | --- |
| `linux_x86_64` | gcc | plain `mex` |
| `macos_arm64` | clang | plain `mex` |
| `windows_x86_64` | MinGW-w64 | statically linked via MATLAB's `mingw64.xml` |

The C is portable ANSI C with no third-party dependencies, so a single
`compile.m` (an unadorned `mex()` call) covers all three.

## Tests

[`test_simplemex.m`](test_simplemex.m) checks scalar/array results, shape
preservation, and the input-type error path, which invokes the shipped MEX and
satisfies the channel's MEX-coverage gate.
