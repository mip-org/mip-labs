# threadedmex

A deliberately trivial **OpenMP-parallel** C MEX:

```matlab
[y, nthreads] = threadedmex(x)   % y = x + 1, computed across OpenMP threads
```

It is the threaded counterpart of [`simplemex`](../../simplemex/1.0.0), and a
**Windows-only test fixture** for `mip`'s MEX file-locking investigation. The
parallel region's real purpose is its side effect: libgomp spins up a persistent
worker-thread pool on the first call and keeps those threads alive (idle) for the
rest of the process. Because the channel's MinGW links `-static`, libgomp is
baked into the `.mexw64`, so those parked threads live in code *inside* the MEX
module and pin it — the expectation being that, unlike `simplemex`, the file
**stays locked even after `clear threadedmex`** (only a MATLAB restart releases
it). Confirming that is the whole point: it isolates the threading as the cause.

## Install

```matlab
mip install --channel mip-org/labs threadedmex
mip load threadedmex
[y, n] = threadedmex([1 2 3]);   % y -> [2 3 4]; n -> threads used
```

## What is shipped

After `mip load`, the `matlab/` directory is on the path:

- `threadedmex.<ext>` — the compiled MEX.

`matlab/threadedmex.c` (the gateway) and `matlab/tmex_worker.c` (the OpenMP
core) are overlaid alongside it but are irrelevant to MATLAB.

## Architecture matrix

| Architecture | Compiler | Notes |
| --- | --- | --- |
| `windows_x86_64` | MinGW-w64 | OpenMP via `gcc -fopenmp`, mex-linked `-lgomp`; statically linked (`mingw64.xml`). |

Windows only **by design**: the file-locking problem under investigation is a
Windows phenomenon — on Linux/macOS a loaded MEX deletes fine. There is no
`[any]` fallback, so install on other platforms is intentionally unsupported.

## Tests

[`test_threadedmex.m`](test_threadedmex.m) checks scalar/array results, shape
preservation, the OpenMP thread count, and the input-type error path, which
invokes the shipped MEX and satisfies the channel's MEX-coverage gate.
