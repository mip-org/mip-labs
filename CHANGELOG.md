# Changelog

## 2026-06-23

- Added `threadedmex` (`1.0.0`, in-repo, windows_x86_64 only): the threaded
  counterpart of `simplemex` — a trivial OpenMP-parallel C MEX (`y = x + 1`
  across threads). A test fixture to confirm that a thread-pool MEX stays
  file-locked even after `clear` (libgomp baked in via MinGW `-static`).

- Added `simplemex` (`1.0.0`, in-repo): a trivial single-threaded C MEX
  (`y = x + 1`) used as a test fixture for mip's package lifecycle and Windows
  MEX file-locking behavior. Plain `mex()` across linux/macos/windows.

- Added `treeweave` (`main`, branch-tracked): MATLAB MEX binding for the
  [treeweave](https://github.com/DiamonDinoia/treeweave) piecewise-polynomial
  approximator. `compile.m` drives treeweave's own CMake build (mwrap gateway +
  the treeweave_c_static C ABI) and stages the generated `tw_*.m` + the MEX into
  `bindings/matlab`. C++20 forces a gcc-toolset ≥ 11 on Linux, clang on macOS,
  and MSVC on Windows (provisioned via per-OS `setup:`; bison/flex for mwrap).
  `numbl_wasm` deferred (the MATLAB API's function-handle callback trampoline
  has no WASM→runtime path under numbl's stateless builtin model).
  `windows_x86_64` dropped: the mwrap generator does not link under MSVC
  (LNK2001 on its flex/bison globals) and the channel's MinGW 8.1 predates
  C++20. linux_x86_64 + macos_arm64 ship.

## 2026-06-22

- Build-request issues opened by an admin (write+ on the repo) now dispatch
  automatically — no `approve` comment needed. README updated.

- Added `fmm2d` (`main`): MATLAB MEX (linux/macos/windows, built from the
  upstream Fortran) plus a `numbl_wasm` build. The WASM build transpiles the
  Fortran to C with fort2c at build time (`matlab/numbl/build_wasm.sh`), so its
  `source.yaml` points at upstream `flatironinstitute/fmm2d` rather than the
  `fmm2d_c_translation` fork.
- Added `numbl_wasm` to the channel's manually dispatchable architectures.

## 2026-06-19

- Created the `mip-org/labs` channel.
- Added `numblsavefig` 1.0.0 (in-repo source): save a MATLAB figure to numbl's
  HDF5 figure format.
