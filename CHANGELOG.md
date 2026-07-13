# Changelog

## 2026-07-13

- Added `matnwb@2.10.0`: NeurodataWithoutBorders/matnwb, the MATLAB interface
  for reading and writing Neurodata Without Borders (NWB) 2.x files. BSD-2-Clause.
  Pure MATLAB (`any`), fetched from upstream tag `v2.10.0`. The `+types` classes
  ship pre-generated for the active NWB schema (2.9.0), so `nwbExport` /
  `nwbRead` round-trip with no code-generation step. Only the root (namespaces
  auto-discovered) and `external_packages/fastsearch` are on the path;
  `tutorials/` is opt-in via `--with tutorials`. Trimmed `docs/` (8 MB rendered
  Sphinx), `logo/`, and `.github/`. `test_matnwb.m` builds an `NwbFile` and
  round-trips it through disk in base MATLAB. See the package README (notably
  the `nwbRead` `savedir` note for read-only installs).

- Added `abct@2025.9`: mikarubi/abct, Mika Rubinov's toolbox for unsupervised
  learning, network science, and imaging/network neuroscience (global
  residualization, Loyvain / co-Loyvain clustering, canonical & co-neighbor
  components, m-UMAP embeddings, degree/dispersion centralities, network
  shrinkage). MIT. Pure MATLAB (`any`), fetched from the upstream repo's
  `abct-matlab/abct/` subdirectory and pinned to release tag `2025.9`.
  `test_abct.m` exercises the toolbox-free core (`degree`, `dispersion`,
  `residualn`, `shrinkage`, `louvains`) in base MATLAB; a few optional code
  paths need add-on toolboxes and are documented in the package README.

- Added `mlxshake@0.3.0`: janklab/MlxShake, exports MATLAB Live Scripts (.mlx)
  to LaTeX/HTML/PDF/Word/Markdown, by Andrew Janke. Pure MATLAB (`any`), fetched
  from upstream tag `v0.3.0`. License is the MathWorks-specific BSD variant
  (`LicenseRef-MathWorks`). Overlays a one-line patch to the vendored
  `Log4jConfigurator` that fixes an upstream log4j crash which otherwise aborts
  every export entry point on recent MATLAB; with it, LaTeX/HTML/PDF/Word export
  work. Markdown export remains broken on recent MATLAB (upstream `_media` image
  drift) and is documented as such. See the package README.

- Added `janklab-core@0.2.2`: janklab/janklab-core, a general-purpose MATLAB
  utility library by Andrew Janke (extended type system, date/time classes,
  validators, misc utilities). Apache-2.0. Fetched from upstream tag `v0.2.2`.
  Native builds (linux/macOS/Windows) compile the single `binsearch_mex` MEX
  (self-contained C, `-static-libgcc` on Linux); an `[any]` fallback ships the
  pure-MATLAB layer only. `mip load` sets up the MATLAB path; the Java-backed
  features (MDBC/SQL, POI Excel, FTP, SLF4M logging) need `init_janklab` to add
  the bundled JARs to the classpath — see the package README.

## 2026-07-09

- Added `openflash@1.0.37`: the MATLAB component (`matlab/` subdirectory) of
  symbiotic-engineering/OpenFLASH — semi-analytical hydrodynamics via the
  matched eigenfunction expansion method (`run_MEEM`). MIT, pure MATLAB,
  `any` arch. Needs the Symbolic Math Toolbox at runtime, so no channel test
  script ships. Packaged as a dependency of MDOcean (see mip-org/mip#337).

## 2026-07-08

- Added `mip` (`project-environments-prototype`, branch-tracked): a prototype
  build of the mip package manager with project-level environments (the
  `mip env` subcommand group; `mipenv.yaml`/`mipenv.lock`). See
  [mip-org/mip#337](https://github.com/mip-org/mip/issues/337). Pure MATLAB
  (`any`); fetched from the `project-environments-prototype` branch of
  [mip-org/mip](https://github.com/mip-org/mip), which ships its own
  `mip.yaml`, so the channel entry is just `source.yaml`.

## 2026-07-07

- Added `meshio` (`main`, branch-tracked): a MATLAB port of the Python
  [meshio](https://github.com/nschloe/meshio) mesh I/O library (gmsh, netgen,
  obj, off, ply, stl, tetgen). Pure MATLAB (`any`); fetched from
  [danfortunato/meshio](https://github.com/danfortunato/meshio), which ships
  its own `mip.yaml`, so the channel entry is just `source.yaml`.

## 2026-06-30

- `fmm2d` (numbl_wasm): define `flong` (`int64_t`) in `fmm2d_c.h`. fort2c emits
  `flong`-typed temps for allocatable-array capacities (the `*_acap` vars), but
  the runtime header only defined `fint`/`fcomplex`, so the generated C failed
  to compile ("undeclared identifier 'flong'").

## 2026-06-25

- `build-package` caller: forward `publish` and `source_repo` to the reusable
  workflow. Test builds (`publish` unchecked) now publish the `.mhl` to a
  rolling `_test-builds` prerelease with a direct download URL, instead of
  leaving it only as a workflow artifact.

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
