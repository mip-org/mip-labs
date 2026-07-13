# matnwb (MatNWB)

MatNWB is a MATLAB interface for reading and writing
[Neurodata Without Borders (NWB)](https://www.nwb.org/) 2.x files — the
standardized format for neurophysiology data. It autogenerates MATLAB classes
from the NWB format schema and provides `nwbRead` / `nwbExport` for round-tripping
NWB files, plus `generateExtension` for working with NWB extensions.

- **Upstream:** https://github.com/NeurodataWithoutBorders/matnwb
- **Documentation:** https://matnwb.readthedocs.io
- **Author:** Neurodata Without Borders
- **License:** BSD-2-Clause
- **Version:** 2.10.0 (upstream tag `v2.10.0`; ships the NWB schema 2.9.0 types)

## Install and load

```matlab
mip install --channel mip-org/labs matnwb
mip load matnwb
```

## What is shipped

`mip load matnwb` puts on the path:

- The package **root**, from which MATLAB discovers the `+types`, `+io`,
  `+file`, `+spec`, `+schemes`, `+util`, `+misc`, `+matnwb` and `+contrib`
  namespaces automatically, along with the top-level functions (`NwbFile`,
  `nwbRead`, `nwbExport`, `nwbInstallExtension`, `generateCore`,
  `generateExtension`, `inspectNwbFile`, `nwbtest`, ...).
- **`external_packages/fastsearch`** — a small helper used by
  `util.loadTimeSeriesData` / `util.loadTimeSeriesTimestamps`.

Also bundled (not on the default path, but used by the tool as needed):

- **`+types`** classes, **pre-generated** for the active NWB schema version
  (2.9.0), so a plain `nwbRead` / `nwbExport` round-trip works with no
  code-generation step.
- **`nwb-schema/`** — the packaged NWB format schemas, needed by
  `generateCore('<version>')` to generate classes for other schema versions.
- **`jar/schema.jar`** — the YAML schema reader used when parsing embedded
  specifications during `nwbRead` (loaded via `javaaddpath` on demand).
- **`+tests/`** — MatNWB's own test suite, driven by `nwbtest`.

The MatNWB **tutorials** (`.mlx` live scripts) are opt-in:

```matlab
mip load matnwb --with tutorials
```

## What is not shipped

Trimmed from the upstream tree because they are not needed at runtime:

- `docs/` — the pre-rendered Sphinx documentation (read it online at
  https://matnwb.readthedocs.io instead).
- `logo/` — image assets used only by the upstream README.
- `.github/` — upstream CI configuration.

## A note on `nwbRead` and the install directory

By default `nwbRead` regenerates the in-memory NWB type classes from the
schema embedded in the file, writing a cache into `misc.getMatnwbDir()` — the
MatNWB install directory. If that directory is **read-only** on your machine,
pass a writable `savedir`, or skip regeneration when the file matches the
shipped schema version:

```matlab
nwb = nwbRead('file.nwb', 'savedir', tempdir);   % regenerate into a writable dir
nwb = nwbRead('file.nwb', 'ignorecache');         % reuse the shipped +types classes
```

## Architecture and static linking

Pure MATLAB — no MEX or native code. The single `[any]` build runs on every
architecture (`linux_x86_64`, `macos_arm64`, `windows_x86_64`, ...). There are
no compiled binaries, so no static-linking concerns.

## Requirements

MATLAB R2019b or newer. The core read/write path uses only base MATLAB (HDF5 is
built in). A working Java runtime (bundled with MATLAB) is used for schema
parsing.

## Tests

`test_matnwb.m` runs after `mip load` via `mip test matnwb`. It builds an
`NwbFile` with subject metadata and a `TimeSeries`, writes it with `nwbExport`,
reads it back with `nwbRead` (both default and `'ignorecache'`), and asserts the
data and metadata round-trip. It also confirms the `+types` classes shipped
pre-generated and that `fastsearch` is on the path — all with base MATLAB only.
