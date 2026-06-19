# numblsavefig

Save a MATLAB figure to numbl's HDF5 figure format — the *"numbl figure HDF5
layout v1"* — so the figure can be opened in numbl's figure viewer, via
`importFigureHdf5`, or in the numbl IDE. It walks the live graphics-object tree
through the public `get`/`properties` API, which is far more robust than parsing
the undocumented `.fig` MAT-file serialization.

- **License:** Apache-2.0
- **Version:** 1.0.0
- **Source:** hosted in-repo in the [`mip-org/labs`](https://github.com/mip-org/mip-labs) channel.
- **Tested with:** MATLAB R2025b (HG2; should work on R2014b+). Base MATLAB only — uses built-in `h5create`/`h5write`/`h5writeatt` plus the low-level `H5*` interface. No toolbox required.

## Install and load

```matlab
mip install --channel mip-org/labs numblsavefig
mip load numblsavefig
```

## Usage

```matlab
plot(1:10, (1:10).^2);
numblsavefig('plot.h5')              % defaults to gcf

surf(peaks);
numblsavefig(gcf, 'surf.h5')         % explicit figure handle
```

Open the resulting `.h5` in the numbl figure viewer
(<https://concept-collection.github.io/numbl-figure-viewer/> — drag the file in)
or read it back with `importFigureHdf5` in the numbl runtime.

## What is shipped

A single function `numblsavefig.m` (plus `example.m` and the test). After
`mip load numblsavefig`, `numblsavefig` is on the path.

Supported graphics:

- **Axes**: `title`/`xlabel`/`ylabel`/`zlabel`, `xlim`/`ylim`/`zlim`, log scales,
  grid, box, `YDir`, colormap (`colormap_data`), `caxis`, 3-D `view`, `legend`.
  Multiple axes are numbered `/axes/1`, `/axes/2`, …
- **Traces**: `line`/`plot`/`plot3`, `surface` → `surf`/`mesh`, `image` →
  `imagesc`, `bar`/`barh`, `patch`, `scatter`.
- `NaN`/`Inf` round-trip natively. Unsupported objects are skipped with a
  warning (never an error).

Not yet covered: `errorbar`, `contour`, `quiver`, `area`, `bar3`,
`subplot`/`tiledlayout` grids, `faceVertexCData`, and chart objects
(`boxchart`/`piechart`/`heatmap`).

## Architecture matrix

Pure MATLAB — built for `any` and runs on every platform. No MEX, no native
code, nothing to statically link.

## Tests

`test_numblsavefig.m` (run by `mip test numblsavefig`) writes figures with
`numblsavefig` and reads them back with base-MATLAB HDF5 functions, asserting
the layout invariants numbl's reader relies on: datasets vs. scalar attributes,
grid orientation (an asymmetric `surf` must reconstruct correctly), NaN
preservation, and 0-based ragged patch faces.
