# mlxshake (mip-org/labs)

[MlxShake](https://github.com/janklab/MlxShake) exports MATLAB Live Scripts
(`.mlx` files) to LaTeX, HTML, PDF, Microsoft Word, and Markdown. By
[Andrew Janke](https://apjanke.net); part of the [Janklab](https://janklab.net)
suite. It began as a productization of Michio Inoue's
[livescript2markdown](https://github.com/minoue-xx/livescript2markdown).

- **Upstream:** https://github.com/janklab/MlxShake
- **Version:** 0.3.0 (upstream tag `v0.3.0`)
- **License:** see the license note below.

> Upstream describes MlxShake as **pre-release beta software** ("probably
> buggy"; not for production use). It is packaged here for the labs channel on
> that understanding.

## Install and load

```matlab
mip install --channel mip-org/labs mlxshake
mip load mlxshake
```

Pure MATLAB — `mip load mlxshake` puts `Mcode/` on the path; everything lives
under the `janklab.mlxshake` namespace.

## Usage

```matlab
% Export a Live Script to HTML (or 'pdf', 'latex', 'msword'):
opts = janklab.mlxshake.MlxExportOptions;
opts.format = 'html';
opts.outFile = 'MyLiveScript.html';
janklab.mlxshake.exportlivescript('MyLiveScript.mlx', opts);

% Just the .mlx -> .tex step:
janklab.mlxshake.mlx2latex('MyLiveScript.mlx', 'MyLiveScript.tex');
```

## Compatibility on recent MATLAB — two changes to be aware of

MlxShake targets ~R2019b. On recent MATLAB it has two issues; this package
addresses the first and documents the second.

1. **log4j initialization crash (patched here).** MlxShake's library
   initializer configures log4j and calls `getAllAppenders().nextElement()`
   without guarding against an empty enumeration. On recent MATLAB (whose
   log4j-1.x API sits over a newer backend) that enumeration comes back empty,
   the call throws `NoSuchElementException`, and because the initializer runs
   at class-load of `MlxshakeBase`, **every** export entry point aborts. This
   package overlays a patched
   `Mcode/+janklab/+mlxshake/+internal/+logger/Log4jConfigurator.m` that guards
   the call with `hasMoreElements` (the only change from upstream v0.3.0;
   marked with a "channel patch" comment). Logging degrades to log4j's default
   console pattern instead of crashing.

2. **Markdown image handling (not fixed).** MlxShake's LaTeX→Markdown step
   expects the Live Editor to emit a `<stem>_images/` figure folder, but recent
   MATLAB emits `<stem>_media/` with different figure filenames. As a result
   **Markdown export (`format='markdown'`, the default) currently fails on
   recent MATLAB**. The direct-export formats below are unaffected. Fixing
   Markdown export requires upstream changes; see the
   [MlxShake repo](https://github.com/janklab/MlxShake).

With the log4j patch, these formats work (verified on R2025b, headless):
`latex`, `html`, `pdf`, and `msword` (the "direct" export formats), plus
`mlx2latex`.

## License

MlxShake is licensed under a **MathWorks-specific variant of the BSD license**
that adds this restriction:

> In all cases, the software is, and all modifications and derivatives of the
> software shall be, licensed to you solely for use in conjunction with
> MathWorks products and service offerings.

Redistribution is permitted (MlxShake carries derivatives of MathWorks-authored
`livescript2markdown` code, which uses the same terms), so this channel may
carry it, but your use is limited to use with MathWorks products. Because it
does not map to a standard SPDX identifier, the manifest uses
`license: "LicenseRef-MathWorks"`.

## What is shipped / not shipped

On the path: `Mcode/` (the `janklab.mlxshake` namespace). Also kept: `bin/`
(the `mlxshake` shell wrapper) and the upstream project files at the root.

Trimmed from the bundle (not needed to run the library): the documentation
trees (`doc/`, `docs/`, `docs-src/`, `doc-project/`), developer scaffolding
(`dev-kit/`, `.github/`), and the ~5 MB `examples/` tree (mostly stale rendered
output). A small `test_fixture.mlx` is bundled for the channel test; upstream's
example Live Scripts remain available in the
[MlxShake repo](https://github.com/janklab/MlxShake).

## Tests

`test_mlxshake.m` verifies the public API is on the path and exports the bundled
`test_fixture.mlx` to LaTeX and HTML (and via `mlx2latex`), asserting non-empty
output. It intentionally skips Markdown export (issue 2 above).
