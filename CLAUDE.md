# CLAUDE.md

Guidance for working in this repository.

## What this is

The `mip-org/labs` MIP package channel — experimental / in-house MATLAB
packages. Packages live under `packages/<name>/<release>/` with a `source.yaml`
and (usually) a `mip.yaml` declaring supported architectures. Builds run one
`(package, architecture)` pair at a time via GitHub Actions.

Unlike most channels, packages here keep their **source in-repo**: the `.m`
files sit directly in the release directory and `source.yaml` has no remote
`source:` key. The prepare step's `fetch_source` is a no-op in that case, and
`overlay_channel_files` copies the release directory (everything except
`source.yaml`) into the build tree.

## Layout

This repo holds only channel-specific content:

- `packages/<name>/<release>/` — package definitions (and, here, the source).
- `.github/workflows/` — thin **caller** workflows. Each owns its event
  triggers (push, schedule, issues, dispatch) and concurrency, then delegates
  all logic to a reusable workflow in `mip-org/mip_channel_tools` via
  `uses: mip-org/mip_channel_tools/.github/workflows/<name>.yml@<ref>` with
  `secrets: inherit`. (`claude.yml` is the exception — a self-contained PR
  assistant.)

The build engine lives in `mip-org/mip_channel_tools`: the reusable workflows,
the `mip-channel` CLI, the MATLAB build scripts (`bundle_one.m`, `test_one.m`),
the Pages site template, and developer notes. To run against a different tooling
branch, edit the `@<ref>` on a caller's `uses:` line.

See `mip-org/mip_channel_tools/adding_a_package.md` for the full package recipe.

## Conventions

- Record every notable change in `CHANGELOG.md`. Keep entries brief.
- Supported channel architectures: `any`, `linux_x86_64`, `macos_arm64`,
  `windows_x86_64`, `numbl_wasm` (a portable `.wasm` built with emcc; for
  packages whose C is transpiled from Fortran at build time, see fmm2d).
- Build requests are submitted via issues (title starts with `build`); each
  body line is `<name>@<release> <architecture>`. See `README.md` for details.
- Channel → repo mapping: channel `mip-org/labs` is served from this repo
  (`mip-org/mip-labs`) at `https://mip-org.github.io/mip-labs/index.json`.
