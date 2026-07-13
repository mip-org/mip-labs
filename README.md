# mip-labs

The **`mip-org/labs`** MIP channel — a home for experimental and in-house
MATLAB packages whose source lives directly in this repository (rather than
being fetched from a separate upstream repo).

Install a package from this channel with:

```matlab
mip install --channel mip-org/labs <name>
mip load <name>
```

## Packages

| Package | Version | Description |
| --- | --- | --- |
| [janklab-core](packages/janklab-core/0.2.2) | 0.2.2 | General-purpose MATLAB utility library (types, date/time, validators, utils); Java-backed DB/Excel/FTP features need `init_janklab`. |
| [numblsavefig](packages/numblsavefig/1.0.0) | 1.0.0 | Save a MATLAB figure to numbl's HDF5 figure format. |
| [openflash](packages/openflash/1.0.37) | 1.0.37 | MATLAB component of OpenFLASH: semi-analytical hydrodynamics via MEEM (needs Symbolic Math Toolbox). |
| [simplemex](packages/simplemex/1.0.0) | 1.0.0 | Trivial single-threaded C MEX (`y = x + 1`); a test fixture for mip's lifecycle and Windows MEX locking. |
| [threadedmex](packages/threadedmex/1.0.0) | 1.0.0 | Trivial OpenMP-parallel C MEX (`y = x + 1`); Windows-only fixture for thread-pool MEX file-locking. |
| [treeweave](packages/treeweave/main) | main | Piecewise-polynomial function approximator (MATLAB MEX binding). |

## How this channel builds

A MIP package channel. Builds run one (package, architecture) at a time. They are triggered automatically on push to `main`, daily via a scheduled probe, or manually via a GitHub issue. The build engine (reusable workflows, the `mip-channel` CLI, the MATLAB build scripts, the Pages site template) lives in [`mip-org/mip_channel_tools`](https://github.com/mip-org/mip_channel_tools); the workflows here are thin callers.

Packages in this channel keep their source **in-repo**: the `.m` files sit
directly in `packages/<name>/<release>/` and `source.yaml` declares no remote
`source:` (the prepare step overlays the channel files as-is).

## Auto-build on push

Pushes to `main` run the `push-build.yml` workflow, which diffs the push and dispatches `build-package.yml` once per `(package, architecture)` pair affected by the change.

A file affects `packages/<name>/<version>` iff its path lies inside that directory. Each affected package expands to every arch declared in its `mip.yaml`, intersected with the channel's supported arches (`any`, `linux_x86_64`, `macos_arm64`, `windows_x86_64`). Recipe-only packages (no channel-side `mip.yaml`) expand to all four.

Changes outside `packages/` (workflows, README) do not trigger any builds. Deleted packages are skipped. The skip-if-unchanged logic still applies — pushes that don't change a package's source hash short-circuit at the prepare step.

## Scheduled rebuild

Daily at 06:00 UTC, `scheduled-build.yml` probes every (package, architecture) pair in the channel by running `mip-channel prepare` for each. A pair "needs rebuilding" iff its `.mhl` is missing on GitHub Releases or its source hash no longer matches. Pairs that need rebuilding are dispatched to `build-package.yml`.

The workflow can also be invoked manually:

```bash
gh workflow run scheduled-build.yml
```

## Submitting a build

Open an issue. The title must start with `Build` (case-insensitive). The body lists one or more build lines:

```
<name>@<release> <architecture>
```

Multiple architectures on one line dispatch multiple builds for that package. Multiple lines dispatch multiple packages. Lines without a package reference are ignored.

Example body:

```
numblsavefig@1.0.0 any
```

Within ~30s the request bot replies with the list of `(package, architecture)` pairs it parsed (or an error list). If an admin — anyone with write access on the repo — opened the issue, the builds dispatch automatically. Otherwise an admin replies `approve` on its own line to dispatch.

### Architecture keywords

- `any` — pure MATLAB; runs on ubuntu.
- `linux_x86_64`, `macos_arm64`, `windows_x86_64` — native; run on the matching OS.
- `all` — expand to every arch declared in the package's `mip.yaml` (intersected with the four above).

### Skip-if-unchanged and `force`

By default, a build that would produce a `.mhl` matching what is already published (same source hash, same metadata) short-circuits. To rebuild anyway, append `force` to a build line:

```
numblsavefig@1.0.0 any force
```

### Approval

Builds dispatch automatically when an admin — anyone with write access on the repo — opens the build issue. For an issue opened by anyone else, builds dispatch only when an admin replies with `approve` on its own line; emoji reactions and `approve` embedded in prose do not count.

## Direct dispatch

The same effect from the command line:

```bash
gh workflow run build-package.yml \
  -f package_path=packages/<name>/<version> \
  -f architecture=<arch> \
  -f force=false
```

Regenerate the channel index without rebuilding:

```bash
gh workflow run assemble-index.yml
```
