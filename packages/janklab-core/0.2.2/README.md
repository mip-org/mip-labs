# janklab-core (mip-org/labs)

[Janklab-core](https://github.com/janklab/janklab-core) is a general-purpose
utility library for MATLAB by [Andrew Janke](https://apjanke.net) — the core
library of the [Janklab](https://janklab.net) suite. It provides an extended
type system, additional date/time classes, more validators, a database toolbox
("MDBC"), advanced Excel I/O and FTP, and a large collection of miscellaneous
utilities.

- **Upstream:** https://github.com/janklab/janklab-core
- **Version:** 0.2.2 (upstream tag `v0.2.2`)
- **License:** Apache-2.0 (bundled third-party libraries carry their own FOSS
  licenses — see below).

## Install and load

```matlab
mip install --channel mip-org/labs janklab-core
mip load janklab-core
```

## What `mip load` sets up

`mip load janklab-core` puts janklab's MATLAB code on the path, mirroring the
directory layout that janklab's own `init_janklab` establishes:

- `Mcode/classes`, `Mcode/commands`, `Mcode/monkeypatch`, `Mcode/toplevel`,
  `Mcode/validators` — the janklab-core library itself (the `jl.*` namespace,
  toplevel utilities, validators, and the `@cell` monkeypatch).
- The three redistributed MATLAB helper libraries under `lib/matlab`
  (`dispstr`, `SLF4M`, `matlab-jarext-inspector`).

After loading, the **pure-MATLAB layer works directly**: the extended type
system, date/time classes, validators, the toplevel utility functions, and the
`binsearch` MEX.

## Java-backed features need `init_janklab`

mip only manipulates the MATLAB path — it does **not** run `javaaddpath`. Several
janklab feature areas are backed by bundled Java libraries and therefore require
the Java classpath to be set up before use:

- the MDBC database toolbox (`jl.sql.*`),
- Excel I/O via Apache POI (`jl.office.excel.*`),
- the FTP client (`jl.net.ftp.*`),
- SwingExplorer, and SLF4M logging (`logger.*`).

To enable these, run janklab's initializer after loading:

```matlab
mip load janklab-core
init_janklab            % adds the bundled JARs to the Java classpath
```

> **Note:** upstream's `init_janklab` currently raises a log4j error
> (`org.apache.log4j.helpers.NullEnumeration`) during its SLF4M logging setup on
> recent MATLAB releases — the bundled SLF4M logging shim only ships JARs for
> R2020a. The pure-MATLAB layer above is unaffected; this only limits the
> logging/Java-backed features on newer MATLAB. This is an upstream issue —
> see the [janklab repo](https://github.com/janklab/janklab-core).

## MEX / architecture matrix

janklab-core ships one native source, `binsearch_mex.c` (a self-contained binary
search used by `jl.algo.binsearch` for `double`/`single` arrays; other types use
the pure-MATLAB `jl.algo.binsearch_mcode` fallback).

| Architecture | MEX built? |
| --- | --- |
| `linux_x86_64`, `macos_arm64`, `windows_x86_64` | yes — `binsearch_mex` |
| `any` (fallback) | no — pure MATLAB only |

The MEX is compiled from source on the channel's own CI runners (upstream's
pre-built binary is stripped before build). It has no third-party dependencies;
on Linux it is linked with `-static-libgcc` so it depends only on OS-provided
libraries. On the `any` fallback build, `jl.algo.binsearch` on `double`/`single`
is unavailable — use `jl.algo.binsearch_mcode` instead.

## Bundled third-party libraries

janklab-core is redistributed with these libraries (all FOSS, redistribution
permitted; kept under `lib/`):

- Apache Commons CSV, Apache Commons Primitives, Apache POI, FastUtil — Apache-2.0
- SwingExplorer — LGPL-3.0
- dispstr — BSD-2-Clause
- SLF4M — Apache-2.0 / BSD

## What is not shipped

Trimmed from the bundle (not needed on an end-user machine): `src/` (the Maven
source for the pre-built `janklab-java` JAR, which ships under `lib/java`),
`M-doc/` (generated docs), and the `dev-kit/`, `scratch/`, and `.github/`
developer scaffolding. Everything under `lib/` (the JARs), `Mcode/`, `examples/`,
`tests/`, and `doc/` is kept.

`examples/` and `tests/` are shipped but off the path by default; add them with:

```matlab
mip load janklab-core --with examples
mip load janklab-core --with tests
```

## Tests

- `test_janklab_core_mex.m` — used by the native builds; exercises the
  `binsearch` MEX (both `double` and `single` paths, satisfying the MEX-coverage
  gate) plus a sampling of the pure-MATLAB utility layer.
- `test_janklab_core.m` — used by the `any` fallback; exercises the
  `binsearch_mcode` fallback and the same utility functions, with no MEX.

Neither test calls `init_janklab`; both cover only the parts that work without
the Java classpath.
