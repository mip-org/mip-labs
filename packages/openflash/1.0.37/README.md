# openflash

The MATLAB component of [OpenFLASH](https://github.com/symbiotic-engineering/OpenFLASH), a semi-analytical hydrodynamics modeling project from the Symbiotic Engineering Analysis (SEA) Lab at Cornell. The MATLAB code implements the matched eigenfunction expansion method (MEEM) via `run_MEEM`.

- **Authors**: Symbiotic Engineering Analysis (SEA) Lab, Cornell University
- **License**: MIT
- **Version**: 1.0.37
- **Repository**: https://github.com/symbiotic-engineering/OpenFLASH

## Install

```matlab
mip install --channel mip-org/labs openflash
mip load openflash
```

## Requirements

`run_MEEM` generates MATLAB functions from symbolic expressions at runtime,
so it **requires the Symbolic Math Toolbox**.

## What is shipped

Only the `matlab/` subdirectory of the upstream repository (the bulk of
OpenFLASH is a Python/Jupyter project). `mip load openflash` puts `src/`
(`run_MEEM`) on the path; `mip load openflash --with tests` adds `test/`
(`test_MEEM`, `convergence_study`).

## Status

The MATLAB port is early-stage and hosted here on the experimental labs
channel. Upstream is actively developed with frequent releases; no channel
test script ships (the CI runners have no Symbolic Math Toolbox).
