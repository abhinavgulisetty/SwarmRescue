# Contributing

Changes should preserve the distinction between a proved exposure model and a
simulated rescue. Start with the [delivery report](RESEARCH_DELIVERY.md) and
[certificate model and proof](docs/research/lag-rescue/method.md).

## Development setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --frozen --extra dev --python 3.12
uv run --frozen --extra dev ruff check .
MUJOCO_GL=egl uv run --frozen --extra dev pytest -ra
```

Linux rendering uses EGL by default. On a minimal Ubuntu installation, install
the Mesa/EGL runtime libraries before importing MuJoCo:

```bash
sudo apt-get update
sudo apt-get install --yes --no-install-recommends \
  libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri
```

CI sets `LIBGL_ALWAYS_SOFTWARE=1` to use Mesa without requiring a GPU. These
system libraries are needed in addition to the Python dependencies.
A C compiler is needed for native firmware
conformance. The full default suite runs headlessly; ARM tests report skips when
`arm-none-eabi-gcc` and `qemu-system-arm` are unavailable. To include the optional
ARM tests, put those tools on `PATH` or supply their installation root:

```bash
EMBER_ARM_TOOL_ROOT=/path/to/arm/root MUJOCO_GL=egl \
  uv run --frozen --extra dev pytest -ra
```

An installation root must contain the executables in `usr/bin/` and, for the
retained Linux toolchain, its shared libraries in `usr/lib/x86_64-linux-gnu/`.
System-installed tools on `PATH` do not need a root override. See also the
[firmware notes](firmware/README.md).
CI checks the preserved release checksums and all five archived mission reports,
runs the native suite on Ubuntu 24.04, and reproduces the certificate study. It
uploads the mission checks, test XML and fresh certificate results. Checking the
archived mission evidence validates its source hashes, egress coordinates and
custody consistency; it does not rerun those physical simulations. Inspect
reported skips rather than assuming a hosted CI run includes ARM execution. The
historical release and post-cleanup test counts refer to their respective
recorded source trees.

## Reproducing the main result

Generate a fresh result directory so the preserved evidence stays intact:

```bash
uv run --frozen --extra dev swarmrescue certificate-evaluation \
  --output-dir outputs/certificate-evaluation
```

The default protocol contains 96 synthetic missions and 288 route candidates.
Its independent NetworkX graph solver checks both actor bounds, including 288
exact integer comparisons. The selected 80 complete missions have 5,352 exhaustive
scenario/timing replays. CI checks these counts, zero integer mismatches, the
frozen method decisions and the generated protocol's source hashes. Execution
times and protocol digests can vary with the machine and dependency metadata.

For certificate changes, add a regression that exercises the relevant assumption
or counterexample. Keep the external graph oracle independent of the optimized
recurrence. Document changes to exposure units, timing intervals, horizon coverage,
scenario persistence or resource accounting alongside the code.

## Evidence and scope

- Keep published receipts, source archives and frozen reports unchanged. Record a
  new version and output directory when an implementation or protocol changes.
- Report the tested model, input data, source revision, dependency versions and
  commands. Distinguish a new experiment from a reproduction of an existing one.
- Preserve the stated limits: finite scenarios and bounded timing do not establish
  calibrated human survival; simulation and QEMU do not establish physical robot
  performance. Historical priority and superiority over other research require
  separate evidence.
- Keep virtual environments, toolchain installations, caches and large generated
  videos out of Git. Distribute packaged binaries and large reproducibility bundles
  as GitHub release assets with checksums. Small reports and the code needed to
  reproduce them belong in the repository.
- Respect upstream source and dataset licenses. Do not vendor the original-author
  reference checkout without verifying its redistribution terms.

## Pull requests

Describe the concrete behavior changed, why it is needed and which checks passed.
Include any changed proof assumptions, failures, skips or simulation tradeoffs.
Use the frozen dependency lock unless intentionally updating dependencies; include
the corresponding lockfile changes when doing so.
