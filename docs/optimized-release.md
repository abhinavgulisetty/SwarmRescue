# Optimized rescue release 0.4.0

This release deploys an exact integer routing accelerator and includes the
focused 60 fps situation demos. The search, local contact and physical evacuation
workflow passes with eight robots in native and live Cortex-M4 modes. All 742
regression tests and runtime Ruff checks pass.

## Measured improvement

The frozen comparison uses 144 fresh models, 24 seeds and six graph families.
Every candidate field matches reference Jacobi exactly. Full routing solves,
including quantization and allocation, average 3.47x faster than compiled
Jacobi (95% CI [3.23, 3.73]) and 2.42x faster than compiled Gauss–Seidel
([2.27, 2.58]). Graph preprocessing is included. Compilation is excluded for
every method.

Generated room and dense graph groups average about 4–5% slower than
Gauss–Seidel. These measurements establish efficiency on the tested integer
solver problems; they do not establish superiority over all navigation systems,
improved rescue quality or calibrated hardware cycles. The
[research report](solver-research.md) preserves all measurements, failures,
source identities, selection rules and confidence intervals.

Production uses path bounds and ordered updates, exact conditional solving for
sticky self-loops, and ordinary Gauss–Seidel on dense non-sticky models. If
accelerated scratch cannot fit beside a robot's existing belief buffers, it
uses Gauss–Seidel within the legacy scratch budget.

## Physical and deployment validation

Both supervised installed seed-42 runs exit with status 0 and evacuate all
eight survivors. They exactly match the previous accepted mission's contact
times, evacuation times, final positions, distances, exposure proxies, robot
positions/arrival times and radio bytes. Last evacuation remains 81.396
simulated seconds. Maximum reported routing passes decrease from 789 to 390.
These are deterministic regression checks, not a fresh physical superiority
experiment.

The live-ARM run makes 25 actual routing requests for two robots. The compiled
service occupies 184,336 RAM bytes, including its 16 KiB stack, below its
200 KiB linker limit. It supports 2,048 states and 8,192 transitions; a
maximum-capacity model and sticky cyclic models are checked against native
fields. The installer exercises an adaptive self-loop solve over QEMU UART
before declaring readiness. Whole-robot firmware and physical-board timing
remain outside this kernel-service deployment.

All four focused installed cases accept their expected outcomes and exit 0.
AddressSanitizer and UBSan report no errors across 800 cold native solves with
identical fixed points. The independent rescue verifier checks coordinates,
ownership, event ordering, firmware work and archived runtime source hashes.

## Run and rebuild

```bash
outputs/deployment/optimized/run-mission
outputs/deployment/optimized/run-mission --firmware-robots 0
outputs/deployment/optimized/venv/bin/swarmrescue cases --out outputs/cases-040
```

Open `outputs/cases-040/index.html` after rendering to explore local contact,
blocked-route replanning, evacuation and stopping without a route. The existing
[smooth demo clips and speaking guide](demos.md) remain available; those bundled
clips were rendered before this release, while the four installed 0.4.0 cases
were separately rerun without rendering.

```bash
uv sync --locked --extra dev
uv build --wheel --out-dir outputs/releases/0.4.0
.venv/bin/python -m swarmrescue.deploy --wheel outputs/releases/0.4.0/swarmrescue-0.4.0-py3-none-any.whl --prefix outputs/deployment/optimized --tool-root outputs/tools/arm/root
```

The earlier validated release stays in `outputs/deployment/rescue/` and
`outputs/releases/0.3.1/`. The new wheel and portable evidence bundle are in
`outputs/releases/0.4.0/`. Python 3.12/Linux, uv and a C compiler are the tested
environment. Dependencies are installed from the hash-locked runtime file;
installation requires cached packages or network access. ARM mode additionally
requires GCC/binutils and QEMU, supplied through `--tool-root`.

The bundle contains the wheel, firmware, exact lock, report/source archives,
complete solver measurements, charts, previous smooth demo pack and checksums.
Its installer creates destination-specific paths. Extract it and use the
included `INSTALL.md`; the packaged deployment record documents the tested
machine rather than defining a relocatable virtual environment.

[Release receipt](releases/0.4.0.json) records hashes, source commit, supervised
exits and acceptance. Source changes are committed locally; nothing was pushed
and no physical board was flashed.
