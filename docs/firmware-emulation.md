# Freestanding Cortex-M4 kernel replay — 2026-10-04

`eval.firmware_replay` generates independent Python integer-reference fixtures and compiles the existing C log-LMDP and deadline kernels with a freestanding guest. The guest checks every final log field, next action, deadline route, objective, feasibility flag, resource fallback, label count and proposal certificate. It emits failure counts through semihosting and exits unsuccessfully if any comparison fails. Corrupted expectations are tested to prove the harness rejects a wrong result.

The target is QEMU's [MPS2 AN386 Cortex-M4 model](https://www.qemu.org/docs/master/system/arm/mps2.html); output and exit use the [Arm semihosting ABI](https://github.com/ARM-software/abi-aa/blob/main/semihosting/semihosting.rst). This is an emulated freestanding CPU execution check, not real-board calibration. Startup initializes data/BSS; a linker script reserves a 16 KiB stack inside a 200 KiB RAM region. No libc, heap, OS, or floating-point kernel runtime is linked. Removing the deadline kernel's `memset` dependency exposed and closed its freestanding-build gap.

## Final results

All 592 tests and runtime Ruff pass, including an actual Cortex-M4 emulator integration test enabled through `EMBER_ARM_TOOL_ROOT`. Native and ARM replays share identical generated fixtures and one source fingerprint:

| Metric | Host | Cortex-M4 QEMU |
| --- | --- | --- |
| Routing cases | 202 | 202 |
| Deadline cases | 600 | 600 |
| Comparisons | 13,610 | 13,610 |
| Mismatches | 0 | 0 |
| Reserved kernel/harness RAM, including stack | Not measured | 32,176 bytes |
| Observed stack high-water during replay interval | Not measured | 272 bytes |
| Reserved stack | Not applicable | 16,384 bytes |
| Text and read-only fixtures | Not measured | 491,956 bytes |

Routing cases include the 500-cell corridor beyond float desirability underflow, unreachable/death boundaries, and 200 random normalized 12-state models. Deadline cases use 200 random six-state two-corridor graphs at capacities 1, 2 and 64; feasible, infeasible and bounded-resource fallbacks are covered. These are kernel development fixtures, not physical worlds, survival evaluations or held-out tests. Text includes large read-only reference tables; it is not the deployed application code size.

The stack canary measures only the replay interval after initialization and before result printing. It is not an exhaustive worst-case stack bound. The ELF has no unresolved runtime symbols. The RAM figure covers routing/deadline scratch, harness globals and the reserved stack; it excludes the actual robot map, GBP, source metadata, transport, sensors, runtime and patient interaction. Whole embedded RAM, calibrated MCU cycles and the full Phase-1 gate remain false in the report. QEMU elapsed time or guest instruction counts are not substituted for hardware cycles.

## Reproduction

Use system tools on `PATH`, or supply an extracted tool root containing `usr/bin/arm-none-eabi-gcc`, `usr/bin/qemu-system-arm` and any local libraries. No downloaded tool binaries or Ubuntu packages are committed. This session used temporary extracted Ubuntu packages under `/tmp/ember-arm-tools/root`; they may need downloading again after a restart. Compiler: ARM GCC 14.2.1; emulator: QEMU 10.2.1. The test skips explicitly if the compiler/emulator are absent and no local tool root is configured.

```bash
.venv/bin/python -m eval.firmware_replay --seeds 0:199 --out outputs/eval/firmware-host-final
.venv/bin/python -m eval.firmware_replay --seeds 0:199 --arm --tool-root /tmp/ember-arm-tools/root --out outputs/eval/firmware-arm-final
EMBER_ARM_TOOL_ROOT=/tmp/ember-arm-tools/root .venv/bin/pytest -q
.venv/bin/ruff check cascade ember eval hazard mcu tests
```

Reports, generated fixture headers, ELF/host binaries, the ARM linker map and matching source archives are retained under ignored `outputs/eval/firmware-{host,arm}-final/`. Both reports use precommit SHA `6fc351f9e837cde4ec7963103774b017fd3b3fb6`, dirty-source fingerprint `0fc543c057c3515c24aac0e3976c5931dd4f85321e917a5942c124186bf93a39` and fixture SHA-256 `3ef2a43479e22c699ca474ad285524ea0f03b1cf79059692fa5e9ae0e1a2f4dd`.

- Host source archive SHA-256: `29563220669c2c89a9e8e01fe9f1f90a8248fc0bf00eaeb766048c4ff3790470`.
- ARM source archive SHA-256: `9ec5f29d4548f144def7a0c308d0ffdb6eca0d2bfed0efc71d780133b30311a4`.
- ARM ELF SHA-256: `d1da28a999f1ce337473fe784dbb8069cdbc8098bf2541c48b0c755d84fc4029`.
