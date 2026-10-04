# EMBER firmware kernels

This directory starts Phase 3 Tier-B validation with a freestanding C99 Q16.16
Bellman sweep. The Python reference is in `mcu/lmdp.py`, and
`tests/test_firmware_lmdp.py` compiles the C kernel as a host shared library and
checks bit-exact outputs.

This is a cross-language conformance gate only. It is not yet evidence of ESP32 or
Cortex-M4 cycle counts. The next Tier-B step is to build the same source under
QEMU-xtensa and Renode/Cortex-M4, then record real cycle counters and memory maps.

## Underflow-safe routing

`log_lmdp.c` / `log_lmdp.h` implement Q16.16 negative-log desirability sweeps
and deterministic policy selection. `log_lmdp_lut.h` stores a 16,388-byte softplus
lookup table in read-only storage. Runtime sweeps and policy extraction require no
floating-point operations, dynamic allocation, logarithms, or exponentials.
The host adapter in `mcu/log_lmdp.py` quantizes models into explicit arena buffers.
`INT32_MAX` denotes unreachable states; finite model costs must fit the remaining
signed Q16.16 range. Route extraction never reconstructs tiny desirabilities.

`tests/test_log_lmdp.py` checks a 500-cell corridor, randomized bit-exact Python/C
sweeps, native policy choices, table consistency, and arena limits. The ablation
schema keeps the original linear Q16 baseline and adds a separate log-Q16
comparison. These remain host conformance and analytical storage checks; QEMU,
Renode, full robot metadata RAM and MCU cycle measurements are still pending.

## Deadline-constrained fallback

`deadline.c` / `deadline.h` provide a proposal-time certificate and an integer
Pareto fallback with a lexicographic fastest-route resource fallback. All CSR
arrays, labels, fast-path workspace, route output and result structures are
supplied by the caller. No runtime floating-point math or allocation is used.
Labels retain both elapsed clock steps and Q16 estimated risk objective; faster
alternatives are not discarded merely because another path has lower risk.
`tests/test_firmware_deadline.py` checks randomized native/reference route,
objective, feasibility, label-cap and proposal-certificate equivalence. Timing
and whole-robot RAM calibration remain separate gates.

## Cortex-M4 emulator replay

The freestanding replay target is now implemented in `replay/`, with a Cortex-M4
startup, semihosting results, bounded linker layout and stack canary. The native
and QEMU development grids each pass 13,610 checks across 202 routing and 600
deadline cases. Run `python -m eval.firmware_replay --arm` with ARM GCC/QEMU on
PATH or a `--tool-root`. See [the emulator report](../docs/firmware-emulation.md).
This closes kernel emulator execution; the earlier pending-emulator statements
above are historical. Whole robot firmware, RAM and measured timing remain open.
