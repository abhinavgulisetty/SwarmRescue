# Routing firmware

These C kernels support native routing and the Cortex-M4 services used by the
verified complete-rescue simulator. They are routing components of a hybrid
host/ARM system. The host supplies sensing, beliefs, assignment and physics.

## Kernels and services

- `lmdp_q16.c` supplies the linear Q16.16 Bellman baseline.
- `log_lmdp.c` and its lookup table supply negative-log integer routing, including
  deterministic policy selection without reconstructing tiny desirabilities.
- `deadline.c` supplies the bounded Pareto fallback and proposal checks.
- `deadline_exact.c` supplies the compiled exact deadline backend.
- `runtime/` supplies native and Cortex-M4 routing services with framed request
  identity and integrity checks. `replay/` supplies the freestanding replay target.

The corresponding `mcu/` adapters build and invoke these kernels. Use
`swarmrescue firmware-build --help` for build options. The existing
`outputs/deployment/complete-rescue-0.7.0/run-mission` launcher uses two persistent
Cortex-M4 workers under QEMU; append `--firmware-robots 0` for native routing.
The retained toolchain is in `outputs/tools/arm/root/`.

## Validation and limits

Firmware conformance and live-service tests remain in `tests/`. Enable the local
ARM tools with `EMBER_ARM_TOOL_ROOT` when running the full suite. The verified
release includes an ARM image and matched native/ARM mission reports; see the
[delivery report](../RESEARCH_DELIVERY.md) and
[current architecture](../docs/architecture.md).

The ARM emulator executes routing instructions. Component memory figures and
reserved stack space do not certify whole-robot RAM. Physical-board cycle
measurements, complete robot firmware and physical-board deployment remain open.
