# Live Cortex-M4 routing — 2026-10-06

The physical mission can delegate the integer route solves of one or two robots to private persistent firmware workers. Each worker receives that robot's current regional model and returns the integer field and actions over framed UART transport. Geometry replacement retains the robot-to-worker binding. Hazard inference, range localization, assignment, radio and fine-geometry refinement still execute on the host; this is a routing service, not whole-robot firmware.

## Runtime and ownership

`firmware/runtime/service.c` uses static storage for at most 2,048 nodes and 8,192 transitions. The freestanding ARM build has no libc, heap or floating-point dependency. Inputs stream directly into those arrays. Each request has a command, sequence number, length and CRC32; responses have matching identity and CRC32. Damaged input is rejected before solving, oversized dimensions are rejected, and bounded solves report nonconvergence rather than silently substituting a host solution. The host checks returned actions against the returned integer field and model; it does not recompute the Bellman field.

`mcu/runtime.py` starts a separate process per selected robot and bounds request writes and reads by a wall-clock timeout. Startup verifies capacities and linked RAM against build metadata; the mission checks the image hash before and after execution. Workers receive an explicit shutdown command and are reaped on completion/error. Planning transport bytes count request and response framing/payloads, excluding startup and shutdown; they are distinct from modeled robot radio airtime.

The ARM transport uses the MPS2 AN386 UART model. Data flows through UART MMIO and QEMU's serial pipe; semihosting is used only to exit the emulator. This matches the primary [QEMU MPS2 board implementation](https://raw.githubusercontent.com/qemu/qemu/master/hw/arm/mps2.c) and [CMSDK UART implementation](https://raw.githubusercontent.com/qemu/qemu/master/hw/char/cmsdk-apb-uart.c).

The linked service reserves **118,800 bytes**, including a 16 KiB stack reservation, under a 200 KiB linker limit. This is static linked extent, not observed stack high-water or whole-robot RAM. RPC wall time includes host/emulator scheduling and transport; it is not a hardware-cycle measurement. The busy-polling UART implementation can consume a host CPU while waiting, so concurrent emulator wall times should not be compared with native solve latency as an algorithm speed result.

## Regression and deployment validation

All **682 tests** pass with the workspace-local ARM toolchain configured; runtime Ruff passes. Persistent host and ARM services return bit-exact reference fields and routes for 20- and 500-node corridors. Repeated models, damaged CRCs, invalid dimensions, capped solves, capacity rejection, changed executable hashes and robot ownership across geometry replacement are covered. The deployed wheel builds its own ARM image outside the source checkout. Its eight-robot seed-42 checkpoint reaches 8/8 sites using two firmware workers and nine planning requests; the source archive matches its installed fingerprint.

Six matched release-checkpoint missions compare zero, one and two firmware workers at seeds 7 and 42. Their results and release hashes are recorded in [the deployment report](deployment.md). These are illustrative development missions, not a fresh sealed evaluation or calibrated survival study.

An earlier concurrent source-checkout batch reached sites but failed the before/after source-identity check; its missing reports are excluded from acceptance evidence. The installed-release comparison fingerprints and archives actual shipped files independently of working-directory Git state. A successful trajectory does not bypass the source-stability requirement.

## Direct commands

```bash
.venv/bin/python -m mcu.runtime --arm --tool-root outputs/tools/arm/root --out outputs/firmware/runtime-arm
.venv/bin/python -m cascade.demo --seed 42 --robots 8 --compact --live-disaster --distributed --region-cells 4 --latest-evidence --range-gbp --range-radio --firmware-robots 2 --firmware-image outputs/firmware/runtime-arm/runtime.elf --firmware-tool-root outputs/tools/arm/root --no-render
```

Omit `--tool-root`/`--firmware-tool-root` when ARM GCC and QEMU are already on `PATH`. `--firmware-platform host` selects the persistent native service for transport debugging. Firmware routing remains opt-in for source missions; the prepared deployment launcher selects two ARM workers explicitly.
