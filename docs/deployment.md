# Local deployment — verified 2026-10-07

SwarmRescue 0.2.0 is deployed locally as a standalone simulator release with two live Cortex-M4 routing workers. The final prefix is `outputs/deployment/final/`; the earlier `outputs/deployment/` installation is retained as a comparison checkpoint. This closes installation and live routing integration for the prototype. It does not close the full research or whole-robot firmware gates in [COMPLETION_GATES.md](../COMPLETION_GATES.md).

## Run the installed release

From the repository root:

```bash
outputs/deployment/final/venv/bin/swarmrescue doctor
outputs/deployment/final/run-mission
```

The launcher runs eight robots at seed 42 with regional independent beliefs, bounded latest evidence, receiver-local range transport and two private ARM planners. It sets a writable cache and writes `outputs/deployment/final/latest-mission.json` plus a matching installed-source archive. Extra options are forwarded, for example `--seed 7 --robots 4 --firmware-robots 1 --report /tmp/mission.json`. An absolute launcher path works from outside the checkout. The recorded acceptance is arrival at survivor sites; it is not physical patient evacuation.

## Rebuild or install elsewhere

Requirements: Linux/POSIX, Python 3.11 or 3.12, `uv`, a native C compiler and the locked runtime dependencies. ARM deployment additionally requires `arm-none-eabi-gcc`, its binutils and `qemu-system-arm` supporting `mps2-an386`. This machine's tools are extracted under `outputs/tools/arm/root`; no system packages were installed. On a machine with system-installed ARM tools, `--tool-root /` uses `/usr/bin` and retains the existing `PATH`.

```bash
uv sync --locked --extra dev
uv build --wheel --out-dir outputs/releases/0.2.0
.venv/bin/python -m swarmrescue.deploy --wheel outputs/releases/0.2.0/swarmrescue-0.2.0-py3-none-any.whl --prefix outputs/deployment/final --tool-root outputs/tools/arm/root
outputs/deployment/final/run-mission
```

Omitting `--tool-root` installs a native simulator deployment without ARM workers. The same installer is exposed as `swarmrescue deploy`. Run it from outside the target virtual environment; it refuses to replace the interpreter environment running the installer. The target prefix is an owned local installation directory. Reinstall after relocating it because its launcher records absolute paths.

The installer reads the exact dependency lock from the wheel and installs with hash verification. It checks native integer routing, MuJoCo, Python and packaged assets from an empty working directory, verifies installed import provenance, builds the packaged ARM service and completes a live UART startup/shutdown handshake before writing a ready deployment record. It creates `deployment.json`, `doctor.json`, `firmware/` and `run-mission`; it does not start a background service or flash a physical board.

The wheel includes Python sources, C headers/kernel sources, linker/replay/runtime assets, the frozen evaluation protocol and runtime lock. Installed execution fingerprints and archives the actual shipped runtime files independently of working-directory Git state. `SWARMRESCUE_CACHE_DIR` selects a writable collapse cache; caches are excluded from installed-source fingerprints. These dependencies and source archives permit replay but do not make emulator time a hardware-cycle measurement.

For a release archive without the Git checkout, extract it and bootstrap the installer in a separate environment:

```bash
uv venv --python 3.12 /tmp/swarmrescue-bootstrap
uv pip install --python /tmp/swarmrescue-bootstrap/bin/python --require-hashes -r runtime-lock.txt
uv pip install --python /tmp/swarmrescue-bootstrap/bin/python --no-deps wheel/swarmrescue-0.2.0-py3-none-any.whl
/tmp/swarmrescue-bootstrap/bin/swarmrescue deploy --wheel wheel/swarmrescue-0.2.0-py3-none-any.whl --prefix ./deployment --tool-root /
./deployment/run-mission
```

The last example assumes the ARM tools are already installed on the destination's `PATH`; use an extracted tool root instead when appropriate. The archive carries release assets and evidence, while the installer creates destination-specific paths and health records.

## Validation

All **682 tests** pass with fresh ARM execution enabled; runtime Ruff passes. Installed health checks, the tactical mission and firmware replay pass outside the checkout. The packaged replay grid makes 2,270 checks across 22 routing and 60 deadline cases with no failures. Earlier larger replay grids remain documented in [firmware emulation](firmware-emulation.md).

Six paired development missions use identical installed sources and compare zero, one and two firmware planners. All reach every assigned site, and arrival-time and search-exposure arrays agree exactly across backends for each seed. Exposure is the existing modeled heat/smoke proxy, not calibrated FED survival. Firmware transport bytes include planning request/response frames and exclude startup/shutdown; they are separate from robot radio airtime.

| Mission | Sites reached | Search exposure sum | Last arrival (sim s) | Firmware requests | Firmware transport bytes | RPC wall time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| seed 42, 0 workers | 8/8 | 21.708731 | 36.412 | 0 | 0 | 0.000 |
| seed 42, 1 workers | 8/8 | 21.708731 | 36.412 | 3 | 83780 | 5.479 |
| seed 42, 2 workers | 8/8 | 21.708731 | 36.412 | 9 | 259822 | 26.095 |
| seed 7, 0 workers | 4/4 | 36.683948 | 45.060 | 0 | 0 | 0.000 |
| seed 7, 1 workers | 4/4 | 36.683948 | 45.060 | 2 | 50792 | 3.859 |
| seed 7, 2 workers | 4/4 | 36.683948 | 45.060 | 8 | 202472 | 19.866 |

The comparison checkpoint fingerprint is `6480bcd3a2ea0fc82d3abeeae08aa950b8020620bf878cbb036de2e6e9e4f06b`. Its wheel is retained in `outputs/releases/0.2.0/checkpoints/mission-comparison/`. The final build adds the installer startup check, keeps the same ARM executable hash, and has its own installed-source fingerprint below. Reports must be attributed to their recorded fingerprint, rather than to another wheel merely sharing the unpublished 0.2.0 version.

The live routing image reserves **118,800 bytes** including 16 KiB reserved stack. Sensors, inference, assignment and geometry refinement still execute on the host; this is not whole-robot memory. RPC wall times include transport and concurrent host scheduling. Details are in [live firmware](live-firmware.md).

The final build was independently rerun from its launcher: **8/8 sites reached**, nine ARM planning requests and 259,822 planning transport bytes. Its installed fingerprint and source archive match the final wheel, and arrival/exposure arrays match the seed-42 control exactly.

## Release identity and artifacts

The saved bundle was checked after the interrupted session: all archived file hashes and external release checksums match. No validation missions or emulator workers remain running. A committed [release receipt](releases/0.2.0.json) retains the acceptance results and hashes even when generated output directories are absent.

- Final wheel SHA256: `9140bf088f889f8c979f20e10b642f5ed74b51ef240f4a4b2d5816a317fd09ef`.
- Final installed-source fingerprint: `fda91e8c8cc9b9011fd9b018d16be3bff813c03fbc8e78d1b52d989612938df3`.
- Runtime lock SHA256: `63521e8f6572f65c931e09f7f9ce0d0c83e52bc384c2b48767dc830335dc13ba`.
- ARM executable SHA256: `ee5d92860e2cfec3c6f9ec808fcf27aac8c1b52fab7723f02c6e1c6a3932c32e`.

The release directory `outputs/releases/0.2.0/` contains the wheel, the portable local-release archive, release acceptance metadata and checksums. The final deployment record is `outputs/deployment/final/deployment.json`; six comparison reports and verified source archives are in `outputs/deployment/validation/`. Generated artifacts and environments are ignored by Git; the committed installer and this guide reproduce them after a push. No Git push, remote publication, global service installation or physical-board flashing was performed. No board or remote endpoint was specified.

Whole-robot firmware, calibrated sensor/tenability models, measured hardware resources/deadlines, deployed learned speedup, physical patient extraction and genuinely unused sealed scientific validation remain open. No whole-project completion percentage is assigned.
