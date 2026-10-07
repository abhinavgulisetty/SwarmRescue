# Complete rescue simulation release — 2026-10-07

Version 0.3.1 completes the implemented search → local contact → physical evacuation workflow and deploys it as a standalone application. Production uses independent robot beliefs, receiver-local range factors, regional integer routing, current structural geometry, and optional private Cortex-M4 routing workers. It uses the tested classical planner rather than depending on an unverified learned speedup. Assignment, sensor simulation, fine geometry and survivor dynamics run on the host; this is an explicit hybrid architecture.

## Run

```bash
outputs/deployment/rescue/run-mission
# Or the native installed workflow:
outputs/deployment/rescue/venv/bin/swarmrescue rescue --seed 42 --robots 8
# Enable the actual rendered simulation:
outputs/deployment/rescue/run-mission --firmware-robots 0 --render --speed 2 --fps 12
```

The prepared launcher selects two live Cortex-M4 workers and writes `latest-mission.json` plus its source archive. `rescue` supplies the complete mission preset; later options override seed, robot count, rendering and firmware settings. `mission` retains the previous source interfaces, including search-only runs when `--evacuate` is omitted. The validated rescue scenes place survivors on the ground level of a dynamically collapsing multi-storey building. Existing multi-storey traversal remains a separate supported mode; upstairs patient evacuation is not established by these ground-level cases.

## What completion means

The implemented software acceptance contract is `physical-rescue-software-v1`: all assigned survivors must be localized, individually contacted, physically moved into an outdoor zone at least 0.5 m beyond the footprint, and remain supported there; all robots must also be physically outside by at least 0.35 m. A parked robot near a beacon is insufficient. Reentry revokes a survivor's safe-zone witness. Contact requires proximity and an unobstructed local sensor ray; ownership is one-to-one. A selected firmware worker must have actually solved live models.

The independent `rescue-evaluation` command checks final coordinates, ownership, event ordering, firmware work and archived source hashes. It rejects missing/indoor/fallen survivors, duplicate owners, corrupted archives and mixtures of runtime versions, even when a report claims success. It can analyze existing reports or run a reproducible seed grid:

```bash
outputs/deployment/rescue/venv/bin/swarmrescue rescue-evaluation --seeds 1,7,19,42 --robots 8 --out outputs/rescue-validation
```

For live ARM validation, additionally provide `--firmware-robots 2 --firmware-image outputs/deployment/rescue/firmware/runtime.elf --tool-root outputs/tools/arm/root`. Run source-based grids only with stable source; installed grids fingerprint their actual shipped runtime files. These are software acceptance/development cases, not a newly sealed scientific experiment.

## Physical model and algorithm decisions

Survivors are 55 kg upright capsule proxies, 0.78 m tall, with three translational degrees of freedom and idealized balance/propulsion. Their configured sliding resistance is 0.08. MuJoCo resolves gravity, inertia and contacts; the controller applies bounded forces and never writes survivor positions after scene initialization. Body orientation/gait, mutual agent collisions and medical physiology are not modeled. This is a reproducible ambulatory surrogate, not a calibrated human biomechanics or survival model.

The initial building runs exposed a contact mismatch: the concrete floor's friction dominated the actor parameter and prevented locomotion. The corrected model gives survivor contact parameters explicit priority and tests movement against the real concrete-floor coefficient. MuJoCo documents the [contact parameter mixing and priority rules](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters) and [applied-force state](https://mujoco.readthedocs.io/en/stable/programming/simulation.html). Structural contacts remain enabled, and an obstructing wall demonstrably prevents rescue.

Escorts follow bounded breadcrumbs from the robot's actual trajectory and wait when their survivor falls behind. The faster controller continues through intermediate samples and preserves tight turning points, slowing at the trajectory tail. Current geometry can replace an outdoor goal when collapse disconnects the previous destination; infeasibility pauses movement and is retried after map changes. Transport failures are not converted into a successful host fallback.

## Validation and release identities

All **708 regression tests** and runtime Ruff pass, including fresh ARM execution. Five 0.3.0 physics acceptance missions pass across four buildings; 0.3.1 corrects the rendered result card without changing the rescue algorithms. Final 0.3.1 native-rendered and live-ARM reruns both pass and are separately archived. They match all eight physical/timing/exposure/radio fields of the checkpoint exactly. The final video correctly reports 8/8 survivors evacuated in 81.4 simulated seconds. All archived sources share the same final fingerprint. The checkpoint native and live-ARM seed-42 runs agree exactly on survivor contact/evacuation times, final positions, traveled distances, modeled exposure, robot positions/arrival times and robot radio bytes. The ARM case performs 25 live planning requests.

| Seed | Robots | ARM workers | Survivors evacuated | Last evacuation (sim s) | Survivor exposure proxy sum | Firmware requests |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 0 | 8/8 | 89.236 | 62.528040 | 0 |
| 19 | 8 | 0 | 8/8 | 119.914 | 41.111137 | 0 |
| 42 | 8 | 0 | 8/8 | 81.396 | 8.825714 | 0 |
| 42 | 8 | 2 | 8/8 | 81.396 | 8.825714 | 25 |
| 7 | 4 | 0 | 4/4 | 65.968 | 63.917079 | 0 |

A slower-controller development checkpoint also rescues all survivors. Against it, the final controller lowers last evacuation time by 27.5% in seed 1, 44.2% in seed 7 and 33.2% in seed 42. Modeled survivor exposure also falls in all three comparisons. These paired examples establish a useful control improvement in the tested scenes, not universal algorithm superiority or medical survival.

The failed first-pass experiments remain in `outputs/experiments/rescue/`; the corrected but slower checkpoint is in `outputs/experiments/rescue-v2/`. Final mission evidence and the independent verification summary are in `outputs/deployment/rescue/validation/`. Release artifacts are in `outputs/releases/0.3.1/`; the committed release receipt preserves the hashes/results without requiring generated directories in Git.

## Rebuild and deploy

```bash
uv sync --locked --extra dev
uv build --wheel --out-dir outputs/releases/0.3.1
.venv/bin/python -m swarmrescue.deploy --wheel outputs/releases/0.3.1/swarmrescue-0.3.1-py3-none-any.whl --prefix outputs/deployment/rescue --tool-root outputs/tools/arm/root
outputs/deployment/rescue/run-mission
```

The installer validates hashed dependencies, assets, native physics/routing and installed provenance from an empty working directory, builds the packaged ARM image and verifies UART startup/shutdown. Readiness is cleared before replacing an installation, so a failed update cannot leave a stale ready record. New launchers select evacuation only when the installed CLI advertises support; older wheels retain their search-only launcher. Extracted tools can be supplied with `--tool-root`, or use `--tool-root /` with system-installed ARM tools; omitting it deploys the native workflow. Python 3.12/Linux is the tested environment.

## Portable bundle

`outputs/releases/0.3.1/swarmrescue-0.3.1-complete-rescue.tar.gz` contains the wheel, exact dependency lock, ARM image/build metadata, independently verified reports/source archives, supporting documents, the rendered demo and checksums. Extract it and install without the checkout:

```bash
uv venv --python 3.12 /tmp/swarmrescue-bootstrap-031
uv pip install --python /tmp/swarmrescue-bootstrap-031/bin/python --require-hashes -r runtime-lock.txt
uv pip install --python /tmp/swarmrescue-bootstrap-031/bin/python --no-deps wheel/swarmrescue-0.3.1-py3-none-any.whl
/tmp/swarmrescue-bootstrap-031/bin/swarmrescue deploy --wheel wheel/swarmrescue-0.3.1-py3-none-any.whl --prefix ./deployment --tool-root /
./deployment/run-mission
```

The example assumes ARM GCC/binutils/QEMU are installed on the destination. Omit `--tool-root /` for native deployment. The installer creates destination-specific absolute paths; the packaged original deployment record is evidence, not a relocatable environment. `CONTENTS-SHA256SUMS` covers archive contents; external `SHA256SUMS` and the committed receipt cover the bundle itself.

Final wheel SHA256: `f6b4f60ef679b96ec607de16ee99028d618cad9594f3f51cd254261f75b1ef37`.
Installed-source fingerprint: `061dadcbe3e9c9a2d0f3b64d77746c7347b827d67848aa59379e38f9c4bae512`.
ARM image SHA256: `ee5d92860e2cfec3c6f9ec808fcf27aac8c1b52fab7723f02c6e1c6a3932c32e`.

No Git push or physical-board flashing was performed. The completed software release does not certify the original proposal's NCA/VoI/failure thresholds, whole-robot embedded RAM or calibrated hardware/sensor/medical performance. The [research ledger](../COMPLETION_GATES.md) retains those independent hypotheses and historical findings; no percentage is substituted for evidence.
