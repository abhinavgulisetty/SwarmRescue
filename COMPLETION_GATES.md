# Research acceptance ledger — 2026-10-07

The original research proposal is **not scientifically certified**. The 0.3.1 software release completes the tested search/contact/physical-evacuation workflow; see [its engineering acceptance and model assumptions](docs/rescue.md). The September 76% score was a historical weighted checkpoint, not a current acceptance result. Passing a component or site-arrival mission does not establish whole-system deployment, survival, or paper contribution gates.

| Requirement | Current evidence | Remaining work |
| --- | --- | --- |
| Local executable deployment | Installable 0.3.1 wheel, hashed dependency lock, isolated installed-source provenance, health check and mission launcher | Remote/physical targets require a specified destination; these are distinct from the local prototype release |
| World, collapse, maps and physical driving | Implemented; regression suite and fresh physical missions pass | Broader deployment validation |
| Separate hazard beliefs, replay protection, radio and native routing | Independent robot states; physical radio delivery; geometry epochs; bounded latest evidence; tested C kernels | Whole firmware integration and bounded metadata |
| Robust integer route extraction | 500-cell underflow corridor; native/reference policy tests; Cortex-M4 replay and live UART planning | Measured hardware deadlines |
| Coupled pose/beacon localization | Joint vector GBP; all 600 development cases converge; mean error improves about 20%; six physical runs reach sites | Range transport and receiver-local estimates now implemented opt-in; host assignment, pose-prior transport, odometry/IMU and pose/hazard coupling remain |
| Sensor and hazard inference | Visible thermal/smoke scans, host raycasts and modeled sensors | Deployed lidar/thermal/gas calibration, temporal/spatial coupling under pose uncertainty |
| NCA contribution | Safeguard implemented; historical speedup about 1x | Training/export and deployed int8 evaluation; the 2x speed claim currently fails |
| VoI and resilience contributions | Implemented component and procedural grids; historical scheduler correction recorded | Fresh frozen whole-system comparisons; physical receiver information gain and failure grid |
| Firmware emulator execution | Replay grids pass; one/two private Cortex-M4 workers now perform live physical-mission routing over framed UART | Full robot firmware, physical-board transport and hardware calibration |
| Less than 200 KiB whole-robot RAM and less than 5% deadline misses | Regional numeric arena under 200 KiB; replay harness reserves 32,176 bytes; live routing service reserves 118,800 bytes including stack | Account for all metadata/map/refinement/sensor buffers and measure target cycles; component figures cannot be added or substituted for a whole image |
| Calibrated survival and patient evacuation | Procedural modeled evaluation; ground-level physical survivor proxies now evacuate to outdoor safe zones | Validate tenability/biomechanics assumptions and extend patient evacuation beyond the tested ground-level surrogate |
| Final sealed validation | Original set already opened; historical replay labeling preserved | Freeze final code/protocol/thresholds before selecting a genuinely unused evaluation set |

The latest full regression run has **708 passes**, including fresh Cortex-M4 replay and live UART checks using workspace-local tools; runtime Ruff passes. The installed tactical mission, installed replay and physical firmware mission pass. Learned speed, real sensor calibration, whole embedded resources, biomechanics calibration, broader evacuation scenarios and final scientific validation remain open. No new whole-project completion percentage is assigned.

Detailed evidence: [deployment](docs/deployment.md), [live firmware](docs/live-firmware.md), [localization](docs/range-localization.md), [firmware emulation](docs/firmware-emulation.md), [architecture](docs/architecture-experiments.md), [sensing](docs/hazard-sensing.md), and [full historical status](FULL_PROJECT_STATUS.md).


Receiver-local range transport is now exercised by eight fresh physical runs and
source-archived loss grids. An opt-in source-coverage target improves 75%-loss
mean error by 25.7% using 2.35× airtime. Known pose priors and host matching remain;
this does not close independent assignment agreement or whole-system timing.
