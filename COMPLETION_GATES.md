# Current completion gates — 2026-10-04

The research project is **not fully complete**. The September 76% score was a historical weighted checkpoint, not a current acceptance result. Passing a component or site-arrival mission does not establish whole-system deployment, survival, or paper contribution gates.

| Requirement | Current evidence | Remaining work |
| --- | --- | --- |
| World, collapse, maps and physical driving | Implemented; regression suite and fresh physical missions pass | Broader deployment validation |
| Separate hazard beliefs, replay protection, radio and native routing | Independent robot states; physical radio delivery; geometry epochs; bounded latest evidence; tested C kernels | Whole firmware integration and bounded metadata |
| Robust integer route extraction | 500-cell underflow corridor; native/reference policy tests; Cortex-M4 replay | Target firmware I/O and measured deadlines |
| Coupled pose/beacon localization | Joint vector GBP; all 600 development cases converge; mean error improves about 20%; six physical runs reach sites | Loss-limited range transport, independent receiver estimates, odometry/IMU and pose/hazard coupling |
| Sensor and hazard inference | Visible thermal/smoke scans, host raycasts and modeled sensors | Deployed lidar/thermal/gas calibration, temporal/spatial coupling under pose uncertainty |
| NCA contribution | Safeguard implemented; historical speedup about 1x | Training/export and deployed int8 evaluation; the 2x speed claim currently fails |
| VoI and resilience contributions | Implemented component and procedural grids; historical scheduler correction recorded | Fresh frozen whole-system comparisons; physical receiver information gain and failure grid |
| Firmware emulator execution | 202 routing + 600 deadline cases pass on host and Cortex-M4; 13,610 comparisons each | Full robot firmware, target transport and hardware calibration |
| Less than 200 KiB whole-robot RAM and less than 5% deadline misses | Regional numeric arena under 200 KiB; replay harness reserves 32,176 bytes | Account for all metadata/map/refinement/sensor buffers and measure target cycles; component figures cannot be added or substituted for a whole image |
| Calibrated survival and patient evacuation | Procedural modeled evaluation; physical missions reach assigned sites | Validate tenability assumptions and implement physical patient extraction/egress |
| Final sealed validation | Original set already opened; historical replay labeling preserved | Freeze final code/protocol/thresholds before selecting a genuinely unused evaluation set |

The latest full regression run passes **592 tests**, including configured Cortex-M4 execution, and runtime Ruff passes. Learned speed, real sensor calibration, whole embedded resources, patient evacuation and final scientific validation remain open. No new whole-project completion percentage is assigned.

Detailed evidence: [localization](docs/range-localization.md), [firmware emulation](docs/firmware-emulation.md), [architecture](docs/architecture-experiments.md), [sensing](docs/hazard-sensing.md), and [full historical status](FULL_PROJECT_STATUS.md).
