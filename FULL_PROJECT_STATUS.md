# EMBER Whole-Project Status

**Audit date:** 2026-09-29
**Current whole-project completion:** **76%**
**Target:** 75%
**Status:** **target reached; project remains incomplete**

This is the controlling completion report. `CHECKPOINT_75.md` measures readiness of
the academic review package; it is not a claim that 75% of the research system is
implemented. The percentage below is weighted by the proposal's fourteen-week plan:
Phase 0 has weight 2/14 and Phases 1--3 each have weight 4/14. Partial credit requires
running code or reproducible evidence, not just design text.

## Weighted audit

| Phase | Weight | Completion | Contribution | Evidence and boundary |
|---|---:|---:|---:|---|
| Phase 0: world and collapse | 14.3% | 100% | 14.3 | Procedural structure, simulated collapse/cache, graph extraction, physical traversal, manifests |
| Phase 1: classical MVP | 28.6% | 88% | 25.1 | Hazard, Tier A, classical LMDP, flooding, B1/B2, lambda selection wired into the policy, and analytical MCU profiling work; full sensors and coupled physical GBP remain |
| Phase 2: contributions | 28.6% | 73% | 20.9 | Safe NCA, mission-level VoI, message aging, progressive collapse, distributed B3 and the 20-world/5-failure/3-scale grid run; NCA training/speed and physical-mission integration remain |
| Phase 3: validation | 28.6% | 55% | 15.7 | Five paired ablations, Q16-vs-float diagnostics, a hash-locked protocol, and all 50 held-out worlds run; QEMU/Renode, physical validation, and final demo remain |
| **Whole project** | **100%** |  | **76.0% (76%)** | **Requested 75% target reached** |

## Phase 2 work now running

- `ember/routing/nca.py` implements a 2,465-byte int8 NCA proposal with a residual
  safeguard. Corrupted weights still converge because rejected proposals fall back to
  Jacobi. This is a global reference safeguard, not yet the proposed local MCU test.
- `ember/voi.py` solves the transposed adjoint system, scores Gaussian updates by
  adjoint-weighted KL, and enforces exact packet-byte token buckets.
- `ember/gbp.py` ages stale edge messages as well as local evidence, with a partition
  and reconnection regression test.
- `ember/consensus.py` implements Metropolis Gaussian average consensus. The schema-6
  procedural evaluator uses it as the distributed B3 risk-inflated A* baseline.
- `eval/phase2.py` benchmarks C2/C3 on 20 development worlds without opening held-out
  seeds.
- `eval/resilience.py` runs five failure levels at 8, 16, and 24 robots across 20 worlds.
- `eval/mcu_profile.py` records fixed-arena high-water and analytical cycle budgets.
- `eval/lambda_selection.py` selects the control temperature using development regret and
  trajectory optimism.
- `eval/ablations.py` runs paired no-VoI, no-GBP, beta-zero, NCA-vs-Jacobi, and
  float-vs-Q16 comparisons over 20 development worlds.
- `eval/heldout.py` verifies a hash-locked protocol before opening all 50 held-out worlds.
- `eval/procedural.py` now gives EMBER a method-specific VoI radio path rather than the
  flooding traffic used by B2.

Current development evidence:

| Check | Result | Decision |
|---|---:|---|
| Safeguarded NCA convergence | 20/20 worlds | Safety component passes |
| Mean Bellman sweep speedup | 1.002x | C2 speed claim fails; 2x gate not met |
| NCA parameter storage | 2,465 bytes | Fits the proposed ~2.5 KB envelope |
| VoI byte reduction vs flooding | 84.2% | Component passes; physical mission integration pending |
| Procedural VoI bytes | 3,560 vs 8,184 | 56.5% mission-level reduction on 24 development scenarios |
| C4 resilience grid | 300 scenarios | 20 worlds × 5 failure levels × 8/16/24 robots; all partitions recovered |
| Tier-A resource profile | 17,200 bytes; 2.74% cycles | Analytical only; zero modeled deadline misses |
| Development lambda selection | 0.25 | Selected over 20 worlds before the frozen held-out run |
| Lambda optimism gap | mean 73.47, maximum 355.99 | Diagnostic works and shows the caveat is material |
| Paired ablation grid | 20 worlds × 5 variants | Complete; full mean route objective 41.667 |
| Mission-level NCA speedup | 1.016x | Still fails the frozen 2x gate |
| Q16 vs float | max field error 2.11e-5; 0% route completion | Fixed point passes, deployable-route gate fails |
| Frozen held-out result | +0.083 exposure, 95% CI [-4.585, 6.482] | 31 wins, 1 tie, 18 losses; superiority gate fails |
| Held-out VoI radio reduction | 55.6% | 3,576 vs 8,047 mean bytes |

The ablations are development results; the final procedural comparison uses held-out
worlds. Neither is a survival, measured embedded timing, physical-swarm, or
paper-superiority claim.

## Phase 3 work now running

- `mcu/lmdp.py` is the bit-exact Python reference for the Q16 sparse Bellman sweep.
- `firmware/lmdp_q16.c` and `firmware/lmdp_q16.h` provide the C99 implementation.
- `tests/test_firmware_lmdp.py` compiles the C code on the host and checks its outputs
  bit-for-bit against Python.
- `eval/frozen_protocol.json` fixes seeds, sampling, lambda, beta, radio budget, and claim
  gates. `eval/heldout.py` refuses modified protocols by SHA-256.
- `outputs/eval/heldout-final.json` contains 150 scenarios from all 50 held-out worlds.

This is the start of Tier B, not Tier-B completion: no QEMU-xtensa or Renode execution,
measured cycle/RAM calibration, or socket-connected live firmware exists yet.

## Work remaining beyond the 75% checkpoint

1. Finish the Phase 1 acceptance gate: full sensor/raycast coverage, physical-swarm GBP,
   and an unseen-seed eight-robot run with measured RAM/deadlines.
2. Integrate safeguarded NCA, message aging, and distributed inference into the physical
   mission; procedural VoI and B3 are already active.
3. Train or explicitly close the NCA experiment. The current 1.002x result must not be
   described as acceleration.
4. Execute the C kernel under at least one proposed emulator and report Tier-A/Tier-B
   cycle and RAM calibration error.
5. Replace the Q16 configuration or scaling strategy that currently underflows route
   extraction despite small absolute field error.

The project has reached **76%** on this weighted audit. That is a checkpoint, not a
completion claim: the physical, embedded-emulator, and fire-tenability boundaries above
remain open.

## Validation commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/python -m eval.phase2 --world-seeds 0:19
.venv/bin/python -m eval.procedural --world-seeds 0:7 --noise-seeds 0:2
.venv/bin/python -m eval.resilience --world-seeds 0:19
.venv/bin/python -m eval.mcu_profile --world-seeds 0:19
.venv/bin/python -m eval.lambda_selection --world-seeds 0:19
.venv/bin/python -m eval.ablations --world-seeds 0:19
.venv/bin/python -m eval.heldout
```

The held-out set was opened through the frozen protocol whose SHA-256 is
`e77caa47310419c8329f5d66c31dfd5b523822ddde21f743d647f28a578d647c`.

## Evaluation validity correction — 2026-10-04

The numerical tables above describe the September implementation. A subsequent
code audit found that procedural VoI adjoint source weights used hidden survivor
truth, while route planning used noisy BLE estimates. Schema 7 removes this
information advantage and uses BLE estimates for communication too. Both radio
policies now use the `radio_loss` seed stream; link-loss draws are still sequential
and packet-order dependent, so eventwise common-random-number coupling remains
unfinished.

The corrected development run uses worlds 0–199 and noise seeds 0–2, with paired
world-cluster bootstrap intervals. Its artifacts are
`outputs/eval/development-fairness-v7.{json,csv}` and
`outputs/eval/development-fairness-v7-source.tar.gz`. Earlier VoI and held-out
figures must not be cited as validation of the corrected scheduler. The original
held-out files are preserved; subsequent uses of seeds 10000–10049 are explicitly
historical replays. The frozen configuration hash does not lock source code.
This correction does not update the historical completion percentage or establish
survival, embedded timing, or physical-swarm performance.

## Deployment and distributed-state milestone — 2026-10-04

Completed code and validation in this continuation:

- `mcu/log_lmdp.py` and `firmware/log_lmdp.{c,h}` replace linear-Q16 underflow
  with integer negative-log desirability and LUT soft-min arithmetic. The legacy
  kernel remains as a comparison. Native C sweeps and actions match Python bit
  for bit. A 500-cell deterministic corridor remains routable even when float
  desirability underflows.
- The 20-world component gate achieves 100% route completion and agreement with
  the float policy, with 6,476 bytes maximum kernel arena use. This closes the
  recorded Q16 route-underflow blocker; it does not establish emulator timing.
- `ember/robot_state.py` gives receivers separate GBP state, packet queues,
  source-factor deduplication, local sensor epochs and cycle budgets. Routing
  reuses the same arena tail across replans. Scratch allocations and host Python
  metadata are explicitly outside measured deployment-RAM claims.
- Procedural schema 9 routes using the assigned robot's map and retains unsent
  local sensor evidence. Loss draws are keyed by absolute slot and directed link
  rather than sequential RNG consumption.
- `ember/physical.py` exposes persistent peer-informed integer replanning.
  `ember/geometry_sensing.py` queries current structural box poses for optical
  occlusion and radio wall counts. Moved/disabled walls and cinematic transparency
  have regression coverage. Non-box geometry is not handled by this helper.

The distributed development experiment covers 20 worlds × 3 noise replicates.
Its exposure difference versus B2 is +34.274, with a world-cluster 95% interval
[-5.567, 90.795]; 8 world wins, 2 ties, 10 losses. Distribution has exposed a
performance gap, not established a superiority result. Artifacts are
`outputs/eval/distributed-log-q16.{json,csv}` and its source archive. Component
ablations are in `outputs/eval/ablations-log-q16.json` (schema 2).

The completion percentage from September has not been re-estimated. Remaining
work includes the full physical demo loop integration and post-collapse
rerasterization, incremental coupled BLE/pose/PDE inference, complete sensors,
calibrated gas/heat tenability, learned int8 acceleration, matched-budget KL-only
comparison, and QEMU/Renode cycle/RAM calibration. QEMU and Renode are not installed
in the current environment. Earlier held-out worlds remain historical evidence.

The full runtime test suite passes. Changed runtime modules pass Ruff. A whole-tree
Ruff invocation additionally finds 263 pre-existing issues in the presentation
builder `scripts/build_review2_ppt.py`; these are not treated as a passing global
lint gate.

## Algorithm innovation and promotion — 2026-10-04

A paired diagnosis located avoidable LMDP detours in worlds 9 and 19. The new
`deadline_lmdp` safeguard preserves feasible proposals and uses bounded integer
Pareto routing only when the proposal cannot complete in time. Its fastest-route
resource fallback prevents label-memory exhaustion from losing feasibility.
Native C certificate and fallback kernels pass randomized bit-exact host tests.

Eighteen initial combinations of mapping, routing and matched-source-budget
communication were compared on disclosed discovery worlds. Six candidates were
checked on worlds 20–59 with three noise replicates. The discovery winner did not
beat legacy EMBER there. A deadline-safeguarded revision was then tested on worlds
0–19 and independently on worlds 60–99; the adjoint/GBP hybrid was promoted as the
procedural development default. No held-out worlds were used for this selection.
These are exploratory development comparisons with pointwise bootstrap intervals,
not a sealed or multiplicity-adjusted paper result.

The original 60 scenarios improved from 95.0% to 98.3% rescue completion, matching
B2 without increasing bytes. The later 120 scenarios matched B2 at 97.78% completion
with about 54% fewer bytes. Raw exposure was lower in mean, but its confidence
interval crossed zero on the latter set. A stable-source 200-world comparison is
available through `python -m eval.innovation`; reports and source archives live
under `outputs/eval/innovation-*`.

The exposure metric now separates actual heat/smoke proxy accumulation from timeout
penalties. This corrects interpretation, not ISO calibration. KL-only and greedy
occupancy-coverage scheduling remain selectable comparison algorithms; the latter
was not promoted. The historical held-out replay pins the legacy router.

No whole-project completion claim or revised completion percentage is made. The
physical integration, coupled localization/PDE factors, full sensors/tenability,
trained acceleration and emulator-calibration gaps listed above remain open.

## Full revised development grid — 2026-10-04

The promoted default has now completed worlds 0–199 × noise seeds 0–2:
600 paired scenarios and 1,800 modeled survivor trials. The source manifest was
unchanged throughout the experiment; the archived source metadata matches it.

| Method | Rescues | Completion | Mean raw exposure proxy | Mean radio bytes |
|---|---:|---:|---:|---:|
| Legacy distributed EMBER | 1,715/1,800 | 95.278% | 15.985184 | 3,575.36 |
| Deadline-safeguarded EMBER | 1,730/1,800 | 96.111% | 15.997711 | 3,575.36 |
| B2 flooding Dijkstra | 1,730/1,800 | 96.111% | 21.426481 | 7,998.72 |
| B3 consensus | 1,730/1,800 | 96.111% | 21.136210 | 7,998.72 |

The safeguard recovers 15 additional rescues with zero rescue-regression scenarios.
Its completion difference from legacy EMBER is +0.833 percentage points, with a
world-cluster 95% bootstrap interval [0.167, 1.833]. Raw exposure changes only
+0.012527 versus legacy, with interval [-0.004133, 0.033213]; recovering previously
unfinished evacuations also adds their actual traversal exposure.

Against B2, raw exposure difference is -5.428770, with interval
[-8.754357, -2.824996]. Against B3 it is -5.138500, with interval
[-7.784759, -2.926294]. Radio traffic falls 55.30% against both baselines. Mean
rescue time is 82.58 seconds, about 4.20 seconds slower than B2 and 4.21 seconds
slower than B3. This is a rescue/exposure/traffic tradeoff, not universal dominance.

There are 85 guard interventions: 15 additional feasible rescues and 70 plans
whose contact time leaves no feasible evacuation path. No label-budget fallback
was needed in this grid; resource fallback remains covered by dedicated tests.

Reports: `outputs/eval/innovation-development-200.json` and
`outputs/eval/innovation-development-200-summary.md`. Exact evaluated source:
`outputs/eval/innovation-development-200-source.tar.gz`, SHA-256
`8ffcb907483eeb3cffb146ec19d8786a844f93a046cf0d70e108c372255b718a`.

All 449 tests pass, runtime Ruff passes, and the tactical mission acceptance check
passes. These are exploratory development results, with pointwise intervals and
no multiplicity adjustment. They do not reopen a sealed test, calibrate ISO
survival, measure MCU timing, or close the physical-integration research gaps.
The September completion percentage remains historical and is not re-estimated.


## Distributed physical integration — 2026-10-04

The optional `cascade.demo --live-disaster --distributed` path now connects live
MuJoCo motion to independent persistent RobotState maps, occlusion-filtered local
scans, geometry-aware radio, bounded source-packet selection, interleaved GBP and
native host integer routing. Source identities survive selection and relays;
posterior marginals are never inserted as independent measurements.

After the first thermal-failure episode, exact current structural poses rebuild
traversability. Direct local evidence transfers only to surviving compatible
cells; spatial factors and routes are rebuilt. Delivered peer observations affect
routes. The historical statement that physical APIs were not wired into the demo
is superseded for this ground-level integration.

Failures encountered during development led to collision-free exterior entry,
a mast/wheel-aware swept robot envelope, a four-metre exterior map margin,
physical-distance deadline certificates and tighter target stopping. Failed
versioned reports remain available. Final seed 7 reaches 4/4 target sites and
seed 42 reaches 8/8; both pass the disclosed demo acceptance checks. These checks
do not satisfy the complete Phase 1 research gate. Site selection changes with
planning geometry, so this is not a paired statistical superiority result.

| Development mission | Sites reached | Delivered peer factors | Radio air bytes | Numeric arena peak |
|---|---:|---:|---:|---:|
| Seed 7 / four robots | 4/4 | 189 | 3,952 | 1,361,628 bytes |
| Seed 42 / eight robots | 8/8 | 882 | 14,784 | 1,230,276 bytes |

All 461 tests and runtime Ruff pass. The final four-robot clip contains 191 decoded
frames at 12 fps, was visually reviewed, and correctly identifies centralized BLE
and the unvalidated survival/timing scope. Each machine report matches its exact
source archive. Detailed evidence: `outputs/eval/physical-integration-summary.md`;
reviewed video: `outputs/demo/distributed-final-reviewed/compact/seed7_end_to_end.mp4`.

Remaining limits: planning is ground-level (cells up to one metre high), BLE stays
centralized, and geometry rebuilds cover the first failure episode only. Continuous
geometry reconstruction and hazard-PDE coupling, coupled range/pose inference,
full sensors, calibrated tenability, physical patient evacuation, trained int8
acceleration and MCU/emulator timing remain unfinished. Numeric arena memory alone
exceeds the 200 KB target; host metadata and temporary buffers add further memory.
Zero analytical inference-tick misses does not establish measured whole-stack
deadlines. The September percentage remains historical; no new completion
percentage or whole-project completion claim is made.


## Lossless topology memory reduction — 2026-10-04

GBP now validates graph endpoints before storing endpoint and reverse-message
indices in the smallest supported unsigned width. Floating-point belief arithmetic
is unchanged. Physical arena capacity is calculated from aligned retained buffers
and the integer router's transition bound.

All 473 tests and runtime Ruff pass. Disclosed seed 7 and seed 42 physical replays
still reach 4/4 and 8/8 target sites; all 40 mission behavior metrics match their
previous runs exactly. Numeric peaks fall from 1,361,628 to 984,756 bytes (seed 7)
and from 1,230,276 to 891,588 bytes (seed 42), reductions of 27.7% and 27.5%.
Allocated arena capacity also falls by about 30%. The 200 KiB gate remains unmet,
and host graph, scratch, packet/provenance metadata and interpreter memory remain
additional. See [the tracked evidence and reproduction commands](docs/compact-topology.md).


## Connected-region architecture checkpoint — 2026-10-04

An opt-in two-level planner (`--region-cells 4`) contracts connected fine cells
for private GBP and native integer routing, then refines certified terrain-aware
corridors using each robot's direct fine readings. The flat planner remains the
default. Live distributed missions now refresh geometry after further material
rubble motion; bounded provenance rejection preserves duplicate history and
reports dropped observation attempts instead of aborting repeated sensing.

All 501 tests and runtime Ruff pass. Final paired development runs reach 4/4 sites
(seed 7) and 8/8 (seeds 19 and 42) with both planners. Compact numeric peaks are
130,952–150,272 bytes, 83.3–86.9% below flat controls. Full-mission exposure improves
in seed 7 but rises slightly in the other two; compact routes finish later.
Radio gains are mixed, and repeated refresh increases traffic over the earlier
one-snapshot protocol. These are three development layouts with one replicate
each, not statistical superiority evidence.

See [the tracked experiment report](docs/architecture-experiments.md) for rejected
versions, paired dose/time/traffic measurements, provenance hashes, reproduction
commands and the reviewed 4/4 clip. The whole embedded RAM gate remains false:
fine geometry, certificate construction/refinement scratch, packet metadata and
Python runtime lie outside the numeric arena. Coupled localization and hazard
geometry, deployed sensors, calibrated tenability, learned acceleration, measured
firmware timing and physical patient evacuation remain unfinished. No new
whole-project completion percentage is claimed.


## Bounded latest-reading inference — 2026-10-04

The opt-in `--latest-evidence` mode replaces each source/cell's old canonical
likelihood instead of retaining every observation epoch. Per-cell sequence
watermarks reject equal and older packets; aging scales both the aggregate and
retained likelihoods before geometry transfer. This is a different statistical
model for a changing field, not exact equivalence to cumulative independent
observations. The default stays accumulation.

All 509 tests and runtime Ruff pass. Ten fresh physical runs on development seeds
7, 19 and 42 pass. In the saturated flat layouts, rejected factor attempts fall
from 20,702 and 4,065 to zero. Regional trajectories/exposure remain unchanged
across three pairs, with numeric peaks of 167,816–187,136 bytes including the
new arena-backed likelihood ledger. Flat exposure increases slightly, and arrival
time changes are mixed; this is continued-sensing evidence, not a survival win.

See [the tracked streaming report](docs/streaming-evidence.md) for the paired
metrics, semantics, replay/aging tests, source hashes and reproduction command.
The source/cell index and identity metadata remain host-side; the whole embedded
RAM and timing gates remain false. Unique source/cell keys can still saturate the
table, and topology-epoch delivery, coupled temporal/range factors, real sensors
and calibrated tenability remain open. No new completion percentage is claimed.


## Map-epoch packet safety — 2026-10-04

Coordinated physical-map rebuilds now advance a uint32 epoch, and source packets
carry that map identity. Receivers reject old or future epochs before interpreting
cell indices, and queued packets are rechecked before fusion. Regional hotspot
floors follow the same rule. Epoch zero preserves the legacy wire bytes; later
epochs use an explicit extension marker and a 16-byte header. All 19 updates still
fit the 244-byte payload, and selection charges the extra eight header bytes.

Rebuild preparation validates and allocates replacement states before consuming
queues or discarding evidence. Allocation failure leaves the current map, epoch,
evidence and pending queue intact. Source sequences continue across maps; epoch
exhaustion fails without wrapping.

All 541 tests and runtime Ruff pass. Five final physical runs pass: regional
latest-reading inference reaches 4/4 sites in seed 7 and 8/8 in seeds 19 and 42;
flat accumulation controls reach 4/4 and 8/8. Normal synchronized delivery has zero
epoch rejections, while adversarial delayed/index-reuse cases are tested directly.
The first length-only wire discriminator was rejected after a truncation ambiguity
was found; its snapshot is retained and its interrupted runs are not acceptance
evidence.

See [the tracked map-epoch report](docs/map-epochs.md) for compatibility, byte
budgets, replay tests, source archives and reproduction. This is a coordinated
host map lifecycle, not distributed map consensus or cross-session replay
protection. Whole embedded RAM, measured timing, deployed sensors, coupled
localization and physical evacuation remain unfinished.

### Independent hazard sensing

Added opt-in periodic visible scans and radio updates without geometry rebuilds.
All 550 tests and runtime Ruff pass; six physical runs reach all assigned sites.
Seed 7 improves, but flat seed 42 becomes slower, more exposed and more expensive
in radio traffic. Periodic sensing remains experimental; a route-switch safeguard
is the next evaluation. See [the sensing report](docs/hazard-sensing.md).
