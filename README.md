# SwarmRescue — EMBER

**E**dge-native **M**essage-passing for **B**ellman-**E**gress **R**outing.

SwarmRescue is a working simulation and firmware-kernel research prototype. Version
0.6.0 installs as a standalone package and supports a complete eight-robot physical MuJoCo
rescue mission with independent receiver beliefs, lossy radio, range-factor localization,
geometry-triggered replanning and live Cortex-M4 integer routing for two robots.
Host code still handles sensors, localization, assignment and fine geometry.
The deployed workflow finds survivor beacons, establishes local contact, and moves
physical survivor proxies into outdoor safe zones. All 779 tests pass (three skips); native/ARM physical missions and independent
verification pass. Human physiology and hardware performance are not calibrated.

[Current faster routing release](docs/native-deadline-release.md), [0.5.0 deadline release](docs/deadline-release.md), [0.4.0 Bellman accelerator](docs/optimized-release.md), [rescue model and contract](docs/rescue.md), [earlier deployment instructions](docs/deployment.md), [research acceptance ledger](COMPLETION_GATES.md),
[historical evidence](FULL_PROJECT_STATUS.md), and [research proposal](PROPOSAL.md).
The September completion percentage is historical and does not describe this release.

## Focused demonstrations

Four short cases explain local contact, a debris-triggered detour, physical
evacuation, and stopping when no route exists. Each has a fixed camera, plain
captions, labelled actors, and an event log. The demos now use continuous
lookahead, bounded wheel acceleration, correct wheel orientation, and 60 fps
at normal playback speed. The local player lets you jump to
the actual decision points:

```bash
.venv/bin/python -m swarmrescue.cli cases --out outputs/cases-smooth
# Open outputs/cases-smooth/index.html in a browser.
```

The cases use authored small layouts and the runtime physics, sensing, integer
planner and escort components. They isolate individual situations; the full
mission still demonstrates the swarm. The source rescue HUD also keeps the scene
visible through completion, with contact/evacuation counts and a short explanation.
See [the demo and speaking guide](docs/demos.md) for clips and presentation notes.

The new interactive deadline case compares two routes on the same hazard map.
At 26 seconds, the hybrid reduces estimated route cost from 110 to 68 while
meeting the deadline. Pause, restart and change the deadline to explain why:

```bash
.venv/bin/python -m swarmrescue.cli deadline-demo --out outputs/deadline-demo
# Open outputs/deadline-demo/index.html in a browser.
```

## Deployed release

The checked local deployment is in `outputs/deployment/fast/`. Run its eight-robot mission:

```bash
outputs/deployment/fast/run-mission
```

The launcher runs the full search/contact/evacuation preset in an isolated installed
environment with two Cortex-M4 QEMU workers.
It records a JSON acceptance report and the installed sources used for that run.
Build/install commands and measured solver results are in [the faster release guide](docs/native-deadline-release.md).
No physical board was flashed and no remote deployment or Git push was performed.

## Research contributions and current evidence

| Proposed contribution | Implemented evidence and remaining gates |
| --- | --- |
| **C1: uncertainty-aware routing** | Independent beliefs feed uncertainty-aware route costs; native integer and emulator policies are cross-checked. Calibrated whole-system survival comparisons remain open. |
| **C2: safeguarded learned acceleration** | The int8 proposer and global safeguard have regression tests. The historical roughly 1x result fails the proposed 2x threshold; deployed int8 acceleration is not established. |
| **C3: adjoint/VoI communication** | Scheduler and procedural comparisons exist. Fresh frozen physical comparisons at matched budgets and receiver information gain remain open. |
| **C4: implicit handling of failures** | Stale precision, radio loss and provenance are modeled. Whole physical failure grids and sealed validation remain open. |

The compiled deadline safeguard is faster on average than every baseline in a
fresh frozen 216-case comparison: **5.90x versus Python A***, **1.044x versus
compiled lazy A*** (95% CI [1.015, 1.076]), and **82.68x versus native HiGHS** on
mutually proven-optimal pairs. It proves 196/198 feasible optima and returns the
same objectives as the previous hybrid in every case. The two low-memory
fallbacks are retained and documented. Per-input and all-known-method superiority
is not claimed. See [complete measurements and limits](docs/native-deadline-release.md).

The production planner now combines exact shortest-path bounds, ordered updates,
and integer self-loop solving. A frozen 144-model comparison measures **3.47x**
faster full routing solves than compiled Jacobi and **2.42x** than Gauss–Seidel,
with identical converged fields. Some small and dense cases are slower than
Gauss–Seidel; this is an average gain on the declared solver distribution.
The native and live-ARM rescue runs retain the prior physical outcomes while
reducing maximum reported routing passes from 789 to 390. See the
[methods, confidence intervals and complete measurements](docs/solver-research.md).

The building and rubble come from procedural MuJoCo collapse simulation; live
thermal damage releases structural welds. Regional distributed missions rerasterize
geometry and preserve compatible factors across map epochs. Per-robot numeric
arenas and the live routing service have component memory accounting; whole-robot
RAM and hardware cycle calibration are not complete. The deployed ARM target is
QEMU MPS2 Cortex-M4, with persistent UART routing requests, rather than a full robot
firmware image. ESP32 and physical-board deployment are not implemented here.

## Running from source

Everything below runs from the repo root against the project venv (Python 3.12).

**Look at a generated building.** Writes four views plus a contact sheet to `outputs/views/`:

```bash
.venv/bin/python -m cascade.view --seed 7                     # intact
.venv/bin/python -m cascade.view --seed 7 --collapsed         # after the blast
.venv/bin/python -m cascade.view --seed 7 --collapsed --cutaway
MUJOCO_GL=glfw .venv/bin/python -m cascade.view --seed 7 --interactive
```

Offscreen commands select EGL automatically on Linux, so they also work without an X11/Wayland
display in CI or over SSH. An explicit `MUJOCO_GL` value is never replaced. The interactive viewer
needs a desktop OpenGL context, requested with `MUJOCO_GL=glfw` as shown above.

**Run the swarm traversal demo.** Writes an annotated `.mp4` to `outputs/demo/`:

```bash
.venv/bin/python -m cascade.demo --seed 7 --robots 4 --speed 4 --fps 24
.venv/bin/python -m cascade.demo --seed 7 --robots 4 --no-render   # numbers only
.venv/bin/python -m cascade.demo --seed 7 --robots 4 --live-disaster
.venv/bin/python -m cascade.demo --seed 7 --robots 4 --compact --live-disaster
```

The checked seed-7 full traversal is `outputs/demo/seed7_swarm.mp4`: 33.6 seconds
at 24 fps, containing the complete 125.9 simulated seconds. All four robots enter,
reach distinct destinations on storeys 1–4, and finish with 100% route completion.
Goal selection spends up to ten metres searching beyond each storey's nearest safe
landing; this keeps the mission substantial without sending a robot through a much
longer secondary stair route merely to maximize path length.

`--compact` is the visual disaster preset: a three-storey 14×10 m structure with
4–6 large bays and roughly 250 members rather than 542. The physics and acceptance
checks are unchanged, but floor loss, facade failure, falling beams, rubble arrest,
thermal damage, rerouting, and robot motion occupy enough of the frame to read
clearly. Compact artifacts are written below `outputs/demo/compact/` so they never
overwrite the research-scale run. It is intentionally paired with
`--live-disaster`; the checked seed-7 clip is
`outputs/demo/compact/seed7_end_to_end.mp4`. Its primary collapse frees 30 large
members, the live thermal event fails five more connected members, and all four
robots finish at distinct interior sites 5.0--10.6 m from their starts. Survivor
sites are sampled from reachable deep-interior cells and withheld from the robots.
After deployment, noisy BLE observations are fused into four position estimates
(0.58 m worst error in the checked run), a centralized deterministic matching gives
each robot a target, and four converged belief routes are installed at runtime.
When thermal damage releases structural welds, every active route is checked against
the updated hazard field. Three exposed routes are replaced and one nearly completed
route is retained. The initial shortest-path-to-EMBER comparison lowers the robots'
aggregate search objective (`mu - 2 sigma`) from 32.106 to 11.321 for this illustrative
seed. This is not a survival-risk result; the clip separately evaluates both routes with
one simulator-truth heat/smoke proxy.

The compact clip is a deliberately paced, silent-caption integration cut: 44 seconds,
30 fps, and exactly 1920x1080. It opens with the Phase 1 integration question and labels
seed 7 as illustrative rather than aggregate evidence. A 35% evidence camera shows the
tracked robot and the failing bay; the 65% tactical pane is explicitly labelled
`EVALUATION VIEW`. Simulator hazard truth is evaluation-only, while the complete
traversability graph is supplied to every route solve in this centralized reference.

The evaluation pane exposes the hidden transformations. BLE range rings collapse into a
fused estimate and 95% confidence ellipse, held-out truth is marked only for evaluation,
and the tracked robot local `mu + beta sigma` belief is drawn as an objective heatmap. The
shortest-path counterfactual remains visible as dashed red while the accepted EMBER route
is solid green and the physical trail keeps the robot livery. Failed structural members
are tinted red while the caption and metric strip show temperature, accumulated exposure,
threshold crossing, and weld release. Robot halos link the tracked subject across both
panes. The large metric strip reports localization error, a common observer-only heat/smoke
proxy, classical Jacobi sweeps, route decisions, and desktop reference latency.

The held results card reports 4/4 localized target sites reached, five thermally failed
structural members, three routes replaced and one retained, 0.58 m worst localization
error, and the change under one pre-ISO observer proxy. It also states that this is a
centralized, classical-Jacobi, desktop-timed illustrative seed—not evidence for FED
survival, distributed GBP/radio, NCA acceleration, VoI, implicit failure handling, or MCU timing. The legacy multi-storey traversal retains its
dollhouse/storey HUD. Encoder padding is now
limited to odd dimensions; the compact preset is emitted without padding at 1920x1080.

`--live-disaster` keeps the cached rubble dynamic, ignites the live heat/smoke field, and lets
accumulated thermal exposure release real structural welds during a bounded forty-eight-second mission.
The secondary-collapse clock starts after twelve seconds, once deployment has cleared the entrance,
so the clip demonstrates hazard-triggered route reevaluation after structural failure rather
than crushing robots at spawn. Fallen members are not yet rerasterized into the planning graph.
It is intentionally separate from the fast mission run: simulating more than 500 live
structural bodies for a full traversal would take tens of minutes. The evidence camera
cuts to the failing bay while the evaluation pane emphasizes the tracked robot and its
rejected, planned, and driven paths. The disaster run passes only when heat fails new
members and every robot
finishes at least 1.5 m inside the facade. In the checked seed-7 run, noisy local thermal
scans independently update each robot Gaussian cost belief and install four converged
initial LMDP routes in the physical waypoint controllers. The later collapse causes three
additional replans and one evidence-based retain decision. The metric strip exposes BLE
localization, the observer proxy, solver convergence, thermal damage, and route decisions,
so the
clip demonstrates the integration without implying that the tactical scalar GBP/radio
reference is already wired into the full multi-storey physics run.

The compact video uses a four-step narrative: deploy for BLE ranging; expose fused
localization and uncertainty; show the causal structural failure; then freeze long enough
to compare rejected and accepted routes before physical execution continues.

The first run for a seed simulates the collapse (~2.5 min) and caches it under `cascade/cache/`;
later runs of that seed reuse it. Editing anything that changes the structure invalidates the cache
by model hash, so the collapse re-runs.

**Run the fast rescue-loop demo.** Four robots localize a survivor from noisy BLE RSSI, the real
heat/smoke solver ignites a blocked shortcut, sparse sensors broadcast canonical Gaussian packets
through a lossy ESP-NOW-style medium, asynchronous scalar GBP propagates the evidence, and both
risk policies are recomputed by the Jacobi LMDP solver. The same run evaluates centralized-shortest
B1 and flooding-Dijkstra B2, and writes a seeded JSON manifest. It can also write an eight-second
tactical MP4 whose HUD values all come from those computations:

```bash
.venv/bin/python -m ember.demo
.venv/bin/python -m ember.demo --no-render   # acceptance check only
```

**Run the 20-world parallel replay.** This builds the paired development worlds used
by the ablation grid and renders a 4×5 command-center wall. Every tile shows a real
multi-storey graph, evolving hazard costs, EMBER/VoI and flooding routes, radio pulses,
and three synchronized survivor markers. The footer reports the aggregate paired result.

```bash
.venv/bin/python -m eval.world_wall
```

The default outputs are `outputs/demo/20_world_command_center.mp4` (12 seconds,
1920×1080 at 24 fps) and `outputs/demo/20_world_command_center.png`.

```bash
.venv/bin/python -m pytest -q          # the test suite
```

**Run the paired tactical evaluation.** This runs B1, B2, and EMBER on the
same truth and noise stream for every seed, writes a full JSON report plus tidy
CSV rows, and bootstraps paired confidence intervals. Seeds 10000--10049 are
sealed by default. The report records unavailable paper metrics as unavailable;
the present heat/smoke dose is not mislabeled as ISO 13571 survival.
B2 now constructs its own last-received mean map directly from delivered
packets; it does not reuse EMBER's GBP belief or uncertainty premium. Localization
coverage uses the covariance ellipse's Mahalanobis statistic, and provenance includes
SHA-256 hashes for both the dirty source diff and `uv.lock`.
Both methods still share the same flooded deliveries and byte count; the protocol marks
`communication_policy_varies_by_method: false`, and method-specific communication
efficiency remains unavailable until a separate EMBER scheduler exists.

```bash
.venv/bin/python -m eval                         # development seeds 0--199
.venv/bin/python -m eval --seeds 0:19            # quick local run
.venv/bin/python -m eval --seeds 0:19 \
  --source-bundle outputs/eval/source.tar.gz       # reconstruct dirty source
.venv/bin/python -m eval --seeds 10000:10049 \
  --allow-held-out                               # only after freezing thresholds
```

The generated claim gate remains false unless EMBER's paired route-dose
confidence interval is strictly below B2's, and remains false until B3 exists.
World geometry and ignition do not yet vary with seed, and that limitation is
machine-readable in the report. The tactical mission remains a smoke test
rather than paper evidence.

**Run procedural world/noise evaluation.** This is the evidence-oriented successor to
the tactical smoke test. Its paired unit is `(world_seed, noise_seed)`; the default
grid is 40 Cascade worlds × 3 noise replicates with three survivors and eight robots.

```bash
.venv/bin/python -m eval.procedural --world-seeds 0:3 --noise-seeds 0:1
.venv/bin/python -m eval.procedural                  # worlds 0--39 × noise 0--2
.venv/bin/python -m eval.heldout                   # hash-locked final protocol
```

Floorplan, connectivity-preserving blockage, ignition, survivor placement, and robot
placement use stable world streams. Sensor, radio, BLE, and method-internal randomness
use separate noise streams. Storey-aware 3-D BLE estimates drive exact minimum-distance
robot assignments; after reaching an estimate, robots follow a deterministic blind graph
search rather than a truth-informed shortcut. Contact delay, survivor waiting exposure,
finite-horizon evacuation, completion, and clairvoyant routing regret are all reported.
Hazard evolution is shared across methods and confidence intervals resample world-level
noise means.

Schema 6 runs `b3_distributed_consensus` and gives EMBER a distinct communication path:
per-robot canonical evidence is merged by Metropolis consensus for B3, while EMBER floods
only adjoint-weighted KL updates admitted by a 64-byte-per-robot token bucket. Consensus
round bytes are not yet charged to B3. Blockage remains connectivity-preserving topological
damage rather than MuJoCo-derived collapse geometry, and exposure remains a pre-ISO
heat/smoke proxy. These limitations are machine-readable and keep the paper claim gate
closed.

## Status

The proposal-weighted whole-project audit is **76%**, clearing the 75% checkpoint. Phase 2 includes the
2,465-byte safeguarded int8 NCA, mission-level adjoint/VoI radio, stale-message aging,
distributed B3, and a 300-scenario resilience grid spanning 20 development worlds, five
failure levels, and 8/16/24 robots. Procedural VoI lowers mean bytes from 8,184 to 3,560
(56.5%) while the exposure confidence interval still crosses zero. The NCA converges but
improves sweep count by only 1.002x in its component benchmark; the paired mission
ablation gives 1.016x, so the speed gate remains false. Phase 1 lambda selection chooses
0.25 on 20 development worlds and the policy now uses it. Phase 3 adds the five required
paired ablations and a hash-locked run over all 50 held-out worlds. Held-out
EMBER-minus-B2 exposure is +0.083 with a 95% interval of [-4.585, 6.482], so superiority
is not established; held-out radio traffic is 55.6% lower. The host-compiled C99 Q16
kernel remains bit-exact with Python, but its route-level development gate fails because
Q16.16 desirability underflows before route extraction. Tier-A profiling reports 17,200
bytes maximum arena high-water, 2.74% maximum modeled cycle utilization, and no modeled
deadline misses. This is not QEMU/Renode evidence. See
[`FULL_PROJECT_STATUS.md`](FULL_PROJECT_STATUS.md) for the weighted audit and remaining
project gates.

Phase 0 is implemented: floor-plan grammar, structural model, collapse by simulation,
dual-representation extraction, and a swarm traversal demo. Phase 1 is now underway with a
deterministic 3-D ground-truth hazard oracle: semantic room fuel loads, Arrhenius combustion,
temperature/smoke transport, and connector-driven stack flow run through matching NumPy and lazy
Taichi CPU backends. Its transport stencil is checked against the analytic periodic 1-D
advection–diffusion solution. A float64 reference LMDP now projects those fields onto the existing
traversability graph, solves the strictly contractive desirability fixed point by plain Jacobi,
and exposes the corresponding local stochastic policy. A first robot-side slice adds noisy,
quantized thermal/gas observations, Gaussian information updates, stale-belief variance growth,
and the C1 `mu + beta*sigma` search/escort cost split. A live tactical demo now shows real hazard
ignition, propagation, sensing, and route relaxation. Thermal exposure also releases the affected
member's real MuJoCo weld constraints exactly once. That bridge now runs inside the bounded
physical-disaster mission: four robots deploy around the live 500-body structure while the real
10 Hz heat/smoke field accumulates exposure and releases structural welds. Every robot now converts
its own noisy local scan into a Gaussian posterior, solves an LMDP, and physically drives the
accepted detour. The run accepts only when all four routes lower aggregate posterior cost and
converge, all hidden survivors are localized within the error gate, thermal failure occurs, and
every robot moves, enters, and finishes. The Tier-A substrate
supplies saturating Q16.16,
fixed-capacity arena allocation, analytical cycle accounting, and deadline fallback. The fast
mission now also supplies an asynchronous scalar GBP reference, Q16 canonical packets,
TTL/deduplicated flooding over a slotted lossy radio, B1/B2 measurements, and seeded run manifests.
Full coupled octree/range-factor GBP remains next. Phase 2 progressive collapse is already
integrated ahead of the rest of that phase. See the 14-week plan and risk register in
[`PROPOSAL.md`](PROPOSAL.md) §7.

### Classical MVP checkpoint — 11/14 slices (79%)

Complete: procedural world, physical collapse/cache, extraction and swarm traversal, dynamic
hazards, the first sensing/belief slice, classical LMDP routing, and the Tier-A MCU substrate.
The fast tactical harness now additionally has experiment manifests, flooding communications, and
B1/B2 baselines. Remaining: the full sensor suite, coupled hazard/range-factor GBP in the physical
swarm, and the eight-robot unseen-seed MVP gate. This is 79% of the classical
MVP engineering checklist, not 79% of the full Phase 2–3 research program. The procedural
evaluation now closes the multi-world, paired-noise, search-to-contact, finite-horizon
trajectory evidence path, but it does not count ISO tenability, B3 consensus traffic,
or physics-collapse coupling as complete.

The first hazard slice blocks vertical transport at slabs except through aligned connectors, but
still treats each storey as one horizontally connected air volume. Partition/door permeability
and mass-conserving horizontal return flow remain before full-world sensor integration.
Combustion constants are stable but not fire-calibrated, and the router's positive heat/smoke
cost is explicitly an oracle proxy—not yet the proposal's ISO 13571 tenability model.

Known limitations of the Phase 0 demo, stated plainly because they bound what it evidences:

- **Robots do not collide with each other.** Mutual deconfliction is a routing-policy problem and
  belongs with the routing policy in Phase 1. With contacts on, a swarm sharing one doorway simply
  queues into it and the run measures the queue rather than the ruin.
- **Settled rubble is baked static.** It cannot shift, tip, or crush a robot, which makes the
  traversal tractable but means this demo says nothing about survival under settling debris.
- **The live sensing model is deliberately local.** All four robots in the disaster clip now
  cooperatively localize procedural survivors from noisy BLE and drive belief-to-LMDP routes, but
  hazard sensing still uses independent radial thermal scans. Full sensor FOV/raycasting and
  physical-swarm communication-aware replanning remain open; the fast tactical harness has scalar
  hazard GBP and lossy flooding, while the standard multi-storey traversal follows offline routes.
- **The collapse is not dynamically calibrated.** A previous 177 MJ peak was almost entirely
  pulverized, non-colliding bodies free-falling below the world; their weight is now cancelled and
  seed 7 reports 1.50 MJ. The corrected run is still time-limited with a small residual kinetic
  energy rather than passing the quiescence gate, so the settled geometry is reproducible but no
  claim is made about matching a real blast's time history.

### Evaluation fairness correction (2026-10-04)

Procedural report schema 7 removes evaluation-only survivor locations from the
VoI adjoint source weights. Communication now uses BLE estimates snapped to graph
nodes, preserving weight when multiple estimates snap to one node. Flooding and
VoI both start from the `radio_loss` substream. Draws remain sequential and depend
on packet order: this is shared-seed coupling, not eventwise paired link loss.
The report records this limitation explicitly.

Run the corrected development comparison with:

```bash
.venv/bin/python -m eval.procedural --world-seeds 0:199 --noise-seeds 0:2 \
  --out outputs/eval/development-fairness-v7.json \
  --csv outputs/eval/development-fairness-v7.csv \
  --source-bundle outputs/eval/development-fairness-v7-source.tar.gz
```

Seeds 10000–10049 were already opened in the September experiment. The protocol
hash locks configuration, not implementation. Runs after this correction are
historical replays, not fresh confirmatory evidence. `python -m eval.heldout` now
labels this explicitly and defaults to `heldout-replay.json` / `heldout-replay.csv`
to preserve the old `heldout-final` artifacts. Earlier numerical results do not
validate the corrected scheduler. ISO tenability, physical distributed inference,
measured embedded timing, and trained NCA acceleration remain separate unfinished
parts of the roadmap.

### Distributed receiver state and integer routing (2026-10-04)

Procedural schema 9 now gives each robot its own GBP buffers, packet queue,
bounded source/sequence/cell provenance set, tick budget, sensor epoch counter,
and log-Q16 routing field. Robot ticks are interleaved. Escort routes use the
assigned robot's delivered evidence rather than a shared robot-0 map. VoI limits
broadcasts while retaining each robot's complete local observations.

Radio loss is keyed by seed, absolute slot, transmitter and receiver. Matching
physical radio events share a loss draw regardless of payload or unrelated
previous transmissions; policies with different schedules encounter different
events within the same noise field. Flood slots advance monotonically. This
supersedes schema 7's sequential radio-loss limitation.

```bash
.venv/bin/python -m eval.procedural --world-seeds 0:19 --noise-seeds 0:2 \
  --out outputs/eval/distributed-log-q16.json \
  --csv outputs/eval/distributed-log-q16.csv
.venv/bin/python -m eval.ablations --world-seeds 0:19 \
  --out outputs/eval/ablations-log-q16.json
```

The ablation schema 2 keeps legacy linear-Q16 failures visible and separately
measures log-Q16 route agreement/completion, value error, and persistent arena
storage. `ember/physical.py` also accepts `robot_state=` for persistent inference
from delivered peer packets and integer physical replanning. Supply
`scene_model=` and `scene_data=` to filter scans by current structural box poses.
`ember/geometry_sensing.py:radio_wall_counts` supplies live wall intersections to
`RadioMedium`. Box raycasts ignore cinematic opacity and track moved/disabled
geometry; non-box geometry is outside their model.

At this earlier API milestone, the rendered loop had not yet been integrated.
The distributed physical integration below supersedes that limitation for the
ground-level demo. Coupled pose/range/PDE factors, the full sensor suite, ISO
tenability, learned int8 acceleration, and emulator timing remain unfinished. Arena accounting covers retained numerical state; Python metadata,
GBP scratch allocations and analytical cycle charges do not establish measured
whole-robot RAM/deadline gates.

### Rescue-clock safeguard and algorithm comparison (2026-10-04)

Procedural schema 10 defaults to `deadline_lmdp`: preserve the integer LMDP
proposal if its discrete edge traversal times fit the remaining evacuation
window; otherwise run bounded integer Pareto routing. Label-cap overflow falls
back to the fastest route. Under the static graph, common contact times and
fixed-speed clock used here, a feasible legacy route is preserved and a feasible
alternative is found whenever one exists. This is a modeled completion guarantee,
not a physical traversal or ISO tenability guarantee.

`firmware/deadline.c` implements the certificate and fallback with caller-provided
buffers, no floating point and no dynamic allocation. Randomized tests compare
routes, objectives, feasibility and resource fallback bit for bit against Python.
Host C conformance does not establish emulator or hardware cycle timing.

The algorithm comparison independently varies communication (`adjoint_kl`,
`kl_only`, `occupancy_coverage`), inference (`gbp`, `canonical`) and routing
(`lmdp`, `risk_astar`, `deadline_pareto`, `deadline_lmdp`). Occupancy coverage uses
exit-conditioned visitation and greedy spatial coverage to reduce packet
redundancy. The discovery winner did not beat legacy EMBER on the first disjoint
set, so it was not promoted. The deadline safeguard was promoted because it
removed avoidable timeouts while preserving feasible legacy routes.

On worlds 0–19 × noise seeds 0–2, safeguarded rescue completion rose from 95.0%
to 98.3%, matching B2 with unchanged sparse-radio traffic. Worlds 60–99 × three
noise seeds showed equal 97.78% completion for B2 and the safeguard, and about
54% fewer radio bytes. The raw exposure difference there was -1.024, with 95%
world-cluster interval [-5.656, 2.420]; exposure superiority is not established on
that set. These are exploratory development results, not sealed confirmation.

`mean_unpenalized_exposure_proxy` reports actual modeled hazard exposure separately
from `mean_dynamic_exposure_proxy`, which includes the disclosed timeout penalty.
Rescue completion is the first selection criterion. Pareto labels and deadline
interventions are recorded separately from Bellman sweeps.

```bash
.venv/bin/python -m eval.innovation --world-seeds 0:199 --noise-seeds 0:2 \
  --algorithms adjoint_kl/gbp/lmdp,adjoint_kl/gbp/deadline_lmdp \
  --out outputs/eval/innovation-development-200.json \
  --source-bundle outputs/eval/innovation-development-200-source.tar.gz
.venv/bin/python -m eval.procedural --world-seeds 0:19 --noise-seeds 0:2
.venv/bin/python -m eval.procedural --escort-policy lmdp  # legacy reproduction
```

The historical held-out replay explicitly pins legacy routing, so changing the
new development default does not silently change its algorithm. Full physical
integration, calibrated tenability, coupled range/pose/time factors, trained NCA
and emulator measurements remain open; this milestone does not complete the full
research proposal.

The full revised grid is now available in
[the 200-world report](outputs/eval/innovation-development-200-summary.md):
1,730/1,800 modeled rescues, matching B2/B3 and recovering 15 rescues over legacy
EMBER with zero scenario regressions. EMBER uses 55.3% fewer radio bytes and has
lower mean raw exposure, while rescue time averages about 4.2 seconds longer than
the baselines. The report includes paired intervals and matching source provenance.
All 449 tests and the tactical acceptance check pass. This is development evidence;
physical integration, calibrated tenability and emulator validation remain open.


### Distributed physical demo integration (2026-10-04)

`cascade.demo --live-disaster --distributed` now runs persistent independent robot
maps, visibility-filtered noisy scans, geometry-aware radio delivery, budgeted GBP
and native host integer routing. After the first thermal-failure episode, it
rebuilds traversability from current structural poses, transfers direct local
factors on surviving cells and replans unfinished robots. BLE localization still
uses the centralized solver.

```bash
.venv/bin/python -m cascade.demo --seed 7 --robots 4 --compact \
  --live-disaster --distributed --out outputs/demo/distributed-review \
  --fps 12 --speed 4 --width 960 --height 865 \
  --report outputs/demo/distributed-review/seed7.json
```

Collision-free exterior entry, a robot envelope covering the mast and wheel sweep,
physical-distance deadline checks and precise target stopping resolve failures
found during development. The final disclosed missions reach 4/4 target sites
(seed 7) and 8/8 (seed 42). All 461 tests and runtime Ruff pass. The rendered clip
was decoded and visually reviewed; reports have matching source archives.

See [the integration report](outputs/eval/physical-integration-summary.md) and
[the reviewed clip](outputs/demo/distributed-final-reviewed/compact/seed7_end_to_end.mp4).
These are ground-level development missions, not a paired superiority experiment
or physical patient evacuation. Planning is limited to cells at most one metre
high; only the first failure episode triggers a geometry rebuild.

The retained numeric arenas use 1.23–1.36 MB, exceeding the 200 KB embedded target
before Python metadata and temporary allocations. Analytical cycle accounting is
not calibrated hardware timing. Continuous geometry/PDE coupling, coupled
localization factors, full sensors, calibrated tenability, trained acceleration
and emulator measurements remain open. The full research project is unfinished.


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

Experimental live sensing is available with `--sensing-interval-s 2` in
`--distributed` missions. It updates hazard beliefs independently of rubble map
refreshes. See [the paired sensing results](docs/hazard-sensing.md), including the
seed 42 regression; the default remains disabled.


## Coupled pose/range localization checkpoint

An opt-in `--range-gbp` localizer now couples uncertain robot poses and beacon
positions, with bounded latest-reading slots, replay protection, log-range RSSI
likelihoods, and explicit nonlinear convergence checks. All 586 tests and runtime
Ruff pass. All 600 paired development cases converge, with mean error falling
from 0.379 m to 0.303 m and 95% coverage rising from 91.7% to 94.8%. Six physical
comparisons reach all sites, but arrival/exposure gains are mixed, so the old
localizer remains the default. Range-factor transport, pose/hazard coupling,
real sensor calibration, whole embedded RAM/timing and evacuation remain open.
See [the localization report](docs/range-localization.md).

## Freestanding firmware emulator checkpoint

Cortex-M4 QEMU replay now passes 13,610 integer-reference checks across 202
routing and 600 deadline cases. The replay reserves 32,176 bytes including its
stack; this is a kernel/harness measurement, not whole-robot RAM. All 592 tests
and runtime Ruff pass. See [firmware emulation](docs/firmware-emulation.md) and
[the current completion gates](COMPLETION_GATES.md) for what remains unfinished.


## Receiver-local range transport — 2026-10-06

`--range-gbp --range-radio` now sends typed RSSI packets through the physical
radio and builds independent receiver factor tables. Replay/epoch checks and
bounded retries preserve source identity; assignments use each robot's own
estimate, with host-coordinated matching. Six physical runs across three pairs reach every site
with unchanged trajectories/exposure; localization airtime is now included.
The 120-case loss grid localizes every team through 75% packet error and
reports full partitions unavailable, without a shared-data fallback. High-loss
error still rises. The full suite has 650 passes and one optional ARM emulator
skip; runtime Ruff passes. Known pose priors, bid transport, temporal pose/hazard
coupling, whole firmware/resources and evacuation remain open. See the
[range transport report](docs/range-transport.md).


### Source-coverage retry tradeoff — 2026-10-06

`--range-target-sources 6` can improve high-loss localization by collecting more
independent source factors within the same retry cap. In paired twenty-world
cases at 75% packet error, mean error falls 25.7%, at 2.35× airtime; all teams
still localize, while two explicitly report the soft coverage target unmet.
Two additional eight-robot physical runs pass with identical behavior and no
extra airtime where complete source coverage is already present. The full suite
has 658 passes and one optional ARM skip; runtime Ruff passes. Defaults remain
unchanged. See [the measured retry tradeoff](docs/range-transport.md).
