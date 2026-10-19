# Current architecture

This guide describes the 0.7.0 complete-rescue workflow. The
[delivery report](../RESEARCH_DELIVERY.md) holds the measured results;
[the model proof](research/lag-rescue/method.md) defines what the verifier certifies.

## Execution and physical world

`swarmrescue rescue` supplies the complete mission preset to
[`cascade/demo.py`](../cascade/demo.py). The preset enables a compact building,
live damage, independent robot beliefs, regional planning, receiver-local range
transport and physical evacuation. Seed and robot count remain CLI options.

`cascade/` generates the structure, simulates collapse in MuJoCo, extracts a
traversability graph and drives robot bodies along planned waypoints. The live
thermal/damage model can release structural members. Geometry changes trigger
graph replacement and route checks. Collapse caches are keyed by the generated
model so they can be regenerated rather than treated as scientific evidence.

`hazard/` supplies the evolving heat/smoke environment. Simulated observations
update robot beliefs; observer-only exposure and simulator truth support
evaluation. Those observer quantities are not calibrated physiological dose.

## Private evidence and shared host responsibilities

[`ember/physical_team.py`](../ember/physical_team.py) connects per-robot state,
physical sensing and modeled lossy radio. Robots hold separate Gaussian hazard
beliefs and provenance; regional maps retain fine direct observations.
`ember/range_team.py`, `range_comms.py` and `range_gbp.py` support receiver-local
range evidence and survivor position estimates.

Planning geometry and map-epoch replacement are supplied by the host. The host
also coordinates assignment, sensor simulation and fine route refinement.
Private beliefs therefore do not imply independent map consensus or a fully
distributed custody agreement. The emulator workers receive routing models,
not complete robot sensor and inference stacks.

## Complete-rescue assignment and routing

[`ember/rescue_planning.py`](../ember/rescue_planning.py) forms bids from each
robot's own risk field and that receiver's survivor estimates. For each pair it
considers risk-weighted and physical-distance approach and exit paths. It checks
each retained combination against the modeled deadline and complete-rescue
certificate, then retains the lowest-cost feasible bid for that pair.

SciPy's rectangular linear assignment implementation chooses a maximum-cardinality
matching over feasible bids, with cost determining the selection among those
matchings. The general interface permits idle robots; the current physical
mission requires a complete one-to-one assignment. This optimizes over supplied
candidate bids, not every possible rescue route.

The physical integration broadcasts one temporally frozen private routing-cost
field across the certificate horizon. It leaves robot and survivor exposure
budgets infinite, while enforcing the available mission time. Its reported
bounds are candidate-model costs, not physical safety thresholds.

Execution subsequently uses the existing belief-based integer LMDP routing and
fine-geometry refinement. New observations and geometry can replace a route.
The initial candidate certificate consequently does not prove a bound for the
entire evolving physical trajectory.

## Contact, custody and physical completion

[`ember/evacuation.py`](../ember/evacuation.py) models survivor proxies and their
escort relationship. Nearness to a beacon is insufficient: local contact
requires proximity and an unobstructed sensor ray. Before committing custody,
the mission must find an exit in the current geometry and obtain a converged
route whose modeled deadline check has not failed. Ownership is one-to-one.

Robots then escort the survivor bodies into supported outdoor safe zones. The
mission records contact and evacuation times, final positions, ownership,
exposure proxies, routing activity and source identity. Ground-level survivor
egress is validated; these cases do not establish upstairs patient evacuation.

[`eval/rescue.py`](../eval/rescue.py) checks reports against the physical
acceptance contract. It independently checks outside coordinates, custody,
event ordering, selected firmware activity and source-archive hashes. A report's
success label alone does not satisfy acceptance.

## Native and ARM routing

`mcu/` wraps the integer routing kernels in `firmware/`. A mission can use native
C or private persistent Cortex-M4 workers under QEMU. The ARM runtime receives
each selected robot's routing model over framed UART, solves it and returns the
field/actions with matching request identity and integrity checks.

The host checks response consistency and lifecycle; sensors, localization,
assignment and physics continue on the host. The release executes actual ARM
routing instructions in an emulator. Component linked-memory figures include a
reserved stack; they do not measure whole-robot RAM, physical-board cycles or
the complete firmware stack of a deployed robot.

## Proof and independent verification

[`ember/rescue_certificate.py`](../ember/rescue_certificate.py) checks a supplied
approach and escort under finitely many hazard scenarios. The survivor waits
during robot approach. Escort segments have declared integer residence-time
intervals; one scenario remains fixed throughout the rescue. The checker finds
each actor's worst cumulative exposure over every admitted timing schedule and
rejects a horizon that omits a possible late completion.

For each fixed scenario, the model is a longest path on a route-stage/time DAG.
Temporal prefixes and a sliding maximum evaluate that graph efficiently and
retain a concrete worst-case timing witness. The integer entrypoint is exact
under its stated rate/horizon bound and unit tick duration.

[`eval/lag_rescue_validation.py`](../eval/lag_rescue_validation.py) independently
materializes the graph and calls NetworkX's `dag_longest_path`. Python integer
edge weights provide the exact comparison. Focused tests also use direct
enumeration and invalid-input/resource-bound cases. The retained protocol and
results identify the checked source.

The proof assumes the supplied scenario set and timing intervals contain the
behavior of interest. Calibration and a containment argument connecting those
inputs to the physical simulator or real people remain separate work. The
current physical runs establish implemented contact and egress behavior; the
graph study establishes the checker's declared-model properties.
