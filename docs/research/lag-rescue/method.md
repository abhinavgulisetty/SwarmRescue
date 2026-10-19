# Complete rescue with bounded follower lag: independent graph verification

Research date: 16 October 2026. This is an executable synthetic study of a
declared uncertainty model. It establishes mathematical and numerical properties
of that model; historical originality, calibrated smoke survival, physical robot
performance and fleet-level optimality are not established.

The implemented candidate is a complete-rescue checker that includes robot
approach, survivor exposure while waiting for contact, and escort with uncertain
segment residence times. One hazard scenario persists through the whole rescue.
The checker certifies every allowed timing schedule rather than a nominal speed
or only the slowest schedule. It can reject a proposed assignment before contact.

## Model and exactness argument

Let `H[s,t,v] >= 0` be tick exposure rates for finitely many fixed scenarios.
The robot waits at its starting node for the declared departure delay and follows
the specified approach with known integer edge durations. The survivor remains
at the victim node throughout this interval. Let `p` be contact time. Escort
edges, indexed `i=0,...,L-1`, hold their origin node `v_i` for an integer duration
`d_i in [a_i,b_i]`, then advance to the next node. During escort the robot waits
with the survivor and shares that tick/node occupancy. This is a grid exposure
convention, not a continuous trajectory model. Edge-interior hazards would need
their own exposure costs.

Assume valid finite intervals and

```
p + sum(b_i) <= available hazard ticks.
```

This check is essential: a truncated search could silently omit an unsafe late
schedule. Every declared duration tuple finishes by the horizon. Refuge arrival
ends exposure accounting; the study explicitly sets refuge hazard to zero.

For a fixed scenario `s`, create vertices `(i,t)` for reachable escort stage and
absolute time. Create an arc `(i,t) -> (i+1,t+d)` for every admitted duration,
with weight

```
sum(H[s,u,v_i] * dt for u in range(t,t+d)).
```

There is a bijection between duration tuples and complete source-to-refuge paths
in this directed acyclic graph. Summing arc weights equals the direct escort
dose for that tuple because the occupied tick ranges are disjoint and exhaustive.
The longest complete path therefore gives the exact worst escort dose in this
scenario. Add the actor's deterministic waiting/approach dose **in the same
scenario**, then maximize over scenarios. Maximizing scenario at each stage would
permit nature to change the source during execution and generally overestimate
the declared worst case.

The production checker's route-stage/time dynamic program represents the same
longest-path problem. Its recurrence is exact by induction: the label at `(i,t)`
is the largest dose among all prefixes reaching that state, and future exposure
and admissible durations depend only on stage, time and fixed scenario. The
optimized sliding-window implementation reduces the repeated duration search;
the independent oracle below does not call or copy that optimized recurrence.

Separate robot and survivor maxima exactly decide the universal conjunction
of their budgets: `for every (scenario,durations), both budgets hold` is equivalent
to each actor's maximum being within its budget. The actors need not attain their
worst cases at the same scenario or timing tuple. Since tick exposure is
nonnegative, terminal cumulative-dose safety also implies all cumulative prefix
budgets hold. Instantaneous thresholds, collision avoidance, changing topology,
missing sources, human refusal to follow and physical stopping failures require
additional constraints or a different model.
Actor budgets are remaining budgets at the modeled starting epoch; prior
exposure must be accounted for separately before supplying those values.

If admissible speed depends on accumulated toxicity or fatigue history, stage
and time need not be a sufficient state. Such dependence is outside this checker.
Its independent residence intervals can conservatively contain a more coupled
physical timing process; exactness is with respect to the declared interval
product, not an uncalibrated human model.

### Sliding-window compilation and resource bound

Let `P_i(t) = dt * sum(H[s,u,v_i] for u < t)` within one fixed scenario, and
let `F_i(u)` be the maximum escort prefix dose arriving at stage `i` at time
`u`. The exact recurrence can be rearranged as

```
F_(i+1)(t) = P_i(t) + max(F_i(u) - P_i(u)
                         for u in [t-b_i, t-a_i] and reachable at stage i).
```

The candidates enter and leave this interval in increasing order as `t`
increases. A monotone queue inserts each preceding state once and removes it
at most once. This removes the per-state enumeration of every dwell duration;
it uses the standard sliding-maximum technique, not new general max-plus
algebra. Every retained maximizer stores its predecessor time, providing a
concrete worst-case duration witness.

For `U` distinct escort origins and horizon `H`, temporal prefixes cost
`O(H*U)` per scenario. With independent integer duration intervals, every
arrival time between the earliest and latest is attainable, so the reachable
stage width is `W_i = 1 + sum_(j<i)(b_j-a_j)`. Queue processing costs
`O(sum_i W_i)` per scenario. Across `S` scenarios the optimization work is
`O(S*(H*U + sum_i W_i))`, in addition to validating/normalizing the supplied
`S*H*N` input and summing pre-pickup exposure. Worst-case
`sum_i W_i <= (L+1)*(H+1)`
when states fit the supplied horizon. The explicit generic graph instead
materializes each duration edge; that additional edge factor is absent in the
compiled evaluator.

Numeric scratch includes `O(H*U)` prefixes, `O(sum_i W_i)` stored witness
parents, and `O(max_i W_i)` frontiers/queues. The implementation reports the
simultaneously live DP arrays, including parents. Scenario inputs, normalization,
temporary input-validation masks and Python metadata are outside that core DP
numeric scratch figure. This is not a
whole-robot memory guarantee. Existing scenario-vector and DAG methods remain
mathematically equivalent; the contribution being tested is this complete-
rescue interface and its compiled implementation, rather than a new generic
longest-path problem.

## External method actually executed

[`eval/lag_rescue_validation.py`](../../../eval/lag_rescue_validation.py) constructs
the complete reachable graph separately for each scenario and calls the installed
**NetworkX 3.6.1** implementation of
[`dag_longest_path`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.dag.dag_longest_path.html).
It adds one to every optimization arc weight to ensure a complete source-to-sink
path is selected even for zero exposure. Every complete path has exactly `L+2`
arcs, so this constant preserves its exposure ordering in real arithmetic. Bounds
are computed by summing the original, unshifted exposure attributes along the
returned witness. Ordinary float64 arithmetic is used; embedded directed-rounding
guarantees are not asserted.

A paired check defines a **separate integer synthetic rate tensor** as
`ceil(H*256).astype(int64)`, with integer tick units and `dt=1`. It calls
`certify_integer_rescue` and builds the same external graph with Python integer
weights and path sums. The wrapper requires nonnegative integer rates, integer
budgets or positive infinity, and `horizon * maximum_rate <= 2**52`. Each
admissible cumulative dose and each finite prefix/max-plus intermediate is then
an exactly representable integer in float64, so the optimized algorithm's
arithmetic is exact on this restricted domain. The external oracle uses arbitrary
precision integers and does not depend on that floating-point argument.
All **288 paired integer cases agree exactly for both actors**, using `==`, with
zero mismatches. This defines and verifies a quantized synthetic objective; it
does not prove that a measured continuous physical hazard is contained in it.

This is an independent use of an external general-purpose implementation, not a
claim to have reproduced a particular paper's optimized routing solver.
The library citation is Hagberg, Schult and Swart,
[*Exploring Network Structure, Dynamics, and Function using NetworkX*, SciPy 2008](https://proceedings.scipy.org/articles/TCWV9851.pdf).
The installed package version, complete source hashes and pre-run parameters
are recorded in [protocol.json](protocol.json); its digest is in
[results.json](results.json).

The route objective inherits finite-scenario robust routing, as studied by
[Yu and Yang, 1998](https://doi.org/10.1016/S0305-0548(97)00085-3), and scenario-aware
path labels are established by
[Raith et al., 2018](https://onlinelibrary.wiley.com/doi/full/10.1002/net.21815).
[Wellman, Ford and Larson, UAI 1995](https://arxiv.org/abs/1302.4987) already address
time-dependent uncertain path costs; the arXiv upload date is 2013, not the
publication year. These are novelty blockers for a general robust-path or
time-expansion claim.

Follower timing is a practical requirement rather than a new observation.
[Fridolf et al., 2014](https://onlinelibrary.wiley.com/doi/full/10.1002/fam.2217)
measure smoke-related walking-speed variation. Their data motivate explicitly
modeling speed uncertainty; this study does not fit its bounds from that paper.
[Nayyar and Wagner, 2025](https://link.springer.com/article/10.1007/s12369-025-01259-w)
study robot-guided evacuee waiting and movement in physical human-subject
experiments. [Chen et al., IROS 2024](https://arxiv.org/abs/2409.19466) implement
human guidance with distance and viewpoint constraints. The present node-duration
model does not reproduce those human models or their experimental results.

## Frozen study and observed results

The deterministic protocol supplies 96 connected 4-by-4 graph missions, each with
one blocked node, one robot, one survivor and one declared absorbing refuge.
NetworkX shortest-path algorithms generate the approach and three simple escort
candidates. Three hazard scenarios per mission contain delayed pulses, decay or
growth. Departure delay, survivor waiting and slower escort are included.
Budgets and parameters are fixed in source before evaluating outcomes.
These are deterministic development instances, not a sealed statistical
confirmation set. Reproducing their exact model outcomes does not establish
containment of new physical hazards or human execution trajectories.

Across **288 candidate routes**, the kernel and external graph bounds differ by
at most **2.23e-15**. False acceptance below means accepting a route whose exact
worst case violates an actor budget within this declared model.

| Method | Accepted candidates | False acceptances | False rejections |
| --- | ---: | ---: | ---: |
| Complete rescue with robust lag | 224 | 0 | 0 |
| Fixed-timing, fixed-scenario robust routing | 255 | 31 | 0 |
| Maximum durations only | 226 | 2 | 0 |
| Pointwise scenario envelope with robust lag | 196 | 0 | 28 |

The fixed-timing comparator adapts established scenario-consistent route
evaluation by fixing escort duration to each segment's lower bound. It is an
ablation of lag uncertainty, not a claim about the best version of any published
algorithm. The pointwise envelope is a conservative ablation that lets the source
change at each tick/location. The external NetworkX comparator is exact and
agrees with the candidate checker; it is not outperformed in solution quality.

Among the supplied candidates, the declared selection rule finds a certified
route in **80 of 96 missions**. Those 80 complete missions were replayed directly
for **all 5,352 scenario-duration combinations**: start, approach, waiting,
contact, escort and refuge arrival. All completed within both budgets. The other
16 missions have no certified candidate among these three choices; the study
does not prove that no feasible route exists elsewhere in the graph.

Complete-call latency measurements in the receipt include input validation and,
for the external method, graph materialization and generic solving. They are
local implementation diagnostics, not an MCU measurement, a broad benchmark or
evidence of superiority over optimized published robust-routing solvers.

Eight focused independent tests cover exhaustive graph/replay agreement, direct
kernel/graph agreement, scenario consistency, predeparture/waiting exposure,
the horizon nonvacuity check, and an example where the longest duration misses
the worst hazard pulse, exact integer comparison with the external oracle,
and the installed CLI without repository test assets.
Reproduce from the project root:

```bash
.venv/bin/python -m pytest -q tests/test_lag_rescue_validation.py
.venv/bin/python -m eval.lag_rescue_validation --output-dir docs/research/lag-rescue
```

The defensible implemented contribution is a complete-rescue verification
interface with exact bounded-lag exposure checking and an independent external
oracle, demonstrated on reproducible synthetic graph missions. Historical
priority and a full physical swarm rescue guarantee remain separate research
questions.
