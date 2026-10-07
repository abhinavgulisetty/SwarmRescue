# Deadline-constrained search research

This extension optimizes the declared static estimated objective (length plus
arrival-node exposure estimate) under a hard, discrete traversal deadline.
It does not read future hazard truth or establish human survival superiority.
The completed 0.4.0 Bellman evaluation remains unchanged.

## Primary research consulted

- [A Fast and Simple Algorithm for the Resource Constrained Shortest Path Problem, ESA 2025](https://drops.dagstuhl.de/storage/00lipics/lipics-vol351-esa2025/LIPIcs.ESA.2025.97/LIPIcs.ESA.2025.97.pdf)
  motivates cost/time heuristics, incumbent pruning and lazy dominance. Our
  lazy A* baseline specializes these principles to positive costs and one time
  resource, where a scalar best closed arrival time suffices. It is a compatible
  implementation, not the authors' full multi-resource/negative-weight solver.
- [Resource Constrained Pathfinding with Enhanced Bidirectional A* Search, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34892/37047)
  is a stronger modern comparator to naive label setting. Its full bidirectional
  algorithm is not claimed as implemented or beaten by this extension.
- [APULSE, November 2025 preprint](https://arxiv.org/html/2511.18236v1)
  combines heuristic and incumbent pruning with approximate time buckets. Our
  exact candidate retains individual arrival times and never discards a label
  solely because it shares a bucket with another label.
- [Topological Value Iteration](https://www.cs.uky.edu/~goldsmit/papers/tvi.pdf)
  is relevant to future component-wise Bellman comparisons. It solves a
  different problem from a resource-constrained route and is not a timing
  comparator in this experiment.
- [SciPy/HiGHS MILP documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
  supports an independent, native compiled mixed-integer baseline on exactly
  the same route objective and time constraint.

## Implemented candidate and baselines

`deadline_route(..., strategy=...)` preserves the legacy `pareto` default and
its firmware conformance. Additional choices are `astar`, `lazy_astar`, `dual_astar`, `bounded_dp`
and `hybrid_exact` and `window_dp`. The production safeguard now chooses `hybrid_exact` when
a proposed route is infeasible; it preserves feasible proposals. The A* methods compute exact cheapest and fastest completion
tables, prune infeasible continuations, and retain improved feasible incumbents
instead of automatically reverting to the fastest route on label exhaustion.

The dual candidate also computes shortest-path tables with integer edge weight
`q_cost + mu * traversal_steps`. For remaining time `B`,
`d_mu(node) - mu * B` is a lower bound on any feasible completion cost for every
nonnegative integer multiplier `mu`. Taking the maximum over these bounds and
the zero-multiplier cost bound remains admissible and consistent in the
time-expanded state space. Multipliers bracket the resource tradeoff and then
refine it; no floating-point rounding is used in a pruning certificate.

A feasible weighted shortest path supplies an incumbent. If its integer cost
equals the lower bound, optimality is certified without storing search labels.
Otherwise exact two-dimensional dominance and A* continue. Reaching the label
limit returns the best known feasible path and a valid objective-gap bound;
it does not claim optimality. Paths exceeding label integer range are handled
conservatively. Certificate sums use int64 and overflow-prone multipliers are
skipped. Positive weights make shortest-path witness chains acyclic.

Tables and labels use caller-owned numeric arena storage and restore its
original prefix. If certificate tables cannot fit, the legacy bounded solver
is used. Python graph objects, dictionaries and heaps remain host memory;
numeric arena accounting is not a whole-MCU RAM claim.

## Validation and experimental boundaries

Six exact methods match an independent time-expanded dynamic-programming
oracle on 200 randomized problems each (1,200 comparisons). Additional capped searches verify
feasibility, route costs, lower bounds, honest optimality flags, arena prefix
preservation and saturated label-range certificates. Legacy firmware behavior
continues to match the original Pareto solver.

Development and fresh frozen evaluation will compare full-call latency,
proven-optimal solution counts, actual estimated-objective gaps, generated labels
and numeric arena peaks. Failed, infeasible and resource-capped cases must stay
in the results. Timing ratios for exact solving compare successful exact pairs;
a quick suboptimal fallback is not timed as an exact solver win. Author-paper
speedup figures are not project measurements.


## Hybrid architecture

The candidate first checks exact fastest and cheapest path certificates. If the
cheapest route misses the deadline, a time-state table with `(deadline+1)*N`
int64 entries supplies an exact Bellman dynamic program when it fits in the
remaining numeric arena. Vectorized scatter-min updates avoid Python label
allocation and support non-unit traversal times. Positive costs make the
recovered witness acyclic. Dead certificate storage is recycled without changing
the caller's arena prefix. When the table cannot fit, dual-bound A* continues.
The separate `bounded_dp` comparator uses the same numeric ceiling and returns
an honest feasible fallback if its table cannot fit.

The initial small-map development experiment had 60 matched cases: four seeds, five families and
three label capacities. Hybrid and bounded DP prove all 45 feasible cases;
legacy Pareto proves 23 and the A* variants prove 43. The legacy mean estimated
objective gap is 6.10%; hybrid has zero gap. Hybrid's full-call speed is only
about 1.23x legacy on mutually exact cases and approximately equal to the other
specialized searches. Native HiGHS MILP is 13.36x slower on these small models.
These figures selected the implementation; they are not held-out results.

## Published implementation comparator

The independent comparator uses the authors' actual NWRCA* code at
[revision 4098543d0c5d78afc977017b5abf6054671a8c43](https://bitbucket.org/s-ahmadi/multiobj/commits/4098543d0c5d78afc977017b5abf6054671a8c43).
AGPL upstream source and binaries are retained only in ignored research outputs,
and are not bundled into this project. Its build uses DIM=2, int32 costs,
GCC -O3 -DPATH and the recommended integer bucket queue without tie-breaking.
Only the explicit resource-budget adapter changes problem input; the search is
unchanged. An unused three-objective dispatcher branch is omitted for DIM=2.

Our external adapter adds a positive one-unit cost/time terminal arc from each
exit, raises the deadline by one and subtracts one from the returned objective.
This exactly preserves the any-exit feasible routes and integer optimum. The
reference returns correct costs/infeasibility on all 20 development maps.
Optional path printing segfaults and the documented hybrid queue raises SIGFPE
on the development tradeoff input, including a double-cost build. Those
configurations are not counted as timing wins. The successful bucket build
reports initialization plus search time; file parsing and IPC are excluded.
This scope is narrower than our full API-call timing. Native reference memory
is unbounded and cannot be advertised as a matched 200KiB whole-MCU comparison.
The adapter records actual exit status and preserves failed cases.

## Exact feasible-time corridor

On larger maps, the full time-state rectangle may exceed the numeric arena.
The candidate also tries a packed exact table before falling back to dual A*.
Let `a(v)` be the fastest time from the start and `b(v)` the fastest time to any
exit. A state with remaining time `r` on a feasible path must satisfy
`b(v) <= r <= deadline - a(v)`. Only this interval is stored for each node;
unreachable nodes have an empty interval. No approximate buckets or cost labels
are discarded. Any continuation from a reachable prefix stays inside these
intervals by the shortest-path triangle inequality, so this restriction
preserves every feasible optimal route. Positive traversal times allow exact
increasing-resource Bellman updates and positive costs prevent witness cycles.

Three int64 index arrays plus the packed int64 cost entries count against the
same caller arena. Transient host arrays are outside whole-MCU accounting.
`window_dp` is an explicit independent comparator using the same corridor
technique, so a candidate win is not attributed to comparison against only an
uncompressed DP. The large-grid family uses opposite-wall endpoints and a
central hazard band: avoiding that band requires extra travel time. Its 1,024–
1,600 nodes make the ordinary time-state table exceed the 200KiB numeric limit.

Reproduce the external reference without bundling its AGPL source:

```bash
python -m eval.build_constrained_reference --out outputs/reference-nwrca
python -m eval.reference_constrained --binary outputs/reference-nwrca/src/bin/roadhog --split development --output outputs/reference-development.json
```

A local pinned upstream checkout can be supplied with `--repository` to avoid a
network clone. Exact source and binary identities are recorded in the build
receipt. The reference compiler and library flags follow the upstream Makefile.


The final development suite contains 72 cases, including the nontrivial large
maps. Hybrid and window DP prove all 54 feasible cases with zero objective gap;
legacy Pareto proves 26 (mean gap 9.26%), ordinary A* 49, lazy A* 48,
dual A* 50 and full-table bounded DP 48. The hybrid is approximately equal to
ordinary/lazy A* in full-call latency on mutually proven-optimal pairs, 1.07x
faster than dual A*, and 14.84x faster than native HiGHS. The development timing
and its seed-bootstrap intervals guide selection only. Evaluation seeds remain
unopened until the source-hashed protocol is committed.
