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
and `hybrid_exact`. The production safeguard now chooses `hybrid_exact` when
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

Five exact methods match an independent time-expanded dynamic-programming
oracle on 200 randomized problems each (1,000 comparisons). Additional capped searches verify
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

The development experiment has 60 matched cases: four seeds, five families and
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
