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
its firmware conformance. Additional choices are `astar`, `lazy_astar` and
`dual_astar`. The A* methods compute exact cheapest and fastest completion
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

Three exact methods match an independent time-expanded dynamic-programming
oracle on 200 randomized problems each. Additional capped searches verify
feasibility, route costs, lower bounds, honest optimality flags, arena prefix
preservation and saturated label-range certificates. Legacy firmware behavior
continues to match the original Pareto solver.

Development and fresh frozen evaluation will compare full-call latency,
proven-optimal solution counts, actual estimated-objective gaps, generated labels
and numeric arena peaks. Failed, infeasible and resource-capped cases must stay
in the results. Timing ratios for exact solving compare successful exact pairs;
a quick suboptimal fallback is not timed as an exact solver win. Author-paper
speedup figures are not project measurements.
