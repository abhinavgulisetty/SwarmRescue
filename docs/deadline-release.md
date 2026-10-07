# Exact deadline routing, release 0.5.0

The evacuation deadline safeguard now replaces an infeasible proposed route
using an exact hybrid: shortest-path certificates, bounded Bellman dynamic
programming, a compressed feasible-time corridor, then integer dual-bound A*
when neither table fits. Feasible proposals retain their existing routes.

## Frozen results

Eight fresh seeds × six graph families × three label capacities produce 144
matched cases (48 distinct maps). There are 126 feasible cases and 18 impossible
cases. Every method receives the same static estimated hazard costs, quantized
objective and discrete deadline. The candidate proves all 126 feasible optima
and correctly rejects all 18 impossible cases, with zero estimated-objective gap.

| Method | Proven optimal / 126 feasible | Mean objective gap | Hybrid speed ratio on mutually exact pairs |
|---|---:|---:|---:|
| Previous Pareto search | 55 | 13.72% | 1.14× |
| Ordinary A* | 112 | 7.59% | 0.948× |
| Lazy A* | 109 | 8.06% | 0.954× |
| Dual-bound A* | 119 | 1.15% | 1.06× |
| Full-table bounded DP | 111 | 7.85% | 0.999× |
| Corridor DP | 126 | 0% | 1.10× |
| Exact hybrid | 126 | 0% | 1.00× |
| Native HiGHS MILP | 123 | 0.24% | 14.33× |

Ratios use three cold full API calls, including quantization, preprocessing and
allocation, and compare only mutually proven-optimal feasible cases. The HiGHS
limit is three seconds per call; its three unproven results are retained and
excluded from exact timing ratios. The hybrid/HiGHS 95% map-seed bootstrap CI is
[13.12, 16.06]; corridor DP [1.05, 1.16]; ordinary A* [0.87, 1.01]. The hybrid is
about 5% slower than ordinary A* on mutually exact pairs, while returning better
routes in 13 cases where A* exhausts its label allowance. It returns cheaper
routes than the previous Pareto search in 21 cases. These case counts include
repeated maps at different label capacities.

The implementation was committed before opening evaluation seeds. Source hashes,
selection rules and comparator settings are sealed in
[the protocol](../eval/constrained_protocol.json). All raw samples, routes, cost
gaps, resource fallbacks, infeasibility and solver statuses are preserved in
[the compressed evaluation](benchmarks/deadline-evaluation.json.gz).
[Summary and family breakdowns](benchmarks/deadline-summary.json) include seed
bootstrap intervals. [The figure](benchmarks/constrained-evaluation.svg) can be
regenerated with `docs/benchmarks/render_constrained.py`.

## Actual published implementation

The authors' [ESA 2025 NWRCA*](https://drops.dagstuhl.de/storage/00lipics/lipics-vol351-esa2025/LIPIcs.ESA.2025.97/LIPIcs.ESA.2025.97.pdf)
implementation, pinned to upstream revision
`4098543d0c5d78afc977017b5abf6054671a8c43`, returns the correct optimum or
infeasibility on all 48 maps. On its 42 feasible maps, its reported native
initialization-plus-search time divided by the hybrid's **full API-call** time
at 4096 labels averages **20.38×**, CI [15.75, 27.61]. This is a conservative
comparison of different timing boundaries: reference parsing and IPC are
excluded, while all candidate preprocessing is included. It is not an
identical-timing-scope or generic RCSP superiority claim.

The frozen reference build uses the recommended integer bucket queue, no
tie-breaking, DIM=2, GCC -O3 and path-construction support. A subsequent fixed
check with the authors' documented default build **without** path support is
also correct on 48/48 and gives **20.16×**, CI [16.24, 25.64], on the same 42
feasible maps. That check reuses already opened inputs and is labeled as a
configuration robustness check, not another independent holdout. Both raw
reports and binary identities are retained. The original optional path printer
segfaults and the hybrid queue raises SIGFPE on a development input; failures
are disclosed, and neither failed configuration is counted as a timing win.

These maps have 12–1600 nodes and large Q16 integer costs. Bucket range/setup
costs influence native reference timing on small maps. These results do not
establish performance on the authors' large road-network benchmarks, negative
weights, multiple resource constraints or all existing methods. Upstream AGPL
code is external and is not bundled in the project. See the
[research and reproducible build recipe](constrained-research.md).

## Understand the route change

```bash
swarmrescue deadline-demo --out outputs/deadline-demo
```

Open `outputs/deadline-demo/index.html`. The same ten corridor choices are
shown for both searches; the deadline slider changes which quiet detours fit.
At 26 seconds, the previous bounded search returns cost 110 and the hybrid
returns cost 68: **38.2% lower estimated cost**, within the same deadline.
All 21 displayed deadline settings match an independent closed-form optimum.
Chrome checks verify slider values, pause/resume/restart, animation and the
390px mobile layout. This schematic explains routing rather than claiming a
physical rescue or survival improvement.

## Limits and validation

The final algorithm suite passes 758 tests with three skips, including 1,200
randomized comparisons against an independent time-expanded oracle. Boundary
tests cover exact arena fit, existing-buffer preservation, disconnected nodes,
large integer costs and honest capped-search certificates. Deployment tests,
new-code Ruff checks and browser interactions pass.

The 200KiB arena limit covers numeric tables and labels. Host Python graphs,
heaps and temporary arrays, and native HiGHS/reference process RAM, are outside
that accounting. The new deadline planner runs on the host; the existing
validated Cortex-M4 service still runs the unchanged Q16 Bellman kernel.
The static modeled objective and traversal clock do not establish measured
survival, future-hazard prediction, complete MCU firmware or medical safety.
