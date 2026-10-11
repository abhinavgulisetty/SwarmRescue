# Compiled exact deadline routing

This release targets full routing-call latency, including graph conversion and
quantization. It replaces Python heap/DP inner loops with a caller-arena C99
kernel, while preserving positive Q16 costs and the same hard traversal clock.
The original 0.5.0 frozen results and standalone installed wheel remain intact;
reproducing that old protocol requires its original wheel, not modified sources.

## Algorithm

The candidate validates and quantizes every call, constructs a fresh CSR graph,
and shares topology/time arrays with the reverse graph. Directed arrival-risk
weights remain separate; no heuristic or graph cache is used across queries.

Exact fastest and cheapest certificates can terminate a solve immediately.
For the time certificate, a verified unit-clock graph uses FIFO layers with
minimum-cost tie relaxation. Every predecessor layer is completed before the
next layer is popped, so time and its lexicographic cost tie remain exact.
Weighted clocks retain heap Dijkstra.

Where allowed, certificate expansion stops when the start is settled. Unknown
node distances use the settled threshold, giving `min(true_distance, threshold)`.
For nonnegative edge cost `c`, consistent `h` implies consistent `min(h,K)`:
if `h(v)>=K`, the right side is at least `K`; otherwise the original triangle
inequality applies. Secondary time/witness data is marked unknown for unsettled
nodes, preventing those nodes from supplying an invented feasible suffix.
These lower bounds safely guide A* and conservatively widen a DP corridor.

If `N*(deadline+1)<=8192`, exact DP avoids label-search overhead. Otherwise,
lazy A* uses the caller's label quota and consistent cost bound. Feasible
cheapest suffixes improve the incumbent before reaching a goal label. Prefix
cycle checks use a temporary high-bit tag in the existing closed-time array;
clocks fit 31 bits, all tags are restored before search resumes, and no extra
numeric allocation is needed. When label search exhausts its allowance, full
or packed exact DP follows if it fits. A low-memory Python fallback preserves
the best known incumbent and reports honest optimality/resource flags.

Packed DP keeps only feasible time intervals. Both implementations skip empty
clock spans; tests cover deadlines near two billion with six actual states.
All sums use int64, unreachable values are excluded before addition, and all
numeric allocations stay within the caller arena and preserve its prefix.
The C kernel has no hidden heap allocation. Host input CSR, output route arrays,
Python objects and native third-party process RAM are outside arena accounting.
This is a host execution change; the Cortex-M4 Bellman service is unchanged.

## Strong comparators and research

The new protocol includes legacy Pareto, standard/lazy/dual Python A*, bounded
full/corridor Python DP, the previous hybrid, **compiled lazy A***, compiled
full/corridor/DP-first methods, and native HiGHS MILP. Compiled baselines share
the same vectorized graph assembly, compiler flags and validated input model.
The conventional compiled A* uses complete Dijkstra fields and scalar lazy
single-resource dominance; it is not slowed with artificial work. The candidate's
unit-clock specialization, truncated certificates, suffix witnesses and
adaptive DP are the separately measured algorithm changes.

The [ESA 2025 paper](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2025.97)
and [enhanced bidirectional A*, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34892)
provide stronger constrained-search references than ordinary shortest-distance
routing. The actual authors' published NWRCA* implementation is retained as an
external comparator. Its license and input-budget adaptation remain documented
in the [previous research report](constrained-research.md). No upstream code is
bundled into this project.

Five cold full-call samples are collected per method in seeded random order.
All imports and compiled libraries are warmed outside measurement equally;
compilation is not mixed into search timing. Exact speed ratios require both
methods to prove the same feasible optimum. Failed, infeasible, timed-out and
resource-capped cases remain in the reports. Confidence intervals resample
whole map seeds, keeping repeated label capacities and families correlated.
A quick suboptimal fallback is never treated as an exact-solver timing win.

Development uses 72000–72003; fresh evaluation uses 82000–82011, six families
and three label capacities. The source-hashed rule is committed before opening
the evaluation seeds. Development tuning is not presented as held-out evidence.
No all-input, all-method, hardware-cycle or measured-survival superiority claim
is justified by this experiment.
