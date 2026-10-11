# Faster exact deadline routing — 0.6.0

The production deadline safeguard now uses compiled exact certificates and
adaptive DP/A* execution. Feasible existing proposals remain stable. The candidate
uses unit-clock FIFO specialization, consistent truncated certificate fields,
cheap-suffix incumbent updates and compressed time corridors. The
[algorithm and proofs](native-deadline-research.md) explain the changes.

## Fresh frozen comparison

Twelve new map seeds × six families × three label capacities produce 216 cases,
72 distinct maps, 198 feasible cases and 18 impossible cases. All methods receive
the same objective, graph, observations, time resource and 200KiB numeric scratch
ceiling. Compiled baselines share the fresh vectorized CSR assembly and -O3 compiler.
The protocol was committed before opening any evaluation seed.

The candidate is faster **on average than every tested baseline**, on mutually
proven-optimal feasible pairs:

| Comparator | Candidate speedup | 95% map-seed bootstrap CI | Comparator proven optimal / 198 |
|---|---:|---:|---:|
| Legacy Pareto | 6.89× | [6.42, 7.43] | 87 |
| Python A* | 5.90× | [5.50, 6.35] | 174 |
| Python lazy A* | 5.76× | [5.42, 6.14] | 172 |
| Python dual A* | 6.86× | [6.17, 7.59] | 182 |
| Python full DP | 5.58× | [5.33, 5.87] | 174 |
| Python corridor DP | 7.50× | [6.55, 8.44] | 195 |
| Previous Python hybrid | 6.90× | [6.23, 7.54] | 196 |
| Compiled lazy A* | **1.044×** | **[1.015, 1.076]** | 171 |
| Compiled full DP | 1.055× | [1.037, 1.075] | 174 |
| Compiled corridor DP | 1.066× | [1.043, 1.088] | 195 |
| Compiled DP hybrid | 1.090× | [1.060, 1.129] | 196 |
| Native HiGHS MILP | 82.68× | [72.24, 94.42] | 192 |

The new candidate proves **196/198 feasible optima** and rejects all 18 impossible
cases correctly. Every returned objective matches the previous hybrid exactly,
so the speed gain does not reduce its route quality. The two remaining cases
are the same 82003 large-grid map at label capacities 32 and 128. Both return an
honest bounded fallback, **51.36% above** the unconstrained-memory oracle optimum.
Their mean gap across all feasible cases is 0.52%. At 4096 labels the optimum is
proved. This is a real remaining memory-limited weakness, not a hidden success.

HiGHS has a three-second call limit and returns no feasible solution in three
feasible cases; these and other unproven results remain in the raw data and are
excluded from exact timing ratios. Timing medians use five cold API calls; the
returned correctness/optimality record is the final sample. Solver imports and
library compilation are excluded for every method. All quantization, fresh
assembly, preprocessing and allocation are included. No query-result or graph
cache gives the candidate a warm-information advantage.

[The sealed protocol](../eval/native_deadline_protocol.json),
[full raw evaluation](benchmarks/native-deadline-evaluation.json.gz),
[summary and family breakdowns](benchmarks/native-deadline-summary.json), and
[comparison figure](benchmarks/native-deadline-evaluation.svg) preserve every
slower, infeasible and capped case. Confidence intervals resample whole map seeds.
Geometric mean wins do **not** imply a win on every individual input or against
every implementation ever published.

## Actual published reference

The authors' default compiled [NWRCA* implementation](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2025.97),
pinned to upstream revision `4098543d0c5d78afc977017b5abf6054671a8c43`, is correct
on all 72 maps. On 66 feasible maps, its reported initialization+search time
is **127.86×** the candidate full-call latency at 4096 labels, CI [116.69, 139.53].
Reference parsing and IPC are excluded while candidate preprocessing is included;
these are deliberately different timing boundaries, not a generic RCSP headline.
The recommended integer bucket queue is used without tie-breaking or path support.
Q16 bucket range/setup overhead matters on these 12–1600-node maps. This does not
establish performance on its large road networks, negative costs or multiple
resources. The [external raw reference](benchmarks/native-deadline-reference.json.gz)
and exact binary hash are retained; upstream AGPL source/binaries are not bundled.

## Validation and use

The final suite passes **779 tests**, with three skips. Independent oracle checks
include 2200 randomized comparisons across eleven strategies, 500 unit-clock
models, 480 capped-search checks and near-two-billion-step clock intervals.
AddressSanitizer and undefined-behavior checks pass on the final C kernel.
The integrated 21-budget deadline demo still matches its closed-form optimum.

```bash
outputs/deployment/fast/run-mission
outputs/deployment/fast/venv/bin/swarmrescue deadline-demo --out outputs/deadline-demo-060
```

The local release is installed separately from 0.5.0. Open the demo's `index.html`
to explain the 38.2% lower estimated route cost under a 26-second deadline, with
pause, restart and deadline selection.

Peak candidate numeric scratch is **204,788 bytes**, below the 204,800-byte limit.
Host input/output arrays, Python objects and third-party process RAM are outside
that accounting. The new deadline kernel executes on the host; the existing
Cortex-M4 Bellman service is unchanged. These results measure static estimated
route cost and host latency, not calibrated survival or hardware cycles.
