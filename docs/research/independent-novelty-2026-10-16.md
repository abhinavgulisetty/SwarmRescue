# Independent novelty investigation: complete-rescue exposure certificates

Research date: 16 October 2026. The README's method claims were not used to
select the direction. Existing source was inspected to establish the implemented
problem, and external primary literature was searched independently.

Follow-up implementation: [release 0.7.0 delivery](../../RESEARCH_DELIVERY.md)
adds bounded escort timing, an exact integer-unit checker, independent NetworkX
verification, unchanged original-author reference execution and full physical
rescue validation. This initial audit remains the record of the first candidate
and its limitations; the follow-up does not establish historical priority.

**Outcome:** a new direction for this repository, with a precise mathematical
model, proved properties, and an executable synthetic demonstrator. Historical
originality of the proposed protocol is **not established**. The core robust
optimization and communication arguments below are established machinery. This
document must not be cited as proof of a first-in-the-literature algorithm.

## 1. Recommended research direction

**Complete-rescue certificates that preserve a shared uncertain hazard source.**

Before a robot accepts a survivor, require a certificate for the complete
trajectory: robot approach, survivor exposure while waiting, pickup, and escort
at the survivor's speed, ending at a specified refuge. Keep the same unknown
hazard source throughout these phases. A certificate represents accumulated
exposure by source coefficients, so phase composition does not invent a new
adversarial fire at each step. A stored, certified continuation can execute
without further radio delivery, within its declared model and time horizon.

The research question is whether this interface permits more complete rescues
than equally safe, independently bounded phases at a fixed communication budget,
and whether its payload can be reduced using the structure of actual building
hazard models. This is narrower and more falsifiable than another risk-aware
planner or a generic value-of-information scheduler.

The appropriate provisional title is:

> **Common-source exposure certificates for communication-limited robotic
> rescue: accounting for waiting survivors and delayed escorts.**

It is a candidate system/protocol contribution. A new general shortest-path,
adjoint, safety-clock, or communication-complexity theorem is not claimed.

## 2. Why it addresses an actual implementation gap

These findings come from executable source, not the README:

| Location | Implemented behavior | Consequence |
| --- | --- | --- |
| [`ember/search.py`](../../ember/search.py), `localize_and_assign` | Host-collected, greedy Euclidean matching; one survivor per robot. The docstring explicitly excludes independent agreement under partitions. | A nearby survivor need not have a feasible complete rescue. This is not a fully decentralized assignment protocol. |
| [`cascade/demo.py`](../../cascade/demo.py), contact/escort branch | Ownership begins before `exterior_route` and the subsequent escort planning call. | Exit feasibility is checked after contact rather than certified before assignment. |
| [`ember/evacuation.py`](../../ember/evacuation.py), `EvacuationConfig` | A physical ambulatory proxy with nominal speed 0.65 m/s and a fixed escort-gap limit. | A robot's passage time is not the survivor's passage time. |
| [`ember/belief.py`](../../ember/belief.py), `routing_cost` | Per-cell `mean + beta * sigma`. | There is no demonstrated complete-rescue exposure certificate under a shared evolving hazard model. This does not mean a sum of Gaussian margins is inherently unsafe. |
| [`ember/physical_team.py`](../../ember/physical_team.py) | Shared host planning geometry and synchronized map replacements. | A certificate must explicitly account for topology/model identity; decentralized map agreement cannot be assumed solved. |

The assignment is also suboptimal even without hazards: with robot coordinates
`(0, 2, 100)` and survivor coordinates `(1, -2, 100)` on a line, stable greedy
edge selection has total distance 5, while an optimal assignment has total 3.
Replacing it with Hungarian matching would fix that example but would not be
research novelty or establish rescue safety.

## 3. Independent prior-art audit

The summaries below identify collisions, rather than declaring an empty
intersection. Preprints are labeled; dates refer to the versions inspected.
External originality cannot be proved by a finite keyword search.

| Primary source | What is already established | Implication |
| --- | --- | --- |
| [Wu, Ramchurn and Chen, IJCAI 2016](https://www.ijcai.org/Proceedings/16/Papers/081.pdf), paper | Joint responder rescue and UAV information gathering under a spreading radioactive hazard, with POMDP planning and decision value of observations. | “Hazard sensing + rescue + VoI” is not a new research claim. |
| [Tihanyi et al., 2022 manuscript, published ECC 2023](https://arxiv.org/html/2103.01840v3), full text | Safe target visiting and exit under stochastic evolving hazards; safety-aware assignment, distributed auctions, and approximation bounds. | “Hazard-aware rescue allocation with guarantees” is occupied. Its modeled targets are visited; explicit survivor waiting/escort exposure is a narrower distinction, not proof of priority. |
| [Guo et al., TAC 2023](https://arxiv.org/abs/2302.05242), abstract | Coupled outbound and safe-return policies under probabilistic temporal tasks, including rescue/transport applications. | Complete outbound/return planning and returnability are established. |
| [Mattuschka et al., July 2025 preprint](https://arxiv.org/html/2507.02500v2), full text, especially §4 | Goal-oriented contamination sensing, an adjoint of integrated spacetime concentration, and dynamic sensor steering. | Neither integrated-dose adjoints nor goal-oriented PDE sensing can be claimed as new here. |
| [Raith et al., Networks 2018](https://onlinelibrary.wiley.com/doi/full/10.1002/net.21815), full text, §4 | Scenario-aware robust path labels and extension/dominance conditions. | Preserving a vector rather than its current maximum is established robust-path machinery. |
| [Yu and Yang, 1998](https://doi.org/10.1016/S0305-0548(97)00085-3), abstract and author-uploaded manuscript located | Minimax shortest paths under a common finite scenario set. | The simplex model below reduces exactly to extreme-source scenarios. Its path objective is inherited, not a new robust optimization formulation. |
| [Yang, Belta and Tron, 2019](https://arxiv.org/abs/1903.03692), preprint | Self-triggered safety control with explicit safe update periods. | Certificate expiry alone is not novel. |
| [Ballotta and Talak, 2025 journal version](https://arxiv.org/abs/2402.09382), abstract | Distributed safety control with communication delay and age-of-information prediction. | Freshness plus a safety monitor is not novel. |
| [Cordeiro, Tardieu and Pautet, ECRTS 2025](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECRTS.2025.5), proceedings and full-text audit | Resource-safe fallback with uncertainty, task criticality and coordinator faults; communication assumptions matter. | A rescue certificate must state its blackout and fallback assumptions rather than merely claiming resilience. |
| [Meriaux and Mahajan, ECC 2025](https://adityam.github.io/files/projects/mdp/conference/2025-ecc.pdf), paper | Robust path planning with incremental information revelation and a sufficient information state. | Preserving common uncertainty across online decisions is established. |
| [Klauck and Podder, 2014](https://arxiv.org/pdf/1402.4312), preprint, §2.2 | Distinguishable communication-matrix rows require distinct deterministic one-way messages. | The payload proof below specializes a standard communication argument. |
| [Shahparvari, posted 4 October 2026](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7560932), SSRN preprint abstract | Sensor tasking for capacitated evacuation decisions with noise, cost, latency and regret. | Decision-aware evacuation sensing is a particularly recent direct competitor. No peer-review status is inferred. |
| [Wang, July 2026](https://arxiv.org/abs/2607.01537), preprint abstract | Re-sensing deadlines from prediction-certificate validity and rollout drift. | A “safety expiry becomes sensing clock” claim is already occupied. |
| [Kemisetti, Shperberg and Sung, August 2026](https://arxiv.org/abs/2608.00792), preprint abstract | Certified temporal macro-actions ending at observations, with contingent planning over uncertain roadmaps. | Temporal path certificates and sense-before-commit policies are established. |

Search families included rescue under stochastic hazards, safe return under
blackouts, dose-constrained uncertain routing, correlated/source uncertainty,
PDE adjoint exposure sensing, expiring certificates, and robust-path communication
lower bounds. Broad searches were followed by primary-paper checks. Papers
identified only through abstracts were not treated as exhaustively compared
algorithms. This is a targeted novelty screen, not a systematic review of every
publication, thesis, patent, or non-English source.

## 4. Exact model and certificate

At the evidence epoch, let a nonnegative field evolve as

\[
 h_k=\bar h_k+A^kB\theta,\qquad
 A,B\ge0,\quad\theta\ge0,\quad\mathbf1^T\theta\le M.
\]

`theta` is one fixed latent source realization for the entire rescue. Columns of
`B` represent possible initial source locations or patterns. `bar h` is a known
nonnegative deterministic background. This is a bounded positive linear model,
**not** the existing nonlinear fire simulator and not a calibrated toxicology
model. Known time-varying propagators can replace `A^k` without changing the
following linear-functional argument. Arbitrary unmodeled ignitions cannot.

For actor `a` (robot or survivor), let the nonnegative row `ell[a,k]` encode
occupancy and exposure weight over tick `k`, including tick duration. Define

\[
 b_a=\sum_k\ell_{a,k}\bar h_k,\qquad
 q_a=\sum_k\ell_{a,k} A^kB.
\]

The survivor's rows include every tick waiting for the robot, any pickup delay,
and the slower escort. The robot's rows include the approach and escort.
Additional delay at a refuge is harmless only if that refuge is modeled as safe;
survivor exposure continues during that delay. Use different dose budgets for
the two actors if needed. Each `K_a` must be the **remaining exposure budget**
after accounting for all exposure before the evidence epoch. Without a prior
exposure bound, this theorem bounds only exposure accumulated from that epoch,
not lifetime dose. Budgets here are synthetic model parameters.

### Proposition 1: exact exposure safety under a fixed source budget

For either actor, the exact worst-case total modeled exposure is

\[
 D_a^*=b_a+M\max_j q_{a,j}.
\]

**Proof.** Nonnegativity gives
`q_a theta <= max(q_a) * sum(theta) <= M max(q_a)`. Putting all mass `M`
on a maximizing coordinate attains the bound; `M=0` is immediate. Therefore
`D_robot* <= K_robot` and `D_survivor* <= K_survivor` are necessary and sufficient
for both modeled total-exposure limits to hold for every admitted source.

If the stored trajectories are traversable, execution follows their declared
timing, and the hazard remains in this fixed model set, further messages are
unnecessary for these inequalities. All packets after commitment may be dropped.
This is a finite-trajectory result to the specified refuge, not an infinite-time
safety or arbitrary-failure theorem. Any instantaneous concentration threshold
requires separate per-tick constraints. A route change requires recertification.

### Proposition 2: exact composition needs more than current maximum exposure

At a phase boundary, add the base terms and source vectors before taking the
maximum: `(b1+b2, q1+q2)`. Set `b=0,M=1`. With `q=(1,0)` and `q'=(0,1)`, both prefixes have
worst-case dose 1. Continuation `r=(1,0)` produces full worst-case doses 2 and 1.
The current maximum-dose scalar cannot support exact arbitrary continuation.
This does not rule out encoding a whole vector in an arbitrary-precision scalar.

The same issue applies to transferring exposure accounting at a handoff, but a
complete ownership-transfer protocol is **not implemented or proved** here.

### Proposition 3: tight abstract payload bound

Suppose a prefix coefficient vector belongs to `{0,...,C}^m` for a positive
integer `C`. A deterministic,
zero-error, fixed-length, one-way message must let a receiver answer
`max(q+r) <= C` for **every** nonnegative integer continuation `r`, without
receiver side information about `q`. The message needs at least

\[
 \left\lceil m\log_2(C+1)\right\rceil\text{ bits}.
\]

**Proof.** For distinct vectors, choose a coordinate `i` with `q_i < q'_i`
(swap names if necessary). Set `r_i=C-q_i` and all other entries to zero.
`q` passes the fixed threshold `C`; `q'` fails it. Hence all `(C+1)^m` vectors
require distinct messages. Base-`C+1` encoding attains the bound. For `C=0`,
the sole vector requires zero payload bits, handled separately.

This is the standard distinguishable-row argument, specialized to this
certificate interface. It does not prove an `m`-counter lower bound for a
particular physical building: attainable vectors and continuations may occupy
a smaller set. A rank/reachability analysis is necessary before applying that
lower bound to smoke dynamics. Protocol headers, route identity and timing are
additional payload.

### Proposition 4: the price of resetting uncertainty between phases

For nonnegative phase vectors `q[1],...,q[S]`, when the denominator is positive,

\[
 1\le
 \frac{\sum_s\|q^{(s)}\|_\infty}
 {\|\sum_s q^{(s)}\|_\infty}
 \le\min(m,S).
\]

**Proof.** The lower bound is the triangle inequality. Each phase maximum is
at most the final maximum, giving the factor `S`. Also the sum of phase maxima
is at most the sum of all coordinates, at most `m` times the final maximum.
Disjoint unit-coordinate phases attain the smaller dimension when `S<=m`;
for `S>=m`, distribute equal total exposure across the `m` coordinates with
disjoint-support phases (zero phases are also allowed). The bound is sharp.

Set `b=0,M=1`. For two isolated source patterns and two exposed ticks, a trajectory may have
vectors `(1,0)` then `(0,1)`. The exact source-consistent dose is 1; the reset
bound is 2. At budget 1.5, reset reasoning rejects a truly robust-feasible
trajectory. This is a mathematical separation, not evidence of a universal
rescue-rate improvement. The inequality itself is elementary established
mixed-norm mathematics.

### Proposition 5: bounded error for upward coefficient rounding

For `delta>0`, replace each `q_j` by `delta*ceil(q_j/delta)`. For `M>0`, this can
only increase the certified dose, by less than `M*delta`; the bound is independent
of `m`. For `M=0`, the source dose error is zero. If `b` is rounded upward on a
grid `delta0>0`, add less than `delta0`. The proof follows directly from
`0 <= max(q_rounded)-max(q) < delta`, multiplied by `M`. Separately rounding `S` phases
can accumulate `S` such errors; exact addition before final rounding avoids that.
Floating-point arithmetic still needs directed error control before this is an
embedded safety guarantee.

## 5. Essential counterexamples and scope limits

1. **No correlation advantage for independent upper boxes.** If every uncertain
   source component can simultaneously take its upper bound, and all responses
   are nonnegative, that same corner maximizes all ticks. Accumulating first
   offers no strict improvement over a correct pointwise box envelope. A total
   source budget must be physically justified, not chosen to make results win.
2. **Positive dynamics do not imply monotone heating.** `A>=0` preserves order
   between fields, but a plume can move or decay. Safe departure times can be
   disconnected. A claimed blackout horizon permitting any delay `0..H` must
   check all those delays, rather than binary-search one endpoint. A stored
   exact departure schedule is a different contract.
3. **A guide-safe route may be survivor-unsafe.** For a hazard that increases
   from 0.4 at robot passage to 0.8 at delayed survivor passage, a 0.6
   instantaneous threshold passes the guide and fails the follower. A fixed
   spatial gap does not supply a progress or time-lag guarantee.
4. **Stopping is not a fire-safety fallback.** If continued waiting exhausts the
   survivor budget, a communication timeout cannot be handled merely by
   stopping. The stored complete continuation must already fit the contract.
5. **Unknown changes invalidate the model premise.** New sources, doors opening,
   collapse, motion delay, or survivor arrest are not made safe by ignoring
   packets. Model/geometry identifiers prevent using a known stale certificate;
   they cannot detect an unobserved physical change by themselves.

## 6. Executable evidence and what it can establish

The isolated demonstrator is [`eval/source_budget_rescue.py`](../../eval/source_budget_rescue.py),
with focused checks in [`tests/test_source_budget_rescue.py`](../../tests/test_source_budget_rescue.py).
The machine-readable run is [`source-budget-results.json`](source-budget-results.json).
It does not replace the project's production controller or claim calibrated
MuJoCo evacuation performance.

The independent oracle starts a forward field simulation at every extreme
source of the simplex and accumulates actor exposures directly. It does not
use the coefficient maximum formula to generate oracle doses. Since a linear
functional on this simplex attains its maximum at an extreme point, this is
complete over the continuous source set for each tested model and schedule.
Finite-model checks do not replace the proofs or verify numerical roundoff
for an MCU deployment.

The study compares the exact common-source bound, a pointwise reset bound, and
a snapshot baseline that freezes the uncertain source response at `k=0` while
retaining the known background. They use the same declared latent source set,
schedules and dose thresholds. Thresholds are deliberately selected from the
per-system 60th percentile of oracle schedule exposures, producing a descriptive
acceptance comparison; they are not independently calibrated or held out. The
reset method is a deliberately conservative
ablation, not the strongest published robust-path baseline. Candidate schedules
are supplied to the checker; global route optimality is not claimed. Detailed
sample counts, errors, witness values and source hashes belong to the JSON
receipt, so they can be regenerated instead of being copied from earlier runs.
Agreement with the extreme-source oracle also confirms the standard finite-
scenario reduction; it supplies no advantage over exact scenario evaluation.

The final development run uses seed `20261016`, 200 independently generated
systems and eight schedules for the **same start, survivor and refuge within
each system**: 1,600 schedules, 3,200 actor doses and 6,400 forward-simulated
extreme-source trajectories. Exact certificate and oracle values differ by at
most `1.78e-15`. The small-domain checks cover 729 coefficient/tick cases and
351 pairwise message-distinction witnesses, with no violations. Thirteen
focused tests and Ruff pass.

| Evaluator | Accepted schedules | Accepted despite oracle infeasibility | Oracle-feasible schedules rejected |
| --- | ---: | ---: | ---: |
| Common-source certificate | 947 | 0 | 0 |
| Pointwise reset | 674 | 0 | 273 |
| Upward-quantized certificate | 663 | 0 | 284 |
| Frozen uncertain-field snapshot | 911 | 205 | 241 |

These are schedule-level descriptive counts under the oracle-derived thresholds
above, not independent rescue trials or a claim of superiority to published
planners. The quantized variant can reject more schedules than the reset
ablation; smaller payload is not a free improvement. The maximum observed
reset/exact exposure ratio is 2.065. Separate witnesses demonstrate omitted
waiting exposure and a positive-system delay profile `[0,1,0]`, whose accepted
departure delays at budget 0.5 are `[true,false,true]`.

Reproduce from this checkout:

```bash
.venv/bin/python -m pytest -q tests/test_source_budget_rescue.py
.venv/bin/python -m eval.source_budget_rescue --output docs/research/source-budget-results.json
```

## 7. What would turn this into an accepted novelty claim

The following are research gates, not requests for permission. They describe
what the available evidence does and does not establish.

1. **Beat the exact reduction, not only the reset ablation.** On a time-expanded
   graph, use each extreme source as one scenario and compare with ordinary
   scenario-vector robust labeling. If the proposed certificate adds no new
   theorem, payload reduction, execution property or empirical capability,
   the algorithmic novelty claim fails.
2. **Justify the hazard set independently.** Fit source mass, unmodeled forcing,
   transport errors and clock/motion envelopes using separate development data.
   Validate containment on unused simulated scenarios. A missed true trajectory
   is a model failure even if all algebraic checks pass. Do not turn a coverage
   estimate into a deterministic guarantee.
3. **Exploit a rescue-specific structure.** Seek a stricter theorem about the
   reachable coefficient/continuation set under waiting plus bounded follower
   lag: can its exact minimal summary be smaller than generic scenario vectors,
   while supporting contact, route change and ownership handoff? The abstract
   payload lower bound above does not solve this question.
4. **Evaluate full missions at matched resources.** Before assignment, certify
   both actor budgets and actual exit reachability. Include unequal team/task
   counts, blocked exits, common versus independent hazard uncertainty, long
   dropouts and survivor delays. Compare distance assignment, full-rescue
   snapshot assignment, robust full-rescue assignment, and the closest
   published stochastic-hazard allocation method with disclosed adaptations.
5. **Separate protocol from hazard proof.** Demonstrate exclusive contact/escort
   ownership, failover and continuation behavior under partitions. A safe dose
   certificate does not solve distributed agreement or handoff atomicity.
6. **Freeze before confirmation.** Hash code, models, budgets, map set, packet
   budget and primary metrics before opening a new confirmation set. Report
   completed robust-feasible rescues and false acceptance/rejection separately;
   no benefit should be manufactured by silently changing uncertainty sets.

The safe present-tense claim is: **we derived and checked a complete-rescue
certificate for a bounded common-source model, with exact composition,
conservatism and payload results.** The stronger claim that its protocol is
historically new, or that it improves full rescue missions on this project,
remains unproved. Relabeling these established ingredients as “proven novelty”
would repeat the project's earlier problem.
