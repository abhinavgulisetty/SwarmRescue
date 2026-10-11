# Claim ledger

Use the scope in the same sentence as the claim. The evidence distinguishes a
mathematical guarantee, a checked finite instance, an experimental estimate,
and a research hypothesis. A numerical result alone does not establish novelty.

| Claim | Evidence | Required qualification |
|---|---|---|
| Positive-cost, single-resource A* completion and DP are exact under the stated assumptions. | Theorems 1–3 in [proofs.md](proofs.md). | Conditional on exact arithmetic, valid inputs and uninterrupted exact completion; not a verified-C claim. |
| Unit-duration FIFO and fixed-threshold distance clipping preserve the required certificate properties. | Lemmas 3–4 in [proofs.md](proofs.md). | Unit clocks must be verified; unknown distances cannot provide suffix witnesses. |
| A capped feasible route has a valid absolute gap bound U−L when its lower bound is valid. | Theorem 4; checked bounds in the saved instance certificates. | No constant-factor approximation guarantee; relative bound requires L>0. |
| All 72 original discrete maps have independently checked optima or infeasibility. | 1,095,724 Bellman equalities in [instance-certificates.json.gz](instance-certificates.json.gz). | Finite corpus; standard-library checker is independently implemented but not formally verified. |
| All 2,808 original saved result records satisfy the checked witness/claim contract. | [evidence-audit.json](evidence-audit.json), including source and input digests. | Includes valid unresolved statuses, not 2,808 successful exact solves; only final-repeat statuses were saved. |
| The candidate proves 196/198 original feasible case optima. | Original raw data plus independent certificates. | 198 cap-specific cases derive from 66 feasible maps; two cases on one map are 51.36% suboptimal. |
| Every solver call in the declared bounded exhaustive domain matches path enumeration. | [exhaustive-checks.json](exhaustive-checks.json): 330,876 native calls and 7,168 public-API calls. | Nodes 1–4, finite cost/time alphabets, fixed starts/goals and deadlines 0–6; excludes large-clock/cap exhaustion coverage. |
| The original candidate has a 1.044× geometric-mean speed ratio against the tested compiled A*. | 171 mutually exact cases in the original sealed evaluation. | One host, 11 informative seed blocks; 49 individual losses; all-method common-support 95% interval includes a tie. |
| The original Python A* comparison has a 5.90× geometric-mean speed ratio. | 174 mutually exact cases in the original evaluation. | Contains implementation/language effects; not an algorithm-only advantage. |
| Numeric scratch stayed below 200 KiB in the recorded candidate runs. | Recorded high-water marks checked against the declared arena. | Excludes graph input/output arrays, Python heap, C stack and external process memory; not whole-device RAM. |
| Fresh confirmation is assessed by the frozen primary endpoint, preserving every repeat. | [protocol](../../eval/paper_confirmation_protocol.json) and confirmation reports. | Three processes on one physical host; approximate inference and success-conditioned support. Read the actual primary conclusion in [manuscript.md](manuscript.md). |
| The fresh candidate proves 606/612 feasible optima on every repeat, versus compiled lazy A*'s 517/612. | 105,840 checked repeat records; [confirmation-summary.json](confirmation-summary.json). | Finite 240-map corpus with 200 KiB numeric scratch and stated label quotas; the previous hybrid also proves 606/612. |
| The fresh candidate is 6.707× faster than the previous Python hybrid on mutually exact pairs. | 606 exact pairs; approximate hierarchical 95% interval [6.328,7.101]. | Same host and pipeline scope; includes language/implementation effects, not a proof of purely algorithmic gains. |
| Fresh compiled-A* latency is compatible with a tie. | Primary ratio 1.011×, approximate 95% interval [0.998,1.023]; registered primary rule fails. | Do not present the earlier 1.044× estimate alone as a reliably replicated advantage. |

## Statements the evidence does not support

- “Faster than every existing algorithm,” “wins on every input,” or a universal
  worst-case latency advantage. Existing slower cases directly contradict the
  per-input version of that claim.
- “Always optimal within 200 KiB.” The two original capped cases contradict it.
- “Improves rescue survival” or “reduces evacuation time.” The verified seed-42
  physical runs match the previous release; their purpose was regression checking.
- “Runs the new deadline solver on a Cortex-M4 with measured cycle speedups.”
  This kernel executes on the host; the embedded Bellman service is separate.
- “Uses at most 200 KiB total memory.” Numeric arena accounting excludes other
  allocations and the stack.
- “127.86× faster than state-of-the-art RCSP generally.” The authors' external
  NWRCA* comparison has different timing boundaries and small-workload setup
  effects. It is supplementary evidence with its own declared adaptation.
- “Formally verified implementation” or “a new theorem establishing algorithmic
  superiority.” The proofs concern a model, and the finite executable checks
  supplement them. The standard mathematical ingredients predate this project.

## What would strengthen the paper further

Component ablations are required before attributing gains separately to FIFO,
clipping, suffix updates or the adaptive selection threshold. An independent
machine and public realistic-network workloads are required for wider empirical
generalization. A fuller literature review is required before claiming a novel
algorithm or first-of-its-kind combination. None of these gaps can be fixed by
stronger wording or additional significance tests on the same small corpus.
