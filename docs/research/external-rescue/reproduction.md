# Original-author stochastic-hazard rescue reference

This reproduction executes the public reference implementation accompanying
Tihanyi, Lu, Karaca and Kamgarpour's *Multi-robot task allocation for safe
planning against stochastic hazard dynamics* ([manuscript](https://arxiv.org/abs/2103.01840),
[ECC 2023 publication](https://doi.org/10.23919/ECC57647.2023.10178126)).

The source is pinned to original-author repository commit
[`ddce51c4dc8c067d350ee89bbf41cea76b401f47`](https://github.com/TihanyiD/multi_alloc/commit/ddce51c4dc8c067d350ee89bbf41cea76b401f47),
dated 12 April 2022, message `'polished'`. That public code predates manuscript
version 3 and the 2023 publication. This is a reproduction of the identified
public implementation; exact equivalence with every final-paper detail is not
assumed. No original source files or supplied case-study files were modified.

The repository has no LICENSE or other license declaration. Its source is
downloaded separately into `/tmp` and is not vendored into SwarmRescue. The
local harness and output receipts are available here; source access remains
through the original repository.

## What was actually executed

1. Run the untouched author entrypoint
   `two_stage_framework/two_stage_framework_case_studies/two_stage_framework.py`
   with its default `case_study_1` and noninteractive Matplotlib plotting.
   This default **loads saved results**, hazard samples and the dynamic-program
   objective table. Its successful execution alone is a cached-example replay.
2. Instantiate and execute the original `Forward_Greedy_Allocator`,
   `Reverse_Greedy_Allocator` and `Brute_Force_Allocator` classes afresh on the
   author's supplied dynamic-program objective table. These are actual calls
   to downloaded author source, not local rewrites of the auctions.
3. For the forward and reverse solutions, execute the original low-level
   policy generator, path simulator and joint-episode success calculation
   afresh. Reuse the supplied hazard episodes. Exhaustive curvature diagnostics
   are omitted because they do not determine the allocation or policy.

The full robot-by-task-subset objective table is not regenerated, and new
hazard episodes are not sampled. Fresh allocation results are checked against
the supplied saved solutions. This distinction is recorded in
[`receipt.json`](receipt.json), together with every tracked source/data SHA256,
runtime package versions, measured current wall times and unchanged-source
checks. Historical time fields stored by the author are labeled separately;
they are not current execution measurements.

The completed run reproduced all three supplied allocations exactly and had
zero absolute error against their saved multiplicative objective values:

| Original method, freshly executed | Multiplicative objective | Joint success on supplied episodes | Fresh policy/path/sample replay |
| --- | ---: | ---: | ---: |
| Forward greedy | 0.6987007497806219 | 0.7422 | 52.50 s |
| Reverse greedy | 0.7166217852020254 | 0.7580 | 55.05 s |
| Brute force | 0.7166217852020254 | Not recomputed for brute force | Not recomputed |

The forward/reverse episode fractions also match their saved results. The
fresh allocation-only calls took about 0.69 ms, 1.32 ms and 4.40 ms,
respectively, because they queried the already supplied objective table.
These timings omit objective-table generation and are not end-to-end solver
times. Reverse greedy matches the brute-force allocation optimum on this
specific supplied instance; this is not a general optimality claim.

## Scope of the baseline

The author problem has three robots, five targets, an evolving stochastic
hazard, 75 time states and 5000 supplied hazard episodes. Robots visit their
assigned targets and exit. The multiplicative allocation objective is the
product of individual safe-mission probabilities. The reported `group` value
is the fraction of the supplied hazard episodes in which all simulated robot
paths avoid the hazard. It is not a held-out estimate and is not the fraction
of physically escorted humans.

No waiting-survivor exposure, pickup duration, human escort speed, exposure-dose
budget or radio-blackout extension is included. No numerical result here is a
comparison between the public implementation and SwarmRescue's complete-rescue
certificate or MuJoCo controller. Any such adaptation must preserve the
original results separately and disclose its changed objective and dynamics.

## Reproduce

Run from the SwarmRescue checkout. Dependencies are isolated from the project:

```bash
git clone https://github.com/TihanyiD/multi_alloc.git /tmp/swarmrescue-tihanyi-original
git -C /tmp/swarmrescue-tihanyi-original checkout ddce51c4dc8c067d350ee89bbf41cea76b401f47
uv venv --python .venv/bin/python /tmp/swarmrescue-tihanyi-env
uv pip install --python /tmp/swarmrescue-tihanyi-env/bin/python numpy==1.26.4 pandas==2.2.3 matplotlib==3.9.4
.venv/bin/python -m eval.external_rescue_reference \
  --source /tmp/swarmrescue-tihanyi-original \
  --python /tmp/swarmrescue-tihanyi-env/bin/python \
  --output docs/research/external-rescue \
  --fresh-policies
```

The harness requires the pinned commit and rejects changes to tracked external
source/data. It fixes one BLAS thread and uses the Agg plotting backend. Source
hashes make the exact supplied samples and cached table reviewable. See
[`original-entrypoint.stdout.txt`](original-entrypoint.stdout.txt),
[`fresh-solvers.stdout.txt`](fresh-solvers.stdout.txt) and
[`author-objective-table.json`](author-objective-table.json) for the reproduced
execution and data interface.
