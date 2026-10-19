# SwarmRescue 0.7.0: complete-rescue verification and execution

Delivery date: 16 October 2026. This release implements a complete-rescue
verification interface, proves its exposure bounds for a declared model, checks
it with an external graph solver, and integrates its frozen-belief specialization
into physical rescue assignment. It does not establish historical priority or
calibrated human survival.

The [portable delivery bundle](outputs/releases/0.7.0/swarmrescue-0.7.0-complete-rescue.tar.gz)
contains the wheel, proof, verification receipts, source archives and
[installation instructions](outputs/releases/0.7.0/INSTALL.md). The
[release manifest](docs/releases/0.7.0.json) records artifact hashes and
which claims remain open.

## Run the installed mission

```bash
outputs/deployment/complete-rescue-0.7.0/run-mission
```

The launcher runs eight robots, receiver-local localization and hazard beliefs,
complete-rescue preflight, local contact, physical survivor escort, geometry
updates and two actual Cortex-M4 routing services under QEMU. To run host
routing instead, append `--firmware-robots 0`. It writes a physical mission
report and its installed-source archive into the deployment directory. It does
not flash a physical board.

The release wheel is
[`outputs/releases/0.7.0/swarmrescue-0.7.0-py3-none-any.whl`](outputs/releases/0.7.0/swarmrescue-0.7.0-py3-none-any.whl).
The isolated deployment uses hashed runtime dependencies and has been checked
from outside the source checkout. Deployment identity is recorded in
[`deployment.json`](outputs/deployment/complete-rescue-0.7.0/deployment.json).

## Concrete contribution and proof

The checker includes robot approach, survivor waiting before pickup, and escort
with independent bounded residence times on each route segment. One hazard
scenario persists throughout the complete rescue. It maximizes over **every**
allowed escort timing, rather than evaluating nominal speed or only the longest
duration. It rejects horizons that omit any admitted late schedule.

For each scenario, the exact problem is a longest path on a route-stage/time
DAG. Temporal prefix sums and a sliding-maximum queue compile that graph without
enumerating every duration edge. The proof, state sufficiency assumptions,
complexity, memory accounting, counterexamples and inherited ingredients are in
[`docs/research/lag-rescue/method.md`](docs/research/lag-rescue/method.md).

[`ember/rescue_certificate.py`](ember/rescue_certificate.py) supplies both a
float64 implementation and an exact integer-unit entrypoint. Integer rates with
`max_rate * horizon <= 2**52` and `dt=1` keep every prefix, max-plus key, sum and
comparison exactly representable. This is an arithmetic guarantee for the
supplied integral model; applying it to real measurements needs a justified
conversion and uncertainty model.

The new physical assignment considers risk-weighted and geometric approach/exit
alternatives, verifies the complete candidate before matching, and minimizes
cost among feasible candidate bids using SciPy's established rectangular linear
assignment implementation. It permits idle robots in unequal/infeasible task
sets. The current physical mission still requires a complete matching. Custody
begins only after a current-geometry, converged exit route also passes its
modeled deadline check.

Two defects were caught and fixed during end-to-end validation: a risk-preferred
detour could exclude a deadline-feasible geometric alternative, and the installed
evaluator initially depended on unshipped repository tests. Both have focused
regressions. No uncertified fallback replaces a failed certificate.

## External verification and original-author execution

An independent implementation materializes the full stage/time graphs and calls
the actual NetworkX `dag_longest_path` implementation. It uses Python integer
weights for exact integer checks; it neither calls nor copies the optimized
kernel recurrence.

| Check | Observed result |
| --- | --- |
| Independent floating graph comparisons | 288 cases; maximum discrepancy `2.22e-15` |
| Independent integer graph comparisons | 288 cases; exact equality for both actors in every case |
| Complete graph missions with a certified candidate | 80 of 96; the remaining 16 have no certified route among the three supplied choices |
| Direct complete-mission scenario/timing replay | 5,352 combinations; all selected rescues finish within both modeled budgets |
| Nominal-timing ablation | 31 false acceptances among 288 candidates |
| Longest-duration-only ablation | 2 false acceptances |
| Pointwise scenario-reset ablation | 28 false rejections |

The graph protocol uses predetermined synthetic instances and budgets. It is a
development study, not calibrated smoke or a sealed statistical confirmation.
The exact external graph solver agrees in solution quality; no superiority to
it or the published routing literature is claimed.

The public original-author implementation of Tihanyi et al.'s stochastic-hazard
task allocation was also downloaded at a pinned commit and executed unchanged.
Fresh original forward, reverse and brute-force allocation calls exactly match
the supplied objective values. Original forward/reverse policies and paths
were regenerated and evaluated on the supplied 5,000 episodes, reproducing joint
success fractions `0.7422` and `0.7580`. The cached objective table and hazard
episodes were reused, as the [reproduction report](docs/research/external-rescue/reproduction.md)
and [receipt](docs/research/external-rescue/receipt.json) explicitly record.

That author problem visits targets and exits; it does not model waiting human
exposure or bounded-lag escort. Its reproduced scores therefore are not a
matched comparison or evidence that this release beats that paper.

## Physical validation and remaining boundaries

Final source-archived physical validation is recorded in the
[acceptance summary](outputs/releases/0.7.0/validation/summary.json). All five
runs pass, with one identical installed-source fingerprint. A successful report
must independently
establish local contact, unique ownership, every survivor physically outside,
every robot outside, valid event ordering, actual selected firmware activity
and archive/source identity. A success label alone is insufficient.

| Seed / assignment | Robots | ARM workers | Physically evacuated | Last evacuation, simulated seconds | Survivor exposure proxy sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 19 / complete | 8 | 0 | 8/8 | 122.394 | 76.824037 |
| 42 / complete | 8 | 0 | 8/8 | 82.928 | 8.289299 |
| 42 / complete | 8 | 2 | 8/8 | 82.928 | 8.289299 |
| 7 / complete | 4 | 0 | 4/4 | 67.274 | 64.104276 |
| 42 / distance control | 8 | 0 | 8/8 | 81.396 | 8.825714 |

The paired native/ARM runs agree exactly on contact and evacuation times,
survivor positions/distance/exposure, robot positions/arrival times and radio
bytes. The ARM services execute 26 live route requests. The matched assignment
comparison reduces modeled survivor exposure by 6.08% and increases last
evacuation time by 1.88%. It demonstrates a measured tradeoff on this development
case, not universal superiority. The exposure proxy is not a physiological dose.

The physical preflight currently uses each robot's private **temporally frozen
routing-cost field**, with exposure thresholds disabled. Its certificates apply
to assignment candidate routes. The physical controller subsequently solves
and replaces routes on evolving geometry, so those preflight bounds do not
certify the executed physical dose or bounded-lag model. Physical completion
validates the software workflow; the graph study validates the exact model
checker. A calibration/containment result connecting them remains open.

The existing capsule-survivor model, shared host geometry, host-coordinated
assignment, omitted mutual collisions, and hybrid host/ARM architecture remain
explicit assumptions. Whole-robot MCU memory, measured physical-board cycles,
deployed sensor calibration, human physiology, partition-safe ownership handoff
and historical originality are not certified by this delivery.

All 933 final regression tests pass with the workspace ARM toolchain enabled;
there are no skipped tests. Changed runtime, verifier and test files pass Ruff.
The broader legacy research tree still has 558 pre-existing Ruff findings in
45 files; it has not been reformatted or silently declared clean.

## Reproduce the scientific checks

```bash
.venv/bin/python -m pytest -q tests/test_rescue_certificate.py tests/test_rescue_planning.py tests/test_lag_rescue_validation.py
.venv/bin/python -m swarmrescue.cli certificate-evaluation --output-dir outputs/certificate-evaluation
```

The [independent literature audit](docs/research/independent-novelty-2026-10-16.md)
identifies the closest prior work and the novelty blockers. Robust scenario
paths, dynamic programming, sliding maxima, safe return, auctions and linear
assignment are inherited methods. The narrow contribution is the implemented
complete-rescue checking/assignment interface and the proved, externally checked
bounded-timing behavior. A claim of historical first use remains unestablished.
