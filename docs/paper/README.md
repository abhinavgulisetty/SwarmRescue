# Research evidence package

Start with [the manuscript](manuscript.md), [the mathematical arguments](proofs.md),
and [the claim ledger](claims.md). This is a reproducible research draft and
evidence artifact. It is not an accepted paper, a completed novelty review, or
a formal verification of the entire implementation.

## Evidence layers

1. Algorithm arguments: [proofs.md](proofs.md) defines the precise static,
   positive-cost, single-resource problem and proves the safe transformations,
   exact completion conditions, conservative bounds and representation limits.
2. Independent instance certificates: [instance-certificates.json.gz](instance-certificates.json.gz)
   embeds the discrete graph and complete Bellman table for each of 72 original
   maps, along with every saved method/capacity result. A checker using only the
   Python standard library verifies 1,095,724 Bellman states and 2,808 results.
3. Bounded exhaustive verification: [exhaustive-checks.json](exhaustive-checks.json)
   records 330,876 native solver calls on every graph in a declared 1–4-node
   symmetric domain, plus 7,168 public-API calls with directed arrival risk and
   multiple goals. The independent oracle enumerates simple paths. This finite
   coverage is not a proof for arbitrary graphs or resource-exhaustion cases.
4. Original benchmark audit: [evidence-audit.json](evidence-audit.json) checks
   source identities, generated inputs, costs, routes, lower bounds and saved
   medians; it includes clustered bootstrap intervals, leave-one-seed-out
   sensitivity, common-success support and Holm-adjusted diagnostic p-values.
   It cannot recover statuses missing from the original timing repeats.
5. Fresh confirmation: [the committed protocol](../../eval/paper_confirmation_protocol.json)
   freezes 40 new seeds, three independent process sessions and seven calls per
   method/case/session. Every repeat's result, route and latency is preserved.
   Source identities are checked before and after each worker. This addresses
   repeat-status loss; it does not establish machine-independent performance.

The [fresh summary](confirmation-summary.json) checks all **105,840 repeats**.
The candidate proves **606/612** feasible optima versus compiled lazy A*'s
**517/612**, but its registered latency endpoint is **not confirmed**: the
1.011× ratio has an approximate 95% interval [0.998,1.023]. Read this result
alongside the favorable original benchmark rather than claiming replicated
compiled-A* speed superiority. [Figure](confirmation.svg) and
[CSV table](confirmation-table.csv) include common-support sensitivity.

All raw records and unfavorable cases are retained. Production routing was not
changed to produce the evidence. The existing 0.6.0 benchmark and deployment
remain reproducible with their sealed source identities.

## Reproduction

Use the repository's Python 3.12 environment with its locked dependencies and
the same C compiler for timing comparisons. Run commands from the repository
root. Build/import warmup is outside measured calls; each call still rebuilds
its graph data. Process startup is not part of the reported routing latency.

```bash
# Check the committed file identities first.
sha256sum -c docs/paper/SHA256SUMS

# Standard-library-only instance verification; no C compiler or solver needed.
python -m eval.routing_certificates docs/paper/instance-certificates.json.gz

# Recreate the original evidence audit and bounded exhaustive checks.
.venv/bin/python -m eval.paper_evidence
.venv/bin/python -m eval.exhaustive_deadline

# Run the fresh timing replication sequentially, without concurrent workloads.
.venv/bin/python -m eval.paper_confirmation --worker 0 --output outputs/paper-reproduction/confirmation-process-0.json.gz
.venv/bin/python -m eval.paper_confirmation --worker 1 --output outputs/paper-reproduction/confirmation-process-1.json.gz
.venv/bin/python -m eval.paper_confirmation --worker 2 --output outputs/paper-reproduction/confirmation-process-2.json.gz
```

Rerun reports are kept under the repository's ignored output directory. The
frozen provenance exporter requires absolute input paths within the repository;
the commands below supply those paths without changing the registered code:

```bash
.venv/bin/python -m eval.paper_confirmation \
  --summarize "$PWD/outputs/paper-reproduction/confirmation-process-0.json.gz" \
              "$PWD/outputs/paper-reproduction/confirmation-process-1.json.gz" \
              "$PWD/outputs/paper-reproduction/confirmation-process-2.json.gz" \
  --output outputs/paper-reproduction/confirmation-summary.json

.venv/bin/python -m pytest tests/test_deadline_routing.py \
  tests/test_constrained_search.py tests/test_routing_certificates.py \
  tests/test_paper_confirmation.py
```

Fresh latencies and process IDs will vary; the semantic witnesses should remain
valid. Never replace committed evidence with only the favorable rerun. If any
solver, generator, checker or protocol-hashed source changes, create a new
versioned protocol instead of weakening its source check.

The independent certificate checker verifies the embedded model. Regenerating
the audit additionally binds that model to the original source-hashed generator
and raw benchmark input digests. Neither operation can independently attest
that an experiment was physically run on the claimed host; external replication
and archived release identities strengthen that provenance.

## Remaining publication work

Establish the novelty of the contribution with a wider literature comparison;
run component ablations and an independent-machine replication; add public
realistic-network workloads, varied deadlines and memory budgets. A physical
rescue improvement claim requires matched multi-seed mission experiments with
equal observations and hazard information. A whole-device memory or embedded
cycle claim requires measurements of that complete implementation on hardware.
The current host deadline kernel and existing Cortex-M4 Bellman service are
different components.

The locally committed preregistration establishes ordering in this repository.
For submission, deposit the protocol, code revision and evidence in a durable
public archive with a timestamp and persistent identifier, and ask an independent
researcher to reproduce the results. Publishing and pushing have not been done.
