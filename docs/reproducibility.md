# Reproducing the recorded evidence

The repository contains two recorded snapshots: the frozen 0.7.0 delivery
validated on 16 October 2026, and the cleaned source validated on 19 October.
The original wheel and mission source archives remain unchanged. The cleaned
checkout keeps the production runtime and verifier while removing obsolete
experiments; its 872-test suite differs from the frozen release's 933-test suite.

## Independent certificate study

From a fresh checkout with Python 3.12 and uv installed:

```bash
uv sync --frozen --extra dev --python 3.12
uv run --frozen --extra dev swarmrescue certificate-evaluation \
  --output-dir outputs/certificate-evaluation
```

The [model proof](research/lag-rescue/method.md),
[frozen protocol](research/lag-rescue/protocol.json) and
[recorded results](research/lag-rescue/results.json) define the comparison.
Expected results are 288 exact integer comparisons with zero mismatches and
5,352 successful scenario/timing replays over 80 selected complete missions.
The other 16 missions have no certified route among their supplied candidates;
they are not proved globally infeasible. Runtime measurements can vary.

## Archived mission acceptance

The five mission reports and their sibling source archives are included in
[the validation directory](../outputs/releases/0.7.0/validation/). Recheck actual
coordinates, custody, event ordering, selected firmware activity and archive
identity without rerunning physics:

```bash
uv run --frozen --extra dev swarmrescue rescue-evaluation --reports \
  outputs/releases/0.7.0/validation/seed19-complete-native.json \
  outputs/releases/0.7.0/validation/seed42-complete-native.json \
  outputs/releases/0.7.0/validation/seed42-complete-arm.json \
  outputs/releases/0.7.0/validation/seed7-complete-native.json \
  outputs/releases/0.7.0/validation/seed42-distance-native.json \
  --out outputs/mission-evidence-check
```

Absolute paths inside frozen reports record the validation machine. The checker
uses the report files supplied above and their adjacent `.source.tar.gz` files,
so those original absolute paths do not need to exist on your machine.

## Packaged delivery

Verify the tracked wheel, portable archive and release receipt:

```bash
(cd outputs/releases/0.7.0 && sha256sum -c SHA256SUMS)
```

Follow the [portable installation instructions](../outputs/releases/0.7.0/INSTALL.md)
to install the wheel or the [GitHub release](https://github.com/abhinavgulisetty/SwarmRescue/releases/tag/v0.7.0)
to download the assets. The archive includes its own per-file checksum list.
The [deployment receipt](../outputs/deployment/complete-rescue-0.7.0/deployment.json)
is provenance for the validated machine, not a relocatable environment.
Environments, compiler installations and simulator caches are regenerated locally.

## Original-author reference and scope

The [reference reproduction](research/external-rescue/reproduction.md) specifies
the pinned external checkout, dependencies and supplied input data. Its source
is not vendored. Its task differs from human waiting and escort, so reproduced
objectives do not establish a matched superiority result.

The graph checker proves properties of declared scenarios and timing bounds.
The simulated mission demonstrates integrated contact and egress. Physical
preflight uses frozen cost beliefs with exposure budgets disabled; these results
do not certify human survival or establish historical algorithmic priority.
