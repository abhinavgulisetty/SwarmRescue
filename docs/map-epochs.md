# Map epochs and delayed-packet safety — 2026-10-04

Coordinated physical-map rebuilds now advance a checked uint32 epoch. Every robot stamps its generated source packets with that epoch. Receivers reject mismatched epochs before checking cell bounds or queue capacity, and recheck queued packets before fusion. Regional hotspot floors also ignore wrong-map packets. Thus a delayed frame cannot interpret an old cell index as a newly assigned voxel. Direct stateless fusion requires an explicitly matching epoch.

## Wire compatibility and budgeting

Epoch zero keeps the exact existing 8-byte header and source-identified legacy wire bytes. For nonzero epochs, the legacy provenance field is zero, an invalid source-factor bitmap reserved as the explicit extension marker. The 16-byte extended header carries the original provenance bitmap and uint32 map epoch. The canonical packet object requires nonzero provenance. Frames without source provenance are no longer accepted as legacy factors; physical RobotState already rejected them before this change.

With the existing 12-byte update format, 19 updates still fit exactly inside the 244-byte payload limit. Epoch frames cost eight additional header bytes per broadcast. Partial-source packet selection charges the actual header size, preserves the epoch, and stays within its byte budget. Radio decode, forwarding and duplicate keys preserve map identity. Legacy-only readers reject valid extended frames instead of interpreting their cell payload as a legacy factor.

The initial length-only discriminator was rejected: truncating an extended payload by four bytes could make it look like a valid legacy packet. The unpromoted snapshot is `outputs/demo/map-epochs/length-discriminator-rejected.source.tar.gz`; its five preliminary missions were interrupted and are not acceptance evidence. The final explicit marker remains in every valid prefix, so truncated epoch frames cannot fall back to legacy decoding. Tests include 1-, 4-, 8-, 13- and 16-byte truncations. Complete trailing-update omission retains the epoch and is not detected by this schema; link integrity/authentication is outside this change.

## Rebuild lifecycle

Preparation validates the lattice and aging variance, checks epoch exhaustion, and allocates replacement robot states before consuming queues or discarding changed-region factors. Allocation failure restores the current graph/region references and leaves epochs, evidence and pending queues intact. Commit transfers direct evidence on surviving geometry, continues source sequences, copies rejection counters, then publishes the replacement graph, robot states and epoch. Uint32 epochs never wrap; exhaustion requires a new session.

This is a coordinated shared-map host reference, not distributed map-version consensus, concurrent rebuild synchronization, or cross-session replay protection. Two independently constructed maps with the same ordinal epoch are not guaranteed to have identical node IDs. Initial legacy epoch-zero traffic assumes the original shared map. Future-epoch packets are discarded, not buffered until a peer map catches up.

## Validation

All 541 tests and runtime Ruff pass. The new tests cover exact legacy bytes, epoch roundtrip/TTL/MTU, invalid and exhausted epochs, explicit zero-provenance marker rules, truncated frames, exact partial-packet budgets, floods and air-byte accounting, wrong-epoch frames with out-of-range indices/full queues, queued replay rejection, regional-floor protection, actual cell-index reuse after voxel removal, and failure-safe preparation for both flat and regional maps with both inference policies. Native integer conformance remains covered.

Final fresh physical regression runs follow. All source archives match their reports and share one source fingerprint. These are three disclosed development layouts, not paired algorithm-superiority or statistical generalization evidence. Normal synchronized transport produces zero epoch rejections; adversarial delayed frames are exercised by the replay tests.

| Seed | Map | Evidence | Sites reached | Final map epoch | Radio air bytes | Numeric peak bytes | Full-mission exposure proxy | Last arrival seconds |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 7 | region4 | latest | 4/4 | 10 | 16,160 | 182,109 | 227.402088 | 45.060 |
| 19 | region4 | latest | 8/8 | 4 | 28,992 | 167,816 | 673.602695 | 29.380 |
| 42 | region4 | latest | 8/8 | 6 | 38,496 | 187,136 | 23.830171 | 30.562 |
| 7 | flat | accumulate | 4/4 | 12 | 17,152 | 984,756 | 239.260655 | 36.652 |
| 42 | flat | accumulate | 8/8 | 6 | 40,224 | 898,100 | 23.517286 | 29.018 |

Heat/smoke exposure is the existing uncalibrated observer proxy, not ISO survival or patient evacuation. Numeric peaks exclude geometry/refinement scratch, Python objects/indexes, and simultaneous old/new arenas during rebuild. Whole embedded RAM, measured timing, sensor deployment, coupled localization and physical evacuation gates remain open. New header overhead changes packet selection; prior checkpoint traffic is historical context rather than a fresh same-source paired comparison.

## Reproduction

```bash
.venv/bin/pytest -q
.venv/bin/ruff check cascade ember eval hazard mcu tests
.venv/bin/python -m cascade.demo --seed 42 --robots 8 --compact --live-disaster --distributed --region-cells 4 --latest-evidence --no-render --report outputs/demo/map-epochs/replay.json
```

## Source archives

- `outputs/demo/map-epochs/seed7-region4-latest-v2.json`; source archive SHA-256 `e669308c237d12a69234cb61c2b225186a3b5a15a1df04001e8bcf2007d491c3`.
- `outputs/demo/map-epochs/seed19-region4-latest-v2.json`; source archive SHA-256 `a355b44dc81efd5260e4ebfaa5a2e4223d498d0b2b705a887ef45e35642dcc7c`.
- `outputs/demo/map-epochs/seed42-region4-latest-v2.json`; source archive SHA-256 `e75c1ab9fc8d1d5e5a5fb5ac612315715a99aa6d48ef419144ea84b4fac9d44d`.
- `outputs/demo/map-epochs/seed7-flat-accumulate-v2.json`; source archive SHA-256 `fd3a3354e21baa9efc3522d0ea49c7be3817e7f2c125cad892d52d4c0f020dff`.
- `outputs/demo/map-epochs/seed42-flat-accumulate-v2.json`; source archive SHA-256 `fd3321b7513514e1edb95bdd75a0143d42bf25f5bc31b0c0db3bec6f1ecd2898`.
