# Bounded streaming evidence — 2026-10-04

`--latest-evidence` is an opt-in physical-mission model that retains one current likelihood per source/cell. Newer packets replace that source's older contribution; equal or older sequence numbers are rejected. This avoids both unbounded repeated-history growth and repeated certainty accumulation from the same source. Accumulation remains the default. These three layouts are development experiments with one noise/radio replicate each, not sealed confirmation.

| Seed | Map | Evidence | Sites reached | Numeric peak bytes | Rejected factor attempts | Factor replacements | Full-mission exposure proxy | Search exposure proxy | Last arrival seconds | Radio air bytes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | region4 | accumulate | 4/4 | 145,245 | 0 | 0 | 227.402088 | 36.683948 | 45.060 | 15,536 |
| 7 | region4 | latest | 4/4 | 182,109 | 0 | 3,553 | 227.402088 | 36.683948 | 45.060 | 15,536 |
| 7 | flat | accumulate | 4/4 | 984,756 | 20,702 | 0 | 239.260655 | 44.332045 | 36.652 | 16,352 |
| 7 | flat | latest | 4/4 | 1,132,212 | 0 | 37,783 | 239.272221 | 44.327471 | 32.912 | 14,912 |
| 19 | region4 | accumulate | 8/8 | 130,952 | 0 | 0 | 673.602695 | 652.876184 | 29.380 | 27,584 |
| 19 | region4 | latest | 8/8 | 167,816 | 0 | 1,981 | 673.602695 | 652.876184 | 29.380 | 27,648 |
| 42 | region4 | accumulate | 8/8 | 150,272 | 0 | 0 | 23.830171 | 21.652159 | 30.562 | 36,416 |
| 42 | region4 | latest | 8/8 | 187,136 | 0 | 5,875 | 23.830171 | 21.652159 | 30.562 | 36,416 |
| 42 | flat | accumulate | 8/8 | 898,100 | 4,065 | 0 | 23.517286 | 21.335645 | 29.018 | 39,008 |
| 42 | flat | latest | 8/8 | 1,045,556 | 0 | 55,677 | 23.560872 | 21.352837 | 30.004 | 39,008 |

The new mode eliminates observed provenance drops in both saturated flat layouts while all ten runs pass. Regional physical exposure and arrival times remain identical across all three pairs; the regional ledger adds exactly 36,864 numeric bytes and peaks at 167,816–187,136 bytes. Flat full-mission exposure rises by 0.011565 (seed 7) and 0.043586 (seed 42), with mixed search exposure and arrival time. Seed 7 arrives 3.740 seconds earlier and uses 1,440 fewer air bytes; seed 42 arrives 0.986 seconds later with unchanged air bytes. No universal performance, survival or calibration improvement is claimed. The option remains experimental because it changes likelihood semantics and requires broader dynamic-field validation.

## Statistical model and invariants

For each cell, canonical precision and information equal the background plus the retained contributions from distinct sources. Replacing `(h_old, J_old)` by `(h_new, J_new)` updates the aggregate by `(h_new - h_old, J_new - J_old)`. Under nonnegative process-variance aging, retained contributions and the background stay positive in precision. Repeated updates do not create additional source/cell slots or additional independent likelihoods.

This is a latest-reading approximation to a changing field, not the original cumulative independent-observation Gaussian model, a coupled temporal factor graph, or a calibrated sensor-bias filter. Independent source errors are still assumed; common sensor bias and robot pose uncertainty remain unmodeled. Regional peak aggregation and monotone delivered-hotspot floors retain their prior approximation boundaries.

Process-variance aging scales every retained contribution by the same ratio used to age the aggregate. Therefore replacing an aged old factor removes its aged contribution rather than subtracting an unaged measurement. Geometry transfer copies these values and sequence watermarks only for retained factors; aggregate posterior marginals are not inserted as fresh evidence.

Arena-backed storage costs 18 bytes per capacity slot: two float64 canonical values and one uint16 sequence. The source/cell slot index and active identity set remain bounded Python metadata outside the numeric arena. Capacity is charged before regional allocation and oversized arenas still fail. New source/cell keys can still exhaust the table; an overflowing packet is rejected whole, including replacements in that packet. Existing-key packets can continue updating at capacity. Locally generated source sequences remain limited to 65,536 packets per session with no wraparound reuse; external senders must obey the same session rule.

The current transport is synchronous across geometry transactions. Packets delayed across topology epochs are not supported by the wire protocol, which lacks a map epoch; this work does not solve that existing limitation. The whole embedded-memory and measured timing gates remain false. Host aging/index/refinement scratch, map geometry, packet objects, Python runtime and analytical cycle constants are not MCU deployment evidence.

## Validation and selection

All 509 tests and runtime Ruff pass. Eight new tests cover 10,000 readings in one slot with bounded precision, independent source sequences, out-of-order/equal-epoch replay rejection, aging before replacement, removed/reindexed geometry, continued known-key updates at capacity, charged arena backing, and actual flat/regional rebuild transfer. First-epoch routes remain unchanged in the paired physical cases; later routes use the changed likelihood model.

Full-mission exposure includes waiting after arrival; search exposure ends at site arrival. Both integrate the uncalibrated heat/smoke proxy at the same cadence. Neither is ISO tenability or patient evacuation. Changes in route and geometry-refresh timing affect radio traffic and subsequent sensed cells. The fresh regional controls share source, layout and initial noise/loss seeds with their latest-reading counterparts. The flat cases also use fresh accumulation controls on the same source; they target the two layouts that previously saturated provenance.

## Reproduction

```bash
.venv/bin/pytest -q
.venv/bin/ruff check cascade ember eval hazard mcu tests
.venv/bin/python -m cascade.demo --seed 42 --robots 8 --compact --live-disaster --distributed --region-cells 4 --latest-evidence --no-render --report outputs/demo/streaming-evidence/replay.json
# Omit --latest-evidence for the accumulation control; set --region-cells 0 for flat geometry.
```

## Source provenance

All ten reports have matching source archives and the same pre-commit source fingerprint. The CLI checks source stability across each run. Generated outputs remain local under ignored `outputs/`; archive hashes follow.

- `outputs/demo/streaming-evidence/seed7-region4-accumulate.json`; matching source archive SHA-256 `7b4b06c8f27baed8f1b510b01c25f29ef54ee2af215c217569b0921855a1197b`.
- `outputs/demo/streaming-evidence/seed7-region4-latest.json`; matching source archive SHA-256 `0260fd21bdcf2781fbbfde2009f8e518f3d373eadfd69bdbcc9a4829032bf7fd`.
- `outputs/demo/streaming-evidence/seed7-flat-accumulate.json`; matching source archive SHA-256 `8a2e7db38d920df19314b5a395b393e783c6f56808fac8955e144132419ec4dc`.
- `outputs/demo/streaming-evidence/seed7-flat-latest.json`; matching source archive SHA-256 `be57d42a986b4bcb168bcb1303acd4a537a5bed28a3bed8bd4caf290437276d8`.
- `outputs/demo/streaming-evidence/seed19-region4-accumulate.json`; matching source archive SHA-256 `2a0df61ec82e0fb3ffa779e61a9fd2c1f1cf63d327c4c8395d5985a6805e16f0`.
- `outputs/demo/streaming-evidence/seed19-region4-latest.json`; matching source archive SHA-256 `12b2f6b2235572542d678db0513f60460fe7c2414fc1ff75209287936121aecd`.
- `outputs/demo/streaming-evidence/seed42-region4-accumulate.json`; matching source archive SHA-256 `5053de7bdd8247b69946e8cc62e30cd857249a013ed63e7816c41f8306e3d95c`.
- `outputs/demo/streaming-evidence/seed42-region4-latest.json`; matching source archive SHA-256 `12fd13e627a8e87fe781ca84e88c41de7915078d1eac1b1f9cbf2c57274fedc7`.
- `outputs/demo/streaming-evidence/seed42-flat-accumulate.json`; matching source archive SHA-256 `b40972129ab66488540bba4018a1928b7b154d48c41bc7af0ee0ad0145f31bf9`.
- `outputs/demo/streaming-evidence/seed42-flat-latest.json`; matching source archive SHA-256 `6ff063e1140c50e27472e43044a553e0441ff3c7d3a18dd89b265abb71fb2b95`.
