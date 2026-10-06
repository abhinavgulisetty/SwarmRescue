# Receiver-local range transport — 2026-10-06

`ember/range_comms.py` adds a distinct fixed-pose-epoch RSSI frame: source ID, uint16 sequence, TTL, uint32 pose epoch, explicit `RNG1` marker/count, and uint16 beacon IDs with float32 measured RSSI. Thirty-seven updates occupy 243 bytes and fit the 244-byte limit. Two zero provenance words ensure Gaussian hazard decoders reject these frames. Explicit counts reject truncation, including omitted complete updates. This is type/length validation, not link authentication or a checksum.

`RadioMedium.flood_packet` accepts an explicitly selected decoder; the default Gaussian wire format remains unchanged. The selected decoder validates a frame before radio time is reserved. Range forwarding uses the existing sensitivity, wall loss, seeded absolute-slot link loss, TTL and relay duplicate handling. Range pose epochs are independent of voxel-map epochs; a compatibility `map_epoch` property serves only as the flood identity key.

`RangeLocalizationTeam` creates separate coupled-GBP factor tables per receiver, for 3–32 robots and 1–32 beacons. Each source ingests its own quantized reading; peers ingest only decoded delivered frames. Retransmission rounds are bounded (default three, maximum eight), reuse the exact sequence IDs, and stop once every receiver has enough factors and full-rank anchor geometry. Reordered and repeated observations cannot become independent evidence. Degenerate geometry or solver failure remains explicit. New pose epochs require new teams, and wrong-epoch packets are rejected before interpreting beacon indices.

Measured pose priors and RSSI calibration remain common input assumptions. The packet carries no posterior, oracle beacon position, pose prior or wall correction. Assignment negotiation and pose-prior transport are outside this layer. The current code is a host reference; per-receiver retained numerical bytes and aggregate team bytes exclude temporary inversions, returned solutions, packet objects, metadata and runtime.

Focused tests and runtime Ruff pass. Tests cover exact wire roundtrips/MTU/TTL, all truncation lengths of a short frame, cross-type rejection, invalid fields, sequence exhaustion, wrong-epoch out-of-range IDs, no-loss equivalence to shared GBP on identical quantized observations, distinct estimates under asymmetric delivery, full partition unavailability, and unchanged likelihoods under 100 retransmissions. They also found that repeated GBP solves changed the weak beacon prior seed; the seed now remains fixed after initial multilateration, so solving again does not redefine the prior.

Physical search integration and fresh paired mission/evaluation evidence follow below. No fully distributed assignment, embedded-memory/timing, survival or final-project-completion gate is claimed here.


## Physical integration and delivery evaluation

`cascade.demo --range-gbp --range-radio` creates independent inference states using the physical team's current radio positions, wall raycasts, keyed slot/link loss and TTL forwarding. `--range-rounds` sets a bounded retransmission limit (default three, maximum eight). The host coordinator commits one-to-one matching from each receiver's local cost row. An assigned robot's target is exactly its own estimate; peer posteriors are never averaged into it. Assignment-bid transport remains unmodeled, and all receivers still start from common known measured-pose priors and RSSI calibration. The physical integration is a single fixed-pose search snapshot, not continuous distributed SLAM.

All localization broadcasts and relays are included in `radio_bytes`, with a separate `range_radio_bytes` subtotal. Per-receiver factor counts, retransmission rounds, retained numeric storage and aggregate team storage are reported. Missing ranges or nonconvergence are explicit failures; no shared-table fallback occurs. The ready test includes anchor rank, so a sufficient count of collinear sources continues retrying.

The full suite has 650 passes and one optional Cortex-M4 test skipped because temporary compiler/emulator binaries from October 4 are absent. Runtime Ruff passes. That skip does not invalidate the archived October 4 emulator checkpoint, but this run is not a new ARM execution result.

Six same-source development physical runs pass:

| Mission | Arrived | Accepted | Maximum BLE error m | Search exposure proxy | Last arrival s | Total radio bytes | Range radio bytes | Range rounds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed19-radio | 8/8 | true | 0.812973 | 658.203850 | 29.854 | 34144 | 4416 | 1 |
| seed19-shared | 8/8 | true | 0.812975 | 658.203850 | 29.854 | 29728 | 0 | 0 |
| seed42-radio | 8/8 | true | 1.151504 | 21.708731 | 36.412 | 48064 | 4416 | 1 |
| seed42-shared | 8/8 | true | 1.151505 | 21.708731 | 36.412 | 43648 | 0 | 0 |
| seed7-radio | 4/4 | true | 0.628147 | 36.683948 | 45.060 | 16880 | 720 | 1 |
| seed7-shared | 4/4 | true | 0.628144 | 36.683948 | 45.060 | 16160 | 0 | 0 |

Every receiver gets all source factors in these three layouts: 16 factors for each four-robot receiver, 64 for each eight-robot receiver. Radio and shared-table arms have identical motion/exposure within each pair; float32 wire quantization changes reported localization errors by only a few micrometres. The extra localization traffic is 720 bytes for four robots and 4,416 bytes for eight. This closes actual range delivery in these layouts, not superiority or a radio-budget-matched comparison. The shared-table controls pay no localization delivery cost.

A separate 120-case development grid uses worlds 0–19, eight robots/beacons, 0.05 m measured-pose noise, 0.45 dB RSSI noise, and packet error rates 0%, 10%, 25%, 50%, 75% and 100%. Both methods use identical float32 measurements. Complete-factor GBP is an information reference, not an equal-budget communication baseline.

| Packet error rate | Available teams | Available receivers | Mean error when solved m | 95% coverage when solved | Mean airtime bytes | Mean rounds |
| --- | --- | --- | --- | --- | --- | --- |
| 0% | 20/20 | 160/160 | 0.192447 | 96.250% | 4416.00 | 1.0 |
| 10% | 20/20 | 160/160 | 0.192447 | 96.250% | 4416.00 | 1.0 |
| 25% | 20/20 | 160/160 | 0.192447 | 96.250% | 4409.10 | 1.0 |
| 50% | 20/20 | 160/160 | 0.194287 | 96.641% | 3998.55 | 1.0 |
| 75% | 20/20 | 160/160 | 0.281545 | 94.453% | 2484.00 | 1.1 |
| 100% | 0/20 | 0/160 | Unavailable | Unavailable | 1656.00 | 3.0 |

No-loss receiver coordinates exactly equal complete-factor GBP. At 75% loss, some coordinates differ by as much as 2.327 m from complete-data estimates, even though enough factors remain to solve. Thus the minimum-count retry rule preserves availability but not complete-data precision. Full partitions retain only the source's own factors and correctly refuse localization. Reduced air bytes at high loss reflect fewer successful relays and do not establish an efficiency gain. These are development-only modeled sensor/link results, not survivor survival or deployed calibration.

Retained storage is 2,496 numeric bytes per four-robot receiver (10,000 aggregate team bytes including sequence counters), or 9,088 per eight-robot receiver (72,736 aggregate bytes). The fixed weak-prior seed adds 64/128 bytes over October 4's reported solver storage. Temporary matrices, returned results, packet objects, Python metadata/runtime and map/refinement storage remain excluded. Whole embedded memory/timing gates remain false.

## Reproduction and archives

```bash
.venv/bin/pytest -q
.venv/bin/ruff check cascade ember eval hazard mcu tests
.venv/bin/python -m eval.range_delivery --world-seeds 0:19 --loss-rates 0,0.1,0.25,0.5,0.75,1 --out outputs/eval/range-delivery-final.json
.venv/bin/python -m cascade.demo --seed 42 --robots 8 --compact --live-disaster --distributed --region-cells 4 --latest-evidence --range-gbp --range-radio --no-render --report outputs/demo/range-transport/replay.json
```

The six final mission archives and delivery-grid archive match their reports. They share precommit SHA `0fc9bfe7362f14538966dc2397853b5c7f633ec1` and dirty-source fingerprint `2d564c13050582f36994c3e037995f58f52ae7de54d0fca10446bf7be157bc41`. The delivery archive SHA-256 is `598500d6427ee0ca4e263a6be0b3ab4e6c545c072db18e0ddcfb4e06e7554f1c`.

Mission source archive SHA-256 values:

- `outputs/demo/range-transport/seed19-radio.json`: `066c43414e4223b0a52834672183b1313b3623c23519528489edccfdee731e47`.
- `outputs/demo/range-transport/seed19-shared.json`: `7a5e20ffd862a5f44e42811cd55d0f87f54f92bf5981631ea5d90a926b72940a`.
- `outputs/demo/range-transport/seed42-radio.json`: `b484ea1a5386b63b5049ca870c1535fab4d55decdc75b2728841ab69bcfad591`.
- `outputs/demo/range-transport/seed42-shared.json`: `3af1b92f355c5ac24d3c440884e35797370e3668e5be43109b8993db670b5a74`.
- `outputs/demo/range-transport/seed7-radio.json`: `97a2ffa20ccbe4741ecb6f3be750ac346a21241305c7aae3c90c581cd33de24a`.
- `outputs/demo/range-transport/seed7-shared.json`: `cbcf83676a4899312e770ed3d4fe1c073cd73c96895b4830214d5929f9b9a3f5`.
