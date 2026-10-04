# Independent hazard sensing — 2026-10-04

`cascade.demo --sensing-interval-s 2` enables experimental periodic visible hazard scans, radio exchange and replanning for unfinished distributed robots. Zero (the default) preserves geometry-triggered sensing. Initial sensing starts the clock; each geometry scan resets it, coalescing simultaneous triggers. Hazard-only scans do not rebuild the graph or advance its epoch, and stop once every robot finishes. Finished robots can still provide peer observations while others search. Reports count hazard-only refreshes and changed routes separately from collapse responses.

The existing raycasts, noisy thermal/smoke observation model, source provenance, byte budgets and private robot beliefs remain in use. No observer-truth dose influences route decisions. Latest-source-cell evidence is recommended for this experiment; cumulative evidence can exhaust its bounded identity table under repeated scans. Latest replacement is an approximate changing-field model, not calibrated temporal GBP. Regional monotone hotspot floors still cannot represent cooling fully.

## First cadence experiment

All 550 tests and runtime Ruff pass. A stationary-map sensor integration test confirms both heating and cooling update flat latest-evidence beliefs without a new epoch or growing the retained ledger. Six development missions pass, using latest evidence in both arms:

| Architecture/seed/arm | Arrived | Search exposure proxy | Last arrival s | Radio bytes | Hazard-only scans | Hazard-only changed routes |
| --- | --- | --- | --- | --- | --- | --- |
| flat42-control | 8/8 | 21.352837 | 30.004 | 40096 | 0 | 0 |
| flat42-periodic | 8/8 | 22.324910 | 36.830 | 87296 | 8 | 33 |
| flat7-control | 4/4 | 44.327471 | 32.912 | 15472 | 0 | 0 |
| flat7-periodic | 4/4 | 36.257395 | 30.564 | 19120 | 3 | 12 |
| region7-control | 4/4 | 36.683948 | 45.060 | 16160 | 0 | 0 |
| region7-periodic | 4/4 | 35.959619 | 37.148 | 23824 | 4 | 10 |

Search exposure accumulates only while a robot is unfinished; it is the existing modeled heat/smoke proxy, not tenability survival or a physical evacuation result. Two-second sensing improves seed 7 in both architectures but worsens flat seed 42 in exposure, arrival and traffic. Thus periodic sensing remains opt-in and is not a universal improvement claim. The next experiment below tests route-switch inertia instead of replacing routes at every scan.

Each JSON and matching `.source.tar.gz` is under ignored `outputs/demo/hazard-cadence/`, with the row name as filename. All six source bundles match their report SHA and dirty-source fingerprint; they share one precommit source snapshot based on `15b3626`. These are two disclosed development seeds, not a sealed evaluation. Extra scans consume additional observation draws and radio slots, so paired seeds do not guarantee identical per-scan noise between schedules. Mission duration remains 48 simulated seconds. Full embedded RAM, measured timing, deployed sensors, coupled localization and physical evacuation gates remain open.

```bash
.venv/bin/python -m cascade.demo --seed 42 --robots 8 --compact --live-disaster --distributed --latest-evidence --sensing-interval-s 2 --hazard-switch-min-gain 0 --no-render --report outputs/demo/hazard-cadence/replay.json
```

## Route-switch inertia experiment

Hazard-only replanning now compares both physical polylines against the same private belief field, integrating nearest-cell risk at at most 0.125 m sample spacing. This avoids treating a route with fewer stored waypoints as inherently safer. A proposed path needs a 15% modeled cost reduction before a switch; a shortening deadline intervention bypasses this threshold. Every geometry response bypasses the threshold. `--hazard-switch-min-gain 0` reproduces unconditional switching; the default threshold is 0.15 when sensing is enabled. Sensing itself remains disabled by default.

The guard uses belief risk only, never observer truth, temperatures, or evaluation exposure. In regional mode it uses the returned coarse regional belief field; the fine direct-reading cache still informs route refinement but is not included in the switch score. This limitation can suppress beneficial fine-scale changes. Nearest-cell line integration is a host approximation, not a calibrated exposure bound, traversability certificate, or MCU kernel. Host SciPy tree and sampling scratch are outside numeric arena accounting.

All 558 tests and runtime Ruff pass. Added checks cover low-gain detours, hot-corridor escapes, comparable sparse/dense waypoints, shortening deadline overrides and invalid thresholds. Seven fresh physical runs share one source fingerprint and pass all mission acceptance checks:

| Architecture/seed/arm | Arrived | Search exposure proxy | Last arrival s | Radio bytes | Hazard-only scans | Changed routes | Suppressed switches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat42-control | 8/8 | 21.352837 | 30.004 | 40096 | 0 | 0 | 0 |
| flat42-guard | 8/8 | 21.655160 | 33.542 | 76864 | 7 | 3 | 29 |
| flat42-naive | 8/8 | 22.324910 | 36.830 | 87296 | 8 | 33 | 0 |
| flat7-control | 4/4 | 44.327471 | 32.912 | 15472 | 0 | 0 | 0 |
| flat7-guard | 4/4 | 44.327472 | 37.372 | 23200 | 4 | 1 | 12 |
| region7-control | 4/4 | 36.683948 | 45.060 | 16160 | 0 | 0 | 0 |
| region7-guard | 4/4 | 36.487894 | 41.448 | 24496 | 6 | 1 | 14 |

In flat seed 42, inertia reduces hazard-only changed routes from 33 to 3, improves last arrival by 3.288 s, reduces search exposure by 3.0%, and reduces radio bytes by 11.9% compared with unconditional switching. It still trails the geometry-only control in exposure, arrival and radio cost. In flat seed 7, it loses the first experiment's exposure gain and arrives 4.460 s later than the control. Regional seed 7 improves arrival by 3.612 s and search exposure by about 0.5% versus its control, but uses more radio traffic. Inertia therefore addresses one failure mode and is not a universally better algorithm or a calibrated threshold. No periodic-sensing setting is promoted to the main default.

All seven final JSON reports and source archives are under ignored `outputs/demo/hazard-inertia/`. Their precommit SHA is `f8dd3a24b33f92ea3b0e7215a958d43c930a65da`; dirty-source fingerprint is `487260ee44e3efbd20107c6309dcf0291d21f180051b9c64d6e1fa93165cb46d`. Controls reproduce their initial metrics; the new-source zero-threshold seed 42 run also reproduces the original unconditional-sensing result. These remain two disclosed development seeds with different scan/noise schedules, without statistical superiority or held-out claims.

```bash
.venv/bin/pytest -q
.venv/bin/ruff check cascade ember eval hazard mcu tests
.venv/bin/python -m cascade.demo --seed 42 --robots 8 --compact --live-disaster --distributed --latest-evidence --sensing-interval-s 2 --hazard-switch-min-gain 0.15 --no-render --report outputs/demo/hazard-inertia/replay.json
```

Final archive SHA-256 values:

- `flat42-control.json`: `bafcf13e0c0e3863448e1f3122a17a1399d0b9c4f70b1d9eb2eb5322d8bec709`.
- `flat42-guard.json`: `f2a949e0ef2bdfea8efc75582a27a29423406f6e8099416cb42a82af29a8ebb4`.
- `flat42-naive.json`: `4c846533c946d20119dcee3eef625c441f84cda82b9e835de9f1ea33be77309a`.
- `flat7-control.json`: `6d68065d80879f25063bc26aa932a3af39a259ce4201ea178aac5cd2f0cb726e`.
- `flat7-guard.json`: `7ebb8381f63f0e4a003e771bdaa069f1700b729a650c496b5ad9a311a0f48d19`.
- `region7-control.json`: `ab20ce421908dcd70138f840d1ae931443d54f3fa237b1e66d54b640981946ce`.
- `region7-guard.json`: `4c06d7947e888bbff7ef9f355be79c574295b2566bf12fdd286be49adcefce2f`.
