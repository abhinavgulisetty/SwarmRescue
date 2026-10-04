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

Search exposure accumulates only while a robot is unfinished; it is the existing modeled heat/smoke proxy, not tenability survival or a physical evacuation result. Two-second sensing improves seed 7 in both architectures but worsens flat seed 42 in exposure, arrival and traffic. Thus periodic sensing remains opt-in and is not a universal improvement claim. The next experiment will test route-switch inertia instead of replacing routes at every scan.

Each JSON and matching `.source.tar.gz` is under ignored `outputs/demo/hazard-cadence/`, with the row name as filename. All six source bundles match their report SHA and dirty-source fingerprint; they share one precommit source snapshot based on `15b3626`. These are two disclosed development seeds, not a sealed evaluation. Extra scans consume additional observation draws and radio slots, so paired seeds do not guarantee identical per-scan noise between schedules. Mission duration remains 48 simulated seconds. Full embedded RAM, measured timing, deployed sensors, coupled localization and physical evacuation gates remain open.

```bash
.venv/bin/python -m cascade.demo --seed 42 --robots 8 --compact --live-disaster --distributed --latest-evidence --sensing-interval-s 2 --no-render --report outputs/demo/hazard-cadence/replay.json
```
