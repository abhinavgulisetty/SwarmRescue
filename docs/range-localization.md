# Coupled pose/range GBP — 2026-10-04

`ember/range_gbp.py` provides a bounded host vector-factor graph. Each beacon position and robot pose is a 2D or 3D Gaussian variable. All beacon factors share the same latent robot poses, anchored by measured-pose priors. Gaussian canonical messages integrate the opposite variable's cavity uncertainty and remove this factor's previous message before updating. Nonlinear relinearization uses damped GBP and a joint-objective line search. Capped or stalled solves report nonconvergence; the physical search rejects them explicitly.

Each source/beacon has one likelihood slot and a signed-64-bit sequence watermark. Newer readings replace the old likelihood; equal or older observations cannot be double-counted. RSSI factors retain Gaussian log-range likelihoods, instead of treating exponentiated distances as additive Gaussian noise. Direct Gaussian range factors remain supported. A fixed pose epoch is required; moving anchors require a new graph. There is no cross-session replay protection or transmitted range-packet protocol.

`localize_and_assign(..., range_gbp=True)` replaces the old Gauss–Newton search solver, and `cascade.demo --range-gbp` exercises it in distributed physical missions. The default remains the old localizer. Simulation search uses known instantaneous robot positions with a small 1e-6 m² prior covariance; the benchmark instead supplies genuinely noisy measured anchors with their 0.05 m standard-deviation priors. Every method receives identical RSSI and measured-pose draws in the benchmark.

## Validation

All 586 tests and runtime Ruff pass. Tests compare linearized means with an independent full joint information solve, exact tree marginals, nonlinear RSSI MAP estimates with independent SciPy least squares, 3D recovery, increased covariance under uncertain anchors, shared-pose updates, bounded likelihood replacement, replay, invalid inputs, degeneracy, iteration caps and search integration. A monkeypatch test proves GBP search does not invoke the old Gauss–Newton solver.

The first Gaussian-distance prototype converged in only 14/15 smoke cases. Its report and archive remain `outputs/eval/localization-gbp-smoke.{json,source.tar.gz}`. The log-range revision converged in 15/15 smoke cases and all 600 final paired development cases: worlds 0–199 × noise seeds 0–2, eight robots and eight beacons per case. RSSI noise levels are 0.15, 0.45 and 1.5 dB; pose noise is 0.05 m. No held-out seeds were used.

| Metric | Classical | Coupled range GBP |
| --- | --- | --- |
| Mean beacon error | 0.379119 m | 0.303076 m |
| Mean case maximum error | 0.699230 m | 0.564050 m |
| 95% ellipse coverage | 91.729% | 94.833% |
| Converged cases | Not diagnosed by old solver | 600/600 |

The paired world mean-error difference is −0.076043 m; the world-cluster bootstrap 95% interval is [−0.082492, −0.069352] m. This is about a 20.1% improvement on this synthetic development distribution. Loopy GBP covariance is approximate, and this empirical coverage result does not establish real sensor calibration, wall-loss/bias robustness, or survivor survival.

Six same-source physical missions pass:

| Mission | Arrived | Maximum BLE error m | Search exposure proxy | Last arrival s | Hazard radio bytes | Localization retained numeric bytes |
| --- | --- | --- | --- | --- | --- | --- |
| seed19-control | 8/8 | 0.797948 | 652.876184 | 29.380 | 28992 | 0 |
| seed19-gbp | 8/8 | 0.812975 | 658.203850 | 29.854 | 29728 | 8960 |
| seed42-control | 8/8 | 1.168789 | 21.652159 | 30.562 | 38496 | 0 |
| seed42-gbp | 8/8 | 1.151505 | 21.708731 | 36.412 | 43648 | 8960 |
| seed7-control | 4/4 | 0.616610 | 36.683948 | 45.060 | 16160 | 0 |
| seed7-gbp | 4/4 | 0.628144 | 36.683948 | 45.060 | 16160 | 2432 |

Physical gains are mixed: seed 42's BLE error improves slightly but last arrival is 5.850 s later, search exposure and hazard traffic rise; seed 19's error/exposure also rise slightly. Seed 7 has the same motion and exposure. Thus the new localizer remains opt-in despite the component error improvement. Hazard traffic excludes localization delivery because search still receives a shared RSSI table.

## Scope and remaining gates

This is joint pose/beacon inference, not pose/hazard coupling, odometry/IMU fusion, deployed sensor calibration or fully distributed localization. Each engine can own a receiver's delivered factors, but the physical integration currently supplies the shared measurement table without loss-limited BLE-factor transport or independent receiver assignments. No beacon truth initializes the solver; its starting seed is algebraic multilateration of received ranges. The final shared-table assignment remains unchanged.

Retained numerical storage is 2,432 bytes for four robots/four beacons and 8,960 bytes for eight/eight. It excludes temporary matrix inversions, line-search arrays, returned estimates, host metadata and runtime, and is not allocated in the routing arena. Whole embedded RAM and timing gates remain false. Default damping/caps and the near-zero simulation pose prior are reference choices, not calibrated deployment settings. Geometry-only hazard sensing remains the default. The full project is not complete.

## Reproduction and source evidence

```bash
.venv/bin/pytest -q
.venv/bin/ruff check cascade ember eval hazard mcu tests
.venv/bin/python -m eval.localization --world-seeds 0:199 --noise-seeds 0:2 --out outputs/eval/localization-gbp-final.json
.venv/bin/python -m cascade.demo --seed 42 --robots 8 --compact --live-disaster --distributed --region-cells 4 --latest-evidence --range-gbp --no-render --report outputs/demo/range-gbp/replay.json
```

All seven final source bundles match their reports. They share precommit SHA `2c3d7cc37cf929d7639235c2dcc8a16bf628a2eb` and dirty-source fingerprint `972988baa8634ae9918782ad956760b0d5045ef547cf76cb49a2b1d8503607e6`.

Archive SHA-256 values:

- `outputs/demo/range-gbp/seed19-control.json`: `73a7412ede8e2c04ba383f4002823a328debf57dc6848692c934c3bb45c346a4`.
- `outputs/demo/range-gbp/seed19-gbp.json`: `20f90aad8427aab640d77c992c422509945009b3ca48b8d09adf763f7dbb81ba`.
- `outputs/demo/range-gbp/seed42-control.json`: `be7864f8d100e9d205c49324037d9fcc35af870838b5a096aba53e1cf9d85580`.
- `outputs/demo/range-gbp/seed42-gbp.json`: `cbc7c7d8c1fbc096f3e210fd89d8e6fcb421a97c52d4c5fb36f473136114737b`.
- `outputs/demo/range-gbp/seed7-control.json`: `09faed2c1e57423d7981be00940ad09312fad5dc0c38890561498a9ccd736f1f`.
- `outputs/demo/range-gbp/seed7-gbp.json`: `980a618581b9174c41793e8868944f41a14a24680be1a0bae73efa88af3fc85b`.
- `outputs/eval/localization-gbp-final.json`: `a06ca3afa8bafa84b50d9ac02ed186d4f6b6a7a2a66b1cc491698e62689666a3`.


The shared-table transport boundary above describes the October 4 checkpoint.
October 6 adds opt-in actual range delivery and receiver-local estimates; see the
[range transport report](range-transport.md). Host matching and shared pose
priors still remain outside a fully distributed search claim.
