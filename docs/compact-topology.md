# Lossless compact GBP topology — 2026-10-04

Graph endpoints and reverse-message indices now use the smallest supported unsigned width (16, 32 or 64 bits), chosen independently from validated index bounds. GBP information, precision, edge variances, update order and floating-point arithmetic remain unchanged. Physical arenas are sized from the actual aligned GBP layout plus an upper bound for reusable integer routing.

| Disclosed mission | Sites reached | Previous numeric peak | Compact numeric peak | Reduction | Allocated capacity before / after |
|---|---:|---:|---:|---:|---:|
| Seed 7, 4 robots | 4/4 | 1,361,628 B | 984,756 B | 27.68% | 1,478,864 / 1,029,330 B |
| Seed 42, 8 robots | 8/8 | 1,230,276 B | 891,588 B | 27.53% | 1,335,384 / 931,714 B |

Validation: all 473 tests pass and runtime Ruff passes. New tests cover both sides of the 65,536-entry node and directed-message boundaries, exact agreement with an int64 topology reference through masked updates and evidence aging, invalid endpoint rejection before unsigned casting, isolated cells, private per-robot buffers, exact aligned storage estimates and a 4,000-node native routing workspace with self loops.

For graphs fitting 16-bit node and message indices, raw persistent GBP storage falls from `16*N + 96*E` to `16*N + 60*E` bytes, plus arena alignment. Routing retains its existing int32 layout. The maximum CSR transition count is `2*E + N`, allowing the allocated arena capacity to fall by about 30% in these scenes.

The 200 KiB embedded gate still fails in these physical scenes. Graph storage, packet queues, factor provenance, temporary host arrays and interpreter metadata are outside this numeric arena accounting. These are disclosed development replays; they do not establish fresh held-out generalization, full physical evacuation, calibrated survival or measured MCU timing.

Reproduce from the repository root:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check cascade ember eval hazard mcu tests
.venv/bin/python -m cascade.demo --seed 7 --robots 4 --compact --live-disaster --distributed --no-render --report outputs/demo/compact-topology/seed7.json
.venv/bin/python -m cascade.demo --seed 42 --robots 8 --compact --live-disaster --distributed --no-render --report outputs/demo/compact-topology/seed42.json
```

## Seed 7

Baseline: `outputs/demo/distributed-final-reviewed/seed7_rendered.json`. Replay: `outputs/demo/compact-topology/seed7.json`.
All 40 mission behavior metrics match exactly, including final positions, route indices, costs, radio deliveries, localization, collapse response, GBP evidence and integer routing sweeps. Video/frame fields differ because the replay skips rendering.
Source archive SHA-256: `03807fd3868f0af2955328685cee1ebd48e8feb332ec3981679d908954e11a7b`.
Run source base commit: `af7b2195c6a29448bd10bc3a687a4f4125c6c0f0`; dirty-source hash: `c245f09bd27de6d423a99d7ee84a2be29efae3204fed0ece8f4ff26eb441c5fb`.

## Seed 42

Baseline: `outputs/demo/compact/seed42_distributed8_v5.json`. Replay: `outputs/demo/compact-topology/seed42.json`.
All 40 mission behavior metrics match exactly, including final positions, route indices, costs, radio deliveries, localization, collapse response, GBP evidence and integer routing sweeps. Video/frame fields differ because the replay skips rendering.
Source archive SHA-256: `7a5f1a9c279a75ecb1e0dd235fa46395c3267c53e0d9cd4c8b0a5147b88e9268`.
Run source base commit: `af7b2195c6a29448bd10bc3a687a4f4125c6c0f0`; dirty-source hash: `c245f09bd27de6d423a99d7ee84a2be29efae3204fed0ece8f4ff26eb441c5fb`.
