# SwarmRescue — EMBER

**E**dge-native **M**essage-passing for **B**ellman-**E**gress **R**outing: a consensus-free rescue
swarm in which the risk map, the route plan, and the radio schedule are one fixed point.

> 📄 **The full project proposal is in [`PROPOSAL.md`](PROPOSAL.md).**

## In one paragraph

8–24 small ground robots search a procedurally generated, physically collapsed, burning
multi-storey building. Every robot's intelligence runs inside a simulated ESP32-class budget
(240 MHz, 520 KB RAM, no FPU, integer arithmetic only). There is no central coordinator, no leader
election, no global clock, and no failure detector. Robots cooperatively estimate a propagating
hazard field by Gaussian belief propagation whose prior *is* the hazard PDE, route survivors by
relaxing a linearly-solvable Bellman equation with a safeguarded neural cellular automaton, and
decide what to transmit by adjoint sensitivity — sending only the information that could change
somebody's route.

## The four claims

| | Contribution |
|---|---|
| **C1** | The hazard belief's covariance enters the routing objective **analytically** (KL-control), and one scalar's sign switches the same operator between risk-averse escort and uncertainty-seeking search. No tuned cost inflation anywhere. |
| **C2** | A ~2.5 KB int8 **safeguarded NCA** accelerates the Bellman relaxation, with a local acceptance test making divergence **impossible for any weights** (Prop. 1). Learning inside a safety-critical embedded loop, made safe by safeguarding rather than verification. |
| **C3** | **Adjoint-weighted value-of-information gossip**: the co-desirability field `z†` (same kernel, transposed graph) tells each robot which cells trajectories actually flow through, so the swarm is loud about a fire in a critical stairwell and silent about one in a dead end. New metric: **nats per byte**. |
| **C4** | **No failure detector.** Robot death and radio partition are the same event to the estimator; unrefreshed beliefs lose precision, and pessimistic routing withdraws from unwatched regions automatically. |

## Constraints honoured

- Simulation only — **no ROS, no ROS 2, no Webots, no stock scenes.**
- MuJoCo 3.x + Taichi + NumPy; PyTorch offline only, for training the NCA.
- The rubble is **not hand-placed**: it is the settled result of a simulated structural collapse.
- Secondary collapses during the demo are **causal** (thermal damage integral releases a real
  MuJoCo weld constraint), never scripted animation.
- Robot-side code is fixed-point only, arena-allocated, and cycle-budgeted; overrunning the budget
  makes a robot act on a stale field, so compute cost changes behaviour.
- Kernels are cross-validated against **real firmware** under QEMU-xtensa (ESP32) and Renode
  (Cortex-M4); 1–2 robots run that firmware live during the final demo.

## Running it

Everything below runs from the repo root against the project venv (Python 3.12).

**Look at a generated building.** Writes four views plus a contact sheet to `outputs/views/`:

```bash
.venv/bin/python -m cascade.view --seed 7                     # intact
.venv/bin/python -m cascade.view --seed 7 --collapsed         # after the blast
.venv/bin/python -m cascade.view --seed 7 --collapsed --cutaway
.venv/bin/python -m cascade.view --seed 7 --interactive       # live MuJoCo window
```

**Run the swarm traversal demo.** Writes an annotated `.mp4` to `outputs/demo/`:

```bash
.venv/bin/python -m cascade.demo --seed 7 --robots 4
.venv/bin/python -m cascade.demo --seed 7 --robots 4 --no-render   # numbers only
```

The video has two panes: a *dollhouse* shot that cuts the building off above whichever storey the
swarm is on, and a *plan* shot with the slabs removed so every robot is visible at once regardless
of storey. Robots carry coloured beacons because a 0.34 m chassis in a 24 × 18 × 12 m ruin is
otherwise a couple of pixels.

The first run for a seed simulates the collapse (~2.5 min) and caches it under `cascade/cache/`;
later runs of that seed reuse it. Editing anything that changes the structure invalidates the cache
by model hash, so the collapse re-runs.

```bash
.venv/bin/python -m pytest -q          # the test suite
```

## Status

Proposal stage, with Phase 0 implemented: floor-plan grammar, structural model, collapse by
simulation, dual-representation extraction, and a swarm traversal demo. See the 14-week plan and
risk register in [`PROPOSAL.md`](PROPOSAL.md) §7.

Known limitations of the Phase 0 demo, stated plainly because they bound what it evidences:

- **Robots do not collide with each other.** Mutual deconfliction is a routing-policy problem and
  belongs with the routing policy in Phase 1. With contacts on, a swarm sharing one doorway simply
  queues into it and the run measures the queue rather than the ruin.
- **Settled rubble is baked static.** It cannot shift, tip, or crush a robot, which makes the
  traversal tractable but means this demo says nothing about survival under settling debris.
- **The robots are teleoperated**, following routes from an offline planner with full knowledge of
  the world. None of EMBER's contributions are exercised yet; that is the point of Phase 1 onward.
- **The collapse is not energy-honest.** Peak kinetic energy exceeds the building's available
  potential energy, so the solver is injecting energy during the violent phase. The settled result
  is plausible and reproducible, but no quantitative claim should rest on the collapse dynamics
  until this is characterised.
