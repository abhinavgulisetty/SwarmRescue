# EMBER: Edge-native Message-passing for Bellman-Egress Routing

### A consensus-free rescue swarm in which the map, the plan, and the radio protocol are one fixed point

**Track:** Simulation only (no hardware). **Team size:** 1–3. **Duration:** ~14 weeks.
**Stack:** MuJoCo 3.x + Taichi + NumPy, fixed-point MCU sandbox, QEMU/Renode firmware validation.
**Explicitly excluded:** ROS, ROS 2, Webots, and any stock demo scene.

---

## 1. Title

**EMBER: Edge-native Message-passing for Bellman-Egress Routing — physics-informed, consensus-free swarm routing on 520 KB microcontrollers under propagating hazards.**

One-sentence thesis:

> If risk mapping, route planning, and communication scheduling are all expressed as *local message passing toward a single fixed point*, then decentralization, fault tolerance, self-healing, and microcontroller feasibility stop being separate engineering problems and become consequences of one mathematical object.

---

## 2. Problem Statement & Gap Analysis

### 2.1 The problem

A multi-storey building has partially collapsed and is on fire. A swarm of 8–24 small ground robots must, with **no central coordinator, no global clock, no reliable radio, and no compute beyond an ESP32-class MCU**:

1. cooperatively estimate a **time-varying 3-D hazard field** (heat, smoke toxicity, imminent secondary collapse) that they can only sample sparsely and noisily;
2. localize survivors from **degraded BLE RSSI**, which is nearly useless from a single robot;
3. continuously maintain **two different routes** per survivor — a *search* route for robots to reach them and an *egress* route to lead them out — where "best" means *most survivable*, not shortest;
4. keep working while robots are crushed, cooked, or radio-isolated, and while the hazard invalidates the map faster than any replanning cycle.

The hard part is not path planning. The hard part is that **the objects that must be shared are too big for the radio, the objects that must be stored are too big for the RAM, and the world changes faster than any plan-then-execute loop**. Every one of those three constraints is usually relaxed away in the literature. We relax none of them.

### 2.2 Why the obvious approaches fail

| Approach | Why it breaks here |
|---|---|
| Centralized A\*/D\*-Lite with a shared map | Needs a coordinator and full-map bandwidth; and even given an *oracle* map it optimizes the wrong objective — the shortest route through a stairwell is the route through a chimney. |
| Flooding gossip of occupancy grids | Bandwidth grows with map size × robots; ESP-NOW gives ~244 B payloads at ~250 kbps shared. A 3-D risk grid never fits. |
| ACO / pheromone stigmergy | Pheromone fields have no uncertainty semantics, converge on the order of the mission length, and have no principled response to a *propagating* hazard front. |
| Tabular / plain deep Q-learning | Non-stationary hazard breaks the MDP assumption; a policy net large enough to generalize does not fit in 520 KB without an FPU. |
| GNN/Transformer decentralized MAPF | State of the art for *this class of problem* (see G1) but assumes GPU-class inference, float32, and benign/static obstacles. |

### 2.3 Gap analysis (four gaps, one empty intersection)

**G1 — Learned decentralized planners assume a datacenter on every robot, and a static world.**
Li, Gama, Ribeiro & Prorok's GNN path planner [1] is the canonical result that *learned local communication* can approach centralized multi-agent path finding quality, and LPAC [2] extends the perception–action–communication loop to coverage. Both train by imitating a centralized expert that presumes global knowledge, run float32 models of megabyte scale, and evaluate in static or slowly-varying workspaces. Nothing in this line addresses (a) a hazard field that *propagates and destroys the graph mid-episode*, or (b) a 520 KB / no-FPU inference budget. Knowledge distillation of such a policy to an MCU is not a matter of quantization: the *expert itself* is undefined when the map is unknowable.

**G2 — Distributed Gaussian belief propagation is used for geometry, not for hazard, and treats bandwidth as nearly free.**
Ortiz, Evans & Davison [3] make the case for GBP as *the* algorithm for distributed inference on emerging decentralized hardware — asynchronous, local, robust to lost messages. Patwardhan, Murai & Davison [4] instantiate it for collaborative multi-robot planning with collision factors. In both, the factor graph encodes *static geometry and kinematics*. No work embeds a **physics-driven hazard PDE as the factor-graph prior**, and no work schedules GBP messages by **how much they would change a downstream routing decision** — messages are sent when beliefs change, not when decisions would change.

**G3 — Neural cellular automata regrow shapes, not policies, and come with no guarantees.**
NCA replaced hand-written CA rules with gradient-trained local updates and are celebrated precisely for *regeneration* after damage [5]; recent work pushes toward universal computation with NCA [6]. But NCA have no control-theoretic semantics and no convergence guarantee — the two things a rescue robot needs most. No NCA has been tied to a **linearly-solvable Bellman fixed point**, and none has been deployed to a microcontroller with a proof that it cannot diverge.

**G4 — TinyML knows how to run one feedforward pass, not an iterative distributed inference.**
The microcontroller-quantization literature [7] is comprehensive on quantizing and scheduling *feedforward CNN inference*. It is silent on the regime we need: an **iterative fixed-point computation in fixed-point arithmetic**, where quantization error interacts with the contraction rate and a stalled iterate is a safety event. Meanwhile the applied-swarm-robotics community names exactly our two constraints — onboard compute and communication hardware/protocols — as the field's blocking limitations, and criticizes the gap between swarm robotics' search-and-rescue ambitions and its abstracted benchmarks [8].

**The empty intersection.** Nobody has built a system where **hazard inference, risk-sensitive routing, and radio scheduling are the same fixed-point computation, executed in fixed-point arithmetic on microcontroller-class hardware, with a convergence guarantee that survives arbitrary message loss and robot death.** That is EMBER.

### 2.4 References

1. Q. Li, F. Gama, A. Ribeiro, A. Prorok. *Graph Neural Networks for Decentralized Multi-Robot Path Planning.* IROS 2020. arXiv:1912.06095.
2. S. Agarwal, R. Muthukrishnan, W. Gosrich, V. Kumar, A. Ribeiro. *LPAC: Learnable Perception–Action–Communication Loops with Applications to Coverage Control.* IEEE Transactions on Robotics, vol. 41, pp. 5986–6005, 2025. arXiv:2401.04855.
3. J. Ortiz, T. Evans, A. J. Davison. *A Visual Introduction to Gaussian Belief Propagation.* arXiv:2107.02308, 2021.
4. A. Patwardhan, R. Murai, A. J. Davison. *Distributing Collaborative Multi-Robot Planning with Gaussian Belief Propagation.* IEEE Robotics and Automation Letters, vol. 8, no. 2, pp. 552–559, Feb. 2023. arXiv:2203.11618.
5. A. Mordvintsev, E. Randazzo, E. Niklasson, M. Levin. *Growing Neural Cellular Automata.* Distill, 2020. doi:10.23915/distill.00023.
6. G. Béna, M. Faldor, D. F. M. Goodman, A. Cully. *A Path to Universal Neural Cellular Automata.* arXiv:2505.13058, 2025.
7. H. A. Abushahla, D. Varam, A. J. N. Panopio, M. I. AlHajri. *Neural Network Quantization for Microcontrollers: A Comprehensive Survey of Methods, Platforms, and Applications.* arXiv:2508.15008, 2025.
8. M. Kegeleirs, M. Birattari. *Towards Applied Swarm Robotics: Current Limitations and Enablers.* Frontiers in Robotics and AI, 2025. doi:10.3389/frobt.2025.1607978.
9. E. Todorov. *Linearly-solvable Markov decision problems* (NIPS 2006) and *Efficient computation of optimal actions* (PNAS, vol. 106, no. 28, pp. 11478–11483, 2009). — the LMDP/KL-control basis of §6.2.
10. ISO 13571:2007, *Life-threatening components of fire — Guidelines for the estimation of time to compromised tenability in fires*, building on D. Purser's Fractional Effective Dose model — the basis of our survivability metric (§8).

---

## 3. Novel Contributions

Four contributions. None is a tweak of an existing method; each is independently testable, and each has a stated falsification criterion in §8.4.

---

### C1 — Uncertainty-to-risk coupling that is *derived*, not tuned: the belief covariance enters the routing objective analytically.

Every risk-aware planner we know of inflates costs by a hand-tuned heuristic ("add 2σ to uncertain cells"). We show that under a KL-control (linearly-solvable MDP) formulation of egress routing, the hazard belief's **variance enters the Bellman operator in closed form**, and that a *single scalar* β continuously interpolates between two behaviours the swarm needs simultaneously:

- **β > 0 (pessimistic):** the route a robot takes when *escorting a survivor* — avoid cells whose danger is uncertain.
- **β < 0 (optimistic):** the route a robot takes when *searching* — uncertainty becomes attractive, giving directed exploration.

Both fields are computed by **the same kernel with one changed constant**, so exploration and escort are not two subsystems but two evaluations of one operator. To our knowledge no prior work derives explore-vs-exploit in rescue routing as a sign flip inside a risk measure (§6.3).

### C2 — Safeguarded Neural Cellular Automaton for Bellman fixed points: learned acceleration with a *provable* impossibility of divergence.

We train a ~2.5 KB int8 NCA whose fixed point is the solution of the linear Bellman equation, and pair it with a **purely local acceptance test** that falls back to a Jacobi step whenever the learned step fails to reduce a local residual. Because plain Jacobi on this system is a contraction in the ∞-norm, the safeguarded iteration is **guaranteed to converge to the unique correct desirability field for any weights θ**, while the learned operator can only improve the *rate* (Proposition 1, §6.4). The safeguard requires no global synchronisation, so the accelerated iteration is still a legal cellular automaton.

This is the missing piece for G3: it makes a learned local rule *safe enough to steer a rescue robot*. It also inverts the usual TinyML story — instead of distilling a big policy into a small net, we distill a **solver's convergence rate** into a small net, which is why 2.5 KB suffices.

### C3 — Adjoint-weighted value-of-information gossip: robots stay silent about hazards that would not change anyone's route.

We compute a **co-desirability (adjoint) field** `z†` using the *same NCA kernel run on the transposed graph*, giving each robot an O(1)-per-cell estimate of `∂ log z / ∂ q(x)` — the routing sensitivity of every cell. Transmission priority for a candidate GBP message is then

```
priority(x) = routing_sensitivity(x) × KL_gain(x)      [nats of decision-relevant information per byte]
```

and a token bucket sends the top-k under a hard byte budget. The consequence is a swarm that is loud about a fire in a stairwell everyone depends on and *completely silent* about a fire in a dead-end room. We claim this specific combination — the adjoint of a KL-control desirability field used as the scheduling priority for Gaussian belief-propagation messages, on a microcontroller — is new; VoI scheduling and adjoint sensitivity each have long histories, and we are precise about which part is the contribution.

We additionally propose **nats-per-byte** (KL reduction achieved at the receiver per byte transmitted) as the correct efficiency metric for swarm communication, in place of raw throughput.

### C4 — No failure detector: robot death and stale information are handled by the inference model itself.

There is no heartbeat protocol, no leader election, no membership consensus, and no explicit failure detection anywhere in EMBER. Time-factors in the factor graph inflate the variance of any belief that stops being refreshed, so a dead robot's contribution **decays smoothly to an uninformative prior**, and (via C1) a pessimistic route automatically starts avoiding the region nobody is watching. Robot failure and radio partition are *the same event* to the estimator, differing only in duration.

We claim this as a contribution in its own right: it converts the standard hardest part of decentralized systems engineering into a property that follows from the choice of prior.

---

## 4. Simulation Environment Design

The environment, **Cascade**, is written from scratch. There is no stock scene, and the rubble is not hand-placed — it is the settled output of a simulated structural failure.

### 4.1 Physics engine choice

**MuJoCo 3.x** via the official `mujoco` Python bindings.

*Why:* deterministic and seedable; a contact model good enough that a wheeled robot on rubble behaves plausibly; runtime-toggleable `equality` constraints (which is exactly the primitive we need for progressive collapse); native `hfield` terrain; scene construction from MJCF XML that we generate programmatically; and no C++ build step for students. PyBullet is a reasonable fallback (noted in the risk register); Isaac Sim is rejected as too heavy for a laptop demo, and Brax/MJX is held in reserve only for batched offline NCA training.

### 4.2 World generation — a five-stage procedural pipeline

`cascade.generate(seed) -> (MJCF, voxel_graph, hazard_init, survivor_spawns)`

**Stage 1 — Floor-plan grammar.** Per storey (3–5 storeys), a recursive binary space partition with stochastic aspect-ratio and minimum-area constraints yields rooms; a corridor spine is carved by a Steiner-tree approximation over room centroids. Doors are placed on shared walls with probability decaying in wall length. A small L-system places **stairwells, elevator shafts, and service ducts** as vertical connectors — these matter enormously later, because they are also the chimneys.

**Stage 2 — Structural graph.** Columns, beams, and slabs are instantiated as MuJoCo bodies joined by `equality/weld` constraints, forming a load-bearing graph. Each member carries a mass, a yield threshold, and a thermal mass.

**Stage 3 — Collapse by simulation, not by hand.** We sample `k` initiating failures from a spatial damage kernel (e.g. a blast centroid or a fatigue seed), deactivate those welds, and **run MuJoCo forward for 5–10 s of wall-clock physics.** Slabs pancake, beams rotate, debris piles form. The rubble field is *emergent and physically consistent* — mass is conserved, piles have angles of repose, voids exist under leaning slabs. The settled state is cached as an MJCF `keyframe` keyed on the seed, so every experiment is exactly reproducible without re-simulating.

> This stage is the direct answer to "no stock environments." The geometry a robot must traverse was produced by a structural collapse, not by an artist.

**Stage 4 — Dual representation extraction.** The settled scene is rasterized into (a) a MuJoCo `hfield` for fine traversability, (b) large surviving slabs kept as independent movable bodies (a robot can be blocked, tipped, or crushed by them), and (c) a **voxel connectivity graph** `G = (V, E)` at ~0.4 m resolution, annotated with slope, step height, and squeeze width. `G` is ground truth for evaluation and is *never* handed to the robots.

**Stage 5 — Population.** Survivors and robot spawns are placed under reachability and difficulty constraints, and the initial fire seed is placed to guarantee at least one *non-trivial routing dilemma*: a scenario in which the shortest egress route and the most survivable egress route differ by more than a factor of 1.5 in expected dose. Seeds that fail this check are rejected, so the benchmark cannot be won by a shortest-path planner.

**Train/test hygiene.** Seeds `0–199` are the development set; seeds `1000–1049` are a **held-out test set opened only for the final demo and the paper's tables**. NCA weights are trained exclusively on the development set. Cross-seed generalization is a headline result, not an afterthought.

### 4.3 Dynamic hazard model

Three coupled physical processes on a 3-D lattice, implemented in **Taichi** (JIT to CPU or GPU) at 10 Hz, interleaved with MuJoCo at 500 Hz.

**(a) Combustion — reaction–diffusion–advection with Arrhenius kinetics.** Fields: temperature `T`, fuel load `F`.

```
∂T/∂t = ∇·(κ ∇T) − u·∇T + H·R(T,F) − h(T − T∞)
∂F/∂t = −R(T,F),        R(T,F) = A · F · exp(−Eₐ / (R_g T))
```

Fuel load is spatially heterogeneous and derived from the floor-plan semantics (an office is not a stairwell), so the fire front is anisotropic and its shape carries information about the building.

**(b) Smoke and toxicity — buoyant advection of a passive scalar with a stack effect.** Smoke concentration `C` obeys advection–diffusion under a velocity field `u`. Rather than solve full Navier–Stokes, we compute `u` from a **1-D buoyancy-pressure model along each vertical connector** (stairwell, shaft, duct) driven by the temperature difference between the shaft and ambient, coupled to horizontal corridor flow by mass conservation.

This one modelling choice is what makes the routing problem genuinely hard and genuinely realistic: **smoke climbs the stairwells much faster than it crosses the corridors, so the topologically shortest egress route is systematically the most lethal one.** The environment does not merely contain a hazard; it contains a *trap that a shortest-path planner walks straight into*.

**(c) Progressive structural degradation — the live hazard injection is causal, not scripted.** Each structural member accumulates thermal damage `D = ∫ g(T(x,t)) dt`. When `D` crosses its threshold, its `equality` constraint is released at runtime (`data.eq_active`), and MuJoCo does the rest: a beam fails, a slab drops, a corridor closes, and possibly a robot is crushed. New rubble is re-rasterized into the `hfield` incrementally.

> This satisfies the "unplanned hazard injected live" requirement in the strongest available sense. The demo operator can also *cause* a collapse interactively (click a column to weaken it), but the collapse itself is always the physics engine's answer, never a scripted animation. Nothing in the robots' software is aware that a "hazard injection event" exists as a category.

**Ground truth is never observable.** `T`, `C`, `F`, and `D` exist only in the simulator. Robots see the noisy, quantized, sparse samples of §4.4.

### 4.4 Survivors and sensor simulation

**Survivors.** A social-force model with a fear-gradient term (herding toward crowds, flight from perceived heat/smoke), plus a fraction of immobile/trapped survivors. Each survivor accumulates a tenability burden per ISO 13571 [10], which we implement with its three distinct fractions rather than one lumped number:

- **FED_asphyxiant** — a time-integrated *dose* from the asphyxiant gases carried by `C` (CO, HCN, with CO₂-driven hyperventilation as a multiplier on uptake rate);
- **FEC_irritant** — a *concentration*-based fraction for sensory/upper-airway irritants, which acts immediately rather than by accumulation and therefore degrades walking speed before it incapacitates;
- **FED_heat** — convected and radiant thermal dose from `T`.

Incapacitation is `max(FED_asph + FED_heat, FEC_irr) ≥ 1`. Keeping the fractions separate is not pedantry: the irritant term is what slows a survivor down mid-escape, which couples survivor *speed* to the hazard field and so makes the egress route's length and its toxicity interact rather than add independently.

> Stated honestly, per the standard: the unity threshold corresponds to compromised tenability for roughly half the exposed population, not for all of it. "Survival rate" in §8 therefore means *fraction of survivors kept below the median-susceptibility tenability limit* — a defensible fire-safety-engineering quantity rather than an arbitrary game score, and one giving every survivor an individually meaningful clock.

**BLE beacons.** Each survivor emits an advertisement; a robot at distance `d` measures

```
RSSI = P_tx − 10 n log₁₀(d) − W·(walls crossed) + X_σ ,   X_σ ~ N(0, σ²) log-normal shadowing
```

with wall count obtained by **raycasting the actual MuJoCo geometry**, plus a multipath term and 1 dB quantization. Single-robot localization from this is deliberately terrible (tens of metres). Multi-robot fusion is the only thing that works — and, crucially, it is not a separate subsystem: **an RSSI reading is just another factor on the same factor graph** (§6.1).

**Robot sensors**, all quantized to plausible ADC depths:

| Sensor | Model | Rate |
|---|---|---|
| 2-D lidar, 16–32 beams | MuJoCo raycast + range noise + dropouts on specular/hot surfaces | 8 Hz |
| Thermal array (AMG8833-class) | 4×4 or 8×8 coarse `T` samples with a blur kernel and a ±2.5 °C bias | 10 Hz |
| CO / gas sensor | scalar sample of `C` at the robot, with first-order lag and drift | 2 Hz |
| IMU | bias random walk + noise | 100 Hz |
| Wheel odometry | integrated with slip that increases on rubble | 50 Hz |

**Robots die**, for reasons: crushing (contact impulse over threshold), thermal kill (`T > T_crit` sustained), battery depletion (with motion- and *compute*-dependent draw), and immobilisation (wedged in rubble). Death is not a coin flip administered by the experiment harness; it is a physical outcome, which is what makes the resilience results meaningful.

### 4.5 Edge-device simulation — three tiers, with cross-validation

This is the requirement most projects fake. Our approach is to make the cheap tier *behaviourally* honest and then **calibrate it against real firmware**.

**Tier A — the fixed-point MCU sandbox (used for all 8–24 robot experiments).**

Every line of robot intelligence is written against a restricted API, `mcu.ops`, and is subject to three simultaneously enforced budgets:

1. **No floating point.** All robot-side arithmetic uses a `Q16.16` fixed-point type with explicit shifts and saturation. Inside any function marked `@on_mcu`, encountering a Python float raises. Hazard beliefs are additionally kept in a **log domain with a shared exponent** so that precision-domain quantities stay representable in 32-bit integers.
2. **Memory: 520 KB, accounted.** Robot state lives in an `MCUArena(520 * 1024)` bump/pool allocator. There is no Python object allocation on the hot path; over-allocation raises `MCUOutOfMemory`. We log RAM high-water mark per robot per run and report it as a first-class result.
3. **Cycles: 240 MHz, accounted, and consequential.** Rather than trace Python bytecode (slow and inaccurate), each `mcu.ops` primitive is *vectorized in NumPy for speed but analytically costed*: a length-`n` MAC over int32 charges `n × cost_mac` cycles from an Xtensa LX6-calibrated table (int mul 1, int div ≈ 30, softfloat 50–100, load/store with cache-miss penalty). Each robot receives a budget of `240e6 × Δt` cycles per control tick; **if a robot overruns, its new action is not applied and the previous action is repeated.** Compute cost therefore changes robot behaviour, which is the entire point of the constraint. A greedy algorithm that is too slow visibly drives into a wall.

**Tier B — real firmware, real cycle counts (validation).** The GBP message kernel, the safeguarded NCA sweep, and the policy softmax are additionally implemented in **fixed-point C** and built for (i) ESP32 under **QEMU-xtensa** as shipped with ESP-IDF (`idf.py qemu`), and (ii) a Cortex-M4 at comparable clock under **Renode**, using the hardware cycle counter and the real `.bss`/heap map. We replay a few hundred recorded ticks through both and report the **Tier-A-vs-Tier-B calibration error** on cycles and RAM.

> Honest limitation, stated up front: full 24-robot firmware co-simulation in real time is not feasible, and we will not claim it. Tier B validates that the Tier-A cost model is not a fiction, on the kernels that dominate the budget.

**Tier C — live co-simulation for the demo.** During the final demo, 1–2 robots run their *actual C firmware* inside QEMU/Renode, exchanging packets with the Python simulator over a socket, while the rest use Tier A. Those robots are visually indistinguishable in behaviour, which is the claim we want to be able to make on stage: **this literally runs on an ESP32.**

---

## 5. System Architecture

### 5.1 Per-robot stack (all of it inside the 520 KB / 240 MHz budget)

```
┌──────────────────────────────────────────────────────────────────┐
│ L5  Actuation      differential-drive controller, traction-aware │
├──────────────────────────────────────────────────────────────────┤
│ L4  Behaviour      role arbitration from field values only:      │
│                    SEARCH (β<0 field) | ESCORT (β>0 field)       │
│                    | RELAY (hold connectivity) — no negotiation  │
├──────────────────────────────────────────────────────────────────┤
│ L3  Routing        safeguarded NCA sweeps → z⁻, z⁺, z†           │
│                    policy = local softmax over neighbour z       │
├──────────────────────────────────────────────────────────────────┤
│ L2  Inference      Gaussian BP over multi-resolution hazard      │
│                    octree + survivor range factors               │
├──────────────────────────────────────────────────────────────────┤
│ L1  Comms          adjoint-weighted VoI gossip, token bucket     │
├──────────────────────────────────────────────────────────────────┤
│ L0  Sensing        quantized lidar/thermal/gas/IMU/odometry      │
└──────────────────────────────────────────────────────────────────┘
```

Two structural properties are worth naming. First, **L4 reads nothing but field values** — role selection is a threshold on locally computed quantities, so there is no task-allocation protocol to deadlock. Second, **L1 depends on L3** (the adjoint field decides what to transmit), which is the unusual loop that makes C3 possible: the router tells the radio what matters.

### 5.2 Memory budget (per robot, of 520 KB)

| Component | Sizing | Bytes |
|---|---|---:|
| Hazard belief octree (η, Λ) | ~1500 active cells × 2 × Q16 + flags | ~12 KB |
| Desirability fields `z⁻`, `z⁺`, `z†` | 3 × 1500 × Q16 | ~18 KB |
| NCA weights + hidden channels | int8 weights (~2.5 KB) + 4 hidden ch. | ~11 KB |
| Survivor tracks | 32 tracks × (pose, Λ, FED estimate, TTL) | ~4 KB |
| Lidar / thermal ring buffers | | ~6 KB |
| Comms TX/RX rings + provenance TTLs | | ~16 KB |
| Local pose graph / odometry | | ~6 KB |
| Stack, libc, scratch | | ~40 KB |
| **Total** | | **≈ 113 KB** |

Roughly 4× headroom. This is deliberate: the headroom is what lets us *raise* resolution when a robot is in a dense region, and it means the budget claim survives the inevitable implementation slippage. The octree is the mechanism — fine cells near the robot, coarse cells far away, with a fixed cell ceiling and LRU eviction of distant subtrees.

### 5.3 Communication protocol

Deliberately austere: **there is exactly one message type, and it is a Gaussian.**

- **PHY model:** range-limited with **material-aware attenuation computed by raycasting the MuJoCo geometry** between transmitter and receiver (wall count → excess path loss). Link failure is therefore *spatially structured and correlated with the building*, not i.i.d. dropout. This matters: i.i.d. dropout is what most swarm papers simulate, and it is far more benign than reality, because independent losses average out while a concrete wall does not.
- **MAC:** slotted CSMA with collisions and backoff, ~250 kbps shared, 1 % ambient packet error, 244 B max payload (ESP-NOW-realistic).
- **Packet:** a header (sender, seq, TTL, provenance bits) plus up to ~8 belief updates of the form `(cell_id: u24, η: Q16, Λ: Q16)` — 24–40 B typical, well inside one MTU.
- **No** leader election, membership service, heartbeat, synchronized round, or global clock. A robot that receives nothing simply keeps iterating on a belief whose variance is growing.

**Data incest / rumour double-counting** is a real hazard in decentralized fusion and we address it explicitly rather than hoping: messages carry provenance bits and a TTL, fusion divides out the common prior (channel-filter style), and where provenance is ambiguous we fall back to **covariance intersection**, which is conservative by construction. Robust (Huber-style) factors bound the damage from outlier sensor readings and from a malfunctioning robot broadcasting nonsense.

### 5.4 Handling the edge constraints

| Constraint | Mechanism |
|---|---|
| No FPU | Everything in Q16.16; hazard beliefs in log domain with shared exponent; NCA in int8 with per-channel scales |
| 520 KB | Multi-resolution octree with hard cell ceiling + LRU eviction; arena allocation; no dynamic Python objects on hot paths |
| 240 MHz | Anytime iteration: `z` fields are *always usable*, just less converged. Cycle budget per tick caps sweeps; a starved robot acts on a stale-but-valid field rather than blocking |
| Intermittent comms | GBP is a fixed-point iteration — a lost message delays convergence and never corrupts it; time-factors decay unrefreshed beliefs (C4) |
| Robot death | No failure detector needed (C4); the field re-relaxes locally around the gap |

The "anytime" property deserves emphasis because it is what makes the compute budget survivable: EMBER has **no plan-then-execute cycle to miss a deadline on**. There is only a field that is continuously being improved and continuously being read.

---

## 6. Core Algorithms

### 6.1 Layer 1 — Physics-informed distributed Gaussian belief propagation over the hazard field

Let `x ∈ V` index cells of the robot's octree, and let the estimated variable be **log hazard intensity** `η(x)` (log domain keeps it positive and makes multiplicative sensor noise additive). The swarm's joint belief is represented by a factor graph whose factors are:

- **Measurement factors** `f_z(η_x)`: thermal-array, gas, and lidar-derived observations, linearized about the current estimate; Huber-robustified.
- **Physics factors** `f_d(η_x, η_x')`: *the discretized hazard PDE is the prior.* One implicit-Euler step of the advection–diffusion operator of §4.3(b) becomes a Gaussian pairwise factor whose precision is **anisotropic**, aligned with the robot's estimate of the stack-effect velocity field. This is the mechanism by which a robot that has never been upstairs can nonetheless *predict* that the stairwell is filling with smoke.
- **Time factors** `f_t(η_x^k, η_x^{k+1})`: a random-walk prior between consecutive time slices, with process noise `Q`. These are what implement C4: an unrefreshed belief loses precision at a known rate.
- **Range factors** `f_r(η_survivor_pos, robot_pose)`: BLE RSSI, linearized each iteration (GBP-as-incremental-smoothing). Survivor localization is *not a separate module* — it is more factors on the same graph, fused by the same message passing, transmitted by the same packets.

All factors are Gaussian, so every message is a pair `(η, Λ)` — an information vector and a precision. Variable-to-factor and factor-to-variable updates are the standard GBP updates:

```
m_{x→f} = Σ_{g ∈ N(x)\f} m_{g→x}
m_{f→x} = marginalize_{V(f)\x} ( f ⊗ ⨂_{y ∈ V(f)\x} m_{y→f} )
belief(x) = ⨂_{f ∈ N(x)} m_{f→x}
```

with all products/sums in the canonical (information) parameterization, so a message *is* an addition of `(η, Λ)` — two Q16 integer adds per cell. That is the whole reason this fits on an MCU.

**Why GBP and not consensus averaging.** GBP is exact on trees and empirically well-behaved on loopy graphs; more importantly it is **asynchronous by construction** and degrades gracefully under lost messages, which is precisely the argument Ortiz et al. [3] make for decentralized hardware. Average-consensus schemes need rounds; rounds need liveness assumptions; liveness assumptions are what a burning building destroys.

**Multi-resolution.** Coarse cells are *not* averages of fine cells — they are separate variables tied to their children by aggregation factors, so precision bookkeeping stays consistent when the octree refines or evicts.

### 6.2 Layer 2 — Egress routing as a linearly-solvable MDP

Here is the core reformulation. On the traversability graph `G`, define a **passive dynamics** `p(x'|x)` — the transition a robot makes with no control effort, weighted by terrain difficulty (slope, step height, squeeze width) from §4.2. Control means *distorting* that distribution, and we charge for the distortion in KL:

```
ℓ(x, u) = q(x) + λ · KL( u(·|x) ‖ p(·|x) )
```

where the state cost `q(x)` is a per-step **hazard rate** — and we choose it to be exactly the ISO 13571 tenability accrual rate of §4.4, so that the quantity being minimized is the quantity being measured:

```
q(x) = Δt · [ dFED_asph/dt (Ĉ) + dFED_heat/dt (T̂) ]        accumulating tenability burden
     + w_D · P̂(collapse at x)                              non-toxic death hazard
     + w_geom · difficulty(x) · (1 + κ · FEC_irr(Ĉ))        traversal time, slowed by irritants
```

The three terms are commensurable because all of them are *rates*: the first two are fractions-of-lethal-dose per second, the third is a collapse probability per second, and the fourth is time spent, which matters precisely because time spent is more dose accrued. The irritant term enters as a multiplier on traversal difficulty rather than as an additive dose, mirroring §4.4 — a route through irritant smoke is punished for being *slow*, and slowness is what converts a survivable corridor into a lethal one. `α`-style tuning exponents are gone: the shape of the toxicity response comes from the standard, not from us.

Define the **desirability** `z(x) = exp(−V(x)/λ)`. Todorov's key result [9] is that this cost structure turns the Bellman equation *linear* — the `min` over actions is solved analytically and vanishes:

```
z(x) = exp(−q(x)/λ) · Σ_{x'} p(x'|x) z(x')        for non-terminal x
z(x) = 1   at egress (V = 0)
z(x) = 0   at absorbing death states
```

In matrix form, with `M = diag(exp(−q/λ))`, this is the sparse linear fixed point `z = M P z + b`. And the optimal policy is a **local normalization of neighbouring desirabilities**:

```
u*(x'|x) = p(x'|x) z(x') / Σ_{x''} p(x''|x) z(x'')
```

Read off what that buys us, because every one of these is a consequence, not an added feature:

1. **No `max`, no `argmax`, no branching** in the update — just 6–26 multiply-accumulates per cell. In the log domain it is adds and a log-sum-exp approximated by a small lookup table. This is *why* it fits in a 240 MHz integer core.
2. **Linear ⇒ decomposable.** A linear fixed point can be relaxed by asynchronous Jacobi/Gauss–Seidel sweeps in any order, on any partition, with stale neighbour values. That is the mathematical licence for full decentralization: robots relaxing overlapping subdomains with out-of-date boundary data still converge to the same field.
3. **Self-healing for free.** No plan is stored anywhere — only a field. A collapse changes `M` and `b` locally; the field re-relaxes locally. This is the "regrowth" property NCA are famous for [5], but here it is a property of the *fixed point*, not a trained behaviour.
4. **The policy is stochastic and calibrated,** not a brittle single path — natural for a swarm that should spread over multiple viable corridors rather than queue on one.
5. **`z` is a dose-weighted survival functional, not a proxy.** By the Feynman–Kac form of the LMDP, `z(x) = E_p[ exp(−D/λ) ]` where `D` is the tenability burden accumulated along a passive-dynamics trajectory from `x` to absorption. Since `q` is now literally the ISO 13571 accrual rate, `D` is literally the survivor's FED at egress — the thing §8 measures. The routing objective and the evaluation metric are the same integral.

**An honest caveat about λ, stated here rather than discovered in review.** `z = E_p[exp(−D/λ)]` is a *soft-min* over trajectories, so by Jensen `V = −λ log z ≤ E_p[D]`: the LMDP value is systematically **optimistic** relative to expected dose, crediting lucky trajectories more than it should. The gap closes as `λ → ∞` (risk-neutral) and widens as `λ → 0` (where `V → min D`, pure best-case). This means **λ is doing double duty** — it is simultaneously the control-cost temperature (how much policy distortion costs, hence how stochastic the swarm's routing is) and the risk attitude toward trajectory-level stochasticity. The two cannot be tuned independently in a pure LMDP; that is a real structural limitation of the formulation, not an implementation gap.

We handle it rather than hide it: `β` in §6.3 corrects for uncertainty in *`q` itself* (belief-level Jensen gap), which is a **different** gap from this trajectory-level one, and we do not claim `β` fixes both. Instead we (a) select `λ` on the development seeds by matching realized FED against the clairvoyant optimum of §8.1 rather than by hand, and (b) report the optimism gap `E_p[D] − V` as a measured diagnostic. If that gap turns out to be large enough to change routes, the principled repair is a risk-sensitive LMDP variant with a separate exponential tilt — flagged in the risk register as the fallback, not silently assumed away.

**Explicitly: this is not A\* with a risk-inflated cost.** A\* returns one path minimizing an expected additive cost. This returns a *field* whose values are exponentially-weighted (entropic) risk measures, whose policy is a distribution, whose update is local and asynchronous, and whose recomputation cost after a change is proportional to the size of the change rather than the size of the map.

### 6.3 C1 in detail — the belief covariance enters the objective analytically

`q(x)` is not known; it is a GBP belief, approximately `q(x) ~ N(μ_q(x), σ_q²(x))`. The temptation is to plug in `E[exp(−q/λ)]`. Doing so gives, exactly,

```
E[ exp(−q/λ) ] = exp( −μ_q/λ + σ_q²/(2λ²) )   ⟹   q_eff = μ_q − σ_q²/(2λ)
```

i.e. **taking the expectation makes the planner *optimistic***: uncertainty *reduces* effective cost. That is a Jensen-gap artifact, and it is exactly wrong for escorting a survivor — but exactly right for exploration. So rather than hide it, we make it the control knob. We define the routing cost as a one-sided confidence bound

```
q_β(x) = μ_q(x) + β · σ_q(x)
```

and compute two desirability fields with the same kernel and one changed constant:

- **`z⁺` with β > 0** — pessimistic, risk-averse. Used for **ESCORT**: never lead a survivor through a cell whose danger is poorly known.
- **`z⁻` with β < 0** — optimistic, uncertainty-seeking. Used for **SEARCH**: robots are *pulled* toward under-observed regions, giving directed exploration with no separate exploration bonus, no frontier detection, and no coverage planner.

The GBP precision `Λ` is already in memory; `σ_q = Λ^{-1/2}` is one shift-and-lookup. There is no tuned inflation heuristic anywhere in EMBER. Exploration and escort are one operator evaluated at two values of β, and the swarm's role arbitration (L4) is a threshold on the *ratio* `z⁺/z⁻`, which is high exactly where the map is confidently safe and low where it is unknown.

### 6.4 C2 in detail — the safeguarded Neural Cellular Automaton

**The problem with plain relaxation.** Jacobi on `z = MPz + b` propagates information one cell per sweep. A 100-cell corridor therefore needs ~100 sweeps before the exit is "visible" at the far end. At 240 MHz with 1500 cells, that is real time we do not have when a beam has just dropped.

**The learned operator.** Let `r^k(x) = (MPz^k + b − z^k)(x)` be the local residual. The NCA state at each cell is `[z, r, h₁..h₄, q, mask]`. The update is

```
z^{k+1}(x) = z^k(x) + N_θ( neighbourhood of state at x )
h^{k+1}(x) = h^k(x) + H_θ( neighbourhood of state at x )
```

where `N_θ, H_θ` are a depthwise-separable 3×3×3 convolution stack (≈ 8 channels, int8, ~2.5 KB of weights) shared by every cell and every robot. The hidden channels `h` are the interesting part: they are free to learn a **coarse-scale carrier** that transports residual information faster than one cell per sweep — in effect an *in-place, learned multigrid V-cycle* with no explicit grid hierarchy and no global operations.

**Training** (offline, on a laptop GPU, PyTorch; never on the robot). Sample hazard fields, graph topologies, and boundary conditions from the Cascade generator (development seeds only), compute the exact `z*` with a sparse direct solve, and minimize an unrolled residual objective

```
L(θ) = Σ_{k=1..K} γ^k ‖ r^k ‖²  +  ‖ z^K − z* ‖²  +  ρ · (quantization + spectral penalties)
```

with `K` ≈ 8–16, randomized topology, randomized `β`, and quantization-aware training so the int8 deployment is the trained artifact. We are distilling a **convergence rate**, not a policy — which is why kilobytes suffice where a distilled MAPF policy would need megabytes.

**Proposition 1 (safeguarded convergence).** Augment each cell's update with the purely local acceptance test

```
accept the NCA step at x  ⟺  |r_new(x)| ≤ (1 − ε)·|r_old(x)|      (ε > 0 small)
otherwise  take a Jacobi step at x:  z(x) ← (MPz + b)(x)
```

Then for **any** weights θ, the safeguarded asynchronous iteration converges to the unique fixed point of `z = MPz + b`.

*Proof sketch.* `M = diag(exp(−q/λ))` with `q ≥ 0` and `P` row-stochastic gives `‖MP‖_∞ ≤ max_x exp(−q(x)/λ) = γ < 1` whenever every trajectory accrues some cost — so `I − MP` is a strictly diagonally dominant M-matrix and the Jacobi operator is a γ-contraction in `‖·‖_∞`, convergent under arbitrary asynchronous update orders and bounded staleness (Bertsekas–Tsitsiklis chaotic relaxation). Accepted NCA steps strictly decrease the local residual by construction of the test; rejected steps are Jacobi steps. Hence the residual `‖r‖_∞` is non-increasing and strictly decreased on every rejected-step sweep, so no adversarial θ can prevent convergence — it can only fail to help. ∎

Three things follow, and they are the reason this contribution matters:

- **Safety.** A learned component steers a rescue robot, and its worst case is "no faster than the classical method." Untrained, corrupted, or out-of-distribution weights cannot produce a wrong field. This is what G3 was missing.
- **Legality as a CA.** The test is local, so the accelerated iteration involves no global synchronisation and remains a genuine cellular automaton — it can run asynchronously across robots and across partition boundaries.
- **Measurable claim.** "Sweeps to residual tolerance, learned vs. Jacobi" is a clean, single-number, falsifiable speedup (§8.4), and it is measured *in the deployed int8 arithmetic*, not in float.

**Fixed-point numerics.** Iterating a contraction in Q16 introduces a quantization floor: the iteration converges to a ball around `z*` of radius `O(δ/(1−γ))` where `δ` is the quantization step. We derive this bound, choose the log-domain shared exponent to keep it below the resolution at which the *policy* changes (the softmax of §6.2 is scale-invariant, which buys real slack), and verify it empirically. **This interaction between fixed-point arithmetic and fixed-point iteration is precisely the regime G4 identified as untreated**, and quantifying it is a secondary contribution.

### 6.5 C3 in detail — the adjoint field and VoI gossip

We want to transmit the belief updates that most change *decisions*, not those that most change *beliefs*. The sensitivity of the desirability at the robot's own position `x_r` to the cost at an arbitrary cell `x` is, to first order,

```
∂ log z(x_r) / ∂ q(x)  ∝  z†(x) · z(x) / λ
```

where `z†` is the **co-desirability**, i.e. the solution of the *adjoint* system `z† = (MP)ᵀ z† + b†` with sources at the robot and survivor positions. Concretely `z†(x)` is the (unnormalized) probability that a trajectory under the current policy *passes through* `x`. So:

> A cell matters to the swarm's decisions exactly insofar as trajectories flow through it.

The implementation cost is close to zero, which is the elegant part: `z†` is computed by **the same NCA kernel run on the transposed graph**, one extra field (~6 KB). Transmission priority for a candidate update at cell `x` is then

```
priority(x) = z†(x) · KL_gain(x),
KL_gain(x) = KL( N(η_new, Λ_new) ‖ N(η_old, Λ_old) )   [closed form: two Q16 ops + one log-table lookup]
```

and a token bucket transmits the top-k under a hard bytes/second budget, with an anti-starvation floor so no region goes permanently unreported.

The predicted behaviour is sharp and easy to demonstrate on stage: a fire in a **stairwell every route depends on** triggers immediate high-priority chatter; an identical fire in a **dead-end room nobody routes through** produces near-silence. Flooding gossip cannot tell these apart, which is exactly why we expect to beat it by one to two orders of magnitude in bytes at equal survival rate.

`z†` earns its memory twice over: it is also the natural **relay-placement objective** (a robot maximizing `z†`-weighted connectivity is holding open the corridor that matters), so the RELAY role in L4 needs no additional machinery either.

### 6.6 How the layers close into one object

| Layer | Object | Update | Radio |
|---|---|---|---|
| Hazard + survivors | GBP beliefs `(η, Λ)` | local message passing | the messages **are** the packets |
| Routing | fields `z⁺, z⁻` | local safeguarded NCA sweeps | boundary values gossiped as beliefs |
| Scheduling | adjoint `z†` | same kernel, transposed graph | decides *which* messages exist |

Three layers, one primitive: *sum incoming local messages, apply a local nonlinearity, repeat.* Robot failure, radio partition, hazard propagation, and secondary collapse are all just perturbations to one fixed-point iteration — which is the claim in §1, and the reason a 520 KB device can run all of it.

---

## 7. Implementation Plan

Fourteen weeks, three reviewable milestones, each with a hard **definition of done** and a **live demo**. Every phase ends with something that runs; there is no phase whose deliverable is only a document.

### Phase 0 — Weeks 1–2 → **Proposal review**

Build the world before building the mind.

- `cascade/` generator: floor-plan grammar, structural graph, MJCF emission.
- **Collapse-by-simulation** working end-to-end; settled states cached as seeded keyframes.
- `hfield` + voxel-graph extraction; a scripted robot drives over rubble without exploding.
- Metric/logging skeleton (Hydra configs, seeded manifests, Polars log schema).

**Done when:** `python -m cascade.demo --seed 7` renders a physically collapsed 4-storey building in MuJoCo and a teleoperated robot traverses three storeys. **Demo:** collapse replay from three seeds, side by side.

### Phase 1 — Weeks 3–6 → **MVP review**

The first end-to-end loop, with the classical solver and no learning.

- Taichi hazard solver: combustion + smoke + **stack effect**; verified against an analytic 1-D advection–diffusion solution.
- Sensor models (lidar, thermal, gas, IMU, odometry) and the **BLE RSSI** model with raycast wall attenuation.
- **Tier-A MCU sandbox**: Q16 type, arena allocator, cycle accounting, deadline scheduler.
- **GBP** over the hazard octree + range factors for survivors.
- **Plain Jacobi** LMDP routing with the β-parameterized cost (C1) — no NCA yet.
- **λ selection and optimism diagnostic**: choose `λ` against the clairvoyant optimum on dev seeds, and measure the trajectory-level gap `E_p[D] − V` (§6.2 caveat) to decide early whether the risk-sensitive variant is needed.
- Comms sim: ESP-NOW-class PHY/MAC with raycast attenuation; **flooding gossip** as the transmit policy.
- Baselines **B1** (centralized A\*/D\*-Lite, perfect comms and full observability) and **B2** (flooding + local Dijkstra).
- 6–8 robots, single hazard, no live collapse.

**Done when:** 8 robots on an unseen seed localize ≥ 3 survivors and route them out, with RAM high-water < 200 KB and deadline-miss rate < 5 %; B1/B2 produce numbers in the same harness. **Demo:** live run with a hazard overlay and a per-robot RAM/cycle HUD; the "shortest route is the chimney" dilemma visibly punishes B1.

> Phase 1 is deliberately a *complete, publishable-negative-result-proof* system on its own. If every subsequent contribution failed, there would still be a working decentralized rescue swarm on an MCU budget to demo. That is the project's main risk mitigation.

### Phase 2 — Weeks 7–10

The three contributions land.

- **NCA training rig** (PyTorch, offline): dev-seed hazard fields, exact `z*` targets, unrolled residual loss, quantization-aware training, export to int8 tables.
- **Safeguarded NCA** in the Tier-A sandbox; measure sweeps-to-tolerance vs. Jacobi; empirically verify Proposition 1 by injecting *deliberately corrupted* weights and confirming convergence still holds.
- **Adjoint field `z†`** + VoI gossip + token bucket; nats-per-byte instrumentation.
- **C4 validation**: kill robots and partition the network with no failure detector in the loop; measure recovery half-life.
- **Progressive collapse** wired to thermal damage (`data.eq_active`), plus the interactive "weaken this column" demo hook.
- Baseline **B3** (consensus log-odds merging + risk-inflated A\*) — the strong "reasonable engineer" baseline.
- Scale to 16–24 robots; full metric harness; ablation grid automated.

**Done when:** the full metric table populates automatically over ≥ 20 dev seeds × 5 robot-failure levels, and the four headline predictions of §8 are either confirmed or refuted with error bars.

### Phase 3 — Weeks 11–14 → **Final demo**

Validation, generalization, and the write-up.

- **Tier B**: fixed-point C kernels; build under QEMU-xtensa (ESP32) and Renode (Cortex-M4); report Tier-A-vs-Tier-B cycle and RAM calibration error.
- **Tier C**: 1–2 robots running real firmware over a socket, live, during the demo.
- **Held-out seeds `1000–1049`** opened for the first time; all paper numbers come from these.
- Ablations: no-VoI, no-NCA, float-vs-Q16, no-GBP (independent beliefs), β = 0.
- *(Stretch, drop first)* Baseline **B4**: a MARL policy (IPPO/QMIX) trained on the same env, then quantized — expected to fail the memory budget or the cross-seed generalization test, either of which is an informative result.
- Paper draft (6–8 pages), reproducibility bundle, demo script.

**Done when:** every table is from held-out seeds, and the live demo survives an *unrehearsed* hazard injection chosen by the audience.

### Scoping by team size

| Team | Scope |
|---|---|
| **1 student** | Phases 0–2 in full, C1 + C2 + C4. Drop B4 and Tier C; Tier B limited to the NCA kernel on Renode/Cortex-M4 only. C3 reduced to a KL-only gossip gate (no adjoint) if week 10 slips. |
| **2 students** | Everything above plus C3 in full and Tier C. Split: environment+physics / inference+routing. |
| **3 students** | Add B4, GPU-batched training via MJX, 32+ robots, and a heterogeneous swarm (a few robots with better sensors) as a bonus study. |

### Risk register

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| MuJoCo collapse sim is unstable or too slow | Med | Reduce member count; substep; cache settled keyframes (already the design). Fallback: PyBullet, or a one-off offline collapse baked to a mesh. |
| NCA gives no real speedup | Med | Proposition 1 means the *system still works* — it degrades to Jacobi. Report the negative result honestly; it is publishable ("learned acceleration does not help below N cells because..."). |
| Loopy GBP oscillates on the hazard graph | Med | Message damping, factor-graph sparsification, robust factors; Gauss–Seidel-style scheduling. Well-trodden mitigations. |
| ESP32 support in Renode is weak | High | Known: use QEMU-xtensa via ESP-IDF for ESP32, Renode with a Cortex-M4 at comparable clock as the second data point. Already the plan. |
| Taichi hazard solver dominates wall-clock | Med | Coarsen the hazard lattice; decouple to 5 Hz; precompute the stack-effect velocity field per seed. |
| LMDP trajectory-level optimism (§6.2 caveat) is large enough to change routes | Med | Measure the gap `E_p[D] − V` directly in Phase 1. If material: adopt a risk-sensitive LMDP with a separate exponential tilt, decoupling risk attitude from the control temperature `λ`. The kernel stays linear, so C2/C3 are unaffected. |
| Scope overrun | High | Phase 1 is a complete demonstrable system by itself; every later contribution is additive and independently droppable. |

**Reproducibility.** Everything seeded; configs in Hydra; a manifest per run recording seed, git SHA, config hash, and library versions; CI runs a 3-robot smoke test on every commit.

---

## 8. Expected Results

These are **pre-registered predictions with falsification criteria**, not results. Nothing here has been measured yet; the point of stating them this sharply is that the project can *fail visibly*, which is what makes it science rather than a demo.

### 8.1 Metrics

| Metric | Definition | Why this one |
|---|---|---|
| **Survival rate** | fraction of survivors reaching egress below the ISO 13571 tenability limit, `max(FED_asph + FED_heat, FEC_irr) < 1` (§4.4) | the actual mission objective, from fire-safety engineering, not a game score |
| **Risk-optimality gap** | realized route's expected dose ÷ dose of the **clairvoyant optimum** (offline solve of the time-expanded MDP with full future hazard knowledge) | a *true* upper bound exists, so "how close to optimal" is answerable rather than relative |
| **Time to first contact** | median time until a survivor is localized to < 2 m | measures the search half of the problem |
| **Comm efficiency** | bytes/robot/s, **and nats-per-byte** (KL reduction at receivers per byte sent) | raw throughput is the wrong axis; C3 is a claim about information per byte |
| **Resilience curve** | survival rate vs. fraction of robots killed (0–60 %) | resilience is a curve, not a number |
| **Recovery half-life** | time for the routing field to return within ε of pre-perturbation risk-optimality after a k-robot kill or a live collapse | the key self-healing claim of C2/§6.2 |
| **Edge feasibility** | RAM high-water (KB), cycles/tick, deadline-miss rate, Tier-A↔Tier-B calibration error | the constraint that makes the project honest |

### 8.2 Baselines

- **B1 — Centralized A\*/D\*-Lite, perfect comms, full observability.** An *oracle* on path length. We expect it to be *beaten on survival* despite its unfair advantages, because shortest ≠ safest and its cost model has no risk term. If it wins on survival too, C1's premise is wrong.
- **B2 — Flooding gossip + local Dijkstra on last-received map.** The naive swarm baseline; expected to saturate the channel.
- **B3 — Consensus log-odds merging + risk-inflated A\*.** The strong baseline a competent engineer would actually build. This is the one that matters.
- **B4 — Quantized MARL (IPPO/QMIX)** *(stretch)*. Expected to fail the memory budget or generalize poorly to held-out seeds.

Plus ablations: **no-VoI** (isolates C3), **no-NCA** (isolates C2), **float-vs-Q16** (isolates the quantization cost), **no-GBP** (isolates cooperative inference), **β = 0** (isolates C1).

### 8.3 Predictions

| Prediction | Rationale |
|---|---|
| **P1 — EMBER beats centralized-oracle A\* (B1) on survival rate, while losing to it on path length.** | B1 optimizes the wrong objective and walks survivors into the stack-effect chimney; the environment generator guarantees such dilemmas exist. Losing on length is *expected and correct*. |
| **P2 — EMBER matches B3's survival rate using 10–100× fewer bytes, and beats B2 outright.** | C3 sends only route-relevant information; B2's traffic scales with map × robots and B3 needs consensus rounds. Most hazard updates provably do not change any policy. |
| **P3 — Safeguarded NCA reaches routing tolerance in ~10× fewer sweeps than Jacobi**, with the advantage *growing* with map diameter. | Jacobi moves information one cell/sweep; learned hidden channels carry it multiscale. If the gain is < 2× the contribution is refuted — but the system still works (Prop. 1). |
| **P4 — EMBER's resilience curve degrades gracefully to ≥ 50 % robot loss, where B2/B3 exhibit a knee.** | No leader, no membership, no failure detector (C4); stale beliefs self-attenuate and the field re-relaxes locally. |
| **P5 — Recovery half-life scales with the *diameter of the perturbation*, not the diameter of the map.** | A local change to `M` and `b` perturbs a contraction locally. This is the sharpest quantitative signature of the whole design, and the easiest to falsify. |
| **P6 — Full stack fits in < 150 KB RAM with < 20 % deadline misses at 10 Hz, and Tier-A cycle estimates land within ~30 % of measured firmware.** | Budget in §5.2 has ~4× headroom; the calibration number is reported whatever it turns out to be. |

### 8.4 What would falsify the whole thing

Stated plainly, because a proposal that cannot be wrong is not worth running:

- **C1 fails** if the β = 0 ablation matches full EMBER on survival — i.e. the derived risk premium does nothing a mean-cost planner does not already do.
- **C2 fails** if the learned NCA gives < 2× sweep reduction at deployment precision, or if int8 quantization error moves the *policy* (not merely the field).
- **C3 fails** if nats-per-byte improves by < 3× over KL-only gating, meaning the adjoint weighting was not carrying the gain.
- **C4 fails** if the resilience curve has a knee at a lower failure fraction than B3's — i.e. implicit failure handling is worse than explicit.
- **The thesis fails** if survival rate turns out to be insensitive to routing quality altogether (e.g. dominated by search time), which would mean we built a beautiful solution to a non-bottleneck. *We test this in Phase 1 before committing to Phase 2*, by sweeping routing quality with an oracle map and checking that survival responds.

### 8.5 Insights we expect to be able to defend

1. **Bandwidth is not the binding constraint; decision-relevance is.** Most of what a swarm transmits cannot change any decision, and the adjoint field identifies which parts those are cheaply. This reframes swarm communication from a compression problem into a sensitivity-analysis problem.
2. **Linear Bellman equations are the natural fit for microcontroller swarms.** Removing the `max` operator is what simultaneously buys asynchrony, decomposability, self-healing, and integer arithmetic. We expect the general lesson — *choose the control formulation for its algebra, not its expressiveness* — to transfer well beyond rescue routing.
3. **Learned components can be made safe by safeguarding rather than by verification.** A local acceptance test with a classical fallback gives an unconditional guarantee for pennies, and is a template for putting learning inside safety-critical embedded loops.
4. **Uncertainty is not a nuisance to be inflated away; it is the exploration signal.** One scalar's sign turns a safety margin into a search drive.

---

## 9. Tools and Libraries

**Explicitly excluded: ROS, ROS 2, Webots, and all stock demo scenes.**

### On-robot (must survive the Tier-A restrictions; no floats, no allocation on hot paths)
| Purpose | Tool |
|---|---|
| Fixed-point arithmetic | custom `Q16.16` on NumPy int32, with saturating ops |
| Memory model | custom `MCUArena` bump/pool allocator, 520 KB hard cap |
| Cycle accounting | custom `mcu.ops` analytic cost model, Xtensa-LX6-calibrated |
| GBP, NCA, policy | pure integer kernels written against `mcu.ops` |

### Simulation
| Purpose | Tool |
|---|---|
| Rigid-body physics, contacts, raycasting, heightfields | **MuJoCo 3.x** (`mujoco`) |
| Hazard PDEs (combustion, smoke, stack effect) | **Taichi** (JIT, CPU/GPU) |
| Array backbone | **NumPy** |
| Graph algorithms (offline: baselines, ground truth) | **NetworkX**, **SciPy.sparse** (`spsolve` for exact `z*`) |
| Optional CPU acceleration | **Numba** |
| Rendering / video | `mujoco.viewer`, **imageio**, **matplotlib** |

### Offline training and analysis (laptop/GPU only — never on the robot)
| Purpose | Tool |
|---|---|
| NCA training, quantization-aware training | **PyTorch** |
| Batched env rollouts for training *(optional)* | **MuJoCo MJX / JAX** |
| MARL baseline B4 *(stretch)* | **PyTorch** + a compact custom IPPO/QMIX |
| Experiment logs, tables | **Polars**, **matplotlib** |

### Firmware validation
| Purpose | Tool |
|---|---|
| ESP32 (Xtensa) emulation | **QEMU-xtensa** via **ESP-IDF** (`idf.py qemu`) |
| Cortex-M4 emulation, cycle counters | **Renode** |
| Kernels | hand-written fixed-point **C** (CMSIS-DSP-style), built with ESP-IDF / arm-none-eabi-gcc |

### Engineering
Hydra (configs) · pytest + Hypothesis (property tests on the fixed-point kernels: contraction, monotonicity, no-overflow) · ruff + mypy · Rich/tqdm · git + a seeded run manifest (seed, git SHA, config hash, library versions) · GitHub Actions running a 3-robot smoke test per commit.

### Repository layout

```
SwarmRescue/
├── cascade/          # world generation: grammar, structural graph, collapse sim, extraction
├── hazard/           # Taichi combustion / smoke / stack-effect solvers, degradation
├── sensors/          # lidar, thermal, gas, IMU, odometry, BLE RSSI
├── mcu/              # Q16, MCUArena, ops cost model, deadline scheduler
├── ember/
│   ├── gbp/          # factor graph, physics factors, range factors, robust factors
│   ├── routing/      # LMDP kernels, safeguarded NCA, adjoint field
│   ├── comms/        # PHY/MAC model, VoI scheduler, token bucket, provenance
│   └── behaviour/    # role arbitration from field values
├── baselines/        # B1 centralized A*/D*-Lite, B2 flooding, B3 consensus, B4 MARL
├── train/            # offline NCA training + QAT + int8 export
├── firmware/         # C kernels, ESP-IDF and Cortex-M4 builds, socket co-sim bridge
├── eval/             # metric harness, clairvoyant optimum solver, ablation grid, plots
└── demo/             # live demo script, HUD overlays, interactive hazard injection
```

---

## Appendix A — What the live final demo looks like

1. **A seed nobody has run before** (from the held-out set) generates a 4-storey building; the collapse is simulated on screen in ~8 s.
2. Sixteen robots enter. The HUD shows, per robot, RAM high-water, cycles/tick, and deadline misses — always inside the ESP32 budget. **Two of the sixteen are running real C firmware in QEMU**, and the audience is invited to identify which; they cannot.
3. Fire seeds in a second-floor office. Smoke climbs the stairwell fast (stack effect). The `z⁻` (optimistic) field visibly pulls robots toward unobserved regions; nobody wrote an exploration policy.
4. Survivors are localized by BLE trilateration that **no single robot could have done** — a side panel shows the single-robot estimate at ±20 m collapsing to ±1.5 m as messages fuse.
5. `z⁺` (pessimistic) egress fields appear as flow arrows. Centralized A\* — running alongside with a perfect map — routes its survivor into the stairwell. EMBER routes around it. The dose counters diverge on screen.
6. **An audience member clicks a column.** It heats, its damage integral crosses threshold, and a slab drops — closing the corridor EMBER was using. No replanning call is made anywhere in the code. The field re-relaxes locally and the flow arrows reverse within a second or two. The recovery half-life is measured live and printed.
7. **Eight of sixteen robots are killed** at once. There is no failure detector to trigger. Stale beliefs lose precision, the pessimistic field withdraws from unwatched regions, and the remaining robots complete the rescue at a graceful cost. The resilience curve gets a new live data point.

**Appendix B — a note on ambition vs. feasibility.** The one honest concern with this proposal is that four contributions is a lot for one semester. That is why Phase 1 is scoped to be a *complete, demonstrable decentralized rescue swarm on an MCU budget with two baselines* before any of C1–C4 lands, and why each contribution is independently droppable with a stated fallback. The worst realistic outcome is a working system, an honest negative result on learned acceleration, and a paper about why the linear-Bellman formulation is the right primitive for edge-native swarms. That outcome still clears the rubric.
