"""Stage 3 -- collapse by simulation, not by hand.

We sample `k` initiating failures from a spatial damage kernel, deactivate
those welds, and run MuJoCo forward.  Slabs pancake, beams rotate, debris
piles form.  Nothing about the resulting rubble field is authored: it is
whatever the contact solver produced, so mass is conserved, piles hold an
angle of repose, and voids exist under leaning slabs.

The settled state is cached under the seed, so every downstream experiment is
reproducible without paying for the collapse again.  The cache key includes a
hash of the model XML: change the generator and stale rubble is discarded
rather than silently reused.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .params import CollapseParams
from .rng import stage_rng
from .structure import MemberKind, Structure

#: Where settled keyframes live.  Regenerable from (seed, generator), so the
#: directory is a cache and is git-ignored.
CACHE_DIR = Path(__file__).resolve().parent / "cache"

#: Cache format version.  Bump to invalidate every stored keyframe.
CACHE_VERSION: int = 1

#: Only load-bearing welds are eligible as initiating failures.  Releasing a
#: partition wall's weld drops one wall and teaches us nothing about collapse.
_INITIATOR_PREFIXES = ("w_foot_col", "w_head_col")

#: Where pulverized members are parked: far below the ground plane, out of the
#: rubble field and out of the render.  They keep their degrees of freedom (a
#: MuJoCo model's body count is fixed at compile time) but interact with
#: nothing.
_DEBRIS_DISPOSAL_Z: float = -500.0

#: How many blast centroids to try before accepting the most damaging one.
_MAX_BLAST_ATTEMPTS: int = 24


@dataclass(frozen=True, slots=True)
class CollapseResult:
    """The settled outcome of one seeded collapse."""

    seed: int
    #: Columns destroyed by the blast itself, before any cascade.
    initiating_members: tuple[str, ...]
    blast_center: tuple[float, float, float]
    qpos: np.ndarray
    steps_simulated: int
    settled_early: bool
    wall_time_s: float
    peak_kinetic_energy_j: float
    final_kinetic_energy_j: float
    model_hash: str
    #: Every member destroyed, initiating failures plus the cascade.
    failed_members: tuple[str, ...] = ()
    #: (step, member) for each cascade failure, in the order they happened.
    cascade: tuple[tuple[int, str], ...] = ()

    @property
    def summary(self) -> str:
        mode = "quiescent" if self.settled_early else "time-limited"
        return (
            f"seed={self.seed} initiating={len(self.initiating_members)} "
            f"cascade={len(self.cascade)} failed={len(self.failed_members)} "
            f"steps={self.steps_simulated} ({mode}) "
            f"peakKE={self.peak_kinetic_energy_j / 1e6:.2f} MJ "
            f"in {self.wall_time_s:.1f} s"
        )


def model_hash(xml: str) -> str:
    """Stable digest of the model, used as part of the cache key."""
    return hashlib.sha256(xml.encode("utf-8")).hexdigest()[:16]


# --- initiating failures -----------------------------------------------------


def sample_initiating_members(
    structure: Structure,
    seed: int,
    params: CollapseParams,
) -> tuple[tuple[str, ...], tuple[float, float, float]]:
    """Choose which columns the blast destroys, from a spatial damage kernel.

    A blast centroid is sampled in the building's lower storeys, and each
    column's destruction probability decays as a Gaussian in its distance from
    that centroid.  Two properties matter and both come from the kernel:

    * Damage is **localised**.  Failures scattered uniformly produce a
      uniformly sagging building, which looks nothing like a partial collapse.
    * Damage is **clustered enough to matter**.  This frame carries a ~3.6x
      static safety margin, so removing a few columns scattered across an
      18-node grid changes nothing; the load simply redistributes.  A collapse
      requires a contiguous group of columns to go at once, leaving a slab
      genuinely unsupported over a full bay.
    """
    rng = stage_rng(seed, "collapse")

    columns = structure.of_kind(MemberKind.COLUMN)
    if not columns:
        return (), (0.0, 0.0, 0.0)

    # Initiate at the base: destroying columns higher up leaves the storeys
    # below intact and the building simply stands on the stump.
    ground = [c for c in columns if c.storey == 0] or list(columns)
    supports = slab_supports(structure)

    xs = [c.center[0] for c in ground]
    ys = [c.center[1] for c in ground]

    best: tuple[str, ...] = ()
    blast = (0.0, 0.0, 0.0)
    for attempt in range(_MAX_BLAST_ATTEMPTS):
        center = (
            float(rng.uniform(min(xs), max(xs))),
            float(rng.uniform(min(ys), max(ys))),
            0.0,
        )
        # Widen the blast on later attempts rather than resampling forever.
        radius = params.blast_radius_m * (1.0 + 0.2 * attempt)
        destroyed = {
            c.name
            for c in ground
            if (c.center[0] - center[0]) ** 2 + (c.center[1] - center[1]) ** 2
            <= radius**2
        }
        if len(destroyed) > len(best):
            best, blast = tuple(sorted(destroyed)), center

        # The precondition for a collapse: at least one slab must lose *every*
        # column beneath it.  A blast that leaves each slab still propped up
        # somewhere redistributes load and the building stands, which is
        # physically correct and useless as a rescue scenario.
        if any(
            cols and cols <= destroyed
            for cols in supports.values()
        ):
            return tuple(sorted(destroyed)), center

    return best, blast


def slab_supports(structure: Structure) -> dict[str, set[str]]:
    """Map each slab to the set of columns bearing directly beneath it."""
    out: dict[str, set[str]] = {}
    for weld in structure.welds:
        if weld.name.startswith("w_head_col") and weld.body_b is not None:
            out.setdefault(weld.body_b, set()).add(weld.body_a)
    return out


def _sq_distance(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    return sum((a[k] - b[k]) ** 2 for k in range(3))


# --- the collapse itself -----------------------------------------------------


def release_welds(
    model: mujoco.MjModel, data: mujoco.MjData, names: tuple[str, ...]
) -> int:
    """Deactivate the named equality constraints.  Returns how many released.

    This is the *only* mechanism by which anything is ever destroyed in
    Cascade, in Phase 0 and in the live thermal-damage collapses of Phase 2
    alike.  There is no scripted animation path to keep it consistent with.
    """
    released = 0
    for name in names:
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eq_id < 0:
            raise KeyError(f"no equality constraint named {name!r} in the model")
        data.eq_active[eq_id] = 0
        released += 1
    return released


def kinetic_energy(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Total kinetic energy, 0.5 * v' M v, in joules."""
    momentum = np.zeros(model.nv, dtype=np.float64)
    mujoco.mj_mulM(model, data, momentum, data.qvel)
    return 0.5 * float(data.qvel @ momentum)


def weld_force_magnitudes(
    model: mujoco.MjModel, data: mujoco.MjData
) -> np.ndarray:
    """Translational force carried by each equality constraint, in newtons.

    MuJoCo lays a weld's rows out contiguously as three translational
    components followed by three rotational ones.  We take the norm of the
    translational triple only, so the result is a force in newtons that can be
    compared against a member's yield threshold without mixing in torques.

    Returns an array of length `model.neq`; inactive constraints read zero.
    """
    forces = np.zeros(model.neq, dtype=np.float64)
    if data.nefc == 0:
        return forces

    is_equality = data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    rows = np.flatnonzero(is_equality)
    if rows.size == 0:
        return forces

    eq_ids = data.efc_id[rows]
    values = data.efc_force[rows]

    # Walk each equality's contiguous row block and take its first three rows.
    start = 0
    while start < rows.size:
        eq_id = eq_ids[start]
        end = start
        while end < rows.size and eq_ids[end] == eq_id:
            end += 1
        block = values[start : min(start + 3, end)]
        forces[eq_id] = float(np.linalg.norm(block))
        start = end
    return forces


def contact_loads(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Summed contact normal force per body, in newtons.

    The compressive load path in this structure runs through *contact*, not
    through the welds: a column rests on the ground and carries the slab
    resting on it, while its welds mostly resist separation and only carry a
    few percent of the weight.  Measuring weld force alone therefore reports
    ~2% of the true axial load, which is the trap this function exists to
    avoid.
    """
    loads = np.zeros(model.nbody, dtype=np.float64)
    wrench = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, wrench)
        normal = abs(float(wrench[0]))
        contact = data.contact[i]
        loads[model.geom_bodyid[contact.geom1]] += normal
        loads[model.geom_bodyid[contact.geom2]] += normal
    return loads


def member_axial_loads(
    model: mujoco.MjModel, data: mujoco.MjData, structure: Structure
) -> dict[str, float]:
    """Estimated axial load carried by each member, in newtons.

    A member in equilibrium receives roughly its axial load at each end (from
    what it carries above and what supports it below), so the summed contact
    normal force is about twice the axial load; the weld contribution is added
    on top because a weld in tension carries load the contacts cannot.
    """
    contact = contact_loads(model, data)
    weld = weld_force_magnitudes(model, data)

    weld_by_member: dict[str, float] = {}
    for w in structure.welds:
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, w.name)
        if eq_id >= 0 and data.eq_active[eq_id]:
            weld_by_member[w.body_a] = max(
                weld_by_member.get(w.body_a, 0.0), float(weld[eq_id])
            )

    out: dict[str, float] = {}
    for member in structure.members:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, member.name)
        if body_id < 0:
            continue
        out[member.name] = 0.5 * float(contact[body_id]) + weld_by_member.get(
            member.name, 0.0
        )
    return out


def find_overloaded_members(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    structure: Structure,
    safety_factor: float,
    already_failed: set[str],
) -> tuple[str, ...]:
    """Members whose axial load exceeds their yield threshold.

    This is what turns a handful of initiating failures into a progressive
    collapse: when a column goes, its share of the load transfers to its
    neighbours, and those that cannot carry it fail in turn.  The cascade is a
    consequence of the load path, not a scripted sequence.

    Only load-bearing members are checked.  Partitions are detailed with a
    deflection head gap (see `structure.WALL_HEAD_GAP_M`), so they carry no
    vertical load and have no structural role to lose -- they simply fall when
    whatever they stand on goes, and the pile they make is the rubble the
    robots must traverse.  Failing them explicitly would only delete that
    rubble, and deleting rubble is what turns a partial collapse into a
    runaway: the debris that should pile up and arrest the cascade vanishes
    instead.
    """
    loads = member_axial_loads(model, data, structure)
    return tuple(
        m.name
        for m in structure.members
        if m.kind.is_load_bearing
        and m.name not in already_failed
        and loads.get(m.name, 0.0) > safety_factor * m.yield_threshold_n
    )


def pulverize(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    structure: Structure,
    names: tuple[str, ...],
) -> None:
    """Destroy the named members: they stop carrying load and stop colliding.

    A real column does not fail by politely detaching -- it crushes and
    buckles, ceasing to occupy the space that held the slab up.  A rigid box
    cannot do that, so a failed member has its welds released, its collision
    geometry disabled, and its body dropped far below the ground plane.  The
    slab above then has genuinely nothing beneath it.

    This mutates `model` (geom contype/conaffinity), so callers that need a
    pristine model must pass a per-run instance.
    """
    welds_by_member: dict[str, list[str]] = {}
    for w in structure.welds:
        welds_by_member.setdefault(w.body_a, []).append(w.name)
        if w.body_b is not None:
            welds_by_member.setdefault(w.body_b, []).append(w.name)

    for name in names:
        for weld_name in welds_by_member.get(name, ()):
            eq_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name
            )
            if eq_id >= 0:
                data.eq_active[eq_id] = 0

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue
        for geom_id in range(
            model.body_geomadr[body_id],
            model.body_geomadr[body_id] + model.body_geomnum[body_id],
        ):
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0

        joint_id = model.body_jntadr[body_id]
        if joint_id >= 0:
            adr = model.jnt_qposadr[joint_id]
            data.qpos[adr + 2] = _DEBRIS_DISPOSAL_Z
            data.qvel[model.jnt_dofadr[joint_id] : model.jnt_dofadr[joint_id] + 6] = 0.0


def simulate_collapse(
    model: mujoco.MjModel,
    structure: Structure,
    seed: int,
    params: CollapseParams,
    xml_hash: str,
) -> CollapseResult:
    """Release the initiating welds and let the physics produce the rubble."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    initiating, blast = sample_initiating_members(structure, seed, params)
    # The blast pulverizes the columns it struck: their welds go *and* they
    # stop occupying the space that held the slab up.
    failed: set[str] = set(initiating)
    pulverize(model, data, structure, initiating)
    mujoco.mj_forward(model, data)

    cascade: list[tuple[int, str]] = []
    # member -> consecutive checks it has been over its threshold
    overload_streak: dict[str, int] = {}
    peak_ke = 0.0
    quiescent_for = 0
    settled_early = False
    step = 0
    started = time.perf_counter()

    for step in range(1, params.n_settle_steps + 1):
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError(
                f"collapse diverged at step {step} for seed {seed}; "
                "the solver configuration or member sizing needs attention"
            )

        if (
            step % params.overload_check_interval == 0
            and len(failed) < params.max_cascade_failures
        ):
            budget = params.max_cascade_failures - len(failed)
            over_now = set(
                find_overloaded_members(
                    model, data, structure, params.overload_safety_factor, failed
                )
            )
            # Sustained overload only: a member that drops back under its
            # threshold has ridden out an impulse and its streak resets.
            for name in list(overload_streak):
                if name not in over_now:
                    del overload_streak[name]
            for name in over_now:
                overload_streak[name] = overload_streak.get(name, 0) + 1

            overloaded = tuple(
                sorted(
                    n
                    for n, streak in overload_streak.items()
                    if streak >= params.overload_persistence_checks
                )
            )[:budget]
            if overloaded:
                # Everything that fails is pulverized, partitions included.
                #
                # Merely detaching a failed member does not work: a free wall
                # still wedged between the floor and the slab above goes on
                # carrying compression exactly as before, so the building
                # stayed up with 46 members "failed".  A member that has
                # yielded must stop bearing load, and for a rigid box the only
                # way to express that is to remove it.
                pulverize(model, data, structure, overloaded)
                failed.update(overloaded)
                for name in overloaded:
                    overload_streak.pop(name, None)
                cascade.extend((step, name) for name in overloaded)

        energy = kinetic_energy(model, data)
        peak_ke = max(peak_ke, energy)

        # Stop as soon as the rubble stops moving; most collapses settle well
        # inside the time budget and the remainder is wasted wall-clock.
        if energy < params.quiescent_kinetic_energy_j:
            quiescent_for += 1
            if quiescent_for >= params.quiescent_ticks:
                settled_early = True
                break
        else:
            quiescent_for = 0

    return CollapseResult(
        seed=seed,
        initiating_members=initiating,
        blast_center=blast,
        qpos=data.qpos.copy(),
        steps_simulated=step,
        settled_early=settled_early,
        wall_time_s=time.perf_counter() - started,
        peak_kinetic_energy_j=peak_ke,
        final_kinetic_energy_j=kinetic_energy(model, data),
        model_hash=xml_hash,
        failed_members=tuple(sorted(failed)),
        cascade=tuple(cascade),
    )


# --- caching -----------------------------------------------------------------


def cache_path(seed: int, xml_hash: str, cache_dir: Path | None = None) -> Path:
    directory = cache_dir or CACHE_DIR
    return directory / f"collapse_v{CACHE_VERSION}_seed{seed}_{xml_hash}.npz"


def save_collapse(result: CollapseResult, cache_dir: Path | None = None) -> Path:
    path = cache_path(result.seed, result.model_hash, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        qpos=result.qpos,
        initiating_members=np.array(result.initiating_members, dtype=object),
        blast_center=np.array(result.blast_center),
        steps_simulated=result.steps_simulated,
        settled_early=result.settled_early,
        wall_time_s=result.wall_time_s,
        peak_kinetic_energy_j=result.peak_kinetic_energy_j,
        final_kinetic_energy_j=result.final_kinetic_energy_j,
        model_hash=result.model_hash,
        seed=result.seed,
        failed_members=np.array(result.failed_members, dtype=object),
        cascade_steps=np.array([s for s, _ in result.cascade], dtype=np.int64),
        cascade_members=np.array([n for _, n in result.cascade], dtype=object),
    )
    return path


def load_collapse(
    seed: int, xml_hash: str, cache_dir: Path | None = None
) -> CollapseResult | None:
    """Return the cached settled state, or None on a miss.

    A miss includes the case where the generator changed: the model hash is
    part of the filename, so stale rubble is never silently reused.
    """
    path = cache_path(seed, xml_hash, cache_dir)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as blob:
        return CollapseResult(
            seed=int(blob["seed"]),
            initiating_members=tuple(blob["initiating_members"].tolist()),
            blast_center=tuple(blob["blast_center"].tolist()),
            qpos=blob["qpos"],
            steps_simulated=int(blob["steps_simulated"]),
            settled_early=bool(blob["settled_early"]),
            wall_time_s=float(blob["wall_time_s"]),
            peak_kinetic_energy_j=float(blob["peak_kinetic_energy_j"]),
            final_kinetic_energy_j=float(blob["final_kinetic_energy_j"]),
            model_hash=str(blob["model_hash"]),
            failed_members=tuple(blob["failed_members"].tolist()),
            cascade=tuple(
                zip(
                    blob["cascade_steps"].tolist(),
                    blob["cascade_members"].tolist(),
                    strict=True,
                )
            ),
        )


def apply_settled_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    structure: Structure,
    result: CollapseResult,
) -> None:
    """Place the world in its post-collapse configuration, at rest.

    Restoring `qpos` alone is not enough: the members destroyed during the
    collapse must be destroyed again, or a cached rubble field would be
    silently propped up by columns that failed in the run that produced it.
    """
    if result.qpos.shape[0] != model.nq:
        raise ValueError(
            f"cached qpos has {result.qpos.shape[0]} entries but the model "
            f"expects {model.nq}; the cache is stale"
        )
    pulverize(model, data, structure, result.failed_members)
    data.qpos[:] = result.qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
