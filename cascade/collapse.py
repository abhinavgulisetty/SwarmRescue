"""Stage 3 -- collapse by simulation, not by hand.

A blast destroys the columns and connections inside its radius (see
`cascade.damage`), and MuJoCo is run forward from there.  Slabs pancake, beams
rotate, debris piles form.  Nothing about the resulting rubble field is
authored: it is whatever the contact solver produced, so mass is conserved,
piles hold an angle of repose, and voids exist under leaning slabs.

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

from .damage import BlastDamage, sample_damage
from .params import CollapseParams
from .structure import Structure

#: Where settled keyframes live.  Regenerable from (seed, generator), so the
#: directory is a cache and is git-ignored.
CACHE_DIR = Path(__file__).resolve().parent / "cache"

#: Cache format version.  Bump to invalidate every stored keyframe.
CACHE_VERSION: int = 2

#: Where pulverized members are parked: far below the ground plane, out of the
#: rubble field and out of the render.  They keep their degrees of freedom (a
#: MuJoCo model's body count is fixed at compile time) but interact with
#: nothing.
_DEBRIS_DISPOSAL_Z: float = -500.0


@dataclass(frozen=True, slots=True)
class CollapseResult:
    """The settled outcome of one seeded collapse."""

    seed: int
    damage: BlastDamage
    qpos: np.ndarray
    steps_simulated: int
    settled_early: bool
    wall_time_s: float
    peak_kinetic_energy_j: float
    final_kinetic_energy_j: float
    model_hash: str

    @property
    def summary(self) -> str:
        mode = "quiescent" if self.settled_early else "time-limited"
        return (
            f"seed={self.seed} {self.damage.summary} | "
            f"steps={self.steps_simulated} ({mode}) "
            f"peakKE={self.peak_kinetic_energy_j / 1e6:.2f} MJ "
            f"in {self.wall_time_s:.1f} s"
        )


def model_hash(xml: str) -> str:
    """Stable digest of the model, used as part of the cache key."""
    return hashlib.sha256(xml.encode("utf-8")).hexdigest()[:16]


# --- instrumentation ---------------------------------------------------------


def kinetic_energy(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Total kinetic energy, 0.5 * v' M v, in joules."""
    momentum = np.zeros(model.nv, dtype=np.float64)
    mujoco.mj_mulM(model, data, momentum, data.qvel)
    return 0.5 * float(data.qvel @ momentum)


def contact_loads(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Summed contact normal force per body, in newtons.

    The compressive load path in this structure runs through *contact*, not
    through the welds: a column rests on the ground and carries the slab
    resting on it, while its welds mostly resist separation and carry only a
    few percent of the weight.  Measuring weld force alone reports ~2% of the
    true axial load, which is the trap this function exists to avoid.
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

    A member in equilibrium receives roughly its axial load at each end, from
    what it carries above and what supports it below, so the summed contact
    normal force is about twice the axial load.
    """
    contact = contact_loads(model, data)
    out: dict[str, float] = {}
    for member in structure.members:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, member.name)
        if body_id >= 0:
            out[member.name] = 0.5 * float(contact[body_id])
    return out


# --- applying damage ---------------------------------------------------------


def release_welds(
    model: mujoco.MjModel, data: mujoco.MjData, names: tuple[str, ...]
) -> int:
    """Deactivate the named equality constraints.  Returns how many released.

    This is the only mechanism by which structure is ever severed in Cascade,
    in Phase 0 and in the live thermal-damage collapses of Phase 2 alike.
    There is no scripted animation path to keep it consistent with.
    """
    released = 0
    for name in names:
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eq_id < 0:
            raise KeyError(f"no equality constraint named {name!r} in the model")
        data.eq_active[eq_id] = 0
        released += 1
    return released


def pulverize(
    model: mujoco.MjModel, data: mujoco.MjData, names: tuple[str, ...]
) -> None:
    """Destroy the named members: they stop carrying load and stop colliding.

    A real column does not fail by politely detaching -- it crushes and
    buckles, ceasing to occupy the space that held the slab up.  A rigid box
    cannot do that, so a destroyed member has its collision geometry disabled
    and its body dropped far below the ground plane.  The slab above then has
    genuinely nothing beneath it.

    Reserved for columns.  Applying it to everything was a mistake worth
    naming: pulverizing failed slabs and partitions deletes the debris field,
    and the debris field is exactly what *arrests* a real collapse -- rubble
    piles up and the falling mass lands on something.  Delete it and every
    collapse runs to completion.

    This mutates `model` (geom contype/conaffinity), so callers needing a
    pristine model must pass a per-run instance.
    """
    for name in names:
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
            dof = model.jnt_dofadr[joint_id]
            data.qvel[dof : dof + 6] = 0.0


def apply_damage(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    damage: BlastDamage,
) -> None:
    """Put the blast's damage into the model, before any physics runs."""
    release_welds(model, data, damage.severed_welds)
    pulverize(model, data, damage.destroyed_columns)
    mujoco.mj_forward(model, data)


# --- the collapse ------------------------------------------------------------


def simulate_collapse(
    model: mujoco.MjModel,
    structure: Structure,
    seed: int,
    params: CollapseParams,
    xml_hash: str,
) -> CollapseResult:
    """Apply the blast and let the physics produce the rubble."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    damage = sample_damage(structure, seed, params)
    apply_damage(model, data, damage)

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
        damage=damage,
        qpos=data.qpos.copy(),
        steps_simulated=step,
        settled_early=settled_early,
        wall_time_s=time.perf_counter() - started,
        peak_kinetic_energy_j=peak_ke,
        final_kinetic_energy_j=kinetic_energy(model, data),
        model_hash=xml_hash,
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
        seed=result.seed,
        model_hash=result.model_hash,
        steps_simulated=result.steps_simulated,
        settled_early=result.settled_early,
        wall_time_s=result.wall_time_s,
        peak_kinetic_energy_j=result.peak_kinetic_energy_j,
        final_kinetic_energy_j=result.final_kinetic_energy_j,
        blast_center=np.array(result.damage.center),
        blast_radius=result.damage.radius_m,
        destroyed_columns=np.array(result.damage.destroyed_columns, dtype=object),
        freed_members=np.array(result.damage.freed_members, dtype=object),
        severed_welds=np.array(result.damage.severed_welds, dtype=object),
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
        damage = BlastDamage(
            center=tuple(blob["blast_center"].tolist()),
            radius_m=float(blob["blast_radius"]),
            destroyed_columns=tuple(blob["destroyed_columns"].tolist()),
            freed_members=tuple(blob["freed_members"].tolist()),
            severed_welds=tuple(blob["severed_welds"].tolist()),
        )
        return CollapseResult(
            seed=int(blob["seed"]),
            damage=damage,
            qpos=blob["qpos"],
            steps_simulated=int(blob["steps_simulated"]),
            settled_early=bool(blob["settled_early"]),
            wall_time_s=float(blob["wall_time_s"]),
            peak_kinetic_energy_j=float(blob["peak_kinetic_energy_j"]),
            final_kinetic_energy_j=float(blob["final_kinetic_energy_j"]),
            model_hash=str(blob["model_hash"]),
        )


def apply_settled_state(
    model: mujoco.MjModel, data: mujoco.MjData, result: CollapseResult
) -> None:
    """Place the world in its post-collapse configuration, at rest.

    Restoring `qpos` alone is not enough: the severed welds and the disabled
    collision geometry are model/data state that positions do not carry, so a
    cached rubble field would otherwise be propped up by columns that the
    blast destroyed.
    """
    if result.qpos.shape[0] != model.nq:
        raise ValueError(
            f"cached qpos has {result.qpos.shape[0]} entries but the model "
            f"expects {model.nq}; the cache is stale"
        )
    apply_damage(model, data, result.damage)
    data.qpos[:] = result.qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def collapse_for_seed(
    model: mujoco.MjModel,
    structure: Structure,
    seed: int,
    params: CollapseParams,
    xml_hash: str,
    *,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> CollapseResult:
    """Cached entry point: simulate the collapse once per (seed, model)."""
    if use_cache:
        cached = load_collapse(seed, xml_hash, cache_dir)
        if cached is not None:
            return cached
    result = simulate_collapse(model, structure, seed, params, xml_hash)
    if use_cache:
        save_collapse(result, cache_dir)
    return result


def rubble_statistics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    structure: Structure,
    reference_z: np.ndarray,
    damage: BlastDamage,
) -> dict[str, float]:
    """Summary of what the collapse actually did, for tests and reporting."""
    destroyed = set(damage.destroyed_columns)
    drops: list[float] = []
    standing = 0
    fallen = 0
    for member in structure.members:
        if member.name in destroyed:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, member.name)
        if body_id < 0:
            continue
        drop = float(reference_z[body_id] - data.xpos[body_id, 2])
        drops.append(drop)
        if drop > 1.0:
            fallen += 1
        else:
            standing += 1

    total = max(1, standing + fallen)
    return {
        "max_drop_m": max(drops) if drops else 0.0,
        "mean_drop_m": float(np.mean(drops)) if drops else 0.0,
        "fallen_fraction": fallen / total,
        "standing_fraction": standing / total,
        "n_destroyed": float(len(destroyed)),
    }
