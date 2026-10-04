"""MJCF emission -- turn a `Structure` into a MuJoCo model.

The XML is built with ElementTree rather than string templates, so a malformed
attribute is a Python error rather than a silent MuJoCo parse failure.

Two MuJoCo details worth stating, because both are load-bearing:

* `equality/weld` is emitted **without** a `relpose`.  Its default has a zero
  quaternion w-component, which instructs the compiler to capture the relative
  pose from the reference configuration -- exactly what "these members are
  currently joined" means.  Supplying an explicit identity relpose instead
  would snap every member onto its neighbour's origin at compile time.
* Every weld is named, because releasing one at runtime through
  `data.eq_active[...]` is the primitive the entire collapse model rests on.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

from .look import add_assets, add_ground, add_lighting, add_visual
from .params import GRAVITY_M_S2, BuildingParams
from .structure import Member, Structure, overlapping_pairs

#: Solver configuration, chosen by benchmark *and* a correctness check.
#:
#: The structure is heavily over-constrained on purpose -- 18 ground columns
#: welded to the world, each slab welded to every column under it -- so the
#: constraint system is redundant and stiff.  That punishes an inexact solver.
#:
#: Measured on the reference 4-storey model (320 members, 360 welds), intact:
#:
#:     Newton30  + pyramidal :  39 ms/step   KE after 1 step = 212 J   converged
#:     Newton100 + elliptic  : 138 ms/step   KE after 1 step = 212 J   converged
#:     CG20      + pyramidal :   7 ms/step   KE after 1 step = 1.98e4 J  DIVERGED
#:
#: CG looks 5x faster and is worthless: it fails to converge, MuJoCo hits
#: mjWARN_CONTACTFULL and auto-resets, and the "fast" number is the speed of a
#: simulation that is silently exploding and restarting every other step.
#:
#: The check that catches this is cheap and belongs in the test suite: after
#: one step from rest, an intact building's kinetic energy must equal the
#: analytic free-fall value 0.5 * M * (g*dt)^2 (215.6 J here).  Anything
#: substantially above that means the solver is inventing energy.  Wall-clock
#: benchmarks alone cannot distinguish "fast" from "wrong".
_SOLVER = "Newton"
#: 30 already converges; 50 leaves margin for the contact-rich collapse phase,
#: where the constraint set is far larger than in the intact benchmark.
_SOLVER_ITERATIONS = "50"
_SOLVER_LS_ITERATIONS = "20"
_FRICTION_CONE = "pyramidal"
#: 3 = frictional contact without torsional/rolling rows.  Torsion buys little
#: for boxy debris and costs two constraint rows per contact.
_CONDIM = "3"

#: Contact softness for structural members.  Stiffer than the MuJoCo default
#: so that slabs stack on rubble instead of sinking into it.
_SOLREF = "0.004 1"
_SOLIMP = "0.95 0.99 0.001"

#: Sliding/torsional friction of broken concrete -- high, so debris piles hold
#: an angle of repose instead of sliding flat.
_FRICTION = "1.0 0.02 0.005"

#: Slop allowed before a weld is considered violated.  Tight: a building that
#: sags visibly while intact looks wrong.
_WELD_SOLREF = "0.002 1"
_WELD_SOLIMP = "0.97 0.999 0.001"


def to_mjcf(
    structure: Structure,
    *,
    model_name: str = "cascade",
    timestep_s: float = 0.002,
    extra_worldbody: list[ET.Element] | None = None,
    extra_actuators: list[ET.Element] | None = None,
) -> str:
    """Serialise `structure` to an MJCF XML string."""
    params = structure.params
    root = ET.Element("mujoco", model=model_name)

    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root,
        "option",
        timestep=f"{timestep_s}",
        gravity=f"0 0 -{GRAVITY_M_S2}",
        integrator="implicitfast",
        cone=_FRICTION_CONE,
        solver=_SOLVER,
        iterations=_SOLVER_ITERATIONS,
        ls_iterations=_SOLVER_LS_ITERATIONS,
    )
    # Contacts and constraints both scale with the member count; let MuJoCo
    # size its arena from the model rather than guessing a byte budget.
    ET.SubElement(root, "size", memory="256M")

    add_visual(root)
    _add_defaults(root)
    add_assets(root)
    world = _add_worldbody(root, structure, params)

    for element in extra_worldbody or ():
        world.append(element)

    _add_equality(root, structure)
    _add_contact_exclusions(root, structure)

    if extra_actuators:
        actuator = ET.SubElement(root, "actuator")
        for element in extra_actuators:
            actuator.append(element)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def _add_defaults(root: ET.Element) -> None:
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "geom",
        type="box",
        condim=_CONDIM,
        friction=_FRICTION,
        solref=_SOLREF,
        solimp=_SOLIMP,
    )


def _add_worldbody(
    root: ET.Element, structure: Structure, params: BuildingParams
) -> ET.Element:
    world = ET.SubElement(root, "worldbody")
    add_lighting(world, params.footprint_m, params.total_height_m)
    add_ground(world, params.footprint_m, _FRICTION)

    for member in structure.members:
        _add_member(world, member)
    return world


def _add_member(world: ET.Element, member: Member) -> None:
    cx, cy, cz = member.center
    hx, hy, hz = member.half_extents
    attrs = {"name": member.name, "pos": f"{cx} {cy} {cz}"}
    if member.quat != (1.0, 0.0, 0.0, 0.0):
        attrs["quat"] = " ".join(f"{q}" for q in member.quat)
    body = ET.SubElement(world, "body", **attrs)
    # Every structural member is a free body.  It is held up by its welds, not
    # by being nailed to the world -- which is what lets it fall when they go.
    ET.SubElement(body, "freejoint", name=f"j_{member.name}")
    ET.SubElement(
        body,
        "geom",
        name=f"g_{member.name}",
        size=f"{hx} {hy} {hz}",
        density=f"{member.density}",
        material=f"mat_{member.kind.value}",
    )


def _add_equality(root: ET.Element, structure: Structure) -> None:
    equality = ET.SubElement(root, "equality")
    for weld in structure.welds:
        attrs = {
            "name": weld.name,
            "body1": weld.body_a,
            "solref": _WELD_SOLREF,
            "solimp": _WELD_SOLIMP,
        }
        # Omitting body2 welds to the world -- a storey-0 column's ground fixing.
        if weld.body_b is not None:
            attrs["body2"] = weld.body_b
        ET.SubElement(equality, "weld", **attrs)


def _add_contact_exclusions(root: ET.Element, structure: Structure) -> None:
    pairs = overlapping_pairs(structure.members)
    if not pairs:
        return
    contact = ET.SubElement(root, "contact")
    for a, b in pairs:
        ET.SubElement(contact, "exclude", body1=a, body2=b)


def write_mjcf(path: str, xml: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(xml)


#: A member that ends the collapse within this of where it was drawn is taken
#: not to have moved, and is written back at its authored pose.
#:
#: Welds in MuJoCo are soft, so four thousand steps of a violent collapse leave
#: every *surviving* member a few centimetres from where it started even when
#: nothing touched it.  That drift is integrator noise and it is not harmless:
#: a stair flight and the landing it meets each drifted about 0.06 m in
#: opposite directions, which turned a flush junction into a 0.12 m lip at the
#: crest of a ramp -- the one place a robot is pitched up on its rear wheels
#: and least able to mount anything.  Every robot in the swarm teetered there
#: and slid back down.
#:
#: The tolerance has to stay small, because sag *is* real damage: a slab whose
#: columns have gone drops, and snapping that back would be a lie about the
#: scenario.  Anything that has moved further than a wheel radius is left
#: exactly where the physics put it.
SNAP_TOLERANCE_M: float = 0.12
#: ...and the same for rotation, as the cosine of half the angle.  About 6
#: degrees; a member that has tipped further has genuinely tipped.
SNAP_TOLERANCE_COS: float = 0.9986


def settled_poses(
    model, data, structure: Structure, *, snap_to_authored: bool = True
) -> dict[str, tuple]:
    """Read each surviving member's world pose out of a settled MjData."""
    import mujoco
    import numpy as np

    out: dict[str, tuple] = {}
    for member in structure.members:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, member.name)
        if body < 0:
            continue
        geom_start = model.body_geomadr[body]
        if geom_start >= 0 and model.geom_contype[geom_start] == 0:
            continue  # pulverized by the blast; it no longer exists
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(data.xmat[body], dtype=np.float64))
        position = np.asarray(data.xpos[body], dtype=np.float64)

        authored_pos = np.asarray(member.center, dtype=np.float64)
        authored_quat = np.asarray(member.quat, dtype=np.float64)
        moved = float(np.linalg.norm(position - authored_pos))
        turned = abs(float(authored_quat @ quat))
        if (
            snap_to_authored
            and moved <= SNAP_TOLERANCE_M
            and turned >= SNAP_TOLERANCE_COS
        ):
            position, quat = authored_pos, authored_quat

        out[member.name] = (
            tuple(float(v) for v in position),
            tuple(float(v) for v in quat),
        )
    return out


def to_settled_mjcf(
    structure: Structure,
    poses: dict[str, tuple],
    *,
    model_name: str = "cascade_settled",
    timestep_s: float = 0.002,
    extra_worldbody: list[ET.Element] | None = None,
    extra_actuators: list[ET.Element] | None = None,
) -> str:
    """Emit the settled scene with the rubble baked in as static geometry.

    After the collapse the debris is at rest, and simulating 500-odd free
    bodies that are not going anywhere costs ~30 ms/step -- half an hour of
    wall-clock to drive a robot across the building.  Freezing them at their
    settled poses leaves the robot as the only dynamic body, and the same
    traversal runs in about a minute.

    The cost is explicit and worth stating rather than burying: rubble in this
    model cannot shift, tip, or crush the robot.  The proposal wants that
    (SS4.2 stage 4), so this is the mode for Phase 0's navigability demo, not
    for evaluating survival under settling debris.  Members absent from
    `poses` were pulverized by the blast and are simply not emitted.
    """
    params = structure.params
    root = ET.Element("mujoco", model=model_name)

    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root,
        "option",
        timestep=f"{timestep_s}",
        gravity=f"0 0 -{GRAVITY_M_S2}",
        integrator="implicitfast",
        cone=_FRICTION_CONE,
        solver=_SOLVER,
        iterations=_SOLVER_ITERATIONS,
        ls_iterations=_SOLVER_LS_ITERATIONS,
    )
    ET.SubElement(root, "size", memory="64M")
    add_visual(root)
    _add_defaults(root)
    add_assets(root)

    world = ET.SubElement(root, "worldbody")
    add_lighting(world, params.footprint_m, params.total_height_m)
    add_ground(world, params.footprint_m, _FRICTION)
    for member in structure.members:
        pose = poses.get(member.name)
        if pose is None:
            continue
        (px, py, pz), quat = pose
        hx, hy, hz = member.half_extents
        ET.SubElement(
            world,
            "geom",
            name=f"g_{member.name}",
            type="box",
            size=f"{hx} {hy} {hz}",
            pos=f"{px} {py} {pz}",
            quat=" ".join(f"{q}" for q in quat),
            material=f"mat_{member.kind.value}",
        )

    for element in extra_worldbody or ():
        world.append(element)
    if extra_actuators:
        actuator = ET.SubElement(root, "actuator")
        for element in extra_actuators:
            actuator.append(element)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")
