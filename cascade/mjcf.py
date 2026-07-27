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

from .floorplan import FloorPlan
from .params import GRAVITY_M_S2, BuildingParams
from .structure import Member, MemberKind, Structure, overlapping_pairs

#: Visual material per member kind: (rgba,) chosen so storeys read clearly in
#: a render and rubble is distinguishable from intact structure.
_MATERIALS: dict[MemberKind, tuple[float, float, float, float]] = {
    MemberKind.COLUMN: (0.46, 0.47, 0.50, 1.0),
    MemberKind.BEAM: (0.38, 0.39, 0.42, 1.0),
    MemberKind.SLAB: (0.62, 0.61, 0.58, 1.0),
    MemberKind.WALL: (0.80, 0.78, 0.73, 1.0),
    MemberKind.STAIR: (0.55, 0.50, 0.44, 1.0),
}

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
    plan: FloorPlan,
    *,
    model_name: str = "cascade",
    timestep_s: float = 0.002,
    extra_worldbody: list[ET.Element] | None = None,
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

    _add_visual(root)
    _add_defaults(root)
    _add_assets(root, plan)
    world = _add_worldbody(root, structure, params, plan)

    for element in extra_worldbody or ():
        world.append(element)

    _add_equality(root, structure)
    _add_contact_exclusions(root, structure)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


#: Offscreen framebuffer size.  MuJoCo defaults to 640x480 and *errors* rather
#: than downscaling when asked for a larger frame, so this has to be declared
#: in the model rather than chosen at render time.
_OFFSCREEN_WIDTH: int = 1920
_OFFSCREEN_HEIGHT: int = 1200


def _add_visual(root: ET.Element) -> None:
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        offwidth=str(_OFFSCREEN_WIDTH),
        offheight=str(_OFFSCREEN_HEIGHT),
    )
    # A little atmospheric haze reads as dust and gives depth cues in a
    # monochrome rubble field, where everything is the same grey.
    ET.SubElement(visual, "map", fogstart="25", fogend="90", haze="0.25")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="8")


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


def _add_assets(root: ET.Element, plan: FloorPlan) -> None:
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.12 0.14 0.18",
        rgb2="0.02 0.02 0.03",
        width="256",
        height="256",
    )
    ET.SubElement(
        asset,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1="0.16 0.17 0.19",
        rgb2="0.21 0.22 0.24",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "material",
        name="ground",
        texture="grid",
        texrepeat="12 12",
        reflectance="0.05",
    )
    for kind, rgba in _MATERIALS.items():
        ET.SubElement(
            asset,
            "material",
            name=f"mat_{kind.value}",
            rgba=" ".join(f"{c}" for c in rgba),
        )


def _add_worldbody(
    root: ET.Element,
    structure: Structure,
    params: BuildingParams,
    plan: FloorPlan,
) -> ET.Element:
    world = ET.SubElement(root, "worldbody")

    width, depth = params.footprint_m
    ET.SubElement(
        world,
        "light",
        name="key",
        pos=f"{width * 0.5} {depth * 0.5} {params.total_height_m + 8.0}",
        dir="0 0 -1",
        directional="true",
        diffuse="0.7 0.7 0.7",
    )
    # The ground plane extends well past the footprint so debris thrown clear
    # of the building still lands on something.
    ET.SubElement(
        world,
        "geom",
        name="ground",
        type="plane",
        size=f"{width} {depth} 0.5",
        pos=f"{width * 0.5} {depth * 0.5} 0",
        material="ground",
        condim="3",
        friction=_FRICTION,
    )

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
