"""The scene's visual identity, in one place.

Split out of `cascade.mjcf` because two different documents need it -- the
dynamic model that gets collapsed and the settled model the robots drive
through -- and because the emitter had accumulated two divergent copies of the
lighting rig.

The palette is doing work, not decoration.  The first cut gave every member
kind a shade of grey between 0.38 and 0.80 under a single overhead lamp, on a
near-black ground under a near-black sky, and the result was genuinely hard to
read: no horizon, so the building floated in a void with no sense of scale; no
value separation, so a slab that had pancaked onto the floor was the same tone
as the floor; and no directional light, so nothing had a lit face and a shaded
face to tell you which way it was pointing.

Three decisions follow from that:

* **A sky with a horizon.**  A gradient from steel blue overhead to pale dust
  at eye level, with the ground fading into the same dust.  It costs nothing
  and it is the difference between a photograph of a building and a render of
  a mesh.
* **Value separation between member kinds**, so frame, floor and partition are
  three distinct tones rather than three greys.
* **One warm hue in the whole building, reserved for the stairs.**  They are
  the only route between storeys and therefore the only thing in frame the
  viewer needs to find; everything else is deliberately neutral so that they
  are the first thing the eye lands on.
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET

from .structure import MemberKind

#: Offscreen framebuffer size.  MuJoCo defaults to 640x480 and *errors* rather
#: than downscaling when asked for a larger frame, so this has to be declared
#: in the model rather than chosen at render time.
OFFSCREEN_WIDTH: int = 1920
OFFSCREEN_HEIGHT: int = 1200


@dataclass(frozen=True, slots=True)
class Surface:
    """How one member kind is shaded."""

    rgb: tuple[float, float, float]
    #: Strength of the specular highlight.  Concrete is not glossy, but a
    #: little specular is what stops a flat-shaded box reading as a cutout.
    specular: float = 0.12
    shininess: float = 0.2
    reflectance: float = 0.0

    def as_attrs(self, name: str) -> dict[str, str]:
        r, g, b = self.rgb
        return {
            "name": name,
            "rgba": f"{r} {g} {b} 1",
            "specular": f"{self.specular}",
            "shininess": f"{self.shininess}",
            "reflectance": f"{self.reflectance}",
        }


#: Structural palette.  Cool neutrals for everything the robot cannot use, one
#: warm brick for the thing it climbs.
MEMBER_SURFACES: dict[MemberKind, Surface] = {
    MemberKind.COLUMN: Surface((0.29, 0.30, 0.33), specular=0.10),
    MemberKind.BEAM: Surface((0.37, 0.38, 0.41), specular=0.10),
    MemberKind.SLAB: Surface((0.66, 0.65, 0.62), specular=0.08, reflectance=0.02),
    MemberKind.WALL: Surface((0.85, 0.83, 0.78), specular=0.06),
    MemberKind.STAIR: Surface((0.60, 0.37, 0.25), specular=0.16, shininess=0.3),
}

#: Sky, top to bottom: overcast steel overhead, dust at the horizon.
SKY_ZENITH: tuple[float, float, float] = (0.33, 0.42, 0.54)
SKY_HORIZON: tuple[float, float, float] = (0.76, 0.73, 0.67)
#: Atmosphere.  Matched to the horizon so the ground plane's far edge dissolves
#: into the sky instead of ending in a hard line across the frame.
HAZE: tuple[float, float, float] = (0.74, 0.71, 0.66)

#: Ground: warm asphalt, two tones close enough together that the grid reads as
#: texture and scale rather than as a chessboard.
GROUND_LIGHT: tuple[float, float, float] = (0.345, 0.335, 0.32)
GROUND_DARK: tuple[float, float, float] = (0.31, 0.30, 0.29)

#: Metres per checker square.  Coarse enough not to alias into moire in the
#: middle distance, fine enough to give the eye a scale reference next to a
#: 0.34 m robot.
GROUND_TILE_M: float = 4.0

#: Key light direction, pointing down and across so vertical faces separate by
#: which way they face.  Straight down -- the previous rig -- lights every wall
#: identically and flattens the building into a plan.
SUN_DIR: tuple[float, float, float] = (-0.45, 0.62, -1.0)
SUN_DIFFUSE: tuple[float, float, float] = (0.74, 0.71, 0.65)
#: Cool fill from the opposite side, so shadowed faces are blue-grey rather
#: than black.  Interiors under a surviving slab are otherwise unreadable.
FILL_DIR: tuple[float, float, float] = (-0.55, -0.40, -0.55)
FILL_DIFFUSE: tuple[float, float, float] = (0.24, 0.28, 0.34)
#: Ambient, applied through the headlight.  Enough that a robot inside a room
#: is lit at all; low enough that the directional lights still model form.
AMBIENT: tuple[float, float, float] = (0.30, 0.31, 0.34)


# --- the overlay ---------------------------------------------------------------
#
# Kept here with the scene palette rather than next to the drawing code, so
# that the frame around the render and the render itself are demonstrably the
# same design and not two that happen to sit next to each other.  Values are
# 0-255, because that is what the imaging library wants.

#: Panel ground.  Near-black and slightly blue, so the warm dust of the render
#: reads as light coming out of the picture.
INK: tuple[int, int, int] = (13, 15, 19)
#: One step up, for the strip the readouts sit on.
INK_RAISED: tuple[int, int, int] = (20, 23, 29)
#: Hairline rules and empty progress track.
RULE: tuple[int, int, int] = (44, 49, 60)
TEXT_PRIMARY: tuple[int, int, int] = (233, 236, 242)
TEXT_MUTED: tuple[int, int, int] = (139, 148, 164)
TEXT_DIM: tuple[int, int, int] = (95, 103, 118)
#: The one accent, taken from the stair material: the same warmth marks the
#: route in the render and the live figures in the overlay.
ACCENT: tuple[int, int, int] = (214, 132, 78)

#: Status colours, keyed by `SwarmMember.status`.
STATUS_COLOURS: dict[str, tuple[int, int, int]] = {
    "arrived": (126, 208, 142),
    "stalled": (226, 122, 110),
    "driving": (168, 178, 196),
    "waiting": (120, 128, 144),
}


def _triple(value: tuple[float, float, float]) -> str:
    return " ".join(f"{c}" for c in value)


def add_visual(root: ET.Element) -> None:
    """Renderer-wide settings: framebuffer, atmosphere, shadow quality."""
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        offwidth=str(OFFSCREEN_WIDTH),
        offheight=str(OFFSCREEN_HEIGHT),
    )
    ET.SubElement(
        visual,
        "headlight",
        ambient=_triple(AMBIENT),
        diffuse="0.18 0.18 0.19",
        specular="0.08 0.08 0.08",
    )
    # Haze reads as dust and, more usefully, gives depth cues in a rubble field
    # where every surface is the same material.  It is also what makes the
    # infinite ground plane dissolve into the sky instead of running to a hard
    # line across the frame.
    # A tight shadow frustum, and a large map to fill it.  The shadow map has a
    # fixed texel budget however far it is stretched, and at the default
    # frustum a 24 m building spreads it thin enough that self-shadowing
    # aliases: the lit face of the block came back covered in black speckle.
    # The cost of tightening it is that debris thrown a long way clear of the
    # building stops casting, which is a fair trade for a clean frame.
    ET.SubElement(visual, "map", haze="0.35", shadowclip="0.5", shadowscale="0.6")
    ET.SubElement(visual, "rgba", haze=f"{_triple(HAZE)} 1")
    ET.SubElement(visual, "quality", shadowsize="8192", offsamples="8")


def add_assets(root: ET.Element) -> None:
    """Skybox, ground texture, and one material per member kind."""
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="sky",
        type="skybox",
        builtin="gradient",
        rgb1=_triple(SKY_ZENITH),
        rgb2=_triple(SKY_HORIZON),
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1=_triple(GROUND_DARK),
        rgb2=_triple(GROUND_LIGHT),
        width="512",
        height="512",
    )
    # `texuniform` makes texrepeat a rate per metre rather than per geom, which
    # is the only sane basis for an unbounded plane.
    tiles_per_m = 1.0 / GROUND_TILE_M
    ET.SubElement(
        asset,
        "material",
        name="ground",
        texture="grid",
        texrepeat=f"{tiles_per_m} {tiles_per_m}",
        texuniform="true",
        specular="0.04",
        shininess="0.1",
    )
    for kind, surface in MEMBER_SURFACES.items():
        ET.SubElement(asset, "material", **surface.as_attrs(f"mat_{kind.value}"))


def add_lighting(world: ET.Element, footprint_m, total_height_m: float) -> None:
    """A warm key with shadows and a cool fill without them."""
    width, depth = footprint_m
    ET.SubElement(
        world,
        "light",
        name="sun",
        pos=f"{width * 0.5} {depth * 0.5} {total_height_m + 12.0}",
        dir=_triple(SUN_DIR),
        directional="true",
        castshadow="true",
        diffuse=_triple(SUN_DIFFUSE),
        specular="0.18 0.18 0.18",
    )
    ET.SubElement(
        world,
        "light",
        name="fill",
        pos=f"{width * 0.5} {depth * 0.5} {total_height_m + 12.0}",
        dir=_triple(FILL_DIR),
        directional="true",
        castshadow="false",
        diffuse=_triple(FILL_DIFFUSE),
        specular="0 0 0",
    )


def add_ground(world: ET.Element, footprint_m, friction: str) -> ET.Element:
    """The ground plane, unbounded.

    Zero half-extents mean infinite, which settles two things at once: debris
    thrown clear of the building always lands on something, and the render
    gains a horizon.  A finite plane sized to the footprint put its own edge
    across the middle of every wide shot, with black sky beyond it -- the
    building appeared to be standing on a tabletop in space.
    """
    width, depth = footprint_m
    return ET.SubElement(
        world,
        "geom",
        name="ground",
        type="plane",
        size=f"0 0 {GROUND_TILE_M}",
        pos=f"{width * 0.5} {depth * 0.5} 0",
        material="ground",
        condim="3",
        friction=friction,
    )
