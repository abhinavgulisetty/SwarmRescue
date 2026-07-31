"""Look at a generated world -- as images, or interactively.

    python -m cascade.view --seed 7                 # intact, writes PNGs
    python -m cascade.view --seed 7 --collapsed     # after the collapse
    python -m cascade.view --seed 7 --interactive   # live MuJoCo window

Rendering exists as a first-class entry point rather than a debugging
afterthought.  Three of the defects found so far -- two missing facades, stairs
running into the ceiling, and a stairwell capped by its own slab -- were
invisible to every numeric check and obvious the moment the model was drawn.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco

from .collapse import apply_settled_state, collapse_for_seed, model_hash
from .floorplan import build_floorplan
from .mjcf import settled_poses, to_mjcf, to_settled_mjcf
from .params import CascadeParams

# Importing .render first fixes the GL backend before mujoco is loaded.
from .render import CameraView, contact_sheet, render_scene, save_png
from .structure import MemberKind, build_structure

#: Members hidden by --cutaway, so the interior is visible from outside.
_CUTAWAY_HIDDEN = frozenset({MemberKind.WALL})


def _views(params, stairwell) -> list[CameraView]:
    width, depth = params.building.footprint_m
    height = params.building.total_height_m
    centre = (0.5 * width, 0.5 * depth, 0.40 * height)
    span = max(width, depth, height)
    sx, sy = stairwell.center
    return [
        CameraView("iso", centre, span * 2.1, 45.0, -20.0),
        CameraView("elevation", centre, span * 2.0, 90.0, -8.0),
        CameraView("plan", (0.5 * width, 0.5 * depth, 0.0), span * 1.9, 90.0, -89.0),
        CameraView("stairwell", (sx, sy, 0.45 * height), span * 0.95, 20.0, -10.0),
    ]


def build_scene(
    seed: int,
    params: CascadeParams,
    *,
    collapsed: bool,
    cutaway: bool,
    use_cache: bool = True,
):
    """Return (model, data, plan) for the requested view of `seed`."""
    plan = build_floorplan(seed, params.building)
    structure = build_structure(plan)

    if cutaway:
        import dataclasses

        keep = {
            m.name for m in structure.members if m.kind not in _CUTAWAY_HIDDEN
        }
        structure = dataclasses.replace(
            structure,
            members=tuple(m for m in structure.members if m.name in keep),
            welds=tuple(
                w
                for w in structure.welds
                if w.body_a in keep and (w.body_b is None or w.body_b in keep)
            ),
        )

    xml = to_mjcf(structure)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if not collapsed:
        return model, data, plan

    collapse = collapse_for_seed(
        model, structure, seed, params.collapse, model_hash(xml), use_cache=use_cache
    )
    print(f"[cascade] {collapse.summary}")
    apply_settled_state(model, data, collapse)

    # Re-emit as static geometry: nothing is moving any more, and a static
    # scene is far lighter for the interactive viewer to run.
    static = mujoco.MjModel.from_xml_string(
        to_settled_mjcf(structure, settled_poses(model, data, structure))
    )
    static_data = mujoco.MjData(static)
    mujoco.mj_forward(static, static_data)
    return static, static_data, plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cascade.view", description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--collapsed", action="store_true", help="show the post-collapse rubble"
    )
    parser.add_argument(
        "--cutaway",
        action="store_true",
        help="hide partitions so the interior and stairs are visible",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="open a live MuJoCo window instead of writing images",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/views"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=620)
    args = parser.parse_args(argv)

    params = CascadeParams()
    model, data, plan = build_scene(
        args.seed, params, collapsed=args.collapsed, cutaway=args.cutaway
    )
    state = "collapsed" if args.collapsed else "intact"

    if args.interactive:
        # Imported as a bare name: `import mujoco.viewer` here would rebind
        # `mujoco` as a local and shadow the module for the rest of main().
        from mujoco import viewer as mujoco_viewer

        print(
            f"[cascade] seed {args.seed} ({state}) -- drag to orbit, "
            "scroll to zoom, Esc to quit"
        )
        mujoco_viewer.launch(model, data)
        return 0

    views = _views(params, plan.stairwells()[0].rect)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        frames = [render_scene(model, data, v, renderer=renderer) for v in views]
    finally:
        renderer.close()

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f"{state}{'_cutaway' if args.cutaway else ''}"
    for view, frame in zip(views, frames, strict=True):
        save_png(args.out / f"seed{args.seed}_{suffix}_{view.name}.png", frame)
    sheet = save_png(
        args.out / f"seed{args.seed}_{suffix}.png", contact_sheet(frames, columns=2)
    )
    print(f"[cascade] wrote {len(frames)} views + contact sheet -> {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
