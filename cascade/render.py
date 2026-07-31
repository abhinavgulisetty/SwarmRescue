"""Offscreen rendering of a Cascade world.

Exists for verification as much as for the demo.  A building generator is very
easy to get subtly, grossly wrong in ways that summary statistics do not
reveal -- room areas can tile perfectly while the rooms themselves are
nonsense -- so being able to look at the thing is a first-class requirement,
not a presentation nicety.

Backend note: on this machine EGL and OSMesa both fail to initialise and GLFW
works, so `MUJOCO_GL` is defaulted to glfw unless the caller has already
chosen.  That has to happen before `mujoco` is imported anywhere.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "glfw")

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

#: Default render size.  Large enough to see rubble structure, small enough
#: that a batch of seeds renders in seconds.
DEFAULT_WIDTH: int = 1280
DEFAULT_HEIGHT: int = 800


@dataclass(frozen=True, slots=True)
class CameraView:
    """A free-camera pose, in MuJoCo's azimuth/elevation convention."""

    name: str
    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float

    def apply(self, camera: mujoco.MjvCamera) -> None:
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = self.lookat
        camera.distance = self.distance
        camera.azimuth = self.azimuth
        camera.elevation = self.elevation


def standard_views(
    footprint_m: tuple[float, float], total_height_m: float
) -> tuple[CameraView, ...]:
    """Views that between them expose the failure modes worth catching."""
    width, depth = footprint_m
    center = (0.5 * width, 0.5 * depth, 0.4 * total_height_m)
    span = max(width, depth, total_height_m)
    return (
        CameraView("iso", center, span * 2.1, 45.0, -20.0),
        CameraView("front", center, span * 2.0, 90.0, -8.0),
        CameraView(
            "plan",
            (0.5 * width, 0.5 * depth, 0.0),
            span * 1.9,
            90.0,
            -89.0,
        ),
    )


def render_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    view: CameraView,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    renderer: mujoco.Renderer | None = None,
    overlay: Callable[[mujoco.MjvScene], None] | None = None,
) -> np.ndarray:
    """Render one frame as an RGB array.

    Pass an existing `renderer` when rendering many frames: constructing one
    allocates a GL context, which is far more expensive than a frame.

    `overlay` is handed the populated scene just before it is drawn, which is
    the only point at which geometry that is not in the model -- a planned
    route, a trail -- can be added to the picture.
    """
    own = renderer is None
    active = renderer or mujoco.Renderer(model, height=height, width=width)
    try:
        camera = mujoco.MjvCamera()
        view.apply(camera)
        active.update_scene(data, camera=camera)
        if overlay is not None:
            overlay(active.scene)
        return active.render()
    finally:
        if own:
            active.close()


def save_png(path: str | Path, image: np.ndarray) -> Path:
    """Write an RGB array to disk, creating parent directories."""
    import imageio.v3 as iio

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, image)
    return out


def contact_sheet(images: list[np.ndarray], columns: int = 2) -> np.ndarray:
    """Tile frames into a single image, padding the last row if needed."""
    if not images:
        raise ValueError("contact_sheet needs at least one image")
    height, width, channels = images[0].shape
    if any(img.shape != images[0].shape for img in images):
        raise ValueError("contact_sheet requires uniformly sized frames")

    rows = (len(images) + columns - 1) // columns
    sheet = np.zeros((rows * height, columns * width, channels), dtype=images[0].dtype)
    for index, image in enumerate(images):
        r, c = divmod(index, columns)
        sheet[r * height : (r + 1) * height, c * width : (c + 1) * width] = image
    return sheet
