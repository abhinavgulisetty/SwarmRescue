"""Cascade -- procedural generation of collapsed, burning multi-storey buildings.

The rubble a robot must traverse is the settled output of a simulated
structural failure, never hand-placed geometry.

MuJoCo chooses its OpenGL implementation while :mod:`mujoco` is imported, not
when a renderer is constructed.  Package initialisation is consequently the
one reliable place to select the offscreen backend for every ``cascade`` entry
point: it runs before ``cascade.demo`` or ``cascade.view`` can import MuJoCo.
An environment value supplied by the caller always wins.
"""

from __future__ import annotations

import os
import sys

# EGL creates a GPU-backed context without connecting to X11 or Wayland.  The
# demo and image viewer are offscreen by default, so this also makes their CLI
# entry points work in CI and over SSH.  Other platforms retain MuJoCo's native
# default.  Set MUJOCO_GL=glfw before Python starts for the interactive viewer.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

__all__ = ["floorplan", "geometry", "params", "rng", "thermal"]
