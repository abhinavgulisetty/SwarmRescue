"""Test-process setup that must run before test modules import MuJoCo."""

from __future__ import annotations

import os
import sys

# Several test modules import mujoco before importing cascade.  MuJoCo freezes
# its GLContext choice during that first import, so package setup cannot repair
# the order afterward.  Mirror cascade's non-destructive Linux default at the
# earliest pytest hook while still allowing `MUJOCO_GL=... pytest` overrides.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
