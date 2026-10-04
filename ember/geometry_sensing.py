"""Segment raycasts against current MuJoCo structural box poses.

The building generator uses oriented boxes. These queries use collision geometry
and current data transforms, independently of cinematic opacity. Robot geoms and
collision-disabled debris are excluded. Non-box geometry is outside this model.
"""

from __future__ import annotations

import mujoco
import numpy as np


def box_raycasts(model, data, origin_m, targets_m) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest structural hit distance and intersected wall-box counts."""
    origin = np.asarray(origin_m, dtype=float)
    targets = np.asarray(targets_m, dtype=float)
    if origin.shape != (3,) or targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("ray origins and targets must be 3D")
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(targets)):
        raise ValueError("ray coordinates must be finite")
    delta = targets - origin
    lengths = np.linalg.norm(delta, axis=1)
    nearest = np.full(len(targets), np.inf)
    walls = np.zeros(len(targets), dtype=np.int32)
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if "robot" in name or (
            model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0
        ):
            continue
        rotation = np.asarray(data.geom_xmat[geom]).reshape(3, 3)
        local_origin = (origin - data.geom_xpos[geom]) @ rotation
        directions = delta @ rotation
        half = model.geom_size[geom]
        parallel = np.abs(directions) < 1e-12
        first = np.divide(
            -half - local_origin, directions,
            out=np.full(directions.shape, -np.inf), where=~parallel,
        )
        second = np.divide(
            half - local_origin, directions,
            out=np.full(directions.shape, np.inf), where=~parallel,
        )
        entry = np.max(np.minimum(first, second), axis=1)
        leave = np.min(np.maximum(first, second), axis=1)
        outside_parallel = np.any(parallel & (np.abs(local_origin) > half), axis=1)
        hit = (
            ~outside_parallel & (entry <= leave) & (leave > 1e-9)
            & (entry < 1.0 - 1e-9) & (lengths > 0.0)
        )
        nearest[hit] = np.minimum(
            nearest[hit], np.maximum(entry[hit], 0) * lengths[hit]
        )
        if name.startswith("g_wall_") or name.startswith("wall_"):
            walls[hit] += 1
    return nearest, walls


def radio_wall_counts(model, data, positions_m) -> np.ndarray:
    """Wall attenuation inputs evaluated at current robot positions and poses."""
    positions = np.asarray(positions_m, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("radio positions must have shape (N, 3)")
    counts = np.zeros((len(positions), len(positions)), dtype=np.int32)
    for sender, origin in enumerate(positions):
        _, counts[sender] = box_raycasts(model, data, origin, positions)
    return counts
