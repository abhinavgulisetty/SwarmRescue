"""Contracts for the 20-world command-center renderer."""

from __future__ import annotations

import numpy as np
import pytest

from eval.world_wall import (
    HEIGHT,
    WIDTH,
    WorldReplay,
    _grid_shape,
    _hazard_color,
    _summary,
    render_frame,
)


def _replay(seed: int = 0) -> WorldReplay:
    return WorldReplay(
        seed=seed,
        positions=np.asarray(((0.0, 0.0), (0.5, 1.0), (1.0, 0.0))),
        edges=np.asarray(((0, 1), (1, 2)), dtype=np.int64),
        exits=(2,),
        survivors=(0,),
        robots=(0,),
        hazard=np.asarray(((0.1, 0.2, 0.3), (0.2, 0.8, 0.4))),
        ember_routes=((0, 1, 2),),
        flooding_routes=((0, 1, 2),),
        ember_objective=2.0,
        flooding_objective=3.0,
        ember_bytes=100,
        flooding_bytes=250,
        inference_sweeps=10,
        routing_sweeps=20,
    )


def test_world_wall_layout_and_summary_support_twenty_parallel_worlds() -> None:
    replays = tuple(_replay(seed) for seed in range(20))

    assert _grid_shape(20) == (5, 4)
    assert _summary(replays).survivor_count == 20
    assert _summary(replays).ember_wins == 20
    assert _summary(replays).byte_reduction == pytest.approx(0.6)
    with pytest.raises(ValueError, match="between 1 and 20"):
        _grid_shape(21)


def test_world_wall_renders_a_stable_rgb_frame() -> None:
    frame = render_frame((_replay(),), elapsed_s=6.0)

    assert frame.shape == (HEIGHT, WIDTH, 3)
    assert frame.dtype == np.uint8
    assert len(set(_hazard_color(value, 0.0, 1.0) for value in (0.0, 0.5, 1.0))) == 3
