"""Phase-2 safeguards for the quantized NCA routing accelerator."""

from __future__ import annotations

import numpy as np

from cascade.extract import TraversabilityGraph, VoxelGrid
from ember.routing import (
    Int8NCAWeights,
    LMDPConfig,
    build_lmdp,
    solve_jacobi,
    solve_safeguarded_nca,
)


def _chain(count: int) -> TraversabilityGraph:
    cells = np.column_stack(
        (
            np.arange(count, dtype=np.int32),
            np.zeros(count, dtype=np.int32),
            np.zeros(count, dtype=np.int32),
        )
    )
    occupied = np.ones((count, 1, 1), dtype=bool)
    return TraversabilityGraph(
        grid=VoxelGrid(
            origin=(0.0, 0.0, 0.0),
            cell=(1.0, 1.0, 1.0),
            occupied=occupied,
            blocked=np.zeros_like(occupied),
        ),
        cells=cells,
        surface_z=np.full(count, 0.5),
        edges=np.asarray(
            [(index, index + 1) for index in range(count - 1)],
            dtype=np.int32,
        ),
        edge_cost=np.ones(count - 1),
    )


def _model(count: int = 12):
    return build_lmdp(
        _chain(count),
        np.concatenate((np.full(count - 1, 0.1), np.zeros(1))),
        (count - 1,),
        config=LMDPConfig(tolerance=1.0e-10, max_sweeps=5_000),
    )


def test_default_int8_nca_is_approximately_2_5_kb_and_matches_jacobi() -> None:
    model = _model()
    classical = solve_jacobi(model)
    accelerated = solve_safeguarded_nca(model)

    assert accelerated.parameter_bytes == 2_465
    assert accelerated.solution.converged
    assert accelerated.accepted_proposals > 0
    assert accelerated.solution.sweeps <= classical.sweeps
    np.testing.assert_allclose(
        accelerated.solution.desirability,
        classical.desirability,
        rtol=0.0,
        atol=3.0e-10,
    )


def test_corrupted_weights_cannot_break_convergence() -> None:
    model = _model()
    rng = np.random.default_rng(41)
    corrupted = Int8NCAWeights(
        rng.integers(-128, 128, (32, 8), dtype=np.int8),
        rng.integers(-128, 128, 32, dtype=np.int8),
        rng.integers(-128, 128, (64, 32), dtype=np.int8),
        rng.integers(-128, 128, 64, dtype=np.int8),
        rng.integers(-128, 128, 64, dtype=np.int8),
    )

    classical = solve_jacobi(model)
    safeguarded = solve_safeguarded_nca(model, corrupted)

    assert safeguarded.solution.converged
    assert safeguarded.rejected_proposals > 0
    assert safeguarded.worst_residual_ratio <= model.contraction_bound + 1.0e-12
    np.testing.assert_allclose(
        safeguarded.solution.desirability,
        classical.desirability,
        rtol=0.0,
        atol=3.0e-10,
    )


def test_nca_keeps_exit_and_death_boundaries_exact() -> None:
    graph = _chain(4)
    model = build_lmdp(
        graph,
        np.array((0.1, 0.1, 0.1, 0.0)),
        exit_nodes=(3,),
        death_nodes=(2,),
    )

    result = solve_safeguarded_nca(model)

    assert result.solution.desirability[3] == 1.0
    assert result.solution.desirability[2] == 0.0
