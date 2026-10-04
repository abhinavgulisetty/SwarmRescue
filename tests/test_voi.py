"""Phase-2 adjoint and value-of-information scheduling contracts."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.extract import TraversabilityGraph, VoxelGrid
from ember.comms import PACKET_HEADER_BYTES, UPDATE_BYTES
from ember.routing import LMDPConfig, build_lmdp
from ember.voi import (
    TokenBucket,
    gaussian_kl,
    schedule_gaussian_updates,
    solve_adjoint,
)


def _chain(count: int = 4):
    cells = np.column_stack(
        (
            np.arange(count, dtype=np.int32),
            np.zeros(count, dtype=np.int32),
            np.zeros(count, dtype=np.int32),
        )
    )
    occupied = np.ones((count, 1, 1), dtype=bool)
    graph = TraversabilityGraph(
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
    return build_lmdp(
        graph,
        np.concatenate((np.full(count - 1, 0.1), np.zeros(1))),
        (count - 1,),
        config=LMDPConfig(tolerance=1.0e-10),
    )


def test_adjoint_converges_on_transposed_lmdp() -> None:
    model = _chain()
    source = np.zeros(model.n_states)
    source[0] = 1.0

    result = solve_adjoint(model, source)

    assert result.converged
    assert result.residual <= model.config.tolerance
    assert np.all(result.field >= 0.0)
    assert result.field[1] > result.field[3]


def test_voi_selects_route_relevant_cell_under_one_update_budget() -> None:
    count = 4
    bucket = TokenBucket(
        capacity_bytes=PACKET_HEADER_BYTES + UPDATE_BYTES,
        refill_bytes=UPDATE_BYTES,
    )
    relevance = np.array((0.1, 4.0, 0.5, 0.0))
    schedule = schedule_gaussian_updates(
        sender=0,
        sequence=7,
        cells=np.arange(count),
        prior_mean=np.zeros(count),
        prior_variance=np.ones(count),
        posterior_mean=np.ones(count),
        posterior_variance=np.full(count, 0.5),
        adjoint=relevance,
        bucket=bucket,
    )

    assert schedule.selected_cells == (1,)
    assert len(schedule.packets) == 1
    assert schedule.bytes_scheduled == PACKET_HEADER_BYTES + UPDATE_BYTES
    assert schedule.nats_per_byte > 0.0
    assert bucket.tokens == 0


def test_zero_relevance_stays_silent_and_refill_is_bounded() -> None:
    bucket = TokenBucket(capacity_bytes=64, refill_bytes=20)
    bucket.consume(60)
    schedule = schedule_gaussian_updates(
        sender=0,
        sequence=0,
        cells=np.arange(2),
        prior_mean=np.zeros(2),
        prior_variance=np.ones(2),
        posterior_mean=np.ones(2),
        posterior_variance=np.full(2, 0.5),
        adjoint=np.zeros(2),
        bucket=bucket,
    )

    assert schedule.packets == ()
    assert schedule.bytes_scheduled == 0
    bucket.refill()
    bucket.refill()
    bucket.refill()
    bucket.refill()
    assert bucket.tokens == bucket.capacity_bytes


def test_gaussian_kl_rejects_nonpositive_variance() -> None:
    with pytest.raises(ValueError, match="positive variance"):
        gaussian_kl(
            np.zeros(1),
            np.ones(1),
            np.zeros(1),
            np.zeros(1),
        )
