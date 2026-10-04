"""Checks for B1/B2 route baselines and common metrics."""

from __future__ import annotations

import numpy as np

from cascade.extract import TraversabilityGraph, VoxelGrid
from ember.baselines import (
    centralized_shortest_baseline,
    flooding_dijkstra_baseline,
    last_received_mean_map,
)
from ember.comms import packetize_gaussians


def _diamond_graph() -> TraversabilityGraph:
    occupied = np.zeros((5, 1, 1), dtype=bool)
    return TraversabilityGraph(
        VoxelGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), occupied, occupied),
        np.column_stack((np.arange(5), np.zeros(5), np.zeros(5))).astype(np.int32),
        np.zeros(5),
        np.asarray(((0, 1), (1, 3), (0, 2), (2, 4), (4, 3))),
        np.ones(5),
    )


def test_b1_shortest_and_b2_shared_hazard_take_different_routes() -> None:
    graph = _diamond_graph()
    truth = np.asarray((0.1, 8.0, 0.1, 0.1, 0.1))
    shared = truth.copy()

    shortest = centralized_shortest_baseline(graph, 0, 3, truth)
    flooding = flooding_dijkstra_baseline(graph, 0, 3, shared, truth, radio_bytes=732)

    assert shortest.route == (0, 1, 3)
    assert flooding.route == (0, 2, 4, 3)
    assert flooding.path_length > shortest.path_length
    assert flooding.hazard_dose < shortest.hazard_dose
    assert flooding.radio_bytes == 732


def test_b2_last_received_map_replaces_cells_without_gbp_coupling() -> None:
    older = packetize_gaussians(
        0,
        0,
        np.asarray((1, 2)),
        np.asarray((2.0, 3.0)),
        np.asarray((0.25, 0.25)),
    )
    newer = packetize_gaussians(
        1,
        0,
        np.asarray((2,)),
        np.asarray((7.0,)),
        np.asarray((4.0,)),
    )

    mean = last_received_mean_map(
        4,
        older + newer,
        prior_mean=0.5,
    )

    np.testing.assert_allclose(mean, (0.5, 2.0, 7.0, 0.5), rtol=1.0e-4)
