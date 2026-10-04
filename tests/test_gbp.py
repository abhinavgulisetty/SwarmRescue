"""Tests for scalar asynchronous Gaussian belief propagation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cascade.extract import TraversabilityGraph, VoxelGrid
from ember.gbp import GaussianBeliefPropagation, GBPConfig


def _line_graph(nodes: int) -> TraversabilityGraph:
    occupied = np.zeros((nodes, 1, 1), dtype=bool)
    return TraversabilityGraph(
        VoxelGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), occupied, occupied),
        np.column_stack(
            (
                np.arange(nodes, dtype=np.int32),
                np.zeros(nodes, dtype=np.int32),
                np.zeros(nodes, dtype=np.int32),
            )
        ),
        np.zeros(nodes),
        np.asarray([(index, index + 1) for index in range(nodes - 1)]),
        np.ones(nodes - 1),
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"edge_variance": 0.0},
        {"damping": 0.0},
        {"damping": 1.1},
        {"tolerance": 0.0},
        {"max_sweeps": 0},
    ),
)
def test_gbp_config_rejects_invalid_values(changes) -> None:
    with pytest.raises(ValueError):
        replace(GBPConfig(), **changes)


def test_tree_gbp_matches_direct_linear_gaussian_solution() -> None:
    graph = _line_graph(3)
    config = GBPConfig(edge_variance=0.5, damping=0.7, tolerance=1.0e-10)
    gbp = GaussianBeliefPropagation(graph, mean=0.0, variance=4.0, config=config)
    gbp.observe(np.asarray((0, 2)), np.asarray((2.0, -1.0)), np.asarray((0.2, 0.4)))

    result = gbp.run()

    precision = np.eye(3) / 4.0
    information = np.zeros(3)
    information[[0, 2]] += np.asarray((2.0 / 0.2, -1.0 / 0.4))
    precision[[0, 2], [0, 2]] += np.asarray((1.0 / 0.2, 1.0 / 0.4))
    for source, target in ((0, 1), (1, 2)):
        precision[source, source] += 2.0
        precision[target, target] += 2.0
        precision[source, target] -= 2.0
        precision[target, source] -= 2.0
    covariance = np.linalg.inv(precision)
    expected_mean = covariance @ information

    assert result.converged
    assert result.belief.mean == pytest.approx(expected_mean, abs=1.0e-8)
    assert result.belief.variance == pytest.approx(np.diag(covariance), abs=1.0e-8)


def test_message_loss_delays_propagation_but_later_sweeps_recover() -> None:
    graph = _line_graph(4)
    gbp = GaussianBeliefPropagation(
        graph,
        mean=0.0,
        variance=2.0,
        config=GBPConfig(damping=1.0, tolerance=1.0e-9),
    )
    gbp.observe(np.asarray((0,)), np.asarray((4.0,)), np.asarray((0.05,)))
    initial = gbp.belief().mean.copy()

    assert gbp.sweep(np.zeros(gbp.source.size, dtype=bool)) == 0.0
    assert gbp.belief().mean == pytest.approx(initial)

    result = gbp.run()
    assert result.converged
    assert result.belief.mean[-1] > 0.0


def test_age_increases_uncertainty_for_an_isolated_cell() -> None:
    gbp = GaussianBeliefPropagation(_line_graph(1), mean=1.0, variance=0.2)
    before = gbp.belief()
    gbp.age(np.asarray((0,)), 0.3)
    after = gbp.belief()

    assert after.mean[0] == pytest.approx(before.mean[0])
    assert after.variance[0] == pytest.approx(0.5)


def test_partitioned_messages_age_and_reconnection_recovers() -> None:
    graph = _line_graph(5)
    config = GBPConfig(damping=1.0, tolerance=1.0e-9)
    gbp = GaussianBeliefPropagation(
        graph,
        mean=0.0,
        variance=2.0,
        config=config,
    )
    gbp.observe(np.asarray((0,)), np.asarray((4.0,)), np.asarray((0.05,)))
    reference = gbp.run()
    assert reference.converged

    partition = (
        ((gbp.source == 1) & (gbp.target == 2))
        | ((gbp.source == 2) & (gbp.target == 1))
    )
    variance_before = gbp.belief().variance.copy()
    for _ in range(20):
        gbp.age_messages(partition, 0.5)
    stale = gbp.belief()

    assert stale.variance[1] > variance_before[1]
    assert stale.variance[2] > variance_before[2]

    recovered = gbp.run()
    assert recovered.converged
    np.testing.assert_allclose(
        recovered.belief.mean,
        reference.belief.mean,
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        recovered.belief.variance,
        reference.belief.variance,
        rtol=0.0,
        atol=1.0e-7,
    )
