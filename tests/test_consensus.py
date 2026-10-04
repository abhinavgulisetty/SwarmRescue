"""Contracts for the distributed B3 Gaussian consensus baseline."""

from __future__ import annotations

import numpy as np
import pytest

from ember.belief import GaussianBelief
from ember.consensus import gaussian_average_consensus


def test_connected_consensus_matches_centralized_information_sum() -> None:
    beliefs = (
        GaussianBelief(np.array((1.0, 0.0)), np.array((1.0, 4.0))),
        GaussianBelief(np.array((3.0, 2.0)), np.array((1.0, 1.0))),
        GaussianBelief(np.array((5.0, 4.0)), np.array((2.0, 2.0))),
    )
    adjacency = np.array(
        (
            (0, 1, 0),
            (1, 0, 1),
            (0, 1, 0),
        ),
        dtype=bool,
    )

    result = gaussian_average_consensus(beliefs, adjacency)

    precision = sum(1.0 / belief.variance for belief in beliefs)
    information = sum(
        belief.mean / belief.variance for belief in beliefs
    )
    assert result.converged
    assert result.component_count == 1
    for belief in result.beliefs:
        np.testing.assert_allclose(
            belief.mean,
            information / precision,
            rtol=0.0,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            belief.variance,
            1.0 / precision,
            rtol=0.0,
            atol=1.0e-7,
        )


def test_partition_converges_to_separate_component_beliefs() -> None:
    beliefs = tuple(
        GaussianBelief(
            np.array((float(index),)),
            np.array((1.0,)),
        )
        for index in range(4)
    )
    adjacency = np.array(
        (
            (0, 1, 0, 0),
            (1, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 0, 1, 0),
        ),
        dtype=bool,
    )

    result = gaussian_average_consensus(beliefs, adjacency)

    assert result.converged
    assert result.component_count == 2
    assert result.beliefs[0].mean[0] == pytest.approx(0.5)
    assert result.beliefs[1].mean[0] == pytest.approx(0.5)
    assert result.beliefs[2].mean[0] == pytest.approx(2.5)
    assert result.beliefs[3].mean[0] == pytest.approx(2.5)
    assert result.beliefs[0].variance[0] == pytest.approx(0.5)
