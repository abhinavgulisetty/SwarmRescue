"""Sparse robot observations and the first Gaussian hazard belief slice."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ember.routing import OracleCostConfig
from hazard import HazardDomain, HazardState


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(values: np.ndarray) -> np.ndarray:
    out = np.array(values, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class HazardSensorConfig:
    """Calibratable thermal and gas sensor errors for one robot."""

    temperature_noise_k: float = 1.0
    temperature_quantization_k: float = 0.25
    temperature_bias_k: float = 0.0
    smoke_noise_kg_m3: float = 0.002
    smoke_quantization_kg_m3: float = 0.001
    smoke_bias_kg_m3: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.temperature_noise_k,
            self.temperature_quantization_k,
            self.temperature_bias_k,
            self.smoke_noise_kg_m3,
            self.smoke_quantization_kg_m3,
            self.smoke_bias_kg_m3,
        )
        _require(
            all(math.isfinite(value) for value in values),
            "sensor errors must be finite",
        )
        _require(
            self.temperature_noise_k >= 0.0,
            "temperature noise cannot be negative",
        )
        _require(
            self.smoke_noise_kg_m3 >= 0.0,
            "smoke noise cannot be negative",
        )
        _require(
            self.temperature_quantization_k > 0.0,
            "temperature quantization must be positive",
        )
        _require(
            self.smoke_quantization_kg_m3 > 0.0,
            "smoke quantization must be positive",
        )


@dataclass(frozen=True, slots=True)
class HazardObservation:
    """Noisy, quantized samples with their diagonal measurement variance."""

    temperature_k: np.ndarray
    smoke_kg_m3: np.ndarray
    temperature_variance: np.ndarray
    smoke_variance: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(
            _readonly(value)
            for value in (
                self.temperature_k,
                self.smoke_kg_m3,
                self.temperature_variance,
                self.smoke_variance,
            )
        )
        shape = arrays[0].shape
        _require(len(shape) == 1, "observations must be vectors")
        _require(
            all(value.shape == shape for value in arrays),
            "observation shapes must match",
        )
        _require(
            all(np.all(np.isfinite(value)) for value in arrays),
            "observations must be finite",
        )
        _require(np.all(arrays[1] >= 0.0), "smoke observations cannot be negative")
        _require(
            np.all(arrays[2] > 0.0) and np.all(arrays[3] > 0.0),
            "observation variance must be positive",
        )
        for name, value in zip(
            (
                "temperature_k",
                "smoke_kg_m3",
                "temperature_variance",
                "smoke_variance",
            ),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, value)


def sample_hazard(
    state: HazardState,
    domain: HazardDomain,
    positions_m: np.ndarray,
    rng: np.random.Generator,
    config: HazardSensorConfig | None = None,
) -> HazardObservation:
    """Sample simulator truth without exposing its fields to robot code."""

    cfg = config or HazardSensorConfig()
    positions = np.asarray(positions_m, dtype=np.float64)
    _require(
        positions.ndim == 2 and positions.shape[1] == 3,
        "positions_m must have shape (N, 3)",
    )
    temperature = state.sample_many(domain, positions, "temperature")
    smoke = state.sample_many(domain, positions, "smoke")
    temperature += cfg.temperature_bias_k + rng.normal(
        0.0, cfg.temperature_noise_k, positions.shape[0]
    )
    smoke += cfg.smoke_bias_kg_m3 + rng.normal(
        0.0, cfg.smoke_noise_kg_m3, positions.shape[0]
    )
    temperature = (
        np.round(temperature / cfg.temperature_quantization_k)
        * cfg.temperature_quantization_k
    )
    smoke = np.maximum(
        np.round(smoke / cfg.smoke_quantization_kg_m3)
        * cfg.smoke_quantization_kg_m3,
        0.0,
    )
    # Quantization error is uniform over one bin, hence variance step^2 / 12.
    temperature_variance = np.full(
        positions.shape[0],
        cfg.temperature_noise_k**2 + cfg.temperature_quantization_k**2 / 12.0,
    )
    smoke_variance = np.full(
        positions.shape[0],
        cfg.smoke_noise_kg_m3**2 + cfg.smoke_quantization_kg_m3**2 / 12.0,
    )
    return HazardObservation(
        temperature,
        smoke,
        temperature_variance,
        smoke_variance,
    )


def observe_state_cost(
    observation: HazardObservation,
    config: OracleCostConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map heat/smoke readings and sensor variance into routing-cost space."""

    cfg = config or OracleCostConfig()
    hot = observation.temperature_k > cfg.heat_onset_k
    mean = (
        cfg.base_step_cost
        + cfg.heat_weight
        * np.maximum(observation.temperature_k - cfg.heat_onset_k, 0.0)
        / cfg.heat_scale_k
        + cfg.smoke_weight * observation.smoke_kg_m3 / cfg.smoke_scale_kg_m3
    )
    variance = (
        hot
        * (cfg.heat_weight / cfg.heat_scale_k) ** 2
        * observation.temperature_variance
        + (cfg.smoke_weight / cfg.smoke_scale_kg_m3) ** 2
        * observation.smoke_variance
    )
    return np.asarray(mean), np.asarray(variance)


@dataclass(frozen=True, slots=True)
class GaussianBelief:
    """Independent Gaussian cells fused by canonical information addition."""

    mean: np.ndarray
    variance: np.ndarray

    def __post_init__(self) -> None:
        mean = _readonly(self.mean)
        variance = _readonly(self.variance)
        _require(
            mean.ndim == 1 and mean.size > 0,
            "belief mean must be a nonempty vector",
        )
        _require(variance.shape == mean.shape, "belief shapes must match")
        _require(np.all(np.isfinite(mean)), "belief mean must be finite")
        _require(
            np.all(np.isfinite(variance)) and np.all(variance > 0.0),
            "belief variance must be finite and positive",
        )
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)

    @classmethod
    def uniform(
        cls, n_cells: int, *, mean: float, variance: float
    ) -> GaussianBelief:
        _require(
            isinstance(n_cells, int)
            and not isinstance(n_cells, bool)
            and n_cells > 0,
            "n_cells must be a positive integer",
        )
        return cls(np.full(n_cells, mean), np.full(n_cells, variance))

    def predict(self, process_variance: float) -> GaussianBelief:
        """Apply the C4 time factor: stale cells become less certain."""

        _require(
            process_variance >= 0.0 and math.isfinite(process_variance),
            "process_variance must be finite and nonnegative",
        )
        return GaussianBelief(self.mean, self.variance + process_variance)

    def update(
        self,
        cells: np.ndarray,
        measurement_mean: np.ndarray,
        measurement_variance: np.ndarray,
    ) -> GaussianBelief:
        """Fuse scalar Gaussian measurements by information addition."""

        indices = np.asarray(cells)
        observed = np.asarray(measurement_mean, dtype=np.float64)
        variance = np.asarray(measurement_variance, dtype=np.float64)
        _require(
            indices.ndim == observed.ndim == variance.ndim == 1,
            "measurement inputs must be vectors",
        )
        _require(
            indices.shape == observed.shape == variance.shape,
            "measurement shapes must match",
        )
        _require(
            np.issubdtype(indices.dtype, np.integer),
            "cell indices must be integers",
        )
        _require(
            np.all((indices >= 0) & (indices < self.mean.size)),
            "cell index outside belief",
        )
        _require(np.all(np.isfinite(observed)), "measurements must be finite")
        _require(
            np.all(np.isfinite(variance)) and np.all(variance > 0.0),
            "measurement variance must be finite and positive",
        )

        precision = 1.0 / self.variance
        information = self.mean * precision
        added_precision = np.zeros_like(precision)
        added_information = np.zeros_like(information)
        np.add.at(added_precision, indices, 1.0 / variance)
        np.add.at(added_information, indices, observed / variance)
        precision += added_precision
        information += added_information
        return GaussianBelief(information / precision, 1.0 / precision)

    def routing_cost(self, beta: float, *, minimum: float = 1.0e-4) -> np.ndarray:
        """Return C1's one-sided confidence cost ``mean + beta * sigma``."""

        _require(math.isfinite(beta), "beta must be finite")
        _require(
            minimum > 0.0 and math.isfinite(minimum),
            "minimum must be finite and positive",
        )
        return np.maximum(self.mean + beta * np.sqrt(self.variance), minimum)
