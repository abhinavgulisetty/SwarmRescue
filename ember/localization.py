"""BLE RSSI sensing and cooperative survivor localization.

The simulator owns the survivor's true position.  Robots receive only noisy
RSSI measurements and combine range constraints with a small Gauss--Newton
solve.  This is the classical Phase-1 reference that the later distributed
GBP range factors must reproduce.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_CHI_SQUARED_95 = {
    2: 5.991464547107979,
    3: 7.814727903251179,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class RSSIConfig:
    reference_rssi_dbm: float = -45.0
    reference_distance_m: float = 1.0
    path_loss_exponent: float = 2.2
    wall_loss_db: float = 5.0
    noise_std_db: float = 1.5
    minimum_distance_m: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.reference_rssi_dbm,
            self.reference_distance_m,
            self.path_loss_exponent,
            self.wall_loss_db,
            self.noise_std_db,
            self.minimum_distance_m,
        )
        _require(
            all(math.isfinite(value) for value in values),
            "RSSI parameters must be finite",
        )
        _require(self.reference_distance_m > 0.0, "reference distance must be positive")
        _require(self.path_loss_exponent > 0.0, "path-loss exponent must be positive")
        _require(self.wall_loss_db >= 0.0, "wall loss cannot be negative")
        _require(self.noise_std_db >= 0.0, "RSSI noise cannot be negative")
        _require(self.minimum_distance_m > 0.0, "minimum distance must be positive")


@dataclass(frozen=True, slots=True)
class LocalizationEstimate:
    position_m: np.ndarray
    covariance_m2: np.ndarray
    residual_rms_m: float
    measurements: int

    def __post_init__(self) -> None:
        position = np.array(self.position_m, dtype=np.float64, copy=True)
        covariance = np.array(self.covariance_m2, dtype=np.float64, copy=True)
        _require(
            position.ndim == 1 and position.size in _CHI_SQUARED_95,
            "position must be a 2-D or 3-D point",
        )
        _require(
            covariance.shape == (position.size, position.size),
            "covariance dimensions must match position",
        )
        _require(np.all(np.isfinite(position)), "position must be finite")
        _require(np.all(np.isfinite(covariance)), "covariance must be finite")
        _require(
            np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-12),
            "covariance must be symmetric positive definite",
        )
        _require(
            np.all(np.linalg.eigvalsh(covariance) > 0.0),
            "covariance must be symmetric positive definite",
        )
        _require(
            self.residual_rms_m >= 0.0 and math.isfinite(self.residual_rms_m),
            "residual must be finite and nonnegative",
        )
        _require(
            self.measurements >= position.size + 1,
            "localization needs at least dimension + 1 measurements",
        )
        position.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "covariance_m2", covariance)

    @property
    def dimension(self) -> int:
        return int(self.position_m.size)

    @property
    def chi_squared_95(self) -> float:
        return _CHI_SQUARED_95[self.dimension]

    @property
    def uncertainty_radius_m(self) -> float:
        """Major-axis radius of the 95% confidence ellipse or ellipsoid."""

        return math.sqrt(
            self.chi_squared_95
            * max(0.0, float(np.linalg.eigvalsh(self.covariance_m2)[-1]))
        )

    def mahalanobis_squared(self, point_m: np.ndarray) -> float:
        """Squared error in the estimate's covariance geometry."""

        point = np.asarray(point_m, dtype=np.float64)
        _require(
            point.shape == self.position_m.shape,
            "point dimension must match estimate",
        )
        _require(np.all(np.isfinite(point)), "point must be finite")
        error = self.position_m - point
        normalized_error = np.linalg.solve(self.covariance_m2, error)
        return float(error @ normalized_error)

    def covers_95(self, point_m: np.ndarray) -> bool:
        """Whether a point lies in the estimate's 95% confidence region."""

        return self.mahalanobis_squared(point_m) <= self.chi_squared_95


def sample_rssi(
    robot_positions_m: np.ndarray,
    survivor_position_m: np.ndarray,
    rng: np.random.Generator,
    *,
    wall_counts: np.ndarray | None = None,
    config: RSSIConfig | None = None,
) -> np.ndarray:
    """Return noisy RSSI values without exposing survivor ground truth downstream."""

    cfg = config or RSSIConfig()
    robots = np.asarray(robot_positions_m, dtype=np.float64)
    survivor = np.asarray(survivor_position_m, dtype=np.float64)
    _require(
        robots.ndim == 2 and robots.shape[1] in _CHI_SQUARED_95,
        "robot positions must have shape (N, 2) or (N, 3)",
    )
    dimension = robots.shape[1]
    _require(
        robots.shape[0] > 0 and survivor.shape == (dimension,),
        "survivor dimension must match robot positions",
    )
    _require(
        np.all(np.isfinite(robots)) and np.all(np.isfinite(survivor)),
        "positions must be finite",
    )
    walls = (
        np.zeros(robots.shape[0])
        if wall_counts is None
        else np.asarray(wall_counts, dtype=np.float64)
    )
    _require(
        walls.shape == (robots.shape[0],), "wall_counts must have one value per robot"
    )
    _require(
        np.all(np.isfinite(walls)) and np.all(walls >= 0.0),
        "wall counts must be finite and nonnegative",
    )
    distance = np.maximum(
        np.linalg.norm(robots - survivor, axis=1), cfg.minimum_distance_m
    )
    mean = (
        cfg.reference_rssi_dbm
        - 10.0 * cfg.path_loss_exponent * np.log10(distance / cfg.reference_distance_m)
        - cfg.wall_loss_db * walls
    )
    return mean + rng.normal(0.0, cfg.noise_std_db, robots.shape[0])


def rssi_to_range(
    rssi_dbm: np.ndarray,
    *,
    wall_counts: np.ndarray | None = None,
    config: RSSIConfig | None = None,
) -> np.ndarray:
    cfg = config or RSSIConfig()
    rssi = np.asarray(rssi_dbm, dtype=np.float64)
    _require(rssi.ndim == 1 and rssi.size > 0, "RSSI values must be a nonempty vector")
    _require(np.all(np.isfinite(rssi)), "RSSI values must be finite")
    walls = (
        np.zeros(rssi.size)
        if wall_counts is None
        else np.asarray(wall_counts, dtype=np.float64)
    )
    _require(walls.shape == rssi.shape, "wall_counts must match RSSI values")
    exponent = (cfg.reference_rssi_dbm - cfg.wall_loss_db * walls - rssi) / (
        10.0 * cfg.path_loss_exponent
    )
    return np.maximum(
        cfg.reference_distance_m * np.power(10.0, exponent), cfg.minimum_distance_m
    )


def localize_rssi(
    robot_positions_m: np.ndarray,
    rssi_dbm: np.ndarray,
    *,
    wall_counts: np.ndarray | None = None,
    config: RSSIConfig | None = None,
) -> LocalizationEstimate:
    """Estimate a 2-D or 3-D survivor position from noisy RSSI ranges."""

    cfg = config or RSSIConfig()
    robots = np.asarray(robot_positions_m, dtype=np.float64)
    _require(
        robots.ndim == 2 and robots.shape[1] in _CHI_SQUARED_95,
        "robot positions must have shape (N, 2) or (N, 3)",
    )
    dimension = robots.shape[1]
    _require(
        robots.shape[0] >= dimension + 1,
        "localization needs at least dimension + 1 robots",
    )
    ranges = rssi_to_range(rssi_dbm, wall_counts=wall_counts, config=cfg)
    _require(
        ranges.shape == (robots.shape[0],), "RSSI values must match robot positions"
    )

    # Circle subtraction supplies a stable linear seed; Gauss--Newton then
    # minimizes the actual range residual rather than the squared-range proxy.
    delta = robots[1:] - robots[0]
    rhs = (
        ranges[0] ** 2
        - ranges[1:] ** 2
        + np.sum(robots[1:] ** 2, axis=1)
        - np.sum(robots[0] ** 2)
    )
    position, *_ = np.linalg.lstsq(2.0 * delta, rhs, rcond=None)
    for _ in range(12):
        offsets = position - robots
        predicted = np.maximum(np.linalg.norm(offsets, axis=1), cfg.minimum_distance_m)
        jacobian = offsets / predicted[:, None]
        step, *_ = np.linalg.lstsq(jacobian, predicted - ranges, rcond=None)
        position -= step
        if np.linalg.norm(step) < 1.0e-6:
            break

    offsets = position - robots
    predicted = np.maximum(np.linalg.norm(offsets, axis=1), cfg.minimum_distance_m)
    residual = predicted - ranges
    jacobian = offsets / predicted[:, None]
    range_sigma = max(
        0.05,
        math.log(10.0)
        * float(np.mean(ranges))
        * max(cfg.noise_std_db, 0.1)
        / (10.0 * cfg.path_loss_exponent),
    )
    information = jacobian.T @ jacobian / range_sigma**2
    _require(
        np.linalg.matrix_rank(information) == dimension,
        "robot geometry does not constrain every localization dimension",
    )
    covariance = np.linalg.inv(information)
    covariance = 0.5 * (covariance + covariance.T)
    return LocalizationEstimate(
        position,
        covariance,
        float(np.sqrt(np.mean(residual**2))),
        robots.shape[0],
    )


__all__ = [
    "LocalizationEstimate",
    "RSSIConfig",
    "localize_rssi",
    "rssi_to_range",
    "sample_rssi",
]
