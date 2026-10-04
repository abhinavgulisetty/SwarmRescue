"""Bounded, incremental vector GBP for coupled robot poses and beacon ranges.

This host reference owns source/beacon likelihood slots and nonlinear Gaussian
messages. It does not implement distributed transport or calibrated sensor bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ember.localization import LocalizationEstimate, RSSIConfig, rssi_to_range


@dataclass(frozen=True, slots=True)
class RangeGBPConfig:
    damping: float = 0.5
    tolerance: float = 1e-7
    max_sweeps: int = 250
    max_linearizations: int = 60
    beacon_prior_variance_m2: float = 1e6

    def __post_init__(self):
        if not math.isfinite(self.damping) or not 0 < self.damping <= 1:
            raise ValueError("damping must be finite in (0, 1]")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        for count in (self.max_sweeps, self.max_linearizations):
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError("iteration caps must be positive integers")
        if (
            not math.isfinite(self.beacon_prior_variance_m2)
            or self.beacon_prior_variance_m2 <= 0
        ):
            raise ValueError("beacon prior variance must be finite and positive")


@dataclass(frozen=True, slots=True)
class RangeGBPResult:
    estimates: tuple[LocalizationEstimate, ...]
    robot_positions_m: np.ndarray
    robot_covariance_m2: np.ndarray
    converged: bool
    sweeps: int
    linearizations: int
    objective: float
    retained_numeric_bytes: int


class RangeFactorGBP:
    """One receiver's fixed source/beacon table with latest-reading watermarks.

    All beacons share the same latent robot pose variables. A range message
    integrates the opposite variable's cavity covariance, rather than treating
    measured anchors as exact. Replayed or reordered sequence IDs never become
    independent observations. A new instance is required for a new pose epoch.
    """

    def __init__(
        self, robot_positions_m, pose_covariance_m2, n_beacons, *, config=None
    ):
        anchors = np.asarray(robot_positions_m, dtype=float)
        covariance = np.asarray(pose_covariance_m2, dtype=float)
        if (
            anchors.ndim != 2
            or anchors.shape[1] not in (2, 3)
            or len(anchors) < anchors.shape[1] + 1
            or not np.all(np.isfinite(anchors))
        ):
            raise ValueError("finite anchors require dimension + 1 robots")
        n, dimension = anchors.shape
        if (
            covariance.shape != (n, dimension, dimension)
            or not np.all(np.isfinite(covariance))
            or not np.allclose(covariance, covariance.swapaxes(-1, -2))
            or np.any(np.linalg.eigvalsh(covariance) <= 0)
        ):
            raise ValueError("pose covariance must be symmetric positive definite")
        if (
            not isinstance(n_beacons, int)
            or isinstance(n_beacons, bool)
            or n_beacons < 1
        ):
            raise ValueError("beacon count must be a positive integer")
        self.config = config or RangeGBPConfig()
        self.anchors = anchors.copy()
        self.pose_precision = np.linalg.inv(covariance)
        self.pose_information = np.einsum("nij,nj->ni", self.pose_precision, anchors)
        self.ranges = np.zeros((n, n_beacons))
        self.variances = np.ones((n, n_beacons))
        self.log_range_variances = np.zeros((n, n_beacons))
        self.sequences = np.full((n, n_beacons), -1, dtype=np.int64)
        self.message_precision = np.zeros((n, n_beacons, 2, dimension, dimension))
        self.message_information = np.zeros((n, n_beacons, 2, dimension))
        self.robot_mean = anchors.copy()
        self.beacon_mean = np.zeros((n_beacons, dimension))
        self._seeded = False
        self.replaced = self.stale = 0

    @property
    def retained_numeric_bytes(self):
        return sum(
            value.nbytes
            for value in vars(self).values()
            if isinstance(value, np.ndarray)
        )

    def observe(self, source, beacon, sequence, distance_m, variance_m2):
        n, b = self.ranges.shape
        if any(
            not isinstance(x, (int, np.integer)) or isinstance(x, bool)
            for x in (source, beacon, sequence)
        ):
            raise ValueError("source, beacon and sequence must be integers")
        if not 0 <= source < n or not 0 <= beacon < b or not 0 <= sequence <= 2**63 - 1:
            raise ValueError("range observation identity is out of bounds")
        if (
            not math.isfinite(distance_m)
            or distance_m <= 0
            or not math.isfinite(variance_m2)
            or variance_m2 <= 0
        ):
            raise ValueError("range and variance must be finite and positive")
        if sequence <= self.sequences[source, beacon]:
            self.stale += 1
            return False
        self.replaced += self.sequences[source, beacon] >= 0
        self.ranges[source, beacon] = distance_m
        self.variances[source, beacon] = variance_m2
        self.log_range_variances[source, beacon] = 0
        self.sequences[source, beacon] = sequence
        # Old linearizations are not independent evidence and must be replaced.
        self.message_precision[source, beacon] = 0
        self.message_information[source, beacon] = 0
        return True

    def observe_rssi(self, source, beacon, sequence, rssi_dbm, *, config=None, walls=0):
        cfg = config or RSSIConfig()
        if not math.isfinite(walls) or walls < 0:
            raise ValueError("wall count must be finite and nonnegative")
        distance = float(
            rssi_to_range(
                np.array([rssi_dbm]), wall_counts=np.array([walls]), config=cfg
            )[0]
        )
        # Preserve the Gaussian log-range likelihood rather than pretending
        # the transformed distance has additive Gaussian noise.
        sigma = max(
            0.05,
            math.log(10) * distance * cfg.noise_std_db / (10 * cfg.path_loss_exponent),
        )
        accepted = self.observe(source, beacon, sequence, distance, sigma**2)
        if accepted:
            self.log_range_variances[source, beacon] = (sigma / distance) ** 2
        return accepted

    def _initialize(self):
        for beacon in range(self.ranges.shape[1]):
            sources = np.flatnonzero(self.sequences[:, beacon] >= 0)
            anchors = self.anchors[sources]
            dimension = anchors.shape[1]
            if len(sources) < dimension + 1:
                raise ValueError("each beacon requires dimension + 1 received ranges")
            delta = 2 * (anchors[1:] - anchors[0])
            if np.linalg.matrix_rank(delta) < dimension:
                raise ValueError("anchor geometry does not constrain every dimension")
            ranges = self.ranges[sources, beacon]
            rhs = (
                ranges[0] ** 2
                - ranges[1:] ** 2
                + np.sum(anchors[1:] ** 2, axis=1)
                - np.sum(anchors[0] ** 2)
            )
            if not self._seeded:
                self.beacon_mean[beacon] = np.linalg.lstsq(delta, rhs, rcond=None)[0]
        self._seeded = True

    def _linear_solve(self, directions, offsets, seed, factor_variance=None):
        variance = self.variances if factor_variance is None else factor_variance
        d = self.anchors.shape[1]
        weak = np.eye(d) / self.config.beacon_prior_variance_m2
        prior_info = seed / self.config.beacon_prior_variance_m2
        active = self.sequences >= 0
        self.message_precision.fill(0)
        self.message_information.fill(0)
        previous_robot, previous_beacon = (
            self.robot_mean.copy(),
            self.beacon_mean.copy(),
        )
        for sweep in range(1, self.config.max_sweeps + 1):
            pose_j = self.pose_precision + self.message_precision[:, :, 0].sum(axis=1)
            pose_h = self.pose_information + self.message_information[:, :, 0].sum(
                axis=1
            )
            beacon_j = weak + self.message_precision[:, :, 1].sum(axis=0)
            beacon_h = prior_info + self.message_information[:, :, 1].sum(axis=0)
            # Remove only this factor's old message before taking its cavity.
            cavity_pose_j = pose_j[:, None] - self.message_precision[:, :, 0]
            cavity_pose_h = pose_h[:, None] - self.message_information[:, :, 0]
            cavity_beacon_j = beacon_j[None] - self.message_precision[:, :, 1]
            cavity_beacon_h = beacon_h[None] - self.message_information[:, :, 1]
            pose_cov = np.linalg.inv(cavity_pose_j)
            beacon_cov = np.linalg.inv(cavity_beacon_j)
            pose_mean = np.einsum("nbij,nbj->nbi", pose_cov, cavity_pose_h)
            beacon_mean = np.einsum("nbij,nbj->nbi", beacon_cov, cavity_beacon_h)
            u = directions
            pose_var = variance + np.einsum("nbi,nbij,nbj->nb", u, beacon_cov, u)
            beacon_var = variance + np.einsum("nbi,nbij,nbj->nb", u, pose_cov, u)
            outer = u[:, :, :, None] * u[:, :, None, :]
            new_j = np.stack(
                (
                    outer / pose_var[:, :, None, None],
                    outer / beacon_var[:, :, None, None],
                ),
                axis=2,
            )
            new_h = np.stack(
                (
                    -u
                    * ((offsets - np.einsum("nbi,nbi->nb", u, beacon_mean)) / pose_var)[
                        :, :, None
                    ],
                    u
                    * ((offsets + np.einsum("nbi,nbi->nb", u, pose_mean)) / beacon_var)[
                        :, :, None
                    ],
                ),
                axis=2,
            )
            new_j *= active[:, :, None, None, None]
            new_h *= active[:, :, None, None]
            message_delta = max(
                float(
                    np.max(np.abs(new_j - self.message_precision) / (1 + np.abs(new_j)))
                ),
                float(
                    np.max(
                        np.abs(new_h - self.message_information) / (1 + np.abs(new_h))
                    )
                ),
            )
            damping = self.config.damping
            self.message_precision += damping * (new_j - self.message_precision)
            self.message_information += damping * (new_h - self.message_information)
            pose_j = self.pose_precision + self.message_precision[:, :, 0].sum(axis=1)
            pose_h = self.pose_information + self.message_information[:, :, 0].sum(
                axis=1
            )
            beacon_j = weak + self.message_precision[:, :, 1].sum(axis=0)
            beacon_h = prior_info + self.message_information[:, :, 1].sum(axis=0)
            self.robot_mean[:] = np.linalg.solve(pose_j, pose_h[..., None])[..., 0]
            self.beacon_mean[:] = np.linalg.solve(beacon_j, beacon_h[..., None])[..., 0]
            delta = max(
                np.max(np.abs(self.robot_mean - previous_robot)),
                np.max(np.abs(self.beacon_mean - previous_beacon)),
            )
            previous_robot[:], previous_beacon[:] = self.robot_mean, self.beacon_mean
            if max(delta, message_delta) < self.config.tolerance:
                return sweep, True, np.linalg.inv(pose_j), np.linalg.inv(beacon_j)
        return sweep, False, np.linalg.inv(pose_j), np.linalg.inv(beacon_j)

    def _objective(self, poses, beacons, seed):
        predicted = np.linalg.norm(beacons[None] - poses[:, None], axis=2)
        residual = predicted - self.ranges
        rssifactor = self.log_range_variances > 0
        safe_range = np.maximum(self.ranges, 1e-6)
        log_residual = np.log(np.maximum(predicted, 1e-6) / safe_range)
        likelihood_cost = np.where(
            rssifactor,
            log_residual**2 / np.maximum(self.log_range_variances, 1e-30),
            residual**2 / self.variances,
        )
        active = self.sequences >= 0
        anchor_error = poses - self.anchors
        return float(
            np.sum((beacons - seed) ** 2) / self.config.beacon_prior_variance_m2
            + np.sum(np.where(active, likelihood_cost, 0))
            + np.einsum("ni,nij,nj->", anchor_error, self.pose_precision, anchor_error)
        )

    def solve(self):
        self._initialize()
        seed = self.beacon_mean.copy()
        total_sweeps, converged = 0, False
        for _linearization in range(1, self.config.max_linearizations + 1):
            poses, beacons = self.robot_mean.copy(), self.beacon_mean.copy()
            difference = beacons[None] - poses[:, None]
            length = np.maximum(np.linalg.norm(difference, axis=2), 1e-6)
            directions = difference / length[:, :, None]
            logfactor = self.log_range_variances > 0
            target = np.where(
                logfactor,
                length * (1 + np.log(np.maximum(self.ranges, 1e-6) / length)),
                self.ranges,
            )
            offsets = target - length + np.einsum("nbi,nbi->nb", directions, difference)
            variance = np.where(
                logfactor, self.log_range_variances * length**2, self.variances
            )
            sweeps, linear_converged, pose_cov, beacon_cov = self._linear_solve(
                directions, offsets, seed, variance
            )
            total_sweeps += sweeps
            step_pose, step_beacon = self.robot_mean - poses, self.beacon_mean - beacons
            old_objective = self._objective(poses, beacons, seed)
            rate = 1.0
            while rate >= 1 / 1024:
                candidate_pose, candidate_beacon = (
                    poses + rate * step_pose,
                    beacons + rate * step_beacon,
                )
                if (
                    self._objective(candidate_pose, candidate_beacon, seed)
                    <= old_objective + 1e-10
                ):
                    break
                rate *= 0.5
            if rate < 1 / 1024:
                self.robot_mean[:], self.beacon_mean[:] = poses, beacons
                break
            self.robot_mean[:], self.beacon_mean[:] = candidate_pose, candidate_beacon
            if (
                linear_converged
                and max(np.max(np.abs(step_pose)), np.max(np.abs(step_beacon))) < 1e-5
            ):
                converged = True
                break
        estimates = []
        for beacon, mean in enumerate(self.beacon_mean):
            active = self.sequences[:, beacon] >= 0
            residual = (
                np.linalg.norm(mean - self.robot_mean[active], axis=1)
                - self.ranges[active, beacon]
            )
            covariance = 0.5 * (beacon_cov[beacon] + beacon_cov[beacon].T)
            estimates.append(
                LocalizationEstimate(
                    mean,
                    covariance,
                    float(np.sqrt(np.mean(residual**2))),
                    int(active.sum()),
                )
            )
        return RangeGBPResult(
            tuple(estimates),
            self.robot_mean.copy(),
            pose_cov.copy(),
            converged,
            total_sweeps,
            _linearization,
            self._objective(self.robot_mean, self.beacon_mean, seed),
            self.retained_numeric_bytes,
        )
