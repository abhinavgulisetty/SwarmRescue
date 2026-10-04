"""Causal bridge from the hazard field to MuJoCo structural failure."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from hazard import HazardDomain, HazardState

from .collapse import release_welds
from .structure import Structure


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class ThermalDamageConfig:
    """Calibratable temperature exposure needed to fail one member."""

    onset_temperature_k: float = 573.15
    failure_exposure_k_s: float = 12_000.0

    def __post_init__(self) -> None:
        _require(
            self.onset_temperature_k > 0.0 and math.isfinite(self.onset_temperature_k),
            "onset_temperature_k must be finite and positive",
        )
        _require(
            self.failure_exposure_k_s > 0.0
            and math.isfinite(self.failure_exposure_k_s),
            "failure_exposure_k_s must be finite and positive",
        )


class ThermalDamage:
    """Accumulate heat exposure and release each failed member's welds once."""

    def __init__(
        self,
        structure: Structure,
        config: ThermalDamageConfig | None = None,
    ) -> None:
        self.structure = structure
        self.config = config or ThermalDamageConfig()
        self.member_names = tuple(member.name for member in structure.members)
        # ponytail: member-centre samples; use volume quadrature after fire calibration.
        self._positions = np.asarray(
            [member.center for member in structure.members], dtype=np.float64
        )
        self._exposure = np.zeros(len(self.member_names), dtype=np.float64)
        self._failed: set[str] = set()
        self._released: set[str] = set()

    @property
    def failed_members(self) -> tuple[str, ...]:
        return tuple(sorted(self._failed))

    @property
    def released_welds(self) -> tuple[str, ...]:
        return tuple(sorted(self._released))

    @property
    def peak_exposure_k_s(self) -> float:
        """Largest accumulated member exposure, for causal reporting."""

        return float(np.max(self._exposure, initial=0.0))

    @property
    def failure_threshold_k_s(self) -> float:
        return self.config.failure_exposure_k_s

    def step(
        self,
        state: HazardState,
        domain: HazardDomain,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dt_seconds: float,
    ) -> tuple[str, ...]:
        """Advance damage and return members that failed during this step."""

        _require(
            dt_seconds >= 0.0 and math.isfinite(dt_seconds),
            "dt_seconds must be finite and nonnegative",
        )
        if dt_seconds == 0.0:
            return ()
        temperature = state.sample_many(
            domain,
            self._positions,
            "temperature",
            outside_value=domain.config.ambient_temperature_k,
        )
        self._exposure += (
            np.maximum(temperature - self.config.onset_temperature_k, 0.0) * dt_seconds
        )
        failed_now = tuple(
            name
            for name, exposure in zip(self.member_names, self._exposure, strict=True)
            if exposure >= self.config.failure_exposure_k_s and name not in self._failed
        )
        if not failed_now:
            return ()

        failed = set(failed_now)
        welds = tuple(
            weld.name
            for weld in self.structure.welds
            if weld.name not in self._released
            and (weld.body_a in failed or weld.body_b in failed)
        )
        release_welds(model, data, welds)
        self._failed.update(failed)
        self._released.update(welds)
        return failed_now
