"""Deterministic combustion and smoke transport on a three-dimensional grid.

This is the first Phase 1 oracle: it owns the unobservable ground-truth
temperature, fuel, and smoke fields.  The explicit finite-difference update is
sub-stepped to satisfy its advection/diffusion stability bound.  A NumPy
backend is kept as a transparent reference and the Taichi backend is loaded
and initialised only when requested, so importing :mod:`hazard` has no JIT or
global-runtime side effects.

Array axes are ``(z, y, x)`` throughout.  Velocity components use world order
``(x, y, z)`` in the last axis.
"""

from __future__ import annotations

import math
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from cascade.params import FUEL_LOAD_KG_M2

if TYPE_CHECKING:
    from cascade.floorplan import FloorPlan

Backend = Literal["numpy", "taichi"]
FieldName = Literal["temperature", "fuel", "smoke"]

_GRAVITY_M_S2 = 9.81
_TAICHI_LOCK = threading.RLock()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(values: np.ndarray, dtype: np.dtype[np.generic]) -> np.ndarray:
    out = np.array(values, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class HazardConfig:
    """Numerical and physical parameters for one hazard world.

    The combustion constants are an intentionally compact Arrhenius model,
    not a calibrated compartment-fire model.  Keeping them named makes that
    limitation measurable and replaceable rather than hiding it in a kernel.
    """

    cell_size_m: float = 0.5
    tick_seconds: float = 0.1
    ambient_temperature_k: float = 293.15
    thermal_diffusivity_m2_s: float = 0.025
    smoke_diffusivity_m2_s: float = 0.060
    heat_loss_s_inv: float = 0.015
    combustion_pre_exponential_s_inv: float = 50.0
    activation_temperature_k: float = 7_200.0
    heat_release_k_per_kg_m3: float = 120.0
    smoke_yield_per_kg_m3: float = 0.12
    stack_discharge_coefficient: float = 0.35
    max_stack_speed_m_s: float = 2.5
    stability_limit: float = 0.42
    max_substeps: int = 256

    def __post_init__(self) -> None:
        _require(
            self.cell_size_m > 0.0 and math.isfinite(self.cell_size_m),
            "cell_size_m must be finite and positive",
        )
        _require(
            self.tick_seconds > 0.0 and math.isfinite(self.tick_seconds),
            "tick_seconds must be finite and positive",
        )
        _require(
            self.ambient_temperature_k > 0.0
            and math.isfinite(self.ambient_temperature_k),
            "ambient_temperature_k must be finite and positive",
        )
        _require(
            self.thermal_diffusivity_m2_s >= 0.0
            and math.isfinite(self.thermal_diffusivity_m2_s),
            "thermal_diffusivity_m2_s must be finite and nonnegative",
        )
        _require(
            self.smoke_diffusivity_m2_s >= 0.0
            and math.isfinite(self.smoke_diffusivity_m2_s),
            "smoke_diffusivity_m2_s must be finite and nonnegative",
        )
        _require(
            self.heat_loss_s_inv >= 0.0 and math.isfinite(self.heat_loss_s_inv),
            "heat_loss_s_inv must be finite and nonnegative",
        )
        _require(
            self.combustion_pre_exponential_s_inv >= 0.0,
            "combustion_pre_exponential_s_inv must be nonnegative",
        )
        _require(
            math.isfinite(self.combustion_pre_exponential_s_inv),
            "combustion_pre_exponential_s_inv must be finite",
        )
        _require(
            self.activation_temperature_k >= 0.0
            and math.isfinite(self.activation_temperature_k),
            "activation_temperature_k must be finite and nonnegative",
        )
        _require(
            self.heat_release_k_per_kg_m3 >= 0.0
            and math.isfinite(self.heat_release_k_per_kg_m3),
            "heat_release_k_per_kg_m3 must be finite and nonnegative",
        )
        _require(
            self.smoke_yield_per_kg_m3 >= 0.0
            and math.isfinite(self.smoke_yield_per_kg_m3),
            "smoke_yield_per_kg_m3 must be finite and nonnegative",
        )
        _require(
            0.0 <= self.stack_discharge_coefficient <= 1.0
            and math.isfinite(self.stack_discharge_coefficient),
            "stack_discharge_coefficient must be finite and in [0, 1]",
        )
        _require(
            self.max_stack_speed_m_s >= 0.0
            and math.isfinite(self.max_stack_speed_m_s),
            "max_stack_speed_m_s must be finite and nonnegative",
        )
        _require(
            0.0 < self.stability_limit < 1.0
            and math.isfinite(self.stability_limit),
            "stability_limit must be finite and in (0, 1)",
        )
        _require(
            isinstance(self.max_substeps, int)
            and not isinstance(self.max_substeps, bool)
            and self.max_substeps >= 1,
            "max_substeps must be a positive integer",
        )


@dataclass(frozen=True, slots=True)
class HazardDomain:
    """Static lattice geometry and room-derived initial fuel density."""

    origin_m: tuple[float, float, float]
    cell_size_m: float
    active: np.ndarray
    connector: np.ndarray
    vertical_open: np.ndarray
    initial_fuel_kg_m3: np.ndarray
    bounds_max_m: tuple[float, float, float] | None = None
    config: HazardConfig | None = None

    def __post_init__(self) -> None:
        active = _readonly(self.active, np.dtype(np.bool_))
        connector = _readonly(self.connector, np.dtype(np.bool_))
        vertical_open = _readonly(self.vertical_open, np.dtype(np.bool_))
        fuel = _readonly(self.initial_fuel_kg_m3, np.dtype(np.float64))
        _require(active.ndim == 3, "hazard domain arrays must be three-dimensional")
        _require(
            connector.shape == active.shape
            and vertical_open.shape == active.shape
            and fuel.shape == active.shape,
            "hazard domain arrays must share one shape",
        )
        origin = np.asarray(self.origin_m, dtype=np.float64)
        grid_extent = self.cell_size_m * np.asarray(
            (active.shape[2], active.shape[1], active.shape[0]), dtype=np.float64
        )
        bounds = self.bounds_max_m or tuple(origin + grid_extent)
        bounds_max = np.asarray(bounds, dtype=np.float64)
        _require(
            origin.shape == (3,)
            and bounds_max.shape == (3,)
            and np.all(np.isfinite(origin))
            and np.all(np.isfinite(bounds_max)),
            "domain bounds must be finite three-dimensional points",
        )
        _require(np.all(bounds_max > origin), "domain bounds must have positive extent")
        _require(
            self.cell_size_m > 0.0 and math.isfinite(self.cell_size_m),
            "cell_size_m must be finite and positive",
        )
        domain_config = self.config or HazardConfig(cell_size_m=self.cell_size_m)
        _require(
            math.isclose(domain_config.cell_size_m, self.cell_size_m),
            "domain config and cell size must match",
        )
        _require(
            np.allclose(bounds_max - origin, grid_extent, rtol=0.0, atol=1.0e-9),
            "physical bounds must exactly tile the configured grid",
        )
        _require(np.all(np.isfinite(fuel)), "initial fuel must be finite")
        _require(np.all(fuel >= 0.0), "initial fuel cannot be negative")
        _require(
            not np.any(connector & ~active),
            "connector cells must be inside the active domain",
        )
        next_active = np.zeros_like(active)
        next_active[:-1] = active[1:]
        _require(
            not np.any(vertical_open & ~(active & next_active)),
            "open vertical faces must join two active cells",
        )
        _require(
            not np.any(vertical_open[-1]),
            "the top boundary cannot contain an open vertical face",
        )
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "connector", connector)
        object.__setattr__(self, "vertical_open", vertical_open)
        object.__setattr__(self, "initial_fuel_kg_m3", fuel)
        object.__setattr__(self, "bounds_max_m", tuple(float(v) for v in bounds_max))
        object.__setattr__(self, "config", domain_config)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.active.shape  # type: ignore[return-value]

    @classmethod
    def from_floorplan(
        cls, plan: FloorPlan, config: HazardConfig | None = None
    ) -> HazardDomain:
        """Rasterize room semantics and aligned shafts onto a regular grid.

        Fuel loads are specified per square metre of floor in ``cascade``.
        Dividing by storey height makes the lattice field a volumetric density
        while preserving each room's total equivalent fuel load.
        """

        cfg = config or HazardConfig()
        cell = cfg.cell_size_m
        footprint = plan.footprint
        height = plan.params.total_height_m
        storey_height = plan.params.storey_height_m
        storey_cells = storey_height / cell
        _require(
            math.isclose(storey_cells, round(storey_cells), abs_tol=1.0e-9),
            "cell_size_m must evenly divide one storey height",
        )
        extents = (footprint.width, footprint.height, height)
        counts: list[int] = []
        for extent in extents:
            cells = extent / cell
            rounded = round(cells)
            _require(
                rounded >= 1 and math.isclose(cells, rounded, abs_tol=1.0e-9),
                "cell_size_m must evenly divide both footprint axes and height",
            )
            counts.append(rounded)
        nx, ny, nz = counts

        x = footprint.x0 + (np.arange(nx, dtype=np.float64) + 0.5) * cell
        y = footprint.y0 + (np.arange(ny, dtype=np.float64) + 0.5) * cell
        z = (np.arange(nz, dtype=np.float64) + 0.5) * cell
        active = (
            (z[:, None, None] < height)
            & (y[None, :, None] < footprint.y1)
            & (x[None, None, :] < footprint.x1)
        )
        fuel = np.zeros((nz, ny, nx), dtype=np.float64)
        connector = np.zeros_like(active)

        for storey in plan.storeys:
            z0 = storey.index * storey_height
            z1 = z0 + storey_height
            z_mask = (z >= z0) & (z < z1)
            for room in storey.rooms:
                rect = room.rect
                x_mask = (x >= rect.x0) & (x < rect.x1)
                y_mask = (y >= rect.y0) & (y < rect.y1)
                cells = z_mask[:, None, None] & y_mask[None, :, None] & x_mask[
                    None, None, :
                ]
                fuel[cells] = FUEL_LOAD_KG_M2[room.kind] / storey_height
                if room.is_connector:
                    connector[cells] = True

        for planned in plan.connectors:
            x_mask = (x >= planned.rect.x0) & (x < planned.rect.x1)
            y_mask = (y >= planned.rect.y0) & (y < planned.rect.y1)
            _require(
                np.any(x_mask) and np.any(y_mask),
                "cell_size_m is too coarse to resolve every vertical connector",
            )

        connector &= active
        fuel[~active] = 0.0
        # ``vertical_open[k]`` is the face from layer k to k+1.  Air mixes
        # within a storey, while a slab boundary is impermeable except where a
        # grammar connector punches a continuous opening through it.
        storey_index = np.floor(z / storey_height).astype(np.int64)
        vertical_open = np.zeros_like(active)
        same_storey = storey_index[:-1] == storey_index[1:]
        vertical_open[:-1] = active[:-1] & active[1:] & (
            same_storey[:, None, None] | (connector[:-1] & connector[1:])
        )
        return cls(
            origin_m=(footprint.x0, footprint.y0, 0.0),
            bounds_max_m=(footprint.x1, footprint.y1, height),
            cell_size_m=cell,
            active=active,
            connector=connector,
            vertical_open=vertical_open,
            initial_fuel_kg_m3=fuel,
            config=cfg,
        )

    def initial_state(self, config: HazardConfig | None = None) -> HazardState:
        cfg = config or self.config or HazardConfig(cell_size_m=self.cell_size_m)
        _require(
            math.isclose(cfg.cell_size_m, self.cell_size_m),
            "config and domain cell sizes must match",
        )
        temperature = np.full(self.shape, cfg.ambient_temperature_k, dtype=np.float64)
        return HazardState(
            temperature_k=temperature,
            fuel_kg_m3=self.initial_fuel_kg_m3,
            smoke_kg_m3=np.zeros(self.shape, dtype=np.float64),
            time_s=0.0,
        )


@dataclass(frozen=True, slots=True)
class HazardState:
    """Immutable snapshot of the simulator-only hazard fields."""

    temperature_k: np.ndarray
    fuel_kg_m3: np.ndarray
    smoke_kg_m3: np.ndarray
    time_s: float = 0.0

    def __post_init__(self) -> None:
        temperature = _readonly(self.temperature_k, np.dtype(np.float64))
        fuel = _readonly(self.fuel_kg_m3, np.dtype(np.float64))
        smoke = _readonly(self.smoke_kg_m3, np.dtype(np.float64))
        _require(temperature.ndim == 3, "hazard fields must be three-dimensional")
        _require(
            fuel.shape == temperature.shape and smoke.shape == temperature.shape,
            "hazard fields must share one shape",
        )
        _require(self.time_s >= 0.0 and math.isfinite(self.time_s), "invalid time_s")
        _require(np.all(np.isfinite(temperature)), "temperature must be finite")
        _require(np.all(np.isfinite(fuel)), "fuel must be finite")
        _require(np.all(np.isfinite(smoke)), "smoke must be finite")
        _require(np.all(temperature > 0.0), "temperature must be above absolute zero")
        _require(np.all(fuel >= 0.0), "fuel cannot be negative")
        _require(np.all(smoke >= 0.0), "smoke cannot be negative")
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "fuel_kg_m3", fuel)
        object.__setattr__(self, "smoke_kg_m3", smoke)

    def ignite(
        self,
        domain: HazardDomain,
        world_xyz_m: tuple[float, float, float],
        *,
        radius_m: float,
        temperature_k: float,
        smoke_kg_m3: float = 0.0,
    ) -> HazardState:
        """Return a snapshot with a deterministic spherical ignition seed."""

        if self.temperature_k.shape != domain.shape:
            raise ValueError("state and domain shapes do not match")
        _require(radius_m > 0.0, "ignition radius must be positive")
        _require(
            temperature_k > 0.0 and math.isfinite(temperature_k),
            "ignition temperature must be finite and positive",
        )
        _require(
            smoke_kg_m3 >= 0.0 and math.isfinite(smoke_kg_m3),
            "ignition smoke must be finite and nonnegative",
        )
        center = np.asarray(world_xyz_m, dtype=np.float64)
        _require(
            center.shape == (3,) and np.all(np.isfinite(center)),
            "ignition centre must be a finite world-space point",
        )

        nz, ny, nx = domain.shape
        ox, oy, oz = domain.origin_m
        cell = domain.cell_size_m
        x = ox + (np.arange(nx) + 0.5) * cell
        y = oy + (np.arange(ny) + 0.5) * cell
        z = oz + (np.arange(nz) + 0.5) * cell
        distance_sq = (
            (x[None, None, :] - center[0]) ** 2
            + (y[None, :, None] - center[1]) ** 2
            + (z[:, None, None] - center[2]) ** 2
        )
        seeded = domain.active & (distance_sq <= radius_m * radius_m)
        _require(np.any(seeded), "ignition sphere does not cover an active cell")

        temperature = np.array(self.temperature_k, copy=True)
        smoke = np.array(self.smoke_kg_m3, copy=True)
        temperature[seeded] = np.maximum(temperature[seeded], temperature_k)
        smoke[seeded] = np.maximum(smoke[seeded], smoke_kg_m3)
        return HazardState(temperature, self.fuel_kg_m3, smoke, self.time_s)

    def sample(
        self,
        domain: HazardDomain,
        world_xyz_m: tuple[float, float, float],
        field: FieldName,
    ) -> float:
        """Trilinearly sample a field at a world-space point."""

        arrays = {
            "temperature": self.temperature_k,
            "fuel": self.fuel_kg_m3,
            "smoke": self.smoke_kg_m3,
        }
        if field not in arrays:
            raise ValueError(f"unknown hazard field: {field!r}")
        if arrays[field].shape != domain.shape:
            raise ValueError("state and domain shapes do not match")

        nx = domain.shape[2]
        ny = domain.shape[1]
        nz = domain.shape[0]
        origin = domain.origin_m
        cell = domain.cell_size_m
        point = np.asarray(world_xyz_m, dtype=np.float64)
        lower = np.asarray(origin, dtype=np.float64)
        upper = np.asarray(domain.bounds_max_m, dtype=np.float64)
        if not np.all(np.isfinite(point)) or np.any(point < lower) or np.any(
            point > upper
        ):
            raise ValueError(f"point {world_xyz_m!r} lies outside the hazard grid")

        fractional = (point - lower) / cell - 0.5
        fractional = np.clip(fractional, 0.0, np.asarray((nx - 1, ny - 1, nz - 1)))
        lo = np.floor(fractional).astype(np.int64)
        hi = np.minimum(lo + 1, np.asarray((nx - 1, ny - 1, nz - 1)))
        weight = fractional - lo
        values = arrays[field]

        result = 0.0
        for dz in (0, 1):
            iz = int(hi[2] if dz else lo[2])
            wz = weight[2] if dz else 1.0 - weight[2]
            for dy in (0, 1):
                iy = int(hi[1] if dy else lo[1])
                wy = weight[1] if dy else 1.0 - weight[1]
                for dx in (0, 1):
                    ix = int(hi[0] if dx else lo[0])
                    wx = weight[0] if dx else 1.0 - weight[0]
                    result += float(values[iz, iy, ix]) * wx * wy * wz
        return result

    def sample_many(
        self,
        domain: HazardDomain,
        world_xyz_m: np.ndarray,
        field: FieldName,
        *,
        outside_value: float | None = None,
    ) -> np.ndarray:
        """Vectorized trilinear samples for sensors and graph-cost projection.

        ``outside_value`` is explicit because different fields have different
        exterior physics: temperature normally uses ambient while fuel and
        smoke use zero.  Omitting it preserves the scalar API's strict bounds.
        """

        arrays = {
            "temperature": self.temperature_k,
            "fuel": self.fuel_kg_m3,
            "smoke": self.smoke_kg_m3,
        }
        if field not in arrays:
            raise ValueError(f"unknown hazard field: {field!r}")
        if arrays[field].shape != domain.shape:
            raise ValueError("state and domain shapes do not match")
        points = np.asarray(world_xyz_m, dtype=np.float64)
        _require(
            points.ndim == 2 and points.shape[1] == 3,
            "world_xyz_m must have shape (N, 3)",
        )
        _require(np.all(np.isfinite(points)), "sample points must be finite")
        if outside_value is not None:
            _require(math.isfinite(outside_value), "outside_value must be finite")

        nz, ny, nx = domain.shape
        cell = domain.cell_size_m
        lower = np.asarray(domain.origin_m, dtype=np.float64)
        upper = np.asarray(domain.bounds_max_m, dtype=np.float64)
        inside = np.all((points >= lower) & (points <= upper), axis=1)

        fractional = (points - lower) / cell - 0.5
        fractional = np.clip(
            fractional,
            0.0,
            np.asarray((nx - 1, ny - 1, nz - 1), dtype=np.float64),
        )
        lo = np.floor(fractional).astype(np.int64)
        hi = np.minimum(lo + 1, np.asarray((nx - 1, ny - 1, nz - 1)))
        weight = fractional - lo

        # A ceil-sized boundary cell can sit beyond a non-divisible footprint.
        # Treat a point whose containing cell is inactive as exterior too.
        containing = np.floor((points - lower) / cell).astype(np.int64)
        containing = np.clip(containing, 0, np.asarray((nx - 1, ny - 1, nz - 1)))
        inside &= domain.active[
            containing[:, 2], containing[:, 1], containing[:, 0]
        ]
        if outside_value is None and not np.all(inside):
            raise ValueError("one or more points lie outside the hazard grid")

        values = arrays[field]
        result = np.zeros(points.shape[0], dtype=np.float64)
        for dz in (0, 1):
            iz = np.where(dz, hi[:, 2], lo[:, 2])
            wz = np.where(dz, weight[:, 2], 1.0 - weight[:, 2])
            for dy in (0, 1):
                iy = np.where(dy, hi[:, 1], lo[:, 1])
                wy = np.where(dy, weight[:, 1], 1.0 - weight[:, 1])
                for dx in (0, 1):
                    ix = np.where(dx, hi[:, 0], lo[:, 0])
                    wx = np.where(dx, weight[:, 0], 1.0 - weight[:, 0])
                    result += values[iz, iy, ix] * wx * wy * wz
        if outside_value is not None:
            result[~inside] = outside_value
        return result


class HazardSolver:
    """Advance one domain with deterministic, stability-limited updates."""

    def __init__(
        self,
        domain: HazardDomain,
        config: HazardConfig | None = None,
        *,
        backend: Backend = "numpy",
    ) -> None:
        self.domain = domain
        self.config = config or domain.config or HazardConfig(
            cell_size_m=domain.cell_size_m
        )
        _require(
            math.isclose(self.config.cell_size_m, domain.cell_size_m),
            "config and domain cell sizes must match",
        )
        if backend not in ("numpy", "taichi"):
            raise ValueError(f"unknown hazard backend: {backend!r}")
        self.backend: Backend = backend

    def stack_velocity(self, temperature_k: np.ndarray) -> np.ndarray:
        """Return the buoyancy-driven velocity field for this temperature.

        Each vertical connector column uses its mean excess temperature to
        drive one upward 1-D pressure head.  Non-connector air remains still in
        this first slice; corridor return flow is a later calibration step.
        """

        if temperature_k.shape != self.domain.shape:
            raise ValueError("temperature and domain shapes do not match")
        velocity = np.zeros((*self.domain.shape, 3), dtype=np.float64)
        ambient = self.config.ambient_temperature_k
        cell = self.domain.cell_size_m
        for j, i in np.argwhere(np.any(self.domain.connector, axis=0)):
            column = self.domain.connector[:, j, i]
            if not np.any(column):
                continue
            excess = np.maximum(temperature_k[column, j, i] - ambient, 0.0)
            mean_excess = float(excess.mean())
            height = max(float(column.sum()) * cell, cell)
            speed = self.config.stack_discharge_coefficient * math.sqrt(
                2.0 * _GRAVITY_M_S2 * height * mean_excess / ambient
            )
            velocity[column, j, i, 2] = min(
                speed, self.config.max_stack_speed_m_s
            )
        return velocity

    def step(self, state: HazardState, ticks: int = 1) -> HazardState:
        """Return a new state after ``ticks`` 10 Hz hazard updates."""

        _require(
            isinstance(ticks, int) and not isinstance(ticks, bool) and ticks >= 0,
            "ticks must be a nonnegative integer",
        )
        if state.temperature_k.shape != self.domain.shape:
            raise ValueError("state and domain shapes do not match")
        if ticks == 0:
            return state

        temperature = np.array(state.temperature_k, copy=True)
        fuel = np.array(state.fuel_kg_m3, copy=True)
        smoke = np.array(state.smoke_kg_m3, copy=True)
        elapsed = state.time_s
        for _ in range(ticks):
            velocity = self.stack_velocity(temperature)
            substeps = self._substep_count(velocity)
            dt = self.config.tick_seconds / substeps
            for _ in range(substeps):
                temperature, fuel, smoke = self._substep(
                    temperature, fuel, smoke, velocity, dt
                )
            elapsed += self.config.tick_seconds
        return HazardState(temperature, fuel, smoke, elapsed)

    def _substep_count(self, velocity: np.ndarray) -> int:
        cell = self.domain.cell_size_m
        max_speed = float(np.max(np.abs(velocity)))
        max_diffusivity = max(
            self.config.thermal_diffusivity_m2_s,
            self.config.smoke_diffusivity_m2_s,
        )
        rate = max_speed / cell + 6.0 * max_diffusivity / (cell * cell)
        if rate == 0.0:
            return 1
        stable_dt = self.config.stability_limit / rate
        count = max(1, math.ceil(self.config.tick_seconds / stable_dt))
        if count > self.config.max_substeps:
            raise RuntimeError(
                f"hazard update needs {count} substeps; max_substeps="
                f"{self.config.max_substeps}"
            )
        return count

    def _substep(
        self,
        temperature: np.ndarray,
        fuel: np.ndarray,
        smoke: np.ndarray,
        velocity: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        safe_temperature = np.maximum(temperature, 1.0)
        rate_constant = cfg.combustion_pre_exponential_s_inv * np.exp(
            -cfg.activation_temperature_k / safe_temperature
        )
        # Exact integration of dF/dt = -kF over this operator-split substep.
        # Unlike explicit clipping, this stays positive for every finite k*dt
        # and does not turn an oversized reaction step into instantaneous burn.
        burned = fuel * -np.expm1(-rate_constant * dt)
        fuel = fuel - burned
        temperature = temperature + cfg.heat_release_k_per_kg_m3 * burned
        smoke = smoke + cfg.smoke_yield_per_kg_m3 * burned

        # Advect heat above ambient, not absolute Kelvin temperature.  A
        # conservative flux of absolute temperature creates heat wherever the
        # deliberately simplified stack velocity has nonzero divergence.
        temperature_excess = np.maximum(
            temperature - cfg.ambient_temperature_k, 0.0
        )
        temperature_excess = self._transport(
            temperature_excess, velocity, cfg.thermal_diffusivity_m2_s, dt
        )
        temperature = cfg.ambient_temperature_k + temperature_excess
        smoke = self._transport(smoke, velocity, cfg.smoke_diffusivity_m2_s, dt)
        temperature -= dt * cfg.heat_loss_s_inv * (
            temperature - cfg.ambient_temperature_k
        )

        active = self.domain.active
        temperature = np.where(
            active,
            np.maximum(temperature, cfg.ambient_temperature_k),
            cfg.ambient_temperature_k,
        )
        fuel = np.where(active, np.maximum(fuel, 0.0), 0.0)
        smoke = np.where(active, np.maximum(smoke, 0.0), 0.0)
        return temperature, fuel, smoke

    def _transport(
        self,
        values: np.ndarray,
        velocity: np.ndarray,
        diffusivity: float,
        dt: float,
    ) -> np.ndarray:
        if self.backend == "taichi":
            return _transport_3d_taichi(
                values,
                self.domain.active,
                self.domain.vertical_open,
                velocity,
                diffusivity,
                dt,
                self.domain.cell_size_m,
            )
        return _transport_3d_numpy(
            values,
            self.domain.active,
            self.domain.vertical_open,
            velocity,
            diffusivity,
            dt,
            self.domain.cell_size_m,
        )


def _axis_neighbours(
    values: np.ndarray,
    active: np.ndarray,
    vertical_open: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    before = np.roll(values, 1, axis=axis)
    after = np.roll(values, -1, axis=axis)
    before_active = np.roll(active, 1, axis=axis)
    after_active = np.roll(active, -1, axis=axis)

    low = [slice(None)] * 3
    low[axis] = 0
    high = [slice(None)] * 3
    high[axis] = -1
    before[tuple(low)] = values[tuple(low)]
    after[tuple(high)] = values[tuple(high)]
    before_active[tuple(low)] = False
    after_active[tuple(high)] = False
    if axis == 0:
        before_active[1:] &= vertical_open[:-1]
        after_active[:-1] &= vertical_open[:-1]
    return (
        np.where(before_active, before, values),
        np.where(after_active, after, values),
        before_active,
        after_active,
    )


def _transport_3d_numpy(
    values: np.ndarray,
    active: np.ndarray,
    vertical_open: np.ndarray,
    velocity: np.ndarray,
    diffusivity: float,
    dt: float,
    cell_size: float,
) -> np.ndarray:
    neighbours = tuple(
        _axis_neighbours(values, active, vertical_open, axis) for axis in range(3)
    )
    laplacian = sum(
        before + after - 2.0 * values
        for before, after, _, _ in neighbours
    ) / (cell_size * cell_size)

    divergence = np.zeros_like(values)
    # Array axes z,y,x correspond to velocity components 2,1,0.
    for axis, component in ((0, 2), (1, 1), (2, 0)):
        before, after, active_before, active_after = neighbours[axis]
        speed = velocity[..., component]
        speed_before = np.roll(speed, 1, axis=axis)
        speed_after = np.roll(speed, -1, axis=axis)

        face_before = 0.5 * (speed_before + speed)
        face_after = 0.5 * (speed + speed_after)
        flux_before = np.where(
            face_before >= 0.0,
            face_before * before,
            face_before * values,
        )
        flux_after = np.where(
            face_after >= 0.0,
            face_after * values,
            face_after * after,
        )
        flux_before = np.where(active & active_before, flux_before, 0.0)
        flux_after = np.where(active & active_after, flux_after, 0.0)
        divergence += (flux_after - flux_before) / cell_size
    updated = values + dt * (diffusivity * laplacian - divergence)
    return np.where(active, updated, values)


def _ensure_taichi() -> None:
    with _TAICHI_LOCK:
        try:
            import taichi as ti
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError("the taichi hazard backend is not installed") from exc

        runtime = ti.lang.impl.get_runtime()
        if runtime.prog is None:
            # Taichi finalises its cache manager even with offline caching off.
            # A stable writable directory avoids both ~/.cache failures and a
            # leaked mkdtemp directory for every initialized process.
            cache_dir = Path(tempfile.gettempdir()) / "swarmrescue-taichi-cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            ti.init(
                arch=ti.cpu,
                default_fp=ti.f64,
                default_ip=ti.i32,
                offline_cache=False,
                offline_cache_file_path=str(cache_dir),
            )
        if ti.lang.impl.current_cfg().arch != ti.cpu:
            raise RuntimeError("the hazard backend requires Taichi's CPU runtime")


def _transport_3d_taichi(
    values: np.ndarray,
    active: np.ndarray,
    vertical_open: np.ndarray,
    velocity: np.ndarray,
    diffusivity: float,
    dt: float,
    cell_size: float,
) -> np.ndarray:
    source = np.ascontiguousarray(values, dtype=np.float64)
    mask = np.ascontiguousarray(active, dtype=np.int32)
    vertical = np.ascontiguousarray(vertical_open, dtype=np.int32)
    flow = np.ascontiguousarray(velocity, dtype=np.float64)
    target = np.empty_like(source)
    with _TAICHI_LOCK:
        _ensure_taichi()
        from ._taichi import transport_3d

        transport_3d(source, mask, vertical, flow, target, diffusivity, dt, cell_size)
    return target


def transport_periodic_1d(
    values: np.ndarray,
    velocity_m_s: float,
    diffusivity_m2_s: float,
    dt_s: float,
    cell_size_m: float,
    *,
    steps: int = 1,
    backend: Backend = "numpy",
) -> np.ndarray:
    """Advance periodic 1-D advection-diffusion for analytic verification.

    This deliberately exposes the same first-order-upwind/second-order-central
    stencil used by the 3-D solver.  A sinusoidal initial condition has a
    closed-form solution, making numerical drift visible in a small unit test.
    """

    source = np.asarray(values, dtype=np.float64)
    _require(source.ndim == 1 and source.size >= 3, "values must be a 1-D grid")
    _require(np.all(np.isfinite(source)), "values must be finite")
    _require(math.isfinite(velocity_m_s), "velocity_m_s must be finite")
    _require(
        diffusivity_m2_s >= 0.0 and math.isfinite(diffusivity_m2_s),
        "diffusivity must be finite and nonnegative",
    )
    _require(dt_s > 0.0 and math.isfinite(dt_s), "dt_s must be finite and positive")
    _require(
        cell_size_m > 0.0 and math.isfinite(cell_size_m),
        "cell_size_m must be finite and positive",
    )
    _require(
        isinstance(steps, int) and not isinstance(steps, bool) and steps >= 0,
        "steps must be a nonnegative integer",
    )
    if backend not in ("numpy", "taichi"):
        raise ValueError(f"unknown hazard backend: {backend!r}")
    stability = (
        abs(velocity_m_s) * dt_s / cell_size_m
        + 2.0 * diffusivity_m2_s * dt_s / (cell_size_m * cell_size_m)
    )
    _require(stability <= 1.0 + 1e-12, "explicit transport step is unstable")

    out = np.array(source, copy=True, order="C")
    if backend == "taichi":
        with _TAICHI_LOCK:
            _ensure_taichi()
            from ._taichi import transport_periodic_1d as kernel

            for _ in range(steps):
                target = np.empty_like(out)
                kernel(
                    out,
                    target,
                    velocity_m_s,
                    diffusivity_m2_s,
                    dt_s,
                    cell_size_m,
                )
                out = target
        return out

    for _ in range(steps):
        before = np.roll(out, 1)
        after = np.roll(out, -1)
        gradient = (
            (out - before) / cell_size_m
            if velocity_m_s >= 0.0
            else (after - out) / cell_size_m
        )
        laplacian = (before - 2.0 * out + after) / (cell_size_m * cell_size_m)
        out = out + dt_s * (
            diffusivity_m2_s * laplacian - velocity_m_s * gradient
        )
    return out
