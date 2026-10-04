"""Physics and geometry contracts for the Phase 1 hazard oracle."""

from __future__ import annotations

import importlib.util
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from cascade.floorplan import FloorPlan, build_floorplan
from cascade.params import FUEL_LOAD_KG_M2, BuildingParams, RoomKind
from hazard import (
    HazardConfig,
    HazardDomain,
    HazardSolver,
    HazardState,
    transport_periodic_1d,
)


@pytest.fixture(scope="module")
def plan() -> FloorPlan:
    return build_floorplan(7, BuildingParams())


@pytest.fixture(scope="module")
def config() -> HazardConfig:
    # A half-metre grid resolves even the service duct and divides the 3 m
    # storey height exactly, which makes vertical-alignment assertions exact.
    return HazardConfig(cell_size_m=0.5, tick_seconds=0.05)


@pytest.fixture(scope="module")
def domain(plan: FloorPlan, config: HazardConfig) -> HazardDomain:
    return HazardDomain.from_floorplan(plan, config)


@pytest.mark.parametrize(
    "overrides",
    [
        {"cell_size_m": 0.0},
        {"tick_seconds": 0.0},
        {"ambient_temperature_k": 0.0},
        {"thermal_diffusivity_m2_s": -0.01},
        {"smoke_diffusivity_m2_s": -0.01},
        {"stack_discharge_coefficient": 1.01},
        {"max_substeps": 0},
        {"cell_size_m": np.inf},
        {"tick_seconds": np.inf},
        {"heat_loss_s_inv": np.inf},
        {"combustion_pre_exponential_s_inv": np.inf},
        {"max_stack_speed_m_s": np.inf},
    ],
    ids=[
        "zero-cell-size",
        "zero-timestep",
        "absolute-zero-ambient",
        "negative-thermal-diffusivity",
        "negative-smoke-diffusivity",
        "invalid-discharge-coefficient",
        "zero-substeps",
        "infinite-cell-size",
        "infinite-timestep",
        "infinite-heat-loss",
        "infinite-reaction-rate",
        "infinite-stack-speed",
    ],
)
def test_config_rejects_nonphysical_values(overrides: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        replace(HazardConfig(), **overrides)


def test_floorplan_semantics_set_fuel_and_aligned_connector_mask(
    plan: FloorPlan, config: HazardConfig, domain: HazardDomain
) -> None:
    state = domain.initial_state(config)

    # Use the largest room of each kind so trilinear samples are comfortably
    # inside one semantic region rather than blending across a partition.
    representatives = {}
    for storey in plan.storeys:
        for room in storey.rooms:
            current = representatives.get(room.kind)
            if current is None or room.rect.area > current[1].rect.area:
                representatives[room.kind] = (storey, room)

    expected_kinds = {
        RoomKind.OFFICE,
        RoomKind.CORRIDOR,
        RoomKind.STAIRWELL,
        RoomKind.SHAFT,
        RoomKind.DUCT,
    }
    assert expected_kinds <= representatives.keys()
    for kind in expected_kinds:
        storey, room = representatives[kind]
        x, y = room.rect.center
        z = (storey.index + 0.5) * plan.params.storey_height_m
        expected = FUEL_LOAD_KG_M2[kind] / plan.params.storey_height_m
        assert state.sample(domain, (x, y, z), field="fuel") == pytest.approx(
            expected, rel=1e-12
        )

    mask = domain.connector
    assert mask.dtype == np.bool_
    assert np.any(mask)
    assert np.any(~mask)
    np.testing.assert_array_equal(mask, np.broadcast_to(mask[0], mask.shape))

    # Every grammar connector centre must rasterize to a continuous vertical
    # column; equality of arbitrary slices alone would not catch an empty mask.
    ox, oy, _ = domain.origin_m
    for connector in plan.connectors:
        x, y = connector.rect.center
        ix = int((x - ox) / domain.cell_size_m)
        iy = int((y - oy) / domain.cell_size_m)
        assert np.all(mask[:, iy, ix])

    layers_per_storey = round(
        plan.params.storey_height_m / domain.cell_size_m
    )
    for storey in range(plan.params.n_storeys - 1):
        below = (storey + 1) * layers_per_storey - 1
        expected_open = mask[below] & mask[below + 1]
        np.testing.assert_array_equal(domain.vertical_open[below], expected_open)
        assert np.any(expected_open)
        assert np.any(domain.active[below] & ~expected_open)


def test_floorplan_rejects_a_cell_size_that_cannot_tile_the_domain(
    plan: FloorPlan,
) -> None:
    with pytest.raises(ValueError, match="evenly divide"):
        HazardDomain.from_floorplan(plan, HazardConfig(cell_size_m=0.7))
    with pytest.raises(ValueError, match="storey height"):
        HazardDomain.from_floorplan(plan, HazardConfig(cell_size_m=1.2))


def test_domain_retains_custom_configuration(plan: FloorPlan) -> None:
    config = HazardConfig(cell_size_m=1.0, ambient_temperature_k=310.0)
    domain = HazardDomain.from_floorplan(plan, config)

    assert domain.config == config
    assert np.all(domain.initial_state().temperature_k == 310.0)
    assert HazardSolver(domain).config == config


def _seeded_state(domain: HazardDomain, config: HazardConfig) -> HazardState:
    initial = domain.initial_state(config)
    source = np.unravel_index(np.argmax(initial.fuel_kg_m3), domain.shape)
    temperature = np.array(initial.temperature_k, copy=True)
    smoke = np.array(initial.smoke_kg_m3, copy=True)
    temperature[source] = 900.0
    smoke[source] = 0.8
    return HazardState(temperature, initial.fuel_kg_m3, smoke)


def test_ignition_returns_a_reproducible_snapshot(
    config: HazardConfig, domain: HazardDomain
) -> None:
    initial = domain.initial_state(config)
    cell = np.argwhere(domain.active & ~domain.connector)[0]
    z, y, x = (int(index) for index in cell)
    ox, oy, oz = domain.origin_m
    world = (
        ox + (x + 0.5) * domain.cell_size_m,
        oy + (y + 0.5) * domain.cell_size_m,
        oz + (z + 0.5) * domain.cell_size_m,
    )
    first = initial.ignite(
        domain,
        world,
        radius_m=0.51 * domain.cell_size_m,
        temperature_k=850.0,
        smoke_kg_m3=0.2,
    )
    second = initial.ignite(
        domain,
        world,
        radius_m=0.51 * domain.cell_size_m,
        temperature_k=850.0,
        smoke_kg_m3=0.2,
    )

    np.testing.assert_array_equal(first.temperature_k, second.temperature_k)
    np.testing.assert_array_equal(first.smoke_kg_m3, second.smoke_kg_m3)
    assert first.temperature_k[z, y, x] == 850.0
    assert first.smoke_kg_m3[z, y, x] == 0.2
    assert initial.temperature_k[z, y, x] == config.ambient_temperature_k
    assert initial.smoke_kg_m3[z, y, x] == 0.0


def test_step_is_deterministic_finite_and_nonnegative(
    config: HazardConfig, domain: HazardDomain
) -> None:
    first = _seeded_state(domain, config)
    second = _seeded_state(domain, config)
    original = (
        first.temperature_k.copy(),
        first.fuel_kg_m3.copy(),
        first.smoke_kg_m3.copy(),
    )
    solver = HazardSolver(domain, config, backend="numpy")

    result_a = solver.step(first, ticks=4)
    result_b = solver.step(second, ticks=4)

    for field in ("temperature_k", "fuel_kg_m3", "smoke_kg_m3"):
        values_a = getattr(result_a, field)
        values_b = getattr(result_b, field)
        np.testing.assert_array_equal(values_a, values_b)
        assert np.all(np.isfinite(values_a))
        assert np.all(values_a >= 0.0)

    # Stepping is functional: callers can retain a snapshot as ground truth.
    np.testing.assert_array_equal(first.temperature_k, original[0])
    np.testing.assert_array_equal(first.fuel_kg_m3, original[1])
    np.testing.assert_array_equal(first.smoke_kg_m3, original[2])
    assert result_a.time_s == pytest.approx(4 * config.tick_seconds)
    assert not np.array_equal(result_a.temperature_k, first.temperature_k)
    assert not np.array_equal(result_a.smoke_kg_m3, first.smoke_kg_m3)


def test_heat_excess_is_not_created_by_stack_transport(
    domain: HazardDomain,
) -> None:
    config = HazardConfig(
        cell_size_m=domain.cell_size_m,
        tick_seconds=0.05,
        thermal_diffusivity_m2_s=0.0,
        smoke_diffusivity_m2_s=0.0,
        heat_loss_s_inv=0.0,
        combustion_pre_exponential_s_inv=0.0,
        stack_discharge_coefficient=0.8,
    )
    temperature = np.full(domain.shape, config.ambient_temperature_k)
    source = np.argwhere(domain.connector)[0]
    temperature[tuple(source)] += 500.0
    state = HazardState(
        temperature,
        np.zeros(domain.shape),
        np.zeros(domain.shape),
    )
    before = float(
        np.sum(state.temperature_k[domain.active] - config.ambient_temperature_k)
    )

    result = HazardSolver(domain, config).step(state, ticks=20)
    after = float(
        np.sum(result.temperature_k[domain.active] - config.ambient_temperature_k)
    )

    assert after == pytest.approx(before, rel=0.0, abs=1.0e-9)


def test_default_combustion_is_gradual_and_positive(domain: HazardDomain) -> None:
    config = HazardConfig(
        cell_size_m=domain.cell_size_m,
        thermal_diffusivity_m2_s=0.0,
        smoke_diffusivity_m2_s=0.0,
        heat_loss_s_inv=0.0,
        stack_discharge_coefficient=0.0,
    )
    initial = domain.initial_state(config)
    source = np.unravel_index(np.argmax(initial.fuel_kg_m3), domain.shape)
    temperature = np.array(initial.temperature_k, copy=True)
    temperature[source] = 900.0
    state = HazardState(temperature, initial.fuel_kg_m3, initial.smoke_kg_m3)

    result = HazardSolver(domain, config).step(state)
    burned_fraction = 1.0 - result.fuel_kg_m3[source] / state.fuel_kg_m3[source]

    assert 0.0 < burned_fraction < 0.01
    assert 900.0 < result.temperature_k[source] < 1_000.0


def test_concurrent_cold_taichi_calls_are_serialized() -> None:
    if importlib.util.find_spec("taichi") is None:
        pytest.skip("Taichi is not installed")
    values = np.linspace(0.0, 1.0, 32, dtype=np.float64)

    def run() -> np.ndarray:
        return transport_periodic_1d(
            values,
            0.1,
            0.01,
            0.001,
            0.1,
            steps=2,
            backend="taichi",
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: run(), range(4)))

    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])


def test_taichi_3d_backend_matches_numpy(
    config: HazardConfig, domain: HazardDomain
) -> None:
    if importlib.util.find_spec("taichi") is None:
        pytest.skip("Taichi is not installed")
    state = _seeded_state(domain, config)
    expected = HazardSolver(domain, config, backend="numpy").step(state, ticks=2)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*locale.getdefaultlocale.*deprecated.*",
            category=DeprecationWarning,
        )
        actual = HazardSolver(domain, config, backend="taichi").step(state, ticks=2)

    np.testing.assert_allclose(actual.temperature_k, expected.temperature_k, atol=1e-12)
    np.testing.assert_allclose(actual.fuel_kg_m3, expected.fuel_kg_m3, atol=1e-12)
    np.testing.assert_allclose(actual.smoke_kg_m3, expected.smoke_kg_m3, atol=1e-12)


@pytest.mark.parametrize("backend", ["numpy", "taichi"])
def test_periodic_transport_matches_analytic_advection_diffusion(
    backend: str,
) -> None:
    if backend == "taichi" and importlib.util.find_spec("taichi") is None:
        pytest.skip("Taichi is not installed")

    count = 128
    length_m = 2.0 * np.pi
    cell_size_m = length_m / count
    x = np.arange(count, dtype=np.float64) * cell_size_m
    velocity_m_s = 0.3
    diffusivity_m2_s = 0.04
    dt_s = 0.001
    steps = 100
    mode = 2.0
    initial = 0.35 + 0.2 * np.sin(mode * x)

    # Taichi 1.7 calls locale.getdefaultlocale during import.  Python 3.12
    # deprecates that call, and this project's strict warning policy would
    # otherwise mistake the dependency warning for a backend failure.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*locale.getdefaultlocale.*deprecated.*",
            category=DeprecationWarning,
        )
        actual = transport_periodic_1d(
            initial,
            velocity_m_s,
            diffusivity_m2_s,
            dt_s,
            cell_size_m,
            steps=steps,
            backend=backend,
        )
    elapsed_s = steps * dt_s
    expected = 0.35 + 0.2 * np.exp(
        -diffusivity_m2_s * mode**2 * elapsed_s
    ) * np.sin(mode * (x - velocity_m_s * elapsed_s))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=7.0e-4)
    assert actual.sum() == pytest.approx(initial.sum(), rel=2e-12)


def test_world_coordinate_sampling_is_trilinear(
    config: HazardConfig, domain: HazardDomain
) -> None:
    initial = domain.initial_state(config)
    z_index, y_index, x_index = np.indices(domain.shape, dtype=np.float64)
    smoke = 0.001 * (100.0 * z_index + 10.0 * y_index + x_index)
    state = HazardState(initial.temperature_k, initial.fuel_kg_m3, smoke)

    # Fractional array indices deliberately exercise all eight neighbours.
    fx, fy, fz = 3.25, 4.5, 2.75
    fractions = (fx, fy, fz)
    world = tuple(
        origin + (fraction + 0.5) * domain.cell_size_m
        for origin, fraction in zip(domain.origin_m, fractions, strict=True)
    )
    expected = 0.001 * (100.0 * fz + 10.0 * fy + fx)

    assert state.sample(domain, world, field="smoke") == pytest.approx(
        expected, abs=1e-12
    )

    samples = state.sample_many(
        domain,
        np.asarray((world, (-1.0, -1.0, -1.0))),
        "smoke",
        outside_value=0.0,
    )
    assert samples[0] == pytest.approx(expected, abs=1e-12)
    assert samples[1] == 0.0
    with pytest.raises(ValueError, match="outside"):
        state.sample_many(
            domain,
            np.asarray((world, (-1.0, -1.0, -1.0))),
            "smoke",
        )


def test_hot_connector_causally_lifts_smoke_faster_than_corridor(
    plan: FloorPlan, domain: HazardDomain
) -> None:
    # Disable combustion and cooling to isolate the temperature-driven stack
    # effect from smoke production and changing source strength.
    config = HazardConfig(
        cell_size_m=domain.cell_size_m,
        tick_seconds=0.05,
        thermal_diffusivity_m2_s=0.0,
        smoke_diffusivity_m2_s=0.02,
        heat_loss_s_inv=0.0,
        combustion_pre_exponential_s_inv=0.0,
    )
    initial = domain.initial_state(config)
    source_z = 1

    connector_cells = np.argwhere(domain.connector[source_z])
    assert connector_cells.size > 0
    connector_y, connector_x = connector_cells[len(connector_cells) // 2]

    corridor_fuel = FUEL_LOAD_KG_M2[RoomKind.CORRIDOR] / (
        plan.params.storey_height_m
    )
    corridor_cells = np.argwhere(
        ~domain.connector[source_z]
        & np.isclose(initial.fuel_kg_m3[source_z], corridor_fuel)
    )
    assert corridor_cells.size > 0
    distances = (corridor_cells[:, 0] - connector_y) ** 2 + (
        corridor_cells[:, 1] - connector_x
    ) ** 2
    corridor_y, corridor_x = corridor_cells[int(np.argmax(distances))]

    def pulse_state(y: int, x: int, *, in_connector: bool) -> HazardState:
        temperature = np.full(domain.shape, config.ambient_temperature_k)
        if in_connector:
            temperature[domain.connector] += 500.0
        else:
            temperature[:, y, x] += 500.0
        smoke = np.zeros(domain.shape, dtype=np.float64)
        smoke[source_z, y, x] = 1.0
        return HazardState(temperature, np.zeros(domain.shape), smoke)

    connector_state = pulse_state(
        int(connector_y), int(connector_x), in_connector=True
    )
    corridor_state = pulse_state(int(corridor_y), int(corridor_x), in_connector=False)
    solver = HazardSolver(domain, config, backend="numpy")

    connector_velocity = solver.stack_velocity(connector_state.temperature_k)
    corridor_velocity = solver.stack_velocity(corridor_state.temperature_k)
    assert connector_velocity[source_z, connector_y, connector_x, 2] > 0.0
    assert corridor_velocity[source_z, corridor_y, corridor_x, 2] == 0.0

    connector_result = solver.step(connector_state, ticks=8)
    corridor_result = solver.step(corridor_state, ticks=8)
    assert connector_result.smoke_kg_m3.sum() == pytest.approx(1.0, abs=1e-12)
    assert corridor_result.smoke_kg_m3.sum() == pytest.approx(1.0, abs=1e-12)
    heights = (np.arange(domain.shape[0]) + 0.5) * domain.cell_size_m

    def smoke_centre_height(state: HazardState) -> float:
        mass_by_height = state.smoke_kg_m3.sum(axis=(1, 2))
        assert mass_by_height.sum() > 0.0
        return float(np.dot(heights, mass_by_height) / mass_by_height.sum())

    connector_height = smoke_centre_height(connector_result)
    corridor_height = smoke_centre_height(corridor_result)
    assert connector_height > corridor_height + 0.5 * domain.cell_size_m
