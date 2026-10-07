"""Persistent runtime conformance, damaged frames and live robot ownership."""

import os
import shutil
import struct

import numpy as np
import pytest

from ember.physical_team import PhysicalRobotTeam
from ember.regions import RegionRobotTeam
from mcu import MCUArena
from mcu.log_lmdp import solve_log_q16
from mcu.native_log_lmdp import solve_native_log_q16
from mcu.runtime import FirmwareBackend, FirmwareTransportError, build_runtime
from tests.test_log_lmdp import _chain
from tests.test_robot_state import _graph


@pytest.fixture(scope="module")
def host_image(tmp_path_factory):
    if not shutil.which("cc"):
        pytest.skip("C compiler unavailable")
    return build_runtime(tmp_path_factory.mktemp("live-host"))[0]


@pytest.mark.parametrize("nodes", [20, 500])
def test_host_runtime_is_bit_exact_across_repeated_models(host_image, nodes):
    with FirmwareBackend(host_image) as backend:
        for cost in (2.0, 0.4):
            model = _chain(nodes, cost)
            expected = solve_log_q16(model)
            result = backend.solve(model)
            assert result.fixed_point and result.sweeps == expected.sweeps
            np.testing.assert_array_equal(
                result.negative_log_desirability, expected.negative_log_desirability
            )
            assert result.route(0) == tuple(range(nodes))
        assert backend.requests == 2 and backend.transport_bytes > 0
    assert backend.closed and backend.process.poll() == 0


def test_crc_rejection_does_not_poison_the_next_valid_request(host_image):
    with FirmwareBackend(host_image) as backend:
        with pytest.raises(FirmwareTransportError, match="status 1"):
            backend._rpc(1, b"bad payload", checksum=0)
        result = backend.solve(_chain(20))
        assert result.fixed_point and result.route(0) == tuple(range(20))


def test_malformed_dimensions_are_rejected_without_reusing_old_policy(host_image):
    with FirmwareBackend(host_image) as backend:
        backend.solve(_chain(20))
        with pytest.raises(FirmwareTransportError, match="status 2"):
            backend._rpc(1, struct.pack("<III", 3000, 0, 1))
        assert backend.solve(_chain(5)).route(0) == tuple(range(5))


def test_firmware_caps_are_reported_without_host_solver_fallback(host_image):
    from dataclasses import replace

    model = _chain(20)
    model = replace(model, config=replace(model.config, max_sweeps=1))
    with FirmwareBackend(host_image) as backend:
        result = backend.solve(model)
        assert not result.fixed_point and result.sweeps == 1 and result.route(0) == ()
        arena = MCUArena(200 * 1024)
        with pytest.raises(ValueError, match="capacity"):
            backend.solve(_chain(2049), arena=arena)
        assert arena.used == 0


def test_corrupted_executable_does_not_start_a_worker(tmp_path, host_image):
    path = tmp_path / "runtime-host"
    shutil.copy2(host_image, path)
    shutil.copy2(host_image.parent / "runtime.json", tmp_path / "runtime.json")
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="build manifest"):
        FirmwareBackend(path)


@pytest.mark.parametrize("regional", [False, True])
def test_robot_backend_binding_survives_graph_replacement(host_image, regional):
    graph = _graph()
    positions = graph.world_positions()[[0, 2]]
    cls = RegionRobotTeam if regional else PhysicalRobotTeam
    with FirmwareBackend(host_image) as backend:
        team = cls(
            graph,
            positions,
            loss_seed=7,
            routing_backends=(backend,),
            **({"region_cells": 1} if regional else {}),
        )
        assert team.robots[0].routing_backend is backend
        assert team.robots[1].routing_backend is None
        team.replace_graph(graph)
        assert team.robots[0].routing_backend is backend
        assert team.map_epoch == 1


def test_robot_state_uses_the_firmware_solution(host_image):
    from ember.robot_state import RobotState

    graph = _graph()
    robot = RobotState(0, graph, np.zeros(3))
    with FirmwareBackend(host_image) as backend:
        robot.routing_backend = backend
        result = robot.plan(_chain(20))
        assert result.fixed_point and backend.requests == 1
        assert result is robot.routing


@pytest.mark.parametrize("strategy", ["auto", "bound_diagonal", "gauss_seidel"])
def test_live_accelerated_service_preserves_fields_and_rejects_bad_strategy(
    host_image, strategy
):
    from eval.solver_benchmark import make_model

    model = make_model(31001, "grid")
    reference = solve_native_log_q16(model)
    with FirmwareBackend(host_image, strategy=strategy) as backend:
        result = backend.solve(model)
        assert result.fixed_point
        np.testing.assert_array_equal(
            result.negative_log_desirability, reference.negative_log_desirability
        )
        assert backend.requests == 1 and backend.transport_bytes > 0
        with pytest.raises(FirmwareTransportError, match="status 2"):
            backend._rpc(3, struct.pack("<IIII", 1, 0, 1, 99))
        assert backend.solve(_chain(20)).route(0) == tuple(range(20))


def test_auto_arm_uart_handles_sticky_and_full_capacity_models(tmp_path):
    from eval.solver_benchmark import make_model

    root = os.environ.get("EMBER_ARM_TOOL_ROOT")
    if not root and not (
        shutil.which("arm-none-eabi-gcc") and shutil.which("qemu-system-arm")
    ):
        pytest.skip("Cortex-M4 tools unavailable")
    executable, metadata = build_runtime(tmp_path, arm=True, tool_root=root)
    assert metadata["kernel_service_ram_bytes"] < 200 * 1024
    full = _chain(2048)
    # Fill every declared edge slot with four equal outgoing transitions.
    from ember.routing import LMDPConfig, LMDPModel

    full = LMDPModel(
        np.arange(2049) * 4,
        np.repeat(full.col_idx, 4),
        np.full(8192, 0.25),
        full.state_cost,
        full.exit_mask,
        full.death_mask,
        LMDPConfig(max_sweeps=10000),
    )
    with FirmwareBackend(
        executable, arm=True, tool_root=root, strategy="auto"
    ) as backend:
        for model in (make_model(31001, "sticky"), full):
            reference = solve_native_log_q16(model)
            result = backend.solve(model)
            assert result.fixed_point
            np.testing.assert_array_equal(
                result.negative_log_desirability, reference.negative_log_desirability
            )
            for start in (0, model.n_states // 2, model.n_states - 1):
                assert result.route(start) == reference.route(start)


def test_arm_uart_runtime_when_toolchain_available(tmp_path):
    root = os.environ.get("EMBER_ARM_TOOL_ROOT")
    if not root and not (
        shutil.which("arm-none-eabi-gcc") and shutil.which("qemu-system-arm")
    ):
        pytest.skip("Cortex-M4 tools unavailable")
    executable, metadata = build_runtime(tmp_path, arm=True, tool_root=root)
    assert metadata["kernel_service_ram_bytes"] < 200 * 1024
    with FirmwareBackend(executable, arm=True, tool_root=root) as backend:
        for nodes in (20, 500):
            expected = solve_log_q16(_chain(nodes))
            result = backend.solve(_chain(nodes))
            np.testing.assert_array_equal(
                result.negative_log_desirability, expected.negative_log_desirability
            )
            assert result.route(0) == tuple(range(nodes))
    assert backend.process.poll() == 0


@pytest.mark.parametrize(
    "args",
    [
        ["--firmware-robots", "3"],
        ["--firmware-robots", "1", "--live-disaster", "--distributed"],
    ],
)
def test_invalid_firmware_mission_configuration_rejected_before_scene(args):
    from cascade.demo import main

    with pytest.raises(SystemExit) as result:
        main(args)
    assert result.value.code == 2
