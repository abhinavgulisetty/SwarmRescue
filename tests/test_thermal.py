"""Thermal hazard must cause real, one-shot constraint failure."""

from __future__ import annotations

import mujoco
import numpy as np

from cascade.floorplan import build_floorplan
from cascade.mjcf import to_mjcf
from cascade.params import BuildingParams
from cascade.structure import build_structure
from cascade.thermal import ThermalDamage, ThermalDamageConfig
from hazard import HazardConfig, HazardDomain, HazardState


def test_hot_member_releases_its_real_mujoco_welds_once() -> None:
    plan = build_floorplan(7, BuildingParams())
    structure = build_structure(plan)
    model = mujoco.MjModel.from_xml_string(to_mjcf(structure))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    domain = HazardDomain.from_floorplan(plan, HazardConfig(cell_size_m=1.0))
    temperature = np.full(domain.shape, 900.0)
    state = HazardState(
        temperature,
        domain.initial_fuel_kg_m3,
        np.zeros(domain.shape),
    )
    damage = ThermalDamage(
        structure,
        ThermalDamageConfig(
            onset_temperature_k=300.0,
            failure_exposure_k_s=1.0,
        ),
    )

    active_before = int(np.count_nonzero(data.eq_active))
    failed = damage.step(state, domain, model, data, 1.0)
    active_after = int(np.count_nonzero(data.eq_active))

    assert failed
    assert damage.released_welds
    assert active_after < active_before
    assert damage.step(state, domain, model, data, 1.0) == ()
    assert int(np.count_nonzero(data.eq_active)) == active_after
