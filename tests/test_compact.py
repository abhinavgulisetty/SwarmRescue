"""The compact demo remains smaller without weakening collapse semantics."""

from __future__ import annotations

from cascade.damage import sample_damage
from cascade.floorplan import build_floorplan
from cascade.params import CascadeParams, compact_demo_params
from cascade.structure import MemberKind, build_structure


def test_compact_preset_has_fewer_larger_visible_members() -> None:
    default_plan = build_floorplan(7, CascadeParams().building)
    compact = compact_demo_params()
    compact_plan = build_floorplan(7, compact.building)
    default_structure = build_structure(default_plan)
    compact_structure = build_structure(compact_plan)

    assert compact.building.n_storeys == 3
    assert compact_plan.footprint.area < default_plan.footprint.area
    assert len(compact_structure.members) < 0.5 * len(default_structure.members)
    assert 4 <= len(compact_plan.bays) <= 6


def test_compact_seed_seven_loses_a_visible_bay_but_not_the_building() -> None:
    params = compact_demo_params()
    structure = build_structure(build_floorplan(7, params.building))
    damage = sample_damage(structure, 7, params.collapse)
    ground_columns = [
        member
        for member in structure.members
        if member.kind is MemberKind.COLUMN and member.storey == 0
    ]
    destroyed = sum(
        member.name in damage.destroyed_columns for member in ground_columns
    )

    assert destroyed >= 4
    assert destroyed < len(ground_columns)
    assert len(damage.freed_members) >= 20
