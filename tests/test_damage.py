"""Tests for blast damage resolution.

`cascade.damage` is deliberately a pure function of geometry and seed, so all
of this runs without starting a physics engine.
"""

from __future__ import annotations

import pytest

from cascade.damage import (
    BlastDamage,
    compute_damage,
    sample_damage,
    slab_supports,
)
from cascade.floorplan import build_floorplan
from cascade.params import BuildingParams, CollapseParams
from cascade.structure import MemberKind, build_structure

SEEDS = (0, 7, 42, 123, 1000)


@pytest.fixture(scope="module")
def structure():
    return build_structure(build_floorplan(7, BuildingParams()))


@pytest.fixture(scope="module")
def collapse_params() -> CollapseParams:
    return CollapseParams()


def _structure_for(seed: int):
    return build_structure(build_floorplan(seed, BuildingParams()))


# --- compute_damage ----------------------------------------------------------


def test_zero_radius_damages_nothing(structure) -> None:
    damage = compute_damage(structure, (12.0, 9.0, 0.0), 0.0)
    assert damage.destroyed_columns == ()
    assert damage.freed_members == ()
    assert damage.severed_welds == ()


def test_damage_is_deterministic(structure) -> None:
    a = compute_damage(structure, (10.0, 8.0, 0.0), 6.0)
    b = compute_damage(structure, (10.0, 8.0, 0.0), 6.0)
    assert a == b


def test_destroyed_and_freed_are_disjoint(structure) -> None:
    damage = compute_damage(structure, (10.0, 8.0, 0.0), 7.0)
    assert set(damage.destroyed_columns).isdisjoint(damage.freed_members)


def test_only_frangible_kinds_are_destroyed(structure) -> None:
    """Slabs and beams survive as debris; columns and partitions do not."""
    damage = compute_damage(structure, (10.0, 8.0, 0.0), 7.0)
    kinds = {m.name: m.kind for m in structure.members}
    for name in damage.destroyed_columns:
        assert kinds[name] in (MemberKind.COLUMN, MemberKind.WALL)
    for name in damage.freed_members:
        assert kinds[name] in (MemberKind.SLAB, MemberKind.BEAM)


def test_damage_is_full_height(structure) -> None:
    """A blast damages a vertical section, not just the storey it sits on."""
    damage = compute_damage(structure, (10.0, 8.0, 0.0), 7.0)
    storeys = {
        m.storey
        for m in structure.members
        if m.name in set(damage.destroyed_columns) and m.kind is MemberKind.COLUMN
    }
    assert len(storeys) > 1, "blast should reach columns on more than one storey"


def test_larger_radius_damages_a_superset(structure) -> None:
    small = compute_damage(structure, (10.0, 8.0, 0.0), 5.0)
    large = compute_damage(structure, (10.0, 8.0, 0.0), 9.0)
    assert set(small.destroyed_columns) <= set(large.destroyed_columns)


def test_enclosing_radius_destroys_every_column(structure) -> None:
    damage = compute_damage(structure, (12.0, 9.0, 0.0), 1000.0)
    columns = {m.name for m in structure.members if m.kind is MemberKind.COLUMN}
    assert columns <= set(damage.destroyed_columns)


def test_welds_touching_a_destroyed_member_are_severed(structure) -> None:
    damage = compute_damage(structure, (10.0, 8.0, 0.0), 7.0)
    destroyed = set(damage.destroyed_columns)
    severed = set(damage.severed_welds)
    for weld in structure.welds:
        touches = weld.body_a in destroyed or (
            weld.body_b is not None and weld.body_b in destroyed
        )
        if touches:
            assert weld.name in severed


def test_boundary_crossing_welds_are_severed(structure) -> None:
    """The damaged section must be able to separate from the intact one."""
    damage = compute_damage(structure, (10.0, 8.0, 0.0), 7.0)
    inside = set(damage.destroyed_columns) | set(damage.freed_members)
    severed = set(damage.severed_welds)
    for weld in structure.welds:
        a_in = weld.body_a in inside
        b_in = weld.body_b in inside if weld.body_b is not None else False
        if a_in != b_in:
            assert weld.name in severed, (
                f"{weld.name} spans the blast boundary but was not severed"
            )


def test_intact_structure_keeps_its_welds(structure) -> None:
    """Nothing outside the blast should lose a connection."""
    damage = compute_damage(structure, (2.0, 2.0, 0.0), 3.0)
    inside = set(damage.destroyed_columns) | set(damage.freed_members)
    for weld in structure.welds:
        a_out = weld.body_a not in inside
        b_out = weld.body_b is None or weld.body_b not in inside
        if a_out and b_out and weld.name in set(damage.severed_welds):
            pytest.fail(f"{weld.name} joins two intact members but was severed")


# --- sample_damage -----------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_sampling_is_deterministic(seed: int, collapse_params) -> None:
    st = _structure_for(seed)
    assert sample_damage(st, seed, collapse_params) == sample_damage(
        st, seed, collapse_params
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_sampled_blast_respects_size_bounds(seed: int, collapse_params) -> None:
    """Enough damage to matter, little enough that a building remains."""
    st = _structure_for(seed)
    damage = sample_damage(st, seed, collapse_params)
    ground = [
        m for m in st.members if m.kind is MemberKind.COLUMN and m.storey == 0
    ]
    destroyed = set(damage.destroyed_columns)
    hit = sum(1 for c in ground if c.name in destroyed)
    fraction = hit / len(ground)
    assert fraction <= collapse_params.max_destroyed_column_fraction + 1e-9, (
        f"seed {seed} destroys {fraction:.0%} of the ground frame"
    )
    assert fraction > 0.0


@pytest.mark.parametrize("seed", SEEDS)
def test_sampled_blast_leaves_a_building_standing(seed: int, collapse_params) -> None:
    st = _structure_for(seed)
    damage = sample_damage(st, seed, collapse_params)
    total = sum(1 for m in st.members if m.kind is MemberKind.COLUMN)
    survivors = total - sum(
        1
        for m in st.members
        if m.kind is MemberKind.COLUMN and m.name in set(damage.destroyed_columns)
    )
    assert survivors > 0, f"seed {seed} destroyed every column"


@pytest.mark.parametrize("seed", SEEDS)
def test_sampled_blast_fully_unsupports_a_slab(seed: int, collapse_params) -> None:
    """The precondition for anything actually falling.

    A blast can destroy plenty of columns and still leave every slab propped
    on a survivor, in which case the building is only stripped of beams and
    partitions and no floor comes down.
    """
    st = _structure_for(seed)
    damage = sample_damage(st, seed, collapse_params)
    supports = slab_supports(st)
    destroyed = set(damage.destroyed_columns)
    assert any(cols and cols <= destroyed for cols in supports.values()), (
        f"seed {seed}: no slab lost all of its supporting columns"
    )


def test_slab_supports_covers_every_bay(structure) -> None:
    """Support is tracked per bay, since a bay's fragments are welded as one."""
    from cascade.damage import _bay_key

    supports = slab_supports(structure)
    bays = {
        _bay_key(m.name)
        for m in structure.members
        if m.kind is MemberKind.SLAB
    }
    assert bays <= set(supports), "some bay's floor plate has no support columns"
    assert all(cols for cols in supports.values())


def test_blast_damage_is_frozen() -> None:
    damage = BlastDamage((0.0, 0.0, 0.0), 1.0, (), (), ())
    with pytest.raises(AttributeError):
        damage.radius_m = 2.0  # type: ignore[misc]
