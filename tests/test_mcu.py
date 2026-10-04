"""Tier-A fixed-point, memory, and deadline invariants."""

import pytest

from mcu import (
    INT32_MAX,
    Q16,
    CycleBudget,
    DeadlineScheduler,
    MCUArena,
    MCUOutOfMemory,
)


def test_q16_multiplies_and_saturates() -> None:
    assert float(Q16.from_float(1.5) * Q16.from_float(2.0)) == 3.0
    assert (Q16(INT32_MAX) + Q16.from_float(1.0)).raw == INT32_MAX


def test_arena_aligns_tracks_high_water_and_fails_closed() -> None:
    arena = MCUArena(16)
    assert len(arena.alloc(3)) == 3
    assert len(arena.alloc(4, alignment=8)) == 4
    assert arena.high_water == 12
    arena.reset()
    assert arena.used == 0 and arena.high_water == 12
    with pytest.raises(MCUOutOfMemory):
        arena.alloc(17)


def test_deadline_miss_repeats_the_previous_action() -> None:
    scheduler = DeadlineScheduler(budget_cycles=10, previous_action="hold")

    def too_slow(budget: CycleBudget) -> str:
        budget.charge(11)
        return "turn"

    assert scheduler.tick(too_slow) == "hold"
    assert scheduler.deadline_misses == 1
    assert scheduler.last_cycles == 11
