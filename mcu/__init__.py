"""Small Tier-A ESP32 budget model used by robot-side kernels."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

Q16_SCALE = 1 << 16
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def _saturate(value: int) -> int:
    return min(INT32_MAX, max(INT32_MIN, value))


@dataclass(frozen=True, slots=True)
class Q16:
    """Signed, saturating Q16.16 scalar."""

    raw: int

    def __post_init__(self) -> None:
        if not isinstance(self.raw, int) or isinstance(self.raw, bool):
            raise TypeError("Q16 raw value must be an integer")
        object.__setattr__(self, "raw", _saturate(self.raw))

    @classmethod
    def from_float(cls, value: float) -> Q16:
        if not math.isfinite(value):
            raise ValueError("Q16 value must be finite")
        return cls(round(value * Q16_SCALE))

    def __float__(self) -> float:
        return self.raw / Q16_SCALE

    def __add__(self, other: Q16) -> Q16:
        if not isinstance(other, Q16):
            return NotImplemented
        return Q16(self.raw + other.raw)

    def __sub__(self, other: Q16) -> Q16:
        if not isinstance(other, Q16):
            return NotImplemented
        return Q16(self.raw - other.raw)

    def __mul__(self, other: Q16) -> Q16:
        if not isinstance(other, Q16):
            return NotImplemented
        return Q16((self.raw * other.raw) >> 16)


class MCUOutOfMemory(MemoryError):  # noqa: N818 - proposal/API name
    """The fixed robot arena cannot satisfy an allocation."""


@dataclass(slots=True)
class MCUArena:
    """Aligned bump arena with a measurable high-water mark."""

    capacity: int = 520 * 1024
    used: int = field(default=0, init=False)
    high_water: int = field(default=0, init=False)
    _storage: bytearray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("arena capacity must be positive")
        self._storage = bytearray(self.capacity)

    def alloc(self, size: int, *, alignment: int = 4) -> memoryview:
        if size < 0 or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("size must be non-negative and alignment a power of two")
        start = (self.used + alignment - 1) & -alignment
        end = start + size
        if end > self.capacity:
            raise MCUOutOfMemory(
                f"arena needs {end} bytes, capacity is {self.capacity}"
            )
        self.used = end
        self.high_water = max(self.high_water, end)
        return memoryview(self._storage)[start:end]

    def reset(self) -> None:
        self.used = 0


@dataclass(slots=True)
class CycleBudget:
    """Analytical cycle counter for one control tick."""

    limit: int
    used: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValueError("cycle limit must be positive")

    def charge(self, count: int, *, cycles_each: int = 1) -> None:
        if count < 0 or cycles_each < 0:
            raise ValueError("cycle costs must be non-negative")
        self.used += count * cycles_each

    @property
    def missed(self) -> bool:
        return self.used > self.limit


Action = TypeVar("Action")


@dataclass(slots=True)
class DeadlineScheduler(Generic[Action]):  # noqa: UP046 - Python 3.11 support
    """Repeat the previous action when a robot exceeds its cycle budget."""

    budget_cycles: int
    previous_action: Action
    deadline_misses: int = field(default=0, init=False)
    last_cycles: int = field(default=0, init=False)

    def tick(self, propose: Callable[[CycleBudget], Action]) -> Action:
        budget = CycleBudget(self.budget_cycles)
        candidate = propose(budget)
        self.last_cycles = budget.used
        if budget.missed:
            self.deadline_misses += 1
            return self.previous_action
        self.previous_action = candidate
        return candidate
