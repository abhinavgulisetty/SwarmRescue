"""Bit-exact Python reference for the Tier-B Q16 LMDP sweep."""

from __future__ import annotations

import numpy as np

from . import INT32_MAX, INT32_MIN, Q16_SCALE


def _sat32(value: int) -> int:
    return min(INT32_MAX, max(INT32_MIN, value))


def q16_lmdp_sweep(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    passive_q16: np.ndarray,
    attenuation_q16: np.ndarray,
    current_q16: np.ndarray,
    exit_mask: np.ndarray,
    death_mask: np.ndarray,
) -> np.ndarray:
    """Run one saturating CSR Bellman sweep using Q16.16 integers."""

    rows = np.asarray(row_ptr)
    columns = np.asarray(col_idx)
    passive = np.asarray(passive_q16)
    attenuation = np.asarray(attenuation_q16)
    current = np.asarray(current_q16)
    exits = np.asarray(exit_mask, dtype=bool)
    deaths = np.asarray(death_mask, dtype=bool)
    n = attenuation.size
    if rows.shape != (n + 1,) or current.shape != (n,):
        raise ValueError("Q16 LMDP state arrays have inconsistent shapes")
    if exits.shape != (n,) or deaths.shape != (n,):
        raise ValueError("Q16 LMDP masks have inconsistent shapes")
    if columns.shape != passive.shape or int(rows[-1]) != columns.size:
        raise ValueError("Q16 LMDP CSR arrays have inconsistent shapes")

    output = np.zeros(n, dtype=np.int32)
    for state in range(n):
        if exits[state]:
            output[state] = Q16_SCALE
            continue
        if deaths[state]:
            continue
        accumulator = 0
        for edge in range(int(rows[state]), int(rows[state + 1])):
            accumulator += int(passive[edge]) * int(current[int(columns[edge])])
        passive_value = _sat32(
            (accumulator + (Q16_SCALE // 2)) // Q16_SCALE
        )
        output[state] = _sat32(
            (
                int(attenuation[state]) * passive_value
                + (Q16_SCALE // 2)
            )
            // Q16_SCALE
        )
    return output


__all__ = ["q16_lmdp_sweep"]
