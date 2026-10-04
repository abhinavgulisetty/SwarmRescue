#include "lmdp_q16.h"

#include <limits.h>
#include <stddef.h>

#define Q16_SCALE 65536LL

static int32_t sat32(int64_t value) {
    if (value > INT32_MAX) {
        return INT32_MAX;
    }
    if (value < INT32_MIN) {
        return INT32_MIN;
    }
    return (int32_t)value;
}

int ember_lmdp_q16_sweep(
    int32_t n_states,
    const int32_t *row_ptr,
    const int32_t *col_idx,
    const int32_t *passive_q16,
    const int32_t *attenuation_q16,
    const int32_t *current_q16,
    const uint8_t *exit_mask,
    const uint8_t *death_mask,
    int32_t *output_q16
) {
    int32_t state;
    if (
        n_states <= 0 || row_ptr == NULL || col_idx == NULL
        || passive_q16 == NULL || attenuation_q16 == NULL
        || current_q16 == NULL || exit_mask == NULL
        || death_mask == NULL || output_q16 == NULL
    ) {
        return -1;
    }
    for (state = 0; state < n_states; ++state) {
        int32_t edge;
        int64_t accumulator = 0;
        int32_t passive_value;
        if (exit_mask[state]) {
            output_q16[state] = (int32_t)Q16_SCALE;
            continue;
        }
        if (death_mask[state]) {
            output_q16[state] = 0;
            continue;
        }
        if (row_ptr[state] < 0 || row_ptr[state + 1] < row_ptr[state]) {
            return -2;
        }
        for (edge = row_ptr[state]; edge < row_ptr[state + 1]; ++edge) {
            if (col_idx[edge] < 0 || col_idx[edge] >= n_states) {
                return -3;
            }
            accumulator += (
                (int64_t)passive_q16[edge]
                * (int64_t)current_q16[col_idx[edge]]
            );
        }
        passive_value = sat32(
            (accumulator + Q16_SCALE / 2) / Q16_SCALE
        );
        output_q16[state] = sat32(
            (
                (int64_t)attenuation_q16[state]
                * (int64_t)passive_value
                + Q16_SCALE / 2
            )
            / Q16_SCALE
        );
    }
    return 0;
}
