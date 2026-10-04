#ifndef EMBER_LMDP_Q16_H
#define EMBER_LMDP_Q16_H

#include <stdint.h>

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
);

#endif
