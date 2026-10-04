#ifndef EMBER_LOG_LMDP_H
#define EMBER_LOG_LMDP_H
#include <stdint.h>
/* Fixed-state buffers must be distinct. INT32_MAX denotes unreachable. */
int ember_log_q16_sweep(
    int32_t n, int32_t n_edges, const int32_t *rows, const int32_t *columns,
    const int32_t *negative_log_passive, const int32_t *cost,
    const int32_t *current, const uint8_t *exits, const uint8_t *deaths,
    int32_t *output);
/* Return -1 for no progressing action, otherwise a node index. */
int32_t ember_log_q16_next(
    int32_t state, int32_t n, const int32_t *rows, const int32_t *columns,
    const int32_t *negative_log_passive, const int32_t *values);
#endif
