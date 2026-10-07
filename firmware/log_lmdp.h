#ifndef EMBER_LOG_LMDP_H
#define EMBER_LOG_LMDP_H
#include <stdint.h>
/* Whole-solve strategies share the same integer row operator. */
enum ember_log_strategy {
    EMBER_LOG_JACOBI = 0,
    EMBER_LOG_GAUSS_SEIDEL = 1,
    EMBER_LOG_BOUND_JACOBI = 2,
    EMBER_LOG_BOUND_ORDERED = 3,
    EMBER_LOG_PRIORITIZED = 4,
    EMBER_LOG_BOUND_PRIORITIZED = 5,
    EMBER_LOG_BOUND_DIAGONAL = 6,
    EMBER_LOG_AUTO = 7
};
typedef struct {
    uint64_t backups;
    uint64_t bound_edges;
    uint32_t sweeps;
    uint32_t fixed_point;
    uint32_t strategy;
} ember_log_stats;
/* Caller-owned scratch in int32 words; -1 for invalid/overflowing sizes. */
int32_t ember_log_q16_workspace(int32_t n, int32_t n_edges, int32_t strategy);
/* Cold start. Negative status rejects input; exhausted cap returns fixed_point=0.
 * Values and workspace must be distinct and must not overlap model arrays.
 * Bound strategies use a reversed Dijkstra supersolution and never skip states
 * whose bound saturates: parallel paths can still give them finite soft costs.
 * Prioritized schedules use a budget of limit*N applied updates; their sweep
 * count is ceil(applied_updates/N), at least one. Backups include queue refresh.
 */
int ember_log_q16_solve(
    int32_t n, int32_t n_edges, const int32_t *rows, const int32_t *columns,
    const int32_t *negative_log_passive, const int32_t *cost,
    const uint8_t *exits, const uint8_t *deaths, int32_t limit, int32_t strategy,
    int32_t *values, int32_t *workspace, int32_t workspace_words,
    ember_log_stats *stats);
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
