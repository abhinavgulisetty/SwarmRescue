#ifndef EMBER_DEADLINE_H
#define EMBER_DEADLINE_H
#include <stdint.h>
typedef struct {
    int32_t node, time, cost, previous;
    uint8_t state; /* 0 dominated, 1 pending, 2 processed */
} ember_deadline_label;
typedef struct {
    int32_t travel_steps, route_length;
    int64_t objective_q16;
    int32_t feasible, optimal, resource_fallback, labels_used;
} ember_deadline_result;
/* All workspace is provided by the caller; route has capacity n. */
int ember_deadline_route(
    int32_t n, int32_t n_edges, const int32_t *rows, const int32_t *columns,
    const int32_t *edge_steps, const int32_t *edge_cost_q16,
    const uint8_t *exits, int32_t start, int32_t deadline, int32_t label_capacity,
    ember_deadline_label *labels, int32_t *fast_time, int64_t *fast_cost,
    int32_t *fast_previous, uint8_t *visited, int32_t *route,
    ember_deadline_result *result);
/* Validate a bounded proposal against the same static traversal clock. */
int ember_deadline_accept(
    int32_t n, int32_t n_edges, const int32_t *rows, const int32_t *columns,
    const int32_t *edge_steps, const uint8_t *exits, int32_t start,
    int32_t deadline, const int32_t *route, int32_t route_length);

#endif
