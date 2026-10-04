#include "log_lmdp.h"
#include "log_lmdp_lut.h"
#include <limits.h>
#include <stddef.h>

static int32_t softplus(int64_t distance) {
    int32_t index, remainder;
    if (distance >= 1048576) return 0;
    index = (int32_t)(distance / 256);
    remainder = (int32_t)(distance % 256);
    return (ember_softplus_lut[index] * (256 - remainder)
        + ember_softplus_lut[index + 1] * remainder + 128) / 256;
}

int ember_log_q16_sweep(
    int32_t n, int32_t n_edges, const int32_t *rows, const int32_t *columns,
    const int32_t *negative_log_passive, const int32_t *cost,
    const int32_t *current, const uint8_t *exits, const uint8_t *deaths,
    int32_t *output) {
    int32_t state, edge;
    if (n <= 0 || n_edges < 0 || !rows || !columns || !negative_log_passive
        || !cost || !current || !exits || !deaths || !output || current == output)
        return -1;
    if (rows[0] != 0 || rows[n] != n_edges) return -2;
    for (state = 0; state < n; ++state) {
        if (rows[state] < 0 || rows[state + 1] < rows[state]
            || rows[state + 1] > n_edges || cost[state] < 0 || current[state] < 0
            || exits[state] > 1 || deaths[state] > 1
            || (exits[state] && deaths[state])) return -2;
    }
    for (edge = 0; edge < n_edges; ++edge) {
        if (columns[edge] < 0 || columns[edge] >= n
            || negative_log_passive[edge] < 0) return -3;
    }
    for (state = 0; state < n; ++state) {
        int64_t aggregate = INT32_MAX;
        if (exits[state]) { output[state] = 0; continue; }
        if (deaths[state]) { output[state] = INT32_MAX; continue; }
        for (edge = rows[state]; edge < rows[state + 1]; ++edge) {
            int64_t term, distance, smallest;
            if (current[columns[edge]] == INT32_MAX
                || negative_log_passive[edge] == INT32_MAX) continue;
            term = (int64_t)current[columns[edge]] + negative_log_passive[edge];
            if (term >= INT32_MAX) continue;
            if (aggregate == INT32_MAX) { aggregate = term; continue; }
            distance = aggregate > term ? aggregate - term : term - aggregate;
            smallest = aggregate < term ? aggregate : term;
            aggregate = smallest - softplus(distance);
        }
        if (aggregate == INT32_MAX) { output[state] = INT32_MAX; continue; }
        aggregate += cost[state];
        output[state] = aggregate >= INT32_MAX ? INT32_MAX
            : (aggregate < 0 ? 0 : (int32_t)aggregate);
    }
    return 0;
}

int32_t ember_log_q16_next(
    int32_t state, int32_t n, const int32_t *rows, const int32_t *columns,
    const int32_t *negative_log_passive, const int32_t *values) {
    int32_t edge, best = -1;
    int64_t best_score = INT64_MAX;
    if (state < 0 || state >= n || !rows || !columns
        || !negative_log_passive || !values || values[state] == INT32_MAX) return -1;
    for (edge = rows[state]; edge < rows[state + 1]; ++edge) {
        int32_t target = columns[edge];
        int64_t score;
        if (target < 0 || target >= n || negative_log_passive[edge] == INT32_MAX
            || values[target] >= values[state]) continue;
        score = (int64_t)values[target] + negative_log_passive[edge];
        if (score < best_score || (score == best_score && target < best)) {
            best_score = score;
            best = target;
        }
    }
    return best;
}
