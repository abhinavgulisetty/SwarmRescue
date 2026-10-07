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

static int validate(
    int32_t n, int32_t n_edges, const int32_t *rows, const int32_t *columns,
    const int32_t *negative_log_passive, const int32_t *cost,
    const uint8_t *exits, const uint8_t *deaths) {
    int32_t state, edge;
    if (n <= 0 || n_edges < 0 || !rows || !columns || !negative_log_passive
        || !cost || !exits || !deaths)
        return -1;
    if (rows[0] != 0 || rows[n] != n_edges) return -2;
    for (state = 0; state < n; ++state) {
        if (rows[state] < 0 || rows[state + 1] < rows[state]
            || rows[state + 1] > n_edges || cost[state] < 0
            || exits[state] > 1 || deaths[state] > 1
            || (exits[state] && deaths[state])) return -2;
    }
    for (edge = 0; edge < n_edges; ++edge) {
        if (columns[edge] < 0 || columns[edge] >= n
            || negative_log_passive[edge] < 0) return -3;
    }
    return 0;
}

static int32_t backup(
    int32_t state, const int32_t *rows, const int32_t *columns,
    const int32_t *negative_log_passive, const int32_t *cost,
    const int32_t *current, const uint8_t *exits, const uint8_t *deaths) {
        int32_t edge;
        int64_t aggregate = INT32_MAX;
        if (exits[state]) return 0;
        if (deaths[state]) return INT32_MAX;
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
        if (aggregate == INT32_MAX) return INT32_MAX;
        aggregate += cost[state];
        return aggregate >= INT32_MAX ? INT32_MAX
            : (aggregate < 0 ? 0 : (int32_t)aggregate);
}

int ember_log_q16_sweep(
    int32_t n, int32_t n_edges, const int32_t *rows, const int32_t *columns,
    const int32_t *negative_log_passive, const int32_t *cost,
    const int32_t *current, const uint8_t *exits, const uint8_t *deaths,
    int32_t *output) {
    int32_t state;
    int status;
    if (!current || !output || current == output) return -1;
    status = validate(n, n_edges, rows, columns, negative_log_passive,
                      cost, exits, deaths);
    if (status) return status;
    for (state = 0; state < n; ++state) if (current[state] < 0) return -2;
    for (state = 0; state < n; ++state) {
        output[state] = backup(state, rows, columns, negative_log_passive,
                               cost, current, exits, deaths);
    }
    return 0;
}

int32_t ember_log_q16_workspace(int32_t n, int32_t e, int32_t strategy) {
    int64_t words;
    int64_t reverse_words;
    if (n <= 0 || e < 0 || strategy < 0 || strategy > 7) return -1;
    reverse_words = (uint64_t)n * (uint64_t)e <= UINT32_MAX ? e : 2 * (int64_t)e;
    words = strategy >= 2 ? 5 * (int64_t)n + 1 + reverse_words : n;
    if ((uint64_t)words > SIZE_MAX / sizeof(int32_t)
        || (uint64_t)n + 1 > SIZE_MAX / sizeof(int32_t)
        || (uint64_t)e > SIZE_MAX / sizeof(int32_t)) return -1;
    return words > INT32_MAX ? -1 : (int32_t)words;
}

static int32_t reverse_size(int32_t n, int32_t e) {
    return (uint64_t)n * (uint64_t)e <= UINT32_MAX ? e : 2 * e;
}

/* Indexed binary heap: each vertex has at most one queued entry. */
static int before(int32_t a, int32_t b, const int32_t *values) {
    return values[a] < values[b] || (values[a] == values[b] && a < b);
}
static void heap_up(int32_t *heap, int32_t *position, const int32_t *values,
                    int32_t at) {
    int32_t node = heap[at];
    while (at > 0) {
        int32_t parent = (at - 1) / 2;
        if (!before(node, heap[parent], values)) break;
        heap[at] = heap[parent]; position[heap[at]] = at; at = parent;
    }
    heap[at] = node; position[node] = at;
}
static int32_t heap_pop(int32_t *heap, int32_t *position,
                        const int32_t *values, int32_t *size) {
    int32_t node = heap[0], at = 0, last = heap[--*size];
    position[node] = -2;
    while (2 * (int64_t)at + 1 < *size) {
        int32_t child = 2 * at + 1;
        if (child + 1 < *size && before(heap[child + 1], heap[child], values))
            ++child;
        if (!before(heap[child], last, values)) break;
        heap[at] = heap[child]; position[heap[at]] = at; at = child;
    }
    if (*size) { heap[at] = last; position[last] = at; }
    return node;
}

static void path_bound(
    int32_t n, int32_t e, const int32_t *rows, const int32_t *columns,
    const int32_t *passive, const int32_t *cost,
    const uint8_t *exits, const uint8_t *deaths, int32_t *values,
    int32_t *work, ember_log_stats *stats, int initialize_bound) {
    int packed = (uint64_t)n * (uint64_t)e <= UINT32_MAX;
    int32_t *reverse_rows = work, *sources = work + n + 1;
    int32_t *edges = packed ? NULL : sources + e;
    int32_t *heap = sources + reverse_size(n, e);
    int32_t *position = heap + n, *order = position + n;
    int32_t size = 0, ordered = 0;
    for (int32_t i = 0; i <= n; ++i) reverse_rows[i] = 0;
    for (int32_t edge = 0; edge < e; ++edge) ++reverse_rows[columns[edge] + 1];
    for (int32_t i = 0; i < n; ++i) {
        reverse_rows[i + 1] += reverse_rows[i];
        position[i] = reverse_rows[i];
    }
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t edge = rows[i]; edge < rows[i + 1]; ++edge) {
            int32_t at = position[columns[edge]]++;
            sources[at] = packed && initialize_bound
                ? (int32_t)((uint32_t)i * (uint32_t)e + (uint32_t)edge) : i;
            if (!packed) edges[at] = edge;
        }
    }
    for (int32_t i = 0; i < n; ++i) position[i] = -1;
    if (!initialize_bound) return;
    for (int32_t i = 0; i < n; ++i) if (exits[i]) {
        heap[size] = i; heap_up(heap, position, values, size++);
    }
    while (size) {
        int32_t target = heap_pop(heap, position, values, &size);
        order[ordered++] = target;
        for (int32_t at = reverse_rows[target]; at < reverse_rows[target + 1]; ++at) {
            int32_t source = packed ? (int32_t)((uint32_t)sources[at] / (uint32_t)e)
                                    : sources[at];
            int32_t edge = packed ? (int32_t)((uint32_t)sources[at] % (uint32_t)e)
                                  : edges[at];
            int64_t candidate;
            ++stats->bound_edges;
            if (exits[source] || deaths[source] || passive[edge] == INT32_MAX)
                continue;
            candidate = (int64_t)values[target] + cost[source] + passive[edge];
            if (candidate >= values[source]) continue;
            values[source] = (int32_t)candidate;
            if (position[source] == -1) {
                heap[size] = source; heap_up(heap, position, values, size++);
            } else if (position[source] >= 0) {
                heap_up(heap, position, values, position[source]);
            }
        }
    }
    /* A saturated path bound is not a proof of soft-Bellman unreachability. */
    for (int32_t i = 0; i < n; ++i) if (position[i] == -1) order[ordered++] = i;
}

static int overlap(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    if (!a_bytes || !b_bytes) return 0;
    return x <= y ? y - x < a_bytes : x - y < b_bytes;
}

/* Residual priority heap; reverse Bellman dependencies, not graph distances. */
static int priority_before(int32_t a, int32_t b, const int32_t *residual) {
    return residual[a] > residual[b] || (residual[a] == residual[b] && a < b);
}
static void priority_up(int32_t *heap, int32_t *position,
                         const int32_t *residual, int32_t at) {
    int32_t node = heap[at];
    while (at > 0) {
        int32_t parent = (at - 1) / 2;
        if (!priority_before(node, heap[parent], residual)) break;
        heap[at] = heap[parent]; position[heap[at]] = at; at = parent;
    }
    heap[at] = node; position[node] = at;
}
static void priority_down(int32_t *heap, int32_t *position,
                           const int32_t *residual, int32_t at, int32_t size) {
    int32_t node = heap[at];
    while (2 * (int64_t)at + 1 < size) {
        int32_t child = 2 * at + 1;
        if (child + 1 < size && priority_before(heap[child + 1], heap[child], residual))
            ++child;
        if (!priority_before(heap[child], node, residual)) break;
        heap[at] = heap[child]; position[heap[at]] = at; at = child;
    }
    heap[at] = node; position[node] = at;
}
static void priority_remove(int32_t node, int32_t *heap, int32_t *position,
                             const int32_t *residual, int32_t *size) {
    int32_t at = position[node], last = heap[--*size];
    position[node] = -1;
    if (at < *size) {
        heap[at] = last; position[last] = at;
        priority_up(heap, position, residual, at);
        priority_down(heap, position, residual, position[last], *size);
    }
}

static void prioritized_solve(
    int32_t n, int32_t e, const int32_t *rows, const int32_t *columns,
    const int32_t *passive, const int32_t *cost,
    const uint8_t *exits, const uint8_t *deaths, int32_t limit,
    int32_t *values, int32_t *work, ember_log_stats *stats, int bound) {
    int packed = bound && (uint64_t)n * (uint64_t)e <= UINT32_MAX;
    int32_t *reverse_rows = work, *sources = work + n + 1;
    int32_t *heap = sources + reverse_size(n, e), *position = heap + n;
    int32_t *residual = position + n, *cached = residual + n;
    int32_t size = 0;
    uint64_t applied = 0, budget = (uint64_t)limit * (uint64_t)n;
    for (int32_t i = 0; i < n; ++i) position[i] = -1;
    for (int32_t i = 0; i < n; ++i) {
        cached[i] = backup(i, rows, columns, passive, cost, values, exits, deaths);
        if (!exits[i] && !deaths[i]) ++stats->backups;
        residual[i] = values[i] - cached[i];
        if (residual[i] > 0) {
            heap[size] = i; priority_up(heap, position, residual, size++);
        }
    }
    while (size && applied < budget) {
        int32_t node = heap[0];
        priority_remove(node, heap, position, residual, &size);
        values[node] = cached[node]; ++applied;
        /* Refresh this row too: a self-loop can change its own backup. */
        for (int32_t at = reverse_rows[node]; at < reverse_rows[node + 1]; ++at) {
            int32_t source = packed ? (int32_t)((uint32_t)sources[at] / (uint32_t)e)
                                    : sources[at];
            if (exits[source] || deaths[source]) continue;
            if (position[source] >= 0)
                priority_remove(source, heap, position, residual, &size);
            cached[source] = backup(source, rows, columns, passive, cost,
                                     values, exits, deaths);
            ++stats->backups;
            residual[source] = values[source] - cached[source];
            if (residual[source] > 0) {
                heap[size] = source;
                priority_up(heap, position, residual, size++);
            }
        }
    }
    stats->fixed_point = size == 0;
    stats->sweeps = (uint32_t)((applied + n - 1) / n);
    if (!stats->sweeps) stats->sweeps = 1;
}

/* Greatest conditional integer fixed point for a sticky self-loop. A one-step
 * upper bound keeps every self term below the saturation discontinuity. Rows
 * crossing that boundary safely use an ordinary update instead. */
static int32_t diagonal_backup(
    int32_t state, const int32_t *rows, const int32_t *columns,
    const int32_t *passive, const int32_t *cost, int32_t *values,
    const uint8_t *exits, const uint8_t *deaths, ember_log_stats *stats) {
    int32_t old = values[state];
    int32_t next = backup(state, rows, columns, passive, cost, values, exits, deaths);
    int sticky = 0;
    if (!exits[state] && !deaths[state]) ++stats->backups;
    if (next == old || exits[state] || deaths[state]) return next;
    for (int32_t edge = rows[state]; edge < rows[state + 1]; ++edge) {
        if (columns[edge] != state || passive[edge] == INT32_MAX) continue;
        if ((int64_t)next + passive[edge] >= INT32_MAX) return next;
        if (passive[edge] <= 6905) sticky = 1; /* p(self) >= approximately 0.9 */
    }
    if (sticky) {
        int32_t low = 0, high = next;
        while (low < high) {
            int32_t middle = low + (int32_t)(((int64_t)high - low + 1) / 2);
            int32_t at_middle;
            values[state] = middle;
            at_middle = backup(state, rows, columns, passive, cost, values, exits, deaths);
            ++stats->backups;
            if (at_middle >= middle) low = middle;
            else high = middle - 1;
        }
        values[state] = old;
        return low;
    }
    return next;
}

int ember_log_q16_solve(
    int32_t n, int32_t e, const int32_t *rows, const int32_t *columns,
    const int32_t *passive, const int32_t *cost,
    const uint8_t *exits, const uint8_t *deaths, int32_t limit, int32_t strategy,
    int32_t *values, int32_t *work, int32_t work_words, ember_log_stats *stats) {
    int32_t need = ember_log_q16_workspace(n, e, strategy);
    int status;
    int32_t *following, *current = values, *order = NULL;
    const void *inputs[6] = {rows, columns, passive, cost, exits, deaths};
    size_t sizes[6];
    if (need < 0 || limit < 1 || !values || !work || !stats || work_words < need)
        return -1;
    sizes[0] = ((size_t)n + 1) * sizeof(int32_t);
    sizes[1] = sizes[2] = (size_t)e * sizeof(int32_t);
    sizes[3] = (size_t)n * sizeof(int32_t); sizes[4] = sizes[5] = (size_t)n;
    if (overlap(values, sizes[3], work, (size_t)need * sizeof(int32_t))
        || overlap(stats, sizeof(*stats), values, sizes[3])
        || overlap(stats, sizeof(*stats), work, (size_t)need * sizeof(int32_t)))
        return -4;
    for (int i = 0; i < 6; ++i) {
        if (overlap(values, sizes[3], inputs[i], sizes[i])
            || overlap(work, (size_t)need * sizeof(int32_t), inputs[i], sizes[i])
            || overlap(stats, sizeof(*stats), inputs[i], sizes[i])) return -4;
    }
    status = validate(n, e, rows, columns, passive, cost, exits, deaths);
    if (status) return status;
    stats->backups = stats->bound_edges = 0;
    stats->sweeps = stats->fixed_point = 0;
    if (strategy == EMBER_LOG_AUTO) {
        int sticky = 0;
        for (int32_t i = 0; i < n && !sticky; ++i) {
            if (exits[i] || deaths[i]) continue;
            for (int32_t edge = rows[i]; edge < rows[i + 1]; ++edge)
                if (columns[edge] == i && passive[edge] <= 6905) sticky = 1;
        }
        strategy = sticky ? EMBER_LOG_BOUND_DIAGONAL
            : ((int64_t)e > 12 * (int64_t)n
                ? EMBER_LOG_GAUSS_SEIDEL : EMBER_LOG_BOUND_ORDERED);
    }
    stats->strategy = (uint32_t)strategy;
    for (int32_t i = 0; i < n; ++i) values[i] = exits[i] ? 0 : INT32_MAX;
    following = work;
    if (strategy >= 2) {
        path_bound(n, e, rows, columns, passive, cost, exits, deaths,
                   values, work, stats, strategy != EMBER_LOG_PRIORITIZED);
        order = work + 3 * n + 1 + reverse_size(n, e);
        following = order + n;
    }
    if (strategy == EMBER_LOG_PRIORITIZED) {
        for (int32_t i = 0; i < n; ++i) values[i] = exits[i] ? 0 : INT32_MAX;
    }
    if (strategy == EMBER_LOG_PRIORITIZED || strategy == EMBER_LOG_BOUND_PRIORITIZED) {
        prioritized_solve(n, e, rows, columns, passive, cost, exits, deaths,
                          limit, values, work, stats, strategy == EMBER_LOG_BOUND_PRIORITIZED);
        return 0;
    }
    for (int32_t sweep = 0; sweep < limit; ++sweep) {
        int fixed = 1;
        for (int32_t at = 0; at < n; ++at) {
            int32_t state = strategy == EMBER_LOG_BOUND_ORDERED
                || strategy == EMBER_LOG_BOUND_DIAGONAL ? order[at] : at;
            int32_t next;
            if (strategy == EMBER_LOG_BOUND_DIAGONAL) {
                next = diagonal_backup(state, rows, columns, passive, cost,
                                         current, exits, deaths, stats);
            } else {
                next = backup(state, rows, columns, passive, cost,
                               current, exits, deaths);
                if (!exits[state] && !deaths[state]) ++stats->backups;
            }
            if (current[state] != next) fixed = 0;
            if (strategy == EMBER_LOG_JACOBI || strategy == EMBER_LOG_BOUND_JACOBI)
                following[state] = next;
            else current[state] = next;
        }
        ++stats->sweeps;
        if (strategy == EMBER_LOG_JACOBI || strategy == EMBER_LOG_BOUND_JACOBI) {
            int32_t *swap = current; current = following; following = swap;
        }
        if (fixed) { stats->fixed_point = 1; break; }
    }
    if (current != values) for (int32_t i = 0; i < n; ++i) values[i] = current[i];
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
