#include "deadline.h"
#include <limits.h>
#include <stddef.h>

static void fast_route(int32_t goal, int32_t n, const int32_t *previous,
    int32_t *route, ember_deadline_result *result) {
    int32_t count = 0, i;
    while (goal >= 0 && count < n) { route[count++] = goal; goal = previous[goal]; }
    for (i = 0; i < count / 2; ++i) {
        int32_t temporary = route[i]; route[i] = route[count - 1 - i];
        route[count - 1 - i] = temporary;
    }
    result->route_length = count;
}

int ember_deadline_route(
    int32_t n, int32_t n_edges, const int32_t *rows, const int32_t *columns,
    const int32_t *edge_steps, const int32_t *edge_cost_q16,
    const uint8_t *exits, int32_t start, int32_t deadline, int32_t label_capacity,
    ember_deadline_label *labels, int32_t *fast_time, int64_t *fast_cost,
    int32_t *fast_previous, uint8_t *visited, int32_t *route,
    ember_deadline_result *result) {
    int32_t state, edge, goal = -1, used = 1;
    if (n <= 0 || n_edges < 0 || start < 0 || start >= n || deadline < 0
        || label_capacity < 1 || !rows || !columns || !edge_steps || !edge_cost_q16
        || !exits || !labels || !fast_time || !fast_cost || !fast_previous
        || !visited || !route || !result) return -1;
    if (rows[0] != 0 || rows[n] != n_edges) return -2;
    for (state = 0; state < n; ++state) {
        if (rows[state] < 0 || rows[state + 1] < rows[state]
            || rows[state + 1] > n_edges || exits[state] > 1) return -2;
    }
    for (edge = 0; edge < n_edges; ++edge) {
        if (columns[edge] < 0 || columns[edge] >= n || edge_steps[edge] < 1
            || edge_cost_q16[edge] < 1) return -3;
    }
    result->travel_steps = 0;
    result->route_length = 0;
    result->objective_q16 = 0;
    result->feasible = 0;
    result->optimal = 0;
    result->resource_fallback = 0;
    result->labels_used = 0;
    for (state = 0; state < n; ++state) {
        fast_time[state] = INT32_MAX; fast_cost[state] = INT64_MAX;
        fast_previous[state] = -1; visited[state] = 0;
    }
    fast_time[start] = 0; fast_cost[start] = 0;
    /* Lexicographic shortest traversal clock, then risk objective, then node ID. */
    for (;;) {
        int32_t current = -1;
        for (state = 0; state < n; ++state) {
            if (visited[state] || fast_time[state] == INT32_MAX) continue;
            if (current < 0 || fast_time[state] < fast_time[current]
                || (fast_time[state] == fast_time[current]
                    && fast_cost[state] < fast_cost[current])) current = state;
        }
        if (current < 0) break;
        visited[current] = 1;
        if (exits[current]) { goal = current; break; }
        for (edge = rows[current]; edge < rows[current + 1]; ++edge) {
            int32_t target = columns[edge];
            int64_t time = (int64_t)fast_time[current] + edge_steps[edge];
            int64_t cost = fast_cost[current] + edge_cost_q16[edge];
            if (time >= INT32_MAX) continue;
            if (time < fast_time[target] || (time == fast_time[target]
                && cost < fast_cost[target])) {
                fast_time[target] = (int32_t)time; fast_cost[target] = cost;
                fast_previous[target] = current;
            }
        }
    }
    if (goal < 0 || fast_time[goal] > deadline) {
        result->optimal = 1;
        if (goal >= 0) {
            result->travel_steps = fast_time[goal]; result->objective_q16 = fast_cost[goal];
            fast_route(goal, n, fast_previous, route, result);
        }
        return 0;
    }
    labels[0].node = start; labels[0].time = labels[0].cost = 0;
    labels[0].previous = -1; labels[0].state = 1;
    for (;;) {
        int32_t current = -1, index;
        for (index = 0; index < used; ++index) {
            if (labels[index].state != 1) continue;
            if (current < 0 || labels[index].cost < labels[current].cost
                || (labels[index].cost == labels[current].cost
                    && labels[index].time < labels[current].time)) current = index;
        }
        if (current < 0) break;
        state = labels[current].node;
        if (exits[state]) {
            int32_t count = 0, parent = current;
            result->travel_steps = labels[current].time;
            result->objective_q16 = labels[current].cost;
            result->feasible = result->optimal = 1; result->labels_used = used;
            while (parent >= 0 && count < n) {
                route[count++] = labels[parent].node; parent = labels[parent].previous;
            }
            for (index = 0; index < count / 2; ++index) {
                int32_t temporary = route[index]; route[index] = route[count - 1 - index];
                route[count - 1 - index] = temporary;
            }
            result->route_length = count;
            return 0;
        }
        labels[current].state = 2;
        for (edge = rows[state]; edge < rows[state + 1]; ++edge) {
            int32_t target = columns[edge], dominated = 0;
            int64_t time = (int64_t)labels[current].time + edge_steps[edge];
            int64_t cost = (int64_t)labels[current].cost + edge_cost_q16[edge];
            if (time > deadline || cost >= INT32_MAX) continue;
            for (index = 0; index < used; ++index) {
                if (labels[index].state && labels[index].node == target
                    && labels[index].time <= time && labels[index].cost <= cost) {
                    dominated = 1; break;
                }
            }
            if (dominated) continue;
            if (used == label_capacity) goto fallback;
            for (index = 0; index < used; ++index) {
                if (labels[index].node == target && labels[index].time >= time
                    && labels[index].cost >= cost) labels[index].state = 0;
            }
            labels[used].node = target; labels[used].time = (int32_t)time;
            labels[used].cost = (int32_t)cost; labels[used].previous = current;
            labels[used].state = 1; ++used;
        }
    }
fallback:
    result->feasible = 1; result->resource_fallback = 1; result->labels_used = used;
    result->travel_steps = fast_time[goal]; result->objective_q16 = fast_cost[goal];
    fast_route(goal, n, fast_previous, route, result);
    return 0;
}

int ember_deadline_accept(
    int32_t n, int32_t n_edges, const int32_t *rows, const int32_t *columns,
    const int32_t *edge_steps, const uint8_t *exits, int32_t start,
    int32_t deadline, const int32_t *route, int32_t route_length) {
    int32_t index;
    int64_t elapsed = 0;
    if (n <= 0 || n_edges < 0 || !rows || !columns || !edge_steps || !exits
        || !route || deadline < 0 || route_length <= 0 || route_length > n
        || start < 0 || start >= n || route[0] != start || rows[0] != 0
        || rows[n] != n_edges) return 0;
    for (index = 0; index < route_length; ++index) {
        int32_t node = route[index], edge, found = 0;
        if (node < 0 || node >= n || rows[node] < 0 || rows[node + 1] < rows[node]
            || rows[node + 1] > n_edges) return 0;
        if (index == route_length - 1) return exits[node] == 1 && elapsed <= deadline;
        for (edge = rows[node]; edge < rows[node + 1]; ++edge) {
            if (columns[edge] == route[index + 1] && edge_steps[edge] > 0) {
                elapsed += edge_steps[edge]; found = 1; break;
            }
        }
        if (!found || elapsed > deadline) return 0;
    }
    return 0;
}
