"""Exact deadline routing with bounded dynamic programming and dual A*.

Positive time and cost edges, one hard time resource. Standard A* and a
single-resource lazy-dominance variant are explicit matched baselines.
Dual A* uses max_mu d_mu(v)-mu*(deadline-elapsed), never learned or bucketed
heuristics. Tables and labels occupy caller-owned numeric scratch; Python graph
objects and heaps remain host memory and are not whole-MCU accounting.
"""

from __future__ import annotations

import heapq
from dataclasses import replace

import numpy as np

from ember.routing.deadline import DeadlineRoute
from mcu import INT32_MAX, Q16_SCALE, MCUArena

INF = np.iinfo(np.int64).max
MAX_DUAL_RUNS = 8


def _vector(storage, n, dtype, fill):
    kind = np.dtype(dtype)
    result = np.frombuffer(
        storage.alloc(n * kind.itemsize, alignment=kind.itemsize), dtype=kind
    )
    result[:] = fill
    return result


def _table(reverse, goals, storage, *, multiplier=None):
    """Reversed lexicographic Dijkstra. None prioritizes time, else cost+mu*time."""
    n = len(reverse)
    primary = _vector(storage, n, np.int64, INF)
    secondary = _vector(storage, n, np.int64, INF)
    following = _vector(storage, n, np.int32, -1)
    # Transient host lists avoid repeated NumPy scalar boxing in the heap.
    # Persistent numeric tables occupy the measured arena; host temporaries
    # are explicitly outside whole-MCU memory accounting.
    first_values = [INF] * n
    second_values = [INF] * n
    next_values = [-1] * n
    queue = []
    for goal in sorted(set(goals)):
        first_values[goal] = second_values[goal] = 0
        heapq.heappush(queue, (0, 0, goal))
    while queue:
        first, second, node = heapq.heappop(queue)
        if (first, second) != (first_values[node], second_values[node]):
            continue
        for source, steps, cost in reverse[node]:
            if multiplier is None:
                candidate = (first + steps, second + cost)
            else:
                candidate = (first + cost + multiplier * steps, second + steps)
            if max(candidate) >= INF:
                continue
            if candidate < (first_values[source], second_values[source]):
                first_values[source], second_values[source] = candidate
                next_values[source] = node
                heapq.heappush(queue, (*candidate, source))
    primary[:] = first_values
    secondary[:] = second_values
    following[:] = next_values
    return primary, secondary, following


def _path(table, start, goals):
    if table[0][start] == INF:
        return ()
    route = [start]
    goal_set = set(goals)
    while route[-1] not in goal_set:
        target = int(table[2][route[-1]])
        if target < 0 or len(route) >= len(table[0]):
            raise RuntimeError("invalid positive-weight shortest-path certificate")
        route.append(target)
    return tuple(route)


def _resource_dp(adjacent, start, goals, deadline, storage):
    """Vectorized exact Bellman DP on remaining time, with no label pruning."""
    n = len(adjacent)
    table = np.frombuffer(
        storage.alloc((deadline + 1) * n * 8, alignment=8), dtype=np.int64
    ).reshape(deadline + 1, n)
    table[:] = INF
    goal_ids = np.array(sorted(set(goals)), dtype=np.int32)
    table[:, goal_ids] = 0
    flat = [
        (u, v, steps, cost) for u, row in enumerate(adjacent) for v, steps, cost in row
    ]
    if flat:
        source, target, duration, weight = np.array(flat, dtype=np.int64).T
        largest_cost = int(np.max(weight))
        if largest_cost * deadline >= INF:
            raise ValueError("dynamic cost exceeds int64 range")
        for remaining in range(1, deadline + 1):
            table[remaining] = table[remaining - 1]
            usable = duration <= remaining
            child = table[remaining - duration[usable], target[usable]]
            finite = child != INF
            # INF is excluded before addition so an unreachable state cannot wrap.
            candidates = child[finite] + weight[usable][finite]
            np.minimum.at(table[remaining], source[usable][finite], candidates)
    objective = int(table[deadline, start])
    if objective == INF:
        return (), 0, 0
    route = [start]
    remaining = deadline
    goals = set(goals)
    while route[-1] not in goals:
        node = route[-1]
        options = [
            (steps, target, cost)
            for target, steps, cost in adjacent[node]
            if steps <= remaining
            and table[remaining - steps, target] != INF
            and int(table[remaining - steps, target]) + cost == table[remaining, node]
        ]
        if not options:
            raise RuntimeError("dynamic-programming witness missing")
        steps, target, _ = min(options)
        route.append(target)
        remaining -= steps
    return tuple(route), deadline - remaining, objective


def _window_dp(adjacent, start, goals, deadline, minimum, maximum, storage):
    """Exact DP restricted to resource states that can occur on feasible paths."""
    n = len(adjacent)
    low = _vector(storage, n, np.int64, 0)
    high = _vector(storage, n, np.int64, 0)
    offsets = _vector(storage, n + 1, np.int64, 0)
    low[:], high[:] = minimum, maximum
    widths = np.maximum(0, high - low + 1)
    offsets[1:] = np.cumsum(widths)
    values = _vector(storage, int(offsets[-1]), np.int64, INF)
    goal_set = set(goals)
    for goal in goal_set:
        values[offsets[goal] : offsets[goal + 1]] = 0
    flat = [(u, v, t, c) for u, row in enumerate(adjacent) for v, t, c in row]
    source, target, duration, weight = np.array(flat, dtype=np.int64).T
    if int(weight.max()) * deadline >= INF:
        raise ValueError("dynamic cost exceeds int64 range")
    remaining = 1
    while remaining <= deadline:
        nodes = np.flatnonzero((low <= remaining) & (remaining <= high))
        if not len(nodes):
            upcoming = low[(low > remaining) & (low <= high)]
            if not len(upcoming):
                break
            remaining = int(upcoming.min())
            continue
        inherited = nodes[low[nodes] < remaining]
        here = offsets[inherited] + remaining - low[inherited]
        values[here] = values[here - 1]
        child_time = remaining - duration
        usable = (
            (low[source] <= remaining)
            & (remaining <= high[source])
            & (low[target] <= child_time)
            & (child_time <= high[target])
        )
        u, v, time_left = source[usable], target[usable], child_time[usable]
        child = values[offsets[v] + time_left - low[v]]
        finite = child != INF
        np.minimum.at(
            values,
            offsets[u[finite]] + remaining - low[u[finite]],
            child[finite] + weight[usable][finite],
        )
        remaining += 1
    best = int(values[offsets[start] + deadline - low[start]])
    if best == INF:
        raise RuntimeError("feasible corridor lost its path")
    route, remaining = [start], deadline
    while route[-1] not in goal_set:
        node = route[-1]
        current = int(values[offsets[node] + remaining - low[node]])
        options = []
        for target_node, steps, cost in adjacent[node]:
            child_time = remaining - steps
            if low[target_node] <= child_time <= high[target_node]:
                child = int(
                    values[offsets[target_node] + child_time - low[target_node]]
                )
                if child != INF and child + cost == current:
                    options.append((steps, target_node))
        if not options:
            raise RuntimeError("corridor witness missing")
        steps, target_node = min(options)
        route.append(target_node)
        remaining -= steps
    return tuple(route), deadline - remaining, best


def constrained_search(
    adjacent, start, goals, deadline, max_labels, *, arena, strategy, legacy_fallback
):
    n = len(adjacent)
    storage = arena if arena is not None else MCUArena(200 * 1024)
    offset = storage.used
    if start in goals:
        return DeadlineRoute(
            (start,),
            0,
            0.0,
            True,
            True,
            False,
            0,
            storage.high_water,
            lower_bound_q16=0,
            search_strategy=strategy,
        )
    # Two exact certificates (40*N bytes), alignment and at least one label.
    if storage.capacity - ((offset + 7) & ~7) < 40 * n + 32:
        return replace(legacy_fallback(), search_strategy="pareto_memory_fallback")
    runs, expanded, used, dynamic_states = 0, 0, 0, 0
    reverse = [[] for _ in adjacent]
    edge_scales = []
    for source, neighbours in enumerate(adjacent):
        for target, steps, cost in neighbours:
            reverse[target].append((source, steps, cost))
            edge_scales.append(max(1, cost // steps))
    best_route, best_time, best_cost, bound = (), 0, 0, 0

    def result(*, feasible=True, optimal=False, fallback=False):
        return DeadlineRoute(
            best_route,
            best_time,
            best_cost / Q16_SCALE,
            feasible,
            optimal,
            fallback,
            used,
            storage.high_water,
            lower_bound_q16=best_cost if optimal and feasible else bound,
            search_strategy=strategy,
            expanded_labels=expanded,
            preprocessing_runs=runs,
            dynamic_states=dynamic_states,
        )

    try:
        fast = _table(reverse, goals, storage)
        runs += 1
        best_route = _path(fast, start, goals)
        if not best_route:
            return result(feasible=False, optimal=True)
        best_time, best_cost = int(fast[0][start]), int(fast[1][start])
        if best_time > deadline:
            return result(feasible=False, optimal=True)
        if best_time == deadline:
            bound = best_cost
            return result(optimal=True)
        dp_bytes = (deadline + 1) * n * 8
        dp_fits = dp_bytes <= storage.capacity - ((offset + 7) & ~7)
        if strategy == "bounded_dp":
            if not dp_fits:
                return result(fallback=True)
            storage.used = offset
            best_route, best_time, best_cost = _resource_dp(
                adjacent, start, goals, deadline, storage
            )
            dynamic_states = (deadline + 1) * n
            return result(optimal=True)
        cheap = _table(reverse, goals, storage, multiplier=0)
        runs += 1
        bound = int(cheap[0][start])
        if cheap[1][start] <= deadline:
            best_route = _path(cheap, start, goals)
            best_time, best_cost = int(cheap[1][start]), int(cheap[0][start])
            return result(optimal=True)
        if strategy == "hybrid_exact" and dp_fits:
            # Tables are dead after this early return; recycle them, preserving
            # the caller's unrelated arena prefix and the same numeric ceiling.
            storage.used = offset
            best_route, best_time, best_cost = _resource_dp(
                adjacent, start, goals, deadline, storage
            )
            dynamic_states = (deadline + 1) * n
            return result(optimal=True)
        if strategy in ("hybrid_exact", "window_dp"):
            certificate_end = storage.used
            if storage.capacity - storage.used >= 20 * n + 8:
                from_start = _table(adjacent, (start,), storage)
                runs += 1
                # Avoid INF subtraction/overflow for disconnected vertices.
                reachable = (fast[0] != INF) & (from_start[0] != INF)
                minimum = np.where(reachable, fast[0], 1).copy()
                maximum = np.where(reachable, deadline - from_start[0], 0).copy()
                states = sum(
                    max(0, int(hi) - int(lo) + 1)
                    for lo, hi in zip(minimum, maximum, strict=True)
                )
                packed_bytes = 24 * n + 8 + 8 * states
                if packed_bytes <= storage.capacity - ((offset + 7) & ~7):
                    storage.used = offset
                    best_route, best_time, best_cost = _window_dp(
                        adjacent, start, goals, deadline, minimum, maximum, storage
                    )
                    dynamic_states = states
                    return result(optimal=True)
                storage.used = certificate_end
            if strategy == "window_dp":
                return result(fallback=True)
        # Reserve the requested label quota before allocating dual tables.
        reserve = min(max_labels, max(1, (storage.capacity - storage.used - 8) // 17))
        tables = []
        if strategy in ("dual_astar", "hybrid_exact"):
            scale = int(np.median(edge_scales)) if edge_scales else 1
            multiplier, low, high = max(1, scale), 0, None
            for _ in range(MAX_DUAL_RUNS):
                if storage.capacity - storage.used - 17 * reserve < 20 * n + 8:
                    break
                # No 64-bit saturation may masquerade as an admissible bound.
                if any(
                    n * (cost + multiplier * steps) >= INF
                    for row in adjacent
                    for _, steps, cost in row
                ):
                    break
                table = _table(reverse, goals, storage, multiplier=multiplier)
                runs += 1
                tables.append((multiplier, table[0]))
                weighted = int(table[0][start])
                travel = int(table[1][start])
                objective = weighted - multiplier * travel
                bound = max(bound, weighted - multiplier * deadline)
                if travel <= deadline:
                    if objective < best_cost:
                        best_route = _path(table, start, goals)
                        best_time, best_cost = travel, objective
                    high = multiplier
                else:
                    low = multiplier
                if bound == best_cost:
                    return result(optimal=True)
                if high is None:
                    multiplier *= 2
                else:
                    if high - low <= 1:
                        break
                    multiplier = (low + high) // 2

        def heuristic(node, elapsed):
            lower = int(cheap[0][node])
            for multiplier, distance in tables:
                lower = max(
                    lower, int(distance[node]) - multiplier * (deadline - elapsed)
                )
            return lower

        if strategy == "lazy_astar":
            if storage.capacity - storage.used < 8 * n + 24:
                return result(fallback=True)
            closed_time = _vector(storage, n, np.int64, INF)
        capacity = min(
            max_labels, max(0, (storage.capacity - ((storage.used + 3) & ~3)) // 17)
        )
        if capacity < 1:
            return result(fallback=True)
        labels = np.frombuffer(storage.alloc(capacity * 16), np.int32).reshape(-1, 4)
        live = np.frombuffer(storage.alloc(capacity), np.uint8)
        live[:] = 0
        labels[0], live[0] = (start, 0, 0, -1), 1
        used = 1
        frontiers = [[] for _ in adjacent]
        frontiers[start].append(0)
        queue = [(heuristic(start, 0), 0, 0)]
        goals = set(goals)
        truncated = False

        def label_path(index):
            path = []
            while index >= 0:
                path.append(int(labels[index, 0]))
                index = int(labels[index, 3])
            return tuple(reversed(path))

        while queue:
            priority, elapsed, index = heapq.heappop(queue)
            if priority >= best_cost:
                return result(optimal=not truncated, fallback=truncated)
            if not live[index]:
                continue
            node, _, cost, _ = (int(x) for x in labels[index])
            if strategy == "lazy_astar":
                if elapsed >= closed_time[node]:
                    continue
                closed_time[node] = elapsed
            expanded += 1
            if node in goals:
                best_route, best_time, best_cost = label_path(index), elapsed, cost
                return result(optimal=not truncated, fallback=truncated)
            for target, steps, edge_cost in adjacent[node]:
                next_time, next_cost = elapsed + steps, cost + edge_cost
                if (
                    fast[0][target] == INF
                    or int(fast[0][target]) + next_time > deadline
                ):
                    continue
                priority = next_cost + heuristic(target, next_time)
                if priority >= best_cost:
                    continue
                if max(next_time, next_cost) >= INT32_MAX:
                    truncated = True
                    continue
                frontier = frontiers[target]
                if strategy != "lazy_astar":
                    if any(
                        live[j]
                        and labels[j, 1] <= next_time
                        and labels[j, 2] <= next_cost
                        for j in frontier
                    ):
                        continue
                elif next_time >= closed_time[target]:
                    continue
                # A cheapest completion certifies a better feasible incumbent.
                if next_time + int(cheap[1][target]) <= deadline:
                    total_cost = next_cost + int(cheap[0][target])
                    if total_cost < best_cost:
                        prefix = label_path(index)
                        suffix = _path(cheap, target, goals)
                        if not set(prefix).intersection(suffix):
                            best_route = (*prefix, *suffix)
                            best_time = next_time + int(cheap[1][target])
                            best_cost = total_cost
                    if priority >= best_cost:
                        continue
                if used == capacity:
                    return result(fallback=True)
                if strategy != "lazy_astar":
                    for j in frontier:
                        if labels[j, 1] >= next_time and labels[j, 2] >= next_cost:
                            live[j] = 0
                labels[used], live[used] = (target, next_time, next_cost, index), 1
                frontier.append(used)
                heapq.heappush(queue, (priority, next_time, used))
                used += 1
        return result(optimal=not truncated, fallback=truncated)
    finally:
        storage.used = offset
