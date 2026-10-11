#include "deadline_exact.h"
#include <limits.h>
#include <string.h>
#define INF INT64_MAX
#define NONE UINT32_MAX
typedef struct {
  uint8_t *base;
  size_t used, capacity, peak;
} Pool;
static void *alloc(Pool *p, size_t bytes, size_t align) {
  size_t at = (p->used + align - 1) & ~(align - 1);
  if (at > p->capacity || bytes > p->capacity - at)
    return NULL;
  p->used = at + bytes;
  if (p->peak < p->used)
    p->peak = p->used;
  return p->base + at;
}
typedef struct {
  int64_t *a, *b;
  uint32_t *next;
} Table;
typedef struct {
  uint32_t *nodes, *pos, count;
  int64_t *a, *b;
} Heap;
static int less(Heap *h, uint32_t a, uint32_t b) {
  return h->a[a] < h->a[b] ||
         (h->a[a] == h->a[b] &&
          (h->b[a] < h->b[b] || (h->b[a] == h->b[b] && a < b)));
}
static void swap(Heap *h, uint32_t a, uint32_t b) {
  uint32_t x = h->nodes[a];
  h->nodes[a] = h->nodes[b];
  h->nodes[b] = x;
  h->pos[h->nodes[a]] = a;
  h->pos[x] = b;
}
static void up(Heap *h, uint32_t i) {
  while (i) {
    uint32_t parent = (i - 1) / 2;
    if (!less(h, h->nodes[i], h->nodes[parent]))
      break;
    swap(h, i, parent);
    i = parent;
  }
}
static void push(Heap *h, uint32_t node) {
  uint32_t i = h->pos[node];
  if (i == NONE) {
    i = h->count++;
    h->nodes[i] = node;
    h->pos[node] = i;
  }
  up(h, i);
}
static uint32_t pop(Heap *h) {
  uint32_t node = h->nodes[0];
  h->pos[node] = NONE;
  h->count--;
  if (h->count) {
    h->nodes[0] = h->nodes[h->count];
    h->pos[h->nodes[0]] = 0;
    uint32_t i = 0;
    for (;;) {
      uint32_t j = 2 * i + 1;
      if (j >= h->count)
        break;
      if (j + 1 < h->count && less(h, h->nodes[j + 1], h->nodes[j]))
        j++;
      if (!less(h, h->nodes[j], h->nodes[i]))
        break;
      swap(h, i, j);
      i = j;
    }
  }
  return node;
}
static int table(Pool *p, uint32_t n, const uint32_t *rows, const uint32_t *to,
                 const uint32_t *duration, const int64_t *weight,
                 const uint8_t *goals, int32_t source, int time_first,
                 int32_t finish, Table *t) {
  t->a = alloc(p, 8ull * n, 8);
  t->b = alloc(p, 8ull * n, 8);
  t->next = alloc(p, 4ull * n, 4);
  if (!t->a || !t->b || !t->next)
    return 0;
  size_t mark = p->used;
  uint32_t *nodes = alloc(p, 4ull * n, 4), *pos = alloc(p, 4ull * n, 4);
  if (!nodes || !pos)
    return 0;
  Heap h = {nodes, pos, 0, t->a, t->b};
  for (uint32_t v = 0; v < n; v++) {
    t->a[v] = t->b[v] = INF;
    t->next[v] = pos[v] = NONE;
  }
  for (uint32_t v = 0; v < n; v++)
    if ((source >= 0 && v == (uint32_t)source) || (source < 0 && goals[v])) {
      t->a[v] = t->b[v] = 0;
      push(&h, v);
    }
  int64_t threshold = INF;
  while (h.count) {
    uint32_t u = pop(&h);
    if (finish >= 0)
      t->next[u] |= 0x80000000u;
    if (finish >= 0 && u == (uint32_t)finish) {
      threshold = t->a[u];
      break;
    }
    for (uint32_t e = rows[u]; e < rows[u + 1]; e++) {
      uint32_t v = to[e];
      int64_t x = time_first ? duration[e] : weight[e],
              y = time_first ? weight[e] : duration[e];
      if (t->a[u] > INF - x || t->b[u] > INF - y)
        continue;
      int64_t a = t->a[u] + x, b = t->b[u] + y;
      if (a < t->a[v] || (a == t->a[v] && b < t->b[v])) {
        t->a[v] = a;
        t->b[v] = b;
        t->next[v] = u;
        push(&h, v);
      }
    }
  }
  if (finish >= 0) {
    /* min(exact_distance, threshold) is admissible and consistent for
       positive edges. Unsettled nodes cannot have distance below threshold.
       Their secondary time is unknown, so no suffix witness is offered. */
    for (uint32_t v = 0; v < n; v++) {
      if (threshold != INF && (t->a[v] == INF || !(t->next[v] & 0x80000000u))) {
        t->a[v] = threshold;
        t->b[v] = INF;
        t->next[v] = NONE;
      } else if (t->next[v] != NONE) {
        t->next[v] &= 0x7fffffffu;
      }
    }
  }
  p->used = mark;
  return 1;
}
static uint32_t witness(uint32_t n, uint32_t start, const uint8_t *goals,
                        const uint32_t *next, uint32_t *route) {
  uint32_t length = 0, u = start;
  for (;;) {
    if (length >= n || u >= n)
      return 0;
    route[length++] = u;
    if (goals[u])
      return length;
    u = next[u];
  }
}
typedef struct {
  int64_t cost, f;
  uint32_t node, time, parent, pad;
} Label;
static int label_less(const Label *l, uint32_t a, uint32_t b) {
  return l[a].f < l[b].f ||
         (l[a].f == l[b].f &&
          (l[a].time < l[b].time || (l[a].time == l[b].time && a < b)));
}
static void label_push(uint32_t *heap, uint32_t *size, const Label *l,
                       uint32_t node) {
  uint32_t i = (*size)++;
  while (i) {
    uint32_t parent = (i - 1) / 2;
    if (!label_less(l, node, heap[parent]))
      break;
    heap[i] = heap[parent];
    i = parent;
  }
  heap[i] = node;
}
static uint32_t label_pop(uint32_t *heap, uint32_t *size, const Label *l) {
  uint32_t answer = heap[0], last = heap[--(*size)];
  if (*size) {
    uint32_t i = 0;
    while (2 * i + 1 < *size) {
      uint32_t j = 2 * i + 1;
      if (j + 1 < *size && label_less(l, heap[j + 1], heap[j]))
        j++;
      if (!label_less(l, heap[j], last))
        break;
      heap[i] = heap[j];
      i = j;
    }
    heap[i] = last;
  }
  return answer;
}
static uint32_t label_path(const Label *l, uint32_t index, uint32_t n,
                           uint32_t *route) {
  uint32_t length = 0;
  while (index != NONE) {
    if (length >= n)
      return 0;
    route[length++] = l[index].node;
    index = l[index].parent;
  }
  for (uint32_t i = 0; i < length / 2; i++) {
    uint32_t x = route[i];
    route[i] = route[length - 1 - i];
    route[length - 1 - i] = x;
  }
  return length;
}
/* 1 proved; 0 capped. Lazy dominance requires the fixed consistent node cost
 * heuristic. */
static int search(Pool *p, uint32_t n, const uint32_t *rows, const uint32_t *to,
                  const uint32_t *duration, const int64_t *weight,
                  const uint8_t *goals, uint32_t start, uint32_t deadline,
                  uint32_t limit, int suffix, Table fast, Table cheap,
                  uint32_t *route, ember_deadline_stats *s) {
  size_t mark = p->used;
  uint32_t *closed = alloc(p, 4ull * n, 4);
  if (!closed) {
    p->used = mark;
    return 0;
  }
  for (uint32_t i = 0; i < n; i++)
    closed[i] = NONE;
  size_t remaining = p->capacity - ((p->used + 7) & ~(size_t)7);
  uint32_t quota = (uint32_t)(remaining / (sizeof(Label) + 4));
  if (quota > limit)
    quota = limit;
  if (!quota) {
    p->used = mark;
    return 0;
  }
  Label *l = alloc(p, sizeof(Label) * quota, 8);
  uint32_t *heap = alloc(p, 4ull * quota, 4);
  if (!l || !heap) {
    p->used = mark;
    return 0;
  }
  uint32_t used = 1, count = 0;
  l[0] = (Label){0, cheap.a[start], start, 0, NONE, 0};
  label_push(heap, &count, l, 0);
  int proved = 0;
  while (count) {
    uint32_t index = label_pop(heap, &count, l);
    Label current = l[index];
    if (current.f >= s->cost) {
      proved = 1;
      break;
    }
    uint32_t u = current.node;
    if (closed[u] <= current.time)
      continue;
    closed[u] = current.time;
    if (goals[u]) {
      s->route_length = label_path(l, index, n, route);
      s->cost = current.cost;
      s->travel = current.time;
      proved = 1;
      break;
    }
    if (suffix && cheap.b[u] <= deadline - current.time &&
        current.cost <= INF - cheap.a[u] &&
        current.cost + cheap.a[u] < s->cost) {
      /* Use spare output space temporarily only if the incumbent improves. */
      uint32_t length = 0, k = index;
      while (k != NONE) {
        length++;
        k = l[k].parent;
      }
      uint32_t v = u, extra = 0;
      int valid = length <= n;
      while (valid && !goals[v]) {
        v = cheap.next[v];
        if (v >= n || ++extra + length > n) {
          valid = 0;
          break;
        }
        k = index;
        while (k != NONE) {
          if (l[k].node == v) {
            valid = 0;
            break;
          }
          k = l[k].parent;
        }
      }
      if (valid) {
        s->route_length = label_path(l, index, n, route);
        v = u;
        while (!goals[v]) {
          v = cheap.next[v];
          route[s->route_length++] = v;
        }
        s->cost = current.cost + cheap.a[u];
        s->travel = current.time + cheap.b[u];
        if (current.f >= s->cost) {
          proved = 1;
          break;
        }
      }
    }
    int capped = 0;
    for (uint32_t e = rows[u]; e < rows[u + 1]; e++) {
      uint32_t v = to[e];
      if (duration[e] > deadline - current.time)
        continue;
      uint32_t time = current.time + duration[e];
      if (fast.a[v] == INF || fast.a[v] > deadline - time ||
          closed[v] <= time || current.cost > INF - weight[e] ||
          cheap.a[v] == INF)
        continue;
      int64_t cost = current.cost + weight[e];
      if (cost > INF - cheap.a[v] || cost + cheap.a[v] >= s->cost)
        continue;
      if (used >= quota) {
        capped = 1;
        break;
      }
      l[used] = (Label){cost, cost + cheap.a[v], v, time, index, 0};
      label_push(heap, &count, l, used++);
    }
    if (capped)
      break;
    if (!count)
      proved = 1;
  }
  s->labels = used;
  p->used = mark;
  return proved;
}
static int64_t value(const int64_t *values, const int64_t *low,
                     const int64_t *high, const size_t *offset, uint32_t v,
                     int64_t time) {
  if (time < low[v] || time > high[v])
    return INF;
  return values[offset[v] + (size_t)(time - low[v])];
}
static int dynamic(Pool *p, uint32_t n, const uint32_t *rows,
                   const uint32_t *to, const uint32_t *duration,
                   const int64_t *weight, const uint8_t *goals, uint32_t start,
                   uint32_t deadline, const int64_t *minimum,
                   const int64_t *maximum, int packed, uint32_t *route,
                   ember_deadline_stats *s) {
  int64_t *low = NULL, *high = NULL;
  size_t *offset = NULL, states = 0;
  if (packed) {
    low = alloc(p, 8ull * n, 8);
    high = alloc(p, 8ull * n, 8);
    offset = alloc(p, sizeof(size_t) * (n + 1ull), sizeof(size_t));
    if (!low || !high || !offset)
      return 0;
    offset[0] = 0;
    for (uint32_t u = 0; u < n; u++) {
      low[u] = minimum[u];
      high[u] = maximum[u];
      size_t width = high[u] >= low[u] ? (size_t)(high[u] - low[u] + 1) : 0;
      if (width > SIZE_MAX - states)
        return 0;
      states += width;
      offset[u + 1] = states;
    }
  } else {
    if (n && (size_t)(deadline + 1) > SIZE_MAX / n)
      return 0;
    states = (size_t)(deadline + 1) * n;
  }
  if (states > SIZE_MAX / 8 || states > UINT32_MAX)
    return 0;
  int64_t *values = alloc(p, 8 * states, 8);
  if (!values)
    return 0;
  for (size_t i = 0; i < states; i++)
    values[i] = INF;
  for (uint32_t remaining = 0;;) {
    int active = 0;
    uint32_t next_time = deadline;
    for (uint32_t u = 0; u < n; u++) {
      if (packed && (remaining < low[u] || remaining > high[u])) {
        if (high[u] >= low[u] && low[u] > remaining && low[u] < next_time)
          next_time = (uint32_t)low[u];
        continue;
      }
      active = 1;
      size_t here =
          packed ? offset[u] + remaining - low[u] : (size_t)remaining * n + u;
      if (goals[u]) {
        values[here] = 0;
        continue;
      }
      int64_t best = remaining && (!packed || remaining > low[u])
                         ? values[packed ? here - 1 : here - n]
                         : INF;
      for (uint32_t e = rows[u]; e < rows[u + 1]; e++) {
        if (duration[e] > remaining)
          continue;
        uint32_t v = to[e], child_time = remaining - duration[e];
        int64_t child = packed ? value(values, low, high, offset, v, child_time)
                               : values[(size_t)child_time * n + v];
        if (child != INF && child <= INF - weight[e] &&
            child + weight[e] < best)
          best = child + weight[e];
      }
      values[here] = best;
    }
    if (remaining == deadline)
      break;
    if (packed && !active && next_time > remaining)
      remaining = next_time;
    else
      remaining++;
  }
  size_t root = packed ? offset[start] + deadline - low[start]
                       : (size_t)deadline * n + start;
  int64_t best = values[root];
  if (best == INF)
    return -1;
  uint32_t u = start, remaining = deadline, length = 0;
  for (;;) {
    if (length >= n)
      return -1;
    route[length++] = u;
    if (goals[u])
      break;
    int found = 0;
    uint32_t chosen = NONE, chosen_time = NONE;
    int64_t current = packed ? value(values, low, high, offset, u, remaining)
                             : values[(size_t)remaining * n + u];
    for (uint32_t e = rows[u]; e < rows[u + 1]; e++) {
      if (duration[e] > remaining)
        continue;
      uint32_t v = to[e], t = remaining - duration[e];
      int64_t child = packed ? value(values, low, high, offset, v, t)
                             : values[(size_t)t * n + v];
      if (child != INF && child <= INF - weight[e] &&
          child + weight[e] == current &&
          (duration[e] < chosen_time ||
           (duration[e] == chosen_time && v < chosen))) {
        found = 1;
        chosen = v;
        chosen_time = duration[e];
      }
    }
    if (!found)
      return -1;
    u = chosen;
    remaining -= chosen_time;
  }
  s->cost = best;
  s->travel = deadline - remaining;
  s->route_length = length;
  s->states = (uint32_t)states;
  return 1;
}
int ember_deadline_exact(uint32_t n, const uint32_t *rows, const uint32_t *to,
                         const uint32_t *duration, const int64_t *weight,
                         const uint32_t *rr, const uint32_t *rt,
                         const uint32_t *rd, const int64_t *rw,
                         const uint8_t *goals, uint32_t start,
                         uint32_t deadline, uint32_t label_limit, uint32_t mode,
                         uint8_t *arena, size_t capacity, size_t offset,
                         uint32_t *route, ember_deadline_stats *s) {
  memset(s, 0, sizeof(*s));
  if (!n || start >= n || offset > capacity || mode > 4 ||
      deadline >= INT32_MAX)
    return -1;
  Pool p = {arena, offset, capacity, offset};
  s->method = mode;
  if (goals[start]) {
    route[0] = start;
    s->route_length = 1;
    s->feasible = s->optimal = 1;
    s->peak = offset;
    return 1;
  }
  Table fast, cheap, forward;
  if (!table(&p, n, rr, rt, rd, rw, goals, -1, 1,
             mode == 4 ? (int32_t)start : -1, &fast))
    goto quota;
  s->runs++;
  if (fast.a[start] == INF) {
    s->optimal = 1;
    goto done;
  }
  s->cost = fast.b[start];
  s->travel = fast.a[start];
  s->route_length = witness(n, start, goals, fast.next, route);
  if (!s->route_length)
    return -1;
  if (s->travel > deadline) {
    s->optimal = 1;
    goto done;
  }
  s->feasible = 1;
  if (s->travel == deadline) {
    s->optimal = 1;
    s->lower_bound = s->cost;
    goto done;
  }
  if (!table(&p, n, rr, rt, rd, rw, goals, -1, 0,
             mode == 4 ? (int32_t)start : -1, &cheap))
    goto quota;
  s->runs++;
  s->lower_bound = cheap.a[start];
  if (cheap.b[start] <= deadline) {
    s->cost = cheap.a[start];
    s->travel = cheap.b[start];
    s->route_length = witness(n, start, goals, cheap.next, route);
    s->optimal = 1;
    goto done;
  }
  if (mode == 3 || mode == 4) {
    uint32_t limit = label_limit;
    if (mode == 4 && limit > 32)
      limit = 32;
    if (search(&p, n, rows, to, duration, weight, goals, start, deadline, limit,
               mode == 4, fast, cheap, route, s)) {
      s->optimal = 1;
      goto done;
    }
    if (mode == 3)
      goto done;
  }
  /* Certificates remain intact if an allocation cannot fit. */
  size_t certificate = p.used, aligned = (offset + 7) & ~(size_t)7;
  size_t full = (size_t)(deadline + 1) * n;
  if (mode != 2 && aligned <= capacity && full <= (capacity - aligned) / 8) {
    p.used = offset;
    int ok = dynamic(&p, n, rows, to, duration, weight, goals, start, deadline,
                     NULL, NULL, 0, route, s);
    if (ok != 1)
      return -1;
    s->optimal = 1;
    goto done;
  }
  if (mode == 1)
    goto done;
  if (!table(&p, n, rows, to, duration, weight, goals, (int32_t)start, 1, -1,
             &forward))
    goto quota;
  s->runs++;
  /* Reuse forward's two fields as persistent interval metadata, keeping them
     after the old certificate prefix. Packed DP may overwrite only that prefix.
   */
  for (uint32_t u = 0; u < n; u++) {
    int64_t from = forward.a[u];
    forward.a[u] = fast.a[u] == INF || from == INF ? 1 : fast.a[u];
    forward.b[u] =
        fast.a[u] == INF || from == INF ? 0 : (int64_t)deadline - from;
  }
  size_t states = 0;
  for (uint32_t u = 0; u < n; u++)
    if (forward.b[u] >= forward.a[u]) {
      size_t width = (size_t)(forward.b[u] - forward.a[u] + 1);
      if (width > SIZE_MAX - states)
        goto quota;
      states += width;
    }
  /* To recycle all arena storage, move intervals to the output-independent
     certificate arrays before DP starts; keep their 16*N bytes at the prefix.
   */
  int64_t *lo = (int64_t *)(arena + aligned), *hi = lo + n;
  memmove(lo, forward.a, 8ull * n);
  memmove(hi, forward.b, 8ull * n);
  if (states > SIZE_MAX / 8 || aligned + 16ull * n > capacity ||
      40ull * n + sizeof(size_t) > capacity - aligned ||
      states > (capacity - aligned - 40ull * n - sizeof(size_t)) / 8) {
    p.used = certificate;
    goto quota;
  }
  p.used = aligned + 16ull * n;
  {
    int ok = dynamic(&p, n, rows, to, duration, weight, goals, start, deadline,
                     lo, hi, 1, route, s);
    if (ok != 1)
      return -1;
    s->optimal = 1;
    goto done;
  }
quota:
  s->peak = p.peak;
  return -2;
done:
  if (s->optimal && s->feasible)
    s->lower_bound = s->cost;
  s->peak = p.peak;
  return 1;
}
